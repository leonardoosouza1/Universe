//! CPU × GPU parity of the fly map: `fly_map.wgsl` against
//! `universe::fly_map::{cpu_step, cpu_read}`.
//!
//! # Why `#[ignore]`
//!
//! Same reason as `retina_digest_parity_test.rs`: this opens a real `wgpu::Device`, so it
//! cannot run on a machine or CI without a GPU. Run it explicitly:
//!
//! ```text
//! HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0 \
//!   cargo test --release --lib -- --ignored fly_map_parity
//! ```
//!
//! # What it proves
//!
//! Three beings with DIFFERENT selections (different bindings, different gains, one with
//! a pruned cell type) run ten chained ticks on the synthetic six-neuron map, with a wave
//! that changes every tick. The commands and the full activation vector are compared
//! against the CPU reference at every tick, not only at the end: the state is recurrent,
//! so a divergence at tick 1 would be washed into a plausible-looking number by tick 10 —
//! comparing every tick is what makes an accumulated error visible.

use std::sync::Arc;

use super::{step_groups, FlyMapGpu, STEP_ROWS};
use crate::config::iara::vision::RETINA_TOTAL;
use crate::fly_map::{
    apply_reflex_readout, cpu_read, cpu_step, pack_selection, EyeFrame, FlyMap, COMMANDS,
    IN_CHANNELS, OFF_GAIN, OFF_OUT_PAIR, OFF_SCALE, OFF_TYPE_MASK, OUT_SLOTS,
};
use crate::gpu::perception::WAVE_U32S;

const N_BEINGS: usize = 3;
const N_TICKS: usize = 10;
const TAU_STEPS: u32 = 4;
/// The band the task fixes. The kernel and the reference do the same multiply-adds in the
/// same order, so the real gap is float rounding on tanh/exp, not summation order.
const TOL: f32 = 1e-5;

/// A deterministic i8 wave, different per being and per tick.
fn synthetic_wave(tick: usize) -> Vec<i8> {
    let mut w = vec![0i8; N_BEINGS * IN_CHANNELS];
    for b in 0..N_BEINGS {
        for ch in 0..IN_CHANNELS {
            let h = (ch as u32)
                .wrapping_mul(2654435761)
                .wrapping_add((b as u32).wrapping_mul(97))
                .wrapping_add((tick as u32).wrapping_mul(131));
            w[b * IN_CHANNELS + ch] = ((h % 255) as i32 - 127) as i8;
        }
    }
    w
}

/// The wave as perception writes it: 4 i8 per u32, `WAVE_U32S` words per being.
fn pack_wave(wave: &[i8]) -> Vec<u32> {
    let mut words = vec![0u32; N_BEINGS * WAVE_U32S];
    for b in 0..N_BEINGS {
        for ch in 0..IN_CHANNELS {
            let byte = wave[b * IN_CHANNELS + ch] as u8 as u32;
            words[b * WAVE_U32S + ch / 4] |= byte << ((ch % 4) * 8);
        }
    }
    words
}

pub(super) async fn open_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await?;
    let mut limits = wgpu::Limits::downlevel_defaults();
    // `fly_step` binds 14 storage buffers (12 + the eye binding's `pixel_of_neuron` and
    // `retina`); the downlevel default is 4. `gpu_evolve`'s device asks for the same 16
    // (`context.rs`).
    limits.max_storage_buffers_per_shader_stage = 16;
    adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("fly-map-parity-test"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .ok()
}

#[test]
#[ignore = "opens a real GPU — run manually: cargo test --release --lib -- --ignored fly_map_parity"]
fn fly_map_parity_gpu_matches_the_cpu_reference() {
    let Some((device, queue)) = pollster::block_on(open_device()) else {
        eprintln!("[fly_map_parity] no GPU available — test skipped");
        return;
    };
    let map = FlyMap::synthetic();
    let layout = map.fly_layout();

    // Three different selections, so a kernel that ignored the per-being tables would
    // produce three identical rows and fail.
    let mut blocks: Vec<Vec<f32>> = Vec::new();
    for b in 0..N_BEINGS {
        let mut block = vec![-1.0f32; layout.block_len()];
        // Being 0 binds channel 5 to group 0; being 1 binds 7 to group 1 and 9 to group 0;
        // being 2 binds nothing to group 0 and channel 11 to group 1.
        match b {
            0 => block[5] = 0.0,
            1 => {
                block[7] = 1.0;
                block[9] = 0.0;
            }
            _ => block[11] = 1.0,
        }
        for s in 0..OUT_SLOTS {
            block[OFF_OUT_PAIR + s] = ((s + b) % 2) as f32;
        }
        block[OFF_GAIN] = 1.0 + b as f32 * 0.5;
        block[OFF_SCALE] = 1.0 - b as f32 * 0.25;
        for t in 0..layout.n_types {
            block[OFF_TYPE_MASK + t] = 1.0;
        }
        if b == 2 {
            block[OFF_TYPE_MASK + 1] = 0.0; // being 2 runs with cell type 1 pruned
        }
        blocks.push(block);
    }
    let refs: Vec<&[f32]> = blocks.iter().map(|v| v.as_slice()).collect();
    let sel = pack_selection(&refs, layout, N_BEINGS);

    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let buf_wave = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fly-parity-wave"),
        size: (N_BEINGS * WAVE_U32S * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // The eye binding is off in this test (`eye_on: false`): a small unread buffer is
    // enough — `--fly-eye` on/off is `fly_map_parity_with_the_eye_on` below.
    let buf_retina_unused = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fly-parity-retina-unused"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let gpu = FlyMapGpu::new(
        Arc::clone(&device),
        Arc::clone(&queue),
        &map,
        N_BEINGS,
        TAU_STEPS,
    )
    .expect("the synthetic map fits in any device");
    assert!(gpu.upload_selection(&sel, N_BEINGS), "selection uploaded");
    gpu.reset_state();

    let alpha = gpu.alpha();
    let n = map.n();
    let mut x = vec![0.0f32; n * N_BEINGS];
    let mut r_a = vec![0.0f32; n * N_BEINGS];
    let mut r_b = vec![0.0f32; n * N_BEINGS];

    for tick in 0..N_TICKS {
        let wave = synthetic_wave(tick);
        queue.write_buffer(&buf_wave, 0, bytemuck::cast_slice(&pack_wave(&wave)));

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fly-parity-tick"),
        });
        gpu.dispatch(&mut enc, N_BEINGS, &buf_wave, false, &buf_retina_unused);
        queue.submit(Some(enc.finish()));
        device.poll(wgpu::Maintain::Wait);
        let gpu_cmd = gpu.read_commands_after_submit(N_BEINGS);
        let gpu_state = gpu.debug_read_state(N_BEINGS);

        cpu_step(&map, &sel, &wave, alpha, &mut x, &r_a, &mut r_b, None);
        std::mem::swap(&mut r_a, &mut r_b);
        let cpu_cmd = cpu_read(&map, &sel, &r_a);

        for j in 0..n {
            for b in 0..N_BEINGS {
                let i = j * N_BEINGS + b;
                assert!(
                    (gpu_state[i] - r_a[i]).abs() < TOL,
                    "tick {tick}, neuron {j}, being {b}: GPU {} vs CPU {}",
                    gpu_state[i],
                    r_a[i]
                );
            }
        }
        for b in 0..N_BEINGS {
            for c in 0..COMMANDS {
                assert!(
                    (gpu_cmd[b][c] - cpu_cmd[b][c]).abs() < TOL,
                    "tick {tick}, being {b}, command {c}: GPU {} vs CPU {}",
                    gpu_cmd[b][c],
                    cpu_cmd[b][c]
                );
            }
        }
    }

    // The test would also pass on a kernel that wrote zeros everywhere, so the last thing
    // it does is check that something actually moved.
    let moved = r_a.iter().filter(|v| v.abs() > 1e-6).count();
    assert!(
        moved > 0,
        "every activation is zero — the map never ran, so the comparison proved nothing"
    );
    let cmd = cpu_read(&map, &sel, &r_a);
    assert!(
        cmd.iter().any(|c| c.iter().any(|v| v.abs() > 1e-6)),
        "every command is zero — the read kernel had nothing to prove"
    );
}

/// A deterministic retina frame, different per being and per tick — same "hash the
/// indices" style as [`synthetic_wave`]. `FlyMap::synthetic_with_eye` only ever reads
/// pixels 5, 900 and `RETINA_TOTAL - 1`, but the whole buffer is filled so a wrong
/// `retina_stride` shows up as SOME neuron reading its NEIGHBOUR's frame instead of
/// silently passing.
fn synthetic_retina(tick: usize) -> Vec<u32> {
    let mut r = vec![0u32; N_BEINGS * RETINA_TOTAL];
    for b in 0..N_BEINGS {
        for p in 0..RETINA_TOTAL {
            let h = (p as u32)
                .wrapping_mul(2246822519)
                .wrapping_add((b as u32).wrapping_mul(83))
                .wrapping_add((tick as u32).wrapping_mul(151));
            let mm = h % 1_200_000; // millimetres, within the 1200 m legacy_encoding covers
            let mat = (h >> 16) % 4; // 0 Grass / 1 Road / 2 Wall / 3 Hole
            r[b * RETINA_TOTAL + p] = (mm << 8) | mat;
        }
    }
    r
}

/// The eye binding (`--fly-eye`, `docs/FLY-MAP-BRAIN.md` "Eye binding") on the real GPU:
/// same three-being, ten-tick, changing-every-tick shape as
/// [`fly_map_parity_gpu_matches_the_cpu_reference`], on `FlyMap::synthetic_with_eye`
/// (neuron 0 -> pixel 5, neuron 3 -> pixel 900, neuron 4 -> pixel `RETINA_TOTAL - 1`) —
/// the SAME three DIFFERENT genome selections as that test, so a kernel that let the eye
/// binding leak into a non-eye-driven neuron (or vice versa) fails here, not just on the
/// CPU-only tests in `fly_map_eye_tests.rs`.
#[test]
#[ignore = "opens a real GPU — run manually: cargo test --release --lib -- --ignored fly_map_parity"]
fn fly_map_parity_with_the_eye_on() {
    let Some((device, queue)) = pollster::block_on(open_device()) else {
        eprintln!("[fly_map_parity] no GPU available — test skipped");
        return;
    };
    let map = FlyMap::synthetic_with_eye();
    let layout = map.fly_layout();
    assert!(
        map.has_pixel_binding(),
        "the map under test must carry the block"
    );

    let mut blocks: Vec<Vec<f32>> = Vec::new();
    for b in 0..N_BEINGS {
        let mut block = vec![-1.0f32; layout.block_len()];
        match b {
            0 => block[5] = 0.0,
            1 => {
                block[7] = 1.0;
                block[9] = 0.0;
            }
            _ => block[11] = 1.0,
        }
        for s in 0..OUT_SLOTS {
            block[OFF_OUT_PAIR + s] = ((s + b) % 2) as f32;
        }
        block[OFF_GAIN] = 1.0 + b as f32 * 0.5;
        block[OFF_SCALE] = 1.0 - b as f32 * 0.25;
        for t in 0..layout.n_types {
            block[OFF_TYPE_MASK + t] = 1.0;
        }
        blocks.push(block);
    }
    let refs: Vec<&[f32]> = blocks.iter().map(|v| v.as_slice()).collect();
    let sel = pack_selection(&refs, layout, N_BEINGS);

    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let buf_wave = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fly-parity-eye-wave"),
        size: (N_BEINGS * WAVE_U32S * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let buf_retina = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fly-parity-eye-retina"),
        size: (N_BEINGS * RETINA_TOTAL * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let gpu = FlyMapGpu::new(
        Arc::clone(&device),
        Arc::clone(&queue),
        &map,
        N_BEINGS,
        TAU_STEPS,
    )
    .expect("the synthetic map fits in any device");
    assert!(gpu.upload_selection(&sel, N_BEINGS), "selection uploaded");
    gpu.reset_state();

    let alpha = gpu.alpha();
    let n = map.n();
    let mut x = vec![0.0f32; n * N_BEINGS];
    let mut r_a = vec![0.0f32; n * N_BEINGS];
    let mut r_b = vec![0.0f32; n * N_BEINGS];
    let pixel_of_neuron = map
        .pixel_of_neuron
        .clone()
        .expect("map carries the pixel-binding block");

    for tick in 0..N_TICKS {
        let wave = synthetic_wave(tick);
        queue.write_buffer(&buf_wave, 0, bytemuck::cast_slice(&pack_wave(&wave)));
        let retina = synthetic_retina(tick);
        queue.write_buffer(&buf_retina, 0, bytemuck::cast_slice(&retina));

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fly-parity-eye-tick"),
        });
        gpu.dispatch(&mut enc, N_BEINGS, &buf_wave, true, &buf_retina);
        queue.submit(Some(enc.finish()));
        device.poll(wgpu::Maintain::Wait);
        let gpu_cmd = gpu.read_commands_after_submit(N_BEINGS);
        let gpu_state = gpu.debug_read_state(N_BEINGS);

        let frame = EyeFrame {
            pixel_of_neuron: &pixel_of_neuron,
            retina: &retina,
            retina_stride: RETINA_TOTAL,
        };
        cpu_step(
            &map,
            &sel,
            &wave,
            alpha,
            &mut x,
            &r_a,
            &mut r_b,
            Some(&frame),
        );
        std::mem::swap(&mut r_a, &mut r_b);
        let cpu_cmd = cpu_read(&map, &sel, &r_a);

        for j in 0..n {
            for b in 0..N_BEINGS {
                let i = j * N_BEINGS + b;
                assert!(
                    (gpu_state[i] - r_a[i]).abs() < TOL,
                    "eye on, tick {tick}, neuron {j}, being {b}: GPU {} vs CPU {}",
                    gpu_state[i],
                    r_a[i]
                );
            }
        }
        for b in 0..N_BEINGS {
            for c in 0..COMMANDS {
                assert!(
                    (gpu_cmd[b][c] - cpu_cmd[b][c]).abs() < TOL,
                    "eye on, tick {tick}, being {b}, command {c}: GPU {} vs CPU {}",
                    gpu_cmd[b][c],
                    cpu_cmd[b][c]
                );
            }
        }
    }

    let moved = r_a.iter().filter(|v| v.abs() > 1e-6).count();
    assert!(
        moved > 0,
        "every activation is zero — the eye-on run never moved, so the comparison proved \
         nothing"
    );
}

/// The reflex readout (`--fly-readout reflex`, `docs/FLY-MAP-BRAIN.md` "Reflex readout")
/// on the real GPU. `apply_reflex_readout` overwrites `out_pair` on the HOST, before
/// `upload_selection` — no kernel edit, because `fly_read` already computes
/// `mean(plus) - mean(minus)` over whatever `out_pair` holds, genome-selected or not.
/// This test is the proof: three beings get three DIFFERENT genome `out_pair` choices,
/// all of them overridden by the SAME `reflex_out` block, and GPU vs CPU still agree —
/// a kernel that read the genome's own pairs instead of the packed override would show
/// three different commands here, not one.
#[test]
#[ignore = "opens a real GPU — run manually: cargo test --release --lib -- --ignored fly_map_parity"]
fn fly_map_parity_with_reflex_readout() {
    let Some((device, queue)) = pollster::block_on(open_device()) else {
        eprintln!("[fly_map_parity] no GPU available — test skipped");
        return;
    };
    let map = FlyMap::synthetic_with_reflex();
    let layout = map.fly_layout();
    let reflex_out = map
        .reflex_out
        .expect("map carries the reflex-readout block");

    let mut blocks: Vec<Vec<f32>> = Vec::new();
    for b in 0..N_BEINGS {
        let mut block = vec![-1.0f32; layout.block_len()];
        match b {
            0 => block[5] = 0.0,
            1 => {
                block[7] = 1.0;
                block[9] = 0.0;
            }
            _ => block[11] = 1.0,
        }
        // Three DIFFERENT genome out_pair choices, deliberately not matching
        // `reflex_out` (which never draws odd/even the same way) — the override below
        // must make all three agree.
        for s in 0..OUT_SLOTS {
            block[OFF_OUT_PAIR + s] = ((s + b) % 2) as f32;
        }
        block[OFF_GAIN] = 1.0 + b as f32 * 0.5;
        block[OFF_SCALE] = 1.0 - b as f32 * 0.25;
        for t in 0..layout.n_types {
            block[OFF_TYPE_MASK + t] = 1.0;
        }
        blocks.push(block);
    }
    let refs: Vec<&[f32]> = blocks.iter().map(|v| v.as_slice()).collect();
    let mut sel = pack_selection(&refs, layout, N_BEINGS);
    let genome_out_pair = sel.out_pair.clone();
    apply_reflex_readout(&mut sel, reflex_out);
    assert_ne!(
        sel.out_pair, genome_out_pair,
        "the synthetic genome blocks were built to disagree with reflex_out; if they \
         already match, this test cannot tell an override from a coincidence"
    );
    for b in 0..N_BEINGS {
        assert_eq!(
            &sel.out_pair[b * OUT_SLOTS..b * OUT_SLOTS + OUT_SLOTS],
            &reflex_out,
            "being {b}: apply_reflex_readout must write the SAME six indices to every being"
        );
    }

    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let buf_wave = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fly-parity-reflex-wave"),
        size: (N_BEINGS * WAVE_U32S * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // The eye binding is off in this test too — the reflex readout and the eye binding
    // are independent extensions, and mixing them would leave this test unable to tell
    // which one produced a given divergence.
    let buf_retina_unused = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fly-parity-reflex-retina-unused"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let gpu = FlyMapGpu::new(
        Arc::clone(&device),
        Arc::clone(&queue),
        &map,
        N_BEINGS,
        TAU_STEPS,
    )
    .expect("the synthetic map fits in any device");
    assert!(gpu.upload_selection(&sel, N_BEINGS), "selection uploaded");
    gpu.reset_state();

    let alpha = gpu.alpha();
    let n = map.n();
    let mut x = vec![0.0f32; n * N_BEINGS];
    let mut r_a = vec![0.0f32; n * N_BEINGS];
    let mut r_b = vec![0.0f32; n * N_BEINGS];

    for tick in 0..N_TICKS {
        let wave = synthetic_wave(tick);
        queue.write_buffer(&buf_wave, 0, bytemuck::cast_slice(&pack_wave(&wave)));

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fly-parity-reflex-tick"),
        });
        gpu.dispatch(&mut enc, N_BEINGS, &buf_wave, false, &buf_retina_unused);
        queue.submit(Some(enc.finish()));
        device.poll(wgpu::Maintain::Wait);
        let gpu_cmd = gpu.read_commands_after_submit(N_BEINGS);
        let gpu_state = gpu.debug_read_state(N_BEINGS);

        cpu_step(&map, &sel, &wave, alpha, &mut x, &r_a, &mut r_b, None);
        std::mem::swap(&mut r_a, &mut r_b);
        let cpu_cmd = cpu_read(&map, &sel, &r_a);

        for j in 0..n {
            for b in 0..N_BEINGS {
                let i = j * N_BEINGS + b;
                assert!(
                    (gpu_state[i] - r_a[i]).abs() < TOL,
                    "reflex readout, tick {tick}, neuron {j}, being {b}: GPU {} vs CPU {}",
                    gpu_state[i],
                    r_a[i]
                );
            }
        }
        for b in 0..N_BEINGS {
            for c in 0..COMMANDS {
                assert!(
                    (gpu_cmd[b][c] - cpu_cmd[b][c]).abs() < TOL,
                    "reflex readout, tick {tick}, being {b}, command {c}: GPU {} vs CPU {}",
                    gpu_cmd[b][c],
                    cpu_cmd[b][c]
                );
            }
        }
    }

    let moved = r_a.iter().filter(|v| v.abs() > 1e-6).count();
    assert!(
        moved > 0,
        "every activation is zero — the reflex-readout run never moved, so the \
         comparison proved nothing"
    );
    let cmd = cpu_read(&map, &sel, &r_a);
    assert!(
        cmd.iter().any(|c| c.iter().any(|v| v.abs() > 1e-6)),
        "every command is zero — the reflex-bound groups had nothing to prove"
    );
}

/// `fly_step` is `@workgroup_size(32, 4)`, and every neuron must be covered.
///
/// This one needs NO GPU, so it runs in the ordinary `cargo test --release --lib`. It is
/// the guard on the pair that can drift apart in silence: the `4` in the kernel's
/// `@workgroup_size` and [`STEP_ROWS`] on the host. Dispatch too few groups and the tail
/// of the connectome never integrates — no error, just neurons frozen at whatever the
/// previous tick left, which is exactly the failure a six-neuron synthetic map in the
/// parity test above would never reach.
#[test]
fn fly_step_dispatch_covers_every_neuron() {
    // The kernel text is the source of truth for the workgroup shape; read it, do not
    // trust the comment next to the constant (CLAUDE.md rule 6).
    let src = include_str!("fly_map.wgsl");
    assert!(
        src.contains("@workgroup_size(32, 4)"),
        "the kernel's workgroup shape changed but STEP_ROWS did not"
    );
    assert_eq!(
        STEP_ROWS, 4,
        "STEP_ROWS must be the 4 of @workgroup_size(32, 4)"
    );

    // 138 639 is the full FlyWire 783 export; the rest are the edges around the split.
    for n in [
        1u32, 3, 4, 5, 6, 127, 128, 138_639, 262_140, 262_141, 1_000_000,
    ] {
        let (gx, gy) = step_groups(n);
        assert!(
            gx <= 65_535 && gy <= 65_535,
            "n={n}: dispatch {gx}x{gy} exceeds the per-dimension cap"
        );
        let covered = u64::from(gx) * u64::from(gy) * u64::from(STEP_ROWS);
        assert!(
            covered >= u64::from(n),
            "n={n}: {gx}x{gy} workgroups cover only {covered} neurons"
        );
    }
    // A map of zero neurons must not ask for zero workgroups — wgpu rejects that.
    assert_eq!(step_groups(0), (1, 1));
}
