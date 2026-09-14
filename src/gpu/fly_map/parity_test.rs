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

use super::FlyMapGpu;
use crate::fly_map::{
    COMMANDS, FlyMap, IN_CHANNELS, OFF_GAIN, OFF_OUT_PAIR, OFF_SCALE, OFF_TYPE_MASK, OUT_SLOTS,
    cpu_read, cpu_step, pack_selection,
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

async fn open_device() -> Option<(wgpu::Device, wgpu::Queue)> {
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
    // `fly_step` binds 12 storage buffers; the downlevel default is 4. `gpu_evolve`'s
    // device asks for the same 16 (`context.rs`).
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
        gpu.dispatch(&mut enc, N_BEINGS, &buf_wave);
        queue.submit(Some(enc.finish()));
        device.poll(wgpu::Maintain::Wait);
        let gpu_cmd = gpu.read_commands_after_submit(N_BEINGS);
        let gpu_state = gpu.debug_read_state(N_BEINGS);

        cpu_step(&map, &sel, &wave, alpha, &mut x, &r_a, &mut r_b);
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
