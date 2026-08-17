//! CPU × GPU parity test of the perception CfC (`retina_digest.wgsl` vs
//! `iara_cfc::CfcCell::step`, via `RetinaPerception`).
//!
//! # Why `#[ignore]`
//!
//! This project does not run real GPU compute inside `cargo test` (grep for
//! `RetinaGpu::new`/`PerceptionGpu::new` inside `#[cfg(test)]` in the rest of the crate
//! finds none) — the kernels are validated by manual binaries (`gpu_validate`,
//! `pista_check`). This test would be the first to open a `wgpu::Device` inside
//! `cargo test`, which would break on machines/CI without a GPU. It stays `#[ignore]` so
//! it does not block `cargo test --release --lib`; run explicitly with:
//!
//! ```text
//! HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0 \
//!   cargo test --release --lib -- --ignored retina_digest_parity
//! ```
//!
//! # What it proves
//!
//! Runs `RetinaDigestGpu` for N ticks with synthetic retinas (varied, including
//! a wall in the central columns) and compares, tick by tick:
//! - ch192-207 (the CfC channel) against the same formula computed on the CPU from the
//!   PREVIOUS hidden state (preserves the 1-tick lag documented in the kernel);
//! - `min_wall_ahead`/`edge_row6`/`edge_row7` against the logic that existed in
//!   `context.rs` before this migration (reproduced here only as a test oracle —
//!   it is not a second production copy, it is the "before" against the "after").
//!
//! The recurrent state h itself (CPU's `pipeline.state.h` vs GPU's `cfc_state`) is
//! validated INDIRECTLY: any divergence in the recurrence shows up in the channels of the
//! NEXT tick, because this iteration's `h_new` becomes the "old h" read in the next
//! dispatch — a test of several chained ticks catches accumulated divergence that a
//! single-tick test would not.

use super::RetinaDigestGpu;
use crate::config::iara::vision::{RETINA_COLS, RETINA_TOTAL};
use crate::gpu::perception::WAVE_U32S;
use crate::gpu::{Pixel, legacy_encoding};
use crate::perception::RetinaPerception;

const N_BEINGS: usize = 2;
const N_TICKS: usize = 4;

/// Generates a deterministic synthetic retina (packed `dist_mm<<8|material`), varying
/// by `(being, tick)` to exercise the CfC with different inputs tick by tick — without
/// that, `h` would converge trivially and the test would not catch recurrence errors.
/// Includes a wall (material=2) in the central columns [42,84) in some pixels to
/// exercise the `min_wall_ahead` path.
fn synthetic_retina(being: usize, tick: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(RETINA_TOTAL);
    let seed = (being as u32)
        .wrapping_mul(97)
        .wrapping_add((tick as u32).wrapping_mul(131));
    for j in 0..RETINA_TOTAL {
        let jx = j as u32;
        let h = jx
            .wrapping_mul(2654435761)
            .wrapping_add(seed)
            .wrapping_mul(0x9E3779B1);
        let dist_mm = (h % 60_000) as u32; // 0..60 m
        let col = j % RETINA_COLS;
        // Forces a wall in the central columns in ~1/4 of the pixels of that band, close
        // (< 20 m), so min_wall_ahead has a real value to compare.
        let material: u32 = if col >= RETINA_COLS / 3 && col < 2 * RETINA_COLS / 3 && (h % 4) == 0 {
            2
        } else {
            h % 4
        };
        let dist_mm = if material == 2 {
            dist_mm % 20_000
        } else {
            dist_mm
        };
        out.push((dist_mm << 8) | material);
    }
    out
}

/// Test oracle: reproduces the logic that `context.rs::tick()` did BEFORE this
/// migration (min_wall_ahead + edge_row6/7 from the raw physical retina). It is not
/// production code — it exists only so the test has a "before" independent of the WGSL
/// to compare against the "after".
fn cpu_wall_and_edges(retina_packed: &[u32]) -> (f32, f32, f32) {
    let mut min_wall_ahead = f32::MAX;
    let mut retina_i8 = vec![0i8; RETINA_TOTAL];
    for (j, &packed) in retina_packed.iter().enumerate() {
        let dist_m = (packed >> 8) as f32 / 1000.0;
        let material = (packed & 0xFF) as u8;
        if material == 2 {
            let col = j % RETINA_COLS;
            if col >= RETINA_COLS / 3 && col < 2 * RETINA_COLS / 3 {
                min_wall_ahead = min_wall_ahead.min(dist_m);
            }
        }
        retina_i8[j] = legacy_encoding(Pixel { dist_m, material });
    }
    let mut edge = [0f32; 2];
    for (k, &row_idx) in [6usize, 7usize].iter().enumerate() {
        let edge_sum: i32 = (0..RETINA_COLS - 1)
            .map(|col| {
                let a = retina_i8[row_idx * RETINA_COLS + col] as i32;
                let b = retina_i8[row_idx * RETINA_COLS + col + 1] as i32;
                (a - b).abs()
            })
            .sum();
        edge[k] = (edge_sum as f32 / 6000.0).clamp(0.0, 1.0);
    }
    (min_wall_ahead, edge[0], edge[1])
}

/// Test oracle: the same hidden→channel conversion formula copied in
/// `retina_digest.wgsl` (approximation x/(1+|x|), NOT exact tanh — see the kernel doc).
fn channel_from_hidden(h: &[f32]) -> [i8; 16] {
    let mut out = [0i8; 16];
    for (i, &hi) in h.iter().enumerate() {
        let abs_h = hi.abs();
        let tanh_h = if abs_h > 10.0 {
            hi.signum() * 0.999
        } else {
            hi / (1.0 + abs_h)
        };
        let val = (tanh_h * 127.0 * 0.3).clamp(-127.0, 127.0);
        out[i] = val as i8;
    }
    out
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
    limits.max_storage_buffers_per_shader_stage = 8;
    adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("retina-digest-parity-test"),
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
#[ignore = "opens a real GPU — run manually: cargo test --release --lib -- --ignored retina_digest_parity"]
fn retina_digest_bate_com_cpu_ao_longo_de_varios_ticks() {
    let Some((device, queue)) = pollster::block_on(open_device()) else {
        eprintln!("[retina_digest_parity] no GPU available — test skipped");
        return;
    };

    let digest = RetinaDigestGpu::new(&device, &queue, N_BEINGS);

    let buf_retina = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test-retina"),
        size: (N_BEINGS * RETINA_TOTAL * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // P1.8: `output` (buf_output_shared) remains a mandatory parameter of
    // `bind_and_dispatch` — the bind group layout is shared with
    // `perception.rs` — but the kernel NO longer writes into its ch192-207 (finding
    // "ch192-207 aliases into the NEXT being", fixed: the 16 values now go
    // to `extra_out[3..19]`, see the comparison below). `buf_output` only needs
    // to exist and be the right size so the bind group does not fail — there is no
    // corruption read to do here anymore, so no readback of it is
    // needed in this test.
    let bytes_output = (N_BEINGS * WAVE_U32S * 4) as u64;
    let buf_output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test-output"),
        size: bytes_output,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // CPU oracle: same fixed seed that RetinaDigestGpu::new() uses internally to
    // upload the weights — both sides start from the SAME CfC weights.
    let mut cpu_percept: Vec<RetinaPerception> =
        (0..N_BEINGS).map(|_| RetinaPerception::new()).collect();

    for tick in 0..N_TICKS {
        let mut retina_data = Vec::with_capacity(N_BEINGS * RETINA_TOTAL);
        let mut retinas_per_being = Vec::with_capacity(N_BEINGS);
        for being in 0..N_BEINGS {
            let r = synthetic_retina(being, tick);
            retina_data.extend_from_slice(&r);
            retinas_per_being.push(r);
        }
        queue.write_buffer(&buf_retina, 0, bytemuck::cast_slice(&retina_data));

        // Expected channel BEFORE advancing the CfC — the kernel writes with the "old" h.
        let expected_channels: Vec<[i8; 16]> = cpu_percept
            .iter()
            .map(|p| channel_from_hidden(&p.pipeline.state.h))
            .collect();

        let mut enc = device.create_command_encoder(&Default::default());
        enc.clear_buffer(&buf_output, 0, Some(bytes_output));
        let bytes_extra = digest.bind_and_dispatch(
            &mut enc,
            &device,
            &queue,
            N_BEINGS,
            &buf_retina,
            &buf_output,
        );
        queue.submit(Some(enc.finish()));
        device.poll(wgpu::Maintain::Wait);

        // ── Reads extra_out: min_wall_ahead/edge_row6/edge_row7 (slots 0-2) and the
        // squashed CfC state "ch192-207" (slots 3-18, finding P1.8 — it stopped
        // aliasing into `buf_output` and now lives here, its own space per being) ──
        let extra = digest.read_after_submit(&device, N_BEINGS, bytes_extra);
        for being in 0..N_BEINGS {
            let (want_wall, want_e6, want_e7) = cpu_wall_and_edges(&retinas_per_being[being]);
            let row = extra[being];
            let (got_wall, got_e6, got_e7) = (row[0], row[1], row[2]);
            // min_wall_ahead: if no wall qualified, the two sides use LARGE but
            // DIFFERENT sentinels (f32::MAX vs 1e9) — we only compare
            // the exact value when a real wall was found.
            if want_wall < 1.0e8 {
                assert!(
                    (got_wall - want_wall).abs() < 0.01,
                    "tick={tick} being={being} min_wall_ahead: GPU={got_wall} CPU={want_wall}"
                );
            } else {
                assert!(
                    got_wall > 1.0e8,
                    "tick={tick} being={being} expected sentinel, GPU={got_wall}"
                );
            }
            assert!(
                (got_e6 - want_e6).abs() < 0.01,
                "tick={tick} being={being} edge_row6: GPU={got_e6} CPU={want_e6}"
            );
            assert!(
                (got_e7 - want_e7).abs() < 0.01,
                "tick={tick} being={being} edge_row7: GPU={got_e7} CPU={want_e7}"
            );
            for ci in 0..16usize {
                let got = row[3 + ci];
                let want = expected_channels[being][ci] as f32;
                assert!(
                    (got - want).abs() <= 1.0,
                    "tick={tick} being={being} ch192+{ci} (now extra_out[{}]): GPU={got} CPU={want} (h_old={:?})",
                    3 + ci,
                    cpu_percept[being].pipeline.state.h
                );
            }
        }

        // ── Advances the CPU's CfC to the next tick (same retina the GPU saw) ──
        for being in 0..N_BEINGS {
            let mut retina_i8 = vec![0i8; RETINA_TOTAL];
            for (j, &packed) in retinas_per_being[being].iter().enumerate() {
                let dist_m = (packed >> 8) as f32 / 1000.0;
                let material = (packed & 0xFF) as u8;
                retina_i8[j] = legacy_encoding(Pixel { dist_m, material });
            }
            cpu_percept[being].perceive(&retina_i8);
        }
    }
}
