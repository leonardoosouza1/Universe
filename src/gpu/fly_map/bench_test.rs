//! Wall-clock cost of `fly_step` on the REAL connectome, per tick, per population size.
//!
//! # Why `#[ignore]`
//!
//! Same reason as `parity_test.rs`: it opens a real `wgpu::Device`. It also needs the
//! exported map on disk, which is 122 MB and lives on the data partition, not in the
//! repository. Run it explicitly:
//!
//! ```text
//! HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0 \
//!   cargo test --release --lib -- --ignored --nocapture fly_map_bench
//! ```
//!
//! # What it measures
//!
//! `dispatch` (both kernels) submitted alone and waited on, 50 ticks, MEDIAN ms/tick. The
//! median and not the mean because the GPU is shared: one evolution run scheduling a batch
//! in the middle of the sample moves the mean and leaves the median alone.
//!
//! The state is recurrent and stays resident, so tick 50 does exactly the work tick 1 did
//! — no warm-up artefact beyond the first dispatch, which is dropped as the pipeline is
//! still being compiled there.
//!
//! `FLY_MAP_BENCH` overrides the map path, which is how the `--min-weight 3` and
//! `_w3_nooptic` exports get measured with the same code.

use std::sync::Arc;
use std::time::Instant;

use super::FlyMapGpu;
use super::parity_test::open_device;
use crate::fly_map::{FlyMap, pack_selection, random_block};
use crate::gpu::perception::WAVE_U32S;

/// The full 783 export: n = 138 639 neurons, m = 15.1 M edges.
const DEFAULT_MAP: &str = "/media/llm-data/datasets/flywire/flymap_783.bin";
/// Populations the run path actually uses: the smoke size and a real generation.
const POPULATIONS: [usize; 2] = [8, 40];
const N_TICKS: usize = 50;
const TAU_STEPS: u32 = 4;

#[test]
#[ignore = "opens a real GPU and needs the exported map — run manually: cargo test --release --lib -- --ignored --nocapture fly_map_bench"]
fn fly_map_bench_step_cost_per_tick() {
    let path = std::env::var("FLY_MAP_BENCH").unwrap_or_else(|_| DEFAULT_MAP.to_string());
    if !std::path::Path::new(&path).exists() {
        eprintln!("[fly_map_bench] map not found at {path} — benchmark skipped");
        return;
    }
    let Some((device, queue)) = pollster::block_on(open_device()) else {
        eprintln!("[fly_map_bench] no GPU available — benchmark skipped");
        return;
    };
    let map = match FlyMap::load(&path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[fly_map_bench] cannot read {path}: {e} — benchmark skipped");
            return;
        }
    };
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    println!(
        "[fly_map_bench] {path}: n={} m={} types={} k_in={} k_out={}",
        map.n(),
        map.m(),
        map.n_types,
        map.k_in,
        map.k_out()
    );

    for b in POPULATIONS {
        match bench_one(&device, &queue, &map, b) {
            Some((per_submit, batched)) => println!(
                "[fly_map_bench] B={b:<3} median {per_submit:.2} ms/tick per submit, \
                 {batched:.2} ms/tick batched ({:.3} ms per being)",
                batched / b as f64
            ),
            None => println!("[fly_map_bench] B={b:<3} skipped (map does not fit)"),
        }
    }
}

/// One population size: build the buffers, run `N_TICKS`, return
/// `(median ms/tick submitted one at a time, ms/tick with all N_TICKS in one submit)`.
///
/// The two differ by the cost of getting a submission scheduled at all, which on a device
/// already saturated by another run is a floor of its own — and a floor that does not
/// shrink when the kernel does. The batched figure divides one submit across `N_TICKS`, so
/// that floor is paid once and what is left is the kernel. Report both: the per-submit
/// number is what a tick of `gpu_evolve` actually waits for, the batched one is what the
/// kernel costs.
fn bench_one(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    map: &FlyMap,
    n_beings: usize,
) -> Option<(f64, f64)> {
    let layout = map.fly_layout();
    // A random selection per being, so the per-being tables are genuinely different and
    // the kernel cannot short-circuit on a uniform population.
    let blocks: Vec<Vec<f32>> = (0..n_beings)
        .map(|b| random_block(0xF1Eu64.wrapping_add(b as u64 * 7919), layout))
        .collect();
    let refs: Vec<&[f32]> = blocks.iter().map(|v| v.as_slice()).collect();
    let sel = pack_selection(&refs, layout, n_beings);

    let gpu = FlyMapGpu::new(
        Arc::clone(device),
        Arc::clone(queue),
        map,
        n_beings,
        TAU_STEPS,
    )?;
    assert!(gpu.upload_selection(&sel, n_beings), "selection uploaded");
    gpu.reset_state();

    // A non-zero wave, written once: the cost of `fly_step` does not depend on the value of
    // the input, and re-uploading it every tick would time the upload instead.
    let words: Vec<u32> = (0..n_beings * WAVE_U32S)
        .map(|i| (i as u32).wrapping_mul(2654435761))
        .collect();
    let buf_wave = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fly-bench-wave"),
        size: (n_beings * WAVE_U32S * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf_wave, 0, bytemuck::cast_slice(&words));
    // This benchmark measures the existing hot path (`--fly-brain`), not the eye binding
    // (`--fly-eye`) — `eye_on: false` and an unread buffer, same as
    // `parity_test::fly_map_parity_gpu_matches_the_cpu_reference`'s `buf_retina_unused`.
    let buf_retina_unused = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fly-bench-retina-unused"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let mut samples: Vec<f64> = Vec::with_capacity(N_TICKS);
    for tick in 0..=N_TICKS {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fly-bench-tick"),
        });
        gpu.dispatch(&mut enc, n_beings, &buf_wave, false, &buf_retina_unused);
        let t0 = Instant::now();
        queue.submit(Some(enc.finish()));
        device.poll(wgpu::Maintain::Wait);
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        // Tick 0 pays for pipeline compilation and the first buffer residency fault.
        if tick > 0 {
            samples.push(dt);
        }
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let per_submit = samples[samples.len() / 2];

    // The same work again, but recorded as ONE encoder. The state is recurrent, so the
    // ticks still chain exactly as above — only the number of submissions changes.
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("fly-bench-batch"),
    });
    for _ in 0..N_TICKS {
        gpu.dispatch(&mut enc, n_beings, &buf_wave, false, &buf_retina_unused);
    }
    let t0 = Instant::now();
    queue.submit(Some(enc.finish()));
    device.poll(wgpu::Maintain::Wait);
    let batched = t0.elapsed().as_secs_f64() * 1000.0 / N_TICKS as f64;

    Some((per_submit, batched))
}
