//! CPU × GPU equivalence of the SLOW BRAIN.
//!
//! The agent that wrote `src/gpu/slow_brain.wgsl` could not run on the GPU, so the kernel
//! is COMPILED and UNVERIFIED. This project has already seen five reports of fabricated
//! proof — including a validation table from a run that segfaulted and a test that
//! compared zero against zero. So the rule here is: the number has to come out of a real
//! execution, and the test has to be able to FAIL.
//!
//! Two specific traps this file avoids on purpose:
//!
//!  1. **Comparing zero against zero.** If the weights are born zeroed, both sides give 0
//!     and the equivalence is vacuous. Here the weights come from `rng::Lcg::symmetric()`
//!     and the program ABORTS if the CPU output has no variation — without signal, there
//!     is nothing to validate.
//!  2. **Testing a single tick.** The slow brain runs 1 time every 8 ticks and has
//!     recurrent state (CfC + fatigue + counter). A single tick exercises neither the
//!     subsample nor the memory. We run hundreds of ticks with a varying input.
//!
//! One single `wgpu::Device` in the process: opening two on the same device has already
//! locked up here.

use std::sync::Arc;
use universe::engine::SlowBrain;
use universe::gpu::slow_brain::SlowBrainGpu;
use universe::rng::Lcg;

const N_BEINGS: usize = 8;
const TICKS: usize = 400;
const DT: f32 = 1.0 / 64.0;

fn main() {
    // ── Compute-only device, no window ──────────────────────────────────────
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        ..Default::default()
    });
    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    })) {
        Some(a) => a,
        None => {
            eprintln!("no GPU adapter");
            std::process::exit(2);
        }
    };
    println!("GPU: {}", adapter.get_info().name);

    let (device, queue) = match pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("slow-equiv"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    )) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("request_device failed: {e:?}");
            std::process::exit(2);
        }
    };
    let device: wgpu::Device = device;
    let queue: wgpu::Queue = queue;
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    // ── Weights: a DIFFERENT set per being ──────────────────────────────────
    // Beings with distinct weights catch per-being indexing errors (`base = i * stride`),
    // which is exactly the bug that has already shown up here: `fat_base = being * 96` in
    // a 192-slot buffer made being N read the slots of being N−1.
    let mut cpus: Vec<SlowBrain> = Vec::with_capacity(N_BEINGS);
    let mut all_weights: Vec<f32> = Vec::new();
    for b in 0..N_BEINGS {
        let mut brain = SlowBrain::new();
        let mut rng = Lcg::new(0x5107_0000 + b as u64);
        let n = brain.all_weights().len();
        let w = rng.weights(n, 0.35);
        brain.set_all_weights(&w);
        all_weights.extend_from_slice(&w);
        cpus.push(brain);
    }
    println!(
        "{} beings · {} weights/being · {} ticks",
        N_BEINGS,
        all_weights.len() / N_BEINGS,
        TICKS
    );

    let gpu = SlowBrainGpu::new_shared(device.clone(), queue.clone(), N_BEINGS);
    gpu.upload_weights(&all_weights, N_BEINGS);

    // ── Input: varies over time AND across beings ───────────────────────────
    let make_input = |t: usize, b: usize| -> [i8; 48] {
        let mut inp = [0i8; 48];
        for (i, slot) in inp.iter_mut().enumerate() {
            let period = 23.0 + (i as f32) * 5.0 + (b as f32) * 3.0;
            *slot = (((t as f32) / period * std::f32::consts::TAU).sin() * 110.0) as i8;
        }
        inp
    };

    let mut worst_abs = 0.0f32;
    let mut worst_tick = 0usize;
    let mut sum_abs = 0.0f64;
    let mut n_comparisons = 0usize;
    let mut cpu_samples: Vec<[f32; 3]> = Vec::new();

    for t in 0..TICKS {
        let inputs: Vec<[i8; 48]> = (0..N_BEINGS).map(|b| make_input(t, b)).collect();

        for (b, brain) in cpus.iter_mut().enumerate() {
            brain.tick(&inputs[b], DT);
        }
        let gpu_out = gpu.forward(&inputs, N_BEINGS, DT);

        if gpu_out.len() != N_BEINGS {
            eprintln!(
                "GPU returned {} outputs, expected {}",
                gpu_out.len(),
                N_BEINGS
            );
            std::process::exit(1);
        }

        for b in 0..N_BEINGS {
            for k in 0..3 {
                let d = (cpus[b].output[k] - gpu_out[b][k]).abs();
                sum_abs += d as f64;
                n_comparisons += 1;
                if d > worst_abs {
                    worst_abs = d;
                    worst_tick = t;
                }
            }
        }
        cpu_samples.push(cpus[0].output);

        if t % 80 == 0 {
            println!(
                "  t={:3}  cpu[0] = {:+.6} {:+.6} {:+.6}   gpu[0] = {:+.6} {:+.6} {:+.6}",
                t,
                cpus[0].output[0],
                cpus[0].output[1],
                cpus[0].output[2],
                gpu_out[0][0],
                gpu_out[0][1],
                gpu_out[0][2],
            );
        }
    }

    // ── Guard against vacuous validation ────────────────────────────────────
    // If the CPU produced no variation, comparing against the GPU proves nothing.
    let mut max_amplitude = 0.0f32;
    for k in 0..3 {
        let mx = cpu_samples.iter().map(|o| o[k]).fold(f32::MIN, f32::max);
        let mn = cpu_samples.iter().map(|o| o[k]).fold(f32::MAX, f32::min);
        max_amplitude = max_amplitude.max(mx - mn);
    }
    println!("\nCPU signal amplitude: {max_amplitude:.6}");
    if max_amplitude < 1e-3 {
        eprintln!("ABORT: the CPU produced no signal — the comparison would be vacuous (0 vs 0).");
        std::process::exit(1);
    }

    let mean = sum_abs / n_comparisons as f64;
    println!("comparisons: {n_comparisons}");
    println!("mean absolute error: {mean:.3e}");
    println!("worst absolute error:  {worst_abs:.3e}  (tick {worst_tick})");

    // Tolerance: the expected difference is only from f32 operation ordering across sums
    // of 64 and 16 terms, with tanh/exp on both sides. 1e-4 is loose for that and tight
    // enough to catch an index, layout or equation error.
    const TOL: f32 = 1e-4;
    if worst_abs <= TOL {
        println!("\nEQUIVALENT — worst error {worst_abs:.3e} ≤ {TOL:.0e}");
    } else {
        println!("\nDIVERGENT — worst error {worst_abs:.3e} > {TOL:.0e}");
        std::process::exit(1);
    }
}
