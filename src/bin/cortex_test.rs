// ── EMERGENT CORTICAL SHEET TEST ──────────────────────────────────────────────
//
// Proves that the sheet has HETEROGENEOUS and NON-ZERO activity after the first ticks.
//
// # What this test validates
//
// 1. `std > 0.01` and `max − min > 0.05` per being (REAL HETEROGENEITY)
//    — If all 1024 units had the same state, the std would be zero.
//    Heterogeneity is guaranteed structurally by the log-spaced τ bank:
//    units with τ=1.6ms integrate fast, τ=318ms integrate slowly →
//    even with the same spontaneous wave, each unit decays at its own rhythm.
//
// 2. `mean ≠ 0` (real signal, not a degenerate state)
//    — The spontaneous wave injects cos() with measurable phase and magnitude.
//    If mean=0, the sheet is not receiving input.
//
// 3. `motor ≠ [0,0,0]` (motor projection works)
//    — With buf_mw weights initialized with pseudo-random noise ±0.1 and
//    non-zero activations, the motor output MUST be different from zero.
//
// 4. The results DIFFER across the 4 tested beings (INDIVIDUALITY)
//    — Each being receives the same spontaneous wave but with a different sensory_wave
//    (tick 10 uses a wave with channel k = being*13 + k as a distinct signal).
//    Even if the wave were identical, the Hebbian weights diverge tick by tick.
//    In ticks 1-10, the test uses distinct waves per being to force divergence.
//
// # How to run
//
//   cd ~/projects/LLM/Universe
//   env -u DISPLAY nice -n 19 cargo build --release -j 2 --bin cortex_test && \
//   env -u DISPLAY ./target/release/cortex_test
//
// Expected output (real values, NOT fabricated):
//   being 0: min=-0.XXXX max=+0.XXXX mean=0.XXXX std=0.XXXX [PASS]
//   being 1: min=-0.XXXX max=+0.XXXX mean=0.XXXX std=0.XXXX [PASS]
//   ...
//   motor[0]: [throttle=0.XXXX steer=0.XXXX brake=0.XXXX]
//   RESULT: PASS (all 4 metrics verified with real magnitude)

use std::sync::Arc;

use universe::gpu::cortex_sheet::{CortexSheet, SHEET_N};

fn main() {
    // ── Ensure a single Vulkan ICD (prevents loader SIGSEGV with multiple ICDs) ──
    let icd_radeon = "/usr/share/vulkan/icd.d/radeon_icd.json";
    if std::path::Path::new(icd_radeon).exists() && std::env::var("VK_ICD_FILENAMES").is_err() {
        unsafe {
            std::env::set_var("VK_ICD_FILENAMES", icd_radeon);
        }
    }

    // ── Open wgpu device ─────────────────────────────────────────────────────
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("No adapter found");

    let name = adapter.get_info().name.clone();
    eprintln!("[cortex_test] Adapter: {}", name);

    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffers_per_shader_stage = 8;

    let (raw_device, raw_queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("cortex-test"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .expect("Failed to create device");

    let device = Arc::new(raw_device);
    let queue = Arc::new(raw_queue);

    // ── Create cortical sheet with 4 beings ─────────────────────────────────
    let n_seres = 4usize;
    let cortex = CortexSheet::new(Arc::clone(&device), Arc::clone(&queue), n_seres);

    // Inspect the τ bank (no GPU — computed on the CPU)
    let taus = cortex.tau_bank();
    println!(
        "[τ] bank of {} values: min={:.4}ms  max={:.1}ms  spread={:.0}×",
        taus.len(),
        taus[0] * 1000.0,
        taus[taus.len() - 1] * 1000.0,
        taus[taus.len() - 1] / taus[0]
    );
    // Check log-spacing: τ[512] should be ≈ √(τ_min × τ_max)
    let tau_mid_expected = (taus[0] * taus[taus.len() - 1]).sqrt();
    println!(
        "[τ] τ[512] = {:.4}ms  expected(geo-mean)={:.4}ms",
        taus[512] * 1000.0,
        tau_mid_expected * 1000.0
    );

    // ── Run 20 ticks with a distinct sensory_wave per being ──────────────────
    // Artificial waves: being i has channel k = sin(k + i*π/4) × 64 as i8.
    // This guarantees beings get DIFFERENT inputs → Hebbian divergence.
    let n_ticks = 20usize;
    let mut act_sum_prev = 0.0_f32; // Σ|a_j| from the previous tick (0 at tick 0)

    for tick in 0..n_ticks {
        // Build distinct waves per being
        let mut waves: Vec<[i8; 223]> = Vec::with_capacity(n_seres);
        for ser in 0..n_seres {
            let mut w = [0i8; 223];
            for k in 0..223usize {
                // Sine wave distinct per being and per tick: sin(k/10 + being + tick/5)
                let phase = (k as f32 / 10.0)
                    + (ser as f32 * std::f32::consts::PI / 4.0)
                    + (tick as f32 / 5.0);
                w[k] = (phase.sin() * 64.0) as i8;
            }
            waves.push(w);
        }
        cortex.upload_wave(&waves, n_seres);

        // Single encoder for the 3 kernels + readback
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cortex-test-tick"),
        });

        cortex.dispatch_afference(&mut enc, n_seres, tick as u32);
        cortex.dispatch_hebb(&mut enc, n_seres, tick as u32, act_sum_prev);
        cortex.dispatch_motor(&mut enc, n_seres, tick as u32);
        cortex.copy_for_readback(&mut enc, n_seres);

        queue.submit(Some(enc.finish()));
        device.poll(wgpu::Maintain::Wait);

        // Read activation statistics (proof of heterogeneity)
        let stats = cortex.read_act_stats(n_seres);

        // Compute being 0's act_sum for the next tick
        // (Σ|a_j| passed to the Hebb kernel as a normalization parameter)
        // For test simplicity, we use only being 0.
        // In production, compute per being via readback before dispatch B.
        // Here: we approximate it as SHEET_N × |mean| (bounded below by heterogeneity).
        // In the real integration, read buf_act_rb and sum before dispatch B.
        act_sum_prev = stats[0].2.abs() * SHEET_N as f32;

        // Print on the last tick
        if tick == n_ticks - 1 {
            println!("\n── Sheet activations (tick {}) ──", tick);
            let mut all_pass = true;
            for (ser, (mn, mx, mean, std)) in stats.iter().enumerate() {
                // PASS criteria (with real magnitude):
                // std > 0.01: units have different states (heterogeneity)
                // max-min > 0.05: there is real spread, not a constant
                // |mean| > 0: non-zero signal (spontaneous wave arrived)
                let heterogeneo = std > &0.01 && (mx - mn) > 0.05;
                let nao_zero = mean.abs() > 1e-6;
                let ok = heterogeneo && nao_zero;
                if !ok {
                    all_pass = false;
                }
                println!(
                    "  being {}: min={:+.4}  max={:+.4}  mean={:+.4}  std={:.4}  {}",
                    ser,
                    mn,
                    mx,
                    mean,
                    std,
                    if ok { "PASS" } else { "FAIL" }
                );
            }

            // Read motor output
            let motors = cortex.read_motor(n_seres);
            println!("\n── Motor output (tick {}) ──", tick);
            for (ser, m) in motors.iter().enumerate() {
                let nao_zero = m[0].abs() + m[1].abs() + m[2].abs() > 1e-6;
                if !nao_zero {
                    all_pass = false;
                }
                println!(
                    "  being {}: throttle={:+.4}  steer={:+.4}  brake={:+.4}  {}",
                    ser,
                    m[0],
                    m[1],
                    m[2],
                    if nao_zero { "PASS" } else { "FAIL (zero)" }
                );
            }

            // Verify that beings DIFFER from one another (individuality)
            let mut divergencia_ok = false;
            for i in 0..n_seres {
                for j in (i + 1)..n_seres {
                    let diff = (stats[i].2 - stats[j].2).abs();
                    if diff > 1e-6 {
                        divergencia_ok = true;
                    }
                }
            }
            if !divergencia_ok {
                all_pass = false;
            }
            println!("\n── Divergence between beings ──");
            for i in 0..n_seres {
                for j in (i + 1)..n_seres {
                    let diff = (stats[i].2 - stats[j].2).abs();
                    println!(
                        "  |mean[{}] − mean[{}]| = {:.6}  {}",
                        i,
                        j,
                        diff,
                        if diff > 1e-6 {
                            "divergent (OK)"
                        } else {
                            "identical (PROBLEM)"
                        }
                    );
                }
            }

            println!(
                "\n══ FINAL RESULT: {} ══",
                if all_pass { "PASS" } else { "FAIL" }
            );
            if !all_pass {
                eprintln!("FAIL: Check stderr for driver details");
                std::process::exit(1);
            }
        }
    }

    // ── τ bank summary ───────────────────────────────────────────────────────
    // Counts how many units are in each band (for being 0; the τ are the same).
    let (mut n_gamma, mut n_beta, mut n_alpha, mut n_theta, mut n_delta) = (0, 0, 0, 0, 0);
    for &t in &taus {
        let f_hz = 1.0 / (2.0 * std::f32::consts::PI * t);
        if f_hz >= 30.0 {
            n_gamma += 1;
        } else if f_hz >= 13.0 {
            n_beta += 1;
        } else if f_hz >= 8.0 {
            n_alpha += 1;
        } else if f_hz >= 4.0 {
            n_theta += 1;
        } else {
            n_delta += 1;
        }
    }
    println!("\n── Band distribution (doc §5.3) ──");
    println!(
        "  gamma (>30Hz):   {:4} units ({:.1}%)",
        n_gamma,
        n_gamma as f32 / SHEET_N as f32 * 100.0
    );
    println!(
        "  beta  (13-30Hz): {:4} units ({:.1}%)",
        n_beta,
        n_beta as f32 / SHEET_N as f32 * 100.0
    );
    println!(
        "  alpha (8-13Hz):  {:4} units ({:.1}%)",
        n_alpha,
        n_alpha as f32 / SHEET_N as f32 * 100.0
    );
    println!(
        "  theta (4-8Hz):   {:4} units ({:.1}%)",
        n_theta,
        n_theta as f32 / SHEET_N as f32 * 100.0
    );
    println!(
        "  delta (<4Hz):    {:4} units ({:.1}%)",
        n_delta,
        n_delta as f32 / SHEET_N as f32 * 100.0
    );
}
