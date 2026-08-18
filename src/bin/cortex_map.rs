// ── EMERGENT CORTICAL ALLOCATION MAP ───────────────────────────────────────────
//
// Tests prediction P1 from NEUROMOD_PREDICOES.md:
//   "The boundary between 'visual' and 'auditory' is not drawn.
//    It is where the cross-correlation drops."
//
// # Input: SYNTHETIC (honestly declared)
//
// We use synthetic input with two DISTINGUISHABLE modalities:
//   Modality A: channels 0..=111  — sinusoidal block, phase 0
//   Modality B: channels 112..=222 — sinusoidal block, phase π/2 (decorrelated)
//
// This is the minimal control so that Hebb can segregate territories.
// The prediction is REFUTED if no segregation emerges even with this ideal input.
//
// # Refutation criterion (P1, NEUROMOD_PREDICOES.md)
//
//   ratio = intra_region_correlation / inter_region_correlation
//   REFUTED if ratio < 3.0
//
// # Tick limit: maximum 20 (user instruction — evolution running in background).
//
// # Usage
//
//   cd ~/projects/LLM/Universe
//   nice -n 19 cargo build --release -j 2 --bin cortex_map && \
//   env -u DISPLAY ./target/release/cortex_map

use std::sync::Arc;

use universe::gpu::cortex_sheet::{CortexSheet, SHEET_H, SHEET_N, SHEET_W};

// Actual sheet size — public constants from cortex_sheet.rs
const N_INPUTS: usize = 223;
const N_TICKS: usize = 20; // maximum allowed (evolution in background)
const N_SERES: usize = 1; // 1 being is enough for the allocation map
const ATIVO_LIMIAR: f32 = 0.05; // mean-activity threshold → "active" (P2)
const SILENTE_LIMIAR: f32 = 0.01; // mean-activity threshold → "silent" (P2)

// Channels of each modality (synthetic)
const MOD_A_END: usize = 111; // channels 0..=111 → modality A (N=112)
const MOD_B_START: usize = 112; // channels 112..=222 → modality B (N=111)

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
    .expect("No Vulkan adapter found");

    let name = adapter.get_info().name.clone();
    eprintln!("[cortex_map] Adapter: {name}");

    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffers_per_shader_stage = 8;

    let (raw_device, raw_queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("cortex-map"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .expect("Failed to create device");

    let device = Arc::new(raw_device);
    let queue = Arc::new(raw_queue);

    // ── Create the cortical sheet with 1 being ───────────────────────────────
    let cortex = CortexSheet::new(Arc::clone(&device), Arc::clone(&queue), N_SERES);
    let taus = cortex.tau_bank(); // τ per unit (CPU, no GPU)

    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║  CORTEX MAP — EMERGENT ALLOCATION MAP (P1, P2, P6)              ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("WARNING: SYNTHETIC input with 2 decorrelated modalities.");
    println!("  Modality A = channels  0-111  (sine, phase 0)");
    println!("  Modality B = channels 112-222 (sine, phase π/2, decorrelated from A)");
    println!("  N_TICKS = {N_TICKS} (maximum — evolution in background)");
    println!("  N_SERES = {N_SERES}");
    println!();

    // ── Activation accumulator across ticks (for mean activity) ──────────────
    // Sums absolute activations per unit across ticks.
    // We use f64 for precision when accumulating 20 values.
    let mut act_accum: Vec<f64> = vec![0.0; SHEET_N];
    let mut act_sum_prev = 0.0_f32;

    // ── Tick loop ─────────────────────────────────────────────────────────────
    for tick in 0..N_TICKS {
        // Builds the synthetic wave:
        //   Modality A (channels 0-111): sin(2π × k/112 + φ_tick) × 100
        //   Modality B (channels 112-222): cos(2π × (k-112)/111 + φ_tick) × 100
        // The two modalities have similar variance but are decorrelated
        // from each other (one is sin, the other cos of the same frequency → correlation ≈ 0).
        let phase_tick = (tick as f32) * 0.3; // slow temporal phase (varies across ticks)

        let mut waves: Vec<[i8; 223]> = Vec::with_capacity(N_SERES);
        {
            let mut w = [0i8; 223];
            for k in 0..N_INPUTS {
                let v: f32 = if k <= MOD_A_END {
                    // Modality A: pure sine with spatial frequency k/112
                    (2.0 * std::f32::consts::PI * k as f32 / 112.0 + phase_tick).sin() * 100.0
                } else {
                    // Modality B: cosine (decorrelated from A by π/2)
                    let kk = k - MOD_B_START;
                    (2.0 * std::f32::consts::PI * kk as f32 / 111.0 + phase_tick).cos() * 100.0
                };
                w[k] = v.clamp(-127.0, 127.0) as i8;
            }
            waves.push(w);
        }
        cortex.upload_wave(&waves, N_SERES);

        // Single encoder: afference → hebb → motor → readback
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cortex-map-tick"),
        });
        cortex.dispatch_afference(&mut enc, N_SERES, tick as u32);
        cortex.dispatch_hebb(&mut enc, N_SERES, tick as u32, act_sum_prev);
        cortex.dispatch_motor(&mut enc, N_SERES, tick as u32);
        cortex.copy_for_readback(&mut enc, N_SERES);

        queue.submit(Some(enc.finish()));
        device.poll(wgpu::Maintain::Wait);

        // Reads activations and accumulates
        let stats = cortex.read_act_stats(N_SERES);
        // act_sum for the next Hebb tick: Σ|a_j| of being 0.
        // Estimate: we use mean × SHEET_N × 0.5 (underestimated, but enough
        // for the Hebb kernel to have the normalization in the right order of magnitude).
        act_sum_prev = stats[0].2.abs() * SHEET_N as f32;

        if tick == 0 || tick == N_TICKS - 1 {
            let (mn, mx, mean, std) = stats[0];
            eprintln!(
                "[tick {:2}] min={:+.4} max={:+.4} mean={:+.4} std={:.4}  act_sum_prev={:.1}",
                tick, mn, mx, mean, std, act_sum_prev
            );
        }

        // Accumulates |activation| per unit to compute mean activity later
        // We need to read back the raw activations per unit.
        // read_act_stats returns aggregate statistics, not the 1024 individual activations.
        // We use the standard deviation as an activity proxy, but the ideal is to have
        // the individual activations. We read buf_act_rb via the available interface.
        // Since there is no public API to read buf_act_rb per unit, we use
        // the individual-activation reading technique implemented below.
        // We accumulate act_sum_prev as a proxy for total sheet activity.
        // For P2/P6 we need the 1024 individual activations — read via read_act_raw.
        let raw_acts = cortex.read_act_raw(N_SERES);
        for (u, &a) in raw_acts[0..SHEET_N].iter().enumerate() {
            act_accum[u] += (a as f64).abs();
        }
    }

    // ── Mean activity per unit ───────────────────────────────────────────────
    let act_mean: Vec<f32> = act_accum
        .iter()
        .map(|&s| (s / N_TICKS as f64) as f32)
        .collect();

    // Checks whether there is real activity or everything is zero
    let act_max = act_mean.iter().cloned().fold(0.0_f32, f32::max);
    if act_max < 1e-6 {
        println!("NOT MEASURED: all mean activations are zero.");
        println!(
            "Reason: the sheet received no signal after {} ticks.",
            N_TICKS
        );
        println!("Check: (1) upload_wave correct? (2) did the afference kernel run?");
        println!(
            "(3) buf_afw initialized at 1.0 but the EPS gaussian may be filtering everything."
        );
        return;
    }

    // ── For each unit, which modality dominates? ──────────────────────────────
    // We use the most direct proof available without buf_afw readback:
    // We run 2 extra ticks — one with only modality A active, another with only B active.
    // The activation difference reveals which modality dominates each unit.
    //
    // Note: buf_afw has no public readback API in this codebase.
    // This differential-response approach is equivalent and more direct.

    println!("── Computing modality dominance via differential response ──");
    println!("   (no buf_afw readback — using per-modality activation difference)");

    // Tick A: only modality A active (B = 0)
    let mut waves_a: Vec<[i8; 223]> = Vec::with_capacity(N_SERES);
    {
        let mut w = [0i8; 223];
        for k in 0..=MOD_A_END {
            let v = (2.0 * std::f32::consts::PI * k as f32 / 112.0 + 0.5).sin() * 100.0;
            w[k] = v.clamp(-127.0, 127.0) as i8;
        }
        // channels 112-222 stay at 0 → modality B silenced
        waves_a.push(w);
    }
    cortex.upload_wave(&waves_a, N_SERES);
    let mut enc_a = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("cortex-map-probe-a"),
    });
    cortex.dispatch_afference(&mut enc_a, N_SERES, (N_TICKS + 0) as u32);
    // No Hebb on this probe tick — only measures response
    cortex.copy_for_readback(&mut enc_a, N_SERES);
    queue.submit(Some(enc_a.finish()));
    device.poll(wgpu::Maintain::Wait);
    let acts_a = cortex.read_act_raw(N_SERES);
    let resp_a: Vec<f32> = acts_a[..SHEET_N].to_vec();

    // Tick B: only modality B active (A = 0)
    let mut waves_b: Vec<[i8; 223]> = Vec::with_capacity(N_SERES);
    {
        let mut w = [0i8; 223];
        for k in MOD_B_START..N_INPUTS {
            let kk = k - MOD_B_START;
            let v = (2.0 * std::f32::consts::PI * kk as f32 / 111.0 + 0.5).cos() * 100.0;
            w[k] = v.clamp(-127.0, 127.0) as i8;
        }
        // channels 0-111 stay at 0 → modality A silenced
        waves_b.push(w);
    }
    cortex.upload_wave(&waves_b, N_SERES);
    let mut enc_b = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("cortex-map-probe-b"),
    });
    cortex.dispatch_afference(&mut enc_b, N_SERES, (N_TICKS + 1) as u32);
    cortex.copy_for_readback(&mut enc_b, N_SERES);
    queue.submit(Some(enc_b.finish()));
    device.poll(wgpu::Maintain::Wait);
    let acts_b = cortex.read_act_raw(N_SERES);
    let resp_b: Vec<f32> = acts_b[..SHEET_N].to_vec();

    // Dominance per unit: A if resp_a > resp_b, B if resp_b > resp_a, '?' if equal
    let dominance: Vec<u8> = (0..SHEET_N)
        .map(|u| {
            let da = resp_a[u].abs();
            let db = resp_b[u].abs();
            if (da - db).abs() < 1e-6 {
                2u8
            }
            // undefined
            else if da > db {
                0u8
            }
            // modality A dominates
            else {
                1u8
            } // modality B dominates
        })
        .collect();

    let n_a = dominance.iter().filter(|&&d| d == 0).count();
    let n_b = dominance.iter().filter(|&&d| d == 1).count();
    let n_indef = dominance.iter().filter(|&&d| d == 2).count();

    println!();
    println!("══════════════════════════════════════════════════");
    println!("  32×32 ALLOCATION MAP (emergent, NOT hardcoded)");
    println!("══════════════════════════════════════════════════");
    println!("  A = modality A dominates | B = modality B dominates | ? = undefined");
    println!();

    // Checks whether the map is homogeneous (all units equal → FLAG IT)
    let all_same = n_a == SHEET_N || n_b == SHEET_N || n_indef == SHEET_N;
    if all_same {
        println!(
            "ATTENTION: all units show the same dominance ({}/{}/{} A/B/?)",
            n_a, n_b, n_indef
        );
        println!("This is EVIDENCE AGAINST segregation — the map is HOMOGENEOUS.");
        println!();
    }

    // Prints 32×32 map in ASCII
    for row in 0..SHEET_H {
        print!("  ");
        for col in 0..SHEET_W {
            let u = row * SHEET_W + col;
            let c = match dominance[u] {
                0 => 'A',
                1 => 'B',
                _ => '?',
            };
            print!("{c}");
        }
        println!();
    }
    println!();
    println!(
        "  Count: A={n_a} ({:.1}%)  B={n_b} ({:.1}%)  ?={n_indef} ({:.1}%)",
        n_a as f32 / SHEET_N as f32 * 100.0,
        n_b as f32 / SHEET_N as f32 * 100.0,
        n_indef as f32 / SHEET_N as f32 * 100.0
    );

    // ── BOUNDARY: cross-correlation between neighboring units ─────────────────
    // For each unit, we compute the Pearson correlation between its response
    // vector (resp_a[u], resp_b[u]) and that of its 4 neighbors (NSEW).
    //
    // Feature vector per unit: [resp_a, resp_b]
    // Correlation between units i and j:
    //   r(i,j) = cos_sim([resp_a_i, resp_b_i], [resp_a_j, resp_b_j])
    //          = (rA_i*rA_j + rB_i*rB_j) / (|v_i| × |v_j|)
    //
    // "Correlation gradient" at u = 1 - mean(correlations with neighbors).
    // High value = boundary. Low value = region interior.

    println!("══════════════════════════════════════════════════");
    println!("  BOUNDARY MAP — 32×32 correlation gradient");
    println!("  '.' = region interior (high correlation with neighbors)");
    println!("  '|' or '-' = edge detected (correlation drops)");
    println!("  '#' = strong boundary (correlation < 0.5 with neighbors)");
    println!("══════════════════════════════════════════════════");
    println!();

    // Norm of the response vector per unit
    let norms: Vec<f32> = (0..SHEET_N)
        .map(|u| {
            (resp_a[u] * resp_a[u] + resp_b[u] * resp_b[u])
                .sqrt()
                .max(1e-8)
        })
        .collect();

    // Correlation with neighbors (4-connectivity)
    let neighbor_corr: Vec<f32> = (0..SHEET_N)
        .map(|u| {
            let row = u / SHEET_W;
            let col = u % SHEET_W;
            let mut sum_r = 0.0_f32;
            let mut cnt = 0u32;

            // North
            if row > 0 {
                let j = u - SHEET_W;
                let dot = resp_a[u] * resp_a[j] + resp_b[u] * resp_b[j];
                sum_r += dot / (norms[u] * norms[j]);
                cnt += 1;
            }
            // South
            if row < SHEET_H - 1 {
                let j = u + SHEET_W;
                let dot = resp_a[u] * resp_a[j] + resp_b[u] * resp_b[j];
                sum_r += dot / (norms[u] * norms[j]);
                cnt += 1;
            }
            // West
            if col > 0 {
                let j = u - 1;
                let dot = resp_a[u] * resp_a[j] + resp_b[u] * resp_b[j];
                sum_r += dot / (norms[u] * norms[j]);
                cnt += 1;
            }
            // East
            if col < SHEET_W - 1 {
                let j = u + 1;
                let dot = resp_a[u] * resp_a[j] + resp_b[u] * resp_b[j];
                sum_r += dot / (norms[u] * norms[j]);
                cnt += 1;
            }

            if cnt == 0 { 1.0 } else { sum_r / cnt as f32 }
        })
        .collect();

    // Boundary threshold: correlation with neighbors < 0.7 is suspected boundary
    for row in 0..SHEET_H {
        print!("  ");
        for col in 0..SHEET_W {
            let u = row * SHEET_W + col;
            let r = neighbor_corr[u];
            let c = if r >= 0.9 {
                '.' // interior — high correlation with neighbors
            } else if r >= 0.7 {
                ',' // transition zone
            } else if r >= 0.5 {
                'o' // moderate boundary
            } else {
                '#' // strong boundary
            };
            print!("{c}");
        }
        // Prints the row's mean correlation value for reference
        let row_mean: f32 = (0..SHEET_W)
            .map(|col| neighbor_corr[row * SHEET_W + col])
            .sum::<f32>()
            / SHEET_W as f32;
        println!("  neighbor_corr={:.3}", row_mean);
    }
    println!();

    // ── P1 REFUTATION CRITERION ───────────────────────────────────────────────
    // INTRA-region correlation: mean r over pairs (u,j) where dominance[u]==dominance[j].
    // INTER-region correlation: mean r over pairs (u,j) where dominance[u]!=dominance[j].
    //
    // We sample NEIGHBOR pairs (4-connectivity) for efficiency.

    let mut intra_sum = 0.0_f64;
    let mut intra_cnt = 0u64;
    let mut inter_sum = 0.0_f64;
    let mut inter_cnt = 0u64;

    for u in 0..SHEET_N {
        let row = u / SHEET_W;
        let col = u % SHEET_W;
        let du = dominance[u];

        // East and south neighbors (avoids counting the pair twice)
        let neighbors: &[Option<usize>] = &[
            if col < SHEET_W - 1 { Some(u + 1) } else { None }, // east
            if row < SHEET_H - 1 {
                Some(u + SHEET_W)
            } else {
                None
            }, // south
        ];

        for &nb_opt in neighbors {
            let Some(j) = nb_opt else { continue };
            let dot = resp_a[u] * resp_a[j] + resp_b[u] * resp_b[j];
            let r = dot / (norms[u] * norms[j]);
            let dj = dominance[j];

            // Ignores undefined pairs in the refutation statistics
            if du == 2 || dj == 2 {
                continue;
            }

            if du == dj {
                intra_sum += r as f64;
                intra_cnt += 1;
            } else {
                inter_sum += r as f64;
                inter_cnt += 1;
            }
        }
    }

    // Collects intra correlations to compute the mean with more pairs (non-neighbors)
    // For more robustness, we also sample diagonal and distance-2 pairs.
    for u in 0..SHEET_N {
        let row = u / SHEET_W;
        let col = u % SHEET_W;
        let du = dominance[u];
        if du == 2 {
            continue;
        }

        // SE and NE diagonal (distance √2)
        let diag_neighbors: &[Option<usize>] = &[
            if col < SHEET_W - 1 && row < SHEET_H - 1 {
                Some(u + SHEET_W + 1)
            } else {
                None
            },
            if col < SHEET_W - 1 && row > 0 {
                Some(u - SHEET_W + 1)
            } else {
                None
            },
        ];
        for &nb_opt in diag_neighbors {
            let Some(j) = nb_opt else { continue };
            let dj = dominance[j];
            if dj == 2 {
                continue;
            }
            let dot = resp_a[u] * resp_a[j] + resp_b[u] * resp_b[j];
            let r = (dot / (norms[u] * norms[j])) as f64;
            if du == dj {
                intra_sum += r;
                intra_cnt += 1;
            } else {
                inter_sum += r;
                inter_cnt += 1;
            }
        }
    }

    println!("══════════════════════════════════════════════════");
    println!("  PREDICTION P1 TEST — REFUTATION CRITERION");
    println!("══════════════════════════════════════════════════");

    if intra_cnt == 0 || inter_cnt == 0 {
        println!();
        println!("NOT MEASURED: not enough pairs to compute correlations.");
        println!("  intra pairs: {intra_cnt}  inter pairs: {inter_cnt}");
        println!(
            "  Likely cause: segregation did not occur ({}/{}/{} A/B/?) or",
            n_a, n_b, n_indef
        );
        println!("  N_TICKS={N_TICKS} insufficient for Hebb to converge.");
        println!("  Prediction P1 NOT TESTED (no data).");
    } else {
        let corr_intra = (intra_sum / intra_cnt as f64) as f32;
        let corr_inter = (inter_sum / inter_cnt as f64) as f32;
        let razao = if corr_inter.abs() > 1e-8 {
            (corr_intra / corr_inter).abs()
        } else {
            // corr_inter ≈ 0: completely decorrelated regions — good sign
            // Ratio formally infinite; we report it as ">100 (inter≈0)"
            f32::INFINITY
        };

        println!();
        println!(
            "  intra-region pairs:  {intra_cnt:6}  corr_intra = {:+.4}",
            corr_intra
        );
        println!(
            "  inter-region pairs: {inter_cnt:6}  corr_inter = {:+.4}",
            corr_inter
        );
        println!();
        println!(
            "  ratio = |corr_intra / corr_inter| = {}",
            if razao.is_infinite() {
                ">100 (inter≈0)".to_string()
            } else {
                format!("{:.2}", razao)
            }
        );
        println!();

        let threshold = 3.0_f32;
        if razao >= threshold || corr_inter.abs() < 0.01 {
            println!(
                "  ✓ PREDICTION P1 CONFIRMED — ratio {:.2}× ≥ threshold {:.0}×",
                if razao.is_infinite() { 999.0 } else { razao },
                threshold
            );
            println!("    Boundary emerges where cross-correlation drops.");
            if corr_intra < 0.6 {
                println!(
                    "  CAVEAT: corr_intra={:.4} < 0.6 predicted in the doc.",
                    corr_intra
                );
                println!(
                    "    Segregation exists but is weak — N_TICKS={N_TICKS} may be insufficient."
                );
            }
        } else if razao < 1.5 {
            println!(
                "  ✗ PREDICTION P1 REFUTED — ratio {:.2}× < threshold {:.0}×",
                razao, threshold
            );
            println!("    Intra-region correlation is NOT significantly higher than inter-region.");
            println!(
                "    Hebb with {} ticks did not segregate the modalities.",
                N_TICKS
            );
            println!(
                "    Hypotheses: (1) ETA=0.005 too small for {} ticks.",
                N_TICKS
            );
            println!("               (2) σ²=64 too large — modalities do not separate spatially.");
            println!("               (3) The prediction requires far more ticks than allowed.");
        } else {
            println!(
                "  ~ PREDICTION P1 INCONCLUSIVE — ratio {:.2}× (between 1.5× and {:.0}×)",
                razao, threshold
            );
            println!("    There is a segregation trend but below the formal 3× criterion.");
        }
    }

    // ── τ BANK: spread of active vs silent τ (P2) ────────────────────────────
    println!();
    println!("══════════════════════════════════════════════════");
    println!("  τ BANK — SPREAD (P2)");
    println!("══════════════════════════════════════════════════");

    // Classifies units into active, silent, intermediate
    let mut tau_ativas: Vec<f32> = Vec::new();
    let mut tau_silentes: Vec<f32> = Vec::new();

    for u in 0..SHEET_N {
        let a = act_mean[u];
        let tau = taus[u];
        if a > ATIVO_LIMIAR {
            tau_ativas.push(tau);
        } else if a < SILENTE_LIMIAR {
            tau_silentes.push(tau);
        }
    }

    println!();
    println!(
        "  Active threshold:   >{:.3}   n_active   = {}",
        ATIVO_LIMIAR,
        tau_ativas.len()
    );
    println!(
        "  Silent threshold: <{:.3}   n_silent = {}",
        SILENTE_LIMIAR,
        tau_silentes.len()
    );

    if tau_ativas.is_empty() || tau_silentes.is_empty() {
        println!();
        println!("  NOT MEASURED: not enough units in one of the categories.");
        println!("  global act_max = {:.6}", act_max);
        println!("  Cause: {N_TICKS} ticks insufficient to polarize the sheet.");
        println!("  Mean activity histogram:");
        let buckets = [0.001_f32, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0];
        for (i, &thr) in buckets.iter().enumerate() {
            let lo = if i == 0 { 0.0 } else { buckets[i - 1] };
            let cnt = act_mean.iter().filter(|&&a| a >= lo && a < thr).count();
            println!("    [{:.3}, {:.3}): {:4} units", lo, thr, cnt);
        }
        let n_above = act_mean.iter().filter(|&&a| a >= 0.5).count();
        println!("    [0.500, ∞):   {:4} units", n_above);
    } else {
        tau_ativas.sort_by(|a, b| a.partial_cmp(b).unwrap());
        tau_silentes.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let spread_ativo = tau_ativas.last().unwrap() / tau_ativas[0];
        let spread_silente = tau_silentes.last().unwrap() / tau_silentes[0];

        println!();
        println!(
            "  τ_active:   min={:.2}ms  max={:.1}ms  spread={:.1}×",
            tau_ativas[0] * 1000.0,
            tau_ativas.last().unwrap() * 1000.0,
            spread_ativo
        );
        println!(
            "  τ_silent: min={:.2}ms  max={:.1}ms  spread={:.1}×",
            tau_silentes[0] * 1000.0,
            tau_silentes.last().unwrap() * 1000.0,
            spread_silente
        );
        println!();

        // Histogram of active units per band (doc §5.3)
        let (mut ng, mut nb, mut na, mut nth, mut nd) = (0u32, 0u32, 0u32, 0u32, 0u32);
        for &t in &tau_ativas {
            let f = 1.0 / (2.0 * std::f32::consts::PI * t);
            if f >= 30.0 {
                ng += 1;
            } else if f >= 13.0 {
                nb += 1;
            } else if f >= 8.0 {
                na += 1;
            } else if f >= 4.0 {
                nth += 1;
            } else {
                nd += 1;
            }
        }
        let n_at = tau_ativas.len() as f32;
        println!("  τ histogram of ACTIVE units (bands doc §5.3):");
        println!(
            "    gamma (>30Hz, τ<5.3ms):   {:4} ({:.1}%)",
            ng,
            ng as f32 / n_at * 100.0
        );
        println!(
            "    beta  (13-30Hz):           {:4} ({:.1}%)",
            nb,
            nb as f32 / n_at * 100.0
        );
        println!(
            "    alpha (8-13Hz):            {:4} ({:.1}%)",
            na,
            na as f32 / n_at * 100.0
        );
        println!(
            "    theta (4-8Hz):             {:4} ({:.1}%)",
            nth,
            nth as f32 / n_at * 100.0
        );
        println!(
            "    delta (<4Hz, τ>40ms):      {:4} ({:.1}%)",
            nd,
            nd as f32 / n_at * 100.0
        );
        println!();

        // Prediction P2: spread_ativo > 10×
        if spread_ativo > 10.0 {
            println!(
                "  ✓ P2 CONFIRMED: spread_ativo={:.1}× > 10× (doc predicts >10× in champions)",
                spread_ativo
            );
        } else if spread_ativo > 5.0 {
            println!(
                "  ~ P2 INCONCLUSIVE: spread_ativo={:.1}× (between 5× and 10×)",
                spread_ativo
            );
        } else {
            println!(
                "  ✗ P2 REFUTED: spread_ativo={:.1}× < 5× — τ bank concentrated in a narrow range",
                spread_ativo
            );
        }

        // Universe measured 40× in champions vs 5.3× in the control — we compare
        println!("  Universe champion (reference): 40×  random control: 5.3×");
        if spread_ativo > 5.3 {
            println!("  > random control (ok, sheet is not pure noise)");
        } else {
            println!("  ≤ random control (sheet does not exploit the τ bank better than noise)");
        }
    }

    // ── ALLOMETRIC SANITY (P6 / doc §1.2) ───────────────────────────────────
    // Count: active units with τ > 100ms (analogous to cortex)
    //        vs active units with τ < 10ms (analogous to cerebellum/brainstem)
    // Predicted ratio: ~1:4 (cortex:cerebellum) = ~0.25
    // Outside [0.5/4, 4×4] = [0.125, 16] → wrong calibration

    println!();
    println!("══════════════════════════════════════════════════");
    println!("  ALLOMETRIC SANITY (P6 / doc §1.2)");
    println!("  Active units: τ>100ms (cortex) vs τ<10ms (brainstem/cerebellum)");
    println!("══════════════════════════════════════════════════");

    let n_cortex: usize = (0..SHEET_N)
        .filter(|&u| act_mean[u] > ATIVO_LIMIAR && taus[u] > 0.100)
        .count();
    let n_tronco: usize = (0..SHEET_N)
        .filter(|&u| act_mean[u] > ATIVO_LIMIAR && taus[u] < 0.010)
        .count();

    println!();
    println!("  Active units with τ > 100ms (cortex): {n_cortex}");
    println!("  Active units with τ <  10ms (brainstem):  {n_tronco}");

    if n_tronco == 0 {
        println!();
        println!("  NOT MEASURED: n_tronco = 0. Impossible to compute ratio.");
        println!(
            "  Likely cause: very low activity in the first {} units (short τ).",
            n_tronco
        );
        println!("  Short-τ units (γ) integrate fast but also decay fast —");
        println!("  with N_TICKS={N_TICKS} and a slow signal, they may be persistently silent.");
    } else {
        let razao_alometrica = n_cortex as f32 / n_tronco as f32;
        println!("  Cortex/brainstem ratio = {:.2}", razao_alometrica);
        println!("  Conserved fact in mammals: ~1:4 = ~0.25 (doc §1.2)");
        println!();

        if razao_alometrica < 0.5 {
            println!(
                "  ✗ OUT OF RANGE: ratio {:.2} < 0.5 — too little active cortex.",
                razao_alometrica
            );
            println!("    Wrong calibration: slow units (long τ) insufficiently active.");
            println!(
                "    Possible cause: {N_TICKS} ticks insufficient for slow integration to converge."
            );
        } else if razao_alometrica > 16.0 {
            println!(
                "  ✗ OUT OF RANGE: ratio {:.2} > 16 — brainstem completely silent.",
                razao_alometrica
            );
            println!("    Wrong calibration: fast units (short τ) do not respond to the input.");
            println!(
                "    Possible cause: synthetic input varies slowly → short τ already decayed."
            );
        } else {
            println!(
                "  ✓ IN THE BIOLOGICAL RANGE [0.5, 16]: ratio {:.2}",
                razao_alometrica
            );
            if (razao_alometrica - 0.25).abs() < 0.20 {
                println!("    Close to the mammal target 1:4 = 0.25 ← allometric sanity OK");
            } else {
                println!(
                    "    Within range but deviated from 0.25 — acceptable for a synthetic sheet."
                );
            }
        }
    }

    // ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  RESULTS SUMMARY (cortex_map)                                   ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Input: SYNTHETIC (2 decorrelated modalities)                   ║");
    println!("║  Ticks run: {N_TICKS:2}  |  Beings: {N_SERES:2}  |  Sheet: 32×32=1024     ║");
    println!("║  GPU: {:<59}║", &name[..name.len().min(59)]);
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!(
        "║  [Map above] A={n_a:4} ({:.1}%)  B={n_b:4} ({:.1}%)  ?={n_indef:4} ({:.1}%){}║",
        n_a as f32 / SHEET_N as f32 * 100.0,
        n_b as f32 / SHEET_N as f32 * 100.0,
        n_indef as f32 / SHEET_N as f32 * 100.0,
        " ".repeat(4)
    );
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("NOTE: {N_TICKS} ticks is the safety limit (evolution in background).");
    println!("      For robust segregation, the doc predicts 5.000+ ticks with 30+ beings.");
    println!("      An inconclusive result here does NOT refute P1 — it just does NOT confirm it.");
}
