// DNA viewer — dump header + weight stats + ASCII heatmap of the ternary .dna.
// Usage: cargo run --release --bin dna_view [PATH.dna]
// Default: ./CHAMPION.dna

use universe::evolution::dna_genesis::{DNA_FAST_SIZE, DNA_SLOW_SIZE, load_champion_dna};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "CHAMPION.dna".to_string());

    let dna = match load_champion_dna(&path) {
        Some(d) => d,
        None => {
            eprintln!(
                "❌ Failed to load {}. File missing or corrupted.",
                path
            );
            std::process::exit(1);
        }
    };

    println!("═══════════════════════════════════════════════════════════════");
    println!("🧬 DNA VIEWER — {}", path);
    println!("═══════════════════════════════════════════════════════════════");
    println!("📋 HEADER");
    println!("  version       : v{}", dna.version);
    println!("  generation    : {}", dna.generation);
    println!("  seed          : {}", dna.seed);
    println!("  track_width   : {:.2} m", dna.track_width);
    println!("  fitness       : {:.2}", dna.fitness);
    println!("  fast_scale    : {:.6}", dna.fast_scale);
    println!("  slow_scale    : {:.6}", dna.slow_scale);
    println!();

    if dna.version >= 4 {
        let sw = dna.slow_weights.as_deref().unwrap_or(&[]);
        let report = iara_genesis::tlv::compression_report(
            &dna.fast_weights,
            sw,
            dna.generation,
            dna.fitness,
        );
        println!("─────────────────────────────────────────────────────────────────");
        println!("🚀 V4 GENERATIVE COMPRESSION");
        println!("  base codec    : {}", report.codec_used);
        println!(
            "  final ratio   : {:.1}:1 ({} original bytes → {} V4 bytes)",
            report.ratio, report.original_size, report.compressed_size
        );
        println!("  fast fidelity : {:.1}%", report.fast_accuracy * 100.0);
        println!("  slow fidelity : {:.1}%", report.slow_accuracy * 100.0);
        println!("  tree splines  : {} corrections", report.tree_corrections);
        println!("─────────────────────────────────────────────────────────────────");
        println!();
    }

    dump_layer(
        "INSTINCT FAST (128 inputs × 64 neurons)",
        &dna.fast_weights,
        128,
        64,
        DNA_FAST_SIZE,
    );

    if let Some(ref sf) = dna.slow_f32 {
        println!("─────────────────────────────────────────────────────────────────");
        println!("🔹 SLOW BRAIN f32 RAW (32 inputs × 8 neurons) — SLWF chunk lossless");
        let nonzero = sf.iter().filter(|&&w| w.abs() > 1e-6).count();
        let neg = sf.iter().filter(|&&w| w < -1e-6).count();
        let pos = sf.iter().filter(|&&w| w > 1e-6).count();
        let total = sf.len();
        let mean_abs: f32 = sf.iter().map(|w| w.abs()).sum::<f32>() / total as f32;
        let max_abs = sf.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        println!("  total         : {}", total);
        println!(
            "  active        : {} ({:.1}%)  neg={} pos={}",
            nonzero,
            nonzero as f32 * 100.0 / total as f32,
            neg,
            pos
        );
        println!("  |w|_mean      : {:.4}", mean_abs);
        println!("  |w|_max       : {:.4}", max_abs);
        println!();
        println!("  heatmap (8 neurons × 32 inputs):");
        for n in 0..8 {
            let off = n * 32;
            let row: String = (0..32)
                .map(|i| {
                    let w = sf[off + i];
                    if w > 0.05 {
                        '+'
                    } else if w < -0.05 {
                        '-'
                    } else if w.abs() > 1e-6 {
                        '·'
                    } else {
                        '.'
                    }
                })
                .collect();
            println!("  N{:02} │{}", n, row);
        }
        println!();
    } else if let Some(ref sw) = dna.slow_weights {
        dump_layer(
            "SLOW BRAIN (32 inputs × 8 neurons) — LEGACY ternary",
            sw,
            32,
            8,
            DNA_SLOW_SIZE,
        );
    } else {
        println!("⚠  slow_weights missing (legacy v1 DNA).");
    }

    println!();
    if let Some(craw) = &dna.cortex_raw {
        println!("─────────────────────────────────────────────────────────────────");
        println!("🧠 CRAW CORTEX f32 ANALYSIS (source of truth — bypass compress_organism)");
        println!("─────────────────────────────────────────────────────────────────");
        cortex_f32_stats(craw);
        println!();
        f32_input_channel_stats(craw, 128, 64);
    } else {
        println!("⚠  No CRAW chunk (pre-V6 DNA) — channel analysis uses legacy ternary fast_weights");
        input_channel_stats(&dna.fast_weights, 128, 64);
    }
}

fn cortex_f32_stats(craw: &[i8]) {
    // Dequantize i8 / 63.5 → f32 (the save's scale).
    let weights: Vec<f32> = craw.iter().map(|&w| w as f32 / 63.5).collect();
    let n = weights.len() as f32;
    let mean: f32 = weights.iter().sum::<f32>() / n;
    let var: f32 = weights.iter().map(|w| (w - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();
    let mut max = f32::MIN;
    let mut min = f32::MAX;
    let mut dead_count = 0;
    let mut sat_count = 0;
    for &w in &weights {
        if w > max {
            max = w;
        }
        if w < min {
            min = w;
        }
        if w.abs() < 0.01 {
            dead_count += 1;
        }
        if w.abs() > 1.5 {
            sat_count += 1;
        }
    }
    println!("  total weights : {}", weights.len());
    println!("  mean / std    : {:.4} / {:.4}", mean, std);
    println!("  range         : [{:.4}, {:.4}]", min, max);
    println!(
        "  dead (|w|<.01): {} ({:.1}%) — dead neurons / ignored channels",
        dead_count,
        dead_count as f32 * 100.0 / n
    );
    println!(
        "  saturated     : {} ({:.1}%) — weight |w|>1.5 (clamp ±2 limit)",
        sat_count,
        sat_count as f32 * 100.0 / n
    );
}

fn f32_input_channel_stats(craw: &[i8], inputs: usize, neurons: usize) {
    if craw.len() != inputs * neurons {
        return;
    }
    let weights: Vec<f32> = craw.iter().map(|&w| w as f32 / 63.5).collect();

    // Per channel: absolute sum of weights cross-neurons (importance) +
    // count of neurons that polarize strongly (>0.3 abs).
    let mut ch_stats: Vec<(usize, f32, usize, f32)> = (0..inputs)
        .map(|c| {
            let mut sum_abs = 0.0_f32;
            let mut polarized = 0usize;
            let mut max_abs = 0.0_f32;
            for n in 0..neurons {
                let w = weights[n * inputs + c];
                sum_abs += w.abs();
                if w.abs() > 0.3 {
                    polarized += 1;
                }
                if w.abs() > max_abs {
                    max_abs = w.abs();
                }
            }
            (c, sum_abs / neurons as f32, polarized, max_abs)
        })
        .collect();

    // TOP 25 channels in use (highest mean |w| cross-neurons).
    println!("  TOP 25 MOST USED channels (mean absolute weight cross-neurons):");
    println!(
        "  {:>4} {:>9} {:>8} {:>8}  bar",
        "ch", "|w|_mean", "|polariz|", "|w|_max"
    );
    let mut top = ch_stats.clone();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (ch, mw, pol, maxw) in top.iter().take(25) {
        let bar_len = (mw * 80.0).clamp(0.0, 40.0) as usize;
        let bar = "█".repeat(bar_len);
        println!(
            "  {:>4} {:>9.4} {:>8} {:>8.3}  {}{}",
            ch,
            mw,
            pol,
            maxw,
            bar,
            channel_label(*ch)
        );
    }

    println!();
    println!("  20 MUTE channels (the network ignores them — mean weight < 0.01):");
    let mut silent = ch_stats.clone();
    silent.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut shown = 0;
    for (ch, mw, _, maxw) in silent.iter() {
        if shown >= 20 {
            break;
        }
        if *mw > 0.05 {
            break;
        } // only show the truly mute ones
        println!(
            "  {:>4} {:>9.4} max={:.3}  {}",
            ch,
            mw,
            maxw,
            channel_label(*ch)
        );
        shown += 1;
    }
}

fn dump_layer(title: &str, weights: &[i8], inputs: usize, neurons: usize, expected_size: usize) {
    println!("─────────────────────────────────────────────────────────────────");
    println!("🔹 {}", title);
    println!("  expected size    : {} weights", expected_size);
    println!("  actual size      : {} weights", weights.len());

    if weights.is_empty() {
        println!("  ⚠  empty");
        return;
    }

    let mut n_neg = 0;
    let mut n_zero = 0;
    let mut n_pos = 0;
    for &w in weights {
        if w < 0 {
            n_neg += 1;
        } else if w == 0 {
            n_zero += 1;
        } else {
            n_pos += 1;
        }
    }
    let total = weights.len() as f32;
    println!("  distribution:");
    println!(
        "    -1 (inhibitory) : {:5} ({:.1}%)",
        n_neg,
        n_neg as f32 * 100.0 / total
    );
    println!(
        "     0 (mute)       : {:5} ({:.1}%)",
        n_zero,
        n_zero as f32 * 100.0 / total
    );
    println!(
        "    +1 (excitatory) : {:5} ({:.1}%)",
        n_pos,
        n_pos as f32 * 100.0 / total
    );

    let active_frac = (n_neg + n_pos) as f32 / total;
    println!(
        "  active density   : {:.1}% (sparsity {:.1}%)",
        active_frac * 100.0,
        (1.0 - active_frac) * 100.0
    );

    println!();
    println!(
        "  heatmap per neuron ({} neurons × {} inputs):",
        neurons, inputs
    );
    println!("  legend: '-'=inhibits  '.'=mute  '+'=excites");
    let samples = neurons.min(16);
    for n in 0..samples {
        print!("  N{:02} │", n);
        for i in 0..inputs.min(64) {
            let idx = n * inputs + i;
            if idx >= weights.len() {
                break;
            }
            let c = match weights[idx] {
                w if w < 0 => '-',
                0 => '.',
                _ => '+',
            };
            print!("{}", c);
        }
        println!();
    }
    if samples < neurons {
        println!("  … ({} additional neurons omitted)", neurons - samples);
    }
}

fn input_channel_stats(weights: &[i8], inputs: usize, neurons: usize) {
    if weights.len() != inputs * neurons {
        return;
    }
    let mut ch_activity: Vec<(usize, f32, i32, i32)> = (0..inputs)
        .map(|c| {
            let mut neg = 0;
            let mut pos = 0;
            for n in 0..neurons {
                let w = weights[n * inputs + c];
                if w < 0 {
                    neg += 1;
                } else if w > 0 {
                    pos += 1;
                }
            }
            let active = (neg + pos) as f32 / neurons as f32;
            (c, active, neg, pos)
        })
        .collect();

    ch_activity.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!();
    println!("  TOP 20 most active channels:");
    println!(
        "  {:>5} {:>8} {:>6} {:>6}  bar",
        "ch", "active%", "neg", "pos"
    );
    for (ch, active, neg, pos) in ch_activity.iter().take(20) {
        let bar_len = (*active * 40.0) as usize;
        let bar = "█".repeat(bar_len);
        println!(
            "  {:>5} {:>7.1}% {:>6} {:>6}  {}{}",
            ch,
            active * 100.0,
            neg,
            pos,
            bar,
            channel_label(*ch)
        );
    }

    println!();
    println!("  20 MUTE channels (lowest activation = the network ignores them):");
    let mut silent = ch_activity.clone();
    silent.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    for (ch, active, neg, pos) in silent.iter().take(20) {
        println!(
            "  {:>5} {:>7.1}% {:>6} {:>6}  {}",
            ch,
            active * 100.0,
            neg,
            pos,
            channel_label(*ch)
        );
    }
}

fn channel_label(ch: usize) -> &'static str {
    match ch {
        0..=7 => "  ← fovea pooled (8 bins center-dense)",
        8..=15 => "  ← peripheral pooled (8 bins)",
        16 => "  ← yaw_rate",
        17 => "  ← pitch_rate",
        18 => "  ← roll_rate",
        28 => "  ← pre_brake (speed×curv)",
        32 => "  ← curv180m lookahead",
        36..=39 => "  ← open_left_width × 4",
        40..=43 => "  ← open_right_width × 4",
        44..=47 => "  ← wall_imminent × 4",
        60..=63 => "  ← BEST_ESCAPE direction × 4",
        64 => "  ← alignment with track",
        65 => "  ← speed",
        66 => "  ← lateral_vel (drift)",
        67 => "  ← hp",
        68 => "  ← rpm_ratio",
        69 => "  ← gear",
        70 => "  ← tire_temp",
        71 => "  ← downforce",
        72 => "  ← in_draft",
        73 => "  ← grass under car",
        74 => "  ← steer_limit dyn",
        75 => "  ← rain wetness",
        76..=77 => "  ← whisker dist L/R",
        78..=79 => "  ← curv40m signed dup",
        80 => "  ← curv100m signed",
        81 => "  ← lane_offset",
        82 => "  ← ttc_inv (imminent collision)",
        83..=84 => "  ← motion L/R (optic flow)",
        85 => "  ← wall_ema_L (proximity)",
        86 => "  ← wall_ema_R (proximity)",
        87 => "  ← slip_signal",
        88 => "  ← proximity_sustain",
        89 => "  ← reverse_state",
        90..=93 => "  ← dark_center × 4 (sliding window)",
        94..=97 => "  ← whiskers ±30/±60°",
        108..=110 => "  ← racing_line offset/derivative/lookahead",
        120..=122 => "  ← memory (last action)",
        124 => "  ← spinning",
        125 => "  ← competitive stress",
        126..=127 => "  ← joy tick/streak",
        _ => "",
    }
}
