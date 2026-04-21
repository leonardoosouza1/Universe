// DNA viewer — dump header + weight stats + heatmap ASCII do .dna ternário.
// Uso: cargo run --release --bin dna_view [CAMINHO.dna]
// Default: ./CHAMPION.dna

use universe::evolution::dna_genesis::{load_champion_dna, DNA_FAST_SIZE, DNA_SLOW_SIZE};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).cloned().unwrap_or_else(|| "CHAMPION.dna".to_string());

    let dna = match load_champion_dna(&path) {
        Some(d) => d,
        None => {
            eprintln!("❌ Falha ao carregar {}. Arquivo inexistente ou corrompido.", path);
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
        let report = iara_genesis::tlv::compression_report(&dna.fast_weights, sw, dna.generation, dna.fitness);
        println!("─────────────────────────────────────────────────────────────────");
        println!("🚀 V4 GENERATIVE COMPRESSION");
        println!("  codec base    : {}", report.codec_used);
        println!("  ratio final   : {:.1}:1 ({} bytes originais → {} bytes V4)", report.ratio, report.original_size, report.compressed_size);
        println!("  fidelid. fast : {:.1}%", report.fast_accuracy * 100.0);
        println!("  fidelid. slow : {:.1}%", report.slow_accuracy * 100.0);
        println!("  tree splines  : {} corrections", report.tree_corrections);
        println!("─────────────────────────────────────────────────────────────────");
        println!();
    }

    dump_layer("INSTINCT FAST (128 inputs × 64 neurons)", &dna.fast_weights, 128, 64, DNA_FAST_SIZE);

    if let Some(ref sw) = dna.slow_weights {
        dump_layer("SLOW BRAIN (32 inputs × 8 neurons)", sw, 32, 8, DNA_SLOW_SIZE);
    } else {
        println!("⚠  slow_weights ausente (DNA legacy v1).");
    }

    println!();
    println!("🧠 INPUT CHANNELS (first layer) — canais sensoriais ativos por neurônio");
    println!("   Forte (>30% ativo): canal que polariza múltiplos neurônios = sinal crítico");
    input_channel_stats(&dna.fast_weights, 128, 64);
}

fn dump_layer(title: &str, weights: &[i8], inputs: usize, neurons: usize, expected_size: usize) {
    println!("─────────────────────────────────────────────────────────────────");
    println!("🔹 {}", title);
    println!("  tamanho esperado : {} pesos", expected_size);
    println!("  tamanho real     : {} pesos", weights.len());

    if weights.is_empty() {
        println!("  ⚠  vazio");
        return;
    }

    let mut n_neg = 0;
    let mut n_zero = 0;
    let mut n_pos = 0;
    for &w in weights {
        if w < 0 { n_neg += 1; }
        else if w == 0 { n_zero += 1; }
        else { n_pos += 1; }
    }
    let total = weights.len() as f32;
    println!("  distribuição:");
    println!("    -1 (inibitório) : {:5} ({:.1}%)", n_neg, n_neg as f32 * 100.0 / total);
    println!("     0 (mudo)       : {:5} ({:.1}%)", n_zero, n_zero as f32 * 100.0 / total);
    println!("    +1 (excitatório): {:5} ({:.1}%)", n_pos, n_pos as f32 * 100.0 / total);

    let active_frac = (n_neg + n_pos) as f32 / total;
    println!("  densidade ativa  : {:.1}% (sparsity {:.1}%)", active_frac * 100.0, (1.0 - active_frac) * 100.0);

    println!();
    println!("  heatmap por neurônio ({} neurônios × {} inputs):", neurons, inputs);
    println!("  legenda: '-'=inibe  '.'=mudo  '+'=excita");
    let samples = neurons.min(16);
    for n in 0..samples {
        print!("  N{:02} │", n);
        for i in 0..inputs.min(64) {
            let idx = n * inputs + i;
            if idx >= weights.len() { break; }
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
        println!("  … ({} neurônios adicionais omitidos)", neurons - samples);
    }
}

fn input_channel_stats(weights: &[i8], inputs: usize, neurons: usize) {
    if weights.len() != inputs * neurons { return; }
    let mut ch_activity: Vec<(usize, f32, i32, i32)> = (0..inputs).map(|c| {
        let mut neg = 0;
        let mut pos = 0;
        for n in 0..neurons {
            let w = weights[n * inputs + c];
            if w < 0 { neg += 1; }
            else if w > 0 { pos += 1; }
        }
        let active = (neg + pos) as f32 / neurons as f32;
        (c, active, neg, pos)
    }).collect();

    ch_activity.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!();
    println!("  TOP 20 canais mais ativos:");
    println!("  {:>5} {:>8} {:>6} {:>6}  bar", "ch", "active%", "neg", "pos");
    for (ch, active, neg, pos) in ch_activity.iter().take(20) {
        let bar_len = (*active * 40.0) as usize;
        let bar = "█".repeat(bar_len);
        println!("  {:>5} {:>7.1}% {:>6} {:>6}  {}{}", ch, active * 100.0, neg, pos, bar, channel_label(*ch));
    }

    println!();
    println!("  20 canais MUDOS (menor ativação = rede ignora):");
    let mut silent = ch_activity.clone();
    silent.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    for (ch, active, neg, pos) in silent.iter().take(20) {
        println!("  {:>5} {:>7.1}% {:>6} {:>6}  {}", ch, active * 100.0, neg, pos, channel_label(*ch));
    }
}

fn channel_label(ch: usize) -> &'static str {
    match ch {
        0..=31   => "  ← retina fóvea",
        32..=63  => "  ← retina periférica",
        64       => "  ← alinhamento c/ pista",
        65       => "  ← velocidade",
        66       => "  ← lateral_vel (drift)",
        67       => "  ← hp",
        79       => "  ← curvatura 25m",
        82       => "  ← ttc_inv (colisão iminente)",
        85       => "  ← proximidade parede L",
        86       => "  ← proximidade parede R",
        94..=97  => "  ← whiskers ±30/±60°",
        98       => "  ← gaze_yaw",
        125      => "  ← stress competitivo",
        _        => "",
    }
}
