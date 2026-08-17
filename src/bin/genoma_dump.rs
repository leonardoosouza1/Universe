//! Text dump of a genome, through the CANONICAL parser (`universe::gpu::genome_io`).
//!
//! Exists because reimplementing the file layout has already cost this project dearly:
//! a second parser read `neuron_activation` believing it was τ and produced a
//! "spread 0.0×" that was a lie. Every genome analysis outside Rust goes
//! through here — Python receives text, never decodes the binary.
//!
//! Usage: genoma_dump --arquivo <genoma.bin> --saida <folder>
//! Writes into the folder: l2.txt (L2_OUT rows × L1_OUT columns), neuron_activation.txt,
//! neuron_alpha.txt, neuron_threshold_l1.txt, tau.txt, meta.txt.

use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut arquivo = String::new();
    let mut saida = String::from("/tmp");
    let mut patch = String::new();
    let mut saida_genoma = String::new();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--arquivo" => {
                i += 1;
                arquivo = args[i].clone();
            }
            "--saida" => {
                i += 1;
                saida = args[i].clone();
            }
            "--patch-l2" => {
                i += 1;
                patch = args[i].clone();
            }
            "--saida-genoma" => {
                i += 1;
                saida_genoma = args[i].clone();
            }
            _ => {}
        }
        i += 1;
    }
    if !patch.is_empty() {
        patch_l2(&arquivo, &patch, &saida_genoma);
        return;
    }
    let g = match universe::gpu::genome_io::load_genome(&arquivo) {
        Some(g) => g,
        None => {
            eprintln!("could not load {arquivo}");
            std::process::exit(1);
        }
    };
    std::fs::create_dir_all(&saida).unwrap();

    let escreve = |nome: &str, v: &[f32], cols: usize| {
        let mut f = std::fs::File::create(format!("{saida}/{nome}")).unwrap();
        for (k, x) in v.iter().enumerate() {
            write!(f, "{x}").unwrap();
            if cols > 0 && (k + 1) % cols == 0 {
                writeln!(f).unwrap();
            } else {
                write!(f, " ").unwrap();
            }
        }
        if cols == 0 {
            writeln!(f).unwrap();
        }
    };

    // Layout of `buf_l2w` in the shader (brain.wgsl): j*L1_OUT + i — one ROW per
    // L2 output, 96 input columns. Written in the same order so Python does
    // not have to guess.
    let l1_out = 96usize;
    escreve("l2.txt", &g.l2, l1_out);
    escreve("neuron_activation.txt", &g.neuron_activation, 0);
    escreve("neuron_alpha.txt", &g.neuron_alpha, 0);
    escreve("neuron_threshold_l1.txt", &g.neuron_threshold_l1, 0);
    escreve("neuron_period_l1.txt", &g.neuron_period_l1, 0);
    escreve("tau.txt", &g.tau, 0);

    let mut m = std::fs::File::create(format!("{saida}/meta.txt")).unwrap();
    writeln!(m, "arquivo {arquivo}").unwrap();
    writeln!(
        m,
        "l1 {} l2 {} slow {} tau {} sensors {}",
        g.l1.len(),
        g.l2.len(),
        g.slow.len(),
        g.tau.len(),
        g.sensors.len()
    )
    .unwrap();
    writeln!(
        m,
        "neuron_alpha {} neuron_period_l1 {} neuron_threshold_l1 {} neuron_activation {}",
        g.neuron_alpha.len(),
        g.neuron_period_l1.len(),
        g.neuron_threshold_l1.len(),
        g.neuron_activation.len()
    )
    .unwrap();
    writeln!(m, "gates_w {}", g.gates_w.len()).unwrap();
    println!(
        "dump at {saida} — l2 {} weights, activation {} values",
        g.l2.len(),
        g.neuron_activation.len()
    );
}

// ── L2 SURGERY ────────────────────────────────────────────────────────────────
// Second mode (`--patch-l2 <linhas.txt> --saida-genoma <novo.bin>`): replaces the
// FIRST L2 rows (the ones producing throttle/steer/brake) with weights
// solved outside, and rewrites the genome through the canonical serializer.
//
// Why it exists: the formula pilot drives better than the champion, but it is not a
// genome — evolution cannot be seeded with it. This is the bridge: freeze L1 and
// recompute only the L2 rows that become action, with the PILOT as the target. If the
// patched genome drives better than the original, the distillation comes back inside
// what evolution can inherit.
//
// The input file has one line per output (0=throttle, 1=steer, 2=brake),
// 96 weights per line, in the SAME order as `buf_l2w` (j*L1_OUT + i).
fn patch_l2(genoma: &str, linhas_txt: &str, saida: &str) {
    let mut g = universe::gpu::genome_io::load_genome(genoma).expect("genome");
    let txt = std::fs::read_to_string(linhas_txt).expect("lines");
    let mut n_linhas = 0usize;
    for (j, linha) in txt.lines().filter(|l| !l.trim().is_empty()).enumerate() {
        let pesos: Vec<f32> = linha
            .split_whitespace()
            .map(|t| t.parse().unwrap())
            .collect();
        assert_eq!(
            pesos.len(),
            96,
            "row {j} has {} weights, expected 96",
            pesos.len()
        );
        g.l2[j * 96..(j + 1) * 96].copy_from_slice(&pesos);
        n_linhas += 1;
    }
    universe::gpu::genome_io::save_genome(&g, saida).expect("write");
    println!("surgery: {n_linhas} L2 row(s) replaced → {saida}");
}
