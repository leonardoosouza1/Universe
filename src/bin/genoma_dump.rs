//! Dump de um genoma em texto, pelo parser CANÔNICO (`universe::gpu::genome_io`).
//!
//! Existe porque reimplementar o layout do arquivo já custou caro neste projeto:
//! um segundo parser leu `neuron_activation` achando que era τ e produziu um
//! "espalhamento 0,0×" que era mentira. Toda análise de genoma fora do Rust passa
//! por aqui — o Python recebe texto, nunca decodifica o binário.
//!
//! Uso: genoma_dump --arquivo <genoma.bin> --saida <pasta>
//! Gera na pasta: l2.txt (L2_OUT linhas × L1_OUT colunas), neuron_activation.txt,
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
            eprintln!("não consegui carregar {arquivo}");
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

    // Layout de `buf_l2w` no shader (brain.wgsl): j*L1_OUT + i — uma LINHA por
    // saída de L2, 96 colunas de entrada. Escrevo na mesma ordem para o Python
    // não precisar adivinhar.
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
        "dump em {saida} — l2 {} pesos, activation {} valores",
        g.l2.len(),
        g.neuron_activation.len()
    );
}

// ── CIRURGIA DE L2 ────────────────────────────────────────────────────────────
// Segundo modo (`--patch-l2 <linhas.txt> --saida-genoma <novo.bin>`): substitui as
// PRIMEIRAS linhas de L2 (as que produzem acelerador/volante/freio) por pesos
// resolvidos fora, e regrava o genoma pelo serializador canônico.
//
// Por que existe: o piloto por fórmula dirige melhor que o campeão, mas não é um
// genoma — não dá pra semear evolução com ele. Isto é a ponte: congela L1 e
// recalcula só as linhas de L2 que viram ação, tendo o PILOTO como alvo. Se o
// genoma remendado dirigir melhor que o original, a destilação volta pra dentro
// do que a evolução consegue herdar.
//
// O arquivo de entrada tem uma linha por saída (0=acelerador, 1=volante, 2=freio),
// 96 pesos por linha, na MESMA ordem de `buf_l2w` (j*L1_OUT + i).
fn patch_l2(genoma: &str, linhas_txt: &str, saida: &str) {
    let mut g = universe::gpu::genome_io::load_genome(genoma).expect("genoma");
    let txt = std::fs::read_to_string(linhas_txt).expect("linhas");
    let mut n_linhas = 0usize;
    for (j, linha) in txt.lines().filter(|l| !l.trim().is_empty()).enumerate() {
        let pesos: Vec<f32> = linha
            .split_whitespace()
            .map(|t| t.parse().unwrap())
            .collect();
        assert_eq!(
            pesos.len(),
            96,
            "linha {j} com {} pesos, esperava 96",
            pesos.len()
        );
        g.l2[j * 96..(j + 1) * 96].copy_from_slice(&pesos);
        n_linhas += 1;
    }
    universe::gpu::genome_io::save_genome(&g, saida).expect("gravar");
    println!("cirurgia: {n_linhas} linha(s) de L2 substituídas → {saida}");
}
