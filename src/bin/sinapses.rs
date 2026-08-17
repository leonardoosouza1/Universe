//! `sinapses` — reads a `.dna` and tells what that brain has learned, in sentences.
//!
//! A synapse is an edge `sensory channel → neuron`. The weight alone is a number in a
//! position; with the channel name and the neuron's role it becomes a sentence about
//! behavior: *"curvature at 70 m → brake, +10.3"* is a driving rule one can discuss.
//!
//! Comparing two `.dna` files (`--contra`), the program shows what CHANGED: which synapses
//! were born, which died and which thickened. It is the closest reading of "when was this
//! synapse generated" one can get without changing the file format — and without touching
//! the format, old DNAs keep loading.
//!
//! ```text
//! cargo run --release --bin sinapses -- CHAMPION.dna
//! cargo run --release --bin sinapses -- CHAMPION.dna --contra CHAMPION_W14m.dna
//! cargo run --release --bin sinapses -- CHAMPION.dna --neuronio 2 --top 15
//! ```

use universe::config::iara::brain::SENSORY_WAVE_SIZE as SWS;
use universe::evolution::dna_genesis;
use universe::neuro::{canal_nome, neuronio_papel};

/// Below this the synapse is considered absent. Sparse priming is born at ±3.0 to ±10.0 and
/// the gaussian mutation walks at σ≈0.02–0.08, so 0.25 separates "connection" from "drift
/// noise" with an order of magnitude of headroom on both sides.
const LIMIAR: f32 = 0.25;

struct Sinapse {
    neuronio: usize,
    canal: usize,
    peso: f32,
}

fn ler(caminho: &str) -> Result<(Vec<f32>, u32, f32), String> {
    let dna = dna_genesis::load_champion_dna(caminho)
        .ok_or_else(|| format!("could not read {caminho} (unrecognized format?)"))?;
    // SAME PREFERENCE AS `spawn.rs:142`: the CRAW chunk carries the real i8 weights
    // (±127 → ±2.0); `fast_weights` in v8 is the compressor's TERNARY {-1,0,+1} version,
    // which divided by 63.5 would give ±0.016 and make every brain look empty. Reading the
    // wrong source here does not raise an error — it produces a silently false report,
    // which is worse.
    let crus: &Vec<i8> = dna.cortex_raw.as_ref().unwrap_or(&dna.fast_weights);
    if crus.is_empty() {
        return Err(format!("{caminho} has no layer-1 weights"));
    }
    if dna.cortex_raw.is_none() {
        eprintln!(
            "⚠️  {caminho} has no CRAW chunk — reading `fast_weights`, which in v8 is \
                   TERNARY. The weights will look ~64× smaller than they are."
        );
    }
    let w = crus.iter().map(|w| *w as f32 / 63.5).collect();
    Ok((w, dna.generation, dna.fitness))
}

fn fortes(pesos: &[f32], filtro_neuronio: Option<usize>) -> Vec<Sinapse> {
    let mut v: Vec<Sinapse> = pesos
        .iter()
        .enumerate()
        .filter_map(|(i, &p)| {
            let (neuronio, canal) = (i / SWS, i % SWS);
            if p.abs() < LIMIAR {
                return None;
            }
            if let Some(f) = filtro_neuronio {
                if neuronio != f {
                    return None;
                }
            }
            Some(Sinapse {
                neuronio,
                canal,
                peso: p,
            })
        })
        .collect();
    v.sort_by(|a, b| b.peso.abs().partial_cmp(&a.peso.abs()).unwrap());
    v
}

fn frase(s: &Sinapse) -> String {
    let canal = canal_nome(s.canal)
        .map(|n| n.to_string())
        // a channel with no located origin in the code stays explicitly unknown: inventing
        // a name in an interpretability report is worse than showing the raw index
        .unwrap_or_else(|| format!("channel {} (no mapped name)", s.canal));
    // The DNA's i8 caps the weight at ±2.0 (127/63.5), so "saturated" is not an adjective
    // of emphasis: it is the information that this connection hit the STORAGE CEILING and
    // the true value may be larger. Priming is born at ±3.0 to ±10.0 — all of that becomes
    // ±2.0 on disk. Calling that "strong" would hide a limit of the format.
    let forca = match s.peso.abs() {
        p if p >= 1.98 => "SATURATED",
        p if p >= 1.20 => "very strong",
        p if p >= 0.60 => "strong",
        _ => "medium",
    };
    let sentido = if s.peso >= 0.0 { "excites" } else { "inhibits" };
    format!(
        "  {:>+8.2}  {:<11} {:<7} {:<38} → n{:<3} {}",
        s.peso,
        forca,
        sentido,
        canal,
        s.neuronio,
        neuronio_papel(s.neuronio)
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arquivo = match args.get(1) {
        Some(a) if !a.starts_with("--") => a.clone(),
        _ => {
            eprintln!(
                "usage: sinapses <file.dna> [--contra other.dna] \
                       [--neuronio N] [--top K]"
            );
            std::process::exit(2);
        }
    };
    let contra = args
        .iter()
        .position(|a| a == "--contra")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let filtro = args
        .iter()
        .position(|a| a == "--neuronio")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok());
    let top: usize = args
        .iter()
        .position(|a| a == "--top")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(25);

    let (pesos, geracao, fitness) = match ler(&arquivo) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    if args.iter().any(|a| a == "--priming") {
        println!("=== {arquivo} ===");
        println!("generated at generation {geracao} · fitness {fitness:.0}");
        compara_priming(&pesos);
        return;
    }

    if args.iter().any(|a| a == "--mapa") {
        println!("=== {arquivo} ===");
        println!("generated at generation {geracao} · fitness {fitness:.0}");
        mapa(&pesos);
        return;
    }

    let ss = fortes(&pesos, filtro);
    let motoras = ss.iter().filter(|s| s.neuronio < 3).count();
    println!("=== {arquivo} ===");
    println!("generated at generation {geracao} · fitness {fitness:.0}");
    println!(
        "{} synapses above |{LIMIAR}| out of {} weights ({:.1}% of the total) · {} on the 3 motor neurons",
        ss.len(),
        pesos.len(),
        ss.len() as f32 / pesos.len() as f32 * 100.0,
        motoras
    );
    println!("\nTHE {top} STRONGEST\n");
    for s in ss.iter().take(top) {
        println!("{}", frase(s));
    }

    if let Some(outro) = contra {
        let (p2, ger2, _) = match ler(&outro) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        };
        if p2.len() != pesos.len() {
            eprintln!(
                "\n⚠️  the two DNAs have different sizes ({} vs {}) — cannot \
                       pair synapse by synapse",
                pesos.len(),
                p2.len()
            );
            std::process::exit(1);
        }
        println!("\n=== WHAT CHANGED FROM {outro} (gen {ger2}) TO {arquivo} (gen {geracao}) ===\n");
        let (mut nasceram, mut morreram, mut mudaram) = (vec![], vec![], vec![]);
        for i in 0..pesos.len() {
            let (a, b) = (p2[i], pesos[i]);
            let s = Sinapse {
                neuronio: i / SWS,
                canal: i % SWS,
                peso: b,
            };
            match (a.abs() >= LIMIAR, b.abs() >= LIMIAR) {
                (false, true) => nasceram.push(s),
                (true, false) => morreram.push(Sinapse { peso: a, ..s }),
                (true, true) if (b - a).abs() > 0.5 => {
                    mudaram.push((s, b - a));
                }
                _ => {}
            }
        }
        nasceram.sort_by(|x, y| y.peso.abs().partial_cmp(&x.peso.abs()).unwrap());
        morreram.sort_by(|x, y| y.peso.abs().partial_cmp(&x.peso.abs()).unwrap());
        mudaram.sort_by(|x, y| y.1.abs().partial_cmp(&x.1.abs()).unwrap());

        println!("BORN ({}):", nasceram.len());
        for s in nasceram.iter().take(10) {
            println!("{}", frase(s));
        }
        println!("\nDIED ({}):", morreram.len());
        for s in morreram.iter().take(10) {
            println!("{}", frase(s));
        }
        println!("\nTHICKENED OR THINNED ({}):", mudaram.len());
        for (s, d) in mudaram.iter().take(10) {
            println!("{}   (Δ {:+.2})", frase(s), d);
        }
    }
}

// ---------------------------------------------------------------------------------------
// BRAIN MAP — where the neurons are and what each one listens to
// ---------------------------------------------------------------------------------------

/// Sensory groups, to know how much of each sense the brain consumes.
/// The ranges come from where the channels are WRITTEN (`motor.rs:455-524, 480-513`).
const GRUPOS: [(&str, &[(usize, usize)]); 6] = [
    ("VISION (retina)", &[(0, 15), (128, 191)]),
    ("curvature ahead", &[(20, 20), (32, 39), (78, 80)]),
    (
        "walls and whiskers",
        &[(16, 31), (76, 77), (94, 97), (114, 115)],
    ),
    (
        "track ahead",
        &[(40, 55), (90, 93), (108, 110), (118, 119)],
    ),
    ("body telemetry", &[(56, 75), (124, 125)]),
    ("motion and TTC", &[(82, 84), (98, 107), (116, 117)]),
];

fn grupo_de(canal: usize) -> &'static str {
    for (nome, faixas) in GRUPOS.iter() {
        if faixas.iter().any(|&(a, b)| canal >= a && canal <= b) {
            return nome;
        }
    }
    "unclassified"
}

/// How much of each sense the brain consumes, and where it keeps that listening.
fn mapa(pesos: &[f32]) {
    let n_neuronios = pesos.len() / SWS;

    println!("\n=== HOW MUCH OF EACH SENSE THE BRAIN CONSUMES ===");
    println!("sum of |weight| per sensory group, over the {n_neuronios} units of layer 1\n");
    let total: f32 = pesos.iter().map(|w| w.abs()).sum();
    let mut linhas: Vec<(f32, String)> = GRUPOS
        .iter()
        .map(|(nome, faixas)| {
            let (mut soma, mut n_canais) = (0.0f32, 0usize);
            for &(a, b) in faixas.iter() {
                n_canais += b - a + 1;
                for ch in a..=b.min(SWS - 1) {
                    for neu in 0..n_neuronios {
                        soma += pesos[neu * SWS + ch].abs();
                    }
                }
            }
            // per channel, otherwise a large group looks important just for being large
            (
                soma / n_canais.max(1) as f32,
                format!(
                    "{:<22} {:>7} channels {:>9.1} total {:>9.2} per channel {:>7.1}%",
                    nome,
                    n_canais,
                    soma,
                    soma / n_canais.max(1) as f32,
                    soma / total.max(1e-9) * 100.0
                ),
            )
        })
        .collect();
    linhas.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    for (_, l) in &linhas {
        println!("  {l}");
    }
    println!("\n  Sorted by weight PER CHANNEL — otherwise a large group would look important");
    println!("  just for having many channels. The '%' column is the share of the total, for comparison.");

    println!("\n=== WHAT EACH NEURON LISTENS TO ===");
    println!("for each unit: the group that dominates its input and the strongest channel\n");
    let mut por_grupo: std::collections::BTreeMap<&str, usize> = Default::default();
    for neu in 0..n_neuronios {
        let fatia = &pesos[neu * SWS..(neu + 1) * SWS];
        let mut soma_grupo: std::collections::BTreeMap<&str, f32> = Default::default();
        for (ch, w) in fatia.iter().enumerate() {
            *soma_grupo.entry(grupo_de(ch)).or_insert(0.0) += w.abs();
        }
        let dono = soma_grupo
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(k, _)| *k)
            .unwrap_or("—");
        *por_grupo.entry(dono).or_insert(0) += 1;
        if neu < 3 {
            let (melhor_ch, melhor_w) = fatia
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
                .map(|(i, w)| (i, *w))
                .unwrap_or((0, 0.0));
            println!(
                "  n{:<3} {:<14} dominates: {:<22} strongest: {:>+6.2} {}",
                neu,
                neuronio_papel(neu),
                dono,
                melhor_w,
                canal_nome(melhor_ch).unwrap_or("(no name)")
            );
        }
    }
    println!("\n  DISTRIBUTION of the {n_neuronios} units by dominant sense:");
    let mut v: Vec<_> = por_grupo.into_iter().collect();
    v.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    for (g, n) in v {
        let barra = "█".repeat((n * 40 / n_neuronios.max(1)).max(1));
        println!("    {:<22} {:>3} units  {}", g, n, barra);
    }
}

// ---------------------------------------------------------------------------------------
// DOES EVOLUTION FIND WHAT THE HUMAN WROTE?
// ---------------------------------------------------------------------------------------

/// Compares the brain with the 15 hand-written priming synapses (`neuro::PRIMING`).
///
/// The question is one of independence: if a brain is born WITHOUT the priming (`esparso`
/// mode) and still converges to the same connections, then the human only anticipated what
/// evolution would find on its own. If it converges to others, evolution has a path of its
/// own — and it is worth seeing which. If it converges to nothing, the priming was the brain.
fn compara_priming(pesos: &[f32]) {
    use universe::neuro::PRIMING;
    println!("\n=== WHAT THE HUMAN WROTE vs WHAT THIS BRAIN HAS ===\n");
    println!(
        "{:>4} {:>7} {:>9}  {:<44} {}",
        "n", "channel", "weight here", "channel", "verdict"
    );
    println!("{}", "-".repeat(92));
    let mut presentes = 0;
    for &(neu, ch) in PRIMING.iter() {
        let w = pesos.get(neu * SWS + ch).copied().unwrap_or(0.0);
        let veredito = if w.abs() >= 1.0 {
            presentes += 1;
            "PRESENT and strong"
        } else if w.abs() >= LIMIAR {
            presentes += 1;
            "present, weak"
        } else {
            "absent"
        };
        println!(
            "{:>4} {:>7} {:>+9.2}  {:<44} {}",
            neu,
            ch,
            w,
            canal_nome(ch).unwrap_or("(no name)"),
            veredito
        );
    }
    println!("{}", "-".repeat(92));
    println!(
        "  {presentes}/{} of the hand-written synapses are in this brain",
        PRIMING.len()
    );

    // and what it has ON THE MOTORS that the human did NOT write
    let mut proprias: Vec<(usize, usize, f32)> = Vec::new();
    for neu in 0..3 {
        for ch in 0..SWS {
            let w = pesos[neu * SWS + ch];
            if w.abs() >= LIMIAR && !PRIMING.contains(&(neu, ch)) {
                proprias.push((neu, ch, w));
            }
        }
    }
    proprias.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());
    println!(
        "\n  MOTOR CONNECTIONS THE HUMAN DID NOT WRITE: {}",
        proprias.len()
    );
    for (neu, ch, w) in proprias.iter().take(10) {
        println!(
            "    {:>+7.2}  n{} {:<16} ← {}",
            w,
            neu,
            neuronio_papel(*neu),
            canal_nome(*ch).unwrap_or("(no name)")
        );
    }
    if proprias.is_empty() {
        println!("    (none — the motors only have what came from the factory)");
    }
}
