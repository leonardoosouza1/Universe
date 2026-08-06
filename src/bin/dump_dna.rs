//! `dump_dna` — o que tem dentro de um `.dna`, antes de acreditar em qualquer leitura.
//!
//! Existe porque um `.dna` v8 guarda os pesos da camada 1 em DOIS lugares com escalas
//! diferentes, e ler o lugar errado não dá erro — dá um relatório silenciosamente falso:
//!
//!   `fast_weights`  versão TERNÁRIA {−1,0,+1} do compressor. Dividida por 63,5 vira ±0,016
//!                   e faz o cérebro inteiro parecer vazio.
//!   chunk `CRAW`    os pesos i8 de verdade, ±127 → ±2,0. É o que `spawn.rs:142` prefere.
//!
//! Foi esse engano que fez `sinapses` reportar "0 sinapses" num campeão que tinha 1637.
//! Rodar isto primeiro custa um segundo e mostra qual fonte o arquivo realmente tem.
//!
//! ```text
//! cargo run --release --bin dump_dna -- CHAMPION.dna
//! ```

use universe::evolution::dna_genesis::load_champion_dna;

fn perfil(rotulo: &str, w: &[i8]) {
    if w.is_empty() {
        println!("  {rotulo:<14} vazio");
        return;
    }
    let nz = w.iter().filter(|&&x| x != 0).count();
    let (mn, mx) = w.iter().fold((0i8, 0i8), |(a, b), &x| (a.min(x), b.max(x)));
    // 8 faixas de 16 no valor absoluto: revela de imediato se o conteúdo é ternário
    // (tudo na primeira faixa) ou i8 cheio (com cauda até 127)
    let mut h = [0usize; 8];
    for &x in w {
        h[(((x as i32).abs() / 16) as usize).min(7)] += 1;
    }
    let ternario = mx <= 1 && mn >= -1;
    println!(
        "  {rotulo:<14} n={:<7} não-zero={:<6} ({:>5.1}%)  faixa=[{:>4},{:>4}]  {}",
        w.len(), nz, nz as f32 / w.len() as f32 * 100.0, mn, mx,
        if ternario { "TERNÁRIO ⚠️  /63.5 aqui dá ±0.016" } else { "i8 cheio" }
    );
    println!("  {:<14} histograma |i8| em faixas de 16: {:?}", "", h);
}

fn main() {
    let arquivo = match std::env::args().nth(1) {
        Some(a) => a,
        None => {
            eprintln!("uso: dump_dna <arquivo.dna>");
            std::process::exit(2);
        }
    };
    let d = match load_champion_dna(&arquivo) {
        Some(d) => d,
        None => {
            eprintln!("não consegui ler {arquivo} (formato não reconhecido?)");
            std::process::exit(1);
        }
    };

    println!("=== {arquivo} ===");
    println!(
        "versão={}  geração={}  fitness={:.0}  semente={}  largura de pista={:.1}m",
        d.version, d.generation, d.fitness, d.seed, d.track_width
    );
    println!("fast_scale={}  slow_scale={}", d.fast_scale, d.slow_scale);

    println!("\nPESOS DA CAMADA 1 — as duas fontes possíveis:");
    perfil("fast_weights", &d.fast_weights);
    match d.cortex_raw.as_ref() {
        Some(c) => perfil("CRAW", c),
        None => println!("  {:<14} ausente", "CRAW"),
    }

    println!("\nOUTROS BLOCOS:");
    let opt = |v: Option<&Vec<i8>>| v.map_or("ausente".to_string(), |x| format!("{} pesos", x.len()));
    println!("  cortex_l2_raw  {}", opt(d.cortex_l2_raw.as_ref()));
    println!("  slow_weights   {}", opt(d.slow_weights.as_ref()));
    println!("  trinity        {}", if d.trinity.is_some() { "presente (portão + reflexo)" }
                                    else { "ausente" });

    let fonte = if d.cortex_raw.is_some() { "CRAW" } else { "fast_weights" };
    println!("\n-> `sinapses` vai ler: {fonte}");
    if d.cortex_raw.is_none() {
        println!("   ⚠️  sem CRAW os pesos saem ~64× menores do que são.");
    }
}
