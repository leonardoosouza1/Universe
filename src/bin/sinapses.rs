//! `sinapses` — lê um `.dna` e conta o que aquele cérebro aprendeu, em frases.
//!
//! Uma sinapse é uma aresta `canal sensorial → neurônio`. O peso sozinho é um número numa
//! posição; com o nome do canal e o papel do neurônio vira uma frase sobre comportamento:
//! *"curvatura a 70 m → freio, +10.3"* é uma regra de pilotagem que dá para discutir.
//!
//! Comparando dois `.dna` (`--contra`), o programa mostra o que MUDOU: quais sinapses
//! nasceram, quais morreram e quais engrossaram. É a leitura mais próxima de "quando esta
//! sinapse foi gerada" que se consegue sem mudar o formato do arquivo — e sem mexer no
//! formato, DNAs antigos continuam carregando.
//!
//! ```text
//! cargo run --release --bin sinapses -- CHAMPION.dna
//! cargo run --release --bin sinapses -- CHAMPION.dna --contra CHAMPION_W14m.dna
//! cargo run --release --bin sinapses -- CHAMPION.dna --neuronio 2 --top 15
//! ```

use universe::config::iara::brain::SENSORY_WAVE_SIZE as SWS;
use universe::evolution::dna_genesis;
use universe::neuro::{canal_nome, neuronio_papel};

/// Abaixo disto a sinapse é considerada ausente. O priming esparso nasce em ±3.0 a ±10.0 e
/// a mutação gaussiana anda em σ≈0.02–0.08, então 0.25 separa "conexão" de "ruído de deriva"
/// com folga de uma ordem de grandeza dos dois lados.
const LIMIAR: f32 = 0.25;

struct Sinapse {
    neuronio: usize,
    canal: usize,
    peso: f32,
}

fn ler(caminho: &str) -> Result<(Vec<f32>, u32, f32), String> {
    let dna = dna_genesis::load_champion_dna(caminho)
        .ok_or_else(|| format!("não consegui ler {caminho} (formato não reconhecido?)"))?;
    // MESMA PREFERÊNCIA DE `spawn.rs:142`: o chunk CRAW carrega os pesos i8 de verdade
    // (±127 → ±2.0); `fast_weights` no v8 é a versão TERNÁRIA {-1,0,+1} do compressor, que
    // dividida por 63.5 daria ±0.016 e faria todo cérebro parecer vazio. Ler a fonte errada
    // aqui não dá erro — dá um relatório silenciosamente falso, que é pior.
    let crus: &Vec<i8> = dna.cortex_raw.as_ref().unwrap_or(&dna.fast_weights);
    if crus.is_empty() {
        return Err(format!("{caminho} não tem pesos da camada 1"));
    }
    if dna.cortex_raw.is_none() {
        eprintln!(
            "⚠️  {caminho} não tem chunk CRAW — lendo `fast_weights`, que no v8 é \
                   TERNÁRIO. Os pesos vão parecer ~64× menores do que são."
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
        // canal sem origem localizada no código fica explícito como desconhecido: inventar
        // um nome num relatório de interpretabilidade é pior que mostrar o índice cru
        .unwrap_or_else(|| format!("canal {} (sem nome mapeado)", s.canal));
    // O i8 do DNA trava o peso em ±2.0 (127/63.5), então "saturada" não é um adjetivo de
    // ênfase: é a informação de que aquela conexão bateu no TETO DO ARMAZENAMENTO e o valor
    // verdadeiro pode ser maior. O priming nasce em ±3.0 a ±10.0 — tudo isso vira ±2.0 no
    // disco. Chamar isso de "forte" esconderia um limite do formato.
    let forca = match s.peso.abs() {
        p if p >= 1.98 => "SATURADA",
        p if p >= 1.20 => "muito forte",
        p if p >= 0.60 => "forte",
        _ => "média",
    };
    let sentido = if s.peso >= 0.0 { "excita" } else { "inibe" };
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
                "uso: sinapses <arquivo.dna> [--contra outro.dna] \
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
        println!("gerado na geração {geracao} · fitness {fitness:.0}");
        compara_priming(&pesos);
        return;
    }

    if args.iter().any(|a| a == "--mapa") {
        println!("=== {arquivo} ===");
        println!("gerado na geração {geracao} · fitness {fitness:.0}");
        mapa(&pesos);
        return;
    }

    let ss = fortes(&pesos, filtro);
    let motoras = ss.iter().filter(|s| s.neuronio < 3).count();
    println!("=== {arquivo} ===");
    println!("gerado na geração {geracao} · fitness {fitness:.0}");
    println!(
        "{} sinapses acima de |{LIMIAR}| em {} pesos ({:.1}% do total) · {} nos 3 neurônios motores",
        ss.len(),
        pesos.len(),
        ss.len() as f32 / pesos.len() as f32 * 100.0,
        motoras
    );
    println!("\nAS {top} MAIS FORTES\n");
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
                "\n⚠️  os dois DNAs têm tamanhos diferentes ({} vs {}) — não dá para \
                       parear sinapse a sinapse",
                pesos.len(),
                p2.len()
            );
            std::process::exit(1);
        }
        println!("\n=== O QUE MUDOU DE {outro} (gen {ger2}) PARA {arquivo} (gen {geracao}) ===\n");
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

        println!("NASCERAM ({}):", nasceram.len());
        for s in nasceram.iter().take(10) {
            println!("{}", frase(s));
        }
        println!("\nMORRERAM ({}):", morreram.len());
        for s in morreram.iter().take(10) {
            println!("{}", frase(s));
        }
        println!("\nENGROSSARAM OU AFINARAM ({}):", mudaram.len());
        for (s, d) in mudaram.iter().take(10) {
            println!("{}   (Δ {:+.2})", frase(s), d);
        }
    }
}

// ---------------------------------------------------------------------------------------
// MAPA DO CÉREBRO — onde estão os neurônios e o que cada um escuta
// ---------------------------------------------------------------------------------------

/// Grupos sensoriais, para saber quanto de cada sentido o cérebro consome.
/// As faixas vêm de onde os canais são ESCRITOS (`motor.rs:455-524, 480-513`).
const GRUPOS: [(&str, &[(usize, usize)]); 6] = [
    ("VISÃO (retina)", &[(0, 15), (128, 191)]),
    ("curvatura à frente", &[(20, 20), (32, 39), (78, 80)]),
    (
        "paredes e bigodes",
        &[(16, 31), (76, 77), (94, 97), (114, 115)],
    ),
    (
        "pista à frente",
        &[(40, 55), (90, 93), (108, 110), (118, 119)],
    ),
    ("telemetria do corpo", &[(56, 75), (124, 125)]),
    ("movimento e TTC", &[(82, 84), (98, 107), (116, 117)]),
];

fn grupo_de(canal: usize) -> &'static str {
    for (nome, faixas) in GRUPOS.iter() {
        if faixas.iter().any(|&(a, b)| canal >= a && canal <= b) {
            return nome;
        }
    }
    "não classificado"
}

/// Quanto de cada sentido o cérebro consome, e onde ele guarda essa escuta.
fn mapa(pesos: &[f32]) {
    let n_neuronios = pesos.len() / SWS;

    println!("\n=== QUANTO DE CADA SENTIDO O CÉREBRO CONSOME ===");
    println!("soma de |peso| por grupo sensorial, sobre as {n_neuronios} unidades da camada 1\n");
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
            // por canal, senão um grupo grande parece importante só por ser grande
            (
                soma / n_canais.max(1) as f32,
                format!(
                    "{:<22} {:>7} canais {:>9.1} total {:>9.2} por canal {:>7.1}%",
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
    println!("\n  Ordenado por peso POR CANAL — senão um grupo grande pareceria importante");
    println!("  só por ter muitos canais. A coluna '%' é a fatia do total, para comparar.");

    println!("\n=== O QUE CADA NEURÔNIO ESCUTA ===");
    println!("para cada unidade: o grupo que domina sua entrada e o canal mais forte\n");
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
                "  n{:<3} {:<14} domina: {:<22} mais forte: {:>+6.2} {}",
                neu,
                neuronio_papel(neu),
                dono,
                melhor_w,
                canal_nome(melhor_ch).unwrap_or("(sem nome)")
            );
        }
    }
    println!("\n  DISTRIBUIÇÃO das {n_neuronios} unidades por sentido dominante:");
    let mut v: Vec<_> = por_grupo.into_iter().collect();
    v.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    for (g, n) in v {
        let barra = "█".repeat((n * 40 / n_neuronios.max(1)).max(1));
        println!("    {:<22} {:>3} unidades  {}", g, n, barra);
    }
}

// ---------------------------------------------------------------------------------------
// A EVOLUÇÃO ACHA O QUE O HUMANO ESCREVEU?
// ---------------------------------------------------------------------------------------

/// Compara o cérebro com as 15 sinapses do priming à mão (`neuro::PRIMING`).
///
/// A pergunta é a da independência: se um cérebro nasce SEM o priming (modo `esparso`) e
/// mesmo assim converge para as mesmas conexões, então o humano só adiantou o que a
/// evolução acharia sozinha. Se converge para outras, a evolução tem um caminho próprio —
/// e vale ver qual. Se não converge para nada, o priming era o cérebro.
fn compara_priming(pesos: &[f32]) {
    use universe::neuro::PRIMING;
    println!("\n=== O QUE O HUMANO ESCREVEU vs O QUE ESTE CÉREBRO TEM ===\n");
    println!(
        "{:>4} {:>7} {:>9}  {:<44} {}",
        "n", "canal", "peso aqui", "canal", "veredito"
    );
    println!("{}", "-".repeat(92));
    let mut presentes = 0;
    for &(neu, ch) in PRIMING.iter() {
        let w = pesos.get(neu * SWS + ch).copied().unwrap_or(0.0);
        let veredito = if w.abs() >= 1.0 {
            presentes += 1;
            "PRESENTE e forte"
        } else if w.abs() >= LIMIAR {
            presentes += 1;
            "presente, fraca"
        } else {
            "ausente"
        };
        println!(
            "{:>4} {:>7} {:>+9.2}  {:<44} {}",
            neu,
            ch,
            w,
            canal_nome(ch).unwrap_or("(sem nome)"),
            veredito
        );
    }
    println!("{}", "-".repeat(92));
    println!(
        "  {presentes}/{} das sinapses escritas à mão estão neste cérebro",
        PRIMING.len()
    );

    // e o que ele tem NOS MOTORES que o humano NÃO escreveu
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
        "\n  CONEXÕES MOTORAS QUE O HUMANO NÃO ESCREVEU: {}",
        proprias.len()
    );
    for (neu, ch, w) in proprias.iter().take(10) {
        println!(
            "    {:>+7.2}  n{} {:<16} ← {}",
            w,
            neu,
            neuronio_papel(*neu),
            canal_nome(*ch).unwrap_or("(sem nome)")
        );
    }
    if proprias.is_empty() {
        println!("    (nenhuma — os motores só têm o que veio de fábrica)");
    }
}
