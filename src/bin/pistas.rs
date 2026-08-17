//! `pistas` — mede a dificuldade REAL de cada pista, em vez de confiar no rótulo.
//!
//! As pistas de `world.rs` vêm rotuladas nos comentários como EASY / MEDIUM / HARD /
//! VERY HARD / EXTREME, e o `TrackProfile` tem um campo `complexity` — que **nunca é lido
//! por ninguém** (só escrito em `config.rs`, `main.rs` e `ui_menu.rs`). Ou seja: a
//! dificuldade do currículo hoje é julgamento humano, não medida.
//!
//! A física dá o eixo certo, e não é a largura. Da varredura de largura medida em
//! 2026-07-30: alargar a pista de 8 para 24 m muda a distância percorrida em 27% e a
//! MORTALIDADE de 58/60 para 33/60 — largura compra margem de erro, não velocidade. O que
//! manda é o RAIO DA CURVA MAIS FECHADA:
//!
//!     v_curva  = sqrt(μ·g·r)                     teto de aderência
//!     d_freio  = (v_reta² − v_curva²) / (2·μ·g)  distância para chegar nele
//!
//! e `d_freio` decide **qual sonda de curvatura ainda chega a tempo** — 20, 40, 70 ou 120 m.
//! É essa a demanda cognitiva de verdade: quanto mais fechada a curva, mais cedo é preciso
//! decidir, e mais ruidosa é a informação disponível naquele horizonte.
//!
//! ```text
//! cargo run --release --bin pistas
//! ```

use bevy::math::Vec2;
use universe::world::track_waypoints_for_seed;

const CELULA: f32 = 3.0; // metros por célula (world.rs:57)
const G: f32 = 9.81;
const MU: f32 = 1.0;
/// As quatro sondas de curvatura do carro (canais 20, 78, 79, 80).
const SONDAS: [f32; 4] = [20.0, 40.0, 70.0, 120.0];

struct Medida {
    comprimento: f32,
    r_min: f32,
    n_curvas: usize,
    angulo_max: f32,
}

fn medir(wps: &[Vec2]) -> Medida {
    let p: Vec<Vec2> = wps.iter().map(|w| *w * CELULA).collect();
    let n = p.len();
    let mut comprimento = 0.0;
    let (mut r_min, mut n_curvas, mut ang_max) = (f32::INFINITY, 0usize, 0.0f32);

    for i in 0..n - 1 {
        let seg = (p[i + 1] - p[i]).length();
        comprimento += seg;
        if i == 0 {
            continue;
        }
        let a = (p[i] - p[i - 1]).normalize_or_zero();
        let b = (p[i + 1] - p[i]).normalize_or_zero();
        let ang = a.dot(b).clamp(-1.0, 1.0).acos(); // radianos
        if ang.to_degrees() < 1.0 {
            continue; // reta
        }
        n_curvas += 1;
        ang_max = ang_max.max(ang.to_degrees());
        // raio pela aproximação de arco, usando o MENOR dos dois segmentos que formam a
        // curva: é ele que limita o quanto dá para abrir a trajetória
        let seg_menor = (p[i] - p[i - 1]).length().min((p[i + 1] - p[i]).length());
        let r = seg_menor / ang;
        r_min = r_min.min(r);
    }
    Medida {
        comprimento,
        r_min,
        n_curvas,
        angulo_max: ang_max,
    }
}

/// Qual sonda ainda chega a tempo para frear de `v_reta` até a curva de raio `r`.
fn sonda_necessaria(r: f32, v_reta: f32, largura: f32) -> (f32, f32, String) {
    let v_curva = (MU * G * (r + largura / 2.0)).sqrt();
    if v_reta <= v_curva {
        return (v_curva, 0.0, "nenhuma — dá para passar sem frear".into());
    }
    let d = (v_reta * v_reta - v_curva * v_curva) / (2.0 * MU * G);
    let s = SONDAS
        .iter()
        .find(|&&x| x >= d)
        .map(|x| format!("{x:.0} m"))
        .unwrap_or_else(|| "NENHUMA — cega".into());
    (v_curva, d, s)
}

/// A demanda REAL de antecipação da pista, curva a curva.
///
/// Supor uma velocidade de reta fixa é frouxo: o que decide é quanto o trecho ANTERIOR
/// deixou acelerar. Uma curva fechada logo depois de outra curva fechada é fácil (já se
/// chega devagar); a mesma curva no fim de uma reta longa é o caso duro. Aqui se percorre a
/// pista acumulando velocidade nas retas com a mesma aderência que segura as curvas, e
/// reporta-se a curva de MAIOR exigência, não a de menor raio.
fn demanda_maxima(p: &[Vec2], largura: f32, v_teto_carro: f32) -> (usize, f32, f32, f32) {
    let n = p.len();
    let mut v = v_teto_carro; // começa otimista; a primeira curva corrige
    let (mut pior_i, mut pior_d, mut pior_r, mut pior_v) = (0usize, 0.0f32, 0.0f32, 0.0f32);
    // duas voltas: a primeira acerta a velocidade de entrada de cada trecho, a segunda mede
    for volta in 0..2 {
        for i in 1..n - 1 {
            let seg_ant = (p[i] - p[i - 1]).length();
            // acelera na reta com a mesma aderência disponível (limite otimista)
            v = (v * v + 2.0 * MU * G * seg_ant).sqrt().min(v_teto_carro);
            let a = (p[i] - p[i - 1]).normalize_or_zero();
            let b = (p[i + 1] - p[i]).normalize_or_zero();
            let ang = a.dot(b).clamp(-1.0, 1.0).acos();
            if ang.to_degrees() < 1.0 {
                continue;
            }
            let r = seg_ant.min((p[i + 1] - p[i]).length()) / ang;
            let v_curva = (MU * G * (r + largura / 2.0)).sqrt();
            if v > v_curva {
                let d = (v * v - v_curva * v_curva) / (2.0 * MU * G);
                if volta == 1 && d > pior_d {
                    pior_d = d;
                    pior_i = i;
                    pior_r = r;
                    pior_v = v;
                }
                v = v_curva; // sai da curva no limite dela
            }
        }
    }
    (pior_i, pior_r, pior_v, pior_d)
}

fn main() {
    // seeds com traçado escrito à mão em world.rs, mais alguns procedurais para comparar
    let seeds: [(u64, &str); 8] = [
        (123, "Dubai Oval"),
        (64, "Interlagos Gateway"),
        (9001, "Procedural Sigma"),
        (512, "Silverstone Flow"),
        (768, "Nurburgring Hills"),
        (256, "Monaco Tight"),
        (1024, "Suzuka Crossing"),
        (42, "semente do campeão"),
    ];
    let largura = 14.0_f32;
    let v_reta = 30.0_f32; // TETO do carro, m/s — a velocidade real de cada trecho é calculada

    println!("DIFICULDADE MEDIDA DAS PISTAS (largura {largura:.0} m · reta a {v_reta:.0} m/s)\n");
    println!("O rótulo EASY/HARD dos comentários é julgamento; isto aqui é conta.\n");
    println!(
        "{:<22} {:>8} {:>7} {:>7} {:>8} {:>9} {:>9}  {}",
        "pista", "compr.", "curvas", "r_min", "r da pior", "v entrada", "d_freio", "sonda exigida"
    );
    println!("{}", "-".repeat(104));

    let mut linhas: Vec<(f32, String)> = Vec::new();
    for (seed, nome) in seeds {
        let wps = track_waypoints_for_seed(seed);
        if wps.len() < 3 {
            continue;
        }
        let m = medir(&wps);
        let p: Vec<Vec2> = wps.iter().map(|w| *w * CELULA).collect();
        let (i_pior, r_pior, v_ent, d_real) = demanda_maxima(&p, largura, v_reta);
        let sonda = SONDAS
            .iter()
            .find(|&&x| x >= d_real)
            .map(|x| format!("{x:.0} m"))
            .unwrap_or_else(|| {
                if d_real > 0.0 {
                    "NENHUMA — cega".into()
                } else {
                    "nenhuma — não precisa frear".into()
                }
            });
        linhas.push((
            d_real,
            format!(
                "{:<22} {:>7.0}m {:>7} {:>7.1}m {:>7.1}m {:>8.1}m/s {:>8.1}m  {}",
                nome, m.comprimento, m.n_curvas, m.r_min, r_pior, v_ent, d_real, sonda
            ),
        ));
    }
    linhas.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap()); // do mais fácil ao mais difícil
    for (_, l) in &linhas {
        println!("{l}");
    }
    println!("{}", "-".repeat(104));
    println!(
        "\nOrdenado pela DEMANDA DE ANTECIPAÇÃO (distância de frenagem exigida na pior curva),"
    );
    println!(
        "não pelo raio mínimo: a mesma curva é fácil depois de outra curva e dura no fim de uma reta."
    );
    if std::env::args().any(|a| a == "--curriculo") {
        let v = std::env::args()
            .position(|a| a == "--vteto")
            .and_then(|i| std::env::args().nth(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(40.0);
        curriculo(largura, v);
        return;
    }
    println!("Quando a sonda exigida passa de 40 m, o carro PRECISA da leitura de 70/120 m —");
    println!("que é a mais antecipada e a mais ruidosa. É aí que o cérebro passa a importar.");
}

// ---------------------------------------------------------------------------------------
// CURRÍCULO: uma rampa de dificuldade medida, não rotulada
// ---------------------------------------------------------------------------------------

/// Monta uma sequência de pistas cuja demanda de antecipação sobe degrau a degrau, cobrindo
/// as quatro sondas do carro. É o que falta hoje: as pistas à mão vivem todas na faixa da
/// sonda de 20 m, e as procedurais pulam direto para 40 m — platô e abismo, sem rampa.
fn curriculo(largura: f32, v_teto: f32) {
    use universe::pista_gen::{SONDAS, gerar};
    println!(
        "\n\n=== CURRÍCULO GERADO (largura {largura:.0} m · teto do carro {v_teto:.0} m/s) ==="
    );
    println!("cada degrau exige uma sonda mais longa que o anterior\n");
    println!(
        "{:>7} {:>9} {:>8} {:>8} {:>9} {:>9}  {}",
        "degrau", "semente", "compr.", "curvas", "r_min", "d_freio", "exige"
    );
    println!("{}", "-".repeat(84));

    let faixas: [(f32, f32); 4] = [(2.0, 18.0), (22.0, 38.0), (45.0, 68.0), (75.0, 115.0)];
    let mut achados = 0;
    for (degrau, (lo, hi)) in faixas.iter().enumerate() {
        let mut ok = false;
        for s in 1u64..=60 {
            if let Some((_, d)) = gerar(s * 7919, *lo, *hi, largura, v_teto, 300) {
                let sonda = d.sonda().map(|x| format!("{x:.0} m")).unwrap_or("—".into());
                println!(
                    "{:>7} {:>9} {:>7.0}m {:>8} {:>8.1}m {:>8.1}m  {} ({})",
                    degrau + 1,
                    s * 7919,
                    d.comprimento,
                    d.n_curvas,
                    d.r_min,
                    d.d_freio,
                    sonda,
                    d.rotulo()
                );
                ok = true;
                achados += 1;
                break;
            }
        }
        if !ok {
            println!(
                "{:>7} {:>9} {:>50}",
                degrau + 1,
                "—",
                format!("NÃO ACHEI pista com d_freio em [{lo:.0}, {hi:.0}] m")
            );
        }
    }
    println!("{}", "-".repeat(84));
    println!("\n{achados}/4 degraus preenchidos. As sondas do carro estão a {SONDAS:?} m.");
    if achados < 4 {
        println!("Degrau vazio NÃO é falha do gerador: pode ser que a geometria não alcance");
        println!("aquela demanda com o teto de velocidade atual — e aí o caminho é motor/marcha");
        println!("(reta mais rápida), não curva mais fechada. Ver a conta em pista_gen.rs.");
    }
}
