//! `pistas` — measures the REAL difficulty of each track, instead of trusting the label.
//!
//! The tracks in `world.rs` come labeled in the comments as EASY / MEDIUM / HARD /
//! VERY HARD / EXTREME, and `TrackProfile` has a `complexity` field — which is **never read
//! by anyone** (only written in `config.rs`, `main.rs` and `ui_menu.rs`). In other words:
//! today's curriculum difficulty is human judgment, not measurement.
//!
//! Physics gives the right axis, and it is not the width. From the width sweep measured on
//! 2026-07-30: widening the track from 8 to 24 m changes the distance travelled by 27% and
//! the MORTALITY from 58/60 to 33/60 — width buys margin for error, not speed. What rules
//! is the RADIUS OF THE TIGHTEST CURVE:
//!
//!     v_curva  = sqrt(μ·g·r)                     grip ceiling
//!     d_freio  = (v_reta² − v_curva²) / (2·μ·g)  distance to reach it
//!
//! and `d_freio` decides **which curvature probe still arrives in time** — 20, 40, 70 or 120 m.
//! That is the true cognitive demand: the tighter the curve, the earlier the decision must
//! be made, and the noisier the information available at that horizon.
//!
//! ```text
//! cargo run --release --bin pistas
//! ```

use bevy::math::Vec2;
use universe::world::track_waypoints_for_seed;

const CELULA: f32 = 3.0; // meters per cell (world.rs:57)
const G: f32 = 9.81;
const MU: f32 = 1.0;
/// The car's four curvature probes (channels 20, 78, 79, 80).
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
        let ang = a.dot(b).clamp(-1.0, 1.0).acos(); // radians
        if ang.to_degrees() < 1.0 {
            continue; // straight
        }
        n_curvas += 1;
        ang_max = ang_max.max(ang.to_degrees());
        // radius by the arc approximation, using the SMALLER of the two segments that form
        // the curve: it is the one that limits how much the trajectory can be widened
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

/// Which probe still arrives in time to brake from `v_reta` down to the curve of radius `r`.
fn sonda_necessaria(r: f32, v_reta: f32, largura: f32) -> (f32, f32, String) {
    let v_curva = (MU * G * (r + largura / 2.0)).sqrt();
    if v_reta <= v_curva {
        return (v_curva, 0.0, "none — passes without braking".into());
    }
    let d = (v_reta * v_reta - v_curva * v_curva) / (2.0 * MU * G);
    let s = SONDAS
        .iter()
        .find(|&&x| x >= d)
        .map(|x| format!("{x:.0} m"))
        .unwrap_or_else(|| "NONE — blind".into());
    (v_curva, d, s)
}

/// The track's REAL anticipation demand, curve by curve.
///
/// Assuming a fixed straight-line speed is sloppy: what decides is how much the PREVIOUS
/// stretch allowed accelerating. A tight curve right after another tight curve is easy (one
/// already arrives slowly); the same curve at the end of a long straight is the hard case.
/// Here the track is traversed accumulating speed on the straights with the same grip that
/// holds the curves, and the curve of HIGHEST demand is reported, not the one with the
/// smallest radius.
fn demanda_maxima(p: &[Vec2], largura: f32, v_teto_carro: f32) -> (usize, f32, f32, f32) {
    let n = p.len();
    let mut v = v_teto_carro; // starts optimistic; the first curve corrects it
    let (mut pior_i, mut pior_d, mut pior_r, mut pior_v) = (0usize, 0.0f32, 0.0f32, 0.0f32);
    // two laps: the first settles the entry speed of each stretch, the second measures
    for volta in 0..2 {
        for i in 1..n - 1 {
            let seg_ant = (p[i] - p[i - 1]).length();
            // accelerates on the straight with the same available grip (optimistic limit)
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
                v = v_curva; // exits the curve at its limit
            }
        }
    }
    (pior_i, pior_r, pior_v, pior_d)
}

fn main() {
    // seeds with hand-written layout in world.rs, plus some procedural ones to compare
    let seeds: [(u64, &str); 8] = [
        (123, "Dubai Oval"),
        (64, "Interlagos Gateway"),
        (9001, "Procedural Sigma"),
        (512, "Silverstone Flow"),
        (768, "Nurburgring Hills"),
        (256, "Monaco Tight"),
        (1024, "Suzuka Crossing"),
        (42, "champion's seed"),
    ];
    let largura = 14.0_f32;
    let v_reta = 30.0_f32; // car's CEILING, m/s — each stretch's real speed is computed

    println!("MEASURED TRACK DIFFICULTY (width {largura:.0} m · straight at {v_reta:.0} m/s)\n");
    println!("The EASY/HARD label in the comments is judgment; this here is arithmetic.\n");
    println!(
        "{:<22} {:>8} {:>7} {:>7} {:>8} {:>9} {:>9}  {}",
        "track", "length", "curves", "r_min", "worst r", "entry v", "d_freio", "required probe"
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
                    "NONE — blind".into()
                } else {
                    "none — no braking needed".into()
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
    linhas.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap()); // from easiest to hardest
    for (_, l) in &linhas {
        println!("{l}");
    }
    println!("{}", "-".repeat(104));
    println!("\nSorted by ANTICIPATION DEMAND (braking distance required at the worst curve),");
    println!(
        "not by minimum radius: the same curve is easy after another curve and hard at the end of a straight."
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
    println!("When the required probe exceeds 40 m, the car NEEDS the 70/120 m reading —");
    println!(
        "which is the most anticipated and the noisiest. That is where the brain starts to matter."
    );
}

// ---------------------------------------------------------------------------------------
// CURRICULUM: a ramp of measured difficulty, not a labeled one
// ---------------------------------------------------------------------------------------

/// Builds a sequence of tracks whose anticipation demand climbs step by step, covering the
/// car's four probes. It is what is missing today: the hand-made tracks all live in the
/// 20 m probe range, and the procedural ones jump straight to 40 m — plateau and abyss,
/// no ramp.
fn curriculo(largura: f32, v_teto: f32) {
    use universe::pista_gen::{SONDAS, gerar};
    println!(
        "\n\n=== GENERATED CURRICULUM (width {largura:.0} m · car ceiling {v_teto:.0} m/s) ==="
    );
    println!("each step requires a longer probe than the previous one\n");
    println!(
        "{:>7} {:>9} {:>8} {:>8} {:>9} {:>9}  {}",
        "step", "seed", "length", "curves", "r_min", "d_freio", "requires"
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
                format!("FOUND NO track with d_freio in [{lo:.0}, {hi:.0}] m")
            );
        }
    }
    println!("{}", "-".repeat(84));
    println!("\n{achados}/4 steps filled. The car's probes are at {SONDAS:?} m.");
    if achados < 4 {
        println!("An empty step is NOT a generator failure: the geometry may not reach");
        println!("that demand with the current speed ceiling — and then the path is engine/gear");
        println!("(faster straight), not tighter curve. See the arithmetic in pista_gen.rs.");
    }
}
