//! Sonda do CÉREBRO LENTO — ele informa alguma coisa?
//!
//! O lento é `input[48] → CfC(64) → Dense1(64→16) → ReLU → Dense2(16→3) → tanh`, e
//! `motor.rs` consome as três saídas como grandezas DISTINTAS: agressão, teto de
//! acelerador e viés de volante.
//!
//! Medido na primeira passada: as três saídas têm correlação +0,9999 entre si e a
//! primeira componente principal explica 100% da variância — é UM canal com três
//! nomes. E `slow[1] > 0` em 0 de 2000 ticks, ou seja `throttle_cap` fica travado em
//! 1,0 e aquele consumidor não existe na prática.
//!
//! Esta sonda agora testa POR QUE, por ablação pareada: a hipótese é que a fadiga do
//! CfC (`fatigue_beta`) injeta uma deriva monotônica igual em todos os 64 canais, e
//! esse modo comum engole a informação por canal antes do Dense1. Controle = mesmo
//! cérebro, mesma entrada, mesma semente, só com a fadiga desligada.
//!
//! Diagnóstico, não treino: CPU, sem GPU, menos de um segundo.

use universe::engine::SlowBrain;

/// Resultado de uma condição: o que decide é INDEPENDÊNCIA, não amplitude.
struct Measurement {
    stddevs: [f32; 3],
    mean_corr: f32,
    pc1_frac: f32,
    frac_pos1: f32,
}

/// Entrada realista: cada canal oscila num período próprio, como sensores de um ser
/// andando na pista. Períodos distintos por canal ⇒ entrada SEM modo comum; qualquer
/// modo comum na saída foi criado pelo cérebro.
fn make_input(t: usize) -> [i8; 48] {
    let mut inp = [0i8; 48];
    for (i, slot) in inp.iter_mut().enumerate() {
        let period = 40.0 + (i as f32) * 7.0;
        let phase = (t as f32) / period * std::f32::consts::TAU;
        *slot = (phase.sin() * 100.0) as i8;
    }
    inp
}

fn run_condition(fatigue_beta: f32, ticks: usize, verbose: bool) -> Measurement {
    let mut brain = SlowBrain::new();
    brain.cfc.fatigue_beta = fatigue_beta;
    let dt = 1.0 / 64.0;

    let mut history: Vec<[f32; 3]> = Vec::with_capacity(ticks);
    for t in 0..ticks {
        brain.tick(&make_input(t), dt);
        history.push(brain.output);
        if verbose && (t % 400 == 0 || t == ticks - 1) {
            println!(
                "  t={:4}  out = {:+8.5} {:+8.5} {:+8.5}",
                t, brain.output[0], brain.output[1], brain.output[2]
            );
        }
    }

    let means: [f32; 3] = {
        let mut m = [0.0; 3];
        for (k, mk) in m.iter_mut().enumerate() {
            *mk = history.iter().map(|o| o[k]).sum::<f32>() / history.len() as f32;
        }
        m
    };
    let mut stddevs = [0.0f32; 3];
    for (k, d) in stddevs.iter_mut().enumerate() {
        let var = history
            .iter()
            .map(|o| (o[k] - means[k]).powi(2))
            .sum::<f32>()
            / history.len() as f32;
        *d = var.sqrt();
    }

    let mut cov = [[0.0f32; 3]; 3];
    for o in &history {
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += (o[i] - means[i]) * (o[j] - means[j]);
            }
        }
    }
    let corr = |i: usize, j: usize| -> f32 {
        let d = (cov[i][i] * cov[j][j]).sqrt();
        if d < 1e-20 { 0.0 } else { cov[i][j] / d }
    };
    let mean_corr = (corr(0, 1).abs() + corr(0, 2).abs() + corr(1, 2).abs()) / 3.0;

    // Fração da variância na 1ª direção principal (iteração de potência, 3×3).
    let trace = cov[0][0] + cov[1][1] + cov[2][2];
    let mut v = [1.0f32, 0.3, -0.5];
    for _ in 0..300 {
        let mut nv = [0.0f32; 3];
        for i in 0..3 {
            for j in 0..3 {
                nv[i] += cov[i][j] * v[j];
            }
        }
        let n = (nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2])
            .sqrt()
            .max(1e-20);
        v = [nv[0] / n, nv[1] / n, nv[2] / n];
    }
    let mut av = [0.0f32; 3];
    for i in 0..3 {
        for j in 0..3 {
            av[i] += cov[i][j] * v[j];
        }
    }
    let lambda = av[0] * v[0] + av[1] * v[1] + av[2] * v[2];

    Measurement {
        stddevs,
        mean_corr,
        pc1_frac: lambda / trace.max(1e-20),
        frac_pos1: history.iter().filter(|o| o[1] > 0.0).count() as f32 / history.len() as f32,
    }
}

fn main() {
    // ---- Estatística dos pesos de nascimento ----
    // Antes de qualquer dinâmica: o init é simétrico? Um init de sinal único transforma
    // cada camada num somador com sinal fixo — todas as saídas viram a MESMA soma.
    {
        let b = SlowBrain::new();
        for (nome, w) in [
            ("dense1.weights", &b.dense1.weights),
            ("dense2.weights", &b.dense2.weights),
        ] {
            let neg = w.iter().filter(|&&v| v < 0.0).count();
            let mn = w.iter().cloned().fold(f32::MAX, f32::min);
            let mx = w.iter().cloned().fold(f32::MIN, f32::max);
            let media = w.iter().sum::<f32>() / w.len() as f32;
            println!(
                "{:<16} n={:<5} negativos={:>5}/{:<5} ({:>5.1}%)  min={:+.4} max={:+.4} media={:+.4}",
                nome,
                w.len(),
                neg,
                w.len(),
                100.0 * neg as f32 / w.len() as f32,
                mn,
                mx,
                media
            );
        }
        println!();
    }

    let ticks = 4000;
    println!("=== COM fadiga (fatigue_beta = 0.8, como está no código) ===");
    let with_fatigue = run_condition(0.8, ticks, true);
    println!("\n=== SEM fadiga (fatigue_beta = 0.0) — controle pareado ===");
    let without_fatigue = run_condition(0.0, ticks, true);

    println!("\n{:-<74}", "");
    println!(
        "{:<34} {:>18} {:>18}",
        "métrica", "with_fatigue fadiga", "without_fatigue fadiga"
    );
    println!("{:-<74}", "");
    for k in 0..3 {
        println!(
            "{:<34} {:>18.6} {:>18.6}",
            format!("desvio out{}", k),
            with_fatigue.stddevs[k],
            without_fatigue.stddevs[k]
        );
    }
    println!(
        "{:<34} {:>18.4} {:>18.4}",
        "|correlação| média entre saídas", with_fatigue.mean_corr, without_fatigue.mean_corr
    );
    println!(
        "{:<34} {:>17.1}% {:>17.1}%",
        "variância na 1ª componente",
        100.0 * with_fatigue.pc1_frac,
        100.0 * without_fatigue.pc1_frac
    );
    println!(
        "{:<34} {:>17.1}% {:>17.1}%",
        "slow[1] > 0 (teto de acelerador)",
        100.0 * with_fatigue.frac_pos1,
        100.0 * without_fatigue.frac_pos1
    );
    println!("{:-<74}", "");

    println!("\nLeitura:");
    println!(
        "  graus de liberdade efetivos — with_fatigue fadiga ≈ {:.2}, without_fatigue fadiga ≈ {:.2}  (de 3)",
        1.0 / with_fatigue.pc1_frac.max(1e-9),
        1.0 / without_fatigue.pc1_frac.max(1e-9)
    );
    if without_fatigue.mean_corr < with_fatigue.mean_corr - 0.05 {
        println!("  ⇒ a FADIGA é a causa do modo comum: desligá-la separa as saídas.");
    } else {
        println!("  ⇒ a fadiga NÃO explica o modo comum — a causa está antes dela.");
    }
}
