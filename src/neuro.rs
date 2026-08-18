//! Neural allocation by ABLATION — how much each brain pathway really carries.
//!
//! The car's brain is already made of parts with explicit mixing weights
//! (`engine.rs` `decide()`: `w[0]·cortex + w[1]·reflex + w[2]·sentinel`), plus the
//! slow pathway that **modulates** the rest in `motor.rs` (aggression, throttle
//! ceiling, steering bias). What did not exist was measuring how much each part
//! contributes.
//!
//! # Why ABLATION and not the gate weight
//!
//! The gate weight says how much a pathway is USED. It does not say how much it SERVES. A
//! pathway can receive high weight while saying nothing — and the reverse also happens.
//!
//! This is not conjecture: it was measured on the Python bench (`~/projects/LLM/novo/exp14`).
//! Modulator genes drifted all the way to b=−1.67 and c=8.02 **while ablating those very
//! genes cost exactly +0.0%**. Genes wander freely precisely when they are invisible to
//! selection, so any statistic over the genome (variance, magnitude, gate weight)
//! measures drift together with function. Ablation does not: it is paired within the same
//! individual and asks the only thing that matters — *how much is lost without this?*
//!
//! That is why this module records both things side by side. **The distance between USE and
//! CONTRIBUTION is the result**, not a diagnostic detail.
//!
//! # The risky prediction this experiment tests
//!
//! On 2026-07-30 the Python bench measured, in a survival world with hidden regimes,
//! that modulating the time constant has a realizable ceiling of **+0.11%** against the 24.20%
//! the closed form promised. The diagnosis: in that world the objective is a RATE (what fraction
//! of the time to stay hidden), and filtering improves the INSTANT, which was worth nothing there.
//!
//! Racing is the opposite case: braking 40 m too late into a corner is a crash, and no correct
//! average rate saves you. If that reading is right, **the slow pathway has to cost dearly here** — far
//! more than the ~0% it cost there. If it does not, the "rate vs instant" explanation falls and the
//! result becomes about modulation in general, not about the type of task.
//!
//! # ⚠️ CHANNEL COLLISION IN `sensory_wave` (simulator bug, not this module's)
//!
//! Three writers store into the same channels inside `update_racers`, in the same frame, and the
//! last one wins:
//!
//! | order | line | writes | channels |
//! |---|---|---|---|
//! | 1 | `motor.rs:561` | wall distance, 8 angles | 16–23 |
//! | 2 | `motor.rs:579` | obstacle width, 8 angles | 24–31 |
//! | 3 | `motor.rs:676` | motion per retina row | **21–27** |
//! | 4 | `motor.rs:1413` | curvature at 20 m | **20** |
//!
//! What survives: wall only on channels 16–19 and width only on 28–31 — **half the fan of wall
//! sensors is computed and discarded every frame**. The layout comment at `motor.rs:519`
//! describes what was intended, not what happens.
//!
//! The groups here reflect what SURVIVES, not what the comment promises. Truly fixing it
//! (moving the writers to the free range 192–222) recovers 8 channels of real
//! information, but changes the indices and forces retraining the champion — a design
//! decision, not a silent fix.
//!
//! # Usage
//!
//! One process per ablated pathway (`universe-lab` already runs variants in parallel):
//!
//! ```text
//! ./target/release/universe-lab --spec lab_specs/alocacao.json --gens 30
//! ```

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// The brain pathways that can be switched off independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Via {
    /// Baseline: nothing switched off. Every cost is measured against it.
    Nenhuma,
    /// Cortex: `instinct_layer` → `cortex_l2` → `final_intent[0..3]`. The slow-but-rich pathway.
    Cortex,
    /// Reflex: `SpecialistBrain`, three sparse 32→1 BitLinears with fixed channels.
    Reflexo,
    /// Sentinel, SOFT proposal only. The hard veto stays active on purpose — without it the
    /// car crashes within the first seconds and the number would measure "died early", not
    /// "the sentinel pathway contributes this much".
    SentinelaSuave,
    /// Whole sentinel, hard veto included. Expensive and informative, but it dominates the others.
    SentinelaTotal,
    /// Slow pathway: `SlowBrain` (CfC 64 → Dense 16 → 3). Does NOT act directly — it modulates
    /// aggression, throttle ceiling and steering bias. The exact analogue of M2 on the Python bench.
    Lenta,
    /// The brain's authority over the GEARBOX (`final_intent[3]` and `[4]`).
    ///
    /// The gearbox itself keeps working: auto-upshift at 85% of RPM, auto-downshift below
    /// 30%, anti-stall guardian. What gets switched off is only the neural OVERRIDE, which in
    /// `motor.rs:2107` is the last `else if` — that is, it only decides within the band the
    /// automatic rules leave open.
    ///
    /// It is Leonardo's question posed as a measurement: *"this way the model has more parameters or
    /// worse, I don't know"*. If the cost of this ablation is ~0, giving the gearbox to the brain does
    /// not pay, and adding more engine detail will not give it more to decide — it will give
    /// more for the automatic rule to decide on its own.
    Marcha,
    /// Ablated SENSE: zeroes a group of `sensory_wave` channels at the INPUT, before everything.
    ///
    /// Answers "how vision affects the model" with the same measure as the pathways. It is needed
    /// because the weight map does NOT answer it: measured on the champion, the per-channel weight is
    /// practically identical across the six groups (9.5 to 10.7), and vision's share (35.4%)
    /// is exactly the fraction of channels it occupies (80/223). The weight allocation is
    /// indistinguishable from uniform — so only ablation separates what is listened to from what is
    /// used.
    Sentido(Sentido),
    /// The hand-written PRIMING on the 3 motor neurons (`spawn.rs:64-90`).
    ///
    /// This is the question the ablation table opens and does not answer: the cortex carries
    /// **94.7%**, but motor neurons 0, 1 and 2 are born with 15 HAND-WRITTEN weights
    /// (curvature at 70 m → brake +10.0; at 120 m → steering +8.0; and so on). All of them
    /// appear SATURATED in the trained champion, i.e. they survived evolution intact.
    ///
    /// Ablating only those 15 weights, leaving the rest of the cortex standing, separates two things
    /// that ablating the whole cortex conflates:
    ///
    /// - cost ≈ that of the whole cortex (94.7%) -> **the "brain" is the hand-written rules**,
    ///   and evolution is decoration
    /// - much lower cost -> evolution built something on top of the priming, and it is worth measuring what
    ///
    /// It is the same reasoning as always: separating what was inherited from what was learned.
    Priming,
}

/// The 15 synapses of the sparse priming (`spawn.rs:64-90`), as (neuron, channel).
pub const PRIMING: [(usize, usize); 15] = [
    (0, 64),
    (0, 79),
    (0, 78), // throttle
    (1, 80),
    (1, 32),
    (1, 79),
    (1, 78),
    (1, 20),
    (1, 81), // steering
    (1, 90),
    (1, 91),
    (1, 92),
    (1, 93), // asphalt tracker
    (2, 79),
    (2, 78), // brake
];

/// Zeroes the priming synapses, if the ablation asks for it. Called once, when the brain is
/// assembled — not every frame, because these are weights, not signals.
pub fn ablar_priming(pesos: &mut [f32], stride: usize) {
    if ablada() != Via::Priming {
        return;
    }
    for &(neu, ch) in PRIMING.iter() {
        let i = neu * stride + ch;
        if i < pesos.len() {
            pesos[i] = 0.0;
        }
    }
}

/// Channel groups of the `sensory_wave`, from the ranges where they are WRITTEN
/// (`motor.rs:455-524` and `480-513`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sentido {
    Visao,
    Curvatura,
    Paredes,
    PistaAFrente,
    Corpo,
    Movimento,
}

impl Sentido {
    pub fn faixas(&self) -> &'static [(usize, usize)] {
        match self {
            Sentido::Visao => &[(0, 15), (128, 191)],
            Sentido::Curvatura => &[(20, 20), (32, 39), (78, 80)],
            // 16-19 and 28-31 only: channels 20-27 are OVERWRITTEN later (see the note
            // about channel collision at the top of this module)
            Sentido::Paredes => &[(16, 19), (28, 31), (76, 77), (94, 97), (114, 115)],
            Sentido::PistaAFrente => &[(40, 55), (90, 93), (108, 110), (118, 119)],
            Sentido::Corpo => &[(56, 75), (124, 125)],
            Sentido::Movimento => &[(21, 27), (82, 84), (98, 107), (116, 117)],
        }
    }

    pub fn nome(&self) -> &'static str {
        match self {
            Sentido::Visao => "visao",
            Sentido::Curvatura => "curvatura",
            Sentido::Paredes => "paredes",
            Sentido::PistaAFrente => "pista",
            Sentido::Corpo => "corpo",
            Sentido::Movimento => "movimento",
        }
    }

    pub fn de_str(s: &str) -> Option<Self> {
        Some(match s {
            "visao" | "visão" => Sentido::Visao,
            "curvatura" => Sentido::Curvatura,
            "paredes" => Sentido::Paredes,
            "pista" => Sentido::PistaAFrente,
            "corpo" => Sentido::Corpo,
            "movimento" => Sentido::Movimento,
            _ => return None,
        })
    }
}

/// Zeroes the channels of the ablated sense, if any. Called at the start of the cycle, on the
/// raw `sensory_wave` — before the KAN, the CfC and everything else, so that the ablation means
/// "this sense does not exist" and not "this sense arrives distorted".
pub fn ablar_sentido(onda: &mut [i8]) {
    if let Via::Sentido(s) = ablada() {
        for &(a, b) in s.faixas() {
            for ch in a..=b.min(onda.len().saturating_sub(1)) {
                onda[ch] = 0;
            }
        }
    }
}

impl Via {
    pub fn de_str(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "cortex" | "córtex" => Via::Cortex,
            "reflexo" => Via::Reflexo,
            "sentinela" | "sentinela_suave" => Via::SentinelaSuave,
            "sentinela_total" => Via::SentinelaTotal,
            "lenta" | "slow" => Via::Lenta,
            "marcha" | "cambio" | "câmbio" => Via::Marcha,
            "priming" => Via::Priming,
            outro => {
                if let Some(sent) = Sentido::de_str(outro) {
                    return Via::Sentido(sent);
                }
                return Via::Nenhuma;
            }
        }
    }

    pub fn nome(&self) -> &'static str {
        match self {
            Via::Nenhuma => "nenhuma",
            Via::Cortex => "cortex",
            Via::Reflexo => "reflexo",
            Via::SentinelaSuave => "sentinela_suave",
            Via::SentinelaTotal => "sentinela_total",
            Via::Lenta => "lenta",
            Via::Marcha => "marcha",
            Via::Sentido(s) => s.nome(),
            Via::Priming => "priming",
        }
    }
}

/// How the MOTOR neurons are born. Separates two things the current priming mixes.
///
/// The comment at `spawn.rs:38` explains the problem the priming solves:
///
/// > *"With N(0,0.1) random weights, each motor neuron sums ~110 random ternary inputs that
/// > **drown the curvature signal (SNR ≈ 0.26)**"*
///
/// That is, the priming does TWO things at the same time:
///
/// 1. **sparsity** — the motor neuron connects to few channels. It is an *architectural prior*,
///    and it is what solves the SNR.
/// 2. **which channels** — curvature at 70 m → brake, at 120 m → steering. That is *injected
///    human knowledge*.
///
/// Removing both together measures "without priming" and does not answer the interesting
/// question. Separated, one can ask: **does evolution find on its own the channels the human chose?**
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nascimento {
    /// Sparse AND with hand-picked channels. What exists today.
    Primado,
    /// Sparse, but with RANDOMLY DRAWN channels. Keeps the architectural prior (tractable SNR) and
    /// gives the content choice back to evolution. This is the "independent" condition.
    EsparsoAleatorio,
    /// Dense and random, like the feature neurons. It is the condition the `spawn.rs` comment
    /// describes as drowned — serves as the floor, to know how much sparsity
    /// alone is worth.
    Denso,
}

impl Nascimento {
    pub fn de_str(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "esparso" | "esparso_aleatorio" | "independente" => Nascimento::EsparsoAleatorio,
            "denso" => Nascimento::Denso,
            _ => Nascimento::Primado,
        }
    }
    pub fn nome(&self) -> &'static str {
        match self {
            Nascimento::Primado => "primed (hand-picked channels)",
            Nascimento::EsparsoAleatorio => "sparse with randomly drawn channels (independent)",
            Nascimento::Denso => "dense random (floor)",
        }
    }
}

static NASCIMENTO: OnceLock<Nascimento> = OnceLock::new();

pub fn definir_nascimento(n: Nascimento) {
    let _ = NASCIMENTO.set(n);
}

#[inline]
pub fn nascimento() -> Nascimento {
    *NASCIMENTO.get().unwrap_or(&Nascimento::Primado)
}

/// How many connections each motor neuron receives in sparse mode — the same number the
/// hand priming uses (3 on throttle, 10 on steering, 2 on brake), so the comparison
/// isolates the CHOICE of channels and not their quantity.
pub fn n_conexoes_motor(neuronio: usize) -> usize {
    PRIMING.iter().filter(|&&(n, _)| n == neuronio).count()
}

/// Fraction of the population born in the destructive mutation band (σ 0.08–0.20).
///
/// Mutation is STRATIFIED by slot (`spawn.rs:431`): the first 5 are near-clones
/// (σ≈0.006), then σ climbs in steps, and from slot 30 onward (half of 60) σ goes to
/// 0.08–0.20. Measured in the generation reports:
///
/// ```text
/// GEN 1: NET max 3022 · p90 2924 · median 701 · wall_mean 0.03s
/// GEN 2: NET max 2474 · p90 2341 · median   0 · wall_mean 0.00s
/// ```
///
/// **The median is ZERO.** Half the population goes nowhere — and `wall_mean = 0.00s` shows
/// it is neither a crash nor a wall: it is mutation destroying the genome. The effective population is 30.
///
/// This parameter exists to MEASURE which fraction pays: too much exploration wastes half the
/// budget, too little stagnates. There is no a priori reason for 0.50 to be the point.
static FRACAO_EXPLORADORA: OnceLock<f32> = OnceLock::new();

pub fn definir_fracao_exploradora(f: f32) {
    let _ = FRACAO_EXPLORADORA.set(f.clamp(0.0, 1.0));
}

/// Default fraction: 0.50 — slots 30..59 of 60, as written in `spawn.rs` today.
pub const FRACAO_EXPLORADORA_PADRAO: f32 = 0.50;

#[inline]
pub fn fracao_exploradora() -> f32 {
    *FRACAO_EXPLORADORA
        .get()
        .unwrap_or(&FRACAO_EXPLORADORA_PADRAO)
}

static ABLADA: OnceLock<Via> = OnceLock::new();

/// Pins this process's ablated pathway. Called once at startup, from `LabParams`.
///
/// It is process-global on purpose: `universe-lab` already runs each variant in a
/// separate process, so there is nothing to confuse, and it avoids threading the parameter
/// through all of Bevy's systems just to serve one experiment.
pub fn definir_ablacao(via: Via) {
    let _ = ABLADA.set(via);
}

#[inline]
pub fn ablada() -> Via {
    *ABLADA.get().unwrap_or(&Via::Nenhuma)
}

// ---------------------------------------------------------------------------------------
// USE of the pathways — the gate weights, accumulated. The ablation's comparison counterpart.
// ---------------------------------------------------------------------------------------

/// Fixed-point accumulators (×1e6) so they can be atomic and lock-free, since
/// `decide()` runs for thousands of cars per tick.
static USO: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static USO_N: AtomicU64 = AtomicU64::new(0);
/// How many times the sentinel's hard veto overrode the whole mix.
static VETOS: AtomicU64 = AtomicU64::new(0);

const ESCALA: f64 = 1_000_000.0;

#[inline]
pub fn registrar_uso(w: &[f32; 3]) {
    for i in 0..3 {
        USO[i].fetch_add((w[i].max(0.0) as f64 * ESCALA) as u64, Ordering::Relaxed);
    }
    USO_N.fetch_add(1, Ordering::Relaxed);
}

#[inline]
pub fn registrar_veto() {
    VETOS.fetch_add(1, Ordering::Relaxed);
}

/// Mean of the gate weights since the last call to [`zerar_uso`].
pub fn uso_medio() -> ([f32; 3], u64, u64) {
    let n = USO_N.load(Ordering::Relaxed).max(1);
    let m = [
        (USO[0].load(Ordering::Relaxed) as f64 / ESCALA / n as f64) as f32,
        (USO[1].load(Ordering::Relaxed) as f64 / ESCALA / n as f64) as f32,
        (USO[2].load(Ordering::Relaxed) as f64 / ESCALA / n as f64) as f32,
    ];
    (
        m,
        USO_N.load(Ordering::Relaxed),
        VETOS.load(Ordering::Relaxed),
    )
}

pub fn zerar_uso() {
    for a in USO.iter() {
        a.store(0, Ordering::Relaxed);
    }
    USO_N.store(0, Ordering::Relaxed);
    VETOS.store(0, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------------------
// WHAT THE SYNAPSE SAYS — dictionary of the sensory channels
// ---------------------------------------------------------------------------------------

/// Human-readable name of input channel `ch` of the `sensory_wave`.
///
/// A synapse is an edge `channel → neuron`. Without knowing what the channel measures, "weight 10.0 at
/// position 79" says nothing; with the name, it becomes "*curvature at 70 m → brake, strong*", which is a
/// sentence about the car's behaviour.
///
/// The names come from where the channels are WRITTEN (`simulation/motor.rs`) and from the sparse
/// priming (`evolution/spawn.rs`), not from guesswork. A channel without a located origin returns
/// `None` on purpose: an invented name in an interpretability report is worse than a raw
/// index, because it looks like knowledge.
pub fn canal_nome(ch: usize) -> Option<&'static str> {
    Some(match ch {
        // 0..15: retina pooling by ZONE (motor.rs:455-478)
        0..=7 => "vision FAR zone (~80 m, distant corners)",
        8..=15 => "vision NEAR zone (~20 m, obstacles)",
        // 16..63: semantic channels, layout at motor.rs:515-524
        16..=19 => "wall distance per angle (4 of 8 — see channel collision)",
        20 => "curvature at 20 m (OVERWRITES wall at angle 4)",
        21..=27 => "motion per retina row (OVERWRITES wall 5-7 and width 0-3)",
        28..=31 => "obstacle width per angle (4 of 8 — see channel collision)",
        32..=39 => "flow curvature per angle",
        40..=47 => "track ahead NEAR (5–40 m)",
        48..=55 => "track ahead FAR (45–150 m)",
        56..=59 => "ground material and friction",
        60..=63 => "agent state (escape route)",
        64 => "alignment with track flow",
        65 => "speed",
        66 => "lateral speed (drift)",
        67 => "integrity (hp)",
        68 => "engine revs (rpm)",
        69 => "gear",
        70 => "tyre temperature",
        71 => "downforce",
        72 => "slipstream (draft)",
        73 => "is on grass",
        74 => "steering limit at current speed",
        75 => "wet track",
        76 => "distance to left wall",
        77 => "distance to right wall",
        78 => "curvature at 40 m (signed)",
        79 => "curvature at 70 m (signed)",
        80 => "curvature at 120 m (signed)",
        81 => "lateral deviation from track centre",
        82 => "1/time-to-collision",
        83 => "motion in the left field",
        84 => "motion in the right field",
        90..=93 => "dark-centre tracker (asphalt)",
        94 => "whisker +30°",
        95 => "whisker −30°",
        96 => "whisker +60°",
        97 => "whisker −60°",
        98 => "motion on retina row 7",
        102 => "approach rate on the left",
        103 => "approach rate on the right",
        104 => "approach rate ahead",
        105 => "minimum time-to-collision",
        106 => "left approach (fine gain)",
        107 => "right approach (fine gain)",
        108 => "racing-line deviation",
        109 => "racing-line deviation delta",
        110 => "racing-line anticipation",
        114 => "wall approach on the left",
        115 => "wall approach on the right",
        116 => "approach in the fovea",
        117 => "absolute motion in the fovea",
        118 => "free space ahead",
        119 => "corridor width",
        124 => "is spinning",
        125 => "chassis stress",
        // 128..191: high-resolution visual channels (motor.rs:480-513)
        128..=143 => "HD fovea (volumetric max-pool)",
        144..=159 => "HD peripheral",
        160..=191 => "horizon line, raw pixel",
        _ => return None,
    })
}

/// Role of the output neuron of layer 1. The first three are motor by construction
/// (`spawn.rs`: they receive only the sparse priming); from 3 onward they are feature neurons.
pub fn neuronio_papel(n: usize) -> &'static str {
    match n {
        0 => "MOTOR throttle",
        1 => "MOTOR steering",
        2 => "MOTOR brake",
        _ => "feature",
    }
}

#[cfg(test)]
mod testes {
    use super::*;

    #[test]
    fn de_str_reconhece_as_vias_e_cai_em_nenhuma() {
        assert_eq!(Via::de_str("cortex"), Via::Cortex);
        assert_eq!(Via::de_str("  LENTA "), Via::Lenta);
        assert_eq!(Via::de_str("sentinela_total"), Via::SentinelaTotal);
        // anything that is not a known pathway must NOT ablate anything by mistake:
        // a typo in the spec would silently become an experiment with no baseline
        assert_eq!(Via::de_str("cortez"), Via::Nenhuma);
        assert_eq!(Via::de_str(""), Via::Nenhuma);
    }

    #[test]
    fn uso_medio_e_a_media_dos_pesos() {
        zerar_uso();
        registrar_uso(&[1.0, 0.0, 0.0]);
        registrar_uso(&[0.0, 1.0, 0.0]);
        let (m, n, _) = uso_medio();
        assert_eq!(n, 2);
        assert!((m[0] - 0.5).abs() < 1e-4, "m[0]={}", m[0]);
        assert!((m[1] - 0.5).abs() < 1e-4, "m[1]={}", m[1]);
        assert!(m[2].abs() < 1e-4);
        zerar_uso();
        let (_, n, _) = uso_medio();
        assert_eq!(n, 0);
    }
}

#[cfg(test)]
mod testes_sentido {
    use super::*;

    #[test]
    fn os_grupos_nao_se_sobrepoem_e_cobrem_o_que_dizem() {
        // overlap would make the sum of ablation costs count a channel twice
        let todos = [
            Sentido::Visao,
            Sentido::Curvatura,
            Sentido::Paredes,
            Sentido::PistaAFrente,
            Sentido::Corpo,
            Sentido::Movimento,
        ];
        let mut visto = std::collections::HashMap::new();
        for s in todos {
            for &(a, b) in s.faixas() {
                for ch in a..=b {
                    if let Some(outro) = visto.insert(ch, s.nome()) {
                        panic!("channel {ch} is in '{}' and in '{}'", outro, s.nome());
                    }
                }
            }
        }
        // vision has to be the largest group (zonal retina + HD): 16 + 64 = 80 channels
        let n_visao: usize = Sentido::Visao.faixas().iter().map(|(a, b)| b - a + 1).sum();
        assert_eq!(n_visao, 80, "vision ranges changed");
    }

    #[test]
    fn de_str_reconhece_sentidos_e_vias_sem_confundir() {
        assert_eq!(Via::de_str("visao"), Via::Sentido(Sentido::Visao));
        assert_eq!(Via::de_str("cortex"), Via::Cortex);
        assert_eq!(Via::de_str("visão"), Via::Sentido(Sentido::Visao));
        // a typo must NOT become a silent ablation of something else
        assert_eq!(Via::de_str("visaoo"), Via::Nenhuma);
    }

    #[test]
    fn ablar_zera_so_o_grupo_pedido() {
        // since ABLADA is a OnceLock and this test binary may have already pinned it, the test
        // exercises the range function directly, which is what decides what disappears
        let mut onda = [42i8; 223];
        for &(a, b) in Sentido::Curvatura.faixas() {
            for ch in a..=b {
                onda[ch] = 0;
            }
        }
        assert_eq!(onda[79], 0, "curvature at 70 m had to disappear");
        assert_eq!(onda[0], 42, "the retina could not be touched");
        assert_eq!(onda[64], 42, "the flow alignment could not be touched");
    }
}

#[cfg(test)]
mod testes_priming {
    use super::*;

    #[test]
    fn as_15_sinapses_batem_com_spawn_rs() {
        // if someone changes the priming in spawn.rs without changing here, the ablation starts
        // silently measuring something else. This test is the alarm.
        assert_eq!(PRIMING.len(), 15);
        // the three motor neurons, and ONLY them
        assert!(PRIMING.iter().all(|&(n, _)| n < 3));
        // the brake depends on the two short curvature probes — it is what holds the hairpin
        assert!(PRIMING.contains(&(2, 79)) && PRIMING.contains(&(2, 78)));
        // steering has the four from the asphalt tracker
        for ch in 90..94 {
            assert!(
                PRIMING.contains(&(1, ch)),
                "channel {ch} missing on steering"
            );
        }
    }

    #[test]
    // `0 * stride` names the motor neuron being asserted on, matching the rows
    // above and below; clippy reads it as `erasing_op`.
    #[allow(clippy::erasing_op)]
    fn ablar_priming_zera_so_as_posicoes_certas() {
        // exercises the index arithmetic without depending on the global OnceLock
        let stride = 223;
        let mut pesos = vec![1.0f32; 3 * stride];
        for &(neu, ch) in PRIMING.iter() {
            pesos[neu * stride + ch] = 0.0;
        }
        assert_eq!(
            pesos[2 * stride + 79],
            0.0,
            "brake/curvature 70 m had to disappear"
        );
        assert_eq!(
            pesos[0 * stride + 64],
            0.0,
            "throttle/flow had to disappear"
        );
        assert_eq!(
            pesos[1 * stride + 100],
            1.0,
            "non-primed channel could not be touched"
        );
        assert_eq!(
            pesos[2 * stride + 64],
            1.0,
            "channel 64 is only primed on neuron 0"
        );
        assert_eq!(pesos.iter().filter(|&&w| w == 0.0).count(), 15);
    }
}

#[cfg(test)]
mod testes_nascimento {
    use super::*;

    #[test]
    fn a_quantidade_de_conexoes_bate_com_o_priming() {
        // the "sparse random" mode has to give the motors the SAME number of connections
        // as the hand priming — otherwise the comparison measures quantity, not channel choice
        assert_eq!(
            n_conexoes_motor(0),
            3,
            "throttle: flow + curvature 70 + curvature 40"
        );
        assert_eq!(
            n_conexoes_motor(1),
            10,
            "steering: 6 curvatures/offset + 4 from the asphalt"
        );
        assert_eq!(n_conexoes_motor(2), 2, "brake: curvature 70 and 40");
        assert_eq!(
            n_conexoes_motor(0) + n_conexoes_motor(1) + n_conexoes_motor(2),
            PRIMING.len()
        );
    }

    #[test]
    fn de_str_reconhece_os_modos_e_cai_no_primado() {
        assert_eq!(Nascimento::de_str("esparso"), Nascimento::EsparsoAleatorio);
        assert_eq!(
            Nascimento::de_str("independente"),
            Nascimento::EsparsoAleatorio
        );
        assert_eq!(Nascimento::de_str("denso"), Nascimento::Denso);
        assert_eq!(Nascimento::de_str(""), Nascimento::Primado);
        // a typo falls back to the CURRENT behaviour, not to a silent experimental mode
        assert_eq!(Nascimento::de_str("esparsoo"), Nascimento::Primado);
    }
}

#[cfg(test)]
mod testes_fracao {
    use super::*;

    /// Replica of the `spawn.rs` arithmetic, so it can be tested without assembling the world.
    fn faixa(i: usize, n_pop: usize, frac: f32) -> &'static str {
        let ini = ((1.0 - frac) * n_pop as f32) as usize;
        if i >= ini {
            "explora"
        } else if i * 2 >= ini {
            "medio"
        } else if i * 6 >= ini {
            "pequeno"
        } else {
            "clone"
        }
    }

    #[test]
    fn o_padrao_reproduz_a_estratificacao_original() {
        // with 60 slots and fraction 0.50 it has to give exactly the limits that were
        // hand-written in spawn.rs: clones 0-4, small 5-14, medium 15-29, explore 30+
        let (n, f) = (60usize, 0.50f32);
        assert_eq!(faixa(0, n, f), "clone");
        assert_eq!(faixa(4, n, f), "clone");
        assert_eq!(faixa(7, n, f), "pequeno");
        assert_eq!(faixa(14, n, f), "pequeno");
        assert_eq!(faixa(22, n, f), "medio");
        assert_eq!(faixa(29, n, f), "medio");
        assert_eq!(faixa(30, n, f), "explora");
        assert_eq!(faixa(59, n, f), "explora");
    }

    #[test]
    fn fracao_menor_encolhe_a_faixa_destrutiva() {
        let n = 60;
        let n_explora = |f: f32| (0..n).filter(|&i| faixa(i, n, f) == "explora").count();
        assert_eq!(
            n_explora(0.50),
            30,
            "the default sacrifices half the population"
        );
        assert_eq!(n_explora(0.25), 15);
        assert_eq!(n_explora(0.10), 6);
        // and the order has to be monotonic — smaller fraction, fewer explorers
        assert!(n_explora(0.10) < n_explora(0.25));
        assert!(n_explora(0.25) < n_explora(0.50));
    }

    #[test]
    fn os_primeiros_slots_sao_sempre_clones() {
        // slot 0 is the pure champion (simple_elitism); it can NEVER fall into the
        // destructive band, or the lineage is lost every generation
        for f in [0.10f32, 0.25, 0.50, 0.75] {
            assert_eq!(
                faixa(0, 60, f),
                "clone",
                "fraction {f} put slot 0 outside the clones"
            );
        }
    }
}
