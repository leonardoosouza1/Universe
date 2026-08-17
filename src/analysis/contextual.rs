//! Contextual sparsity — *who* fires for *this* situation.
//!
//! # Why it exists
//!
//! Leonardo, delivering the contextual-sparsity protocol: *"you are trying to find
//! something that computes the value of the neuron's weight, right? Why not try other ways,
//! and for each kind of brain part?"*
//!
//! The question I had been chasing — *which neuron matters* — is **refuted** in his
//! document (§5): with the 8 most "important" neurons out of 128 the performance is
//! WORSE than the baseline; ~50% of them are needed to get close to the total. The
//! information is distributed. Selecting a fixed subset does not work; what works is projecting.
//!
//! Our own numbers already said this and I did not read them that way: uniform attention
//! (0.52-0.66% per channel), effective rank 75/96, zero deaf or mute neurons.
//!
//! The distinction that saves the analysis, and that names this module:
//!
//! · **static sparsity** — which neurons are globally important → does NOT exist
//! · **contextual sparsity** — which ones fire *for this tick* → exists and is exploitable
//!
//! # The ruler changes with the arithmetic
//!
//! The brain has six parts with distinct numeric natures, and measuring them all with the
//! same ruler was the previous mistake (singular value applied to the whole brain):
//!
//! | part | arithmetic | what to ask |
//! |---|---|---|
//! | sensory KAN | integer (i32) | how many knots left the identity |
//! | CfC | integer + diagonal `decay = exp(-d·dt)` | persistence per τ, dead zone |
//! | L1 | f32 weights × i8 input ÷127 | σ_max gain, contextual sparsity |
//! | L2 | pure f32 | sparsity, saturation |
//! | L1→output bypass | f32, skips L2 | half the output does not go through L2 |
//! | τ bank | f32 with TIME semantics | log distribution, correlation with persistence |
//!
//! # The four experiments
//!
//! Translated from the protocol (E1-E4), in the execution order he defines — E2 first
//! because, if the real sparsity is low, the rest loses meaning.
//!
//! **Trap #1 of the protocol, respected here:** the non-linearity is `tanh`, which never
//! zeroes exactly. The threshold is an explicit parameter and is **always reported** next
//! to the result; `sensibilidade_ao_limiar` measures how much the conclusion depends on it.

use crate::math::stats;

/// Fraction of neurons active per sample, and the distribution of that fraction.
///
/// E2 of the protocol. Answers: is the brain truly sparse?
#[derive(Debug, Clone)]
pub struct Esparsidade {
    pub camada: String,
    /// Threshold used — ALWAYS reported alongside, because it changes all the numbers.
    pub limiar: f32,
    /// Mean of the active fraction per tick (0..1).
    pub fracao_ativa_media: f32,
    /// Deviation of the active fraction across ticks — variation indicates the active set
    /// responds to context; constant indicates it always fires the same amount.
    pub fracao_ativa_sigma: f32,
    pub p5: f32,
    pub p95: f32,
    /// Fraction of samples in which the layer was saturated (|h| > 0.99) — the opposite of
    /// sparse. L2 measured 57.5% here, which is a defect, not sparsity.
    pub saturacao_media: f32,
}

/// How much the active set persists across consecutive ticks.
///
/// E1 of the protocol — the potentially new idea. In our case there is a theoretical
/// prediction: the CfC is literally a state with inertia (`s = s·decay + x·gain`), so
/// long-τ channels should persist more than short-τ ones. If that shows up, it is the first
/// empirical validation that the τ bank does what it promises.
#[derive(Debug, Clone)]
pub struct Persistencia {
    pub camada: String,
    pub limiar: f32,
    /// |A[t] ∩ A[t-1]| / |A[t]| — mean over the ticks.
    pub persistencia: f32,
    /// Same computation against a RANDOM (non-adjacent) sample. It is the mandatory
    /// baseline: without it, high persistence can just be "the layer always fires".
    pub baseline_aleatorio: f32,
    /// `persistencia - baseline`. The protocol's refutation criterion: below 10 points,
    /// the incremental-predictor idea dies.
    pub ganho_temporal: f32,
    /// Mean fraction of the active set that CHANGES per tick — what an incremental
    /// predictor would need to guess.
    pub churn: f32,
}

/// Excess of co-activation over statistical independence.
///
/// E3 of the protocol. It is the empirical basis for dividing the brain into regions: the
/// groups must COME OUT of here (clusters of neurons that fire together), not from a priori
/// design.
#[derive(Debug, Clone)]
pub struct CoAtivacao {
    pub camada: String,
    pub limiar: f32,
    /// Mean P(i and j active) across pairs.
    pub coativacao_media: f32,
    /// Mean P(i)·P(j) — what would be expected if each decided on its own.
    pub esperado_independente: f32,
    /// Difference. Positive = group structure exists.
    pub excesso: f32,
    /// The most coupled pairs: (i, j, excess).
    pub pares_fortes: Vec<(usize, usize, f32)>,
}

/// Distribution of activation frequency per neuron.
///
/// E4 of the protocol. Long tail = a few hot neurons carry almost everything.
#[derive(Debug, Clone)]
pub struct QuenteFrio {
    pub camada: String,
    pub limiar: f32,
    /// Activation frequency of each neuron (0..1), original order.
    pub frequencia: Vec<f32>,
    /// How many neurons cover 80% of all activations.
    pub neuronios_para_80pct: usize,
    /// Neurons that never fired.
    pub nunca_ativos: usize,
    /// Neurons that fired in more than 95% of ticks — always-on ones are bias, not
    /// sensor: they carry no information about the context.
    pub sempre_ativos: usize,
}

/// Binary activity mask: `|h| > threshold`.
fn mascara(amostra: &[f32], limiar: f32) -> Vec<bool> {
    amostra.iter().map(|v| v.abs() > limiar).collect()
}

/// E2 — real sparsity per layer.
pub fn esparsidade(camada: &str, amostras: &[Vec<f32>], limiar: f32) -> Esparsidade {
    let mut fracoes: Vec<f32> = Vec::with_capacity(amostras.len());
    let mut saturacoes: Vec<f32> = Vec::with_capacity(amostras.len());
    for a in amostras {
        if a.is_empty() {
            continue;
        }
        let ativos = a.iter().filter(|v| v.abs() > limiar).count();
        let sat = a.iter().filter(|v| v.abs() > 0.99).count();
        fracoes.push(ativos as f32 / a.len() as f32);
        saturacoes.push(sat as f32 / a.len() as f32);
    }
    let mut ord = fracoes.clone();
    Esparsidade {
        camada: camada.to_string(),
        limiar,
        fracao_ativa_media: stats::mean(&fracoes),
        fracao_ativa_sigma: stats::std_dev(&fracoes),
        p5: stats::percentile(&mut ord, 5.0),
        p95: stats::percentile(&mut ord.clone(), 95.0),
        saturacao_media: stats::mean(&saturacoes),
    }
}

/// E1 — temporal persistence of the active set.
///
/// `passo_aleatorio` defines the offset used as baseline: comparing A[t] with
/// A[t + step] instead of A[t-1]. A step large enough that there is no temporal
/// correlation, but within the same run (same distribution).
pub fn persistencia(
    camada: &str,
    amostras: &[Vec<f32>],
    limiar: f32,
    passo_aleatorio: usize,
) -> Persistencia {
    if amostras.len() < 2 {
        return Persistencia {
            camada: camada.to_string(),
            limiar,
            persistencia: 0.0,
            baseline_aleatorio: 0.0,
            ganho_temporal: 0.0,
            churn: 0.0,
        };
    }
    let mascaras: Vec<Vec<bool>> = amostras.iter().map(|a| mascara(a, limiar)).collect();

    let sobreposicao = |a: &[bool], b: &[bool]| -> Option<f32> {
        let n_a = a.iter().filter(|x| **x).count();
        if n_a == 0 {
            return None;
        } // with no active set, the ratio is undefined
        let inter = a.iter().zip(b).filter(|(x, y)| **x && **y).count();
        Some(inter as f32 / n_a as f32)
    };

    let mut adj: Vec<f32> = Vec::new();
    let mut ale: Vec<f32> = Vec::new();
    let passo = passo_aleatorio.max(2);
    for t in 1..mascaras.len() {
        if let Some(v) = sobreposicao(&mascaras[t], &mascaras[t - 1]) {
            adj.push(v);
        }
        let outro = (t + passo) % mascaras.len();
        if outro != t && outro != t - 1 {
            if let Some(v) = sobreposicao(&mascaras[t], &mascaras[outro]) {
                ale.push(v);
            }
        }
    }
    let p = stats::mean(&adj);
    let b = stats::mean(&ale);
    Persistencia {
        camada: camada.to_string(),
        limiar,
        persistencia: p,
        baseline_aleatorio: b,
        ganho_temporal: p - b,
        churn: 1.0 - p,
    }
}

/// E3 — co-activation above independence.
///
/// Cost O(n² · samples); with n=96 and thousands of samples it is cheap enough to run
/// offline. `max_pares` only limits the report of strong pairs.
pub fn coativacao(
    camada: &str,
    amostras: &[Vec<f32>],
    limiar: f32,
    max_pares: usize,
) -> CoAtivacao {
    let n = amostras.first().map_or(0, |a| a.len());
    if n == 0 || amostras.is_empty() {
        return CoAtivacao {
            camada: camada.to_string(),
            limiar,
            coativacao_media: 0.0,
            esperado_independente: 0.0,
            excesso: 0.0,
            pares_fortes: Vec::new(),
        };
    }
    let mascaras: Vec<Vec<bool>> = amostras.iter().map(|a| mascara(a, limiar)).collect();
    let total = mascaras.len() as f32;

    // Marginals: P(i active).
    let marginal: Vec<f32> = (0..n)
        .map(|i| {
            mascaras
                .iter()
                .filter(|m| m.get(i).copied().unwrap_or(false))
                .count() as f32
                / total
        })
        .collect();

    let mut soma_conj = 0.0f64;
    let mut soma_indep = 0.0f64;
    let mut pares: Vec<(usize, usize, f32)> = Vec::new();
    let mut n_pares = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let conj = mascaras
                .iter()
                .filter(|m| {
                    m.get(i).copied().unwrap_or(false) && m.get(j).copied().unwrap_or(false)
                })
                .count() as f32
                / total;
            let indep = marginal[i] * marginal[j];
            soma_conj += conj as f64;
            soma_indep += indep as f64;
            n_pares += 1;
            let exc = conj - indep;
            if exc.abs() > 0.05 {
                pares.push((i, j, exc));
            }
        }
    }
    pares.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());
    pares.truncate(max_pares);

    let d = n_pares.max(1) as f64;
    let cm = (soma_conj / d) as f32;
    let ei = (soma_indep / d) as f32;
    CoAtivacao {
        camada: camada.to_string(),
        limiar,
        coativacao_media: cm,
        esperado_independente: ei,
        excesso: cm - ei,
        pares_fortes: pares,
    }
}

/// E4 — hot and cold neurons.
pub fn quente_frio(camada: &str, amostras: &[Vec<f32>], limiar: f32) -> QuenteFrio {
    let n = amostras.first().map_or(0, |a| a.len());
    let total = amostras.len().max(1) as f32;
    let frequencia: Vec<f32> = (0..n)
        .map(|i| {
            amostras
                .iter()
                .filter(|a| a.get(i).map_or(false, |v| v.abs() > limiar))
                .count() as f32
                / total
        })
        .collect();

    // How many neurons cover 80% of ALL activations (not 80% of the neurons).
    let mut ord: Vec<f32> = frequencia.clone();
    ord.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let soma: f32 = ord.iter().sum();
    let alvo = soma * 0.8;
    let mut acc = 0.0f32;
    let mut k = 0usize;
    for f in &ord {
        acc += f;
        k += 1;
        if acc >= alvo {
            break;
        }
    }

    QuenteFrio {
        camada: camada.to_string(),
        limiar,
        nunca_ativos: frequencia.iter().filter(|f| **f == 0.0).count(),
        sempre_ativos: frequencia.iter().filter(|f| **f > 0.95).count(),
        neuronios_para_80pct: k,
        frequencia,
    }
}

/// How much the conclusion depends on the chosen threshold.
///
/// Trap #1 of the protocol: with `tanh` nothing zeroes exactly, so the threshold is a
/// choice, and a choice that changes the result must be exposed. Returns the active
/// fraction for each tested threshold.
pub fn sensibilidade_ao_limiar(amostras: &[Vec<f32>], limiares: &[f32]) -> Vec<(f32, f32)> {
    limiares
        .iter()
        .map(|&l| (l, esparsidade("_", amostras, l).fracao_ativa_media))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn amostras_fixas(padrao: &[&[f32]]) -> Vec<Vec<f32>> {
        padrao.iter().map(|p| p.to_vec()).collect()
    }

    #[test]
    fn esparsidade_conta_acima_do_limiar() {
        let a = amostras_fixas(&[&[1.0, 0.0, 0.0, 0.0], &[1.0, 1.0, 0.0, 0.0]]);
        let e = esparsidade("t", &a, 0.5);
        // 1/4 and 2/4 → mean 0.375.
        assert!((e.fracao_ativa_media - 0.375).abs() < 1e-5);
    }

    #[test]
    fn persistencia_maxima_quando_conjunto_nao_muda() {
        let a = amostras_fixas(&[&[1.0, 0.0], &[1.0, 0.0], &[1.0, 0.0], &[1.0, 0.0]]);
        let p = persistencia("t", &a, 0.5, 2);
        assert!((p.persistencia - 1.0).abs() < 1e-5);
        assert!((p.churn - 0.0).abs() < 1e-5);
    }

    #[test]
    fn persistencia_zero_quando_conjunto_alterna() {
        let a = amostras_fixas(&[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 0.0], &[0.0, 1.0]]);
        let p = persistencia("t", &a, 0.5, 2);
        assert!(p.persistencia < 1e-5, "persistence was {}", p.persistencia);
    }

    /// The baseline is what separates "persists" from "always fires". A layer with ALL
    /// neurons always active has persistence 1.0 — and baseline 1.0 too, so temporal
    /// gain zero. Without the baseline, that layer would look like the ideal case.
    #[test]
    fn ganho_temporal_zero_quando_tudo_sempre_ativo() {
        let a = amostras_fixas(&[&[1.0, 1.0], &[1.0, 1.0], &[1.0, 1.0], &[1.0, 1.0]]);
        let p = persistencia("t", &a, 0.5, 2);
        assert!((p.persistencia - 1.0).abs() < 1e-5);
        assert!(
            p.ganho_temporal.abs() < 1e-5,
            "gain was {}",
            p.ganho_temporal
        );
    }

    #[test]
    fn coativacao_detecta_par_acoplado() {
        // 0 and 1 always together; 2 independent.
        let a = amostras_fixas(&[
            &[1.0, 1.0, 0.0],
            &[0.0, 0.0, 1.0],
            &[1.0, 1.0, 1.0],
            &[0.0, 0.0, 0.0],
        ]);
        let c = coativacao("t", &a, 0.5, 5);
        assert!(c.excesso > 0.0, "excess was {}", c.excesso);
        assert!(c.pares_fortes.iter().any(|(i, j, _)| *i == 0 && *j == 1));
    }

    #[test]
    fn quente_frio_separa_sempre_de_nunca() {
        let a = amostras_fixas(&[&[1.0, 0.0, 0.5], &[1.0, 0.0, 0.0], &[1.0, 0.0, 0.9]]);
        let q = quente_frio("t", &a, 0.4);
        assert_eq!(q.sempre_ativos, 1); // neuron 0
        assert_eq!(q.nunca_ativos, 1); // neuron 1
    }

    #[test]
    fn sensibilidade_cai_com_limiar_maior() {
        let a = amostras_fixas(&[&[0.9, 0.5, 0.1], &[0.8, 0.4, 0.05]]);
        let s = sensibilidade_ao_limiar(&a, &[0.01, 0.45, 0.95]);
        assert!(
            s[0].1 >= s[1].1 && s[1].1 >= s[2].1,
            "fraction should fall: {s:?}"
        );
    }
}
