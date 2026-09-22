//! Per-neuron activation metrics + runtime guards (E4).
//!
//! REFACTOR_PLAN.md §E4 / MATH_INVENTORY.md gaps 1-4, 6, 12: the project had ZERO
//! visibility into what happens INSIDE the fast brain (L1=96, L2=48) — only the final
//! output (`final_intent`) and the 3 outputs of the slow one (`participation_ratio` in
//! `main.rs`). This module closes that gap: it receives samples already read from the
//! GPU (the caller, `gpu_evolve/context.rs`, is the one that knows how to read
//! `buf_cfc`/`buf_intent`) and returns per-neuron statistics + alarms ready for the
//! report.
//!
//! # Why the thresholds are not duplicated here
//!
//! The original REFACTOR_PLAN proposed a `pub mod thresholds` LOCAL to this file.
//! E1 already created `crate::math::thresholds` with the same thresholds (different
//! names, same function) — having two places with the "dead neuron threshold" is
//! exactly the kind of duplication that produces silent divergence (one place changes,
//! the other does not). This module imports `math::thresholds` as the single source.
//!
//! # Two cutoffs, two levels
//!
//! There is a distinction worth making explicit because the names are similar:
//! - `DEAD_EPS`/`SATURATION_CUTOFF` — PER-SAMPLE cutoff (is this `h` at this tick
//!   dead/saturated?). Produces `NeuronStats::dead_pct`/`saturated_pct`, a continuous
//!   fraction per neuron.
//! - `DEAD_NEURONS_WARN_PCT`/`SATURATED_NEURONS_WARN_PCT` — PER-LAYER cutoff (what
//!   fraction of the layer's NEURONS is consistently dead/saturated?). A neuron only
//!   enters that count if its own per-sample fraction exceeds
//!   `NEURON_DEAD_CLASSIFY_PCT`/`NEURON_SATURATED_CLASSIFY_PCT` (see the comment on
//!   the constants) — not stacking the two levels would make ANY neuron that died
//!   once, however rarely, count as "dead" in the layer fraction.

use crate::math::{linalg, quant, stats, thresholds};

/// Bins of the layer's aggregated histogram (REFACTOR_PLAN.md §E4: "32-bin histogram").
pub const HISTOGRAM_BINS: usize = 32;

/// PER-SAMPLE saturation cutoff: |h| > 0.99 is considered saturated.
///
/// Distinct from `thresholds::SATURATED_NEURONS_WARN_PCT` (fraction of the LAYER
/// saturated that triggers an alarm). `tanh(2.65) ≈ 0.99` — MATH_INVENTORY.md gap 3.
const SATURATION_CUTOFF: f32 = 0.99;

/// A neuron only enters the "dead neurons of the layer" count if it is dead in at
/// least this fraction of the samples — not the first time it crosses zero. 90% is
/// conservative: even a channel that "rests" much of the time (e.g. a rare-event
/// detector) stays out of this classification.
const NEURON_DEAD_CLASSIFY_PCT: f32 = 90.0;
/// Same for saturation: a neuron stuck at the ceiling in ≥90% of the samples.
const NEURON_SATURATED_CLASSIFY_PCT: f32 = 90.0;

/// Statistics of one neuron, collected over a window of samples (see
/// `collect_activation_stats` for the exact semantics of "sample" — it mixes time and
/// population, it is not a single-being trajectory).
#[derive(Debug, Clone)]
pub struct NeuronStats {
    pub neuron_id: usize,
    pub mean: f32,
    pub sigma: f32,
    pub p5: f32,
    pub p50: f32,
    pub p95: f32,
    /// Fraction of samples with `|h| < DEAD_EPS` — 0.0 to 100.0.
    pub dead_pct: f32,
    /// Fraction of samples with `|h| > 0.99` — 0.0 to 100.0.
    pub saturated_pct: f32,
    /// SNR of quantizing this neuron to i8 at activation scale (127), in dB.
    pub snr_i8: f32,
}

/// Aggregated statistics of a whole layer.
#[derive(Debug, Clone)]
pub struct LayerStats {
    pub layer_name: String,
    /// Number of sample rows used (after discarding NaN/Inf).
    pub n_samples: usize,
    pub neurons: Vec<NeuronStats>,
    /// Largest singular value of the layer's input weights (0.0 if `w_rec` was not
    /// provided). See `collect_activation_stats` for how it is computed from a
    /// rectangular matrix (L1 is not square: 192→96).
    /// `None` when the layer had no weight matrix delivered — not to be confused with zero.
    pub spectral_radius: Option<f32>,
    /// Effective rank (95% of the spectral energy) of the same weights. 0 if `w_rec` absent.
    /// `None` when not measured (see [`Self::spectral_radius`]).
    pub effective_rank: Option<usize>,
    /// Policy entropy (nats), only when `policy_actions` was passed.
    pub policy_entropy: Option<f32>,
    /// Aggregated histogram (all samples of all neurons), 32 bins.
    pub histogram: Vec<u32>,
}

/// Kind of threshold violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlarmKind {
    DeadNeurons,
    Saturated,
    SpectralRadiusHigh,
    SnrLow,
    NanDetected,
    PolicyEntropyLow,
}

/// Alarm: threshold violation detected this generation.
#[derive(Debug, Clone)]
pub struct ActivationAlarm {
    pub layer: String,
    pub kind: AlarmKind,
    pub value: f32,
    pub threshold: f32,
}

/// Singular-value spectrum of `w` (`n_neurons × in_dim`, row-major) via
/// Gram + Jacobi: spectral radius (largest singular value, by power iteration — cheap
/// and it is what matters for the instability alarm) and effective rank (needs the
/// whole spectrum, hence the full Jacobi).
///
/// Returns `(0.0, 0)` if `w` does not have the expected shape.
fn spectral_summary(w: &[f32], n_neurons: usize) -> (f32, usize) {
    if n_neurons == 0 || w.is_empty() || w.len() % n_neurons != 0 {
        return (0.0, 0);
    }
    let in_dim = w.len() / n_neurons;
    if in_dim == 0 {
        return (0.0, 0);
    }
    // `linalg::largest_singular_value` already does gram+power-iteration+sqrt (single
    // source — it is the SAME computation that `gpu::evolution::Genome::random` uses to
    // normalize at initialization, P0.5). The effective rank needs the whole spectrum,
    // so the Gram is recomputed here one extra time for `eigenvalues_symmetric`; L1
    // matrices (96×192) make that negligible next to the cost of the rest of the report.
    let spectral_radius = linalg::largest_singular_value(w, n_neurons, in_dim);
    let gram = linalg::gram_matrix(w, n_neurons, in_dim);
    let eigs = linalg::eigenvalues_symmetric(&gram, n_neurons);
    let mut singular_vals: Vec<f32> = eigs.iter().map(|&e| e.max(0.0).sqrt()).collect();
    singular_vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let effective_rank = linalg::effective_rank(&singular_vals, 0.95);

    (spectral_radius, effective_rank)
}

/// Shannon entropy (nats) of the joint distribution of (throttle, steer, brake),
/// binned into 5×5×5 = 125 cells over `[0,1] × [-1,1] × [0,1]`.
///
/// Binning the THREE outputs together (not one entropy per channel) is intentional: a
/// pinned steering wheel with a varying throttle would still be "low entropy" for what
/// matters (the car always takes the same turn), and looking only at the steering
/// channel would hide that if the other two varied.
fn policy_entropy_nats(actions: &[[f32; 3]]) -> f32 {
    const BINS: usize = 5;
    if actions.is_empty() {
        return 0.0;
    }
    let mut counts = std::collections::HashMap::<(usize, usize, usize), u32>::new();
    for a in actions {
        let bt = ((a[0].clamp(0.0, 1.0) * BINS as f32) as usize).min(BINS - 1);
        let bs = (((a[1].clamp(-1.0, 1.0) + 1.0) * 0.5 * BINS as f32) as usize).min(BINS - 1);
        let bb = ((a[2].clamp(0.0, 1.0) * BINS as f32) as usize).min(BINS - 1);
        *counts.entry((bt, bs, bb)).or_insert(0) += 1;
    }
    let total = actions.len() as f32;
    let mut h = 0.0f32;
    for &c in counts.values() {
        let p = c as f32 / total;
        h -= p * p.ln();
    }
    h
}

/// Collects activation statistics from a set of samples.
///
/// `samples[k]` is a row of `n_neurons` values — EACH ROW may come from a different
/// being and/or a different tick. This module does not distinguish the two: the
/// per-neuron statistics (mean, percentiles, %dead, %saturated) are over the SET of
/// samples, mixing population and time, in the same spirit in which `[lento]`
/// (main.rs) already measures `participation_ratio` over samples of several beings at
/// several ticks. Rows with NaN/Inf are discarded BEFORE getting here (see the runtime
/// guard in `gpu_evolve/context.rs::tick()`), so this function assumes `samples` is
/// already clean — but it still audits and reports an alarm if something slipped
/// through (double layer of defence, negligible cost).
///
/// `w_rec`: the layer's input weights, row-major `[n_neurons × in_dim]` (the name
/// "recurrent" is inherited from the original REFACTOR_PLAN — L1 is not recurrent in
/// the RNN sense, but the sum "CfC(channels 0..96) ++ KAN(channels 96..192) → L1"
/// plays the same role W_rec would in a stability analysis: it is the matrix whose
/// largest singular value decides whether the signal amplifies or attenuates). `None`
/// when the layer has no input weights accessible here (e.g. SlowBrain, whose `W`
/// lives only on the GPU).
///
/// `policy_actions`: (throttle, steer, brake) per sample, in the SAME order/stride as
/// the rows of `samples` — used only for `policy_entropy`; pass `None` for layers that
/// are not the motor output.
pub fn collect_activation_stats(
    layer_name: &str,
    samples: &[Vec<f32>],
    w_rec: Option<&[f32]>,
    n_neurons: usize,
    policy_actions: Option<&[[f32; 3]]>,
) -> (LayerStats, Vec<ActivationAlarm>) {
    let mut alarms: Vec<ActivationAlarm> = Vec::new();

    // Double defence: any NaN/Inf that slipped past the caller's filter is reported
    // (alarm) and the row is excluded from the statistics — without this a single bad
    // sample would contaminate the whole layer's mean/percentile/histogram.
    let mut clean: Vec<&Vec<f32>> = Vec::with_capacity(samples.len());
    let mut n_nan_rows = 0usize;
    for row in samples {
        if row.iter().any(|v| !v.is_finite()) {
            n_nan_rows += 1;
        } else {
            clean.push(row);
        }
    }
    if n_nan_rows > 0 {
        alarms.push(ActivationAlarm {
            layer: layer_name.to_string(),
            kind: AlarmKind::NanDetected,
            value: n_nan_rows as f32,
            threshold: 0.0,
        });
    }

    let n_samples = clean.len();
    let mut neurons: Vec<NeuronStats> = Vec::with_capacity(n_neurons);
    let mut all_vals: Vec<f32> = Vec::with_capacity(n_samples * n_neurons.max(1));

    for j in 0..n_neurons {
        let col: Vec<f32> = clean.iter().filter_map(|row| row.get(j).copied()).collect();
        all_vals.extend_from_slice(&col);

        let mean = stats::mean(&col);
        let sigma = stats::std_dev(&col);
        let mut for_percentile = col.clone();
        let p5 = stats::percentile(&mut for_percentile, 5.0);
        let p50 = stats::percentile(&mut for_percentile, 50.0);
        let p95 = stats::percentile(&mut for_percentile, 95.0);
        let dead_pct = if col.is_empty() {
            0.0
        } else {
            100.0
                * col
                    .iter()
                    .filter(|&&v| v.abs() < thresholds::DEAD_EPS)
                    .count() as f32
                / col.len() as f32
        };
        let saturated_pct = if col.is_empty() {
            0.0
        } else {
            100.0 * col.iter().filter(|&&v| v.abs() > SATURATION_CUTOFF).count() as f32
                / col.len() as f32
        };
        let snr_i8 = quant::snr_db_act(&col);

        neurons.push(NeuronStats {
            neuron_id: j,
            mean,
            sigma,
            p5,
            p50,
            p95,
            dead_pct,
            saturated_pct,
            snr_i8,
        });
    }

    let histogram = stats::histogram(&all_vals, HISTOGRAM_BINS);

    // `None` = the layer had no weight matrix delivered to the analysis. That is NOT the
    // same as "zero spectral radius": this branch used to return (0.0, 0) and the report
    // printed "raio_espectral=0.000 rank_efetivo=0/48", which reads as total collapse of
    // the layer. It cost an inverted conclusion on 2026-08-13 (L2 looked dead when it was
    // merely unmeasured). Absence of measurement now travels as an Option all the way to
    // the report, which prints n/a.
    let (spectral_radius, effective_rank) = match w_rec {
        Some(w) => {
            let (r, k) = spectral_summary(w, n_neurons);
            (Some(r), Some(k))
        }
        None => (None, None),
    };

    let policy_entropy = policy_actions.and_then(|a| {
        if a.is_empty() {
            None
        } else {
            Some(policy_entropy_nats(a))
        }
    });

    // ── Layer alarms (fraction of NEURONS, not of samples — see the module doc) ──
    if n_neurons > 0 {
        let dead_frac_layer = 100.0
            * neurons
                .iter()
                .filter(|s| s.dead_pct >= NEURON_DEAD_CLASSIFY_PCT)
                .count() as f32
            / n_neurons as f32;
        if dead_frac_layer > thresholds::DEAD_NEURONS_WARN_PCT {
            alarms.push(ActivationAlarm {
                layer: layer_name.to_string(),
                kind: AlarmKind::DeadNeurons,
                value: dead_frac_layer,
                threshold: thresholds::DEAD_NEURONS_WARN_PCT,
            });
        }
        let sat_frac_layer = 100.0
            * neurons
                .iter()
                .filter(|s| s.saturated_pct >= NEURON_SATURATED_CLASSIFY_PCT)
                .count() as f32
            / n_neurons as f32;
        if sat_frac_layer > thresholds::SATURATED_NEURONS_WARN_PCT {
            alarms.push(ActivationAlarm {
                layer: layer_name.to_string(),
                kind: AlarmKind::Saturated,
                value: sat_frac_layer,
                threshold: thresholds::SATURATED_NEURONS_WARN_PCT,
            });
        }
    }

    // Only alarms about what was actually measured: a layer with no matrix delivered
    // raises no alarm at all (neither a false positive of "zero", nor a silent false
    // negative).
    if let Some(sr) = spectral_radius {
        if sr > thresholds::SPECTRAL_MAX {
            alarms.push(ActivationAlarm {
                layer: layer_name.to_string(),
                kind: AlarmKind::SpectralRadiusHigh,
                value: sr,
                threshold: thresholds::SPECTRAL_MAX,
            });
        }
    }

    if let Some(min_snr) = neurons
        .iter()
        .map(|s| s.snr_i8)
        .filter(|v| v.is_finite())
        .fold(None, |acc: Option<f32>, v| {
            Some(acc.map_or(v, |a| a.min(v)))
        })
    {
        if min_snr < thresholds::SNR_I8_MIN {
            alarms.push(ActivationAlarm {
                layer: layer_name.to_string(),
                kind: AlarmKind::SnrLow,
                value: min_snr,
                threshold: thresholds::SNR_I8_MIN,
            });
        }
    }

    if let Some(pe) = policy_entropy {
        if pe < thresholds::POLICY_ENTROPY_MIN {
            alarms.push(ActivationAlarm {
                layer: layer_name.to_string(),
                kind: AlarmKind::PolicyEntropyLow,
                value: pe,
                threshold: thresholds::POLICY_ENTROPY_MIN,
            });
        }
    }

    (
        LayerStats {
            layer_name: layer_name.to_string(),
            n_samples,
            neurons,
            spectral_radius,
            effective_rank,
            policy_entropy,
            histogram,
        },
        alarms,
    )
}

/// Prints the stats report to stderr (does not block, does not lock a Mutex — each
/// call only reads the `Vec`/`struct` already assembled by `collect_activation_stats`).
pub fn print_stats_report(stats: &LayerStats, alarms: &[ActivationAlarm]) {
    eprintln!(
        "[ativacao:{}] samples={} neurons={} spectral_radius={} effective_rank={} policy_entropy={}",
        stats.layer_name,
        stats.n_samples,
        stats.neurons.len(),
        stats
            .spectral_radius
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "n/a".to_string()),
        stats
            .effective_rank
            .map(|k| format!("{k}/{}", stats.neurons.len()))
            .unwrap_or_else(|| "n/a (weights not delivered)".to_string()),
        stats
            .policy_entropy
            .map(|v| format!("{v:.3}"))
            .unwrap_or_else(|| "n/a".to_string()),
    );
    for a in alarms {
        eprintln!(
            "[ativacao] ALARM {} {:?}: value={:.3} threshold={:.3}",
            a.layer, a.kind, a.value, a.threshold,
        );
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn linha(vs: &[f32]) -> Vec<f32> {
        vs.to_vec()
    }

    #[test]
    fn neuronio_morto_detectado() {
        // 3 neurons, 10 samples: neuron 0 always ~0 (dead), 1 and 2 vary.
        let mut samples = Vec::new();
        for i in 0..10 {
            let t = i as f32 / 9.0;
            samples.push(linha(&[1e-6, t, -t]));
        }
        let (stats, _alarms) = collect_activation_stats("teste", &samples, None, 3, None);
        assert!(
            stats.neurons[0].dead_pct > 99.0,
            "neuron 0 should be ~100% dead, {}",
            stats.neurons[0].dead_pct
        );
        assert!(stats.neurons[1].dead_pct < 50.0);
    }

    #[test]
    fn neuronio_saturado_detectado() {
        let mut samples = Vec::new();
        for _ in 0..10 {
            samples.push(linha(&[0.999, 0.0]));
        }
        let (stats, alarms) = collect_activation_stats("teste", &samples, None, 2, None);
        assert!(stats.neurons[0].saturated_pct > 99.0);
        assert!(
            alarms.iter().any(|a| a.kind == AlarmKind::Saturated),
            "a layer with 1/2 saturated neurons (50% > 30% warn) should alarm: {:?}",
            alarms
        );
    }

    #[test]
    fn nan_na_amostra_vira_alarme_e_e_excluido() {
        let samples = vec![
            linha(&[0.1, 0.2]),
            linha(&[f32::NAN, 0.2]),
            linha(&[0.3, 0.4]),
        ];
        let (stats, alarms) = collect_activation_stats("teste", &samples, None, 2, None);
        assert_eq!(
            stats.n_samples, 2,
            "the row with NaN must be excluded from the count"
        );
        assert!(alarms.iter().any(|a| a.kind == AlarmKind::NanDetected));
    }

    #[test]
    fn camada_vazia_nao_gera_pane() {
        let (stats, alarms) = collect_activation_stats("vazia", &[], None, 4, None);
        assert_eq!(stats.n_samples, 0);
        assert_eq!(stats.neurons.len(), 4);
        assert!(alarms.iter().all(|a| a.kind != AlarmKind::NanDetected));
    }

    #[test]
    fn spectral_summary_matriz_ortogonal_normalizada() {
        // W = I (4×4): singular values all = 1.0 → spectral radius = 1.0,
        // effective rank = 4 (energy spread equally).
        let n = 4usize;
        let mut w = vec![0.0f32; n * n];
        for i in 0..n {
            w[i * n + i] = 1.0;
        }
        let (sr, er) = spectral_summary(&w, n);
        assert!(
            (sr - 1.0).abs() < 0.05,
            "expected spectral radius ~1.0, got {sr}"
        );
        assert_eq!(er, n);
    }

    #[test]
    fn spectral_summary_matriz_retangular_nao_quadrada() {
        // W (2×4): two identical rows orthogonal to two zero columns.
        // rows=2 < cols=4 — tests the "L1 is not square" path (192→96 in practice).
        let w = vec![2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let (sr, er) = spectral_summary(&w, 2);
        assert!(
            (sr - 3.0).abs() < 0.05,
            "expected largest singular value 3.0, got {sr}"
        );
        assert_eq!(er, 2);
    }

    #[test]
    fn spectral_radius_alto_dispara_alarme() {
        // A row with norm 10 guarantees a singular value > SPECTRAL_MAX(1.0).
        let mut w = vec![0.0f32; 3 * 3];
        w[0] = 10.0;
        let samples = vec![linha(&[0.0, 0.0, 0.0])];
        let (stats, alarms) = collect_activation_stats("teste", &samples, Some(&w), 3, None);
        assert!(stats.spectral_radius.unwrap() > thresholds::SPECTRAL_MAX);
        assert!(alarms
            .iter()
            .any(|a| a.kind == AlarmKind::SpectralRadiusHigh));
    }

    #[test]
    fn entropia_politica_baixa_com_acao_constante() {
        let acoes: Vec<[f32; 3]> = (0..50).map(|_| [0.8_f32, 0.63, 0.0]).collect();
        let mut samples = Vec::new();
        for _ in 0..50 {
            samples.push(linha(&[0.0]));
        }
        let (stats, alarms) = collect_activation_stats("L2", &samples, None, 1, Some(&acoes));
        let pe = stats
            .policy_entropy
            .expect("must compute entropy when policy_actions is Some");
        assert!(
            pe < 0.05,
            "a constant action must have entropy ~0, got {pe}"
        );
        assert!(alarms.iter().any(|a| a.kind == AlarmKind::PolicyEntropyLow));
    }

    #[test]
    fn entropia_politica_alta_com_acao_uniforme() {
        // Sweeps throttle/steer/brake across the whole range — entropy must stay well
        // above the threshold (POLICY_ENTROPY_MIN = 0.5 nats).
        let mut acoes = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    let t = i as f32 / 4.0;
                    let s = (j as f32 / 4.0) * 2.0 - 1.0;
                    let b = k as f32 / 4.0;
                    acoes.push([t, s, b]);
                }
            }
        }
        let samples: Vec<Vec<f32>> = (0..acoes.len()).map(|_| linha(&[0.0])).collect();
        let (stats, _alarms) = collect_activation_stats("L2", &samples, None, 1, Some(&acoes));
        let pe = stats.policy_entropy.unwrap();
        assert!(
            pe > thresholds::POLICY_ENTROPY_MIN,
            "125 uniform distinct actions must have entropy > {}, got {}",
            thresholds::POLICY_ENTROPY_MIN,
            pe
        );
    }

    #[test]
    fn histograma_tem_32_bins_e_soma_total() {
        let samples: Vec<Vec<f32>> = (0..20)
            .map(|i| linha(&[(i as f32 / 19.0) * 2.0 - 1.0]))
            .collect();
        let (stats, _) = collect_activation_stats("teste", &samples, None, 1, None);
        assert_eq!(stats.histogram.len(), HISTOGRAM_BINS);
        assert_eq!(stats.histogram.iter().sum::<u32>(), 20);
    }

    #[test]
    fn sem_w_rec_raio_e_rank_ficam_zero() {
        let samples = vec![linha(&[0.1, 0.2])];
        let (stats, alarms) = collect_activation_stats("SlowBrain", &samples, None, 2, None);
        // No matrix delivered: absence of measurement, not zero (see the field doc).
        assert_eq!(stats.spectral_radius, None);
        assert_eq!(stats.effective_rank, None);
        assert!(!alarms
            .iter()
            .any(|a| a.kind == AlarmKind::SpectralRadiusHigh));
    }
}
