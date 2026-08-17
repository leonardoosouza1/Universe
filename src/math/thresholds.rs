/// Named thresholds for activation analysis and model health.
///
/// ALL of the system's thresholds must be defined here, with documentation
/// explaining why the value was chosen. No external file should use
/// numeric literals for these quantities — import this module's constants.
///
/// Source: REFACTOR_PLAN.md §E1 (thresholds.rs section) and empirical analysis of the
/// Universe experiments (sessions 2026-04-xx to 2026-08-13).

/// Capacity saturation fraction above which the model is "saturated".
///
/// PR/width > SATURATION_WARN → the brain is using >85% of the dimensions;
/// growing may be justified (second condition: real gain via the Lei do Garimpo).
///
/// 0.85 = 85% — chosen in `brain_capacity.rs::growth_verdict` as the default
/// `saturation_threshold`.
pub const SATURATION_WARN: f32 = 0.85;

/// Threshold to consider a neuron "dead" (no activation).
///
/// |h| < DEAD_EPS for ALL ticks in the observation window → dead neuron.
/// 1e-4 is chosen to sit above floating-point noise (1e-7) but below
/// any genuine activation (minimum observed on live channels: ~0.01).
pub const DEAD_EPS: f32 = 1e-4;

/// Maximum spectral radius allowed for W_rec without triggering an alarm.
///
/// |λ_max| > SPECTRAL_MAX → potentially unstable recurrent dynamics.
/// 1.0 is the theoretical threshold (echo-state networks, echo-state property condition).
/// In practice, since the GPU's W_rec operates with a CfC (explicit decay), values
/// slightly above 1.0 are absorbed by the decay, but the alarm serves as a
/// warning for investigation.
pub const SPECTRAL_MAX: f32 = 1.0;

/// Target for L1's largest singular value at genome INITIALIZATION (P0.5,
/// `gpu::evolution::Genome::random`) — the same quantity `SPECTRAL_MAX` audits after
/// the fact (`analysis::activation::spectral_summary`), now used as a birth
/// TARGET instead of an alarm threshold.
///
/// Motivation: 4 independent runs of `Genome::random` with raw i.i.d. σ=0.30
/// measured L1's spectral gain at ~7.0 (7.101/7.050/7.048/6.983 —
/// `docs/PLANO_EVOLUCAO.md` §P0.5). Echo-state network theory asks for
/// `|λ_max| <= ~1` for stable input dynamics; at 7× the sensory signal is
/// amplified even before the first mutation, and evolution is always measured
/// carrying that background bias (the observed champion stayed 99.25% equal to the
/// initial draw — cosine 0.9925 — a sign that selection pushes the bad init along, not fixes it).
///
/// 1.0 == `SPECTRAL_MAX` by definition (the alarm fires ABOVE what the init AIMS at),
/// but kept as a separate constant: the two quantities mean different things
/// (post-hoc alarm threshold vs. birth target) and a future change in one must
/// not silently couple to the other.
pub const SPECTRAL_INIT_TARGET: f32 = 1.0;

/// If true, any NaN detected in the activations aborts the analysis.
///
/// NaN in activations = serious bug (division by zero, gradient overflow, etc.).
/// Keep at `true` in production — detection cost is O(n) per analysis.
pub const NAN_ABORT: bool = true;

/// Minimum acceptable SNR of the i8 quantization (in dB).
///
/// SNR < SNR_I8_MIN → poorly distributed weights (outside the [-2, 2] range) or
/// quantization introducing too much noise.
/// 6.0 dB = signal/noise ratio of 4:1 — below that the model may behave
/// unpredictably after quantization to i8.
pub const SNR_I8_MIN: f32 = 6.0;

/// Fraction of dead neurons above which the alarm fires.
///
/// More than 20% dead → initialization problem, vanishing gradient or
/// mutation rate too high. Empirically validated: healthy models
/// have < 5% dead neurons after a few generations.
pub const DEAD_NEURONS_WARN_PCT: f32 = 20.0;

/// Fraction of saturated neurons above which the alarm fires.
///
/// |h| > 0.99 in > 30% of ticks → the channel is stuck at an extreme and
/// carries no useful gradient information.
pub const SATURATED_NEURONS_WARN_PCT: f32 = 30.0;

/// Minimum entropy of the policy (steer/throttle/brake distribution).
///
/// Entropy < POLICY_ENTROPY_MIN → the agent is taking almost always the
/// same action (locked steering, constant braking, etc.).
/// 0.5 nats is low enough to capture collapse without generating false
/// positives on legitimate sustained maneuvers.
pub const POLICY_ENTROPY_MIN: f32 = 0.5;

// ---------------------------------------------------------------------------
// Unit tests — verify the values are in the expected range.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saturation_warn_entre_zero_e_um() {
        assert!(
            SATURATION_WARN > 0.0 && SATURATION_WARN < 1.0,
            "SATURATION_WARN must be in (0, 1), value = {}",
            SATURATION_WARN
        );
    }

    #[test]
    fn dead_eps_positivo_e_pequeno() {
        assert!(
            DEAD_EPS > 0.0 && DEAD_EPS < 0.01,
            "DEAD_EPS must be small and positive, value = {}",
            DEAD_EPS
        );
    }

    #[test]
    fn spectral_max_positivo() {
        assert!(
            SPECTRAL_MAX > 0.0,
            "SPECTRAL_MAX must be positive, value = {}",
            SPECTRAL_MAX
        );
    }

    #[test]
    fn spectral_init_target_positivo_e_igual_ao_alarme() {
        assert!(
            SPECTRAL_INIT_TARGET > 0.0,
            "SPECTRAL_INIT_TARGET must be positive, value = {}",
            SPECTRAL_INIT_TARGET
        );
        // By documented definition: the alarm fires above what the init aims at.
        assert_eq!(
            SPECTRAL_INIT_TARGET, SPECTRAL_MAX,
            "init target and alarm threshold should coincide numerically by definition"
        );
    }

    #[test]
    fn snr_min_positivo() {
        assert!(
            SNR_I8_MIN > 0.0,
            "SNR_I8_MIN must be positive, value = {}",
            SNR_I8_MIN
        );
    }

    #[test]
    fn dead_neurons_warn_pct_entre_0_e_100() {
        assert!(DEAD_NEURONS_WARN_PCT > 0.0 && DEAD_NEURONS_WARN_PCT <= 100.0);
    }

    #[test]
    fn saturated_neurons_warn_pct_entre_0_e_100() {
        assert!(SATURATED_NEURONS_WARN_PCT > 0.0 && SATURATED_NEURONS_WARN_PCT <= 100.0);
    }

    #[test]
    fn policy_entropy_min_positivo() {
        assert!(POLICY_ENTROPY_MIN > 0.0);
    }
}
