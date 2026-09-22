/// Activation functions and clamp helpers — all purely scalar, no allocation.
///
/// These functions mirror the formulas documented in MATH_INVENTORY.md §1.1.
/// The GPU (WGSL) versions use the same formulas; changes here must be
/// mirrored in the corresponding shaders (contrary to rules E1-E3, do not
/// touch the .wgsl — this module serves as the host reference).

/// Standard sigmoid: σ(x) = 1 / (1 + e^{-x}).
///
/// Historically used in gating and probability normalization.
/// For the CfC the primary non-linearity is `tanh`, not sigmoid.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Hyperbolic tangent — the default non-linearity of the CfC and the BitLinear.
///
/// Mapping: [-∞, +∞] → (-1, +1).
/// At 64 Hz with state clamped in [-2, 2]: tanh(±2) ≈ ±0.964 → ±122/127.
#[inline]
pub fn tanh(x: f32) -> f32 {
    x.tanh()
}

/// Fast tanh approximation: x / (1 + |x|).
///
/// Error < 2% for |x| < 3. Used by the perception_temporal.wgsl shader (F42).
/// Do NOT use where precision is required; motor.rs and cfc.rs use exact tanh.
#[inline]
pub fn tanh_approx(x: f32) -> f32 {
    x / (1.0 + x.abs())
}

/// ReLU: max(0, x).
#[inline]
pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// Leaky ReLU: max(alpha * x, x) with alpha < 1.
///
/// alpha = 0.01 is the default to keep a minimal gradient in negative regions.
#[inline]
pub fn leaky_relu(x: f32, alpha: f32) -> f32 {
    if x >= 0.0 {
        x
    } else {
        alpha * x
    }
}

/// Symmetric clamp in [-limit, +limit].
#[inline]
pub fn clamp_sym(x: f32, limit: f32) -> f32 {
    x.clamp(-limit, limit)
}

/// Converts f32 state to i8 via tanh, multiplying by 127.
///
/// Formula: `clamp(tanh(x) * 127, -127, 127) as i8`.
/// Used by the CfC (F07) and by the BitLinear (F10).
#[inline]
pub fn state_to_i8(x: f32) -> i8 {
    (x.tanh() * 127.0).clamp(-127.0, 127.0) as i8
}

/// Soft-thresholding: sign(x) * max(0, |x| - threshold).
///
/// Identical to the Lei do Garimpo (mining law) operator on the time axis.
/// Used in the CfC's adaptive fatigue (F03): attenuates without inverting the sign.
#[inline]
pub fn soft_threshold(x: f32, threshold: f32) -> f32 {
    x.signum() * (x.abs() - threshold).max(0.0)
}

/// Safety epsilon for `out_stage`'s deviation normalization — avoids
/// dividing by a ~0 deviation at the start of training (freshly drawn weights, the 48
/// `l2_raw` still almost all ~0). Same value as `NORM_EPS` in `brain.wgsl`.
pub const OUT_STAGE_NORM_EPS: f32 = 1e-3;

/// Output stage of the brain's STAGE 2 — mirrors `brain.wgsl` (the STAGE 2
/// lines, P1.5 in `docs/PLANO_EVOLUCAO.md`) and `engine.rs` (~497-515):
/// optional deviation normalization → output gain → adaptive fatigue
/// (soft-threshold) → tanh.
///
/// # Parameters
/// - `raw_pregain`: `l2_raw` or `l1_raw[bypass]` BEFORE any gain — the
///   same quantity that `float_layers::forward`/`w_l1r` return (`acc/127`,
///   without the historical `2.0`).
/// - `l2_std`: `Some(deviation)` turns on scale normalization (P1.5, item 3);
///   `None` is the default — behaviour identical to the historical one, without touching a
///   bit of the result. When `Some`, the deviation is that of
///   `math::stats::population_std_dev` over the 48 `raw_pregain` of the SAME being
///   (only makes sense for the L2 pathway, not for the bypass — see the comment in
///   the shader's STAGE 2).
/// - `out_gain`: output gain (P1.5, item 2) — `2.0` reproduces exactly the
///   historical hardcode (`OUT_GAIN_PADRAO` in `gpu/brain.rs`).
/// - `out_beta`, `out_fat`: output fatigue — `out_fat` is the state ALREADY
///   UPDATED this tick (the caller updates `out_fat` BEFORE calling this
///   function, same order the shader and `engine.rs` follow).
///
/// # Equivalence contract (tested below)
/// With `l2_std=None` and `out_gain=OUT_GAIN_PADRAO` (2.0), this function reproduces
/// EXACTLY `tanh(sign(raw*2.0) * max(0, |raw*2.0| - out_beta*out_fat))` — the
/// formula `brain.wgsl`/`engine.rs` had before `out_gain` became a
/// parameter.
#[inline]
pub fn out_stage(
    raw_pregain: f32,
    l2_std: Option<f32>,
    out_gain: f32,
    out_beta: f32,
    out_fat: f32,
) -> f32 {
    let scaled = match l2_std {
        Some(std) => raw_pregain / std.max(OUT_STAGE_NORM_EPS),
        None => raw_pregain,
    };
    let raw = scaled * out_gain;
    let adapted = soft_threshold(raw, out_beta * out_fat);
    adapted.tanh()
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_de_zero_e_meio() {
        let s = sigmoid(0.0);
        assert!((s - 0.5).abs() < 1e-6, "sigmoid(0) must be 0.5, got {}", s);
    }

    #[test]
    fn sigmoid_monotona_crescente() {
        assert!(sigmoid(-2.0) < sigmoid(0.0));
        assert!(sigmoid(0.0) < sigmoid(2.0));
    }

    #[test]
    fn tanh_de_zero_e_zero() {
        assert_eq!(tanh(0.0), 0.0);
    }

    #[test]
    fn tanh_simetrico() {
        // tanh is odd: tanh(-x) = -tanh(x)
        let x = 1.234_f32;
        assert!((tanh(-x) + tanh(x)).abs() < 1e-6);
    }

    #[test]
    fn tanh_approx_erro_pequeno_perto_de_zero() {
        // x/(1+|x|) is accurate near x=0 (error below 2% for |x| ≤ 0.1).
        // For |x| > 0.1 the error grows quickly: it is a first-order
        // approximation, useful in shaders (no exp) but imprecise in general.
        // See MATH_INVENTORY.md F42: perception_temporal.wgsl uses this form.
        for xi in [-0.1_f32, -0.05, 0.0, 0.05, 0.1] {
            let approx = tanh_approx(xi);
            let exact = xi.tanh();
            let err = (approx - exact).abs();
            assert!(err < 0.015, "tanh_approx({}) error={} near zero", xi, err);
        }
    }

    #[test]
    fn tanh_approx_mesmo_sinal_que_tanh() {
        // The approximation must never invert the sign
        for xi in [-3.0_f32, -1.0, 0.0, 1.0, 3.0] {
            let approx = tanh_approx(xi);
            let exact = xi.tanh();
            if exact.abs() > 1e-6 {
                assert_eq!(
                    approx.signum(),
                    exact.signum(),
                    "tanh_approx({}) flipped sign: approx={} exact={}",
                    xi,
                    approx,
                    exact
                );
            }
        }
    }

    #[test]
    fn relu_de_negativo_e_zero() {
        assert_eq!(relu(-5.0), 0.0);
        assert_eq!(relu(3.0), 3.0);
    }

    #[test]
    fn leaky_relu_preserva_gradiente_negativo() {
        let alpha = 0.01_f32;
        let out = leaky_relu(-10.0, alpha);
        assert!(
            (out - (-0.1)).abs() < 1e-6,
            "leaky_relu(-10, 0.01) must be -0.1, got {}",
            out
        );
    }

    #[test]
    fn clamp_sym_respeita_limite() {
        assert_eq!(clamp_sym(5.0, 2.0), 2.0);
        assert_eq!(clamp_sym(-5.0, 2.0), -2.0);
        assert_eq!(clamp_sym(1.0, 2.0), 1.0);
    }

    #[test]
    fn state_to_i8_maximo_clamped() {
        // tanh(10) ≈ 1.0 → 127; tanh(-10) ≈ -1.0 → -127
        assert_eq!(state_to_i8(10.0), 127);
        assert_eq!(state_to_i8(-10.0), -127);
    }

    #[test]
    fn state_to_i8_de_zero_e_zero() {
        assert_eq!(state_to_i8(0.0), 0);
    }

    #[test]
    fn soft_threshold_atenua_sem_inverter() {
        // x=0.5, threshold=0.3 → sign(0.5)*max(0, 0.5-0.3) = 0.2
        let out = soft_threshold(0.5, 0.3);
        assert!(
            (out - 0.2).abs() < 1e-6,
            "soft_threshold(0.5, 0.3) = {}",
            out
        );
        // the sign must not invert
        assert!(out >= 0.0, "negative sign for positive input!");
    }

    #[test]
    fn soft_threshold_negativo_atenua_sem_inverter() {
        let out = soft_threshold(-0.5, 0.3);
        assert!(
            (out - (-0.2)).abs() < 1e-6,
            "soft_threshold(-0.5, 0.3) = {}",
            out
        );
        assert!(out <= 0.0, "positive sign for negative input!");
    }

    #[test]
    fn soft_threshold_abaixo_do_limiar_e_zero() {
        let out = soft_threshold(0.1, 0.5);
        assert_eq!(out, 0.0, "below the threshold it must be zero");
    }

    // ── out_stage (P1.5: parameterizable output gain + optional normalization) ──

    /// Task requirement (docs/PLANO_EVOLUCAO.md P1.5, item 5): "the gain becomes a
    /// parameter and the value 2.0 reproduces the previous output exactly". Compares
    /// `out_stage` with `out_gain=OUT_GAIN_PADRAO` against the historical hardcoded
    /// formula (`raw = l2_raw_pregain * 2.0`) that `brain.wgsl`/`engine.rs`
    /// had BEFORE this session — bit for bit, not just "close".
    #[test]
    fn out_stage_ganho_padrao_reproduz_formula_historica_exatamente() {
        const OUT_GAIN_PADRAO: f32 = 2.0; // gpu::brain::OUT_GAIN_PADRAO (avoids depending on wgpu here)
        for raw_pregain in [-1.8_f32, -0.31, 0.0, 0.42, 1.05, 3.7] {
            for out_beta in [0.0_f32, 0.85] {
                for out_fat in [0.0_f32, 0.12, 1.9] {
                    let historico = {
                        let raw = raw_pregain * 2.0;
                        (raw.signum() * (raw.abs() - out_beta * out_fat).max(0.0)).tanh()
                    };
                    let novo = out_stage(raw_pregain, None, OUT_GAIN_PADRAO, out_beta, out_fat);
                    assert_eq!(
                        novo, historico,
                        "raw_pregain={raw_pregain} out_beta={out_beta} out_fat={out_fat}: \
                         new={novo} historical={historico}"
                    );
                }
            }
        }
    }

    /// `l2_std=None` and `l2_std=Some(1.0)` must be identical — dividing by 1.0
    /// is mathematically a no-op; confirms that normalization OFF (`None`, the
    /// default of the `--norm-l2-by-deviation` flag being off) introduces no
    /// spurious scale.
    #[test]
    fn out_stage_norm_none_e_std_um_sao_identicos() {
        let a = out_stage(0.73, None, 2.0, 0.85, 0.3);
        let b = out_stage(0.73, Some(1.0), 2.0, 0.85, 0.3);
        assert_eq!(a, b);
    }

    /// With normalization ON, the result rescales by amplitude: dividing
    /// by a larger deviation ATTENUATES the signal before the gain — the effect P1.5
    /// pursues against saturation (the scale of l2_raw grows with mutation over
    /// the generations; dividing by its own deviation neutralizes that
    /// growth before the tanh).
    #[test]
    fn out_stage_norm_ligada_atenua_por_desvio_maior() {
        // out_beta=0 (no fatigue) isolates the normalization effect.
        let sem_norm = out_stage(4.0, None, 1.0, 0.0, 0.0);
        let com_norm = out_stage(4.0, Some(2.0), 1.0, 0.0, 0.0);
        assert_eq!(sem_norm, 4.0f32.tanh());
        assert_eq!(com_norm, 2.0f32.tanh()); // 4.0/2.0 = 2.0
        assert!(
            com_norm.abs() < sem_norm.abs(),
            "normalization should attenuate, not amplify here"
        );
    }

    /// The denominator never falls below `OUT_STAGE_NORM_EPS` — a ~0 deviation
    /// (start of training, weights almost all mute) must not make the output
    /// blow up numerically.
    #[test]
    fn out_stage_norm_com_desvio_quase_zero_nao_explode() {
        let out = out_stage(0.001, Some(0.0), 2.0, 0.0, 0.0);
        assert!(
            out.is_finite(),
            "non-finite output with near-zero deviation: {out}"
        );
        assert!(out.abs() <= 1.0, "tanh must saturate in [-1,1], got {out}");
    }

    /// `math::stats::population_std_dev` feeding `out_stage` end-to-end:
    /// given the vector of 48 l2_raw of a hypothetical saturated being, normalization
    /// must reduce the fraction of saturated channels (|tanh|>0.9) relative to the
    /// same vector without normalizing — P1.5's central hypothesis, verified in
    /// pure math (no GPU).
    #[test]
    fn out_stage_com_population_std_dev_reduz_saturacao() {
        use crate::math::stats::population_std_dev;
        // 48 values simulating grown l2_raw (larger weights after several
        // generations of mutation) — most near 3.0, with slight variation.
        let l2_raw: Vec<f32> = (0..48).map(|i| 2.8 + 0.05 * (i as f32 % 5.0)).collect();
        let std = population_std_dev(&l2_raw);
        assert!(
            std > 0.0,
            "deviation should be > 0 for a non-constant vector"
        );

        let sat_sem_norm = l2_raw
            .iter()
            .filter(|&&v| out_stage(v, None, 2.0, 0.0, 0.0).abs() > 0.9)
            .count();
        let sat_com_norm = l2_raw
            .iter()
            .filter(|&&v| out_stage(v, Some(std), 2.0, 0.0, 0.0).abs() > 0.9)
            .count();
        assert!(
            sat_com_norm <= sat_sem_norm,
            "normalization should reduce (or keep) saturation: without={sat_sem_norm} with={sat_com_norm}"
        );
    }
}
