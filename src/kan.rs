/// IARA v2101 - Axiomatic KAN (Kolmogorov-Arnold) Filter
/// Based on Codex v2101 - Science of Spline-based Inference

pub struct KanAxiomFilter {
    /// B-Spline Basis Coefficients: Ultralight 1D tensors
    pub spline_coeffs: [f32; 16],
}

impl KanAxiomFilter {
    pub fn new() -> Self {
        Self {
            spline_coeffs: [1.0; 16], // Neutral Identity at start
        }
    }

    /// Axiomatic Pulse Filtering: "The machine is physically incapable of the impossible."
    /// This filter uses splines to smooth the 1.58-bit ternary output.
    pub fn filter_pulse(&self, raw_signal: f32) -> f32 {
        // 1. Axiom 1: Velocity Persistence (Zero signals should be ignored)
        if raw_signal.abs() < 0.001 {
            return 0.0;
        }

        // 2. Spline Evaluation: Simplified 1D Spline Mapping
        // Map raw_signal [-1.0, 1.0] to spline index [0, 15]
        let idx = ((raw_signal + 1.0) * 7.5).clamp(0.0, 15.0) as usize;
        let c = self.spline_coeffs[idx];

        // Final signal is the product of ternary intent and spline coefficient.
        // Result: A smooth, continuous action from a quantized source.
        raw_signal * c
    }

    /// Adaptative Spline Shift: "Learning via Spline Deformation"
    pub fn adapt(&mut self, signal_idx: usize, delta: f32) {
        if signal_idx < 16 {
            self.spline_coeffs[signal_idx] += delta * 0.01;
        }
    }
}

/// Tier 2.2 — KAN per-channel SENSORY PRE-FILTER.
/// One independent spline per input channel (128 channels). Each channel learns
/// its optimal response curve (log, saturation, step, sigmoid, linear…).
///
/// Position in the pipeline (BEFORE the CfC):
///   raw_input[128] → KAN per-channel → filtered[128]
///                                       ↓ [0..64]
///                                       → CfC (temporal memory)
///                                       ↓
///                                       → BitLinear(128→64)
///
/// Identity-ish init: all coeffs = 1.0 → `filter_pulse(x) = x` (keeps the
/// pre-KAN behavior; zero regression expected at gen 0).
///
/// RUNTIME-ONLY: does not persist in .dna v3. Coeffs shared at spawn,
/// evolve via future adaptation (reward-modulated). See
/// melhorias/02-cerebro-dormente.md §2.2.
pub struct SensoryPreFilter {
    pub filters: Vec<KanAxiomFilter>,
}

impl SensoryPreFilter {
    pub fn new(num_channels: usize) -> Self {
        let mut filters = Vec::with_capacity(num_channels);
        for _ in 0..num_channels {
            filters.push(KanAxiomFilter::new());
        }
        Self { filters }
    }

    /// Applies the spline per channel. Keeps ternary i8 quantization in the output.
    /// `filter_pulse` already handles zero-signal (axiom 1): |x| < 0.001 → 0.
    pub fn apply(&self, raw_input: &[i8], output: &mut [i8]) {
        let n = raw_input.len().min(output.len()).min(self.filters.len());
        for i in 0..n {
            let raw_f32 = (raw_input[i] as f32) / 127.0;
            let filtered = self.filters[i].filter_pulse(raw_f32);
            output[i] = (filtered * 127.0).clamp(-127.0, 127.0) as i8;
        }
    }
}

/// KAN output head — per-output B-spline on raw action logits.
/// WHY: replaces the hard clamp on throttle/steer/brake (which produces saturated ±1
/// zig-zag) with a smooth learnable curve. Expected inputs in [-2, 2] (pre-clamp);
/// outputs in [-1, 1].
pub const KAN_OUT_KNOTS: usize = 12;
pub const KAN_OUT_DOMAIN: f32 = 2.0; // domain [-DOMAIN, +DOMAIN]

/// Evaluates the linearly-interpolated spline (degree-1 B-spline) directly on a raw
/// knot matrix — same math as `KanOutputHead::eval`, but without requiring the wrapper
/// struct. It exists because `Genome::kan_knots` (gpu/evolution.rs) and the GPU
/// binaries (`gpu_evolve`, `gpu_loop`) carry the knots as raw
/// `[[f32; KAN_OUT_KNOTS]; 3]`, not as `KanOutputHead`.
///
/// Deduplicated (E3, REFACTOR_PLAN.md §2.3): there were TWO byte-identical
/// reimplementations of this formula as local closures in `gpu_evolve.rs` and
/// `gpu_loop.rs` (the latter already commented "identical to KanOutputHead::eval()"
/// without actually sharing code) — single source of truth now.
#[inline]
pub fn eval_knots(knots: &[[f32; KAN_OUT_KNOTS]; 3], out_idx: usize, x: f32) -> f32 {
    let out_idx = out_idx.min(2);
    // maps x in [-DOMAIN, +DOMAIN] to index [0, KAN_OUT_KNOTS-1]
    let xc = x.clamp(-KAN_OUT_DOMAIN, KAN_OUT_DOMAIN);
    let u = (xc + KAN_OUT_DOMAIN) / (2.0 * KAN_OUT_DOMAIN); // [0,1]
    let fpos = u * ((KAN_OUT_KNOTS - 1) as f32);
    let i0 = fpos.floor() as usize;
    let i1 = (i0 + 1).min(KAN_OUT_KNOTS - 1);
    let frac = fpos - (i0 as f32);
    let a = knots[out_idx][i0];
    let b = knots[out_idx][i1];
    (a + (b - a) * frac).clamp(-1.0, 1.0)
}

#[derive(Clone, Debug)]
pub struct KanOutputHead {
    /// Order: [throttle, steer, brake] × KAN_OUT_KNOTS knots each.
    pub knots: [[f32; KAN_OUT_KNOTS]; 3],
}

impl KanOutputHead {
    /// Default identity-like: linearly spread knots + smooth sigmoid at the extremes.
    pub fn new() -> Self {
        let mut knots = [[0.0f32; KAN_OUT_KNOTS]; 3];
        for out in 0..3 {
            for k in 0..KAN_OUT_KNOTS {
                // x in [-DOMAIN, +DOMAIN]
                let t = (k as f32) / ((KAN_OUT_KNOTS - 1) as f32); // [0,1]
                let x = -KAN_OUT_DOMAIN + t * (2.0 * KAN_OUT_DOMAIN);
                // smooth sigmoid mapped to [-1, 1]: tanh(x * 0.9) approximates identity in the
                // middle and saturates gradually at the extremes — removes the clamp(±1) "step".
                knots[out][k] = (x * 0.9).tanh();
            }
        }
        Self { knots }
    }

    /// Evaluates the linearly-interpolated spline (degree-1 B-spline) on output `out_idx`
    /// with input `x`. Output always in [-1, 1].
    #[inline]
    pub fn eval(&self, out_idx: usize, x: f32) -> f32 {
        eval_knots(&self.knots, out_idx, x)
    }

    /// Light knot mutation — called by spawn.rs when the child inherits the head.
    /// `rate` = probability of perturbing each knot; low fixed sigma (0.05) preserves smoothness.
    pub fn mutate_spline_knots<R: FnMut() -> f32>(&mut self, mut rng01: R, rate: f32) {
        // v11: sigma 0.05 → 0.025 (half) to reduce cumulative drift over 200+ gens.
        let sigma = 0.025f32;
        for out in 0..3 {
            for k in 0..KAN_OUT_KNOTS {
                if rng01() < rate {
                    // Cheap Box-Muller via sum of 4 uniforms (std~0.58), scaled by sigma.
                    let u: f32 = (0..4).map(|_| rng01() - 0.5).sum();
                    self.knots[out][k] = (self.knots[out][k] + u * sigma).clamp(-1.0, 1.0);
                }
            }
        }
    }
}

impl Default for KanOutputHead {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_init_preserves_input() {
        let pre = SensoryPreFilter::new(128);
        let mut input = [0i8; 128];
        for i in 0..128 {
            input[i] = ((i as i32 - 64) as i8).saturating_mul(2);
        }
        let mut out = [0i8; 128];
        pre.apply(&input, &mut out);
        for i in 0..128 {
            // coeffs = 1.0 → output ≈ input (±1 from quantization).
            // Zero-signal: |x/127| < 0.001 (i.e. input==0) → 0 (same as input==0 anyway).
            let diff = (out[i] as i32 - input[i] as i32).abs();
            assert!(
                diff <= 1,
                "channel {}: in={} out={} diff={}",
                i,
                input[i],
                out[i],
                diff
            );
        }
    }

    #[test]
    fn zero_input_gives_zero() {
        let pre = SensoryPreFilter::new(4);
        let input = [0i8; 4];
        let mut out = [9i8; 4];
        pre.apply(&input, &mut out);
        assert_eq!(out, [0, 0, 0, 0]);
    }
}
