//! Mixture-of-Experts gate. 16 inputs → 3 softmax outputs.
//! Outputs [w_cortex, w_reflex, w_sentinel] summing to 1.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

pub const GATE_INPUTS: usize = 16;
pub const GATE_OUTPUTS: usize = 3;

#[derive(Debug, Clone)]
pub struct GateBrain {
    pub weights: [f32; GATE_INPUTS * GATE_OUTPUTS], // row-major: out * INPUTS + in
    pub bias: [f32; GATE_OUTPUTS],
}

impl GateBrain {
    /// Initialize uniform: bias=0, weights tiny random.
    /// Result of softmax(0) = [1/3, 1/3, 1/3] when weights are zero — ideal cold-start.
    pub fn new_uniform(seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut weights = [0.0_f32; GATE_INPUTS * GATE_OUTPUTS];
        for w in &mut weights {
            *w = (rng.random::<f32>() - 0.5) * 0.1;
        }
        Self {
            weights,
            bias: [0.0; GATE_OUTPUTS],
        }
    }

    pub fn forward(&self, input: &[f32; GATE_INPUTS]) -> [f32; GATE_OUTPUTS] {
        let mut logits = self.bias;
        for o in 0..GATE_OUTPUTS {
            let row = o * GATE_INPUTS;
            let mut acc = logits[o];
            for i in 0..GATE_INPUTS {
                acc += self.weights[row + i] * input[i];
            }
            logits[o] = acc;
        }
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        let mut exps = [0.0_f32; GATE_OUTPUTS];
        for o in 0..GATE_OUTPUTS {
            let e = (logits[o] - max).exp();
            exps[o] = e;
            sum += e;
        }
        if !sum.is_finite() || sum == 0.0 {
            return [0.0, 0.0, 1.0]; // NaN fallback → sentinel-only
        }
        for o in 0..GATE_OUTPUTS {
            exps[o] /= sum;
        }
        exps
    }

    /// Shannon entropy of softmax output (used for fitness reward — avoid collapse).
    pub fn entropy(weights: &[f32; GATE_OUTPUTS]) -> f32 {
        let mut h = 0.0_f32;
        for &w in weights {
            if w > 1e-6 {
                h -= w * w.ln();
            }
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_init_softmax_near_third() {
        let g = GateBrain {
            weights: [0.0; GATE_INPUTS * GATE_OUTPUTS],
            bias: [0.0; GATE_OUTPUTS],
        };
        let out = g.forward(&[0.0; GATE_INPUTS]);
        for w in out {
            assert!((w - 1.0 / 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let g = GateBrain::new_uniform(42);
        let out = g.forward(&[0.5; GATE_INPUTS]);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn nan_input_falls_back_to_sentinel() {
        let mut g = GateBrain::new_uniform(7);
        for w in &mut g.weights {
            *w = f32::INFINITY;
        }
        let out = g.forward(&[1.0; GATE_INPUTS]);
        assert_eq!(out, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn entropy_max_when_uniform() {
        let h = GateBrain::entropy(&[1.0 / 3.0; 3]);
        assert!((h - (3.0_f32).ln()).abs() < 1e-5);
    }

    #[test]
    fn entropy_zero_when_collapsed() {
        let h = GateBrain::entropy(&[1.0, 0.0, 0.0]);
        assert!(h < 1e-5);
    }
}
