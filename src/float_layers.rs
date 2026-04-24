/// Option 4 A/B test: FloatLinear layer for slow_brain.
///
/// Mirrors BitLinear API, but stores weights at f32 full precision (no scale).
/// Used for the slow_brain cortex (32→8) where the 256-weight ternary space
/// was too coarse for smooth strategic modulation.

pub struct FloatLinear {
    pub weights: Vec<f32>,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl FloatLinear {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            weights: vec![0.0; input_dim * output_dim],
            input_dim,
            output_dim,
        }
    }

    /// out[j] = Σ_i weights[j*N+i] * (input[i] as f32) / 127
    pub fn forward(&self, input: &[i8]) -> Vec<f32> {
        let mut output = vec![0.0f32; self.output_dim];
        let norm_factor = 1.0 / 127.0;
        for out_idx in 0..self.output_dim {
            let offset = out_idx * self.input_dim;
            let mut acc = 0.0f32;
            for in_idx in 0..self.input_dim {
                acc += self.weights[offset + in_idx] * (input[in_idx] as f32);
            }
            output[out_idx] = acc * norm_factor;
        }
        output
    }

    pub fn set_weights(&mut self, src: &[f32]) {
        let n = src.len().min(self.weights.len());
        self.weights[..n].copy_from_slice(&src[..n]);
    }

    /// Box-Muller gaussian mutation per-weight: with probability `rate`,
    /// perturb by N(0, 0.02) and clamp to [-2.0, 2.0].
    pub fn mutate<R: FnMut() -> f32>(&mut self, mut rng01: R, rate: f32) {
        for w in self.weights.iter_mut() {
            if rng01() < rate {
                // Box-Muller from two uniforms in (0, 1].
                let mut u1 = rng01();
                if u1 <= 0.0 { u1 = 1e-7; }
                let u2 = rng01();
                let mag = (-2.0 * u1.ln()).sqrt();
                let z = mag * (2.0 * std::f32::consts::PI * u2).cos();
                *w = (*w + 0.02 * z).clamp(-2.0, 2.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_identity_single_weight() {
        let mut l = FloatLinear::new(1, 1);
        l.weights[0] = 1.0;
        let out = l.forward(&[127]);
        assert!((out[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn set_weights_copies() {
        let mut l = FloatLinear::new(2, 2);
        l.set_weights(&[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(l.weights, vec![0.1, 0.2, 0.3, 0.4]);
    }
}
