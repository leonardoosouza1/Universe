/// Dense layer in pure f32 for continuous learning (no ternary).
/// Used in SlowBrain (CfC → Dense → Dense → output).
/// Weights are f32 (stored contiguously), no scale factors.

pub struct DenseLayer {
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl DenseLayer {
    /// Creates a new layer with zeroed weights and zeroed bias.
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            weights: vec![0.0; input_dim * output_dim],
            bias: vec![0.0; output_dim],
            input_dim,
            output_dim,
        }
    }

    /// Forward pass: output[j] = Σ_i weights[j*N+i] * input[i] + bias[j]
    /// Input expected in [-1, 1] (normalized f32).
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.input_dim);
        let mut output = vec![0.0f32; self.output_dim];
        for out_idx in 0..self.output_dim {
            let offset = out_idx * self.input_dim;
            let mut acc = self.bias[out_idx];
            for in_idx in 0..self.input_dim {
                acc += self.weights[offset + in_idx] * input[in_idx];
            }
            output[out_idx] = acc;
        }
        output
    }

    /// Loads weights from a slice (copies up to min(src.len(), weights.len())).
    pub fn set_weights(&mut self, src: &[f32]) {
        let n = src.len().min(self.weights.len());
        self.weights[..n].copy_from_slice(&src[..n]);
    }

    /// Loads bias from a slice.
    pub fn set_bias(&mut self, src: &[f32]) {
        let n = src.len().min(self.bias.len());
        self.bias[..n].copy_from_slice(&src[..n]);
    }

    /// Gaussian mutation per weight: with probability `rate`,
    /// perturbs by N(0, sigma) and clamps to [-clamp_val, clamp_val].
    pub fn mutate<R: FnMut() -> f32>(
        &mut self,
        mut rng01: R,
        rate: f32,
        sigma: f32,
        clamp_val: f32,
    ) {
        let gauss = |rng01: &mut R| -> f32 {
            let u1 = rng01().max(1e-7);
            let u2 = rng01();
            (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
        };

        for w in self.weights.iter_mut() {
            if rng01() < rate {
                *w = (*w + gauss(&mut rng01) * sigma).clamp(-clamp_val, clamp_val);
            }
        }
        for b in self.bias.iter_mut() {
            if rng01() < rate {
                *b = (*b + gauss(&mut rng01) * sigma).clamp(-clamp_val, clamp_val);
            }
        }
    }

    /// Returns the total parameter count (weights + bias).
    pub fn param_count(&self) -> usize {
        self.weights.len() + self.bias.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_identity() {
        let mut l = DenseLayer::new(2, 2);
        l.weights[0] = 1.0; // w[0,0]
        l.weights[1] = 0.0; // w[0,1]
        l.weights[2] = 0.0; // w[1,0]
        l.weights[3] = 1.0; // w[1,1]
        let input = [0.5, 0.3];
        let out = l.forward(&input);
        assert!((out[0] - 0.5).abs() < 1e-5);
        assert!((out[1] - 0.3).abs() < 1e-5);
    }

    #[test]
    fn bias_adds() {
        let mut l = DenseLayer::new(1, 1);
        l.weights[0] = 0.0;
        l.bias[0] = 0.5;
        let out = l.forward(&[0.2]);
        assert!((out[0] - 0.5).abs() < 1e-5);
    }
}
