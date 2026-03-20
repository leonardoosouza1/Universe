/// IARA v2101 - BitNet 1.58-bit Linear Layer (BitLinear)
/// Based on Codex v2101 - Module II

pub struct BitLinear {
    /// Ternary weights {-1, 0, 1}
    pub weights: Vec<i8>,
    /// Scale factor (gamma) for dequantization
    pub weight_scale: f32,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl BitLinear {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            weights: vec![0; input_dim * output_dim],
            weight_scale: 1.0,
            input_dim,
            output_dim,
        }
    }

    /// Quantization-Aware Transformation
    /// Wq = clamp(round(W / mean(|W|)), -1, 1)
    pub fn quantize_weights(&mut self, full_precision_weights: &[f32]) {
        let abs_mean: f32 = full_precision_weights.iter().map(|w| w.abs()).sum::<f32>() / full_precision_weights.len() as f32;
        self.weight_scale = abs_mean;

        for (i, &w) in full_precision_weights.iter().enumerate() {
            let scaled = w / (abs_mean + 1e-8);
            self.weights[i] = scaled.round().clamp(-1.0, 1.0) as i8;
        }
    }

    /// Inference: Replacement of MatMul with ADD/SUB
    /// y = Sum(x_i * w_i) * scale
    /// Where w_i is {-1, 0, 1}
    pub fn forward(&self, input: &[i8]) -> Vec<f32> {
        let mut output = vec![0.0; self.output_dim];

        for out_idx in 0..self.output_dim {
            let mut accumulator: i32 = 0;
            for in_idx in 0..self.input_dim {
                let w = self.weights[out_idx * self.input_dim + in_idx];
                let x = input[in_idx] as i32;

                match w {
                    1 => accumulator += x,    // Addition
                    -1 => accumulator -= x,   // Subtraction
                    _ => (),                  // Ignore (0)
                }
            }
            // Final dequantization scaling
            output[out_idx] = accumulator as f32 * self.weight_scale;
        }

        output
    }

    /// Mielinização: Reduce resistance of successful paths.
    /// In this ternary model, we slightly increase the scale to prioritize this layer.
    pub fn myelinate(&mut self, success_factor: f32) {
        self.weight_scale += 0.001 * success_factor;
    }
}
