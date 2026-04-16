/// IARA v2101 - SOVEREIGN BIT-LINEAR (BitNet 1.58-bit)
/// "The End of MatMul": 100% Science, Zero Legacy.

pub struct BitLinear {
    /// Ternary Weights: {-1, 0, 1} stored in i8 for memory efficiency.
    pub weights: Vec<i8>,
    /// Gamma Scale: Single float for dequantization per layer (minimal cost).
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

    /// Optimized Inference: 100% Addition/Subtraction.
    /// No float multiplications during accumulation.
    pub fn forward(&self, input: &[i8]) -> Vec<f32> {
        let mut output = vec![0.0; self.output_dim];
        let norm_factor = 1.0 / ((self.input_dim as f32).sqrt() * 128.0);

        for out_idx in 0..self.output_dim {
            let mut accumulator: i32 = 0;
            let offset = out_idx * self.input_dim;

            // SIMD-ready loop (Compiler will optimize this to ADD/SUB instructions)
            for in_idx in 0..self.input_dim {
                let w = self.weights[offset + in_idx];
                let x = input[in_idx] as i32;

                // BRANCHLESS LOGIC: w is {-1, 0, 1}
                // (x * w) becomes: (x if w=1) + (-x if w=-1) + (0 if w=0)
                accumulator += x * (w as i32); 
            }

            // Final scale application (The only float op per output neuron)
            output[out_idx] = (accumulator as f32 * self.weight_scale) * norm_factor;
        }

        output
    }

    /// Mielinização (Science-Based Optimization):
    /// In 2026, we don't 'train', we 'myelinate'.
    /// Successful paths reduce resistance (increase weight_scale).
    pub fn myelinate(&mut self, factor: f32) {
        self.weight_scale *= 1.0 + (factor * 0.005);
    }
    
    /// Quantization: Converts latent intent into ternary sovereignty.
    pub fn quantize(&mut self, full_precision: &[f32]) {
        let abs_mean: f32 = full_precision.iter().map(|w| w.abs()).sum::<f32>() / full_precision.len() as f32;
        self.weight_scale = abs_mean;

        for (i, &w) in full_precision.iter().enumerate() {
            let scaled = w / (abs_mean + 1e-8);
            self.weights[i] = scaled.round().clamp(-1.0, 1.0) as i8;
        }
    }

    pub fn set_scale(&mut self, scale: f32) {
        self.weight_scale = scale;
    }
}
