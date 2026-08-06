/// IARA v2101 - SOVEREIGN BIT-LINEAR (BitNet 1.58-bit)
/// "The End of MatMul": 100% Science, Zero Legacy.

/// Abstração de forward ternário — permite swap entre backends
/// (BitLinear atual, BitLinearPacked 2-bit, BitLinearGpu HIPRTC)
/// sem mexer em `SovereignEngine`. Ver `melhorias/gpu-roadmap.md`.
pub trait BrainForward {
    fn forward_i8(&self, input: &[i8]) -> Vec<f32>;
    fn myelinate(&mut self, factor: f32);
    fn quantize(&mut self, latent: &[f32]);
}

/// CortexLayer (formerly BitLinear ternary): now stores f32 weights at runtime
/// for smooth gradient via gaussian mutation. Disk storage stays compact via
/// quantize_for_save() / dequantize_from_disk() at DNA boundary.
///
/// Backward-compat: API surface preserved (weights, weight_scale, forward,
/// quantize, set_scale, myelinate). Field type changed Vec<i8> → Vec<f32>.
pub struct BitLinear {
    /// Float weights — qualquer valor real (-2..2 típico). Antes ternary {-1,0,1}.
    pub weights: Vec<f32>,
    /// Scale legacy field — kept para API compat. Forward NÃO multiplica scale
    /// (peso já absorve magnitude). Save quantization usa pra DNA backward-compat.
    pub weight_scale: f32,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl BitLinear {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            weights: vec![0.0; input_dim * output_dim],
            weight_scale: 1.0,
            input_dim,
            output_dim,
        }
    }

    /// Forward f32 — output = Σ (w * input/127). Mais expressivo que ternary.
    pub fn forward(&self, input: &[i8]) -> Vec<f32> {
        let mut output = vec![0.0_f32; self.output_dim];
        let norm_factor = 1.0 / 127.0;
        for out_idx in 0..self.output_dim {
            let offset = out_idx * self.input_dim;
            let mut acc = 0.0_f32;
            for in_idx in 0..self.input_dim {
                acc += self.weights[offset + in_idx] * (input[in_idx] as f32);
            }
            output[out_idx] = acc * norm_factor;
        }
        output
    }

    /// Myelinate: no-op em f32 runtime. Antes ternary cortex precisava boost de
    /// scale pra contrabalançar quantização. Agora pesos são contínuos —
    /// multiplicar tudo cumulativamente satura em ±2 e mata o cérebro
    /// (forward vira "porta": output constante = mesma decisão sempre).
    /// weight_scale field mantida por API compat mas só atualiza pra log.
    pub fn myelinate(&mut self, factor: f32) {
        let f = (1.0 + factor * 0.005).min(1.05);
        self.weight_scale = (self.weight_scale * f).min(10.0); // legacy log only
    }

    /// Quantize: agora apenas COPIA o latent f32 nas weights (sem ternarizar).
    /// API name kept para back-compat; semantics evoluíram pra "set weights from latent".
    pub fn quantize(&mut self, full_precision: &[f32]) {
        let n = full_precision.len().min(self.weights.len());
        self.weights[..n].copy_from_slice(&full_precision[..n]);
        // ABLAÇÃO DO PRIMING (ver `neuro.rs`): aplicada AQUI porque `quantize` é o ponto
        // ÚNICO por onde os pesos entram nesta camada — vale para o cérebro aleatório
        // inicial, para o campeão carregado do DNA e para os membros do pool de elite.
        // Zerar noutro lugar deixaria algum desses caminhos passar com o priming intacto.
        crate::neuro::ablar_priming(&mut self.weights, self.input_dim);
        // Update legacy scale field as abs_mean (still used by some DNA paths).
        let abs_mean: f32 = full_precision.iter().map(|w| w.abs()).sum::<f32>() / full_precision.len().max(1) as f32;
        self.weight_scale = abs_mean.max(1e-4);
    }

    pub fn set_scale(&mut self, scale: f32) {
        self.weight_scale = scale;
    }

    /// Quantize f32 weights to i8 for DNA disk. Scale 63.5 cobre range [-2.0, 2.0]
    /// (clamp do mutation) sem saturar. Antes 127 saturava metade do range.
    pub fn quantize_for_save(&self) -> Vec<i8> {
        // `.round()` antes do corte: `as i8` trunca PARA ZERO, o que encolhe toda
        // magnitude sistematicamente (0,5 → 31,75 → 31 → 0,48819). Num laço evolutivo,
        // que salva e recarrega o campeão a cada geração, isso é decaimento acumulado
        // dos pesos, não ruído simétrico. Arredondar deixa o erro em ±0,5/63,5 ≈ 1/127,
        // que é a precisão que o teste deste arquivo sempre declarou esperar.
        self.weights.iter().map(|&w| (w * 63.5).round().clamp(-127.0, 127.0) as i8).collect()
    }

    /// Dequantize i8 disk weights → f32 runtime. i8/63.5 covers [-2.0, 2.0].
    pub fn load_from_quantized(&mut self, q: &[i8]) {
        let n = q.len().min(self.weights.len());
        for i in 0..n { self.weights[i] = q[i] as f32 / 63.5; }
    }
}

impl BrainForward for BitLinear {
    fn forward_i8(&self, input: &[i8]) -> Vec<f32> {
        self.forward(input)
    }
    fn myelinate(&mut self, factor: f32) {
        BitLinear::myelinate(self, factor)
    }
    fn quantize(&mut self, latent: &[f32]) {
        BitLinear::quantize(self, latent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_copies_to_weights() {
        let mut layer = BitLinear::new(4, 2);
        let full = vec![0.5_f32, -0.5, 0.1, -1.0,    0.01, 1.5, -1.2, 0.0];
        layer.quantize(&full);
        for (i, &w) in full.iter().enumerate() {
            assert!((layer.weights[i] - w).abs() < 1e-6);
        }
    }

    #[test]
    fn forward_identity_single_weight() {
        let mut layer = BitLinear::new(1, 1);
        layer.weights[0] = 1.0;
        let out = layer.forward(&[127]);
        assert!((out[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn forward_zero_weight_gives_zero() {
        let layer = BitLinear::new(4, 2);
        let out = layer.forward(&[127, 127, -127, 50]);
        assert!(out.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn save_load_roundtrip() {
        let mut a = BitLinear::new(4, 1);
        a.weights = vec![0.5, -0.3, 0.0, 1.0];
        let q = a.quantize_for_save();
        let mut b = BitLinear::new(4, 1);
        b.load_from_quantized(&q);
        // Lossy ~1/127 precision
        for (orig, recovered) in a.weights.iter().zip(b.weights.iter()) {
            assert!((orig - recovered).abs() < 0.01, "{} vs {}", orig, recovered);
        }
    }
}
