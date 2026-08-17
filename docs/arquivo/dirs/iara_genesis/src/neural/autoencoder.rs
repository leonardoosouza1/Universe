//! # Tiny Autoencoder para compressão de pesos
//!
//! Arquitetura: Linear(N → H) → ReLU → Linear(H → L) | Linear(L → H) → ReLU → Linear(H → N)
//!
//! Onde:
//! - N = dimensão de entrada (8192 para fast_weights)
//! - H = camada oculta (512)
//! - L = latent space (128)
//!
//! Os pesos do autoencoder (codebook) são compartilhados pela população.
//! O latent vector de cada organismo é o DNA comprimido (~256 bytes em f16).
//!
//! ## Treino
//!
//! SGD simples no CPU:
//! - Batch = população inteira (50 organismos)
//! - Loss = MSE + L1 sparsity no latent + ternary accuracy bonus
//! - 1-5 epochs por geração (background, não bloqueia a simulação)

/// Dimensões padrão do autoencoder.
pub const DEFAULT_INPUT_DIM: usize = 8192;
pub const DEFAULT_HIDDEN_DIM: usize = 512;
pub const DEFAULT_LATENT_DIM: usize = 128;
pub const DEFAULT_LEARNING_RATE: f32 = 0.001;
pub const DEFAULT_L1_LAMBDA: f32 = 0.0001;

/// Um autoencoder de 2 camadas para compressão de pesos ternários.
#[derive(Clone)]
pub struct TinyAutoencoder {
    // Encoder: input → hidden → latent
    pub enc_w1: Vec<f32>, // input_dim × hidden_dim
    pub enc_b1: Vec<f32>, // hidden_dim
    pub enc_w2: Vec<f32>, // hidden_dim × latent_dim
    pub enc_b2: Vec<f32>, // latent_dim

    // Decoder: latent → hidden → output
    pub dec_w1: Vec<f32>, // latent_dim × hidden_dim
    pub dec_b1: Vec<f32>, // hidden_dim
    pub dec_w2: Vec<f32>, // hidden_dim × input_dim
    pub dec_b2: Vec<f32>, // input_dim

    pub input_dim: usize,
    pub hidden_dim: usize,
    pub latent_dim: usize,
    pub learning_rate: f32,
    pub l1_lambda: f32,

    /// Número de epochs de treino acumulados.
    pub epochs_trained: u32,
}

impl TinyAutoencoder {
    /// Cria autoencoder com dimensões personalizadas.
    pub fn new(input_dim: usize, hidden_dim: usize, latent_dim: usize) -> Self {
        // Xavier init: U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
        let mut rng_state: u64 = 0xDEAD_BEEF_CAFE_BABE;

        let enc_w1 = xavier_init(input_dim, hidden_dim, &mut rng_state);
        let enc_b1 = vec![0.0; hidden_dim];
        let enc_w2 = xavier_init(hidden_dim, latent_dim, &mut rng_state);
        let enc_b2 = vec![0.0; latent_dim];

        let dec_w1 = xavier_init(latent_dim, hidden_dim, &mut rng_state);
        let dec_b1 = vec![0.0; hidden_dim];
        let dec_w2 = xavier_init(hidden_dim, input_dim, &mut rng_state);
        let dec_b2 = vec![0.0; input_dim];

        Self {
            enc_w1,
            enc_b1,
            enc_w2,
            enc_b2,
            dec_w1,
            dec_b1,
            dec_w2,
            dec_b2,
            input_dim,
            hidden_dim,
            latent_dim,
            learning_rate: DEFAULT_LEARNING_RATE,
            l1_lambda: DEFAULT_L1_LAMBDA,
            epochs_trained: 0,
        }
    }

    /// Cria autoencoder com dimensões padrão (8192 → 512 → 128).
    pub fn default_dims() -> Self {
        Self::new(DEFAULT_INPUT_DIM, DEFAULT_HIDDEN_DIM, DEFAULT_LATENT_DIM)
    }

    /// Encode: pesos ternários → latent vector.
    pub fn encode(&self, input: &[i8]) -> Vec<f32> {
        assert_eq!(input.len(), self.input_dim);

        // Converter i8 → f32 normalizado
        let input_f: Vec<f32> = input.iter().map(|&w| w as f32).collect();

        // Layer 1: z1 = ReLU(W1 · input + b1)
        let hidden = linear_relu(
            &input_f,
            &self.enc_w1,
            &self.enc_b1,
            self.input_dim,
            self.hidden_dim,
        );

        // Layer 2: latent = W2 · z1 + b2 (sem ativação — espaço contínuo)
        let latent = linear_forward(
            &hidden,
            &self.enc_w2,
            &self.enc_b2,
            self.hidden_dim,
            self.latent_dim,
        );

        latent
    }

    /// Decode: latent vector → pesos ternários reconstruídos.
    pub fn decode(&self, latent: &[f32]) -> Vec<i8> {
        assert_eq!(latent.len(), self.latent_dim);

        // Layer 1: z1 = ReLU(W1 · latent + b1)
        let hidden = linear_relu(
            latent,
            &self.dec_w1,
            &self.dec_b1,
            self.latent_dim,
            self.hidden_dim,
        );

        // Layer 2: output = W2 · z1 + b2
        let output = linear_forward(
            &hidden,
            &self.dec_w2,
            &self.dec_b2,
            self.hidden_dim,
            self.input_dim,
        );

        // Quantizar para ternário
        output
            .iter()
            .map(|&v| {
                if v > 0.5 {
                    1i8
                } else if v < -0.5 {
                    -1i8
                } else {
                    0i8
                }
            })
            .collect()
    }

    /// Decode para f32 (sem quantização — útil para treino).
    fn decode_f32(&self, latent: &[f32]) -> Vec<f32> {
        let hidden = linear_relu(
            latent,
            &self.dec_w1,
            &self.dec_b1,
            self.latent_dim,
            self.hidden_dim,
        );
        linear_forward(
            &hidden,
            &self.dec_w2,
            &self.dec_b2,
            self.hidden_dim,
            self.input_dim,
        )
    }

    /// Round-trip: encode → decode.
    pub fn compress_decompress(&self, input: &[i8]) -> Vec<i8> {
        let latent = self.encode(input);
        self.decode(&latent)
    }

    /// Treina o autoencoder com um batch de organismos.
    ///
    /// Retorna (loss_média, accuracy_média) do batch.
    pub fn train_batch(&mut self, batch: &[Vec<i8>]) -> (f32, f32) {
        if batch.is_empty() {
            return (0.0, 0.0);
        }

        let mut total_loss = 0.0_f32;
        let mut total_acc = 0.0_f32;
        let batch_size = batch.len() as f32;

        // Acumuladores de gradientes
        let mut grad_enc_w1 = vec![0.0f32; self.enc_w1.len()];
        let mut grad_enc_b1 = vec![0.0f32; self.enc_b1.len()];
        let mut grad_enc_w2 = vec![0.0f32; self.enc_w2.len()];
        let mut grad_enc_b2 = vec![0.0f32; self.enc_b2.len()];
        let mut grad_dec_w1 = vec![0.0f32; self.dec_w1.len()];
        let mut grad_dec_b1 = vec![0.0f32; self.dec_b1.len()];
        let mut grad_dec_w2 = vec![0.0f32; self.dec_w2.len()];
        let mut grad_dec_b2 = vec![0.0f32; self.dec_b2.len()];

        for sample in batch {
            let input_f: Vec<f32> = sample.iter().map(|&w| w as f32).collect();

            // === FORWARD ===
            // Encoder layer 1
            let enc_h1_pre = linear_forward(
                &input_f,
                &self.enc_w1,
                &self.enc_b1,
                self.input_dim,
                self.hidden_dim,
            );
            let enc_h1: Vec<f32> = enc_h1_pre.iter().map(|&v| v.max(0.0)).collect(); // ReLU
            let relu_mask_enc: Vec<bool> = enc_h1_pre.iter().map(|&v| v > 0.0).collect();

            // Encoder layer 2 (latent)
            let latent = linear_forward(
                &enc_h1,
                &self.enc_w2,
                &self.enc_b2,
                self.hidden_dim,
                self.latent_dim,
            );

            // Decoder layer 1
            let dec_h1_pre = linear_forward(
                &latent,
                &self.dec_w1,
                &self.dec_b1,
                self.latent_dim,
                self.hidden_dim,
            );
            let dec_h1: Vec<f32> = dec_h1_pre.iter().map(|&v| v.max(0.0)).collect();
            let relu_mask_dec: Vec<bool> = dec_h1_pre.iter().map(|&v| v > 0.0).collect();

            // Decoder layer 2 (output)
            let output = linear_forward(
                &dec_h1,
                &self.dec_w2,
                &self.dec_b2,
                self.hidden_dim,
                self.input_dim,
            );

            // === LOSS ===
            // MSE reconstruction loss
            let mse: f32 = input_f
                .iter()
                .zip(output.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                / self.input_dim as f32;

            // L1 sparsity on latent
            let l1: f32 = latent.iter().map(|&v| v.abs()).sum::<f32>() / self.latent_dim as f32;

            let loss = mse + self.l1_lambda * l1;
            total_loss += loss;

            // Accuracy (quantized)
            let quantized: Vec<i8> = output
                .iter()
                .map(|&v| {
                    if v > 0.5 {
                        1
                    } else if v < -0.5 {
                        -1
                    } else {
                        0
                    }
                })
                .collect();
            let correct = sample
                .iter()
                .zip(quantized.iter())
                .filter(|&(&a, &b)| a == b)
                .count();
            total_acc += correct as f32 / self.input_dim as f32;

            // === BACKWARD ===
            // d_loss/d_output = 2(output - input) / N
            let d_output: Vec<f32> = output
                .iter()
                .zip(input_f.iter())
                .map(|(&o, &i)| 2.0 * (o - i) / self.input_dim as f32)
                .collect();

            // Backprop through dec_w2, dec_b2
            let d_dec_h1 = backprop_linear(
                &d_output,
                &dec_h1,
                &self.dec_w2,
                self.hidden_dim,
                self.input_dim,
                &mut grad_dec_w2,
                &mut grad_dec_b2,
            );

            // Backprop through ReLU decoder
            let d_dec_h1_pre: Vec<f32> = d_dec_h1
                .iter()
                .zip(relu_mask_dec.iter())
                .map(|(&d, &active)| if active { d } else { 0.0 })
                .collect();

            // Backprop through dec_w1, dec_b1
            let d_latent_recon = backprop_linear(
                &d_dec_h1_pre,
                &latent,
                &self.dec_w1,
                self.latent_dim,
                self.hidden_dim,
                &mut grad_dec_w1,
                &mut grad_dec_b1,
            );

            // L1 gradient on latent
            let d_latent: Vec<f32> = d_latent_recon
                .iter()
                .zip(latent.iter())
                .map(|(&dr, &l)| {
                    let l1_grad = if l > 0.0 {
                        self.l1_lambda / self.latent_dim as f32
                    } else if l < 0.0 {
                        -self.l1_lambda / self.latent_dim as f32
                    } else {
                        0.0
                    };
                    dr + l1_grad
                })
                .collect();

            // Backprop through enc_w2, enc_b2
            let d_enc_h1 = backprop_linear(
                &d_latent,
                &enc_h1,
                &self.enc_w2,
                self.hidden_dim,
                self.latent_dim,
                &mut grad_enc_w2,
                &mut grad_enc_b2,
            );

            // Backprop through ReLU encoder
            let d_enc_h1_pre: Vec<f32> = d_enc_h1
                .iter()
                .zip(relu_mask_enc.iter())
                .map(|(&d, &active)| if active { d } else { 0.0 })
                .collect();

            // Backprop through enc_w1, enc_b1
            let _d_input = backprop_linear(
                &d_enc_h1_pre,
                &input_f,
                &self.enc_w1,
                self.input_dim,
                self.hidden_dim,
                &mut grad_enc_w1,
                &mut grad_enc_b1,
            );
        }

        // === SGD UPDATE ===
        let lr_scaled = self.learning_rate / batch_size;
        sgd_update(&mut self.enc_w1, &grad_enc_w1, lr_scaled);
        sgd_update(&mut self.enc_b1, &grad_enc_b1, lr_scaled);
        sgd_update(&mut self.enc_w2, &grad_enc_w2, lr_scaled);
        sgd_update(&mut self.enc_b2, &grad_enc_b2, lr_scaled);
        sgd_update(&mut self.dec_w1, &grad_dec_w1, lr_scaled);
        sgd_update(&mut self.dec_b1, &grad_dec_b1, lr_scaled);
        sgd_update(&mut self.dec_w2, &grad_dec_w2, lr_scaled);
        sgd_update(&mut self.dec_b2, &grad_dec_b2, lr_scaled);

        self.epochs_trained += 1;

        (total_loss / batch_size, total_acc / batch_size)
    }

    /// Serializa o latent vector para f16 bytes (compacto).
    pub fn latent_to_f16_bytes(latent: &[f32]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(latent.len() * 2);
        for &v in latent {
            let half = f32_to_f16(v);
            buf.extend_from_slice(&half.to_le_bytes());
        }
        buf
    }

    /// Desserializa latent vector de f16 bytes.
    pub fn latent_from_f16_bytes(buf: &[u8]) -> Vec<f32> {
        let mut latent = Vec::with_capacity(buf.len() / 2);
        for chunk in buf.chunks_exact(2) {
            let half = u16::from_le_bytes([chunk[0], chunk[1]]);
            latent.push(f16_to_f32(half));
        }
        latent
    }

    /// Número total de parâmetros (pesos + biases do autoencoder).
    pub fn num_params(&self) -> usize {
        self.enc_w1.len()
            + self.enc_b1.len()
            + self.enc_w2.len()
            + self.enc_b2.len()
            + self.dec_w1.len()
            + self.dec_b1.len()
            + self.dec_w2.len()
            + self.dec_b2.len()
    }

    /// Tamanho do codebook em bytes.
    pub fn codebook_size_bytes(&self) -> usize {
        self.num_params() * 4 // f32
    }
}

// ============================================================================
// Ops auxiliares
// ============================================================================

/// Forward linear: output[j] = Σ_i input[i] × W[j * input_dim + i] + bias[j]
fn linear_forward(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(out_dim);
    for j in 0..out_dim {
        let mut sum = bias[j];
        let offset = j * in_dim;
        for i in 0..in_dim {
            sum += weights[offset + i] * input[i];
        }
        output.push(sum);
    }
    output
}

/// Forward linear + ReLU.
fn linear_relu(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Vec<f32> {
    let mut output = linear_forward(input, weights, bias, in_dim, out_dim);
    for v in output.iter_mut() {
        *v = v.max(0.0);
    }
    output
}

/// Backprop through linear layer.
/// Returns d_input, accumulates into grad_weights and grad_bias.
fn backprop_linear(
    d_output: &[f32],
    input: &[f32],
    weights: &[f32],
    in_dim: usize,
    out_dim: usize,
    grad_weights: &mut [f32],
    grad_bias: &mut [f32],
) -> Vec<f32> {
    // grad_bias += d_output
    for j in 0..out_dim {
        grad_bias[j] += d_output[j];
    }

    // grad_weights[j * in_dim + i] += d_output[j] * input[i]
    for j in 0..out_dim {
        let offset = j * in_dim;
        let d_j = d_output[j];
        for i in 0..in_dim {
            grad_weights[offset + i] += d_j * input[i];
        }
    }

    // d_input[i] = Σ_j d_output[j] * weights[j * in_dim + i]
    let mut d_input = vec![0.0f32; in_dim];
    for j in 0..out_dim {
        let offset = j * in_dim;
        let d_j = d_output[j];
        for i in 0..in_dim {
            d_input[i] += d_j * weights[offset + i];
        }
    }

    d_input
}

/// SGD: params -= lr × grads
fn sgd_update(params: &mut [f32], grads: &[f32], lr: f32) {
    for (p, g) in params.iter_mut().zip(grads.iter()) {
        *p -= lr * g;
        // Clamp to prevent exploding weights
        if !p.is_finite() {
            *p = 0.0;
        }
    }
}

/// Xavier uniform initialization.
fn xavier_init(fan_in: usize, fan_out: usize, rng: &mut u64) -> Vec<f32> {
    let limit = (6.0 / (fan_in + fan_out) as f64).sqrt() as f32;
    let size = fan_in * fan_out;
    let mut weights = Vec::with_capacity(size);
    for _ in 0..size {
        let u = lcg_uniform(rng);
        weights.push((u * 2.0 - 1.0) * limit);
    }
    weights
}

/// LCG pseudo-random para inicialização (determinístico, sem dep externa).
fn lcg_uniform(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f32) / (u32::MAX as f32)
}

/// Converte f32 → f16 (IEEE 754 half-precision, truncation mode).
fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;

    if exp == 0xFF {
        // Inf/NaN
        return (sign << 15) | 0x7C00 | if frac != 0 { 0x0200 } else { 0 };
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return (sign << 15) | 0x7C00; // overflow → Inf
    }
    if new_exp <= 0 {
        return sign << 15; // underflow → ±0
    }

    (sign << 15) | ((new_exp as u16) << 10) | ((frac >> 13) as u16)
}

/// Converte f16 → f32.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as i32;
    let frac = (h & 0x3FF) as u32;

    if exp == 0x1F {
        // Inf/NaN
        let bits = (sign << 31) | 0x7F800000 | (frac << 13);
        return f32::from_bits(bits);
    }
    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign << 31); // ±0
        }
        // Subnormal
        let mut f = frac;
        let mut e = 1_i32;
        while f & 0x400 == 0 {
            f <<= 1;
            e -= 1;
        }
        f &= 0x3FF;
        let new_exp = (127 - 15 + e) as u32;
        let bits = (sign << 31) | (new_exp << 23) | (f << 13);
        return f32::from_bits(bits);
    }

    // Normal: use i32 to avoid underflow
    let new_exp = (exp - 15 + 127) as u32;
    let bits = (sign << 31) | (new_exp << 23) | (frac << 13);
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autoencoder_creates_with_correct_dims() {
        let ae = TinyAutoencoder::new(256, 64, 16);
        assert_eq!(ae.enc_w1.len(), 256 * 64);
        assert_eq!(ae.enc_w2.len(), 64 * 16);
        assert_eq!(ae.dec_w1.len(), 16 * 64);
        assert_eq!(ae.dec_w2.len(), 64 * 256);
    }

    #[test]
    fn encode_decode_produces_ternary() {
        let ae = TinyAutoencoder::new(256, 64, 16);
        let input: Vec<i8> = (0..256).map(|i| ((i % 3) as i8) - 1).collect();
        let latent = ae.encode(&input);
        assert_eq!(latent.len(), 16);
        let output = ae.decode(&latent);
        assert_eq!(output.len(), 256);
        for &w in &output {
            assert!(w == -1 || w == 0 || w == 1, "Non-ternary: {}", w);
        }
    }

    #[test]
    fn training_reduces_loss() {
        let mut ae = TinyAutoencoder::new(64, 16, 8);
        ae.learning_rate = 0.01;

        // Batch de 5 amostras simples (muitos zeros — fácil de aprender)
        let batch: Vec<Vec<i8>> = (0..5)
            .map(|k| {
                (0..64)
                    .map(|i| if (i + k) % 10 == 0 { 1 } else { 0 })
                    .collect()
            })
            .collect();

        let (loss0, _) = ae.train_batch(&batch);
        // Treinar mais epochs
        let mut last_loss = loss0;
        for _ in 0..20 {
            let (loss, _) = ae.train_batch(&batch);
            last_loss = loss;
        }

        assert!(
            last_loss < loss0,
            "Loss should decrease: {:.4} → {:.4}",
            loss0,
            last_loss
        );
    }

    #[test]
    fn f16_round_trip() {
        let values = [0.0f32, 1.0, -1.0, 0.5, -0.333, 100.0, -0.001];
        for &v in &values {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            let err = (v - back).abs();
            // f16 has ~3 decimal digits of precision
            let tolerance = v.abs() * 0.01 + 0.001;
            assert!(
                err < tolerance,
                "f16 round-trip failed: {} → {} (err={})",
                v,
                back,
                err
            );
        }
    }

    #[test]
    fn latent_f16_serialization() {
        let latent = vec![0.1, -0.5, 1.0, 0.0, -1.0, 0.333];
        let bytes = TinyAutoencoder::latent_to_f16_bytes(&latent);
        assert_eq!(bytes.len(), latent.len() * 2);
        let restored = TinyAutoencoder::latent_from_f16_bytes(&bytes);
        assert_eq!(restored.len(), latent.len());
        for (a, b) in latent.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 0.01, "Mismatch: {} vs {}", a, b);
        }
    }
}
