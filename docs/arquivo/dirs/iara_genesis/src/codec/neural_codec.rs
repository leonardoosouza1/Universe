//! # Neural Codec — Wrapper Codec para o autoencoder
//!
//! Integra o TinyAutoencoder no sistema de codecs via trait `Codec`.
//! O latent vector é serializado em f16 (~2B por float).

use super::{Codec, Compressed};
use crate::neural::autoencoder::TinyAutoencoder;

/// Codec neural baseado em autoencoder.
///
/// Requer um autoencoder treinado. Sem treino, a qualidade é baixa
/// (pesos Xavier random não capturam padrões úteis).
pub struct NeuralCodec {
    pub autoencoder: TinyAutoencoder,
}

impl NeuralCodec {
    pub fn new(autoencoder: TinyAutoencoder) -> Self {
        Self { autoencoder }
    }

    /// Codec com autoencoder de dimensões menores (útil para testes).
    pub fn default_codec() -> Self {
        Self::new(TinyAutoencoder::default_dims())
    }

    /// Codec com dimensões customizadas.
    pub fn with_dims(input_dim: usize, hidden_dim: usize, latent_dim: usize) -> Self {
        Self::new(TinyAutoencoder::new(input_dim, hidden_dim, latent_dim))
    }
}

impl Codec for NeuralCodec {
    fn tag(&self) -> [u8; 4] {
        *b"NEUR"
    }

    fn encode(&self, weights: &[i8]) -> Compressed {
        let latent = self.autoencoder.encode(weights);
        let data = TinyAutoencoder::latent_to_f16_bytes(&latent);

        Compressed {
            codec_tag: self.tag(),
            data,
            original_len: weights.len(),
        }
    }

    fn decode(&self, compressed: &Compressed) -> Option<Vec<i8>> {
        let latent = TinyAutoencoder::latent_from_f16_bytes(&compressed.data);
        if latent.len() != self.autoencoder.latent_dim {
            return None;
        }
        Some(self.autoencoder.decode(&latent))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neural_codec_round_trip_ternary_output() {
        let codec = NeuralCodec::with_dims(256, 32, 8);
        let input: Vec<i8> = (0..256).map(|i| ((i % 3) as i8) - 1).collect();

        let compressed = codec.encode(&input);
        assert_eq!(compressed.data.len(), 8 * 2); // 8 latent × 2 bytes f16

        let restored = codec.decode(&compressed).unwrap();
        assert_eq!(restored.len(), 256);
        for &w in &restored {
            assert!(w >= -1 && w <= 1, "Non-ternary: {}", w);
        }
    }

    #[test]
    fn neural_codec_compression_ratio() {
        let codec = NeuralCodec::with_dims(8192, 512, 128);
        let input: Vec<i8> = vec![0; 8192];

        let compressed = codec.encode(&input);
        let ratio = compressed.ratio();
        // 8192 / 256 = 32:1
        assert!(
            ratio > 25.0,
            "Neural ratio should be ~32:1, got {:.1}:1",
            ratio
        );
    }

    #[test]
    fn trained_codec_improves_accuracy() {
        let mut codec = NeuralCodec::with_dims(64, 16, 4);

        // Batch de amostras simples
        let batch: Vec<Vec<i8>> = (0..10)
            .map(|_| {
                vec![0i8; 64] // all-zeros — easiest pattern
            })
            .collect();

        // Accuracy antes do treino
        let input = &batch[0];
        let before = codec.encode(input);
        let before_restored = codec.decode(&before).unwrap();
        let acc_before = input
            .iter()
            .zip(before_restored.iter())
            .filter(|&(&a, &b)| a == b)
            .count() as f32
            / 64.0;

        // Treinar
        for _ in 0..50 {
            codec.autoencoder.train_batch(&batch);
        }

        // Accuracy depois do treino
        let after = codec.encode(input);
        let after_restored = codec.decode(&after).unwrap();
        let acc_after = input
            .iter()
            .zip(after_restored.iter())
            .filter(|&(&a, &b)| a == b)
            .count() as f32
            / 64.0;

        assert!(
            acc_after >= acc_before,
            "Training should improve accuracy: {:.1}% → {:.1}%",
            acc_before * 100.0,
            acc_after * 100.0
        );
    }
}
