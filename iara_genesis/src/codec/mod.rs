//! # Codec System — Compressão inteligente de pesos ternários
//!
//! Trait unificada para todos os codecs + selector automático (HybridCodec).

pub mod math_codec;
pub mod neural_codec;
pub mod semantic;

/// Resultado de uma compressão: bytes + metadados.
#[derive(Clone, Debug)]
pub struct Compressed {
    /// Identificador do codec usado (4 bytes ASCII).
    pub codec_tag: [u8; 4],
    /// Dados comprimidos.
    pub data: Vec<u8>,
    /// Comprimento original do vetor de pesos.
    pub original_len: usize,
}

impl Compressed {
    /// Serializa para bytes: tag[4] + original_len[4] + data_len[4] + data.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(12 + self.data.len());
        buf.extend_from_slice(&self.codec_tag);
        buf.extend_from_slice(&(self.original_len as u32).to_le_bytes());
        buf.extend_from_slice(&(self.data.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.data);
        buf
    }

    /// Desserializa de bytes.
    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < 12 { return None; }
        let mut codec_tag = [0u8; 4];
        codec_tag.copy_from_slice(&buf[0..4]);
        let original_len = u32::from_le_bytes(buf[4..8].try_into().ok()?) as usize;
        let data_len = u32::from_le_bytes(buf[8..12].try_into().ok()?) as usize;
        if buf.len() < 12 + data_len { return None; }
        let data = buf[12..12 + data_len].to_vec();
        Some(Self { codec_tag, original_len, data })
    }

    /// Ratio de compressão.
    pub fn ratio(&self) -> f64 {
        if self.data.is_empty() { return 0.0; }
        self.original_len as f64 / self.data.len() as f64
    }
}

/// Trait para codecs de compressão de pesos ternários.
pub trait Codec {
    /// Nome do codec (4 bytes ASCII).
    fn tag(&self) -> [u8; 4];

    /// Comprime um vetor de pesos ternários.
    fn encode(&self, weights: &[i8]) -> Compressed;

    /// Descomprime de volta para pesos ternários.
    fn decode(&self, compressed: &Compressed) -> Option<Vec<i8>>;
}

/// Calcula a fidelidade ternária: % de pesos que são idênticos.
pub fn ternary_accuracy(original: &[i8], restored: &[i8]) -> f64 {
    if original.len() != restored.len() || original.is_empty() {
        return 0.0;
    }
    let matches = original.iter().zip(restored.iter())
        .filter(|&(&a, &b)| a == b)
        .count();
    matches as f64 / original.len() as f64
}

/// Calcula o erro funcional: quanto os pesos divergem em magnitude.
/// Útil para quando pesos -1 e 0 são "close enough" mas -1 e +1 não.
pub fn functional_error(original: &[i8], restored: &[i8]) -> f64 {
    if original.len() != restored.len() || original.is_empty() {
        return f64::MAX;
    }
    let sum_sq: f64 = original.iter().zip(restored.iter())
        .map(|(a, b)| {
            let diff = (*a as f64) - (*b as f64);
            diff * diff
        })
        .sum();
    (sum_sq / original.len() as f64).sqrt()
}

/// HybridCodec: testa todos os codecs e escolhe o que produz
/// o menor resultado com fidelidade acima do threshold.
pub struct HybridCodec {
    pub min_accuracy: f64,
}

impl HybridCodec {
    pub fn new() -> Self {
        Self { min_accuracy: 0.90 }
    }

    pub fn with_min_accuracy(min_accuracy: f64) -> Self {
        Self { min_accuracy }
    }

    /// Auto-seleciona o melhor codec para este vetor de pesos.
    pub fn encode(&self, weights: &[i8]) -> Compressed {
        let codecs: Vec<Box<dyn Codec>> = vec![
            Box::new(math_codec::SparseRleCodec),
            Box::new(math_codec::FourierCodec::new(32)),
            Box::new(math_codec::FourierCodec::new(64)),
            Box::new(math_codec::ChebyshevCodec::new(16)),
            Box::new(math_codec::ChebyshevCodec::new(32)),
        ];

        let mut best: Option<Compressed> = None;
        let mut best_score = f64::NEG_INFINITY;

        for codec in &codecs {
            let compressed = codec.encode(weights);
            if let Some(restored) = codec.decode(&compressed) {
                let accuracy = ternary_accuracy(weights, &restored);
                if accuracy >= self.min_accuracy {
                    // Valorizamos compressão lossless imensamente (SparseRLE em sinais esparsos)
                    // Se for 100% perfeito, damos peso 10x à compressão para garantir que vença opções lossy
                    let bonus = if (accuracy - 1.0).abs() < f64::EPSILON { 10.0 } else { 1.0 };
                    let score = accuracy * compressed.ratio() * bonus;
                    if score > best_score {
                        best_score = score;
                        best = Some(compressed);
                    }
                }
            }
        }

        // Fallback: SparseRLE é sempre lossless
        best.unwrap_or_else(|| math_codec::SparseRleCodec.encode(weights))
    }

    /// Decodifica detectando automaticamente o codec pela tag.
    pub fn decode(&self, compressed: &Compressed) -> Option<Vec<i8>> {
        match &compressed.codec_tag {
            b"SRLE" => math_codec::SparseRleCodec.decode(compressed),
            b"FFT1" | b"FFT2" => {
                // Fourier — o número de harmônicos é inferido dos dados
                math_codec::FourierCodec::new(32).decode(compressed)
            },
            b"CHB1" | b"CHB2" => {
                math_codec::ChebyshevCodec::new(16).decode(compressed)
            },
            b"NEUR" => neural_codec::NeuralCodec::default_codec().decode(compressed),
            _ => None,
        }
    }
}

impl Default for HybridCodec {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ternary_accuracy_identical() {
        let a = vec![0i8, 1, -1, 0, 1];
        assert!((ternary_accuracy(&a, &a) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ternary_accuracy_half_match() {
        let a = vec![0i8, 1, -1, 0];
        let b = vec![0i8, -1, 1, 0];
        assert!((ternary_accuracy(&a, &b) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn compressed_serialization() {
        let c = Compressed {
            codec_tag: *b"TEST",
            data: vec![1, 2, 3, 4, 5],
            original_len: 100,
        };
        let bytes = c.to_bytes();
        let c2 = Compressed::from_bytes(&bytes).unwrap();
        assert_eq!(c.codec_tag, c2.codec_tag);
        assert_eq!(c.original_len, c2.original_len);
        assert_eq!(c.data, c2.data);
    }

    #[test]
    fn hybrid_codec_picks_best() {
        let codec = HybridCodec::new();

        // Sparse signal: should prefer SRLE
        let sparse: Vec<i8> = {
            let mut v = vec![0i8; 4096];
            for i in (0..4096).step_by(100) { v[i] = 1; }
            v
        };
        let compressed = codec.encode(&sparse);
        let restored = codec.decode(&compressed).unwrap();
        assert_eq!(sparse, restored, "Hybrid must be lossless for sparse");

        let codec_used = String::from_utf8_lossy(&compressed.codec_tag);
        println!("Codec picked: {}", codec_used);
        println!("Compressed Size: {}", compressed.data.len());
        println!("Ratio: {}", compressed.ratio());

        // Check ratio is good. Sparse RLE ratio on 4096 bytes with 40 non-zeros is very high.
        // Data length might be ~50-100 bytes.
        let ratio = sparse.len() as f64 / compressed.data.len() as f64;
        assert!(ratio > 5.0, "Sparse ratio: {:.1}:1 codec: {}", ratio, codec_used);
    }
}
