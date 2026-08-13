//! # Math-based Codecs
//!
//! Wrappers que implementam o trait `Codec` para Fourier, Chebyshev e Sparse RLE.

use super::{Codec, Compressed};
use crate::math::{fourier, chebyshev, sparse};

// ============================================================================
// Sparse RLE Codec — LOSSLESS
// ============================================================================

/// Codec lossless baseado em Run-Length Encoding para ternários.
pub struct SparseRleCodec;

impl Codec for SparseRleCodec {
    fn tag(&self) -> [u8; 4] { *b"SRLE" }

    fn encode(&self, weights: &[i8]) -> Compressed {
        let rle = sparse::compress(weights);
        let data = sparse::serialize(&rle);
        Compressed {
            codec_tag: self.tag(),
            data,
            original_len: weights.len(),
        }
    }

    fn decode(&self, compressed: &Compressed) -> Option<Vec<i8>> {
        let rle = sparse::deserialize(&compressed.data)?;
        Some(sparse::decompress(&rle))
    }
}

// ============================================================================
// Fourier Codec — LOSSY (alta fidelidade)
// ============================================================================

/// Codec baseado em Fourier DFT: decompõe pesos em harmônicos.
pub struct FourierCodec {
    pub num_harmonics: usize,
}

impl FourierCodec {
    pub fn new(num_harmonics: usize) -> Self {
        Self { num_harmonics }
    }
}

impl Codec for FourierCodec {
    fn tag(&self) -> [u8; 4] {
        if self.num_harmonics <= 32 { *b"FFT1" } else { *b"FFT2" }
    }

    fn encode(&self, weights: &[i8]) -> Compressed {
        if weights.is_empty() {
            return Compressed { codec_tag: self.tag(), data: vec![], original_len: 0 };
        }
        let fc = fourier::analyze(weights, self.num_harmonics);
        let data = fourier::serialize(&fc);
        Compressed {
            codec_tag: self.tag(),
            data,
            original_len: weights.len(),
        }
    }

    fn decode(&self, compressed: &Compressed) -> Option<Vec<i8>> {
        if compressed.original_len == 0 { return Some(vec![]); }
        let fc = fourier::deserialize(&compressed.data)?;
        Some(fourier::synthesize(&fc))
    }
}

// ============================================================================
// Chebyshev Codec — LOSSY (máxima compressão)
// ============================================================================

/// Codec baseado em polinômios de Chebyshev: máxima compressão para pesos suaves.
pub struct ChebyshevCodec {
    pub num_terms: usize,
}

impl ChebyshevCodec {
    pub fn new(num_terms: usize) -> Self {
        Self { num_terms }
    }
}

impl Codec for ChebyshevCodec {
    fn tag(&self) -> [u8; 4] {
        if self.num_terms <= 16 { *b"CHB1" } else { *b"CHB2" }
    }

    fn encode(&self, weights: &[i8]) -> Compressed {
        if weights.is_empty() {
            return Compressed { codec_tag: self.tag(), data: vec![], original_len: 0 };
        }
        let cc = chebyshev::fit(weights, self.num_terms);
        let data = chebyshev::serialize(&cc);
        Compressed {
            codec_tag: self.tag(),
            data,
            original_len: weights.len(),
        }
    }

    fn decode(&self, compressed: &Compressed) -> Option<Vec<i8>> {
        if compressed.original_len == 0 { return Some(vec![]); }
        let cc = chebyshev::deserialize(&compressed.data)?;
        Some(chebyshev::reconstruct(&cc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::ternary_accuracy;

    #[test]
    fn sparse_rle_always_lossless() {
        let weights: Vec<i8> = (0..8192)
            .map(|i| [0, 1, -1, 0, 0][i % 5])
            .collect();
        let codec = SparseRleCodec;
        let compressed = codec.encode(&weights);
        let restored = codec.decode(&compressed).unwrap();
        assert_eq!(weights, restored);
    }

    #[test]
    fn fourier_reasonable_accuracy() {
        // Sinal estruturado
        let weights: Vec<i8> = (0..2048)
            .map(|i| {
                let v = (2.0 * std::f64::consts::PI * 5.0 * i as f64 / 2048.0).sin();
                if v > 0.3 { 1 } else if v < -0.3 { -1 } else { 0 }
            })
            .collect();
        let codec = FourierCodec::new(32);
        let compressed = codec.encode(&weights);
        let restored = codec.decode(&compressed).unwrap();
        let acc = ternary_accuracy(&weights, &restored);
        assert!(acc > 0.70, "Fourier accuracy on sinusoidal: {:.1}%", acc * 100.0);
    }

    #[test]
    fn chebyshev_compresses_heavily() {
        let weights: Vec<i8> = (0..8192).map(|i| ((i % 3) as i8) - 1).collect();
        let codec = ChebyshevCodec::new(16);
        let compressed = codec.encode(&weights);
        let ratio = compressed.ratio();
        assert!(ratio > 50.0, "Chebyshev 16 terms should compress > 50:1, got {:.0}:1", ratio);
    }

    #[test]
    fn all_codecs_handle_empty() {
        let empty: Vec<i8> = vec![];
        let codecs: Vec<Box<dyn Codec>> = vec![
            Box::new(SparseRleCodec),
            Box::new(FourierCodec::new(8)),
            Box::new(ChebyshevCodec::new(4)),
        ];
        for codec in &codecs {
            let compressed = codec.encode(&empty);
            let restored = codec.decode(&compressed).unwrap();
            assert_eq!(restored.len(), 0);
        }
    }
}
