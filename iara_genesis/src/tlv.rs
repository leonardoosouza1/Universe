//! # TLV Integration — Chunks para DNA V4
//!
//! Integra as representações comprimidas no formato TLV (Tag-Length-Value)
//! do DNA V4. Cada componente do iara_genesis mapeia para um chunk V4.
//!
//! ## Novos Chunks
//!
//! | Tag    | Conteúdo | Típico |
//! |--------|----------|--------|
//! | `GENE` | Genome comprimido (HybridCodec) | 400-800B |
//! | `CDBK` | Codebook do autoencoder | ~4KB |
//! | `SOIL` | Campo de energia (Fourier 2D) | ~200B |
//! | `TREE` | Estrutura generativa completa | ~500B |
//! | `BHVR` | Fingerprint comportamental | 12B |

use crate::codec::{Compressed, HybridCodec, ternary_accuracy};
use crate::codec::semantic::BehaviorFingerprint;
use crate::model::tree::{Tree, Sap};
use crate::model::soil::Soil;
use crate::neural::{autoencoder::TinyAutoencoder, codebook};

/// Tags dos chunks iara_genesis no TLV V4.
pub const TAG_GENE: &[u8; 4] = b"GENE";
pub const TAG_CDBK: &[u8; 4] = b"CDBK";
pub const TAG_SOIL: &[u8; 4] = b"SOIL";
pub const TAG_TREE: &[u8; 4] = b"TREE";
pub const TAG_BHVR: &[u8; 4] = b"BHVR";

/// Um chunk TLV genérico.
#[derive(Clone, Debug)]
pub struct Chunk {
    pub tag: [u8; 4],
    pub data: Vec<u8>,
}

impl Chunk {
    pub fn new(tag: [u8; 4], data: Vec<u8>) -> Self {
        Self { tag, data }
    }

    /// Serializa: tag[4] + length[4] + data[length].
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8 + self.data.len());
        buf.extend_from_slice(&self.tag);
        buf.extend_from_slice(&(self.data.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.data);
        buf
    }

    /// Desserializa um chunk de bytes (retorna chunk + bytes consumidos).
    pub fn from_bytes(buf: &[u8]) -> Option<(Self, usize)> {
        if buf.len() < 8 { return None; }

        let mut tag = [0u8; 4];
        tag.copy_from_slice(&buf[0..4]);
        let length = u32::from_le_bytes(buf[4..8].try_into().ok()?) as usize;

        if buf.len() < 8 + length { return None; }
        let data = buf[8..8 + length].to_vec();

        Some((Self { tag, data }, 8 + length))
    }
}

/// Container de chunks para um organismo comprimido.
#[derive(Clone, Debug)]
pub struct GenesisPayload {
    pub chunks: Vec<Chunk>,
}

impl GenesisPayload {
    pub fn new() -> Self {
        Self { chunks: Vec::new() }
    }

    pub fn add_chunk(&mut self, chunk: Chunk) {
        self.chunks.push(chunk);
    }

    /// Encontra um chunk por tag.
    pub fn find_chunk(&self, tag: &[u8; 4]) -> Option<&Chunk> {
        self.chunks.iter().find(|c| &c.tag == tag)
    }

    /// Serializa todos os chunks: magic[4] + version[2] + num_chunks[4] + chunks...
    pub fn to_bytes(&self) -> Vec<u8> {
        let data_size: usize = self.chunks.iter().map(|c| 8 + c.data.len()).sum();
        let mut buf = Vec::with_capacity(10 + data_size);

        buf.extend_from_slice(b"GNSS"); // Genesis magic
        buf.extend_from_slice(&1u16.to_le_bytes()); // version
        buf.extend_from_slice(&(self.chunks.len() as u32).to_le_bytes());

        for chunk in &self.chunks {
            buf.extend_from_slice(&chunk.to_bytes());
        }

        buf
    }

    /// Desserializa de bytes.
    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < 10 { return None; }
        if &buf[0..4] != b"GNSS" { return None; }

        let _version = u16::from_le_bytes(buf[4..6].try_into().ok()?);
        let num_chunks = u32::from_le_bytes(buf[6..10].try_into().ok()?) as usize;

        let mut chunks = Vec::with_capacity(num_chunks);
        let mut off = 10;

        for _ in 0..num_chunks {
            let (chunk, consumed) = Chunk::from_bytes(&buf[off..])?;
            chunks.push(chunk);
            off += consumed;
        }

        Some(Self { chunks })
    }

    /// Tamanho total em bytes.
    pub fn total_size(&self) -> usize {
        10 + self.chunks.iter().map(|c| 8 + c.data.len()).sum::<usize>()
    }
}

impl Default for GenesisPayload {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// High-level API — Compress/Decompress an organism
// ============================================================================

/// Comprime um organismo completo para GenesisPayload.
///
/// Combina: GENE (HybridCodec), TREE, BHVR.
pub fn compress_organism(
    fast_weights: &[i8],
    slow_weights: &[i8],
    generation: u32,
    fitness: f32,
    sap: Sap,
) -> GenesisPayload {
    let mut payload = GenesisPayload::new();

    // 1. GENE: HybridCodec compression (melhor codec automático)
    let codec = HybridCodec::new();
    let compressed = codec.encode(fast_weights);
    payload.add_chunk(Chunk::new(*TAG_GENE, compressed.to_bytes()));

    // 2. TREE: representação generativa completa
    let tree = Tree::plant(fast_weights, slow_weights, generation, fitness, sap);
    payload.add_chunk(Chunk::new(*TAG_TREE, tree.serialize()));

    // 3. BHVR: fingerprint comportamental
    let fp = BehaviorFingerprint::from_weights(fast_weights);
    let mut bhvr_data = Vec::with_capacity(12);
    bhvr_data.extend_from_slice(&fp.to_bytes());
    bhvr_data.extend_from_slice(&fitness.to_le_bytes());
    payload.add_chunk(Chunk::new(*TAG_BHVR, bhvr_data));

    payload
}

/// Descomprime um organismo do GenesisPayload.
///
/// Estratégia: tenta TREE primeiro (mais completo), fallback para GENE.
/// Retorna (fast_weights, slow_weights, generation, fitness).
pub fn decompress_organism(payload: &GenesisPayload) -> Option<(Vec<i8>, Vec<i8>, u32, f32)> {
    // Tentar TREE primeiro (contém fast + slow + metadata)
    if let Some(tree_chunk) = payload.find_chunk(TAG_TREE) {
        let tree = Tree::deserialize(&tree_chunk.data)?;
        let (fast, slow) = tree.harvest();
        return Some((fast, slow, tree.generation, tree.fitness));
    }

    // Fallback: GENE (apenas fast_weights, sem slow)
    if let Some(gene_chunk) = payload.find_chunk(TAG_GENE) {
        let compressed = Compressed::from_bytes(&gene_chunk.data)?;
        let codec = HybridCodec::new();
        let fast = codec.decode(&compressed)?;
        return Some((fast, Vec::new(), 0, 0.0));
    }

    None
}

/// Adiciona um codebook (autoencoder) ao payload.
pub fn attach_codebook(payload: &mut GenesisPayload, autoencoder: &TinyAutoencoder) {
    let data = codebook::serialize(autoencoder);
    payload.add_chunk(Chunk::new(*TAG_CDBK, data));
}

/// Extrai codebook do payload.
pub fn extract_codebook(payload: &GenesisPayload) -> Option<TinyAutoencoder> {
    let chunk = payload.find_chunk(TAG_CDBK)?;
    codebook::deserialize(&chunk.data)
}

/// Adiciona Solo (campo de energia) ao payload.
pub fn attach_soil(payload: &mut GenesisPayload, soil: &Soil) {
    payload.add_chunk(Chunk::new(*TAG_SOIL, soil.serialize()));
}

/// Extrai Solo do payload.
pub fn extract_soil(payload: &GenesisPayload) -> Option<Soil> {
    let chunk = payload.find_chunk(TAG_SOIL)?;
    Soil::deserialize(&chunk.data)
}

/// Report de compressão: métricas detalhadas.
#[derive(Debug)]
pub struct CompressionReport {
    pub original_size: usize,
    pub compressed_size: usize,
    pub ratio: f64,
    pub fast_accuracy: f64,
    pub slow_accuracy: f64,
    pub codec_used: String,
    pub tree_corrections: usize,
}

/// Gera report de compressão detalhado.
pub fn compression_report(
    fast_weights: &[i8],
    slow_weights: &[i8],
    generation: u32,
    fitness: f32,
) -> CompressionReport {
    let original_size = fast_weights.len() + slow_weights.len();

    let payload = compress_organism(fast_weights, slow_weights, generation, fitness, Sap::default());
    let compressed_size = payload.total_size();

    let (restored_fast, restored_slow, _, _) = decompress_organism(&payload)
        .unwrap_or((vec![0; fast_weights.len()], vec![0; slow_weights.len()], 0, 0.0));

    let fast_accuracy = ternary_accuracy(fast_weights, &restored_fast);
    let slow_accuracy = if slow_weights.is_empty() { 1.0 } else {
        ternary_accuracy(slow_weights, &restored_slow)
    };

    let codec_used = payload.find_chunk(TAG_GENE)
        .and_then(|c| Compressed::from_bytes(&c.data))
        .map(|c| String::from_utf8_lossy(&c.codec_tag).to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let tree_corrections = payload.find_chunk(TAG_TREE)
        .and_then(|c| Tree::deserialize(&c.data))
        .map(|t| t.leaves.correction_count)
        .unwrap_or(0);

    CompressionReport {
        original_size,
        compressed_size,
        ratio: original_size as f64 / compressed_size as f64,
        fast_accuracy,
        slow_accuracy,
        codec_used,
        tree_corrections,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_fast() -> Vec<i8> {
        (0..8192).map(|i| {
            let v = (i as f64 * std::f64::consts::PI / 64.0).sin();
            if v > 0.5 { 1 } else if v < -0.5 { -1 } else { 0 }
        }).collect()
    }

    fn sample_slow() -> Vec<i8> {
        (0..256).map(|i| ((i * 5 + 1) % 3 - 1) as i8).collect()
    }

    #[test]
    fn full_pipeline_round_trip() {
        let fast = sample_fast();
        let slow = sample_slow();

        let payload = compress_organism(&fast, &slow, 42, 999.0, Sap::default());

        // Tem todos os chunks
        assert!(payload.find_chunk(TAG_GENE).is_some());
        assert!(payload.find_chunk(TAG_TREE).is_some());
        assert!(payload.find_chunk(TAG_BHVR).is_some());

        // Round-trip
        let (r_fast, r_slow, gen_id, fit) = decompress_organism(&payload).unwrap();
        assert_eq!(gen_id, 42);
        assert!((fit - 999.0).abs() < 1e-3);
        assert_eq!(r_slow, slow); // slow must be lossless
        assert_eq!(r_fast.len(), fast.len());
    }

    #[test]
    fn payload_serialization() {
        let fast = sample_fast();
        let slow = sample_slow();

        let payload = compress_organism(&fast, &slow, 1, 50.0, Sap::default());
        let bytes = payload.to_bytes();
        let restored = GenesisPayload::from_bytes(&bytes).unwrap();

        assert_eq!(restored.chunks.len(), payload.chunks.len());
        assert_eq!(restored.total_size(), payload.total_size());
    }

    #[test]
    fn codebook_attachment() {
        let ae = TinyAutoencoder::new(64, 16, 8);
        let mut payload = GenesisPayload::new();
        attach_codebook(&mut payload, &ae);

        let restored = extract_codebook(&payload).unwrap();
        assert_eq!(restored.input_dim, 64);
        assert_eq!(restored.hidden_dim, 16);
    }

    #[test]
    fn soil_attachment() {
        let soil = Soil::neutral(4, 4);
        let mut payload = GenesisPayload::new();
        attach_soil(&mut payload, &soil);

        let restored = extract_soil(&payload).unwrap();
        assert!((restored.temperature - soil.temperature).abs() < 1e-6);
    }

    #[test]
    fn compression_report_works() {
        let fast = sample_fast();
        let slow = sample_slow();

        let report = compression_report(&fast, &slow, 10, 100.0);

        assert!(report.ratio > 0.5, "Ratio: {:.1}:1", report.ratio);
        assert!(report.slow_accuracy > 0.99, "Slow acc: {:.1}%", report.slow_accuracy * 100.0);
        assert!(report.fast_accuracy > 0.50, "Fast acc: {:.1}%", report.fast_accuracy * 100.0);

        println!("=== Compression Report ===");
        println!("  Original: {} bytes", report.original_size);
        println!("  Compressed: {} bytes", report.compressed_size);
        println!("  Ratio: {:.1}:1", report.ratio);
        println!("  Fast accuracy: {:.1}%", report.fast_accuracy * 100.0);
        println!("  Slow accuracy: {:.1}%", report.slow_accuracy * 100.0);
        println!("  Codec: {}", report.codec_used);
        println!("  Tree corrections: {}", report.tree_corrections);
    }
}
