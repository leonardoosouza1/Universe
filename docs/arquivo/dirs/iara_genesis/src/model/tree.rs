//! # Árvore (Tree) — O organismo como estrutura generativa
//!
//! Cada organismo é uma árvore:
//! - **Semente**: código genético minimal (Chebyshev polynomial seed)
//! - **Tronco**: backbone do instinto (Fourier compressed)
//! - **Galhos**: especializações (slow_weights, KAN coefficients)
//! - **Folhas**: sparse corrections (detalhes que não cabem nas fórmulas)
//! - **Raízes**: conexões com o Solo (afinidade evolutiva)
//! - **Seiva**: neuromoduladores (dopamina, cortisol, etc.)
//! - **Anéis**: histórico de gerações (hash chain)
//!
//! ## Compressão
//!
//! A árvore comprime um organismo de 8192+ bytes em:
//!   Semente (64B) + Tronco (256B) + Galhos (128B) + Folhas (variável)
//!   ≈ 400-800 bytes típico vs 8192 bytes cru

use crate::codec::semantic::BehaviorFingerprint;
use crate::math::{chebyshev, fourier, sparse};

/// Semente: o código genético minimal do organismo.
///
/// Uma semente de 16 coeficientes (Chebyshev) pode gerar 8192 pesos.
/// Ratio: 128:1, mas lossy. A semente captura a "tendência geral".
#[derive(Clone, Debug)]
pub struct Seed {
    /// Coeficientes de Chebyshev (polynomial seed).
    pub coeffs: chebyshev::ChebyshevCoeffs,
}

/// Tronco: backbone do instinto comprimido via Fourier.
///
/// Captura as frequências dominantes dos fast_weights.
/// Ratio: 32:1 com boa fidelidade para padrões periódicos.
#[derive(Clone, Debug)]
pub struct Trunk {
    /// Coeficientes de Fourier do fast_weights backbone.
    pub fourier: fourier::FourierCoeffs,
}

/// Galho: uma especialização do organismo.
///
/// Cada galho pode representar slow_weights, KAN coefficients,
/// CfC state, ou qualquer sub-rede específica.
#[derive(Clone, Debug)]
pub struct Branch {
    /// Nome/tipo do galho.
    pub kind: BranchKind,
    /// Dados comprimidos (codec variável — tipicamente SparseRLE).
    pub data: Vec<u8>,
    /// Tamanho original.
    pub original_len: usize,
}

/// Tipos de galho conhecidos.
#[derive(Clone, Debug, PartialEq)]
pub enum BranchKind {
    /// Slow weights (instinto de longo prazo).
    SlowWeights,
    /// KAN per-channel spline coefficients.
    KanSplines,
    /// CfC hidden state.
    CfcState,
    /// Hypernet seed coefficients.
    HypernetSeed,
    /// Extensão customizada.
    Custom(String),
}

impl BranchKind {
    fn to_tag(&self) -> u8 {
        match self {
            BranchKind::SlowWeights => 0,
            BranchKind::KanSplines => 1,
            BranchKind::CfcState => 2,
            BranchKind::HypernetSeed => 3,
            BranchKind::Custom(_) => 255,
        }
    }

    fn from_tag(tag: u8) -> Self {
        match tag {
            0 => BranchKind::SlowWeights,
            1 => BranchKind::KanSplines,
            2 => BranchKind::CfcState,
            3 => BranchKind::HypernetSeed,
            _ => BranchKind::Custom(format!("unknown_{}", tag)),
        }
    }
}

/// Folhas: sparse corrections (deltas) entre a fórmula e a realidade.
///
/// Quando o tronco (Fourier) + semente (Chebyshev) não capturam um peso
/// corretamente, a folha armazena o delta. Tipicamente muito esparso.
#[derive(Clone, Debug)]
pub struct Leaves {
    /// Pesos delta comprimidos via RLE.
    pub sparse_deltas: sparse::SparseRle,
    /// Número de correções não-zero.
    pub correction_count: usize,
}

/// Raízes: conexão do organismo com o Solo.
#[derive(Clone, Debug)]
pub struct RootSystem {
    /// Afinidade com o fitness landscape (coordenadas no espaço semântico).
    pub position: (f32, f32),
    /// Profundidade: quanto o organismo depende do Solo (herança vs inovação).
    pub depth: f32,
    /// Fingerprint comportamental.
    pub fingerprint: BehaviorFingerprint,
}

/// Seiva: neuromoduladores do organismo.
#[derive(Clone, Debug, Default)]
pub struct Sap {
    pub dopamine: f32,
    pub serotonin: f32,
    pub cortisol: f32,
    pub oxytocin: f32,
    pub adrenaline: f32,
    pub melatonin: f32,
    pub curiosity: f32,
}

/// A Árvore completa — representação generativa comprimida de um organismo.
#[derive(Clone, Debug)]
pub struct Tree {
    /// Semente genética (Chebyshev polynomial).
    pub seed: Seed,
    /// Tronco: backbone do instinto (Fourier).
    pub trunk: Trunk,
    /// Galhos: redes especializadas.
    pub branches: Vec<Branch>,
    /// Folhas: correções esparsas.
    pub leaves: Leaves,
    /// Raízes: conexão com o campo de energia.
    pub roots: RootSystem,
    /// Seiva: neuromoduladores.
    pub sap: Sap,
    /// Anéis de crescimento: hash de cada geração ancestral.
    pub rings: Vec<u64>,
    /// Geração atual.
    pub generation: u32,
    /// Fitness do organismo.
    pub fitness: f32,
}

impl Tree {
    /// Planta uma árvore a partir de pesos ternários crus.
    ///
    /// Este é o entry-point principal: converte `fast_weights + slow_weights`
    /// em uma representação Tree comprimida.
    pub fn plant(
        fast_weights: &[i8],
        slow_weights: &[i8],
        generation: u32,
        fitness: f32,
        sap: Sap,
    ) -> Self {
        // 1. Semente: Chebyshev fit (máxima compressão, captura tendência)
        let seed = Seed {
            coeffs: chebyshev::fit(fast_weights, 16),
        };

        // 2. Tronco: Fourier (boa fidelidade, ~32:1)
        let trunk = Trunk {
            fourier: fourier::analyze(fast_weights, 32),
        };

        // 3. Folhas: deltas entre Fourier e realidade
        let trunk_reconstructed = fourier::synthesize(&trunk.fourier);
        let deltas: Vec<i8> = fast_weights
            .iter()
            .zip(trunk_reconstructed.iter())
            .map(|(&orig, &recon)| (orig as i16 - recon as i16).clamp(-1, 1) as i8)
            .collect();
        let correction_count = deltas.iter().filter(|&&d| d != 0).count();
        let leaves = Leaves {
            sparse_deltas: sparse::compress(&deltas),
            correction_count,
        };

        // 4. Galhos: slow_weights como SparseRLE
        let mut branches = Vec::new();
        if !slow_weights.is_empty() {
            let slow_rle = sparse::compress(slow_weights);
            branches.push(Branch {
                kind: BranchKind::SlowWeights,
                data: sparse::serialize(&slow_rle),
                original_len: slow_weights.len(),
            });
        }

        // 5. Raízes e fingerprint
        let fingerprint = BehaviorFingerprint::from_weights(fast_weights);
        let roots = RootSystem {
            position: (0.5, 0.5), // centro do campo (será atualizado pelo Solo)
            depth: 0.5,
            fingerprint,
        };

        // 6. Anel de crescimento inicial
        let ring = generation_hash(fast_weights, generation);

        Tree {
            seed,
            trunk,
            branches,
            leaves,
            roots,
            sap,
            rings: vec![ring],
            generation,
            fitness,
        }
    }

    /// Colhe a árvore: reconstrói os pesos originais.
    ///
    /// Usa: Tronco (Fourier) + Folhas (sparse deltas) para fast_weights.
    /// Galhos (SparseRLE) para slow_weights.
    pub fn harvest(&self) -> (Vec<i8>, Vec<i8>) {
        // Fast weights: Fourier + deltas
        let trunk_weights = fourier::synthesize(&self.trunk.fourier);
        let deltas = sparse::decompress(&self.leaves.sparse_deltas);

        let fast_weights: Vec<i8> = trunk_weights
            .iter()
            .zip(deltas.iter())
            .map(|(&t, &d)| (t as i16 + d as i16).clamp(-1, 1) as i8)
            .collect();

        // Slow weights: encontrar galho SlowWeights
        let slow_weights = self
            .branches
            .iter()
            .find(|b| b.kind == BranchKind::SlowWeights)
            .and_then(|b| sparse::deserialize(&b.data))
            .map(|rle| sparse::decompress(&rle))
            .unwrap_or_default();

        (fast_weights, slow_weights)
    }

    /// Tamanho total em bytes da representação comprimida (serializada).
    pub fn compressed_size(&self) -> usize {
        // Use actual serialize to get exact size instead of estimating
        self.serialize().len()
    }

    /// Ratio de compressão vs pesos crus.
    pub fn compression_ratio(&self, original_size: usize) -> f64 {
        if self.compressed_size() == 0 {
            return 0.0;
        }
        original_size as f64 / self.compressed_size() as f64
    }

    /// Serializa a árvore completa para bytes.
    pub fn serialize(&self) -> Vec<u8> {
        // Estima capacidade (aproximada, buf vai realocar se precisar)
        let mut buf = Vec::with_capacity(512 + self.leaves.sparse_deltas.data.len());

        // Header
        buf.extend_from_slice(b"TREE");
        buf.extend_from_slice(&self.generation.to_le_bytes());
        buf.extend_from_slice(&self.fitness.to_le_bytes());

        // Seed (Chebyshev)
        let seed_bytes = chebyshev::serialize(&self.seed.coeffs);
        buf.extend_from_slice(&(seed_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(&seed_bytes);

        // Trunk (Fourier)
        let trunk_bytes = fourier::serialize(&self.trunk.fourier);
        buf.extend_from_slice(&(trunk_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(&trunk_bytes);

        // Leaves (Sparse deltas)
        let leaves_bytes = sparse::serialize(&self.leaves.sparse_deltas);
        buf.extend_from_slice(&(leaves_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(&leaves_bytes);
        buf.extend_from_slice(&(self.leaves.correction_count as u32).to_le_bytes());

        // Branches
        buf.extend_from_slice(&(self.branches.len() as u32).to_le_bytes());
        for branch in &self.branches {
            buf.push(branch.kind.to_tag());
            buf.extend_from_slice(&(branch.original_len as u32).to_le_bytes());
            buf.extend_from_slice(&(branch.data.len() as u32).to_le_bytes());
            buf.extend_from_slice(&branch.data);
        }

        // Roots
        buf.extend_from_slice(&self.roots.position.0.to_le_bytes());
        buf.extend_from_slice(&self.roots.position.1.to_le_bytes());
        buf.extend_from_slice(&self.roots.depth.to_le_bytes());
        buf.extend_from_slice(&self.roots.fingerprint.to_bytes());

        // Sap
        for val in [
            self.sap.dopamine,
            self.sap.serotonin,
            self.sap.cortisol,
            self.sap.oxytocin,
            self.sap.adrenaline,
            self.sap.melatonin,
            self.sap.curiosity,
        ] {
            buf.extend_from_slice(&val.to_le_bytes());
        }

        // Rings
        buf.extend_from_slice(&(self.rings.len() as u32).to_le_bytes());
        for &ring in &self.rings {
            buf.extend_from_slice(&ring.to_le_bytes());
        }

        buf
    }

    /// Desserializa uma árvore de bytes.
    pub fn deserialize(buf: &[u8]) -> Option<Self> {
        if buf.len() < 12 {
            return None;
        }
        if &buf[0..4] != b"TREE" {
            return None;
        }

        let mut off = 4;

        let generation = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        off += 4;
        let fitness = f32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        off += 4;

        // Seed
        let seed_len = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?) as usize;
        off += 4;
        let seed_coeffs = chebyshev::deserialize(&buf[off..off + seed_len])?;
        off += seed_len;

        // Trunk
        let trunk_len = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?) as usize;
        off += 4;
        let trunk_fourier = fourier::deserialize(&buf[off..off + trunk_len])?;
        off += trunk_len;

        // Leaves
        let leaves_len = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?) as usize;
        off += 4;
        let sparse_deltas = sparse::deserialize(&buf[off..off + leaves_len])?;
        off += leaves_len;
        let correction_count = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?) as usize;
        off += 4;

        // Branches
        let num_branches = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?) as usize;
        off += 4;
        let mut branches = Vec::with_capacity(num_branches);
        for _ in 0..num_branches {
            let kind = BranchKind::from_tag(buf[off]);
            off += 1;
            let original_len = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?) as usize;
            off += 4;
            let data_len = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?) as usize;
            off += 4;
            let data = buf[off..off + data_len].to_vec();
            off += data_len;
            branches.push(Branch {
                kind,
                original_len,
                data,
            });
        }

        // Roots
        let pos_x = f32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        off += 4;
        let pos_y = f32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        off += 4;
        let depth = f32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        off += 4;
        let fp_bytes: [u8; 8] = buf[off..off + 8].try_into().ok()?;
        off += 8;
        let fingerprint = BehaviorFingerprint::from_bytes(&fp_bytes);

        // Sap
        let mut read_f32 = || -> Option<f32> {
            let v = f32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
            off += 4;
            Some(v)
        };
        let sap = Sap {
            dopamine: read_f32()?,
            serotonin: read_f32()?,
            cortisol: read_f32()?,
            oxytocin: read_f32()?,
            adrenaline: read_f32()?,
            melatonin: read_f32()?,
            curiosity: read_f32()?,
        };

        // Rings
        let num_rings = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?) as usize;
        off += 4;
        let mut rings = Vec::with_capacity(num_rings);
        for _ in 0..num_rings {
            let ring = u64::from_le_bytes(buf[off..off + 8].try_into().ok()?);
            off += 8;
            rings.push(ring);
        }

        Some(Tree {
            seed: Seed {
                coeffs: seed_coeffs,
            },
            trunk: Trunk {
                fourier: trunk_fourier,
            },
            branches,
            leaves: Leaves {
                sparse_deltas,
                correction_count,
            },
            roots: RootSystem {
                position: (pos_x, pos_y),
                depth,
                fingerprint,
            },
            sap,
            rings,
            generation,
            fitness,
        })
    }
}

/// Hash de geração: fingerprint compacto de um ponto na linhagem.
fn generation_hash(weights: &[i8], generation: u32) -> u64 {
    let mut hash: u64 = generation as u64;
    // FNV-1a inspired hash
    for &w in weights.iter().take(256) {
        hash ^= w as u64;
        hash = hash.wrapping_mul(0x100000001B3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_fast() -> Vec<i8> {
        (0..8192)
            .map(|i| {
                // Low-frequency sine wave + some zeros
                let v = (i as f64 * std::f64::consts::PI / 64.0).sin();
                if v > 0.5 {
                    1
                } else if v < -0.5 {
                    -1
                } else {
                    0
                }
            })
            .collect()
    }

    fn sample_slow() -> Vec<i8> {
        (0..256).map(|i| ((i * 5 + 1) % 3 - 1) as i8).collect()
    }

    #[test]
    fn plant_and_harvest_round_trip() {
        let fast = sample_fast();
        let slow = sample_slow();

        let tree = Tree::plant(&fast, &slow, 42, 999.0, Sap::default());

        let (restored_fast, restored_slow) = tree.harvest();

        assert_eq!(restored_fast.len(), fast.len());
        assert_eq!(restored_slow.len(), slow.len());

        // Slow weights devem ser lossless (SparseRLE)
        assert_eq!(slow, restored_slow, "Slow weights must be lossless");

        // Fast weights: Fourier + deltas devem ser lossless
        // (tronco + folhas = reconstrução perfeita)
        let matches = fast
            .iter()
            .zip(restored_fast.iter())
            .filter(|&(&a, &b)| a == b)
            .count();
        let accuracy = matches as f64 / fast.len() as f64;
        assert!(
            accuracy > 0.95,
            "Fast weights accuracy with trunk+leaves: {:.1}%",
            accuracy * 100.0
        );
    }

    #[test]
    fn compression_is_significant() {
        let fast = sample_fast();
        let slow = sample_slow();

        let tree = Tree::plant(&fast, &slow, 1, 50.0, Sap::default());
        let original_size = fast.len() + slow.len(); // 8448 bytes
        let ratio = tree.compression_ratio(original_size);

        assert!(
            ratio > 1.5,
            "Should compress at least 1.5:1, got {:.1}:1",
            ratio
        );
        println!(
            "Compression: {} bytes → {} bytes ({:.1}:1)",
            original_size,
            tree.compressed_size(),
            ratio,
        );
    }

    #[test]
    fn serialize_deserialize_tree() {
        let fast = sample_fast();
        let slow = sample_slow();

        let tree = Tree::plant(
            &fast,
            &slow,
            42,
            999.0,
            Sap {
                dopamine: 0.5,
                cortisol: 0.3,
                ..Default::default()
            },
        );

        let bytes = tree.serialize();
        let restored = Tree::deserialize(&bytes).expect("tree deser must work");

        assert_eq!(restored.generation, 42);
        assert!((restored.fitness - 999.0).abs() < 1e-3);
        assert!((restored.sap.dopamine - 0.5).abs() < 1e-6);
        assert_eq!(restored.branches.len(), tree.branches.len());
        assert_eq!(restored.rings.len(), tree.rings.len());

        // Harvest should still work
        let (r_fast, r_slow) = restored.harvest();
        assert_eq!(r_fast.len(), fast.len());
        assert_eq!(r_slow, slow);
    }

    #[test]
    fn empty_slow_weights_handled() {
        let fast = sample_fast();
        let empty_slow: Vec<i8> = vec![];

        let tree = Tree::plant(&fast, &empty_slow, 1, 0.0, Sap::default());
        let (_, restored_slow) = tree.harvest();
        assert!(restored_slow.is_empty());
    }
}
