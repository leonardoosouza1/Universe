//! # Semantic Compression — Fingerprint comportamental
//!
//! A compressão mais radical: captura o **comportamento** do organismo,
//! não a sua representação interna. Dois organismos com pesos diferentes
//! mas mesmo comportamento recebem o mesmo hash.
//!
//! ## Como funciona
//!
//! 1. **Behavioral Fingerprint**: 64 bits que codificam os comportamentos
//!    primitivos observáveis (vira, freia, acelera, segue centro, etc.)
//!
//! 2. **Semantic Field**: um dicionário de centroides (fingerprint → pesos
//!    comprimidos). Organismos próximos no espaço semântico podem ser
//!    reconstruídos a partir de um template + delta esparso.
//!
//! 3. **Compression**: armazena apenas (fingerprint + delta_from_centroid).
//!    Se o centroide captura 90% do comportamento, o delta é minúsculo.

use crate::math::sparse;

/// Fingerprint comportamental: 64 bits que capturam a essência do organismo.
///
/// Cada bit representa um comportamento primitivo observável:
///   - Bit 0: "vira à esquerda quando parede à direita"
///   - Bit 1: "vira à direita quando parede à esquerda"
///   - Bit 2: "freia quando obstáculo frontal"
///   - Bit 3: "acelera em reta"
///   - ... até 64 comportamentos
///
/// Computado a partir dos pesos via análise de padrão de ativação:
///   peso[i] != 0 em posição significativa → bit correspondente ativado.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BehaviorFingerprint {
    pub hash: u64,
}

impl BehaviorFingerprint {
    /// Computa fingerprint a partir dos pesos ternários.
    ///
    /// Divide os pesos em 64 regiões e calcula a "energia" (soma dos absolutos)
    /// de cada região. Se energia > threshold, bit = 1.
    pub fn from_weights(weights: &[i8]) -> Self {
        if weights.is_empty() {
            return Self { hash: 0 };
        }

        let n = weights.len();
        let region_size = n / 64;
        let mut hash: u64 = 0;

        if region_size == 0 {
            // Para vetores < 64 elementos, usar bits individuais
            for (i, &w) in weights.iter().enumerate().take(64) {
                if w != 0 {
                    hash |= 1u64 << i;
                }
            }
            return Self { hash };
        }

        for bit in 0..64 {
            let start = bit * region_size;
            let end = if bit == 63 { n } else { (bit + 1) * region_size };

            let energy: i32 = weights[start..end].iter()
                .map(|&w| (w as i32).abs())
                .sum();

            // Threshold: pelo menos 10% dos pesos na região são não-zero
            let threshold = (end - start) as i32 / 10;
            if energy > threshold.max(1) {
                hash |= 1u64 << bit;
            }
        }

        Self { hash }
    }

    /// Distância de Hamming entre dois fingerprints (número de bits diferentes).
    pub fn distance(&self, other: &Self) -> u32 {
        (self.hash ^ other.hash).count_ones()
    }

    /// Similaridade normalizada [0, 1]. 1.0 = idênticos.
    pub fn similarity(&self, other: &Self) -> f32 {
        1.0 - (self.distance(other) as f32 / 64.0)
    }

    /// Serializa para 8 bytes.
    pub fn to_bytes(&self) -> [u8; 8] {
        self.hash.to_le_bytes()
    }

    /// Desserializa de 8 bytes.
    pub fn from_bytes(buf: &[u8; 8]) -> Self {
        Self { hash: u64::from_le_bytes(*buf) }
    }
}

/// Centroid: template de referência para um cluster comportamental.
#[derive(Clone, Debug)]
pub struct Centroid {
    /// Fingerprint que identifica este cluster.
    pub fingerprint: BehaviorFingerprint,
    /// Pesos de referência (comprimidos via SparseRLE).
    pub template_compressed: Vec<u8>,
    /// Tamanho original dos pesos.
    pub original_len: usize,
    /// Número de organismos atribuídos a este centroid.
    pub population_count: u32,
}

/// Campo semântico: dicionário de centroides para compressão por template.
#[derive(Clone, Debug)]
pub struct SemanticField {
    pub centroids: Vec<Centroid>,
    pub max_centroids: usize,
}

impl SemanticField {
    /// Cria campo semântico vazio.
    pub fn new(max_centroids: usize) -> Self {
        Self {
            centroids: Vec::new(),
            max_centroids,
        }
    }

    /// Encontra o centroid mais próximo para um fingerprint.
    /// Retorna (índice, distância).
    pub fn find_nearest(&self, fp: &BehaviorFingerprint) -> Option<(usize, u32)> {
        if self.centroids.is_empty() { return None; }

        let mut best_idx = 0;
        let mut best_dist = u32::MAX;

        for (i, centroid) in self.centroids.iter().enumerate() {
            let dist = fp.distance(&centroid.fingerprint);
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        Some((best_idx, best_dist))
    }

    /// Adiciona ou atualiza um centroid.
    ///
    /// Se existe um centroid com distância ≤ threshold, incrementa seu contador.
    /// Senão, adiciona um novo (removendo o menos popular se cheio).
    pub fn update(&mut self, weights: &[i8], merge_threshold: u32) {
        let fp = BehaviorFingerprint::from_weights(weights);

        if let Some((idx, dist)) = self.find_nearest(&fp) {
            if dist <= merge_threshold {
                self.centroids[idx].population_count += 1;
                return;
            }
        }

        // Comprimir pesos usando Sparse RLE
        let rle = sparse::compress(weights);
        let template_compressed = sparse::serialize(&rle);

        let new_centroid = Centroid {
            fingerprint: fp,
            template_compressed,
            original_len: weights.len(),
            population_count: 1,
        };

        if self.centroids.len() < self.max_centroids {
            self.centroids.push(new_centroid);
        } else {
            // Substituir o menos popular
            if let Some(min_idx) = self.centroids.iter()
                .enumerate()
                .min_by_key(|(_, c)| c.population_count)
                .map(|(i, _)| i)
            {
                self.centroids[min_idx] = new_centroid;
            }
        }
    }

    /// Comprime pesos usando o centroid mais próximo + delta esparso.
    ///
    /// Retorna (centroid_index, delta_compressed).
    /// Se nenhum centroid está perto o suficiente, retorna None.
    pub fn compress_with_template(&self, weights: &[i8], max_distance: u32) -> Option<(usize, Vec<u8>)> {
        let fp = BehaviorFingerprint::from_weights(weights);
        let (idx, dist) = self.find_nearest(&fp)?;

        if dist > max_distance {
            return None;
        }

        // Descomprimir template
        let template_rle = sparse::deserialize(&self.centroids[idx].template_compressed)?;
        let template = sparse::decompress(&template_rle);

        // Calcular delta: weights - template
        let delta: Vec<i8> = weights.iter().zip(template.iter())
            .map(|(&w, &t)| (w as i16 - t as i16).clamp(-1, 1) as i8)
            .collect();

        // Comprimir delta (deve ser muito esparso se template é bom)
        let delta_rle = sparse::compress(&delta);
        let delta_bytes = sparse::serialize(&delta_rle);

        Some((idx, delta_bytes))
    }

    /// Descomprime usando centroid + delta.
    pub fn decompress_with_template(&self, centroid_idx: usize, delta_bytes: &[u8]) -> Option<Vec<i8>> {
        if centroid_idx >= self.centroids.len() { return None; }

        let template_rle = sparse::deserialize(&self.centroids[centroid_idx].template_compressed)?;
        let template = sparse::decompress(&template_rle);

        let delta_rle = sparse::deserialize(delta_bytes)?;
        let delta = sparse::decompress(&delta_rle);

        // Somar template + delta e re-quantizar
        let weights: Vec<i8> = template.iter().zip(delta.iter())
            .map(|(&t, &d)| (t as i16 + d as i16).clamp(-1, 1) as i8)
            .collect();

        Some(weights)
    }

    /// Serializa o campo semântico inteiro.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        buf.extend_from_slice(&(self.centroids.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(self.max_centroids as u32).to_le_bytes());

        for centroid in &self.centroids {
            buf.extend_from_slice(&centroid.fingerprint.to_bytes());
            buf.extend_from_slice(&(centroid.original_len as u32).to_le_bytes());
            buf.extend_from_slice(&centroid.population_count.to_le_bytes());
            buf.extend_from_slice(&(centroid.template_compressed.len() as u32).to_le_bytes());
            buf.extend_from_slice(&centroid.template_compressed);
        }

        buf
    }

    /// Desserializa o campo semântico.
    pub fn deserialize(buf: &[u8]) -> Option<Self> {
        if buf.len() < 8 { return None; }

        let num_centroids = u32::from_le_bytes(buf[0..4].try_into().ok()?) as usize;
        let max_centroids = u32::from_le_bytes(buf[4..8].try_into().ok()?) as usize;

        let mut centroids = Vec::with_capacity(num_centroids);
        let mut off = 8;

        for _ in 0..num_centroids {
            if off + 20 > buf.len() { return None; }

            let fp_bytes: [u8; 8] = buf[off..off + 8].try_into().ok()?;
            let fingerprint = BehaviorFingerprint::from_bytes(&fp_bytes);
            off += 8;

            let original_len = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?) as usize;
            off += 4;

            let pop_count = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
            off += 4;

            let data_len = u32::from_le_bytes(buf[off..off + 4].try_into().ok()?) as usize;
            off += 4;

            if off + data_len > buf.len() { return None; }
            let template_compressed = buf[off..off + data_len].to_vec();
            off += data_len;

            centroids.push(Centroid {
                fingerprint,
                template_compressed,
                original_len,
                population_count: pop_count,
            });
        }

        Some(Self { centroids, max_centroids })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_from_all_zeros() {
        let weights = vec![0i8; 256];
        let fp = BehaviorFingerprint::from_weights(&weights);
        assert_eq!(fp.hash, 0, "All-zero weights should produce zero fingerprint");
    }

    #[test]
    fn fingerprint_from_dense_signal() {
        let weights: Vec<i8> = (0..256).map(|_| 1i8).collect();
        let fp = BehaviorFingerprint::from_weights(&weights);
        // Todas as regiões têm energia → todos os bits devem estar ativos
        assert_eq!(fp.hash, u64::MAX, "All-ones weights: {:#018x}", fp.hash);
    }

    #[test]
    fn fingerprint_distance_and_similarity() {
        let fp1 = BehaviorFingerprint { hash: 0xFF };
        let fp2 = BehaviorFingerprint { hash: 0xF0 };
        let dist = fp1.distance(&fp2);
        assert_eq!(dist, 4); // 4 bits differ (0x0F)
        let sim = fp1.similarity(&fp2);
        assert!((sim - (1.0 - 4.0/64.0)).abs() < 1e-6);
    }

    #[test]
    fn fingerprint_serialization() {
        let fp = BehaviorFingerprint { hash: 0xDEAD_BEEF_CAFE_BABE };
        let bytes = fp.to_bytes();
        let restored = BehaviorFingerprint::from_bytes(&bytes);
        assert_eq!(fp, restored);
    }

    #[test]
    fn semantic_field_add_and_find() {
        let mut field = SemanticField::new(4);

        let w1: Vec<i8> = (0..256).map(|i| if i < 128 { 1 } else { 0 }).collect();
        let w2: Vec<i8> = (0..256).map(|i| if i >= 128 { -1 } else { 0 }).collect();

        field.update(&w1, 8);
        field.update(&w2, 8);

        assert_eq!(field.centroids.len(), 2);

        let fp1 = BehaviorFingerprint::from_weights(&w1);
        let (idx, dist) = field.find_nearest(&fp1).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(dist, 0);
    }

    #[test]
    fn template_compression_identical_signal() {
        let mut field = SemanticField::new(4);
        let weights: Vec<i8> = (0..256).map(|i| ((i % 3) as i8) - 1).collect();

        field.update(&weights, 8);

        // Comprimir o mesmo sinal deve dar delta zero
        let (idx, delta_bytes) = field.compress_with_template(&weights, 64).unwrap();
        assert_eq!(idx, 0);

        // Descomprimir deve dar resultado idêntico
        let restored = field.decompress_with_template(0, &delta_bytes).unwrap();
        assert_eq!(weights, restored);
    }

    #[test]
    fn semantic_field_serialization() {
        let mut field = SemanticField::new(8);
        let w1: Vec<i8> = (0..128).map(|i| ((i % 3) as i8) - 1).collect();
        let w2: Vec<i8> = (0..128).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();

        field.update(&w1, 4);
        field.update(&w2, 4);

        let bytes = field.serialize();
        let restored = SemanticField::deserialize(&bytes).unwrap();

        assert_eq!(restored.centroids.len(), 2);
        assert_eq!(restored.max_centroids, 8);
    }
}
