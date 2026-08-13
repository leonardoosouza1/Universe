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
    /// Computa fingerprint a partir dos pesos: divide em 64 regiões e marca as 32 de MAIOR
    /// energia (soma dos absolutos). Critério RELATIVO, por posto.
    ///
    /// # Por que por posto, e não por limiar
    ///
    /// A versão anterior usava limiar ABSOLUTO: `energia > (nº de pesos na região)/10`, com o
    /// comentário "pelo menos 10% dos pesos na região são não-zero". Duas coisas quebravam:
    ///
    /// 1. **Unidades trocadas.** Comparava SOMA DE MAGNITUDES contra CONTAGEM/10. Em
    ///    `cluster_elites` os pesos entram como `w*127.0`, então um único peso não-zero
    ///    contribui 127 e já estoura um limiar de 33. Fator de ~1270× fora.
    /// 2. **Limiar absoluto não serve para pesos densos.** `CHAMPION.dna` tem 52,6% de
    ///    não-zeros; mesmo com as unidades certas, 52,6% passa de 10% em TODA região.
    ///
    /// Resultado medido: `[NICHE] pool=10 clusters=1 largest=10` em **todas** as gerações de
    /// todos os treinos. O fitness-sharing dividia a fitness de todo mundo pelo mesmo 10 —
    /// um fator uniforme, que não separa nada. **A diversidade nunca esteve ligada.**
    ///
    /// Por posto não há limiar para calibrar: metade das regiões acende por construção,
    /// qualquer que seja a densidade ou a escala. É a diferença entre perguntar "esta região
    /// tem muita energia?" (depende de quanto é muito) e "esta região tem mais energia que a
    /// mediana das outras?" (sempre respondível).
    ///
    /// Empate desempata por índice, para o resultado ser determinístico — evolução com
    /// fingerprint não-determinístico já custou caro em outra frente.
    pub fn from_weights(weights: &[i8]) -> Self {
        if weights.is_empty() {
            return Self { hash: 0 };
        }

        let n = weights.len();
        let region_size = n / 64;

        if region_size == 0 {
            // Vetores < 64 elementos: um bit por peso, sem regiões para ranquear.
            let mut hash: u64 = 0;
            for (i, &w) in weights.iter().enumerate().take(64) {
                if w != 0 {
                    hash |= 1u64 << i;
                }
            }
            return Self { hash };
        }

        let energias: Vec<i64> = (0..64).map(|bit| {
            let start = bit * region_size;
            let end = if bit == 63 { n } else { (bit + 1) * region_size };
            weights[start..end].iter().map(|&w| (w as i64).abs()).sum()
        }).collect();

        // Cérebro sem nenhum peso não é "metade aceso", é ausência — mantém hash 0.
        if energias.iter().all(|&e| e == 0) {
            return Self { hash: 0 };
        }

        let mut ordem: Vec<usize> = (0..64).collect();
        ordem.sort_by(|&a, &b| energias[b].cmp(&energias[a]).then(a.cmp(&b)));

        let mut hash: u64 = 0;
        for &bit in ordem.iter().take(32) {
            hash |= 1u64 << bit;
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

    /// Pesos UNIFORMES não carregam informação de região: toda região tem a mesma energia.
    /// Por posto isso vira exatamente metade dos bits (desempate por índice), não `u64::MAX`.
    ///
    /// Este teste afirmava `u64::MAX` e passava por causa da saturação — ele codificava o
    /// bug. O que importa não é quantos bits acendem num sinal chapado, e sim que sinais
    /// DIFERENTES fiquem distinguíveis; é o que os testes abaixo cobram.
    #[test]
    fn fingerprint_from_dense_signal() {
        let weights: Vec<i8> = (0..256).map(|_| 1i8).collect();
        let fp = BehaviorFingerprint::from_weights(&weights);
        assert_eq!(fp.hash.count_ones(), 32, "sinal uniforme: {:#018x}", fp.hash);
    }

    /// O que o fingerprint existe para fazer: separar cérebros diferentes.
    #[test]
    fn fingerprint_distingue_cerebros_na_densidade_real() {
        let fa = BehaviorFingerprint::from_weights(&cerebro_realista(1));
        let fb = BehaviorFingerprint::from_weights(&cerebro_realista(999));
        let fc = BehaviorFingerprint::from_weights(&cerebro_realista(4242));
        for (x, y, n) in [(fa, fb, "1x999"), (fa, fc, "1x4242"), (fb, fc, "999x4242")] {
            assert!(x.distance(&y) > NICHE_DISTANCE_ESPERADA,
                    "{n}: distância {} não separa (limiar de nicho {NICHE_DISTANCE_ESPERADA})",
                    x.distance(&y));
        }
    }

    /// Determinismo: o mesmo cérebro tem de dar sempre o mesmo fingerprint, senão o
    /// clustering vira sorteio a cada geração.
    #[test]
    fn fingerprint_e_deterministico() {
        let c = cerebro_realista(31337);
        assert_eq!(BehaviorFingerprint::from_weights(&c).hash,
                   BehaviorFingerprint::from_weights(&c).hash);
    }

    /// Sensibilidade: mexer numa região tem de mover o fingerprint. Zerar um quarto dos
    /// pesos é uma mudança grande de comportamento e não pode passar despercebida.
    #[test]
    fn fingerprint_reage_a_mudanca_de_regiao() {
        let base = cerebro_realista(5);
        let mut mexido = base.clone();
        for w in mexido.iter_mut().take(base.len() / 4) { *w = 0; }
        let d = BehaviorFingerprint::from_weights(&base)
            .distance(&BehaviorFingerprint::from_weights(&mexido));
        assert!(d > 0, "zerar 25% dos pesos não mudou o fingerprint");
    }

    /// Espelha `NICHE_DISTANCE` de `evolution::pipeline` (8/64). Duplicado porque este crate
    /// não depende daquele; se lá mudar, este teste é o lembrete de revisitar aqui.
    const NICHE_DISTANCE_ESPERADA: u32 = 8;

    /// Gera pesos com a MESMA forma do cérebro real: 21408 pesos ternários (±1 → ±127
    /// pelo `w*127.0` de `cluster_elites`), ~52,6% não-zero, que é a densidade medida em
    /// `CHAMPION.dna`. `semente` muda o padrão sem mudar a densidade.
    fn cerebro_realista(semente: u64) -> Vec<i8> {
        let mut s = semente.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..21408).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            match (s >> 33) % 1000 {
                0..=262 => -127,
                263..=525 => 127,
                _ => 0,
            }
        }).collect()
    }

    /// REGRESSÃO do bug que desligou a diversidade da evolução inteira.
    ///
    /// O limiar ABSOLUTO antigo (`energia > nº_de_pesos/10`) saturava na densidade real e
    /// dois cérebros sem nenhum peso em comum saíam com `hash = u64::MAX` e distância 0.
    /// Medido no campo: `[NICHE] pool=10 clusters=1 largest=10` em **todas** as gerações de
    /// todos os treinos. O fitness-sharing dividia a fitness de todo mundo pelo mesmo 10 —
    /// fator uniforme, que não separa ninguém.
    ///
    /// A conta do erro: 21408 pesos / 64 = 334 por região; limiar `334/10 = 33`; energia com
    /// 52,6% de não-zeros a ±127 ≈ 22.317. **Um único peso não-zero já estourava o limiar.**
    /// A intenção documentada era comparar uma CONTAGEM ("10% dos pesos não-zero"), mas o
    /// código comparava SOMA DE MAGNITUDES — fator ~1270× de unidade errada.
    ///
    /// Este teste reproduz exatamente aquele cálculo e exige que ele NÃO seja mais o que o
    /// código faz.
    #[test]
    fn regressao_fingerprint_nao_satura_mais_na_densidade_real() {
        let a = cerebro_realista(1);
        let b = cerebro_realista(999);
        let nz = a.iter().filter(|&&w| w != 0).count();
        assert!((0.45..0.60).contains(&(nz as f32 / a.len() as f32)),
                "densidade fora do medido em CHAMPION.dna: {nz}");

        // O critério ANTIGO, reproduzido aqui: acendia todos os 64 bits.
        let regiao = a.len() / 64;
        let acesos_criterio_antigo = (0..64).filter(|bit| {
            let (i, f) = (bit * regiao, if *bit == 63 { a.len() } else { (bit + 1) * regiao });
            let energia: i64 = a[i..f].iter().map(|&w| (w as i64).abs()).sum();
            energia > ((f - i) as i64 / 10).max(1)
        }).count();
        assert_eq!(acesos_criterio_antigo, 64, "o critério antigo saturava — este é o bug");

        // O critério ATUAL tem de separar os dois cérebros.
        let (fa, fb) = (BehaviorFingerprint::from_weights(&a),
                        BehaviorFingerprint::from_weights(&b));
        assert_ne!(fa.hash, u64::MAX, "voltou a saturar");
        assert!(fa.distance(&fb) > NICHE_DISTANCE_ESPERADA,
                "cérebros diferentes voltaram a cair no mesmo nicho: distância {}",
                fa.distance(&fb));
    }

    /// O critério corrigido (contagem de não-zeros, como a documentação sempre disse)
    /// também satura nesta densidade — 52,6% passa de 10% em toda região.
    ///
    /// Consertar as UNIDADES não basta: um limiar ABSOLUTO não serve para pesos densos.
    /// É por isso que o conserto tem de ser um critério RELATIVO à própria distribuição.
    #[test]
    fn consertar_so_a_unidade_nao_resolve() {
        let a = cerebro_realista(7);
        let regiao = a.len() / 64;
        let regioes_acima: usize = (0..64).filter(|bit| {
            let (i, f) = (bit * regiao, if *bit == 63 { a.len() } else { (bit + 1) * regiao });
            a[i..f].iter().filter(|&&w| w != 0).count() > (f - i) / 10
        }).count();
        assert_eq!(regioes_acima, 64,
                   "com 52,6% de densidade, contagem>10% também dá todos os bits");
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
