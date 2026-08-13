//! # Universal Compressor — Qualquer dado como equações matemáticas
//!
//! Analisa blocos de dados arbitrários e seleciona automaticamente
//! o motor generativo de menor custo:
//!
//! | Motor      | Quando usar                        | Tipo     |
//! |------------|-------------------------------------|----------|
//! | `ZERO`     | Bloco inteiro é zero                | Lossless |
//! | `BRLE`     | Bytes com runs repetidos            | Lossless |
//! | `DFTS`     | Sinal com periodicidade (Fourier)   | Lossy    |
//! | `DELT`     | Dados suaves (delta coding)         | Lossless |
//! | `LITL`     | Incompressível (entropia máxima)    | Lossless |

use std::f64::consts::PI;

/// Tamanho padrão de bloco: 4 KB
pub const DEFAULT_BLOCK_SIZE: usize = 4096;

// ═══════════════════════════════════════════════════════════════
// Perfil de Bloco — análise espectral de um bloco de bytes
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct BlockProfile {
    /// Entropia de Shannon (0.0 = constante, 8.0 = ruído puro)
    pub entropy: f64,
    /// Fração de bytes zero (0.0 = nenhum, 1.0 = todos)
    pub zero_ratio: f64,
    /// Suavidade: média da diferença absoluta entre bytes consecutivos
    /// (baixo = suave, alto = caótico)
    pub roughness: f64,
    /// Força da periodicidade dominante (0.0 = nenhuma, 1.0 = perfeita)
    pub periodicity: f64,
    /// Melhor motor para este bloco
    pub best_motor: Motor,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Motor {
    /// Bloco inteiro é zero — custo: 0 bytes
    Zero,
    /// Byte RLE — runs de bytes repetidos
    ByteRle,
    /// Delta coding — diferenças entre bytes consecutivos
    Delta,
    /// DFT — Fourier com top-K coeficientes
    Fourier { num_harmonics: usize },
    /// Literal — cópia bruta (incompressível)
    Literal,
}

/// Analisa um bloco de bytes e retorna seu perfil.
pub fn analyze_block(data: &[u8]) -> BlockProfile {
    let n = data.len();
    if n == 0 {
        return BlockProfile {
            entropy: 0.0, zero_ratio: 1.0, roughness: 0.0,
            periodicity: 0.0, best_motor: Motor::Zero,
        };
    }

    // 1. Contagem de frequência por byte
    let mut freq = [0u32; 256];
    for &b in data { freq[b as usize] += 1; }

    // 2. Entropia de Shannon
    let nf = n as f64;
    let entropy = freq.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / nf;
            -p * p.log2()
        })
        .sum::<f64>();

    // 3. Zero ratio
    let zero_ratio = freq[0] as f64 / nf;

    // 4. Roughness (suavidade)
    let roughness = if n > 1 {
        let sum: f64 = data.windows(2)
            .map(|w| (w[1] as f64 - w[0] as f64).abs())
            .sum();
        sum / (n - 1) as f64
    } else { 0.0 };

    // 5. Periodicidade (autocorrelação normalizada no lag dominante)
    let periodicity = estimate_periodicity(data);

    // 6. Seleção de motor
    let best_motor = select_motor(n, entropy, zero_ratio, roughness, periodicity);

    BlockProfile { entropy, zero_ratio, roughness, periodicity, best_motor }
}

fn estimate_periodicity(data: &[u8]) -> f64 {
    let n = data.len();
    if n < 8 { return 0.0; }

    // Média
    let mean = data.iter().map(|&b| b as f64).sum::<f64>() / n as f64;

    // Autocorrelação para lags 1..N/2, normalizada
    let variance: f64 = data.iter().map(|&b| {
        let d = b as f64 - mean;
        d * d
    }).sum::<f64>();

    if variance < 1e-10 { return 0.0; }

    let max_lag = (n / 2).min(256);
    let mut best_corr = 0.0f64;

    for lag in 2..max_lag {
        let mut corr = 0.0f64;
        for i in 0..n - lag {
            corr += (data[i] as f64 - mean) * (data[i + lag] as f64 - mean);
        }
        corr /= variance;
        if corr > best_corr { best_corr = corr; }
    }

    best_corr.clamp(0.0, 1.0)
}

fn select_motor(
    block_len: usize, entropy: f64, zero_ratio: f64,
    roughness: f64, periodicity: f64,
) -> Motor {
    // Tudo zero?
    if zero_ratio > 0.9999 { return Motor::Zero; }

    // Muito repetitivo? (entropia < 1 bit/byte)
    if entropy < 1.5 { return Motor::ByteRle; }

    // Suave? (gradientes, tabelas numéricas)
    if roughness < 3.0 && entropy < 5.0 { return Motor::Delta; }

    // Periódico? (formas de onda, padrões cíclicos)
    if periodicity > 0.3 && entropy < 7.5 {
        // Número de harmônicos proporcional à complexidade
        let k = if periodicity > 0.7 { 16 }
                else if periodicity > 0.5 { 32 }
                else { 64 };
        let k = k.min(block_len / 4);
        return Motor::Fourier { num_harmonics: k };
    }

    // Entropia moderada? Tentar delta
    if entropy < 6.5 { return Motor::Delta; }

    // Incompressível
    Motor::Literal
}

// ═══════════════════════════════════════════════════════════════
// Bloco Comprimido — resultado de um motor
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct CompressedBlock {
    /// Motor usado
    pub motor: Motor,
    /// Dados comprimidos
    pub data: Vec<u8>,
    /// Tamanho original do bloco
    pub original_len: usize,
}

// ═══════════════════════════════════════════════════════════════
// Encoding — cada motor
// ═══════════════════════════════════════════════════════════════

pub fn compress_block(raw: &[u8], profile: &BlockProfile) -> CompressedBlock {
    match profile.best_motor {
        Motor::Zero => compress_zero(raw),
        Motor::ByteRle => compress_byte_rle(raw),
        Motor::Delta => compress_delta(raw),
        Motor::Fourier { num_harmonics } => compress_fourier(raw, num_harmonics),
        Motor::Literal => compress_literal(raw),
    }
}

/// Motor ZERO: bloco inteiro é zero, custo = 0 bytes
fn compress_zero(raw: &[u8]) -> CompressedBlock {
    CompressedBlock { motor: Motor::Zero, data: vec![], original_len: raw.len() }
}

/// Motor LITL: cópia bruta
fn compress_literal(raw: &[u8]) -> CompressedBlock {
    CompressedBlock { motor: Motor::Literal, data: raw.to_vec(), original_len: raw.len() }
}

/// Motor BRLE: Run-Length Encoding de bytes
fn compress_byte_rle(raw: &[u8]) -> CompressedBlock {
    let mut out = Vec::new();
    let n = raw.len();
    let mut i = 0;

    while i < n {
        let val = raw[i];
        let mut run = 1u16;
        while i + (run as usize) < n && raw[i + (run as usize)] == val && run < 255 {
            run += 1;
        }
        out.push(val);
        out.push(run as u8);
        i += run as usize;
    }

    // Se RLE é maior que original, fallback para literal
    if out.len() >= raw.len() {
        return compress_literal(raw);
    }

    CompressedBlock { motor: Motor::ByteRle, data: out, original_len: raw.len() }
}

/// Motor DELT: Delta coding (diferenças entre bytes consecutivos)
fn compress_delta(raw: &[u8]) -> CompressedBlock {
    if raw.is_empty() {
        return CompressedBlock { motor: Motor::Delta, data: vec![], original_len: 0 };
    }

    // Computar deltas
    let mut deltas = Vec::with_capacity(raw.len());
    deltas.push(raw[0]); // primeiro byte literal
    for i in 1..raw.len() {
        deltas.push(raw[i].wrapping_sub(raw[i - 1]));
    }

    // Aplicar RLE nos deltas (deltas tendem a ter mais repetição)
    let mut out = Vec::new();
    let n = deltas.len();
    let mut i = 0;

    while i < n {
        let val = deltas[i];
        let mut run = 1u16;
        while i + (run as usize) < n && deltas[i + (run as usize)] == val && run < 255 {
            run += 1;
        }
        out.push(val);
        out.push(run as u8);
        i += run as usize;
    }

    // Se Delta+RLE é maior que original, fallback
    if out.len() >= raw.len() {
        return compress_literal(raw);
    }

    CompressedBlock { motor: Motor::Delta, data: out, original_len: raw.len() }
}

/// Motor DFTS: Fourier DFT — dados como soma de senóides
fn compress_fourier(raw: &[u8], num_harmonics: usize) -> CompressedBlock {
    let n = raw.len();
    let nf = n as f64;

    // Converter bytes para signal f64 centrado em zero [-128, 127]
    let signal: Vec<f64> = raw.iter().map(|&b| b as f64 - 128.0).collect();

    // DFT: extrair top-K harmônicos por amplitude
    let mut all_harmonics: Vec<(usize, f64, f64)> = Vec::new(); // (freq, amp, phase)

    // DC component
    let dc = signal.iter().sum::<f64>() / nf;

    for k in 1..=n / 2 {
        let kf = k as f64;
        let mut re = 0.0f64;
        let mut im = 0.0f64;

        for (i, &s) in signal.iter().enumerate() {
            let angle = 2.0 * PI * kf * (i as f64) / nf;
            re += s * angle.cos();
            im += s * angle.sin();
        }

        re *= 2.0 / nf;
        im *= 2.0 / nf;

        let amp = (re * re + im * im).sqrt();
        let phase = im.atan2(re);

        all_harmonics.push((k, amp, phase));
    }

    // Ordenar por amplitude (maiores primeiro)
    all_harmonics.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Manter top K
    let k = num_harmonics.min(all_harmonics.len());
    let top_k = &all_harmonics[..k];

    // Serializar: [dc:f32][k:u16][freq:u16, amp:f32, phase:f32] × k
    let mut data = Vec::with_capacity(4 + 2 + k * 10);
    data.extend_from_slice(&(dc as f32).to_le_bytes());
    data.extend_from_slice(&(k as u16).to_le_bytes());

    for &(freq, amp, phase) in top_k {
        data.extend_from_slice(&(freq as u16).to_le_bytes());
        data.extend_from_slice(&(amp as f32).to_le_bytes());
        data.extend_from_slice(&(phase as f32).to_le_bytes());
    }

    // Se Fourier é maior que original, fallback
    if data.len() >= raw.len() {
        return compress_literal(raw);
    }

    CompressedBlock { motor: Motor::Fourier { num_harmonics: k }, data, original_len: n }
}

// ═══════════════════════════════════════════════════════════════
// Decoding — reconstrução de cada motor
// ═══════════════════════════════════════════════════════════════

pub fn decompress_block(block: &CompressedBlock) -> Vec<u8> {
    match block.motor {
        Motor::Zero => vec![0u8; block.original_len],
        Motor::Literal => block.data.clone(),
        Motor::ByteRle => decompress_byte_rle(&block.data, block.original_len),
        Motor::Delta => decompress_delta(&block.data, block.original_len),
        Motor::Fourier { .. } => decompress_fourier(&block.data, block.original_len),
    }
}

fn decompress_byte_rle(data: &[u8], original_len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(original_len);
    let mut i = 0;
    while i + 1 < data.len() && out.len() < original_len {
        let val = data[i];
        let run = data[i + 1] as usize;
        for _ in 0..run.min(original_len - out.len()) {
            out.push(val);
        }
        i += 2;
    }
    out.resize(original_len, 0);
    out
}

fn decompress_delta(data: &[u8], original_len: usize) -> Vec<u8> {
    // Primeiro decodificar RLE dos deltas
    let mut deltas = Vec::with_capacity(original_len);
    let mut i = 0;
    while i + 1 < data.len() && deltas.len() < original_len {
        let val = data[i];
        let run = data[i + 1] as usize;
        for _ in 0..run.min(original_len - deltas.len()) {
            deltas.push(val);
        }
        i += 2;
    }
    deltas.resize(original_len, 0);

    // Reconstruir por acumulação
    let mut out = Vec::with_capacity(original_len);
    if !deltas.is_empty() {
        out.push(deltas[0]); // primeiro byte é literal
        for i in 1..deltas.len() {
            let prev = out[i - 1];
            out.push(prev.wrapping_add(deltas[i]));
        }
    }
    out
}

fn decompress_fourier(data: &[u8], original_len: usize) -> Vec<u8> {
    if data.len() < 6 { return vec![128u8; original_len]; }

    let dc = f32::from_le_bytes(data[0..4].try_into().unwrap()) as f64;
    let k = u16::from_le_bytes(data[4..6].try_into().unwrap()) as usize;

    let mut harmonics = Vec::with_capacity(k);
    for i in 0..k {
        let off = 6 + i * 10;
        if off + 10 > data.len() { break; }
        let freq = u16::from_le_bytes(data[off..off+2].try_into().unwrap()) as usize;
        let amp = f32::from_le_bytes(data[off+2..off+6].try_into().unwrap()) as f64;
        let phase = f32::from_le_bytes(data[off+6..off+10].try_into().unwrap()) as f64;
        harmonics.push((freq, amp, phase));
    }

    let nf = original_len as f64;
    let mut out = Vec::with_capacity(original_len);

    for i in 0..original_len {
        let mut val = dc;
        for &(freq, amp, phase) in &harmonics {
            let angle = 2.0 * PI * (freq as f64) * (i as f64) / nf;
            val += amp * (angle + phase).cos();
        }
        // Converter de volta para [0, 255]
        let byte = (val + 128.0).round().clamp(0.0, 255.0) as u8;
        out.push(byte);
    }

    out
}

// ═══════════════════════════════════════════════════════════════
// Serialização TLV — para gravar no .dna
// ═══════════════════════════════════════════════════════════════

/// Motor tag para o TLV
fn motor_tag(motor: &Motor) -> [u8; 4] {
    match motor {
        Motor::Zero => *b"ZERO",
        Motor::ByteRle => *b"BRLE",
        Motor::Delta => *b"DELT",
        Motor::Fourier { .. } => *b"DFTS",
        Motor::Literal => *b"LITL",
    }
}

fn motor_from_tag(tag: &[u8; 4]) -> Motor {
    match tag {
        b"ZERO" => Motor::Zero,
        b"BRLE" => Motor::ByteRle,
        b"DELT" => Motor::Delta,
        b"DFTS" => Motor::Fourier { num_harmonics: 0 },
        b"LITL" => Motor::Literal,
        _ => Motor::Literal,
    }
}

/// Serializa um CompressedBlock para bytes TLV:
/// [motor_tag:4][original_len:4][data_len:4][data...]
pub fn serialize_block(block: &CompressedBlock) -> Vec<u8> {
    let tag = motor_tag(&block.motor);
    let mut buf = Vec::with_capacity(12 + block.data.len());
    buf.extend_from_slice(&tag);
    buf.extend_from_slice(&(block.original_len as u32).to_le_bytes());
    buf.extend_from_slice(&(block.data.len() as u32).to_le_bytes());
    buf.extend_from_slice(&block.data);
    buf
}

/// Deserializa um CompressedBlock de bytes TLV.
/// Retorna (block, bytes_consumed).
pub fn deserialize_block(buf: &[u8]) -> Option<(CompressedBlock, usize)> {
    if buf.len() < 12 { return None; }
    let mut tag = [0u8; 4];
    tag.copy_from_slice(&buf[0..4]);
    let original_len = u32::from_le_bytes(buf[4..8].try_into().ok()?) as usize;
    let data_len = u32::from_le_bytes(buf[8..12].try_into().ok()?) as usize;

    if buf.len() < 12 + data_len { return None; }
    let data = buf[12..12 + data_len].to_vec();
    let motor = motor_from_tag(&tag);

    Some((CompressedBlock { motor, data, original_len }, 12 + data_len))
}

// ═══════════════════════════════════════════════════════════════
// API de Alto Nível — comprimir/descomprimir arquivo inteiro
// ═══════════════════════════════════════════════════════════════

/// Resultado da compressão de um arquivo.
#[derive(Debug)]
pub struct CompressionResult {
    pub blocks: Vec<CompressedBlock>,
    pub original_size: usize,
    pub compressed_size: usize,
    pub block_count: usize,
    pub motor_stats: MotorStats,
}

#[derive(Debug, Default)]
pub struct MotorStats {
    pub zero_blocks: usize,
    pub rle_blocks: usize,
    pub delta_blocks: usize,
    pub fourier_blocks: usize,
    pub literal_blocks: usize,
}

/// Comprime dados arbitrários usando seleção automática de motores por bloco.
pub fn compress_universal(data: &[u8], block_size: usize) -> CompressionResult {
    let mut blocks = Vec::new();
    let mut stats = MotorStats::default();
    let mut compressed_size = 0usize;

    for chunk in data.chunks(block_size) {
        let profile = analyze_block(chunk);
        let block = compress_block(chunk, &profile);

        compressed_size += 12 + block.data.len(); // TLV overhead

        match block.motor {
            Motor::Zero => stats.zero_blocks += 1,
            Motor::ByteRle => stats.rle_blocks += 1,
            Motor::Delta => stats.delta_blocks += 1,
            Motor::Fourier { .. } => stats.fourier_blocks += 1,
            Motor::Literal => stats.literal_blocks += 1,
        }

        blocks.push(block);
    }

    CompressionResult {
        block_count: blocks.len(),
        original_size: data.len(),
        compressed_size,
        blocks,
        motor_stats: stats,
    }
}

/// Descomprime de volta para dados originais.  
pub fn decompress_universal(blocks: &[CompressedBlock]) -> Vec<u8> {
    let mut out = Vec::new();
    for block in blocks {
        out.extend_from_slice(&decompress_block(block));
    }
    out
}

/// Serializa todos os blocos para um payload binário.
/// Header: [num_blocks:u32][block_size:u32][original_size:u64]
pub fn serialize_payload(result: &CompressionResult, block_size: u32) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(result.block_count as u32).to_le_bytes());
    buf.extend_from_slice(&block_size.to_le_bytes());
    buf.extend_from_slice(&(result.original_size as u64).to_le_bytes());

    for block in &result.blocks {
        buf.extend_from_slice(&serialize_block(block));
    }
    buf
}

/// Deserializa payload de volta para blocos.
pub fn deserialize_payload(buf: &[u8]) -> Option<Vec<CompressedBlock>> {
    if buf.len() < 16 { return None; }

    let num_blocks = u32::from_le_bytes(buf[0..4].try_into().ok()?) as usize;
    let _block_size = u32::from_le_bytes(buf[4..8].try_into().ok()?);
    let _original_size = u64::from_le_bytes(buf[8..16].try_into().ok()?);

    let mut blocks = Vec::with_capacity(num_blocks);
    let mut off = 16;

    for _ in 0..num_blocks {
        let (block, consumed) = deserialize_block(&buf[off..])?;
        blocks.push(block);
        off += consumed;
    }

    Some(blocks)
}

// ═══════════════════════════════════════════════════════════════
// Testes
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_block_round_trip() {
        let data = vec![0u8; 4096];
        let profile = analyze_block(&data);
        assert!(matches!(profile.best_motor, Motor::Zero));
        let block = compress_block(&data, &profile);
        assert_eq!(block.data.len(), 0);
        let restored = decompress_block(&block);
        assert_eq!(data, restored);
    }

    #[test]
    fn rle_block_round_trip() {
        let mut data = vec![0xAA_u8; 2048];
        data.extend_from_slice(&vec![0xBB_u8; 2048]);
        let profile = analyze_block(&data);
        let block = compress_block(&data, &profile);
        let restored = decompress_block(&block);
        assert_eq!(data, restored);
        assert!(block.data.len() < data.len(), "RLE should compress");
    }

    #[test]
    fn delta_block_round_trip() {
        // Dados de gradiente suave: 0, 1, 2, 3, ...
        let data: Vec<u8> = (0..4096).map(|i| (i % 256) as u8).collect();
        let profile = analyze_block(&data);
        let block = compress_block(&data, &profile);
        let restored = decompress_block(&block);
        assert_eq!(data, restored);
    }

    #[test]
    fn literal_block_incompressible() {
        // Dados pseudo-aleatórios (alta entropia)
        let data: Vec<u8> = (0..4096)
            .map(|i| {
                let x = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(1);
                (x >> 33) as u8
            })
            .collect();
        let profile = analyze_block(&data);
        let block = compress_block(&data, &profile);
        let restored = decompress_block(&block);
        assert_eq!(data, restored);
    }

    #[test]
    fn universal_round_trip_mixed_data() {
        let mut data = Vec::new();
        // Bloco de zeros
        data.extend_from_slice(&vec![0u8; 4096]);
        // Bloco de gradiente
        data.extend_from_slice(&(0..4096).map(|i| (i % 256) as u8).collect::<Vec<_>>());
        // Bloco repetitivo
        data.extend_from_slice(&vec![0xFF_u8; 4096]);

        let result = compress_universal(&data, DEFAULT_BLOCK_SIZE);
        let restored = decompress_universal(&result.blocks);

        assert_eq!(data, restored);
        assert!(result.compressed_size < data.len());
    }

    #[test]
    fn serialization_round_trip() {
        let data: Vec<u8> = (0..8192).map(|i| (i % 256) as u8).collect();
        let result = compress_universal(&data, DEFAULT_BLOCK_SIZE);
        let payload = serialize_payload(&result, DEFAULT_BLOCK_SIZE as u32);
        let blocks = deserialize_payload(&payload).unwrap();
        let restored = decompress_universal(&blocks);
        assert_eq!(data, restored);
    }
}
