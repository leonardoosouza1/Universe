//! # Sparse RLE Encoding para pesos ternários
//!
//! Codificação lossless de vetores `{-1, 0, +1}` usando Run-Length Encoding.
//! Ternários têm apenas 3 símbolos → codificação extremamente eficiente.
//!
//! **Fidelidade**: 100% lossless
//! **Ratio**: 2:1 a 50:1 dependendo da esparsidade (tipicamente 70% zeros → ~5:1)
//!
//! ## Formato
//!
//! Cada run é codificado como 1-2 bytes:
//!   - Bits 7-6: símbolo (00 = zero, 01 = +1, 10 = -1, 11 = extensão)
//!   - Bits 5-0: run length - 1 (0..63 → runs de 1 a 64)
//!   - Se run > 64: extensão bytes adicionais
//!
//! Para um vetor de 8192 pesos com 70% zeros:
//!   ~5740 zeros em ~90 runs de zero (média 64) → ~90 bytes
//!   ~2452 não-zeros em ~1200 runs curtas → ~1200 bytes
//!   Total ≈ 1300 bytes (ratio 6.3:1)

/// Resultado da compressão RLE.
#[derive(Clone, Debug)]
pub struct SparseRle {
    /// Dados comprimidos.
    pub data: Vec<u8>,
    /// Comprimento original.
    pub original_len: usize,
}

const SYM_ZERO: u8 = 0b00;    // 0
const SYM_POS: u8 = 0b01;     // +1
const SYM_NEG: u8 = 0b10;     // -1
const SYM_EXT: u8 = 0b11;     // extensão para runs > 64

/// Comprime um vetor de pesos ternários usando RLE.
///
/// **Garantia**: lossless. `decompress(compress(x)) == x` sempre.
pub fn compress(signal: &[i8]) -> SparseRle {
    let mut data = Vec::new();
    let n = signal.len();

    if n == 0 {
        return SparseRle { data, original_len: 0 };
    }

    let mut i = 0;
    while i < n {
        let sym = signal[i];
        let sym_code = match sym {
            0 => SYM_ZERO,
            1 => SYM_POS,
            -1 => SYM_NEG,
            _ => SYM_ZERO, // clamp inválidos para zero
        };

        // Count run length
        let mut run_len = 1usize;
        while i + run_len < n && signal[i + run_len] == sym && run_len < 16384 {
            run_len += 1;
        }

        // Encode run
        if run_len <= 64 {
            // Single byte: [sym:2][len-1:6]
            data.push((sym_code << 6) | ((run_len - 1) as u8));
        } else if run_len <= 16384 {
            // Two bytes: extension header + 14-bit length
            // Byte 1: [EXT:2][high 6 bits of len-1]
            // Byte 2: [low 8 bits of len-1]
            let len_enc = (run_len - 1) as u16;
            data.push((SYM_EXT << 6) | ((len_enc >> 8) as u8 & 0x3F));
            data.push((len_enc & 0xFF) as u8);
            // Extra byte for symbol identity
            data.push(sym_code);
        } else {
            // Shouldn't happen with the 16384 cap above, but safety
            data.push((sym_code << 6) | 63);
            run_len = 64;
        }

        i += run_len;
    }

    SparseRle { data, original_len: n }
}

/// Descomprime RLE de volta para vetor de pesos ternários.
pub fn decompress(rle: &SparseRle) -> Vec<i8> {
    let mut output = Vec::with_capacity(rle.original_len);
    let data = &rle.data;
    let mut i = 0;

    while i < data.len() && output.len() < rle.original_len {
        let byte = data[i];
        let sym_code = byte >> 6;
        let len_part = (byte & 0x3F) as usize;

        if sym_code == SYM_EXT {
            // Extension: 14-bit length + symbol byte
            if i + 2 >= data.len() { break; }
            let high = len_part;
            let low = data[i + 1] as usize;
            let run_len = (high << 8 | low) + 1;
            let ext_sym = data[i + 2];
            let val = sym_to_val(ext_sym);
            for _ in 0..run_len.min(rle.original_len - output.len()) {
                output.push(val);
            }
            i += 3;
        } else {
            let run_len = len_part + 1;
            let val = sym_to_val(sym_code);
            for _ in 0..run_len.min(rle.original_len - output.len()) {
                output.push(val);
            }
            i += 1;
        }
    }

    // Pad with zeros if somehow short
    while output.len() < rle.original_len {
        output.push(0);
    }

    output
}

#[inline]
fn sym_to_val(sym: u8) -> i8 {
    match sym & 0x03 {
        SYM_ZERO => 0,
        SYM_POS => 1,
        SYM_NEG => -1,
        _ => 0,
    }
}

/// Serializa SparseRle para bytes (prefixo com original_len).
pub fn serialize(rle: &SparseRle) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + rle.data.len());
    buf.extend_from_slice(&(rle.original_len as u32).to_le_bytes());
    buf.extend_from_slice(&rle.data);
    buf
}

/// Desserializa SparseRle de bytes.
pub fn deserialize(buf: &[u8]) -> Option<SparseRle> {
    if buf.len() < 4 { return None; }
    let original_len = u32::from_le_bytes(buf[0..4].try_into().ok()?) as usize;
    let data = buf[4..].to_vec();
    Some(SparseRle { data, original_len })
}

/// Calcula a esparsidade (fração de zeros) de um vetor de pesos.
pub fn sparsity(signal: &[i8]) -> f32 {
    if signal.is_empty() { return 0.0; }
    signal.iter().filter(|&&s| s == 0).count() as f32 / signal.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lossless_round_trip_simple() {
        let signal = vec![0, 0, 0, 1, 1, -1, -1, -1, 0, 0, 1, 0];
        let rle = compress(&signal);
        let restored = decompress(&rle);
        assert_eq!(signal, restored);
    }

    #[test]
    fn lossless_round_trip_large_sparse() {
        let n = 8192;
        let mut signal = vec![0i8; n];
        // Punturar com alguns não-zeros
        for i in (0..n).step_by(50) { signal[i] = 1; }
        for i in (25..n).step_by(50) { signal[i] = -1; }

        let rle = compress(&signal);
        let restored = decompress(&rle);

        assert_eq!(signal, restored, "RLE must be lossless");

        let ratio = n as f64 / (rle.data.len() + 4) as f64;
        assert!(ratio > 3.0, "Sparse signal should compress well: {:.1}:1", ratio);
    }

    #[test]
    fn lossless_round_trip_all_zeros() {
        let signal = vec![0i8; 1024];
        let rle = compress(&signal);
        let restored = decompress(&rle);
        assert_eq!(signal, restored);

        // All zeros should compress extremely well
        let ratio = 1024.0 / (rle.data.len() + 4) as f64;
        assert!(ratio > 40.0, "All-zero should compress > 40:1, got {:.1}:1", ratio);
    }

    #[test]
    fn lossless_round_trip_alternating() {
        // Worst case: alternating symbols, no runs
        let signal: Vec<i8> = (0..256).map(|i| [1, 0, -1][i % 3]).collect();
        let rle = compress(&signal);
        let restored = decompress(&rle);
        assert_eq!(signal, restored);
    }

    #[test]
    fn lossless_round_trip_single_element() {
        for val in [-1i8, 0, 1] {
            let signal = vec![val];
            let rle = compress(&signal);
            let restored = decompress(&rle);
            assert_eq!(signal, restored, "Failed for val={}", val);
        }
    }

    #[test]
    fn serialize_deserialize() {
        let signal = vec![0i8, 1, 1, -1, 0, 0, 0];
        let rle = compress(&signal);
        let bytes = serialize(&rle);
        let rle2 = deserialize(&bytes).expect("deser must work");
        let restored = decompress(&rle2);
        assert_eq!(signal, restored);
    }

    #[test]
    fn sparsity_calculation() {
        assert!((sparsity(&[0, 0, 0, 1]) - 0.75).abs() < 1e-6);
        assert!((sparsity(&[1, -1, 1, -1]) - 0.0).abs() < 1e-6);
        assert!((sparsity(&[0, 0, 0, 0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn long_run_extension() {
        // Runs > 64 devem usar extensão
        let signal = vec![0i8; 500];
        let rle = compress(&signal);
        let restored = decompress(&rle);
        assert_eq!(signal, restored, "Long run must survive extension coding");
    }
}
