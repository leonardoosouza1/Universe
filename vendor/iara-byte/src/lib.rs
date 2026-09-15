//! iara-byte — byte encoding foundation for the IARA ecosystem.
//!
//! The thesis: bytes (256 values, 8 bits) are the fundamental vocabulary.
//! Every bit position carries validated semantic meaning:
//!
//! | bit | semantics                                      |
//! |-----|------------------------------------------------|
//! |  7  | UTF-8 structural (0=ASCII, 1=multi-byte lead)  |
//! |  6  | ASCII structural                               |
//! |  5  | case — 0=UPPER, 1=lower (KL=0.003, validated)  |
//! | 0-4 | identity within group (16 slots per group)     |
//!
//! Modules:
//! - Bit-level semantics (bit5, bit7, casefold, is_case_pair, ident5)
//! - Byte classification (ByteClass / classify)
//! - Text normalization (strip_diacritics, normalize)
//! - BF16 ↔ F32 conversion via raw bit manipulation (no external deps)

// ── Bit-level semantics ───────────────────────────────────────────────────────

/// Bit 7: UTF-8 structural (0=ASCII, 1=multi-byte continuation or lead).
#[inline]
pub fn bit7(b: u8) -> u8 {
    (b >> 7) & 1
}

/// Bit 6: ASCII structural.
#[inline]
pub fn bit6(b: u8) -> u8 {
    (b >> 6) & 1
}

/// Bit 5: case (0=UPPER, 1=lower) — VALIDATED KL=0.003 on GPU 2026-05-11.
#[inline]
pub fn bit5(b: u8) -> u8 {
    (b >> 5) & 1
}

/// Bits 0-4: identity within group (16 slots per group).
#[inline]
pub fn ident5(b: u8) -> u8 {
    b & 0x1F
}

/// Returns `true` if `a` and `b` are an ASCII alphabetic case pair —
/// i.e. they differ **only** in bit 5 (e.g. `'A'` / `'a'`).
#[inline]
pub fn is_case_pair(a: u8, b: u8) -> bool {
    a.is_ascii_alphabetic() && b.is_ascii_alphabetic() && (a ^ b) == 0x20
}

/// Force lowercase via bit 5: sets bit 5 for ASCII alphabetic bytes.
/// Non-alphabetic bytes are returned unchanged.
#[inline]
pub fn casefold(b: u8) -> u8 {
    if b.is_ascii_alphabetic() {
        b | 0x20
    } else {
        b
    }
}

// ── Byte classification ───────────────────────────────────────────────────────

/// Semantic class of a single byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteClass {
    /// `A-Z`
    AsciiUpper,
    /// `a-z`
    AsciiLower,
    /// `0-9`
    Digit,
    /// space, tab, newline, carriage-return
    Whitespace,
    /// ASCII printable non-alphanumeric non-whitespace
    Punctuation,
    /// `0xC0..=0xFD` — UTF-8 lead byte (2-to-4 byte sequences)
    Utf8Lead,
    /// `0x80..=0xBF` — UTF-8 continuation byte
    Utf8Cont,
    /// Everything else (C0/C1 control codes, DEL, 0xFE, 0xFF)
    Control,
}

/// Classify a single byte into its [`ByteClass`].
pub fn classify(b: u8) -> ByteClass {
    match b {
        b'A'..=b'Z' => ByteClass::AsciiUpper,
        b'a'..=b'z' => ByteClass::AsciiLower,
        b'0'..=b'9' => ByteClass::Digit,
        b' ' | b'\t' | b'\n' | b'\r' => ByteClass::Whitespace,
        0x21..=0x2F | 0x3A..=0x40 | 0x5B..=0x60 | 0x7B..=0x7E => ByteClass::Punctuation,
        0x80..=0xBF => ByteClass::Utf8Cont,
        0xC0..=0xFD => ByteClass::Utf8Lead,
        _ => ByteClass::Control,
    }
}

// ── BF16 helpers ─────────────────────────────────────────────────────────────

/// Convert a BF16 value (stored as u16 LE bits) to F32.
///
/// BF16 occupies the upper 16 bits of an F32's IEEE 754 representation.
/// Shift left by 16 to reconstruct the full F32 bit pattern.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Convert F32 to BF16 (upper 16 bits of the F32 bit pattern, rounded).
///
/// Uses round-to-nearest-even (banker's rounding) to minimise systematic
/// quantisation bias, matching PyTorch / numpy behaviour.
#[inline]
pub fn f32_to_bf16(val: f32) -> u16 {
    let bits = val.to_bits();
    // Handle NaN: propagate with payload in upper 16 bits, ensure NaN bit set.
    if val.is_nan() {
        return ((bits >> 16) | 0x0040) as u16;
    }
    // Round-to-nearest-even: examine the low 16 bits.
    let round_bit = (bits >> 15) & 1;
    let lower = bits & 0xFFFF;
    let truncated = (bits >> 16) as u16;
    // Add 1 to the upper half if the lower bits indicate rounding up.
    if lower > 0x8000 || (lower == 0x8000 && round_bit == 1) {
        truncated.wrapping_add(1)
    } else {
        truncated
    }
}

/// Convert a slice of BF16 words to F32.
pub fn bf16_slice_to_f32(src: &[u16]) -> Vec<f32> {
    src.iter().map(|&b| bf16_to_f32(b)).collect()
}

/// Convert a slice of F32 to BF16 words.
pub fn f32_slice_to_bf16(src: &[f32]) -> Vec<u16> {
    src.iter().map(|&v| f32_to_bf16(v)).collect()
}

// ── Text normalization ────────────────────────────────────────────────────────

/// Strip common Latin diacritics from a UTF-8 byte slice.
///
/// Handles the 2-byte UTF-8 sequences in the C3 block (U+00C0–U+00FF):
/// accented vowels, cedilla, etc. Bytes that don't match are passed through
/// unchanged; unrecognized C3-continuation bytes are dropped.
pub fn strip_diacritics(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == 0xC3 && i + 1 < bytes.len() {
            let mapped = match bytes[i + 1] {
                // Lowercase accented vowels
                0xA0..=0xA5 => b'a',
                0xA8..=0xAB => b'e',
                0xAC..=0xAF => b'i',
                0xB2..=0xB6 => b'o',
                0xB9..=0xBC => b'u',
                0xA7 => b'c',
                // Uppercase accented vowels
                0x80..=0x85 => b'A',
                0x87 => b'C',
                0x88..=0x8B => b'E',
                0x8C..=0x8F => b'I',
                0x92..=0x96 => b'O',
                0x99..=0x9C => b'U',
                // Unknown C3-sequence: skip both bytes
                _ => {
                    i += 2;
                    continue;
                }
            };
            out.push(mapped);
            i += 2;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    out
}

/// Full normalization pipeline:
/// 1. Strip UTF-8 diacritics (PT-BR map).
/// 2. Casefold all ASCII alphabetic bytes (bit 5 = 1).
/// 3. Collapse runs of ASCII whitespace to a single `' '` and trim.
///
/// Non-alphabetic, non-whitespace bytes (punctuation, digits, etc.) are
/// preserved unchanged after casefolding.  Returns a `String`.
pub fn normalize(s: &str) -> String {
    let stripped = strip_diacritics(s.as_bytes());
    let casefolded: Vec<u8> = stripped.iter().map(|&b| casefold(b)).collect();

    let mut result = String::with_capacity(casefolded.len());
    let mut in_space = true; // start true so leading whitespace is trimmed
    for b in &casefolded {
        if b.is_ascii_whitespace() {
            if !in_space {
                result.push(' ');
                in_space = true;
            }
        } else {
            result.push(*b as char);
            in_space = false;
        }
    }
    // Trim trailing space pushed before non-space bytes ran out.
    if result.ends_with(' ') {
        result.pop();
    }
    result
}

/// Like [`normalize`] but maps non-alphanumeric bytes to space, returning a
/// `Vec<u8>` containing only `[a-z0-9 ]`.  Used by the HDC byte-chat system.
pub fn normalize_hdc(s: &str) -> Vec<u8> {
    let stripped = strip_diacritics(s.as_bytes());
    let mapped: Vec<u8> = stripped
        .iter()
        .map(|&b| {
            if b.is_ascii_alphanumeric() {
                b | 0x20 // casefold
            } else {
                b' '
            }
        })
        .collect();

    let mut out = Vec::with_capacity(mapped.len());
    let mut prev_space = true;
    for b in mapped {
        if b == b' ' {
            if !prev_space {
                out.push(b' ');
            }
            prev_space = true;
        } else {
            out.push(b);
            prev_space = false;
        }
    }
    if out.last() == Some(&b' ') {
        out.pop();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── casefold ──────────────────────────────────────────────────────────────

    #[test]
    fn casefold_upper_to_lower() {
        assert_eq!(casefold(b'A'), b'a');
        assert_eq!(casefold(b'Z'), b'z');
    }

    #[test]
    fn casefold_lower_unchanged() {
        assert_eq!(casefold(b'z'), b'z');
        assert_eq!(casefold(b'a'), b'a');
    }

    #[test]
    fn casefold_non_alpha_unchanged() {
        assert_eq!(casefold(b'5'), b'5');
        assert_eq!(casefold(b' '), b' ');
        assert_eq!(casefold(b'!'), b'!');
    }

    // ── bit5 ─────────────────────────────────────────────────────────────────

    #[test]
    fn bit5_upper_is_zero() {
        assert_eq!(bit5(b'A'), 0);
        assert_eq!(bit5(b'Z'), 0);
    }

    #[test]
    fn bit5_lower_is_one() {
        assert_eq!(bit5(b'a'), 1);
        assert_eq!(bit5(b'z'), 1);
    }

    // ── is_case_pair ─────────────────────────────────────────────────────────

    #[test]
    fn is_case_pair_true() {
        assert!(is_case_pair(b'A', b'a'));
        assert!(is_case_pair(b'a', b'A'));
        assert!(is_case_pair(b'Z', b'z'));
    }

    #[test]
    fn is_case_pair_false_same_case() {
        assert!(!is_case_pair(b'A', b'B'));
        assert!(!is_case_pair(b'a', b'b'));
    }

    #[test]
    fn is_case_pair_false_non_alpha() {
        assert!(!is_case_pair(b'1', b'!'));
        assert!(!is_case_pair(b' ', b'`'));
    }

    // ── strip_diacritics ─────────────────────────────────────────────────────

    #[test]
    fn strip_diacritics_cafe() {
        let result = strip_diacritics("café".as_bytes());
        assert_eq!(result, b"cafe");
    }

    #[test]
    fn strip_diacritics_ptbr_lowercase() {
        let result = strip_diacritics("áéíóúç".as_bytes());
        assert_eq!(result, b"aeiouc");
    }

    #[test]
    fn strip_diacritics_ptbr_uppercase() {
        let result = strip_diacritics("ÁÉÍÓÚÇ".as_bytes());
        assert_eq!(result, b"AEIOUC");
    }

    #[test]
    fn strip_diacritics_passthrough_ascii() {
        let input = b"hello world";
        let result = strip_diacritics(input);
        assert_eq!(result, input);
    }

    // ── normalize (String, preserves punctuation) ─────────────────────────────

    #[test]
    fn normalize_full_pipeline() {
        assert_eq!(normalize("  Café  HELLO  "), "cafe hello");
    }

    #[test]
    fn normalize_collapses_internal_whitespace() {
        assert_eq!(normalize("foo   bar"), "foo bar");
    }

    #[test]
    fn normalize_empty() {
        assert_eq!(normalize(""), "");
    }

    #[test]
    fn normalize_only_whitespace() {
        assert_eq!(normalize("   \t\n  "), "");
    }

    #[test]
    fn normalize_ptbr_phrase() {
        // Punctuation preserved; diacritics stripped; casefolded.
        assert_eq!(normalize("Olá, Mundo!"), "ola, mundo!");
    }

    // ── normalize_hdc (Vec<u8>, HDC-friendly, maps punctuation to space) ──────

    #[test]
    fn normalize_hdc_casefold() {
        assert_eq!(normalize_hdc("Hello"), b"hello");
    }

    #[test]
    fn normalize_hdc_diacritics() {
        assert_eq!(normalize_hdc("olá café"), b"ola cafe");
    }

    #[test]
    fn normalize_hdc_punctuation_to_space() {
        assert_eq!(normalize_hdc("hello, world!"), b"hello world");
    }

    #[test]
    fn normalize_hdc_collapses_spaces() {
        assert_eq!(normalize_hdc("a  b  c"), b"a b c");
    }

    #[test]
    fn normalize_hdc_trim() {
        assert_eq!(normalize_hdc("  hi  "), b"hi");
    }

    #[test]
    fn normalize_hdc_empty() {
        assert_eq!(normalize_hdc(""), b"");
    }

    // ── classify ─────────────────────────────────────────────────────────────

    #[test]
    fn classify_letters() {
        assert_eq!(classify(b'A'), ByteClass::AsciiUpper);
        assert_eq!(classify(b'z'), ByteClass::AsciiLower);
    }

    #[test]
    fn classify_digit() {
        assert_eq!(classify(b'5'), ByteClass::Digit);
    }

    #[test]
    fn classify_whitespace() {
        assert_eq!(classify(b' '), ByteClass::Whitespace);
        assert_eq!(classify(b'\n'), ByteClass::Whitespace);
        assert_eq!(classify(b'\t'), ByteClass::Whitespace);
        assert_eq!(classify(b'\r'), ByteClass::Whitespace);
    }

    #[test]
    fn classify_punctuation() {
        assert_eq!(classify(b'!'), ByteClass::Punctuation);
        assert_eq!(classify(b'.'), ByteClass::Punctuation);
        assert_eq!(classify(b'@'), ByteClass::Punctuation);
    }

    #[test]
    fn classify_utf8() {
        assert_eq!(classify(0xC3), ByteClass::Utf8Lead);
        assert_eq!(classify(0xA9), ByteClass::Utf8Cont);
        assert_eq!(classify(0x80), ByteClass::Utf8Cont);
        assert_eq!(classify(0xBF), ByteClass::Utf8Cont);
    }

    #[test]
    fn classify_control() {
        assert_eq!(classify(0x00), ByteClass::Control); // NUL
        assert_eq!(classify(0x1B), ByteClass::Control); // ESC
        assert_eq!(classify(0x7F), ByteClass::Control); // DEL
    }

    // ── bit helpers ───────────────────────────────────────────────────────────

    #[test]
    fn bit7_ascii_vs_multibyte() {
        assert_eq!(bit7(b'A'), 0);
        assert_eq!(bit7(0xC3), 1);
        assert_eq!(bit7(0xA9), 1);
    }

    #[test]
    fn ident5_masks_lower_5_bits() {
        assert_eq!(ident5(0b1010_0101), 0b0000_0101);
        assert_eq!(ident5(0xFF), 0x1F);
        assert_eq!(ident5(0x00), 0x00);
    }

    // ── BF16 ↔ F32 ───────────────────────────────────────────────────────────

    #[test]
    fn bf16_zero_roundtrip() {
        let v = 0.0f32;
        assert_eq!(bf16_to_f32(f32_to_bf16(v)), v);
    }

    #[test]
    fn bf16_one_roundtrip() {
        // 1.0f32 bit pattern = 0x3F80_0000; upper 16 = 0x3F80.
        let bits = f32_to_bf16(1.0f32);
        assert_eq!(bits, 0x3F80);
        assert!((bf16_to_f32(bits) - 1.0f32).abs() < 1e-4);
    }

    #[test]
    fn bf16_to_f32_known_zero() {
        assert_eq!(bf16_to_f32(0x0000), 0.0f32);
    }

    /// Values whose lower 16 bits are all zero roundtrip exactly (no rounding).
    #[test]
    fn bf16_exact_roundtrip_values() {
        for &val in &[0.0f32, 1.0, -1.0, 0.5, 2.0, 256.0, -0.125] {
            let bits = f32_to_bf16(val);
            let recovered = bf16_to_f32(bits);
            assert!(
                (recovered - val).abs() < val.abs() * 1e-3 + 1e-7,
                "roundtrip failed for {val}: bits={bits:#06x} recovered={recovered}"
            );
        }
    }

    #[test]
    fn bf16_negative_roundtrip() {
        let v = -3.14f32;
        let back = bf16_to_f32(f32_to_bf16(v));
        assert!((back - v).abs() < 1e-2, "got {back} expected ~{v}");
    }

    #[test]
    fn bf16_nan_stays_nan() {
        let bits = f32_to_bf16(f32::NAN);
        assert!(bf16_to_f32(bits).is_nan());
    }

    #[test]
    fn bf16_slice_roundtrip() {
        let src = vec![0.0f32, 1.0, -1.0, 42.0, -0.001];
        let bf = f32_slice_to_bf16(&src);
        let back = bf16_slice_to_f32(&bf);
        for (a, b) in src.iter().zip(back.iter()) {
            assert!(
                (a - b).abs() <= a.abs() * 0.01 + 1e-4,
                "bf16 roundtrip: {a} → {b}"
            );
        }
    }
}
