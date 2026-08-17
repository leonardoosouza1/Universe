/// Quantization and dequantization: i8↔f32 (scale 127), disk (scale 63.5), BF16↔F32.
///
/// Documented in MATH_INVENTORY.md:
/// - F10 (BitLinear): encode/decode with scale 127 for runtime
/// - F15 (DNA serialization): encode/decode with scale 63.5 for disk
/// - F07 (CfC): f32 state → i8 via tanh * 127

/// Runtime scale: i8 activations represent values in [-1.0, 1.0].
/// 127 = max of positive i8. Formula: f32 = i8 / 127.0
pub const I8_SCALE: f32 = 127.0;

/// Disk scale (DNA): L1/L2 weights in [-2.0, 2.0] compressed to i8.
/// 63.5 = 127 / 2.0. Formula: i8 = (f32 * 63.5).round().clamp(-127, 127)
pub const I8_DISK_SCALE: f32 = 63.5;

/// Encodes f32 to i8 with scale 127 (runtime activations).
///
/// Assumes `v` is in range [-1.0, 1.0]. Out-of-range values are clamped.
#[inline]
pub fn encode_i8_act(v: f32) -> i8 {
    (v * I8_SCALE).round().clamp(-127.0, 127.0) as i8
}

/// Decodes i8 back to f32 with scale 127.
#[inline]
pub fn decode_i8_act(v: i8) -> f32 {
    v as f32 / I8_SCALE
}

/// Encodes f32 to i8 with scale 63.5 (disk weights, range [-2.0, 2.0]).
///
/// `.round()` before the clamp avoids accumulated decay over save/load cycles
/// (`as i8` truncation systematically shortens edge values — see F15).
#[inline]
pub fn encode_i8_disk(w: f32) -> i8 {
    (w * I8_DISK_SCALE).round().clamp(-127.0, 127.0) as i8
}

/// Decodes disk i8 back to f32.
#[inline]
pub fn decode_i8_disk(q: i8) -> f32 {
    q as f32 / I8_DISK_SCALE
}

/// Converts a BF16 word (16 bits) to f32.
///
/// BF16 = the 16 most significant bits of f32. Shift 16 bits to the left.
/// Used to transfer weights from GPU (BF16) to host (F32).
#[inline]
pub fn bf16_to_f32(bf16_bits: u16) -> f32 {
    f32::from_bits((bf16_bits as u32) << 16)
}

/// Converts f32 to BF16 with round-to-nearest-even rounding.
///
/// Truncates the 16 least significant mantissa bits, adding 0x8000 for
/// rounding. NaN/Inf are preserved.
#[inline]
pub fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    // Rounding: add 0x7FFF + parity bit (round-to-nearest-even)
    let rounded = bits + 0x7FFF + ((bits >> 16) & 1);
    (rounded >> 16) as u16
}

/// Core of the SNR computation: takes the original and already-reconstructed signal
/// (post decode(encode(·))), whatever scale was used. `snr_db` (disk, scale
/// 63.5) and `snr_db_act` (runtime activations, scale 127) call this core — so
/// the two variants cannot diverge in the formula, only in the quantization scale.
fn snr_db_core(original: &[f32], reconstructed: &[f32]) -> f32 {
    if original.len() != reconstructed.len() || original.is_empty() {
        return 0.0;
    }
    let mut signal_power = 0.0f32;
    let mut noise_power = 0.0f32;
    for (&w, &r) in original.iter().zip(reconstructed.iter()) {
        let err = w - r;
        signal_power += w * w;
        noise_power += err * err;
    }
    if noise_power < 1e-30 {
        // Perfect quantization (weights are zeros or fit exactly)
        return f32::INFINITY;
    }
    if signal_power < 1e-30 {
        // Signal is zero — SNR undefined; returns 0 (worst case)
        return 0.0;
    }
    10.0 * (signal_power / noise_power).log10()
}

/// Quantization SNR in dB: how much noise the i8 conversion introduces.
///
/// `snr_db = 10 * log10(Σ w² / Σ (w - decode(encode(w)))²)`
///
/// High SNR (~48 dB for a uniform signal) indicates the weights are well represented
/// in the range [-2.0, 2.0] with disk scale 63.5.
/// Low SNR indicates many weights are out of range or badly distributed.
pub fn snr_db(original: &[f32], quantized: &[i8]) -> f32 {
    if original.len() != quantized.len() {
        return 0.0;
    }
    let reconstructed: Vec<f32> = quantized.iter().map(|&q| decode_i8_disk(q)).collect();
    snr_db_core(original, &reconstructed)
}

/// ACTIVATION quantization SNR in dB — same formula as `snr_db`, but with the
/// runtime scale (127, range [-1, 1]) instead of the disk scale (63.5, range
/// [-2, 2]). Used by `analysis::activation` (E4) to measure the noise of quantizing
/// the L1/L2 hidden state to i8 before feeding the next layer — using
/// `snr_db` (disk scale) here would give artificially low SNR, because half
/// of the representable range (±[1,2]) is never used by an activation already
/// saturated in tanh (∈ [-1, 1]).
pub fn snr_db_act(samples: &[f32]) -> f32 {
    let reconstructed: Vec<f32> = samples
        .iter()
        .map(|&v| decode_i8_act(encode_i8_act(v)))
        .collect();
    snr_db_core(samples, &reconstructed)
}

/// Round-to-ternary: maps f32 to {-1, 0, +1} with a symmetric threshold.
///
/// `|x| < threshold` → 0 (silence)
/// `x >=  threshold` → +1
/// `x <= -threshold` → -1
///
/// Used in BitNet 1.58-bit (historical; current runtime uses f32 — see layers.rs).
#[inline]
pub fn to_ternary(x: f32, threshold: f32) -> i8 {
    if x >= threshold {
        1
    } else if x <= -threshold {
        -1
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_i8_act_invertivel() {
        // For values that are multiples of 1/127 the round-trip is exact
        let original = [0.5_f32, -0.5, 1.0, -1.0, 0.0];
        for &v in &original {
            let q = encode_i8_act(v);
            let back = decode_i8_act(q);
            let err = (back - v).abs();
            assert!(
                err < 1.0 / I8_SCALE + 1e-6,
                "round-trip error {} for v={}",
                err,
                v
            );
        }
    }

    #[test]
    fn encode_decode_i8_disk_invertivel() {
        // Range [-2, 2]: round-trip must have error < 1/63.5
        let vals = [0.0_f32, 1.0, -1.0, 1.999, -1.999, 0.5, -0.333];
        for &v in &vals {
            let q = encode_i8_disk(v);
            let back = decode_i8_disk(q);
            let err = (back - v).abs();
            assert!(
                err <= 1.0 / I8_DISK_SCALE + 1e-6,
                "disk round-trip error {} for v={}",
                err,
                v
            );
        }
    }

    #[test]
    fn encode_i8_disk_clamp_em_2() {
        // Values outside ±2.0 must be clamped to ±127
        assert_eq!(encode_i8_disk(3.0), 127);
        assert_eq!(encode_i8_disk(-3.0), -127);
    }

    #[test]
    fn bf16_roundtrip_float_simples() {
        // 1.0, 2.0, -1.0 must survive the round-trip (they are exactly representable)
        for &v in &[1.0_f32, 2.0, -1.0, 0.0, 0.5] {
            let bf = f32_to_bf16(v);
            let back = bf16_to_f32(bf);
            let err = (back - v).abs();
            // BF16 has 7 mantissa bits → relative error ≤ 2^-7 ≈ 0.0078
            assert!(
                err <= v.abs() * 0.01 + 1e-7,
                "BF16 round-trip error {} for v={}",
                err,
                v
            );
        }
    }

    #[test]
    fn snr_db_sinal_uniforme_em_torno_de_48db() {
        // Uniform weights in [-2, 2] with 1000 samples → theoretical SNR ~48 dB for
        // a 7-bit ADC (63.5 = 2^6 - 0.5 ≈ 6 effective bits + 1 sign bit):
        // SNR = 6.02 * n_bits + 1.76 ≈ 38-48 dB depending on the distribution.
        let n = 1000usize;
        let weights: Vec<f32> = (0..n)
            .map(|i| (i as f32 / (n - 1) as f32) * 4.0 - 2.0) // uniform [-2, 2]
            .collect();
        let quantized: Vec<i8> = weights.iter().map(|&w| encode_i8_disk(w)).collect();
        let snr = snr_db(&weights, &quantized);
        // For a uniform signal [-2, 2] with 127 levels the expected SNR is ~42-48 dB
        assert!(
            snr > 38.0,
            "SNR for a uniform signal must be > 38 dB, got {} dB",
            snr
        );
        assert!(snr < 60.0, "suspiciously high SNR: {} dB", snr);
    }

    #[test]
    fn snr_db_comprimentos_diferentes_retorna_zero() {
        let w = [1.0_f32; 5];
        let q = [10i8; 3]; // different size
        assert_eq!(snr_db(&w, &q), 0.0);
    }

    #[test]
    fn snr_db_act_sinal_uniforme_em_torno_de_44db() {
        // Uniform activations in [-1, 1] (tanh range) with scale 127 (7 bits + sign):
        // SNR ≈ 6.02*7 + 1.76 ≈ 44 dB, same order of magnitude as snr_db (63.5, [-2,2]).
        let n = 1000usize;
        let acts: Vec<f32> = (0..n)
            .map(|i| (i as f32 / (n - 1) as f32) * 2.0 - 1.0)
            .collect();
        let snr = snr_db_act(&acts);
        assert!(
            snr > 38.0,
            "uniform activation SNR must be > 38 dB, got {} dB",
            snr
        );
        assert!(snr < 60.0, "suspiciously high SNR: {} dB", snr);
    }

    #[test]
    fn snr_db_act_vazio_e_zero() {
        assert_eq!(snr_db_act(&[]), 0.0);
    }

    #[test]
    fn snr_db_act_usa_escala_de_ativacao_nao_de_disco() {
        // v=1.5 is INSIDE the disk range ([-2,2]) but OUTSIDE the activation range
        // ([-1,1]): encode_i8_act clamps at 127 (equivalent to 1.0), so the activation
        // SNR for this signal must be much worse than the disk one — proving the
        // two variants really use different scales, not the same one by mistake.
        let v = [1.5_f32; 50];
        let snr_act = snr_db_act(&v);
        let q_disk: Vec<i8> = v.iter().map(|&w| encode_i8_disk(w)).collect();
        let snr_disk = snr_db(&v, &q_disk);
        assert!(
            snr_act < snr_disk,
            "snr_act ({snr_act}) should be worse than snr_disk ({snr_disk}) for v=1.5 (outside the activation range)"
        );
    }

    #[test]
    fn to_ternary_mapeia_corretamente() {
        let thr = 0.5;
        assert_eq!(to_ternary(0.8, thr), 1);
        assert_eq!(to_ternary(-0.8, thr), -1);
        assert_eq!(to_ternary(0.3, thr), 0);
        assert_eq!(to_ternary(-0.3, thr), 0);
        assert_eq!(to_ternary(0.5, thr), 1); // positive edge
        assert_eq!(to_ternary(-0.5, thr), -1); // negative edge
    }
}
