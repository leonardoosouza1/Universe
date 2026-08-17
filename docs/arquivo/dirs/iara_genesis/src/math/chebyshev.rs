//! # Chebyshev Polynomial Compression
//!
//! Aproxima um vetor de pesos ternários `{-1, 0, +1}` usando polinômios de Chebyshev.
//! Chebyshev é ótimo para pesos "suaves" (gradientes, rampas, sigmoides).
//!
//! **Ratio**: N / (K × 4) — para N=8192 e K=16: **128:1**
//!
//! Mais lossy que Fourier, mas surpreendentemente eficaz para pesos que
//! representam curvas de resposta (splines, KAN coefficients).

/// Coeficientes de Chebyshev para compressão de pesos.
#[derive(Clone, Debug)]
pub struct ChebyshevCoeffs {
    /// Coeficientes c_k do polinômio de Chebyshev de primeiro tipo T_k.
    pub coeffs: Vec<f32>,
    /// Comprimento original do vetor.
    pub original_len: usize,
    /// Threshold para re-quantização ternária.
    pub ternary_threshold: f32,
}

/// Avalia o polinômio de Chebyshev T_k(x) via recorrência:
///   T_0(x) = 1
///   T_1(x) = x
///   T_{k+1}(x) = 2x·T_k(x) - T_{k-1}(x)
#[inline]
fn chebyshev_t(k: usize, x: f64) -> f64 {
    match k {
        0 => 1.0,
        1 => x,
        _ => {
            let mut t_prev2 = 1.0;
            let mut t_prev1 = x;
            for _ in 2..=k {
                let t_curr = 2.0 * x * t_prev1 - t_prev2;
                t_prev2 = t_prev1;
                t_prev1 = t_curr;
            }
            t_prev1
        }
    }
}

/// Ajusta coeficientes de Chebyshev via projeção discreta (quadratura).
///
/// Para cada coeficiente c_k, calcula:
///   c_k = (2/N) × Σ_i signal[i] × T_k(x_i)
///
/// onde x_i = 2i/(N-1) - 1 (mapeamento [0, N-1] → [-1, 1]).
pub fn fit(signal: &[i8], num_terms: usize) -> ChebyshevCoeffs {
    let n = signal.len();
    let nf = n as f64;
    let mut coeffs = Vec::with_capacity(num_terms);

    for k in 0..num_terms {
        let mut ck = 0.0_f64;
        for (i, &s) in signal.iter().enumerate() {
            let x = if n > 1 {
                2.0 * (i as f64) / (nf - 1.0) - 1.0
            } else {
                0.0
            };
            ck += (s as f64) * chebyshev_t(k, x);
        }
        // Normalização: 2/N para k>0, 1/N para k=0
        ck *= if k == 0 { 1.0 / nf } else { 2.0 / nf };
        coeffs.push(ck as f32);
    }

    // Threshold adaptativo: proporção de não-zeros no sinal original
    let nonzero_frac = signal.iter().filter(|&&s| s != 0).count() as f32 / n as f32;
    let threshold = (0.3 + 0.4 * (1.0 - nonzero_frac)).clamp(0.15, 0.85);

    ChebyshevCoeffs {
        coeffs,
        original_len: n,
        ternary_threshold: threshold,
    }
}

/// Reconstrói o vetor de pesos ternários a partir dos coeficientes de Chebyshev.
///
///   signal[i] ≈ quantize(Σ_k c_k × T_k(x_i))
pub fn reconstruct(cc: &ChebyshevCoeffs) -> Vec<i8> {
    let n = cc.original_len;
    let thresh = cc.ternary_threshold as f64;
    let mut output = Vec::with_capacity(n);

    for i in 0..n {
        let x = if n > 1 {
            2.0 * (i as f64) / (n as f64 - 1.0) - 1.0
        } else {
            0.0
        };

        let mut val = 0.0_f64;
        for (k, &ck) in cc.coeffs.iter().enumerate() {
            val += (ck as f64) * chebyshev_t(k, x);
        }

        let quantized = if val > thresh {
            1i8
        } else if val < -thresh {
            -1i8
        } else {
            0i8
        };
        output.push(quantized);
    }

    output
}

/// Serializa coeficientes para bytes.
///
/// Layout:
///   original_len: u32 (4B)
///   ternary_threshold: f32 (4B)
///   num_terms: u32 (4B)
///   coeffs: f32 × num_terms (4B × K)
pub fn serialize(cc: &ChebyshevCoeffs) -> Vec<u8> {
    let k = cc.coeffs.len();
    let mut buf = Vec::with_capacity(12 + k * 4);

    buf.extend_from_slice(&(cc.original_len as u32).to_le_bytes());
    buf.extend_from_slice(&cc.ternary_threshold.to_le_bytes());
    buf.extend_from_slice(&(k as u32).to_le_bytes());

    for &c in &cc.coeffs {
        buf.extend_from_slice(&c.to_le_bytes());
    }

    buf
}

/// Desserializa coeficientes de bytes.
pub fn deserialize(buf: &[u8]) -> Option<ChebyshevCoeffs> {
    if buf.len() < 12 {
        return None;
    }

    let original_len = u32::from_le_bytes(buf[0..4].try_into().ok()?) as usize;
    let ternary_threshold = f32::from_le_bytes(buf[4..8].try_into().ok()?);
    let num_terms = u32::from_le_bytes(buf[8..12].try_into().ok()?) as usize;

    if buf.len() < 12 + num_terms * 4 {
        return None;
    }

    let mut coeffs = Vec::with_capacity(num_terms);
    for k in 0..num_terms {
        let off = 12 + k * 4;
        let c = f32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        if !c.is_finite() {
            return None;
        }
        coeffs.push(c);
    }

    Some(ChebyshevCoeffs {
        coeffs,
        original_len,
        ternary_threshold,
    })
}

/// Tamanho em bytes da serialização.
pub fn serialized_size(num_terms: usize) -> usize {
    12 + num_terms * 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chebyshev_t_known_values() {
        assert!((chebyshev_t(0, 0.5) - 1.0).abs() < 1e-10);
        assert!((chebyshev_t(1, 0.5) - 0.5).abs() < 1e-10);
        // T_2(x) = 2x² - 1 → T_2(0.5) = 2(0.25) - 1 = -0.5
        assert!((chebyshev_t(2, 0.5) - (-0.5)).abs() < 1e-10);
        // T_3(x) = 4x³ - 3x → T_3(0.5) = 4(0.125) - 1.5 = -1.0
        assert!((chebyshev_t(3, 0.5) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn fit_reconstruct_sparse_signal() {
        // Sinal esparso (muitos zeros) — Chebyshev deve pelo menos não explodir
        let n = 512;
        let signal: Vec<i8> = (0..n)
            .map(|i| {
                if i % 20 == 0 {
                    1
                } else if i % 30 == 0 {
                    -1
                } else {
                    0
                }
            })
            .collect();

        let cc = fit(&signal, 32);
        let restored = reconstruct(&cc);

        assert_eq!(restored.len(), n);
        // Chebyshev com poucos termos é lossy; verificar que não é lixo
        let zeros_original = signal.iter().filter(|&&s| s == 0).count();
        let zeros_restored = restored.iter().filter(|&&s| s == 0).count();
        // Pelo menos metade dos zeros devem ser preservados
        assert!(
            zeros_restored > zeros_original / 2,
            "Too many false-positives: orig zeros={}, restored zeros={}",
            zeros_original,
            zeros_restored
        );
    }

    #[test]
    fn serialize_deserialize_round_trip() {
        let signal: Vec<i8> = (0..64).map(|i| ((i % 3) - 1) as i8).collect();
        let cc = fit(&signal, 8);
        let bytes = serialize(&cc);
        let cc2 = deserialize(&bytes).expect("deser must work");

        assert_eq!(cc.original_len, cc2.original_len);
        assert_eq!(cc.coeffs.len(), cc2.coeffs.len());
        for (a, b) in cc.coeffs.iter().zip(cc2.coeffs.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn extreme_compression_ratio() {
        let n = 8192;
        let signal: Vec<i8> = vec![0; n];
        let cc = fit(&signal, 16);
        let bytes = serialize(&cc);
        let ratio = n as f64 / bytes.len() as f64;
        // 16 termos → 12 + 64 = 76 bytes, ratio ~ 107:1
        assert!(
            ratio > 80.0,
            "Ratio for zeros should be huge: {:.0}:1",
            ratio
        );
    }
}
