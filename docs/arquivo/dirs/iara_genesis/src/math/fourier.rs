//! # Fourier Compression para pesos ternários
//!
//! Decompõe um vetor de pesos `{-1, 0, +1}` em coeficientes de Fourier (seno/cosseno).
//! Com K harmônicos, armazenamos 2K floats ao invés de N inteiros.
//!
//! **Ratio**: N / (2K × 4) — para N=8192 e K=32: **32:1**
//!
//! A reconstrução aplica IDFT e re-quantiza para ternário via thresholds adaptativos.

use std::f64::consts::PI;

/// Coeficientes de Fourier reais: pares (a_k, b_k) para cada harmônico k.
#[derive(Clone, Debug)]
pub struct FourierCoeffs {
    /// Pares (a_k, b_k) — cosseno e seno respectivamente.
    pub coeffs: Vec<(f32, f32)>,
    /// Número de pontos originais (necessário para reconstrução).
    pub original_len: usize,
    /// Threshold para re-quantização ternária. Adaptativo baseado na distribuição.
    pub ternary_threshold: f32,
}

/// Analisa um vetor de i8 e extrai K harmônicos de Fourier.
///
/// Usa DFT real-valued (não precisa de FFT full-blown para K << N).
/// Complexidade: O(N × K), que para N=8192 e K=32 é ~260K operações.
pub fn analyze(signal: &[i8], num_harmonics: usize) -> FourierCoeffs {
    let n = signal.len();
    let nf = n as f64;
    let mut coeffs = Vec::with_capacity(num_harmonics);

    for k in 0..num_harmonics {
        let kf = (k + 1) as f64; // k=0 é DC, começamos em 1
        let mut a = 0.0_f64;
        let mut b = 0.0_f64;

        for (i, &s) in signal.iter().enumerate() {
            let angle = 2.0 * PI * kf * (i as f64) / nf;
            let sv = s as f64;
            a += sv * angle.cos();
            b += sv * angle.sin();
        }

        a *= 2.0 / nf;
        b *= 2.0 / nf;
        coeffs.push((a as f32, b as f32));
    }

    // Threshold adaptativo: baseado na amplitude RMS do sinal reconstruído
    let rms = compute_rms_from_coeffs(&coeffs);
    let threshold = (rms * 0.5).clamp(0.1, 0.8);

    FourierCoeffs {
        coeffs,
        original_len: n,
        ternary_threshold: threshold as f32,
    }
}

/// Reconstrói o vetor de pesos ternários a partir dos coeficientes de Fourier.
///
/// Aplica IDFT parcial (soma dos harmônicos) e re-quantiza.
pub fn synthesize(fc: &FourierCoeffs) -> Vec<i8> {
    let n = fc.original_len;
    let nf = n as f64;
    let mut output = Vec::with_capacity(n);
    let thresh = fc.ternary_threshold as f64;

    for i in 0..n {
        let mut val = 0.0_f64;
        for (k, &(a, b)) in fc.coeffs.iter().enumerate() {
            let kf = (k + 1) as f64;
            let angle = 2.0 * PI * kf * (i as f64) / nf;
            val += (a as f64) * angle.cos() + (b as f64) * angle.sin();
        }

        // Re-quantização ternária
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

/// Serializa coeficientes de Fourier para bytes (little-endian).
///
/// Layout:
///   original_len: u32 (4B)
///   ternary_threshold: f32 (4B)
///   num_harmonics: u32 (4B)
///   coeffs: (f32, f32) × num_harmonics (8B × K)
pub fn serialize(fc: &FourierCoeffs) -> Vec<u8> {
    let k = fc.coeffs.len();
    let total = 4 + 4 + 4 + k * 8;
    let mut buf = Vec::with_capacity(total);

    buf.extend_from_slice(&(fc.original_len as u32).to_le_bytes());
    buf.extend_from_slice(&fc.ternary_threshold.to_le_bytes());
    buf.extend_from_slice(&(k as u32).to_le_bytes());

    for &(a, b) in &fc.coeffs {
        buf.extend_from_slice(&a.to_le_bytes());
        buf.extend_from_slice(&b.to_le_bytes());
    }

    buf
}

/// Desserializa coeficientes de Fourier a partir de bytes.
pub fn deserialize(buf: &[u8]) -> Option<FourierCoeffs> {
    if buf.len() < 12 {
        return None;
    }

    let original_len = u32::from_le_bytes(buf[0..4].try_into().ok()?) as usize;
    let ternary_threshold = f32::from_le_bytes(buf[4..8].try_into().ok()?);
    let num_harmonics = u32::from_le_bytes(buf[8..12].try_into().ok()?) as usize;

    if buf.len() < 12 + num_harmonics * 8 {
        return None;
    }

    let mut coeffs = Vec::with_capacity(num_harmonics);
    for k in 0..num_harmonics {
        let off = 12 + k * 8;
        let a = f32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
        let b = f32::from_le_bytes(buf[off + 4..off + 8].try_into().ok()?);
        if !a.is_finite() || !b.is_finite() {
            return None;
        }
        coeffs.push((a, b));
    }

    Some(FourierCoeffs {
        coeffs,
        original_len,
        ternary_threshold,
    })
}

/// Tamanho em bytes da serialização para K harmônicos.
pub fn serialized_size(num_harmonics: usize) -> usize {
    4 + 4 + 4 + num_harmonics * 8
}

/// Calcula RMS (root mean square) da amplitude dos coeficientes.
fn compute_rms_from_coeffs(coeffs: &[(f32, f32)]) -> f64 {
    if coeffs.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = coeffs
        .iter()
        .map(|&(a, b)| {
            let af = a as f64;
            let bf = b as f64;
            af * af + bf * bf
        })
        .sum();
    (sum_sq / coeffs.len() as f64).sqrt()
}

/// Coeficientes de Fourier 2D para campos de energia (Solo).
///
/// Representa um campo 2D como soma de ondas planas:
///   f(x,y) = Σ_k Σ_l c_{k,l} × cos(2π(kx/W + ly/H) + φ_{k,l})
///
/// Armazena (amplitude, fase) por modo.
#[derive(Clone, Debug)]
pub struct FourierField2D {
    pub modes: Vec<(f32, f32)>, // (amplitude, phase)
    pub width: usize,
    pub height: usize,
    pub kx_max: usize,
    pub ky_max: usize,
}

impl FourierField2D {
    /// Cria campo 2D a partir de dados em grid.
    pub fn from_grid(
        data: &[f32],
        width: usize,
        height: usize,
        kx_max: usize,
        ky_max: usize,
    ) -> Self {
        let mut modes = Vec::with_capacity(kx_max * ky_max);
        let wf = width as f64;
        let hf = height as f64;

        for ky in 0..ky_max {
            for kx in 0..kx_max {
                let mut real = 0.0_f64;
                let mut imag = 0.0_f64;

                for y in 0..height {
                    for x in 0..width {
                        let val = data[y * width + x] as f64;
                        let angle = 2.0
                            * PI
                            * ((kx as f64) * (x as f64) / wf + (ky as f64) * (y as f64) / hf);
                        real += val * angle.cos();
                        imag += val * angle.sin();
                    }
                }

                let norm = 2.0 / (wf * hf);
                real *= norm;
                imag *= norm;

                let amplitude = ((real * real + imag * imag).sqrt()) as f32;
                let phase = (imag.atan2(real)) as f32;
                modes.push((amplitude, phase));
            }
        }

        Self {
            modes,
            width,
            height,
            kx_max,
            ky_max,
        }
    }

    /// Reconstrói campo no ponto (x, y) normalizado [0, 1).
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let mut val = 0.0_f32;
        for ky in 0..self.ky_max {
            for kx in 0..self.kx_max {
                let idx = ky * self.kx_max + kx;
                let (amp, phase) = self.modes[idx];
                let angle = 2.0 * std::f32::consts::PI * (kx as f32 * x + ky as f32 * y) + phase;
                val += amp * angle.cos();
            }
        }
        val
    }

    /// Serializa para bytes.
    pub fn serialize(&self) -> Vec<u8> {
        let header = 4 * 4; // width, height, kx_max, ky_max
        let data = self.modes.len() * 8; // (f32, f32) cada
        let mut buf = Vec::with_capacity(header + data);

        buf.extend_from_slice(&(self.width as u32).to_le_bytes());
        buf.extend_from_slice(&(self.height as u32).to_le_bytes());
        buf.extend_from_slice(&(self.kx_max as u32).to_le_bytes());
        buf.extend_from_slice(&(self.ky_max as u32).to_le_bytes());

        for &(amp, phase) in &self.modes {
            buf.extend_from_slice(&amp.to_le_bytes());
            buf.extend_from_slice(&phase.to_le_bytes());
        }

        buf
    }

    /// Desserializa de bytes.
    pub fn deserialize(buf: &[u8]) -> Option<Self> {
        if buf.len() < 16 {
            return None;
        }

        let width = u32::from_le_bytes(buf[0..4].try_into().ok()?) as usize;
        let height = u32::from_le_bytes(buf[4..8].try_into().ok()?) as usize;
        let kx_max = u32::from_le_bytes(buf[8..12].try_into().ok()?) as usize;
        let ky_max = u32::from_le_bytes(buf[12..16].try_into().ok()?) as usize;

        let expected = kx_max * ky_max;
        if buf.len() < 16 + expected * 8 {
            return None;
        }

        let mut modes = Vec::with_capacity(expected);
        for i in 0..expected {
            let off = 16 + i * 8;
            let amp = f32::from_le_bytes(buf[off..off + 4].try_into().ok()?);
            let phase = f32::from_le_bytes(buf[off + 4..off + 8].try_into().ok()?);
            modes.push((amp, phase));
        }

        Some(Self {
            modes,
            width,
            height,
            kx_max,
            ky_max,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analyze_synthesize_preserves_strong_signal() {
        // Sinal com padrão forte: onda senoidal quantizada
        let n = 256;
        let signal: Vec<i8> = (0..n)
            .map(|i| {
                let v = (2.0 * PI * 3.0 * i as f64 / n as f64).sin();
                if v > 0.3 {
                    1
                } else if v < -0.3 {
                    -1
                } else {
                    0
                }
            })
            .collect();

        let fc = analyze(&signal, 8);
        let restored = synthesize(&fc);

        // Para um sinal com periodicidade limpa, Fourier deve capturar bem
        let matches: usize = signal
            .iter()
            .zip(restored.iter())
            .filter(|&(&a, &b)| a == b)
            .count();
        let accuracy = matches as f64 / n as f64;

        assert!(
            accuracy > 0.70,
            "Fourier should capture periodic signal well: {:.1}%",
            accuracy * 100.0
        );
    }

    #[test]
    fn serialize_deserialize_round_trip() {
        let signal: Vec<i8> = (0..128).map(|i| ((i % 3) as i8) - 1).collect();
        let fc = analyze(&signal, 16);
        let bytes = serialize(&fc);
        let fc2 = deserialize(&bytes).expect("deserialize must succeed");

        assert_eq!(fc.original_len, fc2.original_len);
        assert_eq!(fc.coeffs.len(), fc2.coeffs.len());
        for (a, b) in fc.coeffs.iter().zip(fc2.coeffs.iter()) {
            assert!((a.0 - b.0).abs() < 1e-6);
            assert!((a.1 - b.1).abs() < 1e-6);
        }
    }

    #[test]
    fn compression_ratio_is_significant() {
        let n = 8192;
        let signal: Vec<i8> = (0..n).map(|i| ((i * 7 + 3) % 3 - 1) as i8).collect();
        let fc = analyze(&signal, 32);
        let bytes = serialize(&fc);

        let ratio = n as f64 / bytes.len() as f64;
        assert!(ratio > 20.0, "Ratio should be > 20:1, got {:.1}:1", ratio);
    }

    #[test]
    fn fourier2d_round_trip() {
        let w = 8;
        let h = 8;
        let data: Vec<f32> = (0..w * h).map(|i| (i as f32 / 64.0).sin()).collect();
        let field = FourierField2D::from_grid(&data, w, h, 4, 4);

        let bytes = field.serialize();
        let restored = FourierField2D::deserialize(&bytes).unwrap();

        assert_eq!(restored.width, w);
        assert_eq!(restored.height, h);
        assert_eq!(restored.modes.len(), field.modes.len());
    }

    #[test]
    fn all_zeros_handled() {
        let signal = vec![0i8; 1024];
        let fc = analyze(&signal, 8);
        let restored = synthesize(&fc);
        assert!(
            restored.iter().all(|&w| w == 0),
            "All-zero input should restore to all zeros"
        );
    }
}
