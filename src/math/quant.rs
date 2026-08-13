/// Quantização e dequantização: i8↔f32 (escala 127), disco (escala 63.5), BF16↔F32.
///
/// Documentado em MATH_INVENTORY.md:
/// - F10 (BitLinear): encode/decode com escala 127 para runtime
/// - F15 (Serialização DNA): encode/decode com escala 63.5 para disco
/// - F07 (CfC): estado f32 → i8 via tanh * 127

/// Escala de runtime: ativações i8 representam valores em [-1.0, 1.0].
/// 127 = max do i8 positivo. Fórmula: f32 = i8 / 127.0
pub const I8_SCALE: f32 = 127.0;

/// Escala de disco (DNA): pesos L1/L2 em [-2.0, 2.0] comprimidos para i8.
/// 63.5 = 127 / 2.0. Fórmula: i8 = (f32 * 63.5).round().clamp(-127, 127)
pub const I8_DISK_SCALE: f32 = 63.5;

/// Codifica f32 para i8 com escala 127 (ativações runtime).
///
/// Assume que `v` está no range [-1.0, 1.0]. Valores fora do range são clamped.
#[inline]
pub fn encode_i8_act(v: f32) -> i8 {
    (v * I8_SCALE).round().clamp(-127.0, 127.0) as i8
}

/// Decodifica i8 de volta para f32 com escala 127.
#[inline]
pub fn decode_i8_act(v: i8) -> f32 {
    v as f32 / I8_SCALE
}

/// Codifica f32 para i8 com escala 63.5 (pesos em disco, range [-2.0, 2.0]).
///
/// `.round()` antes do clamp evita decaimento acumulado ao salvar/carregar ciclos
/// (truncação `as i8` encurta sistematicamente valores em borda — veja F15).
#[inline]
pub fn encode_i8_disk(w: f32) -> i8 {
    (w * I8_DISK_SCALE).round().clamp(-127.0, 127.0) as i8
}

/// Decodifica i8 de disco de volta para f32.
#[inline]
pub fn decode_i8_disk(q: i8) -> f32 {
    q as f32 / I8_DISK_SCALE
}

/// Converte palavra BF16 (16 bits) para f32.
///
/// BF16 = os 16 bits mais significativos de f32. Deslocamento de 16 bits para a esquerda.
/// Usado para transferir pesos de GPU (BF16) para host (F32).
#[inline]
pub fn bf16_to_f32(bf16_bits: u16) -> f32 {
    f32::from_bits((bf16_bits as u32) << 16)
}

/// Converte f32 para BF16 com arredondamento round-to-nearest-even.
///
/// Trunca os 16 bits menos significativos da mantissa, adicionando 0x8000 para
/// arredondamento. NaN/Inf são preservados.
#[inline]
pub fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    // Arredondamento: adiciona 0x7FFF + bit de paridade (round-to-nearest-even)
    let rounded = bits + 0x7FFF + ((bits >> 16) & 1);
    (rounded >> 16) as u16
}

/// Núcleo do cálculo de SNR: recebe o sinal original e já-reconstruído (pós
/// decode(encode(·))), qualquer que seja a escala usada. `snr_db` (disco, escala
/// 63.5) e `snr_db_act` (ativações runtime, escala 127) chamam este núcleo — assim
/// as duas variantes não podem divergir na fórmula, só na escala de quantização.
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
        // Quantização perfeita (pesos são zeros ou se encaixam exatamente)
        return f32::INFINITY;
    }
    if signal_power < 1e-30 {
        // Sinal é zero — SNR não definida; retorna 0 (pior caso)
        return 0.0;
    }
    10.0 * (signal_power / noise_power).log10()
}

/// SNR de quantização em dB: quanto ruído a conversão i8 introduz.
///
/// `snr_db = 10 * log10(Σ w² / Σ (w - decode(encode(w)))²)`
///
/// SNR alto (~48 dB para sinal uniforme) indica que os pesos são bem representados
/// no range [-2.0, 2.0] com a escala de disco 63.5.
/// SNR baixo indica que muitos pesos estão fora do range ou mal distribuídos.
pub fn snr_db(original: &[f32], quantized: &[i8]) -> f32 {
    if original.len() != quantized.len() {
        return 0.0;
    }
    let reconstructed: Vec<f32> = quantized.iter().map(|&q| decode_i8_disk(q)).collect();
    snr_db_core(original, &reconstructed)
}

/// SNR de quantização de ATIVAÇÕES em dB — mesma fórmula de `snr_db`, mas com a
/// escala de runtime (127, range [-1, 1]) em vez da escala de disco (63.5, range
/// [-2, 2]). Usado por `analysis::activation` (E4) para medir o ruído de quantizar
/// o estado oculto de L1/L2 para i8 antes de alimentar a camada seguinte — usar
/// `snr_db` (escala de disco) aqui daria SNR artificialmente baixo, porque metade
/// da faixa representável (±[1,2]) nunca é usada por uma ativação já saturada em
/// tanh (∈ [-1, 1]).
pub fn snr_db_act(samples: &[f32]) -> f32 {
    let reconstructed: Vec<f32> = samples.iter()
        .map(|&v| decode_i8_act(encode_i8_act(v)))
        .collect();
    snr_db_core(samples, &reconstructed)
}

/// Round-to-ternary: mapeia f32 para {-1, 0, +1} com limiar simétrico.
///
/// `|x| < threshold` → 0 (silêncio)
/// `x >=  threshold` → +1
/// `x <= -threshold` → -1
///
/// Usado no BitNet 1.58-bit (histórico; runtime atual usa f32 — ver layers.rs).
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
// Testes unitários
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_i8_act_invertivel() {
        // Para valores múltiplos de 1/127 o round-trip é exato
        let original = [0.5_f32, -0.5, 1.0, -1.0, 0.0];
        for &v in &original {
            let q = encode_i8_act(v);
            let back = decode_i8_act(q);
            let err = (back - v).abs();
            assert!(err < 1.0 / I8_SCALE + 1e-6,
                "round-trip erro {} para v={}", err, v);
        }
    }

    #[test]
    fn encode_decode_i8_disk_invertivel() {
        // Range [-2, 2]: round-trip deve ter erro < 1/63.5
        let vals = [0.0_f32, 1.0, -1.0, 1.999, -1.999, 0.5, -0.333];
        for &v in &vals {
            let q = encode_i8_disk(v);
            let back = decode_i8_disk(q);
            let err = (back - v).abs();
            assert!(err <= 1.0 / I8_DISK_SCALE + 1e-6,
                "round-trip disco erro {} para v={}", err, v);
        }
    }

    #[test]
    fn encode_i8_disk_clamp_em_2() {
        // Valores fora de ±2.0 devem ser clamped para ±127
        assert_eq!(encode_i8_disk(3.0), 127);
        assert_eq!(encode_i8_disk(-3.0), -127);
    }

    #[test]
    fn bf16_roundtrip_float_simples() {
        // 1.0, 2.0, -1.0 devem sobreviver ao round-trip (são exatamente representáveis)
        for &v in &[1.0_f32, 2.0, -1.0, 0.0, 0.5] {
            let bf = f32_to_bf16(v);
            let back = bf16_to_f32(bf);
            let err = (back - v).abs();
            // BF16 tem 7 bits de mantissa → erro relativo ≤ 2^-7 ≈ 0.0078
            assert!(err <= v.abs() * 0.01 + 1e-7,
                "BF16 round-trip erro {} para v={}", err, v);
        }
    }

    #[test]
    fn snr_db_sinal_uniforme_em_torno_de_48db() {
        // Pesos uniformes em [-2, 2] com 1000 amostras → SNR teórico ~48 dB para
        // ADC de 7 bits (63.5 = 2^6 - 0.5 ≈ 6 bits efetivos + 1 bit sinal):
        // SNR = 6.02 * n_bits + 1.76 ≈ 38-48 dB dependendo da distribuição.
        let n = 1000usize;
        let weights: Vec<f32> = (0..n)
            .map(|i| (i as f32 / (n - 1) as f32) * 4.0 - 2.0) // uniforme [-2, 2]
            .collect();
        let quantized: Vec<i8> = weights.iter().map(|&w| encode_i8_disk(w)).collect();
        let snr = snr_db(&weights, &quantized);
        // Para sinal uniforme [-2, 2] com 127 níveis o SNR esperado é ~42-48 dB
        assert!(snr > 38.0, "SNR para sinal uniforme deve ser > 38 dB, obteve {} dB", snr);
        assert!(snr < 60.0, "SNR suspeito alto: {} dB", snr);
    }

    #[test]
    fn snr_db_comprimentos_diferentes_retorna_zero() {
        let w = [1.0_f32; 5];
        let q = [10i8; 3]; // tamanho diferente
        assert_eq!(snr_db(&w, &q), 0.0);
    }

    #[test]
    fn snr_db_act_sinal_uniforme_em_torno_de_44db() {
        // Ativações uniformes em [-1, 1] (faixa de tanh) com escala 127 (7 bits + sinal):
        // SNR ≈ 6.02*7 + 1.76 ≈ 44 dB, mesma ordem de grandeza de snr_db (63.5, [-2,2]).
        let n = 1000usize;
        let acts: Vec<f32> = (0..n)
            .map(|i| (i as f32 / (n - 1) as f32) * 2.0 - 1.0)
            .collect();
        let snr = snr_db_act(&acts);
        assert!(snr > 38.0, "SNR de ativação uniforme deve ser > 38 dB, obteve {} dB", snr);
        assert!(snr < 60.0, "SNR suspeito alto: {} dB", snr);
    }

    #[test]
    fn snr_db_act_vazio_e_zero() {
        assert_eq!(snr_db_act(&[]), 0.0);
    }

    #[test]
    fn snr_db_act_usa_escala_de_ativacao_nao_de_disco() {
        // v=1.5 está DENTRO da faixa de disco ([-2,2]) mas FORA da faixa de ativação
        // ([-1,1]): encode_i8_act clampa em 127 (equivale a 1.0), então o SNR de
        // ativação para este sinal deve ser bem pior que o de disco — prova que as
        // duas variantes realmente usam escalas diferentes, não a mesma por engano.
        let v = [1.5_f32; 50];
        let snr_act = snr_db_act(&v);
        let q_disk: Vec<i8> = v.iter().map(|&w| encode_i8_disk(w)).collect();
        let snr_disk = snr_db(&v, &q_disk);
        assert!(snr_act < snr_disk,
            "snr_act ({snr_act}) deveria ser pior que snr_disk ({snr_disk}) para v=1.5 (fora da faixa de ativação)");
    }

    #[test]
    fn to_ternary_mapeia_corretamente() {
        let thr = 0.5;
        assert_eq!(to_ternary(0.8, thr), 1);
        assert_eq!(to_ternary(-0.8, thr), -1);
        assert_eq!(to_ternary(0.3, thr), 0);
        assert_eq!(to_ternary(-0.3, thr), 0);
        assert_eq!(to_ternary(0.5, thr), 1);   // borda positiva
        assert_eq!(to_ternary(-0.5, thr), -1); // borda negativa
    }
}
