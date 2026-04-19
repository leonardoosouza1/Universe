/// IARA v2101 - Axiomatic KAN (Kolmogorov-Arnold) Filter
/// Based on Codex v2101 - Science of Spline-based Inference

pub struct KanAxiomFilter {
    /// B-Spline Basis Coefficients: Ultralight 1D tensors
    pub spline_coeffs: [f32; 16],
}

impl KanAxiomFilter {
    pub fn new() -> Self {
        Self {
            spline_coeffs: [1.0; 16], // Neutral Identity at start
        }
    }

    /// Axiomatic Pulse Filtering: "A máquina é fisicamente incapaz do impossível."
    /// This filter uses splines to smooth the 1.58-bit ternary output.
    pub fn filter_pulse(&self, raw_signal: f32) -> f32 {
        // 1. Axiom 1: Velocity Persistence (Zero signals should be ignored)
        if raw_signal.abs() < 0.001 { return 0.0; }

        // 2. Spline Evaluation: Simplified 1D Spline Mapping
        // Map raw_signal [-1.0, 1.0] to spline index [0, 15]
        let idx = ((raw_signal + 1.0) * 7.5).clamp(0.0, 15.0) as usize;
        let c = self.spline_coeffs[idx];
        
        // Final signal is the product of ternary intent and spline coefficient.
        // Result: A smooth, continuous action from a quantized source.
        raw_signal * c
    }

    /// Adaptative Spline Shift: "Learning via Spline Deformation"
    pub fn adapt(&mut self, signal_idx: usize, delta: f32) {
        if signal_idx < 16 {
            self.spline_coeffs[signal_idx] += delta * 0.01;
        }
    }
}

/// Tier 2.2 — KAN per-channel SENSORY PRE-FILTER.
/// Um spline independente por canal de input (128 canais). Cada canal aprende
/// sua curva de resposta ótima (log, saturação, step, sigmoid, linear…).
///
/// Posição no pipeline (ANTES do CfC):
///   raw_input[128] → KAN per-channel → filtered[128]
///                                       ↓ [0..64]
///                                       → CfC (memória temporal)
///                                       ↓
///                                       → BitLinear(128→64)
///
/// Init identity-ish: todos os coeffs = 1.0 → `filter_pulse(x) = x` (mantém
/// comportamento pré-KAN; zero regressão esperada no gen 0).
///
/// RUNTIME-ONLY: não persiste em .dna v3. Coeffs compartilhados no spawn,
/// evoluem via adaptação futura (reward-modulated). Ver
/// melhorias/02-cerebro-dormente.md §2.2.
pub struct SensoryPreFilter {
    pub filters: Vec<KanAxiomFilter>,
}

impl SensoryPreFilter {
    pub fn new(num_channels: usize) -> Self {
        let mut filters = Vec::with_capacity(num_channels);
        for _ in 0..num_channels {
            filters.push(KanAxiomFilter::new());
        }
        Self { filters }
    }

    /// Aplica spline por canal. Mantém quantização ternária i8 no output.
    /// `filter_pulse` já trata zero-signal (axiom 1): |x| < 0.001 → 0.
    pub fn apply(&self, raw_input: &[i8], output: &mut [i8]) {
        let n = raw_input.len().min(output.len()).min(self.filters.len());
        for i in 0..n {
            let raw_f32 = (raw_input[i] as f32) / 127.0;
            let filtered = self.filters[i].filter_pulse(raw_f32);
            output[i] = (filtered * 127.0).clamp(-127.0, 127.0) as i8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_init_preserves_input() {
        let pre = SensoryPreFilter::new(128);
        let mut input = [0i8; 128];
        for i in 0..128 {
            input[i] = ((i as i32 - 64) as i8).saturating_mul(2);
        }
        let mut out = [0i8; 128];
        pre.apply(&input, &mut out);
        for i in 0..128 {
            // coeffs = 1.0 → output ≈ input (±1 por quantização).
            // Zero-signal: |x/127| < 0.001 (i.e. input==0) → 0 (igual ao input==0 de qualquer jeito).
            let diff = (out[i] as i32 - input[i] as i32).abs();
            assert!(diff <= 1, "channel {}: in={} out={} diff={}", i, input[i], out[i], diff);
        }
    }

    #[test]
    fn zero_input_gives_zero() {
        let pre = SensoryPreFilter::new(4);
        let input = [0i8; 4];
        let mut out = [9i8; 4];
        pre.apply(&input, &mut out);
        assert_eq!(out, [0, 0, 0, 0]);
    }
}
