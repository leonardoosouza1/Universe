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

/// KAN output head — per-output B-spline on raw action logits.
/// WHY: substitui clamp duro no throttle/steer/brake (que produz zig-zag ±1 saturado)
/// por curva suave aprendível. Inputs esperados em [-2, 2] (pré-clamp); outputs em [-1, 1].
pub const KAN_OUT_KNOTS: usize = 12;
pub const KAN_OUT_DOMAIN: f32 = 2.0; // domínio [-DOMAIN, +DOMAIN]

#[derive(Clone, Debug)]
pub struct KanOutputHead {
    /// Ordem: [throttle, steer, brake] × KAN_OUT_KNOTS nós cada.
    pub knots: [[f32; KAN_OUT_KNOTS]; 3],
}

impl KanOutputHead {
    /// Default identity-like: knots linearmente espalhados + sigmoid suave no extremo.
    pub fn new() -> Self {
        let mut knots = [[0.0f32; KAN_OUT_KNOTS]; 3];
        for out in 0..3 {
            for k in 0..KAN_OUT_KNOTS {
                // x em [-DOMAIN, +DOMAIN]
                let t = (k as f32) / ((KAN_OUT_KNOTS - 1) as f32); // [0,1]
                let x = -KAN_OUT_DOMAIN + t * (2.0 * KAN_OUT_DOMAIN);
                // sigmoid suave mapeada em [-1, 1]: tanh(x * 0.9) aproxima identidade no meio
                // e satura gradualmente nos extremos — elimina o "degrau" do clamp(±1).
                knots[out][k] = (x * 0.9).tanh();
            }
        }
        Self { knots }
    }

    /// Avalia spline linear-interpolado (B-spline grau 1) no output `out_idx` com input `x`.
    /// Output sempre em [-1, 1].
    #[inline]
    pub fn eval(&self, out_idx: usize, x: f32) -> f32 {
        let out_idx = out_idx.min(2);
        // mapeia x em [-DOMAIN, +DOMAIN] pra índice [0, KAN_OUT_KNOTS-1]
        let xc = x.clamp(-KAN_OUT_DOMAIN, KAN_OUT_DOMAIN);
        let u = (xc + KAN_OUT_DOMAIN) / (2.0 * KAN_OUT_DOMAIN); // [0,1]
        let fpos = u * ((KAN_OUT_KNOTS - 1) as f32);
        let i0 = fpos.floor() as usize;
        let i1 = (i0 + 1).min(KAN_OUT_KNOTS - 1);
        let frac = fpos - (i0 as f32);
        let a = self.knots[out_idx][i0];
        let b = self.knots[out_idx][i1];
        (a + (b - a) * frac).clamp(-1.0, 1.0)
    }

    /// Mutação leve dos knots — chamada pelo spawn.rs quando o filho herda a head.
    /// `rate` = prob de perturbar cada knot; sigma fixo baixo (0.05) preserva suavidade.
    pub fn mutate_spline_knots<R: FnMut() -> f32>(&mut self, mut rng01: R, rate: f32) {
        // v11: sigma 0.05 → 0.025 (metade) pra reduzir drift cumulativo em 200+ gens.
        let sigma = 0.025f32;
        for out in 0..3 {
            for k in 0..KAN_OUT_KNOTS {
                if rng01() < rate {
                    // Box-Muller barato via soma de 4 uniformes (std~0.58), escala por sigma.
                    let u: f32 = (0..4).map(|_| rng01() - 0.5).sum();
                    self.knots[out][k] = (self.knots[out][k] + u * sigma).clamp(-1.0, 1.0);
                }
            }
        }
    }
}

impl Default for KanOutputHead {
    fn default() -> Self { Self::new() }
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
