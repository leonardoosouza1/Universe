/// IARA v2101 - CfC (Closed-form Continuous-time) Cell
/// Based on Codex v2101 - Liquid Neural Networks for Temporal Flow
///
/// Tier 2.1: usado como PRE-PROCESSOR da retina (dim-preserving: 64→64).
/// Fornece memória temporal entre frames — organismo detecta movimento
/// ("parede se aproxima" vs "parede estática") ao invés de reagir apenas
/// ao frame atual.
///
/// Update rule (por canal, in-place):
///   state[i] = state[i] * exp(-decay_rate * dt) + input[i] * gain * dt
///   filtered[i] = tanh(state[i])
///
/// State é CLAMPED em [-clamp_range, +clamp_range] pra evitar blow-up
/// (failure mode documentado em melhorias/02-cerebro-dormente.md§2.1).

pub struct CfCCell {
    pub hidden_size: usize,
    pub state: Vec<f32>,
    pub decay_rate: f32,
    pub input_gain: f32,
    pub clamp_range: f32,
}

impl CfCCell {
    pub fn new(hidden_size: usize) -> Self {
        Self {
            hidden_size,
            state: vec![0.0; hidden_size],
            decay_rate: 0.5,     // exp(-0.5 * dt) por tick
            input_gain: 1.0,
            clamp_range: 2.0,
        }
    }

    /// Reseta o estado temporal (chamado em spawn/death do racer).
    pub fn reset(&mut self) {
        for s in self.state.iter_mut() {
            *s = 0.0;
        }
    }

    /// Update dim-preserving: recebe input i8 [hidden_size], atualiza state,
    /// escreve filtered i8 [hidden_size] = tanh(state) * 127.
    /// Input e output em i8 pra respeitar o pipeline ternário do BitLinear.
    pub fn update(&mut self, input: &[i8], output: &mut [i8], dt: f32) {
        debug_assert_eq!(input.len(), self.hidden_size);
        debug_assert_eq!(output.len(), self.hidden_size);
        let decay = (-self.decay_rate * dt).exp();
        let gain_dt = self.input_gain * dt;
        let clamp = self.clamp_range;
        for i in 0..self.hidden_size {
            let x = (input[i] as f32) / 127.0;
            let mut s = self.state[i] * decay + x * gain_dt;
            if s > clamp { s = clamp; } else if s < -clamp { s = -clamp; }
            self.state[i] = s;
            let filtered = s.tanh();
            output[i] = (filtered * 127.0).clamp(-127.0, 127.0) as i8;
        }
    }

    /// Legacy API (mantida pra não quebrar imports experimentais).
    pub fn flow(&mut self, dt: f32) -> Vec<f32> {
        let decay = (-self.decay_rate * dt).exp();
        for s in self.state.iter_mut() {
            *s *= decay;
        }
        self.state.clone()
    }

    pub fn inject(&mut self, input: &[f32]) {
        for (s, &i) in self.state.iter_mut().zip(input.iter()) {
            *s = (*s + i).clamp(-self.clamp_range, self.clamp_range);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_stabilizes_on_constant_input() {
        let mut cfc = CfCCell::new(4);
        let input = [127i8, 0, -127, 64];
        let mut out = [0i8; 4];
        // 600 passos de 1/60s → 10s virtuais, bem acima do time-constant 1/decay=2s.
        for _ in 0..600 {
            cfc.update(&input, &mut out, 1.0 / 60.0);
        }
        // Steady state: s = x * gain / decay_rate = x * 1.0 / 0.5 = 2x, clamped em ±2.
        // tanh(2) ≈ 0.964 → ≈122 para +127, -122 para -127.
        assert!(out[0] > 100, "sustained +127 input should drive filtered high ({})", out[0]);
        assert!(out[2] < -100, "sustained -127 input should drive filtered low ({})", out[2]);
        assert!(out[1].abs() < 5, "zero input should stay near zero ({})", out[1]);
    }

    #[test]
    fn state_decays_to_zero_without_input() {
        let mut cfc = CfCCell::new(2);
        cfc.state[0] = 1.5;
        cfc.state[1] = -1.5;
        let input = [0i8; 2];
        let mut out = [0i8; 2];
        for _ in 0..1200 {
            cfc.update(&input, &mut out, 1.0 / 60.0);
        }
        assert!(cfc.state[0].abs() < 0.05, "state should decay near zero ({})", cfc.state[0]);
        assert!(cfc.state[1].abs() < 0.05, "state should decay near zero ({})", cfc.state[1]);
    }

    #[test]
    fn reset_zeros_state() {
        let mut cfc = CfCCell::new(8);
        for s in cfc.state.iter_mut() { *s = 1.23; }
        cfc.reset();
        assert!(cfc.state.iter().all(|&s| s == 0.0));
    }
}
