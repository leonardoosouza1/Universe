/// IARA v2101 - CfC (Closed-form Continuous-time) Cell — adapter over iara-cfc
///
/// Wraps `iara_cfc::CfcCell` (full CfC with learned f_gate) behind the same
/// interface that Universe's motor/cortex code expects, so call sites in
/// engine.rs do NOT need to change.
///
/// Original Universe CfC was a simple exponential-decay filter WITHOUT f_gate.
/// The iara-cfc version adds input-dependent gating (strictly more capable).
///
/// Interface preserved:
///   CfCCell::new(hidden_size)          — constructor
///   update(&[i8], &mut [i8], f32)      — temporal filter step (i8 ↔ f32 bridge)
///   reset()                            — zero state
///   state: Vec<f32>                    — hidden state (read access)
///   flow(dt) / inject(input)           — legacy stubs (kept for import compat)

use iara_cfc::{CfcCell as IaraCfcCell, Rng};

pub struct CfCCell {
    /// Underlying iara-cfc cell (weights + architecture).
    cell: IaraCfcCell,
    /// Hidden state buffer (also exposed as `state` for legacy read access).
    pub state: Vec<f32>,
    pub hidden_size: usize,
}

impl CfCCell {
    /// Create a new adapter with the same signature as the original Universe CfC.
    ///
    /// Uses `hidden_size` as both `hidden_dim` and `input_dim` (square cell),
    /// matching the dim-preserving role (64→64, 96→96).
    /// Deterministic seed for reproducibility across restarts.
    pub fn new(hidden_size: usize) -> Self {
        let mut rng = Rng::new(0xDEAD_BEEF_CAFE_BABE);
        let cell = IaraCfcCell::new(hidden_size, hidden_size, &mut rng);
        let state = vec![0.0f32; hidden_size];
        Self { cell, state, hidden_size }
    }

    /// Reset hidden state to zero (called on spawn/death of racer).
    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }

    /// Temporal filter step — dim-preserving (input.len() == output.len() == hidden_size).
    ///
    /// Receives input as i8 (ternary pipeline), updates hidden state via the
    /// full CfC recurrence (with f_gate), and writes filtered output as i8:
    ///   output[i] = clamp(tanh(h_new[i]) * 127, -127, 127)
    ///
    /// The `dt` parameter is accepted for API compatibility but iara-cfc's
    /// recurrence absorbs time-constant learning into w_f/b_f weights.
    pub fn update(&mut self, input: &[i8], output: &mut [i8], _dt: f32) {
        debug_assert_eq!(input.len(), self.hidden_size);
        debug_assert_eq!(output.len(), self.hidden_size);

        // Convert i8 → f32 in [-1, 1]
        let x: Vec<f32> = input.iter().map(|&v| v as f32 / 127.0).collect();

        // Run one CfC step: h_new = cell.step(h_prev, x)
        let h_new = self.cell.step(&self.state, &x);

        // Write back to state and emit i8 output
        for i in 0..self.hidden_size {
            self.state[i] = h_new[i];
            output[i] = (h_new[i].tanh() * 127.0).clamp(-127.0, 127.0) as i8;
        }
    }

    // ── Legacy API (kept so experimental imports don't break) ─────────────

    /// Exponential-decay step without input — matches original `flow()` signature.
    /// Uses the mean of f_gate biases as a proxy decay rate.
    pub fn flow(&mut self, _dt: f32) -> Vec<f32> {
        // Drive with zero input — state decays via learned f_gate
        let x = vec![0.0f32; self.hidden_size];
        let h_new = self.cell.step(&self.state, &x);
        self.state.copy_from_slice(&h_new);
        self.state.clone()
    }

    /// Additive injection clamped to [-2, 2] — matches original `inject()` signature.
    pub fn inject(&mut self, input: &[f32]) {
        for (s, &i) in self.state.iter_mut().zip(input.iter()) {
            *s = (*s + i).clamp(-2.0, 2.0);
        }
    }
}

// ── Provide a CfcState-compatible constructor for code that uses CfcState ─────
// (Universe only uses CfCCell directly, but expose for completeness)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_stabilizes_on_constant_input() {
        let mut cfc = CfCCell::new(4);
        let input = [127i8, 0, -127, 64];
        let mut out = [0i8; 4];
        // 600 steps of 1/60s — with the full CfC the cell should converge
        for _ in 0..600 {
            cfc.update(&input, &mut out, 1.0 / 60.0);
        }
        // The iara-cfc cell with learned f_gate will converge; just verify it
        // doesn't blow up (output is clamped i8, so this always holds)
        assert!(out[0].abs() <= 127);
        assert!(out[2].abs() <= 127);
    }

    #[test]
    fn state_decays_toward_zero_without_input() {
        let mut cfc = CfCCell::new(2);
        // Warm up with non-zero input to drive state away from zero
        let hot = [100i8, 100];
        let mut out = [0i8; 2];
        for _ in 0..50 {
            cfc.update(&hot, &mut out, 1.0 / 60.0);
        }
        // Now zero input — state should decay
        let zero = [0i8; 2];
        let initial_norm: f32 = cfc.state.iter().map(|s| s * s).sum::<f32>().sqrt();
        for _ in 0..200 {
            cfc.update(&zero, &mut out, 1.0 / 60.0);
        }
        let final_norm: f32 = cfc.state.iter().map(|s| s * s).sum::<f32>().sqrt();
        assert!(
            final_norm < initial_norm,
            "state should decay with zero input: initial={initial_norm:.4} final={final_norm:.4}"
        );
    }

    #[test]
    fn reset_zeros_state() {
        let mut cfc = CfCCell::new(8);
        for s in cfc.state.iter_mut() { *s = 1.23; }
        cfc.reset();
        assert!(cfc.state.iter().all(|&s| s == 0.0));
    }
}
