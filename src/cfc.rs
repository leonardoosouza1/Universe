/// IARA v2101 - CfC (Closed-form Continuous-time) Cell
/// Based on Codex v2101 - Liquid Neural Networks for Temporal Flow
///
/// Tier 2.1: used as a PRE-PROCESSOR of the retina (dim-preserving: 64→64).
/// Provides temporal memory between frames — the organism detects motion
/// ("wall approaching" vs "static wall") instead of reacting only to the
/// current frame.
///
/// Update rule (per channel, in-place):
///   state[i] = state[i] * exp(-decay_rate * dt) + input[i] * gain * dt
///   filtered[i] = tanh(state[i])
///
/// State is CLAMPED to [-clamp_range, +clamp_range] to avoid blow-up
/// (failure mode documented in melhorias/02-cerebro-dormente.md§2.1).

/// FASTEST admitted time constant, in seconds.
///
/// 25 ms ≈ 1.6 ticks at 64 Hz. Below that the channel merely reproduces the integration
/// step and τ stops meaning anything — it is the simulator's Nyquist floor, not a matter
/// of taste.
pub const TAU_MIN_S: f32 = 0.025;
/// SLOWEST admitted time constant, in seconds.
///
/// 4 s is double the single τ the brain used to have, so the range contains the old value
/// with slack on both sides and the control does not sit on an edge.
pub const TAU_MAX_S: f32 = 4.0;

/// Inherited uniform decay rate (s⁻¹): τ = 2.0 s on every channel.
pub const DECAY_DEFAULT: f32 = 0.5;
/// Steady-state gain. 2.0 reproduces the old pair (input_gain 1.0 ÷ decay 0.5).
pub const DC_GAIN_DEFAULT: f32 = 2.0;

/// Draws log-uniform decay rates in [1/TAU_MAX_S, 1/TAU_MIN_S] — the "τ bank".
///
/// Log-uniform, not uniform, because the quantity that matters is the ORDER OF MAGNITUDE
/// of the memory. A uniform draw over `d ∈ [0.25; 40]` would put half the channels below
/// τ = 50 ms: 48 nearly identical channels and no context. In log the range gets covered
/// per decade.
///
/// It is the same initialisation Perez-Nieves et al. (2021, Nature Comms) report as the
/// optimal distribution learned in spiking networks, and the form saci's evolution chose
/// on its own when given freedom ([1, 17, 120, 2000, 20000]).
pub fn log_uniform_decays(seed: u64, n: usize) -> Vec<f32> {
    let mut rng = crate::rng::Lcg::new(seed);
    let lo = (1.0 / TAU_MAX_S).ln();
    let hi = (1.0 / TAU_MIN_S).ln();
    (0..n)
        .map(|_| (lo + rng.uniform() * (hi - lo)).exp())
        .collect()
}

#[derive(Clone)]
pub struct CfCCell {
    pub hidden_size: usize,
    pub state: Vec<f32>,
    /// PER-CHANNEL decay rate, in s⁻¹ — the inverse of the time constant (τ = 1/d).
    ///
    /// It used to be a single scalar. It became a vector because of two of our own
    /// measurements: HAVING varied time scales is selected for (τ bank, 40× against the
    /// control's 5.3×) and in saci evolution chose [1, 17, 120, 2000, 20000] on its own,
    /// paying +2.1pp. Swapping τ with the regime is what does NOT pay (+0.11%) — which is
    /// why τ here is inherited and fixed for the being's life, not modulated in real time.
    ///
    /// What was at stake: with a single τ, the being had 2.0 s of memory on ALL channels,
    /// including wall, curvature and speed. At 15 m/s a 2 s filter delivers to the
    /// steering wheel the perception of 30 m behind — the whole curve. Driving fast was
    /// punished by the being's own brain.
    pub decay_rates: Vec<f32>,
    /// Steady-state gain, EQUAL for every channel, independent of τ.
    ///
    /// Without this the experiment would die from an artefact instead of measuring τ: in
    /// `s = s·e^{-d·dt} + x·g·dt` with fixed `g` the steady state is `x·g/d`, so a
    /// τ=25 ms channel comes out 80× weaker than a 2 s one. Evolution would discard the
    /// fast one for being MUTE, not for being fast. With `g·dt = G·(1 - e^{-d·dt})` the
    /// steady state is `x·G` for any τ, and the only difference between channels becomes
    /// the memory.
    pub dc_gain: f32,
    pub clamp_range: f32,
    /// Denoising noise magnitude (byte-lab transfer: SSM_NOISE=0.15 fixed exposure bias).
    /// Injects small perturbations in state to make creatures robust to self-induced drift.
    pub denoise_mag: f32,
    denoise_counter: u32,
    /// Per-channel adaptive fatigue (spike-frequency adaptation).
    ///
    /// A slow state that rises when |state[i]| is high and is subtracted from the
    /// effective input, breaking positive feedback and avoiding an irreversible fixed
    /// point.
    ///
    /// Equation per channel i:
    ///   fatigue[i] += alpha * (|state[i]| - fatigue[i]) * dt
    ///   x_effective = x - beta * fatigue[i]
    ///
    /// alpha=0.3: τ ≈ 3.3s ≈ 210 ticks @ 64Hz (rises slowly, does not mistake tick noise).
    /// beta=0.4: with fatigue=1.8 it subtracts 0.72 from the [-1,1] input — enough to break saturation.
    pub fatigue: Vec<f32>,
    /// Fatigue alpha parameter (integration speed). Default: 0.3.
    pub fatigue_alpha: f32,
    /// Fatigue beta parameter (subtraction strength). Default: 0.4.
    pub fatigue_beta: f32,
}

impl CfCCell {
    pub fn new(hidden_size: usize) -> Self {
        Self {
            hidden_size,
            state: vec![0.0; hidden_size],
            decay_rates: vec![DECAY_DEFAULT; hidden_size], // τ = 2.0 s, the inherited value
            dc_gain: DC_GAIN_DEFAULT,
            clamp_range: 2.0,
            denoise_mag: 0.10, // byte-lab validated: 15% noise fixes exposure bias
            denoise_counter: 0,
            fatigue: vec![0.0; hidden_size],
            fatigue_alpha: 1.0, // τ ≈ 1s @ 64Hz — rises to 63% of the target in ~64 ticks
            // β=0.4: with the soft shrinkage the fixed point for a sustained maximum input
            // is s = 2(1 - β·s) ⇒ s = 2/(1+2β). With β=0.4 that gives s=1.111 (output
            // 102/127, the channel keeps informing); with β=0.8 it gives s=0.769 (output
            // 82) and the CfC stops passing sustained signal, breaking
            // `update_stabilizes_on_constant_input`. 0.4 is the value this file's
            // documentation always declared — the code had drifted to 0.8 without updating
            // either the comment or the test. It remains a sweep parameter, not a sacred
            // constant.
            fatigue_beta: 0.4,
        }
    }

    /// Resets the temporal state (called on racer spawn/death).
    pub fn reset(&mut self) {
        for s in self.state.iter_mut() {
            *s = 0.0;
        }
        // Zero the fatigue along with the state — a newborn being starts with no
        // accumulated fatigue.
        for f in self.fatigue.iter_mut() {
            *f = 0.0;
        }
        self.denoise_counter = 0; // Deterministic denoising start per creature
    }

    /// Replaces the time constants (one per channel, in s⁻¹).
    ///
    /// Silently ignoring a wrong length would be the recipe for a mute bug — hence the
    /// assert: a short vector would leave half the channels at the old τ and the
    /// "heterogeneous" condition would be half-heterogeneous with nobody noticing.
    pub fn set_decay_rates(&mut self, d: &[f32]) {
        assert_eq!(
            d.len(),
            self.hidden_size,
            "decay_rates must cover every channel"
        );
        self.decay_rates.copy_from_slice(d);
    }

    /// Seed the denoising counter with a unique value per car.
    /// Champion clones with identical weights but different denoise phases
    /// take slightly different paths — truly independent attempts.
    pub fn seed_denoise(&mut self, seed: u32) {
        self.denoise_counter = seed;
    }

    /// Dim-preserving update: takes input i8 [hidden_size], updates state,
    /// writes filtered i8 [hidden_size] = tanh(state) * 127.
    /// Input and output in i8 to respect BitLinear's ternary pipeline.
    ///
    /// Includes adaptive fatigue (spike-frequency adaptation): subtracts
    /// beta*fatigue[i] from the effective input before applying the CfC, breaking
    /// saturation fixed points.
    pub fn update(&mut self, input: &[i8], output: &mut [i8], dt: f32) {
        debug_assert_eq!(input.len(), self.hidden_size);
        debug_assert_eq!(output.len(), self.hidden_size);
        let clamp = self.clamp_range;
        let dc_gain = self.dc_gain;
        // Denoising: cheap deterministic noise via counter-based hash (no RNG needed).
        // Byte-lab discovery: 15% state noise during forward pass fixes exposure bias —
        // creatures become robust to self-induced sensory drift from their own actions.
        self.denoise_counter = self.denoise_counter.wrapping_add(1);
        let noise_mag = self.denoise_mag;
        let alpha = self.fatigue_alpha;
        let beta = self.fatigue_beta;
        for i in 0..self.hidden_size {
            let x_raw = (input[i] as f32) / 127.0;
            // Adaptive fatigue as a SOFT THRESHOLD — it attenuates, never inverts.
            //
            // The previous form was `x = x_raw - beta * fatigue`, a raw subtraction. It
            // breaks the two contracts that define a CfC, and this file's tests caught
            // both:
            //   · with no input at all, the fatigue integrates |state| and starts PUSHING
            //     the state negative — the filter creates signal out of nothing;
            //   · with a sustained +127 the fatigue reaches ~1.8 and subtracts 1.44 from
            //     x=1.0, so a constant positive input produces a NEGATIVE output.
            // The original comment recorded the inversion ("negative effective input") as
            // if it were a feature.
            //
            // Spike-frequency adaptation attenuates the response, it does not flip its
            // sign. `sign(x)·max(0, |x| - beta·fat)` is the soft shrinkage — the same
            // operator as the Prospecting Law and OMP, here on the time axis: only what
            // exceeds the accumulated cost passes.
            let threshold = beta * self.fatigue[i];
            let x = x_raw.signum() * (x_raw.abs() - threshold).max(0.0);
            // Per-channel τ. `gain_dt` comes from the SAME decay to keep DC gain = dc_gain
            // at any τ (see the `dc_gain` doc) — one `exp`, no extra transcendental.
            let decay = (-self.decay_rates[i] * dt).exp();
            let gain_dt = dc_gain * (1.0 - decay);
            let mut s = self.state[i] * decay + x * gain_dt;
            // Inject deterministic noise (~15% of channels per tick, cheap hash).
            if noise_mag > 0.0 {
                let h = self
                    .denoise_counter
                    .wrapping_mul(2654435761)
                    .wrapping_add(i as u32);
                if h % 7 == 0 {
                    // ~14% of channels hit per tick
                    // Pseudo-uniform noise in [-noise_mag, +noise_mag] from hash bits.
                    let noise = ((h >> 8) as f32 / 16777216.0 - 0.5) * 2.0 * noise_mag;
                    s += noise;
                }
            }
            if s > clamp {
                s = clamp;
            } else if s < -clamp {
                s = -clamp;
            }
            // Update the fatigue: a low-pass filter over |state| (spike-frequency
            // adaptation).
            // fat += alpha * (|s| - fat) * dt
            self.fatigue[i] += alpha * (s.abs() - self.fatigue[i]) * dt;
            self.state[i] = s;
            output[i] = Self::quantize_state_i8(s);
        }
    }

    /// Quantizes a hidden-state value to the i8 the downstream layer (L1) consumes:
    /// `(tanh(state) * 127).clamp(-127, 127)`. Extracted out of `update()`'s last line
    /// (lab-ui PRD S2) so a CPU-only reference that already HAS a resident state -- e.g.
    /// `brain_step`, replaying a recorded `neural.trace` 'cfc' sample -- reproduces
    /// exactly the L1 input `update()` would have produced for that state, without
    /// re-deriving the state from a wave (which would require the tau bank and the
    /// fatigue history this function does not have) and without a second copy of the
    /// formula.
    pub fn quantize_state_i8(state: f32) -> i8 {
        (state.tanh() * 127.0).clamp(-127.0, 127.0) as i8
    }

    /// Legacy API (kept so experimental imports do not break).
    ///
    /// Decays each channel by ITS OWN τ — it used to be a single decay for all.
    pub fn flow(&mut self, dt: f32) -> Vec<f32> {
        for (s, &d) in self.state.iter_mut().zip(self.decay_rates.iter()) {
            *s *= (-d * dt).exp();
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
        cfc.denoise_mag = 0.0; // disable noise for deterministic test
        let input = [127i8, 0, -127, 64];
        let mut out = [0i8; 4];
        // 600 steps of 1/60s → 10 virtual seconds, well above the time constant
        // 1/decay=2s.
        for _ in 0..600 {
            cfc.update(&input, &mut out, 1.0 / 60.0);
        }
        // Steady state: s = x * gain / decay_rate = x * 1.0 / 0.5 = 2x, clamped to ±2.
        // tanh(2) ≈ 0.964 → ≈122 for +127, -122 for -127.
        assert!(
            out[0] > 100,
            "sustained +127 input should drive filtered high ({})",
            out[0]
        );
        assert!(
            out[2] < -100,
            "sustained -127 input should drive filtered low ({})",
            out[2]
        );
        assert!(
            out[1].abs() < 5,
            "zero input should stay near zero ({})",
            out[1]
        );
    }

    #[test]
    fn state_decays_to_zero_without_input() {
        let mut cfc = CfCCell::new(2);
        cfc.denoise_mag = 0.0; // disable noise for deterministic test
        cfc.state[0] = 1.5;
        cfc.state[1] = -1.5;
        let input = [0i8; 2];
        let mut out = [0i8; 2];
        for _ in 0..1200 {
            cfc.update(&input, &mut out, 1.0 / 60.0);
        }
        assert!(
            cfc.state[0].abs() < 0.05,
            "state should decay near zero ({})",
            cfc.state[0]
        );
        assert!(
            cfc.state[1].abs() < 0.05,
            "state should decay near zero ({})",
            cfc.state[1]
        );
    }

    #[test]
    fn reset_zeros_state() {
        let mut cfc = CfCCell::new(8);
        for s in cfc.state.iter_mut() {
            *s = 1.23;
        }
        cfc.reset();
        assert!(cfc.state.iter().all(|&s| s == 0.0));
    }
}
