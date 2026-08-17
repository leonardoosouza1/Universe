//! Specialist Brains — 3 small coordinated BitLinear (32→1).
//!
//! Inspired by SACI-Trader's SwarmBrain (5 CfC workers + SwarmCoordinator).
//! See `melhorias/specialist-brains.md`.
//!
//! Status: **skeleton**. Not yet connected to `SovereignEngine`. Runtime
//! integration requires:
//!   1) Routing of `sensory_wave` sub-slices per brain (per-channel masks).
//!   2) Refactor of `spawn.rs::build_priming_weights` to generate priming per brain.
//!   3) DNA v4 codec: save 3 ternary blobs + 3 scales.
//!   4) Semantic crossover (parent A → Steer, parent B → Brake).
//!
//! This module exists to lock down the API and allow a comparative micro-benchmark
//! against the current 128→64 monolith.

use crate::float_layers::FloatLinear;
use crate::layers::BrainForward;

/// Input channel mask per specialist (32 ints, each 0..127).
/// Defines WHICH channels of sensory_wave[128] go to each brain.
/// Chosen from the channel map in `motor.rs` and the priming in `spawn.rs`.
pub const STEER_CHANNELS: [usize; 32] = [
    // fovea center (3,4) — frontal retina
    3, 4, // peripheral bins (8..16)
    8, 9, 10, 11, 12, 13, 14, 15, // drift, curvature, slip, whiskers
    66, 78, 79, 80, 87, 94, 95, 96, 97, // lane_offset, optical flow, wall memory
    81, 83, 84, 85, 86, // gaze tracker + dark center (4 duplicates each)
    90, 91, 92, 93, 98, 99, 100, 101,
];

pub const THROTTLE_CHANNELS: [usize; 32] = [
    // flow (64), fovea (3,4), road-hints (67, 76, 77)
    64, 65, 3, 4, 67, 76, 77, // grass, comp-stress, TTC, sustained prox
    73, 125, 82, 88, // speed (65), speed_limit (74)
    74, // peripheral to gate in enclosed environments
    8, 11, 12, 15, // memory, curvature
    85, 86, 78, 79, 80, // near whiskers (cuts throttle)
    94, 95, 96, 97, // retina fovea bins 0..1 (extremes)
    0, 1, 2, 5, // padding telemetry channels
    68, 69, 70,
];

pub const BRAKE_CHANNELS: [usize; 32] = [
    // TTC (82) — primary signal
    82, // whiskers (94..97)
    94, 95, 96, 97, // sustained prox, wall memory
    88, 85, 86, // fovea center (imminent wall)
    3, 4, // peripheral center
    11, 12, // slip, curvature (pre-brake in curve)
    87, 78, 79, 80, // speed
    65, // retina bins
    0, 1, 2, 5, 6, 7, // extra whiskers and mem
    67, 73, // padding
    68, 69, 70, 71, 72, 83, 84,
];

/// Gate (coordinator) — 3 inputs (outputs of the 3 brains) + slow context → 1 weight each.
/// Role: modulation (not replacement). Similar to SACI SwarmCoordinator.policy_head.
pub const GATE_DIM: usize = 8;

/// Swarm of 3 specialists. F32 runtime (FloatLinear) — smooth gradient allows
/// AND composition ("fast AND about to turn = brake"). Ternary quantize only on
/// disk (TRIN DNA chunk = i8×96, dequantize on load). Decision: f32 where a
/// continuum is needed, ternary where a discrete trilemma is needed (future genes).
pub struct SpecialistBrain {
    pub steer: FloatLinear,    // 32 → 1
    pub throttle: FloatLinear, // 32 → 1
    pub brake: FloatLinear,    // 32 → 1
}

impl SpecialistBrain {
    pub fn new() -> Self {
        Self {
            steer: FloatLinear::new(32, 1),
            throttle: FloatLinear::new(32, 1),
            brake: FloatLinear::new(32, 1),
        }
    }

    /// Extracts a 32-channel sub-slice from the full sensory_wave.
    fn gather(input: &[i8], channels: &[usize; 32]) -> [i8; 32] {
        let mut out = [0i8; 32];
        for (i, &ch) in channels.iter().enumerate() {
            if ch < input.len() {
                out[i] = input[ch];
            }
        }
        out
    }

    /// Unified forward. Output compatible with the current BitLinear 128→64
    /// in the first 3 outputs: [throttle, steer, brake].
    pub fn forward_specialist(&self, input: &[i8]) -> [f32; 3] {
        let s_in = Self::gather(input, &STEER_CHANNELS);
        let t_in = Self::gather(input, &THROTTLE_CHANNELS);
        let b_in = Self::gather(input, &BRAKE_CHANNELS);
        let throttle = self.throttle.forward(&t_in)[0];
        let steer = self.steer.forward(&s_in)[0];
        let brake = self.brake.forward(&b_in)[0];
        [throttle, steer, brake]
    }
}

impl BrainForward for SpecialistBrain {
    fn forward_i8(&self, input: &[i8]) -> Vec<f32> {
        self.forward_specialist(input).to_vec()
    }
    fn myelinate(&mut self, factor: f32) {
        // F32 runtime: scale weights uniformly (analog to BitLinear scale bump).
        let f = factor.clamp(0.5, 2.0);
        for w in self.steer.weights.iter_mut() {
            *w *= f;
        }
        for w in self.throttle.weights.iter_mut() {
            *w *= f;
        }
        for w in self.brake.weights.iter_mut() {
            *w *= f;
        }
    }
    fn quantize(&mut self, _latent: &[f32]) {
        // No-op: f32 runtime; quantize happens at DNA save boundary.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_copies_correct_channels() {
        let input: Vec<i8> = (0..128).map(|i| i as i8).collect();
        let got = SpecialistBrain::gather(&input, &STEER_CHANNELS);
        assert_eq!(got[0], STEER_CHANNELS[0] as i8);
        assert_eq!(got[31], STEER_CHANNELS[31] as i8);
    }

    #[test]
    fn forward_produces_three_outputs() {
        let brain = SpecialistBrain::new();
        let input = vec![0i8; 128];
        let out = brain.forward_specialist(&input);
        assert_eq!(out.len(), 3);
        for v in out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn channels_unique_per_brain() {
        let s: std::collections::HashSet<_> = STEER_CHANNELS.iter().collect();
        assert_eq!(s.len(), STEER_CHANNELS.len(), "steer has duplicates");
        let t: std::collections::HashSet<_> = THROTTLE_CHANNELS.iter().collect();
        assert_eq!(t.len(), THROTTLE_CHANNELS.len(), "throttle has duplicates");
        let b: std::collections::HashSet<_> = BRAKE_CHANNELS.iter().collect();
        assert_eq!(b.len(), BRAKE_CHANNELS.len(), "brake has duplicates");
    }
}
