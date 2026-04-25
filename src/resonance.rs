/// IARA v2101 - Harmonic Resonance (Kuramoto Model)
/// Based on Codex v2101 - Module V

use std::f32::consts::PI;

pub struct ResonantNode {
    /// Natural frequency (omega)
    pub natural_freq: f32,
    /// Current phase (theta) between -PI and PI
    pub phase: f32,
    /// Coupling strength (K)
    pub coupling: f32,
}

impl ResonantNode {
    pub fn new(freq: f32) -> Self {
        Self {
            natural_freq: freq,
            phase: 0.0,
            coupling: 0.1,
        }
    }

    /// Kuramoto Update: d_theta/dt = omega + K/N * Sum(sin(theta_j - theta_i))
    /// Simplified for synchronization with an external signal (Acorde de Fase).
    pub fn synchronize(&mut self, external_phase: f32, delta_t: f32) {
        let phase_diff = external_phase - self.phase;
        
        // Sincronia de Fase (Phase-Locked Loop)
        let d_theta = self.natural_freq + self.coupling * phase_diff.sin();
        
        self.phase = (self.phase + d_theta * delta_t) % (2.0 * PI);
        if self.phase > PI { self.phase -= 2.0 * PI; }
        if self.phase < -PI { self.phase += 2.0 * PI; }
    }

    /// Coherence Check: Returns 1.0 if perfectly in sync with the phase
    pub fn get_coherence(&self, target_phase: f32) -> f32 {
        (target_phase - self.phase).cos()
    }
}

/// Kuramoto order parameter: |Σ e^iφ_k| / N. Returns [0, 1].
/// 1.0 = all in phase. 0.0 = uniformly distributed phases (chaotic).
///
/// Returns `1.0` for empty input (calm baseline). Caller is responsible for
/// ensuring oscillators are initialized — `debug_assert!` catches misuse in
/// dev/test, release builds stay lenient.
pub fn coherence(nodes: &[ResonantNode]) -> f32 {
    debug_assert!(!nodes.is_empty(), "coherence called on empty oscillator set");
    if nodes.is_empty() { return 1.0; }
    let n = nodes.len() as f32;
    let (sin_sum, cos_sum) = nodes.iter().fold((0.0_f32, 0.0_f32), |(s, c), node| {
        (s + node.phase.sin(), c + node.phase.cos())
    });
    ((sin_sum * sin_sum + cos_sum * cos_sum).sqrt() / n).min(1.0)
}

#[cfg(test)]
mod coherence_tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn coherence_full_when_aligned() {
        let mut nodes: Vec<ResonantNode> = (0..4).map(|_| ResonantNode::new(1.0)).collect();
        for n in &mut nodes { n.phase = 0.5; }
        assert!((coherence(&nodes) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn coherence_zero_when_uniform() {
        let mut nodes: Vec<ResonantNode> = (0..4).map(|_| ResonantNode::new(1.0)).collect();
        for (i, n) in nodes.iter_mut().enumerate() {
            n.phase = (i as f32) * (2.0 * PI / 4.0);
        }
        assert!(coherence(&nodes) < 0.05);
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn coherence_empty_returns_one() {
        assert_eq!(coherence(&[]), 1.0);
    }
}
