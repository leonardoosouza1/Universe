#![allow(dead_code)]
#![allow(unused_variables)]
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
