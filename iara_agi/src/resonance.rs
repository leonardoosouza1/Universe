#![allow(dead_code)]
#![allow(unused_variables)]
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct ResonantNode {
    pub phase: f32, // Fase interna da onda
    pub frequency: f32,
    pub coupling: f32,
}

impl ResonantNode {
    pub fn new(freq: f32) -> Self {
        Self {
            phase: 0.0,
            frequency: freq,
            coupling: 0.1,
        }
    }

    pub fn synchronize(&mut self, target_phase: f32, delta_t: f32) {
        // Modelo de Kuramoto simplificado: a fase tenta se alinhar ao alvo
        let diff = (target_phase - self.phase).sin();
        self.phase += (self.frequency + self.coupling * diff) * delta_t;
        self.phase %= 2.0 * std::f32::consts::PI;
    }

    pub fn get_coherence(&self, target_phase: f32) -> f32 {
        // Retorna o quão próxima a fase interna está do alvo (0.0 a 1.0)
        (self.phase - target_phase).cos().abs()
    }
}
