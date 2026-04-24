//! Cérebros Especialistas — 3 BitLinear pequenos (32→1) coordenados.
//!
//! Inspirado no SwarmBrain do SACI-Trader (5 CfC workers + SwarmCoordinator).
//! Ver `melhorias/specialist-brains.md`.
//!
//! Status: **skeleton**. Ainda não conectado ao `SovereignEngine`. A integração
//! runtime exige:
//!   1) Roteamento de sub-slices de `sensory_wave` por cérebro (masks por canal).
//!   2) Refactor de `spawn.rs::build_priming_weights` pra gerar priming por brain.
//!   3) DNA v4 codec: salvar 3 blobs ternários + 3 scales.
//!   4) Crossover semântico (pai A → Steer, pai B → Brake).
//!
//! Este módulo existe pra bloquear o API e permitir micro-benchmark comparativo
//! com o monolítico 128→64 atual.

use crate::layers::{BitLinear, BrainForward};

/// Máscara de canais de input por especialista (32 ints, cada 0..127).
/// Define QUAIS canais do sensory_wave[128] vão pra cada cérebro.
/// Escolhidos a partir do mapa de canais em `motor.rs` e do priming em `spawn.rs`.
pub const STEER_CHANNELS: [usize; 32] = [
    // fóvea center (3,4) — retina frontal
    3, 4,
    // periférica bins (8..16)
    8, 9, 10, 11, 12, 13, 14, 15,
    // deriva, curvatura, slip, whiskers
    66, 78, 79, 80, 87,
    94, 95, 96, 97,
    // lane_offset, fluxo óptico, memória parede
    81, 83, 84, 85, 86,
    // gaze tracker + dark center (4 duplicatas cada)
    90, 91, 92, 93, 98, 99, 100, 101,
];

pub const THROTTLE_CHANNELS: [usize; 32] = [
    // fluxo (64), fóvea (3,4), road-hints (67, 76, 77)
    64, 65, 3, 4, 67, 76, 77,
    // grass, comp-stress, TTC, prox sustentada
    73, 125, 82, 88,
    // velocidade (65), speed_limit (74)
    74,
    // periférica para gatear em ambiente fechado
    8, 11, 12, 15,
    // memória, curvatura
    85, 86, 78, 79, 80,
    // whiskers próximos (corta throttle)
    94, 95, 96, 97,
    // retina fóvea bins 0..1 (extremos)
    0, 1, 2, 5,
    // padding canais de telemetria
    68, 69, 70,
];

pub const BRAKE_CHANNELS: [usize; 32] = [
    // TTC (82) — sinal primário
    82,
    // whiskers (94..97)
    94, 95, 96, 97,
    // prox sustentada, memória parede
    88, 85, 86,
    // fóvea centro (parede iminente)
    3, 4,
    // periférica center
    11, 12,
    // slip, curvatura (pre-brake em curva)
    87, 78, 79, 80,
    // velocidade
    65,
    // retina bins
    0, 1, 2, 5, 6, 7,
    // whiskers extras e mem
    67, 73,
    // padding
    68, 69, 70, 71, 72, 83, 84,
];

/// Gate (coordinator) — 3 inputs (outputs dos 3 cérebros) + contexto slow → 1 peso cada.
/// Papel: modulação (não substituição). Similar a SACI SwarmCoordinator.policy_head.
pub const GATE_DIM: usize = 8;

/// Swarm de 3 especialistas.
pub struct SpecialistBrain {
    pub steer: BitLinear,    // 32 → 1
    pub throttle: BitLinear, // 32 → 1
    pub brake: BitLinear,    // 32 → 1
}

impl SpecialistBrain {
    pub fn new() -> Self {
        Self {
            steer: BitLinear::new(32, 1),
            throttle: BitLinear::new(32, 1),
            brake: BitLinear::new(32, 1),
        }
    }

    /// Extrai sub-slice de 32 canais a partir do sensory_wave completo.
    fn gather(input: &[i8], channels: &[usize; 32]) -> [i8; 32] {
        let mut out = [0i8; 32];
        for (i, &ch) in channels.iter().enumerate() {
            if ch < input.len() {
                out[i] = input[ch];
            }
        }
        out
    }

    /// Forward unificado. Output compatível com BitLinear 128→64 atual
    /// nas 3 primeiras saídas: [throttle, steer, brake].
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
        self.steer.myelinate(factor);
        self.throttle.myelinate(factor);
        self.brake.myelinate(factor);
    }
    fn quantize(&mut self, _latent: &[f32]) {
        // Priming por-cérebro acontece em `spawn.rs` (futuro).
        // Quantize global não faz sentido aqui.
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
