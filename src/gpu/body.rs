//! Bounded, heritable vehicle capacities (experimental body schema 1).
//!
//! Development happens once at birth. The resulting values use existing physics
//! inputs; there are no additional GPU buffers or per-tick transfers. These mass
//! and fuel trade-offs are modeling assumptions, not calibrated car specifications.

use super::evolution::Genome;
use super::physics::PhysicsStateGpu;

/// Engine and brake investment, each in [-1, 1]. Zero is the historical body.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
pub struct BodyGenes(pub [f32; 2]);

impl BodyGenes {
    /// Reject invalid checkpoints instead of silently changing inherited DNA.
    pub fn is_valid(self) -> bool {
        self.0
            .iter()
            .all(|v| v.is_finite() && (-1.0..=1.0).contains(v))
    }

    /// Express capacities without touching neural commands or geometry.
    ///
    /// engine = 1 + 0.25*g0; brakes = 1 + 0.25*g1;
    /// mass = base_mass * (1 + 0.15*(engine-1) + 0.05*(brakes-1)).
    /// Suspension stiffness/damping and static wheel loads scale with mass,
    /// preserving rest sag and suspension frequency/damping ratio.
    pub fn express(self, state: &mut PhysicsStateGpu) {
        assert!(self.is_valid(), "invalid body genes");
        if self == Self::default() {
            return; // Preserve every legacy bit, including diagnostic padding.
        }
        let engine = 1.0 + 0.25 * self.0[0];
        let brakes = 1.0 + 0.25 * self.0[1];
        let mass = 1.0 + 0.15 * (engine - 1.0) + 0.05 * (brakes - 1.0);
        state.max_torque *= engine;
        state.brake_torque_max *= brakes;
        state.mass *= mass;
        state.susp_stiffness *= mass;
        state.susp_damping *= mass;
        for wheel in 0..4 {
            state.wheels[2 * wheel][3] *= mass;
        }
    }
}

/// Additional fuel cost relative to the historical engine, applied to load only.
pub fn fuel_load_scale(state: &PhysicsStateGpu) -> f64 {
    f64::from(state.max_torque) / f64::from(super::physics::BASE_ENGINE_TORQUE_NM)
}

/// Mutate inherited bodies after ordinary selection; no draws from the brain RNG.
/// Elites are unchanged. Every child gets small bounded uniform mutations.
pub fn mutate_population(genomes: &mut [Genome], elite: usize, generation: usize, seed: u64) {
    for (index, genome) in genomes.iter_mut().enumerate().skip(elite) {
        let mut state = seed
            ^ (generation as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (index as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
            ^ 0x424F_4459_0000_0001;
        for value in genome.body.0.iter_mut().chain(genome.organs.0.iter_mut()) {
            // SplitMix64: independent stream per offspring and generation.
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let uniform = (z >> 40) as f32 / 16_777_216.0;
            *value = (*value + (2.0 * uniform - 1.0) * 0.1).clamp(-1.0, 1.0);
        }
    }
}

/// Validate the versioned organism DNA block without expressing or simulating it.
pub fn organism_genes_valid(genes: &[f32; 16]) -> bool {
    genes
        .iter()
        .all(|value| value.is_finite() && (-1.0..=1.0).contains(value))
}

/// Opt-in organism mutation on a separate stream. Legacy body/neural draws are unchanged.
pub fn mutate_organism_population(
    genomes: &mut [Genome],
    elite: usize,
    generation: usize,
    seed: u64,
) {
    assert!(
        genomes
            .iter()
            .all(|genome| organism_genes_valid(&genome.organism)),
        "invalid organism DNA"
    );
    for (index, genome) in genomes.iter_mut().enumerate().skip(elite) {
        let mut state = seed
            ^ (generation as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (index as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
            ^ 0x4F52_4741_4E49_534D;
        for value in &mut genome.organism {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let uniform = (z >> 40) as f32 / 16_777_216.0;
            *value = (*value + (2.0 * uniform - 1.0) * 0.1).clamp(-1.0, 1.0);
        }
    }
}

#[cfg(test)]
mod tests;
