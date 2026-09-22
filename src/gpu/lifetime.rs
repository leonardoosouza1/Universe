//! Heritable controls and resident layout for optional lifetime adaptation.
//!
//! The brain kernel keeps this state on the device.  The default genes are useful
//! when the feature is explicitly enabled, while the feature flags remain zero in
//! the normal path so existing brains retain their exact equations.

/// Number of f32 values carried by the lifetime genome block.
pub const N_LIFE_GENES: usize = 16;

/// Resident lifetime state stride, in f32 values, for one being.
pub const LIFETIME_STRIDE: usize = 512;

pub const LIFE_CONFIG_SLOTS: usize = 32;
pub const LIFE_FEEDBACK_OFFSET: usize = 32;
pub const LIFE_STATE_P_OFFSET: usize = 36;
pub const LIFE_STATE_E_OFFSET: usize = 180;
pub const LIFE_PREV_LATENT_OFFSET: usize = 324;
pub const LIFE_PREV_MOTOR_OFFSET: usize = 372;
pub const LIFE_RUNTIME_OFFSET: usize = 375;
/// Money baseline of the online feedback adapter (organism_feedback.wgsl); slots
/// 387..512 are otherwise unused by the brain kernel.
pub const LIFE_AUX_MONEY_OFFSET: usize = 400;
/// Task-progress (body 127) and task-key (body 31) baselines of the same adapter.
pub const LIFE_AUX_PROGRESS_OFFSET: usize = 401;
pub const LIFE_AUX_TASK_KEY_OFFSET: usize = 402;
// Runtime summary slots relative to LIFE_RUNTIME_OFFSET:
// age, world_tick, reward_ema, energy_ema, alive_seconds, update_count,
// l1_capacity_last, l2_capacity_last, elapsed_dt, residual_motor[3].

// Gene slots.  The age values are normalised developmental age in [0, 1], and
// development_rate advances that value in seconds of elapsed brain time.
pub const LIFE_PLASTICITY_RATE: usize = 0;
pub const LIFE_ELIGIBILITY_DECAY: usize = 1;
pub const LIFE_PLASTICITY_CLAMP: usize = 2;
pub const LIFE_RESIDUAL_GAIN: usize = 3;
pub const LIFE_L1_AGE_START: usize = 4;
pub const LIFE_L1_AGE_END: usize = 5;
pub const LIFE_L1_ENERGY_MIN: usize = 6;
pub const LIFE_L1_DENSITY: usize = 7;
pub const LIFE_L2_AGE_START: usize = 8;
pub const LIFE_L2_AGE_END: usize = 9;
pub const LIFE_L2_ENERGY_MIN: usize = 10;
pub const LIFE_L2_DENSITY: usize = 11;
pub const LIFE_REWARD_GAIN: usize = 12;
pub const LIFE_WEIGHT_DECAY: usize = 13;
pub const LIFE_DEVELOPMENT_RATE: usize = 14;
pub const LIFE_ENERGY_EXPONENT: usize = 15;

/// Heritable lifetime controls.  This is a positional block in the v4 genome;
/// changing its order requires a genome version bump.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LifeGenes(pub [f32; N_LIFE_GENES]);

impl LifeGenes {
    /// Conservative bounds used by `is_valid`, mutation and GPU sanitisation.
    pub const MIN: [f32; N_LIFE_GENES] = [
        0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    pub const MAX: [f32; N_LIFE_GENES] = [
        1.0, 16.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0,
    ];

    /// A useful enabled configuration: modest residual learning, full initial
    /// topology capacity, and a ten-second developmental ramp.
    pub const fn default_values() -> Self {
        Self([
            0.05, // plasticity rate (per second)
            2.0,  // eligibility decay (per second)
            0.25, // absolute plasticity clamp
            0.10, // residual motor gain
            0.0,  // L1 developmental start
            1.0,  // L1 developmental end
            0.0,  // L1 minimum energy fraction
            1.0,  // L1 edge density
            0.0,  // L2 developmental start
            1.0,  // L2 developmental end
            0.0,  // L2 minimum energy fraction
            1.0,  // L2 edge density
            1.0,  // reward modulation gain
            0.01, // acquired-weight decay per second
            0.1,  // developmental age per second
            1.0,  // energy exponent
        ])
    }

    /// Returns whether every value is finite and within its schema bounds.
    pub fn is_valid(&self) -> bool {
        self.0
            .iter()
            .enumerate()
            .all(|(i, value)| value.is_finite() && *value >= Self::MIN[i] && *value <= Self::MAX[i])
            && self.0[LIFE_L1_AGE_START] <= self.0[LIFE_L1_AGE_END]
            && self.0[LIFE_L2_AGE_START] <= self.0[LIFE_L2_AGE_END]
    }

    /// Deterministically mutates this block using a private SplitMix-style stream.
    /// `flags` selects the independently evolvable subsets: bit 0 plasticity,
    /// bit 1 development, bit 2 topology. It does not touch the simulator RNG.
    pub fn mutated(self, seed: u64, flags: u32) -> Self {
        let mut state = seed;
        let mut values = self.0;
        for (i, value) in values.iter_mut().enumerate() {
            let active = match i {
                0..=3 | 12..=13 => (flags & 1) != 0,
                4..=6 | 8..=10 | 14..=15 => (flags & 2) != 0,
                7 | 11 => (flags & 4) != 0,
                _ => false,
            };
            if !active {
                continue;
            }
            state = state.wrapping_add(0x9e3779b97f4a7c15).rotate_left(17) ^ 0xbf58476d1ce4e5b9;
            let mixed = state.wrapping_mul(0x94d049bb133111eb).rotate_left(31) ^ (state >> 29);
            let unit = ((mixed >> 11) as f64 / ((1u64 << 53) as f64)) as f32;
            let span = Self::MAX[i] - Self::MIN[i];
            let delta = (unit * 2.0 - 1.0) * span * 0.05;
            *value = (*value + delta).clamp(Self::MIN[i], Self::MAX[i]);
        }
        // Keep the developmental windows ordered after independent mutation.
        if values[LIFE_L1_AGE_START] > values[LIFE_L1_AGE_END] {
            values.swap(LIFE_L1_AGE_START, LIFE_L1_AGE_END);
        }
        if values[LIFE_L2_AGE_START] > values[LIFE_L2_AGE_END] {
            values.swap(LIFE_L2_AGE_START, LIFE_L2_AGE_END);
        }
        Self(values)
    }

    /// Sanitises the genes into the fixed 32-f32 config region.  The second half
    /// mirrors the first half so future shader revisions can consume derived fields
    /// without changing the resident stride.
    pub fn config_slots(&self) -> [f32; LIFE_CONFIG_SLOTS] {
        let mut config = [0.0; LIFE_CONFIG_SLOTS];
        for i in 0..N_LIFE_GENES {
            config[i] = self.0[i].clamp(Self::MIN[i], Self::MAX[i]);
            config[N_LIFE_GENES + i] = config[i];
        }
        config
    }
}

impl Default for LifeGenes {
    fn default() -> Self {
        Self::default_values()
    }
}
