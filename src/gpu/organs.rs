//! Heritable resource and thermal organ capacities.
//!
//! The organ is expressed once at birth, after [`super::body::BodyGenes`] has
//! already expressed the chassis.  Its only runtime field is the cooling gain
//! encoded in the existing `PhysicsStateGpu` padding word; the zero organ keeps
//! the historical state byte-for-byte unchanged.

use super::physics::PhysicsStateGpu;

/// Tank capacity and cooling investment, each bounded to `[-1, 1]`.
///
/// `0` is the legacy organ.  The two scales are independently
/// `1 + 0.25 * gene`.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
pub struct OrganGenes(pub [f32; 2]);

impl OrganGenes {
    /// Reject invalid checkpoints rather than silently changing inherited DNA.
    pub fn is_valid(self) -> bool {
        self.0
            .iter()
            .all(|value| value.is_finite() && (-1.0..=1.0).contains(value))
    }

    /// Resource tank capacity multiplier.
    #[inline]
    pub fn tank_scale(self) -> f32 {
        1.0 + 0.25 * self.0[0]
    }

    /// Engine cooling multiplier.
    #[inline]
    pub fn cooling_scale(self) -> f32 {
        1.0 + 0.25 * self.0[1]
    }

    /// Express the organ over an already expressed body.
    ///
    /// Organ material has an additive cost relative to the historical 1350 kg
    /// chassis: 3% of the tank deviation plus 2% of the cooling deviation.
    /// Suspension stiffness, damping, and the four static wheel loads follow
    /// the resulting mass ratio.  The cooling deviation is stored in the
    /// pre-wind padding word and is consumed only by shader config bit 7.
    pub fn express(self, state: &mut PhysicsStateGpu) {
        assert!(self.is_valid(), "invalid organ genes");
        if self == Self::default() {
            return;
        }

        let old_mass = state.mass;
        let mass_delta =
            1350.0 * (0.03 * (self.tank_scale() - 1.0) + 0.02 * (self.cooling_scale() - 1.0));
        let new_mass = old_mass + mass_delta;
        let mass_ratio = new_mass / old_mass;

        state.mass = new_mass;
        state.susp_stiffness *= mass_ratio;
        state.susp_damping *= mass_ratio;
        for wheel in 0..4 {
            state.wheels[2 * wheel][3] *= mass_ratio;
        }

        // `_pad_before_wind` is a compatibility-preserving transport slot.
        // It stores gain (scale - 1), so zero remains the neutral value.
        state._pad_before_wind = self.cooling_scale() - 1.0;
    }
}
