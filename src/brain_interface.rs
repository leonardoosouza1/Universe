//! Shared contract for swappable GPU brains.
//!
//! The body exposes resident GPU sensor buffers and every brain returns the same
//! three motor channels. Model-specific state remains owned by its GPU backend.

/// Versioned description of the current body's resident observation buffers.
///
/// This is a vehicle-specific transport schema, rather than a claim about the
/// natural fly body. Retina data has 1,764 packed millimetre/material samples;
/// the perception wave has 192 signed-i8 channels packed into GPU words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObservationSchema {
    pub retina_samples: usize,
    pub retina_encoding: &'static str,
    pub wave_channels: usize,
    pub wave_encoding: &'static str,
}

/// Observation contract used by the current GPU body (schema version 1).
pub const OBSERVATION_SCHEMA_V1: ObservationSchema = ObservationSchema {
    retina_samples: crate::config::iara::vision::RETINA_TOTAL,
    retina_encoding: "packed_mm_material",
    wave_channels: crate::config::iara::brain::SENSORY_WAVE_SIZE,
    wave_encoding: "signed_i8_packed",
};

/// GPU sensor buffers produced by the body before a brain dispatch.
///
/// The buffers remain resident on the GPU. `dt` is deliberately a separate
/// dispatch argument in each backend and is expressed in seconds.
pub struct SensorView<'a> {
    pub perception_wave: &'a wgpu::Buffer,
    pub retina: &'a wgpu::Buffer,
}

/// The three normalized motor channels consumed by this vehicle's physics.
///
/// These names and ranges describe the current body adapter; they do not claim
/// to be the actuator dynamics of an actual fly body.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MotorCommand {
    pub throttle: f32,
    pub steer: f32,
    pub brake: f32,
}

impl MotorCommand {
    /// Converts the command to the legacy `[throttle, steer, brake]` layout.
    pub fn as_array(self) -> [f32; 3] {
        [self.throttle, self.steer, self.brake]
    }

    /// Rejects non-finite values before they enter vehicle physics.
    pub fn is_finite(self) -> bool {
        self.throttle.is_finite() && self.steer.is_finite() && self.brake.is_finite()
    }

    /// Clamps a finite command to the vehicle's physical input range.
    pub fn clamped(self) -> Self {
        Self {
            throttle: self.throttle.clamp(0.0, 1.0),
            steer: self.steer.clamp(-1.0, 1.0),
            brake: self.brake.clamp(0.0, 1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motor_command_contract_preserves_order_and_rejects_non_finite() {
        let command = MotorCommand {
            throttle: 0.25,
            steer: -0.5,
            brake: 1.0,
        };
        assert_eq!(command.as_array(), [0.25, -0.5, 1.0]);
        assert!(command.is_finite());
        assert!(
            !MotorCommand {
                steer: f32::NAN,
                ..command
            }
            .is_finite()
        );
    }

    #[test]
    fn observation_schema_v1_pins_resident_shapes() {
        assert_eq!(OBSERVATION_SCHEMA_V1.retina_samples, 1_764);
        assert_eq!(OBSERVATION_SCHEMA_V1.wave_channels, 192);
        assert_eq!(OBSERVATION_SCHEMA_V1.retina_encoding, "packed_mm_material");
        assert_eq!(OBSERVATION_SCHEMA_V1.wave_encoding, "signed_i8_packed");
    }
}
