//! Validated domain values at the brain/body boundary.
//!
//! GPU structs remain compact transport records. Construct these values at the
//! boundary, then explicitly unpack them; do not silently clamp malformed input.

use std::fmt;

/// A field violated a finite-value, unit, or domain contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContractError {
    pub field: &'static str,
    pub expected: &'static str,
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "contract {}: expected {}", self.field, self.expected)
    }
}
impl std::error::Error for ContractError {}

/// Reject NaN and infinity without altering finite values.
pub fn finite(value: f32, field: &'static str) -> Result<f32, ContractError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ContractError {
            field,
            expected: "finite f32",
        })
    }
}

/// Validate a strictly positive physical parameter.
pub fn positive(value: f32, field: &'static str) -> Result<f32, ContractError> {
    finite(value, field)?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err(ContractError {
            field,
            expected: "positive f32",
        })
    }
}

macro_rules! quantity {
    ($name:ident, $unit:literal, $min:expr, $max:expr) => {
        #[doc = concat!("Validated value in ", $unit, ".")]
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $name(f32);
        impl $name {
            /// Validate a value already expressed in the canonical unit.
            pub fn new(value: f32) -> Result<Self, ContractError> {
                finite(value, stringify!($name))?;
                if value < $min || value > $max {
                    return Err(ContractError {
                        field: stringify!($name),
                        expected: concat!("valid domain in ", $unit),
                    });
                }
                Ok(Self(value))
            }
            /// Import only the explicitly declared canonical unit.
            pub fn from_unit(value: f32, unit: &str) -> Result<Self, ContractError> {
                if unit != $unit {
                    return Err(ContractError {
                        field: stringify!($name),
                        expected: $unit,
                    });
                }
                Self::new(value)
            }
            /// Unpack the canonical value for the declared GPU layout.
            pub fn get(self) -> f32 {
                self.0
            }
        }
    };
}

quantity!(SpeedMps, "m/s", 0.0, f32::MAX);
quantity!(Radians, "rad", -f32::MAX, f32::MAX);
quantity!(TemperatureC, "degC", -273.15, f32::MAX);
quantity!(Throttle, "fraction", 0.0, 1.0);
quantity!(Brake, "fraction", 0.0, 1.0);
quantity!(Steering, "signed_fraction", -1.0, 1.0);

impl SpeedMps {
    /// Explicitly convert km/h into m/s.
    pub fn from_kmh(value: f32) -> Result<Self, ContractError> {
        finite(value, "speed_kmh")?;
        Self::new(value / 3.6)
    }
}
impl Radians {
    /// Explicitly convert degrees into radians.
    pub fn from_degrees(value: f32) -> Result<Self, ContractError> {
        finite(value, "angle_degrees")?;
        Self::new(value.to_radians())
    }
}

/// A validated simultaneous motor command, with named semantic channels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VehicleCommand {
    throttle: Throttle,
    steering: Steering,
    brake: Brake,
}
impl VehicleCommand {
    /// Validate all channels before constructing a command.
    pub fn new(throttle: f32, steering: f32, brake: f32) -> Result<Self, ContractError> {
        Ok(Self {
            throttle: Throttle::new(throttle)?,
            steering: Steering::new(steering)?,
            brake: Brake::new(brake)?,
        })
    }
    /// Preserve the established throttle/steering/brake buffer order.
    pub fn as_array(self) -> [f32; 3] {
        [self.throttle.get(), self.steering.get(), self.brake.get()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn units_require_explicit_conversion() {
        assert!(SpeedMps::from_unit(36.0, "km/h").is_err());
        assert_eq!(SpeedMps::from_kmh(36.0).unwrap().get(), 10.0);
        assert!(Radians::from_unit(180.0, "deg").is_err());
        assert_eq!(
            Radians::from_degrees(180.0).unwrap().get(),
            std::f32::consts::PI
        );
        assert!(TemperatureC::from_unit(300.0, "K").is_err());
    }

    #[test]
    fn invalid_motor_commands_are_rejected_without_clamping() {
        for invalid in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            assert!(VehicleCommand::new(invalid, 0.0, 0.0).is_err());
            assert!(VehicleCommand::new(0.0, 0.0, invalid).is_err());
        }
        assert!(VehicleCommand::new(0.0, -1.1, 0.0).is_err());
        assert_eq!(
            VehicleCommand::new(0.25, -0.5, 0.75).unwrap().as_array(),
            [0.25, -0.5, 0.75]
        );
        assert!(TemperatureC::new(-274.0).is_err());
        assert!(SpeedMps::new(-1.0).is_err());
    }
}
