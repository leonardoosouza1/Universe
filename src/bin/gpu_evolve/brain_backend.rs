//! Adapter for the resident GPU brain implementations used by `gpu_evolve`.
//!
//! `LegacyHybrid` deliberately preserves the historical composition: BrainGpu still
//! runs and Fly is an optional command source layered over it. Fly backends share only
//! the body sensor view and motor-command contract; their recurrent state stays local.

use universe::brain_interface::{MotorCommand, SensorView};
use universe::fly_map::spiking::EyeMode;
use universe::gpu::fly_map::{FlyMapGpu, FlySpikingGpu};

/// Parse an experiment parameter without silently selecting a different arm.
pub fn finite_parameter(
    value: Option<&String>,
    name: &str,
    min: f32,
    max: f32,
) -> Result<f32, String> {
    let parsed = value
        .ok_or_else(|| format!("{name} requires a numeric value"))?
        .parse::<f32>()
        .map_err(|_| format!("{name} requires a numeric value"))?;
    if !parsed.is_finite() || parsed < min || parsed > max {
        return Err(format!("{name} must be finite and in [{min}, {max}]"));
    }
    Ok(parsed)
}

/// A missing or nonfinite brain output invalidates a scientific run.
pub fn validate_commands(commands: &[MotorCommand], expected: usize) -> Result<(), String> {
    if commands.len() != expected {
        return Err(format!(
            "brain returned {} commands for {expected} beings",
            commands.len()
        ));
    }
    if let Some(index) = commands.iter().position(|command| !command.is_finite()) {
        return Err(format!("nonfinite brain output for being {index}"));
    }
    Ok(())
}

/// The active optional brain source. Native `BrainGpu` remains outside this enum so the
/// existing hybrid path and its 96-value intent diagnostics remain bit-compatible.
pub enum BrainBackend<'a> {
    LegacyHybrid,
    FlyRate(&'a FlyMapGpu),
    FlySpiking(&'a FlySpikingGpu),
}

impl<'a> BrainBackend<'a> {
    pub fn name(&self) -> &'static str {
        match self {
            Self::LegacyHybrid => "native",
            Self::FlyRate(_) => "fly-rate",
            Self::FlySpiking(_) => "fly-spiking",
        }
    }
    /// Selects the only valid Fly source; callers enforce mutual exclusion at startup.
    pub fn from_options(rate: Option<&'a FlyMapGpu>, spiking: Option<&'a FlySpikingGpu>) -> Self {
        match (rate, spiking) {
            (Some(rate), None) => Self::FlyRate(rate),
            (None, Some(spiking)) => Self::FlySpiking(spiking),
            (None, None) => Self::LegacyHybrid,
            (Some(_), Some(_)) => panic!("Fly rate and spiking backends are mutually exclusive"),
        }
    }

    /// Records the selected Fly brain after body perception has written its buffers.
    pub fn dispatch(
        &self,
        enc: &mut wgpu::CommandEncoder,
        n: usize,
        sensors: SensorView<'_>,
        eye_on: bool,
        dt: f32,
        seed: u64,
    ) {
        match self {
            Self::LegacyHybrid => {}
            Self::FlyRate(fly) => {
                fly.dispatch(enc, n, sensors.perception_wave, eye_on, sensors.retina)
            }
            Self::FlySpiking(fly) => fly.dispatch(
                enc,
                n,
                sensors.retina,
                eye_on,
                EyeMode::Photoreceptor,
                dt,
                seed,
            ),
        }
    }

    /// Reads commands after the shared encoder has been submitted and polled.
    pub fn read_commands_after_submit(&self, n: usize) -> Vec<MotorCommand> {
        let raw = match self {
            Self::LegacyHybrid => return Vec::new(),
            Self::FlyRate(fly) => fly.read_commands_after_submit(n),
            Self::FlySpiking(fly) => fly.read_commands_after_submit(n),
        };
        raw.into_iter()
            .map(|v| MotorCommand {
                throttle: v[0],
                steer: v[1],
                brake: v[2],
            })
            .collect()
    }
}

/// Converts an optional Fly output into safe direct vehicle inputs.
pub fn direct_motor_command(raw: [f32; 3]) -> MotorCommand {
    let command = MotorCommand {
        throttle: raw[0],
        steer: raw[1],
        brake: raw[2],
    };
    if !command.is_finite() {
        eprintln!("[brain-control] invalid Fly command; applying emergency brake");
        MotorCommand {
            brake: 1.0,
            ..Default::default()
        }
    } else {
        command.clamped()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_or_nonfinite_outputs_invalidate_the_run() {
        assert!(validate_commands(&[], 1).is_err());
        assert!(
            validate_commands(
                &[MotorCommand {
                    steer: f32::NAN,
                    ..Default::default()
                }],
                1
            )
            .is_err()
        );
        assert!(validate_commands(&[MotorCommand::default()], 1).is_ok());
    }

    #[test]
    fn invalid_control_parameters_cannot_silently_change_the_experiment() {
        for value in ["NaN", "inf", "-0.1", "1.1", "wrong"] {
            assert!(finite_parameter(Some(&value.to_string()), "alpha", 0.0, 1.0).is_err());
        }
        assert!(finite_parameter(None, "alpha", 0.0, 1.0).is_err());
        assert_eq!(
            finite_parameter(Some(&"0.02".into()), "alpha", 0.0, 1.0),
            Ok(0.02)
        );
    }

    #[test]
    fn direct_command_clamps_and_fails_safe_on_non_finite() {
        assert_eq!(
            direct_motor_command([2.0, -2.0, 0.5]),
            MotorCommand {
                throttle: 1.0,
                steer: -1.0,
                brake: 0.5
            }
        );
        assert_eq!(direct_motor_command([f32::NAN, 0.0, 0.0]).brake, 1.0);
    }
}
