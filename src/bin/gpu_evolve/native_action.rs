//! Pure native motor-command mapping and diagnostic trace contracts.
//!
//! Keeping this transformation outside the GPU tick makes the legacy command math
//! directly testable and lets diagnostics name where a brake command disappeared.

use serde::Serialize;
use universe::kan::KAN_OUT_KNOTS;

/// Selects the final native output transfer function without changing brain weights.
#[derive(Clone, Copy)]
pub enum MotorHead<'a> {
    /// Historical global output spline used by every existing checkpoint.
    Spline(&'a [[f32; KAN_OUT_KNOTS]; 3]),
    /// Diagnostic identity transfer over the already bounded native intent.
    Linear,
}

/// Selects whether negative native throttle can oppose positive modulation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ThrottleArbitration {
    /// Historical ordering: floor the head at zero before adding slow/hip signals.
    Legacy,
    /// Diagnostic ordering: add signed signals, then clamp the combined command.
    #[default]
    Signed,
}

impl ThrottleArbitration {
    fn name(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Signed => "signed",
        }
    }
}

impl MotorHead<'_> {
    fn name(self) -> &'static str {
        match self {
            Self::Spline(_) => "spline",
            Self::Linear => "linear",
        }
    }

    fn map(self, output: usize, raw: f32) -> f32 {
        match self {
            Self::Spline(knots) => universe::kan::eval_knots(knots, output, raw),
            Self::Linear => raw.clamp(-1.0, 1.0),
        }
    }
}

/// Inputs to the behavior-compatible native action mapping, all from one tick.
pub struct NativeActionInput<'a> {
    pub raw_intent: [f32; 3],
    pub slow_output: [f32; 3],
    pub hip_bias: [f32; 3],
    pub previous_steer: f32,
    pub speed_mps: f32,
    pub steering_alpha: f32,
    pub slow_steering_authority: f32,
    pub head: MotorHead<'a>,
    pub throttle_arbitration: ThrottleArbitration,
}

/// Authority of the slow inhibitory pathway over the brake pedal.
///
/// Slow output 1 already represents motor inhibition: positive values lower the
/// throttle ceiling. Giving that same learned signal a brake path closes the missing
/// perception-to-action route without encoding a curve rule in the body. The brain still
/// has to learn when to emit inhibition from its curvature, clearance and body inputs.
const SLOW_BRAKE_AUTHORITY: f32 = 0.8;

/// Motor values in the physics order: throttle, steering and brake.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct MotorCommandValue {
    pub throttle: f32,
    pub steer: f32,
    pub brake: f32,
}

impl MotorCommandValue {
    pub const fn new(throttle: f32, steer: f32, brake: f32) -> Self {
        Self {
            throttle,
            steer,
            brake,
        }
    }

    pub const fn as_array(self) -> [f32; 3] {
        [self.throttle, self.steer, self.brake]
    }
}

/// Names the first rule that removed a requested brake command.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BrakeSuppression {
    #[default]
    None,
    NonpositiveAfterModulation,
    LowSpeedGate,
    ReverseOverride,
}

/// Names the final command override observed before physics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OverrideCause {
    #[default]
    None,
    Reverse,
    PostReverseLockout,
    PhantomPilot,
    SensorPilot,
    SteerScale,
    FixedSteer,
    FixedThrottle,
    Damage,
    DeadBody,
}

/// Pure mapping result before state-machine and diagnostic overrides.
#[derive(Clone, Copy)]
pub struct MappedNativeAction {
    pub raw_intent: [f32; 3],
    pub motor_head: &'static str,
    pub throttle_arbitration: &'static str,
    pub after_head: [f32; 3],
    pub slow_output: [f32; 3],
    pub hip_bias: [f32; 3],
    pub after_modulation: [f32; 3],
    pub steering_filtered: f32,
    pub pre_physics_speed_mps: f32,
    pub brake_suppression: BrakeSuppression,
    pub command: MotorCommandValue,
}

/// Compact selected-being trace attached to the existing sampled diagnostics.
#[derive(Clone, Copy, Serialize)]
pub struct MotorTrace {
    pub source: &'static str,
    pub phase: &'static str,
    pub pre_physics_speed_mps: f32,
    pub raw_intent: [f32; 3],
    pub motor_head: &'static str,
    pub throttle_arbitration: &'static str,
    pub after_head: [f32; 3],
    pub slow_output: [f32; 3],
    pub hip_bias: [f32; 3],
    pub after_modulation: [f32; 3],
    pub steering_filtered: f32,
    pub brake_suppression: BrakeSuppression,
    pub override_cause: OverrideCause,
    pub final_command: [f32; 3],
}

impl MotorTrace {
    pub fn from_mapping(mapped: MappedNativeAction) -> Self {
        Self {
            source: "native_action_mapper_v1",
            phase: "pre_physics_command_assembly",
            pre_physics_speed_mps: mapped.pre_physics_speed_mps,
            raw_intent: mapped.raw_intent,
            motor_head: mapped.motor_head,
            throttle_arbitration: mapped.throttle_arbitration,
            after_head: mapped.after_head,
            slow_output: mapped.slow_output,
            hip_bias: mapped.hip_bias,
            after_modulation: mapped.after_modulation,
            steering_filtered: mapped.steering_filtered,
            brake_suppression: mapped.brake_suppression,
            override_cause: OverrideCause::None,
            final_command: mapped.command.as_array(),
        }
    }

    pub fn with_final_command(mut self, command: MotorCommandValue, cause: OverrideCause) -> Self {
        self.final_command = command.as_array();
        self.override_cause = cause;
        self
    }

    /// Records the reverse controller's brake suppression independently of later overrides.
    pub fn mark_reverse_override(&mut self) {
        self.brake_suppression = BrakeSuppression::ReverseOverride;
    }
}

/// Applies the historical native action math, or the opt-in linear head, without I/O.
pub fn map_native_action(input: NativeActionInput<'_>) -> MappedNativeAction {
    let after_head = [
        input.head.map(0, input.raw_intent[0]),
        input.head.map(1, input.raw_intent[1]),
        input.head.map(2, input.raw_intent[2]),
    ];
    let slow_inhibition = input.slow_output[1].max(0.0);
    let throttle_cap = 1.0 - slow_inhibition * SLOW_BRAKE_AUTHORITY;
    let throttle = match input.throttle_arbitration {
        ThrottleArbitration::Legacy => {
            (after_head[0].max(0.0) + input.slow_output[0] * 0.5 + input.hip_bias[0])
                .clamp(0.0, throttle_cap)
        }
        ThrottleArbitration::Signed => {
            (after_head[0] + input.slow_output[0] * 0.5 + input.hip_bias[0])
                .clamp(0.0, throttle_cap)
        }
    };
    let steer =
        (after_head[1] + input.slow_output[2] * input.slow_steering_authority + input.hip_bias[1])
            .clamp(-1.0, 1.0);
    // Excitatory brake pathways combine after rectification. A negative fast motor
    // logit means "no fast brake"; it must not veto an independent slow inhibition
    // signal. The previous signed sum left the new path practically disconnected when
    // old checkpoints carried a strongly negative fast-brake row.
    let fast_brake = (after_head[2] + input.hip_bias[2]).max(0.0);
    let brake_before_gate = (fast_brake + slow_inhibition * SLOW_BRAKE_AUTHORITY).min(1.0);
    let steering_filtered = (input.previous_steer * (1.0 - input.steering_alpha)
        + steer * input.steering_alpha)
        .clamp(-1.0, 1.0);
    let (brake, brake_suppression) = if brake_before_gate <= 0.0 {
        (0.0, BrakeSuppression::NonpositiveAfterModulation)
    } else if input.speed_mps < 2.0 {
        (0.0, BrakeSuppression::LowSpeedGate)
    } else {
        (brake_before_gate, BrakeSuppression::None)
    };
    MappedNativeAction {
        raw_intent: input.raw_intent,
        motor_head: input.head.name(),
        throttle_arbitration: input.throttle_arbitration.name(),
        after_head,
        slow_output: input.slow_output,
        hip_bias: input.hip_bias,
        after_modulation: [throttle, steer, brake_before_gate],
        steering_filtered,
        pre_physics_speed_mps: input.speed_mps,
        brake_suppression,
        command: MotorCommandValue::new(throttle, steering_filtered, brake),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use universe::evolution::dna_genesis::default_kan_knots;

    fn input(raw_intent: [f32; 3], speed_mps: f32) -> NativeActionInput<'static> {
        NativeActionInput {
            raw_intent,
            slow_output: [0.0; 3],
            hip_bias: [0.0; 3],
            previous_steer: 0.0,
            speed_mps,
            steering_alpha: 0.5,
            slow_steering_authority: 0.2,
            head: MotorHead::Spline(&DEFAULT_KNOTS),
            throttle_arbitration: ThrottleArbitration::Legacy,
        }
    }

    static DEFAULT_KNOTS: [[f32; universe::kan::KAN_OUT_KNOTS]; 3] =
        [[0.0; universe::kan::KAN_OUT_KNOTS]; 3];

    #[test]
    fn legacy_spline_math_matches_existing_command_path() {
        let knots = default_kan_knots();
        let mapped = map_native_action(NativeActionInput {
            head: MotorHead::Spline(&knots),
            ..input([1.0, -1.0, 1.0], 10.0)
        });

        assert!((mapped.after_head[0] - 0.709_620_3).abs() < 1e-6);
        assert!((mapped.command.throttle - 0.709_620_3).abs() < 1e-6);
        assert!((mapped.command.steer + 0.354_810_15).abs() < 1e-6);
        assert!((mapped.command.brake - 0.709_620_3).abs() < 1e-6);
        assert_eq!(mapped.brake_suppression, BrakeSuppression::None);
    }

    #[test]
    fn extracted_legacy_path_matches_the_previous_formula_bit_for_bit() {
        let knots = default_kan_knots();
        let cases: [([f32; 3], [f32; 3], [f32; 3], f32, f32); 2] = [
            (
                [0.7, -0.4, 0.3],
                [0.6, 0.2, -0.5],
                [0.1, -0.1, 0.2],
                0.3,
                12.0,
            ),
            (
                [-0.8, 0.9, -0.7],
                [-0.4, -0.8, 0.7],
                [-0.2, 0.1, -0.1],
                -0.6,
                1.0,
            ),
        ];
        for (raw, slow, hip, previous, speed) in cases {
            let after_head = [
                universe::kan::eval_knots(&knots, 0, raw[0]),
                universe::kan::eval_knots(&knots, 1, raw[1]),
                universe::kan::eval_knots(&knots, 2, raw[2]),
            ];
            let slow_inhibition = slow[1].max(0.0);
            let throttle_cap = 1.0 - slow_inhibition * SLOW_BRAKE_AUTHORITY;
            let expected_throttle =
                (after_head[0].max(0.0) + slow[0] * 0.5 + hip[0]).clamp(0.0, throttle_cap);
            let expected_steer = (previous * 0.5
                + (after_head[1] + slow[2] * 0.2 + hip[1]).clamp(-1.0, 1.0) * 0.5)
                .clamp(-1.0, 1.0);
            let fast_brake = (after_head[2] + hip[2]).max(0.0);
            let brake_before_gate = (fast_brake + slow_inhibition * SLOW_BRAKE_AUTHORITY).min(1.0);
            let expected_brake = if speed < 2.0 { 0.0 } else { brake_before_gate };
            let mapped = map_native_action(NativeActionInput {
                raw_intent: raw,
                slow_output: slow,
                hip_bias: hip,
                previous_steer: previous,
                speed_mps: speed,
                steering_alpha: 0.5,
                slow_steering_authority: 0.2,
                head: MotorHead::Spline(&knots),
                throttle_arbitration: ThrottleArbitration::Legacy,
            });

            assert_eq!(
                mapped.command.throttle.to_bits(),
                expected_throttle.to_bits()
            );
            assert_eq!(mapped.command.steer.to_bits(), expected_steer.to_bits());
            assert_eq!(mapped.command.brake.to_bits(), expected_brake.to_bits());
        }
    }

    #[test]
    fn negative_raw_brake_is_distinct_from_low_speed_suppression() {
        let knots = default_kan_knots();
        let negative = map_native_action(NativeActionInput {
            head: MotorHead::Spline(&knots),
            ..input([0.0, 0.0, -0.8], 20.0)
        });
        let low_speed = map_native_action(NativeActionInput {
            head: MotorHead::Spline(&knots),
            ..input([0.0, 0.0, 0.8], 1.0)
        });

        assert_eq!(negative.command.brake, 0.0);
        assert_eq!(
            negative.brake_suppression,
            BrakeSuppression::NonpositiveAfterModulation
        );
        assert_eq!(low_speed.command.brake, 0.0);
        assert_eq!(low_speed.brake_suppression, BrakeSuppression::LowSpeedGate);
    }

    #[test]
    fn reverse_override_has_its_own_brake_suppression_reason() {
        let knots = default_kan_knots();
        let mapped = map_native_action(NativeActionInput {
            head: MotorHead::Spline(&knots),
            ..input([0.5, 0.0, 0.8], 10.0)
        });
        let mut trace = MotorTrace::from_mapping(mapped);
        trace.mark_reverse_override();
        let trace = trace.with_final_command(
            MotorCommandValue::new(0.75, 0.0, 0.0),
            OverrideCause::Reverse,
        );

        assert_eq!(trace.brake_suppression, BrakeSuppression::ReverseOverride);
        assert_eq!(trace.override_cause, OverrideCause::Reverse);
    }

    #[test]
    fn linear_head_exposes_full_existing_intent_authority() {
        let mapped = map_native_action(NativeActionInput {
            head: MotorHead::Linear,
            ..input([1.0, -1.0, 1.0], 10.0)
        });

        assert_eq!(mapped.after_head, [1.0, -1.0, 1.0]);
        assert_eq!(mapped.command, MotorCommandValue::new(1.0, -0.5, 1.0));
    }

    #[test]
    fn signed_throttle_can_cancel_positive_slow_aggression() {
        let knots = default_kan_knots();
        let legacy = map_native_action(NativeActionInput {
            raw_intent: [-1.0, 0.0, -1.0],
            slow_output: [0.8, 0.0, 0.0],
            head: MotorHead::Spline(&knots),
            ..input([0.0; 3], 10.0)
        });
        let signed = map_native_action(NativeActionInput {
            throttle_arbitration: ThrottleArbitration::Signed,
            raw_intent: [-1.0, 0.0, -1.0],
            slow_output: [0.8, 0.0, 0.0],
            head: MotorHead::Spline(&knots),
            ..input([0.0; 3], 10.0)
        });

        assert_eq!(legacy.command.throttle, 0.4);
        assert_eq!(signed.command.throttle, 0.0);
        assert_eq!(signed.command.brake, 0.0);
    }

    #[test]
    fn slow_motor_inhibition_can_brake_even_when_fast_brake_is_negative() {
        let mapped = map_native_action(NativeActionInput {
            head: MotorHead::Linear,
            raw_intent: [1.0, 0.0, -1.0],
            slow_output: [0.0, 0.75, 0.0],
            ..input([0.0; 3], 20.0)
        });

        assert!((mapped.command.throttle - 0.4).abs() < 1e-6);
        assert!((mapped.command.brake - 0.6).abs() < 1e-6);
        assert_eq!(mapped.brake_suppression, BrakeSuppression::None);
    }
}
