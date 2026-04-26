//! Tesla-style safety layer. Pure rules, no NN.
//! Tier 0 = HARD veto (returns Some(IntentVec) overriding everything).
//! Tier 1/2 = SOFT proposal (gate decides weight).

use bitflags::bitflags;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct IntentVec {
    pub throttle: f32,  // [-1, 1] (negative = reverse)
    pub steer: f32,     // [-1, 1]
    pub brake: f32,     // [0, 1]
}

#[derive(Debug, Clone, Copy)]
pub struct SafetyState {
    pub speed_kmh: f32,
    pub yaw_rate: f32,         // rad/s
    pub slip_avg: f32,         // [0, 1]
    pub ttc_forward: f32,      // s, INF if clear
    pub wall_left: f32,        // raycast distance m
    pub wall_right: f32,
    pub stall_timer: f32,      // s spent below 2 km/h with obstacle ahead
    pub pinado_timer: f32,     // s pinned (V<2 + obstacle)
    pub rain: f32,             // [0, 1]
    pub curv25: f32,           // upcoming curvature
    pub current_thr: f32,      // brain proposed thr (for tier2 TCS check)
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct SafetyFlags: u16 {
        const ANTI_STALL  = 1 << 0;
        const FCA         = 1 << 1;
        const SPIN        = 1 << 2;
        const ESP         = 1 << 3;
        const LANE_KEEP   = 1 << 4;
        const REVERSE_OUT = 1 << 5;
        const TCS         = 1 << 6;
        const AQUA        = 1 << 7;
        const CORNER_LIM  = 1 << 8;
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SentinelOutput {
    pub hard_veto: Option<IntentVec>,
    pub soft_proposal: IntentVec,
    pub flags: SafetyFlags,
}

pub struct Sentinel;

impl Sentinel {
    pub fn evaluate(s: &SafetyState) -> SentinelOutput {
        let mut out = SentinelOutput::default();

        // NaN/Inf hardening — non-finite inputs disable all rules silently otherwise.
        // ttc_forward is allowed to be INF (sentinel: "clear ahead").
        if !(s.yaw_rate.is_finite()
            && s.slip_avg.is_finite()
            && s.speed_kmh.is_finite()
            && s.wall_left.is_finite()
            && s.wall_right.is_finite()
            && s.stall_timer.is_finite()
            && s.pinado_timer.is_finite()
            && s.rain.is_finite()
            && s.curv25.is_finite()
            && s.current_thr.is_finite())
        {
            return out;
        }

        // ----- Tier 0 HARD vetos (priority: spin > FCA > anti-stall) -----
        if s.yaw_rate.abs() > 3.0 {
            out.flags |= SafetyFlags::SPIN;
            out.hard_veto = Some(IntentVec {
                throttle: 0.0,
                brake: 0.5,
                steer: -s.yaw_rate.signum() * 0.5,
            });
            return out;
        }
        if s.ttc_forward < 0.3 && s.speed_kmh > 15.0 {
            out.flags |= SafetyFlags::FCA;
            out.hard_veto = Some(IntentVec { throttle: 0.0, brake: 1.0, steer: 0.0 });
            return out;
        }
        if s.stall_timer > 0.3 && s.speed_kmh < 2.0 && (s.wall_left < 1.0 || s.wall_right < 1.0 || s.ttc_forward < 2.0) {
            out.flags |= SafetyFlags::ANTI_STALL;
            // sign chosen by which wall is closer (escape away from it)
            let escape_steer = if s.wall_left < s.wall_right { 0.5 } else { -0.5 };
            out.hard_veto = Some(IntentVec {
                throttle: -0.5,
                brake: 0.0,
                steer: escape_steer,
            });
            return out;
        }

        // ----- Tier 1 SOFT proposals -----
        let mut prop = IntentVec::default();
        if s.yaw_rate.abs() > 1.5 {
            out.flags |= SafetyFlags::ESP;
            prop.throttle -= 0.7; // counter-throttle proposal
            prop.brake += 0.3;
        }
        if s.wall_left < 0.5 {
            out.flags |= SafetyFlags::LANE_KEEP;
            prop.steer += 0.3;
        } else if s.wall_right < 0.5 {
            out.flags |= SafetyFlags::LANE_KEEP;
            prop.steer -= 0.3;
        }
        if s.pinado_timer > 0.3 && s.speed_kmh < 2.0 {
            out.flags |= SafetyFlags::REVERSE_OUT;
            prop.throttle -= 0.4;
        }

        // ----- Tier 2 SOFT proposals -----
        if s.slip_avg > 0.4 && s.current_thr > 0.5 {
            out.flags |= SafetyFlags::TCS;
            prop.throttle -= 0.4;
        }
        if s.rain > 0.7 && s.speed_kmh > 40.0 {
            out.flags |= SafetyFlags::AQUA;
            prop.throttle -= 0.3;
        }
        if s.curv25 > 30.0 && s.speed_kmh > 60.0 {
            out.flags |= SafetyFlags::CORNER_LIM;
            prop.brake += 0.3;
        }

        prop.throttle = prop.throttle.clamp(-1.0, 1.0);
        prop.steer = prop.steer.clamp(-1.0, 1.0);
        prop.brake = prop.brake.clamp(0.0, 1.0);
        out.soft_proposal = prop;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_state() -> SafetyState {
        SafetyState {
            speed_kmh: 30.0, yaw_rate: 0.0, slip_avg: 0.0, ttc_forward: 10.0,
            wall_left: 5.0, wall_right: 5.0, stall_timer: 0.0, pinado_timer: 0.0,
            rain: 0.0, curv25: 0.0, current_thr: 0.5,
        }
    }

    #[test]
    fn anti_stall_triggers_after_300ms() {
        let mut s = base_state();
        s.speed_kmh = 1.0; s.stall_timer = 0.4; s.wall_left = 0.5;
        let out = Sentinel::evaluate(&s);
        assert!(out.flags.contains(SafetyFlags::ANTI_STALL));
        let v = out.hard_veto.expect("hard veto required");
        assert!(v.throttle < 0.0);
    }

    #[test]
    fn fca_critical_overrides_throttle() {
        let mut s = base_state();
        s.ttc_forward = 0.2; s.speed_kmh = 50.0;
        let out = Sentinel::evaluate(&s);
        assert!(out.flags.contains(SafetyFlags::FCA));
        assert_eq!(out.hard_veto.unwrap().brake, 1.0);
    }

    #[test]
    fn spin_correction_opposite_yaw() {
        let mut s = base_state();
        s.yaw_rate = 3.5;
        let out = Sentinel::evaluate(&s);
        assert!(out.flags.contains(SafetyFlags::SPIN));
        assert!(out.hard_veto.unwrap().steer < 0.0);
    }

    #[test]
    fn no_hard_veto_in_calm_state() {
        let out = Sentinel::evaluate(&base_state());
        assert!(out.hard_veto.is_none());
        assert_eq!(out.flags, SafetyFlags::empty());
    }

    #[test]
    fn esp_soft_proposal_on_moderate_yaw() {
        let mut s = base_state();
        s.yaw_rate = 2.0;
        let out = Sentinel::evaluate(&s);
        assert!(out.flags.contains(SafetyFlags::ESP));
        assert!(out.soft_proposal.brake > 0.0);
    }

    #[test]
    fn tcs_cuts_throttle_on_slip() {
        let mut s = base_state();
        s.slip_avg = 0.5; s.current_thr = 0.8;
        let out = Sentinel::evaluate(&s);
        assert!(out.flags.contains(SafetyFlags::TCS));
        assert!(out.soft_proposal.throttle < 0.0);
    }

    #[test]
    fn priority_spin_over_fca() {
        let mut s = base_state();
        s.yaw_rate = 3.5; s.ttc_forward = 0.1; s.speed_kmh = 40.0;
        let out = Sentinel::evaluate(&s);
        assert!(out.flags.contains(SafetyFlags::SPIN));
        assert!(!out.flags.contains(SafetyFlags::FCA));
    }

    #[test]
    fn nan_input_disables_safety_silently() {
        let mut s = base_state();
        s.yaw_rate = f32::NAN;
        let out = Sentinel::evaluate(&s);
        assert!(out.hard_veto.is_none());
        assert_eq!(out.flags, SafetyFlags::empty());
    }

    #[test]
    fn multiple_soft_flags_compose_and_clamp() {
        // ESP (yaw 2.0) + TCS (slip 0.5, thr 0.8) + AQUA (rain 0.8, speed 50)
        // Combined throttle delta would be -0.7 + -0.4 + -0.3 = -1.4, must clamp to -1.0.
        let mut s = base_state();
        s.yaw_rate = 2.0;
        s.slip_avg = 0.5;
        s.current_thr = 0.8;
        s.rain = 0.8;
        s.speed_kmh = 50.0;
        let out = Sentinel::evaluate(&s);
        assert!(out.flags.contains(SafetyFlags::ESP));
        assert!(out.flags.contains(SafetyFlags::TCS));
        assert!(out.flags.contains(SafetyFlags::AQUA));
        assert_eq!(out.soft_proposal.throttle, -1.0); // clamped
        assert!(out.hard_veto.is_none());
    }
}
