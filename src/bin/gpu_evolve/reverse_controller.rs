//! Pure opt-in reverse-controller state machine.
//!
//! The production legacy branch remains in `context.rs`; this module isolates the
//! corrected controller so trigger, clearance, timeout and cooldown cannot overlap.

use universe::config::iara::reverse_maneuver as config;

pub const MAX_DURATION_S: f32 = config::MAX_DURATION_S;
pub const POST_LOCKOUT_S: f32 = config::POST_LOCKOUT_S;
/// V2 exits only after clearing the entire trigger region, removing overlapping states.
pub const EXIT_CLEAR_M: f32 = config::TRIGGER_DIST_M;

/// Chooses the historical inline controller or the corrected opt-in state machine.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReverseMode {
    #[default]
    Legacy,
    V2,
}

/// Persistent fields needed by the pure reverse controller.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ReverseState {
    pub remaining_s: f32,
    pub post_lockout_s: f32,
    pub failures: u32,
    pub since_exit_s: f32,
}

/// Current observations used to advance the reverse controller once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReverseInput {
    pub dt_s: f32,
    pub forward_distance_m: f32,
    pub speed_mps: f32,
    pub stagnation_ticks: u32,
}

/// Transition emitted by one V2 controller step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReverseEvent {
    #[default]
    None,
    Started,
    Active,
    Cleared,
    TimedOut,
    Cooldown,
}

/// Updated state and whether reverse must reach physics on this tick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReverseStep {
    pub state: ReverseState,
    pub active: bool,
    pub event: ReverseEvent,
    pub reset_stagnation: bool,
}

fn finish(mut state: ReverseState, event: ReverseEvent) -> ReverseStep {
    state.remaining_s = 0.0;
    state.post_lockout_s = POST_LOCKOUT_S;
    state.since_exit_s = 0.0;
    ReverseStep {
        state,
        active: false,
        event,
        reset_stagnation: false,
    }
}

/// Advances the corrected state machine once; a new trigger always reaches physics.
pub fn step_v2(mut state: ReverseState, input: ReverseInput) -> ReverseStep {
    let dt_s = input.dt_s.max(0.0);
    state.since_exit_s += dt_s;
    state.post_lockout_s = (state.post_lockout_s - dt_s).max(0.0);

    if state.remaining_s > 0.0 {
        if input.forward_distance_m >= EXIT_CLEAR_M {
            return finish(state, ReverseEvent::Cleared);
        }
        state.remaining_s = (state.remaining_s - dt_s).max(0.0);
        if state.remaining_s <= 0.0 {
            return finish(state, ReverseEvent::TimedOut);
        }
        return ReverseStep {
            state,
            active: true,
            event: ReverseEvent::Active,
            reset_stagnation: false,
        };
    }

    if state.post_lockout_s > 0.0 {
        return ReverseStep {
            state,
            active: false,
            event: ReverseEvent::Cooldown,
            reset_stagnation: false,
        };
    }

    let triggered = input.forward_distance_m < config::TRIGGER_DIST_M
        && input.speed_mps < config::TRIGGER_SPEED_MAX_MS
        && input.stagnation_ticks > config::STAGNATION_TICKS;
    if !triggered {
        return ReverseStep {
            state,
            active: false,
            event: ReverseEvent::None,
            reset_stagnation: false,
        };
    }

    state.failures = if state.since_exit_s < 5.0 {
        state.failures.saturating_add(1)
    } else {
        0
    };
    let scale = 1.0 + state.failures.min(4) as f32 * 0.75;
    // The first active command consumes this tick. Keeping only the remaining time
    // prevents the physical reverse duration from exceeding the configured bound.
    state.remaining_s = ((config::MIN_DURATION_S * scale).min(MAX_DURATION_S) - dt_s).max(0.0);
    ReverseStep {
        state,
        active: true,
        event: ReverseEvent::Started,
        reset_stagnation: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blocked() -> ReverseInput {
        ReverseInput {
            dt_s: 1.0 / 64.0,
            forward_distance_m: 3.0,
            speed_mps: 0.0,
            stagnation_ticks: 61,
        }
    }

    #[test]
    fn controller_defaults_to_legacy() {
        assert_eq!(ReverseMode::default(), ReverseMode::Legacy);
    }

    #[test]
    fn three_metre_trigger_stays_active_for_the_first_physics_step() {
        let step = step_v2(ReverseState::default(), blocked());

        assert!(step.active);
        assert_eq!(step.event, ReverseEvent::Started);
        assert!(step.state.remaining_s > 0.0);
    }

    #[test]
    fn escalated_reverse_duration_never_exceeds_the_safety_bound() {
        let state = ReverseState {
            failures: 4,
            since_exit_s: 0.0,
            ..ReverseState::default()
        };
        let step = step_v2(state, blocked());

        assert!(step.active);
        assert!(step.state.remaining_s <= MAX_DURATION_S);
    }

    #[test]
    fn timeout_enters_the_same_cooldown_as_clear_exit() {
        let state = ReverseState {
            remaining_s: 0.005,
            ..ReverseState::default()
        };
        let timeout = step_v2(
            state,
            ReverseInput {
                stagnation_ticks: 0,
                ..blocked()
            },
        );

        assert!(!timeout.active);
        assert_eq!(timeout.event, ReverseEvent::TimedOut);
        assert_eq!(timeout.state.post_lockout_s, POST_LOCKOUT_S);
        assert_eq!(timeout.state.since_exit_s, 0.0);
    }

    #[test]
    fn active_reverse_only_clears_outside_the_trigger_region() {
        let state = ReverseState {
            remaining_s: 0.4,
            ..ReverseState::default()
        };
        let still_blocked = step_v2(
            state,
            ReverseInput {
                stagnation_ticks: 0,
                ..blocked()
            },
        );
        let clear = step_v2(
            state,
            ReverseInput {
                forward_distance_m: EXIT_CLEAR_M,
                stagnation_ticks: 0,
                ..blocked()
            },
        );

        assert!(still_blocked.active);
        assert_eq!(clear.event, ReverseEvent::Cleared);
        assert_eq!(clear.state.post_lockout_s, POST_LOCKOUT_S);
    }

    #[test]
    fn physical_reverse_ticks_stay_within_the_duration_bound() {
        let mut step = step_v2(ReverseState::default(), blocked());
        let mut active_ticks = usize::from(step.active);
        while step.state.remaining_s > 0.0 {
            step = step_v2(
                step.state,
                ReverseInput {
                    stagnation_ticks: 0,
                    ..blocked()
                },
            );
            active_ticks += usize::from(step.active);
        }

        let active_seconds = active_ticks as f32 * blocked().dt_s;
        assert!(active_seconds <= MAX_DURATION_S);
        assert_eq!(step.event, ReverseEvent::TimedOut);
        assert_eq!(step.state.post_lockout_s, POST_LOCKOUT_S);
    }
}
