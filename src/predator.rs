//! Scripted predator — a chasing hazard the population cannot out-wait.
//!
//! Why this module exists: the world measured 0% mortality in 34 of 35 generations
//! (difficulty-ladder battery, 2026-08-17) and a closed-form ridge pilot beats the
//! evolved champion — the current pressures are static and a per-tick regression on
//! the sensory wave captures them. A predator is the first pressure whose future
//! position depends on the being's OWN trajectory, so reacting to it needs state.
//!
//! This is a PURE state machine, in the pattern of `gpu_evolve/escada.rs` and
//! `gpu_evolve/fantasma.rs`: no I/O, no statics, no CLI knowledge, no GPU. `step`
//! consumes a state and returns the next one; determinism is total in the inputs.
//!
//! # Integrator contract (src/bin/gpu_evolve/, NOT wired by this module)
//!
//! - Spawn per generation: `Predator::spawn(seed_gen, center, radius, params)` —
//!   same seed ⇒ same spawn for every being of the generation (D2 comparability).
//! - Per tick, CPU-side: `pred = pred.step(&positions, &alive, DT)` with
//!   `DT = 1/64.0` s, then apply the touch rule: `caught(&pred, pos, params)` says
//!   whether a being pays. The COST IS THE INTEGRATOR'S DECISION and must follow the
//!   Crossed-Currency Law: charge life or fuel, never fitness directly (the lesson
//!   fantasma.rs already encodes).
//! - Dose knobs (`speed_cap`, `turn_rate`, catch radius) belong to CLI flags/statics
//!   on the integrator side; `PredatorParams::default()` documents the intended
//!   starting dose only. Escapability by construction: default cap 15 m/s is below
//!   the 18.75 m/s cruise, so a healthy being outruns it in a straight line and the
//!   pressure bites the slow and the cornering — mortality dose stays sweepable.
//! - Units: metres, seconds, radians. World XZ plane, same frame as
//!   `waypoints_world`.

use bevy::math::Vec2;

use crate::rng::Lcg;

/// Dose knobs of the predator. Owned by the CLI/statics on the integrator side.
#[derive(Clone, Copy, Debug)]
pub struct PredatorParams {
    /// Top speed, m/s. Below being cruise (18.75) so straights are an escape.
    pub speed_cap: f32,
    /// Forward acceleration, m/s².
    pub accel: f32,
    /// Max heading change, rad/s. Limits cornering: a tight turn is an escape too.
    pub turn_rate: f32,
    /// A being closer than this is "caught" this tick, metres.
    pub catch_radius_m: f32,
}

impl Default for PredatorParams {
    fn default() -> Self {
        Self {
            speed_cap: 15.0,
            accel: 6.0,
            turn_rate: 1.2,
            catch_radius_m: 3.0,
        }
    }
}

/// Predator state. Heading is carried by `vel`; a stopped predator keeps its last
/// heading in `heading` so restart direction is well-defined.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Predator {
    /// World position, metres.
    pub pos: Vec2,
    /// Velocity, m/s.
    pub vel: Vec2,
    /// Last unit heading — the direction `vel` had when it was last non-zero.
    pub heading: Vec2,
}

impl Predator {
    /// Deterministic spawn: a seed-derived point on the circle of `radius` around
    /// `center`, at rest, heading toward `center`. Same seed ⇒ identical predator.
    pub fn spawn(seed: u64, center: Vec2, radius: f32) -> Self {
        let mut rng = Lcg::new(seed ^ 0x9E3D_A70F); // domain-separated from other seed users
        let angle = rng.uniform() * std::f32::consts::TAU;
        let pos = center + Vec2::new(angle.cos(), angle.sin()) * radius;
        Self {
            pos,
            vel: Vec2::ZERO,
            heading: (center - pos).normalize_or(Vec2::X),
        }
    }

    /// Index of the nearest LIVING target, or `None` if all are dead.
    /// `targets` and `alive` are index-aligned; ties break on the lower index
    /// (`min_by` keeps the first minimum), so the result is deterministic.
    pub fn nearest_living(&self, targets: &[Vec2], alive: &[bool]) -> Option<usize> {
        assert_eq!(
            targets.len(),
            alive.len(),
            "targets and alive mask must be index-aligned"
        );
        targets
            .iter()
            .enumerate()
            .filter(|(i, _)| alive[*i])
            .map(|(i, p)| (i, p.distance(self.pos)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    }

    /// One simulation step of the scripted chase policy. Pure: returns the next
    /// state, never mutates.
    ///
    /// Policy: steer toward the nearest living target, with the heading change
    /// clamped to `turn_rate·dt`; accelerate up to `speed_cap`. With no living
    /// target, coast down to rest on the current heading (drag = `accel`).
    pub fn step(
        &self,
        targets: &[Vec2],
        alive: &[bool],
        dt: f32,
        params: &PredatorParams,
    ) -> Predator {
        let speed = self.vel.length();
        let (heading, new_speed) = match self.nearest_living(targets, alive) {
            Some(i) => {
                let desired = (targets[i] - self.pos).normalize_or(self.heading);
                let heading = rotate_toward(self.heading, desired, params.turn_rate * dt);
                (heading, (speed + params.accel * dt).min(params.speed_cap))
            }
            None => (self.heading, (speed - params.accel * dt).max(0.0)),
        };
        let vel = heading * new_speed;
        Predator {
            pos: self.pos + vel * dt,
            vel,
            heading,
        }
    }

    /// Whether a being at `being_pos` is inside the catch radius this tick.
    /// The integrator decides what being caught COSTS (crossed currency — see the
    /// module header); this function only answers the geometric question.
    pub fn caught(&self, being_pos: Vec2, params: &PredatorParams) -> bool {
        self.pos.distance(being_pos) <= params.catch_radius_m
    }
}

/// Rotate unit vector `from` toward unit vector `to` by at most `max_angle` radians.
/// Both inputs must be unit-length (or zero, which passes through unchanged).
fn rotate_toward(from: Vec2, to: Vec2, max_angle: f32) -> Vec2 {
    let angle = from.angle_to(to); // signed, in (-PI, PI]
    let clamped = angle.clamp(-max_angle, max_angle);
    let (sin, cos) = clamped.sin_cos();
    Vec2::new(from.x * cos - from.y * sin, from.x * sin + from.y * cos)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 64.0;

    #[test]
    fn spawn_is_deterministic_in_the_seed() {
        let a = Predator::spawn(77, Vec2::new(100.0, 50.0), 40.0);
        let b = Predator::spawn(77, Vec2::new(100.0, 50.0), 40.0);
        assert_eq!(a, b, "same seed must reproduce the spawn byte-identically");
        let c = Predator::spawn(78, Vec2::new(100.0, 50.0), 40.0);
        assert_ne!(a.pos, c.pos, "a different seed must move the spawn");
        assert!(
            (a.pos.distance(Vec2::new(100.0, 50.0)) - 40.0).abs() < 1e-3,
            "spawn must sit on the requested circle"
        );
    }

    #[test]
    fn closes_distance_to_a_stationary_target() {
        let params = PredatorParams::default();
        let mut pred = Predator::spawn(7, Vec2::ZERO, 50.0);
        let targets = [Vec2::ZERO];
        let alive = [true];
        let d0 = pred.pos.distance(targets[0]);
        for _ in 0..(4.0 / DT) as usize {
            pred = pred.step(&targets, &alive, DT, &params);
        }
        let d1 = pred.pos.distance(targets[0]);
        // Ceiling math before the assert: 4 s at cap 15 m/s covers 60 m > 50 m
        // spawn distance, minus the ~2.5 s spent accelerating — 20 m of closing is
        // a conservative floor.
        assert!(
            d1 < d0 - 20.0,
            "predator failed to close on the target: {d0:.1} -> {d1:.1} m"
        );
    }

    #[test]
    fn ignores_dead_targets() {
        let params = PredatorParams::default();
        let pred = Predator {
            pos: Vec2::ZERO,
            vel: Vec2::ZERO,
            heading: Vec2::X,
        };
        let targets = [Vec2::new(5.0, 0.0), Vec2::new(100.0, 0.0)];
        // The near target is dead: the far living one must be chosen.
        assert_eq!(pred.nearest_living(&targets, &[false, true]), Some(1));
        assert_eq!(pred.nearest_living(&targets, &[false, false]), None);
    }

    #[test]
    fn heading_change_per_step_is_turn_limited() {
        let params = PredatorParams::default();
        // Target directly BEHIND: worst-case 180-degree demand.
        let mut pred = Predator {
            pos: Vec2::ZERO,
            vel: Vec2::X * 10.0,
            heading: Vec2::X,
        };
        let targets = [Vec2::new(-100.0, 1.0)];
        let alive = [true];
        let max_step = params.turn_rate * DT + 1e-5;
        for _ in 0..200 {
            let next = pred.step(&targets, &alive, DT, &params);
            let delta = pred.heading.angle_to(next.heading).abs();
            assert!(
                delta <= max_step,
                "heading jumped {delta:.5} rad in one step, cap is {max_step:.5}"
            );
            pred = next;
        }
        // After 200 steps (3.125 s at 1.2 rad/s = 3.75 rad available) the U-turn
        // must be complete: now pointing at the target.
        let to_target = (targets[0] - pred.pos).normalize_or(Vec2::X);
        assert!(
            pred.heading.dot(to_target) > 0.99,
            "turn never completed: heading {:?} vs target dir {:?}",
            pred.heading,
            to_target
        );
    }

    #[test]
    fn speed_never_exceeds_cap_and_decays_without_targets() {
        let params = PredatorParams::default();
        let mut pred = Predator::spawn(3, Vec2::ZERO, 30.0);
        let targets = [Vec2::ZERO];
        for _ in 0..1000 {
            pred = pred.step(&targets, &[true], DT, &params);
            assert!(
                pred.vel.length() <= params.speed_cap + 1e-4,
                "speed {} broke the cap {}",
                pred.vel.length(),
                params.speed_cap
            );
        }
        // All targets dead: the predator must coast to rest, not drift forever.
        for _ in 0..1000 {
            pred = pred.step(&targets, &[false], DT, &params);
        }
        assert_eq!(pred.vel.length(), 0.0, "predator must stop with no prey");
    }

    #[test]
    fn caught_is_a_pure_radius_test() {
        let params = PredatorParams::default();
        let pred = Predator {
            pos: Vec2::ZERO,
            vel: Vec2::ZERO,
            heading: Vec2::X,
        };
        assert!(pred.caught(Vec2::new(params.catch_radius_m - 0.1, 0.0), &params));
        assert!(!pred.caught(Vec2::new(params.catch_radius_m + 0.1, 0.0), &params));
    }
}
