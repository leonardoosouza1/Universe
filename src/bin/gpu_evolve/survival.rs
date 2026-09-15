//! Pure survival consequences owned by the body/environment boundary.

pub const HOLE_DAMAGE_FRACTION: f64 = 1.0 / 3.0;
pub const HOLE_DAMAGE_COOLDOWN_TICKS: u32 = 64;
/// Dissipating this much collision energy consumes one complete body-health unit.
/// It corresponds to roughly the kinetic energy of a 1,200 kg vehicle at 41 m/s.
pub const IMPACT_ENERGY_TO_FAILURE_J: f32 = 1_000_000.0;

/// Converts physical collision energy into irreversible body damage in [0, 1].
pub fn impact_damage(energy_loss_j: f32, scale: f32) -> f32 {
    if !energy_loss_j.is_finite() || !scale.is_finite() || energy_loss_j <= 0.0 || scale <= 0.0 {
        return 0.0;
    }
    (energy_loss_j / IMPACT_ENERGY_TO_FAILURE_J * scale).clamp(0.0, 1.0)
}

/// Returns life earned by extending the farthest ordered progress reached so far.
///
/// An odometer cannot fund survival because circling or bouncing against a wall also
/// increases it. Signed progress alone still pays repeated forward/backward motion.
/// The high-water mark pays each unwrapped track metre once.
pub fn life_gain_from_new_progress(
    unwrapped_progress_m: f64,
    best_progress_m: &mut f64,
    life_per_metre: f64,
) -> f64 {
    if !unwrapped_progress_m.is_finite()
        || !best_progress_m.is_finite()
        || !life_per_metre.is_finite()
        || life_per_metre <= 0.0
    {
        return 0.0;
    }
    let frontier = unwrapped_progress_m.max(0.0);
    if frontier <= *best_progress_m {
        return 0.0;
    }
    let delta = frontier - *best_progress_m;
    *best_progress_m = frontier;
    delta * life_per_metre
}

/// Applies one recoverable hole injury and returns whether a new hit occurred.
///
/// The cooldown represents a contact episode rather than charging every physics tick
/// spent over the same hole. Damage scales with the current life capacity, so changing
/// episode length cannot silently make holes harmless or instantly lethal.
pub fn apply_hole_contact(
    life_left: &mut f64,
    cooldown_ticks: &mut u32,
    on_hole: bool,
    life_capacity: f64,
) -> bool {
    if *cooldown_ticks > 0 {
        *cooldown_ticks -= 1;
    }
    if !on_hole || *cooldown_ticks > 0 || *life_left <= 0.0 {
        return false;
    }
    *life_left = (*life_left - life_capacity.max(1.0) * HOLE_DAMAGE_FRACTION).max(0.0);
    *cooldown_ticks = HOLE_DAMAGE_COOLDOWN_TICKS;
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_contact_episode_costs_one_third_of_capacity() {
        let mut life = 900.0;
        let mut cooldown = 0;
        assert!(apply_hole_contact(&mut life, &mut cooldown, true, 900.0));
        assert_eq!(life, 600.0);
        assert_eq!(cooldown, 64);
        assert!(!apply_hole_contact(&mut life, &mut cooldown, true, 900.0));
        assert_eq!(life, 600.0);
    }

    #[test]
    fn leaving_and_waiting_allows_a_later_distinct_hit() {
        let mut life = 900.0;
        let mut cooldown = 0;
        assert!(apply_hole_contact(&mut life, &mut cooldown, true, 900.0));
        for _ in 0..64 {
            assert!(!apply_hole_contact(&mut life, &mut cooldown, false, 900.0));
        }
        assert!(apply_hole_contact(&mut life, &mut cooldown, true, 900.0));
        assert_eq!(life, 300.0);
    }

    #[test]
    fn survival_is_paid_only_for_a_new_progress_frontier() {
        let mut best = 0.0;
        assert_eq!(life_gain_from_new_progress(10.0, &mut best, 12.0), 120.0);
        assert_eq!(life_gain_from_new_progress(4.0, &mut best, 12.0), 0.0);
        assert_eq!(life_gain_from_new_progress(10.0, &mut best, 12.0), 0.0);
        assert_eq!(life_gain_from_new_progress(10.5, &mut best, 12.0), 6.0);
        assert_eq!(best, 10.5);
    }

    #[test]
    fn invalid_progress_never_mints_life() {
        let mut best = 3.0;
        assert_eq!(life_gain_from_new_progress(f64::NAN, &mut best, 12.0), 0.0);
        assert_eq!(life_gain_from_new_progress(4.0, &mut best, f64::NAN), 0.0);
        assert_eq!(best, 3.0);
    }

    #[test]
    fn impact_damage_scales_with_dissipated_energy() {
        assert_eq!(impact_damage(0.0, 1.0), 0.0);
        assert!((impact_damage(250_000.0, 1.0) - 0.25).abs() < 1e-6);
        assert_eq!(impact_damage(2_000_000.0, 1.0), 1.0);
        assert_eq!(impact_damage(f32::NAN, 1.0), 0.0);
    }
}
