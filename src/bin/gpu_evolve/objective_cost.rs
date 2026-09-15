//! Optional pain changes selection, never physical telemetry.

/// Discounts achieved progress by wall exposure and dissipated collision energy.
///
/// Both costs are first converted to metres. The squared Hill-like denominator
///
/// `safe_score = progress / (1 + pain_metres / scale_metres)^2`
///
/// preserves three properties needed by evolution:
/// - zero progress cannot beat movement merely by avoiding pain;
/// - unsafe progress remains ordered instead of many individuals clamping to zero;
/// - once pain is comparable to useful progress, a shorter clean trajectory can win.
///
/// `scale_floor_metres` keeps the first partial sector learnable before meaningful
/// progress exists. Setting both costs to zero preserves the score bit for bit.
pub fn safety_adjusted_score(
    score: f64,
    wall_ticks: u32,
    wall_cost_per_second: f32,
    collision_energy_j: f64,
    impact_cost_per_kilojoule: f32,
    scale_floor_metres: f64,
) -> f64 {
    if wall_cost_per_second <= 0.0 && impact_cost_per_kilojoule <= 0.0 {
        return score;
    }
    let progress = score.max(0.0);
    if progress == 0.0 {
        return 0.0;
    }
    let wall_metres = f64::from(wall_ticks) / 64.0 * f64::from(wall_cost_per_second.max(0.0));
    let impact_metres =
        collision_energy_j.max(0.0) / 1000.0 * f64::from(impact_cost_per_kilojoule.max(0.0));
    let scale = progress.max(scale_floor_metres.max(1.0));
    let ratio = (wall_metres + impact_metres) / scale;
    progress / (1.0 + ratio).powi(2)
}

/// Converts the ordered checkpoint ratchet into metres along the track.
///
/// `checkpoint_count` includes sector zero, which is crossed at spawn. Completed
/// sectors therefore use `count - 1`. Continuous progress is only allowed to fill
/// the current sector; skipping checkpoints cannot jump the score forward.
pub fn ordered_progress_metres(
    checkpoint_count: u32,
    continuous_progress_m: f64,
    lap_length_m: f64,
    sector_count: usize,
) -> f64 {
    if !continuous_progress_m.is_finite()
        || !lap_length_m.is_finite()
        || lap_length_m <= 0.0
        || sector_count == 0
    {
        return 0.0;
    }
    let sector_length = lap_length_m / sector_count as f64;
    let completed = checkpoint_count.saturating_sub(1) as f64;
    let base = completed * sector_length;
    continuous_progress_m
        .max(base)
        .min(base + sector_length)
        .max(0.0)
}

/// Keeps the centered objective in metres while giving up to 25% of the score
/// to road-centre quality. The bounded ratio prevents forward/backward oscillation
/// from manufacturing an unbounded centre bonus.
pub fn centered_progress_metres(ordered_progress_m: f64, centered_progress_m: f64) -> f64 {
    let ordered = ordered_progress_m.max(0.0);
    if ordered == 0.0 {
        return 0.0;
    }
    let quality = (centered_progress_m.max(0.0) / ordered).clamp(0.0, 1.0);
    ordered * (0.75 + 0.25 * quality)
}

#[cfg(test)]
mod tests {
    use super::{centered_progress_metres, ordered_progress_metres, safety_adjusted_score};
    #[test]
    fn pain_discounts_progress_without_making_stasis_optimal() {
        let clean = safety_adjusted_score(500.0, 0, 2.0, 0.0, 1.0, 50.0);
        let crashing = safety_adjusted_score(1_000.0, 0, 2.0, 5_000_000.0, 1.0, 50.0);
        let moving = safety_adjusted_score(5.0, 128, 2.0, 10_000.0, 1.0, 50.0);
        let stopped = safety_adjusted_score(0.0, 0, 2.0, 0.0, 1.0, 50.0);
        assert_eq!(clean, 500.0);
        assert!(crashing < clean);
        assert!(moving > stopped);
    }

    #[test]
    fn disabled_pain_preserves_bits() {
        for score in [0.0_f64, -0.0, 1.123456789, 1e12] {
            assert_eq!(
                safety_adjusted_score(score, u32::MAX, 0.0, f64::MAX, 0.0, 50.0).to_bits(),
                score.to_bits()
            );
        }
    }

    #[test]
    fn one_unit_of_relative_pain_quarters_the_score() {
        assert_eq!(
            safety_adjusted_score(1_000.0, 0, 0.0, 1_000_000.0, 1.0, 50.0),
            250.0
        );
    }

    #[test]
    fn checkpoint_score_has_distance_units_and_no_free_lap_at_spawn() {
        let lap = 1_200.0;
        assert_eq!(ordered_progress_metres(1, 0.0, lap, 24), 0.0);
        assert_eq!(ordered_progress_metres(1, 30.0, lap, 24), 30.0);
        assert_eq!(ordered_progress_metres(1, 500.0, lap, 24), 50.0);
        assert_eq!(ordered_progress_metres(25, 1_230.0, lap, 24), 1_230.0);
    }

    #[test]
    fn invalid_track_geometry_cannot_create_fitness() {
        assert_eq!(ordered_progress_metres(10, 100.0, 0.0, 24), 0.0);
        assert_eq!(ordered_progress_metres(10, f64::NAN, 1_000.0, 24), 0.0);
        assert_eq!(ordered_progress_metres(10, 100.0, 1_000.0, 0), 0.0);
    }

    #[test]
    fn centered_score_is_bounded_and_keeps_metres_as_its_unit() {
        assert_eq!(centered_progress_metres(1_000.0, 1_000.0), 1_000.0);
        assert_eq!(centered_progress_metres(1_000.0, 0.0), 750.0);
        assert_eq!(centered_progress_metres(1_000.0, 50_000.0), 1_000.0);
        assert_eq!(centered_progress_metres(0.0, 50_000.0), 0.0);
    }
}
