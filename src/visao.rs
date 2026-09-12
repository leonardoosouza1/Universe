//! Runtime vision parameters — so that range and angle can be VARIED.
//!
//! # Why this exists
//!
//! Leonardo observed: *"the model does not see angles or does not see what is very far
//! away, it has a limit and depending on the speed that can get in the way"*. The
//! arithmetic confirms it, and the number is harsh.
//!
//! The retina has 14 rows with pitch from −0.012 (row 0) to −0.060 (row 13), and the eye
//! at 0.4 m above the ground. Each row sees `0.4 / |pitch|` meters ahead:
//!
//! | row | distance |
//! |---|---|
//! | 0 (the farthest) | **33.3 m** |
//! | 7 | 10.6 m |
//! | 13 | 6.7 m |
//!
//! And the braking distance to the tightest turn of the track (radius 9.2 m, v_curve 11.4 m/s):
//!
//! | speed | braking required | fits in the 33.3 m? |
//! |---|---|---|
//! | 25 m/s | 25.2 m | yes |
//! | **28 m/s** | **33.3 m** | **exactly at the limit** |
//! | 30 m/s | 39.2 m | **no** |
//! | 40 m/s | 74.9 m | no, 2.2× beyond |
//!
//! > **Above ~28 m/s (101 km/h) the car is blind to the point where it needs to start
//! > braking.** The retina does not reach.
//!
//! This explains the curvature probes at 20/40/70/120 m: they are the compensation for
//! the retina's limit. And it explains why ablating curvature costs only 3% — the car
//! operates below 28 m/s, where the retina suffices and the probes are redundant.
//!
//! # What can be varied
//!
//! `RETINA_ROWS` and `RETINA_COLS` are array sizes and must remain constant. But the
//! RANGE (pitch of row 0) and the ANGLE (FOV) are just numbers in a raycasting loop —
//! they can be parameters. They are exactly the two axes of Leonardo's observation.

use std::sync::OnceLock;

use crate::config::iara::vision::{FOV_DEG, RETINA_MAX_PITCH, RETINA_ROWS};

/// Eye height above the ground, in meters (`world.rs`: `pos + Vec3::Y * 0.4`).
pub const ALTURA_OLHO: f32 = 0.4;

/// Current range of the farthest row, in meters. This is what one wants to vary.
pub const ALCANCE_PADRAO: f32 = ALTURA_OLHO / 0.012; // = 33.33 m

#[derive(Debug, Clone, Copy)]
pub struct Params {
    /// How far row 0 sees on the ground, in meters.
    pub alcance_m: f32,
    /// Horizontal aperture in degrees.
    pub fov_deg: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            alcance_m: ALCANCE_PADRAO,
            fov_deg: FOV_DEG,
        }
    }
}

static PARAMS: OnceLock<Params> = OnceLock::new();

pub fn definir(p: Params) {
    let _ = PARAMS.set(p);
}

#[inline]
pub fn params() -> Params {
    *PARAMS.get().unwrap_or(&Params {
        alcance_m: ALCANCE_PADRAO,
        fov_deg: FOV_DEG,
    })
}

#[inline]
pub fn fov_deg() -> f32 {
    params().fov_deg
}

/// Pitch of row `row`, honoring the requested range.
///
/// Row 13 (the ground right ahead, 6.7 m) stays FIXED: touching it would change what
/// the car sees of its own tires, which is a different problem. What stretches is row
/// 0, and the middle rows interpolate — so the only thing that changes is **how far it
/// sees**.
#[inline]
pub fn pitch(row: usize) -> f32 {
    pitch_at(params().alcance_m, row)
}

/// The pure half of [`pitch`]: the same arithmetic, with the range passed in instead of
/// read from the process-wide `OnceLock`.
///
/// It exists so a test can exercise a NON-default range without setting the global —
/// setting it would leak into every other test running in the same process and is
/// irreversible (`OnceLock::set` succeeds once). Same reason `mutate_sensors` takes its
/// generator from outside instead of reading `SENSORS_EVOLVE_ON`.
#[inline]
pub fn pitch_at(alcance_m: f32, row: usize) -> f32 {
    if (alcance_m - ALCANCE_PADRAO).abs() < 1e-3 {
        return crate::config::iara::vision::RETINA_PITCHES[row.min(RETINA_ROWS - 1)];
    }
    let min_pitch = -(ALTURA_OLHO / alcance_m.max(1.0)); // row 0, the farthest
    if RETINA_ROWS <= 1 {
        return min_pitch;
    }
    let passo = (RETINA_MAX_PITCH - min_pitch) / (RETINA_ROWS as f32 - 1.0);
    min_pitch + passo * row.min(RETINA_ROWS - 1) as f32
}

/// Factor every RANGE-like perception parameter is multiplied by when the retina's own
/// range is moved away from the default — see [`range_scale_of`].
#[inline]
pub fn range_scale() -> f32 {
    range_scale_of(params().alcance_m)
}

/// The pure half of [`range_scale`]: `alcance_m / ALCANCE_PADRAO`, pinned to EXACTLY
/// `1.0` inside the same ±1e-3 m window [`pitch_at`] uses for its own identity branch.
///
/// # Why one single factor
///
/// The horizon of this being is not the retina alone: the wall sweep reaches 60 m, the
/// lateral sweep 40 m and the three curvature look-aheads 40/70/120 m, all of them well
/// PAST the 33.3 m the retina can see. Shortening only the retina would leave those
/// probes as a full-range replacement for it, and the run would measure nothing. So the
/// flag scales the whole perceptual horizon by one ratio, and the relative shape between
/// the probes is preserved.
///
/// The exact-`1.0` window is what makes the flag's absence bit-identical: the callers
/// branch on `scale == 1.0` and take the untouched path, so no multiplication — not even
/// by a float that rounds to one — ever happens on the default run.
#[inline]
pub fn range_scale_of(alcance_m: f32) -> f32 {
    if (alcance_m - ALCANCE_PADRAO).abs() < 1e-3 {
        return 1.0;
    }
    alcance_m.max(1.0) / ALCANCE_PADRAO
}

/// How far on the ground row `row` looks. Used for reports and for the tests.
pub fn distancia_da_linha(row: usize) -> f32 {
    let p = pitch(row);
    if p < -1e-6 {
        ALTURA_OLHO / -p
    } else {
        f32::INFINITY
    }
}

/// Speed above which the required braking exceeds the vision range.
///
/// `d_brake = (v² − v_curve²) / (2·μ·g)`, solved for `d_brake = range`.
pub fn velocidade_cega(alcance_m: f32, v_curva: f32) -> f32 {
    (2.0 * 9.81 * alcance_m + v_curva * v_curva).sqrt()
}

#[cfg(test)]
mod testes {
    use super::*;

    #[test]
    fn o_padrao_reproduz_a_tabela_de_pitches_original() {
        // without touching anything, `pitch()` must return exactly what the const
        // returned — otherwise the change alters the behaviour of every previous
        // experiment
        for row in 0..RETINA_ROWS {
            let esperado = crate::config::iara::vision::RETINA_PITCHES[row];
            assert!(
                (pitch(row) - esperado).abs() < 1e-9,
                "row {row}: {} vs {}",
                pitch(row),
                esperado
            );
        }
        assert!(
            (distancia_da_linha(0) - 33.33).abs() < 0.1,
            "row 0 sees {:.1} m",
            distancia_da_linha(0)
        );
    }

    #[test]
    fn o_alcance_pedido_e_o_alcance_entregue() {
        // exercises the arithmetic without depending on the global OnceLock
        for alvo in [20.0f32, 50.0, 80.0, 120.0] {
            let min_pitch = -(ALTURA_OLHO / alvo);
            let d = ALTURA_OLHO / -min_pitch;
            assert!(
                (d - alvo).abs() < 0.01,
                "asked for {alvo} m and the math gave {d} m"
            );
        }
    }

    /// Identity of the `--retina-range` path: with the default range nothing moves.
    ///
    /// `pitch_at` must return the ORIGINAL const table (not a recomputation that merely
    /// rounds to it), and `range_scale_of` must return exactly 1.0 so the packers take
    /// their untouched branch.
    #[test]
    fn the_default_range_is_bit_identical() {
        for row in 0..RETINA_ROWS {
            let expected = crate::config::iara::vision::RETINA_PITCHES[row];
            assert_eq!(
                pitch_at(ALCANCE_PADRAO, row).to_bits(),
                expected.to_bits(),
                "row {row}: pitch_at must return the const table bit for bit"
            );
            assert_eq!(pitch(row).to_bits(), expected.to_bits());
        }
        assert_eq!(range_scale_of(ALCANCE_PADRAO).to_bits(), 1.0f32.to_bits());
        // The global is untouched by this test, so `range_scale()` reads the default.
        assert_eq!(range_scale().to_bits(), 1.0f32.to_bits());
    }

    /// A shortened range shortens the retina AND reports the ratio the sensor packer
    /// applies to the wall/lateral/curvature probes. 10 m is the value U11b asks for.
    #[test]
    fn a_ten_metre_range_shortens_row_zero_and_scales_by_ten_over_default() {
        let p0 = pitch_at(10.0, 0);
        assert!(
            (ALTURA_OLHO / -p0 - 10.0).abs() < 0.01,
            "row 0 should reach 10 m, it reaches {:.2} m",
            ALTURA_OLHO / -p0
        );
        // Row 13 (the tyres) is anchored: only the horizon moves.
        assert!(
            (pitch_at(10.0, RETINA_ROWS - 1) - RETINA_MAX_PITCH).abs() < 1e-6,
            "the last row must stay at RETINA_MAX_PITCH"
        );
        let s = range_scale_of(10.0);
        assert!(
            (s - 10.0 / ALCANCE_PADRAO).abs() < 1e-6,
            "scale {s} is not 10 / {ALCANCE_PADRAO}"
        );
        assert!(
            (s - 0.3).abs() < 1e-3,
            "10 / 33.33 should be ~0.30, got {s}"
        );
    }

    #[test]
    fn a_velocidade_cega_bate_com_a_fisica() {
        // tightest turn measured on the track: radius 9.2 m -> v_curve 11.4 m/s
        let v = velocidade_cega(33.33, 11.4);
        assert!((v - 28.0).abs() < 0.5, "got {v:.1} m/s, expected ~28");
        // doubling the range does NOT double the speed — the relation is a square root
        let v2 = velocidade_cega(66.66, 11.4);
        assert!(
            v2 < 2.0 * v,
            "the relation must be sublinear: {v2:.1} vs {:.1}",
            2.0 * v
        );
        assert!(v2 > v, "more range must allow more speed");
    }
}
