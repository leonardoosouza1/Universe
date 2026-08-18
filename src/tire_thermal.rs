//! Tyre thermal model — heat accumulation, Newtonian cooling and the grip penalty.
//!
//! Why this module exists: the live GPU pipeline does not simulate tyre temperature at
//! all (`physics_c.wgsl:290` passes it through untouched), and the CPU/Bevy model that
//! does simulate it is dominated by its own cooling term. Measured, with braking
//! excluded, from the coefficients at `src/physics.rs:692-696`:
//!
//! ```text
//! heat = (|slip_ratio| + |slip_angle|) * load * 2e-6 * dt
//! cool = 0.02 * (1 + speed * 0.01) * dt          // CONSTANT in temperature
//!
//! load=3000 slip=0.10 v=18.75 -> heat/cool = 0.025
//! load=3000 slip=0.40 v=18.75 -> heat/cool = 0.101
//! load=4500 slip=0.60 v=18.75 -> heat/cool = 0.227
//! break-even needs slip*load = 11875, i.e. |slip| ~ 4.0 at load 3000 — unreachable
//! ```
//!
//! So the tyre pins at the cold rail 0.0 for the whole race and the grip factor
//! `clamp(1 - |0 - 0.5| * 0.5, 0.7, 1.0)` (`src/physics.rs:566,574`) is the CONSTANT
//! 0.75. That is a flat -25% grip wearing a thermal costume: a DC offset, not a
//! dynamic. Two failures follow from it, and both are the point of this module:
//!
//! 1. A/B of "thermal on" vs "thermal off" measures *less grip everywhere*, not
//!    temperature dynamics — the confound the DC-gain lesson already cost this project.
//! 2. A constant offset is captured perfectly by a per-tick static regression, which is
//!    why the closed-form ridge pilot beats the evolved champion (`docs/DESTILABILITY`).
//!    A state that integrates over ~47 ticks is not.
//!
//! This module is PURE, in the pattern of `resources.rs` / `predator.rs` /
//! `gpu_evolve/escada.rs`: no I/O, no statics, no CLI knowledge, no GPU, no clock.
//!
//! # Integrator contract (src/bin/gpu_evolve/, NOT wired by this module)
//!
//! - Per being, per tick, CPU-side: `temp = step_temp(temp, slip, speed, mu, DT, &p)`
//!   with `DT = 1/64.0`, then multiply the friction coefficient by
//!   `grip_factor(temp, &p)` before the tyre force is computed.
//! - `slip` is the dimensionless aggregate `|slip_ratio| + |slip_angle_rad|` — the same
//!   sum `src/physics.rs:692` already forms. `speed` is m/s, `mu` is the friction
//!   coefficient actually applied.
//! - Feed back the mu you USED (post-factor). Heat is proportional to friction power,
//!   so a tyre that lost grip also generates less heat: that negative feedback is what
//!   bounds the overheat spiral, together with the hard `floor`.
//! - Temperature is NORMALISED, `0.0 = ambient` to `1.0 = T_SCALE_CELSIUS`. That is the
//!   scale ch70 already uses (`perception_temporal.wgsl:82,364` — "media_temp/120°C"),
//!   so the value can be fed to the brain with no rescaling. NOTE the pre-existing
//!   conflict: `src/ui.rs:237` prints `tire_temp * 100.0` as °C. The registry scale
//!   (120) wins here because that is the one that reaches the brain.
//! - `optimal` defaults to 0.5 to stay the single source of truth already set at
//!   `src/gpu/physics.rs:153` and `src/prefabs.rs:54`.
//! - Before any battery: call `normalize_dc` (see the DC-gain section) so the ON arm
//!   has the SAME mean grip as the OFF arm and the A/B isolates the dynamic.
//! - Flag-gated, off by default: this changes the input distribution, so an old
//!   champion needs a paired A/B (CLAUDE.md rule 3).

use crate::rng::Lcg;

/// Celsius at normalised temperature 1.0. Matches the ch70 normaliser documented in
/// `src/gpu/perception_temporal.wgsl:82`, so `grip` and perception share one scale.
pub const T_SCALE_CELSIUS: f32 = 120.0;

/// Ambient / resting temperature, normalised (~18 °C). A parked tyre decays to this.
pub const DEFAULT_AMBIENT: f32 = 0.15;
/// Peak-grip temperature, normalised (60 °C). Same value as `gpu/physics.rs:153`.
pub const DEFAULT_OPTIMAL: f32 = 0.5;
/// Friction-heating coefficient: normalised temperature per second, per unit of
/// `mu * slip * speed`. Calibrated so cruise runs cold and a proper corner arrives at
/// the optimum — see `equilibrium_temp` for the closed form and the table below.
pub const DEFAULT_GEN_COEFF: f32 = 0.145;
/// Cooling at standstill, per second (radiative + conduction into the rim).
pub const DEFAULT_COOL_BASE: f32 = 0.80;
/// Extra cooling per m/s of airflow, per second.
pub const DEFAULT_COOL_SPEED: f32 = 0.03;
/// Gaussian half-width BELOW the optimum: a cold tyre loses grip gently.
pub const DEFAULT_WIDTH_COLD: f32 = 0.35;
/// Gaussian half-width ABOVE the optimum. Deliberately ~2x narrower than the cold
/// side: overheating is punished about twice as fast per degree, which is both real
/// tyre behaviour and the asymmetry that makes over-driving a mistake.
pub const DEFAULT_WIDTH_HOT: f32 = 0.18;
/// Worst-case grip multiplier. Kept at the legacy floor (`src/physics.rs:574`
/// `clamp(0.7, 1.0)`) so tyre forces stay inside the envelope the pipeline is already
/// validated against, and so the overheat feedback loop cannot run away to zero grip.
pub const DEFAULT_FLOOR: f32 = 0.70;
/// Hard cap on the applied factor. Must exceed the DC trim (measured ~1.137 over the
/// default band) or `normalize_dc` cannot reach unit mean.
pub const DEFAULT_MAX_FACTOR: f32 = 1.20;
/// Default spread of the per-being optimum under `heterogeneous`, normalised units.
pub const DEFAULT_OPTIMAL_SPREAD: f32 = 0.08;

/// Dose knobs of the thermal pressure. Owned by the CLI/statics on the integrator
/// side; this module only ever reads them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThermalParams {
    /// Resting temperature, normalised. Temperature never falls below it.
    pub ambient: f32,
    /// Peak-grip temperature, normalised.
    pub optimal: f32,
    /// Heating coefficient, per second per unit `mu * slip * speed`.
    pub gen_coeff: f32,
    /// Cooling rate at standstill, per second.
    pub cool_base: f32,
    /// Additional cooling per m/s, per second.
    pub cool_speed: f32,
    /// Gaussian half-width below the optimum.
    pub width_cold: f32,
    /// Gaussian half-width above the optimum.
    pub width_hot: f32,
    /// Minimum grip multiplier (the curve's asymptote on both sides).
    pub floor: f32,
    /// Maximum applied grip multiplier, after the DC trim.
    pub max_factor: f32,
    /// DC normaliser. 1.0 leaves the raw curve alone; `normalize_dc` sets it so the
    /// mean applied factor over a reference band is exactly 1.0.
    pub gain_trim: f32,
}

impl Default for ThermalParams {
    fn default() -> Self {
        Self {
            ambient: DEFAULT_AMBIENT,
            optimal: DEFAULT_OPTIMAL,
            gen_coeff: DEFAULT_GEN_COEFF,
            cool_base: DEFAULT_COOL_BASE,
            cool_speed: DEFAULT_COOL_SPEED,
            width_cold: DEFAULT_WIDTH_COLD,
            width_hot: DEFAULT_WIDTH_HOT,
            floor: DEFAULT_FLOOR,
            max_factor: DEFAULT_MAX_FACTOR,
            gain_trim: 1.0,
        }
    }
}

impl ThermalParams {
    /// Per-being thermal personality, derived deterministically from `(seed, index)`.
    ///
    /// Only the OPTIMUM moves, by up to `spread` in normalised units; the heating and
    /// cooling coefficients stay shared. Same `(seed, index)` reproduces the same
    /// params byte-identically — the D2 comparability contract.
    ///
    /// The caller MUST follow this with `normalize_dc` (`heterogeneous_normalized`
    /// does both). Heterogenising a parameter without re-normalising the DC gain is
    /// the mistake this project already paid for: a being whose optimum sits away from
    /// the band the track actually visits is simply slower on average, so selection
    /// picks the PARAMETER instead of the behaviour, and the knob becomes a fitness
    /// handicap wearing a diversity costume.
    pub fn heterogeneous(seed: u64, index: usize, spread: f32, base: &ThermalParams) -> Self {
        // Domain-separated from the other seed users (resources.rs, predator.rs). The
        // index is mixed by an odd multiplier rather than XOR-ed in: XOR would only
        // touch the low 6 bits for a 48-being population, and leaning on the LCG to
        // diffuse them is how two beings end up with byte-identical "personalities" —
        // the identical-numbers canary, but silent (CLAUDE.md rule 2).
        let mut rng =
            Lcg::new(seed ^ 0x7_E11A_0000 ^ (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        Self {
            optimal: (base.optimal + rng.symmetric() * spread).clamp(0.05, 0.95),
            ..*base
        }
    }

    /// `heterogeneous` followed by `normalize_dc` over `band` — the form the integrator
    /// should call, because a heterogeneous population with un-normalised DC gain
    /// measures the handicap, not the dynamic.
    pub fn heterogeneous_normalized(
        seed: u64,
        index: usize,
        spread: f32,
        base: &ThermalParams,
        band: &[f32],
    ) -> Self {
        normalize_dc(&Self::heterogeneous(seed, index, spread, base), band)
    }
}

/// Normalised temperature to degrees Celsius, for logs and telemetry only.
pub fn celsius(temp: f32) -> f32 {
    temp * T_SCALE_CELSIUS
}

/// One tick of the thermal state.
///
/// ```text
/// heat(t) = gen_coeff * mu * slip * speed          (friction power, P = mu*N*v_slip)
/// cool(t) = (cool_base + cool_speed * speed) * (temp - ambient)      (Newton's law)
/// temp'   = clamp(temp + (heat - cool) * dt, ambient, 1.0)
/// ```
///
/// The cooling term is PROPORTIONAL to the excess over ambient. That single change is
/// what separates this from the legacy model: a constant cooling term (as in
/// `src/physics.rs:694`) has no equilibrium, so it drives the state onto a rail and
/// holds it there. Newtonian cooling gives a finite equilibrium that depends on how
/// the being is driving, with time constant `1 / (cool_base + cool_speed * speed)` —
/// 0.73 s at 18.75 m/s, about 47 ticks at 64 Hz. That horizon is the point: a
/// per-tick static regression on the sensory wave cannot represent it.
///
/// `slip` is `|slip_ratio| + |slip_angle_rad|`; `speed` is m/s; `mu` is the friction
/// coefficient actually applied this tick. Negative inputs are clamped to zero, so a
/// caller passing a signed slip cannot cool the tyre by sliding.
pub fn step_temp(temp: f32, slip: f32, speed: f32, mu: f32, dt: f32, p: &ThermalParams) -> f32 {
    let slip = slip.max(0.0);
    let speed = speed.max(0.0);
    let mu = mu.max(0.0);
    let heat = p.gen_coeff * mu * slip * speed;
    let cool = (p.cool_base + p.cool_speed * speed) * (temp - p.ambient);
    (temp + (heat - cool) * dt).clamp(p.ambient, 1.0)
}

/// Closed-form steady state of `step_temp` for a sustained regime — the ceiling of the
/// metric, computable BEFORE any run (CLAUDE.md rule 4).
///
/// Measured with the defaults (`mu = 1.0`, speed 18.75 m/s unless stated):
///
/// | regime                  | slip | u*    | grip (raw) |
/// |-------------------------|------|-------|------------|
/// | idle (parked)           | 0.00 | 0.150 | 0.810      |
/// | cruise                  | 0.05 | 0.250 | 0.880      |
/// | proper corner           | 0.15 | 0.449 | 0.994      |
/// | over-driven corner      | 0.30 | 0.749 | 0.745      |
/// | sustained slide         | 0.60 | 1.000 | 0.700      |
/// | corner at 30 m/s        | 0.15 | 0.534 | 0.990      |
///
/// So the tyre only reaches peak grip if the being drives it there, and over-driving
/// costs as much as never warming up. Neither end is reachable by a constant policy.
pub fn equilibrium_temp(slip: f32, speed: f32, mu: f32, p: &ThermalParams) -> f32 {
    let slip = slip.max(0.0);
    let speed = speed.max(0.0);
    let mu = mu.max(0.0);
    let cool = p.cool_base + p.cool_speed * speed;
    if cool <= 0.0 {
        return 1.0;
    }
    (p.ambient + p.gen_coeff * mu * slip * speed / cool).clamp(p.ambient, 1.0)
}

/// The grip curve before the DC trim: an asymmetric Gaussian bell on `[floor, 1.0]`,
/// peaking at exactly 1.0 on `optimal`.
///
/// ```text
/// f_raw(u) = floor + (1 - floor) * exp(-((u - optimal) / w)^2)
/// w = width_cold if u < optimal else width_hot        (width_hot < width_cold)
/// ```
///
/// A bell rather than the legacy triangle (`1 - |u - opt| * 0.5`) because the triangle
/// has a discontinuous derivative at the optimum and a hard clamp on both sides, which
/// makes the gradient of grip with respect to driving style vanish exactly where the
/// interesting decisions are. The bell is smooth everywhere and asymptotes to `floor`.
pub fn grip_factor_raw(temp: f32, p: &ThermalParams) -> f32 {
    let w = if temp < p.optimal {
        p.width_cold
    } else {
        p.width_hot
    };
    let z = (temp - p.optimal) / w.max(1e-6);
    p.floor + (1.0 - p.floor) * (-z * z).exp()
}

/// The grip multiplier to apply to `mu`, after the DC trim and the safety cap.
///
/// With `gain_trim > 1` this EXCEEDS 1.0 inside the optimal window, and that is
/// deliberate: see `normalize_dc`.
pub fn grip_factor(temp: f32, p: &ThermalParams) -> f32 {
    (grip_factor_raw(temp, p) * p.gain_trim).clamp(p.floor * p.gain_trim, p.max_factor)
}

/// A default sample of the temperatures a race actually visits, for `normalize_dc`.
///
/// Spans ambient to the over-driven equilibrium (0.85). The integrator should prefer a
/// histogram measured from a pilot run; this is the honest fallback when there is none.
pub fn reference_band() -> Vec<f32> {
    let (lo, hi, n) = (DEFAULT_AMBIENT, 0.85, 201);
    (0..n)
        .map(|i| lo + (hi - lo) * i as f32 / (n - 1) as f32)
        .collect()
}

/// Mean applied grip factor over `temps` — the DC gain of the model as configured.
///
/// This is the number that must be 1.0 before a battery. If it is 0.88, then "thermal
/// on" is silently also "12% less grip everywhere", and any effect measured against
/// the OFF arm is that offset rather than the dynamic.
pub fn dc_gain(temps: &[f32], p: &ThermalParams) -> f32 {
    if temps.is_empty() {
        return 1.0;
    }
    temps.iter().map(|&t| grip_factor(t, p)).sum::<f32>() / temps.len() as f32
}

/// Return a copy of `p` whose `gain_trim` makes `dc_gain(band, ..) == 1.0`.
///
/// Why unit DC gain matters more than it looks: the raw bell peaks at 1.0 and decays,
/// so its mean over any realistic band is BELOW 1.0 (0.879 over `reference_band` with
/// the defaults). Shipping that means the thermal flag is really a global grip nerf,
/// and the A/B against the OFF arm cannot tell "temperature dynamics help" from "less
/// grip is harder". Trimming to unit mean makes the two arms cost the same on average
/// and differ only in WHEN the grip is available — which is the hypothesis under test.
/// The consequence is that a tyre in its window is slightly BETTER than the baseline
/// (trim ~1.137); that is the model giving back what it takes.
///
/// Fixed-point iteration because `grip_factor` clamps at `max_factor`, so the trim and
/// the measured mean are not exactly reciprocal. Converges in 2-3 passes; 8 is slack.
pub fn normalize_dc(p: &ThermalParams, band: &[f32]) -> ThermalParams {
    let mut out = *p;
    if band.is_empty() {
        return out;
    }
    out.gain_trim = 1.0;
    for _ in 0..8 {
        let g = dc_gain(band, &out);
        if g <= 1e-6 {
            break;
        }
        let next = out.gain_trim / g;
        if (next - out.gain_trim).abs() < 1e-7 {
            break;
        }
        out.gain_trim = next;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT: f32 = 1.0 / 64.0;

    /// Determinism in the seed, and the DC normalisation that must follow heterogeneity.
    #[test]
    fn heterogeneous_params_are_deterministic_in_seed_and_index() {
        let base = ThermalParams::default();
        let a = ThermalParams::heterogeneous(77, 3, DEFAULT_OPTIMAL_SPREAD, &base);
        let b = ThermalParams::heterogeneous(77, 3, DEFAULT_OPTIMAL_SPREAD, &base);
        assert_eq!(
            a, b,
            "same seed and index must reproduce params identically"
        );

        let other_seed = ThermalParams::heterogeneous(78, 3, DEFAULT_OPTIMAL_SPREAD, &base);
        assert_ne!(
            a.optimal, other_seed.optimal,
            "a different seed must move the optimum"
        );
        let other_index = ThermalParams::heterogeneous(77, 4, DEFAULT_OPTIMAL_SPREAD, &base);
        assert_ne!(
            a.optimal, other_index.optimal,
            "a different being must differ"
        );

        // No two beings of a 48-strong population may share an optimum: a silent seed
        // collision would look like diversity and behave like a monoculture.
        let mut optima: Vec<f32> = (0..48)
            .map(|i| ThermalParams::heterogeneous(77, i, DEFAULT_OPTIMAL_SPREAD, &base).optimal)
            .collect();
        optima.sort_by(f32::total_cmp);
        for w in optima.windows(2) {
            assert!(
                w[0] != w[1],
                "two beings share the optimum {}: seed collision",
                w[0]
            );
        }

        // The spread is respected and the shared coefficients are untouched.
        for i in 0..48 {
            let p = ThermalParams::heterogeneous(77, i, DEFAULT_OPTIMAL_SPREAD, &base);
            assert!(
                (p.optimal - base.optimal).abs() <= DEFAULT_OPTIMAL_SPREAD + 1e-6,
                "being {i} optimum {} left the +-{DEFAULT_OPTIMAL_SPREAD} band",
                p.optimal
            );
            assert_eq!(p.gen_coeff, base.gen_coeff, "only the optimum may vary");
            assert_eq!(p.cool_base, base.cool_base, "only the optimum may vary");
        }
    }

    /// The regression that documents WHY this module exists: with the legacy
    /// coefficients (`src/physics.rs:692-696`, braking excluded) the tyre never leaves
    /// the cold rail, so the grip factor is a constant.
    #[test]
    fn legacy_coefficients_pin_the_tyre_at_the_cold_rail() {
        let (load, slip, speed) = (4500.0f32, 0.60f32, 18.75f32); // generous: heavy slide
        let mut temp = 0.30f32; // the legacy initial value, src/physics.rs:27
        for _ in 0..20_000 {
            let heat = slip * load * 0.000_002 * DT;
            let cool = 0.02 * (1.0 + speed * 0.01) * DT;
            temp = (temp + heat - cool).clamp(0.0, 1.0);
        }
        assert_eq!(temp, 0.0, "legacy model must sit on the cold rail");
        // ...and therefore the legacy grip factor is a constant 0.75, a DC offset.
        let legacy_factor: f32 = (1.0 - (temp - 0.5).abs() * 0.5).clamp(0.7, 1.0);
        assert!((legacy_factor - 0.75).abs() < 1e-6, "got {legacy_factor}");

        // This module, same regime, does NOT pin: it settles at a driving-dependent
        // equilibrium strictly inside the range.
        let p = ThermalParams::default();
        let mut t = p.ambient;
        for _ in 0..20_000 {
            t = step_temp(t, 0.15, speed, 1.0, DT, &p);
        }
        assert!(
            t > p.ambient + 0.05 && t < 1.0,
            "settled at {t}, still on a rail"
        );
    }

    /// Temperature converges to the closed form, from both directions, and stays bounded.
    #[test]
    fn temperature_converges_to_the_closed_form_equilibrium() {
        let p = ThermalParams::default();
        for &(slip, speed) in &[
            (0.05f32, 18.75f32),
            (0.15, 18.75),
            (0.30, 18.75),
            (0.15, 30.0),
        ] {
            let expected = equilibrium_temp(slip, speed, 1.0, &p);
            for &start in &[p.ambient, 1.0] {
                let mut t = start;
                for _ in 0..4_000 {
                    // 62 s at 64 Hz, ~85 time constants
                    t = step_temp(t, slip, speed, 1.0, DT, &p);
                    assert!(
                        (p.ambient..=1.0).contains(&t),
                        "temperature {t} left [{}, 1.0]",
                        p.ambient
                    );
                }
                assert!(
                    (t - expected).abs() < 1e-3,
                    "slip {slip} speed {speed} from {start}: settled {t}, closed form {expected}"
                );
            }
        }
        // The regimes must be ORDERED and distinct — identical numbers where they
        // should differ is the canary (CLAUDE.md rule 2).
        let cruise = equilibrium_temp(0.05, 18.75, 1.0, &p);
        let corner = equilibrium_temp(0.15, 18.75, 1.0, &p);
        let hard = equilibrium_temp(0.30, 18.75, 1.0, &p);
        assert!(cruise < corner && corner < hard, "{cruise} {corner} {hard}");
        assert!(
            (corner - p.optimal).abs() < 0.10,
            "a proper corner must reach the window"
        );
    }

    /// A parked being cools to ambient and stops there — no camping exploit, no drift.
    #[test]
    fn a_parked_tyre_decays_to_ambient_and_stays() {
        let p = ThermalParams::default();
        let mut t = 0.95f32;
        for _ in 0..6_000 {
            t = step_temp(t, 0.0, 0.0, 1.0, DT, &p);
        }
        assert!((t - p.ambient).abs() < 1e-4, "parked tyre settled at {t}");
        for _ in 0..1_000 {
            t = step_temp(t, 0.0, 0.0, 1.0, DT, &p);
            assert!(t >= p.ambient - 1e-6, "temperature fell below ambient: {t}");
        }
        // Zero mu means zero friction power: no heat regardless of slip and speed.
        assert_eq!(step_temp(p.ambient, 5.0, 50.0, 0.0, DT, &p), p.ambient);
        // Signed slip cannot be used to cool the tyre below the physical model.
        let hot = 0.8f32;
        assert_eq!(
            step_temp(hot, -3.0, 18.75, 1.0, DT, &p),
            step_temp(hot, 0.0, 18.75, 1.0, DT, &p),
            "negative slip must clamp to zero heat, not become negative heat"
        );
    }

    /// Curve shape: peak on the optimum, bounded by the floor, overheat punished harder.
    #[test]
    fn grip_peaks_at_the_optimum_and_overheat_costs_more_than_cold() {
        let p = ThermalParams::default(); // gain_trim = 1.0: raw shape
        assert!(
            (grip_factor(p.optimal, &p) - 1.0).abs() < 1e-6,
            "peak must be exactly 1.0"
        );

        // Monotone up to the optimum, monotone down after it.
        let mut prev = grip_factor(0.0, &p);
        let mut u = 0.0f32;
        while u < p.optimal - 1e-3 {
            u += 0.01;
            let g = grip_factor(u, &p);
            assert!(
                g >= prev - 1e-6,
                "cold side must rise: {prev} -> {g} at {u}"
            );
            prev = g;
        }
        let mut prev = grip_factor(p.optimal, &p);
        let mut u = p.optimal;
        while u < 1.0 {
            u += 0.01;
            let g = grip_factor(u, &p);
            assert!(g <= prev + 1e-6, "hot side must fall: {prev} -> {g} at {u}");
            prev = g;
        }

        // Bounded by the floor on both extremes.
        for &t in &[0.0f32, 1.0] {
            let g = grip_factor(t, &p);
            assert!(
                g >= p.floor - 1e-6 && g <= 1.0 + 1e-6,
                "factor {g} out of bounds at {t}"
            );
        }

        // Asymmetry: the same distance ABOVE the optimum costs strictly more grip.
        for d in [0.10f32, 0.20, 0.30] {
            let cold = grip_factor(p.optimal - d, &p);
            let hot = grip_factor(p.optimal + d, &p);
            assert!(
                hot < cold,
                "at +-{d}: hot {hot} must cost more than cold {cold}"
            );
        }
    }

    /// The DC-gain lesson, enforced: after `normalize_dc` both a homogeneous and a
    /// heterogeneous population have mean grip 1.0, so an A/B measures the dynamic and
    /// not a hidden global nerf.
    #[test]
    fn dc_normalization_gives_unit_mean_grip_for_every_being() {
        let band = reference_band();
        let base = ThermalParams::default();

        // The raw curve really is a nerf, which is the thing being corrected.
        let raw_gain = dc_gain(&band, &base);
        assert!(
            raw_gain < 0.95,
            "raw DC gain {raw_gain} — if it were already ~1.0 the trim would be pointless"
        );

        let tuned = normalize_dc(&base, &band);
        assert!(
            (dc_gain(&band, &tuned) - 1.0).abs() < 1e-4,
            "homogeneous DC gain {} is not unity",
            dc_gain(&band, &tuned)
        );
        assert!(
            tuned.gain_trim > 1.0,
            "trim must lift the curve, got {}",
            tuned.gain_trim
        );
        assert!(
            tuned.max_factor > tuned.gain_trim,
            "cap {} must clear the trim {} or unit mean is unreachable",
            tuned.max_factor,
            tuned.gain_trim
        );

        // Every being of a heterogeneous population pays the same on average: the
        // optimum moves, the handicap does not.
        for i in 0..48 {
            let p = ThermalParams::heterogeneous_normalized(
                77,
                i,
                DEFAULT_OPTIMAL_SPREAD,
                &base,
                &band,
            );
            let g = dc_gain(&band, &p);
            assert!(
                (g - 1.0).abs() < 1e-3,
                "being {i} has DC gain {g}, not unity"
            );
        }

        // Control: WITHOUT the normalisation the spread does bias beings against each
        // other — the failure mode the contract exists to prevent.
        let biased: Vec<f32> = (0..48)
            .map(|i| dc_gain(&band, &ThermalParams::heterogeneous(77, i, 0.30, &base)))
            .collect();
        let (lo, hi) = biased
            .iter()
            .fold((f32::MAX, f32::MIN), |(a, b), &g| (a.min(g), b.max(g)));
        assert!(
            hi - lo > 1e-3,
            "un-normalised heterogeneity should spread the mean grip ({lo}..{hi})"
        );
    }

    /// The overheat feedback loop is bounded: less grip means less friction power, and
    /// the floor caps the damage. A tyre cannot spiral to zero grip.
    #[test]
    fn overheat_feedback_is_bounded_by_the_floor() {
        let p = normalize_dc(&ThermalParams::default(), &reference_band());
        let mut t = p.ambient;
        let mut worst = f32::MAX;
        for _ in 0..20_000 {
            // Closed loop: mu is scaled by the grip the tyre currently has.
            let mu = 1.0 * grip_factor(t, &p);
            t = step_temp(t, 0.60, 25.0, mu, DT, &p);
            worst = worst.min(grip_factor(t, &p));
        }
        assert!(
            worst >= p.floor * p.gain_trim - 1e-6,
            "grip {worst} fell through the floor {}",
            p.floor * p.gain_trim
        );
        assert!(t <= 1.0, "temperature {t} exceeded the scale");
    }

    #[test]
    fn celsius_matches_the_channel_scale() {
        assert!((celsius(1.0) - T_SCALE_CELSIUS).abs() < 1e-6);
        assert!((celsius(DEFAULT_OPTIMAL) - 60.0).abs() < 1e-6);
    }
}
