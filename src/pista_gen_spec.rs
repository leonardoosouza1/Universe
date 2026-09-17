//! `WorldSpec` -> waypoints, split out of `pista_gen.rs` to keep that file under the
//! 600-line cap (`docs/STYLE.md`) — reuses `pista_gen`'s private-to-the-crate pieces
//! (`Rng`, `tracado`, `SPAN_MAX_CELLS`) rather than duplicating the search machinery.

use bevy::math::Vec2;

use crate::pista_gen::{densificar, medir, tracado, Rng, CELULA, SPAN_MAX_CELLS};

/// Generates waypoints for a `WorldSpec` (`src/world_spec.rs`, `--world` flag) whose
/// non-seed fields are NOT all `None` — the seed-only case is handled by the caller
/// (`world::track_waypoints_for_spec`), which defers straight to
/// `track_waypoints_for_seed` so `--world seed=8005` stays byte-identical to
/// `--tracks 8005`.
///
/// What each field actually moves, honestly:
/// - `curves`: sets `tracado`'s vertex count directly — REAL.
/// - `length`: rejection-sampled against `medir().comprimento` within ±35% — REAL,
///   approximate (the search gives up after `ATTEMPTS` and returns its closest miss).
/// - `radius`: `tracado`'s `raio_base` is scaled from the target and the search keeps
///   the candidate whose measured `r_min` is closest to it — REAL but approximate, and
///   sometimes UNREACHABLE: for seed 8005's own default family (6 vertices, ~1639 m)
///   a 20 000-attempt sweep never found better than r_min≈14.4 m while respecting the
///   curve-count/length defaults below — the seed's own default (12.6 m) is already
///   near that family's floor, so `radius=9` alone cannot be honoured there without
///   also moving `curves`/`length` away from their defaults. The physical floor
///   (`r_min >= 6.0`, `pista_gen`'s own guard) always wins when it IS reachable.
/// - `width`: NOT used here — `medir()` takes it only to compute `d_freio`, which this
///   search does not target. The caller (`world::VoxelWorld::new_from_spec`) applies
///   `spec.width` to the voxel grid itself, so the requested width still reaches the
///   game; it just does not steer which geometry gets generated.
/// - `holes`: NOT used here at all — hole placement is a `world.rs` concern
///   (`track_holes_with_density`), applied after these waypoints exist.
///
/// `default_curves`/`default_length_m`/`default_wps_cells` are the caller's
/// measurement of what THIS seed's *default* generator
/// (`world::track_waypoints_for_seed`) actually produces
/// (`world::track_waypoints_for_spec` computes them via `pista_gen::medir` before
/// calling here) — `WorldSpec`'s own contract is "key omitted = today's hardcoded
/// value", so an omitted `curves`/`length` must target the seed's own default, not a
/// generic constant, and the search must never hand back something WORSE (a wider
/// `r_min`) than that default already is. Found broken 2026-09-17: a fixed default of
/// "10 curves, no length target, `raio_base` 30-200" made
/// `--world seed=8005,radius=9` (only `radius` given) return a 17-curve, 622 m track
/// for a seed whose default is 6 curves at ~1639 m — only the radius floor was
/// supposed to move.
pub fn waypoints_for_spec(
    spec: &crate::world_spec::WorldSpec,
    default_curves: usize,
    default_length_m: f32,
    default_wps_cells: &[Vec2],
) -> Vec<Vec2> {
    const ATTEMPTS: u64 = 4000;
    let width_for_measure = spec.width.unwrap_or(28.0);
    let target_radius = spec.radius.unwrap_or(14.0).max(6.0);
    let target_curves = spec.curves.unwrap_or(default_curves).clamp(4, 24);
    let target_length = spec.length.unwrap_or(default_length_m);

    // Seed the search with the seed's OWN default geometry, not an empty `best`: the
    // loop below only replaces it with something that measures BETTER (smaller
    // `r_min`) while still respecting `target_curves`/`target_length` — so a request
    // this family cannot satisfy falls back to the known-good default instead of to
    // whatever random candidate the loop happened to still be holding when it ran out
    // of attempts (that used to REGRESS past the default: r_min=25 m instead of the
    // default's own 13 m — the bug this whole function's default-derivation fixes).
    let default_d = medir(default_wps_cells, width_for_measure, 40.0);
    let mut best_wps: Vec<Vec2> = default_wps_cells.to_vec();
    let mut best_r_min = default_d.r_min;

    // `target_radius` is a ceiling to beat, not a value to match: "radius=9" means
    // "give me a track with a curve at least as tight as 9 m" (punish harder), so any
    // candidate with `r_min <= target_radius` (and `>= 6.0`, the physical floor)
    // satisfies the request. Vertex count stays within +3 of `target_curves` so the
    // accepted/best candidate's measured curve count actually tracks it, and
    // `raio_base` (the polygon's overall scale — `tracado` places every vertex on a
    // near-circle of that radius) carries the search for `target_radius`, both
    // varying per attempt over the sweep (4000 attempts), same discipline as
    // `pista_gen::gerar` uses for `d_freio`.
    for t in 0..ATTEMPTS {
        let s = spec
            .seed
            .wrapping_mul(2654435761)
            .wrapping_add(t)
            .wrapping_add(0xD15C0_u64); // distinct sub-stream from `gerar`/`gerar_perfil`
        let mut r = Rng::novo(s);
        let n_retas = 1 + (r.proximo() * 3.0) as usize;
        let n_vertices = target_curves + (r.proximo() * 3.0) as usize;
        let raio_base = r.faixa(15.0, 200.0);
        let wps = tracado(s, n_vertices, n_retas, raio_base);
        let d = medir(&wps, width_for_measure, 40.0);
        if d.r_min < 6.0 || d.comprimento < 400.0 {
            continue;
        }
        let (mut mn, mut mx) = (Vec2::new(f32::MAX, f32::MAX), Vec2::new(f32::MIN, f32::MIN));
        for p in &wps {
            mn = mn.min(*p);
            mx = mx.max(*p);
        }
        let span = mx - mn;
        if span.x.max(span.y) > SPAN_MAX_CELLS {
            continue;
        }
        if (d.comprimento - target_length).abs() > target_length * 0.35 {
            continue;
        }
        if d.r_min <= target_radius {
            let wps_m: Vec<Vec2> = wps.iter().map(|p| *p * CELULA).collect();
            return densificar(&wps_m, 23.0);
        }
        if d.r_min < best_r_min {
            best_wps = wps;
            best_r_min = d.r_min;
        }
    }
    // No candidate reached the requested tightness in `ATTEMPTS` tries — report the
    // miss loudly (same discipline as `pista_gen::gerar_perfil`'s fallback). `best_wps`
    // is either a genuine improvement the search found, or (if nothing beat it) the
    // seed's own default geometry, untouched — never a silent "success", and never a
    // regression below what the seed already had.
    eprintln!(
        "[world_spec] seed={} radius target {target_radius:.1}m not reached — \
         tightest candidate found has r_min={best_r_min:.1}m after {ATTEMPTS} attempts \
         (seed's own default is r_min={:.1}m)",
        spec.seed, default_d.r_min
    );
    // Always return METRES here (this function's one output contract, honoured on
    // every path) so the caller's cells-recentring helper (`world.rs`, right after
    // its call into this function) wraps it exactly once — `best_wps` may be a
    // genuine improvement the search found, or (if nothing beat it)
    // `default_wps_cells` unchanged; either way it goes through the same metres round
    // trip, which is idempotent on an already-fitted track (same centre, same span,
    // its own scale factor clamped to 1.0) up to floating-point rounding.
    let wps_m: Vec<Vec2> = best_wps.iter().map(|p| *p * CELULA).collect();
    densificar(&wps_m, 23.0)
}
