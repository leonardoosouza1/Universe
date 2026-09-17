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
///   the candidate whose measured `r_min` is closest to it — REAL but approximate:
///   `raio_base` sets the OVERALL track scale, not any one curve's radius, so this is a
///   correlation the search exploits by rejection, not a hard geometric guarantee.
///   The physical floor (`r_min >= 6.0`, `pista_gen`'s own guard) always wins.
/// - `width`: NOT used here — `medir()` takes it only to compute `d_freio`, which this
///   search does not target. The caller (`world::VoxelWorld::new_from_spec`) applies
///   `spec.width` to the voxel grid itself, so the requested width still reaches the
///   game; it just does not steer which geometry gets generated.
/// - `holes`: NOT used here at all — hole placement is a `world.rs` concern
///   (`track_holes_with_density`), applied after these waypoints exist.
pub fn waypoints_for_spec(spec: &crate::world_spec::WorldSpec) -> Vec<Vec2> {
    const ATTEMPTS: u64 = 4000;
    let width_for_measure = spec.width.unwrap_or(28.0);
    let target_radius = spec.radius.unwrap_or(14.0).max(6.0);
    let target_curves = spec.curves.unwrap_or(10).clamp(4, 24);
    let target_length = spec.length;

    // `target_radius` is a ceiling to beat, not a value to match: "radius=9" means
    // "give me a track with a curve at least as tight as 9 m" (punish harder), so any
    // candidate with `r_min <= target_radius` (and `>= 6.0`, the physical floor)
    // satisfies the request. `raio_base` (tracado's overall scale) barely moves
    // `r_min` — what actually produces a tight local turn is vertex-angle variance,
    // which grows with `n_vertices` — so both `n_vertices` and `raio_base` vary
    // per attempt and the search relies on natural variance plus volume (4000
    // attempts), exactly like `pista_gen::gerar` does for `d_freio`.
    let mut best: Option<(Vec<Vec2>, f32)> = None; // (cells, r_min) — smallest r_min kept
    for t in 0..ATTEMPTS {
        let s = spec
            .seed
            .wrapping_mul(2654435761)
            .wrapping_add(t)
            .wrapping_add(0xD15C0_u64); // distinct sub-stream from `gerar`/`gerar_perfil`
        let mut r = Rng::novo(s);
        let n_retas = 1 + (r.proximo() * 3.0) as usize;
        // More vertices than requested widen the search for a chance local tight
        // turn without changing the reported curve *count* target too much.
        let n_vertices = target_curves + (r.proximo() * (target_curves as f32)) as usize;
        let raio_base = r.faixa(30.0, 200.0);
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
        if let Some(l) = target_length {
            if (d.comprimento - l).abs() > l * 0.35 {
                continue;
            }
        }
        if d.r_min <= target_radius {
            let wps_m: Vec<Vec2> = wps.iter().map(|p| *p * CELULA).collect();
            return densificar(&wps_m, 23.0);
        }
        if best.as_ref().map_or(true, |(_, r_min)| d.r_min < *r_min) {
            best = Some((wps, d.r_min));
        }
    }
    // No candidate reached the requested tightness in `ATTEMPTS` tries — report the
    // miss loudly (same discipline as `pista_gen::gerar_perfil`'s fallback) and return
    // the tightest candidate found, rather than silently claiming the request was met.
    match best {
        Some((wps, r_min)) => {
            eprintln!(
                "[world_spec] seed={} radius target {target_radius:.1}m not reached — \
                 tightest candidate found has r_min={r_min:.1}m after {ATTEMPTS} attempts",
                spec.seed
            );
            let wps_m: Vec<Vec2> = wps.iter().map(|p| *p * CELULA).collect();
            densificar(&wps_m, 23.0)
        }
        None => {
            // Last-resort geometry: same shape family (`tracado`), generic parameters
            // known to clear the physical floor (matches `pista_gen::gerar`'s own
            // defaults). Does NOT call into `world.rs` — `pista_gen`/`pista_gen_spec`
            // stay a dependency of `world`, never the other way (see the module doc
            // comment at the top of `pista_gen.rs`).
            eprintln!(
                "[world_spec] seed={}: no candidate cleared the physical floor \
                 (r_min>=6m, length>=400m, span<={SPAN_MAX_CELLS} cells) in {ATTEMPTS} \
                 attempts — falling back to a generic (non-targeted) track",
                spec.seed
            );
            let wps = tracado(spec.seed, 8, 2, 60.0);
            let wps_m: Vec<Vec2> = wps.iter().map(|p| *p * CELULA).collect();
            densificar(&wps_m, 23.0)
        }
    }
}
