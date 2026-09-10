//! The static map and the pure geometric queries a being's senses need every tick
//! (SPEC §2). No GPU, no I/O — `Trace::to_world` (`trace.rs`) is the only producer
//! in v0; T5's later WGSL port turns these same `Vec`s into read-only buffers
//! without changing this module (plan §1.3, §2).

use crate::constants::{GRID, N_TYPES, TEX};

/// The static map for one taxi world (SPEC §2.1). Buffers are row-major `[iy][ix]`
/// (SPEC §2.3, armadilha 3: swapping the axes produces a transposed map that
/// "almost works" — every value looks plausible, only the geometry is wrong).
#[derive(Debug, Clone, PartialEq)]
pub struct World {
    /// Side length of the square world, metres (`L`, SPEC §3.2 — 200.0 in the
    /// reference trace).
    pub side: f32,
    /// Terrain cost texture, `TEX × TEX` cells, index `iy * TEX + ix`
    /// (SPEC §2.1 `custo`, §2.3 `custo_em`).
    pub cost_tex: Vec<f32>,
    /// District id per cell, `GRID × GRID` cells, index `iy * GRID + ix`
    /// (SPEC §2.1 `campo_dist`, §2.3 `_cel`/`distrito`).
    pub district_field: Vec<u32>,
    /// District (Voronoi) centers, length `K` (SPEC §2.1 `centros`).
    pub centers: Vec<[f32; 2]>,
    /// Tax rate per district, length `K` (SPEC §2.1 `aliq`).
    pub tax: Vec<f32>,
    /// Base price multiplier per district, length `K` (SPEC §2.1 `pmul`/`pmul0`).
    pub price_mul_base: Vec<f32>,
    /// Rent per district, length `K` (SPEC §2.1 `alug`).
    pub rent: Vec<f32>,
    /// Passenger demand per district, length `K` (SPEC §2.1 `demanda`).
    pub demand: Vec<f32>,
    /// Price-cycle phase per district, length `K` (SPEC §2.1 `fase_preco`).
    pub price_phase: Vec<f32>,
    /// Service points, one `Vec` per type, in `TIPOS` order — posto, comida, agua,
    /// descanso, oficina (SPEC §0).
    pub services: [Vec<[f32; 2]>; N_TYPES],
}

impl World {
    /// Terrain cost at a world position (SPEC §2.3, `custo_em`).
    ///
    /// Uses a `TEX − 1 = 63` grid and **round-half-to-even**, NOT `f32::round`
    /// (SPEC §2.3, armadilha 2): `torch.round` on the Python side is banker's
    /// rounding, and `f32::round` is round-half-away-from-zero, so a position
    /// exactly on a cell boundary (`x/L·63 == k + 0.5`) would round to a DIFFERENT
    /// cell than the reference trace on roughly half of those boundaries.
    pub fn terrain_cost_at(&self, x: f32, y: f32) -> f32 {
        let ix = cell_index_round_half_even(x, self.side, TEX);
        let iy = cell_index_round_half_even(y, self.side, TEX);
        self.cost_tex[iy * TEX + ix]
    }

    /// District id at a world position (SPEC §2.3, `_cel`/`distrito`).
    ///
    /// Uses `GRID = 128` (a DIFFERENT resolution than `terrain_cost_at`'s `TEX`) and
    /// truncation (`floor`), not rounding — SPEC §2.3 armadilha 1 is explicit that
    /// the two lookups must stay separate, not share one grid or one rounding rule.
    pub fn district_at(&self, x: f32, y: f32) -> u32 {
        let ix = cell_index_floor(x, self.side, GRID);
        let iy = cell_index_floor(y, self.side, GRID);
        self.district_field[iy * GRID + ix]
    }

    /// Nearest point of each service type, scanning every point of that type
    /// exactly (SPEC §2.3 `_perto_exata`, `cfg.varredura = 'exata'` — the grid
    /// approximation `varredura = 'grade'` is out of scope for v0, SPEC §12).
    /// Returns `(points, distances)`, both indexed in `TIPOS` order (same order as
    /// `self.services`).
    ///
    /// A type with zero points returns `[0.0, 0.0]` and `f32::INFINITY` for that
    /// slot. The SPEC's Python side pads every type to the same length and masks
    /// empty slots with `1e9` (`campo.py:123-127`); the plan explicitly drops that
    /// mask in Rust because each type here scans only its own real count (plan
    /// §3.2) — but an ACTUALLY empty type (no real map has this, but a hand-built
    /// test fixture might) still needs an explicit "nothing found" value, and
    /// `f32::INFINITY` is the sentinel the SPEC already uses for the same idea
    /// elsewhere (`Config::fog_distance`, `nevoa`).
    pub fn nearest_services(&self, x: f32, y: f32) -> ([[f32; 2]; N_TYPES], [f32; N_TYPES]) {
        let mut points = [[0.0f32; 2]; N_TYPES];
        let mut dists = [f32::INFINITY; N_TYPES];
        for (t, type_points) in self.services.iter().enumerate() {
            let mut best_d = f32::INFINITY;
            let mut best_p = [0.0f32; 2];
            for p in type_points {
                let dx = p[0] - x;
                let dy = p[1] - y;
                let d = (dx * dx + dy * dy).sqrt();
                if d < best_d {
                    best_d = d;
                    best_p = *p;
                }
            }
            points[t] = best_p;
            dists[t] = best_d;
        }
        (points, dists)
    }
}

/// `ix = round_half_to_even(p.x / side · (cells − 1)).clamp(0, cells − 1)`
/// (SPEC §2.3, `custo_em`).
fn cell_index_round_half_even(coord: f32, side: f32, cells: usize) -> usize {
    let scaled = coord / side * (cells as f32 - 1.0);
    let rounded = round_half_to_even(scaled) as i64;
    rounded.clamp(0, cells as i64 - 1) as usize
}

/// `ix = floor(p.x / side · cells).clamp(0, cells − 1)` (SPEC §2.3, `_cel`).
fn cell_index_floor(coord: f32, side: f32, cells: usize) -> usize {
    let scaled = (coord / side * cells as f32).floor() as i64;
    scaled.clamp(0, cells as i64 - 1) as usize
}

/// Round-half-to-even ("banker's rounding"), matching `torch.round`'s semantics
/// (SPEC §2.3, armadilha 2). Written out explicitly, rather than calling the
/// standard library's `f32::round_ties_even` (stable since Rust 1.77 and available
/// here — `rustc 1.93.1`), per this task's brief.
fn round_half_to_even(x: f32) -> f32 {
    let floor = x.floor();
    let diff = x - floor;
    if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else {
        // Exactly halfway: round to whichever neighbour is even. `floor as i64` is
        // exact here — cell coordinates never approach 2^53, the range where an f32
        // could lose integer precision.
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{DIRS, G};

    /// A 4×4 toy grid (well under the real `TEX`/`GRID`) used only to check
    /// `round_half_to_even` in isolation, without paying for a full TEX×TEX buffer.
    #[test]
    fn round_half_to_even_matches_torch_round_semantics() {
        assert_eq!(round_half_to_even(0.5), 0.0, "0.5 rounds down to the even 0");
        assert_eq!(round_half_to_even(1.5), 2.0, "1.5 rounds up to the even 2");
        assert_eq!(round_half_to_even(2.5), 2.0, "2.5 rounds down to the even 2");
        assert_eq!(round_half_to_even(-0.5), 0.0, "-0.5 rounds to the even 0");
        assert_eq!(round_half_to_even(-1.5), -2.0, "-1.5 rounds to the even -2");
        assert_eq!(round_half_to_even(0.3), 0.0);
        assert_eq!(round_half_to_even(0.7), 1.0);
    }

    fn tiny_world() -> World {
        // TEX=64, GRID=128 are fixed crate-wide constants (SPEC §0) — every real
        // `World` carries full-size buffers. This fixture fills them uniformly and
        // overrides only the handful of cells each test actually reads, which is
        // cheap (TEX*TEX=4096, GRID*GRID=16384 f32/u32 — kilobytes, not a hand-typed
        // literal).
        World {
            side: 64.0,
            cost_tex: vec![1.0; TEX * TEX],
            district_field: vec![0; GRID * GRID],
            centers: vec![[0.0, 0.0]],
            tax: vec![0.0],
            price_mul_base: vec![1.0],
            rent: vec![0.0],
            demand: vec![0.0],
            price_phase: vec![0.0],
            services: [vec![], vec![], vec![], vec![], vec![]],
        }
    }

    #[test]
    fn terrain_cost_uses_sixty_three_and_ties_to_even() {
        // side = 128.0 (a power of two) so every division below is an EXACT f32
        // operation (halving/quartering just shifts the exponent, no rounding) and
        // the following multiplications by 63.0 stay within f32's 24-bit mantissa —
        // these three points are bit-exact, not "close enough".
        let mut w = tiny_world();
        w.side = 128.0;

        // Point A: x/L*63 = 100/128*63 = 49.21875 -> fractional part .21875 < 0.5,
        // rounds DOWN to 49.
        w.cost_tex[49] = 5.0;
        assert_eq!(w.terrain_cost_at(100.0, 0.0), 5.0, "49.21875 must round down to 49");

        // Point B: x/L*63 = 64/128*63 = 31.5 EXACTLY -> the midpoint of cells 31
        // and 32 the plan's acceptance test asks for. Ties to the EVEN neighbour,
        // 32. NOTE: for THIS specific grid (TEX-1 = 63 = 2^6-1), 31.5 is the ONLY
        // point in [0, L] reachable by exact f32 arithmetic where `x/L*63` lands on
        // a half-integer (a consequence of 63 being one less than a power of two;
        // proved by working the tie equation `63*m ≡ 2^(k-1) (mod 2^k)` for every
        // power-of-two side) — and its even neighbour (32) happens to also be the
        // round-AWAY-from-zero answer for a positive tie, so `f32::round` would
        // pass this specific assertion too. The case that actually tells the two
        // rules apart (an EVEN `n` before the tie, where they disagree) is covered
        // separately and unambiguously, with plain literals and no derived
        // coordinates, by `round_half_to_even_matches_torch_round_semantics` above
        // (`0.5 -> 0`, `2.5 -> 2`: both ties to an EVEN floor, which
        // round-away-from-zero would instead push up to 1 and 3).
        w.cost_tex[31] = 999.0; // the odd neighbour; must NOT be picked
        w.cost_tex[32] = 7.0;
        assert_eq!(
            w.terrain_cost_at(64.0, 0.0),
            7.0,
            "x/L*63 = 31.5 must read the even cell 32, not the odd cell 31"
        );

        // Point C: x/L*63 = 32/128*63 = 15.75 -> fractional part .75 > 0.5, rounds
        // UP to 16.
        w.cost_tex[16] = 9.0;
        assert_eq!(w.terrain_cost_at(32.0, 0.0), 9.0, "15.75 must round up to 16");
    }

    #[test]
    fn district_lookup_truncates_on_a_grid_of_one_twenty_eight() {
        let mut w = tiny_world();
        w.district_field[0 * GRID + 0] = 42; // iy=0, ix=0
        w.district_field[127 * GRID + 127] = 7; // iy=127, ix=127 (clamped corner)
        assert_eq!(w.district_at(0.0, 0.0), 42, "x=0 must land in cell 0");
        assert_eq!(
            w.district_at(w.side, w.side),
            7,
            "x=L must clamp to the last cell, 127, not overflow"
        );
        // A point strictly inside cell 63 (of 128) must NOT round up to 64.
        w.district_field[0 * GRID + 63] = 99;
        w.district_field[0 * GRID + 64] = 1;
        let x_in_63 = (63.9 / 128.0) * w.side;
        assert_eq!(
            w.district_at(x_in_63, 0.0),
            99,
            "truncation, not rounding: 63.9 must floor to cell 63"
        );
    }

    #[test]
    fn texture_rows_are_indexed_by_y_not_x() {
        // side = 128.0 again for exact arithmetic (see the tie test above).
        // x=2 -> x/L*63 = 2/128*63 = 0.984375 -> rounds up to column 1.
        // x=4 -> x/L*63 = 4/128*63 = 1.96875  -> rounds up to column 2.
        // Both fractional parts (.984375, .96875) sit far from the 0.5 boundary,
        // so there is no tie-rounding ambiguity here, only the axis-order question
        // this test exists to catch.
        let mut w = tiny_world();
        w.side = 128.0;
        w.cost_tex[1 * TEX + 2] = 111.0; // iy=1, ix=2
        w.cost_tex[2 * TEX + 1] = 222.0; // iy=2, ix=1
        assert_eq!(
            w.terrain_cost_at(4.0, 2.0), // x -> ix=2, y -> iy=1
            111.0,
            "(x in col 2, y in row 1) must read cost_tex[1*TEX+2], not [2*TEX+1]"
        );
        assert_eq!(
            w.terrain_cost_at(2.0, 4.0), // x -> ix=1, y -> iy=2
            222.0,
            "(x in col 1, y in row 2) must read cost_tex[2*TEX+1], not [1*TEX+2]"
        );

        // side = GRID = 128.0 makes district_at's transform the identity
        // (coord/side*GRID = coord/128*128 = coord), so x=5.x/y=3.x land exactly
        // and unambiguously in columns 5 / rows 3 by construction, no FP risk.
        let mut w2 = tiny_world();
        w2.side = GRID as f32;
        w2.district_field[3 * GRID + 5] = 3; // iy=3, ix=5
        w2.district_field[5 * GRID + 3] = 9; // iy=5, ix=3
        assert_eq!(
            w2.district_at(5.5, 3.5),
            3,
            "district_field must also be indexed [iy][ix], not [ix][iy]"
        );
        assert_eq!(w2.district_at(3.5, 5.5), 9, "and the transposed query must differ");
    }

    #[test]
    fn nearest_service_scans_every_point_of_its_type() {
        let mut w = tiny_world();
        // Type 0 (posto): three points, the middle one closest to the query.
        w.services[0] = vec![[0.0, 0.0], [10.0, 0.0], [100.0, 0.0]];
        let (points, dists) = w.nearest_services(9.0, 0.0);
        assert_eq!(points[0], [10.0, 0.0], "must find the middle point, not the first");
        assert!((dists[0] - 1.0).abs() < 1e-5);

        // A type with no points reports "not found", not a bogus zero distance.
        assert_eq!(dists[1], f32::INFINITY);
        assert_eq!(points[1], [0.0, 0.0]);
    }

    #[test]
    fn nearest_services_returns_every_type_in_tipos_order() {
        let mut w = tiny_world();
        for t in 0..N_TYPES {
            w.services[t] = vec![[t as f32 * 10.0, 0.0]];
        }
        let (points, _dists) = w.nearest_services(0.0, 0.0);
        for t in 0..N_TYPES {
            assert_eq!(points[t], [t as f32 * 10.0, 0.0], "type {t} out of order");
        }
    }

    // Sanity check that `constants::DIRS`/`G` are visible from here (T1's own
    // geometry module leans on them for future callers) and match SPEC §0.
    #[test]
    fn constants_reexports_are_wired() {
        assert_eq!(G, 9);
        assert_eq!(DIRS.len(), G);
    }
}
