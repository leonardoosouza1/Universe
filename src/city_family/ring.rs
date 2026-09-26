//! Ring family: one rounded-rectangle loop with one to three cardinal chords.
//!
//! Every seed gives a different loop size, corner radius, chord count and chord
//! placement. Chords meet the loop at T junctions and cross each other at X
//! junctions; the four corners are circular arcs, so a driver meets curves,
//! straights, T and X junctions and stations spread over several straights.
//! Sizes are chosen so the school's 60 m straight approaches exist before every
//! station and every junction. All numbers are metres in the x/z map frame.

use serde_json::json;

use super::{
    finish_plan, CityLayout, Junction, RoadPiece, SeededRng, CityPlanResult,
    MAX_ARC_RADIUS_M, MIN_ARC_RADIUS_M, MIN_STATION_SEGMENT_M,
};
use crate::world::VoxelWorld;

/// Loop side length range (centreline to centreline).
pub const SIDE_MIN_M: f32 = 480.0;
pub const SIDE_MAX_M: f32 = 720.0;
/// A chord keeps this straight length between its junction and the corner arc.
pub const CHORD_ARC_MARGIN_M: f32 = 70.0;
/// Parallel chords keep this centreline distance.
pub const CHORD_SEPARATION_M: f32 = 150.0;
/// Chord counts drawn by seed: 1..=3.
pub const CHORDS_MIN: u64 = 1;
pub const CHORDS_MAX: u64 = 3;
/// Stations kept per city; the grid city also has five.
pub const STATION_COUNT: usize = 5;
/// Road centrelines stay inside this margin of the 960 m map so the band,
/// the perimeter wall and the outer grass fit within the 3-cell border the grid
/// city also respects.
pub const MAP_MARGIN_M: f32 = 45.0;

/// Deterministic ring layout for a seed on a `size × cell` world.
pub fn layout(world: &VoxelWorld, seed: u64) -> Result<CityLayout, String> {
    let map_m = (world.size as f32 - 1.0) * world.cell_size;
    if map_m < SIDE_MIN_M + 2.0 * MAP_MARGIN_M {
        return Err(format!("ring city needs a map of at least {} m; got {map_m:.0} m", SIDE_MIN_M + 2.0 * MAP_MARGIN_M));
    }
    let mut rng = SeededRng::new(seed ^ 0x52494E47);
    let side_max = SIDE_MAX_M.min(map_m - 2.0 * MAP_MARGIN_M);
    let width = snap(rng.range(SIDE_MIN_M, side_max), world.cell_size);
    let height = snap(rng.range(SIDE_MIN_M, side_max), world.cell_size);
    let radius = snap(rng.range(MIN_ARC_RADIUS_M, MAX_ARC_RADIUS_M), world.cell_size);
    let slack_x = map_m - 2.0 * MAP_MARGIN_M - width;
    let slack_z = map_m - 2.0 * MAP_MARGIN_M - height;
    let min_x = snap(MAP_MARGIN_M + rng.range(0.0, slack_x.max(0.0)), world.cell_size);
    let min_z = snap(MAP_MARGIN_M + rng.range(0.0, slack_z.max(0.0)), world.cell_size);
    let max_x = min_x + width;
    let max_z = min_z + height;

    let chord_count = CHORDS_MIN + rng.below(CHORDS_MAX - CHORDS_MIN + 1);
    // Alternate axes so two chords always produce an X junction.
    let mut vertical_x: Vec<f32> = Vec::new();
    let mut horizontal_z: Vec<f32> = Vec::new();
    for k in 0..chord_count {
        let vertical = k % 2 == 0;
        let (low, high) = if vertical {
            (min_x + radius + CHORD_ARC_MARGIN_M, max_x - radius - CHORD_ARC_MARGIN_M)
        } else {
            (min_z + radius + CHORD_ARC_MARGIN_M, max_z - radius - CHORD_ARC_MARGIN_M)
        };
        let existing = if vertical { &vertical_x } else { &horizontal_z };
        let mut placed = None;
        for _ in 0..64 {
            let candidate = snap(rng.range(low, high), world.cell_size);
            if existing.iter().all(|c| (c - candidate).abs() >= CHORD_SEPARATION_M) {
                placed = Some(candidate);
                break;
            }
        }
        let Some(position) = placed else {
            return Err(format!("ring seed {seed}: no room for chord {k} between {low:.0} and {high:.0} m"));
        };
        if vertical {
            vertical_x.push(position);
        } else {
            horizontal_z.push(position);
        }
    }
    vertical_x.sort_by(f32::total_cmp);
    horizontal_z.sort_by(f32::total_cmp);

    use std::f32::consts::{FRAC_PI_2, PI};
    let mut pieces: Vec<(String, RoadPiece)> = vec![
        ("ring-north".into(), RoadPiece::Straight { a: [min_x + radius, min_z], b: [max_x - radius, min_z] }),
        ("ring-east".into(), RoadPiece::Straight { a: [max_x, min_z + radius], b: [max_x, max_z - radius] }),
        ("ring-south".into(), RoadPiece::Straight { a: [max_x - radius, max_z], b: [min_x + radius, max_z] }),
        ("ring-west".into(), RoadPiece::Straight { a: [min_x, max_z - radius], b: [min_x, min_z + radius] }),
        ("corner-nw".into(), RoadPiece::Arc { centre: [min_x + radius, min_z + radius], radius, start_rad: PI, sweep_rad: FRAC_PI_2 }),
        ("corner-ne".into(), RoadPiece::Arc { centre: [max_x - radius, min_z + radius], radius, start_rad: 3.0 * FRAC_PI_2, sweep_rad: FRAC_PI_2 }),
        ("corner-se".into(), RoadPiece::Arc { centre: [max_x - radius, max_z - radius], radius, start_rad: 0.0, sweep_rad: FRAC_PI_2 }),
        ("corner-sw".into(), RoadPiece::Arc { centre: [min_x + radius, max_z - radius], radius, start_rad: FRAC_PI_2, sweep_rad: FRAC_PI_2 }),
    ];
    for (k, x) in vertical_x.iter().enumerate() {
        pieces.push((format!("chord-v{k}"), RoadPiece::Straight { a: [*x, min_z], b: [*x, max_z] }));
    }
    for (k, z) in horizontal_z.iter().enumerate() {
        pieces.push((format!("chord-h{k}"), RoadPiece::Straight { a: [min_x, *z], b: [max_x, *z] }));
    }

    let mut junctions: Vec<Junction> = Vec::new();
    for x in &vertical_x {
        // arms: +x, +z, -x, -z
        junctions.push(Junction { centre: [*x, min_z], arms: [true, true, true, false] });
        junctions.push(Junction { centre: [*x, max_z], arms: [true, false, true, true] });
        for z in &horizontal_z {
            junctions.push(Junction { centre: [*x, *z], arms: [true, true, true, true] });
        }
    }
    for z in &horizontal_z {
        junctions.push(Junction { centre: [min_x, *z], arms: [true, true, false, true] });
        junctions.push(Junction { centre: [max_x, *z], arms: [false, true, true, true] });
    }
    junctions.sort_by(|a, b| a.centre[1].total_cmp(&b.centre[1]).then(a.centre[0].total_cmp(&b.centre[0])));

    // Stations: midpoints of straight sub-segments between junctions or arc
    // tangent points, long enough for a 60 m straight approach on both sides.
    let mut segments: Vec<([f32; 2], [f32; 2])> = Vec::new();
    for (_, piece) in &pieces {
        let RoadPiece::Straight { a, b } = *piece else { continue };
        let along = [b[0] - a[0], b[1] - a[1]];
        let len = along[0].hypot(along[1]);
        let dir = [along[0] / len, along[1] / len];
        let mut cuts: Vec<f32> = junctions
            .iter()
            .filter_map(|j| {
                let rel = [j.centre[0] - a[0], j.centre[1] - a[1]];
                let t = rel[0] * dir[0] + rel[1] * dir[1];
                let off = (rel[0] - dir[0] * t).hypot(rel[1] - dir[1] * t);
                (off < 0.5 && t > -0.5 && t < len + 0.5).then_some(t.clamp(0.0, len))
            })
            .collect();
        cuts.push(0.0);
        cuts.push(len);
        cuts.sort_by(f32::total_cmp);
        for pair in cuts.windows(2) {
            if pair[1] - pair[0] >= MIN_STATION_SEGMENT_M {
                segments.push((
                    [a[0] + dir[0] * pair[0], a[1] + dir[1] * pair[0]],
                    [a[0] + dir[0] * pair[1], a[1] + dir[1] * pair[1]],
                ));
            }
        }
    }
    if segments.len() < 2 {
        return Err(format!("ring seed {seed}: only {} straight segments of {MIN_STATION_SEGMENT_M} m for stations", segments.len()));
    }
    let mut order: Vec<usize> = (0..segments.len()).collect();
    for i in (1..order.len()).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        order.swap(i, j);
    }
    let stations: Vec<[f32; 2]> = order
        .iter()
        .take(STATION_COUNT)
        .map(|&i| {
            let (a, b) = segments[i];
            [snap((a[0] + b[0]) * 0.5, world.cell_size), snap((a[1] + b[1]) * 0.5, world.cell_size)]
        })
        .collect();

    let signature = format!(
        "ring:{seed:016x}:{width:.0}x{height:.0}:r{radius:.0}:v{}:h{}",
        vertical_x.iter().map(|v| format!("{v:.0}")).collect::<Vec<_>>().join("/"),
        horizontal_z.iter().map(|v| format!("{v:.0}")).collect::<Vec<_>>().join("/"),
    );
    Ok(CityLayout {
        family: "ring",
        seed,
        pieces,
        junctions,
        stations,
        extent_m: [[min_x, min_z], [max_x, max_z]],
        generator: json!({
            "loop_min_m": [min_x, min_z], "loop_max_m": [max_x, max_z],
            "width_m": width, "height_m": height, "corner_radius_m": radius,
            "chords_vertical_x_m": vertical_x, "chords_horizontal_z_m": horizontal_z,
            "station_segment_min_m": MIN_STATION_SEGMENT_M,
            "candidate_station_segments": segments.len(),
        }),
        signature,
    })
}

/// Build the ring city into `world` and return its plan.
pub fn build(world: &mut VoxelWorld, seed: u64) -> CityPlanResult {
    let layout = layout(world, seed)?;
    finish_plan(world, &layout)
}

/// Round a metre value to the nearest cell centre so every centreline passes
/// through cell centres and the 5-cell band is symmetric.
fn snap(value_m: f32, cell: f32) -> f32 {
    (value_m / cell).round() * cell
}
