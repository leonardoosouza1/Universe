//! Seeded city families beyond the orthogonal grid of `organism_city`.
//!
//! Host-only static geometry. Every family writes the same immutable
//! `VoxelWorld` fields as `organism_city::build_city` (grid, flow,
//! min_dist_to_road, waypoints, start position) and returns the same `CityPlan`
//! (stations, traffic lights, versioned metadata), so the GPU upload, the route
//! preprocessing, the traffic kernel and the Lab UI cell-grid renderer consume a
//! family city unchanged. No vehicle dynamics, controller policy or reward lives
//! here. The reference grid city stays byte-identical: nothing in this module
//! runs unless a family other than `grid`/`compact`/`wide` is selected.
//!
//! Geometry contract (metres, x east / z south on the Lab UI map, cell 3 m):
//! - roads are 5 cells (15 m) wide bands around straight or circular-arc
//!   centrelines; a cell is road when its centre is closer than 7.5 m to a
//!   centreline, exactly the band `build_city` paints for a street;
//! - inside the city extent every non-road cell is a solid wall (buildings);
//!   one wall ring surrounds the extent; outside is grass, as in the grid city;
//! - lanes are two-way. The lane on the right-hand side of the travel direction
//!   `t` (`dot(offset, (-t.z, t.x)) > 0` in the x/z frame) carries `+t`; junction
//!   boxes carry zero flow. The grid city's own horizontal streets use the
//!   opposite side; see `docs/CITY-FAMILY-RING.md` for that finding;
//! - junctions are crossings of cardinal straights only (T or X), so the school
//!   helpers that walk a straight cardinal lane keep working; arcs never host a
//!   junction or a station;
//! - stations sit on the centreline at the midpoint of a straight segment at
//!   least `MIN_STATION_SEGMENT_M` long, so a 60 m straight approach exists on
//!   both sides of every station;
//! - seeds with `seed % 10 == 9` are held-out evaluation seeds (`seed_split`).

use bevy::prelude::{Vec2, Vec3};
use serde_json::{Value, json};

use crate::gpu::traffic::TrafficLight;
use crate::organism_city::CityPlan;
use crate::world::{VoxelType, VoxelWorld};

mod contract;
pub mod ring;
#[cfg(test)]
mod tests;
mod topology;

pub use contract::{finish_plan, metadata, validate};
pub(crate) use topology::road_at;
pub use topology::{
    RouteStats, detect_junctions, road_edge_distance, route_stats, spawn_slots_generic,
};

/// Result type shared by the family builders.
pub type CityPlanResult = Result<CityPlan, String>;

/// Metadata schema written by every family in this module.
pub const CITY_FAMILY_SCHEMA_VERSION: &str = "organism-city-v2";
/// Road band width in cells; identical to the grid city.
pub const ROAD_WIDTH_CELLS: i32 = 5;
/// Half road width in metres (5 cells × 3 m / 2).
pub const ROAD_HALF_WIDTH_M: f32 = 7.5;
/// Half side of the zero-flow junction box, one road half width.
pub const JUNCTION_BOX_HALF_M: f32 = 7.5;
/// Smallest corner radius a family may emit (centreline radius).
pub const MIN_ARC_RADIUS_M: f32 = 45.0;
/// Largest corner radius; keeps arc cross-runs below `JUNCTION_RUN_CELLS`.
pub const MAX_ARC_RADIUS_M: f32 = 90.0;
/// A station needs a straight segment at least this long (≥ 60 m either side).
pub const MIN_STATION_SEGMENT_M: f32 = 130.0;
/// A road cell whose row run and column run both reach this many cells is
/// inside a junction box. Straights have a cross run of 5 cells; the widest
/// arc cross-run at radius 90 m is 18 cells.
pub const JUNCTION_RUN_CELLS: usize = 20;
/// Birth slots keep this distance from any junction centre.
pub const SPAWN_JUNCTION_CLEARANCE_M: f32 = 18.0;
/// Traffic-light schedule shared with the grid city (64 Hz ticks).
pub const GREEN_TICKS: u32 = 14 * 64;
pub const YELLOW_TICKS: u32 = 2 * 64;

/// Held-out split by seed: one seed in ten is reserved for evaluation.
pub fn seed_split(seed: u64) -> &'static str {
    if seed % 10 == 9 { "eval" } else { "train" }
}

/// SplitMix64: deterministic, dependency-free, one stream per seed.
pub struct SeededRng(u64);

impl SeededRng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.unit()
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n.max(1)
    }
}

/// Nearest point of a road piece to a query point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Nearest {
    pub distance_m: f32,
    pub foot: [f32; 2],
    /// Unit tangent of the piece at `foot`, along the piece direction.
    pub tangent: [f32; 2],
}

/// One centreline piece. Arcs sweep counter-clockwise in the `atan2(z, x)`
/// convention of the x/z map frame; a sweep is at most a half turn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RoadPiece {
    Straight {
        a: [f32; 2],
        b: [f32; 2],
    },
    Arc {
        centre: [f32; 2],
        radius: f32,
        start_rad: f32,
        sweep_rad: f32,
    },
}

impl RoadPiece {
    pub fn length_m(&self) -> f32 {
        match *self {
            Self::Straight { a, b } => (b[0] - a[0]).hypot(b[1] - a[1]),
            Self::Arc {
                radius, sweep_rad, ..
            } => radius * sweep_rad,
        }
    }

    pub fn start_point(&self) -> [f32; 2] {
        match *self {
            Self::Straight { a, .. } => a,
            Self::Arc {
                centre,
                radius,
                start_rad,
                ..
            } => [
                centre[0] + radius * start_rad.cos(),
                centre[1] + radius * start_rad.sin(),
            ],
        }
    }

    pub fn end_point(&self) -> [f32; 2] {
        match *self {
            Self::Straight { b, .. } => b,
            Self::Arc {
                centre,
                radius,
                start_rad,
                sweep_rad,
            } => {
                let angle = start_rad + sweep_rad;
                [
                    centre[0] + radius * angle.cos(),
                    centre[1] + radius * angle.sin(),
                ]
            }
        }
    }

    pub fn nearest(&self, p: [f32; 2]) -> Nearest {
        match *self {
            Self::Straight { a, b } => {
                let d = [b[0] - a[0], b[1] - a[1]];
                let len_sq = d[0] * d[0] + d[1] * d[1];
                let t = if len_sq > 0.0 {
                    (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / len_sq).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let foot = [a[0] + d[0] * t, a[1] + d[1] * t];
                let len = len_sq.sqrt().max(1e-6);
                Nearest {
                    distance_m: (p[0] - foot[0]).hypot(p[1] - foot[1]),
                    foot,
                    tangent: [d[0] / len, d[1] / len],
                }
            }
            Self::Arc {
                centre,
                radius,
                start_rad,
                sweep_rad,
            } => {
                let v = [p[0] - centre[0], p[1] - centre[1]];
                let r = v[0].hypot(v[1]);
                let angle = v[1].atan2(v[0]);
                let rel = (angle - start_rad).rem_euclid(std::f32::consts::TAU);
                if r > 1e-3 && rel <= sweep_rad {
                    let foot = [centre[0] + radius * v[0] / r, centre[1] + radius * v[1] / r];
                    Nearest {
                        distance_m: (r - radius).abs(),
                        foot,
                        tangent: [-angle.sin(), angle.cos()],
                    }
                } else {
                    let candidates = [
                        (self.start_point(), start_rad),
                        (self.end_point(), start_rad + sweep_rad),
                    ];
                    let mut best: Option<Nearest> = None;
                    for (point, at) in candidates {
                        let distance_m = (p[0] - point[0]).hypot(p[1] - point[1]);
                        if best.map_or(true, |b| distance_m < b.distance_m) {
                            best = Some(Nearest {
                                distance_m,
                                foot: point,
                                tangent: [-at.sin(), at.cos()],
                            });
                        }
                    }
                    best.expect("two arc endpoints")
                }
            }
        }
    }
}

/// Cardinal arm directions of a junction in index order: +x, +z, −x, −z.
pub const ARM_DIRECTIONS: [[f32; 2]; 4] = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]];
pub const ARM_NAMES: [&str; 4] = ["+x", "+z", "-x", "-z"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JunctionKind {
    T,
    X,
}

impl JunctionKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::T => "T",
            Self::X => "X",
        }
    }
}

/// A crossing of cardinal straights; `arms[k]` says a road leaves the centre
/// along `ARM_DIRECTIONS[k]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Junction {
    pub centre: [f32; 2],
    pub arms: [bool; 4],
}

impl Junction {
    pub fn kind(&self) -> JunctionKind {
        if self.arms.iter().filter(|arm| **arm).count() >= 4 {
            JunctionKind::X
        } else {
            JunctionKind::T
        }
    }

    pub fn arm_count(&self) -> usize {
        self.arms.iter().filter(|arm| **arm).count()
    }
}

/// Immutable centreline layout of one family city, before rasterisation.
#[derive(Clone, Debug)]
pub struct CityLayout {
    pub family: &'static str,
    pub seed: u64,
    pub pieces: Vec<(String, RoadPiece)>,
    pub junctions: Vec<Junction>,
    /// `stations[0]` is the depot and the world start position.
    pub stations: Vec<[f32; 2]>,
    /// Bounding box of the road centrelines: `[min, max]`.
    pub extent_m: [[f32; 2]; 2],
    pub generator: Value,
    pub signature: String,
}

impl CityLayout {
    pub fn nearest(&self, p: [f32; 2]) -> Nearest {
        let mut best: Option<Nearest> = None;
        for (_, piece) in &self.pieces {
            let n = piece.nearest(p);
            if best.map_or(true, |b| n.distance_m < b.distance_m) {
                best = Some(n);
            }
        }
        best.expect("layout has at least one road piece")
    }

    pub fn inside_junction_box(&self, p: [f32; 2]) -> bool {
        self.junctions.iter().any(|j| {
            (p[0] - j.centre[0]).abs() < JUNCTION_BOX_HALF_M
                && (p[1] - j.centre[1]).abs() < JUNCTION_BOX_HALF_M
        })
    }

    /// Lane flow at a road point: `+t` on the right-hand side of `t`, `−t` on
    /// the left, zero inside a junction box. Independent of piece orientation.
    pub fn lane_flow(&self, p: [f32; 2]) -> Vec2 {
        if self.inside_junction_box(p) {
            return Vec2::ZERO;
        }
        let n = self.nearest(p);
        let t = n.tangent;
        let offset = [p[0] - n.foot[0], p[1] - n.foot[1]];
        let right = [-t[1], t[0]];
        if offset[0] * right[0] + offset[1] * right[1] >= 0.0 {
            Vec2::new(t[0], t[1])
        } else {
            Vec2::new(-t[0], -t[1])
        }
    }
}

/// Paint the layout into the immutable world fields. Same materials, band
/// width, wall interior, perimeter wall and outer grass as the grid city.
pub fn rasterize(world: &mut VoxelWorld, layout: &CityLayout) {
    let size = world.size as i32;
    let cell = world.cell_size;
    world.grid.fill(VoxelType::Grass);
    world.flow.fill(Vec2::ZERO);
    world.height_map.fill(0.0);
    world.banking.fill(0.0);
    world.min_dist_to_road.fill(0.0);

    let half_cells = ROAD_WIDTH_CELLS / 2;
    let extent_min_col = (layout.extent_m[0][0] / cell).round() as i32 - half_cells;
    let extent_min_row = (layout.extent_m[0][1] / cell).round() as i32 - half_cells;
    let extent_max_col = (layout.extent_m[1][0] / cell).round() as i32 + half_cells;
    let extent_max_row = (layout.extent_m[1][1] / cell).round() as i32 + half_cells;

    for row in 0..size {
        for col in 0..size {
            let p = [col as f32 * cell, row as f32 * cell];
            let index = (row * size + col) as usize;
            let n = layout.nearest(p);
            let in_extent = (extent_min_col..=extent_max_col).contains(&col)
                && (extent_min_row..=extent_max_row).contains(&row);
            let in_perimeter_box = (extent_min_col - 1..=extent_max_col + 1).contains(&col)
                && (extent_min_row - 1..=extent_max_row + 1).contains(&row);
            if n.distance_m < ROAD_HALF_WIDTH_M {
                world.grid[index] = VoxelType::Road;
                world.flow[index] = layout.lane_flow(p);
                world.min_dist_to_road[index] = 0.0;
            } else if in_perimeter_box && !in_extent {
                world.grid[index] = VoxelType::Wall;
                world.min_dist_to_road[index] = 1.0;
            } else if in_extent {
                world.grid[index] = VoxelType::Wall;
                world.min_dist_to_road[index] = n.distance_m / cell;
            } else {
                world.min_dist_to_road[index] = n.distance_m / cell;
            }
        }
    }

    let (min, max) = (layout.extent_m[0], layout.extent_m[1]);
    let loop_points = vec![
        Vec2::new(min[0], min[1]),
        Vec2::new(max[0], min[1]),
        Vec2::new(max[0], max[1]),
        Vec2::new(min[0], max[1]),
        Vec2::new(min[0], min[1]),
    ];
    world.waypoints_world = loop_points;
    world.waypoint_arcs.clear();
    world.waypoint_arcs.push(0.0);
    for pair in world.waypoints_world.windows(2) {
        let previous = *world.waypoint_arcs.last().unwrap();
        world
            .waypoint_arcs
            .push(previous + pair[1].distance(pair[0]));
    }
    world.lap_distance = *world.waypoint_arcs.last().unwrap_or(&0.0);
    let depot = layout.stations[0];
    world.start_pos = Vec3::new(depot[0], 0.5, depot[1]);
    world.start_progress = 0.0;
    world.racing_line = Default::default();
    world.proposal_valid = false;
}

/// Station ABI shared with the grid city: position, stock capacities, counters.
pub fn station_records(layout: &CityLayout) -> Vec<[f32; 8]> {
    layout
        .stations
        .iter()
        .map(|s| [s[0], s[1], 50_000_000.0, 200.0, 500.0, 0.0, 0.0, 0.0])
        .collect()
}

/// One light per junction arm, on the stop line 10.5 m before the centre.
/// Opposite arms share a phase; perpendicular arms never share green.
pub fn traffic_lights(layout: &CityLayout) -> (Vec<TrafficLight>, Vec<Value>) {
    let cycle = 2 * (GREEN_TICKS + YELLOW_TICKS);
    let mut lights = Vec::new();
    let mut metadata = Vec::new();
    for (index, junction) in layout.junctions.iter().enumerate() {
        let axis_offset = if index % 2 == 0 {
            0
        } else {
            GREEN_TICKS + YELLOW_TICKS
        };
        for (arm, dir) in ARM_DIRECTIONS.iter().enumerate() {
            // A vehicle approaching along `dir` arrives from the opposite arm.
            let from = (arm + 2) % 4;
            if !junction.arms[from] {
                continue;
            }
            let stop = [
                junction.centre[0] - dir[0] * (ROAD_HALF_WIDTH_M + 3.0),
                junction.centre[1] - dir[1] * (ROAD_HALF_WIDTH_M + 3.0),
            ];
            let offset = (axis_offset
                + if dir[1] != 0.0 {
                    GREEN_TICKS + YELLOW_TICKS
                } else {
                    0
                })
                % cycle;
            lights.push(TrafficLight {
                position: stop,
                approach: *dir,
                half_width: ROAD_HALF_WIDTH_M,
                phase_offset_ticks: offset,
                green_ticks: GREEN_TICKS,
                yellow_ticks: YELLOW_TICKS,
            });
            metadata.push(json!({
                "junction": format!("j{index}"),
                "approach_arm": ARM_NAMES[arm],
                "position": [stop[0], stop[1]],
                "approach": [dir[0], dir[1]],
                "half_width_m": ROAD_HALF_WIDTH_M,
                "phase_offset_ticks": offset,
                "green_ticks": GREEN_TICKS,
                "yellow_ticks": YELLOW_TICKS,
                "cycle_ticks": cycle,
                "phases": [
                    {"name": "green", "duration_s": 14.0},
                    {"name": "yellow", "duration_s": 2.0},
                    {"name": "red", "duration_s": 16.0}
                ]
            }));
        }
    }
    (lights, metadata)
}
