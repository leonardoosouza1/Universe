//! Deterministic static city geometry for `organism-taxi-v1`.
//!
//! This module edits only the immutable `VoxelWorld` geometry. Streets are
//! axis-aligned, connected, and described by the same coordinates exported to
//! the UI; no vehicle dynamics or controller policy lives here.

use bevy::prelude::{Vec2, Vec3};
use serde_json::{json, Value};

use crate::world::{VoxelType, VoxelWorld};
use std::sync::atomic::{AtomicU8, Ordering};

pub const CITY_GEOMETRY_VERSION: &str = "organism-city-v1";
const ROAD_WIDTH_CELLS: i32 = 5;
const STREET_CENTERS: [i32; 5] = [32, 96, 160, 224, 288];

/// Lane-side convention of the grid family.
///
/// `Legacy` is the v1 rule: horizontal streets flow +X on the `row <= centre`
/// side and vertical streets flow +Z on the `col <= centre` side. Along the lane
/// frame `l = (-t.z, t.x)` that puts horizontal traffic at `-3 m` and vertical
/// traffic at `+3 m`, so a being sees the mirror image of the street depending on
/// its axis (finding F1; measured on 2026-09-25 as 1.2-4.6 m/s on vertical streets
/// against 4.4-10.3 m/s on horizontal ones for the same genome,
/// `docs/experiments/CONTRACT-BENCH-20260925.md`). `Consistent` applies the ring
/// family rule on every street: offset along `(-t.z, t.x) > 0` flows `+t`.
/// The default stays `Legacy` so recorded champions keep their world; the choice
/// is exported in the city metadata as `lane_rule`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneRule {
    Legacy = 0,
    Consistent = 1,
}

impl LaneRule {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "legacy" => Some(Self::Legacy),
            "consistent" => Some(Self::Consistent),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Consistent => "consistent",
        }
    }
}

static LANE_RULE: AtomicU8 = AtomicU8::new(LaneRule::Legacy as u8);

/// Process-wide lane rule used by `build_city` (set once from the CLI).
pub fn set_lane_rule(rule: LaneRule) {
    LANE_RULE.store(rule as u8, Ordering::SeqCst);
}

pub fn lane_rule() -> LaneRule {
    if LANE_RULE.load(Ordering::SeqCst) == LaneRule::Consistent as u8 {
        LaneRule::Consistent
    } else {
        LaneRule::Legacy
    }
}

/// Supported deterministic city families.
///
/// Families vary only the spacing of the connected orthogonal grid. They do not
/// claim to reproduce a block grammar, curves, ramps, or MetaDrive's generator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CityFamily {
    Grid,
    Compact,
    Wide,
    /// Rounded loop with cardinal chords; built by `crate::city_family::ring`.
    Ring,
}

impl CityFamily {
    /// Stable metadata name used by callers and experiment manifests.
    pub fn name(self) -> &'static str {
        match self {
            Self::Grid => "grid",
            Self::Compact => "compact",
            Self::Wide => "wide",
            Self::Ring => "ring",
        }
    }

    /// Parse the stable family name used by configuration layers.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "grid" => Some(Self::Grid),
            "compact" => Some(Self::Compact),
            "wide" => Some(Self::Wide),
            "ring" => Some(Self::Ring),
            _ => None,
        }
    }
}

pub struct CityPlan {
    pub stations: Vec<[f32; 8]>,
    traffic_lights: Vec<crate::gpu::traffic::TrafficLight>,
    metadata: Value,
}

impl CityPlan {
    /// Assemble a plan from a family builder outside this module.
    pub fn from_parts(
        stations: Vec<[f32; 8]>,
        traffic_lights: Vec<crate::gpu::traffic::TrafficLight>,
        metadata: Value,
    ) -> Self {
        Self { stations, traffic_lights, metadata }
    }

    pub fn metadata(&self) -> &Value {
        &self.metadata
    }

    pub fn traffic_lights(&self) -> Vec<crate::gpu::traffic::TrafficLight> {
        self.traffic_lights.clone()
    }
}

/// One birth slot: lane centre, heading along the lane flow, never on a station.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CitySpawn {
    pub position: [f32; 3],
    pub forward: [f32; 3],
    pub depot_distance_m: f32,
}

/// Minimum distance between two birth slots; the car body is about 4.5 m long.
pub const SPAWN_MIN_SEPARATION_M: f32 = 8.0;
/// Minimum distance from any station centre; pickup/service act within 4 m.
pub const SPAWN_STATION_CLEARANCE_M: f32 = 18.0;
/// Distance from an intersection centre to the first slot along a street.
const SPAWN_JUNCTION_MARGIN_M: f32 = 18.0;
/// Spacing between consecutive slots in one lane.
const SPAWN_LANE_SPACING_M: f32 = 12.0;

/// Street centre lines in metres, the same filter `build_city` applies.
pub fn street_centres_m(world: &VoxelWorld) -> Vec<f32> {
    street_centers_for_world(world)
        .into_iter()
        .map(|c| c as f32 * world.cell_size)
        .collect()
}

fn street_centers(size: i32) -> Vec<i32> {
    STREET_CENTERS
        .iter()
        .copied()
        .filter(|v| *v >= 3 && *v < size - 3)
        .collect()
}

fn line_centers(world: &VoxelWorld, horizontal: bool) -> Vec<i32> {
    let size = world.size as i32;
    let counts = (0..size)
        .map(|line| {
            (0..size)
                .filter(|other| {
                    let (row, col) = if horizontal { (line, *other) } else { (*other, line) };
                    world.grid[(row * size + col) as usize] == VoxelType::Road
                })
                .count() as i32
        })
        .collect::<Vec<_>>();
    let threshold = (size / 3).max(ROAD_WIDTH_CELLS * 2);
    let mut centers = Vec::new();
    let mut run_start = None;
    for (line, count) in counts.into_iter().enumerate() {
        if count >= threshold {
            run_start.get_or_insert(line as i32);
        } else if let Some(start) = run_start.take() {
            centers.push((start + line as i32 - 1) / 2);
        }
    }
    if let Some(start) = run_start {
        centers.push((start + size - 1) / 2);
    }
    centers
}

/// True when the road cells are exactly the grid this module paints for the
/// recovered street centres. A family city (`crate::city_family`) has arcs or
/// missing streets and fails this test even when its rows and columns happen
/// to line up; a non-city world fails it too.
pub fn is_orthogonal_grid_city(world: &VoxelWorld) -> bool {
    let columns = line_centers(world, false);
    let rows = line_centers(world, true);
    if columns.len() < 3 || columns != rows {
        return false;
    }
    let size = world.size as i32;
    let half = ROAD_WIDTH_CELLS / 2;
    let extent_min = columns[0] - half;
    let extent_max = columns[columns.len() - 1] + half;
    for row in 0..size {
        for col in 0..size {
            let in_extent = (extent_min..=extent_max).contains(&row) && (extent_min..=extent_max).contains(&col);
            let on_street = columns.iter().any(|c| (col - *c).abs() <= half || (row - *c).abs() <= half);
            let expected_road = in_extent && on_street;
            let is_road = world.grid[(row * size + col) as usize] == VoxelType::Road;
            if expected_road != is_road {
                return false;
            }
        }
    }
    true
}

/// Recover street centres from the road bands written into `world`.
///
/// This keeps school geometry tied to the actual city instance without a global
/// variant or thread-local override. Non-city worlds retain the legacy centres.
fn street_centers_for_world(world: &VoxelWorld) -> Vec<i32> {
    let columns = line_centers(world, false);
    let rows = line_centers(world, true);
    if columns.len() >= 3 && columns == rows {
        columns
    } else {
        street_centers(world.size as i32)
    }
}

fn mix_seed(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn variant_centers(size: i32, seed: u64, family: CityFamily) -> Result<Vec<i32>, String> {
    if size < 24 {
        return Err(format!("city grid requires at least 24 cells; got {size}"));
    }
    let ratios: [f32; 5] = match family {
        CityFamily::Grid => [0.10, 0.30, 0.50, 0.70, 0.90],
        CityFamily::Compact => [0.20, 0.35, 0.50, 0.65, 0.80],
        CityFamily::Wide => [0.06, 0.28, 0.50, 0.72, 0.94],
        CityFamily::Ring => return Err("ring family is built by city_family::ring, not by street centres".into()),
    };
    let jitter = match family {
        CityFamily::Grid => 4,
        CityFamily::Compact => 3,
        CityFamily::Wide => 5,
        CityFamily::Ring => 0,
    };
    let mut centers = ratios
        .into_iter()
        .enumerate()
        .map(|(index, ratio)| {
            let base = ((size - 1) as f32 * ratio).round() as i32;
            let span = (jitter * 2 + 1) as u64;
            let offset = (mix_seed(seed.wrapping_add(index as u64)) % span) as i32 - jitter;
            base + offset
        })
        .collect::<Vec<_>>();
    centers.sort_unstable();
    if centers
        .iter()
        .any(|center| *center < 3 || *center >= size - 3)
        || centers.windows(2).any(|pair| pair[1] - pair[0] <= ROAD_WIDTH_CELLS)
    {
        return Err(format!("{family:?} seed {seed} does not fit a connected city grid of {size} cells"));
    }
    Ok(centers)
}

/// Deterministic birth slots on lane centres.
///
/// Candidates lie one cell either side of every street centre line (4.5 m from
/// the nearest wall face), 18 m plus multiples of 12 m from each intersection
/// centre, the opposite lane staggered by 6 m so cross-lane pairs stay 8.5 m
/// apart. The slot cell and its eight neighbours must be road, the heading is the
/// lane flow, nothing is born within 18 m of a station, and slots are ordered by
/// distance to the depot so a small population stays near it. The old lattice
/// scan put slot 0 on the depot (instant pickup) and everyone else 1.5 m from
/// the perimeter wall.
pub fn spawn_slots(
    world: &VoxelWorld,
    stations: &[[f32; 8]],
    n: usize,
) -> Result<Vec<CitySpawn>, String> {
    if n == 0 {
        return Ok(Vec::new());
    }
    if !is_orthogonal_grid_city(world) {
        return crate::city_family::spawn_slots_generic(world, stations, n);
    }
    let cell = world.cell_size;
    let size = world.size as i32;
    let centers = street_centers_for_world(world);
    let depot = world.start_pos;
    let mut candidates: Vec<CitySpawn> = Vec::new();
    let mut consider = |x_m: f32, z_m: f32| {
        let col = (x_m / cell).round() as i32;
        let row = (z_m / cell).round() as i32;
        if col < 1 || row < 1 || col >= size - 1 || row >= size - 1 {
            return;
        }
        for dr in -1..=1 {
            for dc in -1..=1 {
                if world.grid[((row + dr) * size + col + dc) as usize] != VoxelType::Road {
                    return;
                }
            }
        }
        let flow = world.flow[(row * size + col) as usize];
        if flow.length_squared() < 0.5 {
            return;
        }
        if stations
            .iter()
            .any(|s| (s[0] - x_m).hypot(s[1] - z_m) < SPAWN_STATION_CLEARANCE_M)
        {
            return;
        }
        candidates.push(CitySpawn {
            position: [x_m, 0.5, z_m],
            forward: [flow.x, 0.0, flow.y],
            depot_distance_m: (x_m - depot.x).hypot(z_m - depot.z),
        });
    };
    for pair in centers.windows(2) {
        let (a, b) = (pair[0] as f32 * cell, pair[1] as f32 * cell);
        for lane in [-1.0f32, 1.0] {
            let stagger = if lane < 0.0 { 0.0 } else { SPAWN_LANE_SPACING_M * 0.5 };
            let mut along = a + SPAWN_JUNCTION_MARGIN_M + stagger;
            while along <= b - SPAWN_JUNCTION_MARGIN_M {
                for center in &centers {
                    let c = *center as f32 * cell;
                    consider(along, c + lane * cell);
                    consider(c + lane * cell, along);
                }
                along += SPAWN_LANE_SPACING_M;
            }
        }
    }
    candidates.sort_by(|p, q| {
        p.depot_distance_m
            .total_cmp(&q.depot_distance_m)
            .then(p.position[0].total_cmp(&q.position[0]))
            .then(p.position[2].total_cmp(&q.position[2]))
    });
    let mut slots: Vec<CitySpawn> = Vec::with_capacity(n);
    for candidate in candidates {
        if slots.len() == n {
            break;
        }
        if slots.iter().all(|p| {
            (p.position[0] - candidate.position[0]).hypot(p.position[2] - candidate.position[2])
                >= SPAWN_MIN_SEPARATION_M
        }) {
            slots.push(candidate);
        }
    }
    if slots.len() == n {
        Ok(slots)
    } else {
        Err(format!(
            "city has only {} separated lane spawn slots; requested {n}",
            slots.len()
        ))
    }
}

/// Per-generation rotation of the being -> slot map, so no lineage keeps the
/// slot nearest the passenger. Being `i` uses slot `(i + rotation) % n`.
pub fn slot_rotation(generation: u64, seed: u64, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let h = generation
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(seed ^ 0x5851_F42D_4C95_7F2D)
        .wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    ((h >> 23) % n as u64) as usize
}

/// Replace the legacy cross overlay with a connected orthogonal city grid.
pub fn build_city(world: &mut VoxelWorld) -> CityPlan {
    let centers = street_centers(world.size as i32);
    assert!(
        centers.len() >= 3,
        "city grid requires at least three streets"
    );
    build_city_with_centers(world, centers, None, lane_rule())
}

/// `build_city` with an explicit lane rule; tests use it so that they never
/// touch the process-wide setting.
pub fn build_city_with_lane_rule(world: &mut VoxelWorld, rule: LaneRule) -> CityPlan {
    let centers = street_centers(world.size as i32);
    build_city_with_centers(world, centers, None, rule)
}

/// Build a deterministic seeded grid variant and return its geometry contract.
///
/// The seed affects only the integer centre-line jitter within the selected
/// spacing family. The returned plan updates the world before deriving stations,
/// traffic lights, waypoints, spawn capacity, and lesson candidates. Families are
/// axis-aligned grid spacing variants; this is not a full MetaDrive-style generator.
pub fn build_city_variant(
    world: &mut VoxelWorld,
    seed: u64,
    family: CityFamily,
) -> Result<CityPlan, String> {
    if family == CityFamily::Ring {
        return crate::city_family::ring::build(world, seed);
    }
    let centers = variant_centers(world.size as i32, seed, family)?;
    Ok(build_city_with_centers(world, centers, Some((seed, family)), lane_rule()))
}

/// Validate the static geometry contract before the first generation uses it.
///
/// This checks that recovered road-band centres agree with the metadata emitted
/// by the builder, all stations are connected, every station pair has a finite
/// road route, and n beings have separated lane slots. It performs only host
/// geometry preprocessing; it does not tick a vehicle or run a model.
pub fn validate_city_postconditions(
    world: &VoxelWorld,
    plan: &CityPlan,
    n: usize,
) -> Result<(), String> {
    if plan.metadata.get("schema_version").and_then(Value::as_str)
        == Some(crate::city_family::CITY_FAMILY_SCHEMA_VERSION)
    {
        return crate::city_family::validate(world, plan, n);
    }
    let recovered = street_centers_for_world(world);
    let streets = plan
        .metadata
        .get("streets")
        .and_then(Value::as_array)
        .ok_or("city metadata is missing streets")?;
    let mut metadata_centers = Vec::new();
    for street in streets {
        let center = street
            .get("center_cell")
            .and_then(Value::as_i64)
            .ok_or("city metadata has a street without center_cell")?;
        let center = i32::try_from(center).map_err(|_| "city street center is out of range")?;
        if !metadata_centers.contains(&center) {
            metadata_centers.push(center);
        }
    }
    metadata_centers.sort_unstable();
    if recovered != metadata_centers || recovered.len() < 3 {
        return Err(format!(
            "city street centres are not recoverable from roads: metadata={metadata_centers:?}, recovered={recovered:?}"
        ));
    }
    let lights = plan.traffic_lights();
    let expected_lights = recovered.len() * recovered.len() * 4;
    if lights.len() != expected_lights {
        return Err(format!(
            "city has {} traffic lights; expected {expected_lights}",
            lights.len()
        ));
    }
    if lights.iter().any(|light| {
        !light.position.iter().chain(light.approach.iter()).all(|value| value.is_finite())
            || !light.half_width.is_finite()
            || light.half_width <= 0.0
            || (light.approach[0] * light.approach[0] + light.approach[1] * light.approach[1] - 1.0).abs() > 0.0001
            || light.green_ticks == 0
            || light.yellow_ticks == 0
    }) {
        return Err("city contains an invalid traffic light".to_string());
    }
    crate::organism_scenario::validate_city(world, &plan.stations)?;
    let packed = crate::gpu::taxi_routes::pack_distance_fields(world, &plan.stations)
        .map_err(|error| format!("city route preprocessing failed: {error}"))?;
    let cells = world.size * world.size;
    let expected_route_values = 4 + plan.stations.len() * cells;
    if packed.len() != expected_route_values {
        return Err(format!(
            "city route preprocessing returned {} values; expected {expected_route_values}",
            packed.len()
        ));
    }
    for destination in 0..plan.stations.len() {
        for (origin, station) in plan.stations.iter().enumerate() {
            let col = (station[0] / world.cell_size).round();
            let row = (station[1] / world.cell_size).round();
            if !col.is_finite() || !row.is_finite() || col < 0.0 || row < 0.0
                || col >= world.size as f32 || row >= world.size as f32
            {
                return Err(format!("city station {origin} is outside route grid"));
            }
            let cell = row as usize * world.size + col as usize;
            let distance = packed[4 + destination * cells + cell];
            if !distance.is_finite() || distance < 0.0 {
                return Err(format!(
                    "city route {origin}->{destination} is unreachable"
                ));
            }
        }
    }
    spawn_slots(world, &plan.stations, n)
        .map(|_| ())
        .map_err(|error| format!("city spawn contract failed: {error}"))
}

fn build_city_with_centers(
    world: &mut VoxelWorld,
    centers: Vec<i32>,
    variant: Option<(u64, CityFamily)>,
    rule: LaneRule,
) -> CityPlan {
    let size = world.size as i32;
    let cell = world.cell_size;
    assert!(
        centers.len() >= 3,
        "city grid requires at least three streets"
    );

    world.grid.fill(VoxelType::Grass);
    world.flow.fill(Vec2::ZERO);
    world.height_map.fill(0.0);
    world.banking.fill(0.0);
    world.min_dist_to_road.fill(0.0);

    let half = ROAD_WIDTH_CELLS / 2;
    let extent_min = centers[0] - half;
    let extent_max = centers[centers.len() - 1] + half;
    for row in 0..size {
        for col in 0..size {
            let in_extent_row = (extent_min..=extent_max).contains(&row);
            let in_extent_col = (extent_min..=extent_max).contains(&col);
            let vertical = in_extent_row && centers.iter().any(|c| (col - *c).abs() <= half);
            let horizontal = in_extent_col && centers.iter().any(|c| (row - *c).abs() <= half);
            let index = row as usize * world.size + col as usize;
            let perimeter = row == extent_min - 1
                || row == extent_max + 1
                || col == extent_min - 1
                || col == extent_max + 1;
            if perimeter && (extent_min - 1..=extent_max + 1).contains(&row) && (extent_min - 1..=extent_max + 1).contains(&col) {
                world.grid[index] = VoxelType::Wall;
                world.min_dist_to_road[index] = 1.0;
            } else if vertical || horizontal {
                world.grid[index] = VoxelType::Road;
                world.flow[index] = match (horizontal, vertical) {
                    (true, true) => Vec2::ZERO,
                    (true, false) => {
                        let center = centers
                            .iter()
                            .min_by_key(|c| (row - **c).abs())
                            .copied()
                            .unwrap_or(row);
                        // Consistent: t = +X, l = (0, 1); offset along l = row - centre.
                        let east = match rule {
                            LaneRule::Legacy => row <= center,
                            LaneRule::Consistent => row > center,
                        };
                        if east {
                            Vec2::X
                        } else {
                            -Vec2::X
                        }
                    }
                    (false, true) => {
                        let center = centers
                            .iter()
                            .min_by_key(|c| (col - **c).abs())
                            .copied()
                            .unwrap_or(col);
                        // Consistent: t = +Z, l = (-1, 0); offset along l = centre - col.
                        let south = match rule {
                            LaneRule::Legacy => col <= center,
                            LaneRule::Consistent => col < center,
                        };
                        if south {
                            Vec2::Y
                        } else {
                            -Vec2::Y
                        }
                    }
                    (false, false) => Vec2::ZERO,
                };
                world.min_dist_to_road[index] = 0.0;
            } else {
                if in_extent_row && in_extent_col { world.grid[index] = VoxelType::Wall; }
                let distance_cells = centers
                    .iter()
                    .flat_map(|c| [(*c - col).abs(), (*c - row).abs()])
                    .min()
                    .unwrap_or(size);
                world.min_dist_to_road[index] = distance_cells as f32;
            }
        }
    }

    let left = centers[0];
    let right = *centers.last().unwrap();
    let loop_points = vec![
        Vec2::new(left as f32 * cell, left as f32 * cell),
        Vec2::new(right as f32 * cell, left as f32 * cell),
        Vec2::new(right as f32 * cell, right as f32 * cell),
        Vec2::new(left as f32 * cell, right as f32 * cell),
        Vec2::new(left as f32 * cell, left as f32 * cell),
    ];
    let spawn = loop_points[0];
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
    world.start_pos = Vec3::new(spawn.x, 0.5, spawn.y);
    world.start_progress = 0.0;
    // The old race-specific ideal line would describe a false curved circuit.
    world.racing_line = Default::default();
    world.proposal_valid = false;

    let mut stations = centers
        .iter()
        .enumerate()
        .map(|(index, center)| {
            [
                *center as f32 * cell,
                centers[(index * 2 + 1) % centers.len()] as f32 * cell,
                50_000_000.0,
                200.0,
                500.0,
                0.0,
                0.0,
                0.0,
            ]
        })
        .collect::<Vec<_>>();
    stations[0][0] = world.start_pos.x;
    stations[0][1] = world.start_pos.z;

    let streets = centers
        .iter()
        .enumerate()
        .flat_map(|(index, center)| {
            [
                json!({"id":format!("v{index}"),"axis":"z","center_cell":center,
                    "start_cell":extent_min,"end_cell_exclusive":extent_max+1,
                    "lanes":2,"width_m":ROAD_WIDTH_CELLS as f32*cell}),
                json!({"id":format!("h{index}"),"axis":"x","center_cell":center,
                    "start_cell":extent_min,"end_cell_exclusive":extent_max+1,
                    "lanes":2,"width_m":ROAD_WIDTH_CELLS as f32*cell}),
            ]
        })
        .collect::<Vec<_>>();
    let intersections = centers
        .iter()
        .enumerate()
        .flat_map(|(row, z)| {
            centers.iter().enumerate().map(move |(col, x)| {
                json!({"id":format!("i{row}_{col}"),"x_m":*x as f32*cell,"z_m":*z as f32*cell,
                    "traffic_light":format!("tl{row}_{col}")})
            })
        })
        .collect::<Vec<_>>();
    let green_ticks = 14 * 64;
    let yellow_ticks = 2 * 64;
    let half_width = ROAD_WIDTH_CELLS as f32 * cell * 0.5;
    let mut traffic_lights = Vec::new();
    let mut traffic_metadata = Vec::new();
    for (row, z) in centers.iter().enumerate() {
        for (col, x) in centers.iter().enumerate() {
            let center = Vec2::new(*x as f32 * cell, *z as f32 * cell);
            let axis_offset = if (row + col) % 2 == 0 {
                0
            } else {
                green_ticks + yellow_ticks
            };
            for (approach, stop) in [
                (Vec2::X, center - Vec2::X * (half_width + cell)),
                (-Vec2::X, center + Vec2::X * (half_width + cell)),
                (Vec2::Y, center - Vec2::Y * (half_width + cell)),
                (-Vec2::Y, center + Vec2::Y * (half_width + cell)),
            ] {
                // Perpendicular approaches must never share a green phase.
                let approach_offset = (axis_offset + if approach.y != 0.0 { green_ticks + yellow_ticks } else { 0 }) % (2 * (green_ticks + yellow_ticks));
                traffic_lights.push(crate::gpu::traffic::TrafficLight {
                    position: [stop.x, stop.y],
                    approach: [approach.x, approach.y],
                    half_width,
                    phase_offset_ticks: approach_offset as u32,
                    green_ticks,
                    yellow_ticks,
                });
                traffic_metadata.push(json!({"position":[stop.x,stop.y],"approach":[approach.x,approach.y],
                    "half_width_m":half_width,"phase_offset_ticks":approach_offset,
                    "green_ticks":green_ticks,"yellow_ticks":yellow_ticks,"cycle_ticks":2*(green_ticks+yellow_ticks),
                    "phases":[{"name":"green","duration_s":14.0},
                        {"name":"yellow","duration_s":2.0},
                        {"name":"red","duration_s":16.0}]}));
            }
        }
    }
    let blocks = centers
        .windows(2)
        .enumerate()
        .flat_map(|(row, rows)| {
            centers.windows(2).enumerate().map(move |(col, cols)| {
                json!({"id":format!("b{row}_{col}"),
                    "min_cell":[rows[0]+half+1,cols[0]+half+1],
                    "max_cell_exclusive":[rows[1]-half,cols[1]-half],
                    "land_use":"urban_block"})
            })
        })
        .collect::<Vec<_>>();
    let mut metadata = json!({"schema_version":CITY_GEOMETRY_VERSION,"size":world.size,
        "cell_size_m":cell,"road_width_cells":ROAD_WIDTH_CELLS,
        "extent_cells":[extent_min,extent_max+1],"streets":streets,
        "blocks":blocks,"intersections":intersections,"traffic_lights":traffic_metadata,
        "lane_axes":{"x":"east_west","z":"north_south"},
        "lane_rule":rule.name(),
        "spawn":{"x_m":world.start_pos.x,"z_m":world.start_pos.z,"safe":true}});
    if let Some((seed, family)) = variant {
        let center_signature = centers
            .iter()
            .map(|center| center.to_string())
            .collect::<Vec<_>>()
            .join(",");
        metadata["variant"] = json!({
            "family": family.name(),
            "seed": seed,
            "signature": format!("{}:{seed:016x}:{center_signature}", family.name()),
            "limitations": "axis-aligned grid spacing variation only; no curves, ramps, block grammar, or MetaDrive-style procedural city",
        });
    }

    CityPlan {
        stations,
        traffic_lights,
        metadata,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_contract_has_connected_axes_and_versioned_signals() {
        let _quiet = crate::world::silence_map_diag();
        let mut world = VoxelWorld::new(7, 28.0);
        let plan = build_city(&mut world);
        crate::organism_scenario::validate_city(&world, &plan.stations).unwrap();
        assert_eq!(
            world.grid[world.size / 2 * world.size + world.size / 2],
            VoxelType::Road
        );
        assert_eq!(world.racing_line.points.len(), 0);
        assert_eq!(plan.stations[0][0], world.start_pos.x);
        assert_eq!(plan.stations[0][1], world.start_pos.z);
        let metadata = plan.metadata();
        assert_eq!(metadata["schema_version"], CITY_GEOMETRY_VERSION);
        assert_eq!(metadata["lane_axes"]["x"], "east_west");
        assert_eq!(metadata["traffic_lights"].as_array().unwrap().len(), 100);
        assert_eq!(metadata["traffic_lights"][0]["green_ticks"], 896);
        assert_eq!(metadata["traffic_lights"][0]["yellow_ticks"], 128);
        let lights = plan.traffic_lights();
        for junction in lights.chunks_exact(4) {
            assert_eq!(junction[0].phase_offset_ticks, junction[1].phase_offset_ticks);
            assert_eq!(junction[2].phase_offset_ticks, junction[3].phase_offset_ticks);
            assert_eq!((junction[2].phase_offset_ticks + 2048 - junction[0].phase_offset_ticks) % 2048, 1024);
        }
    }

    #[test]
    fn consistent_lane_rule_puts_every_street_on_the_ring_side() {
        // Ring convention (city_family): offset along (-t.z, t.x) > 0 => flow +t.
        let mut world = VoxelWorld::new(1, 20.0);
        let plan = build_city_with_lane_rule(&mut world, LaneRule::Consistent);
        assert_eq!(plan.metadata["lane_rule"], "consistent");
        let size = world.size as i32;
        let centers = street_centers(size);
        let half = ROAD_WIDTH_CELLS / 2;
        let mut checked = 0;
        for row in 0..size {
            for col in 0..size {
                let flow = world.flow[(row * size + col) as usize];
                if flow.length_squared() < 0.5 {
                    continue;
                }
                let horizontal = flow.x.abs() > 0.5;
                let (line, centre) = if horizontal {
                    (row, *centers.iter().min_by_key(|c| (row - **c).abs()).unwrap())
                } else {
                    (col, *centers.iter().min_by_key(|c| (col - **c).abs()).unwrap())
                };
                assert!((line - centre).abs() <= half);
                // t = flow; l = (-t.z, t.x); offset = (col, row) - centre along l.
                let offset = if horizontal {
                    (row - centre) as f32 * flow.x
                } else {
                    -((col - centre) as f32) * flow.y
                };
                assert!(offset > 0.0 || (offset == 0.0 && (flow.x < 0.0 || flow.y < 0.0)),
                    "cell ({row},{col}) flow {flow:?} sits on the wrong side of its centre {centre}");
                checked += 1;
            }
        }
        assert!(checked > 10_000, "checked only {checked} lane cells");
        // The legacy rule mirrors horizontal streets: same cell, opposite flow.
        let mut legacy = VoxelWorld::new(1, 20.0);
        build_city_with_lane_rule(&mut legacy, LaneRule::Legacy);
        let probe = (STREET_CENTERS[1] - 1) * size + STREET_CENTERS[1] + 20; // horizontal lane, row < centre
        assert_eq!(legacy.flow[probe as usize].x, 1.0);
        assert_eq!(world.flow[probe as usize].x, -1.0);
        let probe_v = (STREET_CENTERS[1] + 20) * size + STREET_CENTERS[1] - 1; // vertical lane, col < centre
        assert_eq!(legacy.flow[probe_v as usize].y, 1.0);
        assert_eq!(world.flow[probe_v as usize].y, 1.0);
    }

    #[test]
    fn spawn_slots_sit_on_lanes_away_from_walls_and_stations() {
        let _quiet = crate::world::silence_map_diag();
        let mut world = VoxelWorld::new(7, 28.0);
        let plan = build_city(&mut world);
        let n = 12;
        let slots = spawn_slots(&world, &plan.stations, n).unwrap();
        assert_eq!(slots.len(), n);
        let size = world.size as i32;
        let cell = world.cell_size;
        for (k, slot) in slots.iter().enumerate() {
            let col = (slot.position[0] / cell).round() as i32;
            let row = (slot.position[2] / cell).round() as i32;
            assert_eq!(world.grid[(row * size + col) as usize], VoxelType::Road, "slot {k} off road");
            // Nearest wall cell centre at least two cells away: 4.5 m to its face.
            let mut nearest_wall = f32::MAX;
            for r in (row - 4).max(0)..(row + 5).min(size) {
                for c in (col - 4).max(0)..(col + 5).min(size) {
                    if world.grid[(r * size + c) as usize] == VoxelType::Wall {
                        let d = ((c as f32 * cell) - slot.position[0])
                            .hypot((r as f32 * cell) - slot.position[2]);
                        nearest_wall = nearest_wall.min(d);
                    }
                }
            }
            assert!(nearest_wall >= 2.0 * cell, "slot {k} hugs a wall: {nearest_wall} m");
            let axis_aligned = (slot.forward[0].abs() - 1.0).abs() < 1e-6 && slot.forward[2] == 0.0
                || (slot.forward[2].abs() - 1.0).abs() < 1e-6 && slot.forward[0] == 0.0;
            assert!(axis_aligned, "slot {k} heading {:?}", slot.forward);
            for station in &plan.stations {
                let d = (station[0] - slot.position[0]).hypot(station[1] - slot.position[2]);
                assert!(d >= SPAWN_STATION_CLEARANCE_M, "slot {k} within {d} m of a station");
            }
            assert!(slot.depot_distance_m >= SPAWN_STATION_CLEARANCE_M);
        }
        for (i, p) in slots.iter().enumerate() {
            for q in &slots[i + 1..] {
                let d = (p.position[0] - q.position[0]).hypot(p.position[2] - q.position[2]);
                assert!(d >= SPAWN_MIN_SEPARATION_M, "slots overlap: {d} m");
            }
        }
        assert!(slots.windows(2).all(|w| w[0].depot_distance_m <= w[1].depot_distance_m));
        assert_eq!(spawn_slots(&world, &plan.stations, n).unwrap(), slots, "deterministic");
        assert!(spawn_slots(&world, &plan.stations, 100_000).is_err());
    }

    #[test]
    fn slot_rotation_is_deterministic_and_covers_every_slot() {
        let n = 12;
        assert_eq!(slot_rotation(3, 77, n), slot_rotation(3, 77, n));
        let mut seen = std::collections::HashSet::new();
        for generation in 0..200u64 {
            seen.insert(slot_rotation(generation, 77, n));
        }
        assert_eq!(seen.len(), n, "every slot is used within 200 generations");
        assert_ne!(slot_rotation(1, 77, n), slot_rotation(1, 78, n) , "seed changes the map");
        assert_eq!(slot_rotation(5, 1, 0), 0);
    }
}
