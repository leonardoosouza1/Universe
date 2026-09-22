//! Deterministic static city geometry for `organism-taxi-v1`.
//!
//! This module edits only the immutable `VoxelWorld` geometry. Streets are
//! axis-aligned, connected, and described by the same coordinates exported to
//! the UI; no vehicle dynamics or controller policy lives here.

use bevy::prelude::{Vec2, Vec3};
use serde_json::{json, Value};

use crate::world::{VoxelType, VoxelWorld};

pub const CITY_GEOMETRY_VERSION: &str = "organism-city-v1";
const ROAD_WIDTH_CELLS: i32 = 5;
const STREET_CENTERS: [i32; 5] = [32, 96, 160, 224, 288];

pub struct CityPlan {
    pub stations: Vec<[f32; 8]>,
    traffic_lights: Vec<crate::gpu::traffic::TrafficLight>,
    metadata: Value,
}

impl CityPlan {
    pub fn metadata(&self) -> &Value {
        &self.metadata
    }

    pub fn traffic_lights(&self) -> Vec<crate::gpu::traffic::TrafficLight> {
        self.traffic_lights.clone()
    }
}

/// Deterministic separated spawn positions around the depot intersection.
pub fn spawn_positions(world: &VoxelWorld, n: usize) -> Result<Vec<[f32; 3]>, String> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut positions = Vec::with_capacity(n);
    let start_col = (world.start_pos.x / world.cell_size).round() as usize;
    let start_row = (world.start_pos.z / world.cell_size).round() as usize;
    if start_row < world.size
        && start_col < world.size
        && world.grid[start_row * world.size + start_col] == VoxelType::Road
    {
        positions.push([world.start_pos.x, 0.5, world.start_pos.z]);
    }
    // Candidate cells are deterministic and six metres apart in the scan
    // lattice. The distance check also handles the depot candidate cleanly.
    for row in (0..world.size).step_by(2) {
        for col in (0..world.size).step_by(2) {
            if positions.len() == n {
                return Ok(positions);
            }
            if world.grid[row * world.size + col] != VoxelType::Road {
                continue;
            }
            let candidate = [
                col as f32 * world.cell_size,
                0.5,
                row as f32 * world.cell_size,
            ];
            if positions
                .iter()
                .all(|p| (p[0] - candidate[0]).hypot(p[2] - candidate[2]) >= 6.0)
            {
                positions.push(candidate);
            }
        }
    }
    if positions.len() == n {
        Ok(positions)
    } else {
        Err(format!(
            "city has only {} separated road spawn slots; requested {n}",
            positions.len()
        ))
    }
}

/// Replace the legacy cross overlay with a connected orthogonal city grid.
pub fn build_city(world: &mut VoxelWorld) -> CityPlan {
    let size = world.size as i32;
    let cell = world.cell_size;
    let centers = STREET_CENTERS
        .iter()
        .copied()
        .filter(|v| *v >= 3 && *v < size - 3)
        .collect::<Vec<_>>();
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
                        if row <= center {
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
                        if col <= center {
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
    let metadata = json!({"schema_version":CITY_GEOMETRY_VERSION,"size":world.size,
        "cell_size_m":cell,"road_width_cells":ROAD_WIDTH_CELLS,
        "extent_cells":[extent_min,extent_max+1],"streets":streets,
        "blocks":blocks,"intersections":intersections,"traffic_lights":traffic_metadata,
        "lane_axes":{"x":"east_west","z":"north_south"},
        "spawn":{"x_m":world.start_pos.x,"z_m":world.start_pos.z,"safe":true}});

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
        let spawns = spawn_positions(&world, 8).unwrap();
        assert_eq!(spawns.len(), 8);
        assert!(spawns
            .windows(2)
            .all(|pair| { (pair[0][0] - pair[1][0]).hypot(pair[0][2] - pair[1][2]) >= 6.0 }));
    }
}
