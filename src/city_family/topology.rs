//! Road-cell topology recovered from a rasterised world: junction boxes, edge
//! distances, generic spawn slots and station route statistics. Host only.


use super::{Junction, ARM_DIRECTIONS, JUNCTION_RUN_CELLS, ROAD_HALF_WIDTH_M, SPAWN_JUNCTION_CLEARANCE_M};
use crate::organism_city::{CitySpawn, SPAWN_MIN_SEPARATION_M, SPAWN_STATION_CLEARANCE_M};
use crate::world::{VoxelType, VoxelWorld};

pub(crate) fn road_at(world: &VoxelWorld, col: i32, row: i32) -> bool {
    let size = world.size as i32;
    col >= 0
        && row >= 0
        && col < size
        && row < size
        && world.grid[(row * size + col) as usize] == VoxelType::Road
}

/// Recover junction boxes from the road cells alone: a road cell whose row run
/// and column run both reach `JUNCTION_RUN_CELLS` lies where two cardinal
/// straights cross. Connected boxes collapse to their centre cell; arms are
/// probed 13.5 m from the centre. Works on the grid city (25 X junctions) and
/// on every family in this module; arcs never qualify.
pub fn detect_junctions(world: &VoxelWorld) -> Vec<Junction> {
    let size = world.size as i32;
    let cell = world.cell_size;
    let index = |col: i32, row: i32| (row * size + col) as usize;
    let mut row_run = vec![0u32; world.grid.len()];
    let mut col_run = vec![0u32; world.grid.len()];
    for row in 0..size {
        let mut start = 0;
        while start < size {
            if !road_at(world, start, row) {
                start += 1;
                continue;
            }
            let mut end = start;
            while end < size && road_at(world, end, row) {
                end += 1;
            }
            for col in start..end {
                row_run[index(col, row)] = (end - start) as u32;
            }
            start = end;
        }
    }
    for col in 0..size {
        let mut start = 0;
        while start < size {
            if !road_at(world, col, start) {
                start += 1;
                continue;
            }
            let mut end = start;
            while end < size && road_at(world, col, end) {
                end += 1;
            }
            for row in start..end {
                col_run[index(col, row)] = (end - start) as u32;
            }
            start = end;
        }
    }
    let threshold = JUNCTION_RUN_CELLS as u32;
    let is_box = |col: i32, row: i32| {
        road_at(world, col, row)
            && row_run[index(col, row)] >= threshold
            && col_run[index(col, row)] >= threshold
    };
    let mut visited = vec![false; world.grid.len()];
    let mut junctions = Vec::new();
    for row in 0..size {
        for col in 0..size {
            if visited[index(col, row)] || !is_box(col, row) {
                continue;
            }
            let mut stack = vec![(col, row)];
            visited[index(col, row)] = true;
            let (mut sum_col, mut sum_row, mut count) = (0i64, 0i64, 0i64);
            while let Some((c, r)) = stack.pop() {
                sum_col += c as i64;
                sum_row += r as i64;
                count += 1;
                for (dc, dr) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let (nc, nr) = (c + dc, r + dr);
                    if nc < 0 || nr < 0 || nc >= size || nr >= size {
                        continue;
                    }
                    if !visited[index(nc, nr)] && is_box(nc, nr) {
                        visited[index(nc, nr)] = true;
                        stack.push((nc, nr));
                    }
                }
            }
            let centre_col = ((sum_col as f64 / count as f64).round()) as i32;
            let centre_row = ((sum_row as f64 / count as f64).round()) as i32;
            let centre = [centre_col as f32 * cell, centre_row as f32 * cell];
            let probe_m = ROAD_HALF_WIDTH_M + 2.0 * cell;
            let mut arms = [false; 4];
            for (k, dir) in ARM_DIRECTIONS.iter().enumerate() {
                let pc = ((centre[0] + dir[0] * probe_m) / cell).round() as i32;
                let pr = ((centre[1] + dir[1] * probe_m) / cell).round() as i32;
                arms[k] = road_at(world, pc, pr)
                    && world.flow[index(pc, pr)].length_squared() > 0.5;
            }
            junctions.push(Junction { centre, arms });
        }
    }
    junctions.sort_by(|a, b| {
        a.centre[1]
            .total_cmp(&b.centre[1])
            .then(a.centre[0].total_cmp(&b.centre[0]))
    });
    junctions
}

/// Distance in cells from every road cell to the nearest non-road cell
/// (grid edge counts as non-road). Non-road cells hold 0.
pub fn road_edge_distance(world: &VoxelWorld) -> Vec<u32> {
    let size = world.size as i32;
    let index = |col: i32, row: i32| (row * size + col) as usize;
    let mut distance = vec![u32::MAX; world.grid.len()];
    let mut queue = std::collections::VecDeque::new();
    for row in 0..size {
        for col in 0..size {
            let edge = col == 0 || row == 0 || col == size - 1 || row == size - 1;
            if !road_at(world, col, row) {
                distance[index(col, row)] = 0;
                queue.push_back((col, row));
            } else if edge {
                distance[index(col, row)] = 1;
                queue.push_back((col, row));
            }
        }
    }
    while let Some((col, row)) = queue.pop_front() {
        let d = distance[index(col, row)];
        for (dc, dr) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let (nc, nr) = (col + dc, row + dr);
            if nc < 0 || nr < 0 || nc >= size || nr >= size {
                continue;
            }
            if distance[index(nc, nr)] == u32::MAX {
                distance[index(nc, nr)] = d + 1;
                queue.push_back((nc, nr));
            }
        }
    }
    for value in &mut distance {
        if *value == u32::MAX {
            *value = 0;
        }
    }
    distance
}

/// Birth slots for a non-grid city: lane-centre cells (two cells from the road
/// edge, so the car body and its 3×3 footprint are on road), carrying a lane
/// flow, at least 18 m from every station and junction centre, ordered by
/// distance to the depot and thinned to 8 m separation. The grid city keeps its
/// own lattice walk in `organism_city::spawn_slots`.
pub fn spawn_slots_generic(
    world: &VoxelWorld,
    stations: &[[f32; 8]],
    n: usize,
) -> Result<Vec<CitySpawn>, String> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let size = world.size as i32;
    let cell = world.cell_size;
    let edge = road_edge_distance(world);
    let junctions = detect_junctions(world);
    let depot = world.start_pos;
    let mut candidates: Vec<CitySpawn> = Vec::new();
    for row in 0..size {
        for col in 0..size {
            let index = (row * size + col) as usize;
            if edge[index] != 2 {
                continue;
            }
            let flow = world.flow[index];
            if flow.length_squared() < 0.5 {
                continue;
            }
            let (x_m, z_m) = (col as f32 * cell, row as f32 * cell);
            if stations
                .iter()
                .any(|s| (s[0] - x_m).hypot(s[1] - z_m) < SPAWN_STATION_CLEARANCE_M)
            {
                continue;
            }
            if junctions
                .iter()
                .any(|j| (j.centre[0] - x_m).hypot(j.centre[1] - z_m) < SPAWN_JUNCTION_CLEARANCE_M)
            {
                continue;
            }
            candidates.push(CitySpawn {
                position: [x_m, 0.5, z_m],
                forward: [flow.x, 0.0, flow.y],
                depot_distance_m: (x_m - depot.x).hypot(z_m - depot.z),
            });
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
            "family city has only {} separated lane spawn slots; requested {n}",
            slots.len()
        ))
    }
}

/// Station-pair road statistics from the immutable route fields.
pub struct RouteStats {
    pub min_pair_road_m: f32,
    pub max_pair_road_m: f32,
    /// Ordered pairs satisfying the cross-city course thresholds: road ≥ 300 m,
    /// straight-line ≥ 200 m, both axis differences ≥ 60 m.
    pub cross_city_capable_pairs: usize,
    pub unreachable_pairs: usize,
}

pub fn route_stats(world: &VoxelWorld, stations: &[[f32; 8]]) -> Result<RouteStats, String> {
    let packed = crate::gpu::taxi_routes::pack_distance_fields(world, stations)
        .map_err(|e| format!("route preprocessing failed: {e}"))?;
    let cells = world.size * world.size;
    let cell_index = |s: &[f32; 8]| {
        let col = (s[0] / world.cell_size).round() as usize;
        let row = (s[1] / world.cell_size).round() as usize;
        row * world.size + col
    };
    let mut stats = RouteStats {
        min_pair_road_m: f32::INFINITY,
        max_pair_road_m: 0.0,
        cross_city_capable_pairs: 0,
        unreachable_pairs: 0,
    };
    for (origin, from) in stations.iter().enumerate() {
        for (destination, to) in stations.iter().enumerate() {
            if origin == destination {
                continue;
            }
            let road = packed[4 + destination * cells + cell_index(from)];
            if !road.is_finite() || road < 0.0 {
                stats.unreachable_pairs += 1;
                continue;
            }
            stats.min_pair_road_m = stats.min_pair_road_m.min(road);
            stats.max_pair_road_m = stats.max_pair_road_m.max(road);
            let dx = (to[0] - from[0]).abs();
            let dz = (to[1] - from[1]).abs();
            if road >= 300.0 && dx.hypot(dz) >= 200.0 && dx >= 60.0 && dz >= 60.0 {
                stats.cross_city_capable_pairs += 1;
            }
        }
    }
    if stations.len() < 2 {
        stats.min_pair_road_m = 0.0;
    }
    Ok(stats)
}
