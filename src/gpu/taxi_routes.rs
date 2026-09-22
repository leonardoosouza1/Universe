//! Immutable road distance fields. Geometry preprocessing only; all per-tick
//! evaluation lives in taxi.wgsl. Four-neighbor distances have metre units.
use crate::world::{VoxelType, VoxelWorld};
use std::collections::VecDeque;
pub const ROAD_POTENTIAL_VERSION: &str = "road-distance-v1";
pub const UNREACHABLE: f32 = -1.0;

pub fn pack_distance_fields(world: &VoxelWorld, stations: &[[f32; 8]]) -> Result<Vec<f32>, &'static str> {
    let size = world.size;
    if size == 0 || !world.cell_size.is_finite() || world.cell_size <= 0.0 || world.grid.len() != size * size {
        return Err("invalid route grid");
    }
    let cells = size * size;
    let mut result = vec![size as f32, world.cell_size, stations.len() as f32, 1.0];
    for station in stations {
        let x = (station[0] / world.cell_size).floor();
        let z = (station[1] / world.cell_size).floor();
        if !x.is_finite() || !z.is_finite() || x < 0.0 || z < 0.0 || x >= size as f32 || z >= size as f32 {
            return Err("station outside route grid");
        }
        let target = z as usize * size + x as usize;
        if world.grid[target] != VoxelType::Road { return Err("station is not on a road"); }
        let mut distances = vec![UNREACHABLE; cells];
        distances[target] = 0.0;
        let mut queue = VecDeque::from([target]);
        while let Some(index) = queue.pop_front() {
            let row = index / size;
            let col = index % size;
            for (dx, dz) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                let nx = col as isize + dx;
                let nz = row as isize + dz;
                if nx < 0 || nz < 0 || nx >= size as isize || nz >= size as isize { continue; }
                let next = nz as usize * size + nx as usize;
                if world.grid[next] == VoxelType::Road && distances[next] == UNREACHABLE {
                    distances[next] = distances[index] + world.cell_size;
                    queue.push_back(next);
                }
            }
        }
        result.extend(distances);
    }
    Ok(result)
}
