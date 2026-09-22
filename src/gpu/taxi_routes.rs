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
        // Cell i is centred on i*cell_size (world.rs::get_voxel_at); round, never floor.
        let x = (station[0] / world.cell_size).round();
        let z = (station[1] / world.cell_size).round();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::VoxelType;

    /// Station 1.0 m left of a cell centre: `floor` picked the neighbouring cell,
    /// `round` (world.rs convention) keeps the centre cell. Static geometry only.
    #[test]
    fn station_cell_uses_round_half_away_from_zero() {
        let _quiet = crate::world::silence_map_diag();
        let mut road = VoxelWorld::new(0, 15.0);
        road.size = 5;
        road.cell_size = 3.0;
        road.grid = vec![VoxelType::Road; 25];
        let stations = [[5.0, 6.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [12.0, 12.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]];
        let packed = pack_distance_fields(&road, &stations).unwrap();
        assert_eq!(&packed[..4], &[5.0, 3.0, 2.0, 1.0]);
        let field = &packed[4..4 + 25];
        // 5.0/3 = 1.67 -> cell 2 (round), not 1 (floor); 6.0/3 = 2.
        assert_eq!(field[2 * 5 + 2], 0.0, "target is the rounded cell");
        assert_eq!(field[2 * 5 + 1], 3.0, "floor's cell is one step away");
        // Exact tie 4.5/3 = 1.5 -> 2 (half away from zero), matching f32::round.
        let tie = [[4.5, 4.5, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [12.0, 12.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]];
        let packed = pack_distance_fields(&road, &tie).unwrap();
        assert_eq!(packed[4 + 2 * 5 + 2], 0.0);
    }
}
