//! Versioned organism-taxi scenario geometry and interface contract.
use crate::world::{VoxelType, VoxelWorld};
use serde_json::{json, Value};

/// Fixed semantics; legacy race sensory channels retain their original meanings.
pub const SCENARIO_ID: &str = "organism-taxi-v1";
/// The final 32 sensory channels are explicitly reassigned by this scenario.
pub const OBSERVATION_ID: &str = crate::gpu::traffic::TRAFFIC_OBSERVATION_ID;
/// Motor 0..2, shift 3..4, service 5, pickup 6, dropoff 7, near-stop reverse request 8.
pub const ACTION_ID: &str = "vehicle-taxi-intent96-active9-v1";

/// Export only the static world occupancy used by the scenario viewer.
///
/// Grass is implicit; each run is `[row, start_col, end_col_exclusive,
/// material_u32]` with Road=1, Wall=2 and Hole=3. This is geometry IO only and
/// never includes positions, velocities, actions or other simulated state.
pub fn export_world_geometry(world: &VoxelWorld) -> Value {
    json!({
        "encoding": "row-runs-v1",
        "size": world.size,
        "cell_size_m": world.cell_size,
        "runs": encode_row_runs(world.size, &world.grid),
    })
}

fn encode_row_runs(size: usize, grid: &[VoxelType]) -> Vec<Value> {
    let mut runs = Vec::new();
    for row in 0..size {
        let mut col = 0;
        while col < size {
            let material = grid
                .get(row.saturating_mul(size).saturating_add(col))
                .copied()
                .unwrap_or(VoxelType::Grass);
            let material_u32 = match material {
                VoxelType::Grass => 0,
                VoxelType::Road => 1,
                VoxelType::Wall => 2,
                VoxelType::Hole => 3,
            };
            if material_u32 == 0 {
                col += 1;
                continue;
            }
            let start = col;
            col += 1;
            while col < size {
                let next = grid
                    .get(row.saturating_mul(size).saturating_add(col))
                    .copied()
                    .unwrap_or(VoxelType::Grass);
                let next_u32 = match next {
                    VoxelType::Grass => 0,
                    VoxelType::Road => 1,
                    VoxelType::Wall => 2,
                    VoxelType::Hole => 3,
                };
                if next_u32 != material_u32 {
                    break;
                }
                col += 1;
            }
            runs.push(json!([row, start, col, material_u32]));
        }
    }
    runs
}

/// Add two connected cross streets to the existing closed road network.
/// Static geometry construction is host work; simulated dynamics remain on GPU.
pub fn connect_city(world: &mut VoxelWorld) -> Vec<[f32; 8]> {
    crate::organism_city::build_city(world).stations
}

/// Reject a station disconnected from the spawn via traversable cells.
pub fn validate_city(world: &VoxelWorld, stations: &[[f32; 8]]) -> Result<(), String> {
    let cell = |x: f32, z: f32| -> Option<usize> {
        if !x.is_finite() || !z.is_finite() || x < 0.0 || z < 0.0 {
            return None;
        }
        let x = (x / world.cell_size).round() as usize;
        let z = (z / world.cell_size).round() as usize;
        (x < world.size && z < world.size).then_some(z * world.size + x)
    };
    let start = cell(world.start_pos.x, world.start_pos.z).ok_or("spawn outside city")?;
    let mut visited = vec![false; world.grid.len()];
    let mut queue = std::collections::VecDeque::from([start]);
    visited[start] = true;
    while let Some(index) = queue.pop_front() {
        let x = index % world.size;
        let z = index / world.size;
        for (nx, nz) in [
            (x.wrapping_sub(1), z),
            (x + 1, z),
            (x, z.wrapping_sub(1)),
            (x, z + 1),
        ] {
            if nx < world.size && nz < world.size {
                let next = nz * world.size + nx;
                if !visited[next] && world.grid[next] == VoxelType::Road {
                    visited[next] = true;
                    queue.push_back(next);
                }
            }
        }
    }
    for (i, station) in stations.iter().enumerate() {
        let index = cell(station[0], station[1]).ok_or("station outside city")?;
        if world.grid[index] != VoxelType::Road || !visited[index] {
            return Err(format!("station {i} is inaccessible"));
        }
    }
    Ok(())
}

/// Episode score: completed work, bounded active-task progress, and survival;
/// dying early cannot avoid an age cost. Slot127 is recomputed from pose and
/// destination each tick, never accumulated by the taxi adapter.
pub fn fitness(body: &crate::gpu::organism::OrganismState) -> f64 {
    let v = &body.values;
    f64::from(
        1000.0 * v[68] + v[127] + v[64] + 0.001 * v[3] + 10.0 * v[29]
            - 100.0 * (1.0 - v[8])
            - 100.0 * (1.0 - v[29]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_runs_encode_non_grass_contiguous_materials() {
        let grid = vec![
            VoxelType::Grass,
            VoxelType::Road,
            VoxelType::Road,
            VoxelType::Wall,
            VoxelType::Hole,
            VoxelType::Grass,
            VoxelType::Hole,
            VoxelType::Hole,
        ];
        let runs = encode_row_runs(4, &grid);
        assert_eq!(
            runs,
            vec![
                json!([0, 1, 3, 1]),
                json!([0, 3, 4, 2]),
                json!([1, 0, 1, 3]),
                json!([1, 2, 4, 3]),
            ]
        );
    }
}
