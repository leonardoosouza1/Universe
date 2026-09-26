//! Lossless terrain facts needed by scene replay viewers.

use crate::world::{VoxelType, VoxelWorld};
use serde_json::{Value, json};

/// Snapshot the material cells that are hazards in the recorded world.
/// Coordinates are integer grid indices, in the same row-major order as `VoxelWorld::grid`.
pub fn snapshot(world: &VoxelWorld) -> Value {
    let hole_cells: Vec<[usize; 2]> = world
        .grid
        .iter()
        .enumerate()
        .filter_map(|(index, material)| {
            if *material == VoxelType::Hole {
                Some([index % world.size, index / world.size])
            } else {
                None
            }
        })
        .collect();
    json!({
        "schema_version": 1,
        "cell_size_m": world.cell_size,
        "grid_size": world.size,
        "alignment": "cell_center",
        "hole_cells": hole_cells,
    })
}

#[cfg(test)]
mod tests {
    use super::snapshot;
    use crate::world::{VoxelType, VoxelWorld, silence_map_diag};

    #[test]
    fn snapshot_reports_actual_holes_and_schema() {
        let _quiet = silence_map_diag();
        let mut world = VoxelWorld::new(123, 28.0);
        world.grid.fill(VoxelType::Road);
        world.grid[7 * world.size + 4] = VoxelType::Hole;
        world.grid[7 * world.size + 5] = VoxelType::Hole;
        let value = snapshot(&world);
        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["cell_size_m"], 3.0);
        assert_eq!(value["grid_size"], world.size);
        let cells = value["hole_cells"].as_array().unwrap();
        assert_eq!(cells.len(), 2);
        for cell in cells {
            let x = cell[0].as_u64().unwrap() as usize;
            let z = cell[1].as_u64().unwrap() as usize;
            assert_eq!(world.grid[z * world.size + x], VoxelType::Hole);
        }
    }

    #[test]
    fn no_holes_snapshot_is_empty() {
        let _quiet = silence_map_diag();
        let mut world = VoxelWorld::new(123, 28.0);
        world.grid.fill(VoxelType::Road);
        assert!(
            snapshot(&world)["hole_cells"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }
}
