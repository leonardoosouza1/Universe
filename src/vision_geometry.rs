//! Physical ray geometry shared by the CPU retina and the GPU implementation.

use bevy::prelude::Vec3;

use crate::gpu::Pixel;
use crate::world::{VoxelType, VoxelWorld};

const MAX_DISTANCE_M: f32 = 1200.0;
const MAX_STEPS: usize = 1024;

fn cell_index(coordinate: f32, cell_size: f32) -> i32 {
    // A voxel's centre is at index * cell_size; boundaries are half a cell away.
    (coordinate / cell_size).round() as i32
}

fn material(world: &VoxelWorld, x: i32, z: i32) -> VoxelType {
    if x >= 0 && z >= 0 && x < world.size as i32 && z < world.size as i32 {
        world.grid[z as usize * world.size + x as usize]
    } else {
        VoxelType::Wall
    }
}

fn ground_material(world: &VoxelWorld, x: f32, z: f32) -> VoxelType {
    material(
        world,
        cell_index(x, world.cell_size),
        cell_index(z, world.cell_size),
    )
}

/// Trace the nearest physical surface along `dir`.
///
/// `dir` must be normalized. Wall cells are opaque during DDA traversal. Hole cells
/// are ground materials and are returned only when the ray reaches the ground plane.
pub fn trace(world: &VoxelWorld, eye: Vec3, dir: Vec3) -> Pixel {
    let horizontal = Vec2::new(dir.x, dir.z);
    let horizontal_len = horizontal.length();
    let ground_distance = if dir.y < -1e-6 {
        (-eye.y / dir.y).max(0.0)
    } else {
        f32::INFINITY
    };
    let limit = ground_distance.min(MAX_DISTANCE_M);

    if ground_material(world, eye.x, eye.z) == VoxelType::Wall {
        return Pixel {
            dist_m: 0.0,
            material: VoxelType::Wall as u8,
        };
    }
    if horizontal_len <= 1e-6 {
        let material = if ground_distance <= MAX_DISTANCE_M {
            ground_material(world, eye.x, eye.z)
        } else {
            VoxelType::Road
        };
        return Pixel {
            dist_m: limit,
            material: material as u8,
        };
    }

    let ray = horizontal / horizontal_len;
    let start_x = eye.x / world.cell_size + 0.5;
    let start_z = eye.z / world.cell_size + 0.5;
    let mut map_x = cell_index(eye.x, world.cell_size);
    let mut map_z = cell_index(eye.z, world.cell_size);

    let delta_x = if ray.x.abs() <= 1e-6 {
        f32::INFINITY
    } else {
        1.0 / ray.x.abs()
    };
    let delta_z = if ray.y.abs() <= 1e-6 {
        f32::INFINITY
    } else {
        1.0 / ray.y.abs()
    };
    let step_x = if ray.x < 0.0 { -1 } else { 1 };
    let step_z = if ray.y < 0.0 { -1 } else { 1 };
    let mut side_x = if ray.x < 0.0 {
        (start_x - map_x as f32) * delta_x
    } else {
        (map_x as f32 + 1.0 - start_x) * delta_x
    };
    let mut side_z = if ray.y < 0.0 {
        (start_z - map_z as f32) * delta_z
    } else {
        (map_z as f32 + 1.0 - start_z) * delta_z
    };

    for _ in 0..MAX_STEPS {
        let side_distance = if side_x < side_z {
            map_x += step_x;
            let d = side_x;
            side_x += delta_x;
            d
        } else {
            map_z += step_z;
            let d = side_z;
            side_z += delta_z;
            d
        };
        let distance = side_distance * world.cell_size / horizontal_len;
        if distance > limit {
            break;
        }
        if material(world, map_x, map_z) == VoxelType::Wall {
            return Pixel {
                dist_m: distance,
                material: VoxelType::Wall as u8,
            };
        }
    }

    if ground_distance <= MAX_DISTANCE_M {
        Pixel {
            dist_m: ground_distance,
            material: ground_material(
                world,
                eye.x + dir.x * ground_distance,
                eye.z + dir.z * ground_distance,
            ) as u8,
        }
    } else {
        Pixel {
            dist_m: MAX_DISTANCE_M,
            material: VoxelType::Road as u8,
        }
    }
}

use bevy::prelude::Vec2;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::silence_map_diag;

    fn road_world() -> VoxelWorld {
        let _quiet = silence_map_diag();
        let mut world = VoxelWorld::new(0, 28.0);
        world.grid.fill(VoxelType::Road);
        world
    }

    #[test]
    fn occupied_start_is_zero_distance_wall() {
        let mut world = road_world();
        let x = world.size / 2;
        let z = world.size / 2;
        world.grid[z * world.size + x] = VoxelType::Wall;
        let eye = Vec3::new(x as f32 * world.cell_size, 1.0, z as f32 * world.cell_size);
        let px = trace(&world, eye, Vec3::X);
        assert_eq!(px.dist_m, 0.0);
        assert_eq!(px.material, VoxelType::Wall as u8);
        assert_eq!(trace(&world, eye, -Vec3::Y).dist_m, 0.0);
        assert_eq!(
            trace(&world, Vec3::new(-1.5, 1.0, eye.z), Vec3::X).dist_m,
            0.0
        );
    }

    #[test]
    fn wall_distance_is_three_dimensional_and_left_right_symmetric() {
        let mut world = road_world();
        let z = world.size / 2;
        let x = world.size / 2 + 16;
        world.grid[z * world.size + x] = VoxelType::Wall;
        let eye = Vec3::new((world.size / 2) as f32 * 3.0, 10.0, z as f32 * 3.0);
        let pitch: f32 = 0.6;
        let horizontal = pitch.cos();
        let expected = 46.5 / horizontal;
        let right = trace(&world, eye, Vec3::new(horizontal, pitch.sin(), 0.0));
        let left_x = world.size / 2 - 16;
        world.grid[z * world.size + left_x] = VoxelType::Wall;
        let left = trace(&world, eye, Vec3::new(-horizontal, pitch.sin(), 0.0));
        assert!(
            (right.dist_m - expected).abs() < 1e-4,
            "{} != {}",
            right.dist_m,
            expected
        );
        assert!((left.dist_m - expected).abs() < 1e-4);
    }

    #[test]
    fn hole_is_ground_material_and_does_not_occlude() {
        let mut world = road_world();
        let z = world.size / 2;
        let x0 = world.size / 2;
        world.grid[z * world.size + x0 + 3] = VoxelType::Hole;
        world.grid[z * world.size + x0 + 6] = VoxelType::Wall;
        let eye = Vec3::new(x0 as f32 * 3.0, 1.0, z as f32 * 3.0);
        let direction = Vec3::new(1.0, -0.1, 0.0).normalize();
        let px = trace(&world, eye, direction);
        assert_eq!(px.material, VoxelType::Hole as u8);
        assert!((px.dist_m - 101.0_f32.sqrt()).abs() < 1e-4);
    }

    #[test]
    fn hole_does_not_occlude_a_shallow_ray_to_a_wall() {
        let mut world = road_world();
        let z = world.size / 2;
        let x0 = world.size / 2;
        world.grid[z * world.size + x0 + 3] = VoxelType::Hole;
        world.grid[z * world.size + x0 + 6] = VoxelType::Wall;
        let eye = Vec3::new(x0 as f32 * 3.0, 1.0, z as f32 * 3.0);
        let px = trace(&world, eye, Vec3::X);
        assert_eq!(px.material, VoxelType::Wall as u8);
        assert!((px.dist_m - 16.5).abs() < 1e-4);
    }
}
