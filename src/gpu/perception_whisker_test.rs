//! Physical fixture for body-relative diagonal sensing, independent of the oracle.
use super::*;
use crate::gpu::{Pose, RetinaGpu};
use crate::world::{VoxelType, VoxelWorld};
use bevy::math::{Vec2, Vec3};

#[test]
#[ignore = "requires exclusive GPU access"]
fn diagonal_whiskers_rotate_with_body_despite_zero_legacy_phase() {
    let mut world = VoxelWorld::new(42, 28.0);
    world.grid.fill(VoxelType::Grass);
    // From (30,30), a +30-degree probe hits the cell centered at (36,33)
    // first at the 7 m sample. A 90-degree body turn moves it to the -60 probe.
    let index = 11 * world.size + 12;
    world.grid[index] = VoxelType::Wall;
    let retina = RetinaGpu::new(&world, 2).expect("GPU required for this explicit test");
    let perception = PerceptionGpu::new(retina.device(), retina.queue(), &world, 2)
        .expect("perception pipeline");
    let poses = [Vec2::X, Vec2::Y].map(|forward| Pose {
        pos: Vec3::new(30.0, 0.0, 30.0),
        forward,
        right: Vec2::new(-forward.y, forward.x),
        gaze_yaw: 0.0,
    });
    retina.render(&poses);
    let inputs = poses.map(|p| PosePerception {
        pos_x: p.pos.x,
        pos_y: p.pos.y,
        pos_z: p.pos.z,
        phase: 0.0,
        fwd_x: p.forward.x,
        fwd_z: p.forward.y,
        rgt_x: p.right.x,
        rgt_z: p.right.y,
    });
    let waves = perception.reduce(
        retina.device(),
        retina.queue(),
        &inputs,
        retina.output_buffer(),
    );
    assert_eq!(&waves[0][94..98], &[38, 0, 0, 0]);
    assert_eq!(&waves[1][94..98], &[0, 0, 0, 38]);
}
