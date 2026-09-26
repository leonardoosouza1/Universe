//! Independent analytic fixtures and full-ray CPU/GPU parity for road optics.
use crate::config::iara::vision::{RETINA_COLS, RETINA_ROWS};
use crate::{
    gpu::{Pose, RetinaGpu},
    visao, vision_geometry,
    world::{VoxelType, VoxelWorld},
};
use bevy::math::{Vec2, Vec3};

#[test]
#[ignore = "requires a real compute adapter; run explicitly for retinal changes"]
fn road_optics_preserve_distance_material_and_visibility_on_gpu() {
    let mut world = VoxelWorld::new(42, 28.0);
    world.grid.fill(VoxelType::Road);
    let center = world.size / 2;
    // A bilateral corridor, a transverse wall 60m away and a floor hole 12m away.
    for z in center - 30..=center + 30 {
        world.grid[z * world.size + center + 20] = VoxelType::Wall;
        world.grid[z * world.size + center - 20] = VoxelType::Wall;
    }
    world.grid[center * world.size + center + 4] = VoxelType::Hole;
    let origin = center as f32 * world.cell_size;
    let poses = [
        Pose {
            pos: Vec3::new(origin + 0.23, 0.5, origin + 0.17),
            forward: Vec2::X,
            right: Vec2::Y,
            gaze_yaw: 0.0,
        },
        Pose {
            pos: Vec3::new(origin + 0.23, 0.5, origin + 0.17),
            forward: -Vec2::X,
            right: -Vec2::Y,
            gaze_yaw: 0.16,
        },
        Pose {
            pos: Vec3::new(origin + 60.0, 0.5, origin),
            forward: Vec2::X,
            right: Vec2::Y,
            gaze_yaw: 0.0,
        },
        Pose {
            pos: Vec3::new(-1.5, 0.5, origin),
            forward: Vec2::X,
            right: Vec2::Y,
            gaze_yaw: 0.0,
        },
        Pose {
            pos: Vec3::new(origin + 120.0, 0.5, origin),
            forward: Vec2::X,
            right: Vec2::Y,
            gaze_yaw: 0.0,
        },
    ];
    let gpu = RetinaGpu::new(&world, poses.len()).expect("real GPU required, not a skipped pass");
    let pixels = gpu.render_road_view(&poses, 120.0);
    let mut compared = 0;
    for (being, pose) in poses.iter().enumerate() {
        let eye = pose.pos + Vec3::Y * visao::ALTURA_OLHO;
        for row in 0..RETINA_ROWS {
            let pitch = visao::road_view_pitch(120.0, row);
            for col in 0..RETINA_COLS {
                let u = col as f32 / (RETINA_COLS - 1) as f32 * 2.0 - 1.0;
                let yaw = if row == 0 {
                    (u * 1.25).tan() / 1.25f32.tan() * 60f32.to_radians() + pose.gaze_yaw
                } else {
                    u * 60f32.to_radians()
                };
                let horizontal = (pose.forward * yaw.cos() + pose.right * yaw.sin()) * pitch.cos();
                let direction = Vec3::new(horizontal.x, pitch.sin(), horizontal.y);
                let expected = vision_geometry::trace(&world, eye, direction);
                let actual = pixels[being][row * RETINA_COLS + col];
                assert_eq!(
                    actual.material, expected.material,
                    "material being {being} ray {row}:{col}, {actual:?}/{expected:?}"
                );
                assert!(
                    (actual.dist_m - expected.dist_m).abs() < 0.004,
                    "distance being {being} ray {row}:{col}, {actual:?}/{expected:?}"
                );
                compared += 1;
            }
        }
    }
    // The transverse wall is genuinely visible beyond the old 33m ground intercept.
    let far = pixels[0][RETINA_COLS / 2];
    assert_eq!(far.material, VoxelType::Wall as u8);
    assert!(far.dist_m > 58.0 && far.dist_m < 59.0, "{far:?}");
    assert!(
        pixels[2..4]
            .iter()
            .flatten()
            .all(|p| p.dist_m == 0.0 && p.material == 2)
    );
    let ground = pixels[4][RETINA_COLS / 2];
    assert_eq!(ground.material, 1);
    assert!(
        (ground.dist_m * visao::road_view_pitch(120.0, 0).cos() - 120.0).abs() < 0.004,
        "requested 120m ground range must include 0.5m body height: {ground:?}"
    );
    println!(
        "{compared} physical rays agree with CPU: material exact, distance <4mm; wall visible at {:.3}m",
        far.dist_m
    );
}
