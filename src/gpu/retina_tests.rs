//! Numerical retinal encoding and physical occlusion regressions.
use super::*;
use crate::config::iara::vision::{RETINA_COLS, RETINA_TOTAL};
use crate::world::{VoxelType, VoxelWorld};
use bevy::math::{Vec2, Vec3};

#[test]
#[ignore = "requires exclusive GPU access"]
fn encoded_dispatch_readback_completes_after_submission() {
    let world = VoxelWorld::new(42, 28.0);
    let gpu = RetinaGpu::new(&world, 1).expect("GPU required for explicit hardware test");
    let poses = [Pose {
        pos: Vec3::new(30.0, 0.0, 30.0),
        forward: Vec2::X,
        right: Vec2::Y,
        gaze_yaw: 0.0,
    }];
    let expected = gpu.render_legacy(&poses);
    let mut encoder = gpu.device().create_command_encoder(&Default::default());
    let count = gpu.dispatch_into(&mut encoder, &poses);
    gpu.queue().submit(Some(encoder.finish()));
    let actual = gpu.read_legacy_after_submit(count);
    assert_eq!(actual[0].as_slice(), expected[0].as_slice());
}

#[cfg(test)]
mod being_tests {
    use super::*;

    /// [`BEING_MATERIAL`] must encode to a value none of the four grid materials ever
    /// reach, AT ANY distance — not only at the one distance a test happens to pick — so
    /// this checks the closed-form ranges directly instead of sampling a few points.
    /// `docs/RETINA-BEINGS.md` §2 only asks for "the same distance"; the -29 chosen in
    /// `legacy_encoding` clears the stronger bar for free (see the comment there).
    #[test]
    fn being_encoding_never_collides_with_a_grid_material() {
        let being = legacy_encoding(Pixel {
            dist_m: 42.0, // arbitrary — the encoding is flat over distance
            material: BEING_MATERIAL,
        });
        assert_eq!(being, -29);

        // Sample every material densely over the full 0..1200 m the kernel clamps to,
        // and assert none of them ever equals -29.
        let mut d = 0.0f32;
        while d <= 1200.0 {
            for material in [0u8, 1, 2, 3] {
                let v = legacy_encoding(Pixel {
                    dist_m: d,
                    material,
                });
                assert_ne!(
                    v, being,
                    "material {material} at {d} m encodes to {v}, colliding with Being"
                );
            }
            d += 0.5;
        }
    }

    /// A material code outside 0-4 is a bug (the retina kernel never produces one), not
    /// an environment failure — `legacy_encoding` must reject it loudly instead of
    /// silently falling back to the Hole formula, which is what the old `_ =>` arm did.
    #[test]
    #[should_panic(expected = "is not Grass/Road/Wall/Hole/Being")]
    fn legacy_encoding_rejects_an_unknown_material() {
        let _ = legacy_encoding(Pixel {
            dist_m: 10.0,
            material: 5,
        });
    }

    /// Guards the WGSL mirror the way `docs/STYLE.md` §4 asks when a shared constant
    /// cannot be `use`d across the Rust/WGSL boundary: if either literal changes without
    /// the other, this fails instead of the two silently drifting apart — the exact bug
    /// class `docs/STYLE.md` §4 documents for `WAVE_U32S`.
    #[test]
    fn being_radius_wgsl_mirror_matches_the_config_constant() {
        let expected = crate::config::iara::vision::BEING_RADIUS_M;
        assert_eq!(
            expected, 0.775,
            "update the WGSL literal below together with this constant"
        );
        let src = include_str!("retina.wgsl");
        assert!(
            src.contains("const BEING_RADIUS: f32 = 0.775;"),
            "retina.wgsl's BEING_RADIUS literal no longer matches \
             config::iara::vision::BEING_RADIUS_M = {expected}"
        );
    }
}

/// `docs/RETINA-BEINGS.md` §3.3: "the geometry is right, not just plausible" — beings at
/// known positions, on the real GPU kernel AND the CPU reference at once, so a bug only
/// one side has cannot hide behind the other looking right.
///
/// `#[ignore]`d for the same reason as `gpu/fly_map/parity_test.rs`: opens a real
/// `wgpu::Device`. Run explicitly:
/// `HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0 cargo test --release --lib -- \
///   --ignored being_geometry`
#[cfg(test)]
mod being_geometry_gpu_tests {
    use super::*;

    /// `VoxelWorld::new`'s track is recentred near `(GRID_CELLS/2, GRID_CELLS/2)` in cell
    /// units, i.e. world (480, 480) m, and its widest recorded span is on the order of
    /// 330 m (`src/world.rs:1157`, "radius up to ~110 cells"). A
    /// point near a grid corner —
    /// world (30, 30) m here, cell (10, 10) — is over 600 m from the track centre, well
    /// outside the Road/Wall band `VoxelWorld::new` paints
    /// (`grid[idx] = if dist_m < track_width/2 {Road} else if ... {Wall} else {Grass}`),
    /// so it is Grass in every direction for hundreds of metres: an open field to place
    /// synthetic beings in without a real track wall interfering.
    fn open_field_world() -> VoxelWorld {
        VoxelWorld::new(42, 10.0)
    }

    /// The observer: at the open-field point above, facing world +x
    /// (`forward = (1, 0)` in the (x, z) plane `render_binocular_retina` reads), gaze
    /// centred (no saccade).
    fn observer() -> Pose {
        Pose {
            pos: Vec3::new(30.0, 0.0, 30.0),
            forward: Vec2::new(1.0, 0.0),
            right: Vec2::new(0.0, 1.0),
            gaze_yaw: 0.0,
        }
    }

    fn cpu_retina(world: &VoxelWorld, eye: Pose, others: &[Vec3]) -> [i8; RETINA_TOTAL] {
        world.render_binocular_retina(
            eye.pos,
            Vec3::new(eye.forward.x, 0.0, eye.forward.y),
            Vec3::new(eye.right.x, 0.0, eye.right.y),
            eye.gaze_yaw,
            others,
        )
    }

    #[test]
    #[ignore = "opens a real GPU — run manually: cargo test --release --lib -- --ignored being_geometry"]
    fn being_directly_ahead_at_10m_appears_in_the_central_columns() {
        let Some(world_gpu) = RetinaGpu::new(&open_field_world(), 4) else {
            eprintln!("[being_geometry] no GPU available — test skipped");
            return;
        };
        let world = open_field_world();
        let eye = observer();
        let being_pos = Vec3::new(eye.pos.x + 10.0, 0.0, eye.pos.z);
        let poses = vec![
            eye,
            Pose {
                pos: being_pos,
                ..eye
            },
        ];

        let gpu_out = world_gpu.render_with_beings(&poses);
        // Row 0 is the fovea; normalized_i == 0 sits at column (RETINA_COLS - 1) / 2. A
        // small window around it, not one exact index, because the tangent warp puts the
        // true centre at a non-integer column.
        let row0 = &gpu_out[0][0..RETINA_COLS];
        let center = (RETINA_COLS - 1) / 2;
        let hit = (center - 3..=center + 3)
            .map(|c| row0[c])
            .find(|px| px.material == BEING_MATERIAL);
        let px = hit.unwrap_or_else(|| {
            panic!("no pixel in row 0's central columns saw the being 10 m ahead")
        });
        assert!(
            (px.dist_m - 10.0).abs() <= 3.0, // one cell (cell_size = 3.0 m)
            "GPU distance to the being = {} m, expected 10 m +/- one cell",
            px.dist_m
        );

        let cpu = cpu_retina(&world, eye, &[being_pos]);
        assert!(
            (center - 3..=center + 3).any(|c| cpu[c] == -29),
            "CPU reference saw no Being pixel in row 0's central columns either"
        );
    }

    #[test]
    #[ignore = "opens a real GPU — run manually: cargo test --release --lib -- --ignored being_geometry"]
    fn being_at_90_degrees_is_outside_the_120_degree_field_of_view() {
        let Some(world_gpu) = RetinaGpu::new(&open_field_world(), 4) else {
            eprintln!("[being_geometry] no GPU available — test skipped");
            return;
        };
        let world = open_field_world();
        let eye = observer();
        // Exactly along `right` — 90 deg off `forward` — 10 m out. The field of view is
        // 120 deg total (+/- 60 deg half), so no ray can point within 30 deg of this
        // bearing; at 10 m that is metres of lateral slack, far past `BEING_RADIUS_M`.
        let being_pos = Vec3::new(eye.pos.x, 0.0, eye.pos.z + 10.0);
        let poses = vec![
            eye,
            Pose {
                pos: being_pos,
                ..eye
            },
        ];

        let gpu_out = world_gpu.render_with_beings(&poses);
        assert!(
            gpu_out[0].iter().all(|px| px.material != BEING_MATERIAL),
            "a being at 90 deg, outside the 120 deg FOV, showed up in the GPU retina"
        );

        let cpu = cpu_retina(&world, eye, &[being_pos]);
        assert!(
            cpu.iter().all(|&v| v != -29),
            "a being at 90 deg, outside the 120 deg FOV, showed up in the CPU retina"
        );
    }

    #[test]
    #[ignore = "opens a real GPU — run manually: cargo test --release --lib -- --ignored being_geometry"]
    fn being_behind_a_wall_stays_invisible() {
        let world = open_field_world();
        let Some(world_gpu) = RetinaGpu::new(&world, 4) else {
            eprintln!("[being_geometry] no GPU available — test skipped");
            return;
        };
        // Near the grid's negative-x edge (`GRID_CELLS = 320`, cell_size = 3.0 m, so
        // world x = 0 is cell 0): `get_voxel_at`/`voxel_at` return Wall for any position
        // outside 0..320 cells, a deterministic "wall" independent of the procedural
        // track's own geometry (`src/world.rs::get_voxel_at`, mirrored in
        // `retina.wgsl::voxel_at`). Facing -x, the ray leaves the grid within about one
        // cell (~3 m); a being 50 m further out, on the SAME ray, sits far behind it.
        let eye = Pose {
            pos: Vec3::new(1.0, 0.0, 1.0),
            forward: Vec2::new(-1.0, 0.0),
            right: Vec2::new(0.0, 1.0),
            gaze_yaw: 0.0,
        };
        let being_pos = Vec3::new(eye.pos.x - 50.0, 0.0, eye.pos.z);
        let poses = vec![
            eye,
            Pose {
                pos: being_pos,
                ..eye
            },
        ];

        let gpu_out = world_gpu.render_with_beings(&poses);
        let row0 = &gpu_out[0][0..RETINA_COLS];
        let center = (RETINA_COLS - 1) / 2;
        for c in center - 3..=center + 3 {
            assert_ne!(
                row0[c].material, BEING_MATERIAL,
                "column {c}: a being 50 m behind the grid-edge wall was seen instead of \
                 the wall — the nearer DDA hit did not win"
            );
            assert!(
                row0[c].dist_m < 10.0,
                "column {c}: expected the near wall (~3 m), got {} m — the being's 50 m \
                 leaked through",
                row0[c].dist_m
            );
        }

        let cpu = cpu_retina(&world, eye, &[being_pos]);
        for c in center - 3..=center + 3 {
            assert_ne!(
                cpu[c], -29,
                "CPU reference: column {c} saw the being through the wall"
            );
        }
    }
}
