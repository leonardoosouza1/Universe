//! Atmospheric horizon decoration: sea plane, mountain ring, sun.
//!
//! Pure visual scenery — never collides with cars (cars stay inside the
//! 720m voxel grid; mountains sit ~1500m out). All entities are tagged with
//! `SimulationEntity` so race reset cleans them up.

use bevy::prelude::*;
use crate::SimulationEntity;

/// Cheap deterministic PRNG — splitmix64. Same pattern as
/// `world::procedural_waypoints`. Returns f32 in [0, 1).
fn make_rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    move || {
        s = s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z = z ^ (z >> 31);
        ((z >> 11) as f64 / (1u64 << 53) as f64) as f32
    }
}

/// Spawn sea plane, mountain ring, and sun. Returns the spawned entity IDs
/// so the caller can register them in `RaceSettings::world_entities` for
/// cleanup on race reset.
pub fn spawn_scenery(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) -> Vec<Entity> {
    let mut entities: Vec<Entity> = Vec::new();

    // World center: voxel grid 240×240 cells × 3m -> 720m, center (360, _, 360).
    // Grid 320 cells × 3m = 960m world. Center at (480, _, 480).
    let cx = 480.0_f32;
    let cz = 480.0_f32;

    // ---- 1. Sea plane (10000m × 10000m, just below grass) ----
    let sea_e = commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(10000.0, 10000.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.10, 0.28, 0.45),
            perceptual_roughness: 0.15,
            metallic: 0.1,
            ..default()
        })),
        Transform::from_xyz(cx, -2.0, cz),
        SimulationEntity,
    )).id();
    entities.push(sea_e);

    // ---- 2. Mountain ring — 24 cones around the play area at radius ~1500m ----
    let ring_radius = 1500.0_f32;
    let n_mountains = 24;
    let mountain_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.25, 0.32, 0.30),
        perceptual_roughness: 0.95,
        metallic: 0.0,
        ..default()
    });

    let mut rng = make_rng(0xA1CE_CA57_u64);
    for i in 0..n_mountains {
        let theta = (i as f32 / n_mountains as f32) * std::f32::consts::TAU;
        let radius_jitter = (rng() - 0.5) * 200.0; // ±100m
        let r = ring_radius + radius_jitter;
        let base_radius = 80.0 + rng() * 70.0;   // 80..150
        let height = 200.0 + rng() * 200.0;      // 200..400

        let x = cx + r * theta.cos();
        let z = cz + r * theta.sin();
        // Cone sits on Y=0 (its base center is at the entity origin offset by
        // height/2 in default builder anchor — Bevy 0.16 default anchors the
        // cone such that origin is between base and tip; place origin so base
        // sits ~at sea level).
        let y = height * 0.5 - 2.0;

        let e = commands.spawn((
            Mesh3d(meshes.add(Cone::new(base_radius, height))),
            MeshMaterial3d(mountain_mat.clone()),
            Transform::from_xyz(x, y, z),
            SimulationEntity,
        )).id();
        entities.push(e);
    }

    // ---- 3. Sun + halo (sphere + larger ghost glow) ----
    let dir = Vec3::new(0.0, 600.0, -120.0).normalize();
    let sun_pos = Vec3::new(cx, 0.0, cz) + dir * 4000.0;

    // Sun core (bright disc)
    let sun_e = commands.spawn((
        Mesh3d(meshes.add(Sphere::new(120.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.95, 0.7),
            emissive: LinearRgba::new(15.0, 13.0, 8.0, 1.0).into(),
            unlit: true,
            ..default()
        })),
        Transform::from_translation(sun_pos),
        SimulationEntity,
    )).id();
    entities.push(sun_e);

    // Sun halo (ghost sphere, larger + softer — bloom amplifica)
    let halo_e = commands.spawn((
        Mesh3d(meshes.add(Sphere::new(280.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(1.0, 0.92, 0.65, 0.4),
            emissive: LinearRgba::new(3.5, 2.8, 1.5, 1.0).into(),
            unlit: true,
            alpha_mode: AlphaMode::Add, // additive blending = glow
            ..default()
        })),
        Transform::from_translation(sun_pos),
        SimulationEntity,
    )).id();
    entities.push(halo_e);

    // ---- 3b. Moon (oposto ao sol, prateado) ----
    let moon_dir = Vec3::new(0.0, 300.0, 200.0).normalize();
    let moon_pos = Vec3::new(cx, 0.0, cz) + moon_dir * 3500.0;
    let moon_e = commands.spawn((
        Mesh3d(meshes.add(Sphere::new(70.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.92, 0.94, 0.98),
            emissive: LinearRgba::new(2.8, 3.0, 3.5, 1.0).into(),
            unlit: true,
            ..default()
        })),
        Transform::from_translation(moon_pos),
        SimulationEntity,
    )).id();
    entities.push(moon_e);

    // ---- 4. URBAN CITY RING — buildings ao redor da pista (vibe street race) ----
    // Anel de prédios entre o grid (720m) e as montanhas (1500m). Boxes de
    // alturas variadas, cores de concreto/vidro, placement aleatório no anel.
    let n_buildings = 80;
    let inner_r = 380.0_f32; // logo fora do grid 720m
    let outer_r = 900.0_f32; // antes das montanhas 1500m
    let mut bld_rng = make_rng(0xC1_D7_BE_EF_u64);

    for _ in 0..n_buildings {
        let theta = bld_rng() * std::f32::consts::TAU;
        let r = inner_r + bld_rng() * (outer_r - inner_r);
        let bx = cx + r * theta.cos();
        let bz = cz + r * theta.sin();
        let bw = 12.0 + bld_rng() * 18.0;   // 12..30m wide
        let bd = 12.0 + bld_rng() * 18.0;   // 12..30m deep
        let bh = 25.0 + bld_rng() * 110.0;  // 25..135m tall (highrises mix)
        // Cor: tons concreto cinza/bege/azulado pra variedade urbana.
        let cv = bld_rng();
        let color = if cv < 0.33 {
            Color::srgb(0.55, 0.55, 0.58) // concreto
        } else if cv < 0.66 {
            Color::srgb(0.62, 0.58, 0.50) // bege
        } else {
            Color::srgb(0.40, 0.46, 0.55) // glass blue
        };
        let mat = materials.add(StandardMaterial {
            base_color: color,
            perceptual_roughness: 0.6,
            metallic: 0.15,
            reflectance: 0.4,
            ..default()
        });
        let e = commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(bw, bh, bd))),
            MeshMaterial3d(mat),
            Transform::from_xyz(bx, bh * 0.5 - 2.0, bz),
            SimulationEntity,
        )).id();
        entities.push(e);

        // Janelas iluminadas (emissive cube no topo) — só pros prédios mais altos.
        if bh > 80.0 && bld_rng() < 0.4 {
            let win_mat = materials.add(StandardMaterial {
                base_color: Color::srgb(1.0, 0.9, 0.6),
                emissive: LinearRgba::new(2.0, 1.6, 0.8, 1.0).into(),
                unlit: true,
                ..default()
            });
            let we = commands.spawn((
                Mesh3d(meshes.add(Cuboid::new(bw * 0.8, 4.0, bd * 0.8))),
                MeshMaterial3d(win_mat),
                Transform::from_xyz(bx, bh - 10.0, bz),
                SimulationEntity,
            )).id();
            entities.push(we);
        }
    }

    entities
}

/// Streetlights along track + buildings INSIDE the track loop interior.
/// Reads voxel world's waypoints and min_dist_to_road to place city decorations
/// only where they don't interfere with the racing surface.
pub fn spawn_track_decorations(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    world: &crate::world::VoxelWorld,
) -> Vec<Entity> {
    let mut entities: Vec<Entity> = Vec::new();
    if world.waypoints_world.is_empty() { return entities; }

    // ---- Streetlights along track (every 8th waypoint, alternating sides) ----
    let lamp_post_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.18, 0.18, 0.20),
        perceptual_roughness: 0.4,
        metallic: 0.7,
        ..default()
    });
    let lamp_glow_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.85, 0.55),
        emissive: LinearRgba::new(8.0, 6.0, 2.5, 1.0).into(),
        unlit: true,
        ..default()
    });
    let post_mesh = meshes.add(Cuboid::new(0.4, 8.0, 0.4));
    let lamp_mesh = meshes.add(Sphere::new(0.6));
    let wps = &world.waypoints_world;
    let stride = 8usize;
    for i in (0..wps.len()-1).step_by(stride) {
        let p = wps[i];
        let p_next = wps[(i + 1).min(wps.len() - 1)];
        let dir = (p_next - p).normalize_or_zero();
        let perp = Vec2::new(-dir.y, dir.x);
        let side = if (i / stride) % 2 == 0 { 1.0 } else { -1.0 };
        let lamp_pos_2d = p + perp * (world.cell_size * 12.0) * side; // ~36m off-center
        let lamp_pos = Vec3::new(lamp_pos_2d.x, 4.0, lamp_pos_2d.y);
        let post_e = commands.spawn((
            Mesh3d(post_mesh.clone()),
            MeshMaterial3d(lamp_post_mat.clone()),
            Transform::from_translation(lamp_pos),
            SimulationEntity,
        )).id();
        entities.push(post_e);
        let lamp_e = commands.spawn((
            Mesh3d(lamp_mesh.clone()),
            MeshMaterial3d(lamp_glow_mat.clone()),
            Transform::from_xyz(lamp_pos.x, 8.0, lamp_pos.z),
            PointLight {
                color: Color::srgb(1.0, 0.88, 0.6),
                intensity: 80_000.0,
                range: 25.0,
                shadows_enabled: false,
                ..default()
            },
            SimulationEntity,
        )).id();
        entities.push(lamp_e);
    }

    // ---- Interior buildings: spawn inside the track loop where there's grass ----
    // Strategy: sample random cells in grid; keep only those with grass voxel AND
    // dist >= 25 cells from road (well inside loop interior or far from track).
    let mut bld_rng = make_rng(0xCAFE_BEEF_u64 ^ wps.len() as u64);
    let mut placed = 0;
    let target = 60;
    let mut tries = 0;
    while placed < target && tries < 2000 {
        tries += 1;
        let cx_cell = (bld_rng() * world.size as f32) as usize;
        let cy_cell = (bld_rng() * world.size as f32) as usize;
        if cx_cell >= world.size || cy_cell >= world.size { continue; }
        let idx = cy_cell * world.size + cx_cell;
        if world.grid[idx] != crate::world::VoxelType::Grass { continue; }
        let d = world.min_dist_to_road[idx];
        if d < 25.0 { continue; } // muito perto da pista — não bloqueia
        // Building dims
        let bw = 8.0 + bld_rng() * 12.0;
        let bd = 8.0 + bld_rng() * 12.0;
        let bh = 15.0 + bld_rng() * 60.0;
        let bx = cx_cell as f32 * world.cell_size + world.cell_size * 0.5;
        let bz = cy_cell as f32 * world.cell_size + world.cell_size * 0.5;
        let cv = bld_rng();
        let color = if cv < 0.33 {
            Color::srgb(0.50, 0.52, 0.55)
        } else if cv < 0.66 {
            Color::srgb(0.60, 0.55, 0.48)
        } else {
            Color::srgb(0.38, 0.44, 0.52)
        };
        let mat = materials.add(StandardMaterial {
            base_color: color,
            perceptual_roughness: 0.55,
            metallic: 0.2,
            reflectance: 0.45,
            ..default()
        });
        let e = commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(bw, bh, bd))),
            MeshMaterial3d(mat),
            Transform::from_xyz(bx, bh * 0.5, bz),
            SimulationEntity,
        )).id();
        entities.push(e);
        // Janelas iluminadas em prédios altos
        if bh > 50.0 && bld_rng() < 0.5 {
            let win_mat = materials.add(StandardMaterial {
                base_color: Color::srgb(1.0, 0.92, 0.65),
                emissive: LinearRgba::new(3.0, 2.4, 1.2, 1.0).into(),
                unlit: true,
                ..default()
            });
            let we = commands.spawn((
                Mesh3d(meshes.add(Cuboid::new(bw * 0.7, 3.0, bd * 0.7))),
                MeshMaterial3d(win_mat),
                Transform::from_xyz(bx, bh - 8.0, bz),
                SimulationEntity,
            )).id();
            entities.push(we);
        }
        placed += 1;
    }

    entities
}
