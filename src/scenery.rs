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
    let cx = 360.0_f32;
    let cz = 360.0_f32;

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

    // ---- 3. Sun: bright glowing sphere far in the sky ----
    // Direction from world center toward DirectionalLight pos (360, 600, 240).
    // Light is at world (360, 600, 240) and world center is (360, 0, 360):
    // direction = (0, 600, -120) normalized.
    let dir = Vec3::new(0.0, 600.0, -120.0).normalize();
    let sun_pos = Vec3::new(cx, 0.0, cz) + dir * 4000.0;

    let sun_e = commands.spawn((
        Mesh3d(meshes.add(Sphere::new(80.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(1.0, 0.95, 0.7),
            emissive: LinearRgba::new(8.0, 7.5, 5.5, 1.0).into(),
            unlit: true,
            ..default()
        })),
        Transform::from_translation(sun_pos),
        SimulationEntity,
    )).id();
    entities.push(sun_e);

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
