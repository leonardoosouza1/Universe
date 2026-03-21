use bevy::prelude::*;
use crate::simulation::Racer;
use crate::world::{VoxelWorld, VoxelWorldResource};
use rand::prelude::*;
use std::fs::File;
use std::io::Write;

#[derive(Resource)]
pub struct RaceSettings {
    pub generation: usize,
    pub seed: u64,
    pub best_weights: Option<Vec<i8>>,
    pub selected_racer: Option<Entity>,
    pub world_entities: Vec<Entity>,
    pub should_reset: bool,
}

pub struct EvolutionPlugin;

impl Plugin for EvolutionPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, evolution_pipeline);
    }
}

pub fn spawn_swarm(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    race: &RaceSettings,
    world: &VoxelWorld,
) {
    let mut rng = rand::rng();
    
    // MODELO F1 DETALHADO (O QUE FUNCIONAVA)
    let chassis = meshes.add(Cuboid::new(3.0, 0.4, 1.2)); 
    let wing = meshes.add(Cuboid::new(0.6, 0.8, 2.0)); 
    let cockpit = meshes.add(Sphere::new(0.4));
    
    let names = ["Senna", "Hamilton", "Prost", "Schumi", "Alonso", "Vettel", "Max", "Lauda"];

    for i in 0..20 {
        let mut engine = crate::engine::SovereignEngine::new();
        if let Some(ref best) = race.best_weights {
            let mutated: Vec<f32> = best.iter().map(|&w| if rng.random_bool(0.12) { rng.random_range(-1.0..1.0) } else { w as f32 }).collect();
            engine.instinct_layer.quantize(&mutated);
        } else {
            let weights: Vec<f32> = (0..128*64).map(|_| rng.random_range(-1.0..1.0)).collect();
            engine.instinct_layer.quantize(&weights);
        }

        let color = if i == 0 && race.best_weights.is_some() { 
            Color::srgb(0.0, 1.0, 0.8) // Cyan Sovereign
        } else { 
            Color::srgb(rng.random::<f32>(), rng.random::<f32>(), rng.random::<f32>()) 
        };

        let racer_mat = materials.add(StandardMaterial { 
            base_color: color, 
            emissive: LinearRgba::from(color) * 5.0, 
            perceptual_roughness: 0.1,
            metallic: 0.9,
            ..default() 
        });

        commands.spawn((
            Racer { 
                name: format!("{}-{}", names[i % names.len()], rng.random_range(10..99)),
                engine, 
                is_dead: false, 
                distance: 0.0, 
                velocity: 0.0, 
                angle: 0.0, 
                angular_velocity: 0.0, 
                gear: 1, 
                rpm: 0.0, 
                stagnation: 0.0, 
                prev_pos: world.start_pos,
                // --- INITIALIZE 2026 FIELDS ---
                slip_angle: 0.0,
                g_force: 0.0,
                anticipation: [0.0; 5],
            },
            Mesh3d(chassis.clone()),
            MeshMaterial3d(racer_mat.clone()),
            Transform::from_translation(world.start_pos + Vec3::new((i as f32 - 10.0) * 1.8, 0.0, 0.0)),
        )).with_children(|p| {
            // Asa Traseira (Downforce Metafísico)
            p.spawn((Mesh3d(wing.clone()), MeshMaterial3d(racer_mat.clone()), Transform::from_xyz(-1.2, 0.4, 0.0)));
            // Cockpit (O Trono da Consciência)
            p.spawn((
                Mesh3d(cockpit.clone()), 
                MeshMaterial3d(materials.add(StandardMaterial { base_color: Color::WHITE, emissive: LinearRgba::WHITE * 2.0, ..default() })), 
                Transform::from_xyz(0.5, 0.3, 0.0)
            ));
        });
    }

}

fn evolution_pipeline(
    mut commands: Commands,
    mut race: ResMut<RaceSettings>,
    mut world_res: ResMut<VoxelWorldResource>,
    query: Query<(Entity, &Racer)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if query.is_empty() { return; }
    let all_dead = query.iter().all(|(_, r)| r.is_dead);

    if all_dead || race.should_reset {
        if let Some((_, best)) = query.iter().max_by(|a, b| a.1.distance.partial_cmp(&b.1.distance).unwrap()) {
            let weights = best.engine.instinct_layer.weights.clone();
            race.best_weights = Some(weights.clone());
            if let Ok(mut file) = File::create("CHAMPION.dna") {
                let _ = file.write_all(&weights.iter().map(|&w| w as u8).collect::<Vec<u8>>());
            }
        }

        // Cleanup
        for e in &race.world_entities { commands.entity(*e).despawn(); }
        for (e, _) in query.iter() { commands.entity(e).despawn(); }

        // RESET ESTÁVEL
        race.generation += 1;
        race.should_reset = false;
        race.selected_racer = None;

        let world = VoxelWorld::new(race.seed);
        let (r_mesh, w_mesh) = world.generate_mesh();
        let road_e = commands.spawn((Mesh3d(meshes.add(r_mesh)), MeshMaterial3d(materials.add(StandardMaterial { perceptual_roughness: 0.9, ..default() })))).id();
        let wall_e = commands.spawn((Mesh3d(meshes.add(w_mesh)), MeshMaterial3d(materials.add(StandardMaterial { base_color: Color::srgb(0.8, 0.1, 0.1), emissive: LinearRgba::RED, ..default() })))).id();
        
        race.world_entities = vec![road_e, wall_e];
        spawn_swarm(&mut commands, &mut meshes, &mut materials, &race, &world);
        world_res.0 = world;
    }
}
