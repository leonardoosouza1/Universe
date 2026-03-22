use bevy::prelude::*;
use crate::simulation::Racer;
use crate::world::{VoxelWorld, VoxelWorldResource};
use crate::environment::EnvironmentalState;
use rand::prelude::*;
use std::fs::File;
use std::io::{Read, Write};

#[derive(Resource)]
pub struct RaceSettings {
    pub generation: usize,
    pub seed: u64,
    pub best_weights: Option<Vec<i8>>,
    pub best_scale: f32,
    pub selected_racer: Option<Entity>,
    pub world_entities: Vec<Entity>,
    pub should_reset: bool,
    pub camera_distance: f32,
    pub is_free_cam: bool,
}

#[derive(Resource)]
pub struct DevSettings {
    pub mutation_rate: f32,
    pub pain_sensitivity: f32,
    pub wave_amplitude: f32,
    pub flow_influence: f32,
}

impl Default for DevSettings {
    fn default() -> Self {
        Self {
            mutation_rate: 0.12,
            pain_sensitivity: 1.0,
            wave_amplitude: 70.0,
            flow_influence: 0.5,
        }
    }
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
    dev: &DevSettings,
) {
    let mut rng = rand::rng();
    let wing = meshes.add(Cuboid::new(0.6, 0.8, 2.0)); 
    let cockpit = meshes.add(Sphere::new(0.4));
    let names = ["Senna", "Hamilton", "Prost", "Schumi", "Alonso", "Vettel", "Max", "Lauda", "Leclerc", "Norris"];

    let start_flow = world.get_flow_at(world.start_pos.x, world.start_pos.z);
    let start_angle = start_flow.y.atan2(start_flow.x);

    let mut current_best_w = race.best_weights.clone();
    let mut current_best_s = race.best_scale;

    if current_best_w.is_none() {
        if let Ok(mut file) = File::open("CHAMPION.dna") {
            let mut buffer = Vec::new();
            if file.read_to_end(&mut buffer).is_ok() && buffer.len() >= 4 {
                let scale_bytes: [u8; 4] = buffer[0..4].try_into().unwrap_or([0,0,128,63]);
                current_best_s = f32::from_le_bytes(scale_bytes);
                current_best_w = Some(buffer[4..].iter().map(|&b| b as i8).collect());
            }
        }
    }

    for i in 0..40 {
        let mut engine = crate::engine::SovereignEngine::new();
        if let Some(ref best) = current_best_w {
            let mutated: Vec<f32> = best.iter().map(|&w| {
                if rng.random_bool(dev.mutation_rate as f64) { rng.random_range(-1.0..1.0) } else { w as f32 }
            }).collect();
            engine.instinct_layer.quantize(&mutated);
            engine.instinct_layer.set_scale(current_best_s);
        } else {
            // --- PRIMING GENÉTICO (A CONSCIÊNCIA INICIAL) ---
            let mut weights = vec![0.0f32; 128 * 64];
            for w in weights.iter_mut() { *w = rng.random_range(-0.1..0.1); }
            
            // Axioma 1: Seguir o Vão Livre (Input 42 -> Steer Output 1)
            weights[1 * 128 + 42] = 1.0; 
            // Axioma 2: Manter Alinhamento (Input 43 -> Steer Output 1)
            weights[1 * 128 + 43] = 0.8;
            // Axioma 3: Acelerar se estiver alinhado (Input 43 -> Pedal Output 0)
            weights[0 * 128 + 43] = 0.5;

            engine.instinct_layer.quantize(&weights);
        }

        let color = if i == 0 && current_best_w.is_some() { Color::srgb(0.0, 1.0, 0.8) } else { Color::srgb(rng.random::<f32>(), rng.random::<f32>(), rng.random::<f32>()) };
        let racer_mat = materials.add(StandardMaterial { base_color: color, emissive: LinearRgba::from(color) * 4.0, perceptual_roughness: 0.1, metallic: 0.9, ..default() });

        let row = i / 2;
        let col = i % 2;
        let spawn_pos = world.start_pos 
            - Vec3::new(start_angle.cos(), 0.0, start_angle.sin()) * (row as f32 * 8.0)
            + Vec3::new(-start_angle.sin(), 0.0, start_angle.cos()) * ((col as f32 - 0.5) * 6.0);

        let racer_entity = if i % 3 == 0 {
            crate::prefabs::spawn_chassis_argo(commands, Transform::from_translation(spawn_pos).with_rotation(Quat::from_rotation_y(-start_angle)), &mut *meshes, racer_mat.clone())
        } else if i % 3 == 1 {
            crate::prefabs::spawn_chassis_sandero(commands, Transform::from_translation(spawn_pos).with_rotation(Quat::from_rotation_y(-start_angle)), &mut *meshes, racer_mat.clone())
        } else {
            crate::prefabs::spawn_chassis_ka(commands, Transform::from_translation(spawn_pos).with_rotation(Quat::from_rotation_y(-start_angle)), &mut *meshes, racer_mat.clone())
        };

        commands.entity(racer_entity).insert(
            Racer { 
                name: format!("{}-{}", names[i % names.len()], rng.random_range(10..99)),
                engine, is_dead: false, distance: 0.0, phase: 0.0, amplitude: 0.0,
                pain_memories: Vec::new(), stagnation: 0.0, prev_pos: spawn_pos, fuel: 1.0, hp: 100.0, total_laps: 0,
                memory: [0.0; 3], visual_cortex: [0; 64],
                tire_temp: 0.0, engine_power: 1.0, grass_time: 0.0, aquaplaning_timer: 0.0,
            }
        ).with_children(|p| {
            p.spawn((Mesh3d(wing.clone()), MeshMaterial3d(racer_mat.clone()), Transform::from_xyz(-1.2, 0.4, 0.0)));
            p.spawn((Mesh3d(cockpit.clone()), MeshMaterial3d(materials.add(StandardMaterial { base_color: Color::WHITE, emissive: LinearRgba::WHITE * 2.0, ..default() })), Transform::from_xyz(0.5, 0.3, 0.0)));
            
            // FAROL DO CAMPO VISUAL (FOV de 135 graus, alcance 40 metros igual à Retina KAN)
            p.spawn((
                SpotLight {
                    color: Color::WHITE,
                    intensity: 150_000.0, // Luz alta para alcance prolongado (150m)
                    range: 150.0,
                    radius: 0.0,
                    shadows_enabled: false,
                    inner_angle: 0.8,
                    outer_angle: 1.178, // (135 graus) / 2 em radianos
                    ..default()
                },
                Transform::from_xyz(1.5, 0.5, 0.0) // No nariz do carro
                    .with_rotation(Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2).mul_quat(Quat::from_rotation_x(-0.05))) // Vira pra frente (-Z -> +X)
            ));
        });
    }
}

fn evolution_pipeline(
    mut commands: Commands,
    mut race: ResMut<RaceSettings>,
    dev: Res<DevSettings>,
    mut world_res: ResMut<VoxelWorldResource>,
    mut env: ResMut<EnvironmentalState>,
    query: Query<(Entity, &Racer, &crate::physics::EngineComponent)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    time: Res<Time>,
) {
    // Tick do timer ambiental
    env.tick(time.delta_secs());
    if query.is_empty() { return; }
    let winner = query.iter().find(|(_, r, _)| r.total_laps >= 5);
    let all_dead = query.iter().all(|(_, r, _)| r.is_dead);

    if winner.is_some() || all_dead || race.should_reset {
        // --- RELATÓRIO DO GÊNESIS (TELEMETRIA COMPLETA) ---
        if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("race_reports.md") {
            use std::io::Write;
            let reason = if winner.is_some() { "Vencedor Encontrado" } else if all_dead { "Extinção (Colisão)" } else { "Tempo Esgotado" };
            let _ = writeln!(file, "\n## GERAÇÃO {} | Motivo do Fim: {} | Duração: {:.1}s", race.generation, reason, env.generation_time);
            let _ = writeln!(file, "**Ambiente:** Asfalto {:.0}°C | Chuva: {:.0}% | Vento: ({:.1}, {:.1}) m/s | Hora: {:.1}h | μ_base: 1.3 × grip_env: {:.2}", 
                env.asphalt_temp, env.weather_wetness * 100.0, env.wind.x, env.wind.z, env.time_of_day, env.combined_grip_factor());
            
            let mut sorted: Vec<_> = query.iter().collect();
            sorted.sort_by(|a, b| {
                let calc_score = |r: &Racer| -> f32 {
                    if r.distance <= 0.0 { 
                        r.distance * 100.0 
                    } else {
                        // Recompensa velocidade, mas penaliza severamente ficar na grama e perder vida (HP)
                        let base = r.distance.powi(2) / r.engine.current_time.max(0.1);
                        let hp_penalty = (100.0 - r.hp) * 10.0; // Perder 100 HP = -1000 pontos
                        let death_penalty = if r.is_dead { 500.0 } else { 0.0 };
                        base - hp_penalty - death_penalty - (r.grass_time * 50.0)
                    }
                };
                let score_a = calc_score(a.1);
                let score_b = calc_score(b.1);
                score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
            });
            for (i, (_, r, engine_comp)) in sorted.iter().enumerate() {
                let status = if r.is_dead { "M" } else { "V" };
                let speed_kmh = r.amplitude * 3.6;
                let gear_label = match engine_comp.current_gear {
                    0 => "R".to_string(),
                    1 => "N".to_string(),
                    g => (g - 1).to_string(),
                };
                // Estresse é o canal 125, mas não guardamos no Racer. 
                // Por enquanto vamos focar no que temos: Pneus e Marchas.
                let _ = writeln!(file, "{:02}. [{}] **{}** | Dist: {:.1}m | EngPwr: {:.0}% | Pneu: {:.1}°C | Grama: {:.1}s | Vel: {:.0}km/h | Marcha: {} | Freio: {:.0}%", 
                    i+1, status, r.name, r.distance, r.engine_power * 100.0, 
                    r.tire_temp * 100.0, r.grass_time, speed_kmh, gear_label, r.memory[2] * 100.0);
            }
        }

        let best_racer = if let Some((_, w, _)) = winner { 
            Some(w) 
        } else { 
            query.iter().max_by(|a, b| {
                let calc_score = |r: &Racer| -> f32 {
                    if r.distance <= 0.0 { 
                        r.distance * 100.0 
                    } else {
                        let base = r.distance.powi(2) / r.engine.current_time.max(0.1);
                        let hp_penalty = (100.0 - r.hp) * 10.0;
                        let death_penalty = if r.is_dead { 500.0 } else { 0.0 };
                        base - hp_penalty - death_penalty - (r.grass_time * 50.0)
                    }
                };
                let score_a = calc_score(a.1);
                let score_b = calc_score(b.1);
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            }).map(|(_, r, _)| r) 
        };
        if let Some(best) = best_racer {
            let weights = best.engine.instinct_layer.weights.clone();
            let scale = best.engine.instinct_layer.weight_scale;
            race.best_weights = Some(weights.clone());
            race.best_scale = scale;
            if let Ok(mut file) = File::create("CHAMPION.dna") { 
                let _ = file.write_all(&scale.to_le_bytes());
                let _ = file.write_all(&weights.iter().map(|&w| w as u8).collect::<Vec<u8>>()); 
            }
        }
        for e in &race.world_entities { if let Ok(mut e_cmd) = commands.get_entity(*e) { e_cmd.despawn(); } }
        for (e, _, _) in query.iter() { commands.entity(e).despawn(); }
        race.generation += 1; race.should_reset = false; race.selected_racer = None;
        let world = VoxelWorld::new(race.seed);
        let (r_mesh, g_mesh, w_mesh) = world.generate_mesh();
        let road_e = commands.spawn((Mesh3d(meshes.add(r_mesh)), MeshMaterial3d(materials.add(StandardMaterial { base_color: Color::srgb(0.15, 0.15, 0.18), perceptual_roughness: 0.85, ..default() })))).id();
        let grass_e = commands.spawn((Mesh3d(meshes.add(g_mesh)), MeshMaterial3d(materials.add(StandardMaterial { base_color: Color::srgb(0.1, 0.3, 0.1), perceptual_roughness: 0.9, ..default() })))).id();
        let wall_e = commands.spawn((Mesh3d(meshes.add(w_mesh)), MeshMaterial3d(materials.add(StandardMaterial { base_color: Color::srgb(0.7, 0.7, 0.7), metallic: 1.0, perceptual_roughness: 0.1, ..default() })))).id();
        race.world_entities = vec![road_e, grass_e, wall_e];
        // Randomiza condições ambientais (mild para as primeiras 5 gerações = bootstrap)
        env.randomize(race.generation <= 5);
        spawn_swarm(&mut commands, &mut meshes, &mut materials, &race, &world, &dev);
        world_res.0 = world;
    }
}
