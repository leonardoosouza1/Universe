use bevy::prelude::*;
use crate::simulation::Racer;
use crate::world::{VoxelWorld, VoxelWorldResource};
use crate::environment::EnvironmentalState;
use crate::config::UniverseConfig;
use rand::prelude::*;
use std::fs::File;
use std::io::{Read, Write};

#[derive(Resource)]
pub struct RaceSettings {
    pub generation: usize,
    pub seed: u64,
    pub best_weights: Option<Vec<i8>>,
    pub best_scale: f32,
    pub best_slow_weights: Option<Vec<i8>>,
    pub best_slow_scale: f32,
    pub selected_racer: Option<Entity>,
    pub world_entities: Vec<Entity>,
    pub should_reset: bool,
    pub hard_reset: bool,
    pub show_confirm_modal: bool,
    pub camera_distance: f32,
    pub is_free_cam: bool,
    pub death_markers: Vec<Vec3>,
    pub global_best_distance: f32,
    pub global_best_position: Option<Vec3>,
    pub champion_distance: f32,  // Distância do DNA salvo em CHAMPION.dna (só muda quando DNA é coroado)
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
            mutation_rate: 0.15,
            pain_sensitivity: 1.0,
            wave_amplitude: 70.0,
            flow_influence: 0.5,
        }
    }
}

pub struct EvolutionPlugin;

impl Plugin for EvolutionPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, evolution_pipeline.run_if(in_state(crate::AppState::Simulation)));
    }
}

pub fn spawn_swarm(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    race: &RaceSettings,
    world: &VoxelWorld,
    dev: &DevSettings,
    config: &UniverseConfig,
) {
    let mut rng = rand::rng();
    let wing = meshes.add(Cuboid::new(0.6, 0.8, 2.0)); 
    let cockpit = meshes.add(Sphere::new(0.4));
    let names = ["Senna", "Hamilton", "Prost", "Schumi", "Alonso", "Vettel", "Max", "Lauda", "Leclerc", "Norris"];
    let car_models = ["Porsche 911 GT3", "Ferrari 296 GT3", "McLaren 720S GT3", "Aston Martin Vantage", "Audi R8 LMS"];

    let start_flow = world.get_flow_at(world.start_pos.x, world.start_pos.z);
    let start_angle = start_flow.y.atan2(start_flow.x);

    let mut current_best_w = race.best_weights.clone();
    let mut current_best_s = race.best_scale;
    let mut current_best_slow_w = race.best_slow_weights.clone();
    let mut current_best_slow_s = race.best_slow_scale;

    // DNA format v2: [fast_scale:4][fast_w:8192][slow_scale:4][slow_w:256] = 8456 bytes
    // DNA format v1 (legacy): [fast_scale:4][fast_w:8192] = 8196 bytes — slow fica neutro
    if current_best_w.is_none() {
        if let Ok(mut file) = File::open("CHAMPION.dna") {
            let mut buffer = Vec::new();
            if file.read_to_end(&mut buffer).is_ok() && buffer.len() >= 4 {
                let scale_bytes: [u8; 4] = buffer[0..4].try_into().unwrap_or([0,0,128,63]);
                current_best_s = f32::from_le_bytes(scale_bytes);
                let fast_end = 4 + 128 * 64;
                if buffer.len() >= fast_end {
                    current_best_w = Some(buffer[4..fast_end].iter().map(|&b| b as i8).collect());
                    // Slow brain: se DNA v2, carrega; senão, fica None (= priming neutro)
                    if buffer.len() >= fast_end + 4 + 32 * 8 {
                        let ss_bytes: [u8; 4] = buffer[fast_end..fast_end + 4].try_into().unwrap_or([0,0,128,63]);
                        current_best_slow_s = f32::from_le_bytes(ss_bytes);
                        current_best_slow_w = Some(buffer[fast_end + 4..].iter().map(|&b| b as i8).collect());
                    }
                }
            }
        }
    }

    for i in 0..config.swarm_count {
        let mut engine = crate::engine::SovereignEngine::new();
        if let Some(ref best) = current_best_w {
            // ELITISMO: Piloto #0 é um CLONE PERFEITO do campeão (0% mutação)
            // Os restantes são filhos mutantes. Isso garante que o melhor DNA nunca se perde.
            let is_elite_clone = i == 0;
            
            if is_elite_clone {
                // Clone puro: copia os pesos exatamente como são
                let exact: Vec<f32> = best.iter().map(|&w| w as f32).collect();
                engine.instinct_layer.quantize(&exact);
                engine.instinct_layer.set_scale(current_best_s);
                
                if let Some(ref best_slow) = current_best_slow_w {
                    let exact_slow: Vec<f32> = best_slow.iter().map(|&w| w as f32).collect();
                    engine.slow_brain.layer.quantize(&exact_slow);
                    engine.slow_brain.layer.set_scale(current_best_slow_s);
                }
            } else {
                // Filho mutante: herda DNA com mutações aleatórias
                let mutated: Vec<f32> = best.iter().map(|&w| {
                    if rng.random_bool(dev.mutation_rate as f64) { rng.random_range(-1.0..1.0) } else { w as f32 }
                }).collect();
                engine.instinct_layer.quantize(&mutated);
                engine.instinct_layer.set_scale(current_best_s);

                if let Some(ref best_slow) = current_best_slow_w {
                    let mutated_slow: Vec<f32> = best_slow.iter().map(|&w| {
                        if rng.random_bool(dev.mutation_rate as f64) { rng.random_range(-1.0..1.0) } else { w as f32 }
                    }).collect();
                    engine.slow_brain.layer.quantize(&mutated_slow);
                    engine.slow_brain.layer.set_scale(current_best_slow_s);
                }
            }
        } else {
            // --- PRIMING GENÉTICO (A CONSCIÊNCIA INICIAL) ---
            let mut weights = vec![0.0f32; 128 * 64];
            for w in weights.iter_mut() { *w = rng.random_range(-1.0..1.0); }
            
            // --- AXIOMATA DE SOBREVIVÊNCIA (magnitudes grandes p/ sobreviver quantize ternária) ---

            // A1: Wall à esquerda (indices 0..5 horizon, 32..37 ground) → vira à DIREITA
            // Pixel alto = wall perto. Peso positivo → steer positivo. (Confirmar sinal em sim.)
            for i in 0..6 {
                weights[1 * 128 + i] = 5.0;         // horizon left-edge
                weights[1 * 128 + (32 + i)] = 5.0;  // ground left-edge
            }

            // A2: Wall à direita (indices 26..31 horizon, 58..63 ground) → vira à ESQUERDA
            for i in 26..32 {
                weights[1 * 128 + i] = -5.0;
                weights[1 * 128 + (32 + i)] = -5.0;
            }

            // A3: Alinhamento com fluxo da pista (canal 64) → acelera
            weights[0 * 128 + 64] = 5.0;

            // A4: Se alinhado E pista livre (centro retina 15 e 47, pixel BAIXO = estrada livre),
            // acelera mais. Peso NEGATIVO porque pixel alto = parede.
            weights[0 * 128 + 15] = -5.0; // Aumentado de -3.0
            weights[0 * 128 + 47] = -5.0;

            // A5: HP baixo ou DOR (canais 67 e 73) -> Reduz throttle
            weights[0 * 128 + 73] = -2.0; 

            // A6: Comp-stress alto (canal 125) → throttle (ataca quem está na frente)
            weights[0 * 128 + 125] = 3.0;

            // A7: REFLEXO DE FRENAGEM DE EMERGÊNCIA (Pan-Vision Central)
            // Mapeia 12 pixels centrais (Horizonte + Chão) para o canal de FREIO (2)
            for i in 13..19 {
                weights[2 * 128 + i] = 10.0;        // Ground center
                weights[2 * 128 + (32 + i)] = 10.0; // Horizon center
            }

            engine.instinct_layer.quantize(&weights);
        }

        let color = if i == 0 && current_best_w.is_some() { Color::srgb(0.0, 1.0, 0.8) } else { Color::srgb(rng.random::<f32>(), rng.random::<f32>(), rng.random::<f32>()) };
        let racer_mat = materials.add(StandardMaterial { base_color: color, emissive: LinearRgba::from(color) * 4.0, perceptual_roughness: 0.1, metallic: 0.9, ..default() });

        let row = i / 2;
        let col = i % 2;
        let spawn_pos = world.start_pos 
            - Vec3::new(start_angle.cos(), 0.0, start_angle.sin()) * (row as f32 * 8.0)
            + Vec3::new(-start_angle.sin(), 0.0, start_angle.cos()) * ((col as f32 - 0.5) * 6.0);

        // As meshes GT3 originais apontam nativamente para +X (Right).
        // Se a pista flui em +X (start_angle = 0), a rotação é 0.
        // from_rotation_y(-start_angle) alinha o nariz (+X) perfeitamente com o vetor de fluxo.
        let spawn_rot = Quat::from_rotation_y(-start_angle);
        let (racer_entity, car_model) = match i % 5 {
            0 => (crate::prefabs::spawn_porsche_911_gt3(commands, Transform::from_translation(spawn_pos).with_rotation(spawn_rot), &mut *meshes, racer_mat.clone()), car_models[0]),
            1 => (crate::prefabs::spawn_ferrari_296_gt3(commands, Transform::from_translation(spawn_pos).with_rotation(spawn_rot), &mut *meshes, racer_mat.clone()), car_models[1]),
            2 => (crate::prefabs::spawn_mclaren_720s_gt3(commands, Transform::from_translation(spawn_pos).with_rotation(spawn_rot), &mut *meshes, racer_mat.clone()), car_models[2]),
            3 => (crate::prefabs::spawn_aston_vantage_gt3(commands, Transform::from_translation(spawn_pos).with_rotation(spawn_rot), &mut *meshes, racer_mat.clone()), car_models[3]),
            _ => (crate::prefabs::spawn_audi_r8_lms_gt3(commands, Transform::from_translation(spawn_pos).with_rotation(spawn_rot), &mut *meshes, racer_mat.clone()), car_models[4]),
        };

        commands.entity(racer_entity).insert(
            Racer { 
                name: if i == 0 && current_best_w.is_some() { "👑 CHAMPION".to_string() } else { format!("{}-{}", names[i % names.len()], rng.random_range(10..99)) },
                car_model: car_model.to_string(),
                engine, is_dead: false, distance: 0.0, phase: 0.0, amplitude: 0.0,
                pain_memories: Vec::new(), stagnation: 0.0, prev_pos: spawn_pos, fuel: 1.0, hp: 100.0, total_laps: 0,
                memory: [0.0; 5], visual_cortex: [0; 64],
                tire_temp: 0.0, engine_power: 1.0, grass_time: 0.0, wall_time: 0.0, aquaplaning_timer: 0.0,
                joy_integral: 0.0, joy_streak: 0.0, wall_hits: 0, wall_cooldown: 0.0, last_waypoint: 0,
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

fn calc_fitness(r: &Racer) -> f32 {
    // ===================================================
    // FITNESS = DISTÂNCIA NA PISTA É REI
    // ===================================================
    // Quem andou mais longe na pista vence. Ponto final.
    // Joy serve apenas como desempate entre carros na mesma distância.
    
    let lap_bonus = r.total_laps as f32 * 50000.0;
    
    // Distância é o fator dominante: cada metro vale muito
    // Exponencial para premiar quem vai LONGE (curvas valem mais que retas)
    let dist_score = if r.distance > 0.0 { 
        r.distance * 100.0 + r.distance.powf(1.3) // Linear forte + bônus exponencial
    } else { 
        r.distance * 500.0 // Penalidade brutal por andar pra trás
    };
    
    // Joy é apenas desempate (~1% do peso da distância)
    let joy = r.joy_integral * 5.0;
    
    // Penalidade por morte: proporcional a quão pouco andou (morrer cedo = pior)
    let death_penalty = if r.is_dead { 
        (100.0 - r.distance.max(0.0)).max(0.0) * 10.0
    } else { 
        0.0 
    };
    
    // PENALIDADE POR WALL-RIDING: Cada segundo encostado na parede DESTRÓI fitness
    // Um carro que andou 50m mas raspou 2s na parede: 50*100 - 2*500 = 4000
    // Um carro que andou 40m sem tocar: 40*100 = 4000 → Empatam!
    let wall_penalty = r.wall_time * 500.0;
    
    lap_bonus + dist_score + joy - death_penalty - wall_penalty
}

fn evolution_pipeline(
    mut commands: Commands,
    mut race: ResMut<RaceSettings>,
    dev: Res<DevSettings>,
    mut world_res: ResMut<VoxelWorldResource>,
    mut env: ResMut<EnvironmentalState>,
    query: Query<(Entity, &Racer, &crate::physics::EngineComponent, &crate::physics::PhysicsBody)>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    config: Res<UniverseConfig>,
    time: Res<Time>,
) {
    // Tick do timer ambiental (Sincronizado com os mesmos clamps do simulador físico)
    let dt = time.delta_secs().min(0.02);
    env.tick(dt);
    if query.is_empty() { return; }
    let winner = query.iter().find(|(_, r, _, _)| r.total_laps >= 5);
    let all_dead = query.iter().all(|(_, r, _, _)| r.is_dead);

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
                let score_a = calc_fitness(a.1);
                let score_b = calc_fitness(b.1);
                score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
            });
            for (i, (_, r, engine_comp, phys_body)) in sorted.iter().enumerate() {
                let status = if r.is_dead { "M" } else { "V" };
                let speed_kmh = r.amplitude * 3.6;
                let gear_label = match engine_comp.current_gear {
                    0 => "R".to_string(),
                    1 => "N".to_string(),
                    g => (g - 1).to_string(),
                };
                let yaw_rate = phys_body.angular_velocity.y;
                let rl_omega = phys_body.wheels[2].angular_velocity;
                let rr_omega = phys_body.wheels[3].angular_velocity;
                let rl_slip = phys_body.wheels[2].slip_ratio;
                let rr_slip = phys_body.wheels[3].slip_ratio;
                let _ = writeln!(file, "{:02}. [{}] **{}** | D:{:.1}m | P:{:.0}% | V:{:.0}km/h | G:{} | RPM:{:.0} | YAW:{:.2}r/s | ω_RL:{:.1} ω_RR:{:.1} | SR_RL:{:.2} SR_RR:{:.2} | Thr:{:.0}% Brk:{:.0}% Trq:{:.0}Nm", 
                    i+1, status, r.name, r.distance, r.engine_power * 100.0, 
                    speed_kmh, gear_label, engine_comp.current_rpm, yaw_rate,
                    rl_omega, rr_omega, rl_slip, rr_slip,
                    engine_comp.throttle * 100.0, engine_comp.brake * 100.0, engine_comp.current_drive_torque);
            }
        }

        // SELEÇÃO POR DISTÂNCIA PURA: quem andou mais longe na pista vence
        let best_racer = if let Some((_, w, _, _)) = winner { 
            Some(w) 
        } else { 
            query.iter().max_by(|a, b| {
                a.1.distance.partial_cmp(&b.1.distance).unwrap_or(std::cmp::Ordering::Equal)
            }).map(|(_, r, _, _)| r)
        };
        if let Some(best) = best_racer {
            let this_gen_best_dist = best.distance;
            
            // ELITISMO REAL: Só coroa se SUPEROU a distância do campeão anterior
            // champion_distance é separado de global_best_distance (que é visual)
            let should_crown = this_gen_best_dist > race.champion_distance
                || race.best_weights.is_none()
                || winner.is_some();

            if should_crown {
                race.champion_distance = this_gen_best_dist; // Atualiza o limiar do campeão
                let weights = best.engine.instinct_layer.weights.clone();
                let scale = best.engine.instinct_layer.weight_scale;
                let slow_weights = best.engine.slow_brain.layer.weights.clone();
                let slow_scale = best.engine.slow_brain.layer.weight_scale;
                race.best_weights = Some(weights.clone());
                race.best_scale = scale;
                race.best_slow_weights = Some(slow_weights.clone());
                race.best_slow_scale = slow_scale;
                if let Ok(mut file) = File::create("CHAMPION.dna") {
                    let _ = file.write_all(&scale.to_le_bytes());
                    let _ = file.write_all(&weights.iter().map(|&w| w as u8).collect::<Vec<u8>>());
                    let _ = file.write_all(&slow_scale.to_le_bytes());
                    let _ = file.write_all(&slow_weights.iter().map(|&w| w as u8).collect::<Vec<u8>>());
                }
                if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("race_reports.md") {
                    use std::io::Write;
                    let _ = writeln!(file, "👑 **NOVO CAMPEÃO COROADO!** Distância: {:.1}m (Gen {}) — Superou {:.1}m anterior", this_gen_best_dist, race.generation, race.champion_distance);
                }
            } else {
                if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("race_reports.md") {
                    use std::io::Write;
                    let _ = writeln!(file, "📉 Geração fraca ({:.1}m vs campeão {:.1}m) — DNA PRESERVADO", this_gen_best_dist, race.champion_distance);
                }
            }
        }
        
        // Limpar o peso do MVP legado (HARD RESET DE PESOS CORROMPIDOS)
        if race.hard_reset {
            race.best_weights = None;
            race.best_slow_weights = None;
            race.death_markers.clear();
            let _ = std::fs::remove_file("CHAMPION.dna");
        }
        if race.should_reset {
            race.death_markers.clear();
        }
        
        for e in &race.world_entities { if let Ok(mut e_cmd) = commands.get_entity(*e) { e_cmd.despawn(); } }
        for (e, _, _, _) in query.iter() { commands.entity(e).despawn(); }
        race.generation += 1; race.should_reset = false; race.hard_reset = false; race.selected_racer = None;
        let world = VoxelWorld::new(race.seed, config.current_track().track_width);
        let (r_mesh, g_mesh, w_mesh) = world.generate_mesh();
        let road_e = commands.spawn((Mesh3d(meshes.add(r_mesh)), MeshMaterial3d(materials.add(StandardMaterial { base_color: Color::srgb(0.15, 0.15, 0.18), perceptual_roughness: 0.85, ..default() })))).id();
        let grass_e = commands.spawn((Mesh3d(meshes.add(g_mesh)), MeshMaterial3d(materials.add(StandardMaterial { base_color: Color::srgb(0.1, 0.3, 0.1), perceptual_roughness: 0.9, ..default() })))).id();
        let wall_e = commands.spawn((Mesh3d(meshes.add(w_mesh)), MeshMaterial3d(materials.add(StandardMaterial { base_color: Color::srgb(0.7, 0.7, 0.7), metallic: 1.0, perceptual_roughness: 0.1, ..default() })))).id();
        race.world_entities = vec![road_e, grass_e, wall_e];
        // Randomiza condições ambientais (mild para as primeiras 5 gerações = bootstrap)
        env.randomize(race.generation <= 5);
        spawn_swarm(&mut commands, &mut meshes, &mut materials, &race, &world, &dev, &config);
        world_res.0 = world;
    }
}
