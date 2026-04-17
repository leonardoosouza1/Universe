use bevy::prelude::*;
use crate::simulation::Racer;
use crate::world::VoxelWorld;
use crate::config::UniverseConfig;
use rand::prelude::*;
use std::io::Write;

use super::dna::{EliteDna, load_champion_dna};
use super::logs::open_report_log;
use super::settings::{RaceSettings, DevSettings};

pub fn spawn_swarm(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    race: &mut RaceSettings,
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

    // Mutation rate com decay exponencial por geração: convergência gradual.
    // Floor em 0.005 mantém exploração residual mesmo após 100+ gerações.
    let effective_mutation_rate = (dev.mutation_rate * 0.97_f32.powi(race.generation as i32 - 1))
        .max(0.005) as f64;

    // Carrega CHAMPION.dna (v3 com magic "IARA" ou legacy v1/v2).
    if current_best_w.is_none() {
        if let Some(dna) = load_champion_dna("CHAMPION.dna") {
            current_best_w = Some(dna.fast_weights);
            current_best_s = dna.fast_scale;
            if let Some(sw) = dna.slow_weights {
                current_best_slow_w = Some(sw);
                current_best_slow_s = dna.slow_scale;
            }
            // Se o DNA foi treinado em outra seed/largura, loga aviso e descarta champion_distance
            // (senão o carro novo nunca consegue coroar num track diferente).
            if dna.version >= 3 {
                let same_track = dna.seed as u64 == race.seed
                    && (dna.track_width - config.current_track().track_width).abs() < 0.1;
                if !same_track {
                    race.champion_distance = 0.0;
                    if let Some(mut file) = open_report_log() {
                        let _ = writeln!(file, "⚠️ CHAMPION.dna veio de outra pista (seed {} width {:.1}) — threshold resetado", dna.seed, dna.track_width);
                    }
                }
            }
        }
    }

    // Helper: tournament k=3 seleciona 1 pai entre k competidores aleatórios do pool.
    let tournament = |pool: &[EliteDna], rng: &mut rand::rngs::ThreadRng| -> usize {
        let mut best_idx = rng.random_range(0..pool.len());
        for _ in 1..3 {
            let cand = rng.random_range(0..pool.len());
            if pool[cand].fitness > pool[best_idx].fitness { best_idx = cand; }
        }
        best_idx
    };

    // Helper: uniform crossover (50/50 por gene) entre dois pais.
    let crossover = |a: &[i8], b: &[i8], rng: &mut rand::rngs::ThreadRng| -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(&wa, &wb)| {
            if rng.random_bool(0.5) { wa as f32 } else { wb as f32 }
        }).collect()
    };

    let pool_size = race.elite_pool.len();

    for i in 0..config.swarm_count {
        let mut engine = crate::engine::SovereignEngine::new();
        if pool_size >= 2 {
            // --- REPRODUÇÃO SEXUAL: Tournament + Crossover + Mutação ---
            // Primeiros N slots (até 5): clones perfeitos do top-N (elitismo múltiplo).
            // Restantes: torneio k=3 para escolher 2 pais, crossover uniforme, mutação.
            let elite_clones = pool_size.min(5);

            if i < elite_clones {
                let elite = &race.elite_pool[i];
                let exact: Vec<f32> = elite.fast_weights.iter().map(|&w| w as f32).collect();
                engine.instinct_layer.quantize(&exact);
                engine.instinct_layer.set_scale(elite.fast_scale);
                let exact_slow: Vec<f32> = elite.slow_weights.iter().map(|&w| w as f32).collect();
                engine.slow_brain.layer.quantize(&exact_slow);
                engine.slow_brain.layer.set_scale(elite.slow_scale);
            } else {
                let p1 = tournament(&race.elite_pool, &mut rng);
                let mut p2 = tournament(&race.elite_pool, &mut rng);
                if p2 == p1 && pool_size > 1 { p2 = (p1 + 1) % pool_size; }
                let parent_a = &race.elite_pool[p1];
                let parent_b = &race.elite_pool[p2];

                let crossed_fast = crossover(&parent_a.fast_weights, &parent_b.fast_weights, &mut rng);
                let mutated_fast: Vec<f32> = crossed_fast.into_iter().map(|w| {
                    if rng.random_bool(effective_mutation_rate) { rng.random_range(-1.0..1.0) } else { w }
                }).collect();
                engine.instinct_layer.quantize(&mutated_fast);
                engine.instinct_layer.set_scale((parent_a.fast_scale + parent_b.fast_scale) * 0.5);

                let crossed_slow = crossover(&parent_a.slow_weights, &parent_b.slow_weights, &mut rng);
                let mutated_slow: Vec<f32> = crossed_slow.into_iter().map(|w| {
                    if rng.random_bool(effective_mutation_rate) { rng.random_range(-1.0..1.0) } else { w }
                }).collect();
                engine.slow_brain.layer.quantize(&mutated_slow);
                engine.slow_brain.layer.set_scale((parent_a.slow_scale + parent_b.slow_scale) * 0.5);
            }
        } else if let Some(ref best) = current_best_w {
            // --- FALLBACK: DNA de CHAMPION.dna sem pool (primeira geração após boot) ---
            if i == 0 {
                let exact: Vec<f32> = best.iter().map(|&w| w as f32).collect();
                engine.instinct_layer.quantize(&exact);
                engine.instinct_layer.set_scale(current_best_s);
                if let Some(ref best_slow) = current_best_slow_w {
                    let exact_slow: Vec<f32> = best_slow.iter().map(|&w| w as f32).collect();
                    engine.slow_brain.layer.quantize(&exact_slow);
                    engine.slow_brain.layer.set_scale(current_best_slow_s);
                }
            } else {
                let mutated: Vec<f32> = best.iter().map(|&w| {
                    if rng.random_bool(effective_mutation_rate) { rng.random_range(-1.0..1.0) } else { w as f32 }
                }).collect();
                engine.instinct_layer.quantize(&mutated);
                engine.instinct_layer.set_scale(current_best_s);
                if let Some(ref best_slow) = current_best_slow_w {
                    let mutated_slow: Vec<f32> = best_slow.iter().map(|&w| {
                        if rng.random_bool(effective_mutation_rate) { rng.random_range(-1.0..1.0) } else { w as f32 }
                    }).collect();
                    engine.slow_brain.layer.quantize(&mutated_slow);
                    engine.slow_brain.layer.set_scale(current_best_slow_s);
                }
            }
        } else {
            // --- PRIMING GENÉTICO (A CONSCIÊNCIA INICIAL) ---
            // Mar de ZEROS com axiomas ILUMINADOS.
            // Antes: ruído uniforme [-1,1] + axiomas — quantize ternária produzia ±1 aleatório
            // em 99% dos pesos, diluindo os axiomas. Agora axiomas dominam quantize isolados.
            let mut weights = vec![0.0f32; 128 * 64];

            // --- AXIOMATA DE SOBREVIVÊNCIA (magnitudes grandes p/ sobreviver quantize ternária) ---

            // A1: Wall à esquerda (indices 0..5 horizon, 32..37 ground) → vira à DIREITA (positivo)
            for i in 0..6 {
                weights[1 * 128 + i] = 15.0;         // Grito de sobrevivência: foge da esquerda
                weights[1 * 128 + (32 + i)] = 15.0;
            }

            // A2: Wall à direita (indices 26..31 horizon, 58..63 ground) → vira à ESQUERDA (negativo)
            for i in 26..32 {
                weights[1 * 128 + i] = -15.0;        // Grito de sobrevivência: foge da direita
                weights[1 * 128 + (32 + i)] = -15.0;
            }

            // A3: Alinhamento com fluxo da pista (canal 64) → acelera
            weights[0 * 128 + 64] = 5.0;

            // A4: Se a pista está livre no centro (valores baixos), acelera! Se ver parede no centro, corta o motor.
            // Pan-Vision Central (10..21 e 42..53)
            for i in 10..22 {
                weights[0 * 128 + i] = -8.0;         // Corta aceleração (Throttle = 0) ao ver parede
                weights[0 * 128 + (32 + i)] = -8.0;
            }

            // A5: HP baixo ou DOR (canais 67 e 73) -> Reduz throttle severamente
            weights[0 * 128 + 73] = -5.0;

            // A6: Comp-stress alto (canal 125) → throttle (ataca quem está na frente)
            weights[0 * 128 + 125] = 3.0;

            // A7: REFLEXO DE FRENAGEM PÂNICO (Pan-Vision Central)
            // Mapeia 12 pixels centrais (Horizonte + Chão) para o canal de FREIO (2)
            for i in 12..20 {
                weights[2 * 128 + i] = 20.0;        // Freia com força total se parede na frente
                weights[2 * 128 + (32 + i)] = 20.0;
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
                joy_integral: 0.0, joy_streak: 0.0, wall_hits: 0, wall_cooldown: 0.0, last_waypoint: 0, track_progress: 0.0, grace_ticks: 0,
                reverse_timer: 0.0,
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
