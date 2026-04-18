use bevy::prelude::*;
use crate::simulation::Racer;
use crate::world::VoxelWorld;
use crate::config::UniverseConfig;
use rand::prelude::*;
use std::io::Write;

use super::dna::{EliteDna, load_champion_dna};
use super::logs::open_report_log;
use super::settings::{RaceSettings, DevSettings};

/// Constrói o vetor de pesos do instinct_layer pré-semeado com os axiomas A1–A15.
/// Usado tanto na geração 0 (sem pool/DNA) quanto periodicamente em gens seguintes
/// para reinjetar instintos perdidos por deriva genética.
fn build_priming_weights() -> Vec<f32> {
    let mut weights = vec![0.0f32; 128 * 64];

    // IMPORTANTE: a FÓVEA (índices 0..31) agora é SACÁDICA — pixel_i não mapeia
    // pra um ângulo fixo do carro (gira com gaze_yaw). Priming direcional que assume
    // "pixel 0 = esquerda" causaria o carro a ser ATRAÍDO pra parede quando a cabeça
    // vira. SOLUÇÃO: priming direcional só na PERIFÉRICA (32..63, sempre forward-aligned).
    //
    // A1: Wall à esquerda (periférica 32..37) → vira à DIREITA (positivo)
    for i in 0..6 {
        weights[1 * 128 + (32 + i)] = 15.0;
    }
    // A2: Wall à direita (periférica 58..63) → vira à ESQUERDA (negativo)
    for i in 26..32 {
        weights[1 * 128 + (32 + i)] = -15.0;
    }
    // A3: Alinhamento com fluxo (canal 64) → acelera
    weights[0 * 128 + 64] = 5.0;
    // A4: Parede no centro (periférica 42..54) → corta throttle forte
    for i in 10..22 {
        weights[0 * 128 + (32 + i)] = -10.0;
    }
    // A4d: Centro EXATO da periférica (46..50) → BRAKE forte, sem steer.
    for i in 14..18 {
        weights[2 * 128 + (32 + i)] = 14.0;
    }
    // A5: Dor de grama (73) → reduz throttle
    weights[0 * 128 + 73] = -5.0;
    // A6: Comp-stress (125) → acelera
    weights[0 * 128 + 125] = 3.0;
    // A7: Freio pânico estreito (periférica 46..50)
    for i in 14..18 {
        weights[2 * 128 + (32 + i)] = 8.0;
    }
    // A8: Deriva lateral (66) → contra-steer
    weights[1 * 128 + 66] = -5.0;
    // A9: Curvatura antecipada (78/79/80)
    weights[1 * 128 + 78] = -6.0;
    weights[1 * 128 + 79] = -10.0;
    weights[1 * 128 + 80] = -4.0;
    // A10: Lane offset (81) → centraliza
    weights[1 * 128 + 81] = 4.0;
    // A11: Urgência de colisão (82) → freia + corta motor
    weights[2 * 128 + 82] = 10.0;
    weights[0 * 128 + 82] = -6.0;
    // A12: Fluxo óptico (83/84) → steer anti-aproximação
    weights[1 * 128 + 83] = 5.0;
    weights[1 * 128 + 84] = -5.0;
    // A13: Memória de parede (85/86)
    weights[1 * 128 + 85] = 8.0;
    weights[1 * 128 + 86] = -8.0;
    // A14: Slip angle (87) → contra-volante
    weights[1 * 128 + 87] = 4.0;
    // A15: Proximidade sustentada (88) → corta throttle
    weights[0 * 128 + 88] = -7.0;

    // A16 (NOVO): DARK-CENTER TRACKER (canais 90-93) → SIGA O ESCURO.
    // Resposta à crítica do usuário: "o carro não tenta centralizar o preto".
    // Canal = centro de massa angular dos pixels de pista, normalizado [-1..+1].
    // Positivo ⇒ pista à direita ⇒ steer RIGHT. Negativo ⇒ pista à esquerda ⇒ LEFT.
    // Duplicado em 4 canais para que, após quantize ternária (todos ±1), este axioma
    // some 4× o sinal de um axioma normal — vira o maior vetor de steering na rede.
    for i in 90..94 {
        weights[1 * 128 + i] = 15.0;
    }

    // A17 (NOVO): WHISKERS DIAGONAIS (canais 94-97) — reflexo de curva fechada.
    //   94 = forward-left 30°  | 95 = forward-right 30°
    //   96 = forward-left 60°  | 97 = forward-right 60°
    //  Valor 127 = parede encostando naquele ângulo.
    //  - Parede diagonal à ESQUERDA (94, 96) → steer DIREITA (positivo) + BRAKE.
    //  - Parede diagonal à DIREITA  (95, 97) → steer ESQUERDA (negativo) + BRAKE.
    //  Magnitudes altas (12-18) pra sobreviver quantize e ganhar peso contra A16 quando
    //  a retina vê "aberto na diagonal" mas a parede está fechando.
    weights[1 * 128 + 94] = 18.0;   // FL 30° → steer RIGHT forte
    weights[1 * 128 + 96] = 14.0;   // FL 60° → steer RIGHT moderado
    weights[1 * 128 + 95] = -18.0;  // FR 30° → steer LEFT forte
    weights[1 * 128 + 97] = -14.0;  // FR 60° → steer LEFT moderado
    // BRAKE em qualquer whisker próximo (todos os 4 → saída 2 = brake)
    weights[2 * 128 + 94] = 10.0;
    weights[2 * 128 + 95] = 10.0;
    weights[2 * 128 + 96] = 8.0;
    weights[2 * 128 + 97] = 8.0;
    // Corta throttle também (saída 0 = accel)
    weights[0 * 128 + 94] = -8.0;
    weights[0 * 128 + 95] = -8.0;

    // A18 (NOVO): GAZE TRACKER (canais 98-101) — fóvea sacádica aponta pro caminho aberto.
    // Valor = ângulo da fóvea normalizado [-127..+127] mapeando [-60°..+60°].
    // Positivo ⇒ pista aberta à direita ⇒ steer DIREITA. Negativo ⇒ LEFT.
    // Com MEMÓRIA: se visão fica toda branca (parede), gaze mantém última direção
    // conhecida — carro continua virando pra onde viu pista pela última vez.
    // Duplicado 4× (igual A16) pra dominar após quantize ternária: na gen 1 este vira
    // o maior vetor de steering, rede aleatória não consegue anular.
    for i in 98..102 {
        weights[1 * 128 + i] = 14.0;
    }

    weights
}

/// Spawna carros no índice `[idx_start..idx_end)`. Permite spawn em blocos entre frames
/// (evita freeze de spawn de 2000 entidades em 1 frame).
pub fn spawn_swarm(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    race: &mut RaceSettings,
    world: &VoxelWorld,
    dev: &DevSettings,
    config: &UniverseConfig,
    idx_start: usize,
    idx_end: usize,
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

    // Helper: BURST MUTATION por canal de entrada.
    // Layout do fast layer: weights[out * 128 + in], 64 outputs × 128 inputs.
    // Burst = escolhe um canal de entrada c e substitui TODAS as 64 conexões
    // (para cada output) por valores aleatórios em [-1, 1]. Quebra neutralidade do BitNet:
    // mutação pontual em layer ternário tem alta probabilidade de virar 0 após quantize —
    // burst força variação coerente de um sensor inteiro.
    let burst_mutate_input_channel = |weights: &mut Vec<f32>, rng: &mut rand::rngs::ThreadRng| {
        let n_outputs = 64usize;
        let n_inputs = 128usize;
        let c = rng.random_range(0..n_inputs);
        for out in 0..n_outputs {
            let idx = out * n_inputs + c;
            if idx < weights.len() {
                weights[idx] = rng.random_range(-1.0..1.0);
            }
        }
    };

    let pool_size = race.elite_pool.len();

    for i in idx_start..idx_end.min(config.swarm_count) {
        let mut engine = crate::engine::SovereignEngine::new();

        // --- HALL OF FAME (CHAMPION FALLBACK) ---
        // Slot 0: SEMPRE carrega o CHAMPION DNA se existe. Imune à deriva geracional
        // (até mesmo se o elite_pool inteiro for ruim, o HoF preserva o melhor de todos os tempos).
        let slot_is_champion = i == 0 && current_best_w.is_some();

        if slot_is_champion {
            if let Some(ref champion) = current_best_w {
                let exact: Vec<f32> = champion.iter().map(|&w| w as f32).collect();
                engine.instinct_layer.quantize(&exact);
                engine.instinct_layer.set_scale(current_best_s);
                if let Some(ref champion_slow) = current_best_slow_w {
                    let exact_slow: Vec<f32> = champion_slow.iter().map(|&w| w as f32).collect();
                    engine.slow_brain.layer.quantize(&exact_slow);
                    engine.slow_brain.layer.set_scale(current_best_slow_s);
                }
            }
        } else if pool_size >= 2 {
            // --- REPRODUÇÃO DIVERSIFICADA NO GRID (30 slots de largada) ---
            // Gargalo anterior: só 5 grid spawns × 2 elite clones = 2 DNAs testados
            // da largada por gen. DNA de 935m ficava na população distribuída sem
            // nunca ser validado em "start from zero". Agora espalhamos o pool
            // inteiro pelos 30 slots grid:
            //   Slot 0:      CHAMPION (já tratado acima)
            //   Slot 1-10:   top-10 elite clones (cada elite testado no grid)
            //   Slot 11-20:  PRIMING REFRESH (instintos frescos na largada)
            //   Slot 21-29:  tournament crossover (filhos diversos no grid)
            //   Slot 30+:    tournament crossover (distributed, cobertura de pista)
            let elite_clones = pool_size.min(10);
            let refresh_slots = 10;
            let elite_range_end = 1 + elite_clones; // [1, 1+elite_clones)

            if i >= 1 && i < elite_range_end {
                let elite = &race.elite_pool[i - 1];
                let exact: Vec<f32> = elite.fast_weights.iter().map(|&w| w as f32).collect();
                engine.instinct_layer.quantize(&exact);
                engine.instinct_layer.set_scale(elite.fast_scale);
                let exact_slow: Vec<f32> = elite.slow_weights.iter().map(|&w| w as f32).collect();
                engine.slow_brain.layer.quantize(&exact_slow);
                engine.slow_brain.layer.set_scale(elite.slow_scale);
            } else if i < elite_range_end + refresh_slots {
                // PRIMING REFRESH: axiomas frescos + ruído leve para diversidade.
                // Preserva instintos mas ainda permite exploração local.
                let priming = build_priming_weights();
                let jittered: Vec<f32> = priming.into_iter().map(|w| {
                    if rng.random_bool(0.005) { rng.random_range(-1.0..1.0) } else { w }
                }).collect();
                engine.instinct_layer.quantize(&jittered);
                // slow_brain fica zerado (estado neutro, evolução descobre uso)
            } else {
                let p1 = tournament(&race.elite_pool, &mut rng);
                let mut p2 = tournament(&race.elite_pool, &mut rng);
                if p2 == p1 && pool_size > 1 { p2 = (p1 + 1) % pool_size; }
                let parent_a = &race.elite_pool[p1];
                let parent_b = &race.elite_pool[p2];

                let crossed_fast = crossover(&parent_a.fast_weights, &parent_b.fast_weights, &mut rng);
                let mut mutated_fast: Vec<f32> = crossed_fast.into_iter().map(|w| {
                    if rng.random_bool(effective_mutation_rate) { rng.random_range(-1.0..1.0) } else { w }
                }).collect();
                // 20% dos slots crossover sofrem burst por canal de entrada (além da mutação pontual).
                if rng.random_bool(0.20) {
                    burst_mutate_input_channel(&mut mutated_fast, &mut rng);
                }
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
            // --- PRIMING GENÉTICO (GERAÇÃO 0, SEM POOL/DNA) ---
            // Axiomas A1–A15 definidos em build_priming_weights(). Magnitudes grandes
            // pra sobreviver quantize ternária (bits ±1).
            let weights = build_priming_weights();
            engine.instinct_layer.quantize(&weights);
        }

        let color = if i == 0 && current_best_w.is_some() { Color::srgb(0.0, 1.0, 0.8) } else { Color::srgb(rng.random::<f32>(), rng.random::<f32>(), rng.random::<f32>()) };
        let racer_mat = materials.add(StandardMaterial { base_color: color, emissive: LinearRgba::from(color) * 4.0, perceptual_roughness: 0.1, metallic: 0.9, ..default() });

        // --- SPAWN DISTRIBUÍDO AO LONGO DA PISTA (Andrychowicz et al, 2017) ---
        // Carros 0..29 (champion + 10 elites + 10 priming + 9 crossover) começam na
        // largada — 30 tentativas independentes do DNA bom, o suficiente pra que pelo
        // menos um sobreviva à largada e prove skill transferível.
        // Carros 30+ spawnam em pontos espalhados pela pista (coverage).
        // Fitness usa NET progress (track_progress - spawn_progress), então carros
        // que nasceram na curva e sobreviveram 30m PROVARAM habilidade de curva.
        // Isso quebra o platô deceptivo de "só reta dá fitness alto".
        const GRID_SPAWNS: usize = 30;
        let (spawn_pos, spawn_rot, spawn_progress) = if i < GRID_SPAWNS {
            // Grid tradicional na largada (grade 2 colunas × 15 fileiras).
            // Mantém a largura original (~6m) pra caber na pista; ocupa mais profundidade
            // (15 × 6m = 90m de grid atrás da largada — cabe numa reta de Interlagos).
            let row = i / 2;
            let col = i % 2;
            let p = world.start_pos
                - Vec3::new(start_angle.cos(), 0.0, start_angle.sin()) * (row as f32 * 6.0)
                + Vec3::new(-start_angle.sin(), 0.0, start_angle.cos()) * ((col as f32 - 0.5) * 6.0);
            (p, Quat::from_rotation_y(-start_angle), 0.0)
        } else {
            // Distribui ao longo dos waypoints. Fração = 25%, 50%, 75%, 87%... em faixa circular.
            let distributed_idx = (i - GRID_SPAWNS) as f32;
            let total_distributed = (config.swarm_count - GRID_SPAWNS).max(1) as f32;
            let fraction = 0.15 + (distributed_idx / total_distributed) * 0.7; // [0.15..0.85]
            let wp_count = world.waypoints_world.len().max(2);
            let target_wp = ((fraction * wp_count as f32) as usize).min(wp_count - 2);
            let wp = world.waypoints_world[target_wp];
            let wp_next = world.waypoints_world[target_wp + 1];
            let dir = (wp_next - wp).normalize_or_zero();
            let p = Vec3::new(wp.x, world.start_pos.y, wp.y);
            let angle = dir.y.atan2(dir.x);
            let progress = world.get_track_progress(p.x, p.z);
            (p, Quat::from_rotation_y(-angle), progress)
        };
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
                joy_integral: 0.0, joy_streak: 0.0, wall_hits: 0, wall_cooldown: 0.0, last_waypoint: 0, track_progress: spawn_progress, grace_ticks: 0,
                reverse_timer: 0.0,
                progress_checkpoint: spawn_progress,
                progress_stall: 0.0,
                wall_ema_l: 15.0,
                wall_ema_r: 15.0,
                proximity_sustain: 0.0,
                post_reverse_lockout: 0.0,
                spawn_progress,
                path: vec![spawn_pos],
                path_timer: 0.0,
                brain_trace: Vec::with_capacity(15),
                trace_timer: 0.0,
                gaze_yaw: 0.0,
                max_progress: spawn_progress,
                slow_time: 0.0,
                race_time: 0.0,
                prev_lap_prog: spawn_progress.rem_euclid(world.lap_distance.max(1.0)),
                post_lap_grace: 0,
                post_lap_checkpoint: 0.0,
                death_reason: None,
                death_frame: 0,
                death_pos: Vec3::ZERO,
                peak_speed: 0.0,
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
