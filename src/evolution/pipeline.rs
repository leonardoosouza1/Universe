use bevy::prelude::*;
use crate::simulation::Racer;
use crate::world::{VoxelWorld, VoxelWorldResource};
use crate::environment::EnvironmentalState;
use crate::config::UniverseConfig;

use super::dna::{EliteDna, save_champion_dna};
use super::logs::open_report_log;
use super::settings::{RaceSettings, DevSettings};
use super::spawn::spawn_swarm;

pub(super) fn calc_fitness(r: &Racer) -> f32 {
    // ===================================================
    // FITNESS SIMPLIFICADO — sinal limpo para seleção.
    // ===================================================
    // Estrutura:
    //   base    = progresso NA PISTA (domínio 100× maior que desempates)
    //   lap     = bônus por voltas completas (50k cada)
    //   joy     = desempate fino entre carros equidistantes (~1%)
    //   wall    = regularização suave (não fitness-killer)
    //   death   = linear em "quão pouco andou antes de morrer"
    //
    // Removido: exponencial `prog^1.3` (criava discontinuidade artificial),
    //           peso de wall 500→50 (wall-riding agora custa, mas não apaga 40m de progresso).

    let prog = r.track_progress.max(0.0);
    let lap_bonus = r.total_laps as f32 * 50000.0;
    let joy = r.joy_integral * 1.0;  // Desempate fino
    let wall_penalty = r.wall_time * 50.0;
    let death_penalty = if r.is_dead { (100.0 - prog).max(0.0) * 10.0 } else { 0.0 };

    lap_bonus + prog * 100.0 + joy - wall_penalty - death_penalty
}

pub(super) fn evolution_pipeline(
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
        // Rank global por fitness — reutilizado pelo pool elite + relatório.
        let mut sorted: Vec<_> = query.iter().collect();
        sorted.sort_by(|a, b| {
            let score_a = calc_fitness(a.1);
            let score_b = calc_fitness(b.1);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // --- POOL ELITE (top-5) para tournament na próxima geração ---
        race.elite_pool.clear();
        for (_, r, _, _) in sorted.iter().take(5) {
            race.elite_pool.push(EliteDna {
                fast_weights: r.engine.instinct_layer.weights.clone(),
                fast_scale: r.engine.instinct_layer.weight_scale,
                slow_weights: r.engine.slow_brain.layer.weights.clone(),
                slow_scale: r.engine.slow_brain.layer.weight_scale,
                fitness: calc_fitness(r),
            });
        }

        // --- RELATÓRIO DO GÊNESIS (TELEMETRIA COMPLETA) ---
        if let Some(mut file) = open_report_log() {
            use std::io::Write;
            let reason = if winner.is_some() { "Vencedor Encontrado" } else if all_dead { "Extinção (Colisão)" } else { "Tempo Esgotado" };
            let _ = writeln!(file, "\n## GERAÇÃO {} | Motivo do Fim: {} | Duração: {:.1}s", race.generation, reason, env.generation_time);
            let _ = writeln!(file, "**Ambiente:** Asfalto {:.0}°C | Chuva: {:.0}% | Vento: ({:.1}, {:.1}) m/s | Hora: {:.1}h | μ_base: 1.3 × grip_env: {:.2}",
                env.asphalt_temp, env.weather_wetness * 100.0, env.wind.x, env.wind.z, env.time_of_day, env.combined_grip_factor());

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
                let _ = writeln!(file, "{:02}. [{}] **{}** | TP:{:.1}m (D:{:.1}m) | P:{:.0}% | V:{:.0}km/h | G:{} | RPM:{:.0} | YAW:{:.2}r/s | ω_RL:{:.1} ω_RR:{:.1} | SR_RL:{:.2} SR_RR:{:.2} | Thr:{:.0}% Brk:{:.0}% Trq:{:.0}Nm",
                    i+1, status, r.name, r.track_progress, r.distance, r.engine_power * 100.0,
                    speed_kmh, gear_label, engine_comp.current_rpm, yaw_rate,
                    rl_omega, rr_omega, rl_slip, rr_slip,
                    engine_comp.throttle * 100.0, engine_comp.brake * 100.0, engine_comp.current_drive_torque);
            }
        }

        // SELEÇÃO POR PROGRESSO REAL NA PISTA: quem avançou mais na polilinha vence
        let best_racer = if let Some((_, w, _, _)) = winner {
            Some(w)
        } else {
            query.iter().max_by(|a, b| {
                a.1.track_progress.partial_cmp(&b.1.track_progress).unwrap_or(std::cmp::Ordering::Equal)
            }).map(|(_, r, _, _)| r)
        };
        if let Some(best) = best_racer {
            let this_gen_best_dist = best.track_progress;

            // ELITISMO REAL: Só coroa se SUPEROU o progresso REAL na pista do campeão anterior
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
                let _ = save_champion_dna(
                    "CHAMPION.dna",
                    &weights, scale,
                    &slow_weights, slow_scale,
                    race.generation as u32,
                    race.seed as u32,
                    config.current_track().track_width,
                    this_gen_best_dist,
                );
                if let Some(mut file) = open_report_log() {
                    use std::io::Write;
                    let _ = writeln!(file, "👑 **NOVO CAMPEÃO COROADO!** Distância: {:.1}m (Gen {}) — Superou {:.1}m anterior", this_gen_best_dist, race.generation, race.champion_distance);
                }
            } else {
                if let Some(mut file) = open_report_log() {
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

        // --- CURRICULUM ADAPTATIVO DE LARGURA ---
        // Pista começa LARGA (base + 8m) e estreita conforme o campeão completa voltas reais.
        // champion_laps ≈ champion_distance / lap_distance (usa último world conhecido como ref).
        // Cada volta completada reduz o bônus em 2m até chegar na base configurada.
        let base_width = config.current_track().track_width;
        let champion_laps = (race.champion_distance / world_res.0.lap_distance.max(1.0)).max(0.0);
        let width_bonus = (8.0 - champion_laps * 2.0).max(0.0);
        let effective_width = base_width + width_bonus;
        let world = VoxelWorld::new(race.seed, effective_width);
        let (r_mesh, g_mesh, w_mesh) = world.generate_mesh();
        let road_e = commands.spawn((Mesh3d(meshes.add(r_mesh)), MeshMaterial3d(materials.add(StandardMaterial { base_color: Color::srgb(0.15, 0.15, 0.18), perceptual_roughness: 0.85, ..default() })))).id();
        let grass_e = commands.spawn((Mesh3d(meshes.add(g_mesh)), MeshMaterial3d(materials.add(StandardMaterial { base_color: Color::srgb(0.1, 0.3, 0.1), perceptual_roughness: 0.9, ..default() })))).id();
        let wall_e = commands.spawn((Mesh3d(meshes.add(w_mesh)), MeshMaterial3d(materials.add(StandardMaterial { base_color: Color::srgb(0.7, 0.7, 0.7), metallic: 1.0, perceptual_roughness: 0.1, ..default() })))).id();
        race.world_entities = vec![road_e, grass_e, wall_e];
        // Curriculum invertido: condições MILD até o campeão completar ao menos meia volta.
        // Evita pressão ambiental antes da política básica existir.
        let mild_mode = race.champion_distance < world_res.0.lap_distance * 0.5;
        env.randomize(mild_mode);
        spawn_swarm(&mut commands, &mut meshes, &mut materials, &mut race, &world, &dev, &config);
        world_res.0 = world;
    }
}
