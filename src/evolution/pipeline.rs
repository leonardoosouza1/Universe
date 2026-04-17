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

    // NET progress: o quanto o carro andou DESDE seu spawn. Fundamental com distributed
    // spawn — senão quem nasce no waypoint 80% e não anda nada tem fitness alto "de graça".
    let prog = (r.track_progress - r.spawn_progress).max(0.0);
    let lap_bonus = r.total_laps as f32 * 50000.0;
    let joy = r.joy_integral * 1.0;  // Desempate fino
    // Wall penalty 50 → 200: raspar parede agora custa MUITO mais progresso equivalente.
    // 1s de wall_time = 200pts = 2m de track_progress deletados.
    let wall_penalty = r.wall_time * 200.0;
    // Wall hits: cada contato durante grace custa 500pts adicionais (fora do grace morre direto).
    let hit_penalty = r.wall_hits as f32 * 500.0;
    let death_penalty = if r.is_dead { (100.0 - prog).max(0.0) * 10.0 } else { 0.0 };

    lap_bonus + prog * 100.0 + joy - wall_penalty - hit_penalty - death_penalty
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
    // Geração encerra por QUALQUER dos:
    //  - winner (5 voltas)
    //  - all_dead (não adianta esperar ninguém vivo)
    //  - timeout configurável (carros vivos tiveram tempo de acumular laps)
    //  - reset manual
    let all_dead = query.iter().all(|(_, r, _, _)| r.is_dead);
    let timeout_reached = env.generation_time >= config.generation_timeout_s;

    if winner.is_some() || all_dead || timeout_reached || race.should_reset {
        // Rank global por fitness — reutilizado pelo pool elite + relatório.
        let mut sorted: Vec<_> = query.iter().collect();
        sorted.sort_by(|a, b| {
            let score_a = calc_fitness(a.1);
            let score_b = calc_fitness(b.1);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // --- POOL ELITE (top-10) para tournament na próxima geração ---
        // Expandido de 5 → 10: dobra diversidade genética. Gargalo anterior era
        // que o DNA bom (carros distribuídos a 935m) só estava em 5 slots e
        // raramente migrava pro grid (5 grid spawns × 2 elite clones = 2 testes/gen).
        race.elite_pool.clear();
        for (_, r, _, _) in sorted.iter().take(10) {
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
            let reason = if winner.is_some() { "Vencedor Encontrado" }
                else if all_dead { "Extinção (Colisão)" }
                else if timeout_reached { "Timeout Atingido" }
                else { "Reset Manual" };
            let _ = writeln!(file, "\n## GERAÇÃO {} | Motivo do Fim: {} | Duração: {:.1}s", race.generation, reason, env.generation_time);
            let _ = writeln!(file, "**Ambiente:** Asfalto {:.0}°C | Chuva: {:.0}% | Vento: ({:.1}, {:.1}) m/s | Hora: {:.1}h | μ_base: 1.3 × grip_env: {:.2}",
                env.asphalt_temp, env.weather_wetness * 100.0, env.wind.x, env.wind.z, env.time_of_day, env.combined_grip_factor());

            // --- REPORT OTIMIZADO PARA 500+ CARROS ---
            // Antes: 500 linhas por gen × ~200 gens = 100k linhas em race_reports.md (~2MB/ciclo).
            // Agora: TOP 15 (os que importam) + BOTTOM 3 (pra ver como estão morrendo os piores)
            // + SUMMARY AGREGADO (medianas, taxa de morte, NET p50/p90).
            // Reduz arquivo em ~20× mantendo sinal de debug útil.
            let total = sorted.len();
            let top_n = 15.min(total);
            let bot_n = if total > top_n + 3 { 3 } else { 0 };
            let dump_row = |file: &mut std::fs::File, rank: usize, r: &Racer, engine_comp: &crate::physics::EngineComponent, phys_body: &crate::physics::PhysicsBody| {
                let status = if r.is_dead { "M" } else { "V" };
                let speed_kmh = r.amplitude * 3.6;
                let gear_label = match engine_comp.current_gear {
                    0 => "R".to_string(),
                    1 => "N".to_string(),
                    g => (g - 1).to_string(),
                };
                let yaw_rate = phys_body.angular_velocity.y;
                let rl_slip = phys_body.wheels[2].slip_ratio;
                let rr_slip = phys_body.wheels[3].slip_ratio;
                let net = (r.track_progress - r.spawn_progress).max(0.0);
                let grid_tag = if r.spawn_progress < 1.0 { "G" } else { "D" };
                let _ = writeln!(file, "{:03}. [{}{}] **{}** | NET:{:.1}m | V:{:.0}km/h | G:{} | RPM:{:.0} | YAW:{:.2}r/s | SR:{:.2}/{:.2} | Thr:{:.0}% Brk:{:.0}%",
                    rank, grid_tag, status, r.name, net, speed_kmh, gear_label, engine_comp.current_rpm, yaw_rate,
                    rl_slip, rr_slip,
                    engine_comp.throttle * 100.0, engine_comp.brake * 100.0);
            };

            let _ = writeln!(file, "\n### 🏁 TOP {}", top_n);
            for (i, (_, r, engine_comp, phys_body)) in sorted.iter().take(top_n).enumerate() {
                dump_row(&mut file, i + 1, r, engine_comp, phys_body);
            }
            if bot_n > 0 {
                let _ = writeln!(file, "\n### 🪦 BOTTOM {}", bot_n);
                for (idx, (_, r, engine_comp, phys_body)) in sorted.iter().rev().take(bot_n).enumerate() {
                    dump_row(&mut file, total - idx, r, engine_comp, phys_body);
                }
            }

            // Summary agregado: distribuição de NET/fitness/survival.
            let nets: Vec<f32> = sorted.iter().map(|(_, r, _, _)| (r.track_progress - r.spawn_progress).max(0.0)).collect();
            let deaths = sorted.iter().filter(|(_, r, _, _)| r.is_dead).count();
            let fits: Vec<f32> = sorted.iter().map(|(_, r, _, _)| calc_fitness(r)).collect();
            let median = |v: &[f32]| -> f32 { if v.is_empty() { 0.0 } else { v[v.len()/2] } };
            let p90 = |v: &[f32]| -> f32 { if v.is_empty() { 0.0 } else { v[(v.len()*1/10).min(v.len()-1)] } }; // v já ordenado desc
            let mean = |v: &[f32]| -> f32 { if v.is_empty() { 0.0 } else { v.iter().sum::<f32>() / v.len() as f32 } };
            let grid_count = sorted.iter().filter(|(_, r, _, _)| r.spawn_progress < 1.0).count();
            let _ = writeln!(file, "\n### 📊 SUMMARY | N={} | Deaths: {}/{} ({:.0}%) | Grid: {} | Distrib: {}",
                total, deaths, total, (deaths as f32 / total as f32) * 100.0, grid_count, total - grid_count);
            let _ = writeln!(file, "NET — max: {:.1}m | p90: {:.1}m | median: {:.1}m | mean: {:.1}m",
                nets.first().copied().unwrap_or(0.0), p90(&nets), median(&nets), mean(&nets));
            let _ = writeln!(file, "Fitness — max: {:.0} | p90: {:.0} | median: {:.0} | mean: {:.0}",
                fits.first().copied().unwrap_or(0.0), p90(&fits), median(&fits), mean(&fits));

            // --- DIAGNÓSTICO CEREBRAL DO TOP-3 (NET) ---
            // Mostra o histórico dos 3 melhores. Se o volante não responde quando
            // a parede chega, vai aparecer aqui: wall_prox=120 mas raw_steer≈0.
            let _ = writeln!(file, "\n### 🧠 BRAIN TRACE — Top 3 (NET)");
            for (i, (_, r, _, _)) in sorted.iter().take(3).enumerate() {
                let net = (r.track_progress - r.spawn_progress).max(0.0);
                let _ = writeln!(file, "\n#### {}. {} — NET {:.1}m | {} snapshots", i+1, r.name, net, r.brain_trace.len());
                let _ = writeln!(file, "| t(s) | V(km/h) | WL | WR | TTC | Curv25 | Retina_C | raw_thr | raw_steer | post_damp | phys_steer | steer_lim | risk_damp | TOP5 canais |");
                let _ = writeln!(file, "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
                for s in r.brain_trace.iter() {
                    let top5s = s.top_channels.iter()
                        .filter(|(_,v)| *v != 0)
                        .map(|(i,v)| format!("c{}={}", i, v))
                        .collect::<Vec<_>>().join(" ");
                    let _ = writeln!(file, "| {:.1} | {:.0} | {} | {} | {} | {} | {} | {:+.2} | {:+.2} | {:+.2} | {:+.2} | {:.2} | {:.2} | {} |",
                        s.t, s.speed_kmh, s.wall_prox_l, s.wall_prox_r, s.ttc_inv,
                        s.curv_25, s.retina_center_max,
                        s.raw_throttle, s.raw_steer, s.post_damp_steer,
                        s.physical_steer, s.steer_limit, s.risk_damp, top5s);
                }
            }
            let _ = writeln!(file, "");
        }

        // --- MELHOR DA GERAÇÃO (qualquer spawn) — só pra visualização ---
        // Separado do best_racer (que filtra grid-spawn pra coroação).
        // Aqui queremos o traçado de quem mais andou, pra mostrar evolução no mapa.
        if let Some((_, gen_best, _, _)) = query.iter().max_by(|a, b| {
            let na = (a.1.track_progress - a.1.spawn_progress).max(0.0);
            let nb = (b.1.track_progress - b.1.spawn_progress).max(0.0);
            na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            race.last_gen_path = gen_best.path.clone();
            race.last_gen_net = (gen_best.track_progress - gen_best.spawn_progress).max(0.0);
            race.last_gen_was_grid = gen_best.spawn_progress < 1.0;
        }

        // SELEÇÃO POR NET PROGRESS, RESTRITA A GRID-SPAWNS.
        // Motivo: carros com spawn distribuído viram "especialistas" de um trecho fácil
        // e falham quando o DNA é recarregado em slot 0 (spawn na largada). O campeão
        // DEVE ser um carro que provou saber sair da largada. Distributed cars ainda
        // contribuem genes via elite_pool/tournament — só não podem ser CHAMPION.
        let net_prog = |r: &Racer| (r.track_progress - r.spawn_progress).max(0.0);
        let is_grid_spawn = |r: &Racer| r.spawn_progress < 1.0;
        let best_racer = if let Some((_, w, _, _)) = winner {
            Some(w)
        } else {
            query.iter()
                .filter(|(_, r, _, _)| is_grid_spawn(r))
                .max_by(|a, b| {
                    net_prog(a.1).partial_cmp(&net_prog(b.1)).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(_, r, _, _)| r)
        };

        // --- DECAY DO THRESHOLD (anti-deadlock) ---
        // Se 20 gens consecutivas não coroam, o campeão salvo provavelmente não é
        // replicável (foi lucky-spec de um trecho). Reduz o threshold 5%/gen após 20
        // gens sem coroação, permitindo que um "campeão pior mas CONSISTENTE" assuma.
        if race.gens_since_crown >= 20 {
            race.champion_distance *= 0.95;
        }

        if let Some(best) = best_racer {
            let this_gen_best_dist = net_prog(best);

            // ELITISMO REAL: Só coroa se SUPEROU o progresso REAL na pista do campeão anterior
            // champion_distance é separado de global_best_distance (que é visual)
            let should_crown = this_gen_best_dist > race.champion_distance
                || race.best_weights.is_none()
                || winner.is_some();

            if should_crown {
                race.champion_distance = this_gen_best_dist; // Atualiza o limiar do campeão
                race.gens_since_crown = 0; // Resetar contador de deadlock
                // Guarda o traçado do campeão para visualizar na próxima geração.
                race.champion_path = best.path.clone();
                race.champion_net_dist = this_gen_best_dist;
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
                race.gens_since_crown += 1;
                if let Some(mut file) = open_report_log() {
                    use std::io::Write;
                    let decay_note = if race.gens_since_crown >= 20 { " [THRESHOLD EM DECAY -5%/gen]" } else { "" };
                    let _ = writeln!(file, "📉 Geração fraca ({:.1}m vs campeão {:.1}m) — DNA PRESERVADO (gens sem coroar: {}){}", this_gen_best_dist, race.champion_distance, race.gens_since_crown, decay_note);
                }
            }
        } else {
            // Nenhum grid-spawn teve progresso válido — continua contando deadlock
            race.gens_since_crown += 1;
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
        // Pista começa um pouco mais larga (base + 4m) e estreita rápido com voltas do campeão.
        // Cada volta completada reduz o bônus em 2m — piso (base) atingido em 2 voltas reais.
        let base_width = config.current_track().track_width;
        let champion_laps = (race.champion_distance / world_res.0.lap_distance.max(1.0)).max(0.0);
        let width_bonus = (4.0 - champion_laps * 2.0).max(0.0);
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
