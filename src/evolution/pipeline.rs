use bevy::prelude::*;
use crate::simulation::Racer;
use crate::world::{VoxelWorld, VoxelWorldResource};
use crate::environment::EnvironmentalState;
use crate::config::UniverseConfig;

use super::dna::{EliteDna, save_champion_dna};
use super::logs::open_report_log;
use super::settings::{RaceSettings, DevSettings};
use super::spawn::spawn_swarm;

/// Decide nível ambiental da próxima geração.
/// Muta `gens_until_validation` e `is_validation_gen` de forma pura/testável.
/// Retorna (level, is_validation_gen_flag).
pub fn decide_next_env_level(
    gens_until_validation: &mut u32,
    champion_distance: f32,
    lap_distance: f32,
    current_level: u8,
) -> (u8, bool) {
    if *gens_until_validation <= 1 {
        *gens_until_validation = 3;
        (0, true)
    } else {
        *gens_until_validation -= 1;
        let target = EnvironmentalState::level_for_champion(champion_distance, lap_distance);
        // RAMP GRADUAL: só sobe 1 nível por gen. Salto 1→3 matava toda população
        // (champion especializado em level 1 não sobrevivia asphalt frio + chuva).
        let capped = target.min(current_level.saturating_add(1));
        (capped, false)
    }
}

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

    // Proximity bonus: incentiva FECHAR a volta. Quadrático em NET até saturar em 500m.
    // Carro com 480m (quase 1 volta) ganha 9200 pts extras = gap claro vs 400m (6400) ou
    // 300m (3600). Satura em 10000 pra não competir com lap_bonus 50k.
    // Depois de uma volta completa, lap_bonus domina e proximity fica estável.
    let prox_frac = (prog / 500.0).clamp(0.0, 1.0);
    let proximity_bonus = prox_frac * prox_frac * 10000.0;
    // Wall penalty 50 → 200: raspar parede agora custa MUITO mais progresso equivalente.
    // 1s de wall_time = 200pts = 2m de track_progress deletados.
    let wall_penalty = r.wall_time * 200.0;
    // Wall hits: cada contato durante grace custa 500pts adicionais (fora do grace morre direto).
    let hit_penalty = r.wall_hits as f32 * 500.0;
    // Death penalty REWORKED: antes `(100-prog)*10` capava em 1000pts e zerava após 100m
    // — dying ficava de graça depois. Agora: 2000 flat + 5% do prog (perdeu potencial).
    // 800m prog + death = 2000 + 400 = 2400pts. 800m prog survived = 0 penalty + SURVIVAL_BONUS.
    let death_penalty = if r.is_dead { 2000.0 + prog * 0.05 } else { 0.0 };
    // Survival bonus: sobreviver até final da geração tem valor flat. Quebra incentivo de
    // "acelera geral + morre" quando progresso é equivalente a quem vai devagar e chega.
    let survival_bonus = if r.is_dead { 0.0 } else { 3000.0 };

    lap_bonus + prog * 100.0 + proximity_bonus + joy + survival_bonus - wall_penalty - hit_penalty - death_penalty
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
    // Tick do timer ambiental: usa virtual delta cheio (SEM cap). Motor+física agora rodam
    // em FixedUpdate (substep N× por frame em time_scale alto), env aqui acompanha o tempo
    // virtual total — generation_timeout em virtual-seconds escala certinho.
    let dt = time.delta_secs();
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

        // --- POOL ELITE PERSISTENTE (Hall of Fame) ---
        // ANTES: clear + top-10 da gen atual. Gen fraca contaminava pool → cascata
        // de colapso (620m gen 6 → pool lixo → gen 7 fraca também).
        // AGORA: merge — só entra se bater pior slot do pool. Gens fracas não apagam
        // DNA bom de gens anteriores. Champion + top elites ficam guardados.
        for (_, r, _, _) in sorted.iter().take(10) {
            let new_fit = calc_fitness(r);
            let candidate = EliteDna {
                fast_weights: r.engine.instinct_layer.weights.clone(),
                fast_scale: r.engine.instinct_layer.weight_scale,
                slow_weights: r.engine.slow_brain.layer.weights.clone(),
                slow_scale: r.engine.slow_brain.layer.weight_scale,
                fitness: new_fit,
            };
            if race.elite_pool.len() < 10 {
                race.elite_pool.push(candidate);
            } else {
                // Acha pior do pool. Se novo é melhor, substitui.
                let (worst_idx, worst_fit) = race.elite_pool.iter().enumerate()
                    .min_by(|a, b| a.1.fitness.partial_cmp(&b.1.fitness).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, e)| (i, e.fitness))
                    .unwrap_or((0, f32::NEG_INFINITY));
                if new_fit > worst_fit {
                    race.elite_pool[worst_idx] = candidate;
                }
            }
        }
        // Re-sort pool desc por fitness (slot 0 = melhor de todos os tempos).
        race.elite_pool.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));

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

            // Lap completion distribution — quantos completaram 1, 2, 3+ voltas
            let laps_0 = sorted.iter().filter(|(_, r, _, _)| r.total_laps == 0).count();
            let laps_1 = sorted.iter().filter(|(_, r, _, _)| r.total_laps == 1).count();
            let laps_2 = sorted.iter().filter(|(_, r, _, _)| r.total_laps == 2).count();
            let laps_3p = sorted.iter().filter(|(_, r, _, _)| r.total_laps >= 3).count();
            let _ = writeln!(file, "Laps — 0v: {} | 1v: {} | 2v: {} | 3+v: {}", laps_0, laps_1, laps_2, laps_3p);

            // Death reason breakdown — categórico via DeathReason enum.
            let mut d_hp = 0; let mut d_pin = 0; let mut d_bat = 0;
            let mut d_stag = 0; let mut d_sbld = 0; let mut d_fuel = 0; let mut d_unk = 0;
            for (_, r, _, _) in sorted.iter() {
                if !r.is_dead { continue; }
                match r.death_reason {
                    Some(crate::simulation::racer::DeathReason::HpZero) => d_hp += 1,
                    Some(crate::simulation::racer::DeathReason::Pinado) => d_pin += 1,
                    Some(crate::simulation::racer::DeathReason::Batida) => d_bat += 1,
                    Some(crate::simulation::racer::DeathReason::Stagnation) => d_stag += 1,
                    Some(crate::simulation::racer::DeathReason::SpeedBleed) => d_sbld += 1,
                    Some(crate::simulation::racer::DeathReason::Fuel) => d_fuel += 1,
                    None => d_unk += 1,
                }
            }
            let _ = writeln!(file, "Death reasons — HP:{} PINADO:{} BATIDA:{} STAG:{} SBLD:{} FUEL:{} ?:{}",
                d_hp, d_pin, d_bat, d_stag, d_sbld, d_fuel, d_unk);

            // --- DEATH BY ZONE (waypoint segment) ---
            // Agrupa mortes por segmento da pista. Identifica "curva que mata".
            let world = &world_res.0;
            let n_segs = world.waypoints_world.len().saturating_sub(1).max(1);
            let mut deaths_per_seg = vec![0u32; n_segs];
            for (_, r, _, _) in sorted.iter() {
                if r.is_dead {
                    let seg = world.waypoint_zone(r.death_pos.x, r.death_pos.z).min(n_segs - 1);
                    deaths_per_seg[seg] += 1;
                }
            }
            let _ = writeln!(file, "\n### 💀 DEATHS POR SEGMENTO (waypoint i → i+1)");
            for (i, count) in deaths_per_seg.iter().enumerate() {
                let p1 = world.waypoints_world[i];
                let p2 = world.waypoints_world[i+1];
                let bar = "█".repeat((((*count as f32) / (total as f32).max(1.0)) * 30.0) as usize);
                let _ = writeln!(file, "seg[{}] ({:.0},{:.0})→({:.0},{:.0}): {:3}  {}",
                    i, p1.x, p1.y, p2.x, p2.y, count, bar);
            }

            // --- ASCII MAP com death markers + finish positions ---
            let death_points: Vec<Vec3> = sorted.iter()
                .filter(|(_, r, _, _)| r.is_dead)
                .map(|(_, r, _, _)| r.death_pos).collect();
            let alive_positions: Vec<Vec3> = sorted.iter()
                .filter(|(_, r, _, _)| !r.is_dead)
                .take(30)
                .map(|(_, _, _, body)| {
                    let _ = body;
                    Vec3::ZERO
                }).collect();
            let map = world.render_ascii_map(100, 40, &death_points, &alive_positions);
            let _ = writeln!(file, "\n### 🗺️  MAPA (100×40) | `.`=grass `=`=road `#`=wall `0-7`=waypoints `X`=morte");
            let _ = writeln!(file, "```");
            let _ = write!(file, "{}", map);
            let _ = writeln!(file, "```");

            // --- PER-RACER ENHANCED (top + qualquer com lap) ---
            let with_lap: Vec<_> = sorted.iter().filter(|(_, r, _, _)| r.total_laps > 0).collect();
            if !with_lap.is_empty() {
                let _ = writeln!(file, "\n### 🏆 RACERS COM VOLTAS COMPLETADAS (todos)");
                let _ = writeln!(file, "| Nome | Laps | NET(m) | Peak(km/h) | Walls | Death | Frame | Zone |");
                let _ = writeln!(file, "|---|---|---|---|---|---|---|---|");
                for (_, r, _, _) in with_lap.iter() {
                    let net = (r.track_progress - r.spawn_progress).max(0.0);
                    let death = r.death_reason.map(|d| d.label()).unwrap_or("VIVO");
                    let zone = if r.is_dead { world.waypoint_zone(r.death_pos.x, r.death_pos.z) as i32 } else { -1 };
                    let _ = writeln!(file, "| {} | {} | {:.0} | {:.0} | {} | {} | {} | {} |",
                        r.name, r.total_laps, net, r.peak_speed * 3.6, r.wall_hits, death, r.death_frame,
                        if zone < 0 { "-".to_string() } else { zone.to_string() });
                }
            }

            // Console summary compacto pra feedback ao vivo.
            println!("📊 GEN {} | Laps: 0v={} 1v={} 2v={} 3+v={} | NET max={:.0}m median={:.0}m | Deaths {}/{}",
                race.generation, laps_0, laps_1, laps_2, laps_3p,
                nets.first().copied().unwrap_or(0.0), median(&nets), deaths, total);

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

            // COROAÇÃO GATED POR VALIDATION GEN
            // Só coroa em gens rodadas em env fixo (level 0) — evita lucky-env bias.
            // Winner com 5 voltas sempre coroa (é prova robusta).
            // best_weights None = primeiro DNA, coroa sempre pra bootstrap.
            let is_validation = race.env_level == 0;
            // VALIDATION tolera regressão 15% se DNA for CONSISTENTE (completou ≥1 lap).
            // Motivo: champion lucky-spec em level 1 (ex: 4082m) nunca mais era batido
            // após curriculum subir pra level 2+, travando evolução. 15% de folga
            // permite DNA que fez 3500m+1lap assumir, mantendo pressão seletiva mas
            // destravando deadlock precoce.
            let val_threshold = race.champion_distance * 0.85;
            let val_ok = is_validation
                && this_gen_best_dist > val_threshold
                && best.total_laps >= 1;
            let should_crown = winner.is_some()
                || race.best_weights.is_none()
                || val_ok;

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
        // CURRICULUM LEVEL + VALIDATION GEN (ver decide_next_env_level)
        let champ_dist = race.champion_distance;
        let lap_dist = world_res.0.lap_distance;
        let current_lvl = race.env_level;
        let (next_level, is_val) = decide_next_env_level(
            &mut race.gens_until_validation,
            champ_dist,
            lap_dist,
            current_lvl,
        );
        race.is_validation_gen = is_val;
        race.env_level = next_level;
        env.randomize_level(next_level);
        // Spawn batched: agenda jobs, drain_spawn_queue processa em blocos por frame.
        race.pending_spawn = Some((0, config.swarm_count));
        world_res.0 = world;
    }
}

#[cfg(test)]
mod tests {
    use super::decide_next_env_level;

    #[test]
    fn validation_cycle_is_every_three_gens() {
        let mut counter: u32 = 3;
        let lap = 900.0;
        let (lvl, val) = decide_next_env_level(&mut counter, 0.0, lap, 1);
        assert_eq!((lvl, val), (1, false));
        assert_eq!(counter, 2);
        let (_, val) = decide_next_env_level(&mut counter, 0.0, lap, 1);
        assert!(!val);
        assert_eq!(counter, 1);
        let (lvl, val) = decide_next_env_level(&mut counter, 0.0, lap, 1);
        assert_eq!((lvl, val), (0, true));
        assert_eq!(counter, 3);
    }

    #[test]
    fn validation_level_is_always_zero() {
        let mut counter: u32 = 1;
        let (lvl, val) = decide_next_env_level(&mut counter, 2000.0, 900.0, 3);
        assert_eq!(lvl, 0);
        assert!(val);
    }

    #[test]
    fn non_validation_respects_champion_distance() {
        let mut counter: u32 = 3;
        let (lvl, _) = decide_next_env_level(&mut counter, 100.0, 900.0, 1);
        assert_eq!(lvl, 1);
        let mut counter: u32 = 3;
        let (lvl, _) = decide_next_env_level(&mut counter, 700.0, 900.0, 1);
        assert_eq!(lvl, 2);
        let mut counter: u32 = 3;
        let (lvl, _) = decide_next_env_level(&mut counter, 1500.0, 900.0, 2);
        assert_eq!(lvl, 3);
    }

    #[test]
    fn ramp_caps_at_current_plus_one() {
        // Champion salta pra level 3 mas current=1 → capado em 2
        let mut counter: u32 = 3;
        let (lvl, _) = decide_next_env_level(&mut counter, 2000.0, 900.0, 1);
        assert_eq!(lvl, 2);
        // current=2 já → level 3 liberado
        let mut counter: u32 = 3;
        let (lvl, _) = decide_next_env_level(&mut counter, 2000.0, 900.0, 2);
        assert_eq!(lvl, 3);
    }
}
