use bevy::prelude::*;
use crate::simulation::Racer;
use crate::world::{VoxelWorld, VoxelWorldResource};
use crate::environment::EnvironmentalState;
use crate::config::UniverseConfig;
use crate::SimulationEntity;
use iara_genesis::codec::semantic::BehaviorFingerprint;

use super::dna_genesis::{EliteDna, save_champion_dna_full, apply_catastrophic_scars, decay_scars};

// Fitness-sharing niching threshold: parents/elites with Hamming distance
// > NICHE_DISTANCE on their 64-bit BehaviorFingerprint are considered
// different behavioral clusters. 8/64 = 12.5% disagreement — empirically
// tight enough to split genuinely different strategies but loose enough to
// group minor mutations of the same lineage.
// v14: tentou 10, piorou (tudo 1 cluster). Revertido pra 8.
pub(super) const NICHE_DISTANCE: u32 = 8;

/// Clusters elites by Hamming distance on BehaviorFingerprint.
/// Returns (cluster_id_per_elite, cluster_size_per_elite).
/// Greedy single-pass: elite joins the first existing cluster whose
/// representative is within NICHE_DISTANCE, otherwise opens a new one.
pub(super) fn cluster_elites(elites: &[EliteDna]) -> (Vec<usize>, Vec<usize>) {
    let fps: Vec<BehaviorFingerprint> = elites.iter()
        .map(|e| BehaviorFingerprint::from_weights(&e.fast_weights))
        .collect();
    let mut reps: Vec<BehaviorFingerprint> = Vec::new();
    let mut ids: Vec<usize> = Vec::with_capacity(elites.len());
    for fp in fps.iter() {
        let mut found: Option<usize> = None;
        for (i, r) in reps.iter().enumerate() {
            if fp.distance(r) <= NICHE_DISTANCE { found = Some(i); break; }
        }
        match found {
            Some(i) => ids.push(i),
            None => { reps.push(*fp); ids.push(reps.len() - 1); }
        }
    }
    let mut sizes = vec![0usize; reps.len()];
    for &id in ids.iter() { sizes[id] += 1; }
    let per_elite: Vec<usize> = ids.iter().map(|&id| sizes[id]).collect();
    (ids, per_elite)
}
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

// Tier 1.1 — constantes de reward shaping denso.
// Calibradas pra cada sinal contribuir 1-5% da fitness de um champion típico.
const K_SMOOTH: f32 = 100.0;   // steering jitter: penalidade até -500
const K_SPEED: f32 = 0.5;      // speed_on_track: 30 m/s × 120s × 0.5 ≈ 1800 max
const K_NOVEL: f32 = 50.0;     // sqrt(voxels): sqrt(2000) × 50 ≈ 2236 max
const K_STAB: f32 = 200.0;     // stability_integral: 0..500 range

// Tier 6.9 — Phi (IIT) proxy como pressão seletiva.
// Cheap integrated-information approximation from brain_trace snapshots:
// correlation magnitude between sensory channels and motor outputs, averaged.
// Higher = retina/probe channels jointly drive actions (integrated architecture).
// Alvo: 1-3% da fitness de um champion típico (~500-1500 pts).
const K_PHI: f32 = 300.0;
const K_AESTH: f32 = 200.0;

fn aesthetic_score(telem: &crate::simulation::racer::RacerTelemetry) -> f32 {
    let steer: Vec<f32> = telem.steering_history.iter().copied().collect();
    if steer.len() < 3 { return 0.0; }
    let mut jerk_sum = 0.0f32;
    let mut jerk_n = 0usize;
    for i in 2..steer.len() {
        let d1 = steer[i] - steer[i - 1];
        let d0 = steer[i - 1] - steer[i - 2];
        jerk_sum += (d1 - d0).abs();
        jerk_n += 1;
    }
    let jerk = if jerk_n > 0 { jerk_sum / jerk_n as f32 } else { 0.0 };
    let speeds: Vec<f32> = telem.speed_history.iter()
        .zip(telem.on_road_mask.iter())
        .filter_map(|(&s, &on)| if on { Some(s) } else { None })
        .collect();
    let speed_var = if speeds.len() >= 2 {
        let mean = speeds.iter().sum::<f32>() / speeds.len() as f32;
        speeds.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / speeds.len() as f32
    } else { 0.0 };
    let line_consistency = 1.0 / (1.0 + speed_var);
    let score = (1.0 / (1.0 + jerk * 10.0)) * 0.5 + line_consistency * 0.5;
    score.clamp(0.0, 1.0)
}

/// Compute a cheap integrated-information proxy from brain_trace snapshots.
///
/// Uses sensor channels (wall_prox_l/r, ttc_inv, curv_25, retina_center_max) and
/// action channels (raw_throttle, raw_steer, raw_brake). Metric = mean |Pearson
/// correlation| across all sensor×action pairs. Returns 0.0 on NaN, zero variance,
/// or < 4 samples (cov undefined / too noisy).
///
/// Note: spec suggested `log(det(cov_joint)/prod(var_marginal))`, but with 5+3
/// heterogeneous channels and ≤15 samples, correlation-sum is numerically stable
/// and captures the same "joint coupling beyond per-channel independence" signal.
fn compute_phi_proxy(r: &Racer) -> f32 {
    let trace = &r.brain_trace;
    let n = trace.len();
    if n < 4 { return 0.0; }

    // Sensor matrix (5 channels), action matrix (3 channels).
    let mut sensors: [Vec<f32>; 5] = Default::default();
    let mut actions: [Vec<f32>; 3] = Default::default();
    for s in trace.iter() {
        sensors[0].push(s.wall_prox_l as f32);
        sensors[1].push(s.wall_prox_r as f32);
        sensors[2].push(s.ttc_inv as f32);
        sensors[3].push(s.curv_25 as f32);
        sensors[4].push(s.retina_center_max as f32);
        actions[0].push(s.raw_throttle);
        actions[1].push(s.raw_steer);
        actions[2].push(s.raw_brake);
    }

    fn mean_var(xs: &[f32]) -> (f32, f32) {
        let n = xs.len() as f32;
        let m = xs.iter().sum::<f32>() / n;
        let v = xs.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / n;
        (m, v)
    }
    fn corr(xs: &[f32], ys: &[f32]) -> f32 {
        let n = xs.len() as f32;
        let (mx, vx) = mean_var(xs);
        let (my, vy) = mean_var(ys);
        if vx <= 1e-6 || vy <= 1e-6 { return 0.0; }
        let cov: f32 = xs.iter().zip(ys.iter())
            .map(|(x, y)| (x - mx) * (y - my)).sum::<f32>() / n;
        let c = cov / (vx.sqrt() * vy.sqrt());
        if c.is_nan() || c.is_infinite() { 0.0 } else { c }
    }

    let mut sum_abs = 0.0f32;
    let mut pairs = 0u32;
    for s in sensors.iter() {
        for a in actions.iter() {
            sum_abs += corr(s, a).abs();
            pairs += 1;
        }
    }
    if pairs == 0 { return 0.0; }
    let phi = sum_abs / pairs as f32;
    if phi.is_nan() || phi.is_infinite() { 0.0 } else { phi.clamp(0.0, 1.0) }
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
    // Tier 4.3 — Nicho-splitting (simpler version, shared-pool).
    //   niche == 0 (defensive):  wall_penalty × 2.0  — pune raspar parede forte.
    //   niche == 1 (aggressive): speed_reward × 1.5  — premia velocidade on-road.
    // Pool único: ambos competem pelas mesmas vagas elite → cross-breed natural
    // via tournament/mate-preference. Lineage fica introspectável pelo campo.
    let wall_mult: f32 = if r.niche == 0 { 2.0 } else { 1.0 };
    let speed_mult: f32 = if r.niche == 1 { 1.5 } else { 1.0 };
    let wall_penalty = r.wall_time * 200.0 * wall_mult;
    // Wall hits: cada contato durante grace custa 500pts adicionais (fora do grace morre direto).
    let hit_penalty = r.wall_hits as f32 * 500.0;
    // Death penalty REWORKED: antes `(100-prog)*10` capava em 1000pts e zerava após 100m
    // — dying ficava de graça depois. Agora: 2000 flat + 5% do prog (perdeu potencial).
    // 800m prog + death = 2000 + 400 = 2400pts. 800m prog survived = 0 penalty + SURVIVAL_BONUS.
    let death_penalty = if r.is_dead { 2000.0 + prog * 0.05 } else { 0.0 };
    // Survival bonus: sobreviver até final da geração tem valor flat. Quebra incentivo de
    // "acelera geral + morre" quando progresso é equivalente a quem vai devagar e chega.
    let survival_bonus = if r.is_dead { 0.0 } else { 3000.0 };

    // --- TIER 1.1: SINAIS DENSOS DE REWARD SHAPING ---
    // Cada um premia behavior que correlaciona com progresso de longo prazo.
    // Pisos mantidos baixos pra não dominar lap_bonus (50k) em champions fluentes.
    //
    // 1) Smoothness: penaliza jitter de volante (clamp piso -500 — não é prêmio infinito ficar parado).
    let steering_var = r.telemetry.steering_variance();
    let smoothness_reward = (-steering_var * K_SMOOTH).max(-500.0);
    // 2) Speed-on-track: velocidade acumulada ON-ROAD (anti-grass, anti-wall).
    let speed_reward = r.telemetry.speed_on_track() * K_SPEED * speed_mult;
    // 3) Novelty: sqrt(voxels únicos on-road). On-road obrigatório = anti circle-exploit.
    let unique_voxels = r.telemetry.visited_voxels.len();
    let novelty_bonus = (unique_voxels as f32).sqrt() * K_NOVEL;
    // 4) Stability: integral de (1 - |yaw_rate|) on-road — premia dirigibilidade controlada.
    let stability_reward = r.telemetry.stability_integral() * K_STAB;

    // Tier 6.9 — Phi (IIT) proxy: integra sensores×ações na fitness.
    // Max teórico ~1.0 × 300 = 300pts. Típico 0.2-0.5 → 60-150pts (≈1-3%).
    let phi_reward = compute_phi_proxy(r) * K_PHI;
    // Tier 6.7 — Aesthetic judge: baixo jerk + linha consistente.
    let aesthetic_reward = aesthetic_score(&r.telemetry) * K_AESTH;

    lap_bonus
        + prog * 100.0
        + proximity_bonus
        + joy
        + survival_bonus
        + smoothness_reward
        + speed_reward
        + novelty_bonus
        + stability_reward
        + phi_reward
        + aesthetic_reward
        - wall_penalty
        - hit_penalty
        - death_penalty
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
        // --- TIER 6.6: DECAY GLOBAL DE SCARS na borda de geração ---
        // Antes de injetar novos candidatos, decrementa TTL de todo scar_mask existente.
        // Scars com TTL 0 ficam livres pra mutação novamente.
        let mut total_active_scars_before = 0usize;
        let mut total_active_scars_after = 0usize;
        for elite in race.elite_pool.iter_mut() {
            total_active_scars_before += elite.scar_mask.iter().filter(|&&t| t > 0).count();
            total_active_scars_after += decay_scars(&mut elite.scar_mask);
        }

        // Se HOUVE qualquer BATIDA neste gen, a linhagem aprende com o trauma:
        // Scars propagam pros elites top-K (proxy de "paciência evolutiva" — se
        // o top-N tem parentesco com quem bateu, marcar pesos ativos).
        // Simples: conta BATIDAs em toda a população.
        let gen_batidas = sorted.iter().filter(|(_, r, _, _)|
            matches!(r.death_reason, Some(crate::simulation::racer::DeathReason::Batida))
        ).count();

        let mut scars_created_this_gen = 0usize;
        for (rank, (_, r, _, _)) in sorted.iter().take(10).enumerate() {
            let new_fit = calc_fitness(r);
            let fast_weights: Vec<i8> = r.engine.instinct_layer.weights.clone();
            let mut scar_mask = vec![0u8; fast_weights.len()];
            // 1) O próprio top-10 morreu em BATIDA? Marca direto.
            let own_batida = matches!(
                r.death_reason,
                Some(crate::simulation::racer::DeathReason::Batida)
            );
            // 2) Houve BATIDA na população e este é top-3 (linhagem dominante)?
            //    Proxy: população afeta pool inteiro via reprodução compartilhada.
            let lineage_batida = gen_batidas > 0 && rank < 3;
            if own_batida || lineage_batida {
                let seed_bias = fast_weights.iter().map(|&w| w as i64).sum::<i64>() as u64
                    ^ (race.generation as u64).wrapping_mul(0xA5A5_A5A5)
                    ^ (rank as u64).wrapping_mul(0x1234_5678);
                let created = apply_catastrophic_scars(&mut scar_mask, &fast_weights, seed_bias);
                scars_created_this_gen += created;
            }
            let candidate = EliteDna {
                fast_weights,
                fast_scale: r.engine.instinct_layer.weight_scale,
                slow_weights: r.engine.slow_brain.layer.weights.clone(),
                slow_scale: r.engine.slow_brain.layer.weight_scale,
                fitness: new_fit,
                scar_mask,
                // Tier 4.2: captura estado hormonal médio ao final da vida.
                cortisol_mean: r.engine.metabolism.mean_cortisol(),
                dopamine_mean: r.engine.metabolism.mean_dopamine(),
                curiosity_mean: r.engine.metabolism.mean_curiosity(),
                // Tier 6.3 LITE: propaga morpho genes do racer vivo pra elite pool.
                morpho: r.morpho,
                // v11: captura knots do KAN output head pra inheritance entre gens.
                kan_knots: r.kan_out.knots,
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
        // --- FITNESS-SHARING NICHING (behavioral fingerprint) ---
        // Cluster the merged pool by 64-bit BehaviorFingerprint Hamming distance
        // (≤ NICHE_DISTANCE = same niche). Divide raw fitness by cluster size before
        // sorting for elitism: a lone explorer with fitness 10k beats a 3-clone
        // dominant strategy at 12k each (12k/3 = 4k). Prevents convergence to a
        // single genotype and preserves alternate strategies for future crossover.
        let (_pre_ids, pre_sizes) = cluster_elites(&race.elite_pool);
        let mut order: Vec<usize> = (0..race.elite_pool.len()).collect();
        order.sort_by(|&i, &j| {
            let fi = race.elite_pool[i].fitness / pre_sizes[i].max(1) as f32;
            let fj = race.elite_pool[j].fitness / pre_sizes[j].max(1) as f32;
            fj.partial_cmp(&fi).unwrap_or(std::cmp::Ordering::Equal)
        });
        let reordered: Vec<EliteDna> = order.iter().map(|&i| race.elite_pool[i].clone()).collect();
        race.elite_pool = reordered;
        // Re-cluster after reorder so cluster ids correspond to the new positions
        // (spawn.rs reads race.elite_clusters in tournament rejection sampling).
        let (ids_post, sizes_post) = cluster_elites(&race.elite_pool);
        race.elite_clusters = ids_post;
        let cluster_count = sizes_post.iter().copied().max().map(|_| {
            let mut seen = std::collections::BTreeSet::new();
            for &c in race.elite_clusters.iter() { seen.insert(c); }
            seen.len()
        }).unwrap_or(0);
        let largest_cluster_size = sizes_post.iter().copied().max().unwrap_or(0);
        // Telemetry — small one-line stderr log per generation (parallel to the
        // existing println! GEN summary). Concise so it can be grepped easily.
        eprintln!("[NICHE] gen={} pool={} clusters={} largest={} thr={}",
            race.generation, race.elite_pool.len(), cluster_count, largest_cluster_size, NICHE_DISTANCE);

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
            let mut d_stag = 0; let mut d_sbld = 0; let mut d_fuel = 0;
            let mut d_hole = 0; let mut d_npc = 0; let mut d_unk = 0;
            for (_, r, _, _) in sorted.iter() {
                if !r.is_dead { continue; }
                match r.death_reason {
                    Some(crate::simulation::racer::DeathReason::HpZero) => d_hp += 1,
                    Some(crate::simulation::racer::DeathReason::Pinado) => d_pin += 1,
                    Some(crate::simulation::racer::DeathReason::Batida) => d_bat += 1,
                    Some(crate::simulation::racer::DeathReason::Stagnation) => d_stag += 1,
                    Some(crate::simulation::racer::DeathReason::SpeedBleed) => d_sbld += 1,
                    Some(crate::simulation::racer::DeathReason::Fuel) => d_fuel += 1,
                    Some(crate::simulation::racer::DeathReason::Hole) => d_hole += 1,
                    Some(crate::simulation::racer::DeathReason::NpcHit) => d_npc += 1,
                    None => d_unk += 1,
                }
            }
            let _ = writeln!(file, "Death reasons — HP:{} PINADO:{} BATIDA:{} STAG:{} SBLD:{} FUEL:{} HOLE:{} NPC:{} ?:{}",
                d_hp, d_pin, d_bat, d_stag, d_sbld, d_fuel, d_hole, d_npc, d_unk);

            // Tier 4.3 — NICHO-SPLITTING METRICS.
            // Diferencial de pressão: defensive pune wall_time 2×, aggressive premia
            // speed_on_track 1.5×. Esperamos divergência estatística: defensive com
            // wall_time MENOR, aggressive com speed_on_track MAIOR.
            let mut def_n = 0usize; let mut agg_n = 0usize;
            let mut def_wall_sum = 0.0f32; let mut agg_wall_sum = 0.0f32;
            let mut def_speed_sum = 0.0f32; let mut agg_speed_sum = 0.0f32;
            let mut def_fit_sum = 0.0f32; let mut agg_fit_sum = 0.0f32;
            for (_, r, _, _) in sorted.iter() {
                let f = calc_fitness(r);
                let sp = r.telemetry.speed_on_track();
                if r.niche == 0 {
                    def_n += 1; def_wall_sum += r.wall_time; def_speed_sum += sp; def_fit_sum += f;
                } else {
                    agg_n += 1; agg_wall_sum += r.wall_time; agg_speed_sum += sp; agg_fit_sum += f;
                }
            }
            let safe = |s: f32, n: usize| -> f32 { if n == 0 { 0.0 } else { s / n as f32 } };
            let _ = writeln!(file, "Niche — DEF n={} wall_mean={:.2}s speed_mean={:.1} fit_mean={:.0} | AGG n={} wall_mean={:.2}s speed_mean={:.1} fit_mean={:.0}",
                def_n, safe(def_wall_sum, def_n), safe(def_speed_sum, def_n), safe(def_fit_sum, def_n),
                agg_n, safe(agg_wall_sum, agg_n), safe(agg_speed_sum, agg_n), safe(agg_fit_sum, agg_n));

            // Tier 6.6 — Weight Scars telemetry.
            // pool_active_scars é o estado FINAL do pool (pós-decay + pós-inserção).
            // - total_active_scars_before: scars herdados do início desta gen.
            // - total_active_scars_after: sobreviveram ao decay (TTL>0 após -1).
            // - new_this_gen: scars recém-criados por BATIDA (antes de push no pool).
            // - pool_active: soma final (após merge top-10 e sort).
            let pool_active_scars: usize = race.elite_pool.iter()
                .map(|e| e.scar_mask.iter().filter(|&&t| t > 0).count())
                .sum();
            let pool_max_ttl: u8 = race.elite_pool.iter()
                .flat_map(|e| e.scar_mask.iter().copied())
                .max().unwrap_or(0);
            let _ = writeln!(file, "Scars — new_this_gen: {} | pool_active: {} | max_ttl: {} (pre_decay: {}, post_decay: {})",
                scars_created_this_gen, pool_active_scars, pool_max_ttl, total_active_scars_before, total_active_scars_after);

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
            // FIX Bug 2: NET max real (não do top-fitness; rewards múltiplos desacoplam fitness e NET).
            let net_max_real: f32 = nets.iter().cloned().fold(0.0_f32, f32::max);
            println!("📊 GEN {} | Laps: 0v={} 1v={} 2v={} 3+v={} | NET max={:.0}m median={:.0}m | Deaths {}/{}",
                race.generation, laps_0, laps_1, laps_2, laps_3p,
                net_max_real, median(&nets), deaths, total);

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
            // FIX 3 ELITISMO RÍGIDO: validation só coroa se melhor QUE o champion (não 85%).
            // Tolerância 0.85 permitia saw-tooth regressivo — cada validation aceitava
            // DNA 15% pior, acumulando erosão. Agora só substitui se realmente superou.
            // Decay automático pós-20 gens (linha 329) continua destravando deadlocks.
            // FIX coroação: removido `total_laps >= 1`. Lap = 1299m; carros
            // chegam a 1139m NET (87% volta) mas nunca fecham. Threshold trava em
            // bootstrap 280m apesar de evolução 4x. NET puro > champion_distance
            // basta — elitism rígido (linha acima) evita regressão.
            let val_ok = is_validation
                && this_gen_best_dist > race.champion_distance;
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
                // FIX Bug 1: persistir morpho + hypernet do champion (Tier 6.3/6.4 LITE).
                let _ = save_champion_dna_full(
                    "CHAMPION.dna",
                    &weights, scale,
                    &slow_weights, slow_scale,
                    race.generation as u32,
                    race.seed as u32,
                    config.current_track().track_width,
                    this_gen_best_dist,
                    race.current_morpho,
                    race.current_hypernet,
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

        // --- TIER 1.2: FITNESS HISTORY + EMERGENCY BRAKE ---
        // Registra mean-fitness da geração que acabou (ring buffer 100 gens).
        // Emergency brake: se (gen > 30) e mean recente-5 < mean around-30 × 0.5,
        // ativa boost counter de 10 gens (axiom_strength piso 0.5 pra forçar recuperação).
        let gen_fitness: f32 = {
            let n = sorted.len();
            if n == 0 { 0.0 }
            else { sorted.iter().map(|(_, r, _, _)| calc_fitness(r)).sum::<f32>() / n as f32 }
        };
        if race.fitness_history.len() == super::settings::FITNESS_HISTORY_CAP {
            race.fitness_history.pop_front();
        }
        race.fitness_history.push_back(gen_fitness);
        if race.axiom_boost_counter > 0 {
            race.axiom_boost_counter -= 1;
        }
        if race.generation > 30 && race.fitness_history.len() >= 30 {
            let hist_len = race.fitness_history.len();
            let recent_slice: Vec<f32> = race.fitness_history.iter().rev().take(5).copied().collect();
            let recent_mean = recent_slice.iter().sum::<f32>() / recent_slice.len() as f32;
            // "around 30" = janela de 5 gens centrada no offset 30 a partir do fim.
            let start = hist_len.saturating_sub(32);
            let end = hist_len.saturating_sub(27).max(start + 1).min(hist_len);
            let ref_slice = &race.fitness_history.iter().copied().collect::<Vec<f32>>()[start..end];
            let ref_mean = ref_slice.iter().sum::<f32>() / ref_slice.len().max(1) as f32;
            if recent_mean < ref_mean * 0.5 && race.axiom_boost_counter == 0 {
                race.axiom_boost_counter = 10;
                if let Some(mut file) = open_report_log() {
                    use std::io::Write;
                    let _ = writeln!(file, "🚨 EMERGENCY BRAKE: mean_fit recente {:.0} < {:.0} × 0.5 — axiom_strength floor 0.5 por 10 gens",
                        recent_mean, ref_mean);
                }
            }
        }

        for e in &race.world_entities { if let Ok(mut e_cmd) = commands.get_entity(*e) { e_cmd.despawn(); } }
        for (e, _, _, _) in query.iter() { commands.entity(e).despawn(); }
        race.generation += 1; race.should_reset = false; race.hard_reset = false; race.selected_racer = None;

        // CURRICULUM LEVEL + VALIDATION GEN (ver decide_next_env_level)
        // Calculado ANTES do world rebuild pra Tier 5.3 pinar validation no track 0.
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

        // --- TIER 5.3 MULTI-TRACK ROTATION ---
        // Decide qual track a PRÓXIMA geração vai rodar. `race.generation` já foi incrementado
        // acima. Validation gens pinam em tracks[0] pra comparação honesta de campeão.
        let next_gen_idx = race.generation as usize;
        let next_track = config.track_for_gen(next_gen_idx, is_val);
        let track_changed = config.rotate_tracks && next_track.seed != race.seed;
        if track_changed {
            // Campeão salvo foi num track diferente — threshold vira armadilha
            // (não replica numa pista nova). Decay 0.8 força rebaseline rápido
            // sem apagar o DNA (best_weights preservado).
            race.champion_distance *= 0.8;
        }
        race.seed = next_track.seed;

        // --- CURRICULUM ADAPTATIVO DE LARGURA ---
        // Pista começa um pouco mais larga (base + 4m) e estreita rápido com voltas do campeão.
        // Cada volta completada reduz o bônus em 2m — piso (base) atingido em 2 voltas reais.
        let base_width = next_track.track_width;
        let champion_laps = (race.champion_distance / world_res.0.lap_distance.max(1.0)).max(0.0);
        let width_bonus = (4.0 - champion_laps * 2.0).max(0.0);
        let effective_width = base_width + width_bonus;
        if config.rotate_tracks {
            println!("🔀 TRACK ROTATION gen={} → \"{}\" (seed={}, width={:.1}m, val={})",
                next_gen_idx, next_track.name, next_track.seed, effective_width, is_val);
            if let Some(mut file) = open_report_log() {
                use std::io::Write;
                let _ = writeln!(file, "🔀 **TRACK**: {} (seed={}, base_width={:.1}m, validation={})",
                    next_track.name, next_track.seed, base_width, is_val);
            }
        }
        let world = VoxelWorld::new(next_track.seed, effective_width);
        let (r_mesh, g_mesh, w_mesh) = world.generate_mesh();
        // FIX: SimulationEntity marker garante que cleanup_simulation (OnExit) despawne
        // estas meshes se o usuário voltar pro menu. Sem ele, sair+reentrar sim deixa
        // o mapa da geração anterior visível por baixo do novo.
        let road_e = commands.spawn((Mesh3d(meshes.add(r_mesh)), MeshMaterial3d(materials.add(StandardMaterial { base_color: Color::srgb(0.15, 0.15, 0.18), perceptual_roughness: 0.85, ..default() })), SimulationEntity)).id();
        let grass_e = commands.spawn((Mesh3d(meshes.add(g_mesh)), MeshMaterial3d(materials.add(StandardMaterial { base_color: Color::srgb(0.1, 0.3, 0.1), perceptual_roughness: 0.9, ..default() })), SimulationEntity)).id();
        let wall_e = commands.spawn((Mesh3d(meshes.add(w_mesh)), MeshMaterial3d(materials.add(StandardMaterial { base_color: Color::srgb(0.7, 0.7, 0.7), metallic: 1.0, perceptual_roughness: 0.1, ..default() })), SimulationEntity)).id();
        race.world_entities = vec![road_e, grass_e, wall_e];
        // Spawn batched: agenda jobs, drain_spawn_queue processa em blocos por frame.
        race.pending_spawn = Some((0, config.swarm_count));
        world_res.0 = world;

        // Tier 5.4 — NPCs obstáculos móveis spawnam DEPOIS do world rebuild
        // (precisa de waypoints_world válidos). Gate curricular (gen >= NPC_ACTIVATION_GEN)
        // está dentro de spawn_npcs. Validation gens limparam npcs via reset_fixed.
        env.spawn_npcs(&world_res.0.waypoints_world);
        if !env.npcs.is_empty() {
            println!("🚧 NPCS gen={} count={} waypoints={}",
                race.generation, env.npcs.len(), world_res.0.waypoints_world.len());
        }
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
