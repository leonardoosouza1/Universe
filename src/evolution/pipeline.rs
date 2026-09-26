use crate::SimulationEntity;
use crate::config::UniverseConfig;
use crate::environment::EnvironmentalState;
use crate::simulation::Racer;
use crate::world::{VoxelWorld, VoxelWorldResource};
use bevy::prelude::*;
use iara_genesis::codec::semantic::BehaviorFingerprint;

use super::dna_genesis::{EliteDna, apply_catastrophic_scars, decay_scars};

// Fitness-sharing niching threshold: parents/elites with Hamming distance
// > NICHE_DISTANCE on their 64-bit BehaviorFingerprint are considered
// different behavioral clusters. 8/64 = 12.5% disagreement — empirically
// tight enough to split genuinely different strategies but loose enough to
// group minor mutations of the same lineage.
// v14: tried 10, got worse (everything 1 cluster). Reverted to 8.
//
// ⚠️ THAT CALIBRATION IS NULL. It was done when `BehaviorFingerprint` SATURATED and every
// pairwise distance was 0 — any threshold gave 1 cluster. The experiment did not measure
// what it thought it measured. See `iara_genesis/src/codec/semantic.rs`.
//
// MEASURED after the fingerprint fix (`telemetry/probe_dist`):
//   gen 1-3: dist_med 5.0-5.3 · dist_max 10
//   gen 4-7: dist_med 2.4-2.6 · dist_max 8
//
// In other words: the threshold 8 is 1.5-3× LARGER than the typical distance between elites.
// It erases all the diversity that exists — and it does exist, because `dist_med` drops from
// 5.3 to 2.4 but never reaches zero. Hence `clusters=1` in every generation.
pub(super) const NICHE_DISTANCE: u32 = 8;

/// ADAPTIVE niche threshold: the median of the pool's own pairwise distances.
///
/// Same principle that fixed the fingerprint — **relative criterion, not absolute**. A
/// fixed threshold needs to be calibrated against a distance scale that changes over the
/// course of evolution (5.3 at the start, 2.4 after converging); the median tracks it on
/// its own.
///
/// With the median as threshold, ~half of the pairs count as the same niche and the other
/// half separate — so `clusters` stays between 2 and ~pool/2 by construction, instead of
/// collapsing.
///
/// `NICHE_DISTANCE` becomes a CEILING: never loosen beyond what the old calibration allowed.
pub(super) fn limiar_de_nicho(dists: &[u32]) -> u32 {
    if dists.is_empty() {
        return NICHE_DISTANCE;
    }
    let mut d: Vec<u32> = dists.to_vec();
    d.sort_unstable();
    d[d.len() / 2].min(NICHE_DISTANCE)
}

/// All pairwise Hamming distances between the elites.
///
/// Used to tell whether `clusters=1` is genuine monoculture or a loose threshold: the
/// cluster count alone cannot distinguish the two cases.
pub(super) fn distancias_entre_elites(elites: &[EliteDna]) -> Vec<u32> {
    let fps: Vec<BehaviorFingerprint> = elites
        .iter()
        .map(|e| {
            let q: Vec<i8> = e
                .fast_weights
                .iter()
                .map(|&w| (w * 127.0).clamp(-127.0, 127.0) as i8)
                .collect();
            BehaviorFingerprint::from_weights(&q)
        })
        .collect();
    let mut out = Vec::new();
    for i in 0..fps.len() {
        for j in (i + 1)..fps.len() {
            out.push(fps[i].distance(&fps[j]));
        }
    }
    out
}

/// Clusters elites by Hamming distance on BehaviorFingerprint.
/// Returns (cluster_id_per_elite, cluster_size_per_elite).
/// Greedy single-pass: elite joins the first existing cluster whose
/// representative is within the threshold, otherwise opens a new one.
///
/// The threshold is ADAPTIVE (`limiar_de_nicho`): the median of the pool's own distances,
/// capped by `NICHE_DISTANCE`. A fixed threshold does not work because the distance scale
/// changes over the course of evolution — measured 5.3 in the first generations and 2.4
/// after converging.
pub(super) fn cluster_elites(elites: &[EliteDna], adaptativo: bool) -> (Vec<usize>, Vec<usize>) {
    let fps: Vec<BehaviorFingerprint> = elites
        .iter()
        .map(|e| {
            let q: Vec<i8> = e
                .fast_weights
                .iter()
                .map(|&w| (w * 127.0).clamp(-127.0, 127.0) as i8)
                .collect();
            BehaviorFingerprint::from_weights(&q)
        })
        .collect();
    let mut pares = Vec::new();
    for i in 0..fps.len() {
        for j in (i + 1)..fps.len() {
            pares.push(fps[i].distance(&fps[j]));
        }
    }
    let limiar = if adaptativo {
        limiar_de_nicho(&pares)
    } else {
        NICHE_DISTANCE
    };
    let mut reps: Vec<BehaviorFingerprint> = Vec::new();
    let mut ids: Vec<usize> = Vec::with_capacity(elites.len());
    for fp in fps.iter() {
        let mut found: Option<usize> = None;
        for (i, r) in reps.iter().enumerate() {
            if fp.distance(r) <= limiar {
                found = Some(i);
                break;
            }
        }
        match found {
            Some(i) => ids.push(i),
            None => {
                reps.push(*fp);
                ids.push(reps.len() - 1);
            }
        }
    }
    let mut sizes = vec![0usize; reps.len()];
    for &id in ids.iter() {
        sizes[id] += 1;
    }
    let per_elite: Vec<usize> = ids.iter().map(|&id| sizes[id]).collect();
    (ids, per_elite)
}
use super::logs::open_report_log;
use super::settings::{DevSettings, RaceSettings};

/// Decides the environmental level of the next generation.
/// Mutates `gens_until_validation` and `is_validation_gen` in a pure/testable way.
/// Returns (level, is_validation_gen_flag).
/// Fraction of the population that must complete a lap for the curriculum to ADVANCE.
pub const FRACTION_TO_RISE: f32 = 0.30;
/// Below this fraction the curriculum RETREATS — the current level is out of reach.
pub const FRACTION_TO_FALL: f32 = 0.05;

pub fn decide_next_env_level(
    gens_until_validation: &mut u32,
    champion_distance: f32,
    lap_distance: f32,
    current_level: u8,
) -> (u8, bool) {
    // compatibility: without population data, behaves as before
    decide_next_env_level_pop(
        gens_until_validation,
        champion_distance,
        lap_distance,
        current_level,
        None,
    )
}

/// Decides the next environment level by the POPULATION, not by the champion.
///
/// # What was wrong
///
/// `level_for_champion` looks only at `champion_distance / lap_distance`. Measured in the
/// ablation runs: in generation 1 one car did 1.93 laps in the dry, which puts the target at
/// level 3, and the whole population goes along. The result, gen by gen:
///
/// | gen | level | completed ≥1 lap | mortality |
/// |---|---|---|---|
/// | 1 | 1 | **23.6/60** | 70% |
/// | 2 | 2 | 1.2/60 | 98% |
/// | 3 | 3 | 0.6/60 | 99% |
///
/// And it is not just "hard". At level 3 the draw goes to 15°C asphalt with a soaked track,
/// which drops grip to ~0.33. With that, the braking distance required at the tightest curve
/// goes to **130.9 m**, and the car's longest curvature probe reaches **120 m**:
///
/// > The car becomes PHYSICALLY BLIND to the curve. No brain can solve it, and what gets
/// > measured there is noise divided by noise.
///
/// # The two fixes
///
/// 1. **Advance by the population.** Only goes up if at least [`FRACTION_TO_RISE`] complete
///    a lap; retreats if fewer than [`FRACTION_TO_FALL`] manage. With hysteresis between
///    the two, to avoid oscillating.
/// 2. **Retreating is possible.** Before, the level only went up; one lucky generation
///    trapped the population in an environment it could not sustain, with no way back.
///
/// `frac_completou` = fraction of beings that closed ≥1 lap in the generation that just
/// ended. `None` = no data (keeps the old behavior, for the existing tests).
pub fn decide_next_env_level_pop(
    gens_until_validation: &mut u32,
    champion_distance: f32,
    lap_distance: f32,
    current_level: u8,
    frac_completou: Option<f32>,
) -> (u8, bool) {
    if *gens_until_validation <= 1 {
        *gens_until_validation = 3;
        return (0, true);
    }
    *gens_until_validation -= 1;

    let Some(frac) = frac_completou else {
        // legacy path: target by the champion, climbing at most 1 level per generation
        let target = EnvironmentalState::level_for_champion(champion_distance, lap_distance);
        return (target.min(current_level.saturating_add(1)), false);
    };

    let proximo = if frac >= FRACTION_TO_RISE {
        // the population consolidated this level: one step up, never two
        current_level.saturating_add(1).min(3)
    } else if frac < FRACTION_TO_FALL {
        // nobody sustains it: retreat to where something can be learned
        current_level.saturating_sub(1).max(1)
    } else {
        current_level.max(1) // consolidation zone: stay where it is
    };
    (proximo, false)
}

// The six reward-shaping constants (K_SMOOTH, K_SPEED, K_NOVEL, K_STAB, K_PHI,
// K_AESTH) were removed: each appeared ONCE in the file, in its own definition.
// The fitness in use is pure distance, and that list gave the impression of a
// multi-term objective that does not exist. If they become necessary again, they come
// back together with their consumer — a constant without a consumer is documentation
// of an intention, not code.

fn aesthetic_score(telem: &crate::simulation::racer::RacerTelemetry) -> f32 {
    let steer: Vec<f32> = telem.steering_history.iter().copied().collect();
    if steer.len() < 3 {
        return 0.0;
    }
    let mut jerk_sum = 0.0f32;
    let mut jerk_n = 0usize;
    for i in 2..steer.len() {
        let d1 = steer[i] - steer[i - 1];
        let d0 = steer[i - 1] - steer[i - 2];
        jerk_sum += (d1 - d0).abs();
        jerk_n += 1;
    }
    let jerk = if jerk_n > 0 {
        jerk_sum / jerk_n as f32
    } else {
        0.0
    };
    let speeds: Vec<f32> = telem
        .speed_history
        .iter()
        .zip(telem.on_road_mask.iter())
        .filter_map(|(&s, &on)| if on { Some(s) } else { None })
        .collect();
    let speed_var = if speeds.len() >= 2 {
        let mean = speeds.iter().sum::<f32>() / speeds.len() as f32;
        speeds.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / speeds.len() as f32
    } else {
        0.0
    };
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
    if n < 4 {
        return 0.0;
    }

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
        if vx <= 1e-6 || vy <= 1e-6 {
            return 0.0;
        }
        let cov: f32 = xs
            .iter()
            .zip(ys.iter())
            .map(|(x, y)| (x - mx) * (y - my))
            .sum::<f32>()
            / n;
        let c = cov / (vx.sqrt() * vy.sqrt());
        if c.is_nan() || c.is_infinite() {
            0.0
        } else {
            c
        }
    }

    let mut sum_abs = 0.0f32;
    let mut pairs = 0u32;
    for s in sensors.iter() {
        for a in actions.iter() {
            sum_abs += corr(s, a).abs();
            pairs += 1;
        }
    }
    if pairs == 0 {
        return 0.0;
    }
    let phi = sum_abs / pairs as f32;
    if phi.is_nan() || phi.is_infinite() {
        0.0
    } else {
        phi.clamp(0.0, 1.0)
    }
}

/// Lap bonus — integer (old) or continuous in fractional laps.
///
/// Separated from `calc_fitness` because `Racer` has ~60 fields and cannot be constructed
/// in a test; and because the step this function contains deserved to be testable on its own.
///
/// `continuo == false` or `lap_dist <= 0` (world has not yet written the length) return
/// the old behavior, which is what keeps the A/B comparison honest.
pub(super) fn bonus_de_volta(
    prog: f32,
    lap_dist: f32,
    total_laps: usize,
    bonus: f32,
    continuo: bool,
) -> f32 {
    if continuo && lap_dist > 0.0 {
        (prog / lap_dist) * bonus
    } else {
        total_laps as f32 * bonus
    }
}

pub(super) fn calc_fitness(r: &Racer, track_width: f32, lab: &crate::lab::LabParams) -> f32 {
    // ===================================================
    // SIMPLIFIED FITNESS — clean signal for selection.
    // ===================================================
    // Structure:
    //   base    = progress ON THE TRACK (domain 100× larger than tie-breakers)
    //   lap     = bonus per completed lap (50k each)
    //   joy     = fine tie-breaker between equidistant cars (~1%)
    //   wall    = soft regularization (not a fitness-killer, scale-aware for narrow tracks)
    //   death   = linear in "how little it moved before dying"
    //
    // Removed: exponential `prog^1.3` (created an artificial discontinuity),
    //          wall weight 500→50 (wall-riding now costs, but does not erase 40m of progress).

    // NET progress: how far the car moved SINCE its spawn. Fundamental with distributed
    // spawn — otherwise whoever spawns at the 80% waypoint and moves nothing gets high
    // fitness "for free".
    let prog = (r.track_progress - r.spawn_progress).max(0.0);
    // THE STEP INSIDE THE FITNESS.
    //
    // `total_laps` is an INTEGER and `lap_bonus` is worth 50,000, while `prog` reaches
    // ~5,800 m — the bonus dominates progress by ~9×. Selection then barely distinguishes
    // cars within the same lap bucket:
    //
    //   3.01 laps -> 155,000 · 3.99 laps -> 156,600 (**+1%**) · 4.00 laps -> 206,600 (+32%)
    //
    // The population was exactly there: best car at 3.72 laps of a physical ceiling of
    // 4.36. Going from 3.0 to 3.99 — the improvement one WANTS to select — was worth 1%.
    // It is the same step that hid the laps in the report, now inside the SELECTION SIGNAL.
    //
    // Continuous keeps the scale and the incentive to close laps, without the step. Kept
    // behind a flag so the A/B comparison is possible; `lap_dist_m == 0` (world has not yet
    // written the length) also falls back to the old behavior.
    let lap_bonus = bonus_de_volta(
        prog,
        r.lap_dist_m,
        r.total_laps,
        lab.lap_bonus,
        lab.lap_bonus_continuo,
    );
    let joy = r.joy_integral * 1.0; // Fine tie-breaker

    // Proximity bonus: incentivizes CLOSING the lap. Quadratic in NET until saturating at
    // 500m. A car at 480m (almost 1 lap) earns 9200 extra pts = clear gap vs 400m (6400) or
    // 300m (3600). Saturates at 10000 so it does not compete with lap_bonus 50k.
    // After a full lap, lap_bonus dominates and proximity stays stable.
    let prox_frac = (prog / 500.0).clamp(0.0, 1.0);
    let proximity_bonus = prox_frac * prox_frac * 10000.0;
    // Wall penalty: scales with track_width. Narrow tracks (<14m) have unavoidable walls.
    // Wide tracks (>20m) can avoid walls → larger penalty.
    // Formula: base 200 × (track_width / 14.0)^0.7 — grows slowly with width.
    let width_penalty_scale = (track_width / 14.0).powf(0.7).clamp(0.5, 2.0);
    // Tier 4.3 — Niche-splitting (simpler version, shared-pool).
    //   niche == 0 (defensive):  wall_penalty × 2.0  — punishes wall-scraping hard.
    //   niche == 1 (aggressive): speed_reward × 1.5  — rewards on-road speed.
    // Single pool: both compete for the same elite slots → natural cross-breed
    // via tournament/mate-preference. Lineage stays introspectable through the field.
    let wall_mult: f32 = if r.niche == 0 { 2.0 } else { 1.0 };
    let speed_mult: f32 = if r.niche == 1 { 1.5 } else { 1.0 };
    let wall_penalty = r.wall_time * 200.0 * width_penalty_scale * wall_mult;
    // Wall hits: each contact during grace costs 500 additional pts (outside grace it dies outright).
    let hit_penalty = r.wall_hits as f32 * 500.0;
    // Death penalty REWORKED: before, `(100-prog)*10` capped at 1000pts and zeroed after 100m
    // — dying was free after that. Now: 2000 flat + 5% of prog (lost potential).
    // 800m prog + death = 2000 + 400 = 2400pts. 800m prog survived = 0 penalty + SURVIVAL_BONUS.
    let death_penalty = if r.is_dead {
        lab.death_penalty + prog * 0.05
    } else {
        0.0
    };
    // Survival bonus: surviving to the end of the generation has flat value. Breaks the
    // "floor it + die" incentive when progress is equivalent to going slow and arriving.
    let survival_bonus = if r.is_dead { 0.0 } else { lab.survival_bonus };

    // --- TIER 1.1: DENSE REWARD-SHAPING SIGNALS ---
    // Each rewards behavior that correlates with long-term progress.
    // Floors kept low so they do not dominate lap_bonus (50k) in fluent champions.
    //
    // 1) Smoothness: penalizes steering jitter (clamp floor -500 — standing still is not an infinite prize).
    let steering_var = r.telemetry.steering_variance();
    let smoothness_reward = (-steering_var * lab.k_smooth).max(-500.0);
    // 2) Speed-on-track: accumulated ON-ROAD speed (anti-grass, anti-wall).
    let speed_reward = r.telemetry.speed_on_track() * lab.k_speed * speed_mult;
    // 3) Novelty: sqrt(unique on-road voxels). On-road mandatory = anti circle-exploit.
    let unique_voxels = r.telemetry.visited_voxels.len();
    let novelty_bonus = (unique_voxels as f32).sqrt() * lab.k_novel;
    // 4) Stability: integral of (1 - |yaw_rate|) on-road — rewards controlled drivability.
    let stability_reward = r.telemetry.stability_integral() * lab.k_stab;

    // Tier 6.9 — Phi (IIT) proxy: integrates sensors×actions into the fitness.
    // Theoretical max ~1.0 × 300 = 300pts. Typical 0.2-0.5 → 60-150pts (≈1-3%).
    let phi_reward = compute_phi_proxy(r) * lab.k_phi;
    // Tier 6.7 — Aesthetic judge: low jerk + consistent line.
    let aesthetic_reward = aesthetic_score(&r.telemetry) * lab.k_aesth;

    // Curve bonus (byte-lab transfer: "specialist paradox" — reward curve mastery).
    // Each major curve passed earns a step bonus. Distances are approximate midpoints
    // of tight turns on typical seed 9001 tracks. Works on any track: if a stretch
    // has no curve at that distance, the car just gets free points for going straight.
    let curve_thresholds: &[(f32, f32)] = &[
        (200.0, 2000.0),   // first gentle curve
        (500.0, 4000.0),   // first tight curve (~50°)
        (620.0, 5000.0),   // second tight curve (~52°)
        (760.0, 6000.0),   // third tight curve (~48°)
        (1000.0, 8000.0),  // mid-lap curves
        (1300.0, 10000.0), // far curves
    ];
    let curve_bonus: f32 = curve_thresholds
        .iter()
        .filter(|(threshold, _)| prog >= *threshold)
        .map(|(_, bonus)| bonus)
        .sum();

    lap_bonus
        + prog * 100.0
        + proximity_bonus
        + curve_bonus
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
    _dev: Res<DevSettings>,
    mut world_res: ResMut<VoxelWorldResource>,
    mut env: ResMut<EnvironmentalState>,
    query: Query<(
        Entity,
        &Racer,
        &crate::physics::EngineComponent,
        &crate::physics::PhysicsBody,
    )>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    config: Res<UniverseConfig>,
    time: Res<Time>,
    lab: Res<crate::lab::LabParams>,
    hl: Res<crate::HeadlessMode>,
) {
    // Environmental timer tick: uses the full virtual delta (NO cap). Engine+physics now run
    // in FixedUpdate (substep N× per frame at high time_scale); env here tracks total virtual
    // time — generation_timeout in virtual-seconds scales exactly right.
    let dt = time.delta_secs();
    env.tick(dt);
    if query.is_empty() {
        return;
    }
    // LIFE MODE: limit raised from 10 to 200 laps.
    // With long life (HP_INICIAL=300, regeneration 1 HP/s while advancing, FUEL=real clock),
    // the first being to complete 10 laps triggered the generation before the timeout — and
    // that cut the generation too early, preventing the other beings from accumulating
    // selection data. At 200 laps (>10× the time of a typical generation of ~600 s /
    // 60-70 s/lap ≈ 9 laps), in practice the generation always runs to the timeout — the
    // desired behavior. If an extremely competent being does >200 laps in a single run, the
    // generation ends normally. The field of 10 became a sentinel for "end when nobody is
    // learning anymore".
    let winner = query.iter().find(|(_, r, _, _)| r.total_laps >= 200);
    // Generation ends on ANY of:
    //  - winner (200 laps — sentinel practically unreachable in generation time)
    //  - all_dead (no point waiting with nobody alive)
    //  - configurable timeout (living cars had time to accumulate laps)
    //  - manual reset
    let all_dead = query.iter().all(|(_, r, _, _)| r.is_dead);
    // Timeout only for WIDE tracks (>30m) — narrow needs patience for cars
    // to learn. Wide stuck = timeout cuts the runaway.
    let track_w = config.current_track().track_width;
    let timeout_reached = if hl.enabled || track_w > 30.0 {
        env.generation_time >= config.generation_timeout_s
    } else {
        false
    };

    if winner.is_some() || all_dead || timeout_reached || race.should_reset {
        // Global rank by fitness — reused by the elite pool + report.
        let mut sorted: Vec<_> = query.iter().collect();
        sorted.sort_by(|a, b| {
            let score_a = calc_fitness(a.1, track_w, &lab);
            let score_b = calc_fitness(b.1, track_w, &lab);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // --- PERSISTENT ELITE POOL (Hall of Fame) ---
        // BEFORE: clear + top-10 of the current gen. A weak gen contaminated the pool →
        // collapse cascade (620m gen 6 → garbage pool → gen 7 weak too).
        // NOW: merge — only enters if it beats the pool's worst slot. Weak gens do not
        // erase good DNA from earlier gens. Champion + top elites stay stored.
        // --- TIER 6.6: GLOBAL SCAR DECAY at the generation boundary ---
        // Before injecting new candidates, decrement the TTL of every existing scar_mask.
        // Scars with TTL 0 become free for mutation again.
        let mut total_active_scars_before = 0usize;
        let mut total_active_scars_after = 0usize;
        for elite in race.elite_pool.iter_mut() {
            total_active_scars_before += elite.scar_mask.iter().filter(|&&t| t > 0).count();
            total_active_scars_after += decay_scars(&mut elite.scar_mask);
        }

        // If there WAS any CRASH (BATIDA) this gen, the lineage learns from the trauma:
        // scars propagate to the top-K elites (proxy for "evolutionary patience" — if
        // the top-N is related to whoever crashed, mark active weights).
        // Simple: count BATIDAs across the whole population.
        let gen_batidas = sorted
            .iter()
            .filter(|(_, r, _, _)| {
                matches!(
                    r.death_reason,
                    Some(crate::simulation::racer::DeathReason::Batida)
                )
            })
            .count();

        let mut scars_created_this_gen = 0usize;
        for (rank, (_, r, _, _)) in sorted.iter().take(10).enumerate() {
            let new_fit = calc_fitness(r, track_w, &lab);
            // ---- Trinity fitness penalties / rewards ----
            const LN3: f32 = 1.0986123; // ln(3) — max entropy of 3-class softmax
            let hv = r.engine.hard_veto_count as f32;
            let so = r.engine.soft_override_count as f32;
            let avg_entropy = if r.engine.gate_entropy_n > 0 {
                r.engine.gate_entropy_acc / (r.engine.gate_entropy_n as f32)
            } else {
                0.0
            };
            let entropy_norm = (avg_entropy / LN3).clamp(0.0, 1.0);
            let chaos_alive_bonus = if r.engine.last_chaos > 0.5 && !r.is_dead {
                10.0
            } else {
                0.0
            };
            let new_fit = new_fit - hv * 5.0 - so * 0.3 + entropy_norm * 20.0 + chaos_alive_bonus;
            // ---- end Trinity fitness ----
            let fast_weights: Vec<f32> = r.engine.instinct_layer.weights.clone();
            let mut scar_mask = vec![0u8; fast_weights.len()];
            // 1) Did this top-10 member itself die in a BATIDA (crash)? Mark directly.
            let own_batida = matches!(
                r.death_reason,
                Some(crate::simulation::racer::DeathReason::Batida)
            );
            // 2) Was there a BATIDA in the population and is this top-3 (dominant lineage)?
            //    Proxy: the population affects the whole pool via shared reproduction.
            let lineage_batida = gen_batidas > 0 && rank < 3;
            if own_batida || lineage_batida {
                let q: Vec<i8> = fast_weights
                    .iter()
                    .map(|&w| (w * 127.0).clamp(-127.0, 127.0) as i8)
                    .collect();
                let seed_bias = q.iter().map(|&w| w as i64).sum::<i64>() as u64
                    ^ (race.generation as u64).wrapping_mul(0xA5A5_A5A5)
                    ^ (rank as u64).wrapping_mul(0x1234_5678);
                let created = apply_catastrophic_scars(&mut scar_mask, &q, seed_bias);
                scars_created_this_gen += created;
            }
            let candidate = EliteDna {
                fast_weights,
                fast_scale: r.engine.instinct_layer.weight_scale,
                // Option 4: FloatLinear runtime → lossy quantize to i8 for DNA storage.
                // Clamp range [-2,2] → i8 [-127,127].
                slow_weights: r.engine.slow_brain.all_weights(),
                slow_scale: 1.0,
                fitness: new_fit,
                scar_mask,
                // Tier 4.2: captures mean hormonal state at the end of life.
                cortisol_mean: r.engine.metabolism.mean_cortisol(),
                dopamine_mean: r.engine.metabolism.mean_dopamine(),
                curiosity_mean: r.engine.metabolism.mean_curiosity(),
                // Tier 6.3 LITE: propagates morpho genes from the living racer to the elite pool.
                morpho: r.morpho,
                cortex_l2: r.engine.cortex_l2.weights.clone(),
                car_model_idx: r.car_model_idx,
                // v11: captures the KAN output head knots for inheritance across gens.
                kan_knots: r.kan_out.knots,
                perception_weights: r.engine.retina_perception.weights_to_vec(),
            };
            // Guard: only add to the pool if the racer reached a minimum distance (sanity check)
            // Prevents total-crash DNA (<=0m) from contaminating the pool.
            // v8b: 150m→100m. Cars on a 14m track do ~100m; threshold selective but not too much.
            // v10: remove is_dead filter — genetics preserves high fitness regardless of death.
            // Elite selection is fitness-based, not liveness-based. Dead agent at 917m > live at 50m.
            if r.track_progress < 25.0 {
                continue;
            }
            if race.elite_pool.len() < 10 {
                race.elite_pool.push(candidate);
            } else {
                // Find the pool's worst. If the new one is better, replace it.
                let (worst_idx, worst_fit) = race
                    .elite_pool
                    .iter()
                    .enumerate()
                    .min_by(|a, b| {
                        a.1.fitness
                            .partial_cmp(&b.1.fitness)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
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
        let (_pre_ids, pre_sizes) = cluster_elites(&race.elite_pool, lab.nicho_adaptativo);
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
        let (ids_post, sizes_post) = cluster_elites(&race.elite_pool, lab.nicho_adaptativo);
        race.elite_clusters = ids_post;
        let cluster_count = sizes_post
            .iter()
            .copied()
            .max()
            .map(|_| {
                let mut seen = std::collections::BTreeSet::new();
                for &c in race.elite_clusters.iter() {
                    seen.insert(c);
                }
                seen.len()
            })
            .unwrap_or(0);
        let largest_cluster_size = sizes_post.iter().copied().max().unwrap_or(0);
        // Telemetry — small one-line stderr log per generation (parallel to the
        // existing println! GEN summary). Concise so it can be grepped easily.
        // REAL DISTANCE between the elites, not just the cluster count.
        //
        // `clusters=1` alone is ambiguous: it can be genuine monoculture OR a too-loose
        // threshold. The distance distribution separates the two — if the typical distance
        // is 3 and the threshold is 8, the threshold is erasing diversity that EXISTS.
        //
        // It matters because the threshold's historical calibration ("v14: tried 10, got
        // worse — everything 1 cluster") was done when the fingerprint SATURATED and every
        // distance was 0. That conclusion is null, and this line is what allows redoing it
        // with real data.
        let dists = distancias_entre_elites(&race.elite_pool);
        let (d_med, d_max) = if dists.is_empty() {
            (0.0, 0)
        } else {
            (
                dists.iter().sum::<u32>() as f32 / dists.len() as f32,
                dists.iter().copied().max().unwrap_or(0),
            )
        };
        let thr_usado = if lab.nicho_adaptativo {
            limiar_de_nicho(&dists)
        } else {
            NICHE_DISTANCE
        };
        eprintln!(
            "[NICHE] gen={} pool={} clusters={} largest={} thr={} dist_med={:.1} dist_max={}",
            race.generation,
            race.elite_pool.len(),
            cluster_count,
            largest_cluster_size,
            thr_usado,
            d_med,
            d_max
        );

        // --- GENESIS REPORT (FULL TELEMETRY) ---
        if let Some(mut file) = open_report_log() {
            use std::io::Write;
            let reason = if winner.is_some() {
                "Winner Found"
            } else if all_dead {
                "Extinction (Collision)"
            } else if timeout_reached {
                "Timeout Reached"
            } else {
                "Manual Reset"
            };
            let _ = writeln!(
                file,
                "\n## GENERATION {} | End Reason: {} | Duration: {:.1}s",
                race.generation, reason, env.generation_time
            );
            let _ = writeln!(
                file,
                "**Environment:** Asphalt {:.0}°C | Rain: {:.0}% | Wind: ({:.1}, {:.1}) m/s | Time: {:.1}h | μ_base: 1.3 × grip_env: {:.2}",
                env.asphalt_temp,
                env.weather_wetness * 100.0,
                env.wind.x,
                env.wind.z,
                env.time_of_day,
                env.combined_grip_factor()
            );

            // --- REPORT OPTIMIZED FOR 500+ CARS ---
            // Before: 500 lines per gen × ~200 gens = 100k lines in race_reports.md (~2MB/cycle).
            // Now: TOP 15 (the ones that matter) + BOTTOM 3 (to see how the worst are dying)
            // + AGGREGATED SUMMARY (medians, death rate, NET p50/p90).
            // Shrinks the file ~20× while keeping useful debug signal.
            let total = sorted.len();
            let top_n = 15.min(total);
            let bot_n = if total > top_n + 3 { 3 } else { 0 };
            let dump_row = |file: &mut std::fs::File,
                            rank: usize,
                            r: &Racer,
                            engine_comp: &crate::physics::EngineComponent,
                            phys_body: &crate::physics::PhysicsBody| {
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
                let _ = writeln!(
                    file,
                    "{:03}. [{}{}] **{}** | NET:{:.1}m | V:{:.0}km/h | G:{} | RPM:{:.0} | YAW:{:.2}r/s | SR:{:.2}/{:.2} | Thr:{:.0}% Brk:{:.0}%",
                    rank,
                    grid_tag,
                    status,
                    r.name,
                    net,
                    speed_kmh,
                    gear_label,
                    engine_comp.current_rpm,
                    yaw_rate,
                    rl_slip,
                    rr_slip,
                    engine_comp.throttle * 100.0,
                    engine_comp.brake * 100.0
                );
            };

            let _ = writeln!(file, "\n### 🏁 TOP {}", top_n);
            for (i, (_, r, engine_comp, phys_body)) in sorted.iter().take(top_n).enumerate() {
                dump_row(&mut file, i + 1, r, engine_comp, phys_body);
            }
            if bot_n > 0 {
                let _ = writeln!(file, "\n### 🪦 BOTTOM {}", bot_n);
                for (idx, (_, r, engine_comp, phys_body)) in
                    sorted.iter().rev().take(bot_n).enumerate()
                {
                    dump_row(&mut file, total - idx, r, engine_comp, phys_body);
                }
            }

            // Aggregated summary: NET/fitness/survival distribution.
            let nets: Vec<f32> = sorted
                .iter()
                .map(|(_, r, _, _)| (r.track_progress - r.spawn_progress).max(0.0))
                .collect();
            let deaths = sorted.iter().filter(|(_, r, _, _)| r.is_dead).count();
            let fits: Vec<f32> = sorted
                .iter()
                .map(|(_, r, _, _)| calc_fitness(r, track_w, &lab))
                .collect();
            let median = |v: &[f32]| -> f32 { if v.is_empty() { 0.0 } else { v[v.len() / 2] } };
            let p90 = |v: &[f32]| -> f32 {
                if v.is_empty() {
                    0.0
                } else {
                    v[(v.len() * 1 / 10).min(v.len() - 1)]
                }
            }; // v already sorted desc
            let mean = |v: &[f32]| -> f32 {
                if v.is_empty() {
                    0.0
                } else {
                    v.iter().sum::<f32>() / v.len() as f32
                }
            };
            let grid_count = sorted
                .iter()
                .filter(|(_, r, _, _)| r.spawn_progress < 1.0)
                .count();
            let _ = writeln!(
                file,
                "\n### 📊 SUMMARY | N={} | Deaths: {}/{} ({:.0}%) | Grid: {} | Distrib: {}",
                total,
                deaths,
                total,
                (deaths as f32 / total as f32) * 100.0,
                grid_count,
                total - grid_count
            );
            let _ = writeln!(
                file,
                "NET — max: {:.1}m | p90: {:.1}m | median: {:.1}m | mean: {:.1}m",
                nets.first().copied().unwrap_or(0.0),
                p90(&nets),
                median(&nets),
                mean(&nets)
            );
            let _ = writeln!(
                file,
                "Fitness — max: {:.0} | p90: {:.0} | median: {:.0} | mean: {:.0}",
                fits.first().copied().unwrap_or(0.0),
                p90(&fits),
                median(&fits),
                mean(&fits)
            );

            // Lap completion distribution — how many completed 1, 2, 3+ laps
            // stored in RaceSettings because the curriculum is decided outside, in another scope
            race.frac_completou_volta = if total > 0 {
                sorted
                    .iter()
                    .filter(|(_, r, _, _)| r.total_laps >= 1)
                    .count() as f32
                    / total as f32
            } else {
                0.0
            };
            let laps_0 = sorted
                .iter()
                .filter(|(_, r, _, _)| r.total_laps == 0)
                .count();
            let laps_1 = sorted
                .iter()
                .filter(|(_, r, _, _)| r.total_laps == 1)
                .count();
            let laps_2 = sorted
                .iter()
                .filter(|(_, r, _, _)| r.total_laps == 2)
                .count();
            let laps_3p = sorted
                .iter()
                .filter(|(_, r, _, _)| r.total_laps >= 3)
                .count();
            let _ = writeln!(
                file,
                "Laps — 0v: {} | 1v: {} | 2v: {} | 3+v: {}",
                laps_0, laps_1, laps_2, laps_3p
            );

            // Death reason breakdown — categorical via the DeathReason enum.
            let mut d_hp = 0;
            let mut d_pin = 0;
            let mut d_bat = 0;
            let mut d_stag = 0;
            let mut d_sbld = 0;
            let mut d_fuel = 0;
            let mut d_hole = 0;
            let mut d_npc = 0;
            let mut d_unk = 0;
            for (_, r, _, _) in sorted.iter() {
                if !r.is_dead {
                    continue;
                }
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
            let _ = writeln!(
                file,
                "Death reasons — HP:{} PINADO:{} BATIDA:{} STAG:{} SBLD:{} FUEL:{} HOLE:{} NPC:{} ?:{}",
                d_hp, d_pin, d_bat, d_stag, d_sbld, d_fuel, d_hole, d_npc, d_unk
            );

            // Tier 4.3 — NICHE-SPLITTING METRICS.
            // Pressure differential: defensive punishes wall_time 2×, aggressive rewards
            // speed_on_track 1.5×. We expect statistical divergence: defensive with
            // LOWER wall_time, aggressive with HIGHER speed_on_track.
            let mut def_n = 0usize;
            let mut agg_n = 0usize;
            let mut def_wall_sum = 0.0f32;
            let mut agg_wall_sum = 0.0f32;
            let mut def_speed_sum = 0.0f32;
            let mut agg_speed_sum = 0.0f32;
            let mut def_fit_sum = 0.0f32;
            let mut agg_fit_sum = 0.0f32;
            for (_, r, _, _) in sorted.iter() {
                let f = calc_fitness(r, track_w, &lab);
                let sp = r.telemetry.speed_on_track();
                if r.niche == 0 {
                    def_n += 1;
                    def_wall_sum += r.wall_time;
                    def_speed_sum += sp;
                    def_fit_sum += f;
                } else {
                    agg_n += 1;
                    agg_wall_sum += r.wall_time;
                    agg_speed_sum += sp;
                    agg_fit_sum += f;
                }
            }
            let safe = |s: f32, n: usize| -> f32 { if n == 0 { 0.0 } else { s / n as f32 } };
            let _ = writeln!(
                file,
                "Niche — DEF n={} wall_mean={:.2}s speed_mean={:.1} fit_mean={:.0} | AGG n={} wall_mean={:.2}s speed_mean={:.1} fit_mean={:.0}",
                def_n,
                safe(def_wall_sum, def_n),
                safe(def_speed_sum, def_n),
                safe(def_fit_sum, def_n),
                agg_n,
                safe(agg_wall_sum, agg_n),
                safe(agg_speed_sum, agg_n),
                safe(agg_fit_sum, agg_n)
            );

            // Tier 6.6 — Weight Scars telemetry.
            // pool_active_scars is the pool's FINAL state (post-decay + post-insertion).
            // - total_active_scars_before: scars inherited at the start of this gen.
            // - total_active_scars_after: survived the decay (TTL>0 after -1).
            // - new_this_gen: scars freshly created by BATIDA (before push into the pool).
            // - pool_active: final sum (after top-10 merge and sort).
            let pool_active_scars: usize = race
                .elite_pool
                .iter()
                .map(|e| e.scar_mask.iter().filter(|&&t| t > 0).count())
                .sum();
            let pool_max_ttl: u8 = race
                .elite_pool
                .iter()
                .flat_map(|e| e.scar_mask.iter().copied())
                .max()
                .unwrap_or(0);
            let _ = writeln!(
                file,
                "Scars — new_this_gen: {} | pool_active: {} | max_ttl: {} (pre_decay: {}, post_decay: {})",
                scars_created_this_gen,
                pool_active_scars,
                pool_max_ttl,
                total_active_scars_before,
                total_active_scars_after
            );

            // --- DEATH BY ZONE (waypoint segment) ---
            // Groups deaths by track segment. Identifies the "killer curve".
            let world = &world_res.0;
            let n_segs = world.waypoints_world.len().saturating_sub(1).max(1);
            let mut deaths_per_seg = vec![0u32; n_segs];
            for (_, r, _, _) in sorted.iter() {
                if r.is_dead {
                    let seg = world
                        .waypoint_zone(r.death_pos.x, r.death_pos.z)
                        .min(n_segs - 1);
                    deaths_per_seg[seg] += 1;
                }
            }
            let _ = writeln!(file, "\n### 💀 DEATHS PER SEGMENT (waypoint i → i+1)");
            for (i, count) in deaths_per_seg.iter().enumerate() {
                let p1 = world.waypoints_world[i];
                let p2 = world.waypoints_world[i + 1];
                let bar = "█".repeat((((*count as f32) / (total as f32).max(1.0)) * 30.0) as usize);
                let _ = writeln!(
                    file,
                    "seg[{}] ({:.0},{:.0})→({:.0},{:.0}): {:3}  {}",
                    i, p1.x, p1.y, p2.x, p2.y, count, bar
                );
            }

            // --- ASCII MAP with death markers + finish positions ---
            let death_points: Vec<Vec3> = sorted
                .iter()
                .filter(|(_, r, _, _)| r.is_dead)
                .map(|(_, r, _, _)| r.death_pos)
                .collect();
            let alive_positions: Vec<Vec3> = sorted
                .iter()
                .filter(|(_, r, _, _)| !r.is_dead)
                .take(30)
                .map(|(_, _, _, body)| {
                    let _ = body;
                    Vec3::ZERO
                })
                .collect();
            let map = world.render_ascii_map(100, 40, &death_points, &alive_positions);
            let _ = writeln!(
                file,
                "\n### 🗺️  MAP (100×40) | `.`=grass `=`=road `#`=wall `0-7`=waypoints `X`=death"
            );
            let _ = writeln!(file, "```");
            let _ = write!(file, "{}", map);
            let _ = writeln!(file, "```");

            // --- PER-RACER ENHANCED (top + anyone with a lap) ---
            let with_lap: Vec<_> = sorted
                .iter()
                .filter(|(_, r, _, _)| r.total_laps > 0)
                .collect();
            if !with_lap.is_empty() {
                let _ = writeln!(file, "\n### 🏆 RACERS WITH COMPLETED LAPS (all)");
                let _ = writeln!(
                    file,
                    "| Name | Laps | NET(m) | Peak(km/h) | Walls | Death | Frame | Zone |"
                );
                let _ = writeln!(file, "|---|---|---|---|---|---|---|---|");
                for (_, r, _, _) in with_lap.iter() {
                    let net = (r.track_progress - r.spawn_progress).max(0.0);
                    let death = r.death_reason.map(|d| d.label()).unwrap_or("ALIVE");
                    let zone = if r.is_dead {
                        world.waypoint_zone(r.death_pos.x, r.death_pos.z) as i32
                    } else {
                        -1
                    };
                    let _ = writeln!(
                        file,
                        "| {} | {} | {:.0} | {:.0} | {} | {} | {} | {} |",
                        r.name,
                        r.total_laps,
                        net,
                        r.peak_speed * 3.6,
                        r.wall_hits,
                        death,
                        r.death_frame,
                        if zone < 0 {
                            "-".to_string()
                        } else {
                            zone.to_string()
                        }
                    );
                }
            }

            // Compact console summary for live feedback.
            // FIX Bug 2: real NET max (not the top-fitness one; multiple rewards decouple fitness and NET).
            let net_max_real: f32 = nets.iter().cloned().fold(0.0_f32, f32::max);
            println!(
                "📊 GEN {} | Laps: 0v={} 1v={} 2v={} 3+v={} | NET max={:.0}m median={:.0}m | Deaths {}/{}",
                race.generation,
                laps_0,
                laps_1,
                laps_2,
                laps_3p,
                net_max_real,
                median(&nets),
                deaths,
                total
            );

            // USE of the pathways (mean gate weight) next to the ablated pathway. Greppable line.
            // Use is NOT contribution: the pair (use here, ablation cost across variants) is the
            // result of the allocation experiment — see `neuro.rs` and `lab_specs/alocacao.json`.
            {
                let (uso, n, vetos) = crate::neuro::uso_medio();
                println!(
                    "🧠 ALOC gen={} ablada={} uso_cortex={:.3} uso_reflexo={:.3} \
                          uso_sentinela={:.3} vetos={} n={} net_max={:.0} net_med={:.0}",
                    race.generation,
                    crate::neuro::ablada().nome(),
                    uso[0],
                    uso[1],
                    uso[2],
                    vetos,
                    n,
                    net_max_real,
                    median(&nets)
                );
                crate::neuro::zerar_uso();
            }

            // --- BRAIN DIAGNOSTICS OF THE TOP-3 (NET) ---
            // Shows the history of the 3 best. If the steering does not respond when
            // the wall arrives, it will show up here: wall_prox=120 but raw_steer≈0.
            let _ = writeln!(file, "\n### 🧠 BRAIN TRACE — Top 3 (NET)");
            for (i, (_, r, _, _)) in sorted.iter().take(3).enumerate() {
                let net = (r.track_progress - r.spawn_progress).max(0.0);
                let _ = writeln!(
                    file,
                    "\n#### {}. {} — NET {:.1}m | {} snapshots",
                    i + 1,
                    r.name,
                    net,
                    r.brain_trace.len()
                );
                let _ = writeln!(
                    file,
                    "| t(s) | V(km/h) | WL | WR | TTC | Curv25 | Retina_C | raw_thr | raw_steer | post_damp | phys_steer | steer_lim | risk_damp | TOP5 channels |"
                );
                let _ = writeln!(
                    file,
                    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
                );
                for s in r.brain_trace.iter() {
                    let top5s = s
                        .top_channels
                        .iter()
                        .filter(|(_, v)| *v != 0)
                        .map(|(i, v)| format!("c{}={}", i, v))
                        .collect::<Vec<_>>()
                        .join(" ");
                    let _ = writeln!(
                        file,
                        "| {:.1} | {:.0} | {} | {} | {} | {} | {} | {:+.2} | {:+.2} | {:+.2} | {:+.2} | {:.2} | {:.2} | {} |",
                        s.t,
                        s.speed_kmh,
                        s.wall_prox_l,
                        s.wall_prox_r,
                        s.ttc_inv,
                        s.curv_25,
                        s.retina_center_max,
                        s.raw_throttle,
                        s.raw_steer,
                        s.post_damp_steer,
                        s.physical_steer,
                        s.steer_limit,
                        s.risk_damp,
                        top5s
                    );
                }
            }
            let _ = writeln!(file, "");
        }

        // --- BEST OF THE GENERATION (any spawn) — visualization only ---
        // Separate from best_racer (which filters grid-spawn for crowning).
        // Here we want the path of whoever traveled the most, to show evolution on the map.
        if let Some((_, gen_best, _, _)) = query.iter().max_by(|a, b| {
            let na = (a.1.track_progress - a.1.spawn_progress).max(0.0);
            let nb = (b.1.track_progress - b.1.spawn_progress).max(0.0);
            na.partial_cmp(&nb).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            race.last_gen_path = gen_best.path.clone();
            race.last_gen_net = (gen_best.track_progress - gen_best.spawn_progress).max(0.0);
            race.last_gen_was_grid = gen_best.spawn_progress < 1.0;
            // Debug: dump best racer's brain trace for vision analysis (15Hz sampling)
            let run_id = format!("gen_{}", race.generation);
            crate::evolution::telemetry::save_brain_trace(
                &run_id,
                &gen_best.name,
                &gen_best.brain_trace,
            );
        }

        // SELECTION BY NET PROGRESS, RESTRICTED TO GRID-SPAWNS.
        // Reason: cars with distributed spawn become "specialists" of an easy stretch
        // and fail when the DNA is reloaded into slot 0 (spawn at the start line). The
        // champion MUST be a car that proved it can leave the start line. Distributed cars
        // still contribute genes via elite_pool/tournament — they just cannot be CHAMPION.
        let net_prog = |r: &Racer| (r.track_progress - r.spawn_progress).max(0.0);
        let is_grid_spawn = |r: &Racer| r.spawn_progress < 1.0;
        let best_racer = if let Some((_, w, _, _)) = winner {
            Some(w)
        } else {
            query
                .iter()
                .filter(|(_, r, _, _)| is_grid_spawn(r))
                .max_by(|a, b| {
                    net_prog(a.1)
                        .partial_cmp(&net_prog(b.1))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(_, r, _, _)| r)
        };

        // --- THRESHOLD DECAY (anti-deadlock) ---
        // If 20 consecutive gens do not crown, the saved champion is probably not
        // replicable (it was a lucky-spec of one stretch). Reduces the threshold 5%/gen
        // after 20 gens without crowning, letting a "worse but CONSISTENT champion" take over.
        if race.gens_since_crown >= 20 {
            race.champion_distance *= 0.95;
        }

        if let Some(best) = best_racer {
            let this_gen_best_dist = net_prog(best);

            // CROWNING GATED BY VALIDATION GEN
            // Only crowns in gens run in a fixed env (level 0) — avoids lucky-env bias.
            // Winner with 5 laps always crowns (robust proof).
            // best_weights None = first DNA, always crowns for bootstrap.
            let is_validation = race.env_level == 0;
            // FIX 3 STRICT ELITISM: validation only crowns if better THAN the champion (not 85%).
            // Tolerance 0.85 allowed a regressive saw-tooth — each validation accepted
            // DNA 15% worse, accumulating erosion. Now it only replaces if it truly surpassed.
            // Automatic decay after 20 gens (line 329) keeps unlocking deadlocks.
            // Crowning FIX: removed `total_laps >= 1`. Lap = 1299m; cars
            // reach 1139m NET (87% of a lap) but never close it. Threshold got stuck at
            // bootstrap 280m despite 4x evolution. Pure NET > champion_distance
            // suffices — strict elitism (line above) prevents regression.
            let val_ok = is_validation && this_gen_best_dist > race.champion_distance;
            // SIMPLE ELITISM: ignores the validation-gate. Any gen where the best-racer
            // beats champion_distance → crowns. See melhorias/simple-elitism.md.
            #[cfg(feature = "simple_elitism")]
            let simple_ok = this_gen_best_dist > race.champion_distance;
            #[cfg(not(feature = "simple_elitism"))]
            let simple_ok = false;
            let should_crown =
                winner.is_some() || race.best_weights.is_none() || val_ok || simple_ok;

            // Guard removed: the ≥10% improvement requirement over best_ever_dist (which
            // never decays) blocked incremental gains after 664m — the champion had to
            // reach 730m in one go. champion_distance (which DOES decay) already ensures
            // enough elitism via val_ok / simple_ok above.
            if should_crown {
                race.champion_distance = this_gen_best_dist; // Update the champion threshold
                race.gens_since_crown = 0; // Reset the deadlock counter
                // Stores the champion's path to visualize in the next generation.
                race.champion_path = best.path.clone();
                race.champion_net_dist = this_gen_best_dist;
                let weights = best.engine.instinct_layer.weights.clone();
                let scale = best.engine.instinct_layer.weight_scale;
                // Option 4: FloatLinear → lossy i8 encoding at scale 63.5 for DNA storage.
                let slow_weights: Vec<f32> = best.engine.slow_brain.all_weights();
                let slow_scale: f32 = 1.0;
                race.best_weights = Some(weights.clone());
                race.best_car_model_idx = Some(best.car_model_idx);
                race.best_l2_weights = Some(best.engine.cortex_l2.weights.clone());
                race.best_scale = scale;
                race.best_slow_weights = Some(slow_weights.clone());
                race.best_slow_scale = slow_scale;
                // Trinity: capture champion gate + reflex weights alongside cortex.
                race.best_gate_weights = Some(best.engine.gate.weights.to_vec());
                race.best_gate_bias = Some(best.engine.gate.bias.to_vec());
                let mut reflex_w: Vec<f32> = Vec::with_capacity(96);
                reflex_w.extend_from_slice(&best.engine.reflex.steer.weights);
                reflex_w.extend_from_slice(&best.engine.reflex.throttle.weights);
                reflex_w.extend_from_slice(&best.engine.reflex.brake.weights);
                race.best_reflex_weights = Some(reflex_w);
                // Trinity weights snapshot for DNA save (v5 TRIN chunk).
                let trinity_save = {
                    let mut gate_w_arr = [0.0f32; 48];
                    gate_w_arr.copy_from_slice(&best.engine.gate.weights);
                    let mut gate_b_arr = [0.0f32; 3];
                    gate_b_arr.copy_from_slice(&best.engine.gate.bias);
                    // Quantize f32 weights → i8 [-127..127] for compact disk storage.
                    // Range mapping: w * 127, clamped. Lossy but DNA file stays 96 bytes.
                    let q = |w: f32| -> i8 { (w * 127.0).clamp(-127.0, 127.0) as i8 };
                    let mut steer_arr = [0i8; 32];
                    let mut thr_arr = [0i8; 32];
                    let mut brake_arr = [0i8; 32];
                    for (i, &w) in best.engine.reflex.steer.weights.iter().enumerate().take(32) {
                        steer_arr[i] = q(w);
                    }
                    for (i, &w) in best
                        .engine
                        .reflex
                        .throttle
                        .weights
                        .iter()
                        .enumerate()
                        .take(32)
                    {
                        thr_arr[i] = q(w);
                    }
                    for (i, &w) in best.engine.reflex.brake.weights.iter().enumerate().take(32) {
                        brake_arr[i] = q(w);
                    }
                    crate::evolution::dna_genesis::TrinityWeights {
                        gate_weights: gate_w_arr,
                        gate_bias: gate_b_arr,
                        reflex_steer: steer_arr,
                        reflex_throttle: thr_arr,
                        reflex_brake: brake_arr,
                    }
                };
                // FIX Bug 1: persist the champion's morpho + hypernet + trinity.
                // Cortex f32 runtime — quantize to i8 on save with scale 63.5
                // (covers the ±2.0 range of the mutation clamp without saturating).
                let weights_q: Vec<i8> = weights
                    .iter()
                    .map(|&w| (w * 63.5).clamp(-127.0, 127.0) as i8)
                    .collect();
                // V7: cortex_l2 also quantized for disk.
                let l2_q: Vec<i8> = best
                    .engine
                    .cortex_l2
                    .weights
                    .iter()
                    .map(|&w| (w * 63.5).clamp(-127.0, 127.0) as i8)
                    .collect();
                // Keep ternary for now. i8 quantization caused regression.
                let slow_weights_q: Vec<i8> = slow_weights
                    .iter()
                    .map(|&w| {
                        if w > 0.15 {
                            1i8
                        } else if w < -0.15 {
                            -1i8
                        } else {
                            0i8
                        }
                    })
                    .collect();
                // Save TWICE: generic CHAMPION.dna (latest run) + per-width
                // CHAMPION_W{width}m.dna (persistent per-track memory, does not unlearn).
                let track_w = config.current_track().track_width;
                let per_width = format!("CHAMPION_W{}m.dna", track_w as i32);
                // LINEAGE: one snapshot per crowning, with the generation in the name. It is
                // what allows answering "when was this synapse born" — `sinapses --contra`
                // between two consecutive snapshots shows what was born, died and thickened.
                // Chosen over recording the birth generation INSIDE the .dna: that would
                // require a new block in the format and an extra pass in the mutation hot
                // path, and this branch's legacy DNA tests already fail. Copying a file
                // carries no risk at all and gives the same information, outside the simulation.
                let _ = std::fs::create_dir_all("linhagem");
                let snapshot = format!("linhagem/gen{:05}.dna", race.generation);
                for path in &["CHAMPION.dna", per_width.as_str(), snapshot.as_str()] {
                    let _ = crate::evolution::dna_genesis::save_champion_dna_v7(
                        path,
                        &weights_q,
                        scale,
                        &slow_weights_q,
                        slow_scale,
                        race.generation as u32,
                        race.seed as u32,
                        track_w,
                        this_gen_best_dist,
                        race.current_morpho,
                        race.current_hypernet,
                        best.kan_out.knots,
                        Some(&trinity_save),
                        Some(&l2_q),
                        Some(&slow_weights),
                    );
                }
                // Update hard checkpoint
                race.best_ever_dist = this_gen_best_dist.max(race.best_ever_dist);
                if let Some(mut file) = open_report_log() {
                    use std::io::Write;
                    let _ = writeln!(
                        file,
                        "👑 **NEW CHAMPION CROWNED!** Distance: {:.1}m (Gen {}) — Surpassed previous {:.1}m",
                        this_gen_best_dist, race.generation, race.champion_distance
                    );
                }
            } else {
                race.gens_since_crown += 1;
                if let Some(mut file) = open_report_log() {
                    use std::io::Write;
                    let decay_note = if race.gens_since_crown >= 20 {
                        " [THRESHOLD IN DECAY -5%/gen]"
                    } else {
                        ""
                    };
                    let _ = writeln!(
                        file,
                        "📉 Weak generation ({:.1}m vs champion {:.1}m) — DNA PRESERVED (gens without crowning: {}){}",
                        this_gen_best_dist,
                        race.champion_distance,
                        race.gens_since_crown,
                        decay_note
                    );
                }
            }
        } else {
            // No grid-spawn had valid progress — keep counting the deadlock
            race.gens_since_crown += 1;
        }

        // Clear the legacy MVP weights (HARD RESET OF CORRUPTED WEIGHTS)
        if race.hard_reset {
            race.best_weights = None;
            race.best_car_model_idx = None;
            race.best_slow_weights = None;
            race.best_gate_weights = None;
            race.best_gate_bias = None;
            race.best_reflex_weights = None;
            race.death_markers.clear();
            let _ = std::fs::remove_file("CHAMPION.dna");
        }
        if race.should_reset {
            race.death_markers.clear();
        }

        // --- TIER 1.2: FITNESS HISTORY + EMERGENCY BRAKE ---
        // Records the mean fitness of the generation that just ended (100-gen ring buffer).
        // Emergency brake: if (gen > 30) and recent-5 mean < around-30 mean × 0.5,
        // activates a 10-gen boost counter (axiom_strength floor 0.5 to force recovery).
        let gen_fitness: f32 = {
            let n = sorted.len();
            if n == 0 {
                0.0
            } else {
                sorted
                    .iter()
                    .map(|(_, r, _, _)| calc_fitness(r, track_w, &lab))
                    .sum::<f32>()
                    / n as f32
            }
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
            let recent_slice: Vec<f32> =
                race.fitness_history.iter().rev().take(5).copied().collect();
            let recent_mean = recent_slice.iter().sum::<f32>() / recent_slice.len() as f32;
            // "around 30" = 5-gen window centered at offset 30 from the end.
            let start = hist_len.saturating_sub(32);
            let end = hist_len.saturating_sub(27).max(start + 1).min(hist_len);
            let ref_slice = &race.fitness_history.iter().copied().collect::<Vec<f32>>()[start..end];
            let ref_mean = ref_slice.iter().sum::<f32>() / ref_slice.len().max(1) as f32;
            if recent_mean < ref_mean * 0.5 && race.axiom_boost_counter == 0 {
                race.axiom_boost_counter = 10;
                if let Some(mut file) = open_report_log() {
                    use std::io::Write;
                    let _ = writeln!(
                        file,
                        "🚨 EMERGENCY BRAKE: recent mean_fit {:.0} < {:.0} × 0.5 — axiom_strength floor 0.5 for 10 gens",
                        recent_mean, ref_mean
                    );
                }
            }
        }

        for e in &race.world_entities {
            if let Ok(mut e_cmd) = commands.get_entity(*e) {
                e_cmd.despawn();
            }
        }
        for (e, _, _, _) in query.iter() {
            commands.entity(e).despawn();
        }

        // Close this generation's replay writer before incrementing the counter.
        // The next update_racers will open a new file for gen N+1.
        if let Some(ref mut rw) = race.replay_writer {
            let _ = rw.flush();
        }
        race.replay_writer = None;

        race.generation += 1;
        race.should_reset = false;
        race.hard_reset = false;
        race.selected_racer = None;

        // CURRICULUM LEVEL + VALIDATION GEN (see decide_next_env_level)
        // Computed BEFORE the world rebuild so Tier 5.3 can pin validation on track 0.
        let champ_dist = race.champion_distance;
        let lap_dist = world_res.0.lap_distance;
        let current_lvl = race.env_level;
        // fraction of the POPULATION that closed ≥1 lap — that is what governs the
        // curriculum now, not the champion's distance (see decide_next_env_level_pop)
        let frac_completou = Some(race.frac_completou_volta);
        let (next_level, is_val) = decide_next_env_level_pop(
            &mut race.gens_until_validation,
            champ_dist,
            lap_dist,
            current_lvl,
            frac_completou,
        );
        race.is_validation_gen = is_val;
        race.env_level = next_level;
        env.randomize_level(next_level);

        // --- TIER 5.3 MULTI-TRACK ROTATION ---
        // Decides which track the NEXT generation will run. `race.generation` was already
        // incremented above. Validation gens pin to tracks[0] for honest champion comparison.
        let next_gen_idx = race.generation as usize;
        let next_track = config.track_for_gen(next_gen_idx, is_val);
        let track_changed = config.rotate_tracks && next_track.seed != race.seed;
        if track_changed {
            // Saved champion was on a different track — the threshold becomes a trap
            // (does not replicate on a new track). Decay 0.8 forces a quick rebaseline
            // without erasing the DNA (best_weights preserved).
            race.champion_distance *= 0.8;
        }
        race.seed = next_track.seed;

        // --- ADAPTIVE WIDTH CURRICULUM ---
        // main.rs creates the initial world with track_width + 4.0 (wall margin).
        // The pipeline must use the SAME convention: full_width = base + 4.0.
        // Curriculum narrows from full_width by up to 4m as champion completes laps.
        // Floor: never narrower than base_width (the user-specified width).
        let base_width = next_track.track_width;
        let full_width = base_width + 4.0; // match main.rs convention
        // NARROWING BY THE POPULATION, not by the champion.
        //
        // It was `champion_laps * 0.5`, with `champion_laps = champion_distance / lap_distance`.
        // Same flaw as the environment curriculum, in a second mechanism: one car that did
        // 1.93 laps narrowed 0.96 m for all 60. Measured before this fix: the track went from
        // 18.0 to 17.1 m between gen 1 and 2, and the lap-completing fraction fell from 39%
        // to 7%.
        //
        // Now the target comes from the FRACTION of the population that closes a lap, with the
        // same dead zone as the environment curriculum (nothing below FRACTION_TO_RISE), and
        // the narrowing moves at most 0.25 m per generation — in BOTH directions, so it can
        // widen back when the population cannot sustain it.
        let alvo = 4.0 * ((race.frac_completou_volta - FRACTION_TO_RISE) / 0.50).clamp(0.0, 1.0);
        let passo = 0.25;
        race.estreitamento += (alvo - race.estreitamento).clamp(-passo, passo);
        let narrowing = race.estreitamento.clamp(0.0, 4.0);
        let effective_width = (full_width - narrowing).max(base_width);
        if config.rotate_tracks {
            println!(
                "🔀 TRACK ROTATION gen={} → \"{}\" (seed={}, width={:.1}m, val={})",
                next_gen_idx, next_track.name, next_track.seed, effective_width, is_val
            );
            if let Some(mut file) = open_report_log() {
                use std::io::Write;
                let _ = writeln!(
                    file,
                    "🔀 **TRACK**: {} (seed={}, base_width={:.1}m, validation={})",
                    next_track.name, next_track.seed, base_width, is_val
                );
            }
        }

        // Telemetry — always run, not only under rotate_tracks
        let nets: Vec<f32> = sorted
            .iter()
            .map(|(_, r, _, _)| r.track_progress.max(0.0))
            .collect();
        let n = nets.len().max(1);
        let avg_dist = nets.iter().sum::<f32>() / n as f32;
        let mut nets_sorted = nets.clone();
        nets_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p50_dist = nets_sorted[n / 2];
        let p90_dist = nets_sorted[(n * 9 / 10).min(n - 1)];
        let crashes = sorted.iter().filter(|(_, r, _, _)| r.is_dead).count();
        let speed_bleeds = sorted
            .iter()
            .filter(|(_, r, _, _)| {
                matches!(
                    r.death_reason,
                    Some(crate::simulation::racer::DeathReason::SpeedBleed)
                )
            })
            .count();

        // Count death reasons
        let mut death_counts: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for (_, r, _, _) in &sorted {
            if let Some(ref reason) = r.death_reason {
                *death_counts.entry(format!("{:?}", reason)).or_insert(0) += 1;
            }
        }

        let laps: usize = sorted.first().map(|(_, r, _, _)| r.total_laps).unwrap_or(0);
        let champ_dist = sorted
            .first()
            .map(|(_, r, _, _)| r.track_progress)
            .unwrap_or(0.0);
        let best_fitness = sorted
            .first()
            .map(|(_, r, _, _)| calc_fitness(r, track_w, &lab))
            .unwrap_or(0.0);
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        // Extract values from race BEFORE using race.telemetry
        let run_id_copy = race.telemetry.run_id.clone();
        let gen_num = race.generation;
        let track_seed = race.seed;
        let is_val = race.is_validation_gen;
        let env_temp = env.asphalt_temp;
        let env_wetness = env.weather_wetness;
        let gen_dur = env.generation_time;
        // Now it is safe to call write_gen
        race.telemetry.write_gen(serde_json::json!({
            "run_id": run_id_copy,
            "gen": gen_num,
            "champion_dist_m": champ_dist,
            "best_fitness": best_fitness,
            "avg_dist_m": avg_dist,
            "p50_dist_m": p50_dist,
            "p90_dist_m": p90_dist,
            "crashes": crashes,
            "speed_bleeds": speed_bleeds,
            "survivors": n - crashes,
            "lap_count": laps,
            "track_seed": track_seed,
            "track_val": is_val,
            "env_temp": env_temp,
            "env_wetness": env_wetness,
            "gen_duration_secs": gen_dur,
            "ts_unix": ts,
            "death_reasons": death_counts,
        }));

        // DEBUG: Disabled brain trace export for performance. Re-enable after bottleneck identified.
        /*
        // Export champion brain trace as JSON for vision replay
        if let Some((_, best, _, _)) = sorted.first() {
            if !best.brain_trace.is_empty() {
                let trace_json: Vec<serde_json::Value> = best.brain_trace.iter().map(|snap| {
                    serde_json::json!({
                        "t": snap.t,
                        "speed_kmh": snap.speed_kmh,
                        "wall_prox_l": snap.wall_prox_l,
                        "wall_prox_r": snap.wall_prox_r,
                        "ttc_inv": snap.ttc_inv,
                        "curv_25": snap.curv_25,
                        "raw_throttle": snap.raw_throttle,
                        "raw_steer": snap.raw_steer,
                        "raw_brake": snap.raw_brake,
                        "retina_fov": snap.retina_fov.to_vec(),
                        "sensory_compact": snap.sensory_compact.to_vec(),
                    })
                }).collect();

                let trace_path = format!("logs/champion_gen_{:04}_brain_trace.json", gen_num);
                if let Ok(mut file) = std::fs::File::create(&trace_path) {
                    if let Ok(json_str) = serde_json::to_string(&trace_json) {
                        let _ = std::io::Write::write_all(&mut file, json_str.as_bytes());
                    }
                }
            }
        }
        */

        let world = VoxelWorld::new(next_track.seed, effective_width);
        if hl.enabled {
            race.world_entities = Vec::new();
        } else {
            let (r_mesh, g_mesh, w_mesh) = world.generate_mesh();
            let road_e = commands
                .spawn((
                    Mesh3d(meshes.add(r_mesh)),
                    MeshMaterial3d(materials.add(StandardMaterial {
                        base_color: Color::srgb(0.15, 0.15, 0.18),
                        perceptual_roughness: 0.85,
                        ..default()
                    })),
                    SimulationEntity,
                ))
                .id();
            let grass_e = commands
                .spawn((
                    Mesh3d(meshes.add(g_mesh)),
                    MeshMaterial3d(materials.add(StandardMaterial {
                        base_color: Color::srgb(0.1, 0.3, 0.1),
                        perceptual_roughness: 0.9,
                        ..default()
                    })),
                    SimulationEntity,
                ))
                .id();
            let wall_e = commands
                .spawn((
                    Mesh3d(meshes.add(w_mesh)),
                    MeshMaterial3d(materials.add(StandardMaterial {
                        base_color: Color::srgb(0.7, 0.7, 0.7),
                        metallic: 1.0,
                        perceptual_roughness: 0.1,
                        ..default()
                    })),
                    SimulationEntity,
                ))
                .id();
            race.world_entities = vec![road_e, grass_e, wall_e];
        }
        // Batched spawn: schedules jobs, drain_spawn_queue processes them in blocks per frame.
        race.pending_spawn = Some((0, config.swarm_count));
        world_res.0 = world;

        // Tier 5.4 — moving-obstacle NPCs spawn AFTER the world rebuild
        // (needs valid waypoints_world). The curriculum gate (gen >= NPC_ACTIVATION_GEN)
        // is inside spawn_npcs. Validation gens cleared npcs via reset_fixed.
        env.spawn_npcs(&world_res.0.waypoints_world);
        if !env.npcs.is_empty() {
            println!(
                "🚧 NPCS gen={} count={} waypoints={}",
                race.generation,
                env.npcs.len(),
                world_res.0.waypoints_world.len()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{NICHE_DISTANCE, bonus_de_volta, decide_next_env_level, limiar_de_nicho};

    /// The fixed threshold of 8 erases the diversity we MEASURED to exist.
    ///
    /// Real distances between elites (`telemetry/probe_dist`, after the fingerprint
    /// fix): median 2.4–5.3 and maximum 8–10. With threshold 8, everything falls into the
    /// first cluster — hence `clusters=1` in every generation of every training run.
    #[test]
    fn limiar_fixo_e_grande_demais_para_a_distancia_medida() {
        // sample with the shape measured in generations 4-7: median 2.4, maximum 8
        let medido = [1, 2, 2, 2, 3, 3, 4, 8];
        assert!(
            NICHE_DISTANCE > 2 * 2,
            "8 is more than double the measured median"
        );
        assert!(
            limiar_de_nicho(&medido) < NICHE_DISTANCE,
            "the adaptive one must tighten when the population converges"
        );
    }

    /// The adaptive one TRACKS the scale: it tightens when the population converges and
    /// loosens when it is diverse. That is what an absolute threshold cannot do.
    #[test]
    fn limiar_adaptativo_acompanha_a_escala() {
        let cedo = [4, 5, 5, 5, 6, 6, 7, 10]; // gen 1-3: median ~5
        let tarde = [1, 2, 2, 2, 3, 3, 4, 8]; // gen 4-7: median ~2.4
        assert!(limiar_de_nicho(&cedo) > limiar_de_nicho(&tarde));
    }

    /// Never loosen beyond what the old calibration allowed.
    #[test]
    fn limiar_adaptativo_respeita_o_teto() {
        let dispersos = [30, 40, 50, 60];
        assert_eq!(limiar_de_nicho(&dispersos), NICHE_DISTANCE);
    }

    /// A degenerate pool (all identical) must not explode into N clusters of 1.
    #[test]
    fn pool_identico_continua_um_cluster() {
        assert_eq!(limiar_de_nicho(&[0, 0, 0, 0]), 0);
    }

    /// With no pairs (pool of 0 or 1) it returns the old threshold, without dividing by zero.
    #[test]
    fn pool_vazio_cai_no_limiar_antigo() {
        assert_eq!(limiar_de_nicho(&[]), NICHE_DISTANCE);
    }

    const PISTA: f32 = 1658.0;
    const BONUS: f32 = 50_000.0;

    /// The step: with integer counting, going from 3.01 to 3.99 laps is barely rewarded, and
    /// that is exactly where the population was (best car at 3.72 of a physical ceiling of 4.36).
    ///
    /// Without this measured, "evolution stagnated" looked like the brain's fault. It was
    /// selection that could not see the improvement.
    #[test]
    fn contagem_inteira_e_quase_cega_dentro_do_balde() {
        let a = bonus_de_volta(3.01 * PISTA, PISTA, 3, BONUS, false);
        let b = bonus_de_volta(3.99 * PISTA, PISTA, 3, BONUS, false);
        assert_eq!(a, b, "inside the bucket the bonus is literally identical");

        // Only raw progress separates the two, and it is ~9× smaller than the bonus.
        let dif_prog = (3.99 - 3.01) * PISTA;
        assert!(
            dif_prog / a < 0.02,
            "the remaining signal is {:.1}% — practically noise",
            dif_prog / a * 100.0
        );
    }

    /// The continuous version sees the same improvement with signal proportional to the real gain.
    #[test]
    fn continuo_enxerga_a_melhoria_dentro_do_balde() {
        let a = bonus_de_volta(3.01 * PISTA, PISTA, 3, BONUS, true);
        let b = bonus_de_volta(3.99 * PISTA, PISTA, 3, BONUS, true);
        let ganho = (b - a) / a;
        assert!(ganho > 0.30, "gain of only {:.1}%", ganho * 100.0);
    }

    /// Strict monotonicity: every extra meter must be worth more. It is what guarantees
    /// no plateau is left where selection goes blind.
    #[test]
    fn continuo_e_estritamente_monotono() {
        let mut ant = f32::NEG_INFINITY;
        for i in 0..400 {
            let prog = i as f32 * 15.0;
            let v = bonus_de_volta(prog, PISTA, (prog / PISTA) as usize, BONUS, true);
            assert!(v > ant || prog == 0.0, "plateau at prog={prog}");
            ant = v;
        }
    }

    /// Integer counting has plateaus — the direct proof of what the continuous version fixes.
    #[test]
    fn contagem_inteira_tem_patamares() {
        let patamares = (0..400)
            .filter(|i| {
                let (p1, p2) = (*i as f32 * 15.0, (*i + 1) as f32 * 15.0);
                bonus_de_volta(p1, PISTA, (p1 / PISTA) as usize, BONUS, false)
                    == bonus_de_volta(p2, PISTA, (p2 / PISTA) as usize, BONUS, false)
            })
            .count();
        assert!(
            patamares > 350,
            "expected nearly everything on a plateau, got {patamares}"
        );
    }

    /// Compatibility: without the flag, or before the world writes the length, nothing changes.
    #[test]
    fn sem_flag_ou_sem_comprimento_mantem_o_antigo() {
        assert_eq!(bonus_de_volta(5000.0, PISTA, 3, BONUS, false), 3.0 * BONUS);
        assert_eq!(bonus_de_volta(5000.0, 0.0, 3, BONUS, true), 3.0 * BONUS);
    }

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
        // Champion jumps to level 3 but current=1 → capped at 2
        let mut counter: u32 = 3;
        let (lvl, _) = decide_next_env_level(&mut counter, 2000.0, 900.0, 1);
        assert_eq!(lvl, 2);
        // current=2 already → level 3 unlocked
        let mut counter: u32 = 3;
        let (lvl, _) = decide_next_env_level(&mut counter, 2000.0, 900.0, 2);
        assert_eq!(lvl, 3);
    }
}

#[cfg(test)]
mod testes_curriculo_populacao {
    use super::{FRACTION_TO_FALL, FRACTION_TO_RISE, decide_next_env_level_pop};

    /// The real case that broke the training: one lucky car did 1.93 laps and dragged all 60
    /// to level 3, where grip drops to 0.33 and the required braking distance (130 m) exceeds
    /// the reach of the longest sensor (120 m). Measured: 23.6/60 completed a lap -> 1.2/60.
    #[test]
    fn campeao_de_sorte_nao_arrasta_a_populacao() {
        let mut val = 5;
        // champion did 1.93 laps (3198 m on a 1658 lap), but only 8% of the population closed one
        let (nivel, _) = decide_next_env_level_pop(&mut val, 3198.0, 1658.0, 1, Some(0.08));
        assert_eq!(nivel, 1, "8% of the population does not sustain a level up");
    }

    #[test]
    fn sobe_so_quando_a_populacao_consolida() {
        let mut val = 5;
        let (n, _) =
            decide_next_env_level_pop(&mut val, 0.0, 1658.0, 1, Some(FRACTION_TO_RISE + 0.05));
        assert_eq!(
            n, 2,
            "with the fraction above the threshold, climb one step"
        );
    }

    #[test]
    fn sobe_no_maximo_um_degrau_por_geracao() {
        let mut val = 5;
        let (n, _) = decide_next_env_level_pop(&mut val, 99999.0, 1658.0, 1, Some(0.95));
        assert_eq!(n, 2, "never two steps at once, no matter how good it was");
    }

    #[test]
    fn recua_quando_ninguem_sustenta() {
        let mut val = 5;
        let (n, _) =
            decide_next_env_level_pop(&mut val, 3198.0, 1658.0, 3, Some(FRACTION_TO_FALL / 2.0));
        assert_eq!(n, 2, "with nobody completing a lap, the level must retreat");
    }

    #[test]
    fn ha_histerese_entre_subir_e_descer() {
        // in the middle band it stays put: without this the level would oscillate every
        // generation and the population would never consolidate anything
        let meio = (FRACTION_TO_FALL + FRACTION_TO_RISE) / 2.0;
        let mut val = 5;
        let (n, _) = decide_next_env_level_pop(&mut val, 0.0, 1658.0, 2, Some(meio));
        assert_eq!(n, 2);
    }

    #[test]
    fn nunca_recua_abaixo_de_um_nem_passa_de_tres() {
        let mut val = 5;
        let (n, _) = decide_next_env_level_pop(&mut val, 0.0, 1658.0, 1, Some(0.0));
        assert_eq!(n, 1, "the floor is level 1");
        let mut val = 5;
        let (n, _) = decide_next_env_level_pop(&mut val, 0.0, 1658.0, 3, Some(1.0));
        assert_eq!(n, 3, "the ceiling is level 3");
    }

    #[test]
    fn sem_dado_de_populacao_mantem_o_comportamento_antigo() {
        // the legacy tests call the population-less version; its answer must not change
        let mut val = 5;
        let (n, _) = decide_next_env_level_pop(&mut val, 0.0, 1658.0, 1, None);
        assert_eq!(n, 1);
    }
}

#[cfg(test)]
mod testes_largura_populacao {
    use super::FRACTION_TO_RISE;

    /// Replica of the `pipeline.rs` logic so it can be tested without building the whole world.
    fn passo_largura(estreitamento: &mut f32, frac: f32) -> f32 {
        let alvo = 4.0 * ((frac - FRACTION_TO_RISE) / 0.50).clamp(0.0, 1.0);
        *estreitamento += (alvo - *estreitamento).clamp(-0.25, 0.25);
        estreitamento.clamp(0.0, 4.0)
    }

    #[test]
    fn campeao_de_sorte_nao_estreita_a_pista() {
        // the real case: 8% of the population closed a lap and the track narrowed anyway,
        // because ONE car had done 1.93 laps
        let mut e = 0.0;
        for _ in 0..10 {
            passo_largura(&mut e, 0.08);
        }
        assert_eq!(
            e, 0.0,
            "with 8% of the population, the track must not narrow"
        );
    }

    #[test]
    fn estreita_devagar_quando_a_populacao_sustenta() {
        let mut e = 0.0;
        let d1 = passo_largura(&mut e, 0.80);
        assert!(
            (d1 - 0.25).abs() < 1e-6,
            "at most 0.25 m per generation, got {d1}"
        );
        for _ in 0..40 {
            passo_largura(&mut e, 0.80);
        }
        assert!(
            (e - 4.0).abs() < 1e-3,
            "with 80% consolidated it reaches the 4 m ceiling, got {e}"
        );
    }

    #[test]
    fn alarga_de_volta_quando_a_populacao_desaba() {
        let mut e = 0.0;
        for _ in 0..40 {
            passo_largura(&mut e, 0.80);
        } // narrowed down to 4 m
        assert!(e > 3.9);
        for _ in 0..40 {
            passo_largura(&mut e, 0.02);
        } // population collapsed
        assert_eq!(
            e, 0.0,
            "the track must widen back — before, that was impossible"
        );
    }

    #[test]
    fn nunca_passa_do_teto_de_quatro_metros() {
        let mut e = 0.0;
        for _ in 0..100 {
            passo_largura(&mut e, 1.0);
        }
        assert!(e <= 4.0 + 1e-6);
    }
}
