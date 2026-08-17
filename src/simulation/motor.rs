use super::racer::{Racer, Skidmark};
use crate::config::iara::vision::{RETINA_COLS, RETINA_ROWS, RETINA_TOTAL};
use crate::environment::EnvironmentalState;
use crate::world::{VoxelType, VoxelWorldResource};
use bevy::ecs::system::SystemParam;
use bevy::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

/// Groups the two GPU contexts (retina + perception) into a single SystemParam.
///
/// Bevy 0.16 enforces a 16-parameter limit per system; update_racers was already
/// at the limit with 16. By packing the two GPU resources into this wrapper, we
/// return to 16 parameters without losing mutable access to either of them.
#[derive(SystemParam)]
pub(super) struct GpuContexts<'w> {
    /// Retina GPU context — lives across ticks. Created in `simulation/mod.rs`.
    pub retina: ResMut<'w, crate::gpu::RetinaGpuRes>,
    /// Perception GPU context — shares the retina's device/queue.
    /// Created on demand when lab.gpu_core=true.
    pub percept: ResMut<'w, crate::gpu::perception::PerceptionGpuRes>,
}

// --- Tier 6.5: Time Dilation metrics (process-wide) ---
// Count brain firings vs skips accumulated since boot. Logged 1x/gen
// in pipeline (accessed via getters). Zero overhead (atomic relaxed).
pub(crate) static TIME_DILATION_FIRES: AtomicU64 = AtomicU64::new(0);
pub(crate) static TIME_DILATION_SKIPS: AtomicU64 = AtomicU64::new(0);

pub fn time_dilation_metrics() -> (u64, u64) {
    (
        TIME_DILATION_FIRES.load(AtomicOrdering::Relaxed),
        TIME_DILATION_SKIPS.load(AtomicOrdering::Relaxed),
    )
}

/// Tier 1.3 — Axiom as external prior (Ng-Harada-Russell 1999 potential-based shaping).
///
/// Procedural policy mirroring the 17-18 priming axioms used in
/// `build_priming_weights()` (spawn.rs). Returns an additive bias for each of the
/// 64 outputs of the instinct_layer — it is added to `final_intent` after the
/// brain's forward pass, scaled by `race.current_axiom_strength` (decay 1.0→0.0
/// over 50 gens). NHR theorem: potential-based bias does not change the optimal
/// policy, it only accelerates convergence.
///
/// Output layout (same as the priming block):
///   0 = accel/throttle, 1 = steer (+right / -left), 2 = brake.
///
/// Input channels consulted:
///   32..37  = left periphery (A1 wall-left)         42..53 = center (A4 wall-front)
///   58..63  = right periphery (A2 wall-right)       46..49 = exact center (A4d/A7 brake)
///   64      = flow alignment (A3)                   73     = grass pain (A5)
///   66      = lateral drift (A8)                    78..80 = curvature (A9)
///   81      = lane offset (A10)                     82     = collision urgency (A11)
///   83,84   = optical flow (A12)                    85,86  = wall memory L/R (A13)
///   87      = slip angle (A14)                      88     = proximity sustain (A15)
///   90..93  = dark-center tracker (A16)             94..97 = diagonal whiskers (A17)
///   98..101 = gaze tracker (A18)                  108 = racing-line offset (A21)
/// Builds a 32-dim i8 input slice for the reflex SpecialistBrain by sampling key sensory channels.
/// Conservative subset — enough for a simple safety-aware reflex.
fn slow_input_for_reflex(
    sensory_wave: &[i8; crate::config::iara::brain::SENSORY_WAVE_SIZE],
) -> [i8; 32] {
    let mut out = [0_i8; 32];
    for (dst, &src) in out.iter_mut().zip(sensory_wave.iter().take(32)) {
        *dst = src;
    }
    out
}

fn axiom_policy(
    inputs: &[i8; crate::config::iara::brain::SENSORY_WAVE_SIZE],
    lab: &crate::lab::LabParams,
) -> [f32; 96] {
    let mut bias = [0.0f32; 96];
    let WALL_T: i8 = lab.wall_t_threshold; // ~0.31 normalized — wall close but not touching

    // A1: left wall (periph bins 8..11) → steer right (+)
    let wall_l: i32 = (8..12).map(|i| inputs[i] as i32).sum();
    if wall_l > WALL_T as i32 * 2 {
        bias[1] += lab.axiom_wall_avoidance;
    }
    // A2: right wall (periph bins 12..15) → steer left (-)
    let wall_r: i32 = (12..16).map(|i| inputs[i] as i32).sum();
    if wall_r > WALL_T as i32 * 2 {
        bias[1] -= lab.axiom_wall_avoidance;
    }
    // A3: alignment with flow (64) → accelerate
    bias[0] += (inputs[64] as f32 / 127.0) * 0.15;
    // A4: wall at center (periph bins 10..13 = center of the periphery) → cut throttle
    let wall_c: i32 = (10..14).map(|i| inputs[i] as i32).sum();
    if wall_c > WALL_T as i32 * 2 {
        bias[0] -= 0.25;
    }
    // A4d/A7: fovea center (bins 3..5) → strong brake (wall straight ahead)
    let wall_front: i32 = (3..6).map(|i| inputs[i] as i32).sum();
    if wall_front > WALL_T as i32 * 2 {
        bias[2] += 0.30;
    }
    // A5: grass pain (73) → reduce throttle
    if inputs[73] > WALL_T {
        bias[0] -= 0.15;
    }
    // A8: lateral drift (66) → counter-steer
    bias[1] -= (inputs[66] as f32 / 127.0) * 0.10;
    // A9: anticipated curvature (78/79/80). FIX v3: sign was inverted.
    // cross(forward, flow_ahead) xz-plane: + = curve to the RIGHT. Steer + = right.
    // v4: boost coeffs 0.08/0.18/0.06 → 0.20/0.40/0.25 to break the 1st-corner plateau
    // (330m). Lookahead 10/25/50m creates predictive steering instead of reactive.
    bias[1] += (inputs[78] as f32 / 127.0) * 0.20;
    bias[1] += (inputs[79] as f32 / 127.0) * 0.32; // v5: 0.40→0.32 (less overshoot)
    bias[1] += (inputs[80] as f32 / 127.0) * 0.25;
    // v5: pre-brake only on a SEVERE curve (threshold) instead of always proportional.
    // Tested 0.20 → regressed (305m swing, 15.4 bleeds). 0.10 is sweet spot.
    let curv_mag = ((inputs[79] as f32).abs() + (inputs[80] as f32).abs()) / 254.0;
    bias[0] -= curv_mag * 0.10; // Stable at 0.10; aggressive cut causes volatility
    // A10: lane offset (81) — positive value = more space to the right ⇒ I'm on the LEFT ⇒
    // I need to turn RIGHT = positive steer. Coef boosted 0.08→0.22 (v2) to become
    // the primary centering signal on narrow tracks.
    bias[1] += (inputs[81] as f32 / 127.0) * 0.22;
    // A11: collision urgency (82) → brake + cut engine
    bias[2] += (inputs[82] as f32 / 127.0) * 0.20;
    bias[0] -= (inputs[82] as f32 / 127.0) * 0.12;
    // A12: optical flow L/R (83/84) → anti-approach steer
    bias[1] += (inputs[83] as f32 / 127.0) * 0.08;
    bias[1] -= (inputs[84] as f32 / 127.0) * 0.08;
    // A13: wall memory L/R (85/86)
    bias[1] += (inputs[85] as f32 / 127.0) * 0.12;
    bias[1] -= (inputs[86] as f32 / 127.0) * 0.12;
    // A14: slip angle (87) → counter-steer. v5: 0.08→0.20 (stabilize corner exit).
    bias[1] += (inputs[87] as f32 / 127.0) * 0.20;
    // A15: sustained proximity (88) → cut throttle
    if inputs[88] > WALL_T {
        bias[0] -= 0.12;
    }
    // A16: dark-center tracker (90..93) → follow the dark (track center)
    let dark_center: i32 = (90..94).map(|i| inputs[i] as i32).sum();
    bias[1] += (dark_center as f32 / (127.0 * 4.0)) * lab.axiom_dark_center;
    // A17: diagonal whiskers (94..97) — wall close on a diagonal
    //   94 = FL 30°, 95 = FR 30°, 96 = FL 60°, 97 = FR 60°
    if inputs[94] > WALL_T {
        bias[1] += lab.axiom_whisker;
        bias[2] += lab.axiom_whisker * 0.67;
        bias[0] -= lab.axiom_whisker * 0.53;
    }
    if inputs[96] > WALL_T {
        bias[1] += lab.axiom_whisker * 0.80;
        bias[2] += lab.axiom_whisker * 0.53;
    }
    if inputs[95] > WALL_T {
        bias[1] -= lab.axiom_whisker;
        bias[2] += lab.axiom_whisker * 0.67;
        bias[0] -= lab.axiom_whisker * 0.53;
    }
    if inputs[97] > WALL_T {
        bias[1] -= lab.axiom_whisker * 0.80;
        bias[2] += lab.axiom_whisker * 0.53;
    }
    // A18: gaze tracker — DISABLED in axiom_policy.
    // Channels 98-101 do NOT contain gaze: ch98=motion_norm (always ≥0 → bias RIGHT),
    // ch101-102=curvature lookahead. This caused a systematic rightward turn.
    // The dark-center A16 (ch 90-93) already plays the track-following role.
    // let gaze: i32 = (98..102).map(|i| inputs[i] as i32).sum();
    // bias[1] += (gaze as f32 / (127.0 * 4.0)) * 0.25;

    // A19 REVERTED v6b: tested over 200 gens — champion peak 621m vs v5 736m.
    // Anti-stall forced throttle in post-curve states that should not accelerate.
    // Decay window 20→40 and rate 0.95→0.98 KEPT (pipeline.rs).

    // A20: drift commit — high slip + high curvature ahead + minimum speed →
    // commit with throttle instead of chickening out. Channels: 87 slip, |79|+|80|/254 curv, 65 speed.
    let slip_abs = (inputs[87] as f32 / 127.0).abs();
    let curv_mag_a20 = ((inputs[79] as f32).abs() + (inputs[80] as f32).abs()) / 254.0;
    let speed_norm = (inputs[65] as f32 / 127.0).abs();
    if slip_abs > 0.5 && curv_mag_a20 > 0.4 && speed_norm > 0.35 {
        bias[0] += 0.18;
        bias[2] -= 0.15;
    }

    // A21: racing line follow. Channel 108 = signed lateral offset (positive =
    // car to the LEFT of the ideal line → needs to steer RIGHT to close the gap).
    // Convention matches A10 (channel 81): positive → positive bias[1] (steer +right).
    // Axiom decay (race.current_axiom_strength) is applied at the call site, so
    // here we only seed the prior. v13: 0.35→0.20 (HOLE=5181 in v12, overcorrection
    // toward the apex pulled the car outward in a tight curve).
    // v14: 0.20→0.26 for narrow tracks — precision on centerline critical
    const CH_LINE: usize = 108;
    let line_err = inputs[CH_LINE] as f32 / 127.0;
    bias[1] += line_err * 0.26;

    // A9 Extended: 7-distance lookahead steering guidance (20m to 350m).
    // Weight decreases with distance: immediate corners require urgent correction.
    bias[1] += (inputs[20] as f32 / 127.0) * 0.35; // 20m nearest (urgency max)
    bias[1] += (inputs[79] as f32 / 127.0) * 0.18; // 70m (reorg from c100 slot)
    bias[1] += (inputs[101] as f32 / 127.0) * 0.12; // 260m far horizon
    bias[1] += (inputs[102] as f32 / 127.0) * 0.06; // 350m ultra-far (stabilize)

    // A-TTC: coefs drastically reduced for debug
    let ttc_norm = inputs[105] as f32 / 127.0;
    if ttc_norm > 0.85 {
        // threshold raised — only in CRITICAL danger
        bias[2] += ttc_norm * 0.15; // very weak brake
        bias[0] -= ttc_norm * 0.05; // very weak throttle cut
    }

    // A-Depth: disabled — always 0
    let _depth_near_px = (inputs[52] as f32 + inputs[53] as f32) / (2.0 * 127.0);
    // if depth_near_px > 0.3 { bias[0] -= depth_near_px * 0.15; }

    // A-MotionRow0: disabled for stability (motion channels adding noise under SPEED_BLEED)
    let _near_motion = inputs[21] as f32 / 127.0;
    // if _near_motion > 0.6 {
    //     bias[2] += _near_motion * 0.06;  // brake hint
    //     bias[0] -= _near_motion * 0.03;  // throttle cut
    // }

    // A-HeadingAlign: correct heading when misaligned with road flow.
    // Detects spinning/drifting via yaw rate (ch 16) + poor alignment (inverse of ch 64).
    // If yaw is high but flow alignment is low, apply steering damping.
    // If flow alignment is poor but yaw is neutral, apply bias toward road flow.
    let yaw_rate_norm = (inputs[16] as f32).abs() / 127.0;
    let flow_align_norm = (inputs[64] as f32 / 127.0).max(0.0); // 0=misaligned, 1=aligned
    if flow_align_norm < 0.5 && yaw_rate_norm > 0.4 {
        // Spinning + poor alignment: dampen steering override
        bias[1] *= 0.70;
    }
    if flow_align_norm < 0.6 {
        // Generally misaligned: use road flow hint (ch 83 left, ch 84 right optical flow)
        let flow_l = (inputs[83] as f32 / 127.0).max(0.0);
        let flow_r = (inputs[84] as f32 / 127.0).max(0.0);
        if flow_l > flow_r {
            bias[1] += (flow_l - flow_r) * 0.15; // road curving left, steer left
        } else if flow_r > flow_l {
            bias[1] -= (flow_r - flow_l) * 0.15; // road curving right, steer right
        }
    }

    bias
}

pub(super) fn update_racers(
    mut commands: Commands,
    mut query: Query<(
        Entity,
        &mut Racer,
        &mut Transform,
        &mut MeshMaterial3d<StandardMaterial>,
        Option<&mut crate::physics::EngineComponent>,
        Option<&mut crate::physics::PhysicsBody>,
    )>,
    world_res: Res<VoxelWorldResource>,
    env: Res<EnvironmentalState>,
    dev: Res<crate::evolution::DevSettings>,
    mut race: ResMut<crate::evolution::RaceSettings>,
    time: Res<Time>,
    lab: Res<crate::lab::LabParams>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut skidmark_cache: Local<Option<(Handle<Mesh>, Handle<StandardMaterial>)>>,
    mut retina_tick_counter: Local<u32>,
    // Cache of previous lateral-offset-from-racing-line per entity so the
    // derivative channel (109) can be computed without extending Racer
    // (spawn.rs is frozen by task constraints).
    mut prev_line_offset: Local<HashMap<Entity, f32>>,
    mut gizmos: Gizmos,
    mut perf_accum: Local<(u32, f64, f64, f64, f64)>,
    mut gpu: GpuContexts<'_>,
) {
    let t_frame_start = std::time::Instant::now();
    let dt = time.delta_secs();
    let vdt = dt;
    let world = &world_res.0;

    // --- REPLAY WRITER: opens the file at the start of each generation ---
    // Creates telemetry/gen_<N>/motor.replay the first time update_racers runs per generation.
    // The flush/close happens in evolution_pipeline when race.generation increments.
    if race.replay_writer.is_none() {
        let dir = std::path::PathBuf::from(format!("telemetry/gen_{}", race.generation));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("motor.replay");
        match crate::replay::ReplayWriter::create(
            &path,
            crate::replay::SOURCE_MOTOR,
            5, // top-5 racers
            64.0,
            race.seed,
        ) {
            Ok(w) => race.replay_writer = Some(w),
            Err(e) => eprintln!("[replay] Failed to open {:?}: {}", path, e),
        }
    }

    // --- TOGGLE GIZMOS (KeyG) ---
    // Wrap the entire visual gizmos block (flag, paths, pillars) in dev.show_gizmos.
    // Allows clearing the UI for "game mode" when the player just wants to drive.
    if dev.show_gizmos {
        // --- RENDER MARKER-ZERO (THE GLOBAL CHAMPION'S FLAG) ---
        if let Some(flag_pos) = race.global_best_position {
            // 15-meter pole staked into the ground
            gizmos.line(
                flag_pos,
                flag_pos + Vec3::Y * 15.0,
                Color::srgb(1.0, 0.84, 0.0),
            );
            // Golden globe/flag at the top
            gizmos.sphere(flag_pos + Vec3::Y * 15.0, 2.0, Color::srgb(1.0, 0.84, 0.0));
            // Ring on the ground for visibility from afar
            gizmos.circle(
                Isometry3d::new(flag_pos, Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
                5.0,
                Color::srgba(1.0, 0.84, 0.0, 0.6),
            );
        }

        // --- LAST CHAMPION'S PATH (magenta, persists until surpassed) ---
        if race.champion_path.len() >= 2 {
            let path = &race.champion_path;
            let y_off = Vec3::Y * 1.5;
            for w in path.windows(2) {
                gizmos.line(w[0] + y_off, w[1] + y_off, Color::srgba(1.0, 0.2, 0.9, 0.9));
            }
            if let Some(&start) = path.first() {
                gizmos.sphere(start + Vec3::Y * 2.0, 1.5, Color::srgb(1.0, 0.84, 0.0)); // 👑 gold
                gizmos.line(start, start + Vec3::Y * 12.0, Color::srgb(1.0, 0.84, 0.0));
            }
            if let Some(&end) = path.last() {
                gizmos.sphere(end + Vec3::Y * 2.0, 1.5, Color::srgb(1.0, 0.1, 0.1)); // 🔴 end
                gizmos.line(end, end + Vec3::Y * 12.0, Color::srgb(1.0, 0.1, 0.1));
            }
        }

        // --- BEST OF THE LAST GENERATION'S PATH ---
        // 🔵 CYAN = grid-spawn (eligible for coronation — if it beats magenta, it becomes champion)
        // 🟠 ORANGE = distributed spawn (does NOT crown — just a specialist of one stretch)
        // So if orange passes magenta, you know it's fake progress.
        if race.last_gen_path.len() >= 2 {
            let path = &race.last_gen_path;
            let y_off = Vec3::Y * 0.6;
            let (line_col, sphere_col) = if race.last_gen_was_grid {
                (
                    Color::srgba(0.0, 0.9, 1.0, 0.75),
                    Color::srgb(0.0, 0.9, 1.0),
                )
            } else {
                (
                    Color::srgba(1.0, 0.5, 0.0, 0.75),
                    Color::srgb(1.0, 0.5, 0.0),
                )
            };
            for w in path.windows(2) {
                gizmos.line(w[0] + y_off, w[1] + y_off, line_col);
            }
            if let Some(&start) = path.first() {
                gizmos.sphere(start + Vec3::Y * 0.8, 1.0, sphere_col);
            }
            if let Some(&end) = path.last() {
                gizmos.sphere(end + Vec3::Y * 0.8, 1.0, sphere_col);
            }
        }

        // --- RENDER TRACK WAYPOINTS/CHECKPOINTS ---
        // Decimate: the procedural track has 100+ waypoints (Catmull-Rom). Showing all
        // = 100+ pillars crossing through walls (gizmos drawn on top). Show only 1 in
        // every 8 (density of the original anchors) and height reduced to 3m.
        let wps = &world.waypoints_world;
        let stride = (wps.len() / 18).max(1); // ~14-18 pillars
        for i in (0..wps.len()).step_by(stride) {
            let wp = Vec3::new(wps[i].x, 0.5, wps[i].y);

            // Checkpoint pillar (3m tall, thin).
            let pillar_color = if i == 0 {
                Color::srgb(0.0, 1.0, 0.3) // Green = Start/Finish
            } else {
                Color::srgba(0.0, 0.7, 1.0, 0.5) // Subtle cyan
            };
            gizmos.line(wp, wp + Vec3::Y * 3.0, pillar_color);
            gizmos.sphere(wp + Vec3::Y * 3.5, 0.8, pillar_color);
        }
    } // end dev.show_gizmos

    // --- PRE-COMPUTATION (First read-only pass over the same query) ---
    let mut best_score = -10000.0;
    let mut current_best = race.selected_racer.unwrap_or(Entity::PLACEHOLDER);
    let mut racer_positions: Vec<(Entity, Vec3, Vec3)> = Vec::new();

    for (e, r, t, _, _, _) in query.iter() {
        // Update the Global Champion's Flag if this pilot beat the all-time mark
        // 0.05 added to avoid visual micro-updates
        if r.track_progress > race.global_best_distance + 0.05 {
            race.global_best_distance = r.track_progress;
            race.global_best_position = Some(t.translation);
        }

        let score = if r.track_progress < 15.0 {
            r.track_progress * 10.0 - 10000.0
        } else {
            let avg_speed = r.track_progress / r.engine.current_time.max(0.1);
            let lap_bonus = r.total_laps as f32 * 10000.0;
            lap_bonus + (r.track_progress * 100.0) + (avg_speed * 50.0) + (r.hp * 2.0)
        };
        if score > best_score {
            best_score = score;
            current_best = e;
        }
        if !r.is_dead {
            racer_positions.push((e, t.translation, t.right().as_vec3()));
        }
    }

    // --- PARALLEL: PRECOMPUTE RETINAS OF ALL LIVING CARS ---
    // Retina = 64 DDA raycasts × 2000 cars = 128k rays/tick. At time_scale=10 that's
    // ~10 substeps/real frame → 1.28M rays/frame. Heavy even parallelized via rayon.
    //
    // Optimization: cache + skip. Recompute retina only every RETINA_SKIP ticks.
    // RETINA_SKIP auto-computed in config based on RETINA_ROWS×COLS to avoid CPU bottleneck.
    // A car at 60 m/s moves ~3m between recomputes — tolerable for perception (walls 3m).
    let retina_skip = crate::config::iara::vision::RETINA_SKIP;
    *retina_tick_counter = retina_tick_counter.wrapping_add(1);
    let recompute_retinas = (*retina_tick_counter % retina_skip) == 0;

    let t_retina = std::time::Instant::now();

    // --- RETINA + PERCEPTION CHAINED (gpu_core=true) or retina alone (gpu_core=false) ---
    //
    // Before (problem): two independent encoders:
    //   1. retina: encoder + submit + poll(Wait)  → ~5 ms/tick
    //   2. perception: encoder + submit + poll(Wait) → +0.76 ms/tick unnecessary
    //
    // After (solution): ONE single encoder for retina → perception:
    //   1. dispatch_into(enc, ...)          ← records retina into the external encoder
    //   2. bind_and_dispatch(enc, ...)      ← records perception into the SAME encoder
    //   3. queue.submit(enc.finish())       ← ONE submit
    //   4. device.poll(Wait)               ← ONE block
    //   5. read_legacy_after_submit(n)     ← reads retina (already in readback via copy in the encoder)
    //   6. read_after_submit(n, bytes)     ← reads perception (likewise)
    //
    // Estimated saving: ~0.76 ms/tick eliminated (encoder overhead + perception's extra poll).
    // The retina poll is mandatory anyway — we add no latency.
    //
    // Fallback: if gpu_core=false or GPU unavailable, falls back to CPU without panic.

    let forcar_cpu = std::env::var("IARA_RETINA_CPU").is_ok();

    // Collect living beings and their poses (shared between GPU and CPU paths).
    let retina_inputs: Vec<(Entity, Vec3, Vec3, Vec3)> = if recompute_retinas {
        query
            .iter()
            .filter(|(_, r, _, _, _, _)| !r.is_dead)
            .map(|(e, _, t, _, _, _)| (e, t.translation, t.right().as_vec3(), t.back().as_vec3()))
            .collect()
    } else {
        Vec::new()
    };

    // percept_waves: filled by the chained path (gpu_core=true) or stays None.
    let mut percept_waves: Option<
        HashMap<Entity, [i8; crate::config::iara::brain::SENSORY_WAVE_SIZE]>,
    > = None;

    let retinas: HashMap<Entity, [i8; RETINA_TOTAL]> = if recompute_retinas {
        // ── Chained GPU path (retina → perception, ONE submit) ──────────────────
        let resultado_gpu: Option<(
            Vec<[i8; RETINA_TOTAL]>,
            Option<Vec<[i8; crate::config::iara::brain::SENSORY_WAVE_SIZE]>>,
        )> = if forcar_cpu {
            None
        } else if let Some(retina_ctx) = gpu.retina.get_or_create(world, retina_inputs.len()) {
            let poses: Vec<crate::gpu::Pose> = retina_inputs
                .iter()
                .map(|&(_, pos, fwd, right)| crate::gpu::Pose {
                    // The caller's `+0.4` is PRESERVED on purpose: together with the `+0.4`
                    // inside the kernel it yields the same eye the CPU always used (1.3 m with
                    // the car at y=0.5). Documented in visao.rs — fixing it is a separate experiment.
                    pos: pos + Vec3::Y * 0.4,
                    forward: Vec2::new(fwd.x, fwd.z),
                    right: Vec2::new(right.x, right.z),
                    gaze_yaw: 0.0,
                })
                .collect();

            if lab.gpu_core {
                // ── Chaining: retina → perception in a single encoder ─────────────
                // Builds perception poses (phase + forward/right). Same order as
                // retina_inputs, so the being's index matches the index in buf_output.
                let poses_percept: Vec<crate::gpu::perception::PosePerception> = query
                    .iter()
                    .filter(|(_, r, _, _, _, _)| !r.is_dead)
                    .map(|(_, r, t, _, _, _)| {
                        let fwd = t.right().as_vec3();
                        let rgt = t.back().as_vec3();
                        crate::gpu::perception::pose_from_transform(
                            t.translation,
                            Vec2::new(fwd.x, fwd.z),
                            Vec2::new(rgt.x, rgt.z),
                            r.phase,
                        )
                    })
                    .collect();

                let sig = crate::gpu::world_signature(world);
                // get_or_create borrows the retina's device/queue (no second VkDevice).
                // Perception is only available if the GPU retina is present.
                let n_alive = retina_inputs.len();
                let percept_ok = gpu
                    .percept
                    .get_or_create(retina_ctx.device(), retina_ctx.queue(), world, sig, n_alive)
                    .is_some();

                if percept_ok {
                    // Creates the encoder with the retina's device (dispatch_into uses self.device/queue
                    // internally — no extra parameters, no borrow conflict).
                    let mut enc = retina_ctx
                        .device()
                        .create_command_encoder(&Default::default());

                    // dispatch retina → enc (no submit); uses self.device/queue internally.
                    let n_dispatched = retina_ctx.dispatch_into(&mut enc, &poses);

                    // dispatch perception → enc (no submit); perception uses the retina's buf_output
                    // already in the encoder as its source — wgpu's implicit pipeline barrier guarantees
                    // the previous compute pass (retina) finishes before this one starts.
                    let pctx = gpu.percept.ctx.as_ref().unwrap();
                    let bytes_percept = pctx.bind_and_dispatch(
                        &mut enc,
                        retina_ctx.device(),
                        retina_ctx.queue(),
                        &poses_percept,
                        retina_ctx.output_buffer(),
                    );

                    // ONE submit for both dispatches. ONE poll — eliminates the
                    // extra poll that cost ~0.76 ms/tick in the previous perception.
                    retina_ctx.queue().submit(Some(enc.finish()));
                    retina_ctx.device().poll(wgpu::Maintain::Wait);

                    // Reads retina (buf_readback was already copied in the encoder by dispatch_into).
                    let retina_legada = retina_ctx.read_legacy_after_submit(n_dispatched);

                    // Reads perception (buf_readback was already copied in the encoder by bind_and_dispatch).
                    let waves = pctx.read_after_submit(retina_ctx.device(), n_alive, bytes_percept);

                    if retina_legada.len() == n_alive && waves.len() == n_alive {
                        Some((retina_legada, Some(waves)))
                    } else {
                        // Size divergence: deaths between iters — fallback without perception.
                        Some((retina_legada, None))
                    }
                } else {
                    // GPU perception unavailable — retina alone with its own submit.
                    let saida = retina_ctx.render_legacy(&poses);
                    if saida.len() == retina_inputs.len() {
                        let conv: Vec<[i8; RETINA_TOTAL]> = saida
                            .into_iter()
                            .map(|v| {
                                let mut r = [-128i8; RETINA_TOTAL];
                                r[..v.len().min(RETINA_TOTAL)]
                                    .copy_from_slice(&v[..v.len().min(RETINA_TOTAL)]);
                                r
                            })
                            .collect();
                        Some((conv, None))
                    } else {
                        None
                    }
                }
            } else {
                // gpu_core=false — normal GPU retina (own encoder), no GPU perception.
                // RETINA ON THE GPU. Measured: 57% of simulation time was here. 31.7× vs CPU.
                let saida = retina_ctx.render_legacy(&poses);
                if saida.len() == retina_inputs.len() {
                    let conv: Vec<[i8; RETINA_TOTAL]> = saida
                        .into_iter()
                        .map(|v| {
                            let mut r = [-128i8; RETINA_TOTAL];
                            r[..v.len().min(RETINA_TOTAL)]
                                .copy_from_slice(&v[..v.len().min(RETINA_TOTAL)]);
                            r
                        })
                        .collect();
                    Some((conv, None))
                } else {
                    None
                }
            }
        } else {
            None
        };

        match resultado_gpu {
            Some((retina_legada, waves)) => {
                // Associates percept_waves with entities (same order as retina_inputs).
                if let Some(w) = waves {
                    let entities_alive: Vec<Entity> =
                        retina_inputs.iter().map(|&(e, _, _, _)| e).collect();
                    if w.len() == entities_alive.len() {
                        percept_waves =
                            Some(entities_alive.iter().copied().zip(w.into_iter()).collect());
                    }
                }
                // Retinas mapped by entity.
                retina_inputs
                    .iter()
                    .zip(retina_legada.into_iter())
                    .map(|(&(e, _, _, _), r)| (e, r))
                    .collect()
            }
            None => {
                // CPU fallback — parallel retina via rayon.
                // `IARA_RETINA_CPU=1` forces this path for A/B comparison.
                retina_inputs
                    .par_iter()
                    .map(|&(e, pos, fwd, right)| {
                        (
                            e,
                            world.render_binocular_retina(pos + Vec3::Y * 0.4, fwd, right, 0.0),
                        )
                    })
                    .collect()
            }
        }
    } else {
        // Reuse: takes visual_cortex (retina from the last recompute) of each car.
        query
            .iter()
            .filter(|(_, r, _, _, _, _)| !r.is_dead)
            .map(|(e, r, _, _, _, _)| (e, r.visual_cortex))
            .collect()
    };

    // Accumulated time: chained retina + perception add into the same slot.
    perf_accum.1 += t_retina.elapsed().as_secs_f64();
    // percept_waves was already filled above when gpu_core=true and chaining worked.
    // The perf_accum.2 slot (perception/inject) keeps being accounted in the per-being loop below.

    // --- PRECOMPUTE RANK + SPATIAL HASH (eliminates O(N²) from the main loop) ---
    // Before: each car sorted everyone (O(N log N) × N) + comp_stress/draft scan O(N) × N
    //      = ~4M ops/tick for 2000 cars. Now: 1 sort O(N log N) + O(1) lookup.
    //   Spatial hash 20m cells: neighbor query returns only 9 cells (avg ~8 cars each).
    let mut rank_sorted: Vec<(Entity, f32)> =
        racer_positions.iter().map(|&(e, p, _)| (e, p.x)).collect();
    rank_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let rank_by_entity: HashMap<Entity, usize> = rank_sorted
        .iter()
        .enumerate()
        .map(|(i, &(e, _))| (e, i))
        .collect();

    const SPATIAL_CELL: f32 = 20.0;
    let mut spatial: HashMap<(i32, i32), Vec<(Entity, Vec3, Vec3)>> = HashMap::new();
    for &entry in &racer_positions {
        let cx = (entry.1.x / SPATIAL_CELL).floor() as i32;
        let cz = (entry.1.z / SPATIAL_CELL).floor() as i32;
        spatial.entry((cx, cz)).or_default().push(entry);
    }

    for (entity, mut racer, mut transform, mat_handle, mut opt_engine, mut opt_body) in
        query.iter_mut()
    {
        if racer.is_dead {
            if let Some(engine_comp) = opt_engine.as_mut() {
                engine_comp.throttle = 0.0;
            }
            if let Some(body) = opt_body.as_mut() {
                body.velocity *= 0.95_f32.powf(dt * 60.0); // Strong friction: stops in ~2s
                body.angular_velocity *= 0.9_f32.powf(dt * 60.0);
                if body.velocity.length() < 0.5 {
                    body.velocity = Vec3::ZERO;
                }
            }
            continue;
        }

        // Tick-based grace: immune to time_scale. ~600 frames = 10 real seconds @ 60Hz.
        racer.grace_ticks = racer.grace_ticks.saturating_add(1);
        // race_time in VIRTUAL SECONDS: at 8x time_scale, 1 real second = 8 virtual seconds here.
        // It is the clock the USER sees on screen. Grace punishments use this.
        racer.race_time += vdt;

        let pos = transform.translation;
        let flow = world.get_flow_at(pos.x, pos.z);
        let track_phase = flow.y.atan2(flow.x);

        // --- 1. DYNAMIC VISION (WITH DAY/NIGHT CYCLE) ---
        // When gpu_core=true and percept_waves is available, starts with the channels
        // computed by the GPU (WGSL perception). The CPU then overwrites ON TOP
        // the temporal/inter-being channels documented in gpu/perception.rs:CANAIS_NA_CPU.
        // When gpu_core=false or GPU unavailable, starts at zeros as before.
        let mut sensory_wave: [i8; crate::config::iara::brain::SENSORY_WAVE_SIZE] =
            if let Some(ref pw) = percept_waves {
                pw.get(&entity)
                    .copied()
                    .unwrap_or([0i8; crate::config::iara::brain::SENSORY_WAVE_SIZE])
            } else {
                [0i8; crate::config::iara::brain::SENSORY_WAVE_SIZE]
            };

        // --- STEREOSCOPIC BIOLOGICAL VISION (Double Fovea) ---
        let forward = transform.right().as_vec3();
        let right = transform.back().as_vec3();

        // Retina FIXED forward (saccade disabled). Data precomputed IN PARALLEL above
        // via rayon par_iter — this serial loop only consumes the per-entity result.
        let _ = (forward, right);
        let retina_binocular = *retinas.get(&entity).unwrap_or(&[-128i8; RETINA_TOTAL]);
        // POOLED RETINA — Volumetric: max-pool across ALL rows for each bin.
        // If there is a wall at ANY depth layer, the brain knows.
        // Retina HD v2: FOV_BINS adapted dynamically (8 bins, continuous with no gaps)
        let c = RETINA_COLS;
        let cols_per_fov_bin = c / 8;
        let fov_bins: [(usize, usize); 8] = [
            (0, cols_per_fov_bin),
            (cols_per_fov_bin, 2 * cols_per_fov_bin),
            (2 * cols_per_fov_bin, 3 * cols_per_fov_bin),
            (3 * cols_per_fov_bin, 4 * cols_per_fov_bin),
            (4 * cols_per_fov_bin, 5 * cols_per_fov_bin),
            (5 * cols_per_fov_bin, 6 * cols_per_fov_bin),
            (6 * cols_per_fov_bin, 7 * cols_per_fov_bin),
            (7 * cols_per_fov_bin, c), // Last bin reaches the end to cover the remainder
        ];
        // === ZONE-BASED POOLING (ch 0-15): preserve depth + curve shape ===
        // FAR zone (ch 0-7): pool rows 0-4 (~80m ahead, distant curves)
        for (bin, &(lo, hi)) in fov_bins.iter().enumerate() {
            let mut fov_max: i8 = -128;
            for row in 0..5 {
                // FAR rows only (rows 0-4)
                for p in lo..hi {
                    let f = retina_binocular[row * RETINA_COLS + p];
                    if f > fov_max {
                        fov_max = f;
                    }
                }
            }
            sensory_wave[bin] = fov_max;
        }
        // NEAR zone (ch 8-15): pool rows 10-13 (~20m ahead, obstacles to avoid)
        for bin in 0..8 {
            let cols_per_bin = RETINA_COLS / 8;
            let start = bin * cols_per_bin;
            let mut per_max: i8 = -128;
            for row in 10..RETINA_ROWS {
                // NEAR rows only (rows 10-13)
                for k in 0..cols_per_bin {
                    let p = retina_binocular[row * RETINA_COLS + start + k];
                    if p > per_max {
                        per_max = p;
                    }
                }
            }
            sensory_wave[8 + bin] = per_max;
        }
        // --- HD VISUAL CHANNELS (128..191) — High resolution for the expanded brain ---
        // ch 128..143: HD Fovea (16 bins × 4 cols, volumetric max-pool)
        let cols_per_bin_hd = RETINA_COLS / 16;
        for bin in 0..16usize {
            let lo = bin * cols_per_bin_hd;
            let hi = lo + cols_per_bin_hd;
            let mut fov_max: i8 = -128;
            for row in 0..RETINA_ROWS {
                for p in lo..hi {
                    let f = retina_binocular[row * RETINA_COLS + p];
                    if f > fov_max {
                        fov_max = f;
                    }
                }
            }
            sensory_wave[128 + bin] = fov_max;
        }
        // ch 144..159: HD Periphery (16 bins × 4 cols, uniform)
        let cols_per_bin_hd = RETINA_COLS / 16;
        for bin in 0..16usize {
            let lo = bin * cols_per_bin_hd;
            let hi = lo + cols_per_bin_hd;
            let mut per_max: i8 = -128;
            for row in 0..RETINA_ROWS {
                for p in lo..hi {
                    let f = retina_binocular[row * RETINA_COLS + p];
                    if f > per_max {
                        per_max = f;
                    }
                }
            }
            sensory_wave[144 + bin] = per_max;
        }
        // ch 160..191: Raw horizon pixels. Subsample to 32 cols.
        let horizon_row = RETINA_ROWS / 2;
        let step = RETINA_COLS / 32;
        for i in 0..32usize {
            sensory_wave[160 + i] = retina_binocular[horizon_row * RETINA_COLS + i * step];
        }
        // === 64 SEMANTIC SENSORY CHANNELS (ch 16-63, no raycasts) ===
        // All queries via world.get_voxel_at(), get_height_at(), get_flow_at() — O(1) ops.
        //
        // Layout:
        // ch 16-23:  wall_distance_fov (8 YAW angles left→right, 60m range)
        // ch 24-31:  wall_width_fov (obstacle width per YAW angle)
        // ch 32-39:  wall_curvature_fov (flow curvature per YAW angle)
        // ch 40-47:  road_ahead NEAR (5-40m, reactive steering)
        // ch 48-55:  road_ahead FAR (45-150m, strategic — closes old 40-80m gap)
        // ch 56-63:  ground_material + agent_state

        let forward_2d = Vec2::new(forward.x, forward.z);
        let right_2d = Vec2::new(right.x, right.z);

        // Helper: YAW-swept direction for a given angle index (0..8).
        // Sweeps ±50° (covering inner 100° of 120° FOV), left→right.
        // FIX: the old code varied PITCH instead of YAW — all 8 sensors
        // looked in the same lateral direction. Now it sweeps horizontally.
        let fov_half_rad: f32 = 50.0_f32.to_radians(); // ±50° lateral sweep
        let yaw_dir = |angle_idx: usize| -> Vec2 {
            let frac = (angle_idx as f32 - 3.5) / 3.5; // -1.0 (left) → +1.0 (right)
            let yaw = frac * fov_half_rad;
            let dx = forward_2d.x * yaw.cos() + right_2d.x * yaw.sin();
            let dz = forward_2d.y * yaw.cos() + right_2d.y * yaw.sin();
            Vec2::new(dx, dz).normalize_or_zero()
        };

        // --- ch 16-23: wall_distance_fov (8 YAW angles, 60m range) ---
        // Sweep LEFT→RIGHT detecting wall proximity at each lateral angle.
        // Range 60m (was 24m) — at 20 m/s gives 3s reaction time for curves.
        const WALL_SAMPLES: [f32; 8] = [3.0, 8.0, 15.0, 25.0, 35.0, 45.0, 52.0, 60.0];
        const WALL_MAX_DIST: f32 = 60.0;
        for angle_idx in 0..8 {
            let dir = yaw_dir(angle_idx);
            let mut wall_dist_min = f32::MAX;
            for &sample_dist in WALL_SAMPLES.iter() {
                let sx = pos.x + dir.x * sample_dist;
                let sz = pos.z + dir.y * sample_dist;
                let voxel = world.get_voxel_at(sx, sz);
                if voxel == VoxelType::Wall || voxel == VoxelType::Hole {
                    wall_dist_min = wall_dist_min.min(sample_dist);
                }
            }
            // Encode: 0=no wall in 60m, 127=touching wall, 1-126=linear
            let dist_encoded = if wall_dist_min >= WALL_MAX_DIST {
                0i8
            } else {
                ((127.0 * (WALL_MAX_DIST - wall_dist_min) / WALL_MAX_DIST).clamp(1.0, 127.0)) as i8
            };
            sensory_wave[16 + angle_idx] = dist_encoded;
        }

        // --- ch 24-31: wall_width_fov (obstacle width per YAW angle) ---
        // For each lateral angle, measure obstacle width at 15m ahead.
        for angle_idx in 0..8 {
            let dir = yaw_dir(angle_idx);
            let perp = Vec2::new(-dir.y, dir.x); // perpendicular (2D)
            let mut width = 0.0;
            for &offset in [-3.0_f32, -2.0, -1.0, 1.0, 2.0, 3.0].iter() {
                let px = pos.x + dir.x * 15.0 + perp.x * offset;
                let pz = pos.z + dir.y * 15.0 + perp.y * offset;
                let voxel = world.get_voxel_at(px, pz);
                if voxel == VoxelType::Wall || voxel == VoxelType::Hole {
                    width += offset.abs();
                }
            }
            let width_encoded = (width / 6.0 * 127.0).clamp(0.0, 127.0) as i8;
            sensory_wave[24 + angle_idx] = width_encoded;
        }

        // --- ch 32-39: wall_curvature_fov (flow curvature per YAW angle) ---
        // At each lateral angle, sample flow field 10-22m ahead → cross product
        // with forward = local track curvature. Positive = curving right, neg = left.
        for angle_idx in 0..8 {
            let dir = yaw_dir(angle_idx);
            let dist_ahead = 10.0 + (angle_idx as f32) * 1.5;
            let sx = pos.x + dir.x * dist_ahead;
            let sz = pos.z + dir.y * dist_ahead;
            let flow_local = world.get_flow_at(sx, sz);
            let cross = forward_2d.x * flow_local.y - forward_2d.y * flow_local.x;
            sensory_wave[32 + angle_idx] = (cross * 127.0).clamp(-127.0, 127.0) as i8;
        }

        // === NEAR ZONE (5-40m) vs FAR ZONE (45-150m) ===
        // Two depth layers for lookahead: reactive + strategic.
        // FAR zone starts at 45m (was 80m) — closes the old 40-80m gap.

        // --- ch 40-47: NEAR ZONE (5-40m lookahead - reactive steering) ---
        for dist_idx in 0..8 {
            let look_dist = 5.0 + dist_idx as f32 * 5.0; // 5, 10, 15, ..., 40m
            let probe_pos = pos + forward * look_dist;
            let voxel = world.get_voxel_at(probe_pos.x, probe_pos.z);
            let flow = world.get_flow_at(probe_pos.x, probe_pos.z);

            let road_quality = match voxel {
                VoxelType::Road => 60,
                VoxelType::Grass => -60,
                VoxelType::Wall => 127,
                VoxelType::Hole => -127,
            };

            let flow_mag = flow.length();
            sensory_wave[40 + dist_idx] =
                ((road_quality as f32 * (0.5 + flow_mag * 0.5)).clamp(-127.0, 127.0)) as i8;
        }

        // --- ch 48-55: FAR ZONE (45-150m lookahead - strategic planning) ---
        // Detects curves, crests, turns. Starts at 45m to close old 40-80m gap.
        for step in 0..8 {
            let horizon_dist = 45.0 + step as f32 * 15.0; // 45, 60, 75, 90, 105, 120, 135, 150m
            let apex_pos = pos + forward * horizon_dist;

            let height_now = world.get_height_at(apex_pos.x, apex_pos.z);
            let height_ahead =
                world.get_height_at(apex_pos.x + forward.x * 10.0, apex_pos.z + forward.z * 10.0);
            let height_delta = height_ahead - height_now;

            let normal = world.get_surface_normal(apex_pos.x, apex_pos.z);
            let banking = normal.x.atan2(normal.y);

            let voxel_far = world.get_voxel_at(apex_pos.x, apex_pos.z);
            let road_presence = match voxel_far {
                VoxelType::Road => 1.0,
                VoxelType::Grass => 0.3,
                VoxelType::Wall => -1.0,
                VoxelType::Hole => -0.5,
            };

            let apex_signal = (height_delta * 50.0 + banking * 40.0 + road_presence * 30.0)
                .clamp(-127.0, 127.0) as i8;
            sensory_wave[48 + step] = apex_signal;
        }

        // --- ch 56-63: ground_material + material lookahead ---
        // ch 56: material at the current point (Road=0, Grass=64, Wall=100, Hole=-127)
        // ch 57: material 3m ahead   — anticipates imminent surface transition
        // ch 58: material 8m ahead   — medium horizon (complements road_ahead NEAR)
        // ch 59: material 20m ahead  — long horizon (complements road_ahead FAR)
        //
        // BEFORE: ch 57, 58, 59 were exact copies of ch 56 (material_code).
        // Measured variance (120 samples, 2026-08-04): cor(56,57)=1.0 — exact duplicate.
        // AFTER: each channel looks at a distinct distance ahead, giving 4 temporally
        // separated pieces of surface-change information.
        //
        // IMPACT: invalidates trained DNA (channels 57-59 change semantics).
        // Update the CPU oracle in gpu_validate --percepcao (ch 48-59 compared).
        let material_encode = |vox: VoxelType| -> i8 {
            match vox {
                VoxelType::Road => 0,
                VoxelType::Grass => 64,
                VoxelType::Wall => 100,
                VoxelType::Hole => -127,
            }
        };
        let current_voxel = world.get_voxel_at(pos.x, pos.z);
        sensory_wave[56] = material_encode(current_voxel); // current position
        // lookaheads: sample ahead along the forward vector
        let probe_3m = pos + forward * 3.0;
        let probe_8m = pos + forward * 8.0;
        let probe_20m = pos + forward * 20.0;
        sensory_wave[57] = material_encode(world.get_voxel_at(probe_3m.x, probe_3m.z));
        sensory_wave[58] = material_encode(world.get_voxel_at(probe_8m.x, probe_8m.z));
        sensory_wave[59] = material_encode(world.get_voxel_at(probe_20m.x, probe_20m.z));

        // ch 60-63: agent state (speed, throttle, steer intent - will be filled below)
        // Placeholder for telemetry channels
        for i in 0..4 {
            sensory_wave[60 + i] = 0;
        }

        // Motion per-row: vertical motion structure (complements hemispheric ch 83/84).
        // Row 0 (near pitch) + high motion = nearby object moving = urgent.
        // Row 7 (far pitch) + motion = distant curve.
        for row in 0..RETINA_ROWS {
            let motion_sum: i32 = (0..RETINA_COLS)
                .map(|col| {
                    let cur = retina_binocular[row * RETINA_COLS + col] as i32;
                    let prev = racer.visual_cortex[row * RETINA_COLS + col] as i32;
                    (cur - prev).abs()
                })
                .sum();
            // Theoretical max: 255 * 126 = 32130
            let motion_norm = ((motion_sum as f32 / 8000.0) * 127.0).min(127.0) as i8;
            if row < 7 {
                sensory_wave[21 + row] = motion_norm; // ch 21-27 = rows 0-6
            } else {
                sensory_wave[98] = motion_norm; // ch 98 = row 7
            }
        }

        // Edge density per-row: detects boundaries via horizontal gradient (simplified Sobel).
        // High density = road edge, wall boundary, lane marking.
        for row in 0..RETINA_ROWS {
            let edge_sum: i32 = (0..RETINA_COLS - 1)
                .map(|col| {
                    let a = retina_binocular[row * RETINA_COLS + col] as i32;
                    let b = retina_binocular[row * RETINA_COLS + col + 1] as i32;
                    (a - b).abs()
                })
                .sum();
            // Theoretical max: 255 * 125 = 31875
            let edge_norm = ((edge_sum as f32 / 6000.0) * 127.0).min(127.0) as i8;
            let slot: usize = match row {
                0 => 29,
                1 => 30,
                2 => 31,
                3 => 33,
                4 => 34,
                5 => 35,
                6 => 99,
                7 => 100,
                _ => 0,
            };
            sensory_wave[slot] = edge_norm;
        }

        // --- SACCADE WITH MEMORY: weighted average of dark pixels + hold if blind ---
        // Uses ALL dark pixels (negative value = road) in the frontal sector (±60°),
        // weights by "how dark" and computes the angular centroid. If NOTHING is dark
        // (all wall/grass = vision blocked), KEEPS the previous gaze — it's the memory
        // of "the track was to the right a moment ago, keep going there to find it".
        // 180°/32 pixels ≈ 5.625°/pixel. ±60° ⇒ indices 6..26.
        let per_fov = std::f32::consts::PI;
        let mut dark_angle_sum: f32 = 0.0;
        let mut dark_weight_sum: f32 = 0.0;
        for i in 12..52 {
            let v =
                retina_binocular[((RETINA_ROWS / 2).saturating_sub(1)) * RETINA_COLS + i] as i32; // Row 3 (40m foresight)
            if v < 0 {
                let strength = (-v) as f32;
                let angle = ((i as f32 / (RETINA_COLS as f32 - 1.0)) * 2.0 - 1.0) * (per_fov / 2.0);
                dark_angle_sum += angle * strength;
                dark_weight_sum += strength;
            }
        }
        // Saccade/gaze_yaw DISABLED (user feedback: "problema é o foco" [the problem is focus]).
        // We keep dark_center below as steer priming (channels 90-93) —
        // it already carries the "track goes there, turn there" signal without rotating the eye.
        racer.gaze_yaw = 0.0;
        // Channels 98-100: motion/edge now computed above
        sensory_wave[101] = 0;
        let _ = (dark_angle_sum, dark_weight_sum, per_fov);

        // --- DARK-CENTER TRACKER (Channels 90-93): "Center the Black" — 4 variants ---
        // Angular center of mass of the DARK pixels (track) in the retina.
        // Positive = track goes right → steer RIGHT.
        // Negative = track goes left → steer LEFT.
        //
        // BEFORE (2026-08-04): 4× the same row-3 centroid — exact duplicate (corr=1.0).
        // The original reason was "quadruple vote in the ternary network", but the runtime
        // migrated to F32 and the duplication is no longer useful — 3 channels were waste.
        //
        // AFTER: 4 windows of increasing width over distinct rows to capture
        // the same quantity ("track is to the left/right") at distance scales:
        //   ch 90: W=8,  row 3 (~40m)  — curve anticipation
        //   ch 91: W=16, row 5 (~25m)  — medium horizon, wide window smooths noise
        //   ch 92: pure L-R signal (left_dark – right_dark), normalized
        //          — complements the centroid: works even when the centroid lands on a wall
        //   ch 93: W=8,  row 7 (~15m)  — imminent reaction, curve already in the immediate field
        //
        // IMPACT: invalidates DNA trained on channels 91-93 (semantics change).
        // 2026-08-04: channels 91-93 were constant (corr=1.0 with ch90) → variance lost.
        let half_col = (RETINA_COLS / 2) as f32 - 0.5; // 31.5

        // Helper: sliding-window blob finder for a specific row.
        // Returns the angular centroid normalized in [-1,1].
        let dark_center_for_row = |row: usize, w_size: usize| -> f32 {
            let row_off = row * RETINA_COLS;
            let search_lo = RETINA_COLS / 4; // idx 16
            let search_hi = (RETINA_COLS * 3 / 4).saturating_sub(w_size); // idx 48-w
            let mut best_sum: f32 = -1.0;
            let mut best_centroid: f32 = 0.0;
            for start in search_lo..=search_hi {
                let mut sum: f32 = 0.0;
                let mut weighted: f32 = 0.0;
                for k in 0..w_size {
                    let v = retina_binocular[row_off + start + k] as i32;
                    let d = (-v as f32).max(0.0);
                    sum += d;
                    weighted += d * (start + k) as f32;
                }
                if sum > best_sum {
                    best_sum = sum;
                    best_centroid = if sum > 1.0 {
                        weighted / sum
                    } else {
                        (start + w_size / 2) as f32
                    };
                }
            }
            if best_sum > 1.0 {
                ((best_centroid - half_col) / half_col).clamp(-1.0, 1.0)
            } else {
                0.0
            }
        };

        // ch 90: W=8, row 3 (~40m) — same computation as before, anticipation signal
        let dc_40m = dark_center_for_row((RETINA_ROWS / 2).saturating_sub(1), 8);
        sensory_wave[90] = (dc_40m * 127.0).clamp(-127.0, 127.0) as i8;

        // ch 91: W=16, row 5 (~25m) — wider window, medium horizon
        let dc_25m = dark_center_for_row(5.min(RETINA_ROWS - 1), 16);
        sensory_wave[91] = (dc_25m * 127.0).clamp(-127.0, 127.0) as i8;

        // ch 92: L-R darkness difference at row 3 — information NOT in the centroid:
        // if there is lots of dark on both sides (wide straight), the centroid sits at 0 but
        // LR=0 too, while in an asymmetric curve LR diverges from the centroid.
        // (open_l_norm/open_r_norm computed below, but we use the raw sums here)
        // BUG FIX 2 (user report): "modelo escolhe menor lado" [model picks the smaller side].
        // EXPLICIT channel: sum of "darkness" (road = negative i8) per half.
        let mut left_dark_sum: i32 = 0;
        let mut right_dark_sum: i32 = 0;
        let half = RETINA_COLS / 2;
        let row3_off = ((RETINA_ROWS / 2).saturating_sub(1)) * RETINA_COLS;
        for i in 0..half {
            let v = retina_binocular[row3_off + i] as i32;
            if v < 0 {
                left_dark_sum += -v;
            }
        }
        for i in half..RETINA_COLS {
            let v = retina_binocular[row3_off + i] as i32;
            if v < 0 {
                right_dark_sum += -v;
            }
        }
        // Normalized difference: + = more track on the left, - = more track on the right
        // Subtracts opposite to steer: track on the left = turn left = negative steer signal.
        let lr_diff_norm = ((right_dark_sum - left_dark_sum) as f32 / 4096.0).clamp(-1.0, 1.0);
        sensory_wave[92] = (lr_diff_norm * 127.0).clamp(-127.0, 127.0) as i8;

        // ch 93: W=8, row 7 (~15m) — imminent reaction, same window but closer
        let dc_15m = dark_center_for_row(7.min(RETINA_ROWS - 1), 8);
        sensory_wave[93] = (dc_15m * 127.0).clamp(-127.0, 127.0) as i8;

        // Max possible: 32 pixels × 128 = 4096. Normalize for open_l/r used in trapped.
        let open_l_norm = (left_dark_sum as f32 / 4096.0).clamp(0.0, 1.0);
        let open_r_norm = (right_dark_sum as f32 / 4096.0).clamp(0.0, 1.0);

        // --- ch 36-39: OPEN_L per row (left opening at 4 distances) ---
        // BEFORE: 4× row-3 open_l_i8 — exact duplicate (corr=1.0).
        // AFTER: rows 3/5/7/9 to capture the dark on the left at distinct distances.
        //   ch 36: row 3 (~40m)   — same as before, long anticipation
        //   ch 37: row 5 (~25m)   — medium horizon
        //   ch 38: row 7 (~15m)   — near reaction
        //   ch 39: row 9 (~8m)    — imminent (closest available in the retina)
        // IMPACT: invalidates DNA on ch 37-39 (semantics change).
        let dark_sum_left_at_row = |row: usize| -> f32 {
            let off = row.min(RETINA_ROWS - 1) * RETINA_COLS;
            let mut s: i32 = 0;
            for i in 0..half {
                let v = retina_binocular[off + i] as i32;
                if v < 0 {
                    s += -v;
                }
            }
            (s as f32 / 4096.0).clamp(0.0, 1.0)
        };
        sensory_wave[36] = (dark_sum_left_at_row(3) * 127.0) as i8; // ~40m
        sensory_wave[37] = (dark_sum_left_at_row(5) * 127.0) as i8; // ~25m
        sensory_wave[38] = (dark_sum_left_at_row(7) * 127.0) as i8; // ~15m
        sensory_wave[39] = (dark_sum_left_at_row(9.min(RETINA_ROWS - 1)) * 127.0) as i8; // ~8m

        // --- ch 40-43: OPEN_R per row (right opening at 4 distances) ---
        // BEFORE: 4× row-3 open_r_i8 — exact duplicate (corr=1.0).
        // AFTER: same idea as ch 36-39, but the right half of the retina.
        // IMPACT: invalidates DNA on ch 41-43 (semantics change).
        let dark_sum_right_at_row = |row: usize| -> f32 {
            let off = row.min(RETINA_ROWS - 1) * RETINA_COLS;
            let mut s: i32 = 0;
            for i in half..RETINA_COLS {
                let v = retina_binocular[off + i] as i32;
                if v < 0 {
                    s += -v;
                }
            }
            (s as f32 / 4096.0).clamp(0.0, 1.0)
        };
        sensory_wave[40] = (dark_sum_right_at_row(3) * 127.0) as i8; // ~40m
        sensory_wave[41] = (dark_sum_right_at_row(5) * 127.0) as i8; // ~25m
        sensory_wave[42] = (dark_sum_right_at_row(7) * 127.0) as i8; // ~15m
        sensory_wave[43] = (dark_sum_right_at_row(9.min(RETINA_ROWS - 1)) * 127.0) as i8; // ~8m

        // BEST ESCAPE direction — RESTRICT to FRONTAL field.
        // User bug fix: 180° periphery caught track BEHIND, model turned backward crashing.
        // Frontal-only = car only considers "escape toward where I'm going".
        // --- ch 60-63: ESCAPE_DIR per row (escape direction at 4 distances) ---
        // BEFORE: 4× row-3 escape_i8 — exact duplicate (corr=1.0).
        // AFTER: escape computed at rows 3/5/7/9 for decision depth:
        //   ch 60: row 3 (~40m)  — long-corridor anticipation
        //   ch 61: row 5 (~25m)  — medium horizon
        //   ch 62: row 7 (~15m)  — near decision
        //   ch 63: row 9 (~8m)   — imminent reflex
        // IMPACT: invalidates DNA on ch 61-63 (semantics change).
        let escape_at_row = |row: usize| -> i8 {
            let row_off = row.min(RETINA_ROWS - 1) * RETINA_COLS;
            let mut b_start = 0usize;
            let mut b_len = 0usize;
            let mut c_start = 0usize;
            let mut c_len = 0usize;
            for i in (RETINA_COLS / 4)..(RETINA_COLS * 3 / 4) {
                let v = retina_binocular[row_off + i] as i32;
                if v < -10 {
                    if c_len == 0 {
                        c_start = i;
                    }
                    c_len += 1;
                    if c_len > b_len {
                        b_start = c_start;
                        b_len = c_len;
                    }
                } else {
                    c_len = 0;
                }
            }
            let norm = if b_len >= 2 {
                let center = b_start as f32 + (b_len as f32 - 1.0) * 0.5;
                ((center - 31.5) / 31.5).clamp(-1.0, 1.0) * 127.0
            } else {
                0.0
            };
            norm as i8
        };
        sensory_wave[60] = escape_at_row(3); // ~40m
        sensory_wave[61] = escape_at_row(5); // ~25m
        sensory_wave[62] = escape_at_row(7); // ~15m
        sensory_wave[63] = escape_at_row(9.min(RETINA_ROWS - 1)); // ~8m

        // BUG FIX 3 (user report): "pista pequena bate direto" [small track crashes head-on].
        // --- ch 44-47: WALL_IMMINENT decomposed into 4 angular regions ---
        // BEFORE: 4× the same max-pixel in the central fovea — exact duplicate (corr=1.0).
        // AFTER: 4 retina regions to detect a wall by angle of arrival:
        //   ch 44: central fovea (cols 5/16..11/16) — same as before
        //   ch 45: narrow fovea (cols 6/16..10/16, closest row = row_near)
        //   ch 46: left side (cols 0..1/4) — wall coming from the left
        //   ch 47: right side (cols 3/4..COLS) — wall coming from the right
        // IMPACT: invalidates DNA on ch 45-47 (semantics change).
        let reflex_row = (RETINA_ROWS / 2).saturating_sub(1);
        let near_row = RETINA_ROWS.saturating_sub(2); // closest row available

        // ch 44: central fovea, row 3 (~30m) — keeps current behavior
        let mut wall_central_max: i32 = -128;
        for i in (RETINA_COLS * 5 / 16)..(RETINA_COLS * 11 / 16) {
            let v = retina_binocular[reflex_row * RETINA_COLS + i] as i32;
            if v > wall_central_max {
                wall_central_max = v;
            }
        }
        let wall_imminent_i8: i8 = if wall_central_max > 110 {
            127
        } else if wall_central_max > 80 {
            90
        } else if wall_central_max > 40 {
            50
        } else {
            0
        };
        sensory_wave[44] = wall_imminent_i8;

        // ch 45: narrow fovea (cols 6/16..10/16), closest row — super-imminent collision
        let mut wall_near_max: i32 = -128;
        for i in (RETINA_COLS * 6 / 16)..(RETINA_COLS * 10 / 16) {
            let v = retina_binocular[near_row * RETINA_COLS + i] as i32;
            if v > wall_near_max {
                wall_near_max = v;
            }
        }
        sensory_wave[45] = if wall_near_max > 110 {
            127
        } else if wall_near_max > 80 {
            90
        } else if wall_near_max > 40 {
            50
        } else {
            0
        };

        // ch 46: left side (cols 0..COLS/4), row 3 — wall coming from the left
        let mut wall_left_max: i32 = -128;
        for i in 0..(RETINA_COLS / 4) {
            let v = retina_binocular[reflex_row * RETINA_COLS + i] as i32;
            if v > wall_left_max {
                wall_left_max = v;
            }
        }
        sensory_wave[46] = if wall_left_max > 80 {
            90
        } else if wall_left_max > 40 {
            50
        } else {
            0
        };

        // ch 47: right side (cols 3COLS/4..COLS), row 3 — wall coming from the right
        let mut wall_right_max: i32 = -128;
        for i in (RETINA_COLS * 3 / 4)..RETINA_COLS {
            let v = retina_binocular[reflex_row * RETINA_COLS + i] as i32;
            if v > wall_right_max {
                wall_right_max = v;
            }
        }
        sensory_wave[47] = if wall_right_max > 80 {
            90
        } else if wall_right_max > 40 {
            50
        } else {
            0
        };

        // ch 48..51 = TRAPPED_SCORE decomposed into 4 components.
        // BEFORE: 4× the same composite value — exact duplicate (corr=1.0).
        // AFTER: exposes the 3 individual factors + composite result:
        //   ch 48: normalized wall_pressure [0..127] — frontal wall pressure
        //   ch 49: stuck_speed [0..127] — how stopped the car is (1 - v/25km/h)
        //   ch 50: lr_blocked [0..127]  — lack of lateral dark (no L+R exit)
        //   ch 51: final composite (original result) — emergency heuristic
        // IMPACT: invalidates DNA on ch 49-51 (semantics change).
        let speed_kmh_now = ((sensory_wave[65] as f32 / 127.0) * 80.0 * 3.6).max(0.0);
        let wall_pressure = wall_imminent_i8 as f32;
        let stuck_speed = (1.0 - (speed_kmh_now / 25.0).clamp(0.0, 1.0)) * 127.0;
        let lr_blocked = (1.0 - (open_l_norm + open_r_norm) * 0.5).clamp(0.0, 1.0) * 127.0;
        let trapped_i8: i8 = if wall_pressure >= 80.0 {
            (stuck_speed * 0.3 + wall_pressure * 0.5 + lr_blocked * 0.2).clamp(0.0, 127.0) as i8
        } else {
            0
        };
        sensory_wave[48] = wall_pressure.clamp(0.0, 127.0) as i8;
        sensory_wave[49] = stuck_speed.clamp(0.0, 127.0) as i8;
        sensory_wave[50] = lr_blocked.clamp(0.0, 127.0) as i8;
        sensory_wave[51] = trapped_i8;

        // --- OPTICAL FLOW + APPROACH DETECTION ---
        // Motion sums (ch 83/84) — any motion.
        // Approach signed (ch 114/115) — wall APPROACHING from outside toward center.
        // Pixel value RISES = wall getting closer (near = positive retina).
        // Signed ΔV per hemisphere: + = approaching, - = receding.
        let mut motion_left: i32 = 0;
        let mut motion_right: i32 = 0;
        let mut approach_left: i32 = 0; // signed delta sum (positive = wall coming)
        let mut approach_right: i32 = 0;
        // Optical flow uses rows 3 and 4 (horizon/sky) to detect approaching walls.
        // The ground (rows 0, 1) is constant and generates no useful motion delta for walls.
        for i in 0..half {
            let dr_f = retina_binocular[((RETINA_ROWS / 2).saturating_sub(1)) * RETINA_COLS + i]
                as i32
                - racer.visual_cortex[((RETINA_ROWS / 2).saturating_sub(1)) * RETINA_COLS + i]
                    as i32;
            let dr_p = retina_binocular[((RETINA_ROWS / 2) + 1) * RETINA_COLS + i] as i32
                - racer.visual_cortex[((RETINA_ROWS / 2) + 1) * RETINA_COLS + i] as i32;
            motion_left += dr_f.abs() + dr_p.abs();
            approach_left += dr_p;
        }
        for i in half..RETINA_COLS {
            let dr_f = retina_binocular[((RETINA_ROWS / 2).saturating_sub(1)) * RETINA_COLS + i]
                as i32
                - racer.visual_cortex[((RETINA_ROWS / 2).saturating_sub(1)) * RETINA_COLS + i]
                    as i32;
            let dr_p = retina_binocular[((RETINA_ROWS / 2) + 1) * RETINA_COLS + i] as i32
                - racer.visual_cortex[((RETINA_ROWS / 2) + 1) * RETINA_COLS + i] as i32;
            motion_right += dr_f.abs() + dr_p.abs();
            approach_right += dr_p;
        }
        sensory_wave[83] = ((motion_left / 128).min(127)) as i8;
        sensory_wave[84] = ((motion_right / 128).min(127)) as i8;
        // Approach signed: scale 32 px × delta±255 = ±8160. Norm /64 → [-127, 127].
        sensory_wave[114] = (approach_left / 64).clamp(-127, 127) as i8;
        sensory_wave[115] = (approach_right / 64).clamp(-127, 127) as i8;

        // FOVEAL OPTICAL FLOW (ch 116/117) — forward expansion = looming wall.
        // rays 24..40 = ±~15° frontal (fovea TAN_K 1.25 dense, 64 cols).
        let mut fovea_approach: i32 = 0;
        let mut fovea_motion_abs: i32 = 0;
        for i in 24..40 {
            let dr = retina_binocular[((RETINA_ROWS / 2).saturating_sub(1)) * RETINA_COLS + i]
                as i32
                - racer.visual_cortex[((RETINA_ROWS / 2).saturating_sub(1)) * RETINA_COLS + i]
                    as i32;
            fovea_approach += dr; // + = wall coming in front
            fovea_motion_abs += dr.abs();
        }
        sensory_wave[116] = (fovea_approach / 32).clamp(-127, 127) as i8;
        sensory_wave[117] = (fovea_motion_abs / 32).min(127) as i8;

        // --- DEPTH-PER-ROW (channels 52-59): explicit depth per retina row ---
        // Each retina row is pitched to detect objects at a different distance.
        // Row 0 (pitch=-0.10 rad) sees near; Row 7 (pitch=+0.05 rad) sees far.
        // Channel = sum of all WALL pixels (high positive values) in that row,
        // normalized. Gives the model a clear signal: "wall near" vs "wall far".
        // IMPORTANT: capped at 8 rows (ch 52-59) — NOT RETINA_ROWS (14).
        // Rows 8-13 would write to ch 60-65, overwriting escape_i8 in ch 60-63 (vital steering signal).
        for row in 0..8 {
            let wall_sum: i32 = (0..RETINA_COLS)
                .map(|col| {
                    let v = retina_binocular[row * RETINA_COLS + col] as i32;
                    v.max(0) // Only wall/hole (positive), ignores road (negative) and grass (~0-40)
                })
                .sum();
            // Normalize: theoretical max = 126 cols * 127 = 16002
            let depth_normalized = ((wall_sum as f32 / 4000.0) * 127.0).min(127.0) as i8;
            sensory_wave[52 + row] = depth_normalized;
        }

        // --- WALL APPROACH RATE (channels 102-107): dD/dt in m/s ---
        // Detects whether the wall is approaching (positive) or receding (negative).
        // Essential for high-speed reflexes — the model needs rate, not just distance.
        let DT = 1.0 / 64.0_f32;
        // ch 76-77 normalizes by 40.0 (expanded from 15m→40m, BUG 1 fix 2026-08-04).
        // Semantics: 0 = touching, 127 = 40m free. dist = (ch/127) * 40.
        let dist_l = (sensory_wave[76] as f32 / 127.0) * 40.0; // left lateral distance (m)
        let dist_r = (sensory_wave[77] as f32 / 127.0) * 40.0; // right lateral distance (m)

        let approach_rate_l = (racer.wall_dist_prev_l - dist_l) / DT;
        let approach_rate_r = (racer.wall_dist_prev_r - dist_r) / DT;

        // Frontal approach: uses row 3 (40m foresight) central wall pixels as dist_front proxy
        let reflex_row = (RETINA_ROWS / 2).saturating_sub(1);
        let front_wall_px: i32 = (RETINA_COLS / 3..(2 * RETINA_COLS / 3))
            .map(|col| retina_binocular[reflex_row * RETINA_COLS + col] as i32)
            .filter(|&v| v > 0)
            .sum();
        let dist_front_now = if front_wall_px > 0 {
            (30.0 * (1.0 - (front_wall_px as f32 / 2000.0).min(1.0))).max(1.0)
        } else {
            50.0
        };
        let approach_rate_f = (racer.wall_dist_prev_f - dist_front_now) / DT;

        sensory_wave[102] = (approach_rate_l.clamp(-30.0, 30.0) / 30.0 * 127.0) as i8;
        sensory_wave[103] = (approach_rate_r.clamp(-30.0, 30.0) / 30.0 * 127.0) as i8;
        sensory_wave[104] = (approach_rate_f.clamp(-30.0, 30.0) / 30.0 * 127.0) as i8;

        // Minimum TTC (time to contact): D / v_approach (0..2s → 127=imminent, 0=safe)
        let ttc_l = if approach_rate_l > 1.0 {
            (dist_l / approach_rate_l).min(2.0)
        } else {
            2.0
        };
        let ttc_r = if approach_rate_r > 1.0 {
            (dist_r / approach_rate_r).min(2.0)
        } else {
            2.0
        };
        let ttc_f = if approach_rate_f > 1.0 {
            (dist_front_now / approach_rate_f).min(2.0)
        } else {
            2.0
        };
        let ttc_min = ttc_l.min(ttc_r).min(ttc_f).clamp(0.0, 2.0);
        sensory_wave[105] = (127.0 * (1.0 - ttc_min / 2.0)) as i8;

        // Delta_dist (simple derivative, ±15m/s range)
        sensory_wave[106] = (approach_rate_l.clamp(-15.0, 15.0) / 15.0 * 64.0) as i8;
        sensory_wave[107] = (approach_rate_r.clamp(-15.0, 15.0) / 15.0 * 64.0) as i8;

        // Update prev for the next frame
        racer.wall_dist_prev_l = dist_l;
        racer.wall_dist_prev_r = dist_r;
        racer.wall_dist_prev_f = dist_front_now;

        racer.visual_cortex = retina_binocular;

        // --- VISUAL CHAOS (Fear of Obstacles) ---
        // The model should seek the "Black" (Track). Seeing lots of "White" (Wall) causes Panic.
        // Post pooled retina: iterates 16 bins (8 fovea + 8 periphery).
        // --- VISUAL ASSESSMENT: FEAR VS HORIZON ---
        let mut visual_stress = 0.0;
        let mut visual_freedom = 0.0;
        for i in 0..16 {
            let v = sensory_wave[i] as f32;
            if v > 0.0 {
                visual_stress += v / 127.0;
            } else if v < -50.0 {
                // Very negative pixel = free track or very distant wall (Cyan)
                visual_freedom += v.abs() / 128.0;
            }
        }
        visual_stress /= 16.0; // Average whiteness (0.0 to 1.0)
        visual_freedom /= 16.0; // Average free space (0.0 to 1.0)

        // Soft punishment for staring at walls, but STRONG REWARD for looking at the horizon.
        // This encourages the model to turn the wheel until it finds the long path.
        racer.joy_integral += (visual_freedom * 15.0 - visual_stress * 5.0) * dt;

        // [PANIC DISABLED] - The model went blind with noise when approaching
        // walls, which caused the "freezing" from lack of information.
        let _panic_level = 0.0;

        // --- NIGHT NOISE: Environment uncertainty only ---
        use rand::prelude::*;
        let mut rng =
            rand::rngs::StdRng::seed_from_u64(racer.stable_seed + racer.grace_ticks as u64);

        if env.is_night() {
            let magnitude = 15;
            for i in 0..16 {
                let noise = rng.random_range(-magnitude..magnitude);
                sensory_wave[i] = sensory_wave[i].saturating_add(noise);
            }
        }

        // --- 2. INSTINCT AND TELEMETRY CHANNELS (Sensoriometry) ---
        let alignment = (racer.phase - track_phase).cos();
        sensory_wave[64] = (alignment * 127.0) as i8;

        let speed = if let Some(ref body) = opt_body {
            body.velocity.length()
        } else {
            0.0
        };
        let avg_tire_temp = if let Some(ref body) = opt_body {
            body.wheels.iter().map(|w| w.tire_temp).sum::<f32>() / 4.0
        } else {
            0.3
        };
        sensory_wave[65] = (speed * 2.0).clamp(0.0, 127.0) as i8;

        let mut is_drifting = false;
        if let Some(ref body) = opt_body {
            let lateral_dir = transform.back().as_vec3();
            let lateral_vel = body.velocity.dot(lateral_dir);
            sensory_wave[66] = (lateral_vel * 10.0).clamp(-127.0, 127.0) as i8;
            if lateral_vel.abs() > 4.0 && speed > 5.0 && pos.y < 1.0 {
                is_drifting = true;
            }
        }

        // Perception of Mechanical Pain (HP), Engine Strain (RPM) and Current Gear
        // FIXED: normalizes by HP_INICIAL instead of fixed 100.0. With HP_INICIAL=1000
        // and the old base of 100.0, the channel would stay saturated at 127 forever (HP > 100
        // → ratio > 1.0 → clamp → 127 = constant). The brain needs to feel the real HP.
        sensory_wave[67] =
            ((racer.hp / crate::simulation::racer::HP_INICIAL) * 127.0).clamp(-128.0, 127.0) as i8;
        // --- RANK + COMP_STRESS via precomputed data ---
        let my_rank = rank_by_entity.get(&entity).copied().unwrap_or(0);
        let is_leader = my_rank == 0;

        // Channel 125: Competitive Stress (0 leader, 127 behind someone close, 64 mid-pack)
        // Spatial hash query: only 9 surrounding cells (~8 cars each) instead of the whole N.
        let mut comp_stress = 0;
        if !is_leader {
            comp_stress = 64;
            let cx = (pos.x / SPATIAL_CELL).floor() as i32;
            let cz = (pos.z / SPATIAL_CELL).floor() as i32;
            'outer: for dx in -1..=1 {
                for dz in -1..=1 {
                    if let Some(cell) = spatial.get(&(cx + dx, cz + dz)) {
                        for &(other_e, other_pos, _) in cell {
                            if other_e == entity {
                                continue;
                            }
                            let to_other = other_pos - pos;
                            if to_other.length() < 20.0
                                && to_other.normalize_or_zero().dot(forward) > 0.3
                            {
                                comp_stress = 127;
                                break 'outer;
                            }
                        }
                    }
                }
            }
        }
        sensory_wave[125] = comp_stress as i8;

        // Channel 124: Centrifugal Pain (Anti-Spin)
        // If spinning a lot (>1.5 rad/s) with little speed (<2.0 m/s), MAXIMUM PAIN.
        let is_spinning = if let Some(body) = &opt_body {
            body.angular_velocity.y.abs() > 1.5 && speed < 2.0
        } else {
            false
        };
        sensory_wave[124] = if is_spinning { -128 } else { 0 };

        // --- SENSORY-REFERENTIAL AGITATION (STRESS NOISE) ---
        // Injects exploratory noise into the sensors when stressed to escape stuck states
        if comp_stress > 0 {
            let noise_amplitude = (comp_stress as f32 / 127.0) * 5.0;
            for i in 0..16 {
                let noise = (rng.random::<f32>() - 0.5) * noise_amplitude;
                sensory_wave[i] = (sensory_wave[i] as f32 + noise).clamp(-128.0, 127.0) as i8;
            }
        }

        // --- ORIGINAL CHANNELS ---
        if let Some(engine) = &opt_engine {
            let rpm_ratio = engine.current_rpm / engine.max_rpm;
            sensory_wave[68] = (rpm_ratio * 127.0).clamp(0.0, 127.0) as i8;
            sensory_wave[69] = (engine.current_gear as i8 * 20).clamp(0, 127);
        }

        // --- NEW TELEMETRY SENSORS (PROPRIOCEPTION) ---
        // Channel 70: Tire Thermal Load (0-127 = 0%-100%)
        sensory_wave[70] = (avg_tire_temp * 127.0).clamp(0.0, 127.0) as i8;

        // Channel 71: Downforce Sensor (D = 0.5 * ρ * v² * Cl * A)
        let downforce_normalized = (0.5 * 1.225 * speed * speed * 0.18 * 3.0 * 2.0) / 15000.0; // Normalized ~15kN max
        sensory_wave[71] = (downforce_normalized * 127.0).clamp(0.0, 127.0) as i8;

        // Channel 72: Drafting/Slipstream (127 if a pilot ahead within 10m)
        // Spatial hash: only 9 cells instead of the full N.
        let mut in_draft = false;
        let cx = (pos.x / SPATIAL_CELL).floor() as i32;
        let cz = (pos.z / SPATIAL_CELL).floor() as i32;
        'draft: for dx in -1..=1 {
            for dz in -1..=1 {
                if let Some(cell) = spatial.get(&(cx + dx, cz + dz)) {
                    for &(other_e, other_pos, _) in cell {
                        if other_e == entity {
                            continue;
                        }
                        let to_other = other_pos - pos;
                        let dist = to_other.length();
                        let dot_forward = to_other.normalize_or_zero().dot(forward);
                        if dist < 10.0 && dot_forward > 0.5 {
                            in_draft = true;
                            break 'draft;
                        }
                    }
                }
            }
        }
        sensory_wave[72] = if in_draft { 127 } else { 0 };

        // voxel_current: material under the car — used here (ch 73), in dopamine (1540),
        // in the grass penalty (2525), in the hole (2538) and in telemetry (2653).
        let voxel_current = world.get_voxel_at(pos.x, pos.z);

        // Channel 73: Available corridor (total free track width, without depending on ch 76/77).
        // Computed directly here (ch 76/77 are written LATER this tick — not available).
        // Normalized: 0 = 0m corridor (boxed in), 127 = corridor ≥30m (wide track).
        //
        // BEFORE: "on_grass" (-128 on grass, 0 off). Measured variance = 0.0 in 120
        // track samples (almost never on grass during races), dead channel.
        // The "on grass" info is already in ch 56 (material_code = 64 on grass).
        // AFTER: total corridor width — info that ch 76/77 hold SEPARATELY (left/right)
        // but the brain needs as a SUM for "how much space do I have overall".
        // Example: 18m track → corridor ≈ 18m → 18/30 × 127 ≈ 76.
        // IMPACT: invalidates trained DNA; ch 73 changes semantics.
        {
            let left_perp_73 = Vec3::new(-forward.z, 0.0, forward.x).normalize_or_zero();
            let right_perp_73 = -left_perp_73;
            let mut dist_l_73 = 30.0f32;
            let mut dist_r_73 = 30.0f32;
            for &sd in [1.0f32, 3.0, 6.0, 9.0, 12.0, 16.0, 20.0, 26.0, 30.0].iter() {
                let lp = pos + left_perp_73 * sd;
                let rp = pos + right_perp_73 * sd;
                if dist_l_73 == 30.0
                    && matches!(
                        world.get_voxel_at(lp.x, lp.z),
                        VoxelType::Wall | VoxelType::Hole
                    )
                {
                    dist_l_73 = sd;
                }
                if dist_r_73 == 30.0
                    && matches!(
                        world.get_voxel_at(rp.x, rp.z),
                        VoxelType::Wall | VoxelType::Hole
                    )
                {
                    dist_r_73 = sd;
                }
            }
            let corredor = (dist_l_73 + dist_r_73).min(30.0);
            sensory_wave[73] = ((corredor / 30.0) * 127.0).clamp(0.0, 127.0) as i8;
        }

        // Channel 74: Steering Weight (Dynamic Caster)
        let steer_limit = 1.0 / (1.0 + speed * 0.02);
        sensory_wave[74] = (steer_limit * 127.0).clamp(0.0, 127.0) as i8;

        // Channel 75: Weather / Wetness
        sensory_wave[75] = (env.weather_wetness * 127.0).clamp(0.0, 127.0) as i8;

        // --- LATERAL PROXIMITY SENSORS (no raycasts) ---
        // Sampled grid queries: looks for walls at 90° left/right
        let mut dist_l = 15.0;
        let mut dist_r = 15.0;
        let left_perp = Vec3::new(-forward.z, 0.0, forward.x).normalize_or_zero();
        let right_perp = -left_perp;

        for sample_dist in [1.0, 3.0, 6.0, 9.0, 12.0, 15.0].iter() {
            let left_probe = pos + left_perp * sample_dist;
            let right_probe = pos + right_perp * sample_dist;
            if dist_l == 15.0
                && (world.get_voxel_at(left_probe.x, left_probe.z) == VoxelType::Wall
                    || world.get_voxel_at(left_probe.x, left_probe.z) == VoxelType::Hole)
            {
                dist_l = *sample_dist;
            }
            if dist_r == 15.0
                && (world.get_voxel_at(right_probe.x, right_probe.z) == VoxelType::Wall
                    || world.get_voxel_at(right_probe.x, right_probe.z) == VoxelType::Hole)
            {
                dist_r = *sample_dist;
            }
        }
        sensory_wave[76] = ((dist_l / 15.0) * 127.0).clamp(0.0, 127.0) as i8;
        sensory_wave[77] = ((dist_r / 15.0) * 127.0).clamp(0.0, 127.0) as i8;

        // --- DIAGONAL WHISKERS (channels 94-97) — grid-based proximity ---
        // Sample 4 diagonal directions (±30°, ±60°) with grid queries
        const WHISKER_RANGE: f32 = 10.0;
        let ang30 = std::f32::consts::FRAC_PI_6;
        let ang60 = std::f32::consts::FRAC_PI_3;

        let measure_whisker = |angle_offset: f32| -> i8 {
            let angle = racer.phase + angle_offset;
            let dir = Vec3::new(angle.cos(), 0.0, angle.sin());
            let mut min_dist = WHISKER_RANGE;
            for sample_dist in [1.0, 3.0, 5.0, 7.0, 10.0].iter() {
                let probe = pos + dir * sample_dist;
                if world.get_voxel_at(probe.x, probe.z) == VoxelType::Wall
                    || world.get_voxel_at(probe.x, probe.z) == VoxelType::Hole
                {
                    min_dist = min_dist.min(*sample_dist);
                }
            }
            ((1.0 - (min_dist / WHISKER_RANGE).clamp(0.0, 1.0)) * 127.0) as i8
        };
        sensory_wave[94] = measure_whisker(ang30);
        sensory_wave[95] = measure_whisker(-ang30);
        sensory_wave[96] = measure_whisker(ang60);
        sensory_wave[97] = measure_whisker(-ang60);

        // --- ANTICIPATED CURVATURE (TRACK LOOKAHEAD) ---
        // Samples the flow field ahead at multiple distances and measures the angular
        // deviation from the current heading. Positive = track curves LEFT;
        // negative = curves RIGHT. It's the "internal GPS" of the next ~5s.
        for (slot, look_dist) in [
            (78usize, 10.0_f32),
            (79, 25.0),
            (80, 100.0),
            (19, 75.0),
            (20, 150.0),
        ] {
            let sample = pos + forward * look_dist;
            let flow_ahead = world.get_flow_at(sample.x, sample.z);
            // cross-z (2D perp dot): forward × flow_ahead
            let cross = forward_2d.x * flow_ahead.y - forward_2d.y * flow_ahead.x;
            sensory_wave[slot] = (cross * 127.0).clamp(-127.0, 127.0) as i8;
        }

        // Channel 81: LANE OFFSET — lateral asymmetry for track position [-1..+1].
        // Positive = more space on the right → I'm on the LEFT of the track.
        // Negative = more space on the left → I'm on the RIGHT of the track.
        let lane_denom = (dist_l + dist_r).max(0.1);
        let lane_offset = (dist_r - dist_l) / lane_denom;
        sensory_wave[81] = (lane_offset * 127.0).clamp(-127.0, 127.0) as i8;

        // --- RACING LINE SENSORS (channels 108/109/110) ---
        // Teacher signal for the network to learn the ideal line (wide-apex-wide).
        // 108 = lateral offset (m, signed, clamped to ±half_width, then scaled).
        //       Positive = car to the LEFT of the line → needs to turn RIGHT.
        // 109 = offset derivative (previous offset - current). Positive = closing
        //       the distance to the line (good). Encodes "I'm correcting in the right direction".
        // 110 = lookahead offset ~5m ahead on the ideal line. Anticipates turn-in
        //       when the instantaneous offset is 0 but the line has already started curving.
        let pos2d = Vec2::new(pos.x, pos.z);
        let fwd2d = Vec2::new(forward.x, forward.z);
        let line_offset_m = if !world.racing_line.is_empty() {
            world.racing_line.lateral_offset_from_line(pos2d, fwd2d)
        } else {
            0.0
        };
        let hw = world.racing_line.half_width.max(1.0);
        let line_off_norm = (line_offset_m / hw).clamp(-1.0, 1.0);
        sensory_wave[108] = (line_off_norm * 127.0).clamp(-127.0, 127.0) as i8;

        let prev_off = prev_line_offset
            .get(&entity)
            .copied()
            .unwrap_or(line_offset_m);
        // "previous offset - current": positive when the error shrank this tick.
        let d_off = (prev_off - line_offset_m) / hw;
        sensory_wave[109] = (d_off * 127.0 * 8.0).clamp(-127.0, 127.0) as i8; // gain 8x because per-tick delta is small
        prev_line_offset.insert(entity, line_offset_m);

        let lookahead_m = if !world.racing_line.is_empty() {
            world.racing_line.lookahead_offset(pos2d, 5.0)
        } else {
            0.0
        };
        sensory_wave[110] = ((lookahead_m / hw).clamp(-1.0, 1.0) * 127.0) as i8;

        // Channel 82: FRONTAL COLLISION URGENCY — grid-based forward probe (no raycast).
        // Sampled ahead to find closest wall, compute inverse TTC in a single signal [0..127].
        let mut fwd_probe = 30.0f32;
        for sample_dist in [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0].iter() {
            let probe = pos + forward * sample_dist;
            if world.get_voxel_at(probe.x, probe.z) == VoxelType::Wall
                || world.get_voxel_at(probe.x, probe.z) == VoxelType::Hole
            {
                fwd_probe = *sample_dist;
                break;
            }
        }
        let ttc_inv = (speed / fwd_probe.max(0.5)).clamp(0.0, 10.0); // 10 = ~0.1s of TTC
        sensory_wave[82] = ((ttc_inv / 10.0) * 127.0) as i8;

        // ── WIDTH-INVARIANT CHANNELS (Tesla-like generalization) ──
        // Without this, DNA over-fits absolute pixel intensity → 28m DNA doesn't work at 16m.
        // Normalize by half_width = same reaction on any track.
        // ch 118: forward_wall_dist / hw — "X track-widths ahead" (clamp 0..3)
        let fwd_norm = (fwd_probe / hw).clamp(0.0, 3.0);
        sensory_wave[118] = ((fwd_norm / 3.0) * 127.0).clamp(0.0, 127.0) as i8;
        // ch 119: lateral corridor / hw — "lateral free space in track-widths"
        let corridor = (dist_l + dist_r) / (2.0 * hw);
        sensory_wave[119] = (corridor.clamp(0.0, 2.0) * 63.5).clamp(0.0, 127.0) as i8;
        // ch 123: whisker min dist / hw — closest wall on either side, normalized
        let whisker_min = dist_l.min(dist_r).max(0.1);
        let whisker_norm = (whisker_min / hw).clamp(0.0, 1.5);
        sensory_wave[123] = ((whisker_norm / 1.5) * 127.0).clamp(0.0, 127.0) as i8;

        // --- PERSISTENT SENSORY MEMORY ---
        // EMA (alpha ~0.15/frame @ 60Hz ≈ ~0.5s half-life) smooths noise spikes
        // and keeps the "wall nearby" sensation even if the momentary raycast fails.
        let ema_alpha = (dt * 3.0).clamp(0.0, 1.0);
        racer.wall_ema_l = racer.wall_ema_l * (1.0 - ema_alpha) + dist_l * ema_alpha;
        racer.wall_ema_r = racer.wall_ema_r * (1.0 - ema_alpha) + dist_r * ema_alpha;
        // Inverted: 127 = touching, 0 = 15m+ free. Persistent "alert" channel.
        sensory_wave[85] = ((1.0 - (racer.wall_ema_l / 15.0).clamp(0.0, 1.0)) * 127.0) as i8;
        sensory_wave[86] = ((1.0 - (racer.wall_ema_r / 15.0).clamp(0.0, 1.0)) * 127.0) as i8;

        // Channel 87: SLIP ANGLE — angle between velocity and forward (drift/aquaplaning signal).
        // Positive = velocity points to the right of heading (drift to the right).
        let slip_signal = if let Some(ref body) = opt_body {
            if body.velocity.length_squared() > 1.0 {
                let v2d = Vec2::new(body.velocity.x, body.velocity.z).normalize_or_zero();
                let f2d = forward_2d.normalize_or_zero();
                // cross-z: f × v
                f2d.x * v2d.y - f2d.y * v2d.x
            } else {
                0.0
            }
        } else {
            0.0
        };
        sensory_wave[87] = (slip_signal * 127.0).clamp(-127.0, 127.0) as i8;

        // Channel 88: SUSTAINED PROXIMITY — accumulates time scraping either side (<3m).
        // Signals "I've been scraping too long, need to move away".
        let raspando = racer.wall_ema_l < 3.0 || racer.wall_ema_r < 3.0;
        if raspando {
            racer.proximity_sustain = (racer.proximity_sustain + dt).min(3.0);
        } else {
            racer.proximity_sustain = (racer.proximity_sustain - dt * 0.5).max(0.0);
        }
        sensory_wave[88] = ((racer.proximity_sustain / 3.0) * 127.0) as i8;

        // Channel 89: reverse/cooldown state from the PREVIOUS frame. 127 in reverse, 64 in cooldown, 0 normal.
        // (the actual timer update happens further below, after the network decides)
        sensory_wave[89] = if racer.reverse_timer > 0.0 {
            127
        } else if racer.post_reverse_lockout > 0.0 {
            64
        } else {
            0
        };

        // --- DRIFT SKIDMARKS ---
        if is_drifting && rng.random::<f32>() < 0.15 {
            // rate limit to ~10 FPS
            let (mark_mesh, mark_mat) = skidmark_cache.get_or_insert_with(|| {
                (
                    meshes.add(Rectangle::new(1.0, 1.0)),
                    materials.add(StandardMaterial {
                        base_color: Color::srgba(0.0, 0.0, 0.0, 0.7),
                        alpha_mode: AlphaMode::Blend,
                        unlit: true,
                        ..default()
                    }),
                )
            });

            let mark_length = (speed * 0.06).max(0.5);
            let mut mark_t = *transform;
            mark_t.translation = pos - mark_t.right().as_vec3() * 0.9;
            mark_t.translation.y = 0.51;
            mark_t.rotation *= Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
            mark_t.scale = Vec3::new(mark_length, 0.4, 1.0);

            let _e = commands
                .spawn((
                    Mesh3d(mark_mesh.clone()),
                    MeshMaterial3d(mark_mat.clone()),
                    mark_t,
                    Skidmark { timer: 5.0 },
                ))
                .id();
            // Not adding to world_entities because they decay on their own and
            // cleaning them up explicitly could block the CPU unnecessarily during reset.
        }

        // Temporal memory
        sensory_wave[120] = (racer.memory[0] * 127.0) as i8;
        sensory_wave[121] = (racer.memory[1] * 127.0) as i8;
        sensory_wave[122] = (racer.memory[2] * 127.0) as i8;

        // --- DOPAMINERGIC SYSTEM (Continuous Reward Shaping) ---
        // joy_tick = aligned_progress * surface. Streak compounds flow state.
        // Evolution selects whoever MAXIMIZES the integral — grass/wall are ALLOWED, they just yield less/cost.
        let progress_speed = (speed * alignment.max(0.0) / 30.0).clamp(0.0, 1.0); // 30m/s ≈ 108km/h = peak
        let surface_factor = match voxel_current {
            VoxelType::Road => 1.0,
            VoxelType::Grass => 0.2,
            VoxelType::Wall => -0.5,
            VoxelType::Hole => -1.0, // Hole is worse than wall — reinforces avoidance.
        };
        let joy_tick = progress_speed * surface_factor;
        if joy_tick > 0.5 {
            racer.joy_streak = (racer.joy_streak + dt).min(5.0);
        } else {
            racer.joy_streak = (racer.joy_streak - dt * 2.0).max(0.0); // decays 2x faster
        }
        let dopamine_mult = 1.0 + racer.joy_streak * 0.5; // up to 3.5x after 5s of flow
        racer.joy_integral += joy_tick * dopamine_mult * dt;

        // Channels 126/127: exposes dopamine as a sensory signal (evolvable meta-cognition)
        sensory_wave[126] = (joy_tick * 127.0).clamp(-127.0, 127.0) as i8;
        sensory_wave[127] = ((racer.joy_streak / 5.0) * 127.0) as i8;

        // --- TIER 3.1: DOPAMINE (Reward Prediction Error) ---
        // actual_reward ≈ delta_track_progress[m] * 10 - wall_contact_cost.
        // Proxy for delta: speed * alignment * dt (forward component of advance).
        // wall_contact_cost fires on wall surface (voxel == Wall).
        let delta_track_progress = (speed * alignment.max(0.0) * dt).max(0.0);
        let wall_contact_cost = if matches!(voxel_current, VoxelType::Wall) {
            5.0 * dt
        } else {
            0.0
        };
        let actual_reward = delta_track_progress * 10.0 - wall_contact_cost;
        racer.engine.metabolism.update_dopamine(actual_reward);

        // --- TIER 3.2: CORTISOL (integral fear) ---
        // Integrates proximity of the nearest wall between left/right EMA.
        // wall_ema_l/r are already distances in meters (decay ~1s).
        let min_wall_distance = racer.wall_ema_l.min(racer.wall_ema_r);
        racer
            .engine
            .metabolism
            .update_cortisol(min_wall_distance, dt);

        // --- TIER 3.3: CURIOSITY (retina novelty) ---
        // Measures visual_cortex variance over the last 16 frames. Monotony → it rises.
        // visual_cortex was updated just above in this tick (recompute branch)
        // or preserves the previous substep's value (fallback). Both cases serve
        // as a valid retinal signal for the curiosity history.
        let retina_snapshot = racer.visual_cortex;
        racer
            .engine
            .metabolism
            .update_curiosity(&retina_snapshot, dt);

        // --- 3. SOVEREIGN EXECUTION ---
        // Tier 6.5 — Time Dilation under stress:
        //   low cortisol (<0.5)  → skip_n = 2 (brain fires 30Hz, compute savings).
        //   high cortisol (≥0.5) → skip_n = 1 (brain fires 60Hz, maximum attention under danger).
        // Racer preserves engine.final_intent between ticks → skipped frames reuse the last decision.
        // Gated behind gen ≥ 10 so as not to destabilize the evolutionary bootstrap.
        racer.brain_tick_counter = racer.brain_tick_counter.wrapping_add(1);
        let cortisol = racer.engine.metabolism.cortisol;
        // Time dilation disabled: skip_n=2 at gen>=10 reduced brain freq to
        // 30Hz and degraded DNA reload (cars need to react at 60Hz to cap).
        let _ = cortisol;
        let skip_n: u8 = 1;
        let should_fire = (racer.brain_tick_counter % skip_n) == 0;

        // --- PERCEPTION STAGE v2: full upgrade (with retrain) ---
        // Added channels (trained from scratch):
        //   16..19 = yaw_rate × 4 (ternary weight)
        //   20..23 = pitch_rate × 4
        //   24..27 = roll_rate × 4
        //   28..31 = pre_brake_signal × 4 (speed × |curv|)
        //   78     = curv25_signed (cortex + axiom)
        //   79     = curv25_signed (Sentinel + axiom — duplicated)
        //   80     = curv60_signed (lookahead)
        // Proprioception: yaw/pitch/roll into sensory wave
        {
            let yaw = if let Some(ref body) = opt_body {
                body.angular_velocity.y
            } else {
                0.0
            };
            let pitch_r = if let Some(ref body) = opt_body {
                body.angular_velocity.x
            } else {
                0.0
            };
            let roll = if let Some(ref body) = opt_body {
                body.angular_velocity.z
            } else {
                0.0
            };
            let yaw_i8 = ((yaw / 3.0).clamp(-1.0, 1.0) * 127.0) as i8;
            let pitch_i8 = ((pitch_r / 3.0).clamp(-1.0, 1.0) * 127.0) as i8;
            let roll_i8 = ((roll / 3.0).clamp(-1.0, 1.0) * 127.0) as i8;
            sensory_wave[16] = yaw_i8;
            sensory_wave[17] = pitch_i8;
            sensory_wave[18] = roll_i8;
        }

        // Curvature lookahead — signed angles (7 distances for better curve prediction).
        // Hoisted out of inner block so governor can access these values downstream.
        let curv_at = |ahead_m: f32| -> f32 {
            let probe = Vec3::new(pos.x, 0.0, pos.z) + forward * ahead_m;
            let f = world.get_flow_at(probe.x, probe.z);
            if f.length_squared() > 1e-4 {
                let cross_y = forward.x * f.y - forward.z * f.x;
                let dot = (forward.x * f.x + forward.z * f.y).clamp(-1.0, 1.0);
                cross_y.atan2(dot)
            } else {
                0.0
            }
        };
        let yn = std::f32::consts::FRAC_PI_2;
        // 7 distances: near/mid/far progression for granular curve perception.
        let c20_signed = curv_at(20.0);
        let c40_signed = curv_at(40.0);
        let c70_signed = curv_at(70.0);
        let c120_signed = curv_at(120.0);
        let c180_signed = curv_at(180.0);
        let c260_signed = curv_at(260.0);
        // c350_signed is no longer used here: ch111 (350m) migrated to
        // perception_temporal.wgsl (curv_at_d), eliminating the CPU/GPU write conflict.

        {
            let c20 = ((c20_signed / yn).clamp(-1.0, 1.0) * 127.0) as i8;
            let c40 = ((c40_signed / yn).clamp(-1.0, 1.0) * 127.0) as i8;
            let c70 = ((c70_signed / yn).clamp(-1.0, 1.0) * 127.0) as i8;
            let c120 = ((c120_signed / yn).clamp(-1.0, 1.0) * 127.0) as i8;
            let c180 = ((c180_signed / yn).clamp(-1.0, 1.0) * 127.0) as i8;
            let c260 = ((c260_signed / yn).clamp(-1.0, 1.0) * 127.0) as i8;

            // Ch 20, 78, 79, 80, 32, 101 (6 lookahead channels).
            // ch111 (350m horizon) was moved to perception_temporal.wgsl:curv_at_d(350)
            // — write conflict eliminated; now only the GPU writes ch111.
            sensory_wave[20] = c20; // 20m nearest
            sensory_wave[78] = c40; // 40m near
            sensory_wave[79] = c70; // 70m mid
            sensory_wave[80] = c120; // 120m mid-far
            sensory_wave[32] = c180; // 180m far
            sensory_wave[101] = c260; // 260m very far

            // Pre-brake: speed × max curv across horizons. Weights decay: nearest highest.
            let speed_norm = (speed / 80.0).clamp(0.0, 1.0);
            let curv_abs_max = (c40_signed.abs() / yn)
                .max(c70_signed.abs() / yn * 0.90)
                .max(c120_signed.abs() / yn * 0.70)
                .max(c180_signed.abs() / yn * 0.50)
                .clamp(0.0, 1.0);
            let pre_brake = (speed_norm * curv_abs_max * 127.0).clamp(0.0, 127.0) as i8;
            sensory_wave[28] = pre_brake;
        }

        if should_fire {
            let t_percept = std::time::Instant::now();
            racer.engine.retina_perception.inject_into_wave(
                &retina_binocular,
                &mut sensory_wave,
                192,
                0.3,
            );
            perf_accum.2 += t_percept.elapsed().as_secs_f64();

            let t_brain = std::time::Instant::now();
            racer.engine.execute_cycle(&sensory_wave, track_phase, dt);
            perf_accum.3 += t_brain.elapsed().as_secs_f64();

            // DEBUG: capture spawn behavior (first 1.0s of each racer)
            let debug_gen = race.generation;
            let debug_tick = (racer.race_time * 64.0) as u32;
            if racer.race_time < 1.0 {
                let steer_pre_axiom = racer.engine.final_intent.get(1).copied().unwrap_or(0.0);
                let throttle = racer.engine.final_intent.get(0).copied().unwrap_or(0.0);
                let sensory_sample = format!(
                    "[{},{},{},{},{}]",
                    sensory_wave[0],
                    sensory_wave[1],
                    sensory_wave[75],
                    sensory_wave[82],
                    sensory_wave[118]
                );
                race.telemetry.write_debug(serde_json::json!({
                    "gen": debug_gen,
                    "tick": debug_tick,
                    "entity_id": entity.index(),
                    "pos": format!("({:.1},{:.1})", pos.x, pos.z),
                    "steer_pre_axiom": format!("{:.3}", steer_pre_axiom),
                    "throttle": format!("{:.3}", throttle),
                    "sensory_sample": sensory_sample,
                    "track_progress": format!("{:.1}", racer.track_progress)
                }));
            }

            // Tier 1.3 — Axiom as external prior (potential-based shaping).
            // Injects procedural bias into the brain output AFTER the forward pass. Scale
            // decays with the generation (race.current_axiom_strength: 1.0 gen 0 → 0.0 gen 50+).
            // Clamp to [-1, 1] preserves the range of the tanh already applied to final_intent.
            let ax_strength = race.current_axiom_strength;
            if ax_strength > 1e-4 {
                let bias = axiom_policy(&sensory_wave, &lab);
                let n = racer.engine.final_intent.len().min(64);
                for i in 0..n {
                    let v = racer.engine.final_intent[i] + bias[i] * ax_strength;
                    racer.engine.final_intent[i] = v.clamp(-1.0, 1.0);
                }

                // DEBUG: post-axiom steer
                if racer.race_time < 1.0 {
                    let steer_post_axiom = racer.engine.final_intent.get(1).copied().unwrap_or(0.0);
                    let axiom_steer_bias = bias.get(1).copied().unwrap_or(0.0);
                    race.telemetry.write_debug(serde_json::json!({
                        "gen": debug_gen,
                        "tick": debug_tick,
                        "entity_id": entity.index(),
                        "steer_post_axiom": format!("{:.3}", steer_post_axiom),
                        "axiom_steer_bias": format!("{:.3}", axiom_steer_bias),
                        "axiom_strength": format!("{:.3}", ax_strength)
                    }));
                }
            }
            // ---- Trinity decide (Gate + Reflex + Sentinel) ----
            let cortex_intent = [
                racer.engine.final_intent.get(0).copied().unwrap_or(0.0),
                racer.engine.final_intent.get(1).copied().unwrap_or(0.0),
                racer.engine.final_intent.get(2).copied().unwrap_or(0.0),
            ];
            let speed_kmh = speed * 3.6;
            // Wall raycast proxies derived from retina samples (i8 0..127). Larger sum = closer.
            let horizon_row = RETINA_ROWS / 2;
            let wall_l_proxy = {
                let s: i32 = ((RETINA_COLS / 2)..(RETINA_COLS * 38 / 64))
                    .map(|i| retina_binocular[horizon_row * RETINA_COLS + i] as i32)
                    .sum();
                let norm = (s as f32 / (6.0 * 127.0)).clamp(0.0, 1.0);
                (1.0 - norm) * 5.0
            };
            let wall_r_proxy = {
                let s: i32 = ((RETINA_COLS * 58 / 64)..RETINA_COLS)
                    .map(|i| retina_binocular[horizon_row * RETINA_COLS + i] as i32)
                    .sum();
                let norm = (s as f32 / (6.0 * 127.0)).clamp(0.0, 1.0);
                (1.0 - norm) * 5.0
            };
            let fwd_obstacle_norm: f32 = {
                let s: i32 = (10..22).map(|i| sensory_wave[i] as i32).sum();
                (s as f32 / (12.0 * 127.0)).clamp(0.0, 1.0)
            };
            let ttc_forward = if speed > 0.5 {
                let clear_m = (1.0 - fwd_obstacle_norm) * 30.0;
                (clear_m / speed).clamp(0.0, 99.0)
            } else {
                99.0
            };
            // yaw_rate: pull from body angular velocity if available (cheap local).
            let yaw_rate: f32 = if let Some(ref body) = opt_body {
                body.angular_velocity.y
            } else {
                0.0
            };
            // slip_avg: read directly from wheel state (1-tick lag from physics, fine).
            let slip_avg: f32 = if let Some(ref body) = opt_body {
                body.wheels
                    .iter()
                    .map(|w| w.slip_ratio.abs() + w.slip_angle.abs())
                    .sum::<f32>()
                    / 4.0
            } else {
                0.0
            };
            // curv25 read from sensory_wave[79..80] (filled in proprio block above the brain fire).
            let curv25: f32 =
                ((sensory_wave[79] as f32).abs() + (sensory_wave[80] as f32).abs()) / 254.0 * 60.0;
            let safety_state = crate::safety::SafetyState {
                speed_kmh,
                yaw_rate,
                slip_avg,
                ttc_forward,
                wall_left: wall_l_proxy,
                wall_right: wall_r_proxy,
                // slow_time increments 1.0/tick @ ~64Hz when speed<2m/s; convert to real seconds.
                stall_timer: racer.slow_time / 64.0,
                // wall_time already in real seconds; gate on near-stop so it reads "pinned vs scraping".
                pinado_timer: if speed_kmh < 8.0 {
                    racer.wall_time
                } else {
                    0.0
                },
                rain: env.weather_wetness,
                curv25,
                current_thr: cortex_intent[0],
            };
            let chaos_in = crate::chaos::ChaosInput {
                yaw_rate,
                slip_avg,
                retina_temporal_var: 0.0, // wired later from motion_left/motion_right
            };
            let reflex_in = slow_input_for_reflex(&sensory_wave);
            let trinity = racer
                .engine
                .decide(cortex_intent, &reflex_in, &safety_state, &chaos_in);
            // Mutate final_intent so downstream readers see gate-blended values.
            if let Some(slot) = racer.engine.final_intent.get_mut(0) {
                *slot = trinity[0];
            }
            if let Some(slot) = racer.engine.final_intent.get_mut(1) {
                *slot = trinity[1];
            }
            if let Some(slot) = racer.engine.final_intent.get_mut(2) {
                *slot = trinity[2];
            }
            // ---- end Trinity ----
            // Updates last_action with the fresh output (throttle, steer).
            let ti = racer.engine.final_intent.get(0).copied().unwrap_or(0.0);
            let si = racer.engine.final_intent.get(1).copied().unwrap_or(0.0);
            racer.last_action = (ti, si);
            TIME_DILATION_FIRES.fetch_add(1, AtomicOrdering::Relaxed);
        } else {
            // Skipped: engine.final_intent already holds the last intent (persists between ticks).
            // Nothing to do — downstream will read the same values.
            TIME_DILATION_SKIPS.fetch_add(1, AtomicOrdering::Relaxed);
        }

        // --- SLOW CORTEX (System 2) — aggregates context and emits modulating bias ---
        // 32-dim input: homeostatic state + pooled retina + previous fast outputs.
        // Runs subsampled (1× every 32 ticks). Output in `racer.engine.slow_brain.output`.
        let _lap_progress = if world.lap_distance > 0.0 {
            (racer.distance / world.lap_distance).fract()
        } else {
            0.0
        };
        let mut slow_input = [0i8; 48];
        // Channels 32..47: HD visual bins (from sensory_wave 128..143)
        for i in 0..16 {
            slow_input[32 + i] = sensory_wave[128 + i];
        }
        slow_input[0] = sensory_wave[126]; // joy_tick
        slow_input[1] = sensory_wave[127]; // joy_streak
        // FIXED: normalizes by HP_INICIAL instead of a fixed 100.0.
        // With HP_INICIAL=1000 and a fixed 100.0 divisor, the channel would sit at 127 forever.
        slow_input[2] =
            ((racer.hp / crate::simulation::racer::HP_INICIAL) * 127.0).clamp(0.0, 127.0) as i8; // hp
        // FUEL GAUGE = FRACTION OF THE TANK, not absolute units.
        // It used to be `fuel * 127` with a clamp: with the birth tank at 1000 the channel
        // stayed pinned at 127 forever — the brain had a sensor that was a constant.
        // Even with the tank at 3.0 it would only leave saturation after 2/3 was spent.
        // Normalized by the initial tank, it informs across the whole range, which is what
        // a fuel needle does.
        slow_input[3] =
            ((racer.fuel / crate::simulation::racer::FUEL_INICIAL) * 127.0).clamp(0.0, 127.0) as i8;
        slow_input[4] = ((speed / 80.0) * 127.0).clamp(0.0, 127.0) as i8; // speed / 80m/s
        slow_input[5] = (racer.tire_temp * 127.0).clamp(0.0, 127.0) as i8;
        slow_input[6] = (racer.engine_power * 127.0).clamp(0.0, 127.0) as i8;
        slow_input[7] = (alignment * 127.0) as i8;
        // BUG FIX: Second Lap Amnesia.
        // We omit lap_progress and total_laps from the model input. They caused a discontinuity
        // in the network state when the race completed a lap, making the car "forget" how to drive.
        slow_input[8] = 0;
        slow_input[9] = 0;
        slow_input[10] = ((racer.wall_time / 5.0) * 127.0).clamp(0.0, 127.0) as i8;
        slow_input[11] = ((racer.grass_time / 5.0) * 127.0).clamp(0.0, 127.0) as i8;
        slow_input[12] = (racer.stagnation * 127.0).clamp(0.0, 127.0) as i8;
        slow_input[13] = if env.is_night() { 127 } else { 0 };
        // 14..21: pooled retina — up to 4 regions, volumetric max-pool across all rows.
        // Adapts to the real RETINA_COLS: with 16 cols → 4 regions of 4 bins.
        // With 8 cols → 4 regions of 2 bins. Avoids OOB when RETINA_COLS < 32.
        let n_regions = 4usize.min(RETINA_COLS);
        let bins_per_region = RETINA_COLS / n_regions.max(1);
        for reg in 0..n_regions {
            let mut max_near: i8 = -128;
            let mut max_far: i8 = -128;
            for row in 0..RETINA_ROWS {
                for j in 0..bins_per_region {
                    let col = reg * bins_per_region + j;
                    if col >= RETINA_COLS {
                        break;
                    }
                    let v = racer.visual_cortex[row * RETINA_COLS + col];
                    if row < RETINA_ROWS / 2 {
                        if v > max_near {
                            max_near = v;
                        }
                    } else {
                        if v > max_far {
                            max_far = v;
                        }
                    }
                }
            }
            slow_input[14 + reg] = max_near;
            slow_input[18 + reg] = max_far;
        }
        // 22..25: latest fast-brain intents (throttle, steer, brake, shift)
        slow_input[22] = (racer.memory[0] * 127.0) as i8;
        slow_input[23] = (racer.memory[1] * 127.0) as i8;
        slow_input[24] = (racer.memory[2] * 127.0) as i8;
        slow_input[25] = (racer.engine.final_intent[3] * 127.0) as i8;
        // 26: self-recurrent memory (slow[0] from the previous tick)
        slow_input[26] = (racer.engine.slow_brain.output[0] * 127.0) as i8;
        // 27..28, 30: CURVATURE LOOKAHEAD — track flow 30/60/90m ahead.
        // Signed yaw [-π/2, +π/2] → normalized to [-127, +127]. Negative = curve to the left,
        // positive = right. System 2 now KNOWS the next curve turns and in which direction,
        // and can pre-brake and pre-turn the wheel before the fast brain sees the wall.
        let forward_vec = forward; // alias already available
        let pos_fwd = Vec3::new(pos.x, 0.0, pos.z);
        let signed_yaw_to_flow = |ahead_m: f32| -> f32 {
            let probe = pos_fwd + forward_vec * ahead_m;
            let flow2 = world.get_flow_at(probe.x, probe.z);
            if flow2.length_squared() < 1e-4 {
                return 0.0;
            }
            let flow3 = Vec3::new(flow2.x, 0.0, flow2.y).normalize_or_zero();
            let cross_y = forward_vec.x * flow3.z - forward_vec.z * flow3.x;
            let dot = forward_vec.dot(flow3);
            cross_y.atan2(dot.max(-1.0).min(1.0))
        };
        let curv_30 = signed_yaw_to_flow(30.0);
        let curv_60 = signed_yaw_to_flow(60.0);
        let curv_90 = signed_yaw_to_flow(90.0);
        let yaw_norm = std::f32::consts::FRAC_PI_2;
        slow_input[27] = ((curv_30 / yaw_norm).clamp(-1.0, 1.0) * 127.0) as i8;
        slow_input[28] = ((curv_60 / yaw_norm).clamp(-1.0, 1.0) * 127.0) as i8;
        slow_input[29] = if racer.grace_ticks < 2560 { 127 } else { 0 }; // in_grace (tick-based, sync with motor grace)
        slow_input[30] = ((curv_90 / yaw_norm).clamp(-1.0, 1.0) * 127.0) as i8;
        // channel 31: normalized frontal wall distance (0 = touching, 127 = free >6m), via grid query
        let mut forward_wall_dist_sensor = 30.0f32;
        for sample_dist in [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0].iter() {
            let probe = pos + transform.right().as_vec3() * sample_dist;
            if world.get_voxel_at(probe.x, probe.z) == VoxelType::Wall
                || world.get_voxel_at(probe.x, probe.z) == VoxelType::Hole
            {
                forward_wall_dist_sensor = *sample_dist;
                break;
            }
        }
        slow_input[31] = ((forward_wall_dist_sensor / 6.0).clamp(0.0, 1.0) * 127.0) as i8;
        racer.engine.slow_brain.tick(&slow_input, dt);

        // Use the final_intent stabilized by the CfC and KAN instead of raw weights (avoids ZigZag)
        let intent = &racer.engine.final_intent;

        // ================================================================
        // === SYMBOLIC LAYER: SURVIVAL REFLEXES ==========================
        // ================================================================
        // The neural network has random weights. Without a base "instinct"
        // to move, it never bootstraps. The Symbolic Reflex guarantees that:
        //   1) The car ALWAYS tries to move forward (minimum throttle)
        //   2) Brake only activates on EXPLICIT positive intent (>0.5)
        //   3) The AI modulates INTENSITY and DIRECTION, it does not decide WHETHER to move
        //
        // Evolution selects whoever modulates BEST, not whoever discovers
        // that the accelerator needs pressing.
        // ================================================================

        // Throttle 100% neural — the network has to LEARN to accelerate via A3 priming (flow alignment).
        // Removed the 0.3 minimum survival_throttle: it removed selective pressure on learning throttle.
        // KAN output head: smooth per-channel spline before the final clamp — eliminates the hard ±1
        // saturation of the raw logit ([-2,2] expected) and gives a per-car evolvable curve.
        let neural_throttle = racer.kan_out.eval(0, intent[0]);
        // Symmetric throttle FIX: BEFORE it was (n+1)/2 → a 0-output became a 50% default pedal.
        // Cars accelerated even when the cortex output was neutral/negative. Now:
        //   neural < 0 → pedal 0 (does not accelerate)
        //   neural > 0 → proportional pedal (full at 1.0)
        // Balances with brake (line 1245: brake threshold 0.2).
        let base_pedal = neural_throttle.max(0.0).clamp(0.0, 1.0);

        // SLOW BRAIN MODULATION RE-ENABLED — System 2 active.
        // SLOW-PATHWAY ABLATION (see `neuro.rs`): zeroing the output returns exactly the
        // neutral of this modulation — aggression 0, cap 1.0, bias 0 — i.e., the car running
        // on the fast pathway alone. It is the same paired ablation as the Python bench, and
        // here it tests the prediction that MODULATING pays off in an INSTANT task (racing),
        // unlike the RATE task (survival), where the realizable ceiling measured +0.11%.
        let neutro = [0.0f32; 3];
        let slow: &[f32; 3] = if crate::neuro::ablada() == crate::neuro::Via::Lenta {
            &neutro
        } else {
            &racer.engine.slow_brain.output
        };
        let aggression: f32 = slow[0] * 0.5; // Bias [-0.5, 0.5]
        let throttle_cap: f32 = 1.0 - slow[1].max(0.0) * 0.8; // Range [0.2, 1.0], Neutral at 0.0
        // v3: slow[2] repurposed: risk_damp→steer_bias.
        // OLD risk_damp was DAMPENING steer in curves (counterproductive for tight turns).
        // NEW steer_bias: System-2 anticipatory steering from curvature lookahead (30/60/90m).
        // Signed: positive = steer right, negative = steer left. Range [-0.20, 0.20].
        let steer_bias: f32 = slow[2] * 0.20;
        let risk_damp: f32 = 0.0; // Disabled: was weakening steer exactly when curves need it most
        let cortisol = racer.engine.metabolism.risk_aversion();

        // Tier 3.2: fear brakes (throttle -30% max) and softens the wheel (steer -20% max).
        let cortisol_throttle_mult = 1.0 - cortisol * 0.3;
        let cortisol_steer_mult = 1.0 - cortisol * 0.2;
        let pedal_intent = (base_pedal * (1.0 + aggression * 0.5) * cortisol_throttle_mult)
            .clamp(0.0, throttle_cap);
        // KAN spline on the raw steer BEFORE smoothing (v8) — smooths the curve per evolved car.
        let steer_spline = racer.kan_out.eval(1, intent[1]);
        // Gain compensation: the brain speaks in "real physical units" — we multiply
        // by the inverse of dynamic_steer_limit (cap 2×) BEFORE the clamp, so that neural
        // ±1 saturation corresponds to true full-lock at any speed.
        // See melhorias/gain-asymmetry.md. OFF by default (invalidates legacy champion scales).
        #[cfg(feature = "gain_comp_steer")]
        let gain_comp = (1.0 + speed * 0.02).min(2.0);
        #[cfg(not(feature = "gain_comp_steer"))]
        let gain_comp = 1.0_f32;
        let steer_raw_unclamped = (steer_spline + steer_bias) * cortisol_steer_mult * gain_comp;
        // v8: temporal smoothing via prev_steer_mem (memory[1] snapshot).
        // Fights winner-take-all ±1 saturation alternating frame by frame.
        let prev_steer_mem = racer.memory[1];
        let new_steer = steer_raw_unclamped.clamp(-1.0, 1.0);
        // v3: 0.65/0.35→0.50/0.50 — faster steer response for tight curves.
        // Old: ~180ms lag (3.6m blind @ 20 m/s). New: ~125ms (2.5m).
        let steer_raw = prev_steer_mem * 0.50 + new_steer * 0.50;
        let steer_intent = steer_raw.clamp(-1.0, 1.0) * (1.0 - risk_damp);

        // Override system: brake has priority if the intent is clear (>0.2).
        // KAN spline on the raw brake before the threshold — same reason as throttle/steer.
        let brake_spline = racer.kan_out.eval(2, intent[2]);
        // Brake symmetric: central ±0.05 dead-zone instead of the asymmetric 0.2 threshold.
        // Allows fine modulation but avoids the constant-brake of freshborn cars (spline≈0).
        // See melhorias/gain-asymmetry.md §Throttle → brake asymmetry.
        #[cfg(feature = "brake_symmetric")]
        let raw_brake = {
            let dead = 0.05_f32;
            if brake_spline > dead {
                ((brake_spline - dead) / (1.0 - dead)).clamp(0.0, 1.0)
            } else {
                0.0
            }
        };
        #[cfg(not(feature = "brake_symmetric"))]
        let raw_brake = {
            // FIX: it was a 0.2 threshold — asymmetric vs the always-engaged throttle pedal.
            // Cars did not brake even with brake priming +14/+15 because the cortex output
            // rarely exceeds 0.2. Symmetric 0.05 dead-zone (same as brake_symmetric).
            let dead = 0.05_f32;
            if brake_spline > dead {
                ((brake_spline - dead) / (1.0 - dead)).clamp(0.0, 1.0)
            } else {
                0.0
            }
        };
        let combined_brake = raw_brake.clamp(0.0, 1.0);
        let mut brake_intent = if speed < 2.0 { 0.0 } else { combined_brake };
        let mut pedal_intent_override: Option<f32> = None;

        // ================================================================
        // === CURVATURE SPEED GOVERNOR (physics-based, NOT evolved)     ===
        // ================================================================
        // Tesla doesn't evolve braking — it calculates v² ≤ μ·g·r.
        // Estimate turn radius from curvature probes, compute max safe
        // cornering speed, brake proportionally if overspeeding.
        //
        // r_est = ahead_dist / |angle|  (arc approximation)
        // v_safe = sqrt(μ · g · r_est × C)
        //
        // C (concentration) corrects for tight turns packed into a fraction
        // of the lookahead distance. Short probes (40m) give more accurate
        // radii than long ones (120m). 20m probe skipped — too sensitive
        // to moderate curves that the car can actually negotiate.
        //
        // Calibration: hairpin r_real≈17m. 40m probe θ=0.79 → r_est=50.6m.
        // C=0.5 → r_eff=25.3m → v_safe=15.8m/s. Progressive braking from
        // 25→16 m/s over the approach. Neural net adds on top via gate.
        {
            const G: f32 = 9.81;
            const MIN_ANGLE: f32 = 0.05; // ~2.9° — below this, road is straight
            const GOV_MIN_SPEED: f32 = 5.0; // don't govern below 18 km/h

            // Per-probe: (distance, angle, concentration_factor)
            // Short probes: C closer to 1.0 (arc approx is more accurate)
            // Long probes: C lower (curve may be packed into fraction of d)
            let probes: [(f32, f32, f32); 3] = [
                (40.0, c40_signed.abs(), 0.5),   // primary braking zone
                (70.0, c70_signed.abs(), 0.45),  // pre-decel zone
                (120.0, c120_signed.abs(), 0.4), // early warning
            ];

            if speed > GOV_MIN_SPEED {
                let mut gov_brake = 0.0_f32;

                for &(dist, angle, concentration) in &probes {
                    if angle < MIN_ANGLE {
                        continue;
                    }

                    let r_eff = (dist / angle) * concentration;
                    let v_safe = (G * r_eff).sqrt(); // μ=1.0 implicit

                    if speed > v_safe {
                        let overspeed = (speed - v_safe) / v_safe;
                        let time_to_curve = dist / speed;

                        let urgency = if time_to_curve < 1.0 {
                            1.0
                        } else if time_to_curve < 3.0 {
                            1.0 - (time_to_curve - 1.0) / 2.0 * 0.7
                        } else {
                            0.3
                        };

                        let probe_brake = (overspeed * urgency).clamp(0.0, 1.0);
                        gov_brake = gov_brake.max(probe_brake);
                    }
                }

                brake_intent = brake_intent.max(gov_brake);

                if gov_brake > 0.5 {
                    pedal_intent_override = Some(0.0);
                }
            }
        }

        // SENTINEL ANTI-COLLISION (ALWAYS-ON) — Tesla AEB-like emergency brake.
        // Physical rule: the car CANNOT crash. If TTC < threshold, force brake +
        // throttle cut. Does not depend on DNA, it is pure geometric prediction.
        //
        // Multi-tier severity:
        //   TTC < 0.3s → brake 1.0 + throttle 0 (last resort)
        //   TTC < 0.6s → brake 0.8 (preemptive)
        //   TTC < 1.0s + closing → brake 0.5 (early warning)
        {
            // [SENTINEL DISABLED BY REQUEST]
            // The automatic emergency brake (AEB) was making the model "freeze"
            // near walls, preventing it from learning to dodge on its own.
            // The model now has total freedom to crash or dodge using the neural network.

            /*
            const SENTINEL_TTC_HARD: f32 = 0.3;
            const SENTINEL_TTC_BRAKE: f32 = 0.6;
            const SENTINEL_TTC_WARN: f32 = 1.0;
            const SENTINEL_MIN_SPEED: f32 = 3.0; // m/s ~ 11 km/h
            if speed > SENTINEL_MIN_SPEED {
                // Grid-based forward distance (no raycast)
                let mut fwd = 50.0f32;
                for sample_dist in [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0].iter() {
                    let probe = pos + transform.right().as_vec3() * sample_dist;
                    if world.get_voxel_at(probe.x, probe.z) == VoxelType::Wall
                       || world.get_voxel_at(probe.x, probe.z) == VoxelType::Hole {
                        fwd = *sample_dist;
                        break;
                    }
                }
                let ttc = fwd / speed;
                if ttc < SENTINEL_TTC_HARD {
                    brake_intent = 1.0;
                    pedal_intent_override = Some(0.0);
                } else if ttc < SENTINEL_TTC_BRAKE {
                    let severity = 1.0 - (ttc - SENTINEL_TTC_HARD)
                        / (SENTINEL_TTC_BRAKE - SENTINEL_TTC_HARD);
                    brake_intent = brake_intent.max(0.8 * severity);
                } else if ttc < SENTINEL_TTC_WARN {
                    let severity = 1.0 - (ttc - SENTINEL_TTC_BRAKE)
                        / (SENTINEL_TTC_WARN - SENTINEL_TTC_BRAKE);
                    brake_intent = brake_intent.max(0.5 * severity);
                }
            }
            // PATH PROJECTION SENTINEL (grid-based trajectory predict, no raycast).
            if speed > 5.0 {
                let steer_now = racer.memory[1]; // -1..+1 last steer
                let yaw_rate = steer_now * speed * 0.05; // approx rad/s
                let mut sim_x = pos.x;
                let mut sim_z = pos.z;
                let mut sim_phase = racer.phase;
                let dt_proj = 0.4;
                let mut min_clearance = f32::INFINITY;
                for _ in 0..3 {
                    sim_phase += yaw_rate * dt_proj;
                    sim_x += sim_phase.cos() * speed * dt_proj;
                    sim_z += sim_phase.sin() * speed * dt_proj;
                    // Grid query at trajectory point
                    let mut clr = 8.0f32;
                    let sim_dir = Vec3::new(sim_phase.cos(), 0.0, sim_phase.sin());
                    for sample_dist in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].iter() {
                        let probe = Vec3::new(sim_x, pos.y, sim_z) + sim_dir * sample_dist;
                        if world.get_voxel_at(probe.x, probe.z) == VoxelType::Wall
                           || world.get_voxel_at(probe.x, probe.z) == VoxelType::Hole {
                            clr = *sample_dist;
                            break;
                        }
                    }
                    if clr < min_clearance { min_clearance = clr; }
                }
                if min_clearance < 2.0 {
                    let severity = (1.0 - min_clearance / 2.0).clamp(0.0, 1.0);
                    brake_intent = brake_intent.max(0.6 * severity);
                }
            }

            // LATERAL anti-collision (grid-based, no raycast)
            if speed > SENTINEL_MIN_SPEED {
                let left_perp = Vec3::new(-transform.right().as_vec3().z, 0.0, transform.right().as_vec3().x).normalize_or_zero();
                let right_perp = -left_perp;
                let mut lat_l = 10.0f32;
                let mut lat_r = 10.0f32;
                for sample_dist in [1.0, 3.0, 5.0, 7.0, 10.0].iter() {
                    let left_probe = pos + left_perp * sample_dist;
                    let right_probe = pos + right_perp * sample_dist;
                    if lat_l == 10.0f32 && (world.get_voxel_at(left_probe.x, left_probe.z) == VoxelType::Wall
                       || world.get_voxel_at(left_probe.x, left_probe.z) == VoxelType::Hole) {
                        lat_l = *sample_dist;
                    }
                    if lat_r == 10.0f32 && (world.get_voxel_at(right_probe.x, right_probe.z) == VoxelType::Wall
                       || world.get_voxel_at(right_probe.x, right_probe.z) == VoxelType::Hole) {
                        lat_r = *sample_dist;
                    }
                }
                let lat_min = lat_l.min(lat_r);
                if lat_min < 1.5 {
                    let severity = 1.0 - lat_min / 1.5;
                    brake_intent = brake_intent.max(0.4 * severity);
                }
            }
            */
        }

        // Modulates the accelerator: if braking, reduces the pedal proportionally.
        // Sentinel hard override (TTC <0.3s) forces pedal=0.
        let pedal_after_sentinel = pedal_intent_override.unwrap_or(pedal_intent);
        let final_pedal_intent = (pedal_after_sentinel - brake_intent).max(0.0);

        // --- EMERGENCY ESCAPE SYSTEM (ECAS) & STIGMERGY ---
        // If the wall is close (< 4m) on the side, the physical reflex FORCES the wheel to the opposite side
        // Visual feedback: draws red arrows indicating the walls' "repulsion"
        let mut emerg_steer = 0.0;
        // ECAS danger 4m→2.5m: on a narrow track, 4m was firing constantly and fighting
        // learned steering. Now only a reflex in real emergency (<2.5m lateral).
        let danger_dist = 2.5;
        // Grid-based proximity check (no raycasts)
        let left_perp = Vec3::new(
            -transform.right().as_vec3().z,
            0.0,
            transform.right().as_vec3().x,
        )
        .normalize_or_zero();
        let right_perp = -left_perp;
        let mut p_dist_l = 10.0f32;
        let mut p_dist_r = 10.0f32;
        for sample_dist in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0].iter() {
            let left_probe = pos + left_perp * sample_dist;
            let right_probe = pos + right_perp * sample_dist;
            if p_dist_l == 10.0f32
                && (world.get_voxel_at(left_probe.x, left_probe.z) == VoxelType::Wall
                    || world.get_voxel_at(left_probe.x, left_probe.z) == VoxelType::Hole)
            {
                p_dist_l = *sample_dist;
            }
            if p_dist_r == 10.0f32
                && (world.get_voxel_at(right_probe.x, right_probe.z) == VoxelType::Wall
                    || world.get_voxel_at(right_probe.x, right_probe.z) == VoxelType::Hole)
            {
                p_dist_r = *sample_dist;
            }
        }

        // ECAS IS NOW A HINT, NOT AN OVERRIDE. Severity multiplier 2.0 → 0.5.
        // The network still has to LEARN to turn; ECAS only nudges the wheel at impact time.
        if p_dist_l < danger_dist {
            let severity = 1.0 - (p_dist_l / danger_dist);
            emerg_steer -= severity * 0.5;
            gizmos.arrow(
                pos - transform.right().as_vec3() * p_dist_l,
                pos,
                Color::srgba(1.0, 0.0, 0.0, severity),
            );
        }
        if p_dist_r < danger_dist {
            let severity = 1.0 - (p_dist_r / danger_dist);
            emerg_steer += severity * 0.5;
            gizmos.arrow(
                pos + transform.right().as_vec3() * p_dist_r,
                pos,
                Color::srgba(1.0, 0.0, 0.0, severity),
            );
        }

        // ================================================================
        // === FRONTAL SAFETY & AUTOMATIC REVERSE =========================
        // ================================================================
        // Frontal probe: how close the wall is in the direction of the car's nose.
        // Two uses:
        //   1) SAFETY CAP: forward throttle rises linearly with the clearance ahead.
        //      Touching the wall => zero pedal; 6m+ of clearance => free pedal.
        //      Prevents the AI from wedging itself accelerating into concrete.
        //   2) REVERSE TRIGGER: if very close (<2m) and practically stopped,
        //      fires a 1.5s reverse_timer. During reverse, shift goes to
        //      gear 0 and the pedal applies recoil torque. Visually: orange arrow.
        const SAFE_DIST: f32 = 3.0; // minimum clearance for full throttle (narrow-track friendly)
        const REVERSE_TRIGGER_DIST: f32 = 3.0; // detects earlier (was 2.5)
        const REVERSE_MIN_DURATION: f32 = 0.4; // short tap (was 3.0)
        const REVERSE_MAX_DURATION: f32 = 1.5; // hard cap (was 5.0)
        const REVERSE_EXIT_CLEAR: f32 = 2.5; // only needs to back off 2.5m (was 4.0)
        const REVERSE_PEDAL: f32 = 0.75; // >0.5 guarantees stall recovery
        const POST_REVERSE_LOCKOUT: f32 = 0.6; // short cooldown (was 1.5)

        // Grid-based forward wall distance (no raycast)
        let mut forward_wall_dist = 30.0f32;
        for sample_dist in [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0].iter() {
            let probe = pos + transform.right().as_vec3() * sample_dist;
            if world.get_voxel_at(probe.x, probe.z) == VoxelType::Wall
                || world.get_voxel_at(probe.x, probe.z) == VoxelType::Hole
            {
                forward_wall_dist = *sample_dist;
                break;
            }
        }
        // [SAFETY CAP DISABLED] - The accelerator is no longer cut near the wall.
        // The neural network decides whether to accelerate or not.
        let safety_cap: f32 = 1.0;

        // --- CONTROLLED REVERSE (3-point turn on a narrow track) ---
        // User: "saber virar e dar ré pra ajustar ângulo se preciso" [know how to turn and
        // reverse to adjust the angle if needed].
        // STRICT trigger: frontal wall <2.5m + L+R lateral blocked (<3m
        // both sides) + speed near zero + 1s+ stuck. Avoids a forward/reverse loop.
        let mut lat_l_check = 10.0f32;
        let mut lat_r_check = 10.0f32;
        for sample_dist in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].iter() {
            let left_probe = pos + left_perp * sample_dist;
            let right_probe = pos + right_perp * sample_dist;
            if lat_l_check == 10.0f32
                && (world.get_voxel_at(left_probe.x, left_probe.z) == VoxelType::Wall
                    || world.get_voxel_at(left_probe.x, left_probe.z) == VoxelType::Hole)
            {
                lat_l_check = *sample_dist;
            }
            if lat_r_check == 10.0f32
                && (world.get_voxel_at(right_probe.x, right_probe.z) == VoxelType::Wall
                    || world.get_voxel_at(right_probe.x, right_probe.z) == VoxelType::Hole)
            {
                lat_r_check = *sample_dist;
            }
        }
        let both_lat_blocked = lat_l_check < 4.0 && lat_r_check < 4.0;
        if racer.reverse_timer <= 0.0
            && racer.post_reverse_lockout <= 0.0
            && forward_wall_dist < REVERSE_TRIGGER_DIST
            && both_lat_blocked
            && speed < 1.5
            && racer.stagnation > 60.0
        {
            racer.reverse_timer = REVERSE_MIN_DURATION;
        }
        if racer.reverse_timer > 0.0 {
            racer.reverse_timer -= dt;
            if forward_wall_dist > REVERSE_EXIT_CLEAR || racer.reverse_timer > REVERSE_MAX_DURATION
            {
                racer.reverse_timer = 0.0;
                racer.post_reverse_lockout = POST_REVERSE_LOCKOUT;
            }
        }
        if racer.post_reverse_lockout > 0.0 {
            racer.post_reverse_lockout -= dt;
        }
        let is_reversing = racer.reverse_timer > 0.0;

        // (channel 89 was already set before execute_cycle, using the previous frame's state)

        if forward_wall_dist < SAFE_DIST {
            let severity = 1.0 - safety_cap;
            let front_point = pos + transform.right().as_vec3() * forward_wall_dist;
            gizmos.arrow(
                front_point,
                pos,
                Color::srgba(1.0, 0.6, 0.0, severity.max(0.2)),
            );
        }
        if is_reversing {
            gizmos.sphere(
                pos + Vec3::new(0.0, 2.0, 0.0),
                0.5,
                Color::srgb(1.0, 0.4, 0.0),
            );
        }

        // Spectral Escape (Stigmergy) Removed - The AI now learns via the network and natural selection

        // --- TIER 3.3: EXPLORATION NOISE (curiosity > 0.7 && stuck) ---
        // When curiosity is high AND the organism is stagnant (>60 frames without max_progress),
        // injects gaussian noise into the steering to escape local optima.
        // scale 0.1 * curiosity (cap 0.1). Uses rand::rng() (consistent with motor.rs).
        let curiosity = racer.engine.metabolism.exploration_drive();
        let exploration_steer = if curiosity > 0.7 && racer.stagnation > 60.0 && !is_reversing {
            use rand::prelude::*;
            let mut rng =
                rand::rngs::StdRng::seed_from_u64(racer.stable_seed + racer.grace_ticks as u64);
            // Approximate Box-Muller via sum of 4 uniforms (determinism-friendly,
            // avoids a rand_distr dep). Mean 0, std ≈ 0.58.
            let u: f32 = (0..4).map(|_| rng.random::<f32>() - 0.5).sum();
            u * 0.1 * curiosity
        } else {
            0.0
        };

        // During reverse, forces steer = 0 (straight backup) to avoid rotating into another wall.
        // The network regains control when reverse ends.
        let final_steer_intent = if is_reversing {
            0.0
        } else {
            (steer_intent + emerg_steer + exploration_steer).clamp(-1.0, 1.0)
        };

        // Gears: smart auto-shift based on RPM and speed
        // GEARBOX-AUTHORITY ABLATION (see `neuro.rs`): zeroes only the NEURAL intent.
        // The gearbox stays whole — auto-upshift, auto-downshift and anti-stall keep
        // working. Measures exactly what the brain adds to the automatic rule.
        let ablar_marcha = crate::neuro::ablada() == crate::neuro::Via::Marcha;
        let shift_up_intent = if ablar_marcha {
            0.0
        } else {
            racer.engine.final_intent[3]
        };
        let shift_down_intent = if ablar_marcha {
            0.0
        } else {
            racer.engine.final_intent[4]
        };
        let mut shift_req = 0;

        if let Some(ref eng) = opt_engine {
            // The clutch is slipping if wheel speed is low compared to the revs
            let wheel_idx = eng.current_gear;
            let wheel_ratio = if wheel_idx <= 1 {
                0.0
            } else {
                12.0 - ((wheel_idx - 2) as f32 * 2.0)
            };
            let wheel_rpm = speed * wheel_ratio.max(3.0) * 30.0;
            let is_clutch_engaged = wheel_rpm > 1200.0 && eng.current_gear >= 2;

            // Active reverse maneuver: pushes gear to 0 (Reverse) and holds it there.
            if is_reversing {
                if eng.current_gear > 0 {
                    shift_req = -1;
                }
            }
            // Auto-clutch: leaves Neutral/Reverse into 1st ALWAYS when stopped (disabled in reverse)
            else if eng.current_gear <= 1 {
                shift_req = 1;
            }
            // Guardian Anti-Stall: if stopped but in a high gear (2nd, 3rd), DOWNSHIFT IMMEDIATELY!
            // Prevents the AI from killing the engine by sitting in long gears without moving.
            else if eng.current_gear > 2 && speed < 3.0 {
                shift_req = -1;
            }
            // Auto-upshift: shifts up ONLY if the clutch is engaged
            else if is_clutch_engaged
                && eng.current_rpm > eng.max_rpm * 0.85
                && eng.current_gear < 6
            {
                shift_req = 1;
            }
            // Auto-downshift: shifts down at critical RPM
            else if is_clutch_engaged
                && eng.current_rpm < eng.max_rpm * 0.30
                && eng.current_gear > 2
                && speed > 3.0
            {
                shift_req = -1;
            }
            // AI may override with a manual shift ONLY at engaged speed (is_clutch_engaged)
            else if is_clutch_engaged {
                if shift_up_intent > 0.5 && racer.memory[3] <= 0.5 && eng.current_gear < 6 {
                    shift_req = 1;
                } else if shift_down_intent > 0.5 && racer.memory[4] <= 0.5 && eng.current_gear > 2
                {
                    shift_req = -1;
                }
            }
        }

        racer.memory[0] = pedal_intent;
        racer.memory[1] = steer_raw; // v8: RNN state; read via ch121/slow23
        racer.memory[2] = brake_intent;
        racer.memory[3] = shift_up_intent;
        racer.memory[4] = shift_down_intent;

        // --- ACTUATOR INERTIA (Fast Hydraulic Steering) ---
        // In reverse: applies a fixed reverse pedal and releases the brake, overriding the neural intent.
        // In normal mode: multiplies by safety_cap (zero if touching the wall, 1 if >6m).
        // Pedal: during reverse, applies fixed REVERSE_PEDAL. In post-reverse lockout, caps at 0.8 (was 0.3)
        // v8b: raised from 0.3→0.8 so as not to kill acceleration during early learning.
        let pedal_target = if is_reversing {
            REVERSE_PEDAL
        } else if racer.post_reverse_lockout > 0.0 {
            (final_pedal_intent * safety_cap).min(0.8)
        } else {
            final_pedal_intent * safety_cap
        };
        let brake_target = if is_reversing { 0.0 } else { brake_intent };
        racer.memory[0] += (pedal_target - racer.memory[0]) * (dt * 10.0).min(1.0);
        racer.memory[1] += (final_steer_intent - racer.memory[1]) * (dt * 20.0).min(1.0); // From 8.0 to 20.0 (Nearly instantaneous)
        racer.memory[2] += (brake_target - racer.memory[2]) * (dt * 15.0).min(1.0);

        let mut physical_throttle = racer.memory[0];
        let physical_steer = racer.memory[1]; // No steering lock! Total focus on the curves
        let physical_brake = racer.memory[2];

        // --- DIAGNOSTIC: BRAIN SNAPSHOT (~2Hz) ---
        // Captures inputs/outputs and attenuations for later dump in the reports.
        racer.trace_timer -= dt;
        if racer.trace_timer <= 0.0 && racer.brain_trace.len() < 1800 {
            // ~30s @ 60Hz
            racer.trace_timer = 0.066; // 15Hz tracing for frame-by-frame vision debug
            // Top-5 most active channels (|value|)
            let sw = &sensory_wave;
            let mut pairs: Vec<(u8, i8)> = (0..128u8).map(|i| (i, sw[i as usize])).collect();
            pairs.sort_by(|a, b| (b.1 as i16).abs().cmp(&(a.1 as i16).abs()));
            let mut top5 = [(0u8, 0i8); 5];
            for (k, p) in pairs.iter().take(5).enumerate() {
                top5[k] = *p;
            }

            // Peak of the retina's central channels (10..22) — wall straight ahead
            let _retina_center_max = (10..22usize)
                .chain((32 + 10)..(32 + 22))
                .map(|i| sw[i].unsigned_abs() as i8)
                .max()
                .unwrap_or(0);

            // dynamic_steer_limit and risk_damp already computed above
            let slow = &racer.engine.slow_brain.output;
            let _snap_risk_damp = slow[2].max(0.0).min(0.3);
            let snap_steer_limit = 1.0 / (1.0 + speed * 0.02);
            let _snap_phys_steer = physical_steer * snap_steer_limit;

            let _engine_time = racer.engine.current_time;
            let _ri_thr = racer.engine.final_intent[0];
            let _ri_str = racer.engine.final_intent[1];
            let _ri_brk = racer.engine.final_intent[2];
            // Compress retina into FOV_BINS (already computed into sw[0..FOV_BINS])
            const FOV_BINS: usize = 8;
            const RETINA_ROWS_LOCAL: usize = 14;
            let mut retina_fov = vec![0i8; FOV_BINS * RETINA_ROWS_LOCAL];
            // sw[0..FOV_BINS] are fovea FOV bins. Distribute across rows for visualization.
            // For now, repeat the fovea bins (one per RETINA_ROW).
            for row in 0..RETINA_ROWS_LOCAL {
                for bin in 0..FOV_BINS {
                    retina_fov[row * FOV_BINS + bin] = sw[bin];
                }
            }

            // Compact sensory channels: speed, alignment, curvature, walls, etc
            // `sensory_compact` was removed: 10 channels were filled every tick and
            // NONE was ever read. The compiler had been warning for a while, hidden among 26 warnings.
            // The same channels remain available in `sensory_wave` for whoever needs them.

            // Capture sensory snapshot for visualization/diagnostics (UI panel access)
            for i in 0..64 {
                racer.sensory_snapshot[i] = sensory_wave[i];
            }

            // === VISION VALIDATION DIAGNOSTIC (disabled for now) ===
            // Debug diagnostics can be re-enabled with conditional logging if needed.
            // For now, sensory_snapshot is captured for UI visualization.

            // DEBUG: Disabled brain_trace sampling temporarily for performance profiling.
            // Re-enable after identifying bottleneck.
            /*
            racer.brain_trace.push(crate::simulation::racer::BrainSnapshot {
                t: engine_time,
                speed_kmh: speed * 3.6,
                wall_prox_l: sw[85],
                wall_prox_r: sw[86],
                ttc_inv: sw[82],
                curv_25: sw[79],
                retina_center_max,
                raw_throttle: ri_thr,
                raw_steer: ri_str,
                raw_brake: ri_brk,
                post_damp_steer: final_steer_intent,
                physical_steer: snap_phys_steer,
                steer_limit: snap_steer_limit,
                risk_damp: snap_risk_damp,
                top_channels: top5,
                retina_fov,
                sensory_compact,
            });
            */
        }

        // BOS removed — the penalty system already punishes overlap
        // The symbolic layer guarantees brake only activates on explicit intent

        // --- PROGRESSIVE MECHANICAL DAMAGE AND STRESS THERMODYNAMICS ---
        // LIMP MODE in reverse: does not multiply by engine_power, guaranteeing enough torque
        // to unstick from the wall even at low HP (keeps a minimum floor of 0.6x pedal).
        if !is_reversing {
            physical_throttle *= racer.engine_power;
        } else {
            physical_throttle *= racer.engine_power.max(0.6);
        }

        // Overlap penalty — DISABLED v7
        // Was killing post-grace when cars learned the throttle+brake combo.
        // let conflict = !is_reversing && final_pedal_intent > 0.4 && brake_intent > 0.4;
        // if conflict && racer.grace_ticks >= 2560 {
        //     racer.hp -= dt * 5.0;
        //     gizmos.sphere(pos + Vec3::new(0.0, 1.5, 0.0), 0.3, Color::srgb(1.0, 0.5, 0.0));
        // }

        // --- OVER-REV PENALTY & CLUTCH BURN — DISABLED v7
        let in_grace = racer.grace_ticks < 2560;
        // if let Some(ref eng) = opt_engine {
        //     // Redline: punish only after grace (Gen 1 needs time to learn rev control)
        //     if !in_grace && eng.current_rpm > eng.max_rpm * 0.95 {
        //         racer.hp -= dt * 5.0;
        //     }
        //     // Burnout: reduced and only outside grace
        //     if !in_grace && speed < 1.0 && eng.current_rpm > eng.max_rpm * 0.7 {
        //         racer.hp -= dt * 2.0; // was 5.0
        //     }
        // }

        // --- STAGNATION / SLOW-DRIVING PENALTY — DISABLED v7
        // Was primary post-grace killer (5 HP/s when speed < 1 m/s)
        // if !in_grace && !is_reversing && speed < 1.0 {
        //     racer.hp -= dt * 5.0;
        // }

        // --- TIRE THERMAL MODEL (now per-wheel in physics.rs) ---
        // Read the wheel average for the Racer's telemetry
        if let Some(ref body) = opt_body {
            racer.tire_temp = body.wheels.iter().map(|w| w.tire_temp).sum::<f32>() / 4.0;
        }

        // --- GRASS: Time Penalty (Constant Neural "PAIN") ---
        if voxel_current == VoxelType::Grass {
            racer.grass_time += dt;
        }

        // --- HOLE: severe damage (LIFE MODE) or instant death (default) ---
        // VoxelType::Hole under the car's center → strong penalty. Forces the network to
        // actively dodge holes (distinct retina pixel ~90, while road ~-80 and wall ~127).
        // Grace window v5: 180→640 frames (3s→10s). Gen 1 needs more time to learn.
        //
        // LIFE MODE: falling into a hole does not kill instantly — it takes 100 HP (⅓ of the
        // 300 tank). At 100 HP of damage per fall, 3 falls kill. That is still a severe
        // punishment but recoverable: a being that avoids holes after falling can regenerate
        // with progress. A 1s cooldown avoids "100 HP/s of hole" if the car gets stuck inside.
        if voxel_current == VoxelType::Hole && !racer.is_dead && racer.grace_ticks >= 2560 {
            if lab.modo_vida {
                // Hole damage: 100 HP per event (cooldown via wall_cooldown so it does not repeat).
                // wall_cooldown serves here as a generic cooldown — the car leaves the hole in <1s.
                if racer.wall_cooldown == 0.0 {
                    racer.hp -= 100.0;
                    racer.wall_cooldown = 1.0; // 1 second of cooldown between hole damage events
                    // Pre-set the death label as Hole so that if hp hits zero here,
                    // the central logic (L~2373) finds death_reason already filled and does
                    // not overwrite it with HpZero — preserving the real cause in the autopsy.
                    if racer.hp <= 0.0 && racer.death_reason.is_none() {
                        racer.death_reason = Some(crate::simulation::racer::DeathReason::Hole);
                    }
                    println!(
                        "⚠️ {} HOLE (modo_vida) hp={:.0} frame {}",
                        racer.name, racer.hp, racer.grace_ticks
                    );
                }
                // If HP reached zero from the fall, normal death via the central logic (L~2373).
            } else {
                racer.is_dead = true;
                racer.hp = 0.0;
                racer.death_reason = Some(crate::simulation::racer::DeathReason::Hole);
                racer.death_frame = racer.grace_ticks;
                racer.death_pos = pos;
                race.death_markers.push(pos);
                if let Some(mat) = materials.get_mut(mat_handle.0.id()) {
                    mat.base_color = Color::BLACK;
                }
                commands.entity(entity).insert(Visibility::Hidden);
                println!(
                    "💀 {} HOLE frame {} pos=({:.1},{:.1})",
                    racer.name, racer.grace_ticks, pos.x, pos.z
                );
            }
        }

        // --- NPC HIT — DISABLED v7
        // if !racer.is_dead && racer.grace_ticks >= 2560 && env.npc_hit(Vec2::new(pos.x, pos.z)) && racer.wall_cooldown == 0.0 {
        //     racer.telemetry.hit_npc_count += 1;
        //     racer.wall_cooldown = 0.8;
        //     racer.hp -= 50.0;
        //     if racer.hp <= 0.0 {
        //         racer.is_dead = true;
        //         racer.hp = 0.0;
        //         racer.death_reason = Some(crate::simulation::racer::DeathReason::NpcHit);
        //         racer.death_frame = racer.grace_ticks;
        //         racer.death_pos = pos;
        //         race.death_markers.push(pos);
        //         if let Some(mat) = materials.get_mut(mat_handle.0.id()) { mat.base_color = Color::BLACK; }
        //         commands.entity(entity).insert(Visibility::Hidden);
        //         println!("💀 {} NPC_HIT frame {}", racer.name, racer.grace_ticks);
        //     }
        // }

        // --- CASTER DYNAMICS (SPEED-DEPENDENT STEERING WEIGHT) ---
        // Steer_limit = MaxSteer / (1.0 + speed * 0.02)
        // Prevents turning 90° at 200km/h, forcing F1 smoothness
        let dynamic_steer_limit = 1.0 / (1.0 + speed * 0.02);

        // engine_power degrades with HP (Progressive Damage). If HP hits zero, the engine dies (0.0 power).
        // FIXED: normalizes by HP_INICIAL instead of 100.0. With HP_INICIAL=1000 and a fixed 100.0
        // divisor, engine_power would stay pinned at 1.0 until HP fell below 100 — the being would feel
        // no degradation at all through the first 900 HP of damage. Now degradation starts early and is proportional.
        racer.engine_power = (racer.hp / crate::simulation::racer::HP_INICIAL).clamp(0.0, 1.0);

        // Death by CATASTROPHIC mechanical failure (HP <= 0.0) or time running out.
        if racer.hp <= 0.0 && !racer.is_dead {
            racer.is_dead = true;
            if racer.death_reason.is_none() {
                racer.death_reason = Some(crate::simulation::racer::DeathReason::HpZero);
                racer.death_frame = racer.grace_ticks;
                racer.death_pos = pos;
            }
            // Creates a PAIN MEMORY in the world for the next generations
            race.death_markers.push(pos);

            if let Some(mat) = materials.get_mut(mat_handle.0.id()) {
                mat.base_color = Color::BLACK;
            }
            // --- HIDE-ON-DEATH: frees the screen with 500 cars ---
            // We keep the entity (the pipeline needs to read Racer for fitness/pool),
            // but hide the mesh + children. InheritedVisibility propagates to the wing,
            // cockpit and headlight. Render cost: zero.
            commands.entity(entity).insert(Visibility::Hidden);
        } else if !racer.is_dead {
            // Color reflects mechanical state: red = damaged, cyan normal
            if let Some(mat) = materials.get_mut(mat_handle.0.id()) {
                let health_ratio =
                    (racer.hp / crate::simulation::racer::HP_INICIAL).clamp(0.0, 1.0);
                mat.base_color = Color::srgb(1.0, health_ratio, health_ratio);
            }
        }

        // --- 4. ENGINE AND PHYSICS ---
        let flow_pull = (track_phase - racer.phase).sin() * dev.flow_influence;
        let steering_speed = (physical_steer * dynamic_steer_limit) * 2.5 + flow_pull;

        if let (Some(engine_comp), Some(body)) = (&mut opt_engine, &mut opt_body) {
            engine_comp.throttle = physical_throttle;
            engine_comp.brake = physical_brake;
            // Limits the actual rear-wheel angle depending on speed
            engine_comp.steer = physical_steer * dynamic_steer_limit;
            engine_comp.manual_shift_req = shift_req;

            // Physics system (physics.rs) handles: throttle→torque→forces→integration
            // Just set the control inputs; physics.rs will calculate realistic response

            // Stalled engine: does NOT kill instantly!
            // The AI needs to learn the start sequence (downshift to gear 0/1 + hold the accelerator).
            // Drains HP slowly (starter battery) as evolutionary pressure to restart quickly.
            // Grace period: Gen 1 may have a stalled engine during bootstrap without battery drain.
            if !in_grace && engine_comp.is_stalled {
                racer.hp -= if lab.modo_vida { 0.1 } else { 1.0 } * dt; // life mode: cost, not execution
            }

            // PHYSICAL YAW ENABLED!
            // The car's rotation is now dominated entirely by the tire forces in physics.rs.
            // The AI turns the front wheel -> generates lateral tire force -> generates torque -> turns the chassis.

            racer.amplitude = body.velocity.length();
            if racer.amplitude > racer.peak_speed {
                racer.peak_speed = racer.amplitude;
            }
            racer.phase = transform.rotation.to_euler(EulerRot::YXZ).0 * -1.0;

            // --- TIER 1.1: DENSE TELEMETRY (reward shaping) ---
            // Samples every fixed tick (64Hz). voxel==Road only rewards useful exploration (anti-novelty-exploit).
            let tel_on_road = voxel_current == VoxelType::Road;
            racer.telemetry.record(
                final_steer_intent,
                body.velocity.length(),
                body.angular_velocity.y,
                tel_on_road,
                transform.translation.x,
                transform.translation.z,
            );

            // --- REPLAY SAMPLING @ 16Hz, TOP-5 RACERS ---
            if *retina_tick_counter % 4 == 0 {
                let my_rank = rank_by_entity.get(&entity).copied().unwrap_or(999);
                if my_rank < 5 {
                    use crate::evolution::telemetry::{ReplayFrame, name_to_id};
                    let yaw = transform.rotation.to_euler(EulerRot::YXZ).0;
                    let frame = ReplayFrame {
                        frame_tick: racer.grace_ticks,
                        generation: race.generation as u16,
                        rank: my_rank as u8,
                        flags: if racer.is_dead { 1 } else { 0 },
                        pos_x: pos.x,
                        pos_z: pos.z,
                        heading: yaw,
                        speed_ms: racer.amplitude,
                        throttle: racer.memory[0],
                        steer: racer.memory[1],
                        brake: racer.memory[2],
                        hp: racer.hp,
                        distance_m: racer.distance,
                        track_progress: racer.track_progress,
                        total_laps: racer.total_laps as u16,
                        racer_id: name_to_id(&racer.name),
                        joy_tick: sensory_wave[126] as f32 / 127.0,
                        sensory_wave,
                        _reserved: [0u8; 8],
                    };
                    race.telemetry.write_frame(&frame);

                    // --- NEW REPLAY (128 bytes/frame, crate::replay format) ---
                    // Records in parallel with the legacy replay for motor vs cortex_evolve comparison.
                    // Read by: `replay_analyze --replay telemetry/gen_N/motor.replay`
                    let axiom_str_q = (race.current_axiom_strength * 127.0).clamp(0.0, 127.0) as u8;
                    if let Some(ref mut rw) = race.replay_writer {
                        // Normalized 2D forward for the HTML player (2D canvas uses x/z = x/y)
                        let fwd_3 = transform.right().as_vec3();
                        let fwd_2 = [fwd_3.x, fwd_3.z];

                        // Safety flags: bit0=governor (pedal_override present), bit1=sentinel,
                        // bit2=ecas (emerg_steer active), bit3=reversing
                        // Sentinel is disabled on this path — bit1 always 0 here.
                        let sf: u8 = (pedal_intent_override.is_some() as u8)
                            | 0u8 << 1  // sentinel disabled in this code
                            | ((emerg_steer.abs() > 0.01) as u8) << 2
                            | (is_reversing as u8) << 3;

                        // flow_align: sensory_wave[64] = (alignment * 127) as i8
                        let flow_align = sensory_wave[64] as f32 / 127.0;

                        // Compact retina: samples 8 equally spaced channels along the horizon
                        let step_fov = (RETINA_COLS / 8).max(1);
                        let horizon = RETINA_ROWS / 2;
                        let mut retina_fov = [0i8; 8];
                        let mut retina_peri = [0i8; 8];
                        for k in 0..8 {
                            retina_fov[k] = retina_binocular[horizon * RETINA_COLS + k * step_fov];
                            let peri_row = (RETINA_ROWS * 3 / 4).min(RETINA_ROWS - 1);
                            retina_peri[k] =
                                retina_binocular[peri_row * RETINA_COLS + k * step_fov];
                        }

                        // Saturated dimensions in final_intent (|x| > 0.95)
                        let dims_sat = racer
                            .engine
                            .final_intent
                            .iter()
                            .filter(|&&v| v.abs() > 0.95)
                            .count()
                            .min(255) as u8;

                        let replay_frame = crate::replay::ReplayFrame {
                            pos: [pos.x, pos.y, pos.z],
                            fwd: fwd_2,
                            speed,
                            yaw_rate: body.angular_velocity.y,
                            arc_prog: racer.track_progress,
                            laps: racer.total_laps as u16,
                            gear: opt_engine
                                .as_ref()
                                .map(|e| e.current_gear as u8)
                                .unwrap_or(0),
                            hp_norm: ((racer.hp / crate::simulation::racer::HP_INICIAL) * 127.0)
                                .clamp(0.0, 127.0) as u8,
                            fuel_norm: ((racer.fuel / crate::simulation::racer::FUEL_INICIAL)
                                * 127.0)
                                .clamp(0.0, 127.0) as u8,
                            tick: racer.grace_ticks,
                            dist_wall_l: dist_l,
                            dist_wall_r: dist_r,
                            dist_wall_f: forward_wall_dist,
                            curv_40m: (c40_signed / std::f32::consts::PI).clamp(-1.0, 1.0),
                            curv_70m: (c70_signed / std::f32::consts::PI).clamp(-1.0, 1.0),
                            curv_120m: (c120_signed / std::f32::consts::PI).clamp(-1.0, 1.0),
                            flow_align,
                            line_offset: line_offset_m,
                            intent_thr: pedal_intent,
                            intent_str: steer_intent,
                            intent_brk: combined_brake,
                            final_thr: physical_throttle,
                            final_str: physical_steer,
                            final_brk: physical_brake,
                            dims_saturated: dims_sat,
                            intent_mag_q: ((racer
                                .engine
                                .final_intent
                                .iter()
                                .map(|v| v.abs())
                                .sum::<f32>()
                                / racer.engine.final_intent.len().max(1) as f32)
                                * 127.0)
                                .clamp(0.0, 127.0) as u8,
                            axiom_str_q,
                            safety_flags: sf,
                            retina_fov,
                            retina_peri,
                            _pad: [0u8; 8],
                        };
                        let _ = rw.write_frame(my_rank as u8, &replay_frame);
                    }
                }
            }

            // True distance: projection of velocity onto the track direction (Flow)
            let flow_3d = Vec3::new(flow.x, 0.0, flow.y).normalize_or_zero();
            let track_velocity = body.velocity.dot(flow_3d);

            // Fitness punishment for Zigzag + Grass Penalty
            let lateral_velocity = body.velocity.cross(flow_3d).length();

            // ONLY ACCUMULATES DISTANCE IF NOT ON THE WALL
            // This completely eliminates the "productive ricochet" strategy
            let on_wall = world.get_voxel_at(transform.translation.x, transform.translation.z)
                == VoxelType::Wall;
            if !on_wall {
                racer.distance += (track_velocity - lateral_velocity * 0.5) * dt;
            }

            // --- PATH SAMPLING (~4Hz) to draw the champion's line ---
            racer.path_timer -= dt;
            if racer.path_timer <= 0.0 {
                racer.path_timer = 0.25;
                if racer.path.len() < 1500 {
                    racer.path.push(transform.translation);
                }
            }

            // --- ANTI-PLATEAU / ANTI-DONUT ---
            // Going 10s without gaining 3m → drains 10HP/s (death in ~10s of looping).
            // Loosened for narrow tracks: the car needs time to maneuver in a tight curve,
            // and 5s/20HP/s used to kill anyone who stopped to think in a 90° L.
            if racer.track_progress > racer.progress_checkpoint + 3.0 {
                racer.progress_checkpoint = racer.track_progress;
                racer.progress_stall = 0.0;
            } else {
                racer.progress_stall += dt;
            }
            if !in_grace && !is_reversing && racer.progress_stall > 10.0 {
                // LIFE MODE: 0.2 HP/s (was 1.0 HP/s).
                // At HP_INICIAL=1000 and 0.2 HP/s it would take ~5000 s to kill — never kills on its own.
                // It was 1.0 HP/s → killed in ~300 s (with HP=300), still a hidden executioner.
                // The real pressure in modo_vida is HUNGER (fuel), not the anti-donut.
                racer.hp -= if lab.modo_vida { 0.2 } else { 10.0 } * dt;
            }

            // Severe Punishment for Driving the Wrong Way — DISABLED v7
            // Was killing post-grace cars for backing up. With base throttle bias, forward should be natural.
            // if !in_grace && !is_reversing && racer.distance < -5.0 {
            //     racer.hp -= 100.0 * dt;
            // }

            // Punishment for doing donuts (Spinning in place)
            // Grace period: Gen 1 may spin during bootstrap without dying.
            if !in_grace && body.angular_velocity.y.abs() > 1.5 && track_velocity.abs() < 5.0 {
                // LIFE MODE: 0.5 HP/s (was 3.0 HP/s → killed in ~100 s of continuous donut with HP=300).
                // At 0.5 HP/s and HP_INICIAL=1000 it would take ~2000 s of UNINTERRUPTED spinning to kill.
                // In practice the fuel runs out first: ~150 s without a lap = FUEL. Donut becomes a fine, not death.
                racer.hp -= if lab.modo_vida { 0.5 } else { 30.0 } * dt;
            }

            // Tire Wear Punishment (Thermal Destruction): Avoid Free Burnouts
            // If the wheels are slipping wildly (huge Slip Ratio) and the car barely advances
            // Grace period: Gen 1 may skid during initial spin-up without penalty.
            let avg_slip = body
                .wheels
                .iter()
                .map(|w| w.slip_ratio.abs() + w.slip_angle.abs())
                .sum::<f32>()
                / 4.0;
            if !in_grace && avg_slip > 0.5 && track_velocity.abs() < 15.0 {
                // "Blowing the Tires": drains life if skidding pointlessly while accelerating in place.
                // LIFE MODE: avg_slip × 0.2 HP/s (was avg_slip × 1.0 HP/s).
                // With typical avg_slip of ~1.0 and HP_INICIAL=1000: 0.2 HP/s → ~5000 s to kill.
                // At 1.0 HP/s the same slip killed in ~300 s (with HP=300) — another hidden executioner.
                racer.hp -= avg_slip * if lab.modo_vida { 0.2 } else { 10.0 } * dt;
            }

            // === DRIFT REWARD (clean drifting) ===
            // Car sliding BUT keeping speed in the track direction = bonus dopamine.
            // Requires: high average slip_angle + real forward speed + engaged throttle + ok alignment.
            // Stationary burnout (above) keeps being punished — the two do not overlap.
            let avg_slip_angle = body.wheels.iter().map(|w| w.slip_angle.abs()).sum::<f32>() / 4.0;
            if avg_slip_angle > 0.15
                && track_velocity > 10.0
                && pedal_intent > 0.3
                && alignment > 0.3
            {
                // Scale: drift magnitude × speed × alignment with the track.
                // Cap at 0.6 rad (~34°) so as not to reward absurd spins.
                let drift_bonus = avg_slip_angle.min(0.6) * track_velocity * alignment * 0.08;
                racer.joy_integral += drift_bonus * dt;
                // Visible smoke: doubles skidmark chance during a clean drift.
                if rng.random::<f32>() < 0.25 {
                    racer.tire_temp = (racer.tire_temp + 0.02 * dt).min(1.0);
                }
            }

            // === DRAFTING: 30% drag reduction if behind another car ===
            if in_draft {
                let drag_reduction =
                    body.velocity.normalize_or_zero() * body.velocity.length_squared() * 0.18 * 0.3;
                body.forces += drag_reduction;
            }
        } else {
            // Fallback kinematic
            let target_amp = pedal_intent * dev.wave_amplitude;
            racer.amplitude = racer.amplitude + (target_amp - racer.amplitude) * 2.0 * dt;
            racer.phase = racer.phase + steering_speed * dt;
            let wave_vector =
                Vec3::new(racer.phase.cos(), 0.0, racer.phase.sin()) * racer.amplitude;
            transform.translation += wave_vector * dt;
            transform.rotation = Quat::from_rotation_y(-racer.phase);
            racer.distance += (wave_vector * dt).length();
        }

        racer.wall_cooldown = (racer.wall_cooldown - dt).max(0.0);
        let mut hit_wall = false;

        // Checks collisions using the same physical probes so the bounce cannot save it
        if world.get_voxel_at(transform.translation.x, transform.translation.z) == VoxelType::Wall {
            hit_wall = true;
        } else {
            // If the center is free, checks the 4 corners (nose, rear, sides)
            // Approximate chassis constants (wheelbase, track_width) — ignores opt_body for now.
            let wb = 2.6_f32;
            let tw = 1.6_f32;

            let pos = transform.translation;
            let forward = transform.right().as_vec3();
            let right = -transform.back().as_vec3();

            let probe_points = [
                pos + forward * (wb * 0.5), // Nose
                pos - forward * (wb * 0.5), // Rear
                pos + right * (tw * 0.5),   // Right side
                pos - right * (tw * 0.5),   // Left side
            ];

            for p in &probe_points {
                if world.get_voxel_at(p.x, p.z) == VoxelType::Wall {
                    hit_wall = true;
                    break;
                }
            }
        }

        // GRACE in REAL frames — virtual time is chaotic in the first frames (irregular
        // FPS in a large swarm, Bevy's max_delta, etc).
        // v3: 3200 frames (50s). It was 7200 (112.5s) = 93.7% of the generation protected,
        // creatures crawling at 1-2 m/s became immortal. Now 50s of bootstrap +
        // 70s of real selective pressure (speed bleed, wall damage, etc.).
        let out_of_grace = racer.grace_ticks >= 3200;

        // === POST-LAP GRACE (computed early — the wall check below needs HP grace) ===
        if racer.post_lap_grace > 0 {
            racer.post_lap_grace = racer.post_lap_grace.saturating_sub(1);
        }
        let advanced_past_line = racer.post_lap_checkpoint > 0.0
            && racer.track_progress >= racer.post_lap_checkpoint + 25.0;
        let in_post_lap_grace =
            racer.post_lap_grace > 0 || (racer.post_lap_checkpoint > 0.0 && !advanced_past_line);
        let in_post_lap_hp_grace = racer.post_lap_grace > 480;

        if hit_wall && !racer.is_dead {
            racer.wall_time += dt; // capped dt; wall_time in REAL seconds (slow, stable).
            // PINADO: 1.6s REAL scraping the wall = death (was 0.8s). More tolerance.
            // LIFE MODE: scraping the wall does not kill, it hurts. Crashing becomes a cost, not the end.
            let pinned =
                out_of_grace && !in_post_lap_hp_grace && racer.wall_time > 1.6 && !lab.modo_vida;
            // LIFE MODE: scraping costs 0.3 HP/s (was 2.0 HP/s → killed in ~150 s continuous with HP=300).
            // At 0.3 HP/s and HP_INICIAL=1000 it would take ~3333 s of UNINTERRUPTED scraping to kill.
            // Scraping the wall is a real cost but not a sentence — whoever dies by wall now
            // did so because they stayed pressed against it for several minutes without getting out.
            // Death label: it was possible for wall damage to zero the HP and the death to come out
            // as HpZero (detected at L2373 before any specific label), since the
            // `is_dead = true` here only happens if hp<=0.0 after the `max(0.0)`. FIXED:
            // we set death_reason BEFORE checking is_dead (Pinado label guaranteed).
            if lab.modo_vida && out_of_grace && racer.wall_time > 1.6 {
                racer.hp = (racer.hp - 0.3 * dt).max(0.0);
                if racer.hp <= 0.0 && !racer.is_dead {
                    racer.is_dead = true;
                    racer.death_reason = Some(crate::simulation::racer::DeathReason::Pinado);
                    racer.death_frame = racer.grace_ticks;
                    racer.death_pos = pos;
                    race.death_markers.push(pos);
                    commands.entity(entity).insert(Visibility::Hidden);
                }
            }
            if pinned {
                racer.is_dead = true;
                racer.hp = 0.0;
                racer.death_reason = Some(crate::simulation::racer::DeathReason::Pinado);
                racer.death_frame = racer.grace_ticks;
                racer.death_pos = pos;
                race.death_markers.push(pos);
                if let Some(mat) = materials.get_mut(mat_handle.0.id()) {
                    mat.base_color = Color::BLACK;
                }
                commands.entity(entity).insert(Visibility::Hidden);
                println!(
                    "💀 {} PINADO frame {} (wall_time={:.2}s)",
                    racer.name, racer.grace_ticks, racer.wall_time
                );
            } else if racer.wall_cooldown == 0.0 {
                racer.wall_hits += 1;
                racer.wall_cooldown = 0.6;
                // Damage proportional to the NORMAL component of the velocity into the wall.
                // Parallel scrape (velocity dot wall_normal ≈ 0) = 8 dmg.
                // Head-on hit at 30m/s = 60 dmg (out_of_grace) or 35 (grace).
                // Approximation: uses lateral_vel (normal to the track) as a proxy for the normal velocity.
                let impact_speed = if let Some(ref body) = opt_body {
                    let lateral_dir = transform.back().as_vec3();
                    body.velocity.dot(lateral_dir).abs()
                } else {
                    10.0
                };
                // WALL DAMAGE TURNED OFF (v7 decision: it was the main post-grace killer).
                //
                // The full damage computation lived here — impact speed, per-phase
                // ceiling, post-lap attenuation — and ended in a commented-out line
                // (`// racer.hp -= dmg;`). Ten lines producing a discarded number.
                //
                // Consequence the number hides and is worth recording: without this damage the
                // beings practically never die from crashing (3 crashes in 11,450 measured
                // deaths; 98.7% die from getting slow). If it ever comes back, it comes back as
                // a measured parameter, not as a commented-out line.
                let _ = (impact_speed, in_post_lap_hp_grace);
                if racer.hp <= 0.0 {
                    racer.is_dead = true;
                    racer.hp = 0.0;
                    racer.death_reason = Some(crate::simulation::racer::DeathReason::Batida);
                    racer.death_frame = racer.grace_ticks;
                    racer.death_pos = pos;
                    race.death_markers.push(pos);
                    if let Some(mat) = materials.get_mut(mat_handle.0.id()) {
                        mat.base_color = Color::BLACK;
                    }
                    commands.entity(entity).insert(Visibility::Hidden);
                    println!(
                        "💀 {} BATIDA frame {} ({} hits)",
                        racer.name, racer.grace_ticks, racer.wall_hits
                    );
                } else if let Some(mat) = materials.get_mut(mat_handle.0.id()) {
                    let health_ratio =
                        (racer.hp / crate::simulation::racer::HP_INICIAL).clamp(0.0, 1.0);
                    mat.base_color = Color::srgb(1.0, health_ratio, health_ratio);
                }
            }
        } else {
            // Slow decay: dt * 0.1 = needs ~10 continuous real seconds without a hit to zero out.
            racer.wall_time = (racer.wall_time - dt * 0.1).max(0.0);
        }

        // --- CHECKPOINT RECOVERY (HEAL GATE) ---
        // Every 25m of track (real arc-length), a checkpoint is reached.
        // User feedback: cars were dying before reaching the next landmark.
        //  - Interval 50→25m: more frequent healing, they survive while learning.
        //  - HP boost 25→40: compensates a wall hit (60 dmg) in 2 waypoints (before: 3).
        //  - Full stagnation reset: stagnation/progress_stall/progress_checkpoint.
        //    Before, only wall_hits was zeroed — the car still died from old progress_stall
        //    even when crossing a waypoint.
        let current_wp = (racer.track_progress / 25.0).floor().max(0.0) as usize;

        if current_wp > racer.last_waypoint {
            racer.last_waypoint = current_wp;
            // HEALING CEILING = BIRTH HP, not a fixed 100.
            // It was `.min(100.0)`, and that made `HP_INICIAL` nearly INERT: whatever
            // the birth value (100,000 or 300), the first pass through a
            // checkpoint pinned the HP at 100 and the real scale went back to 100 forever.
            // Measured: the death distribution is identical with HP=100,000 and with HP=300
            // (HP ~59%, SpeedBleed ~40% in both), which is only possible if the birth
            // value does not matter.
            let teto = crate::simulation::racer::HP_INICIAL;
            racer.hp = (racer.hp + 0.4 * teto).min(teto);
            racer.wall_hits = 0;
            racer.stagnation = 0.0;
            racer.progress_stall = 0.0;
            racer.progress_checkpoint = racer.track_progress;
            if let Some(mat) = materials.get_mut(mat_handle.0.id()) {
                mat.base_color = Color::srgb(0.5, 1.0, 1.0);
            }
        }

        // === SLOW REGENERATION (LIFE MODE) ===
        // When the being is PROGRESSING on the track (advanced >0.05 m since the previous frame),
        // it recovers HP slowly. This makes damage RECOVERABLE: a scratch on the wall is not
        // a countdown clock — as long as the being keeps moving.
        //
        // Rate: 1.0 HP/s while advancing. At HP_INICIAL=1000:
        //   - a being that ALWAYS moves regenerates 1000 HP in 1000 s (≈14 laps of ~70 s).
        //   - for practical purposes: each ~70 s lap = +70 HP of regeneration.
        // Since the summed drains in the worst case (scraping wall 0.3 + donut 0.5 + stag 0.5)
        // reach 1.3 HP/s, a being that advances EVEN while taking hits sits at a near-perfect
        // equilibrium (−0.3 HP/s net). Without advancing, the FUEL runs out first (~150 s of tank).
        //
        // Condition: requires modo_vida + out_of_grace + real progress in the frame.
        // Does not regenerate in grace so as not to mask bootstrap problems.
        if lab.modo_vida && out_of_grace && !racer.is_dead {
            let advancing = racer.track_progress > racer.progress_checkpoint;
            if advancing {
                let teto_regen = crate::simulation::racer::HP_INICIAL;
                racer.hp = (racer.hp + 1.0 * dt).min(teto_regen);
            }
        }

        // === MONOTONIC STAGNATION (in real FRAMES) ===
        // max_progress ONLY GROWS. No advance > 2m in 180 frames (≈3 real seconds) = death.
        // Frame-based = stable regardless of time_scale / FPS / Bevy's max_delta.
        let prev_max = racer.max_progress;
        racer.max_progress = racer.max_progress.max(racer.track_progress);
        if out_of_grace && !in_post_lap_grace {
            if racer.max_progress > prev_max + 0.01 {
                // v2: checkpoint 2m→0.5m (stagnation resets on fine progress, not just big chunks).
                // Cars that go slowly but advance do not die from STAG.
                if racer.max_progress > racer.progress_checkpoint + 0.5 {
                    racer.progress_checkpoint = racer.max_progress;
                    racer.stagnation = 0.0;
                } else {
                    racer.stagnation += 1.0;
                }
            } else {
                racer.stagnation += 1.0;
            }
            // v8: Stagnation kill RE-ENABLED. Cars that do not advance 0.5m in 480 frames
            // (≈7.5s) die. Without this, cars sit against walls until the 120s
            // timeout, wasting evolution time.
            // LIFE MODE: stagnating does not kill IMMEDIATELY, it drains 0.5 HP/s (was 3.0 HP/s).
            // It was `.max(1.0)` — it kept HP at 1 forever, pinning engine_power at
            // 0.01: the being stayed "alive" but engineless, stuck indefinitely. FIXED: uses
            // `.max(0.0)` and lets the being die via the central hp<=0 logic at L~2373.
            // At 0.5 HP/s and HP_INICIAL=1000: ~2000 s of TOTAL stagnation to kill (HUNGER
            // normally arrives first at the tank's ~150 s, so stag is never the real clock).
            if racer.stagnation > 480.0 && lab.modo_vida {
                racer.hp = (racer.hp - 0.5 * dt).max(0.0);
            } else if racer.stagnation > 480.0 {
                racer.is_dead = true;
                racer.hp = 0.0;
                racer.death_reason = Some(crate::simulation::racer::DeathReason::Stagnation);
                racer.death_frame = racer.grace_ticks;
                racer.death_pos = pos;
                race.death_markers.push(pos);
                if let Some(mat) = materials.get_mut(mat_handle.0.id()) {
                    mat.base_color = Color::BLACK;
                }
                commands.entity(entity).insert(Visibility::Hidden);
            }
        } else {
            racer.stagnation = 0.0;
            racer.progress_checkpoint = racer.max_progress;
        }

        // === SPEED BLEED (RE-ENABLED v8) ===
        // Penalizes stopped cars. Forces evolution: learn to move or die.
        let cur_speed = if let Some(ref body) = opt_body {
            body.velocity.length()
        } else {
            0.0
        };
        // v3: speed threshold 2.0→4.0 (crawlers at 1-2 m/s now count as stalled).
        // making_progress threshold 0.005→0.05 (require meaningful forward motion, not micro-creep).
        let making_progress = racer.max_progress > prev_max + 0.05;
        if out_of_grace && !in_post_lap_grace && cur_speed < 4.0 && !making_progress {
            racer.slow_time += 1.0;
            if racer.slow_time > 180.0 {
                // v8: 360→180 frames (~2.8s) before speed bleed
                // LIFE MODE: slowness does NOT take life. At 2 HP/s it still killed 24-50 of 60,
                // because 300 HP runs out in 150 s and the generation has 220 s — it was still
                // execution by slowness, just slower. What kills now is HUNGER.
                if !lab.modo_vida {
                    racer.hp -= 60.0 * dt;
                }
                // BUG FIXED: check `!racer.is_dead` before overwriting the label.
                // Before, if the wall (modo_vida, L~2630) zeroed the HP and set is_dead=true
                // with death_reason=Pinado in the same frame, this block still entered
                // (hp<=0.0 is true) and OVERWROTE the label with SpeedBleed — wrong autopsy.
                // Now only set SpeedBleed if the car has not yet been marked dead.
                if racer.hp <= 0.0 && !racer.is_dead {
                    racer.is_dead = true;
                    racer.hp = 0.0;
                    racer.death_reason = Some(crate::simulation::racer::DeathReason::SpeedBleed);
                    racer.death_frame = racer.grace_ticks;
                    racer.death_pos = pos;
                    race.death_markers.push(pos);
                    if let Some(mat) = materials.get_mut(mat_handle.0.id()) {
                        mat.base_color = Color::BLACK;
                    }
                    commands.entity(entity).insert(Visibility::Hidden);
                }
            }
        } else {
            racer.slow_time = (racer.slow_time - 2.0).max(0.0);
        }

        // Base fuel burn + quadratic in the pedal. With survival_pressure: cornering costs more.
        #[cfg(feature = "survival_pressure")]
        let corner_cost = {
            let steer_abs = physical_steer.abs();
            steer_abs * pedal_intent * 0.003
        };
        #[cfg(not(feature = "survival_pressure"))]
        let corner_cost = 0.0_f32;
        // LIFE MODE: fuel becomes the true clock. In the default, consumption gives ~1500 s
        // of tank against a 220 s generation — i.e., no one ever starved
        // (`FUEL: 0` in every autopsy). Multiplying by 10, the full tank is worth ~150 s,
        // little more than two ~60 s laps: **you must complete a lap to eat.**
        let mult_fome = if lab.modo_vida { 10.0 } else { 1.0 };
        racer.fuel -= (0.002 + pedal_intent.powi(2) * 0.004 + corner_cost) * mult_fome * dt;

        // Idle brake penalty: hard brake + stopped drains a small amount of HP.
        // A hunter without prey cannot hibernate by braking.
        // Grace period: Gen 1 may use the brake during bootstrap without penalty.
        #[cfg(feature = "survival_pressure")]
        if !in_grace && brake_intent > 0.5 && speed < 3.0 && !is_reversing {
            racer.hp -= 2.0 * dt;
        }
        if racer.fuel <= 0.0 && !racer.is_dead {
            racer.is_dead = true;
            racer.death_reason = Some(crate::simulation::racer::DeathReason::Fuel);
            racer.death_frame = racer.grace_ticks;
            racer.death_pos = pos;
            commands.entity(entity).insert(Visibility::Hidden);
        }

        // Lap Counting — DIRECTION-AGNOSTIC via wrap detection + MONOTONIC track_progress.
        //
        // Critical bug fix: before, track_progress = lap_prog + total_laps*lap_dist. When a car
        // crossed the line BUT the lap did not count (the x-axis check failed OR required_progress
        // blocked distributed spawn), track_progress REGRESSED from ~450 to ~0 → max_progress
        // stopped → stagnation killed. The car vanished "at the finish line" without completing the lap.
        //
        // Now: track_progress accumulates delta_arc ALWAYS (detects forward/backward wrap in the arc).
        // Fitness uses (track_progress - spawn_progress), total_laps is DECOUPLED and only increments
        // when NET travel >= (total_laps+1) * 0.85 * lap_dist — without cheating distributed spawn.
        if !world.waypoints_world.is_empty() && world.lap_distance > 0.0 {
            let cur_lap_prog =
                world.get_track_progress(transform.translation.x, transform.translation.z);
            let prev_lap_prog = racer.prev_lap_prog;
            let lap_dist = world.lap_distance;
            racer.lap_dist_m = lap_dist; // for calc_fitness to compute fractional laps

            // Signed delta with wrap detection. If the arc "jumped" from >0.5*lap to <0.5*lap: forward wrap.
            let mut delta = cur_lap_prog - prev_lap_prog;
            let wrapped_forward = delta < -lap_dist * 0.5;
            let wrapped_backward = delta > lap_dist * 0.5;
            if wrapped_forward {
                delta += lap_dist;
            } else if wrapped_backward {
                delta -= lap_dist;
            }

            // Cap delta to avoid an absurd jump (get_track_progress glitch or teleport).
            let delta_clamped = delta.clamp(-lap_dist * 0.1, lap_dist * 0.1);
            racer.track_progress += delta_clamped;

            if wrapped_forward {
                let net = racer.track_progress - racer.spawn_progress;
                let required = (racer.total_laps as f32 + 1.0) * lap_dist * 0.85;
                if net >= required {
                    racer.total_laps += 1;
                    racer.distance = 0.0;
                    // REFUELING. It was `racer.fuel = 1.0`, and that did NOT refuel: the
                    // spawn gives `fuel: 1000.0` (spawn.rs:829), so assigning 1.0 on the 3rd lap
                    // is a 1000× DUMP disguised as a full tank. With consumption of
                    // ~0.006/s there were ~166 s of life left — i.e., whoever was good enough
                    // to reach the 3rd lap was sentenced for it.
                    // Measured: `3+v=0` in ALL generations of all races.
                    // Now it restores to the birth level, whatever it is.
                    // LIFE MODE: every lap refuels — the lap IS the hunt. Outside it,
                    // keeps the old behavior (every 3 laps) so as not to change an old experiment.
                    if lab.modo_vida || racer.total_laps % 3 == 0 {
                        racer.fuel = crate::simulation::racer::FUEL_INICIAL;
                    }
                    // Immediate post-lap grace: the car crosses the line while slowing, max_progress
                    // freezes at ~lap_dist, it needs a window to re-accelerate without dying.
                    // 240f @ 60fps = 4 real seconds — enough to leave the stall without rewarding laziness.
                    racer.post_lap_grace = 600;
                    racer.post_lap_checkpoint = racer.track_progress;
                    racer.stagnation = 0.0;
                    racer.slow_time = 0.0;
                    racer.progress_checkpoint = racer.max_progress;
                    // FIXED: it was a hardcoded `.min(100.0)` — it ignored HP_INICIAL=300 and
                    // collapsed the ceiling back into the same waypoint bug (see racer.rs L36).
                    // Now uses HP_INICIAL as the ceiling, same as the 25m checkpoint (L~2695).
                    // Lap heal = 50 absolute HP (≈17% of the 300 tank). Moderate by
                    // design: healthy life comes from FREQUENT LAPS, not one big top-up.
                    let teto_lap = crate::simulation::racer::HP_INICIAL;
                    racer.hp = (racer.hp + 50.0).min(teto_lap);
                    racer.wall_hits = 0;
                    racer.wall_time = 0.0;
                    println!(
                        "🏁 LAP {} | {} | gen_frame={} | pos=({:.1},{:.1}) | net_prog={:.0}m",
                        racer.total_laps,
                        racer.name,
                        racer.grace_ticks,
                        transform.translation.x,
                        transform.translation.z,
                        net
                    );
                }
                // WRAP without LAP silenced — distributed spawn generates noise; a real lap already signals progress.
            }
            racer.prev_lap_prog = cur_lap_prog;
        }

        // --- 6. NEURAL ESPIONAGE (FIELD-OF-VIEW GIZMOS) ---
        // Toggle KeyH (dev.show_vision) — hides the green cones for a clean visual.
        if dev.show_vision
            && (race.selected_racer.is_some() && race.selected_racer.unwrap() == entity
                || (race.selected_racer.is_none() && current_best == entity))
        {
            let forward = *transform.right();
            let up = *transform.up();
            let eye_pos = transform.translation + Vec3::Y * 0.4;

            // Visually: ONLY the green 180° tan-warp fovea. Peripheral + whiskers keep
            // feeding the brain (channels 32-63 + 94-97) but the gizmo stays clean.
            let fov_half = 90.0_f32.to_radians();
            const TAN_K: f32 = 1.25_f32;
            let tan_k_val = TAN_K.tan();
            let fovea_len = 60.0;
            let fovea_color = Color::srgba(0.0, 1.0, 0.0, 0.22);

            for i in 0..32 {
                let normalized_i = (i as f32 / 31.0) * 2.0 - 1.0;
                let warped = (normalized_i * TAN_K).tan() / tan_k_val;
                let yaw_angle = warped * fov_half + racer.gaze_yaw;
                let ray_dir = Quat::from_axis_angle(up, yaw_angle) * forward;
                gizmos.line(eye_pos, eye_pos + ray_dir * fovea_len, fovea_color);
            }
            let _ = up;
        }
    }

    // --- PERF LOG (every 640 ticks ≈ 10s sim time) ---
    let frame_us = t_frame_start.elapsed().as_secs_f64();
    perf_accum.4 += frame_us;
    perf_accum.0 += 1;
    if perf_accum.0 >= 640 {
        let n = perf_accum.0 as f64;
        println!(
            "⏱ PERF/640t | total={:.1}ms retina={:.1}ms percept={:.2}ms brain={:.1}ms | per_tick: {:.2}ms",
            perf_accum.4 * 1000.0,
            perf_accum.1 * 1000.0,
            perf_accum.2 * 1000.0,
            perf_accum.3 * 1000.0,
            perf_accum.4 / n * 1000.0,
        );
        *perf_accum = (0, 0.0, 0.0, 0.0, 0.0);
    }
}
