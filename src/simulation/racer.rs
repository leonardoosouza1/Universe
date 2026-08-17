use crate::config::iara::vision::RETINA_TOTAL;
use crate::engine::SovereignEngine;
use bevy::prelude::*;
use std::collections::{HashSet, VecDeque};

/// Tier 1.1 — Dense reward shaping.
/// Ring buffers (128 samples ≈ 2s @ 64Hz FixedUpdate) + exploration voxel grid.
/// Populated in motor.rs every tick, consumed in pipeline::calc_fitness.
pub const TELEMETRY_CAP: usize = 128;
pub const TELEMETRY_DT: f32 = 1.0 / 64.0; // FixedUpdate step — used in the integrals.
pub const TELEMETRY_VOXEL_SIZE: f32 = 10.0; // 10m cell for novelty.

/// # SURVIVAL PRESSURE — the two scales that currently turn it off
///
/// These two values were inline in `spawn.rs:829` as `fuel: 1000.0, hp: 100000.0`,
/// **exactly 1000× the scale the rest of the code assumes**. Three pieces of evidence:
///
/// | where | what it assumes |
/// |---|---|
/// | `motor.rs:2370` `hp -= 1.0*dt` "~100s until death" | HP ≈ 100 |
/// | `motor.rs:2470` `hp -= 30.0*dt` "dies in ~3s" · max wall damage 40 | HP ≈ 100 |
/// | `motor.rs:1572` `slow_input[3] = fuel * 127.0` **with clamp at 127** | fuel ∈ [0,1] |
///
/// The third is the most expensive: with `fuel = 1000` that channel stays saturated at 127
/// forever, and the brain has a fuel sensor that is a CONSTANT — zero information.
///
/// Measured consequence: the cars only die from crashes (hole, wall, stagnation). HP and
/// fuel never bite. There is no survival tension the way the project describes.
///
/// # Current values: pressure ON, starting generous
///
/// Leonardo's choice (*"a more life-friendly environment, then make it harder"*): **3× the
/// coherent scale**, not the coherent scale outright. The reason is the failure mode
/// documented in `PROJETO_DE_MUNDO.md` — with 70–99% mortality already present, turning on
/// full pressure at once would make the world a trap, and nothing is learned in a trap.
///
/// ⚠️ **MEASURED CORRECTION:** `HP_INICIAL` was nearly INERT. The checkpoint healing in
/// `motor.rs` did `hp = (hp + 40).min(100.0)` — ceiling FIXED at 100 —, so the first
/// pass through a checkpoint (every 25 m) pinned HP at 100 whatever the birth value
/// was. Proof: the death-cause distribution is identical with HP=100.000 and with
/// HP=300 (HP ~59%, SpeedBleed ~40% in both). The healing ceiling now follows this constant.
///
/// The real HP economy is: **+40% of the tank per 25 m traveled**, against drains of
/// 1/s (battery), 30/s (spinning in place) and 60/s (speed < 4 m/s without progress).
/// In other words: **move or die**. And that is what the autopsy shows — 98,7% of deaths are
/// HP or SpeedBleed, and CRASH is 3 in 11.450.
///
/// What changes relative to the previous infinity:
/// - the `slow_input[3]` channel stops being constant and starts truly informing — it had
///   to be normalized by the initial tank (`motor.rs`), otherwise it would only leave
///   saturation after 2/3 spent
/// - skidding (`hp -= slip*10*dt`) starts to cost: ~10 s of continuous wheelspin kills
/// - crashing starts to accumulate: wall damage goes up to 40, so ~8 hard hits kill
///
/// To TIGHTEN later, the next step is `HP_INICIAL = 100.0` and `FUEL_INICIAL = 1.0`, which
/// is the scale the damages assume — and it closes a beautiful tension: at 0,006/s the 1.0
/// tank lasts ~167 s ≈ **2,7 laps**, while refueling only comes on the **3rd**. The being has
/// to be efficient enough to reach the pump before running dry. With today's 3.0 tank
/// it lasts ~8 laps, so fuel still does not bite — generous on purpose. Tighten only when
/// the population sustains the current level — the same criterion as the environment
/// curriculum (`pipeline::decide_next_env_level_pop`), for the same reason.
pub const FUEL_INICIAL: f32 = 3.0;
// HP_INICIAL raised from 300 to 1000.
//
// Worst-case computation (all drains simultaneous, no regeneration):
//   stalled engine    0,1 HP/s
//   anti-donut        0,2 HP/s
//   donut/spinning    0,5 HP/s
//   skidding          0,2 HP/s  (avg_slip ≈ 1,0)
//   wall scraping     0,3 HP/s
//   stagnation        0,5 HP/s
//   MAX TOTAL:        1,8 HP/s  → HP=1000 lasts ~556 s ≈ 9 min
//
// Typical bad case (wall + donut + stag, no stall/slip):
//   0,3 + 0,5 + 0,5 = 1,3 HP/s → ~770 s ≈ 12 min
//
// With regeneration (1,0 HP/s while advancing): the balance is -0,3 HP/s in the worst active case.
//
// The real clock is still FUEL (~150 s / tank of FUEL_INICIAL=3.0 at base consumption
// of ~0,002/s × mult_fome=10). None of the HP drains can kill before the tank
// runs dry — exactly what the user asked for: "we won't let it die, the
// tests are relevant with the models living long and doing several laps".
pub const HP_INICIAL: f32 = 1000.0;

#[derive(Clone)]
pub struct RacerTelemetry {
    pub steering_history: VecDeque<f32>,     // steer_intent [-1, 1]
    pub speed_history: VecDeque<f32>,        // m/s
    pub on_road_mask: VecDeque<bool>,        // true when voxel == Road
    pub yaw_rate_history: VecDeque<f32>,     // rad/s
    pub visited_voxels: HashSet<(i32, i32)>, // (cx, cz) cells visited ON-ROAD
    pub hit_npc_count: u32,                  // Tier 5.4 — collisions with moving NPC obstacles
}

impl RacerTelemetry {
    pub fn new() -> Self {
        Self {
            steering_history: VecDeque::with_capacity(TELEMETRY_CAP),
            speed_history: VecDeque::with_capacity(TELEMETRY_CAP),
            on_road_mask: VecDeque::with_capacity(TELEMETRY_CAP),
            yaw_rate_history: VecDeque::with_capacity(TELEMETRY_CAP),
            visited_voxels: HashSet::new(),
            hit_npc_count: 0,
        }
    }

    fn push_capped<T>(buf: &mut VecDeque<T>, val: T) {
        if buf.len() == TELEMETRY_CAP {
            buf.pop_front();
        }
        buf.push_back(val);
    }

    pub fn record(
        &mut self,
        steer: f32,
        speed: f32,
        yaw_rate: f32,
        on_road: bool,
        pos_x: f32,
        pos_z: f32,
    ) {
        Self::push_capped(&mut self.steering_history, steer);
        Self::push_capped(&mut self.speed_history, speed);
        Self::push_capped(&mut self.yaw_rate_history, yaw_rate);
        Self::push_capped(&mut self.on_road_mask, on_road);
        if on_road {
            let cx = (pos_x / TELEMETRY_VOXEL_SIZE).floor() as i32;
            let cz = (pos_z / TELEMETRY_VOXEL_SIZE).floor() as i32;
            self.visited_voxels.insert((cx, cz));
        }
    }

    /// Population variance of the steering history (jitter).
    pub fn steering_variance(&self) -> f32 {
        let n = self.steering_history.len();
        if n < 2 {
            return 0.0;
        }
        let mean: f32 = self.steering_history.iter().sum::<f32>() / n as f32;
        let var: f32 = self
            .steering_history
            .iter()
            .map(|v| {
                let d = v - mean;
                d * d
            })
            .sum::<f32>()
            / n as f32;
        var
    }

    /// Σ speed * on_road * DT — "useful" distance traveled on the track (≈meters on-road).
    pub fn speed_on_track(&self) -> f32 {
        self.speed_history
            .iter()
            .zip(self.on_road_mask.iter())
            .map(|(&s, &on)| if on { s * TELEMETRY_DT } else { 0.0 })
            .sum()
    }

    /// Σ (1 - |yaw_rate|.min(1)) * DT on-road — rewards yaw control.
    pub fn stability_integral(&self) -> f32 {
        self.yaw_rate_history
            .iter()
            .zip(self.on_road_mask.iter())
            .map(|(rate, on)| {
                if *on {
                    (1.0 - rate.abs().min(1.0)) * TELEMETRY_DT
                } else {
                    0.0
                }
            })
            .sum()
    }
}

impl Default for RacerTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeathReason {
    HpZero,     // HP drained outside wall event
    Pinado,     // Wall-time > 0.8s
    Batida,     // Wall hit chain → HP 0
    Stagnation, // max_progress did not grow
    SpeedBleed, // < 2 m/s + no progress → HP drain
    Fuel,       // fuel <= 0
    Hole,       // Fell into a hole (VoxelType::Hole) → instant death
    NpcHit,     // Violent collision with NPC obstacle → HP 0
}

impl DeathReason {
    pub fn label(&self) -> &'static str {
        match self {
            DeathReason::HpZero => "HP",
            DeathReason::Pinado => "PINADO",
            DeathReason::Batida => "BATIDA",
            DeathReason::Stagnation => "STAG",
            DeathReason::SpeedBleed => "SBLD",
            DeathReason::Fuel => "FUEL",
            DeathReason::Hole => "HOLE",
            DeathReason::NpcHit => "NPC",
        }
    }
}

#[derive(Component)]
pub struct Racer {
    pub name: String,
    pub car_model: String,
    pub car_model_idx: u8,
    pub engine: SovereignEngine,
    pub is_dead: bool,
    pub amplitude: f32,
    pub phase: f32,
    pub pain_memories: Vec<PainSynapse>,
    pub prev_pos: Vec3,
    pub distance: f32,
    pub fuel: f32,
    pub hp: f32,
    pub total_laps: usize,
    pub stagnation: f32,
    pub memory: [f32; 5], // [throttle, steer, brake, prev_shift_up, prev_shift_down]
    pub visual_cortex: [i8; RETINA_TOTAL],
    // --- NEW FIELDS (Sovereign Drive Evolution) ---
    pub tire_temp: f32,                  // Tire thermal load 0.0-1.0
    pub engine_power: f32,               // Mechanical degradation 0.1-1.0 (1.0 = full)
    pub grass_time: f32,                 // Accumulated time on grass (fitness penalty)
    pub wall_time: f32,                  // Accumulated time scraping the wall (fitness penalty)
    pub aquaplaning_timer: f32,          // Aquaplaning timer (0.5s of control loss)
    pub joy_integral: f32, // Accumulated dopamine: Σ(joy_tick * streak_mult * dt) — fitness driver
    pub joy_streak: f32,   // Continuous time in flow state (cap 5s); resets on "pain"
    pub wall_hits: u32,    // Wall impact counter (3 strikes = dead)
    pub wall_cooldown: f32, // Time until a new wall hit is registered
    pub last_waypoint: usize, // Last waypoint reached, for healing
    pub track_progress: f32, // Real progress along the track polyline (arc-length)
    pub grace_ticks: u32, // Frame counter since spawn; immune to time_scale (real grace of ~10s)
    pub reverse_timer: f32, // >0 while in an emergency reverse maneuver; decrements with dt
    pub progress_checkpoint: f32, // Last track_progress that marked real advance (+3m)
    pub progress_stall: f32, // Accumulated seconds WITHOUT gaining 3m of track (anti-donut)
    pub wall_ema_l: f32,  // EMA of distance to the left wall (decay ~1s) — persistent sensation
    pub wall_ema_r: f32,  // EMA of distance to the right wall
    pub wall_dist_prev_l: f32, // Distance to the left wall in the previous frame (m) — for the approach-rate computation
    pub wall_dist_prev_r: f32, // Distance to the right wall in the previous frame (m)
    pub wall_dist_prev_f: f32, // Distance to the front wall in the previous frame (m)
    pub proximity_sustain: f32, // Continuous seconds scraping some side (sustained alert)
    pub post_reverse_lockout: f32, // Post-reverse cooldown with capped throttle (avoids "reverse-accelerate-crash")
    pub spawn_progress: f32, // track_progress of the spawn position — fitness uses (track_progress - spawn_progress)
    pub path: Vec<Vec3>, // Position history (sampled ~4Hz) — used to draw the champion's racing line
    pub path_timer: f32, // Cooldown between path samples
    pub brain_trace: Vec<BrainSnapshot>, // Periodic brain snapshots (diagnostics)
    pub trace_timer: f32, // Cooldown between snapshots
    pub gaze_yaw: f32,   // Fovea angle relative to forward (rad). Updated by reactive
    // saccade — points to the most open path detected by the peripheral vision.
    // Clamp ±π/3. Channel 98 exposes it to the network.
    /// Lap length in meters, copied from `world.lap_distance` together with the
    /// progress update. Exists so `calc_fitness` can compute FRACTIONAL LAPS without
    /// needing to see the world. Zero = not written yet, in which case fitness falls back
    /// to the old behavior (integer count) — which keeps the A/B comparison honest.
    pub lap_dist_m: f32,
    pub max_progress: f32, // Maximum track_progress ever reached — MONOTONIC. Immune to oscillation.
    // Basis of the stagnation check: no growth > 2m in 1.5 virtual seconds, death.
    pub slow_time: f32, // VIRTUAL SECONDS with speed < 3 m/s after grace. > 0.5s → heavy bleed.
    pub race_time: f32, // VIRTUAL SECONDS alive — immune to time_scale. Used for real grace
    // (grace_ticks counts real frames, inadequate when the sim runs at 8x).
    pub prev_lap_prog: f32, // In-lap arc position of the previous frame — used to detect wrap
    // direction-agnostic (prev > 0.85*lap_dist && cur < 0.15*lap_dist).
    pub post_lap_grace: u32, // Remaining frames of immunity to STAGNATION/SPEED_BLEED after a lap.
    // A car that completes a lap stops at waypoint[0] → max_progress freezes →
    // it used to die of stagnation before re-accelerating. Grace gives time to leave.
    pub post_lap_checkpoint: f32, // track_progress at the instant of the last lap. Grace only ends when
    // the car advances 25m BEYOND that checkpoint — guarantees it leaves the start line.
    pub death_reason: Option<DeathReason>, // Causal death reason (autopsy).
    pub death_frame: u32,                  // grace_ticks at the instant of death (≈ real frame).
    pub death_pos: Vec3,                   // World position of the death (zone on the track).
    pub peak_speed: f32,                   // Maximum recorded speed (m/s) for the report.
    pub telemetry: RacerTelemetry,         // Tier 1.1: dense ring buffers for reward shaping.
    // --- Tier 6.5: Time Dilation under stress ---
    /// Tick counter to modulate the brain's inference frequency.
    /// Wraps at u8 (256 ticks ≈ 4s @ 64Hz — more than enough for skip mod).
    pub brain_tick_counter: u8,
    /// Last action (throttle, steer) emitted by the brain. Reused when
    /// skip_n > 1 to keep the command stable between inferences (time dilation).
    /// Note: engine.final_intent[] already persists between ticks as a member of the SovereignEngine,
    /// but we keep a canonical copy here for the Tier 6.5 documentation contract.
    pub last_action: (f32, f32),
    // --- Tier 4.3: Niche-splitting (runtime-only, NOT serialized) ---
    /// 0 = defensive (penalizes wall_time 2x), 1 = aggressive (rewards speed_on_track 1.5x).
    /// Alternated at spawn (i % 2). Shared-pool: both niches compete for the same elites,
    /// cross-breed emerges naturally. Annotation-only for lineage introspection.
    pub niche: u8,
    // --- Tier 5.1 LITE: Multi-car social perception ---
    /// Relative positions (dx, dy) of the 3 nearest rivals, normalized by
    /// a 50m clamp → [-1.0, 1.0]. Layout: [dx0, dy0, dx1, dy1, dx2, dy2], ordered
    /// from nearest to farthest. Empty slots stay at 0.0.
    /// Data layer only — populated via `EnvironmentalState::closest_racers` but still
    /// NOT wired into motor/sensor sampling (wire-up in a follow-up). Retina channels 102-107.
    pub perceived_rivals: [f32; 6],
    // --- Tier 6.3 LITE: morphological co-evolution (body params from the DNA) ---
    /// Body genes (mass, wheelbase, downforce, grip_mult) loaded from the DNA at spawn.
    /// For now only STORED — physical wiring in a follow-up. See `crate::evolution::dna_genesis::MorphoGenes`.
    pub morpho: crate::evolution::dna_genesis::MorphoGenes,
    // --- KAN output head: smooth spline per output (throttle/steer/brake) ---
    /// WHY: eliminates the saturated ±1 zig-zag of the hard clamp; evolved via mutate_spline_knots.
    pub kan_out: crate::kan::KanOutputHead,
    /// Stable seed for the local PRNG (deterministic sensory/exploration noise).
    pub stable_seed: u64,
    /// Last sensory snapshot (64 semantic channels) for visualization/diagnostics.
    /// Updated each frame in update_racers() for UI to display what agent sees.
    pub sensory_snapshot: [i8; 64],
}

/// Diagnostic frame of the racer's brain.
/// Captured at ~2Hz and dumped in the reports at key moments (death, champion).
/// Reveals whether the sensory signal is arriving, whether the wheel responds, and
/// how much was attenuated between the neuron and the physics.
#[derive(Clone)]
pub struct BrainSnapshot {
    pub t: f32,                      // Car's lifetime
    pub speed_kmh: f32,              // Speed
    pub wall_prox_l: i8,             // Channel 85: left wall proximity (0=far, 127=touching)
    pub wall_prox_r: i8,             // Channel 86: right wall proximity
    pub ttc_inv: i8,                 // Channel 82: frontal collision urgency
    pub curv_25: i8,                 // Channel 79: anticipated curvature 25m
    pub retina_center_max: i8, // Peak of the retina's central channels (10..22) — wall in the center
    pub raw_throttle: f32,     // final_intent[0] from the brain
    pub raw_steer: f32,        // final_intent[1] (pre-attenuation)
    pub raw_brake: f32,        // final_intent[2]
    pub post_damp_steer: f32,  // steer_intent after the slow brain's risk_damp
    pub physical_steer: f32,   // What ACTUALLY reached the wheel (with dynamic_steer_limit)
    pub steer_limit: f32,      // Dynamic factor 1/(1+v*0.02)
    pub risk_damp: f32,        // Slow brain attenuation (0..0.3)
    pub top_channels: [(u8, i8); 5], // Top-5 most activated channels (|value|), with index+value

    // --- Vision replay data (compressed retina + sensory channels) ---
    pub retina_fov: Vec<i8>, // FOV_BINS × RETINA_ROWS (dynamic size, supports variable retina dimensions)
    pub sensory_compact: [i8; 32], // Top-32 telemetry channels: speed, alignment, curvature, walls, etc
}

pub struct PainSynapse {
    pub position: Vec3,
    pub pain_phase: f32,
    pub intensity: f32,
}

#[derive(Component)]
pub struct Skidmark {
    pub timer: f32,
}
