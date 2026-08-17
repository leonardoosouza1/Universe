use super::dna_genesis::{EliteDna, HypernetSeed, MorphoGenes};
use bevy::prelude::*;
use std::collections::VecDeque;

pub const FITNESS_HISTORY_CAP: usize = 100;

#[derive(Resource)]
pub struct RaceSettings {
    pub generation: usize,
    pub seed: u64,
    pub best_weights: Option<Vec<f32>>,
    pub best_car_model_idx: Option<u8>,
    pub best_l2_weights: Option<Vec<f32>>,
    pub best_scale: f32,
    pub best_slow_weights: Option<Vec<f32>>,
    pub best_slow_scale: f32,
    pub best_gate_weights: Option<Vec<f32>>,
    pub best_gate_bias: Option<Vec<f32>>,
    pub best_reflex_weights: Option<Vec<f32>>, // f32 runtime; quantize only when saving TRIN into the DNA
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
    pub champion_distance: f32, // Distance of the DNA saved in CHAMPION.dna (only changes when a DNA is crowned)
    pub best_ever_dist: f32,    // Guards against regression — only updates with a 10% improvement
    pub elite_pool: Vec<EliteDna>, // Top-5 DNAs of the last generation (tournament pool)
    // Niching: cluster id parallel to elite_pool (same index). Populated by
    // evolution_pipeline after clustering by BehaviorFingerprint (Hamming ≤ 8).
    // Used by spawn.rs in tournaments to reject pairs from the same cluster.
    pub elite_clusters: Vec<usize>,
    pub champion_path: Vec<Vec3>, // Path of the last champion (spawn → last position)
    pub champion_net_dist: f32,   // NET distance of the last champion (for the path label)
    pub gens_since_crown: u32,    // Consecutive gens without crowning — unlocks the threshold if > N
    /// Fraction of the POPULATION that completed ≥1 lap in the previous generation.
    /// Governs the environment curriculum (see `pipeline::decide_next_env_level_pop`). It
    /// used to be governed by the CHAMPION's distance, and a single lucky car took all 60
    /// to a level where grip drops to 0.33 and the required braking (130 m) exceeds the
    /// range of the longest sensor (120 m) — blindness, not difficulty.
    pub frac_completou_volta: f32,
    /// How much the track is narrowed relative to full width, in metres.
    /// Persists across generations and moves at most 0.25 m at a time, in both
    /// directions — it can WIDEN back when the population cannot sustain it. See the
    /// width curriculum in `pipeline.rs`, which used to be driven by the champion's
    /// distance.
    pub estreitamento: f32,
    pub last_gen_path: Vec<Vec3>, // Path of the last gen's best (always refreshed)
    pub last_gen_net: f32,        // NET of the last gen's best
    pub last_gen_was_grid: bool,  // true if the best was born on the grid (eligible for crowning)
    pub is_validation_gen: bool,  // next gen runs a fixed env (level 0); only validates the DNA without noise
    pub gens_until_validation: u32, // counts down to the next validation (default: every 3 gens)
    pub env_level: u8,            // curriculum level applied in the current gen (0..=3)
    // Batched spawn: avoids a freeze when spawning N cars at once. Each gen sets
    // Some((0, swarm_count)); drain_spawn_queue consumes it in blocks of SPAWN_BATCH_SIZE
    // per frame until it becomes None. The sim keeps running between batches — the UI
    // stays responsive.
    pub pending_spawn: Option<(usize, usize)>,
    // Tier 1.2 — mean-fitness history (last 100 gens) for the emergency brake.
    pub fitness_history: VecDeque<f32>,
    // Tier 1.2 — if > 0, the next N gens use a minimum axiom_strength of 0.5 (rescue).
    pub axiom_boost_counter: u32,
    // Tier 1.3 — Axiom as an external prior (potential-based shaping, Ng-Harada-Russell 1999).
    // axiom_policy() in motor.rs adds an additive bias to the brain's output scaled by this value.
    // Updated by spawn.rs every gen with the same decay curve: (1.0 - gen/50).max(0),
    // raised to 0.5 if axiom_boost_counter > 0. Consumed by update_racers after execute_cycle.
    pub current_axiom_strength: f32,
    // Tier 6.3/6.4 — morpho + hypernet of the current champion, persisted via save_champion_dna_full.
    pub current_morpho: MorphoGenes,
    pub current_hypernet: HypernetSeed,
    // Telemetry: per-generation data recording (NDJSON stats + binary replay)
    pub telemetry: super::telemetry::TelemetryWriter,
    /// Writer of the new replay format (128 bytes/frame) — recorded when
    /// lab.record_replay=true. None = no recording. Opened at the start of each
    /// generation and closed (flushed) at the end by the evolution_pipeline.
    pub replay_writer: Option<crate::replay::ReplayWriter>,
}

#[derive(Resource)]
pub struct DevSettings {
    pub mutation_rate: f32,
    pub pain_sensitivity: f32,
    pub wave_amplitude: f32,
    pub flow_influence: f32,
    /// Toggle visual gizmos: champion path, waypoint pillars, flag, last-gen path.
    /// Hotkey G (toggle). Default ON for dev, can hide for clean racing-game look.
    pub show_gizmos: bool,
    /// Toggle vision cones (green fovea raycast) on selected racer. KeyH.
    pub show_vision: bool,
}

impl Default for DevSettings {
    fn default() -> Self {
        Self {
            mutation_rate: 0.008,
            pain_sensitivity: 1.0,
            wave_amplitude: 70.0,
            flow_influence: 0.5,
            show_gizmos: true,
            show_vision: true,
        }
    }
}
