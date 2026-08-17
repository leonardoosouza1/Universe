use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};

#[derive(Resource, Debug, Serialize, Deserialize, Clone)]
pub struct PhysicsSettings {
    pub enable_stall: bool,
    pub enable_thermal: bool,
    pub enable_aero_drag: bool,
    pub enable_downforce: bool,
    pub enable_tcs: bool,
    pub enable_abs: bool,
    pub enable_aquaplaning: bool,
    pub enable_tire_wear: bool,
}

impl Default for PhysicsSettings {
    fn default() -> Self {
        Self {
            enable_stall: true,
            enable_thermal: true,
            enable_aero_drag: true,
            enable_downforce: true,
            enable_tcs: true,
            enable_abs: true,
            enable_aquaplaning: true,
            enable_tire_wear: true,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrackProfile {
    pub name: String,
    pub seed: u64,
    pub track_width: f32,
    pub complexity: f32,
}

#[derive(Resource, Debug, Serialize, Deserialize, Clone)]
pub struct UniverseConfig {
    pub physics: PhysicsSettings,
    pub tracks: Vec<TrackProfile>,
    pub selected_track_index: usize,
    pub swarm_count: usize,
    /// Simulation speed multiplier (1.0 = real time, 10.0 = 10x faster)
    pub time_scale: f32,
    /// Maximum duration of each generation in simulation seconds.
    /// The generation ends when: someone completes 5 laps, OR this timeout is reached.
    /// Dead cars stay in the scene (hidden) but do not end the gen — the living ones
    /// keep accumulating distance until time runs out.
    #[serde(default = "default_generation_timeout_s")]
    pub generation_timeout_s: f32,
    /// Tier 5.3: track rotation per generation. If false (default), always uses
    /// `selected_track_index`. If true, alternates between `tracks` in blocks of
    /// `gens_per_track` and pins `tracks[0]` on validation gens (env_level 0).
    /// Can be enabled via CLI `--rotate` or env `UNIVERSE_ROTATE=1`.
    #[serde(default)]
    pub rotate_tracks: bool,
    #[serde(default = "default_gens_per_track")]
    pub gens_per_track: usize,
    /// FIX UI vs headless mismatch: saves the flags of the last training so the UI
    /// reproduces the same track the DNA saw. Without this, DNA trained with --easy
    /// --no-holes ran on a hard track with holes in the UI → bad cars.
    #[serde(default)]
    pub easy_track: bool,
    #[serde(default)]
    pub no_holes: bool,
}

fn default_generation_timeout_s() -> f32 {
    120.0
}
fn default_gens_per_track() -> usize {
    3
}

impl Default for UniverseConfig {
    fn default() -> Self {
        Self {
            physics: PhysicsSettings::default(),
            // Order = curricular progression (easy → very hard).
            // track_width: wider = more forgiveness for bad mutants.
            tracks: vec![
                // Default = procedural seed 9001. Matches DNA training seeds — UI mode loads
                // CHAMPION.dna trained on this layout instead of mismatching with hardcoded.
                TrackProfile {
                    name: "Procedural Sigma".to_string(),
                    seed: 9001,
                    track_width: 14.0,
                    complexity: 0.5,
                },
                TrackProfile {
                    name: "Dubai Oval".to_string(),
                    seed: 123,
                    track_width: 15.0,
                    complexity: 0.3,
                },
                TrackProfile {
                    name: "Interlagos Gateway".to_string(),
                    seed: 64,
                    track_width: 13.0,
                    complexity: 1.0,
                },
                TrackProfile {
                    name: "Monaco Tight".to_string(),
                    seed: 256,
                    track_width: 11.0,
                    complexity: 2.0,
                },
                TrackProfile {
                    name: "Silverstone Flow".to_string(),
                    seed: 512,
                    track_width: 12.0,
                    complexity: 1.8,
                },
                TrackProfile {
                    name: "Nurburgring Hills".to_string(),
                    seed: 768,
                    track_width: 12.0,
                    complexity: 2.5,
                },
                TrackProfile {
                    name: "Suzuka Crossing".to_string(),
                    seed: 1024,
                    track_width: 10.0,
                    complexity: 3.0,
                },
            ],
            selected_track_index: 0,
            swarm_count: 50,
            time_scale: 3.0,
            generation_timeout_s: 120.0,
            rotate_tracks: false,
            gens_per_track: 3,
            easy_track: false,
            no_holes: false,
        }
    }
}

impl UniverseConfig {
    pub fn load_or_create() -> Self {
        if let Ok(mut file) = File::open("universe_config.json") {
            let mut contents = String::new();
            if file.read_to_string(&mut contents).is_ok() {
                if let Ok(config) = serde_json::from_str(&contents) {
                    return config;
                }
            }
        }
        let config = Self::default();
        let _ = config.save();
        config
    }

    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        let mut file = File::create("universe_config.json")?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    pub fn current_track(&self) -> &TrackProfile {
        &self.tracks[self
            .selected_track_index
            .min(self.tracks.len().saturating_sub(1))]
    }

    /// Tier 5.3: returns the track for a given generation.
    /// - `rotate_tracks == false`: always `current_track()` (back-compat).
    /// - `is_validation_gen == true`: pinned to `tracks[0]` (stable env → honest
    ///   champion comparison across gens).
    /// - Otherwise, rotation by block: idx = (gen / gens_per_track) % tracks.len().
    pub fn track_for_gen(&self, gen_idx: usize, is_validation_gen: bool) -> &TrackProfile {
        if !self.rotate_tracks || self.tracks.is_empty() {
            return self.current_track();
        }
        if is_validation_gen {
            return &self.tracks[0];
        }
        let per = self.gens_per_track.max(1);
        let idx = (gen_idx / per) % self.tracks.len();
        &self.tracks[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_with_rotate() -> UniverseConfig {
        let mut c = UniverseConfig::default();
        c.rotate_tracks = true;
        c.gens_per_track = 2;
        c
    }

    #[test]
    fn rotation_cycles_by_block() {
        let c = cfg_with_rotate();
        let n = c.tracks.len();
        // Block size = 2, so gens 0,1 → track 0; gens 2,3 → track 1; ...
        assert_eq!(c.track_for_gen(0, false).name, c.tracks[0].name);
        assert_eq!(c.track_for_gen(1, false).name, c.tracks[0].name);
        assert_eq!(c.track_for_gen(2, false).name, c.tracks[1 % n].name);
        assert_eq!(c.track_for_gen(3, false).name, c.tracks[1 % n].name);
        assert_eq!(c.track_for_gen(4, false).name, c.tracks[2 % n].name);
    }

    #[test]
    fn validation_gen_pins_track_zero() {
        let c = cfg_with_rotate();
        // In any gen, if it is a validation one, track 0.
        for g in [0usize, 7, 13, 42] {
            assert_eq!(c.track_for_gen(g, true).name, c.tracks[0].name);
        }
    }

    #[test]
    fn rotate_false_keeps_current_track() {
        let c = UniverseConfig::default(); // rotate_tracks = false
        for g in [0usize, 5, 100] {
            assert_eq!(c.track_for_gen(g, false).name, c.current_track().name);
            assert_eq!(c.track_for_gen(g, true).name, c.current_track().name);
        }
    }
}

// ============================================================================
// IARA — REFERENCE CONSTANTS HUB
// ============================================================================
// Centralized documentation of the system's parameters.
// The values below MIRROR the ones that exist in the original modules.
//
// To change a parameter:
//   1. Change the value HERE
//   2. Change it in the corresponding original file (indicated in the comments)
//
// When you want the whole system to derive from here automatically,
// just swap the `use` in the original module for `use crate::config::iara::*`.
// ============================================================================

pub mod iara {
    // ── VISION (source: src/world.rs:12-18) ────────────────────────────
    pub mod vision {
        // ORIGINAL
        // pub const RETINA_ROWS: usize = 14;
        // pub const RETINA_COLS: usize = 126;
        // pub const RETINA_TOTAL: usize = RETINA_ROWS * RETINA_COLS;
        // pub const FOV_DEG: f32 = 140.0;
        // pub const RETINA_SKIP: u32 = 2;
        // pub const RETINA_MIN_PITCH: f32 = -0.04;
        // pub const RETINA_MAX_PITCH: f32 = -0.005;

        // POOLING FIX: max-pool ALL rows destroys curve shape. Use zone-based pooling instead.
        // 14 rows kept (no perf hit), but ch 0-15 now pool by zone (preserve depth + shape).
        // - ch 0-7: FAR rows (0-4, ~80m) = distant curves/crests (what's ahead?)
        // - ch 8-15: NEAR rows (10-14, ~20m) = close obstacles (must avoid now!)
        pub const RETINA_ROWS: usize = 14;
        pub const RETINA_COLS: usize = 126;
        pub const RETINA_TOTAL: usize = RETINA_ROWS * RETINA_COLS;
        pub const FOV_DEG: f32 = 120.0;
        pub const RETINA_SKIP: u32 = 1;
        pub const RETINA_MIN_PITCH: f32 = -0.012; // row 0: horizon/far (~33m ground @ 0.4m eye)
        pub const RETINA_MAX_PITCH: f32 = -0.06; // row 13: ground/near

        const fn compute_pitches() -> [f32; RETINA_ROWS] {
            let mut pitches = [0.0_f32; RETINA_ROWS];
            if RETINA_ROWS <= 1 {
                pitches[0] = RETINA_MIN_PITCH;
                return pitches;
            }
            let step = (RETINA_MAX_PITCH - RETINA_MIN_PITCH) / (RETINA_ROWS as f32 - 1.0);
            let mut i = 0;
            while i < RETINA_ROWS {
                pitches[i] = RETINA_MIN_PITCH + step * i as f32;
                i += 1;
            }
            pitches
        }

        /// Pitch angles per retina row (rad), auto-computed.
        /// Row 0 = high horizon (-0.10 rad), Row N-1 = ground (+0.05 rad).
        pub const RETINA_PITCHES: [f32; RETINA_ROWS] = compute_pitches();
    }

    // ── BRAIN (source: src/simulation/motor.rs, src/engine.rs) ─────────
    pub mod brain {

        /// Computes the sensory wave size automatically.
        /// Structure: 0-127 (fovea, peripheral, telemetry, lookahead)
        ///          + 160-191 (retina horizon scan: 32 samples)
        /// Total = 128 + 32 = 160, but margin for safety = 160 + RETINA_COLS
        const fn compute_sensory_size() -> usize {
            // 160 base + 32 horizon = 192. This is EVERYTHING perception writes: the last
            // block of `perception.wgsl` is "BLOCK Q: ch 160-191".
            //
            // There were 31 more channels "spare (perception injection etc.)" that were never
            // filled by any block. Measured by the vision audit: channels
            // 192..222 are ZERO in every being, always. The effect is not neutral — the
            // first layer had 31 × 96 = 2.976 weights wired to a constant zero, and
            // evolution mutated those weights every generation, spending search on parameters
            // that cannot influence anything.
            //
            // Reserving space for the future costs dearly when the future never arrives.
            192
        }

        /// Sensory wave = array of i8 channels.
        /// It is NOT the raw retina — it goes through pooling in motor.rs.
        /// Size computed automatically based on RETINA_ROWS/COLS.
        pub const SENSORY_WAVE_SIZE: usize = compute_sensory_size();

        /// Fast Brain: BitLinear 192→96 (instinct layer, ternary weights in the DNA).
        pub const L1_NEURONS: usize = 96;

        /// Cortex L2: 96→48 (deep composition).
        pub const L2_NEURONS: usize = 48;

        /// Cortex L3: 48→24 (pre-output).
        pub const L3_NEURONS: usize = 24;

        /// Slow Brain: CfC(4) + Dense(64→16) + Dense(16→3) (slow cortex, runs every N ticks).
        pub const SLOW_BRAIN_CFC_DIM: usize = 4;
        pub const SLOW_BRAIN_D1_IN: usize = 64;
        pub const SLOW_BRAIN_D1_OUT: usize = 16;
        pub const SLOW_BRAIN_D2_IN: usize = 16;
        pub const SLOW_BRAIN_D2_OUT: usize = 3; // throttle, steer, brake
        pub const SLOW_BRAIN_TICK_INTERVAL: u32 = 8; // Runs 1x every 8 ticks (~8Hz @ 64Hz)

        /// Legacy constants (for backward compatibility)
        pub const SLOW_BRAIN_INPUTS: usize = 48;
        pub const SLOW_BRAIN_OUTPUTS: usize = 8;
    }

    // ── GENETICS / DNA (source: src/evolution/dna_genesis.rs) ──────────
    pub mod genetics {
        use super::brain::*;

        // Current version of the DNA format
        pub const DNA_VERSION: u16 = 8;

        // Weight sizes (computed from the architecture)
        pub const DNA_FAST_SIZE: usize = SENSORY_WAVE_SIZE * L1_NEURONS; // 192 × 96 = 18.432
        // DNA v8: SlowBrain full (dense1 + dense2 + biases + scales)
        // dense1.weights(1024) + dense1.bias(16) + dense1.scale(4) + dense2.weights(48) + dense2.bias(3) + dense2.scale(4) = 1099
        pub const DNA_SLOW_SIZE: usize = (SLOW_BRAIN_D1_IN * SLOW_BRAIN_D1_OUT)
            + SLOW_BRAIN_D1_OUT
            + 4
            + (SLOW_BRAIN_D2_IN * SLOW_BRAIN_D2_OUT)
            + SLOW_BRAIN_D2_OUT
            + 4; // 1099

        // Legacy (compatibility with old DNAs)
        pub const DNA_FAST_SIZE_LEGACY: usize = 128 * 64; // 8.192
        pub const DNA_SLOW_SIZE_LEGACY: usize = 32 * 8; // 256

        // Scars (trauma immutability — Tier 6.6)
        pub const SCAR_TTL_INITIAL: u8 = 5; // locks for 5 generations
        pub const SCAR_FRACTION: f32 = 0.10; // 10% of non-zero weights become scars

        // Protected axiom zone (spawn.rs:17)
        pub const PROTECTED_ZONE_SIZE: usize = 819; // ~4.4% of DNA_FAST_SIZE

        // Weight Mutation
        pub const DEFAULT_MUTATION_RATE: f32 = 0.008; // 0.8% flip chance per bit

        // Morphological Mutation (Body)
        pub const MORPHO_MASS_RANGE: (f32, f32) = (700.0, 1400.0);
        pub const MORPHO_WHEELBASE_RANGE: (f32, f32) = (1.5, 3.0);
        pub const MORPHO_DOWNFORCE_RANGE: (f32, f32) = (0.5, 2.0);
        pub const MORPHO_GRIP_RANGE: (f32, f32) = (0.8, 1.2);
        pub const MORPHO_MUTATION_RATE: f32 = 0.05; // 5% per gene per gen
        pub const MORPHO_SIGMA_FRAC: f32 = 0.05; // σ = 5% of the range

        // Hyper-network (Weight Compression)
        pub const HYPERNET_COEFFS: usize = 64;
        pub const HYPERNET_EXPAND_SIZE: usize = 2048;
        pub const HYPERNET_MUTATION_FLIP_RATE: f32 = 0.02;
        pub const HYPERNET_SIGMA: f32 = 0.05;
        pub const HYPERNET_TERNARY_THRESH: f32 = 0.3;

        // Metrics (settings.rs:5)
        pub const FITNESS_HISTORY_CAP: usize = 100;
    }

    // ── SIMULATION (source: src/simulation/motor.rs) ───────────────────
    pub mod simulation {
        // Timing
        pub const TICK_RATE: f32 = 64.0;
        pub const TIME_STEP: f32 = 1.0 / TICK_RATE;

        // Punishment Thresholds (Survival Policy)
        pub const STAGNATION_TIMEOUT_FRAMES: u32 = 960; // ~15s without progress = death
        pub const WALL_DAMAGE_BASE: f32 = 4.0;
        pub const WALL_DAMAGE_MAX: f32 = 40.0;
        pub const WALL_DAMAGE_GRACE: f32 = 20.0;
        // Reduced 30→15 for narrow tracks: models need time to navigate tight corners
        pub const SPEED_BLEED_DAMAGE: f32 = 15.0;
        pub const WALL_PIN_TIME_S: f32 = 1.6; // Seconds pinned to the wall before dying
        pub const HP_GRACE_FRAMES: u32 = 480; // Post-lap protection frames

        // Recovery
        pub const WAYPOINT_DIST_M: f32 = 25.0; // Distance between checkpoints (meters)
        pub const HEAL_AMOUNT: f32 = 40.0; // HP recovered per checkpoint

        // Axioms (Instinct Injection)
        pub const AXIOM_DECAY_GENS: f32 = 50.0; // Axioms fade after N generations
    }

    /// Emergency reverse maneuver: front wall very close + stopped + stagnated
    /// triggers reverse for a minimum time, with a post-reverse cooldown. Replica of `motor.rs`,
    /// ported to the three headless GPU loops (`gpu_evolve`, `gpu_loop`, `cortex_evolve`).
    ///
    /// Deduplicated (E3, REFACTOR_PLAN.md §2.3): there were THREE local copies of these
    /// constants — and they had DIVERGED: `gpu_evolve.rs` and `gpu_loop.rs` declared
    /// `REVERSE_TRIGGER_DIST = 3.0` but never USED it (the trigger's `if` had the
    /// literal `4.0` hardcoded next to it — the constant was dead, confirmed by the warning
    /// `constant REVERSE_TRIGGER_DIST is never used`). Only `cortex_evolve.rs` actually
    /// referenced the constant, and its value was `4.0` — that is, the REAL behaviour of the three
    /// binaries was always `4.0`; the `3.0` was a dead value that lied about the trigger.
    /// The single constant below uses the value effectively in force (`4.0`).
    pub mod reverse_maneuver {
        /// Front distance (m) below which reverse is triggered (with stopped+stagnated).
        pub const TRIGGER_DIST_M: f32 = 4.0;
        /// Minimum duration of the reverse maneuver, in seconds.
        pub const MIN_DURATION_S: f32 = 0.4;
        /// Maximum duration of the reverse maneuver, in seconds (safety timeout).
        pub const MAX_DURATION_S: f32 = 1.5;
        /// Front distance (m) from which the reverse is considered "clear" and ends.
        pub const EXIT_CLEAR_M: f32 = 2.5;
        /// Post-reverse cooldown, in seconds: throttle capped, new reverse blocked.
        pub const POST_LOCKOUT_S: f32 = 0.6;
        /// Stagnation ticks (speed < 1.5 m/s) required before enabling the trigger.
        /// 60 ticks ≈ 1 s @ 64 Hz.
        pub const STAGNATION_TICKS: u32 = 60;
        /// Maximum speed (m/s) for the reverse trigger to consider the being "stopped".
        pub const TRIGGER_SPEED_MAX_MS: f32 = 1.5;
        /// Fixed pedal (throttle) during active reverse — same value as `motor.rs`.
        pub const PEDAL: f32 = 0.75;
    }
}
