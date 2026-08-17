use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Resource, Debug, Serialize, Deserialize, Clone)]
pub struct LabParams {
    // Metadata
    pub experiment_name: String,
    pub variant_id: u32,

    // Axiom policy
    pub axiom_wall_avoidance: f32,
    pub axiom_dark_center: f32,
    pub axiom_whisker: f32,
    pub axiom_racing_line: f32,
    pub axiom_decay_divisor: f32,
    pub axiom_floor: f32,

    // Fitness weights
    pub k_smooth: f32,
    pub k_speed: f32,
    pub k_novel: f32,
    pub k_stab: f32,
    pub k_phi: f32,
    pub k_aesth: f32,
    pub lap_bonus: f32,
    pub survival_bonus: f32,
    pub death_penalty: f32,

    /// Brain pathway to TURN OFF in this process (see `neuro.rs`). Empty = baseline.
    /// `#[serde(default)]` because old specs lack the field and must keep loading.
    #[serde(default)]
    pub via_ablada: String,

    /// How the MOTOR neurons are born (see `neuro::Nascimento`):
    /// empty/"primado" = hand-picked channels · "esparso" = same number of connections
    /// with RANDOMLY DRAWN channels (independent) · "denso" = random with no sparsity (floor).
    #[serde(default)]
    pub nascimento: String,

    /// How far the retina's farthest row sees, in meters. 0 = default (33.3 m).
    /// Above ~28 m/s the required braking exceeds those 33.3 m and the car goes blind to
    /// the braking point — see `visao.rs`.
    #[serde(default)]
    pub visao_alcance_m: f32,
    /// Horizontal aperture of the retina in degrees. 0 = default (120°).
    #[serde(default)]
    pub visao_fov_deg: f32,

    /// Fraction of the population born in the DESTRUCTIVE mutation band (σ 0.08–0.20).
    /// 0 = default (0.50, that is, slots 30-59 of 60).
    ///
    /// Measured: with the default fraction the population MEDIAN is **zero meters** — half
    /// never leaves the spot, and `wall_mean = 0.00s` confirms it is not a crash, it is
    /// destructive mutation. The effective population is 30, not 60.
    #[serde(default)]
    pub fracao_exploradora: f32,

    /// If `true`, the lap bonus is CONTINUOUS (fractional laps) instead of an integer
    /// count. 0/absent = old behaviour, so it can be compared.
    ///
    /// # The step inside the fitness
    ///
    /// `lap_bonus = total_laps as f32 * 50000` uses the INTEGER count, while `prog` reaches
    /// ~5800 m. The bonus dominates progress by ~9×, so within the same lap bucket the
    /// selection sees almost nothing:
    ///
    /// | car | fitness | vs the 3.01 one |
    /// |---|---|---|
    /// | 3.01 laps | 155.000 | — |
    /// | 3.99 laps | 156.600 | **+1%** |
    /// | 4.00 laps | 206.600 | +32% |
    ///
    /// In other words: **going from 3.0 to 3.99 laps is barely rewarded**, and that is
    /// exactly where the population was (best car at 3.72 of a physical ceiling of 4.36).
    /// The metric step that hid the laps ([[feedback_metrica_de_degrau]]) was also INSIDE
    /// the selection signal, blinding the evolution in the range where it needed to see.
    ///
    /// Continuous: `lap_bonus * (prog / lap_length)`. Same scale, same incentive to close
    /// laps, without the step.
    #[serde(default)]
    pub lap_bonus_continuo: bool,

    /// If `true`, the niche threshold is ADAPTIVE (median of the pool distances) instead of
    /// the fixed `NICHE_DISTANCE = 8`. Absent/false = old behaviour, for comparison.
    ///
    /// Measured after fixing the fingerprint: the typical distance between elites is 2.4–5.3
    /// and the maximum 8–10 — the fixed threshold of 8 is 1.5–3× larger than the typical one
    /// and swallows everything into a single cluster, in every generation of every training
    /// run. The historical calibration ("v14: tried 10, got worse") is void: it was done
    /// when the fingerprint saturated and every distance was 0.
    #[serde(default)]
    pub nicho_adaptativo: bool,

    /// Enables the perception kernels (and eventually brain/physics) on the GPU.
    ///
    /// When `true`, the `percepcao.wgsl` kernel is chained AFTER the already existing GPU
    /// retina. The physical retina buffer stays in VRAM — no CPU→GPU round-trip on that
    /// path. The temporal channels (motion, approach_rate, EMA, racing-line, etc.) keep
    /// being written by the CPU ON TOP of the kernel's result, exactly as documented in
    /// `gpu/perception.rs::CANAIS_NA_CPU`.
    ///
    /// Default `false` to preserve the old behaviour in every experiment already run.
    /// Enable it by passing `"gpu_core": true` in the run's params.json.
    #[serde(default)]
    pub gpu_core: bool,

    /// LIFE MODE: stop killing for slowness and let fuel be the clock.
    ///
    /// Autopsy with the retina on the GPU, 2 generations of 60 beings: **`BATIDA: 0`**.
    /// Nobody dies crashing. They die of `SpeedBleed` (43 and 57 of 60) — they bleed for
    /// moving slowly, 60 HP/s after 2.8 s below 4 m/s. It is execution for slowness, not
    /// selection by danger.
    ///
    /// Leonardo: *"we can stop killing the models and leave them free on the track; they
    /// can crash and lose life, but dying would take a long time. They would need fuel, so
    /// they have to hunt — which would be the laps"*.
    ///
    /// With the flag on:
    ///   * SpeedBleed drops from 60 to 2 HP/s — it takes ~2.5 min standing still to kill, not 5 s;
    ///   * death by stagnation and by scraping the wall stop killing and become DAMAGE;
    ///   * fuel becomes the real clock and the lap refuels (it used to be every 3 laps).
    ///
    /// Life then ends for not hunting, which is the pressure the thesis wants.
    #[serde(default)]
    pub modo_vida: bool,

    // Survival thresholds
    pub speed_bleed_damage: f32,
    pub stagnation_secs: f32,
    pub wall_t_threshold: i8,
}

impl Default for LabParams {
    fn default() -> Self {
        Self {
            experiment_name: "default".to_string(),
            variant_id: 0,
            axiom_wall_avoidance: 0.18,
            axiom_dark_center: 0.45,
            axiom_whisker: 0.15,
            axiom_racing_line: 0.26,
            axiom_decay_divisor: 50.0,
            axiom_floor: 0.35,
            k_smooth: 100.0,
            k_speed: 0.5,
            k_novel: 50.0,
            k_stab: 200.0,
            k_phi: 300.0,
            k_aesth: 200.0,
            lap_bonus: 50000.0,
            survival_bonus: 3000.0,
            death_penalty: 2000.0,
            via_ablada: String::new(),
            nascimento: String::new(),
            visao_alcance_m: 0.0,
            visao_fov_deg: 0.0,
            fracao_exploradora: 0.0,
            lap_bonus_continuo: false,
            nicho_adaptativo: false,
            gpu_core: false,
            modo_vida: false,
            speed_bleed_damage: 15.0,
            stagnation_secs: 15.0,
            wall_t_threshold: 40,
        }
    }
}

impl LabParams {
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }
}

use bevy::prelude::Resource;
