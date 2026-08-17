use crate::config::iara::vision::RETINA_TOTAL;
/// IARA v2101 - Digital Metabolism & Homeostasis
/// Based on Codex v2101 - Module I
///
/// Tier 3.1+3.2: biological drives added (dopamine = RPE, cortisol = integral fear).
/// All new fields are RUNTIME-ONLY — reset on spawn (Default::default). They do NOT persist
/// in .dna v3.
use std::collections::VecDeque;

/// Tier 2.3 — episodic memory trace used during sleep consolidation.
/// Captured during wake only when something salient happens (big reward swing
/// or wall-contact); drained into slow_brain via Hebbian during sleep.
#[derive(Clone, Debug)]
pub struct EpisodicTrace {
    pub retina: [i8; RETINA_TOTAL],
    pub action: [f32; 2],
    pub reward: f32,
}

/// Max episodic ring capacity (cap per spec 2.3).
pub const EPISODIC_CAP: usize = 64;

pub struct Metabolism {
    /// Digital Adenosine: accumulation of computational fatigue (0.0 to 1.0)
    pub adenosine: f32,
    /// Recovery rate during "Sleep" or low activity
    pub recovery_rate: f32,
    /// Current State: Active or Sleeping (Night Dojo)
    pub is_sleeping: bool,
    /// VRAM/Cache stress level
    pub cache_stress: f32,

    // --- Tier 3.1: Dopamine = Reward Prediction Error ---
    /// Instantaneous dopamine [-1.0, 1.0] — RPE signal (actual - predicted reward).
    /// Positive spike on good surprise, dip on bad surprise, ~0 on expected reward.
    pub dopamine: f32,
    /// Predicted reward — running average (TD-learning α=0.1) of the received reward.
    pub predicted_reward: f32,

    // --- Tier 3.2: Cortisol = integral fear/stress ---
    /// Cortisol [0.0, 1.0] — integrates wall proximity over time.
    /// High cortisol → defensive behavior (throttle ↓, steering ↓, risk_damp ↑).
    pub cortisol: f32,

    // --- Tier 3.3: Curiosity = novelty in the retina ---
    /// Curiosity [0.0, 1.0] — rises when the retina is monotonous (low variance
    /// over the 16-frame history). The motor uses it to inject exploration noise
    /// when also stuck (stagnation > 60 frames).
    pub curiosity: f32,
    /// Ring buffer of the last 16 frames of the binocular retina (64 channels).
    /// Rotate_right(1) + insert at index 0. Compute variance across history.
    pub retina_history: [[i8; RETINA_TOTAL]; 16],

    // ------------------------------------------------------------------
    // Tier 4.2 — Partial epigenetics (hormonal running means)
    // ------------------------------------------------------------------
    // RUNTIME-ONLY accumulators. Reset on spawn (Default). Do NOT persist in .dna.
    // Captured by EliteDna at the end of the gen and 10% inherited by the child on spawn.
    pub cortisol_sum: f64,
    pub dopamine_sum: f64,
    pub curiosity_sum: f64,
    pub tick_count: u64,

    // ------------------------------------------------------------------
    // Tier 2.3 — Episodic memory (consolidated during sleep)
    // ------------------------------------------------------------------
    /// Ring buffer of salient traces (cap 64). Push when
    /// |reward| > 0.1 OR wall_hit. Drained during sleep by consolidate_episodic.
    /// RUNTIME-only. Does not persist in .dna.
    pub episodic_buffer: VecDeque<EpisodicTrace>,
    /// Guard to avoid consolidating multiple times in the same sleep cycle.
    /// Reset when adenosine < 0.5 (the organism has woken up enough).
    pub sleep_consolidated: bool,

    // ------------------------------------------------------------------
    // Tier 6.8 — Mirror neurons / observational learning
    // ------------------------------------------------------------------
    /// Counts how many ticks the mirror hook fired (cosine(retina, champion_mean) > 0.8).
    /// RUNTIME-only. The fitness hook may read it to reward learning by imitation.
    pub mirror_activation_count: u32,

    // ------------------------------------------------------------------
    // Tier 6.1 — Predictive coding as primary drive
    // ------------------------------------------------------------------
    /// Retina prediction error at the current tick (mean L1, ~[0, 2]).
    /// Updated every tick by the engine; feeds curiosity as an intrinsic drive.
    /// RUNTIME-only.
    pub prediction_error: f32,
    /// Running sum of prediction_error for the lifetime mean (diagnostics).
    pub prediction_error_sum: f64,
    pub prediction_error_samples: u64,
}

impl Metabolism {
    pub fn new() -> Self {
        Self::default()
    }

    /// Simulate the passage of time and the cost of action.
    /// Every operation in the engine adds to adenosine.
    pub fn accumulate_fatigue(&mut self, cost: f32) {
        if !self.is_sleeping {
            self.adenosine += cost;
            self.cache_stress += cost * 0.5; // VRAM fragmentation proxy

            // Check Control Barrier Function (CBF)
            if self.adenosine >= 1.0 || self.cache_stress >= 1.0 {
                self.trigger_sleep();
            }
        }
    }

    fn trigger_sleep(&mut self) {
        self.is_sleeping = true;
    }

    /// Night Dojo: Consolidation and Cache Clearing
    pub fn update(&mut self, delta_t: f32) {
        if self.is_sleeping {
            self.adenosine -= self.recovery_rate * delta_t;
            self.cache_stress -= self.recovery_rate * 2.0 * delta_t;

            if self.adenosine <= 0.1 {
                self.is_sleeping = false;
            }
        } else {
            // Natural slow recovery if not fully stressed
            if self.adenosine > 0.0 {
                self.adenosine -= 0.001 * delta_t;
            }
        }

        // Clamp values
        self.adenosine = self.adenosine.clamp(0.0, 1.5);
        self.cache_stress = self.cache_stress.clamp(0.0, 1.5);
    }

    /// Check if the system can accept new tasks (Axiom: No processing during sleep)
    pub fn can_process(&self) -> bool {
        !self.is_sleeping
    }

    // ------------------------------------------------------------------
    // Tier 3.1 — Dopamine (RPE)
    // ------------------------------------------------------------------

    /// Updates dopamine via TD-error: dopamine = actual - predicted (clamped).
    /// predicted_reward is a running average (α=0.1).
    pub fn update_dopamine(&mut self, actual_reward: f32) {
        let prediction_error = actual_reward - self.predicted_reward;
        self.dopamine = prediction_error.clamp(-1.0, 1.0);
        self.predicted_reward = 0.9 * self.predicted_reward + 0.1 * actual_reward;
        // Tier 4.2: accumulates the hormonal sum (runtime-only, read by EliteDna).
        // tick_count is incremented in update_cortisol (per-tick anchor).
        self.dopamine_sum += self.dopamine as f64;
    }

    /// Plasticity multiplier for myelination.
    /// Tight clamp [0.5, 1.5] to avoid overflow in the cumulative scale × 1.005^N
    /// (see BitLinear::myelinate: it already has a 10x cap but dopamine must not amplify further).
    pub fn learning_modulation(&self) -> f32 {
        (1.0 + self.dopamine * 0.5).clamp(0.5, 1.5)
    }

    // ------------------------------------------------------------------
    // Tier 3.2 — Cortisol (integral fear)
    // ------------------------------------------------------------------

    /// Integrates wall proximity over time with decay (~2s of memory).
    /// min_wall_distance in meters (smallest lateral distance).
    pub fn update_cortisol(&mut self, min_wall_distance: f32, dt: f32) {
        let threat = (1.0 / (min_wall_distance + 1.0)).min(1.0);
        self.cortisol = (self.cortisol * 0.95 + threat * dt * 2.0).clamp(0.0, 1.0);
        // Tier 4.2: accumulates the sum + increments tick_count (single per-tick anchor).
        self.cortisol_sum += self.cortisol as f64;
        self.tick_count = self.tick_count.saturating_add(1);
    }

    /// Current risk aversion (0.0 = calm, 1.0 = terrified).
    pub fn risk_aversion(&self) -> f32 {
        self.cortisol
    }

    // ------------------------------------------------------------------
    // Tier 3.3 — Curiosity (novelty in the retina)
    // ------------------------------------------------------------------

    /// Updates curiosity based on the retina variance over the last
    /// 16 frames. Low variance (monotony) → curiosity rises.
    /// High variance (sensory novelty) → curiosity relaxes slowly.
    ///
    /// `current_retina` is the post-frame binocular visual_cortex (64 i8 channels).
    pub fn update_curiosity(&mut self, current_retina: &[i8; RETINA_TOTAL], dt: f32) {
        let mut var_acc = 0.0f32;
        let n = self.retina_history.len() as f32;
        for ch in 0..RETINA_TOTAL {
            let mut sum = 0.0f32;
            for frame in self.retina_history.iter() {
                sum += frame[ch] as f32 / 127.0;
            }
            let mean = sum / n;
            for frame in self.retina_history.iter() {
                let d = (frame[ch] as f32 / 127.0) - mean;
                var_acc += d * d;
            }
        }
        let novelty = (var_acc / (n * RETINA_TOTAL as f32)).sqrt().min(1.0);

        // Monotony (1 - novelty) feeds curiosity at a rate of 0.5/s.
        let delta = (1.0 - novelty) * dt * 0.5;
        // Gentle decay so it does not lock at 1.0 when the retina barely ever changes.
        self.curiosity = (self.curiosity + delta - dt * 0.05).clamp(0.0, 1.0);
        // Tier 4.2: accumulates the sum (tick_count already incremented in update_cortisol).
        self.curiosity_sum += self.curiosity as f64;

        // Ring buffer: rotate + insert at slot 0.
        self.retina_history.rotate_right(1);
        self.retina_history[0] = *current_retina;
    }

    /// Exploration drive (0..1). Used by the motor to inject noise.
    pub fn exploration_drive(&self) -> f32 {
        self.curiosity
    }

    // ------------------------------------------------------------------
    // Tier 4.2 — Partial epigenetics (means + baseline seeding)
    // ------------------------------------------------------------------

    /// Running mean of cortisol over the organism's lifetime.
    pub fn mean_cortisol(&self) -> f32 {
        (self.cortisol_sum / self.tick_count.max(1) as f64) as f32
    }
    /// Running mean of dopamine (RPE) over the lifetime.
    pub fn mean_dopamine(&self) -> f32 {
        (self.dopamine_sum / self.tick_count.max(1) as f64) as f32
    }
    /// Running mean of curiosity over the lifetime.
    pub fn mean_curiosity(&self) -> f32 {
        (self.curiosity_sum / self.tick_count.max(1) as f64) as f32
    }

    // ------------------------------------------------------------------
    // Tier 6.1 — Predictive coding drive
    // ------------------------------------------------------------------

    /// Hook called by the engine after computing the mean L1 prediction error.
    /// Feeds curiosity as an intrinsic drive (clamp [0,1]) and accumulates the sum
    /// for the lifetime mean. The small gain (0.01) avoids saturating curiosity in one tick;
    /// persistent prediction errors reinforce exploration.
    pub fn apply_prediction_error(&mut self, prediction_error: f32) {
        self.curiosity = (self.curiosity + prediction_error * 0.01).clamp(0.0, 1.0);
        self.prediction_error_sum += prediction_error as f64;
        self.prediction_error_samples = self.prediction_error_samples.saturating_add(1);
    }

    /// Lifetime mean of prediction_error (diagnostics / fitness hook).
    pub fn prediction_error_mean(&self) -> f32 {
        (self.prediction_error_sum / self.prediction_error_samples.max(1) as f64) as f32
    }

    /// Tier 4.2: initializes hormonal baselines inherited from the parent(s).
    /// 10% of the mean metabolic state passes to the child at birth.
    /// - initial cortisol  = parent_mean_cortisol * 0.1   (trauma baseline)
    /// - predicted_reward  = parent_mean_dopamine * 0.1   (mood baseline)
    /// - current dopamine  = 0.0                          (fresh RPE)
    /// - initial curiosity = parent_mean_curiosity * 0.1  (exploratory tendency)
    // ------------------------------------------------------------------
    // Tier 2.3 — Episodic memory API
    // ------------------------------------------------------------------

    /// Captures an episodic trace if it is salient.
    /// Condition: |reward| > 0.1 OR wall_hit. Ring buffer with cap EPISODIC_CAP:
    /// pushes the oldest one out when full.
    pub fn record_episode(
        &mut self,
        retina: &[i8; RETINA_TOTAL],
        action: [f32; 2],
        reward: f32,
        wall_hit: bool,
    ) {
        if reward.abs() <= 0.1 && !wall_hit {
            return;
        }
        if self.episodic_buffer.len() >= EPISODIC_CAP {
            self.episodic_buffer.pop_front();
        }
        self.episodic_buffer.push_back(EpisodicTrace {
            retina: *retina,
            action,
            reward,
        });
    }

    pub fn seed_epigenetic(
        &mut self,
        parent_cortisol: f32,
        parent_dopamine: f32,
        parent_curiosity: f32,
    ) {
        const INHERITANCE: f32 = 0.1;
        self.cortisol = (parent_cortisol * INHERITANCE).clamp(0.0, 1.0);
        self.dopamine = 0.0;
        self.predicted_reward = (parent_dopamine * INHERITANCE).clamp(-1.0, 1.0);
        self.curiosity = (parent_curiosity * INHERITANCE).clamp(0.0, 1.0);
    }
}

impl Default for Metabolism {
    fn default() -> Self {
        Self {
            adenosine: 0.0,
            recovery_rate: 0.05,
            is_sleeping: false,
            cache_stress: 0.0,
            dopamine: 0.0,
            predicted_reward: 0.0,
            cortisol: 0.0,
            curiosity: 0.0,
            retina_history: [[0i8; RETINA_TOTAL]; 16],
            cortisol_sum: 0.0,
            dopamine_sum: 0.0,
            curiosity_sum: 0.0,
            tick_count: 0,
            episodic_buffer: VecDeque::with_capacity(EPISODIC_CAP),
            sleep_consolidated: false,
            mirror_activation_count: 0,
            prediction_error: 0.0,
            prediction_error_sum: 0.0,
            prediction_error_samples: 0,
        }
    }
}
