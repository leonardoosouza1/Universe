/// IARA v2101 - Digital Metabolism & Homeostasis
/// Based on Codex v2101 - Module I
///
/// Tier 3.1+3.2: drives biológicos adicionados (dopamina = RPE, cortisol = medo integral).
/// Todos novos campos são RUNTIME-ONLY — resetam em spawn (Default::default). NÃO persistem
/// no .dna v3.

use std::collections::VecDeque;
use crate::config::iara::vision::RETINA_TOTAL;

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

    // --- Tier 3.1: Dopamina = Reward Prediction Error ---
    /// Dopamina momentânea [-1.0, 1.0] — sinal de RPE (actual - predicted reward).
    /// Spike positivo em surpresa boa, dip em surpresa ruim, ~0 em reward esperado.
    pub dopamine: f32,
    /// Reward previsto — running average (TD-learning α=0.1) do reward recebido.
    pub predicted_reward: f32,

    // --- Tier 3.2: Cortisol = medo/stress integral ---
    /// Cortisol [0.0, 1.0] — integra proximidade-de-parede no tempo.
    /// Alto cortisol → comportamento defensivo (throttle ↓, steering ↓, risk_damp ↑).
    pub cortisol: f32,

    // --- Tier 3.3: Curiosidade = novelty em retina ---
    /// Curiosidade [0.0, 1.0] — sobe quando retina está monótona (baixa variância
    /// no histórico de 16 frames). Motor usa para injetar ruído de exploração
    /// quando também está stuck (stagnation > 60 frames).
    pub curiosity: f32,
    /// Ring buffer dos últimos 16 frames da retina binocular (64 canais).
    /// Rotate_right(1) + insert no índice 0. Compute variance across history.
    pub retina_history: [[i8; RETINA_TOTAL]; 16],

    // ------------------------------------------------------------------
    // Tier 4.2 — Epigenética parcial (running means hormonais)
    // ------------------------------------------------------------------
    // Acumuladores RUNTIME-ONLY. Reseta em spawn (Default). NÃO persistem em .dna.
    // Capturados por EliteDna ao final da gen e herdados 10% pelo filho no spawn.
    pub cortisol_sum: f64,
    pub dopamine_sum: f64,
    pub curiosity_sum: f64,
    pub tick_count: u64,

    // ------------------------------------------------------------------
    // Tier 2.3 — Memória episódica (consolidada em sleep)
    // ------------------------------------------------------------------
    /// Ring buffer de traces salientes (cap 64). Push quando
    /// |reward| > 0.1 OU wall_hit. Drenado em sleep por consolidate_episodic.
    /// RUNTIME-only. Não persiste em .dna.
    pub episodic_buffer: VecDeque<EpisodicTrace>,
    /// Guard p/ evitar consolidar múltiplas vezes no mesmo ciclo de sono.
    /// Reset quando adenosine < 0.5 (organismo acordou o suficiente).
    pub sleep_consolidated: bool,

    // ------------------------------------------------------------------
    // Tier 6.8 — Mirror neurons / observational learning
    // ------------------------------------------------------------------
    /// Conta quantos ticks o mirror hook disparou (cosine(retina, champion_mean) > 0.8).
    /// RUNTIME-only. Fitness hook pode ler pra bonificar aprendizado por imitação.
    pub mirror_activation_count: u32,

    // ------------------------------------------------------------------
    // Tier 6.1 — Predictive coding como drive primário
    // ------------------------------------------------------------------
    /// Erro de predição da retina no tick atual (L1 médio, ~[0, 2]).
    /// Atualizado a cada tick pelo engine; feeds curiosity como drive intrínseco.
    /// RUNTIME-only.
    pub prediction_error: f32,
    /// Soma running do prediction_error p/ média lifetime (diagnóstico).
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
        }
 else {
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
    // Tier 3.1 — Dopamina (RPE)
    // ------------------------------------------------------------------

    /// Atualiza dopamina via TD-error: dopamine = actual - predicted (clamped).
    /// predicted_reward é running average (α=0.1).
    pub fn update_dopamine(&mut self, actual_reward: f32) {
        let prediction_error = actual_reward - self.predicted_reward;
        self.dopamine = prediction_error.clamp(-1.0, 1.0);
        self.predicted_reward = 0.9 * self.predicted_reward + 0.1 * actual_reward;
        // Tier 4.2: acumula soma hormonal (runtime-only, lido por EliteDna).
        // tick_count é incrementado em update_cortisol (anchor por tick).
        self.dopamine_sum += self.dopamine as f64;
    }

    /// Multiplicador de plasticidade pra myelination.
    /// Clamp apertado [0.5, 1.5] pra evitar overflow em scale × 1.005^N acumulativo
    /// (ver BitLinear::myelinate: já tem cap 10x mas dopamina não deve amplificar mais).
    pub fn learning_modulation(&self) -> f32 {
        (1.0 + self.dopamine * 0.5).clamp(0.5, 1.5)
    }

    // ------------------------------------------------------------------
    // Tier 3.2 — Cortisol (medo integral)
    // ------------------------------------------------------------------

    /// Integra proximidade de parede no tempo com decay (~2s de memória).
    /// min_wall_distance em metros (menor distância lateral).
    pub fn update_cortisol(&mut self, min_wall_distance: f32, dt: f32) {
        let threat = (1.0 / (min_wall_distance + 1.0)).min(1.0);
        self.cortisol = (self.cortisol * 0.95 + threat * dt * 2.0).clamp(0.0, 1.0);
        // Tier 4.2: acumula soma + incrementa tick_count (anchor único por tick).
        self.cortisol_sum += self.cortisol as f64;
        self.tick_count = self.tick_count.saturating_add(1);
    }

    /// Aversão a risco atual (0.0 = calmo, 1.0 = aterrorizado).
    pub fn risk_aversion(&self) -> f32 {
        self.cortisol
    }

    // ------------------------------------------------------------------
    // Tier 3.3 — Curiosidade (novelty em retina)
    // ------------------------------------------------------------------

    /// Atualiza curiosidade com base na variância da retina ao longo dos
    /// últimos 16 frames. Baixa variância (monotonia) → curiosity sobe.
    /// Alta variância (novidade sensorial) → curiosity relaxa lentamente.
    ///
    /// `current_retina` é o visual_cortex binocular pós-frame (64 canais i8).
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

        // Monotonia (1 - novelty) alimenta curiosity com rate 0.5/s.
        let delta = (1.0 - novelty) * dt * 0.5;
        // Decay suave pra não travar em 1.0 quando retina muda pouco por sempre.
        self.curiosity = (self.curiosity + delta - dt * 0.05).clamp(0.0, 1.0);
        // Tier 4.2: acumula soma (tick_count já incrementado em update_cortisol).
        self.curiosity_sum += self.curiosity as f64;

        // Ring buffer: rotate + inserir no slot 0.
        self.retina_history.rotate_right(1);
        self.retina_history[0] = *current_retina;
    }

    /// Drive de exploração (0..1). Usado pelo motor para injetar ruído.
    pub fn exploration_drive(&self) -> f32 {
        self.curiosity
    }

    // ------------------------------------------------------------------
    // Tier 4.2 — Epigenética parcial (means + seeding de baselines)
    // ------------------------------------------------------------------

    /// Média running do cortisol ao longo da vida do organismo.
    pub fn mean_cortisol(&self) -> f32 {
        (self.cortisol_sum / self.tick_count.max(1) as f64) as f32
    }
    /// Média running da dopamina (RPE) ao longo da vida.
    pub fn mean_dopamine(&self) -> f32 {
        (self.dopamine_sum / self.tick_count.max(1) as f64) as f32
    }
    /// Média running da curiosidade ao longo da vida.
    pub fn mean_curiosity(&self) -> f32 {
        (self.curiosity_sum / self.tick_count.max(1) as f64) as f32
    }

    // ------------------------------------------------------------------
    // Tier 6.1 — Predictive coding drive
    // ------------------------------------------------------------------

    /// Hook chamado pelo engine após computar o prediction error L1 médio.
    /// Feeds como drive intrínseco em curiosity (clamp [0,1]) e acumula soma
    /// p/ média lifetime. Pequeno ganho (0.01) evita saturar curiosity num tick;
    /// erros persistentes de predição reforçam exploração.
    pub fn apply_prediction_error(&mut self, prediction_error: f32) {
        self.curiosity = (self.curiosity + prediction_error * 0.01).clamp(0.0, 1.0);
        self.prediction_error_sum += prediction_error as f64;
        self.prediction_error_samples = self.prediction_error_samples.saturating_add(1);
    }

    /// Média lifetime do prediction_error (diagnóstico / fitness hook).
    pub fn prediction_error_mean(&self) -> f32 {
        (self.prediction_error_sum / self.prediction_error_samples.max(1) as f64) as f32
    }

    /// Tier 4.2: inicializa baselines hormonais herdados do(s) parent(s).
    /// 10% do estado metabólico médio passa pro filho ao nascer.
    /// - cortisol inicial = parent_mean_cortisol * 0.1   (trauma baseline)
    /// - predicted_reward = parent_mean_dopamine * 0.1   (mood baseline)
    /// - dopamine atual   = 0.0                          (RPE fresco)
    /// - curiosity inicial= parent_mean_curiosity * 0.1  (tendência exploratória)
    // ------------------------------------------------------------------
    // Tier 2.3 — Memória episódica API
    // ------------------------------------------------------------------

    /// Captura um trace episódico se for saliente.
    /// Condição: |reward| > 0.1 OU wall_hit. Ring buffer com cap EPISODIC_CAP:
    /// empurra o mais antigo pra fora quando cheio.
    pub fn record_episode(&mut self, retina: &[i8; RETINA_TOTAL], action: [f32; 2], reward: f32, wall_hit: bool) {
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

    pub fn seed_epigenetic(&mut self, parent_cortisol: f32, parent_dopamine: f32, parent_curiosity: f32) {
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
