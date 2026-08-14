use bevy::prelude::*;
use std::collections::VecDeque;
use super::dna_genesis::{EliteDna, MorphoGenes, HypernetSeed};

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
    pub best_reflex_weights: Option<Vec<f32>>, // f32 runtime; quantize só ao salvar TRIN no DNA
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
    pub champion_distance: f32,  // Distância do DNA salvo em CHAMPION.dna (só muda quando DNA é coroado)
    pub best_ever_dist: f32,     // Protege contra regressão — só atualiza com 10% melhoria
    pub elite_pool: Vec<EliteDna>,  // Top-5 DNAs da última geração (tournament pool)
    // Niching: cluster id paralelo a elite_pool (mesmo índice). Populado por
    // evolution_pipeline após clustering por BehaviorFingerprint (Hamming ≤ 8).
    // Usado por spawn.rs em tournament para rejeitar pares do mesmo cluster.
    pub elite_clusters: Vec<usize>,
    pub champion_path: Vec<Vec3>,   // Traçado do último campeão (spawn → última posição)
    pub champion_net_dist: f32,     // Distância NET do último campeão (para label do traçado)
    pub gens_since_crown: u32,      // Gens consecutivas sem coroar — destrava threshold se > N
    /// Fração da POPULAÇÃO que fechou ≥1 volta na geração anterior.
    /// Rege o currículo de ambiente (ver `pipeline::decide_next_env_level_pop`). Antes ele
    /// era regido pela distância do CAMPEÃO, e um único carro de sorte levava os 60 para um
    /// nível em que a aderência cai a 0,33 e a frenagem exigida (130 m) passa do alcance do
    /// sensor mais longo (120 m) — cegueira, não dificuldade.
    pub frac_completou_volta: f32,
    /// Quanto a pista está estreitada em relação à largura cheia, em metros.
    /// Persiste entre gerações e anda no máximo 0,25 m por vez, nos dois sentidos — pode
    /// ALARGAR de volta quando a população não sustenta. Ver o currículo de largura em
    /// `pipeline.rs`, que antes era dirigido pela distância do campeão.
    pub estreitamento: f32,
    pub last_gen_path: Vec<Vec3>,   // Traçado do melhor da última gen (refresca sempre)
    pub last_gen_net: f32,          // NET do melhor da última gen
    pub last_gen_was_grid: bool,    // true se o melhor nasceu na largada (elegível pra coroar)
    pub is_validation_gen: bool,    // próxima gen roda env fixo (level 0); só valida DNA sem ruído
    pub gens_until_validation: u32, // conta até próxima validation (default: cada 3 gens)
    pub env_level: u8,              // nível curricular aplicado na gen atual (0..=3)
    // Batched spawn: evita freeze ao spawnar N carros de uma vez. Cada gen seta
    // Some((0, swarm_count)); drain_spawn_queue consome em blocos de SPAWN_BATCH_SIZE
    // por frame até virar None. Sim continua rodando entre batches — UI fica responsiva.
    pub pending_spawn: Option<(usize, usize)>,
    // Tier 1.2 — histórico de fitness média (últimas 100 gens) pra emergency brake.
    pub fitness_history: VecDeque<f32>,
    // Tier 1.2 — se > 0, próximas N gens usam axiom_strength mínimo 0.5 (socorro).
    pub axiom_boost_counter: u32,
    // Tier 1.3 — Axiom como prior externo (potential-based shaping, Ng-Harada-Russell 1999).
    // axiom_policy() em motor.rs soma bias aditivo ao output do brain escalado por este valor.
    // Atualizado pela spawn.rs a cada gen com a mesma curva de decay: (1.0 - gen/50).max(0),
    // elevado a 0.5 se axiom_boost_counter > 0. Consumido por update_racers após execute_cycle.
    pub current_axiom_strength: f32,
    // Tier 6.3/6.4 — morpho + hypernet do champion atual, persistidos via save_champion_dna_full.
    pub current_morpho: MorphoGenes,
    pub current_hypernet: HypernetSeed,
    // Telemetry: gravação de dados por geração (NDJSON stats + binary replay)
    pub telemetry: super::telemetry::TelemetryWriter,
    /// Escritor do novo formato de replay (128 bytes/frame) — gravado quando
    /// lab.record_replay=true. None = sem gravação. Aberto no início de cada geração
    /// e fechado (flush) no final pelo evolution_pipeline.
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
    /// Toggle vision cones (fovea raycast verde) on selected racer. KeyH.
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
