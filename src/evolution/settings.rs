use bevy::prelude::*;
use std::collections::VecDeque;
use super::dna::{EliteDna, MorphoGenes, HypernetSeed};

pub const FITNESS_HISTORY_CAP: usize = 100;

#[derive(Resource)]
pub struct RaceSettings {
    pub generation: usize,
    pub seed: u64,
    pub best_weights: Option<Vec<i8>>,
    pub best_scale: f32,
    pub best_slow_weights: Option<Vec<i8>>,
    pub best_slow_scale: f32,
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
    pub elite_pool: Vec<EliteDna>,  // Top-5 DNAs da última geração (tournament pool)
    pub champion_path: Vec<Vec3>,   // Traçado do último campeão (spawn → última posição)
    pub champion_net_dist: f32,     // Distância NET do último campeão (para label do traçado)
    pub gens_since_crown: u32,      // Gens consecutivas sem coroar — destrava threshold se > N
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
}

#[derive(Resource)]
pub struct DevSettings {
    pub mutation_rate: f32,
    pub pain_sensitivity: f32,
    pub wave_amplitude: f32,
    pub flow_influence: f32,
}

impl Default for DevSettings {
    fn default() -> Self {
        Self {
            mutation_rate: 0.008,  // v4: 0.015 ainda erodia axiomas (gen 49 dna_view mostrou
                                   // canais 79/82/85-86 apagados). 0.008 = ~65 pesos/gen = 49%
                                   // menos erosão. Combinado com reinject_axioms preserva
                                   // sinais críticos indefinidamente.
            pain_sensitivity: 1.0,
            wave_amplitude: 70.0,
            flow_influence: 0.5,
        }
    }
}
