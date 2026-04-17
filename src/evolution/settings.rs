use bevy::prelude::*;
use super::dna::EliteDna;

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
            mutation_rate: 0.04,   // Era 0.15 (1228 pesos randomizados/gen = destruia aprendizado).
                                   // Agora ~328/gen no gen 0, decai para ~40/gen após 50 gens.
            pain_sensitivity: 1.0,
            wave_amplitude: 70.0,
            flow_influence: 0.5,
        }
    }
}
