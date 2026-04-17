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
    pub champion_path: Vec<Vec3>,   // Traçado do último campeão (spawn → última posição)
    pub champion_net_dist: f32,     // Distância NET do último campeão (para label do traçado)
    pub gens_since_crown: u32,      // Gens consecutivas sem coroar — destrava threshold se > N
    pub last_gen_path: Vec<Vec3>,   // Traçado do melhor da última gen (refresca sempre)
    pub last_gen_net: f32,          // NET do melhor da última gen
    pub last_gen_was_grid: bool,    // true se o melhor nasceu na largada (elegível pra coroar)
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
            mutation_rate: 0.015,  // v1: 0.15 destruia aprendizado. v2: 0.04 ainda erodia DNA
                                   // — usuário observou gen 2 com maioria batendo na parede.
                                   // v3 (agora): 0.015 → ~123 pesos randomizados/gen no gen 1,
                                   // decai pra ~40/gen após 40 gens. DNA preservado em 98.5%.
            pain_sensitivity: 1.0,
            wave_amplitude: 70.0,
            flow_influence: 0.5,
        }
    }
}
