use bevy::prelude::*;

pub mod dna;
pub mod logs;
pub mod pipeline;
pub mod settings;
pub mod spawn;

pub use dna::{
    EliteDna, DnaPayload,
    DNA_MAGIC, DNA_VERSION, DNA_FAST_SIZE, DNA_SLOW_SIZE, DNA_HEADER_SIZE, DNA_V3_TOTAL,
    save_champion_dna, load_champion_dna,
};
pub use logs::open_report_log;
pub use settings::{RaceSettings, DevSettings};
pub use spawn::spawn_swarm;

pub struct EvolutionPlugin;

impl Plugin for EvolutionPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, pipeline::evolution_pipeline.run_if(in_state(crate::AppState::Simulation)));
    }
}
