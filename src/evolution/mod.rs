use bevy::prelude::*;

pub mod dna;
pub mod dna_genesis;
pub mod logs;
pub mod pipeline;
pub mod settings;
pub mod spawn;
pub mod telemetry;

pub use dna_genesis::{
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
        app.add_systems(Update, (
            pipeline::evolution_pipeline,
            drain_spawn_queue,
        ).run_if(in_state(crate::AppState::Simulation)));
    }
}

/// Tamanho do bloco de spawn por frame. 2000 carros / 150 = ~14 frames ≈ 233ms @ 60fps.
/// Ajustável: maior = termina spawn mais rápido mas stutter maior cada frame; menor = suave
/// mas demora. 150 equilibra: ~3-4ms/frame de overhead, sem percepção de travada.
const SPAWN_BATCH_SIZE: usize = 150;

fn drain_spawn_queue(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut race: ResMut<RaceSettings>,
    world_res: Res<crate::world::VoxelWorldResource>,
    dev: Res<DevSettings>,
    config: Res<crate::config::UniverseConfig>,
) {
    let Some((next_i, total)) = race.pending_spawn else { return; };
    let batch_end = (next_i + SPAWN_BATCH_SIZE).min(total);
    let world = &world_res.0;
    spawn_swarm(
        &mut commands, &mut meshes, &mut materials, &mut race,
        world, &dev, &config,
        next_i, batch_end,
    );
    race.pending_spawn = if batch_end >= total { None } else { Some((batch_end, total)) };
}
