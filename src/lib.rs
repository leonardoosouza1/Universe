pub mod brain_capacity;
pub mod visao;
pub mod pista_gen;
pub mod neuro;
pub mod lab;
pub mod safety;
pub mod chaos;
pub mod gate;
pub mod engine;
pub mod world;
/// Núcleo de simulação em GPU, sem parte gráfica (Bevy vira só o visualizador).
pub mod gpu;
pub mod types;
pub mod layers;
pub mod specialist;
pub mod float_layers;
pub mod dense_layer;
pub mod kan;
pub mod resonance;
pub mod cfc;
/// Gerador determinístico único — ver `rng.rs` para o bug de sinal que motivou.
pub mod rng;
pub mod perception;
pub mod metabolism;
pub mod memory;
pub mod simulation;
pub mod replay;
pub mod ui;
pub mod evolution;
pub mod reality;
pub mod repominer;
pub mod mesh;
pub mod physics;
pub mod prefabs;
pub mod environment;
pub mod config;
pub mod ui_menu;
pub mod substrate;
pub mod scenery;
pub mod visualization;

use bevy::prelude::*;

#[derive(States, Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
pub enum AppState {
    #[default]
    MainMenu,
    Simulation,
}

#[derive(Component)]
pub struct SimulationEntity;

#[derive(Resource, Clone, Copy)]
pub struct HeadlessMode {
    pub enabled: bool,
    pub gens_target: u32,
}
