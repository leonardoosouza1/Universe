pub mod brain;
pub mod brain_capacity;
/// Pure numeric core — functions with no GPU/Bevy dependency (E1).
pub mod math;
/// Activation metrics + runtime guards, built on top of `math::*` (E4).
pub mod analysis;
pub mod visao;
pub mod pista_gen;
pub mod neuro;
pub mod lab;
pub mod safety;
pub mod chaos;
pub mod gate;
pub mod engine;
pub mod world;
/// GPU simulation core, with no graphics (Bevy is only the visualizer on top).
pub mod gpu;
pub mod types;
pub mod layers;
pub mod specialist;
pub mod float_layers;
pub mod dense_layer;
pub mod kan;
pub mod resonance;
pub mod cfc;
/// The single deterministic generator — see `rng.rs` for the sign bug that motivated it.
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
