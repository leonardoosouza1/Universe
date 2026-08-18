/// Activation metrics + runtime guards, built on top of `math::*` (E4).
pub mod analysis;
pub mod brain;
pub mod brain_capacity;
pub mod cfc;
pub mod chaos;
pub mod config;
pub mod dense_layer;
pub mod engine;
pub mod environment;
pub mod evolution;
pub mod float_layers;
pub mod gate;
/// GPU simulation core, with no graphics (Bevy is only the visualizer on top).
pub mod gpu;
pub mod kan;
pub mod lab;
pub mod layers;
/// Pure numeric core — functions with no GPU/Bevy dependency (E1).
pub mod math;
pub mod memory;
pub mod mesh;
pub mod metabolism;
pub mod neuro;
pub mod perception;
pub mod physics;
pub mod pista_gen;
/// D9 scripted predator — pure chase policy, no GPU (hostile-world spec).
pub mod predator;
pub mod prefabs;
pub mod reality;
pub mod replay;
pub mod repominer;
pub mod resonance;
/// D7 scarce resource — energy stations, pure state machine, no GPU.
pub mod resources;
/// The single deterministic generator — see `rng.rs` for the sign bug that motivated it.
pub mod rng;
pub mod safety;
pub mod scenery;
pub mod simulation;
pub mod specialist;
pub mod substrate;
/// Tyre thermal model — heat, Newtonian cooling and the grip curve, no GPU.
pub mod tire_thermal;
pub mod types;
pub mod ui;
pub mod ui_menu;
pub mod visao;
pub mod visualization;
pub mod world;

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
