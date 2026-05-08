pub mod lab;
pub mod safety;
pub mod chaos;
pub mod gate;
pub mod engine;
pub mod world;
pub mod types;
pub mod layers;
pub mod specialist;
pub mod float_layers;
pub mod kan;
pub mod resonance;
pub mod cfc;
pub mod metabolism;
pub mod memory;
pub mod simulation;
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

use bevy::prelude::*;

#[derive(States, Debug, Clone, Copy, Eq, PartialEq, Hash, Default)]
pub enum AppState {
    #[default]
    MainMenu,
    Simulation,
}

#[derive(Component)]
pub struct SimulationEntity;
