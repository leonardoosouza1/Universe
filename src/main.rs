use bevy::prelude::*;
use bevy::core_pipeline::bloom::Bloom;
use bevy::core_pipeline::tonemapping::Tonemapping;

pub mod simulation;
pub mod ui;
pub mod evolution;
pub mod engine;
pub mod metabolism;
pub mod layers;
pub mod cfc;
pub mod kan;
pub mod resonance;
pub mod memory;
pub mod types;
pub mod reality;
pub mod repominer;
pub mod mesh;
pub mod periodic_table;
pub mod world; 

use simulation::{AtomPlugin, MainCamera};
use ui::UiPlugin;
use evolution::{EvolutionPlugin, RaceSettings, DevSettings, ElementMaterials};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window { 
                title: "IARA v2101 - QUANTUM MATERIALS FORGE".into(), 
                ..default() 
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.005, 0.005, 0.01))) // Fundo ultra escuro para contraste
        .insert_resource(RaceSettings {
            generation: 1,
            best_weights: None,
            best_scale: 1000,
            best_recipe: String::from("Synthesizing Carbon-Silver Lattice..."),
            best_fitness: 0,
            should_reset: false,
            fossils: Vec::new(),
        })
        .insert_resource(DevSettings { time_scale: 1.0, ..default() })
        .insert_resource(world::VoxelWorldResource(world::VoxelWorld::new(42)))
        .add_plugins((AtomPlugin, UiPlugin, EvolutionPlugin))
        .add_systems(Startup, setup_scene)
        .run();
}

fn setup_scene(
    mut commands: Commands, 
    _race: Res<RaceSettings>, 
    _dev: Res<DevSettings>, 
    mut _meshes: ResMut<Assets<Mesh>>, 
    mut _materials: ResMut<Assets<StandardMaterial>>,
    mut _el_materials: ResMut<ElementMaterials>,
    _world_res: Res<world::VoxelWorldResource>,
) {
    // ILUMINAÇÃO CINEMATOGRÁFICA
    commands.insert_resource(AmbientLight { 
        brightness: 50.0, // Reduzido para não queimar as cores
        color: Color::srgb(0.5, 0.7, 1.0), 
        ..default() 
    });
    
    // Luz de Borda (Rim Light) para os átomos
    commands.spawn((
        DirectionalLight {
            illuminance: 2500.0,
            shadows_enabled: true,
            color: Color::srgb(1.0, 0.9, 0.8),
            ..default()
        },
        Transform::from_xyz(200.0, 500.0, 200.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    commands.spawn((
        Camera3d::default(),
        Camera { hdr: true, ..default() },
        Tonemapping::TonyMcMapface,
        Bloom::default(),
        MainCamera { yaw: 0.5, pitch: 0.2, dist: 600.0 },
    ));
    
    // O spawn do multiverso agora é chamado pelo EvolutionPlugin
}
