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
            primary_window: Some(Window { title: "IARA v2101 - IRON FORGE".into(), ..default() }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(0.0, 0.0, 0.01)))
        .insert_resource(RaceSettings {
            generation: 1,
            best_weights: None,
            best_scale: 1.0,
            best_recipe: String::from("Sincronizando Valência..."),
            best_fitness: 0.0,
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
    race: Res<RaceSettings>, 
    dev: Res<DevSettings>, 
    mut meshes: ResMut<Assets<Mesh>>, 
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut el_materials: ResMut<ElementMaterials>,
    world_res: Res<world::VoxelWorldResource>,
) {
    commands.insert_resource(AmbientLight { brightness: 1000.0, color: Color::WHITE, ..default() });

    // Câmera Única Cinematográfica
    commands.spawn((
        Camera3d::default(),
        Camera { hdr: true, ..default() },
        Tonemapping::TonyMcMapface,
        Bloom::default(),
        MainCamera { yaw: 0.0, pitch: 0.5, dist: 600.0 },
    ));
    
    evolution::spawn_multiverse(&mut commands, &mut meshes, &mut materials, &mut el_materials, &race, &dev, &world_res.0);
}
