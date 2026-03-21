/// IARA v2101 - SOVEREIGN F1 ENGINE (RESTORED & MODULAR)
/// Arquitetura limpa com a alma da versão que funcionava.

use bevy::prelude::*;
use universe::world::{VoxelWorld, VoxelWorldResource};
use universe::simulation::RacerPlugin;
use universe::ui::{UiPlugin, LeaderboardUI};
use universe::evolution::{EvolutionPlugin, RaceSettings, spawn_swarm};

fn main() {
    let initial_seed = 101;
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.05, 0.05, 0.1)))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window { title: "IARA v2101 - Sovereign F1 Engine".into(), ..default() }),
            ..default()
        }))
        .insert_resource(RaceSettings {
            generation: 1,
            seed: initial_seed,
            best_weights: None,
            selected_racer: None,
            world_entities: Vec::new(),
            should_reset: false,
        })
        .insert_resource(VoxelWorldResource(VoxelWorld::new(initial_seed)))
        .add_plugins((RacerPlugin, UiPlugin, EvolutionPlugin))
        .add_systems(Startup, (setup_scene, setup_hud_container))
        .run();
}

fn setup_hud_container(mut commands: Commands) {
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Px(10.0), top: Val::Px(10.0),
            width: Val::Px(260.0), flex_direction: FlexDirection::Column,
            padding: UiRect::all(Val::Px(10.0)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.85)),
        LeaderboardUI,
    ));
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut race: ResMut<RaceSettings>,
    world_res: Res<VoxelWorldResource>,
) {
    // Iluminação Metafísica: Escuridão profunda com luzes de acento neon
    commands.spawn((
        DirectionalLight { 
            illuminance: 12000.0, 
            shadows_enabled: true, 
            color: Color::srgb(0.8, 0.9, 1.0),
            ..default() 
        },
        Transform::from_xyz(100.0, 400.0, 100.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    
    // Ambient Light baixa para destacar os carros e as bordas de energia
    commands.insert_resource(AmbientLight { brightness: 150.0, color: Color::srgb(0.1, 0.1, 0.2), ..default() });

    let world = &world_res.0;
    let (r_mesh, w_mesh) = world.generate_mesh();
    
    // Pista: Materiais com reflexo metálico e profundidade
    let road_e = commands.spawn((
        Mesh3d(meshes.add(r_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial { 
            perceptual_roughness: 0.2, 
            metallic: 0.8,
            base_color: Color::srgb(0.1, 0.1, 0.1),
            ..default() 
        })),
    )).id();
    
    // Muros: Barreiras de Energia Pura
    let wall_e = commands.spawn((
        Mesh3d(meshes.add(w_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial { 
            base_color: Color::srgb(1.0, 0.1, 0.3), 
            emissive: LinearRgba::from(Color::srgb(1.0, 0.1, 0.2)) * 10.0, 
            ..default() 
        })),
    )).id();

    race.world_entities = vec![road_e, wall_e];
    spawn_swarm(&mut commands, &mut meshes, &mut materials, &race, world);
    
    // Câmera Inicial
    commands.spawn(Camera3d::default());
}
