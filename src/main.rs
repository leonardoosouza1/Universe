use bevy::prelude::*;
use universe::world::{VoxelWorld, VoxelWorldResource};
use universe::simulation::RacerPlugin;
use universe::ui::{UiPlugin, LeaderboardUI};
use universe::evolution::{EvolutionPlugin, RaceSettings, DevSettings, spawn_swarm};
use universe::environment::EnvironmentalState;

fn main() {
    let initial_seed = 42;
    let mut app = App::new();
    
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window { title: "IARA v2101 - Sovereign Dev Engine".into(), ..default() }),
        ..default()
    }))
    .insert_resource(ClearColor(Color::srgb(0.05, 0.05, 0.1)))
    .insert_resource(RaceSettings {
        generation: 1,
        seed: initial_seed,
        best_weights: None,
        best_scale: 1.0,
        selected_racer: None,
        world_entities: Vec::new(),
        should_reset: false,
        hard_reset: false,
        show_confirm_modal: false,
        camera_distance: 150.0,
        is_free_cam: false,
    })
    .insert_resource(DevSettings::default())
    .insert_resource(VoxelWorldResource(VoxelWorld::new(initial_seed)))
    .insert_resource(EnvironmentalState::default())
    .add_plugins((RacerPlugin, UiPlugin, EvolutionPlugin, universe::physics::PhysicsPlugin))
    .add_systems(Startup, (setup_scene, setup_hud_container))
    .run();
}

fn setup_hud_container(mut commands: Commands) {
    commands.spawn((
        Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            ..default()
        },
        LeaderboardUI,
    ));
}

fn setup_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    race: Res<RaceSettings>,
    dev: Res<DevSettings>,
    world_res: Res<VoxelWorldResource>,
) {
    commands.spawn((
        DirectionalLight { 
            illuminance: 25000.0, 
            shadows_enabled: true, 
            color: Color::srgb(1.0, 0.98, 0.95),
            ..default() 
        },
        Transform::from_xyz(150.0, 450.0, 150.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    
    // Diminuindo a luz ambiente para simulação noturna severa (Deixa os espasmos mortais do freio e faróis puros visíveis)
    commands.insert_resource(AmbientLight { brightness: 25.0, color: Color::srgb(0.8, 0.85, 1.0), ..default() });

    let world = &world_res.0;
    let (r_mesh, g_mesh, w_mesh) = world.generate_mesh();
    
    let _road_e = commands.spawn((
        Mesh3d(meshes.add(r_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial { base_color: Color::srgb(0.15, 0.15, 0.18), perceptual_roughness: 0.85, ..default() })),
    )).id();

    let _grass_e = commands.spawn((
        Mesh3d(meshes.add(g_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial { base_color: Color::srgb(0.1, 0.3, 0.1), perceptual_roughness: 0.9, ..default() })),
    )).id();
    
    let _wall_e = commands.spawn((
        Mesh3d(meshes.add(w_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial { base_color: Color::srgb(0.7, 0.7, 0.7), metallic: 1.0, perceptual_roughness: 0.1, ..default() })),
    )).id();

    // spawn_swarm agora exige &DevSettings
    spawn_swarm(&mut commands, &mut meshes, &mut materials, &race, world, &dev);

    // ADICIONAR CÂMERA (CORREÇÃO DA TELA PRETA)
    commands.spawn(Camera3d::default());
}
