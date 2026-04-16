use bevy::prelude::*;
use universe::world::{VoxelWorld, VoxelWorldResource};
use universe::simulation::RacerPlugin;
use universe::ui::{UiPlugin, LeaderboardUI};
use universe::evolution::{EvolutionPlugin, RaceSettings, DevSettings, spawn_swarm};
use universe::environment::EnvironmentalState;
use universe::config::UniverseConfig;
use universe::ui_menu::MenuPlugin;
use universe::{AppState, SimulationEntity};

fn main() {
    let config = UniverseConfig::load_or_create();
    let initial_track = config.current_track();
    let initial_seed = initial_track.seed;
    let initial_width = initial_track.track_width;
    
    let mut app = App::new();
    
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window { title: "IARA v2101 - Sovereign Dev Engine".into(), ..default() }),
        ..default()
    }))
    .insert_resource(ClearColor(Color::srgb(0.05, 0.05, 0.1)))
    .init_state::<AppState>()
    .insert_resource(config)
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
    .insert_resource(VoxelWorldResource(VoxelWorld::new(initial_seed, initial_width)))
    .insert_resource(EnvironmentalState::default())
    .add_plugins((RacerPlugin, UiPlugin, EvolutionPlugin, universe::physics::PhysicsPlugin, MenuPlugin))
    .add_systems(Startup, setup_main_camera)
    .add_systems(OnEnter(AppState::Simulation), (setup_scene, setup_hud_container))
    .add_systems(OnExit(AppState::Simulation), cleanup_simulation)
    .run();
}

fn cleanup_simulation(
    mut commands: Commands,
    query: Query<Entity, With<SimulationEntity>>,
    racer_query: Query<Entity, With<universe::simulation::Racer>>,
) {
    for e in &query {
        commands.entity(e).despawn();
    }
    for e in &racer_query {
        commands.entity(e).despawn();
    }
}

fn setup_main_camera(mut commands: Commands) {
    commands.spawn((Camera3d::default(), Transform::from_xyz(0.0, 100.0, 300.0).looking_at(Vec3::ZERO, Vec3::Y)));
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
    mut race: ResMut<RaceSettings>,
    dev: Res<DevSettings>,
    mut world_res: ResMut<VoxelWorldResource>,
    config: Res<UniverseConfig>,
) {
    let current_track = config.current_track();
    
    // Atualizar o mundo voxel com a seed do track selecionado
    let world = VoxelWorld::new(current_track.seed, current_track.track_width);
    world_res.0 = world;
    
    // Atualizar RaceSettings
    race.seed = current_track.seed;
    race.generation = 1;

    commands.spawn((
        DirectionalLight { 
            illuminance: 25000.0, 
            shadows_enabled: true, 
            color: Color::srgb(1.0, 0.98, 0.95),
            ..default() 
        },
        Transform::from_xyz(150.0, 450.0, 150.0).looking_at(Vec3::ZERO, Vec3::Y),
        SimulationEntity,
    ));
    
    commands.insert_resource(AmbientLight { brightness: 25.0, color: Color::srgb(0.8, 0.85, 1.0), ..default() });

    let world = &world_res.0;
    let (r_mesh, g_mesh, w_mesh) = world.generate_mesh();
    
    commands.spawn((
        Mesh3d(meshes.add(r_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial { base_color: Color::srgb(0.15, 0.15, 0.18), perceptual_roughness: 0.85, ..default() })),
        SimulationEntity,
    ));

    commands.spawn((
        Mesh3d(meshes.add(g_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial { base_color: Color::srgb(0.1, 0.3, 0.1), perceptual_roughness: 0.9, ..default() })),
        SimulationEntity,
    ));
    
    commands.spawn((
        Mesh3d(meshes.add(w_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial { base_color: Color::srgb(0.7, 0.7, 0.7), metallic: 1.0, perceptual_roughness: 0.1, ..default() })),
        SimulationEntity,
    ));

    spawn_swarm(&mut commands, &mut meshes, &mut materials, &race, world, &dev, &config);
}
