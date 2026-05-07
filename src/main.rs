use bevy::prelude::*;
use universe::world::{VoxelWorld, VoxelWorldResource};
use universe::simulation::RacerPlugin;
use universe::ui::{UiPlugin, LeaderboardUI};
use universe::evolution::{EvolutionPlugin, RaceSettings, DevSettings};
use universe::environment::EnvironmentalState;
use universe::config::UniverseConfig;
use universe::ui_menu::MenuPlugin;
use universe::{AppState, SimulationEntity};

#[derive(Resource, Clone, Copy)]
pub struct HeadlessMode {
    pub enabled: bool,
    pub gens_target: u32,
}

/// UI auto-start: pula menu e entra em Simulation direto.
/// Ativa quando `--seed N` passa em modo não-headless.
#[derive(Resource, Clone, Copy)]
pub struct UiAutoStart(pub bool);

fn main() {
    let mut config = UniverseConfig::load_or_create();
    let args: Vec<String> = std::env::args().collect();
    let headless = args.iter().any(|a| a == "--headless");
    let gens_target: u32 = args.iter().position(|a| a == "--gens")
        .and_then(|i| args.get(i+1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let swarm_override: Option<usize> = args.iter().position(|a| a == "--swarm")
        .and_then(|i| args.get(i+1))
        .and_then(|s| s.parse().ok());
    let time_scale_override: Option<f32> = args.iter().position(|a| a == "--speed")
        .and_then(|i| args.get(i+1))
        .and_then(|s| s.parse().ok());
    let seed_override: Option<u64> = args.iter().position(|a| a == "--seed")
        .and_then(|i| args.get(i+1))
        .and_then(|s| s.parse().ok());
    let width_override: Option<f32> = args.iter().position(|a| a == "--width")
        .and_then(|i| args.get(i+1))
        .and_then(|s| s.parse().ok());
    let no_holes = args.iter().any(|a| a == "--no-holes");
    let easy_track = args.iter().any(|a| a == "--easy");
    let city_mode = args.iter().any(|a| a == "--city");
    // FIX UI vs headless: flags persistem em universe_config.json. UI carrega
    // mesma config que último headless training → DNA e pista combinam.
    let final_no_holes = no_holes || config.no_holes;
    let final_easy = easy_track || config.easy_track;
    if final_no_holes {
        universe::world::NO_HOLES.store(true, std::sync::atomic::Ordering::Relaxed);
        println!("🕳  HOLES DISABLED");
    }
    if final_easy {
        universe::world::EASY_TRACK.store(true, std::sync::atomic::Ordering::Relaxed);
        println!("📏 EASY TRACK validation (oval-friendly)");
    }
    if city_mode {
        println!("🏙️  CITY MODE | navigation task enabled");
    }
    // Persiste se CLI passou — próximo run UI herda automatic.
    if no_holes { config.no_holes = true; }
    if easy_track { config.easy_track = true; }
    if city_mode { config.city_mode = true; }
    let _ = config.save();
    if let Some(n) = swarm_override { config.swarm_count = n; }
    if let Some(s) = time_scale_override { config.time_scale = s; }
    if let Some(seed) = seed_override {
        let w = width_override.unwrap_or(28.0);
        config.tracks.insert(0, universe::config::TrackProfile {
            name: format!("Procedural-{seed}-w{}", w as u32),
            seed,
            track_width: w,
            complexity: 0.5,
        });
        config.selected_track_index = 0;
        config.rotate_tracks = false;
        println!("🌱 SEED OVERRIDE | seed={} width={}m (procedural track)", seed, w);
    } else if let Some(w) = width_override {
        // Apenas largura override sem seed — aplica em current track.
        if let Some(t) = config.tracks.get_mut(config.selected_track_index) {
            t.track_width = w;
        }
        println!("📏 WIDTH OVERRIDE | width={}m", w);
    }
    if headless { config.time_scale = config.time_scale.max(8.0); }

    // Tier 5.3: ativa rotação de pistas via CLI (`--rotate`) ou env (UNIVERSE_ROTATE=1).
    let rotate_flag = args.iter().any(|a| a == "--rotate")
        || std::env::var("UNIVERSE_ROTATE").ok().map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false);
    if rotate_flag {
        config.rotate_tracks = true;
        println!("🔀 MULTI-TRACK ROTATION ENABLED | tracks={} | gens_per_track={}",
            config.tracks.len(), config.gens_per_track);
    }

    let initial_track = config.current_track();
    let initial_seed = initial_track.seed;
    let initial_width = initial_track.track_width + 4.0;

    let mut app = App::new();

    let window = if headless {
        Window {
            title: "IARA v2101 - HEADLESS".into(),
            visible: false,
            resolution: (320.0, 240.0).into(),
            ..default()
        }
    } else {
        Window { title: "IARA v2101 - Sovereign Dev Engine".into(), ..default() }
    };
    app.insert_resource(HeadlessMode { enabled: headless, gens_target });
    // UI auto-start quando --seed ou --city passou em modo não-headless.
    let ui_auto_start = (seed_override.is_some() || city_mode) && !headless;
    app.insert_resource(UiAutoStart(ui_auto_start));
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(window),
        ..default()
    }))
    .insert_resource(ClearColor(Color::srgb(0.42, 0.65, 0.88))) // céu azul claro
    .init_state::<AppState>()
    .insert_resource(config)
    .insert_resource(RaceSettings {
        generation: 1,
        seed: initial_seed,
        best_weights: None,
        best_car_model_idx: None,
        best_l2_weights: None,
        best_l3_weights: None,
        best_scale: 1.0,
        best_slow_weights: None,
        best_slow_scale: 1.0,
        best_gate_weights: None,
        best_gate_bias: None,
        best_reflex_weights: None,
        selected_racer: None,
        world_entities: Vec::new(),
        should_reset: false,
        hard_reset: false,
        show_confirm_modal: false,
        camera_distance: 28.0, // cinematic chase (≤80 = chase, >80 = top-down legacy)
        is_free_cam: false,
        death_markers: Vec::new(),
        global_best_distance: 0.0,
        global_best_position: None,
        champion_distance: 0.0,
        best_ever_dist: 0.0,
        elite_pool: Vec::new(),
        elite_clusters: Vec::new(),
        champion_path: Vec::new(),
        champion_net_dist: 0.0,
        gens_since_crown: 0,
        last_gen_path: Vec::new(),
        last_gen_net: 0.0,
        last_gen_was_grid: false,
        is_validation_gen: false,
        gens_until_validation: 3,
        pending_spawn: None,
        env_level: 1,
        fitness_history: std::collections::VecDeque::with_capacity(100),
        axiom_boost_counter: 0,
        current_axiom_strength: 1.0,
        current_morpho: universe::evolution::dna_genesis::MorphoGenes::default(),
        current_hypernet: universe::evolution::dna_genesis::HypernetSeed::default(),
        telemetry: universe::evolution::telemetry::TelemetryWriter::new(
            universe::evolution::telemetry::gen_run_id()
        ),
    })
    .insert_resource(DevSettings::default())
    .insert_resource(VoxelWorldResource(VoxelWorld::new(initial_seed, initial_width), None))
    .insert_resource(EnvironmentalState::default())
    .add_plugins((RacerPlugin, UiPlugin, EvolutionPlugin, universe::physics::PhysicsPlugin, MenuPlugin))
    .add_plugins(bevy_atmosphere::plugin::AtmospherePlugin)
    .add_systems(Startup, (setup_main_camera, apply_time_scale, headless_autostart))
    .add_systems(OnEnter(AppState::Simulation), (setup_scene, setup_hud_container, apply_time_scale))
    .add_systems(OnExit(AppState::Simulation), cleanup_simulation)
    .add_systems(Update, headless_exit_check)
    .run();
}

fn headless_autostart(
    hl: Res<HeadlessMode>,
    ui_auto: Res<UiAutoStart>,
    mut next_state: ResMut<NextState<AppState>>,
) {
    if ui_auto.0 {
        println!("🎮 UI AUTO-START | --seed ou --city detectado, entrando direto na simulação");
        next_state.set(AppState::Simulation);
        return;
    }
    if hl.enabled {
        println!("🤖 HEADLESS MODE | gens_target={} | swarm={} | speed={}×",
            hl.gens_target,
            "?", "?");
        next_state.set(AppState::Simulation);
    }
}

fn headless_exit_check(
    hl: Res<HeadlessMode>,
    race: Res<universe::evolution::RaceSettings>,
    mut exit: EventWriter<bevy::app::AppExit>,
) {
    if !hl.enabled { return; }
    if race.generation as u32 > hl.gens_target {
        println!("🤖 HEADLESS DONE | {} gens completed. Exiting.", hl.gens_target);
        exit.write(bevy::app::AppExit::Success);
    }
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
    use bevy::core_pipeline::bloom::Bloom;
    use bevy::core_pipeline::tonemapping::Tonemapping;
    use bevy::core_pipeline::motion_blur::MotionBlur;
    use bevy::pbr::{ScreenSpaceAmbientOcclusion, ScreenSpaceAmbientOcclusionQualityLevel,
                    VolumetricFog, ScreenSpaceReflections};
    use bevy_atmosphere::prelude::AtmosphereCamera;

    commands.spawn((
        Camera3d::default(),
        Camera {
            hdr: true, // bloom + tonemap chain
            ..default()
        },
        Msaa::Off, // SSAO requires MSAA off; AA via fxaa/taa instead
        Tonemapping::AcesFitted,
        Bloom::NATURAL,
        MotionBlur {
            shutter_angle: 0.5,
            samples: 2,
            ..default()
        },
        ScreenSpaceAmbientOcclusion {
            quality_level: ScreenSpaceAmbientOcclusionQualityLevel::Medium,
            ..default()
        },
        ScreenSpaceReflections::default(),
        VolumetricFog {
            ambient_intensity: 0.05,
            jitter: 0.5,
            ..default()
        },
        AtmosphereCamera::default(),
        Transform::from_xyz(0.0, 100.0, 300.0).looking_at(Vec3::ZERO, Vec3::Y),
        bevy::pbr::DistanceFog {
            color: Color::srgb(0.55, 0.72, 0.88),
            falloff: bevy::pbr::FogFalloff::Linear { start: 600.0, end: 2500.0 },
            ..default()
        },
    ));
}

fn apply_time_scale(
    config: Res<UniverseConfig>,
    mut time: ResMut<Time<Virtual>>,
) {
    time.set_relative_speed(config.time_scale);
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
    // Mesmo bônus de +4m que gen 2+ usa quando champion_distance == 0.
    let world = VoxelWorld::new(current_track.seed, current_track.track_width + 4.0);
    world_res.0 = world;
    
    // Atualizar RaceSettings
    race.seed = current_track.seed;
    race.generation = 1;

    commands.spawn((
        DirectionalLight {
            illuminance: 40000.0,
            shadows_enabled: true,
            color: Color::srgb(1.0, 0.96, 0.88),
            ..default()
        },
        bevy::pbr::VolumetricLight, // god rays do sol via VolumetricFog
        Transform::from_xyz(360.0, 600.0, 240.0).looking_at(Vec3::new(360.0, 0.0, 360.0), Vec3::Y),
        SimulationEntity,
    ));

    commands.insert_resource(AmbientLight {
        brightness: 180.0,
        color: Color::srgb(0.65, 0.78, 0.95), // azul céu indireto
        ..default()
    });

    let world = &world_res.0;
    let (r_mesh, g_mesh, w_mesh) = world.generate_mesh();
    
    // Registra os IDs em race.world_entities para que o evolution_pipeline possa
    // despawnar estas 3 meshes no fim da geração. Sem isso, cada reset criaria
    // uma pista NOVA sobreposta à anterior (efeito "duas pistas").
    let road_e = commands.spawn((
        Mesh3d(meshes.add(r_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.13, 0.13, 0.15),
            perceptual_roughness: 0.65,
            metallic: 0.05,
            reflectance: 0.4,
            ..default()
        })),
        SimulationEntity,
    )).id();

    let grass_e = commands.spawn((
        Mesh3d(meshes.add(g_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.18, 0.42, 0.18),
            perceptual_roughness: 0.95,
            metallic: 0.0,
            ..default()
        })),
        SimulationEntity,
    )).id();

    let wall_e = commands.spawn((
        Mesh3d(meshes.add(w_mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.92, 0.93, 0.95), // concreto branco F1
            perceptual_roughness: 0.55,
            metallic: 0.0,
            reflectance: 0.3,
            ..default()
        })),
        SimulationEntity,
    )).id();

    let mut world_entities = vec![road_e, grass_e, wall_e];
    // Atmospheric horizon: sea plane, mountain ring, sun. Decorative only —
    // sits far beyond the 720m voxel grid, never collides with cars.
    let scenery_entities = universe::scenery::spawn_scenery(&mut commands, &mut meshes, &mut materials);
    world_entities.extend(scenery_entities);
    // Streetlights ao longo da pista + buildings INSIDE the track loop interior (cidade).
    let track_dec = universe::scenery::spawn_track_decorations(
        &mut commands, &mut meshes, &mut materials, &world_res.0
    );
    world_entities.extend(track_dec);

    // City mode: generate navigation state and spawn traffic lights
    if config.city_mode {
        let city_nav = universe::city::CityNavState::new_city(320, 3.0, config.city_blocks as usize);
        let traffic_lights = universe::scenery::spawn_traffic_lights(&mut commands, &mut meshes, &mut materials, &city_nav);
        world_entities.extend(traffic_lights);
        world_res.1 = Some(city_nav);
    }

    race.world_entities = world_entities;

    // Spawn batched — não bloqueia frame inicial. drain_spawn_queue consome progressivamente.
    race.pending_spawn = Some((0, config.swarm_count));
}
