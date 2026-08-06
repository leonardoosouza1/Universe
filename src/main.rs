use bevy::prelude::*;
use universe::world::{VoxelWorld, VoxelWorldResource};
use universe::simulation::RacerPlugin;
use universe::ui::{UiPlugin, LeaderboardUI};
use universe::evolution::{EvolutionPlugin, RaceSettings, DevSettings};
use universe::environment::EnvironmentalState;
use universe::config::UniverseConfig;
use universe::ui_menu::MenuPlugin;
use universe::lab::LabParams;
use universe::{AppState, SimulationEntity, HeadlessMode};

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
    let lab_params_path: Option<String> = args.iter().position(|a| a == "--lab-params")
        .and_then(|i| args.get(i+1))
        .map(|s| s.to_string());
    let experiment_name: Option<String> = args.iter().position(|a| a == "--experiment-name")
        .and_then(|i| args.get(i+1))
        .map(|s| s.to_string());
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
    // Persiste se CLI passou — próximo run UI herda automatic.
    if no_holes { config.no_holes = true; }
    if easy_track { config.easy_track = true; }
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
    if headless && time_scale_override.is_none() { config.time_scale = config.time_scale.max(8.0); }

    // Load lab params if provided
    let mut lab_params = if let Some(path) = lab_params_path {
        LabParams::load(&path).unwrap_or_else(|e| {
            eprintln!("⚠️  Failed to load lab params from {}: {}. Using defaults.", path, e);
            LabParams::default()
        })
    } else {
        LabParams::default()
    };
    // Override experiment name if provided via CLI
    if let Some(name) = experiment_name {
        lab_params.experiment_name = name;
    }

    // Ablação de via (ver `neuro.rs`). Fixada uma vez, no arranque, e global de processo: o
    // `universe-lab` roda cada variante num processo separado, então não há o que confundir.
    // Impresso SEMPRE, inclusive na linha de base — um experimento de ablação em que não se
    // vê qual via caiu é um experimento sem rótulo.
    let via = universe::neuro::Via::de_str(&lab_params.via_ablada);
    universe::neuro::definir_ablacao(via);
    if !lab_params.via_ablada.trim().is_empty() && via == universe::neuro::Via::Nenhuma {
        eprintln!("⚠️  via_ablada=\"{}\" não é uma via conhecida — rodando SEM ablação. \
                   Vias: cortex · reflexo · sentinela · sentinela_total · lenta",
                  lab_params.via_ablada);
    }
    println!("🧠 via ablada: {}", via.nome());

    // Modo de nascimento dos neurônios motores (ver `neuro::Nascimento`). Impresso sempre:
    // um experimento sobre independência do modelo em que não se vê como ele nasceu é um
    // experimento sem rótulo.
    let nasc = universe::neuro::Nascimento::de_str(&lab_params.nascimento);
    universe::neuro::definir_nascimento(nasc);
    println!("🌱 nascimento dos motores: {}", nasc.nome());

    // Parâmetros da visão (ver `visao.rs`). Zero = mantém o padrão, para que nenhum
    // experimento antigo mude de comportamento por omissão.
    let vp = universe::visao::Params {
        alcance_m: if lab_params.visao_alcance_m > 0.0 { lab_params.visao_alcance_m }
                   else { universe::visao::ALCANCE_PADRAO },
        fov_deg: if lab_params.visao_fov_deg > 0.0 { lab_params.visao_fov_deg }
                 else { universe::config::iara::vision::FOV_DEG },
    };
    universe::visao::definir(vp);
    // Fração da população na faixa de mutação destrutiva (ver `neuro.rs`). Medido: com o
    // padrão de 0,50 a MEDIANA da população é ZERO metros — metade não sai do lugar, e
    // `wall_mean = 0,00s` mostra que não é batida: é a mutação destruindo o genoma.
    let fexp = if lab_params.fracao_exploradora > 0.0 { lab_params.fracao_exploradora }
               else { universe::neuro::FRACAO_EXPLORADORA_PADRAO };
    universe::neuro::definir_fracao_exploradora(fexp);
    println!("🎲 fração exploradora (σ alto): {:.0}%", fexp * 100.0);
    println!("👁  visão: alcance {:.1} m · abertura {:.0}° · cega acima de {:.1} m/s",
             vp.alcance_m, vp.fov_deg, universe::visao::velocidade_cega(vp.alcance_m, 11.4));

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

    app.insert_resource(HeadlessMode { enabled: headless, gens_target });
    app.insert_resource(lab_params.clone());
    // UI auto-start quando --seed passou em modo não-headless.
    app.insert_resource(UiAutoStart(seed_override.is_some() && !headless));
    if headless {
        // HEADLESS DE VERDADE — sem janela, sem GPU, sem swap chain.
        //
        // Antes, `--headless` só passava `visible: false` e ainda criava janela real no X.
        // Duas consequências medidas:
        //
        //   1. As DUAS corridas do A/B de nicho morreram no mesmo panic —
        //      `Couldn't get swap chain texture ... swap chain has been lost`, em
        //      `bevy_render::view::window::prepare_windows` — aos 27 e 28 de 30 gerações.
        //      Corrida de dias é impossível assim, e são dias que a evolução exige: 1000
        //      gerações custam 2,4 dias, e as raposas de Belyaev viraram dóceis em 60.
        //   2. Prendia a simulação a um display X existente (`xvfb-run` não resolvia).
        //
        // Nada em headless usa GPU: não existe um único `.wgsl` no repositório nem pipeline
        // de compute, e a retina (`world::render_binocular_retina`) é raycast DDA em CPU
        // sobre voxels, paralelizado por rayon. A janela era peso morto que derrubava tudo.
        //
        // `backends: None` faz o wgpu não criar dispositivo; os tipos de asset (Mesh,
        // StandardMaterial) seguem registrados como DADOS, que é o único uso que o código
        // headless faz deles.
        app.add_plugins(
            DefaultPlugins
                .build()
                .disable::<bevy::winit::WinitPlugin>()
                .set(bevy::render::RenderPlugin {
                    render_creation: bevy::render::settings::RenderCreation::Automatic(
                        bevy::render::settings::WgpuSettings {
                            backends: None,
                            ..default()
                        },
                    ),
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: bevy::window::ExitCondition::DontExit,
                    ..default()
                }),
        );
        // Sem Winit não há laço de eventos: o ScheduleRunner passa a bombear os frames.
        app.add_plugins(bevy::app::ScheduleRunnerPlugin::run_loop(
            std::time::Duration::ZERO,
        ));
    } else {
        app.add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "IARA v2101 - Sovereign Dev Engine".into(),
                ..default()
            }),
            ..default()
        }));
        app.insert_resource(ClearColor(Color::srgb(0.42, 0.65, 0.88)));
    }
    app.init_state::<AppState>()
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
        frac_completou_volta: 0.0,
        estreitamento: 0.0,
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
        telemetry: {
            let lab_copy = lab_params.clone();
            universe::evolution::telemetry::TelemetryWriter::new(
                universe::evolution::telemetry::gen_run_id(),
                &lab_copy
            )
        },
        // Replay writer (novo formato 128 bytes/frame) — iniciado no começo de cada
        // geração pelo motor.rs quando lab.record_replay = true.
        replay_writer: None,
    })
    .insert_resource(DevSettings::default())
    .insert_resource(VoxelWorldResource(VoxelWorld::new(initial_seed, initial_width)))
    .insert_resource(EnvironmentalState::default());

    if headless {
        app.add_plugins((RacerPlugin, EvolutionPlugin, universe::physics::PhysicsPlugin))
            .add_systems(Startup, (apply_time_scale, headless_autostart))
            .add_systems(OnEnter(AppState::Simulation), (setup_scene_headless, apply_time_scale))
            .add_systems(OnExit(AppState::Simulation), cleanup_simulation)
            .add_systems(Update, headless_exit_check);
    } else {
        app.add_plugins((RacerPlugin, UiPlugin, EvolutionPlugin, universe::physics::PhysicsPlugin, MenuPlugin))
            .add_plugins(bevy_atmosphere::plugin::AtmospherePlugin)
            .add_systems(Startup, (setup_main_camera, apply_time_scale, headless_autostart))
            .add_systems(OnEnter(AppState::Simulation), (setup_scene, setup_hud_container, apply_time_scale))
            .add_systems(OnExit(AppState::Simulation), cleanup_simulation)
            .add_systems(Update, headless_exit_check);
    }

    app.run();
}

fn headless_autostart(
    hl: Res<HeadlessMode>,
    ui_auto: Res<UiAutoStart>,
    mut next_state: ResMut<NextState<AppState>>,
) {
    if ui_auto.0 {
        println!("🎮 UI AUTO-START | --seed passou em modo gráfico, entrando direto na simulação");
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

fn setup_scene_headless(
    mut race: ResMut<RaceSettings>,
    mut world_res: ResMut<VoxelWorldResource>,
    config: Res<UniverseConfig>,
) {
    let current_track = config.current_track();
    let world = VoxelWorld::new(current_track.seed, current_track.track_width + 4.0);
    world_res.0 = world;
    race.seed = current_track.seed;
    race.generation = 1;
    race.world_entities = Vec::new();
    race.pending_spawn = Some((0, config.swarm_count));
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
    race.world_entities = world_entities;

    // Spawn batched — não bloqueia frame inicial. drain_spawn_queue consome progressivamente.
    race.pending_spawn = Some((0, config.swarm_count));
}
