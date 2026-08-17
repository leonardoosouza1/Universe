use bevy::prelude::*;

pub mod motor;
pub mod racer;
pub mod racing_line;

pub use racer::{PainSynapse, Racer, Skidmark};
pub use racing_line::RacingLine;

/// Label para ordering de sistemas — necessário porque update_racers tem muitos parâmetros
/// e o Bevy 0.16 não aceita referências diretas à função além do limite de tupla.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct UpdateRacersSet;

pub struct RacerPlugin;

impl Plugin for RacerPlugin {
    fn build(&self, app: &mut App) {
        // Contexto de GPU da retina — vive entre ticks (recriar por tick anularia o ganho).
        app.init_resource::<crate::gpu::RetinaGpuRes>();
        // Contexto de GPU da percepção — compartilha device/queue da retina.
        // Criado sob demanda quando lab.gpu_core=true; Default é None (zero overhead se desligado).
        app.init_resource::<crate::gpu::perception::PerceptionGpuRes>();
        // Contexto de GPU do cérebro — forward pass neural na GPU.
        // Criado sob demanda quando lab.gpu_core=true.
        app.init_resource::<crate::gpu::brain::BrainGpuRes>();
        // Contexto de GPU da física — 3 kernels encadeados (A→B→C) na GPU.
        // Criado sob demanda quando lab.gpu_core=true.
        app.init_resource::<crate::gpu::physics::PhysicsGpuRes>();
        // Motor → FixedUpdate antes da física: brain e physics sincronizados.
        // Time_scale agora escala Virtual → Bevy roda N substeps de FixedUpdate por frame
        // real. Brain tick N× POR frame (antes só 1×), sem tunneling em time_scale alto.
        // Tier 5.4: tick_npcs_system roda ANTES de update_racers via set label.
        // update_racers tem 17 parâmetros — o Bevy 0.16 não aceita before/in_set em
        // fn items além de certo limite. Solução: label em tick_npcs_system e ordering
        // sobre o set, garantindo NPC→racer sem tocar na assinatura da fn pesada.
        app.configure_sets(FixedUpdate, UpdateRacersSet);
        app.add_systems(
            FixedUpdate,
            (
                tick_npcs_system.before(crate::physics::PhysicsSet),
                motor::update_racers.before(crate::physics::PhysicsSet),
            ),
        )
        .add_systems(
            Update,
            (camera_system, skidmark_cleanup_system, toggle_gizmos_system),
        );
    }
}

/// KeyG toggles dev gizmos (track pillars, paths, flags) for clean racing-game look.
/// KeyH toggles vision cones (fovea raycast verde no carro selecionado).
fn toggle_gizmos_system(
    keys: Res<ButtonInput<KeyCode>>,
    mut dev: ResMut<crate::evolution::DevSettings>,
) {
    if keys.just_pressed(KeyCode::KeyG) {
        dev.show_gizmos = !dev.show_gizmos;
        println!(
            "🎮 Gizmos {}",
            if dev.show_gizmos {
                "ON"
            } else {
                "OFF (clean racing view)"
            }
        );
    }
    if keys.just_pressed(KeyCode::KeyH) {
        dev.show_vision = !dev.show_vision;
        println!(
            "👁  Vision cones {}",
            if dev.show_vision { "ON" } else { "OFF" }
        );
    }
}

/// Tier 5.4 — Avança todos os NPCs obstáculos um passo de dt por tick fixed.
/// No-op se `env.npcs` vazio (gate curricular) ou mundo sem waypoints.
fn tick_npcs_system(
    mut env: ResMut<crate::environment::EnvironmentalState>,
    world: Res<crate::world::VoxelWorldResource>,
    time: Res<Time>,
) {
    if env.npcs.is_empty() || world.0.waypoints_world.is_empty() {
        return;
    }
    let dt = time.delta_secs();
    let wps = world.0.waypoints_world.clone();
    env.tick_npcs(&wps, dt);
}

fn skidmark_cleanup_system(
    mut commands: Commands,
    mut query: Query<(Entity, &mut Skidmark)>,
    time: Res<Time>,
) {
    let dt = time.delta_secs();
    for (e, mut skid) in query.iter_mut() {
        skid.timer -= dt;
        if skid.timer <= 0.0 {
            commands.entity(e).despawn();
        }
    }
}

fn camera_system(
    mut query: Query<&mut Transform, With<Camera3d>>,
    racer_query: Query<(Entity, &Racer, &Transform), Without<Camera3d>>,
    mut race: ResMut<crate::evolution::RaceSettings>,
    mut scroll_evr: EventReader<bevy::input::mouse::MouseWheel>,
    mut motion_evr: EventReader<bevy::input::mouse::MouseMotion>,
    mouse_btn: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
) {
    let dt = time.delta_secs();

    // Zoom
    for ev in scroll_evr.read() {
        race.camera_distance = (race.camera_distance - ev.y * 15.0).clamp(10.0, 400.0);
    }

    // Toggle free cam
    if keys.just_pressed(KeyCode::KeyC) {
        race.is_free_cam = !race.is_free_cam;
    }

    // Ler os movimentos do mouse para não acumular
    let mut mouse_delta = Vec2::ZERO;
    for ev in motion_evr.read() {
        if mouse_btn.pressed(MouseButton::Right) {
            mouse_delta += ev.delta;
        }
    }

    if let Ok(mut cam_t) = query.single_mut() {
        if race.is_free_cam {
            let mut move_dir = Vec3::ZERO;
            let forward = cam_t.forward().as_vec3();
            let right = cam_t.right().as_vec3();
            let up = Vec3::Y;

            if keys.pressed(KeyCode::KeyW) {
                move_dir += forward;
            }
            if keys.pressed(KeyCode::KeyS) {
                move_dir -= forward;
            }
            if keys.pressed(KeyCode::KeyA) {
                move_dir -= right;
            }
            if keys.pressed(KeyCode::KeyD) {
                move_dir += right;
            }
            if keys.pressed(KeyCode::KeyE) {
                move_dir += up;
            }
            if keys.pressed(KeyCode::KeyQ) {
                move_dir -= up;
            }

            if move_dir.length_squared() > 0.0 {
                let speed = if keys.pressed(KeyCode::ShiftLeft) {
                    300.0
                } else {
                    100.0
                };
                cam_t.translation += move_dir.normalize() * speed * dt;
            }

            // Rotação com o Mouse (Segurando Botão Direito)
            if mouse_delta.length_squared() > 0.0 {
                let yaw = -mouse_delta.x * 0.003;
                let pitch = -mouse_delta.y * 0.003;

                // Rotaciona Yaw (global Y) - usamos rotação ao redor do Y do mundo para não bugar o horizonte
                cam_t.rotate_axis(Dir3::Y, yaw);
                // Rotaciona Pitch (local X)
                cam_t.rotate_local_x(pitch);
            }

            // Mantendo fallback para as setas do teclado
            if keys.pressed(KeyCode::ArrowLeft) {
                cam_t.rotate_axis(Dir3::Y, 2.0 * dt);
            }
            if keys.pressed(KeyCode::ArrowRight) {
                cam_t.rotate_axis(Dir3::Y, -2.0 * dt);
            }
            if keys.pressed(KeyCode::ArrowUp) {
                cam_t.rotate_local_x(2.0 * dt);
            }
            if keys.pressed(KeyCode::ArrowDown) {
                cam_t.rotate_local_x(-2.0 * dt);
            }
        } else {
            let mut target = race.selected_racer;
            if target.is_none() {
                if let Some((e, _, _)) = racer_query
                    .iter()
                    .filter(|(_, r, _)| !r.is_dead)
                    .max_by(|a, b| a.1.distance.partial_cmp(&b.1.distance).unwrap())
                {
                    target = Some(e);
                }
            }
            if let Some(e) = target {
                if let Ok((_, _, t)) = racer_query.get(e) {
                    // Cinematic chase: behind car along -forward, slightly above, looks at point ahead.
                    // camera_distance: 10..400. Mapped: <=80 = chase atrás, >80 = top-down legacy.
                    let chase = race.camera_distance <= 80.0;
                    let goal = if chase {
                        // forward = right() per existing convention (see motor.rs:376)
                        let fwd = t.right().as_vec3();
                        let chase_dist = race.camera_distance.max(8.0);
                        let height = chase_dist * 0.35 + 1.5;
                        t.translation - fwd * chase_dist + Vec3::Y * height
                    } else {
                        let offset = Vec3::new(-0.6, 0.8, -0.6).normalize() * race.camera_distance;
                        t.translation + offset
                    };
                    cam_t.translation = cam_t
                        .translation
                        .lerp(goal, if chase { 0.10 } else { 0.05 });
                    let look_target = if chase {
                        // Aponta 12m à frente do carro pra leve antecipação cinematográfica.
                        let fwd = t.right().as_vec3();
                        t.translation + fwd * 12.0 + Vec3::Y * 0.5
                    } else {
                        t.translation
                    };
                    cam_t.look_at(look_target, Vec3::Y);
                }
            }
        }
    }
}
