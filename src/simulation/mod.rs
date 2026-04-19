use bevy::prelude::*;

pub mod racer;
pub mod motor;

pub use racer::{Racer, PainSynapse, Skidmark};

pub struct RacerPlugin;

impl Plugin for RacerPlugin {
    fn build(&self, app: &mut App) {
        // Motor → FixedUpdate antes da física: brain e physics sincronizados.
        // Time_scale agora escala Virtual → Bevy roda N substeps de FixedUpdate por frame
        // real. Brain tick N× POR frame (antes só 1×), sem tunneling em time_scale alto.
        // Tier 5.4: tick_npcs_system roda ANTES de update_racers pra racers leem posições
        // frescas dos obstáculos móveis.
        app.add_systems(FixedUpdate, (
                tick_npcs_system.before(motor::update_racers),
                motor::update_racers.before(crate::physics::PhysicsSet),
            ))
           .add_systems(Update, (camera_system, skidmark_cleanup_system));
    }
}

/// Tier 5.4 — Avança todos os NPCs obstáculos um passo de dt por tick fixed.
/// No-op se `env.npcs` vazio (gate curricular) ou mundo sem waypoints.
fn tick_npcs_system(
    mut env: ResMut<crate::environment::EnvironmentalState>,
    world: Res<crate::world::VoxelWorldResource>,
    time: Res<Time>,
) {
    if env.npcs.is_empty() || world.0.waypoints_world.is_empty() { return; }
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
        if skid.timer <= 0.0 { commands.entity(e).despawn(); }
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

            if keys.pressed(KeyCode::KeyW) { move_dir += forward; }
            if keys.pressed(KeyCode::KeyS) { move_dir -= forward; }
            if keys.pressed(KeyCode::KeyA) { move_dir -= right; }
            if keys.pressed(KeyCode::KeyD) { move_dir += right; }
            if keys.pressed(KeyCode::KeyE) { move_dir += up; }
            if keys.pressed(KeyCode::KeyQ) { move_dir -= up; }

            if move_dir.length_squared() > 0.0 {
                let speed = if keys.pressed(KeyCode::ShiftLeft) { 300.0 } else { 100.0 };
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
            if keys.pressed(KeyCode::ArrowLeft) { cam_t.rotate_axis(Dir3::Y, 2.0 * dt); }
            if keys.pressed(KeyCode::ArrowRight) { cam_t.rotate_axis(Dir3::Y, -2.0 * dt); }
            if keys.pressed(KeyCode::ArrowUp) { cam_t.rotate_local_x(2.0 * dt); }
            if keys.pressed(KeyCode::ArrowDown) { cam_t.rotate_local_x(-2.0 * dt); }
        } else {
            let mut target = race.selected_racer;
            if target.is_none() {
                if let Some((e, _, _)) = racer_query.iter().filter(|(_, r, _)| !r.is_dead).max_by(|a, b| a.1.distance.partial_cmp(&b.1.distance).unwrap()) {
                    target = Some(e);
                }
            }
            if let Some(e) = target {
                if let Ok((_, _, t)) = racer_query.get(e) {
                    let offset = Vec3::new(-0.6, 0.8, -0.6).normalize() * race.camera_distance;
                    let goal = t.translation + offset;
                    cam_t.translation = cam_t.translation.lerp(goal, 0.05);
                    cam_t.look_at(t.translation, Vec3::Y);
                }
            }
        }
    }
}
