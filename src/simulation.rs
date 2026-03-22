use bevy::prelude::*;
use crate::engine::SovereignEngine;
use crate::world::{VoxelType, VoxelWorldResource};

#[derive(Component)]
pub struct Racer {
    pub name: String,
    pub engine: SovereignEngine,
    pub is_dead: bool,
    pub distance: f32,
    pub amplitude: f32, 
    pub phase: f32,     
    pub pain_memories: Vec<PainSynapse>,
    pub prev_pos: Vec3,
    pub fuel: f32,
    pub total_laps: usize,
    pub stagnation: f32,
    pub memory: [f32; 3], // [throttle, steer, brake]
}

pub struct PainSynapse {
    pub position: Vec3,
    pub pain_phase: f32, 
    pub intensity: f32,
}

#[derive(Component)]
pub struct Skidmark {
    pub timer: f32,
}

pub struct RacerPlugin;

impl Plugin for RacerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (update_racers, camera_system, skidmark_cleanup_system));
    }
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

fn update_racers(
    mut commands: Commands,
    mut query: Query<(
        &mut Racer, 
        &mut Transform, 
        &mut MeshMaterial3d<StandardMaterial>,
        Option<&mut crate::physics::EngineComponent>,
        Option<&mut crate::physics::PhysicsBody>
    )>,
    world_res: Res<VoxelWorldResource>,
    dev: Res<crate::evolution::DevSettings>,
    time: Res<Time>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut race: ResMut<crate::evolution::RaceSettings>,
    mut skidmark_cache: Local<Option<(Handle<Mesh>, Handle<StandardMaterial>)>>,
) {
    let dt = time.delta_secs().min(0.02);
    let world = &world_res.0;

    for (mut racer, mut transform, mat_handle, mut opt_engine, mut opt_body) in query.iter_mut() {
        if racer.is_dead { 
            if let Some(engine_comp) = opt_engine.as_mut() { engine_comp.throttle = 0.0; }
            if let Some(body) = opt_body.as_mut() { body.velocity *= 0.8; body.angular_velocity *= 0.8; }
            continue; 
        }

        let pos = transform.translation;
        let flow = world.get_flow_at(pos.x, pos.z);
        let track_phase = flow.y.atan2(flow.x);

        // --- 1. VISÃO DINÂMICA (HUMAN-LIKE) ---
        let vision_range = 25.0 + racer.amplitude * 1.8;
        let mut sensory_wave = vec![0i8; 128];
        let num_rays = 21;
        let mut max_harmony_dist = 0.0;
        let mut harmony_angle = 0.0;

        for i in 0..num_rays {
            let angle_offset = (i as f32 / (num_rays - 1) as f32 - 0.5) * 2.8; 
            let d = world.raycast(pos.x, pos.z, racer.phase + angle_offset, vision_range);
            let energy = (vision_range / d.max(1.0)).powi(2);
            sensory_wave[i] = (energy * 0.2).clamp(0.0, 127.0) as i8;
            if d > max_harmony_dist { max_harmony_dist = d; harmony_angle = angle_offset; }
        }

        // --- 2. CANAIS DE INSTINTO E TELEMETRIA ---
        sensory_wave[42] = (harmony_angle * 100.0).clamp(-127.0, 127.0) as i8;
        let alignment = (racer.phase - track_phase).cos();
        sensory_wave[43] = (alignment * 127.0) as i8;

        let speed = if let Some(ref body) = opt_body { body.velocity.length() } else { 0.0 };
        sensory_wave[45] = (speed * 2.0).clamp(0.0, 127.0) as i8; 
        
        let mut is_drifting = false;
        if let Some(ref body) = opt_body {
            let lateral_dir = transform.back().as_vec3();
            let lateral_vel = body.velocity.dot(lateral_dir);
            sensory_wave[46] = (lateral_vel * 10.0).clamp(-127.0, 127.0) as i8;
            if lateral_vel.abs() > 4.0 && speed > 5.0 && pos.y < 1.0 { is_drifting = true; }
        }

        if is_drifting {
            let (mark_mesh, mark_mat) = skidmark_cache.get_or_insert_with(|| {
                (
                    meshes.add(Rectangle::new(1.0, 1.0)),
                    materials.add(StandardMaterial {
                        base_color: Color::srgba(0.0, 0.0, 0.0, 0.7),
                        alpha_mode: AlphaMode::Blend,
                        unlit: true,
                        ..default()
                    })
                )
            });

            let mark_length = (speed * 0.06).max(0.5); // Stretches enough to cover the 1/20s gap
            let mut mark_t = *transform;
            mark_t.translation = pos - mark_t.right().as_vec3() * 0.9;
            mark_t.translation.y = 0.51;
            mark_t.rotation *= Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
            mark_t.scale = Vec3::new(mark_length, 0.4, 1.0);
            
            let e = commands.spawn((Mesh3d(mark_mesh.clone()), MeshMaterial3d(mark_mat.clone()), mark_t, Skidmark { timer: 5.0 })).id();
            race.world_entities.push(e);
        }

        // Memória temporal
        sensory_wave[120] = (racer.memory[0] * 127.0) as i8;
        sensory_wave[121] = (racer.memory[1] * 127.0) as i8;
        sensory_wave[122] = (racer.memory[2] * 127.0) as i8;

        // --- 3. EXECUÇÃO SOBERANA ---
        racer.engine.execute_cycle(&sensory_wave, track_phase, dt);
        let intent = racer.engine.instinct_layer.forward(&sensory_wave);
        
        let pedal_intent = (intent[0] + 0.5).clamp(0.0, 1.0);
        let steer_intent = intent[1].clamp(-1.0, 1.0);
        let brake_intent = intent[2].clamp(0.0, 1.0);

        racer.memory = [pedal_intent, steer_intent, brake_intent];

        // --- 4. ENGINE E FÍSICA (NOVO) ---
        let flow_pull = (track_phase - racer.phase).sin() * dev.flow_influence;
        // Physical steering (Yaw rotation speed)
        let steering_speed = steer_intent * 2.5 + flow_pull; // Diminuído de 6.0 para 2.5 para evitar "zerinhos" (spin-outs) instantâneos

        if let (Some(mut engine_comp), Some(mut body)) = (opt_engine, opt_body) {
            engine_comp.throttle = pedal_intent;
            body.angular_velocity.y = steering_speed;
            
            // Re-read phase and amplitude from physical translation/velocity
            racer.amplitude = body.velocity.length();
            racer.phase = transform.rotation.to_euler(EulerRot::YXZ).0 * -1.0; 
            
            racer.distance += racer.amplitude * dt;
        } else {
            // Fallback kinematic
            let target_amp = pedal_intent * dev.wave_amplitude;
            racer.amplitude = racer.amplitude + (target_amp - racer.amplitude) * 2.0 * dt;
            racer.phase = racer.phase + steering_speed * dt;
            let wave_vector = Vec3::new(racer.phase.cos(), 0.0, racer.phase.sin()) * racer.amplitude;
            transform.translation += wave_vector * dt;
            transform.rotation = Quat::from_rotation_y(-racer.phase);
            racer.distance += (wave_vector * dt).length();
        }

        racer.prev_pos = transform.translation;

        // --- 5. DOR E TRAUMA ---
        let voxel = world.get_voxel_at(transform.translation.x, transform.translation.z);
        if voxel == VoxelType::Wall || voxel == VoxelType::Grass || (racer.engine.current_time > 2.0 && alignment < 0.0) {
            racer.is_dead = true;
            if let Some(mat) = materials.get_mut(mat_handle.0.id()) { mat.base_color = Color::BLACK; }
        }
        
        if racer.amplitude < 3.0 && racer.engine.current_time > 8.0 { 
            racer.stagnation += dt;
            if racer.stagnation > 1.0 { racer.is_dead = true; }
        } else { racer.stagnation = 0.0; }

        racer.fuel -= (0.002 + pedal_intent.powi(2) * 0.004) * dt;
        if racer.fuel <= 0.0 { racer.is_dead = true; }
        
        // Contagem de Voltas
        let lap_line_x = world.start_pos.x;
        if racer.prev_pos.x < lap_line_x && transform.translation.x >= lap_line_x && racer.distance > 400.0 {
            racer.total_laps += 1; racer.distance = 0.0;
            if racer.total_laps % 3 == 0 { racer.fuel = 1.0; }
        }
    }
}

fn camera_system(
    mut query: Query<&mut Transform, With<Camera3d>>,
    racer_query: Query<(Entity, &Racer, &Transform), Without<Camera3d>>,
    mut race: ResMut<crate::evolution::RaceSettings>,
    mut scroll_evr: EventReader<bevy::input::mouse::MouseWheel>,
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

    if let Ok(mut cam_t) = query.get_single_mut() {
        if race.is_free_cam {
            let mut move_dir = Vec3::ZERO;
            let forward = cam_t.forward().as_vec3();
            let right = cam_t.right().as_vec3();
            if keys.pressed(KeyCode::KeyW) { move_dir += forward; }
            if keys.pressed(KeyCode::KeyS) { move_dir -= forward; }
            if keys.pressed(KeyCode::KeyA) { move_dir -= right; }
            if keys.pressed(KeyCode::KeyD) { move_dir += right; }
            
            if move_dir.length_squared() > 0.0 {
                let speed = if keys.pressed(KeyCode::ShiftLeft) { 300.0 } else { 100.0 };
                cam_t.translation += move_dir.normalize() * speed * dt;
            }
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
