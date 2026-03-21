use bevy::prelude::*;
use crate::engine::SovereignEngine;
use crate::world::{VoxelType, VoxelWorldResource};

#[derive(Component)]
pub struct Racer {
    pub name: String,
    pub engine: SovereignEngine,
    pub is_dead: bool,
    pub distance: f32,
    pub velocity: f32,
    pub angle: f32,
    pub angular_velocity: f32,
    pub gear: usize,
    pub rpm: f32,
    pub stagnation: f32,
    pub prev_pos: Vec3,
    // --- SENTIDOS METAFÍSICOS (2026) ---
    pub slip_angle: f32,
    pub g_force: f32,
    pub anticipation: [f32; 5],
}

pub struct RacerPlugin;

impl Plugin for RacerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (update_racers, camera_system));
    }
}

fn update_racers(
    mut query: Query<(&mut Racer, &mut Transform, &mut MeshMaterial3d<StandardMaterial>)>,
    world_res: Res<VoxelWorldResource>,
    time: Res<Time>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let dt = time.delta_secs();
    let world = &world_res.0;

    for (mut racer, mut transform, mat_handle) in query.iter_mut() {
        if racer.is_dead { continue; }

        let pos = transform.translation;
        let flow = world.get_flow_at(pos.x, pos.z);
        let track_angle = flow.y.atan2(flow.x);

        // --- 1. SENSORIAMENTO HÁPTICO E ANTECIPAÇÃO ---
        let mut sensory = vec![0i8; 128];
        let ray_angles = [-1.2, -0.6, 0.0, 0.6, 1.2];
        
        for (i, &ra) in ray_angles.iter().enumerate() {
            let d = world.raycast(pos.x, pos.z, racer.angle + ra, 50.0);
            
            // Proatividade: Gradiente de distância (quão rápido o muro se aproxima)
            let delta_d = d - racer.anticipation[i];
            racer.anticipation[i] = d;
            
            sensory[i] = (d * 2.5) as i8;
            sensory[i + 10] = (delta_d * 10.0).clamp(-127.0, 127.0) as i8;
        }

        // --- 2. SENTIDOS HUMANOS (PROPRIOCEPÇÃO) ---
        // Slip Angle: O "tato" da traseira saindo
        let velocity_vec = (transform.translation - racer.prev_pos) / dt;
        let actual_heading = velocity_vec.z.atan2(velocity_vec.x);
        racer.slip_angle = (actual_heading - racer.angle).sin();
        sensory[20] = (racer.slip_angle * 127.0) as i8;

        // Força-G: Pressão lateral sentida pelo piloto
        racer.g_force = racer.angular_velocity * racer.velocity * 0.1;
        sensory[21] = (racer.g_force * 60.0).clamp(-127.0, 127.0) as i8;

        // Erro de Trajetória e Alinhamento
        let yaw_error = (track_angle - racer.angle).sin();
        sensory[5] = (yaw_error * 127.0) as i8;
        let alignment = (racer.angle - track_angle).cos();
        sensory[6] = (alignment * 100.0) as i8;

        // Telemetria básica
        sensory[8] = racer.gear as i8 * 20;
        sensory[9] = (racer.rpm * 127.0) as i8;

        // --- 3. EXECUÇÃO SOBERANA ---
        racer.engine.execute_cycle(&sensory, track_angle, dt as f32);
        let intent = racer.engine.instinct_layer.forward(&sensory);
        
        // --- 4. DINÂMICA DE TORQUE E ADAPTAÇÃO (Trail Braking) ---
        let pedal = intent[0].clamp(-1.0, 1.0); 
        let steer_input = intent[1].clamp(-1.0, 1.0);
        let shift_intent = intent[2].clamp(-1.0, 1.0);

        // Auto-adaptação: Troca de marchas neural
        if shift_intent > 0.5 && racer.rpm > 0.82 && racer.gear < 8 { racer.gear += 1; racer.rpm = 0.45; }
        if shift_intent < -0.5 && racer.gear > 1 { racer.gear -= 1; racer.rpm = 0.95; }

        let gear_ratio = 1.0 / (racer.gear as f32);
        
        // Lógica de "Tirar o pé": Se o pedal for levemente negativo (-0.2 a 0), 
        // o carro apenas desacelera por compressão, mantendo mais velocidade em curvas.
        let torque = if pedal > 0.0 { 
            (pedal + 0.9) * 22.0 * gear_ratio 
        } else if pedal > -0.2 {
            -2.0 // Apenas tira o pé (Engine Braking)
        } else { 
            pedal * 120.0 // Freio pesado
        };
        
        // Física calibrada para 2026
        racer.rpm = (racer.velocity / (racer.gear as f32 * 6.5)).clamp(0.0, 1.0);
        racer.angular_velocity = (racer.angular_velocity + steer_input * 12.0 * dt).clamp(-5.0, 5.0) * 0.91;
        racer.velocity = (racer.velocity + torque * dt).clamp(0.0, 55.0) * 0.988;
        racer.angle += racer.angular_velocity * dt;

        transform.translation += Vec3::new(racer.angle.cos(), 0.0, racer.angle.sin()) * racer.velocity * dt;
        transform.rotation = Quat::from_rotation_y(-racer.angle);
        
        racer.distance += pos.distance(racer.prev_pos);
        racer.prev_pos = transform.translation;

        // Morte por colisão ou contramão
        if world.get_voxel_at(pos.x, pos.z) == VoxelType::Wall || alignment < -0.4 {
            racer.is_dead = true;
            if let Some(mat) = materials.get_mut(mat_handle.0.id()) {
                mat.base_color = Color::BLACK; mat.emissive = LinearRgba::BLACK;
            }
        }
        
        // Estagnação: Se o piloto não progridir, ele morre (Metabolismo)
        if racer.velocity < 1.0 {
            racer.stagnation += dt;
            if racer.stagnation > 3.0 { racer.is_dead = true; }
        } else { racer.stagnation = 0.0; }
    }
}

fn camera_system(
    mut query: Query<&mut Transform, With<Camera3d>>,
    racer_query: Query<(Entity, &Racer, &Transform), Without<Camera3d>>,
    race_settings: Res<crate::evolution::RaceSettings>,
) {
    let mut target = race_settings.selected_racer;
    if target.is_none() {
        if let Some((e, _, _)) = racer_query.iter().filter(|(_, r, _)| !r.is_dead).max_by(|a, b| a.1.distance.partial_cmp(&b.1.distance).unwrap()) {
            target = Some(e);
        }
    }
    if let Some(e) = target {
        if let Ok((_, _, t)) = racer_query.get(e) {
            let mut cam_t = query.single_mut().unwrap();
            let goal = t.translation + Vec3::new(-45.0, 55.0, -45.0);
            cam_t.translation = cam_t.translation.lerp(goal, 0.1);
            cam_t.look_at(t.translation, Vec3::Y);
        }
    }
}

