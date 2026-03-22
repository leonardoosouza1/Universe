use bevy::prelude::*;
use crate::world::VoxelWorldResource;

#[derive(Component)]
pub struct TireComponent {
    pub b: f32, // Stiffness factor
    pub c: f32, // Shape factor
    pub d: f32, // Peak value
}

impl Default for TireComponent {
    fn default() -> Self {
        Self { b: 10.0, c: 1.9, d: 50000.0 }
    }
}

#[derive(Component)]
pub struct SuspensionComponent {
    pub stiffness: f32,
    pub damping: f32,
    pub rest_length: f32,
    pub max_travel: f32,
    pub raycast_dir: Vec3,
    pub prev_compression: f32,
}

#[derive(Component)]
pub struct EngineComponent {
    pub max_torque: f32,
    pub current_rpm: f32,
    pub max_rpm: f32,
    pub throttle: f32,
    pub brake: f32,
}

#[derive(Component, PartialEq, Clone, Copy)]
pub enum DrivetrainType {
    FWD,
    RWD,
    AWD,
}

#[derive(Component)]
pub struct VehicleChassis {
    pub drivetrain: DrivetrainType,
    pub mass: f32,
    pub com_offset: Vec3,
    pub drag_coefficient: f32,
    pub wheel_base: f32,
}

#[derive(Component, Default)]
pub struct PhysicsBody {
    pub velocity: Vec3,
    pub angular_velocity: Vec3,
    pub forces: Vec3,
    pub torques: Vec3,
}

pub struct PhysicsPlugin;

impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (
            reset_forces,
            suspension_system,
            powertrain_system,
            aerodynamics_system,
            integration_system,
        ).chain());
    }
}

fn reset_forces(mut query: Query<&mut PhysicsBody>) {
    for mut body in query.iter_mut() {
        // Apply gravity
        body.forces = Vec3::new(0.0, -9.81 * 1000.0, 0.0);
        body.torques = Vec3::ZERO;
    }
}

fn suspension_system(
    mut query: Query<(&mut SuspensionComponent, &mut PhysicsBody, &Transform)>,
    time: Res<Time>,
) {
    let dt = time.delta_secs();
    
    for (mut suspension, mut body, transform) in query.iter_mut() {
        let origin = transform.translation;
        let ground_y = 0.5; // Simulate a rough raycast to the floor
        
        let height = origin.y;
        let compression = suspension.rest_length - (height - ground_y);
        
        if compression > 0.0 && compression < suspension.max_travel {
            let spring_force = suspension.stiffness * compression;
            let compression_vel = (compression - suspension.prev_compression) / dt.max(0.001);
            let damper_force = suspension.damping * compression_vel;
            
            let total_force = (spring_force + damper_force).max(0.0);
            body.forces += Vec3::Y * total_force;
        }
        
        suspension.prev_compression = compression;
    }
}

fn powertrain_system(
    mut query: Query<(&mut EngineComponent, &mut PhysicsBody, &Transform, &VehicleChassis)>,
    world_res: Res<VoxelWorldResource>,
    time: Res<Time>,
) {
    let dt = time.delta_secs().max(0.001);
    for (mut engine, mut body, transform, chassis) in query.iter_mut() {
        let speed = body.velocity.length();
        let speed_kmh = speed * 3.6;
        
        // --- VIRTUAL GEARBOX (Transmissão Mecânica) ---
        let gears = [15.0, 10.0, 7.5, 5.5, 4.2, 3.5];
        let mut current_gear = 5;
        let shift_points = [60.0, 110.0, 160.0, 220.0, 280.0];
        for (i, &sp) in shift_points.iter().enumerate() {
            if speed_kmh < sp { current_gear = i; break; }
        }
        let gear_ratio = gears[current_gear];
                         
        // RPM Dinâmico
        engine.current_rpm = (speed * gear_ratio * 12.0).clamp(800.0, engine.max_rpm);
        let rpm_ratio = engine.current_rpm / engine.max_rpm;
        
        // Curva de Torque Realística
        let torque_multiplier = 1.0 - (rpm_ratio - 0.6).powi(2) * 2.0;
        let torque = engine.max_torque * torque_multiplier.clamp(0.6, 1.0) * engine.throttle * gear_ratio;  
        
        let forward_dir = transform.right().as_vec3();
        let tractive_force = forward_dir * torque;
        body.forces += tractive_force;
        
        // --- AERODINÂMICA (ARRASTO E DOWNFORCE) ---
        // Vento frontal como parede: F_d = v^2 * Cd * 1.5
        let drag_force = speed * speed * chassis.drag_coefficient * 1.5; 
        let aero_drag = -body.velocity.normalize_or_zero() * drag_force;
        body.forces += aero_drag;
        
        // Carga Vertical (Peso Estático + Downforce)
        let downforce = speed * speed * chassis.drag_coefficient * 2.0; // Aerofólio empurrando pra baixo
        let normal_load = chassis.mass * 9.81 + downforce; // O peso dinâmico do carro
        
        let lateral_dir = transform.back().as_vec3();
        let lateral_vel = body.velocity.dot(lateral_dir);
        let slip_angle = lateral_vel.atan2(body.velocity.dot(forward_dir).max(0.1));
        
        let voxel = world_res.0.get_voxel_at(transform.translation.x, transform.translation.z);
        let grip_multiplier = if voxel == crate::world::VoxelType::Grass { 0.4 } else { 1.0 };
        
        let p_b = 12.0;   // Rigidez da borracha
        let p_c = 1.9;
        // P_d = Max Grip. Agora dependente do peso efetivo (Pneus grudam mais em alta velocidade pela gravidade+downforce)
        let p_d = normal_load * 1.5 * grip_multiplier; 
        
        let lateral_force_mag = p_d * (p_c * (p_b * slip_angle).atan()).sin();
        
        // Anti-Jitter
        let max_needed_force = (lateral_vel.abs() * chassis.mass) / dt;
        let applied_force_mag = lateral_force_mag.abs().min(max_needed_force) * lateral_force_mag.signum();
        
        let lateral_force = -lateral_dir * applied_force_mag;
        body.forces += lateral_force;
        
        // Braking
        let speed = body.velocity.length();
        if engine.brake > 0.0 && speed > 0.01 {
            let max_brake_force = (speed * chassis.mass) / dt;
            let applied_brake = (engine.brake * chassis.mass * 15.0).min(max_brake_force);
            let brake_force = -body.velocity.normalize_or_zero() * applied_brake;
            body.forces += brake_force;
        }
    }
}

fn aerodynamics_system(
    mut query: Query<(&VehicleChassis, &mut PhysicsBody)>,
) {
    for (chassis, mut body) in query.iter_mut() {
        let speed_sq = body.velocity.length_squared();
        let dir = body.velocity.normalize_or_zero();
        let drag = -dir * chassis.drag_coefficient * speed_sq;
        let downforce = Vec3::NEG_Y * chassis.drag_coefficient * speed_sq * 1.5;
        body.forces += drag + downforce;
    }
}

fn integration_system(
    mut query: Query<(&mut PhysicsBody, &VehicleChassis, &mut Transform)>,
    time: Res<Time>,
) {
    let dt = time.delta_secs();
    if dt == 0.0 { return; }
    
    for (mut body, chassis, mut transform) in query.iter_mut() {
        let acceleration = body.forces / chassis.mass;
        body.velocity += acceleration * dt;
        
        let angular_accel = body.torques / (chassis.mass * 2.0); // Simplified inertia
        body.angular_velocity += angular_accel * dt;
        
        transform.translation += body.velocity * dt;
        
        // Ground constraint
        if transform.translation.y < 0.5 {
            transform.translation.y = 0.5;
            body.velocity.y = body.velocity.y.max(0.0);
        }
        
        // Yaw
        transform.rotate_y(body.angular_velocity.y * dt);
        body.angular_velocity *= 0.90;
    }
}
