use bevy::prelude::*;
use crate::world::VoxelWorldResource;
use crate::environment::EnvironmentalState;

#[derive(Component)]
pub struct TireComponent {
    pub b: f32, // Stiffness factor (Pacejka B)
    pub c: f32, // Shape factor (Pacejka C)
    pub d: f32, // Peak value (Pacejka D)
    pub cornering_stiffness: f32, // C_alpha (N/rad) para Força Lateral
}

impl Default for TireComponent {
    fn default() -> Self {
        Self { b: 10.0, c: 1.9, d: 50000.0, cornering_stiffness: 55000.0 }
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
    pub current_gear: usize,
    pub shift_timer: f32,
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
            powertrain_and_tire_system,
            integration_system,
        ).chain());
    }
}

fn reset_forces(mut query: Query<&mut PhysicsBody>) {
    for mut body in query.iter_mut() {
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
        let height = transform.translation.y;
        let ground_y = 0.5;
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

/// Sistema unificado de Powertrain + Pneus + Aerodinâmica.
/// Corrige o bug anterior onde arrasto era aplicado DUAS vezes.
fn powertrain_and_tire_system(
    mut query: Query<(&mut EngineComponent, &mut PhysicsBody, &Transform, &VehicleChassis)>,
    world_res: Res<VoxelWorldResource>,
    env: Res<EnvironmentalState>,
    time: Res<Time>,
) {
    let dt = time.delta_secs().max(0.001);
    for (mut engine, mut body, transform, chassis) in query.iter_mut() {
        let speed = body.velocity.length();
        let speed_kmh = speed * 3.6;
        
        // --- TRANSMISSÃO AUTOMÁTICA COM DELAY MECÂNICO ---
        let gears = [15.0, 10.0, 7.5, 5.5, 4.2, 3.5];
        let shift_points = [60.0, 110.0, 160.0, 220.0, 280.0];
        
        let mut target_gear = engine.current_gear;
        for (i, &sp) in shift_points.iter().enumerate() {
            if speed_kmh < sp { target_gear = i; break; }
            if i == shift_points.len() - 1 { target_gear = 5; }
        }
        
        if target_gear != engine.current_gear && engine.shift_timer <= 0.0 {
            engine.current_gear = target_gear;
            engine.shift_timer = 0.3;
        }
        
        if engine.shift_timer > 0.0 {
            engine.shift_timer -= dt;
        }

        let gear_ratio = gears[engine.current_gear];
                         
        // RPM Dinâmico
        engine.current_rpm = (speed * gear_ratio * 12.0).clamp(800.0, engine.max_rpm);
        let rpm_ratio = engine.current_rpm / engine.max_rpm;
        
        // Curva de Torque Realística
        let torque_multiplier = 1.0 - (rpm_ratio - 0.6).powi(2) * 2.0;
        
        // Corte de Potência (Embreagem)
        let clutch_engagement = if engine.shift_timer > 0.0 { 0.0 } else { 1.0 };
        let drive_torque = engine.max_torque * torque_multiplier.clamp(0.6, 1.0) * engine.throttle * gear_ratio * clutch_engagement;  
        
        let forward_dir = transform.right().as_vec3();
        let lateral_dir = transform.back().as_vec3();
        
        // --- VENTO AMBIENTAL ---
        body.forces += env.wind * chassis.mass * 0.5;
        
        // --- AERODINÂMICA UNIFICADA (Aplicada UMA ÚNICA VEZ) ---
        // Arrasto: F_d = 0.5 * ρ * v² * Cd * A  (ρ=1.225, A≈2.2m²)
        let frontal_area = 2.2;
        let aero_drag_mag = 0.5 * 1.225 * speed * speed * chassis.drag_coefficient * frontal_area;
        let drag_dir = -body.velocity.normalize_or_zero();
        body.forces += drag_dir * aero_drag_mag;
        
        // Downforce: empurra o carro contra o chão (aumenta grip em alta velocidade)
        // Cl ≈ Cd * 2.5 para carros com aerofólio
        let cl = chassis.drag_coefficient * 2.5;
        let downforce = 0.5 * 1.225 * speed * speed * cl * frontal_area;
        let normal_load = chassis.mass * 9.81 + downforce;
        
        // --- GRIP: Terreno × Ambiente ---
        let voxel = world_res.0.get_voxel_at(transform.translation.x, transform.translation.z);
        let terrain_grip = if voxel == crate::world::VoxelType::Grass { 0.4 } else { 1.0 };
        let env_grip = env.combined_grip_factor();
        let total_grip = terrain_grip * env_grip;
        
        // === COEFICIENTE DE ATRITO PARA PNEUS DE CORRIDA ===
        // Pneus slick em asfalto seco: μ ≈ 1.2-1.5
        // (O μ=0.85 anterior era para pneus normais de rua — MUITO baixo para corrida)
        let mu_base = 1.3; // Pneu racing slick em asfalto seco
        let mu = mu_base * total_grip;
        
        // --- FRICTION CIRCLE (CÍRCULO DE KAMM) ---
        let lateral_vel = body.velocity.dot(lateral_dir);
        let forward_vel = body.velocity.dot(forward_dir).max(0.1);
        let slip_angle = lateral_vel.atan2(forward_vel);
        
        // Slip Ratio
        let wheel_speed = if drive_torque > 0.0 { forward_vel + drive_torque * 0.001 } else { forward_vel };
        let slip_ratio = if forward_vel > 0.5 {
            (wheel_speed - forward_vel) / forward_vel.max(0.5)
        } else {
            engine.throttle * 0.3
        };
        
        // Pacejka Longitudinal
        let f_long_raw = (normal_load * mu) * (1.65 * (10.0 * slip_ratio).atan()).sin();
        
        // Pacejka Lateral
        let f_lat_raw = (normal_load * mu) * (1.9 * (12.0 * slip_angle).atan()).sin();
        
        // === CÍRCULO DE KAMM: sqrt(F_long² + F_lat²) <= μ * N ===
        let max_total_force = mu * normal_load;
        let combined_raw = (f_long_raw * f_long_raw + f_lat_raw * f_lat_raw).sqrt();
        
        let (f_long_final, f_lat_final) = if combined_raw > max_total_force && combined_raw > 0.001 {
            let scale = max_total_force / combined_raw;
            (f_long_raw * scale, f_lat_raw * scale)
        } else {
            (f_long_raw, f_lat_raw)
        };
        
        // === TRAÇÃO: A força do motor é roteada PELO pneu, não diretamente ===
        // O motor gera torque → cria slip ratio → pneu converte em força via Pacejka.
        // A tração real é o MÍNIMO entre o que o motor pede e o que o pneu aguenta.
        let engine_force = drive_torque;
        let tire_traction_limit = f_long_final.abs();
        let actual_traction = engine_force.min(tire_traction_limit);
        body.forces += forward_dir * actual_traction;
        
        // === FORÇA LATERAL (Grip de Curva) ===
        let max_needed_force = (lateral_vel.abs() * chassis.mass) / dt;
        let applied_lat = f_lat_final.abs().min(max_needed_force) * f_lat_final.signum();
        
        // Aquaplanagem: perde quase toda direção
        let aquaplaning_factor = if env.is_aquaplaning(speed) { 0.1 } else { 1.0 };
        body.forces += -lateral_dir * applied_lat * aquaplaning_factor;
        
        // === FRENAGEM ===
        let speed = body.velocity.length();
        if engine.brake > 0.0 && speed > 0.01 {
            let max_brake_force = (speed * chassis.mass) / dt;
            let brake_grip_limit = mu * normal_load; // Freio também limitado pelo grip
            let applied_brake = (engine.brake * chassis.mass * 15.0).min(max_brake_force).min(brake_grip_limit);
            let brake_dir = -body.velocity.normalize_or_zero();
            body.forces += brake_dir * applied_brake;
        }
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
        
        let angular_accel = body.torques / (chassis.mass * 2.0);
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
