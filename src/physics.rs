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
    pub manual_shift_req: i8, // -1 (down), 0 (none), 1 (up)
    pub is_stalled: bool,
    pub starter_timer: f32, // Tempo segurando o motor de arranque (precisa de 1.0s para ligar)
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
    pub cg_height: f32, // Altura do centro de gravidade (metros)
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

/// Sistema Hardcore de Powertrain + Pneus + Aerodinâmica + Transferência de Peso.
/// Baseado em abstrações matemáticas de simuladores profissionais (Assetto Corsa, rFactor).
fn powertrain_and_tire_system(
    mut query: Query<(&mut EngineComponent, &mut PhysicsBody, &Transform, &VehicleChassis)>,
    world_res: Res<VoxelWorldResource>,
    env: Res<EnvironmentalState>,
    time: Res<Time>,
) {
    let dt = time.delta_secs().max(0.001);
    for (mut engine, mut body, transform, chassis) in query.iter_mut() {
        let speed = body.velocity.length();
        let forward_dir = transform.right().as_vec3();
        let lateral_dir = transform.back().as_vec3();
        let forward_vel = body.velocity.dot(forward_dir); // Pode ser NEGATIVO (ré)
        let lateral_vel = body.velocity.dot(lateral_dir);
        
        // Gears: [0=Reverse, 1=Neutral, 2=1st, 3=2nd, 4=3rd, 5=4th, 6=5th]
        let gears = [-10.0, 0.0, 12.0, 8.0, 6.0, 4.5, 3.5];
        
        // ========================================================
        // === A) MOTOR DE ARRANQUE MANUAL (STRICT STALL) ========
        // ========================================================
        // Se o motor está afogado, a IA PRECISA:
        //   1. Estar em marcha 0 ou 1 (embreagem desacopla as rodas)
        //   2. Segurar o acelerador > 50% por 1.0 segundo
        // Se tentar dar partida em marcha alta com carro parado, o motor afoga de novo.
        if engine.is_stalled {
            // Partida permitida apenas em Ré (0), Neutro (1) ou 1ª (2)
            if engine.current_gear <= 2 && engine.throttle > 0.5 {
                engine.starter_timer += dt;
                if engine.starter_timer >= 1.0 {
                    engine.is_stalled = false;
                    engine.starter_timer = 0.0;
                    engine.current_rpm = 800.0;
                }
            } else {
                engine.starter_timer = 0.0;
            }
            // Motor parado não gera torque nem freio motor
        }
        
        // ========================================================
        // === TRANSMISSÃO MANUAL CONTROLADA PELA IA ==============
        // ========================================================
        if engine.shift_timer <= 0.0 {
            if engine.manual_shift_req == 1 && engine.current_gear < 5 {
                engine.current_gear += 1;
                engine.shift_timer = 0.3;
            } else if engine.manual_shift_req == -1 && engine.current_gear > 0 {
                let next_ratio = gears[engine.current_gear - 1];
                let expected_rpm = forward_vel.abs() * next_ratio * 30.0;
                if expected_rpm < engine.max_rpm + 200.0 {
                    engine.current_gear -= 1;
                    engine.shift_timer = 0.3;
                }
            }
        }
        engine.manual_shift_req = 0;
        
        if engine.shift_timer > 0.0 {
            engine.shift_timer -= dt;
        }

        let gear_ratio = gears[engine.current_gear];
        
        // ========================================================
        // === RPM DINÂMICO + CÁLCULO DE TORQUE ===================
        // ========================================================
        // RPM é calculado pela velocidade das rodas motrizes (forward_vel pode ser negativo!)
        let mut raw_rpm = forward_vel.abs() * gear_ratio * 30.0;
        
        let mut clutch_slip = false;
        // Neutro (1) ou Auto-embreagem em Ré (0) ou 1ª (2) com baixa velocidade
        if engine.current_gear == 1 || (engine.current_gear <= 2 && forward_vel.abs() < 5.0) {
            clutch_slip = true;
            raw_rpm = 800.0 + (engine.throttle * 2500.0);
        }

        if engine.shift_timer > 0.0 {
            raw_rpm = 800.0; // Embreagem pressionada durante troca
        } else if !clutch_slip && !engine.is_stalled && raw_rpm < 550.0 {
            engine.is_stalled = true; // AFOGOU! RPM caiu demais com roda acoplada
            engine.starter_timer = 0.0;
        }

        let drive_torque;
        if engine.is_stalled {
            engine.current_rpm = 0.0;
            drive_torque = 0.0;
        } else {
            engine.current_rpm = raw_rpm.clamp(800.0, engine.max_rpm + 250.0);
            let rpm_ratio = engine.current_rpm / engine.max_rpm;
            
            // Curva de Torque Realística (pico em ~60% do RPM máximo)
            let mut torque_multiplier = 1.0 - (rpm_ratio - 0.6).powi(2) * 2.0;
            if rpm_ratio > 1.0 {
                torque_multiplier = 0.0; // REV LIMITER: corta injeção
            } else if rpm_ratio < 0.2 {
                torque_multiplier *= rpm_ratio / 0.2; // Bogs down em RPM muito baixo
            }
            torque_multiplier = torque_multiplier.clamp(0.0, 1.0);
            
            let clutch_engagement = if engine.shift_timer > 0.0 || clutch_slip { 
                if clutch_slip { (forward_vel.abs() / 5.0).clamp(0.0, 1.0) } else { 0.0 }
            } else { 1.0 };
            
            drive_torque = engine.max_torque * torque_multiplier * engine.throttle * gear_ratio * clutch_engagement * 3.0;
        }
        
        // ========================================================
        // === VENTO AMBIENTAL ====================================
        // ========================================================
        body.forces += env.wind * chassis.mass * 0.5;
        
        // ========================================================
        // === AERODINÂMICA UNIFICADA =============================
        // ========================================================
        let frontal_area = 2.2;
        let aero_drag_mag = 0.5 * 1.225 * speed * speed * chassis.drag_coefficient * frontal_area;
        let drag_dir = -body.velocity.normalize_or_zero();
        body.forces += drag_dir * aero_drag_mag;
        
        // Downforce
        let cl = chassis.drag_coefficient * 2.5;
        let downforce = 0.5 * 1.225 * speed * speed * cl * frontal_area;
        let base_normal_load = chassis.mass * 9.81 + downforce;
        
        // ========================================================
        // === C) ROLLING RESISTANCE & ENGINE BRAKING ==============
        // ========================================================
        // Resistência de rolamento: F_rr = C_rr * N (C_rr ≈ 0.015 para pneus slick)
        let c_rr = 0.015;
        let rolling_resistance = c_rr * base_normal_load;
        if speed > 0.05 {
            let rr_dir = -body.velocity.normalize_or_zero();
            body.forces += rr_dir * rolling_resistance;
        }
        
        // Freio Motor (Engine Braking): quando throttle = 0 e motor ligado,
        // o atrito interno do motor freia pelas rodas motrizes.
        if !engine.is_stalled && engine.throttle < 0.05 && forward_vel.abs() > 1.0 {
            let engine_brake_torque = gear_ratio * 80.0; // Torque de atrito interno
            let engine_brake_force = engine_brake_torque.min(forward_vel.abs() * chassis.mass / dt);
            let brake_dir = -forward_dir * forward_vel.signum();
            body.forces += brake_dir * engine_brake_force;
        }
        
        // ========================================================
        // === D) TRANSFERÊNCIA DE PESO (WEIGHT TRANSFER) =========
        // ========================================================
        // Aceleração longitudinal do frame anterior (aproximada pela velocidade)
        let accel_long = if dt > 0.0 { (body.forces.dot(forward_dir)) / chassis.mass } else { 0.0 };
        let accel_lat = if dt > 0.0 { (body.forces.dot(lateral_dir)) / chassis.mass } else { 0.0 };
        
        // ΔW = (M * a * h_cg) / L
        let weight_transfer_long = (chassis.mass * accel_long * chassis.cg_height) / chassis.wheel_base;
        let _weight_transfer_lat = (chassis.mass * accel_lat * chassis.cg_height) / 1.5; // Track width ≈ 1.5m
        
        // Normal load nas rodas traseiras (motrizes para RWD/FWD)
        // Aceleração: peso vai pra trás -> traseiras ganham grip
        // Frenagem: peso vai pra frente -> traseiras perdem grip
        let rear_load = (base_normal_load * 0.5) + weight_transfer_long;
        let front_load = (base_normal_load * 0.5) - weight_transfer_long;
        
        // Para FWD usa front_load, para RWD usa rear_load
        let drive_wheel_load = match chassis.drivetrain {
            DrivetrainType::FWD => front_load.max(100.0),
            DrivetrainType::RWD => rear_load.max(100.0),
            DrivetrainType::AWD => base_normal_load.max(100.0),
        };
        
        // ========================================================
        // === GRIP: Terreno × Ambiente ===========================
        // ========================================================
        let voxel = world_res.0.get_voxel_at(transform.translation.x, transform.translation.z);
        let terrain_grip = if voxel == crate::world::VoxelType::Grass { 0.4 } else { 1.0 };
        let env_grip = env.combined_grip_factor();
        let total_grip = terrain_grip * env_grip;
        
        let mu_base = 1.3; // Pneu racing slick em asfalto seco
        let mu = mu_base * total_grip;
        
        // ========================================================
        // === B) TRUE SLIP RATIO (Pacejka Contínuo) ==============
        // ========================================================
        // Funciona para velocidades negativas (ré) sem quebrar a math.
        // v_ref = |forward_vel| clamped a no mínimo 1.0 (evita divisão por zero)
        let v_ref = forward_vel.abs().max(1.0);
        
        // Velocidade angular da roda (proporcional ao torque do motor)
        let wheel_speed = forward_vel + if !engine.is_stalled { drive_torque * 0.001 } else { 0.0 };
        let slip_ratio = ((wheel_speed - forward_vel) / v_ref).clamp(-1.0, 1.0);
        
        // Slip Angle (ângulo de deriva)
        let slip_angle = lateral_vel.atan2(v_ref);
        
        // Pacejka Longitudinal (usa carga da roda motriz com weight transfer)
        let f_long_raw = (drive_wheel_load * mu) * (1.65 * (10.0 * slip_ratio).atan()).sin();
        
        // Pacejka Lateral (usa carga total)
        let f_lat_raw = (base_normal_load * mu) * (1.9 * (12.0 * slip_angle).atan()).sin();
        
        // === CÍRCULO DE KAMM: sqrt(F_long² + F_lat²) <= μ * N ===
        let max_total_force = mu * drive_wheel_load;
        let combined_raw = (f_long_raw * f_long_raw + f_lat_raw * f_lat_raw).sqrt();
        
        let (f_long_final, f_lat_final) = if combined_raw > max_total_force && combined_raw > 0.001 {
            let scale = max_total_force / combined_raw;
            (f_long_raw * scale, f_lat_raw * scale)
        } else {
            (f_long_raw, f_lat_raw)
        };
        
        // === TRAÇÃO: Motor -> Pneu -> Solo ===
        if !engine.is_stalled {
            let engine_force = drive_torque;
            let tire_traction_limit = f_long_final.abs();
            let actual_traction = engine_force.min(tire_traction_limit);
            body.forces += forward_dir * actual_traction;
        }
        
        // === FORÇA LATERAL (Grip de Curva) ===
        let max_needed_force = (lateral_vel.abs() * chassis.mass) / dt;
        let applied_lat = f_lat_final.abs().min(max_needed_force) * f_lat_final.signum();
        let aquaplaning_factor = if env.is_aquaplaning(speed) { 0.1 } else { 1.0 };
        body.forces += -lateral_dir * applied_lat * aquaplaning_factor;
        
        // ========================================================
        // === E) DIFERENCIAL & YAW VECTORING (LSD) ===============
        // ========================================================
        // Aceleração durante curva: o diferencial de deslizamento limitado (LSD)
        // causa um torque de yaw induzido. Quanto mais throttle + steering + slip,
        // mais o carro tende a sobre-esterçar (power oversteer).
        if !engine.is_stalled && engine.throttle > 0.1 && lateral_vel.abs() > 0.5 {
            let steering_input = lateral_vel.signum(); // Direção da curva
            let lsd_yaw = engine.throttle * slip_ratio.abs() * steering_input * 0.8;
            body.torques += Vec3::Y * lsd_yaw * chassis.mass;
        }
        
        // === FRENAGEM ===
        let speed = body.velocity.length();
        if engine.brake > 0.0 && speed > 0.01 {
            let max_brake_force = (speed * chassis.mass) / dt;
            // Freio usa carga do eixo dianteiro (onde o peso transfere na frenagem)
            let front_brake_load = front_load.max(100.0);
            let brake_grip_limit = mu * front_brake_load * 2.0; // *2 porque ambos os eixos fream
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
