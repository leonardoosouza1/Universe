use bevy::prelude::*;
use crate::world::VoxelWorldResource;
use crate::environment::EnvironmentalState;
use crate::config::UniverseConfig;

// ============================================================
// === WHEEL STATE: Per-wheel physics (4 wheels per car)     ===
// ============================================================
#[derive(Clone, Debug)]
pub struct WheelState {
    pub angular_velocity: f32, // rad/s (rotação da roda)
    pub slip_ratio: f32,       // Longitudinal slip (-1 to 1)
    pub slip_angle: f32,       // Lateral slip (rad)
    pub normal_load: f32,      // N (carga vertical nesta roda)
    pub tire_temp: f32,        // 0.0-1.0
    pub suspension_compression: f32, // metros
    pub prev_compression: f32, // Para damper
}

impl Default for WheelState {
    fn default() -> Self {
        Self {
            angular_velocity: 0.0,
            slip_ratio: 0.0,
            slip_angle: 0.0,
            normal_load: 3000.0,
            tire_temp: 0.3,
            suspension_compression: 0.0,
            prev_compression: 0.0,
        }
    }
}

// ============================================================
// === COMPONENTS                                            ===
// ============================================================

#[derive(Component)]
pub struct TireComponent {
    pub mu_peak: f32,           // Coeficiente de atrito de pico (1.5-1.8 para GT3 slick)
    pub cornering_stiffness: f32, // N/rad
    pub wheel_radius: f32,       // metros (0.33 GT3)
    pub wheel_inertia: f32,      // kg·m² (I = m*r², ~1.5 para roda+pneu GT3)
    pub optimal_temp: f32,       // Temperatura ideal (0.5 = meio da faixa)
}

impl Default for TireComponent {
    fn default() -> Self {
        Self {
            mu_peak: 1.6,
            cornering_stiffness: 55000.0,
            wheel_radius: 0.33,
            wheel_inertia: 1.5,
            optimal_temp: 0.5,
        }
    }
}

#[derive(Component)]
pub struct SuspensionComponent {
    pub stiffness: f32,        // N/m (spring rate)
    pub damping: f32,          // N·s/m
    pub rest_length: f32,      // metros
    pub max_travel: f32,       // metros
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
    pub steer: f32,
    pub current_drive_torque: f32,
    pub current_gear: usize,
    pub shift_timer: f32,
    pub manual_shift_req: i8,
    pub is_stalled: bool,
    pub starter_timer: f32,
    // Novos campos
    pub flywheel_inertia: f32,   // kg·m² (momento de inércia do volante do motor)
    pub engine_temp: f32,        // °C (temperatura do motor: partida=60, ideal=90, overheat=110+)
    pub oil_temp: f32,           // °C
}

#[derive(Component, PartialEq, Clone, Copy)]
pub enum DrivetrainType { FWD, RWD, AWD }

#[derive(Component)]
pub struct VehicleChassis {
    pub drivetrain: DrivetrainType,
    pub mass: f32,
    pub com_offset: Vec3,
    pub drag_coefficient: f32,
    pub wheel_base: f32,
    pub cg_height: f32,
    pub track_width: f32,        // NOVO: Largura entre rodas (1.5-1.6m)
    pub lsd_lock_ratio: f32,     // NOVO: 0.0=open, 1.0=locked differential
    pub brake_torque_max: f32,   // NOVO: Torque máximo de frenagem (Nm)
}

#[derive(Component)]
pub struct PhysicsBody {
    pub velocity: Vec3,
    pub angular_velocity: Vec3,
    pub forces: Vec3,
    pub torques: Vec3,
    // === 4 RODAS INDEPENDENTES ===
    pub wheels: [WheelState; 4], // [FL, FR, RL, RR]
}

impl Default for PhysicsBody {
    fn default() -> Self {
        Self {
            velocity: Vec3::ZERO,
            angular_velocity: Vec3::ZERO,
            forces: Vec3::ZERO,
            torques: Vec3::ZERO,
            wheels: [
                WheelState::default(),
                WheelState::default(),
                WheelState::default(),
                WheelState::default(),
            ],
        }
    }
}

// ============================================================
// === PLUGIN                                                ===
// ============================================================
pub struct PhysicsPlugin;

impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(FixedUpdate, (
            reset_forces,
            suspension_and_load_system,
            powertrain_and_tire_system,
            collision_system,
            integration_system,
        ).chain());
    }
}

// ============================================================
// === 1. RESET FORCES (Gravidade correta: F = m * g)       ===
// ============================================================
fn reset_forces(mut query: Query<(&mut PhysicsBody, &VehicleChassis)>) {
    for (mut body, chassis) in query.iter_mut() {
        // FIX #1: Gravidade correta usando massa real do chassi
        body.forces = Vec3::new(0.0, -9.81 * chassis.mass, 0.0);
        body.torques = Vec3::ZERO;
    }
}

// ============================================================
// === 2. SUSPENSÃO PER-WHEEL + CARGA NORMAL               ===
// ============================================================
fn suspension_and_load_system(
    mut query: Query<(&SuspensionComponent, &mut PhysicsBody, &VehicleChassis, &Transform)>,
    world_res: Res<VoxelWorldResource>,
    config: Res<UniverseConfig>,
    time: Res<Time>,
) {
    let dt = time.delta_secs().max(0.001);

    for (suspension, mut body, chassis, transform) in query.iter_mut() {
        let world = &world_res.0;
        let pos = transform.translation;
        let forward = transform.right().as_vec3();
        let right_dir = -transform.back().as_vec3(); // lateral real (right)

        // Posições das 4 rodas no espaço local → global
        let half_wb = chassis.wheel_base * 0.5;
        let half_tw = chassis.track_width * 0.5;

        let wheel_offsets = [
            forward * half_wb - right_dir * half_tw,  // FL
            forward * half_wb + right_dir * half_tw,  // FR
            -forward * half_wb - right_dir * half_tw, // RL
            -forward * half_wb + right_dir * half_tw, // RR
        ];

        let total_weight = chassis.mass * 9.81;
        
        let speed = body.velocity.length();
        // GT3 Downforce realístico: Force ~3.0 * v^2 
        let aero_downforce = if config.physics.enable_downforce { 3.0 * speed.powi(2) } else { 0.0 };
        let downforce_per_wheel = aero_downforce * 0.25;

        for (i, offset) in wheel_offsets.iter().enumerate() {
            let wheel_pos = pos + *offset;

            // Altura do terreno (usa heightmap se existir)
            let ground_y = world.get_height_at(wheel_pos.x, wheel_pos.z);
            let height_above_ground = wheel_pos.y - ground_y;

            let compression = suspension.rest_length - height_above_ground;

            if compression > 0.0 && compression < suspension.max_travel {
                let spring_force = suspension.stiffness * compression;
                let compression_vel = (compression - body.wheels[i].prev_compression) / dt;
                let damper_force = suspension.damping * compression_vel;
                
                // Downforce empurra fisicamente a roda contra a mola para mais grip!
                let total_force = (spring_force + damper_force).max(0.0) + downforce_per_wheel;

                // Garantir um mínimo de grip e prevenir explosões de salto (bug "hovercraft")
                let static_load = chassis.mass * 9.81 / 4.0;
                let clamped_force = total_force.clamp(static_load * 0.2, static_load * 6.0);

                body.forces.y += clamped_force;
                body.wheels[i].normal_load = clamped_force;
                body.wheels[i].suspension_compression = compression;
            } else {
                // Roda no ar
                body.wheels[i].normal_load = 0.0;
                body.wheels[i].suspension_compression = 0.0;
            }

            body.wheels[i].prev_compression = compression;
        }

        // Se nenhuma roda calculou carga (fallback), distribuir peso igualmente
        let total_load: f32 = body.wheels.iter().map(|w| w.normal_load).sum();
        if total_load < 100.0 {
            for w in body.wheels.iter_mut() {
                w.normal_load = total_weight * 0.25;
            }
        }
    }
}

// ============================================================
// === 3. POWERTRAIN + PNEUS + AERO (O CORAÇÃO)            ===
// ============================================================
fn powertrain_and_tire_system(
    mut query: Query<(&mut EngineComponent, &mut PhysicsBody, &Transform, &VehicleChassis, &TireComponent)>,
    world_res: Res<VoxelWorldResource>,
    env: Res<EnvironmentalState>,
    config: Res<UniverseConfig>,
    time: Res<Time>,
) {
    let dt = time.delta_secs().max(0.001);

    for (mut engine, mut body, transform, chassis, tire) in query.iter_mut() {
        let speed = body.velocity.length();
        let forward_dir = transform.right().as_vec3();
        let lateral_dir = transform.back().as_vec3();
        let forward_vel = body.velocity.dot(forward_dir);
        let lateral_vel = body.velocity.dot(lateral_dir);

        // ========================================================
        // === GEARS                                             ===
        // ========================================================
        let gears: [f32; 7] = [-3.5, 0.0, 3.8, 2.6, 1.9, 1.45, 1.15];
        let final_drive = 3.7;

        // ========================================================
        // === A) MOTOR DE ARRANQUE (STALL RECOVERY)             ===
        // ========================================================
        if engine.is_stalled && config.physics.enable_stall {
            if engine.current_gear <= 2 && engine.throttle > 0.5 {
                engine.starter_timer += dt;
                if engine.starter_timer >= 1.0 {
                    engine.is_stalled = false;
                    engine.starter_timer = 0.0;
                    engine.current_rpm = 800.0;
                    // Resetar rodas paradas
                    for w in body.wheels.iter_mut() {
                        w.angular_velocity = 0.0;
                    }
                }
            } else {
                engine.starter_timer = 0.0;
            }
        }

        // ========================================================
        // === B) TRANSMISSÃO                                    ===
        // ========================================================
        if engine.shift_timer <= 0.0 {
            if engine.manual_shift_req == 1 && engine.current_gear < 6 {
                engine.current_gear += 1;
                engine.shift_timer = 0.15; // Troca rápida sequential (150ms)
            } else if engine.manual_shift_req == -1 && engine.current_gear > 0 {
                let next_ratio = gears[engine.current_gear - 1].abs();
                let expected_rpm = forward_vel.abs() * next_ratio * final_drive / (2.0 * std::f32::consts::PI * tire.wheel_radius) * 60.0;
                if expected_rpm < engine.max_rpm + 500.0 {
                    engine.current_gear -= 1;
                    engine.shift_timer = 0.15;
                }
            }
        }
        engine.manual_shift_req = 0;

        if engine.shift_timer > 0.0 {
            engine.shift_timer -= dt;
        }

        let gear_ratio = gears[engine.current_gear];
        let total_ratio = gear_ratio * final_drive;

        // ========================================================
        // === C) RPM + CLUTCH + TORQUE                          ===
        // ========================================================
        let _gear_ratio_abs = gear_ratio.abs();
        let total_ratio_abs = total_ratio.abs();

        // Velocidade angular média das rodas motrizes
        let drive_wheel_avg_omega = match chassis.drivetrain {
            DrivetrainType::RWD => (body.wheels[2].angular_velocity + body.wheels[3].angular_velocity) * 0.5,
            DrivetrainType::FWD => (body.wheels[0].angular_velocity + body.wheels[1].angular_velocity) * 0.5,
            DrivetrainType::AWD => body.wheels.iter().map(|w| w.angular_velocity).sum::<f32>() * 0.25,
        };

        // RPM baseado na roda via razão de transmissão
        let wheel_rpm = drive_wheel_avg_omega.abs() * total_ratio_abs * 60.0 / (2.0 * std::f32::consts::PI);
        let free_rev_rpm = 800.0 + engine.throttle * (engine.max_rpm - 800.0);

        // Auto-clutch: patina abaixo de 1500 RPM da roda
        let clutch_blend = if engine.current_gear == 1 {
            0.0
        } else if wheel_rpm < 1500.0 {
            (wheel_rpm / 1500.0).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let mut raw_rpm = free_rev_rpm * (1.0 - clutch_blend) + wheel_rpm * clutch_blend;
        let clutch_slip = clutch_blend < 0.95;

        // Flywheel inertia durante troca de marcha
        if engine.shift_timer > 0.0 {
            // RPM decai naturalmente pelo flywheel durante a troca
            let rpm_decay = engine.current_rpm * dt / engine.flywheel_inertia.max(0.05);
            raw_rpm = (engine.current_rpm - rpm_decay * 50.0).max(800.0);
        } else if config.physics.enable_stall && !clutch_slip && !engine.is_stalled && raw_rpm < 550.0 {
            engine.is_stalled = true;
            engine.starter_timer = 0.0;
        }

        // ========================================================
        // === D) ENGINE THERMAL MODEL                           ===
        // ========================================================
        if !engine.is_stalled && config.physics.enable_thermal {
            let rpm_ratio = engine.current_rpm / engine.max_rpm;
            // Aquecimento: proporcional a RPM e throttle
            let heat_gen = rpm_ratio * engine.throttle * 15.0 * dt; // °C/s
            // Resfriamento: proporcional à velocidade (airflow) + radiador base
            let airflow_cooling = (speed * 0.3 + 2.0) * dt; // °C/s
            engine.engine_temp = (engine.engine_temp + heat_gen - airflow_cooling).clamp(20.0, 150.0);
            engine.oil_temp = engine.oil_temp + (engine.engine_temp - engine.oil_temp) * 0.01 * dt;
        } else {
            // Motor parado esfria lentamente
            engine.engine_temp = (engine.engine_temp - 1.0 * dt).max(20.0);
        }

        // Power reduction from overheating (>105°C starts losing power)
        let thermal_power = if config.physics.enable_thermal && engine.engine_temp > 105.0 {
            (1.0 - (engine.engine_temp - 105.0) / 45.0).clamp(0.1, 1.0)
        } else {
            1.0
        };

        let drive_torque;
        if engine.is_stalled {
            engine.current_rpm = 0.0;
            drive_torque = 0.0;
        } else {
            engine.current_rpm = raw_rpm.clamp(800.0, engine.max_rpm + 250.0);
            let rpm_ratio = engine.current_rpm / engine.max_rpm;

            // Curva de Torque Realística (pico ~60% RPM)
            let mut torque_multiplier = 1.0 - (rpm_ratio - 0.6).powi(2) * 2.0;
            if rpm_ratio > 1.0 {
                torque_multiplier = 0.0; // Rev limiter
            } else if rpm_ratio < 0.2 {
                torque_multiplier *= rpm_ratio / 0.2;
            }
            torque_multiplier = torque_multiplier.clamp(0.0, 1.0);

            // Embreagem: transfere torque proporcional ao engagement
            let clutch_engagement = if engine.shift_timer > 0.0 { 0.0 } else { 1.0 };

            // GT3 Traction Control System (TCS)
            let mut tcs_cut = 1.0;
            if config.physics.enable_tcs && chassis.drivetrain == DrivetrainType::RWD {
                let slip_avg = (body.wheels[2].slip_ratio.max(0.0) + body.wheels[3].slip_ratio.max(0.0)) * 0.5;
                if slip_avg > 0.15 {
                    tcs_cut = 1.0 - ((slip_avg - 0.15) * 5.0).clamp(0.0, 0.9); // Corta até 90% da aceleração
                }
            }

            drive_torque = engine.max_torque * torque_multiplier * (engine.throttle * tcs_cut)
                * total_ratio * clutch_engagement * thermal_power;
                
            engine.current_drive_torque = drive_torque;
        }

        // ========================================================
        // === E) AERODINÂMICA                                   ===
        // ========================================================
        // Vento
        body.forces += env.wind * chassis.mass * 0.5;

        // Arrasto
        let frontal_area = 2.2;
        let aero_drag = if config.physics.enable_aero_drag { 0.5 * 1.225 * speed * speed * chassis.drag_coefficient * frontal_area } else { 0.0 };
        let vel_dir = body.velocity.normalize_or_zero();
        body.forces += -vel_dir * aero_drag;

        // Downforce
        let cl = chassis.drag_coefficient * 2.5;
        let downforce = if config.physics.enable_downforce { 0.5 * 1.225 * speed * speed * cl * frontal_area } else { 0.0 };
        body.forces.y -= downforce;

        // Rolling resistance
        let total_load: f32 = body.wheels.iter().map(|w| w.normal_load).sum();
        let rolling_resistance = 0.015 * total_load;
        if speed > 0.05 {
            body.forces += -vel_dir * rolling_resistance;
        }

        // Engine braking
        if !engine.is_stalled && engine.throttle < 0.05 && forward_vel.abs() > 1.0 {
            let engine_brake = total_ratio_abs * 80.0;
            let max_brake = forward_vel.abs() * chassis.mass / dt;
            let actual = engine_brake.min(max_brake);
            body.forces += -forward_dir * forward_vel.signum() * actual;
        }

        // ========================================================
        // === F) GRIP: Terreno × Ambiente × Temperatura         ===
        // ========================================================
        let voxel = world_res.0.get_voxel_at(transform.translation.x, transform.translation.z);
        let terrain_grip = if voxel == crate::world::VoxelType::Grass { 0.4 } else { 1.0 };
        let env_grip = env.combined_grip_factor();

        // ========================================================
        // === G) PER-WHEEL TIRE PHYSICS (O CORAÇÃO REAL)       ===
        // ========================================================
        let drive_indices: Vec<usize> = match chassis.drivetrain {
            DrivetrainType::RWD => vec![2, 3],
            DrivetrainType::FWD => vec![0, 1],
            DrivetrainType::AWD => vec![0, 1, 2, 3],
        };
        let num_drive_wheels = drive_indices.len() as f32;

        // LSD: Distribuir torque entre rodas motrizes
        let mut wheel_torques = [0.0f32; 4];
        if drive_indices.len() == 2 {
            let left_idx = drive_indices[0];
            let right_idx = drive_indices[1];
            let delta_omega = (body.wheels[right_idx].angular_velocity
                - body.wheels[left_idx].angular_velocity).abs();

            // LSD: quanto maior o lock_ratio, mais igual a distribuição
            let lock_factor = (chassis.lsd_lock_ratio * delta_omega * 0.5).clamp(0.0, 0.4);
            let split_left = 0.5 + lock_factor * (body.wheels[right_idx].angular_velocity - body.wheels[left_idx].angular_velocity).signum();
            let split_right = 1.0 - split_left;

            wheel_torques[left_idx] = drive_torque * split_left;
            wheel_torques[right_idx] = drive_torque * split_right;
        } else {
            for &idx in &drive_indices {
                wheel_torques[idx] = drive_torque / num_drive_wheels;
            }
        }

        // Brake torque distribuído em todas as 4 rodas (60% front, 40% rear bias)
        let brake_torque_total = engine.brake * chassis.brake_torque_max;
        let brake_per_wheel = [
            brake_torque_total * 0.30,  // FL
            brake_torque_total * 0.30,  // FR
            brake_torque_total * 0.20,  // RL
            brake_torque_total * 0.20,  // RR
        ];

        let mut total_long_force = 0.0f32;
        let mut total_lat_force = 0.0f32;
        let mut total_yaw_torque = 0.0f32;

        let half_tw = chassis.track_width * 0.5;
        let half_wb = chassis.wheel_base * 0.5;

        // Posição relativa de cada roda para cálculo de yaw
        let wheel_long_pos = [half_wb, half_wb, -half_wb, -half_wb]; // + = frente
        let wheel_lat_pos = [-half_tw, half_tw, -half_tw, half_tw]; // + = direita

        for i in 0..4 {
            let w = &body.wheels[i];
            let load = w.normal_load.max(10.0);

            // Mu ajustado por temperatura do pneu
            let temp_factor = if config.physics.enable_thermal { 1.0 - (w.tire_temp - tire.optimal_temp).abs() * 0.5 } else { 1.0 };
            let mu = tire.mu_peak * terrain_grip * env_grip * temp_factor.clamp(0.7, 1.0);

            // Velocidade linear da roda vs velocidade do carro naquele ponto
            let wheel_linear_speed = w.angular_velocity * tire.wheel_radius;

            // Velocidade do carro no ponto de contato da roda (considerando yaw)
            let raw_fwd_vel = forward_vel + body.angular_velocity.y * wheel_lat_pos[i];
            let raw_lat_vel = lateral_vel - body.angular_velocity.y * wheel_long_pos[i];

            // Speed Sensitive Steering (Assistente de Direção)
            // Impede que a IA vire a roda 35 graus a 200 km/h (o que causa spin imediato e perda total de controle).
            let max_steer_deg = if speed < 15.0 { 
                35.0 
            } else { 
                (35.0 - (speed - 15.0) * 0.4).max(12.0) // Suaviza até um mínimo de 12 graus em velocidades monstruosas
            };
            let wheel_steer = if i < 2 { engine.steer * max_steer_deg.to_radians() } else { 0.0 };
            let cos_s = wheel_steer.cos();
            let sin_s = wheel_steer.sin();

            // Rotação da velocidade para o referencial da roda
            let contact_forward_vel = raw_fwd_vel * cos_s + raw_lat_vel * sin_s;
            let contact_lateral_vel = -raw_fwd_vel * sin_s + raw_lat_vel * cos_s;

            // ========================================================
            // === SLIP RATIO REAL: SR = (ω·r - v) / max(|ω·r|, |v|) ===
            // ========================================================
            let v_ref = wheel_linear_speed.abs().max(contact_forward_vel.abs()).max(0.5);
            let slip_ratio = ((wheel_linear_speed - contact_forward_vel) / v_ref).clamp(-1.0, 1.0);

            // Pacejka longitudinal
            let b_long = 10.0;
            let c_long = 1.5; // Reduzido de 1.65 para não perder tanto grip (mantém 70%) no burnout
            let f_long = load * mu * (c_long * (b_long * slip_ratio).atan()).sin();

            // Slip Angle & f_lat
            let mut slip_angle = 0.0;
            let f_lat;
            if contact_forward_vel.abs() > 0.5 {
                slip_angle = contact_lateral_vel.atan2(contact_forward_vel.abs());
                let b_lat = 12.0;
                let c_lat = 1.35; // Reduzido de 1.9. Mantém ~85% do grip quando está 100% de lado, curando o "bug do gelo" lateral
                f_lat = load * mu * (c_lat * (b_lat * slip_angle).atan()).sin();
            } else {
                // Bug Gelo Fix: Grip sintético viscoso a baixas velocidades
                f_lat = (contact_lateral_vel * load * mu * 2.0).clamp(-load * mu, load * mu);
            }

            // Círculo de Kamm: limitar força total combinada
            let max_force = mu * load;
            let combined = (f_long * f_long + f_lat * f_lat).sqrt();
            let (f_long_final, f_lat_final) = if combined > max_force && combined > 0.001 {
                let scale = max_force / combined;
                (f_long * scale, f_lat * scale)
            } else {
                (f_long, f_lat)
            };

            // ========================================================
            // === WHEEL ANGULAR VELOCITY INTEGRATION (STABLE)      ===
            // ========================================================
            let tire_reaction_torque = f_long_final * tire.wheel_radius;
            
            // Torque base antes do freio
            let base_torque = wheel_torques[i] - tire_reaction_torque;
            let mut expected_omega = w.angular_velocity + (base_torque / tire.wheel_inertia) * dt;
            
            // O freio atua exclusivamente para atritar e zerar a rotação da roda (prevenindo trepidação ABS)
            let brake_torque = brake_per_wheel[i]; 
            if brake_torque > 0.0 {
                let stop_torque = (expected_omega.abs() * tire.wheel_inertia) / dt; // Torque necessário p/ parar a roda
                let actual_brake_torque = if config.physics.enable_abs { brake_torque.min(stop_torque) } else { brake_torque };
                
                if expected_omega > 0.0 {
                    expected_omega -= (actual_brake_torque / tire.wheel_inertia) * dt;
                } else if expected_omega < 0.0 {
                    expected_omega += (actual_brake_torque / tire.wheel_inertia) * dt;
                }
            }
            
            let new_omega = expected_omega;

            // Aquaplanagem
            let aquaplaning = if config.physics.enable_aquaplaning && env.is_aquaplaning(speed) { 0.1 } else { 1.0 };

            // Rotacionar as forças da roda de volta para o referencial do chassi
            let chassis_f_long = f_long_final * cos_s - f_lat_final * sin_s;
            let chassis_f_lat = f_long_final * sin_s + f_lat_final * cos_s;

            // Accumulate forces on chassis
            total_long_force += chassis_f_long * aquaplaning;
            total_lat_force += chassis_f_lat * aquaplaning;

            // TRUE PHYSICS YAW
            // Torque em Y = Braço perpendicular X Força Orto.
            // Pela regra da mão direita no Bevy (Y para Cima):
            // - Força longitudinal (X) na roda da direita (+Z) puxa o nariz para esquerda (+Y)
            // - Força lateral (Z) na frente (+X) empurra o nariz para direita (-Y)
            total_yaw_torque += wheel_lat_pos[i] * chassis_f_long * aquaplaning;
            total_yaw_torque -= wheel_long_pos[i] * chassis_f_lat * aquaplaning;
            
            // Tire thermal model
            if config.physics.enable_thermal {
                let slip_heat = (slip_ratio.abs() + slip_angle.abs()) * load * 0.000002 * dt;
                let brake_heat = brake_per_wheel[i] * w.angular_velocity.abs() * 0.000001 * dt;
                let cooling = 0.02 * (1.0 + speed * 0.01) * dt;
                body.wheels[i].tire_temp = (body.wheels[i].tire_temp + slip_heat + brake_heat - cooling).clamp(0.0, 1.0);
            }

            // Writeback (clone para evitar borrow issue)
            let wheel_mut = &mut body.wheels[i];
            wheel_mut.angular_velocity = new_omega;
            wheel_mut.slip_ratio = slip_ratio;
            wheel_mut.slip_angle = slip_angle;
        }

        // Aplicar forças acumuladas ao chassi
        body.forces += forward_dir * total_long_force;
        body.forces += -lateral_dir * total_lat_force;

        // ========================================================
        // === MOMENTO DE INÉRCIA CORRETO                       ===
        // ========================================================
        let yaw_inertia = chassis.mass * (chassis.wheel_base * 0.5).powi(2);
        // Aplica o yaw torque puramente baseado na física dos pneus
        body.torques.y += total_yaw_torque;

        // A frenagem agora é 100% física via Pacejka slip_ratio nas rodas individuais.
        // Removido o hack antigo de frenagem global que aplicava força artificial duas vezes.

        // Store yaw_inertia for integration (stash it in angular_velocity temporarily scaled)
        // We use torques.y and will divide by inertia in integration
        // Hack: store inertia in torques.x which we don't use for anything else
        body.torques.x = yaw_inertia;
    }
}

// ============================================================
// === 4. COLLISION WITH NORMAL VECTOR                       ===
// ============================================================
fn collision_system(
    mut query: Query<(&mut PhysicsBody, &VehicleChassis, &mut Transform)>,
    world_res: Res<VoxelWorldResource>,
) {
    let world = &world_res.0;

    for (mut body, chassis, mut transform) in query.iter_mut() {
        let pos = transform.translation;
        let forward = transform.right().as_vec3();
        let right = -transform.back().as_vec3();

        // Checar pontos de contato do carro (nariz, traseira, laterais)
        let probe_points = [
            pos + forward * (chassis.wheel_base * 0.5),   // Nariz
            pos - forward * (chassis.wheel_base * 0.5),   // Traseira
            pos + right * (chassis.track_width * 0.5),      // Lateral direita
            pos - right * (chassis.track_width * 0.5),      // Lateral esquerda
        ];

        for probe in &probe_points {
            let voxel = world.get_voxel_at(probe.x, probe.z);
            if voxel == crate::world::VoxelType::Wall {
                // Calcular normal da colisão (direção do carro → parede)
                let penetration_dir = (*probe - pos).normalize_or_zero();
                let vel_into_wall = body.velocity.dot(penetration_dir);

                if vel_into_wall > 0.0 {
                    // Coeficiente de restituição muito baixo (0.05) para parede não parecer "elástica".
                    let restitution = 0.05;
                    let impulse = -(1.0 + restitution) * vel_into_wall * chassis.mass;

                    body.velocity += penetration_dir * (impulse / chassis.mass);

                    // Tangential friction: wall-scraping costs speed (discourage wall-riding strategy)
                    let tangential_vel = body.velocity - penetration_dir * body.velocity.dot(penetration_dir);
                    body.velocity -= tangential_vel * 0.25; // 25% speed loss per contact frame along the wall

                    // Glancing yaw: wall pushes nose away (realistic bounce + helps AI unstick)
                    // Cross product of penetration_dir (wall normal-ish) with forward gives yaw sign
                    let yaw_kick = forward.cross(penetration_dir).y;
                    body.angular_velocity.y += yaw_kick * vel_into_wall * 0.05;

                    // Empurrar para fora da parede
                    transform.translation -= penetration_dir * 0.5;
                }
            }
        }

        // Ground clamp com heightmap
        let ground_y = world.get_height_at(pos.x, pos.z) + 0.5;
        if transform.translation.y < ground_y {
            transform.translation.y = ground_y;
            body.velocity.y = body.velocity.y.max(0.0);
        }
    }
}

// ============================================================
// === 5. INTEGRATION (Euler Semi-Implícito)                ===
// ============================================================
fn integration_system(
    mut query: Query<(&mut PhysicsBody, &VehicleChassis, &mut Transform)>,
    config: Res<UniverseConfig>,
    time: Res<Time>,
) {
    let dt = time.delta_secs();
    if dt == 0.0 { return; }

    for (mut body, chassis, mut transform) in query.iter_mut() {
        let speed = body.velocity.length();
        let aero_drag = if config.physics.enable_aero_drag { 0.5 * 1.225 * chassis.drag_coefficient * 2.5 * speed.powi(2) } else { 0.0 };
        let drag_force = -body.velocity.normalize_or_zero() * aero_drag;

        // Linear
        let acceleration = (body.forces + drag_force) / chassis.mass;
        body.velocity += acceleration * dt;
        transform.translation += body.velocity * dt;

        // Angular Damping: Damping de yaw escala com a velocidade para matar speed-wobbles em altas velocidades
        let base_yaw_damping = 800.0;
        let speed_yaw_damping = speed * 100.0; 
        let total_yaw_damping = base_yaw_damping + speed_yaw_damping;
        let drag_torque_y = -body.angular_velocity.y * total_yaw_damping;

        // Angular: usar yaw_inertia real (armazenada em torques.x)
        let yaw_inertia = if body.torques.x > 0.0 { body.torques.x } else { chassis.mass * 2.0 };
        let angular_accel_y = (body.torques.y + drag_torque_y) / yaw_inertia;
        body.angular_velocity.y += angular_accel_y * dt;

        // Ground constraint (handled in collision but double-check)
        if transform.translation.y < 0.5 {
            transform.translation.y = 0.5;
            body.velocity.y = body.velocity.y.max(0.0);
        }

        // Yaw rotation
        transform.rotate_y(body.angular_velocity.y * dt);
        transform.rotation = transform.rotation.normalize();

        // Angular damping (tire drag + aero)
        body.angular_velocity *= 0.92_f32.powf(dt * 60.0);
    }
}
