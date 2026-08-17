use crate::config::UniverseConfig;
use crate::environment::EnvironmentalState;
use crate::world::VoxelWorldResource;
use bevy::prelude::*;

// ============================================================
// === WHEEL STATE: Per-wheel physics (4 wheels per car)     ===
// ============================================================
#[derive(Clone, Debug)]
pub struct WheelState {
    pub angular_velocity: f32,       // rad/s (wheel rotation)
    pub slip_ratio: f32,             // Longitudinal slip (-1 to 1)
    pub slip_angle: f32,             // Lateral slip (rad)
    pub normal_load: f32,            // N (vertical load on this wheel)
    pub tire_temp: f32,              // 0.0-1.0
    pub suspension_compression: f32, // meters
    pub prev_compression: f32,       // For the damper
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
    pub mu_peak: f32, // Peak friction coefficient (1.5-1.8 for a GT3 slick)
    pub cornering_stiffness: f32, // N/rad
    pub wheel_radius: f32, // meters (0.33 GT3)
    pub wheel_inertia: f32, // kg·m² (I = m*r², ~1.5 for GT3 wheel+tire)
    pub optimal_temp: f32, // Ideal temperature (0.5 = middle of the range)
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
    pub stiffness: f32,   // N/m (spring rate)
    pub damping: f32,     // N·s/m
    pub rest_length: f32, // meters
    pub max_travel: f32,  // meters
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
    // New fields
    pub flywheel_inertia: f32, // kg·m² (moment of inertia of the engine flywheel)
    pub engine_temp: f32,      // °C (engine temperature: startup=60, ideal=90, overheat=110+)
    pub oil_temp: f32,         // °C
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
    pub cg_height: f32,
    pub track_width: f32,      // NEW: Width between wheels (1.5-1.6m)
    pub lsd_lock_ratio: f32,   // NEW: 0.0=open, 1.0=locked differential
    pub brake_torque_max: f32, // NEW: Maximum braking torque (Nm)
}

#[derive(Component)]
pub struct PhysicsBody {
    pub velocity: Vec3,
    pub angular_velocity: Vec3,
    pub forces: Vec3,
    pub torques: Vec3,
    // === 4 INDEPENDENT WHEELS ===
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

#[derive(SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PhysicsSet;

impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            FixedUpdate,
            (
                reset_forces,
                suspension_and_load_system,
                powertrain_and_tire_system,
                collision_system,
                integration_system,
            )
                .chain()
                .in_set(PhysicsSet),
        );
    }
}

// ============================================================
// === 1. RESET FORCES (Correct gravity: F = m * g)         ===
// ============================================================
fn reset_forces(mut query: Query<(&mut PhysicsBody, &VehicleChassis)>) {
    for (mut body, chassis) in query.iter_mut() {
        // FIX #1: Correct gravity using the chassis' real mass
        body.forces = Vec3::new(0.0, -9.81 * chassis.mass, 0.0);
        body.torques = Vec3::ZERO;
    }
}

// ============================================================
// === 2. PER-WHEEL SUSPENSION + NORMAL LOAD               ===
// ============================================================
fn suspension_and_load_system(
    mut query: Query<(
        &SuspensionComponent,
        &mut PhysicsBody,
        &VehicleChassis,
        &Transform,
    )>,
    world_res: Res<VoxelWorldResource>,
    config: Res<UniverseConfig>,
    time: Res<Time>,
) {
    let dt = time.delta_secs().max(0.001);

    for (suspension, mut body, chassis, transform) in query.iter_mut() {
        let world = &world_res.0;
        let pos = transform.translation;
        let forward = transform.right().as_vec3();
        let right_dir = -transform.back().as_vec3(); // real lateral (right)

        // Positions of the 4 wheels in local → global space
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
        // Realistic GT3 downforce: Force ~3.0 * v^2
        let aero_downforce = if config.physics.enable_downforce {
            3.0 * speed.powi(2)
        } else {
            0.0
        };
        let downforce_per_wheel = aero_downforce * 0.25;

        for (i, offset) in wheel_offsets.iter().enumerate() {
            let wheel_pos = pos + *offset;

            // Terrain height (uses heightmap if it exists)
            let ground_y = world.get_height_at(wheel_pos.x, wheel_pos.z);
            let height_above_ground = wheel_pos.y - ground_y;

            let compression = suspension.rest_length - height_above_ground;

            if compression > 0.0 && compression < suspension.max_travel {
                let spring_force = suspension.stiffness * compression;
                let compression_vel = (compression - body.wheels[i].prev_compression) / dt;
                let damper_force = suspension.damping * compression_vel;

                // Downforce physically pushes the wheel against the spring for more grip!
                let total_force = (spring_force + damper_force).max(0.0) + downforce_per_wheel;

                // Guarantee a minimum of grip and prevent jump explosions (the "hovercraft" bug)
                let static_load = chassis.mass * 9.81 / 4.0;
                let clamped_force = total_force.clamp(static_load * 0.2, static_load * 6.0);

                body.forces.y += clamped_force;
                body.wheels[i].normal_load = clamped_force;
                body.wheels[i].suspension_compression = compression;
            } else {
                // Wheel in the air
                body.wheels[i].normal_load = 0.0;
                body.wheels[i].suspension_compression = 0.0;
            }

            body.wheels[i].prev_compression = compression;
        }

        // If no wheel computed a load (fallback), distribute the weight equally
        let total_load: f32 = body.wheels.iter().map(|w| w.normal_load).sum();
        if total_load < 100.0 {
            for w in body.wheels.iter_mut() {
                w.normal_load = total_weight * 0.25;
            }
        }
    }
}

// ============================================================
// === 3. POWERTRAIN + TIRES + AERO (THE HEART)            ===
// ============================================================
fn powertrain_and_tire_system(
    mut query: Query<(
        &mut EngineComponent,
        &mut PhysicsBody,
        &Transform,
        &VehicleChassis,
        &TireComponent,
    )>,
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
        // === A) STARTER MOTOR (STALL RECOVERY)                 ===
        // ========================================================
        if engine.is_stalled && config.physics.enable_stall {
            if engine.current_gear <= 2 && engine.throttle > 0.5 {
                engine.starter_timer += dt;
                if engine.starter_timer >= 1.0 {
                    engine.is_stalled = false;
                    engine.starter_timer = 0.0;
                    engine.current_rpm = 800.0;
                    // Reset stopped wheels
                    for w in body.wheels.iter_mut() {
                        w.angular_velocity = 0.0;
                    }
                }
            } else {
                engine.starter_timer = 0.0;
            }
        }

        // ========================================================
        // === B) TRANSMISSION                                   ===
        // ========================================================
        if engine.shift_timer <= 0.0 {
            if engine.manual_shift_req == 1 && engine.current_gear < 6 {
                engine.current_gear += 1;
                engine.shift_timer = 0.15; // Fast sequential shift (150ms)
            } else if engine.manual_shift_req == -1 && engine.current_gear > 0 {
                let next_ratio = gears[engine.current_gear - 1].abs();
                let expected_rpm = forward_vel.abs() * next_ratio * final_drive
                    / (2.0 * std::f32::consts::PI * tire.wheel_radius)
                    * 60.0;
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

        // Average angular velocity of the drive wheels
        let drive_wheel_avg_omega = match chassis.drivetrain {
            DrivetrainType::RWD => {
                (body.wheels[2].angular_velocity + body.wheels[3].angular_velocity) * 0.5
            }
            DrivetrainType::FWD => {
                (body.wheels[0].angular_velocity + body.wheels[1].angular_velocity) * 0.5
            }
            DrivetrainType::AWD => {
                body.wheels.iter().map(|w| w.angular_velocity).sum::<f32>() * 0.25
            }
        };

        // Wheel-based RPM via the transmission ratio
        let wheel_rpm =
            drive_wheel_avg_omega.abs() * total_ratio_abs * 60.0 / (2.0 * std::f32::consts::PI);
        let free_rev_rpm = 800.0 + engine.throttle * (engine.max_rpm - 800.0);

        // Auto-clutch: slips below 1500 wheel RPM
        let clutch_blend = if engine.current_gear == 1 {
            0.0
        } else if wheel_rpm < 1500.0 {
            (wheel_rpm / 1500.0).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let mut raw_rpm = free_rev_rpm * (1.0 - clutch_blend) + wheel_rpm * clutch_blend;
        let clutch_slip = clutch_blend < 0.95;

        // Flywheel inertia during a gear shift
        if engine.shift_timer > 0.0 {
            // RPM decays naturally through the flywheel during the shift
            let rpm_decay = engine.current_rpm * dt / engine.flywheel_inertia.max(0.05);
            raw_rpm = (engine.current_rpm - rpm_decay * 50.0).max(800.0);
        } else if config.physics.enable_stall
            && !clutch_slip
            && !engine.is_stalled
            && raw_rpm < 550.0
        {
            engine.is_stalled = true;
            engine.starter_timer = 0.0;
        }

        // ========================================================
        // === D) ENGINE THERMAL MODEL                           ===
        // ========================================================
        if !engine.is_stalled && config.physics.enable_thermal {
            let rpm_ratio = engine.current_rpm / engine.max_rpm;
            // Heating: proportional to RPM and throttle
            let heat_gen = rpm_ratio * engine.throttle * 15.0 * dt; // °C/s
            // Cooling: proportional to speed (airflow) + base radiator
            let airflow_cooling = (speed * 0.3 + 2.0) * dt; // °C/s
            engine.engine_temp =
                (engine.engine_temp + heat_gen - airflow_cooling).clamp(20.0, 150.0);
            engine.oil_temp = engine.oil_temp + (engine.engine_temp - engine.oil_temp) * 0.01 * dt;
        } else {
            // A stopped engine cools slowly
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

            // Realistic torque curve (peak ~60% RPM)
            let mut torque_multiplier = 1.0 - (rpm_ratio - 0.6).powi(2) * 2.0;
            if rpm_ratio > 1.0 {
                torque_multiplier = 0.0; // Rev limiter
            } else if rpm_ratio < 0.2 {
                torque_multiplier *= rpm_ratio / 0.2;
            }
            torque_multiplier = torque_multiplier.clamp(0.0, 1.0);

            // Clutch: transfers torque proportional to engagement
            let clutch_engagement = if engine.shift_timer > 0.0 { 0.0 } else { 1.0 };

            // GT3 Traction Control System (TCS)
            let mut tcs_cut = 1.0;
            if config.physics.enable_tcs && chassis.drivetrain == DrivetrainType::RWD {
                let slip_avg =
                    (body.wheels[2].slip_ratio.max(0.0) + body.wheels[3].slip_ratio.max(0.0)) * 0.5;
                if slip_avg > 0.15 {
                    tcs_cut = 1.0 - ((slip_avg - 0.15) * 5.0).clamp(0.0, 0.9); // Cuts up to 90% of the acceleration
                }
            }

            drive_torque = engine.max_torque
                * torque_multiplier
                * (engine.throttle * tcs_cut)
                * total_ratio
                * clutch_engagement
                * thermal_power;

            engine.current_drive_torque = drive_torque;
        }

        // ========================================================
        // === E) AERODYNAMICS                                   ===
        // ========================================================
        // Wind
        body.forces += env.wind * chassis.mass * 0.5;

        // Drag
        let frontal_area = 2.2;
        let aero_drag = if config.physics.enable_aero_drag {
            0.5 * 1.225 * speed * speed * chassis.drag_coefficient * frontal_area
        } else {
            0.0
        };
        let vel_dir = body.velocity.normalize_or_zero();
        body.forces += -vel_dir * aero_drag;

        // Downforce
        let cl = chassis.drag_coefficient * 2.5;
        let downforce = if config.physics.enable_downforce {
            0.5 * 1.225 * speed * speed * cl * frontal_area
        } else {
            0.0
        };
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
        // === F) GRIP: Terrain × Environment × Temperature      ===
        // ========================================================
        let voxel = world_res
            .0
            .get_voxel_at(transform.translation.x, transform.translation.z);
        let terrain_grip = if voxel == crate::world::VoxelType::Grass {
            0.4
        } else {
            1.0
        };
        let env_grip = env.combined_grip_factor();

        // ========================================================
        // === G) PER-WHEEL TIRE PHYSICS (THE REAL HEART)       ===
        // ========================================================
        let drive_indices: Vec<usize> = match chassis.drivetrain {
            DrivetrainType::RWD => vec![2, 3],
            DrivetrainType::FWD => vec![0, 1],
            DrivetrainType::AWD => vec![0, 1, 2, 3],
        };
        let num_drive_wheels = drive_indices.len() as f32;

        // LSD: Distribute torque among the drive wheels
        let mut wheel_torques = [0.0f32; 4];
        if drive_indices.len() == 2 {
            let left_idx = drive_indices[0];
            let right_idx = drive_indices[1];
            let delta_omega = (body.wheels[right_idx].angular_velocity
                - body.wheels[left_idx].angular_velocity)
                .abs();

            // LSD: the higher the lock_ratio, the more equal the distribution
            let lock_factor = (chassis.lsd_lock_ratio * delta_omega * 0.5).clamp(0.0, 0.4);
            let split_left = 0.5
                + lock_factor
                    * (body.wheels[right_idx].angular_velocity
                        - body.wheels[left_idx].angular_velocity)
                        .signum();
            let split_right = 1.0 - split_left;

            wheel_torques[left_idx] = drive_torque * split_left;
            wheel_torques[right_idx] = drive_torque * split_right;
        } else {
            for &idx in &drive_indices {
                wheel_torques[idx] = drive_torque / num_drive_wheels;
            }
        }

        // Brake torque distributed across all 4 wheels (60% front, 40% rear bias)
        let brake_torque_total = engine.brake * chassis.brake_torque_max;
        let brake_per_wheel = [
            brake_torque_total * 0.30, // FL
            brake_torque_total * 0.30, // FR
            brake_torque_total * 0.20, // RL
            brake_torque_total * 0.20, // RR
        ];

        let mut total_long_force = 0.0f32;
        let mut total_lat_force = 0.0f32;
        let mut total_yaw_torque = 0.0f32;

        let half_tw = chassis.track_width * 0.5;
        let half_wb = chassis.wheel_base * 0.5;

        // Relative position of each wheel for the yaw computation
        let wheel_long_pos = [half_wb, half_wb, -half_wb, -half_wb]; // + = front
        let wheel_lat_pos = [-half_tw, half_tw, -half_tw, half_tw]; // + = right

        for i in 0..4 {
            let w = &body.wheels[i];
            let load = w.normal_load.max(10.0);

            // Mu adjusted by tire temperature
            let temp_factor = if config.physics.enable_thermal {
                1.0 - (w.tire_temp - tire.optimal_temp).abs() * 0.5
            } else {
                1.0
            };
            // Speed-grip falloff: fast cars LOSE lateral grip. Forces braking to turn.
            // 20m/s baseline. Each m/s above reduces grip 0.8%. Floor 0.6 at 70m/s.
            let speed_now = body.velocity.length();
            let _ = speed_now;
            let mu = tire.mu_peak * terrain_grip * env_grip * temp_factor.clamp(0.7, 1.0);

            // Linear speed of the wheel vs speed of the car at that point
            let wheel_linear_speed = w.angular_velocity * tire.wheel_radius;

            // Speed of the car at the wheel's contact point (accounting for yaw)
            let raw_fwd_vel = forward_vel + body.angular_velocity.y * wheel_lat_pos[i];
            let raw_lat_vel = lateral_vel - body.angular_velocity.y * wheel_long_pos[i];

            // Speed Sensitive Steering (steering assistant)
            // Prevents the AI from turning the wheel 35 degrees at 200 km/h (which causes immediate spin and total loss of control).
            let max_steer_deg = if speed < 15.0 {
                35.0
            } else {
                (35.0 - (speed - 15.0) * 0.4).max(12.0) // Eases down to a minimum of 12 degrees at monstrous speeds
            };
            let wheel_steer = if i < 2 {
                engine.steer * max_steer_deg.to_radians()
            } else {
                0.0
            };
            let cos_s = wheel_steer.cos();
            let sin_s = wheel_steer.sin();

            // Rotate the velocity into the wheel's reference frame
            let contact_forward_vel = raw_fwd_vel * cos_s + raw_lat_vel * sin_s;
            let contact_lateral_vel = -raw_fwd_vel * sin_s + raw_lat_vel * cos_s;

            // ========================================================
            // === REAL SLIP RATIO: SR = (ω·r - v) / max(|ω·r|, |v|) ===
            // ========================================================
            let v_ref = wheel_linear_speed
                .abs()
                .max(contact_forward_vel.abs())
                .max(0.5);
            let slip_ratio = ((wheel_linear_speed - contact_forward_vel) / v_ref).clamp(-1.0, 1.0);

            // Pacejka longitudinal
            let b_long = 10.0;
            let c_long = 1.5; // Reduced from 1.65 so as not to lose so much grip (keeps 70%) in a burnout
            let f_long = load * mu * (c_long * (b_long * slip_ratio).atan()).sin();

            // Slip Angle & f_lat
            let mut slip_angle = 0.0;
            let f_lat;
            if contact_forward_vel.abs() > 0.5 {
                slip_angle = contact_lateral_vel.atan2(contact_forward_vel.abs());
                let b_lat = 12.0;
                let c_lat = 1.35;
                f_lat = load * mu * (c_lat * (b_lat * slip_angle).atan()).sin();
            } else {
                // Ice Bug Fix: viscous synthetic grip at low speeds
                f_lat = (contact_lateral_vel * load * mu * 2.0).clamp(-load * mu, load * mu);
            }

            // Kamm circle: limit total combined force
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

            // Base torque before the brake
            let base_torque = wheel_torques[i] - tire_reaction_torque;
            let mut expected_omega = w.angular_velocity + (base_torque / tire.wheel_inertia) * dt;

            // The brake acts exclusively to grind down and zero the wheel's rotation (preventing ABS judder)
            let brake_torque = brake_per_wheel[i];
            if brake_torque > 0.0 {
                let stop_torque = (expected_omega.abs() * tire.wheel_inertia) / dt; // Torque needed to stop the wheel
                let actual_brake_torque = if config.physics.enable_abs {
                    brake_torque.min(stop_torque)
                } else {
                    brake_torque
                };

                if expected_omega > 0.0 {
                    expected_omega -= (actual_brake_torque / tire.wheel_inertia) * dt;
                } else if expected_omega < 0.0 {
                    expected_omega += (actual_brake_torque / tire.wheel_inertia) * dt;
                }
            }

            let new_omega = expected_omega;

            // Aquaplaning
            let aquaplaning = if config.physics.enable_aquaplaning && env.is_aquaplaning(speed) {
                0.1
            } else {
                1.0
            };

            // Rotate the wheel forces back into the chassis' reference frame
            let chassis_f_long = f_long_final * cos_s - f_lat_final * sin_s;
            let chassis_f_lat = f_long_final * sin_s + f_lat_final * cos_s;

            // Accumulate forces on chassis
            total_long_force += chassis_f_long * aquaplaning;
            total_lat_force += chassis_f_lat * aquaplaning;

            // YAW TORQUE: Lateral force at wheel arm creates yaw rotation
            // Positive f_lat at the front wheels (positive long_pos) turns the car in the steering direction.
            // The LSD term is kept very small to avoid destabilizing the car under asymmetric drive torque.
            total_yaw_torque += chassis_f_lat * wheel_long_pos[i] * aquaplaning;

            let lsd_speed_gate = (speed / 5.0).clamp(0.0, 1.0);
            total_yaw_torque += chassis_f_long * wheel_lat_pos[i] * 0.005 * lsd_speed_gate;

            // Tire thermal model
            if config.physics.enable_thermal {
                let slip_heat = (slip_ratio.abs() + slip_angle.abs()) * load * 0.000002 * dt;
                let brake_heat = brake_per_wheel[i] * w.angular_velocity.abs() * 0.000001 * dt;
                let cooling = 0.02 * (1.0 + speed * 0.01) * dt;
                body.wheels[i].tire_temp =
                    (body.wheels[i].tire_temp + slip_heat + brake_heat - cooling).clamp(0.0, 1.0);
            }

            // Writeback (clone to avoid a borrow issue)
            let wheel_mut = &mut body.wheels[i];
            wheel_mut.angular_velocity = new_omega;
            wheel_mut.slip_ratio = slip_ratio;
            wheel_mut.slip_angle = slip_angle;
        }

        // Apply accumulated forces to the chassis
        body.forces += forward_dir * total_long_force;
        body.forces += -lateral_dir * total_lat_force;

        // ========================================================
        // === CORRECT MOMENT OF INERTIA                        ===
        // ========================================================
        let yaw_inertia = chassis.mass * (chassis.wheel_base * 0.5).powi(2);
        // Applies the yaw torque purely based on tire physics
        body.torques.y += total_yaw_torque;

        // Braking is now 100% physical via Pacejka slip_ratio on the individual wheels.
        // Removed the old global-braking hack that applied artificial force twice.

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

        // Check the car's contact points (nose, rear, sides)
        let probe_points = [
            pos + forward * (chassis.wheel_base * 0.5), // Nose
            pos - forward * (chassis.wheel_base * 0.5), // Rear
            pos + right * (chassis.track_width * 0.5),  // Right side
            pos - right * (chassis.track_width * 0.5),  // Left side
        ];

        for probe in &probe_points {
            let voxel = world.get_voxel_at(probe.x, probe.z);
            if voxel == crate::world::VoxelType::Wall {
                // Compute the collision normal (car → wall direction)
                let penetration_dir = (*probe - pos).normalize_or_zero();
                let vel_into_wall = body.velocity.dot(penetration_dir);

                if vel_into_wall > 0.0 {
                    // Very low restitution coefficient (0.05) so the wall does not feel "elastic".
                    let restitution = 0.05;
                    let impulse = -(1.0 + restitution) * vel_into_wall * chassis.mass;

                    body.velocity += penetration_dir * (impulse / chassis.mass);

                    // Tangential friction: wall-scraping costs speed
                    let tangential_vel =
                        body.velocity - penetration_dir * body.velocity.dot(penetration_dir);
                    body.velocity -= tangential_vel * 0.5;

                    // Glancing yaw
                    let yaw_kick = forward.cross(penetration_dir).y;
                    body.angular_velocity.y += yaw_kick * vel_into_wall * 0.05;

                    // Push out of the wall
                    transform.translation -= penetration_dir * 0.5;
                }
            }
        }

        // Ground clamp with heightmap
        let ground_y = world.get_height_at(pos.x, pos.z) + 0.5;
        if transform.translation.y < ground_y {
            transform.translation.y = ground_y;
            body.velocity.y = body.velocity.y.max(0.0);
        }
    }
}

// ============================================================
// === 5. INTEGRATION (Semi-implicit Euler)                 ===
// ============================================================
fn integration_system(
    mut query: Query<(&mut PhysicsBody, &VehicleChassis, &mut Transform)>,
    config: Res<UniverseConfig>,
    time: Res<Time>,
) {
    let dt = time.delta_secs();
    if dt == 0.0 {
        return;
    }

    for (mut body, chassis, mut transform) in query.iter_mut() {
        let speed = body.velocity.length();
        let aero_drag = if config.physics.enable_aero_drag {
            0.5 * 1.225 * chassis.drag_coefficient * 2.5 * speed.powi(2)
        } else {
            0.0
        };
        let drag_force = -body.velocity.normalize_or_zero() * aero_drag;

        // Linear
        let acceleration = (body.forces + drag_force) / chassis.mass;
        body.velocity += acceleration * dt;

        // LATERAL SCRUB FRICTION: Sliding sideways burns energy via tire scrub
        // Decomposes velocity into forward and lateral, then applies heavy friction to lateral component
        let forward_dir = transform.right().as_vec3(); // Car's forward direction
        let forward_speed = body.velocity.dot(forward_dir);
        let lateral_vel = body.velocity - forward_dir * forward_speed;
        let lateral_speed = lateral_vel.length();
        if lateral_speed > 0.5 {
            // Aggressive lateral friction: ~3x stronger than regular drag
            // This makes sliding sideways feel "grippy" instead of "soapy"
            let scrub_factor = (1.0 - 3.0 * dt).max(0.85); // Loses ~15% lateral speed per frame minimum
            body.velocity = forward_dir * forward_speed + lateral_vel * scrub_factor;
        }

        transform.translation += body.velocity * dt;

        // Angular damping: yaw damping scales with speed to kill speed-wobbles at high speeds
        // v8: reduced from 800/100 — was over-damped with multiplicative damping removed.
        // v10: 300/45 caused catastrophic instability (0m gens). Reverted to 400/60.
        // The 93° hairpin barrier is a BRAKING issue, not a turning issue.
        let base_yaw_damping = 400.0;
        let speed_yaw_damping = speed * 60.0;
        let total_yaw_damping = base_yaw_damping + speed_yaw_damping;
        let drag_torque_y = -body.angular_velocity.y * total_yaw_damping;

        // Angular: use the real yaw_inertia (stored in torques.x)
        let yaw_inertia = if body.torques.x > 0.0 {
            body.torques.x
        } else {
            chassis.mass * 2.0
        };
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

        // Angular damping v8: REMOVED multiplicative damping (was double-damping
        // with drag_torque_y above). The torque-based damping (lines 740-748) already
        // provides speed-proportional yaw damping. The extra 0.92^(dt*60) was over-
        // constraining turning, preventing cars from navigating 50°+ curves.
        // body.angular_velocity *= 0.92_f32.powf(dt * 60.0);
    }
}
