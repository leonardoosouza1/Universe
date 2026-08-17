use crate::physics::{
    DrivetrainType, EngineComponent, PhysicsBody, SuspensionComponent, TireComponent,
    VehicleChassis,
};
use bevy::prelude::*;

pub fn spawn_porsche_911_gt3(
    commands: &mut Commands,
    transform: Transform,
    meshes: &mut Assets<Mesh>,
    material: Handle<StandardMaterial>,
) -> Entity {
    let chassis_mesh = meshes.add(Cuboid::new(4.5, 1.0, 1.9));

    commands
        .spawn((
            Mesh3d(chassis_mesh),
            MeshMaterial3d(material),
            transform,
            PhysicsBody::default(),
            VehicleChassis {
                drivetrain: DrivetrainType::RWD,
                mass: 1250.0,
                com_offset: Vec3::new(0.0, -0.3, -0.2),
                drag_coefficient: 0.35,
                wheel_base: 2.507,
                cg_height: 0.40,
                track_width: 1.55,
                lsd_lock_ratio: 0.6, // Porsche tem LSD agressivo
                brake_torque_max: 15000.0,
            },
            EngineComponent {
                max_torque: 500.0,
                current_rpm: 800.0,
                max_rpm: 9250.0,
                throttle: 0.0,
                brake: 0.0,
                steer: 0.0,
                current_drive_torque: 0.0,
                current_gear: 2,
                shift_timer: 0.0,
                manual_shift_req: 0,
                is_stalled: false,
                starter_timer: 0.0,
                flywheel_inertia: 0.12, // Flat-6 leve
                engine_temp: 60.0,
                oil_temp: 50.0,
            },
            TireComponent {
                mu_peak: 1.6,
                cornering_stiffness: 55000.0,
                wheel_radius: 0.33,
                wheel_inertia: 1.5,
                optimal_temp: 0.5,
            },
            SuspensionComponent {
                stiffness: 80000.0,
                damping: 6000.0,
                rest_length: 0.4,
                max_travel: 0.8,
                raycast_dir: Vec3::NEG_Y,
                prev_compression: 0.0,
            },
        ))
        .id()
}

pub fn spawn_ferrari_296_gt3(
    commands: &mut Commands,
    transform: Transform,
    meshes: &mut Assets<Mesh>,
    material: Handle<StandardMaterial>,
) -> Entity {
    let e = spawn_porsche_911_gt3(commands, transform, meshes, material.clone());
    commands
        .entity(e)
        .insert(VehicleChassis {
            drivetrain: DrivetrainType::RWD,
            mass: 1250.0,
            com_offset: Vec3::new(0.0, -0.35, 0.1),
            drag_coefficient: 0.38,
            wheel_base: 2.660,
            cg_height: 0.38,
            track_width: 1.58,
            lsd_lock_ratio: 0.5,
            brake_torque_max: 15000.0,
        })
        .insert(EngineComponent {
            max_torque: 710.0,
            current_rpm: 800.0,
            max_rpm: 8000.0,
            throttle: 0.0,
            brake: 0.0,
            steer: 0.0,
            current_drive_torque: 0.0,
            current_gear: 2,
            shift_timer: 0.0,
            manual_shift_req: 0,
            is_stalled: false,
            starter_timer: 0.0,
            flywheel_inertia: 0.15, // Turbo V6 mais pesado
            engine_temp: 60.0,
            oil_temp: 50.0,
        });
    e
}

pub fn spawn_mclaren_720s_gt3(
    commands: &mut Commands,
    transform: Transform,
    meshes: &mut Assets<Mesh>,
    material: Handle<StandardMaterial>,
) -> Entity {
    let e = spawn_porsche_911_gt3(commands, transform, meshes, material.clone());
    commands
        .entity(e)
        .insert(VehicleChassis {
            drivetrain: DrivetrainType::RWD,
            mass: 1265.0,
            com_offset: Vec3::new(0.0, -0.4, 0.15),
            drag_coefficient: 0.36,
            wheel_base: 2.696,
            cg_height: 0.35,
            track_width: 1.60,
            lsd_lock_ratio: 0.45,
            brake_torque_max: 15000.0,
        })
        .insert(EngineComponent {
            max_torque: 700.0,
            current_rpm: 800.0,
            max_rpm: 7500.0,
            throttle: 0.0,
            brake: 0.0,
            steer: 0.0,
            current_drive_torque: 0.0,
            current_gear: 2,
            shift_timer: 0.0,
            manual_shift_req: 0,
            is_stalled: false,
            starter_timer: 0.0,
            flywheel_inertia: 0.14,
            engine_temp: 60.0,
            oil_temp: 50.0,
        });
    e
}

pub fn spawn_aston_vantage_gt3(
    commands: &mut Commands,
    transform: Transform,
    meshes: &mut Assets<Mesh>,
    material: Handle<StandardMaterial>,
) -> Entity {
    let e = spawn_porsche_911_gt3(commands, transform, meshes, material.clone());
    commands
        .entity(e)
        .insert(VehicleChassis {
            drivetrain: DrivetrainType::RWD,
            mass: 1285.0,
            com_offset: Vec3::new(0.0, -0.3, 0.4),
            drag_coefficient: 0.37,
            wheel_base: 2.704,
            cg_height: 0.42,
            track_width: 1.56,
            lsd_lock_ratio: 0.55,
            brake_torque_max: 15000.0,
        })
        .insert(EngineComponent {
            max_torque: 700.0,
            current_rpm: 800.0,
            max_rpm: 7200.0,
            throttle: 0.0,
            brake: 0.0,
            steer: 0.0,
            current_drive_torque: 0.0,
            current_gear: 2,
            shift_timer: 0.0,
            manual_shift_req: 0,
            is_stalled: false,
            starter_timer: 0.0,
            flywheel_inertia: 0.16, // V8 pesado
            engine_temp: 60.0,
            oil_temp: 50.0,
        });
    e
}

pub fn spawn_audi_r8_lms_gt3(
    commands: &mut Commands,
    transform: Transform,
    meshes: &mut Assets<Mesh>,
    material: Handle<StandardMaterial>,
) -> Entity {
    let e = spawn_porsche_911_gt3(commands, transform, meshes, material.clone());
    commands
        .entity(e)
        .insert(VehicleChassis {
            drivetrain: DrivetrainType::RWD,
            mass: 1235.0,
            com_offset: Vec3::new(0.0, -0.35, 0.2),
            drag_coefficient: 0.39,
            wheel_base: 2.650,
            cg_height: 0.39,
            track_width: 1.57,
            lsd_lock_ratio: 0.4,
            brake_torque_max: 15000.0,
        })
        .insert(EngineComponent {
            max_torque: 550.0,
            current_rpm: 800.0,
            max_rpm: 8500.0,
            throttle: 0.0,
            brake: 0.0,
            steer: 0.0,
            current_drive_torque: 0.0,
            current_gear: 2,
            shift_timer: 0.0,
            manual_shift_req: 0,
            is_stalled: false,
            starter_timer: 0.0,
            flywheel_inertia: 0.18, // V10 pesado
            engine_temp: 60.0,
            oil_temp: 50.0,
        });
    e
}
