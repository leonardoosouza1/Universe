use bevy::prelude::*;
use crate::physics::{VehicleChassis, EngineComponent, DrivetrainType, TireComponent, SuspensionComponent, PhysicsBody};

pub fn spawn_porsche_911_gt3(
    commands: &mut Commands,
    transform: Transform,
    meshes: &mut Assets<Mesh>,
    material: Handle<StandardMaterial>,
) -> Entity {
    let chassis_mesh = meshes.add(Cuboid::new(4.5, 1.0, 1.9));

    commands.spawn((
        Mesh3d(chassis_mesh),
        MeshMaterial3d(material),
        transform,
        PhysicsBody::default(),
        VehicleChassis {
            drivetrain: DrivetrainType::RWD,
            mass: 1250.0,
            com_offset: Vec3::new(0.0, -0.3, -0.2), // Engine is in the rear
            drag_coefficient: 0.35, // High downforce = high drag
            wheel_base: 2.507,
            cg_height: 0.40, 
        },
        EngineComponent {
            max_torque: 500.0,
            current_rpm: 800.0,
            max_rpm: 9250.0, // High revving flat-6
            throttle: 0.0,
            brake: 0.0,
            current_gear: 2, 
            shift_timer: 0.0,
            manual_shift_req: 0,
            is_stalled: false,
            starter_timer: 0.0,
        },
        TireComponent::default(),
        SuspensionComponent {
            stiffness: 80000.0, // Race stiffness
            damping: 6000.0,
            rest_length: 0.4,
            max_travel: 0.8,
            raycast_dir: Vec3::NEG_Y,
            prev_compression: 0.0,
        }
    )).id()
}

pub fn spawn_ferrari_296_gt3(
    commands: &mut Commands,
    transform: Transform,
    meshes: &mut Assets<Mesh>,
    material: Handle<StandardMaterial>,
) -> Entity {
    let e = spawn_porsche_911_gt3(commands, transform, meshes, material.clone());
    commands.entity(e).insert(VehicleChassis {
            drivetrain: DrivetrainType::RWD,
            mass: 1250.0,
            com_offset: Vec3::new(0.0, -0.35, 0.1), // Mid engine
            drag_coefficient: 0.38,
            wheel_base: 2.660,
            cg_height: 0.38,
    }).insert(EngineComponent {
            max_torque: 710.0, // Turbo V6
            current_rpm: 800.0,
            max_rpm: 8000.0,
            throttle: 0.0,
            brake: 0.0,
            current_gear: 2, 
            shift_timer: 0.0,
            manual_shift_req: 0,
            is_stalled: false,
            starter_timer: 0.0,
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
    commands.entity(e).insert(VehicleChassis {
            drivetrain: DrivetrainType::RWD,
            mass: 1265.0,
            com_offset: Vec3::new(0.0, -0.4, 0.15),
            drag_coefficient: 0.36,
            wheel_base: 2.696,
            cg_height: 0.35, // Very low CG
    }).insert(EngineComponent {
            max_torque: 700.0,
            current_rpm: 800.0,
            max_rpm: 7500.0,
            throttle: 0.0,
            brake: 0.0,
            current_gear: 2, 
            shift_timer: 0.0,
            manual_shift_req: 0,
            is_stalled: false,
            starter_timer: 0.0,
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
    commands.entity(e).insert(VehicleChassis {
            drivetrain: DrivetrainType::RWD,
            mass: 1285.0,
            com_offset: Vec3::new(0.0, -0.3, 0.4), // Front mid engine
            drag_coefficient: 0.37,
            wheel_base: 2.704,
            cg_height: 0.42,
    }).insert(EngineComponent {
            max_torque: 700.0, // Twin turbo V8
            current_rpm: 800.0,
            max_rpm: 7200.0,
            throttle: 0.0,
            brake: 0.0,
            current_gear: 2, 
            shift_timer: 0.0,
            manual_shift_req: 0,
            is_stalled: false,
            starter_timer: 0.0,
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
    commands.entity(e).insert(VehicleChassis {
            drivetrain: DrivetrainType::RWD,
            mass: 1235.0,
            com_offset: Vec3::new(0.0, -0.35, 0.2), // Mid engine
            drag_coefficient: 0.39,
            wheel_base: 2.650,
            cg_height: 0.39, 
    }).insert(EngineComponent {
            max_torque: 550.0, // N/A V10
            current_rpm: 800.0,
            max_rpm: 8500.0,
            throttle: 0.0,
            brake: 0.0,
            current_gear: 2, 
            shift_timer: 0.0,
            manual_shift_req: 0,
            is_stalled: false,
            starter_timer: 0.0,
    });
    e
}
