use bevy::prelude::*;
use crate::physics::{VehicleChassis, EngineComponent, DrivetrainType, TireComponent, SuspensionComponent, PhysicsBody};

pub fn spawn_chassis_argo(
    commands: &mut Commands,
    transform: Transform,
    meshes: &mut Assets<Mesh>,
    material: Handle<StandardMaterial>,
) -> Entity {
    let chassis_mesh = meshes.add(Cuboid::new(4.0, 1.0, 1.8));

    commands.spawn((
        Mesh3d(chassis_mesh),
        MeshMaterial3d(material),
        transform,
        PhysicsBody::default(),
        VehicleChassis {
            drivetrain: DrivetrainType::FWD,
            mass: 1100.0,
            com_offset: Vec3::new(0.0, -0.2, 0.5), // Front heavy
            drag_coefficient: 0.18, // Aerodinâmica de superesportivo
            wheel_base: 2.5,
        },
        EngineComponent {
            max_torque: 450.0, // Torque agressivo real (0-100 mais rápido)
            current_rpm: 800.0,
            max_rpm: 6500.0,
            throttle: 0.0,
            brake: 0.0,
            current_gear: 0,
            shift_timer: 0.0,
            manual_shift_req: 0,
            is_stalled: false,
        },
        TireComponent::default(),
        SuspensionComponent {
            stiffness: 40000.0,
            damping: 3000.0,
            rest_length: 0.5,
            max_travel: 1.0,
            raycast_dir: Vec3::NEG_Y,
            prev_compression: 0.0,
        }
    )).id()
}

pub fn spawn_chassis_sandero(
    commands: &mut Commands,
    transform: Transform,
    meshes: &mut Assets<Mesh>,
    material: Handle<StandardMaterial>,
) -> Entity {
    let e = spawn_chassis_argo(commands, transform, meshes, material.clone());
    commands.entity(e).insert(VehicleChassis {
            drivetrain: DrivetrainType::FWD,
            mass: 1050.0,
            com_offset: Vec3::new(0.0, -0.1, 0.4),
            drag_coefficient: 0.19,
            wheel_base: 2.6,
    });
    e
}

pub fn spawn_chassis_ka(
    commands: &mut Commands,
    transform: Transform,
    meshes: &mut Assets<Mesh>,
    material: Handle<StandardMaterial>,
) -> Entity {
    let e = spawn_chassis_argo(commands, transform, meshes, material.clone());
    commands.entity(e).insert(VehicleChassis {
            drivetrain: DrivetrainType::FWD,
            mass: 980.0,
            com_offset: Vec3::new(0.0, -0.3, 0.6),
            drag_coefficient: 0.20,
            wheel_base: 2.4,
    });
    e
}
