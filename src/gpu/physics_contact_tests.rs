use super::{ContactDiagnostics, PhysicsGpu, PhysicsStateGpu, CONTACT_DIAGNOSTICS_FLAG};
use bytemuck::Zeroable;

#[test]
fn diagnostic_flag_uses_bit_six_after_existing_flags() {
    assert_eq!(CONTACT_DIAGNOSTICS_FLAG, 64);
}

#[test]
fn zero_padding_decodes_as_no_contact() {
    let state = PhysicsStateGpu::zeroed();
    assert_eq!(
        state.contact_diagnostics(),
        ContactDiagnostics {
            contact_mask: 0,
            impact_speed_mps: 0.0,
            impulse_ns: 0.0,
            energy_loss_j: 0.0,
        }
    );
}

#[test]
fn padding_values_decode_without_reinterpreting_units() {
    let mut state = PhysicsStateGpu::zeroed();
    state._pad_pos = 5.0;
    state._pad_fwd = 12.5;
    state._pad_lat = 16875.0;
    state._pad_dt = 42000.0;
    assert_eq!(
        state.contact_diagnostics(),
        ContactDiagnostics {
            contact_mask: 5,
            impact_speed_mps: 12.5,
            impulse_ns: 16875.0,
            energy_loss_j: 42000.0,
        }
    );
}

/// Manual fixture for the three-kernel path. It is ignored because it opens Vulkan;
/// root can run it exclusively with `cargo test --release --lib -- --ignored
/// physics_contact_tests` after the sequential GPU battery.
#[test]
#[ignore = "opens a real GPU; run manually during the sequential battery"]
fn straight_line_brake_and_contact_diagnostics() {
    let n = 64usize;
    // Road everywhere is material 1; a single wall voxel is material 2.
    let mut grid = vec![1u8; n * n];
    let hmap = vec![0.0f32; n * n];
    let gpu_road = PhysicsGpu::new(&grid, &hmap, n, 3.0, 2).expect("GPU unavailable");
    // Front probe (x≈32) sees one wall voxel while the car moves +X.
    grid[32 * n + 32] = 2;
    let gpu_wall = PhysicsGpu::new(&grid, &hmap, n, 3.0, 2).expect("GPU unavailable");
    let make_car = || {
        let mut car = super::make_physics_state([96.0, 0.5, 96.0], [1.0, 0.0, 0.0]);
        car.vel = [10.0, 0.0, 0.0];
        for wheel in car.wheels.iter_mut().step_by(2) {
            wheel[0] = 10.0 / car.wheel_radius;
        }
        car
    };
    eprintln!("physics contract GPU adapter: {}", gpu_road.adapter);
    // Rejected input must leave every record untouched and the context reusable.
    let mut invalid_batch = [make_car(), make_car()];
    invalid_batch[1].brake_cmd = 1.1;
    let before = bytemuck::cast_slice::<_, u8>(&invalid_batch).to_vec();
    assert!(!gpu_road.step(&mut invalid_batch));
    assert_eq!(bytemuck::cast_slice::<_, u8>(&invalid_batch), before);
    assert!(gpu_road.step_raw(&invalid_batch).is_empty());
    let mut coast = make_car();
    coast.throttle = 0.0;
    coast.brake_cmd = 0.0;
    // With no wall in the probes, diagnostics on/off must leave trajectory identical.
    let mut off = make_car();
    off.config_flags &= !CONTACT_DIAGNOSTICS_FLAG;
    let mut on = make_car();
    on.config_flags |= CONTACT_DIAGNOSTICS_FLAG;
    for _ in 0..20 {
        assert!(gpu_road.step(std::slice::from_mut(&mut off)));
        assert!(gpu_road.step(std::slice::from_mut(&mut on)));
    }
    assert_eq!(off.vel, on.vel);
    assert_eq!(off.pos, on.pos);
    assert_eq!(
        off.contact_diagnostics(),
        ContactDiagnostics {
            contact_mask: 0,
            impact_speed_mps: 0.0,
            impulse_ns: 0.0,
            energy_loss_j: 0.0
        }
    );
    assert_eq!(
        on.contact_diagnostics(),
        ContactDiagnostics {
            contact_mask: 0,
            impact_speed_mps: 0.0,
            impulse_ns: 0.0,
            energy_loss_j: 0.0
        }
    );
    let mut coast = make_car();
    let mut braking = make_car();
    braking.brake_cmd = 0.5;
    for _ in 0..200 {
        assert!(gpu_road.step(std::slice::from_mut(&mut coast)));
    }
    for _ in 0..200 {
        assert!(gpu_road.step(std::slice::from_mut(&mut braking)));
    }
    assert!(
        braking.vel[0].abs() < coast.vel[0].abs(),
        "brake did not slow car"
    );
    // A ground clamp must not be reported as wall-collision energy.
    let mut falling = make_car();
    falling.pos[1] = 0.1;
    falling.vel[1] = -3.0;
    falling.config_flags |= CONTACT_DIAGNOSTICS_FLAG;
    assert!(gpu_road.step(std::slice::from_mut(&mut falling)));
    assert_eq!(falling.contact_diagnostics().contact_mask, 0);
    assert_eq!(falling.contact_diagnostics().energy_loss_j, 0.0);
    // Parked against a wall: contact is real, but impact speed remains zero.
    let mut parked = super::make_physics_state([96.0, 0.5, 96.0], [1.0, 0.0, 0.0]);
    parked.config_flags |= CONTACT_DIAGNOSTICS_FLAG;
    assert!(gpu_wall.step(std::slice::from_mut(&mut parked)));
    assert_ne!(
        parked.contact_diagnostics().contact_mask,
        0,
        "wall probe did not register"
    );
    assert_eq!(parked.contact_diagnostics().impact_speed_mps, 0.0);
    let mut impact = make_car();
    impact.config_flags |= CONTACT_DIAGNOSTICS_FLAG;
    assert!(gpu_wall.step(std::slice::from_mut(&mut impact)));
    assert_ne!(
        impact.contact_diagnostics().contact_mask,
        0,
        "wall probe did not register"
    );
    assert!(impact.contact_diagnostics().impact_speed_mps > 0.0);
    assert!(impact.contact_diagnostics().impulse_ns > 0.0);
}
