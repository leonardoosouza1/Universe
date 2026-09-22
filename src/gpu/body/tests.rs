use super::*;
use crate::gpu::{
    evolution::{evolve, EvoParams},
    physics::make_physics_state,
};

#[test]
fn neutral_body_preserves_all_physics_bytes() {
    let original = make_physics_state([5.0, 1.0, 6.0], [1.0, 0.0, 0.0]);
    let mut expressed = original;
    BodyGenes::default().express(&mut expressed);
    assert_eq!(
        bytemuck::bytes_of(&original),
        bytemuck::bytes_of(&expressed)
    );
    assert_eq!(fuel_load_scale(&expressed), 1.0);
}

#[test]
fn capacity_has_mass_and_fuel_costs_without_changing_actions() {
    let base = make_physics_state([0.0; 3], [1.0, 0.0, 0.0]);
    for engine in [-1.0, 0.0, 1.0] {
        for brakes in [-1.0, 0.0, 1.0] {
            let mut p = base;
            BodyGenes([engine, brakes]).express(&mut p);
            assert!((1282.4..=1417.6).contains(&p.mass));
            assert!((465.0..=775.0).contains(&p.max_torque));
            assert!((3000.0..=5000.0).contains(&p.brake_torque_max));
            assert!((p.susp_stiffness / p.mass - base.susp_stiffness / base.mass).abs() < 1e-5);
            assert!((p.susp_damping / p.mass - base.susp_damping / base.mass).abs() < 1e-6);
            assert!((p.wheels[0][3] * 4.0 - p.mass * 9.81).abs() < 0.01);
            assert_eq!((p.throttle, p.brake_cmd, p.steer), (0.0, 0.0, 0.0));
            assert!((fuel_load_scale(&p) - (1.0 + 0.25 * f64::from(engine))).abs() < 1e-7);
        }
    }
    assert!(!BodyGenes([f32::NAN, 0.0]).is_valid());
    assert!(!BodyGenes([0.0, f32::INFINITY]).is_valid());
    assert!(!BodyGenes([1.01, 0.0]).is_valid());
}

#[test]
fn body_follows_selection_and_mutation_preserves_elites_and_brain() {
    let mut parents: Vec<_> = (0..4).map(Genome::random).collect();
    for (i, parent) in parents.iter_mut().enumerate() {
        parent.body = BodyGenes([i as f32 / 4.0, -0.5]);
    }
    let params = EvoParams {
        n_elite: 1,
        ..Default::default()
    };
    let selected = evolve(&parents, &[0.0, 2.0, 1.0, 10.0], &params, 3, 7);
    assert_eq!(selected[0].body, parents[3].body);
    assert!(selected
        .iter()
        .all(|g| parents.iter().any(|p| p.body == g.body)));
    let mut a = selected.clone();
    let mut b = selected.clone();
    mutate_population(&mut a, 1, 3, 7);
    mutate_population(&mut b, 1, 3, 7);
    assert_eq!(a[0].body, selected[0].body);
    for i in 1..a.len() {
        assert_eq!(a[i].body, b[i].body);
        assert_ne!(a[i].body, selected[i].body);
        assert_eq!(a[i].l1, selected[i].l1);
        assert_eq!(a[i].l2, selected[i].l2);
        assert_eq!(a[i].tau, selected[i].tau);
    }
    mutate_population(&mut b, 1, 3, 8);
    assert_ne!(a[1].body, b[1].body);
    for generation in 0..1000 {
        mutate_population(&mut a, 1, generation, 7);
    }
    assert!(a.iter().all(|g| g.body.is_valid()));
}

#[test]
#[ignore = "requires a Vulkan GPU; run explicitly with the driver override"]
fn body_capacities_reach_gpu_physics() {
    use crate::gpu::physics::{pack_config_flags, PhysicsGpu};
    let gpu = PhysicsGpu::new(&vec![1; 64 * 64], &vec![0.0; 64 * 64], 64, 3.0, 4)
        .expect("a real GPU is required");
    let base = make_physics_state([80.0, 0.5, 80.0], [1.0, 0.0, 0.0]);
    let mut states = [base; 4];
    BodyGenes([-1.0, 0.0]).express(&mut states[0]);
    BodyGenes([1.0, 0.0]).express(&mut states[1]);
    BodyGenes([0.0, -1.0]).express(&mut states[2]);
    BodyGenes([0.0, 1.0]).express(&mut states[3]);
    for (i, p) in states.iter_mut().enumerate() {
        p.config_flags = pack_config_flags(false, false, false, false, false, false);
        p.vel = [10.0, 0.0, 0.0];
        p.current_rpm = 4000.0;
        p.current_gear = 3;
        for wheel in 0..4 {
            p.wheels[2 * wheel][0] = 10.0 / p.wheel_radius;
        }
        p.throttle = if i < 2 { 0.5 } else { 0.0 };
        p.brake_cmd = if i >= 2 { 0.2 } else { 0.0 };
    }
    assert!(gpu.step(&mut states));
    assert!(
        states[1].drive_torque_out > states[0].drive_torque_out,
        "engine genes must affect actual GPU torque"
    );
    // Wheel integration sees braking before chassis velocity sees changed tyre slip.
    assert!(
        states[3].wheels[0][0] < states[2].wheels[0][0],
        "brake genes must affect actual GPU wheel deceleration"
    );
    assert_eq!(states[1].max_torque, 775.0);
    assert_eq!(states[3].brake_torque_max, 5000.0);
    println!(
        "GPU={} engine torques={}/{} wheel speeds={}/{}",
        gpu.adapter,
        states[0].drive_torque_out,
        states[1].drive_torque_out,
        states[2].wheels[0][0],
        states[3].wheels[0][0]
    );
}

#[test]
fn body_genome_envelope_roundtrips_fly_and_rejects_corruption() {
    use crate::gpu::genome_io::{load_genome, save_genome};
    let path = std::env::temp_dir().join(format!("universe_body_v3_{}.bin", std::process::id()));
    let path = path.to_str().unwrap();
    let mut genome = Genome::random(43);
    genome.fly = (0..237).map(|i| i as f32 / 237.0).collect();
    save_genome(&genome, path).unwrap();
    let neutral = std::fs::read(path).unwrap();
    assert_eq!(&neutral[4..8], &2u32.to_le_bytes());
    assert_eq!(load_genome(path).unwrap().body, BodyGenes::default());
    genome.body = BodyGenes([0.3, -0.7]);
    save_genome(&genome, path).unwrap();
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[4..8], &3u32.to_le_bytes());
    assert_eq!(
        &bytes[20..],
        &neutral[8..],
        "v2 payload must remain unchanged"
    );
    let decoded = load_genome(path).unwrap();
    assert_eq!(decoded.body, genome.body);
    assert_eq!(decoded.fly, genome.fly);
    save_genome(&decoded, path).unwrap();
    assert_eq!(
        std::fs::read(path).unwrap(),
        bytes,
        "all genes must round trip"
    );
    for cut in [4, 7, 8, 12, 16, 19, bytes.len() - 1, bytes.len() - 4] {
        std::fs::write(path, &bytes[..cut]).unwrap();
        assert!(load_genome(path).is_none(), "accepted truncation at {cut}");
    }
    for value in [f32::NAN, f32::INFINITY, 1.01] {
        let mut bad = bytes.clone();
        bad[12..16].copy_from_slice(&value.to_le_bytes());
        std::fs::write(path, bad).unwrap();
        assert!(load_genome(path).is_none());
    }
    let mut bad = bytes.clone();
    bad[4..8].copy_from_slice(&99u32.to_le_bytes());
    std::fs::write(path, bad).unwrap();
    assert!(load_genome(path).is_none());
    let mut short_payload = bytes[..100].to_vec();
    short_payload[8..12].copy_from_slice(&80u32.to_le_bytes());
    std::fs::write(path, short_payload).unwrap();
    assert!(load_genome(path).is_none());
    genome.body = BodyGenes([f32::NAN, 0.0]);
    assert!(save_genome(&genome, path).is_err());
    std::fs::remove_file(path).unwrap();
}
