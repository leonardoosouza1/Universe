use super::*;
const POWERTRAIN: u32 = 1 << 13;

#[test]
#[ignore = "physical GPU only"]
fn brake_priority_preserves_requested_authority() {
    let f = Fixture::new();
    f.reset(0.0);
    let mut intents = vec![0.0f32; N * 96];
    // The first pair reproduces the recorded corner after its spline head.
    let throttle = [0.6178072, 0.9, 0.4, 0.0];
    let brake = [0.653184, 0.2, 0.0000000447, 0.0];
    for b in 0..N {
        intents[b * 96] = throttle[b];
        intents[b * 96 + 2] = brake[b];
    }
    f.queue
        .write_buffer(&f.action.buf_intent, 0, bytemuck::cast_slice(&intents));
    for priority in [false, true] {
        let mut flags = FLAG_LINEAR_HEAD | FLAG_ALLOW_LOW_SPEED_BRAKE | FLAG_EXCLUSIVE_PEDALS;
        if priority
            && std::env::var("UNIVERSE_CONTRACT_FAULT").as_deref() != Ok("cancel-brake-priority")
        {
            flags |= FLAG_BRAKE_PRIORITY;
        }
        let mut enc = f.device.create_command_encoder(&Default::default());
        f.action
            .encode(
                &f.device,
                &f.queue,
                &mut enc,
                f.physics.input_buffer(),
                N,
                ActionParams {
                    n: N as u32,
                    flags,
                    steering_alpha: 1.0,
                    slow_steering_authority: 0.0,
                    gearbox_mode: GEARBOX_OFF,
                    gear_cap: 6,
                    dt: DT,
                    _pad: 0.0,
                },
            )
            .unwrap();
        f.queue.submit(Some(enc.finish()));
        let rows = f.physics.read_resident_state(N).unwrap();
        if priority {
            assert!(
                (rows[0].brake_cmd - 0.653184).abs() < 1e-6,
                "brake priority must preserve requested authority"
            );
            assert!(
                (rows[1].brake_cmd - 0.2).abs() < 1e-6,
                "even stronger throttle must not cancel brake"
            );
            assert_eq!(rows[0].throttle, 0.0);
            assert_eq!(rows[1].throttle, 0.0);
            assert!(
                (rows[2].throttle - 0.4).abs() < 1e-6,
                "spline zero roundoff must not inhibit drive"
            );
            assert_eq!(rows[2].brake_cmd, 0.0);
            assert_eq!(rows[3].brake_cmd, 0.0);
        } else {
            assert!(
                (rows[0].brake_cmd - 0.0353768).abs() < 1e-6,
                "legacy arbitration must remain unchanged"
            );
            assert_eq!(rows[1].brake_cmd, 0.0);
        }
    }
    let mut enc = f.device.create_command_encoder(&Default::default());
    assert_eq!(
        f.action.encode(
            &f.device,
            &f.queue,
            &mut enc,
            f.physics.input_buffer(),
            N,
            ActionParams {
                n: N as u32,
                flags: FLAG_BRAKE_PRIORITY,
                steering_alpha: 1.0,
                slow_steering_authority: 0.0,
                gearbox_mode: GEARBOX_OFF,
                gear_cap: 6,
                dt: DT,
                _pad: 0.0
            }
        ),
        Err(
            "organism-action: brake priority requires exclusive low-speed pedals and unsigned brake"
        )
    );
}

fn cars(f: &Fixture, gear: u32, stalled: u32) -> Vec<PhysicsStateGpu> {
    f.reset(0.0);
    let mut rows = f.physics.read_resident_state(N).unwrap();
    for car in &mut rows {
        car.config_flags |= POWERTRAIN | STABLE_WHEEL_FLAGS;
        car.current_gear = gear;
        car.current_rpm = if stalled == 0 { 800.0 } else { 0.0 };
        car.is_stalled = stalled;
        car.throttle = 0.4;
    }
    rows
}
fn physical_step(f: &Fixture) {
    let mut enc = f.device.create_command_encoder(&Default::default());
    f.physics
        .encode_step(&mut enc, f.physics.input_buffer(), N)
        .unwrap();
    f.orientation
        .encode_feedback_to_input(
            &f.device,
            &mut enc,
            f.physics.output_buffer(),
            f.physics.input_buffer(),
            N,
        )
        .unwrap();
    f.queue.submit(Some(enc.finish()));
}
#[test]
#[ignore = "physical GPU only"]
fn load_stall_and_no_ghost_drive() {
    let f = Fixture::new();
    let mut rows = cars(&f, 6, 0);
    if std::env::var("UNIVERSE_CONTRACT_FAULT").as_deref() == Ok("unreachable-stall") {
        for row in &mut rows {
            row.config_flags &= !POWERTRAIN;
        }
    }
    f.physics.upload_resident_state(&rows).unwrap();
    physical_step(&f);
    let result = f.physics.read_resident_state(N).unwrap();
    for car in result {
        assert_eq!(
            car.is_stalled, 1,
            "high gear at zero coupled RPM must stall"
        );
        assert_eq!(car.drive_torque_out, 0.0, "off engine must not drive");
        assert_eq!(car.current_rpm, 0.0);
    }
}

fn snapshot_body(f: &Fixture) -> Vec<crate::gpu::organism::OrganismState> {
    let mut enc = f.device.create_command_encoder(&Default::default());
    f.body.encode_snapshot(&mut enc, N).unwrap();
    f.queue.submit(Some(enc.finish()));
    f.body.read_snapshot(&f.device, N).unwrap()
}
fn set_intents(f: &Fixture, start: f32, reverse: f32, throttle: f32) {
    let mut intents = vec![0.0f32; N * 96];
    for b in 0..N {
        intents[b * 96] = throttle;
        intents[b * 96 + 8] = reverse;
        intents[b * 96 + 9] = start;
    }
    f.queue
        .write_buffer(&f.action.buf_intent, 0, bytemuck::cast_slice(&intents));
}
fn action_step(f: &Fixture, mode: u32, post: bool) {
    let mut enc = f.device.create_command_encoder(&Default::default());
    f.action
        .encode(
            &f.device,
            &f.queue,
            &mut enc,
            f.physics.input_buffer(),
            N,
            ActionParams {
                n: N as u32,
                flags: FLAG_LINEAR_HEAD | FLAG_ALLOW_LOW_SPEED_BRAKE | FLAG_EXCLUSIVE_PEDALS,
                steering_alpha: 1.0,
                slow_steering_authority: 0.0,
                gearbox_mode: mode,
                gear_cap: 6,
                dt: DT,
                _pad: 0.0,
            },
        )
        .unwrap();
    f.body
        .encode_pre(&f.device, &mut enc, f.physics.input_buffer(), N, DT)
        .unwrap();
    f.physics
        .encode_step(&mut enc, f.physics.input_buffer(), N)
        .unwrap();
    if post {
        f.body
            .encode_post(&f.device, &mut enc, f.physics.output_buffer(), N, DT)
            .unwrap();
    }
    f.orientation
        .encode_feedback_to_input(
            &f.device,
            &mut enc,
            f.physics.output_buffer(),
            f.physics.input_buffer(),
            N,
        )
        .unwrap();
    f.queue.submit(Some(enc.finish()));
}
#[test]
#[ignore = "physical GPU only"]
fn paid_start_refusal_reset_and_reverse_contract() {
    let f = Fixture::new();
    let mut rows = cars(&f, 2, 1);
    f.physics.upload_resident_state(&rows).unwrap();
    for b in 0..N {
        f.action.reset_action_slot(&f.queue, b, 2).unwrap();
    }
    let mut body = snapshot_body(&f);
    for row in &mut body {
        row.values[0] = 10_000.0;
    }
    body[1].values[0] = 10.0; // Cannot afford one crank interval plus basal reserve.
    body[2].values[29] = 0.0; // Dead body cannot authorize a crank.
    f.body
        .restore_bytes(&f.queue, bytemuck::cast_slice(&body))
        .unwrap();
    set_intents(&f, 1.0, 0.0, 0.0);
    for _ in 0..32 {
        action_step(&f, GEARBOX_NEURAL, false);
    }
    let halfway = f.physics.read_resident_state(N).unwrap();
    assert!((halfway[0].starter_timer - 0.5).abs() < 1e-5);
    assert_eq!(halfway[0].is_stalled, 1);
    assert_eq!(
        halfway[1].starter_timer, 0.0,
        "unpaid crank must not advance"
    );
    assert_eq!(halfway[2].starter_timer, 0.0, "dead crank must not advance");
    set_intents(&f, 0.0, 0.0, 0.0);
    action_step(&f, GEARBOX_NEURAL, false);
    assert_eq!(
        f.physics.read_resident_state(N).unwrap()[0].starter_timer,
        0.0,
        "released request must reset, not pause"
    );
    set_intents(&f, 1.0, 0.0, 0.0);
    for _ in 0..63 {
        action_step(&f, GEARBOX_NEURAL, false);
    }
    assert_eq!(
        f.physics.read_resident_state(N).unwrap()[0].is_stalled,
        1,
        "no early restart"
    );
    action_step(&f, GEARBOX_NEURAL, false);
    let result = f.physics.read_resident_state(N).unwrap();
    let paid = snapshot_body(&f);
    assert_eq!(result[0].is_stalled, 0, "paid full second must restart");
    assert_eq!(
        result[0].drive_torque_out, 0.0,
        "starter is not a throttle command"
    );
    assert!(
        (10_000.0 - paid[0].values[0] - 2250.0).abs() < 1.0,
        "aborted and complete cranks each cost energy"
    );
    assert_eq!(
        paid[1].values[0], 10.0,
        "no partial payment on refused crank"
    );
    assert_eq!(result[1].is_stalled, 1);
    action_step(&f, GEARBOX_NEURAL, false);
    assert!(
        (snapshot_body(&f)[0].values[0] - paid[0].values[0]).abs() < 0.001,
        "running engine must not pay starter twice"
    );

    // Reverse starts while stopped and must stay latched during cranking even
    // when the raw reverse request goes low halfway through.
    rows = cars(&f, 0, 1);
    f.physics.upload_resident_state(&rows).unwrap();
    for b in 0..N {
        f.action.reset_action_slot(&f.queue, b, 0).unwrap();
    }
    set_intents(&f, 1.0, 1.0, 0.0);
    action_step(&f, GEARBOX_NEURAL, false);
    set_intents(&f, 1.0, 0.0, 0.0);
    for _ in 0..63 {
        action_step(&f, GEARBOX_NEURAL, false);
    }
    let restarted = f.physics.read_resident_state(N).unwrap();
    assert_eq!(
        restarted[0].current_gear, 0,
        "reverse must latch throughout crank"
    );
    assert_eq!(restarted[0].is_stalled, 0);
    set_intents(&f, 0.0, 1.0, 0.3);
    for _ in 0..256 {
        action_step(&f, GEARBOX_NEURAL, true);
    }
    let reversed = f.physics.read_resident_state(N).unwrap();
    assert!(
        reversed[0].pos[0] < rows[0].pos[0] - 1.0,
        "reverse must move backwards"
    );
    assert!(reversed[0].vel[0] < -0.5, "reverse signed velocity");
    eprintln!(
        "REVERSE displacement_m={} speed_mps={}",
        reversed[0].pos[0] - rows[0].pos[0],
        reversed[0].vel[0]
    );
}
#[test]
#[ignore = "physical GPU only"]
fn gearbox_shift_reverse_lockout_and_birth_contract() {
    let f = Fixture::new();
    let mut rows = cars(&f, 3, 0);
    for b in 0..N {
        f.action.reset_action_slot(&f.queue, b, 3).unwrap();
    }
    f.physics.upload_resident_state(&rows).unwrap();
    set_intents(&f, 0.0, 0.0, 0.3);
    action_step(&f, GEARBOX_AUTO, true);
    let down = f.physics.read_resident_state(N).unwrap();
    assert_eq!(
        down[0].current_gear, 2,
        "automatic must downshift below3.5m/s"
    );
    assert!(
        down[0].shift_timer > 0.3,
        "gear change must interrupt physical torque"
    );
    assert_eq!(down[0].drive_torque_out, 0.0);
    for _ in 0..128 {
        action_step(&f, GEARBOX_AUTO, true);
    }
    assert!(
        f.physics.read_resident_state(N).unwrap()[0].pos[0] > rows[0].pos[0] + 0.3,
        "first gear resumes propulsion after shift"
    );
    for car in &mut rows {
        car.current_gear = 2;
        car.vel[0] = 4.0;
        for wheel in car.wheels.iter_mut().step_by(2) {
            wheel[0] = 4.0 / car.wheel_radius;
        }
    }
    for b in 0..N {
        f.action.reset_action_slot(&f.queue, b, 2).unwrap();
    }
    f.physics.upload_resident_state(&rows).unwrap();
    set_intents(&f, 0.0, 1.0, 0.0);
    action_step(&f, GEARBOX_NEURAL, true);
    assert_eq!(
        f.physics.read_resident_state(N).unwrap()[0].current_gear,
        2,
        "reverse must reject moving forward"
    );
    let mut child = make_physics_state([96.0, 0.5, 96.0], [1.0, 0.0, 0.0]);
    crate::gpu::physics::inherit_resident_profile(&rows[0], &mut child);
    assert_eq!(child.current_gear, 2);
    assert_eq!(child.current_rpm, 800.0);
    assert_ne!(child.config_flags & POWERTRAIN, 0);
    child.max_torque = 0.0;
    child.throttle = 1.0;
    f.physics.upload_resident_state(&[child; N]).unwrap();
    physical_step(&f);
    assert_eq!(
        f.physics.read_resident_state(N).unwrap()[0].is_stalled,
        1,
        "zero usable capacity switches engine off"
    );
}

#[test]
#[ignore = "physical GPU only"]
fn engine_sensors_reach_native_start_head() {
    use crate::gpu::{organism_sensors::OrganismSensorsGpu, perception::PerceptionGpu};
    use crate::simulation::racing_line::RacingLine;
    let mut f = Fixture::new();
    let mut rows = cars(&f, 2, 1);
    rows[0].starter_timer = 0.5;
    rows[0].shift_timer = 0.175;
    rows[1].is_stalled = 0;
    rows[2].config_flags &= !POWERTRAIN; // Legacy meanings stay zero without a racing line.
    f.physics.upload_resident_state(&rows).unwrap();
    let mut world = crate::world::VoxelWorld::new(17, 28.0);
    world.grid.fill(crate::world::VoxelType::Road);
    let percept = PerceptionGpu::new(&f.device, &f.queue, &world, N).unwrap();
    let line = RacingLine {
        points: vec![],
        tangents: vec![],
        arc: vec![],
        total_len: 0.0,
        half_width: 14.0,
    };
    let sensors = OrganismSensorsGpu::new(
        &f.device,
        &vec![1u32; world.size * world.size],
        world.size,
        world.cell_size,
        &line,
        &[[10.0, 30.0, 60.0]; N],
        false,
        true,
    )
    .unwrap();
    let buffer = |size| {
        f.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let digest = buffer((N * 19 * 4) as u64);
    let poses = buffer((N * 32) as u64);
    let pixels = buffer((N * crate::config::iara::vision::RETINA_TOTAL * 4) as u64);
    let mut enc = f.device.create_command_encoder(&Default::default());
    sensors
        .encode(
            &f.device,
            &mut enc,
            f.physics.input_buffer(),
            &f.body.buf_state,
            &digest,
            percept.temporal_buffer(),
            &poses,
            percept.poses_buffer(),
            &f.action.buf_intent,
            N,
            DT,
        )
        .unwrap();
    let bytes = percept.encode_resident(&mut enc, &f.device, N, &pixels);
    f.queue.submit(Some(enc.finish()));
    let wave = percept.read_after_submit(&f.device, N, bytes);
    assert_eq!(wave[0][78], 127);
    assert_eq!(wave[0][79], 63);
    assert_eq!(wave[0][80], 63);
    assert_eq!(wave[1][78], 0);
    assert_eq!(wave[2][78], 0, "legacy must not change sensor semantics");
    let mut l1 = vec![0.0f32; N * 96 * 192];
    let mut l2 = vec![0.0f32; N * 48 * 96];
    for b in 0..N {
        l1[b * 96 * 192 + 48 * 192 + 78] = 2.0;
        l2[b * 48 * 96 + 9 * 96 + 48] = 2.0;
    }
    f.queue
        .write_buffer(&f.brain.buf_l1w, 0, bytemuck::cast_slice(&l1));
    f.queue
        .write_buffer(&f.brain.buf_l2w, 0, bytemuck::cast_slice(&l2));
    // Fixed weights are a wiring fixture, not a learned restart controller.
    for tick in 0..128 {
        let mut enc = f.device.create_command_encoder(&Default::default());
        f.brain.set_world_tick(tick);
        f.brain
            .bind_and_dispatch(&mut enc, N, &[], DT, Some(percept.buf_output()));
        f.queue.submit(Some(enc.finish()));
    }
    let (_, _, intent) = f.snapshot();
    eprintln!(
        "START sensory={:?} intents={:?}",
        wave.iter().map(|w| w[78]).collect::<Vec<_>>(),
        intent.iter().map(|i| i[9]).collect::<Vec<_>>()
    );
    assert!(
        intent[0][9] > 0.5,
        "engine-off sensor must reach native starter output"
    );
    // Production CfC has deterministic denoising; zero wave is not zero latent.
    // Validate the actual start threshold, not a fictitious noiseless network.
    assert!(
        intent[1][9] <= 0.5,
        "running engine control stimulus must not request start"
    );
}

#[test]
#[ignore = "physical GPU only"]
fn city_judge_allows_turns_and_charges_engine_events() {
    use crate::gpu::school::*;
    let f = Fixture::new();
    let mut rows = cars(&f, 2, 0);
    rows[0].current_gear = 6; // One demonstrated running->off transition.
    rows[2].is_stalled = 1;
    rows[2]._pad_wind = 1.0; // A prepaid crank interval.
    f.physics.upload_resident_state(&rows).unwrap();
    let geometry = LessonGeometry {
        lesson: Lesson::Delivery,
        start: [96.0, 96.0],
        forward: [1.0, 0.0],
        target: [480.0, 480.0],
        exit: [96.0, 600.0],
        exit_dir: [-1.0, 0.0],
        wrong_exit: [0.0, 0.0],
        depot_distance_m: 0.0,
    };
    let mut school = SchoolGpu::new(
        &f.device,
        N,
        LessonContract::default(),
        64,
        true,
        false,
        false,
    );
    school.cross_city = true;
    school.reset(&f.queue, &[geometry; N]).unwrap();
    let mut enc = f.device.create_command_encoder(&Default::default());
    school
        .initialize(
            &f.device,
            &mut enc,
            f.physics.input_buffer(),
            f.physics.output_buffer(),
            &f.body.buf_state,
            &f.action.buf_intent,
            N,
        )
        .unwrap();
    f.queue.submit(Some(enc.finish()));
    assert_eq!(
        snapshot_body(&f)[0].values[65],
        0.0,
        "cross-city starts empty"
    );
    for tick in 0..64 {
        let mut enc = f.device.create_command_encoder(&Default::default());
        f.physics
            .encode_step(&mut enc, f.physics.input_buffer(), N)
            .unwrap();
        school
            .encode(
                &f.device,
                &mut enc,
                f.physics.input_buffer(),
                f.physics.output_buffer(),
                &f.body.buf_state,
                &f.action.buf_intent,
                N,
                tick,
            )
            .unwrap();
        f.orientation
            .encode_feedback_to_input(
                &f.device,
                &mut enc,
                f.physics.output_buffer(),
                f.physics.input_buffer(),
                N,
            )
            .unwrap();
        f.queue.submit(Some(enc.finish()));
        if tick == 0 {
            let r = school.download(&f.device, &f.queue, N).unwrap();
            assert_eq!(
                r[3][R_STATUS], 0.0,
                "urban approach must allow lateral offset >30m before pickup"
            );
        }
    }
    let r = school.download(&f.device, &f.queue, N).unwrap();
    assert_eq!(
        r[0][R_ENGINE_OFF_EVENTS], 1.0,
        "off transition counted once"
    );
    assert_eq!(r[1][R_ENGINE_OFF_EVENTS], 0.0);
    assert!(
        (r[0][R_SCORE] - r[1][R_SCORE] + 5.0).abs() < 0.01,
        "stall penalty must reach selection score"
    );
    assert!((r[2][R_STARTER_PAID_J] - 1500.0).abs() < 0.01);
    assert!(
        (r[2][R_SCORE] - r[1][R_SCORE] + 1.0).abs() < 0.01,
        "paid cranking penalty must reach selection score"
    );
    assert_eq!(r[0][R_ACCEPTED_PICKUP_TICK], -1.0);
    assert_eq!(r[0][R_ACCEPTED_DROPOFF_TICK], -1.0);
}
