//! Real-GPU integration contracts, not a CPU oracle or a learned taxi policy.
//! Fixed weights make each boundary interpretable; all neural and physical
//! transitions run in the production shaders on a shared device.
use super::*;
use crate::gpu::{
    organism::{CONFIG_DISABLE_OLD_THERMAL, CONFIG_STATIC_REST, OrganismGpu},
    organism_action::*,
    organism_orientation::OrganismOrientationGpu,
    physics::{
        BOUNDED_TCS_FLAG, IMPLICIT_WHEEL_FLAG, PhysicsGpu, PhysicsStateGpu, STABLE_WHEEL_FLAGS,
        make_physics_state,
    },
};

const N: usize = 4;
const DT: f32 = 1.0 / 64.0;

#[path = "vision_flow_tests.rs"]
mod vision;

struct Fixture {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    brain: BrainGpu,
    action: OrganismActionGpu,
    body: OrganismGpu,
    physics: PhysicsGpu,
    orientation: OrganismOrientationGpu,
    wave: wgpu::Buffer,
    stable_wheels: bool,
}

impl Fixture {
    fn new() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("CONTRACT FAIL: real Vulkan GPU unavailable; no skip/fallback");
        let info = adapter.get_info();
        assert!(
            matches!(
                info.device_type,
                wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
            ),
            "CONTRACT FAIL: software/nonphysical adapter: {info:?}"
        );
        eprintln!("CONTRACT GPU: {} ({:?})", info.name, info.device_type);
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("native-flow-contract"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        ))
        .expect("CONTRACT FAIL: GPU device");
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let mut brain = BrainGpu::new(device.clone(), queue.clone(), info.name.clone(), N);
        // Explicit fixture configuration. This never changes training defaults.
        brain.out_gain = 1.0;
        brain.out_beta = 0.0;
        brain.norm_l2_by_std = false;
        let mut l1 = vec![0.0f32; N * 96 * 192];
        let mut l2 = vec![0.0f32; N * 48 * 96];
        for b in 0..N {
            l1[b * 96 * 192 + 96] = 1.0; // KAN-only input -> throttle neuron.
            l1[b * 96 * 192 + 192] = 1.0; // CfC visual channel0 -> steering neuron.
            l1[b * 96 * 192 + 2 * 192 + 97] = 1.0; // Stop stimulus.
            for j in 0..3 {
                l2[b * 48 * 96 + j * 96 + j] = 2.0;
            }
        }
        // Deliberate mutation for the gate's negative-control execution only.
        // It must fail at the L2 boundary, never be accepted as a passing suite.
        if std::env::var("UNIVERSE_CONTRACT_FAULT").as_deref() == Ok("disconnect-l2") {
            l2.fill(0.0);
        }
        queue.write_buffer(&brain.buf_l1w, 0, bytemuck::cast_slice(&l1));
        queue.write_buffer(&brain.buf_l2w, 0, bytemuck::cast_slice(&l2));
        queue.write_buffer(
            &brain.buf_kan,
            0,
            bytemuck::cast_slice(&vec![1.0f32; N * 192 * 16]),
        );
        brain.upload_tau(&vec![4.0f32; N * 96], N);
        let action = OrganismActionGpu::new(&device, N);
        // Identity spline over its declared [-2,2] domain, with the ordinary
        // production output clamp. We exercise the spline, not linear-head bypass.
        let knots = std::array::from_fn(|i| -2.0 + 4.0 * i as f32 / 11.0);
        action.upload_heads(&queue, &vec![[knots; 3]; N]).unwrap();
        let mut body = OrganismGpu::new(&device, &queue, N);
        body.set_config_flags(CONFIG_DISABLE_OLD_THERMAL | CONFIG_STATIC_REST)
            .unwrap();
        let physics = PhysicsGpu::new_shared(
            device.clone(),
            queue.clone(),
            info.name,
            &vec![1u8; 128 * 128],
            &vec![0.0f32; 128 * 128],
            128,
            3.0,
            N,
        )
        .unwrap();
        let orientation = OrganismOrientationGpu::new(&device);
        let wave = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("controlled-resident-sensory-wave"),
            size: (N * 48 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            device,
            queue,
            brain,
            action,
            body,
            physics,
            orientation,
            wave,
            stable_wheels: false,
        }
    }

    fn reset(&self, speed: f32) {
        self.body.reset(&self.queue, &[[0.0; 16]; N]).unwrap();
        let cars: Vec<_> = (0..N)
            .map(|b| {
                let mut car =
                    make_physics_state([96.0, 0.5, 96.0 + b as f32 * 24.0], [1.0, 0.0, 0.0]);
                car.vel = [speed, 0.0, 0.0];
                if std::env::var("UNIVERSE_CONTRACT_FAULT").as_deref() != Ok("legacy-tire-frame") {
                    car.config_flags |= crate::gpu::physics::POWER_CONSISTENT_STEERING_FLAG;
                }
                if self.stable_wheels {
                    car.config_flags |= STABLE_WHEEL_FLAGS;
                }
                for wheel in car.wheels.iter_mut().step_by(2) {
                    wheel[0] = speed / car.wheel_radius;
                }
                car
            })
            .collect();
        self.physics.upload_resident_state(&cars).unwrap();
        for b in 0..N {
            self.action.reset_action_slot(&self.queue, b, 2).unwrap();
        }
        let mut enc = self.device.create_command_encoder(&Default::default());
        // Reset all acquired neural state, including counters, between fixtures.
        for buffer in [
            &self.brain.buf_cfc,
            &self.brain.buf_fatigue,
            &self.brain.buf_neuron_mem,
        ] {
            enc.clear_buffer(buffer, 0, None);
        }
        self.queue.submit(Some(enc.finish()));
    }

    fn stimulus(&self, values: [[i8; 3]; N]) {
        let mut packed = [0u32; N * 48];
        for (b, values) in values.iter().enumerate() {
            for (channel, value) in [96usize, 0, 97].into_iter().zip(values) {
                packed[b * 48 + channel / 4] |= (*value as u8 as u32) << ((channel % 4) * 8);
            }
        }
        self.queue
            .write_buffer(&self.wave, 0, bytemuck::cast_slice(&packed));
    }

    fn step(&mut self, tick: u32) {
        self.step_from(tick, None);
    }

    fn step_from(&mut self, tick: u32, sensory: Option<&wgpu::Buffer>) {
        self.brain.set_world_tick(tick);
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.brain
            .bind_and_dispatch(&mut enc, N, &[], DT, Some(sensory.unwrap_or(&self.wave)));
        enc.copy_buffer_to_buffer(
            self.brain.buf_intent(),
            0,
            &self.action.buf_intent,
            0,
            (N * 96 * 4) as u64,
        );
        self.action
            .encode(
                &self.device,
                &self.queue,
                &mut enc,
                self.physics.input_buffer(),
                N,
                ActionParams {
                    n: N as u32,
                    flags: FLAG_SIGNED_THROTTLE
                        | FLAG_ALLOW_LOW_SPEED_BRAKE
                        | FLAG_EXCLUSIVE_PEDALS,
                    steering_alpha: 0.25,
                    slow_steering_authority: 0.5,
                    gearbox_mode: GEARBOX_AUTO,
                    gear_cap: 3,
                    dt: DT,
                    _pad: 0.0,
                },
            )
            .unwrap();
        self.body
            .encode_pre(&self.device, &mut enc, self.physics.input_buffer(), N, DT)
            .unwrap();
        self.physics
            .encode_step(&mut enc, self.physics.input_buffer(), N)
            .unwrap();
        self.orientation
            .encode_rotation_gpu(&self.device, &mut enc, self.physics.output_buffer(), N, DT)
            .unwrap();
        self.body
            .encode_post(&self.device, &mut enc, self.physics.output_buffer(), N, DT)
            .unwrap();
        self.orientation
            .encode_feedback_to_input(
                &self.device,
                &mut enc,
                self.physics.output_buffer(),
                self.physics.input_buffer(),
                N,
            )
            .unwrap();
        self.queue.submit(Some(enc.finish()));
    }

    fn snapshot(&self) -> (Vec<PhysicsStateGpu>, BrainObservation, Vec<[f32; 96]>) {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("contract-stages"),
            size: OBSERVATION_BYTES as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.brain
            .copy_observation_into(&mut enc, &staging, 0)
            .unwrap();
        self.queue.submit(Some(enc.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let bytes = staging.slice(..).get_mapped_range();
        let stages = decode_observation(&bytes).expect("CONTRACT FAIL: invalid neural stages");
        drop(bytes);
        staging.unmap();
        let ticket = self.brain.request_intent_read(N).unwrap();
        self.device.poll(wgpu::Maintain::Wait);
        let intent = self.brain.collect_intent_read(N, ticket);
        (self.physics.read_resident_state(N).unwrap(), stages, intent)
    }
}

#[test]
#[ignore = "real GPU only; run scripts/check_native_flow_gpu.py"]
fn native_brain_action_body_flow_contract() {
    run_native_flow_contract(false);
}

#[test]
#[ignore = "real GPU only; run scripts/check_native_flow_gpu.py"]
fn native_brain_action_body_flow_contract_stable_wheels() {
    run_native_flow_contract(true);
}

fn run_native_flow_contract(stable_wheels: bool) {
    let mut f = Fixture::new();
    f.stable_wheels = stable_wheels;
    f.reset(5.0);
    f.stimulus([[127, 0, 0], [0, 0, 127], [0, 127, 0], [0, -127, 0]]);
    f.brain.set_observed_being(Some(0));
    f.step(1);
    let (cars, stage, intent) = f.snapshot();
    assert!(
        (stage.l1_matmul[0] - 1.0).abs() < 1e-5,
        "wave->KAN->L1 throttle"
    );
    assert_eq!(stage.l1_to_l2[0], 96, "L1 tanh->signed i8 quantization");
    assert!(
        (stage.l2_matmul[0] - 1.511811).abs() < 1e-5,
        "L2->intent throttle: disconnected or misindexed L2"
    );
    assert!(
        (0.905..0.909).contains(&intent[0][0]),
        "L2 activation throttle"
    );
    assert!(
        (cars[0].throttle - intent[0][0]).abs() < 1e-5,
        "intent->spline->applied throttle"
    );
    assert!(
        cars[0].brake_cmd < 1e-5 && cars[1].throttle < 1e-5,
        "exclusive pedal contract"
    );
    assert!(cars[1].brake_cmd > 0.90, "brake signal lost in action/body");
    for tick in 2..=128 {
        f.step(tick);
    }
    let (cars, _, intent) = f.snapshot();
    let speed = |c: &PhysicsStateGpu| c.vel[0].hypot(c.vel[2]);
    eprintln!(
        "CONTRACT 128 ticks: speeds={:?}, headings={:?}",
        cars.iter().map(speed).collect::<Vec<_>>(),
        cars.iter().map(|c| c.fwd).collect::<Vec<_>>()
    );
    assert!(
        speed(&cars[0]) > 6.0 && cars[0].pos[0] > 106.0,
        "sustained neural throttle must propel the body"
    );
    assert!(
        speed(&cars[1]) < 1.0,
        "neural brake must stop the moving body"
    );
    assert!(
        intent[2][1] > 0.2 && intent[3][1] < -0.2,
        "signed visual CfC->L1->L2 steering lost"
    );
    assert!(
        cars[2].fwd[2] * cars[3].fwd[2] < 0.0,
        "opposite steering stimuli must produce opposite yaw"
    );
    assert!(
        speed(&cars[2]) <= 5.1 && speed(&cars[3]) <= 5.1,
        "passive steering must not accelerate a coasting body: {} / {} m/s from 5 m/s",
        speed(&cars[2]),
        speed(&cars[3])
    );
    for (i, car) in cars.iter().enumerate() {
        car.validate_contract()
            .unwrap_or_else(|e| panic!("invalid physical slot{i}: {e}"));
        assert!(car.throttle * car.brake_cmd < 1e-6, "pedal overlap slot{i}");
    }
    let mut enc = f.device.create_command_encoder(&Default::default());
    f.body.encode_snapshot(&mut enc, N).unwrap();
    f.queue.submit(Some(enc.finish()));
    let bodies = f.body.read_snapshot(&f.device, N).unwrap();
    assert!(
        bodies[0].values[0] < bodies[1].values[0],
        "powered body must consume more energy than braking control"
    );
    assert!(
        bodies[0].values[9] > 293.15,
        "powered body must generate heat"
    );

    // A previously powered actuator must cut immediately when the neural brake
    // takes over; a recovery filter must never leave residual drive torque.
    f.stimulus([[0, 0, 127]; N]);
    f.step(129);
    let (stopping, _, _) = f.snapshot();
    assert_eq!(stopping[0].throttle, 0.0);
    assert_eq!(
        stopping[0].drive_torque_out, 0.0,
        "TCS recovery retained powered torque under braking"
    );
    assert!(stopping[0].brake_cmd > 0.9);
    f.stimulus([[127, 0, 0], [0, 0, 127], [0, 127, 0], [0, -127, 0]]);

    // Repeat from the same reset: acquired state must not leak between episodes.
    f.reset(5.0);
    f.step(1);
    let (_, replay_stage, replay_intent) = f.snapshot();
    assert_eq!(
        stage, replay_stage,
        "reset must reproduce the stage trace exactly"
    );
    assert!((0.905..0.909).contains(&replay_intent[0][0]));

    // The default adaptation must be observable, rather than silently confused
    // with an action/body fault. No expected taxi competence follows from this.
    f.brain.out_beta = 0.85;
    for tick in 2..=128 {
        f.step(tick);
    }
    let (_, adapted, adapted_intent) = f.snapshot();
    assert!(
        (0.22..0.24).contains(&adapted.l2_adapted[0]),
        "output adaptation boundary changed"
    );
    assert!(
        adapted_intent[0][0] < 0.25,
        "sustained output attenuation must be exposed"
    );
    eprintln!(
        "CONTRACT PASS: native stages, signed sensory path, actions, body, physics, reset and adaptation"
    );
}

#[test]
#[ignore = "real GPU only; run scripts/check_native_flow_gpu.py"]
fn native_motor_physics_is_equivalent_in_four_headings() {
    let mut f = Fixture::new();
    f.reset(4.0);
    let headings = [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ];
    let initial: Vec<_> = headings
        .iter()
        .map(|&heading| {
            let mut car = make_physics_state([96.0, 0.5, 96.0], heading);
            car.vel = [4.0 * heading[0], 0.0, 4.0 * heading[2]];
            car.config_flags |=
                crate::gpu::physics::POWER_CONSISTENT_STEERING_FLAG | STABLE_WHEEL_FLAGS;
            match std::env::var("UNIVERSE_CONTRACT_FAULT").as_deref() {
                Ok("explicit-wheel-step") => car.config_flags &= !IMPLICIT_WHEEL_FLAG,
                Ok("instant-tcs-recovery") => car.config_flags &= !BOUNDED_TCS_FLAG,
                _ => {}
            }
            for wheel in car.wheels.iter_mut().step_by(2) {
                wheel[0] = 4.0 / car.wheel_radius;
            }
            car
        })
        .collect();
    f.physics.upload_resident_state(&initial).unwrap();
    f.stimulus([[127, 64, 0]; N]);
    f.brain.set_observed_being(Some(0));
    for tick in 1..=128 {
        f.step(tick);
        if std::env::var_os("UNIVERSE_ROTATION_TRACE").is_some()
            && [1, 2, 4, 8, 16, 32, 64, 128].contains(&tick)
        {
            let (cars, _, intents) = f.snapshot();
            eprintln!(
                "ROTATION tick={tick}: {:?}",
                cars.iter()
                    .map(|c| (
                        c.vel,
                        c.fwd,
                        c.pos,
                        c.av_y,
                        c.current_rpm,
                        c.current_gear,
                        c.drive_torque_out,
                        c.wheels[4],
                        c.throttle,
                        c.brake_cmd,
                        c.steer
                    ))
                    .collect::<Vec<_>>()
            );
            eprintln!(
                "ROTATION intent={:?}",
                intents.iter().map(|v| &v[..3]).collect::<Vec<_>>()
            );
        }
    }
    let (cars, _, intents) = f.snapshot();
    // Metamorphic comparison of GPU results; no host dynamics/physics oracle.
    let rotate = |v: [f32; 2], heading: usize| match heading {
        0 => v,
        1 => [-v[1], v[0]],
        2 => [-v[0], -v[1]],
        _ => [v[1], -v[0]],
    };
    for b in 0..N {
        let expected_pos = rotate([cars[0].pos[0] - 96.0, cars[0].pos[2] - 96.0], b);
        let expected_vel = rotate([cars[0].vel[0], cars[0].vel[2]], b);
        let expected_fwd = rotate([cars[0].fwd[0], cars[0].fwd[2]], b);
        for (actual, expected) in [
            ([cars[b].pos[0] - 96.0, cars[b].pos[2] - 96.0], expected_pos),
            ([cars[b].vel[0], cars[b].vel[2]], expected_vel),
            ([cars[b].fwd[0], cars[b].fwd[2]], expected_fwd),
        ] {
            for axis in 0..2 {
                assert!(
                    (actual[axis] - expected[axis]).abs() < 0.002,
                    "heading-dependent dynamics: slot={b}, actual={actual:?}, expected={expected:?}"
                );
            }
        }
        for channel in 0..3 {
            assert_eq!(intents[b][channel], intents[0][channel]);
        }
        assert!((cars[b].throttle - cars[0].throttle).abs() < 1e-6);
        assert!((cars[b].steer - cars[0].steer).abs() < 1e-6);
    }
    eprintln!(
        "CONTRACT PASS: native brain/action/body/physics are rotation-equivalent over 128 ticks"
    );
}

#[test]
#[ignore = "real GPU diagnostic; timestep sensitivity is not learned competence"]
fn physics_timestep_refinement_diagnostic() {
    run_physics_refinement(std::env::var_os("UNIVERSE_IMPLICIT_WHEEL_DIAGNOSTIC").is_some());
}

#[test]
#[ignore = "real GPU only; run scripts/check_native_flow_gpu.py"]
fn stable_wheels_timestep_refinement_contract() {
    let states = run_physics_refinement(true);
    // Compare actual GPU trajectories at successively refined timesteps. This
    // checks a fixed 2-second manoeuvre, not universal convergence of all scenes.
    let distance =
        |a: &PhysicsStateGpu, b: &PhysicsStateGpu| (a.pos[0] - b.pos[0]).hypot(a.pos[2] - b.pos[2]);
    let speed_delta =
        (states[0].vel[0] - states[3].vel[0]).hypot(states[0].vel[2] - states[3].vel[2]);
    assert!(
        distance(&states[0], &states[3]) < 0.1 && speed_delta < 0.2,
        "production timestep exceeds trajectory tolerance against 8 subdivisions"
    );
    assert!(
        distance(&states[2], &states[3]) < distance(&states[1], &states[2])
            && distance(&states[1], &states[2]) < distance(&states[0], &states[1]),
        "successive GPU timestep refinements must approach one trajectory"
    );
}

fn run_physics_refinement(implicit: bool) -> Vec<PhysicsStateGpu> {
    let f = Fixture::new();
    let mut states = Vec::new();
    for subdivisions in [1u32, 2, 4, 8] {
        let dt = DT / subdivisions as f32;
        let headings = [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ];
        let initial: Vec<_> = headings
            .iter()
            .map(|&heading| {
                let mut car = make_physics_state([96.0, 0.5, 96.0], heading);
                car.vel = [4.0 * heading[0], 0.0, 4.0 * heading[2]];
                car.config_flags |= crate::gpu::physics::POWER_CONSISTENT_STEERING_FLAG;
                if implicit {
                    car.config_flags |= STABLE_WHEEL_FLAGS;
                }
                car.throttle = 0.5;
                car.steer = 0.3;
                car.dt = dt;
                for wheel in car.wheels.iter_mut().step_by(2) {
                    wheel[0] = 4.0 / car.wheel_radius;
                }
                car
            })
            .collect();
        f.physics.upload_resident_state(&initial).unwrap();
        for frame in 0..128 {
            let mut enc = f.device.create_command_encoder(&Default::default());
            for _ in 0..subdivisions {
                f.physics
                    .encode_step(&mut enc, f.physics.input_buffer(), N)
                    .unwrap();
                f.orientation
                    .encode_rotation_gpu(&f.device, &mut enc, f.physics.output_buffer(), N, dt)
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
            }
            f.queue.submit(Some(enc.finish()));
            if frame == 127 {
                let cars = f.physics.read_resident_state(N).unwrap();
                eprintln!(
                    "REFINEMENT subdivisions={subdivisions}: {:?}",
                    cars.iter()
                        .map(|c| (c.pos, c.vel, c.current_rpm, c.wheels[4][0], c.wheels[4][1]))
                        .collect::<Vec<_>>()
                );
                assert!(cars.iter().all(|c| c.current_rpm.is_finite()));
                states.push(cars[0]);
            }
        }
    }
    states
}

#[test]
#[ignore = "real GPU only; run scripts/check_native_flow_gpu.py"]
fn stable_tcs_does_not_filter_a_non_slipping_pedal_request() {
    let f = Fixture::new();
    let initial: Vec<_> = (0..N)
        .map(|b| {
            let mut car = make_physics_state([96.0, 0.5, 96.0], [1.0, 0.0, 0.0]);
            car.vel = [4.0, 0.0, 0.0];
            car.throttle = 0.25;
            car.config_flags |=
                STABLE_WHEEL_FLAGS | crate::gpu::physics::POWER_CONSISTENT_STEERING_FLAG;
            if b % 2 == 1 {
                car.config_flags &= !(1 << 4);
            }
            for wheel in car.wheels.iter_mut().step_by(2) {
                wheel[0] = 4.0 / car.wheel_radius;
            }
            car
        })
        .collect();
    for _ in 0..2 {
        // Upload is an episode reset, including acquired traction-control state.
        f.physics.upload_resident_state(&initial).unwrap();
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
        let cars = f.physics.read_resident_state(N).unwrap();
        assert!(cars[1].drive_torque_out > 0.0);
        assert!(
            (cars[0].drive_torque_out - cars[1].drive_torque_out).abs() < 0.001,
            "TCS must not attenuate a non-slipping pedal request: {} vs {} Nm",
            cars[0].drive_torque_out,
            cars[1].drive_torque_out
        );
    }
}

#[test]
#[ignore = "real GPU only; run scripts/check_native_flow_gpu.py"]
fn stable_tcs_state_survives_checkpoint_and_resets_at_birth() {
    use crate::gpu::state_snapshot;
    let f = Fixture::new();
    let mut car = make_physics_state([96.0, 0.5, 96.0], [1.0, 0.0, 0.0]);
    car.vel = [4.0, 0.0, 0.0];
    car.throttle = 0.5;
    car.config_flags |= STABLE_WHEEL_FLAGS | crate::gpu::physics::POWER_CONSISTENT_STEERING_FLAG;
    // Explicit initial spinning-wheel condition engages TCS on the real GPU.
    for wheel in car.wheels.iter_mut().step_by(2) {
        wheel[0] = 30.0;
        wheel[1] = 0.5;
    }
    f.physics.upload_resident_state(&[car; N]).unwrap();
    let step = || {
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
    };
    let capture = || {
        let mut enc = f.device.create_command_encoder(&Default::default());
        let pending = state_snapshot::encode_capture(
            &f.device,
            &mut enc,
            &f.physics.resident_snapshot_buffers(),
        )
        .unwrap();
        f.queue.submit(Some(enc.finish()));
        pending.read(&f.device).unwrap()
    };
    step();
    let saved = capture();
    let mid = &saved
        .buffers
        .iter()
        .find(|b| b.name == "physics.mid")
        .unwrap()
        .bytes;
    let gain = f32::from_le_bytes(mid[120..124].try_into().unwrap());
    assert!(
        (0.09..0.99).contains(&gain),
        "fixture must actually engage TCS"
    );
    step();
    let continued = capture();
    state_snapshot::restore(&f.queue, &f.physics.resident_snapshot_buffers(), &saved).unwrap();
    step();
    assert_eq!(
        capture(),
        continued,
        "restored controller state must reproduce continuation exactly"
    );
    f.physics.upload_resident_slot(0, &car).unwrap();
    let reset = capture();
    let after = &reset
        .buffers
        .iter()
        .find(|b| b.name == "physics.mid")
        .unwrap()
        .bytes;
    assert_eq!(
        u32::from_le_bytes(after[124..128].try_into().unwrap()),
        0,
        "newborn TCS must be uninitialized"
    );
    let before = &continued
        .buffers
        .iter()
        .find(|b| b.name == "physics.mid")
        .unwrap()
        .bytes;
    assert_eq!(
        &after[128..],
        &before[128..],
        "birth reset must preserve surviving slots"
    );
}

#[test]
#[ignore = "real GPU only; run scripts/check_native_flow_gpu.py"]
fn resident_birth_retains_physics_profile_and_child_body() {
    use crate::gpu::physics::*;
    let f = Fixture::new();
    let mut parent = make_physics_state([96.0, 0.5, 96.0], [1.0, 0.0, 0.0]);
    let mask = CONTACT_DIAGNOSTICS_FLAG
        | (1 << 8)
        | STATIC_REST_FLAG
        | POWER_CONSISTENT_STEERING_FLAG
        | STABLE_WHEEL_FLAGS;
    parent.config_flags |= mask;
    parent.mass = 1600.0;
    parent.max_torque = 800.0;
    let mut child = make_physics_state([96.0, 0.5, 96.0], [1.0, 0.0, 0.0]);
    child.mass = 1200.0;
    child.max_torque = 500.0;
    child.config_flags |= 1 << 7;
    if std::env::var("UNIVERSE_CONTRACT_FAULT").as_deref() != Ok("lost-birth-profile") {
        inherit_resident_profile(&parent, &mut child);
    }
    f.physics.upload_resident_state(&[parent; N]).unwrap();
    f.physics.upload_resident_slot(1, &child).unwrap();
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
    let cars = f.physics.read_resident_state(N).unwrap();
    assert_eq!(
        cars[1].config_flags & mask,
        mask,
        "newborn lost the resident physics profile"
    );
    assert_eq!(cars[1].mass, 1200.0);
    assert_eq!(cars[1].max_torque, 500.0);
    assert_ne!(
        cars[1].config_flags & (1 << 7),
        0,
        "newborn morphology flag was erased"
    );
    assert_eq!(cars[0].mass, 1600.0, "birth altered parent body");
}

#[path = "powertrain_tests.rs"]
mod powertrain;
