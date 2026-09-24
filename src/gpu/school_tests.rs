//! Real-GPU judge fixtures for the organism school (docs/SCHOOL-LESSONS-20260922.md §6.2).
//! Synthetic pose streams are written straight into the physics/body records; the
//! judge kernel alone runs. No CPU oracle of the kernel, no software adapter.
use crate::gpu::school::*;

struct Fixture {
    device: wgpu::Device,
    queue: wgpu::Queue,
    school: SchoolGpu,
    input: wgpu::Buffer,
    output: wgpu::Buffer,
    body: wgpu::Buffer,
    intent: wgpu::Buffer,
    tick: u32,
}

const CONTRACT: LessonContract = LessonContract {
    goal_radius_m: 3.0,
    area_radius_m: 4.0,
    v_min_mps: 3.0,
    v_stop_mps: 0.3,
    hold_ticks: 128,
    corridor_half_m: 6.0,
    overshoot_along_m: 40.0,
    propelled_min_ticks: 16,
    box_half_m: 7.5,
    exit_overshoot_m: 12.0,
    align_min_cos: 0.7,
    gate_radius_m: 3.0,
    move_distance_m: 30.0,
    exit_gate_distance_m: 12.0,
};

/// Start (93,114) heading +z on the vertical street x=96; move/stop target 30 m ahead.
fn straight(lesson: Lesson) -> LessonGeometry {
    LessonGeometry {
        lesson,
        start: [93.0, 114.0],
        forward: [0.0, 1.0],
        target: [93.0, 144.0],
        exit: [93.0, 144.0],
        exit_dir: [0.0, 1.0],
        wrong_exit: [0.0, 0.0],
        depot_distance_m: 18.0,
    }
}

/// Approach (93,250.5) heading +z into junction (96,288); near exit east at (108,285)
/// heading +x, far exit west at (84,291) heading -x.
fn turn(lesson: Lesson) -> LessonGeometry {
    let near = ([108.0, 285.0], [1.0, 0.0]);
    let far = ([84.0, 291.0], [-1.0, 0.0]);
    let (exit, exit_dir, wrong) = if lesson == Lesson::TurnNear {
        (near.0, near.1, far.0)
    } else {
        (far.0, far.1, near.0)
    };
    LessonGeometry {
        lesson,
        start: [93.0, 250.5],
        forward: [0.0, 1.0],
        target: [96.0, 288.0],
        exit,
        exit_dir,
        wrong_exit: wrong,
        depot_distance_m: 192.0,
    }
}

fn service(lesson: Lesson) -> LessonGeometry {
    LessonGeometry {
        lesson,
        start: [93.0, 60.0],
        forward: [0.0, 1.0],
        target: [96.0, 96.0],
        exit: [288.0, 672.0],
        exit_dir: [0.0, 1.0],
        wrong_exit: [0.0, 0.0],
        depot_distance_m: 0.0,
    }
}

#[derive(Clone, Copy)]
struct Step {
    pos: [f32; 2],
    vel: [f32; 2],
    heading: [f32; 2],
    mask: f32,
    throttle: f32,
    alive: f32,
    integrity: f32,
    pickup: bool,
    dropoff: bool,
}

impl Step {
    fn at(pos: [f32; 2], vel: [f32; 2]) -> Self {
        let speed = (vel[0] * vel[0] + vel[1] * vel[1]).sqrt();
        let heading = if speed > 1e-6 {
            [vel[0] / speed, vel[1] / speed]
        } else {
            [0.0, 1.0]
        };
        Self {
            pos,
            vel,
            heading,
            mask: 0.0,
            throttle: 0.5,
            alive: 1.0,
            integrity: 1.0,
            pickup: false,
            dropoff: false,
        }
    }
    fn heading(mut self, h: [f32; 2]) -> Self {
        self.heading = h;
        self
    }
    fn mask(mut self, m: f32) -> Self {
        self.mask = m;
        self
    }
    fn throttle(mut self, t: f32) -> Self {
        self.throttle = t;
        self
    }
    fn dead(mut self) -> Self {
        self.alive = 0.0;
        self
    }
    fn pickup(mut self) -> Self {
        self.pickup = true;
        self
    }
    fn dropoff(mut self) -> Self {
        self.dropoff = true;
        self
    }
}

impl Fixture {
    fn new(geometry: LessonGeometry, horizon: u32) -> Self {
        Self::new_with(geometry, horizon, false)
    }

    fn new_with(geometry: LessonGeometry, horizon: u32, stop_shaping: bool) -> Self {
        Self::new_with_flags(geometry, horizon, stop_shaping, false)
    }

    fn new_with_flags(
        geometry: LessonGeometry,
        horizon: u32,
        stop_shaping: bool,
        stop_envelope: bool,
    ) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("real Vulkan GPU required");
        assert_ne!(
            adapter.get_info().device_type,
            wgpu::DeviceType::Cpu,
            "CPU fallback forbidden"
        );
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("organism-school-fixtures"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        ))
        .expect("real GPU device");
        let make = |floats: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("school-fixture"),
                size: (floats * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let input = make(PHYSICS_INPUT_STRIDE);
        let output = make(PHYSICS_OUTPUT_STRIDE);
        let body = make(BODY_STRIDE);
        let intent = make(96);
        let school =
            SchoolGpu::new(&device, 1, CONTRACT, horizon, true, stop_shaping, stop_envelope);
        school.reset(&queue, &[geometry]).unwrap();
        Self {
            device,
            queue,
            school,
            input,
            output,
            body,
            intent,
            tick: 0,
        }
    }

    fn step(&mut self, s: Step) {
        self.step_with_body(s, |_| {});
    }

    fn step_with_body(&mut self, s: Step, mutate: impl FnOnce(&mut [f32; BODY_STRIDE])) {
        let mut input = [0.0f32; PHYSICS_INPUT_STRIDE];
        input[51] = s.throttle;
        let mut output = [0.0f32; PHYSICS_OUTPUT_STRIDE];
        output[0] = s.vel[0];
        output[2] = s.vel[1];
        output[36] = s.pos[0];
        output[38] = s.pos[1];
        output[39] = s.mask;
        output[40] = s.heading[0];
        output[42] = s.heading[1];
        output[55] = if s.mask > 0.0 { 500.0 } else { 0.0 };
        let mut body = [0.0f32; BODY_STRIDE];
        body[0] = 1.0e6 - self.tick as f32 * 10.0;
        body[8] = s.integrity;
        body[29] = s.alive;
        body[98] = self.tick as f32 * 0.1;
        mutate(&mut body);
        let mut intent = [0.0f32; 96];
        intent[6] = s.pickup as u8 as f32;
        intent[7] = s.dropoff as u8 as f32;
        self.queue
            .write_buffer(&self.input, 0, bytemuck::cast_slice(&input));
        self.queue
            .write_buffer(&self.output, 0, bytemuck::cast_slice(&output));
        // Body slots the judge owns are rewritten by the kernel; everything else is fixed.
        self.queue
            .write_buffer(&self.body, 0, bytemuck::cast_slice(&body));
        self.queue
            .write_buffer(&self.intent, 0, bytemuck::cast_slice(&intent));
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.school
            .encode(
                &self.device,
                &mut encoder,
                &self.input,
                &self.output,
                &self.body,
                &self.intent,
                1,
                self.tick,
            )
            .unwrap();
        self.queue.submit(Some(encoder.finish()));
        self.tick += 1;
    }

    fn run(&mut self, steps: impl IntoIterator<Item = Step>) {
        for s in steps {
            self.step(s);
        }
    }

    fn record(&self) -> [f32; SCHOOL_STRIDE_F32] {
        self.school.download(&self.device, &self.queue, 1).unwrap()[0]
    }

    fn body(&self) -> [f32; BODY_STRIDE] {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("school-body-staging"),
            size: (BODY_STRIDE * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.body, 0, &staging, 0, (BODY_STRIDE * 4) as u64);
        self.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let mut out = [0.0f32; BODY_STRIDE];
        out.copy_from_slice(bytemuck::cast_slice(&data));
        drop(data);
        staging.unmap();
        out
    }
}

#[test]
#[ignore = "real GPU"]
fn service_lessons_initialize_task_state_and_record_transitions() {
    for lesson in [Lesson::ApproachPassenger, Lesson::Pickup, Lesson::Delivery] {
        let geometry = service(lesson);
        let mut fixture = Fixture::new(geometry, 256);
        fixture.step(Step::at(geometry.start, [0.0, 0.0]).throttle(0.0));
        let body = fixture.body();
        let record = fixture.record();
        assert_eq!(&body[70..72], &geometry.target);
        assert_eq!(&body[66..68], &geometry.exit);
        assert_eq!((body[65], body[90]), (0.0, 1.0));
        assert!(body[72] > 0.0 && body[73] > 10.0);
        assert_eq!(record[R_PASSENGER_ID], body[72]);
        assert_eq!(record[R_DELIVERY_START], 0.0);
        assert_eq!(
            (
                record[R_ACCEPTED_PICKUP_TICK],
                record[R_ACCEPTED_DROPOFF_TICK]
            ),
            (-1.0, -1.0)
        );
    }

    let geometry = service(Lesson::NavigateWithPassenger);
    let mut fixture = Fixture::new(geometry, 256);
    fixture.step(Step::at(geometry.start, [0.0, 0.0]).throttle(0.0));
    let body = fixture.body();
    assert_eq!((body[65], body[90]), (1.0, 0.0));
    assert_eq!(
        &body[80..82],
        &[
            geometry.exit[0] - geometry.start[0],
            geometry.exit[1] - geometry.start[1]
        ]
    );

    // The school observer records the native requests and the authoritative taxi
    // state transitions; it never synthesizes pickup or delivery itself.
    let geometry = service(Lesson::Delivery);
    let mut fixture = Fixture::new(geometry, 256);
    fixture.step(Step::at(geometry.start, [0.0, 0.0]).throttle(0.0));
    let passenger_id = fixture.record()[R_PASSENGER_ID];
    fixture.step_with_body(
        Step::at(geometry.target, [0.0, 0.0]).pickup().throttle(0.0),
        |body| {
            body[65] = 1.0;
            body[66] = geometry.exit[0];
            body[67] = geometry.exit[1];
            body[70] = geometry.target[0];
            body[71] = geometry.target[1];
            body[72] = passenger_id;
            body[90] = 0.0;
        },
    );
    let body = fixture.body();
    assert_eq!(
        &body[80..82],
        &[
            geometry.exit[0] - geometry.target[0],
            geometry.exit[1] - geometry.target[1]
        ]
    );
    let record = fixture.record();
    assert_eq!(record[R_PICKUP_REQUESTS], 1.0);
    assert_eq!(record[R_ACCEPTED_PICKUP_TICK], 1.0);

    fixture.step_with_body(
        Step::at(geometry.target, [0.0, 0.0])
            .dropoff()
            .throttle(0.0),
        |body| {
            body[65] = 1.0;
            body[72] = passenger_id;
            body[69] = 2.0;
        },
    );
    assert_eq!(fixture.record()[R_REFUSAL_CODE], 2.0);
    fixture.step_with_body(
        Step::at(geometry.exit, [0.0, 0.0]).dropoff().throttle(0.0),
        |body| {
            body[68] = 1.0;
            body[75] = passenger_id;
        },
    );
    let record = fixture.record();
    assert_eq!(record[R_DROPOFF_REQUESTS], 2.0);
    assert_eq!(record[R_ACCEPTED_DROPOFF_TICK], 3.0);
    assert_eq!(record[R_DELIVERY_DELTA], 1.0);
}

fn init_service(fixture: &mut Fixture, geometry: LessonGeometry) -> f32 {
    fixture.step(Step::at(geometry.start, [0.0, 0.0]).throttle(0.0));
    fixture.record()[R_PASSENGER_ID]
}

fn service_drive(
    fixture: &mut Fixture,
    geometry: LessonGeometry,
    onboard: bool,
    passenger_id: f32,
    progress: f32,
) {
    for tick in 1..=16 {
        let a = tick as f32 / 16.0;
        let pos = [
            geometry.start[0] + (geometry.target[0] - geometry.start[0]) * a,
            geometry.start[1] + (geometry.target[1] - geometry.start[1]) * a,
        ];
        fixture.step_with_body(Step::at(pos, [0.0, 4.0]), |body| {
            body[65] = onboard as u8 as f32;
            body[72] = passenger_id;
            body[127] = progress * a;
        });
    }
}

fn service_hold(
    fixture: &mut Fixture,
    position: [f32; 2],
    onboard: bool,
    passenger_id: f32,
    progress: f32,
) {
    for _ in 0..64 {
        fixture.step_with_body(Step::at(position, [0.0, 0.0]).throttle(0.0), |body| {
            body[65] = onboard as u8 as f32;
            body[72] = passenger_id;
            body[127] = progress;
        });
    }
}

#[test]
#[ignore = "real GPU"]
fn all_service_lessons_require_their_declared_positive_phase_sequence() {
    let g = service(Lesson::ApproachPassenger);
    let mut f = Fixture::new(g, 512);
    let id = init_service(&mut f, g);
    service_drive(&mut f, g, false, id, 95.0);
    service_hold(&mut f, g.target, false, id, 100.0);
    assert_eq!(status(&f.record()), (STATUS_SUCCESS, 0));

    let g = service(Lesson::Pickup);
    let mut f = Fixture::new(g, 512);
    let id = init_service(&mut f, g);
    service_drive(&mut f, g, false, id, 95.0);
    service_hold(&mut f, g.target, false, id, 100.0);
    f.step_with_body(
        Step::at(g.target, [0.0, 0.0]).pickup().throttle(0.0),
        |body| {
            body[65] = 1.0;
            body[72] = id;
            body[127] = 100.0;
        },
    );
    assert_eq!(status(&f.record()), (STATUS_SUCCESS, 0));

    let g = service(Lesson::NavigateWithPassenger);
    let mut f = Fixture::new(g, 512);
    let id = init_service(&mut f, g);
    service_drive(
        &mut f,
        LessonGeometry {
            target: g.exit,
            ..g
        },
        true,
        id,
        195.0,
    );
    service_hold(&mut f, g.exit, true, id, 200.0);
    assert_eq!(status(&f.record()), (STATUS_SUCCESS, 0));

    let g = service(Lesson::Delivery);
    let mut f = Fixture::new(g, 1024);
    let id = init_service(&mut f, g);
    service_drive(&mut f, g, false, id, 95.0);
    service_hold(&mut f, g.target, false, id, 100.0);
    f.step_with_body(
        Step::at(g.target, [0.0, 0.0]).pickup().throttle(0.0),
        |body| {
            body[65] = 1.0;
            body[72] = id;
            body[127] = 100.0;
        },
    );
    service_drive(
        &mut f,
        LessonGeometry {
            start: g.target,
            target: g.exit,
            ..g
        },
        true,
        id,
        200.0,
    );
    service_hold(&mut f, g.exit, true, id, 200.0);
    f.step_with_body(
        Step::at(g.exit, [0.0, 0.0]).dropoff().throttle(0.0),
        |body| {
            body[68] = 1.0;
            body[75] = id;
            body[127] = 200.0;
        },
    );
    let r = f.record();
    assert_eq!(status(&r), (STATUS_SUCCESS, 0));
    assert_eq!(r[R_DELIVERY_DELTA], 1.0);
    assert!(r[R_SCORE] >= 1200.0, "{r:?}");
}

#[test]
#[ignore = "real GPU"]
fn service_failures_are_explicit_and_horizon_is_not_success() {
    let g = service(Lesson::ApproachPassenger);
    let mut f = Fixture::new(g, 32);
    init_service(&mut f, g);
    f.step(Step::at(g.start, [0.0, 0.0]).mask(1.0));
    assert_eq!(status(&f.record()), (STATUS_FAILED, 1));

    let mut f = Fixture::new(g, 32);
    init_service(&mut f, g);
    f.step(Step::at(g.start, [0.0, 0.0]).dead());
    assert_eq!(status(&f.record()), (STATUS_FAILED, 2));

    let mut f = Fixture::new(g, 32);
    init_service(&mut f, g);
    f.step(Step::at([g.start[0] + 10.0, g.start[1]], [0.0, 2.0]));
    assert_eq!(status(&f.record()), (STATUS_FAILED, 3));

    let mut f = Fixture::new(g, 2);
    init_service(&mut f, g);
    f.step(Step::at(g.start, [0.0, 0.0]).throttle(0.0));
    assert_eq!(status(&f.record()), (STATUS_FAILED, 9));

    let pickup = service(Lesson::Pickup);
    let mut f = Fixture::new(pickup, 64);
    init_service(&mut f, pickup);
    f.step_with_body(Step::at(pickup.start, [0.0, 0.0]).pickup(), |body| {
        body[69] = 2.0
    });
    assert_eq!(status(&f.record()), (STATUS_FAILED, 15));

    let mut f = Fixture::new(pickup, 64);
    init_service(&mut f, pickup);
    f.step_with_body(Step::at(pickup.target, [1.0, 0.0]).pickup(), |body| {
        body[69] = 1.0
    });
    assert_eq!(status(&f.record()), (STATUS_FAILED, 10));

    let mut f = Fixture::new(g, 3);
    let id = init_service(&mut f, g);
    f.step_with_body(Step::at(g.target, [0.0, 1.0]), |body| {
        body[72] = id;
        body[127] = 100.0;
    });
    f.step_with_body(Step::at(g.target, [0.0, 1.0]), |body| {
        body[72] = id;
        body[127] = 100.0;
    });
    assert_eq!(status(&f.record()), (STATUS_FAILED, 14));

    let navigate = service(Lesson::NavigateWithPassenger);
    let mut f = Fixture::new(navigate, 64);
    init_service(&mut f, navigate);
    f.step(Step::at(navigate.exit, [0.0, 0.0]).throttle(0.0));
    assert_eq!(status(&f.record()), (STATUS_FAILED, 11));

    let mut f = Fixture::new(navigate, 64);
    let id = init_service(&mut f, navigate);
    f.step_with_body(Step::at(navigate.start, [0.0, 1.0]), |body| {
        body[65] = 1.0;
        body[72] = id + 1.0;
    });
    assert_eq!(status(&f.record()), (STATUS_FAILED, 13));

    let delivery = service(Lesson::Delivery);
    let mut f = Fixture::new(delivery, 512);
    let id = init_service(&mut f, delivery);
    service_drive(&mut f, delivery, false, id, 95.0);
    service_hold(&mut f, delivery.target, false, id, 100.0);
    f.step_with_body(Step::at(delivery.target, [0.0, 0.0]).pickup(), |body| {
        body[65] = 1.0;
        body[72] = id;
        body[127] = 100.0;
    });
    f.step_with_body(Step::at(delivery.target, [0.0, 0.0]).dropoff(), |body| {
        body[65] = 1.0;
        body[72] = id;
        body[69] = 2.0;
        body[127] = 100.0;
    });
    assert_eq!(status(&f.record()), (STATUS_FAILED, 12));

    let mut f = Fixture::new(g, 3);
    init_service(&mut f, g);
    f.step_with_body(
        Step::at([g.start[0], g.start[1] + 2.0], [0.0, 2.0]),
        |body| body[127] = 10.0,
    );
    f.step_with_body(
        Step::at([g.start[0], g.start[1] + 4.0], [0.0, 2.0]),
        |body| body[127] = 20.0,
    );
    assert_eq!(status(&f.record()), (STATUS_TRUNCATED, 0));
}

fn status(r: &[f32; SCHOOL_STRIDE_F32]) -> (u32, u32) {
    (r[R_STATUS] as u32, r[R_FAIL_CODE] as u32)
}

/// Drive straight from the start at `speed` for `ticks`, one 1/64 s step each.
fn straight_run(start: [f32; 2], dir: [f32; 2], speed: f32, ticks: usize) -> Vec<Step> {
    (0..ticks)
        .map(|t| {
            let d = speed * (t as f32 + 1.0) / 64.0;
            Step::at(
                [start[0] + dir[0] * d, start[1] + dir[1] * d],
                [dir[0] * speed, dir[1] * speed],
            )
        })
        .collect()
}

#[test]
#[ignore = "real GPU"]
fn move_success_requires_reaching_the_target_under_own_throttle() {
    let mut f = Fixture::new(straight(Lesson::Move), 1280);
    f.run(straight_run([93.0, 114.0], [0.0, 1.0], 5.0, 400));
    let r = f.record();
    assert_eq!(status(&r), (STATUS_SUCCESS, 0), "{r:?}");
    // 27 m at 5 m/s is reached on tick 345 (0-based), then the record freezes.
    assert_eq!(r[R_OUTCOME_TICK], 345.0, "{r:?}");
    assert!(r[R_PROGRESS] > 0.85 && r[R_SCORE] > 1400.0, "{r:?}");
    // Repeated credit: driving on after the outcome changes nothing.
    let frozen = r;
    f.run(straight_run([93.0, 144.0], [0.0, 1.0], 5.0, 50));
    assert_eq!(f.record(), frozen);
    // The observation still carries the sub-goal after the outcome.
    let body = f.body();
    assert_eq!((body[90], body[65]), (1.0, 0.0));
    assert!((body[70] - 93.0).abs() < 1e-4 && (body[71] - 144.0).abs() < 1e-4);
    // Consumption is frozen with the outcome: 345 ticks × 10 J judged, 449 ticks × 10 J lived.
    assert_eq!(r[R_ENERGY_OUTCOME], 1.0e6 - 3450.0, "{r:?}");
    assert!((r[R_DISTANCE_OUTCOME] - 34.5).abs() < 1e-3, "{r:?}");
    let json = record_json(&r, body[0], body[98]);
    assert_eq!(json["energy_spent_j"], 3450.0);
    assert_eq!(json["energy_episode_j"], 4490.0);
}

#[test]
#[ignore = "real GPU"]
fn move_without_throttle_is_not_a_success() {
    let mut f = Fixture::new(straight(Lesson::Move), 1280);
    f.run(
        straight_run([93.0, 114.0], [0.0, 1.0], 5.0, 400)
            .into_iter()
            .map(|s| s.throttle(0.0)),
    );
    let r = f.record();
    assert_eq!(status(&r), (STATUS_RUNNING, 0), "{r:?}");
    assert_eq!(r[R_PROPELLED], 0.0);
    assert!(r[R_PROGRESS] > 0.85);
}

#[test]
#[ignore = "real GPU"]
fn move_contact_fails_immediately_and_counts_energy() {
    let mut f = Fixture::new(straight(Lesson::Move), 1280);
    let mut steps = straight_run([93.0, 114.0], [0.0, 1.0], 5.0, 100);
    steps[60] = steps[60].mask(4.0);
    f.run(steps);
    let r = f.record();
    assert_eq!(status(&r), (STATUS_FAILED, 1), "{r:?}");
    assert_eq!(r[R_OUTCOME_TICK], 60.0);
    assert_eq!(r[R_CONTACT_TICKS], 1.0);
    assert_eq!(r[R_COLLISION_J], 500.0);
    assert!(r[R_SCORE] < 0.0, "{r:?}");
}

#[test]
#[ignore = "real GPU"]
fn move_death_corridor_overshoot_and_birth_inside_target_have_their_codes() {
    let mut f = Fixture::new(straight(Lesson::Move), 1280);
    let mut steps = straight_run([93.0, 114.0], [0.0, 1.0], 5.0, 40);
    steps[30] = steps[30].dead();
    f.run(steps);
    assert_eq!(status(&f.record()), (STATUS_FAILED, 2));

    let mut f = Fixture::new(straight(Lesson::Move), 1280);
    f.run([
        Step::at([93.0, 120.0], [0.0, 5.0]),
        Step::at([100.0, 121.0], [5.0, 0.0]),
    ]);
    assert_eq!(status(&f.record()), (STATUS_FAILED, 3));

    let mut f = Fixture::new(straight(Lesson::Move), 1280);
    f.run([
        Step::at([93.0, 120.0], [0.0, 5.0]),
        Step::at([93.0, 155.0], [0.0, 5.0]).throttle(0.0),
    ]);
    assert_eq!(status(&f.record()), (STATUS_FAILED, 5));

    let mut f = Fixture::new(straight(Lesson::Move), 1280);
    f.run([Step::at([93.0, 143.0], [0.0, 0.0])]);
    assert_eq!(status(&f.record()), (STATUS_FAILED, 4));
}

#[test]
#[ignore = "real GPU"]
fn truncation_at_horizon_keeps_progress() {
    let mut f = Fixture::new(straight(Lesson::Move), 100);
    f.run(straight_run([93.0, 114.0], [0.0, 1.0], 2.0, 120));
    let r = f.record();
    assert_eq!(status(&r), (STATUS_TRUNCATED, 0), "{r:?}");
    assert_eq!(r[R_OUTCOME_TICK], 99.0);
    assert_eq!(r[R_ELAPSED], 100.0);
    assert!(
        (r[R_PROGRESS] - (100.0 * 2.0 / 64.0) / 30.0).abs() < 0.02,
        "{r:?}"
    );
}

fn stop_approach(entry_speed: f32) -> Vec<Step> {
    // Accelerate to entry_speed over the first 20 m, then brake inside the area.
    let mut steps = Vec::new();
    let mut z = 114.0f32;
    let mut speed = 0.0f32;
    while z < 140.5 {
        speed = (speed + 0.2).min(entry_speed);
        z += speed / 64.0;
        steps.push(Step::at([93.0, z], [0.0, speed]));
    }
    steps
}

#[test]
#[ignore = "real GPU"]
fn stop_success_needs_speed_then_a_held_stop_inside_the_area() {
    let mut f = Fixture::new(straight(Lesson::Stop), 1280);
    let approach = stop_approach(6.0);
    let z = approach.last().unwrap().pos[1];
    f.run(approach);
    assert_eq!(f.record()[R_PHASE], 2.0);
    // 100 ticks stopped is not enough; leaving the area resets the dwell.
    f.run((0..100).map(|_| Step::at([93.0, z + 1.0], [0.0, 0.0]).throttle(0.0)));
    assert_eq!(f.record()[R_DWELL], 100.0);
    f.run([Step::at([93.0, z + 6.0], [0.0, 1.0]).throttle(0.0)]);
    assert_eq!(f.record()[R_DWELL], 0.0);
    f.run((0..128).map(|_| Step::at([93.0, z + 2.0], [0.0, 0.0]).throttle(0.0)));
    let r = f.record();
    assert_eq!(status(&r), (STATUS_SUCCESS, 0), "{r:?}");
    assert!(r[R_ENTRY_SPEED] > 5.5, "{r:?}");
}

#[test]
#[ignore = "real GPU"]
fn stop_crawling_in_dying_in_or_crashing_in_are_failures() {
    let mut f = Fixture::new(straight(Lesson::Stop), 1280);
    f.run(stop_approach(2.0));
    assert_eq!(status(&f.record()), (STATUS_FAILED, 8));

    let mut f = Fixture::new(straight(Lesson::Stop), 1280);
    let approach = stop_approach(6.0);
    let z = approach.last().unwrap().pos[1];
    f.run(approach);
    f.run((0..50).map(|_| Step::at([93.0, z], [0.0, 0.0]).throttle(0.0)));
    f.run([Step::at([93.0, z], [0.0, 0.0]).dead()]);
    assert_eq!(status(&f.record()), (STATUS_FAILED, 2));

    let mut f = Fixture::new(straight(Lesson::Stop), 1280);
    let approach = stop_approach(6.0);
    let z = approach.last().unwrap().pos[1];
    f.run(approach);
    f.run([Step::at([93.0, z + 0.5], [0.0, 0.0]).mask(1.0)]);
    assert_eq!(status(&f.record()), (STATUS_FAILED, 1));
}

fn turn_path(exit: [f32; 2], exit_dir: [f32; 2]) -> Vec<Step> {
    // Straight to the box edge, then a straight diagonal to 6 m before the gate,
    // then along the exit lane.
    let mut steps = straight_run([93.0, 250.5], [0.0, 1.0], 5.0, 384);
    let corner = [exit[0] - exit_dir[0] * 6.0, exit[1] - exit_dir[1] * 6.0];
    let last = steps.last().unwrap().pos;
    let d = ((corner[0] - last[0]).powi(2) + (corner[1] - last[1]).powi(2)).sqrt();
    let n = (d / (4.0 / 64.0)).ceil() as usize;
    for t in 1..=n {
        let a = t as f32 / n as f32;
        let pos = [
            last[0] + (corner[0] - last[0]) * a,
            last[1] + (corner[1] - last[1]) * a,
        ];
        let v = [
            (corner[0] - last[0]) / d * 4.0,
            (corner[1] - last[1]) / d * 4.0,
        ];
        steps.push(Step::at(pos, v));
    }
    for t in 1..=200 {
        let d = 4.0 * t as f32 / 64.0;
        steps.push(Step::at(
            [corner[0] + exit_dir[0] * d, corner[1] + exit_dir[1] * d],
            [exit_dir[0] * 4.0, exit_dir[1] * 4.0],
        ));
    }
    steps
}

#[test]
#[ignore = "real GPU"]
fn turn_near_and_far_succeed_on_their_own_gate_and_fail_on_the_other() {
    for lesson in [Lesson::TurnNear, Lesson::TurnFar] {
        let g = turn(lesson);
        let mut f = Fixture::new(g, 2000);
        f.run(turn_path(g.exit, g.exit_dir));
        let r = f.record();
        assert_eq!(status(&r), (STATUS_SUCCESS, 0), "{lesson:?} {r:?}");
        assert_eq!(r[R_PHASE], 1.0);
        assert!(r[R_PROGRESS] > 0.9, "{lesson:?} {r:?}");

        let other = turn(if lesson == Lesson::TurnNear {
            Lesson::TurnFar
        } else {
            Lesson::TurnNear
        });
        let mut f = Fixture::new(g, 2000);
        f.run(turn_path(other.exit, other.exit_dir));
        assert_eq!(status(&f.record()), (STATUS_FAILED, 7), "{lesson:?}");
    }
}

#[test]
#[ignore = "real GPU"]
fn turn_straight_through_misaligned_or_born_in_box_are_not_successes() {
    let g = turn(Lesson::TurnNear);
    let mut f = Fixture::new(g, 2000);
    f.run(straight_run([93.0, 250.5], [0.0, 1.0], 5.0, 700));
    let r = f.record();
    assert_eq!(status(&r), (STATUS_FAILED, 6), "{r:?}");

    // Reaching the gate sideways (heading +z at an east-bound gate) is not a turn.
    let mut f = Fixture::new(g, 2000);
    f.run(straight_run([93.0, 250.5], [0.0, 1.0], 5.0, 384));
    f.run([Step::at([108.0, 285.0], [0.0, 1.0]).heading([0.0, 1.0])]);
    assert_eq!(status(&f.record()), (STATUS_RUNNING, 0));
    f.run([Step::at([108.0, 285.0], [1.0, 0.0])]);
    assert_eq!(status(&f.record()), (STATUS_SUCCESS, 0));

    let mut f = Fixture::new(g, 2000);
    f.run([Step::at([96.0, 288.0], [0.0, 0.0])]);
    assert_eq!(status(&f.record()), (STATUS_FAILED, 4));
}

#[test]
#[ignore = "real GPU"]
fn turn_subgoal_switches_from_junction_to_exit_when_the_box_is_entered() {
    let g = turn(Lesson::TurnNear);
    let mut f = Fixture::new(g, 2000);
    f.run([Step::at([93.0, 260.0], [0.0, 5.0])]);
    let body = f.body();
    assert!(
        (body[70] - 96.0).abs() < 1e-4 && (body[71] - 288.0).abs() < 1e-4,
        "{:?}",
        &body[70..72]
    );
    f.run([Step::at([93.0, 281.0], [0.0, 5.0])]);
    let body = f.body();
    assert!(
        (body[70] - 108.0).abs() < 1e-4 && (body[71] - 285.0).abs() < 1e-4,
        "{:?}",
        &body[70..72]
    );
    assert!((body[80] - 15.0).abs() < 1e-4 && (body[81] - 4.0).abs() < 1e-4);
}

/// Enter at 6 m/s, creep 50 ticks at 1 m/s inside, hold 40 ticks at 0.2 m/s, leave, then time out.
fn stop_slow_then_leave(f: &mut Fixture) {
    let approach = stop_approach(6.0);
    let z = approach.last().unwrap().pos[1];
    f.run(approach);
    assert_eq!(f.record()[R_PHASE], 2.0);
    f.run((0..50).map(|_| Step::at([93.0, z + 1.0], [0.0, 1.0]).throttle(0.0)));
    // 0.2 m/s is under v_stop (0.3): the dwell counts and the slowest speed stays 0.2.
    f.run((0..40).map(|_| Step::at([93.0, z + 2.0], [0.0, 0.2]).throttle(0.0)));
    // z + 10 is 6.5 m past the area centre, outside the 4 m area: dwell must not resume.
    f.run([Step::at([93.0, z + 10.0], [0.0, 1.0]).throttle(0.0)]);
    while f.record()[R_STATUS] == STATUS_RUNNING as f32 {
        f.run([Step::at([93.0, z + 10.0], [0.0, 0.0]).throttle(0.0)]);
    }
}

#[test]
#[ignore = "real GPU"]
fn stop_shaping_records_always_and_scores_only_with_the_flag() {
    // docs/SCHOOL-STOP-SHAPING-20260922.md §4 step 2: same stream, flag off then on.
    let mut plain = Fixture::new_with(straight(Lesson::Stop), 600, false);
    stop_slow_then_leave(&mut plain);
    let r = plain.record();
    assert_eq!(status(&r), (STATUS_TRUNCATED, 0), "{r:?}");
    assert!(
        (r[R_MIN_SPEED_INSIDE] - 0.2).abs() < 0.02
            && r[R_MAX_DWELL] == 40.0
            && r[R_MAX_NEAR_DWELL] == 40.0,
        "{r:?}"
    );
    assert!(
        (r[R_SCORE] - 500.0 * r[R_PROGRESS]).abs() < 1.0,
        "plain score must stay the Stage 2 formula: {r:?}"
    );

    let mut shaped = Fixture::new_with(straight(Lesson::Stop), 600, true);
    stop_slow_then_leave(&mut shaped);
    let r = shaped.record();
    assert_eq!(status(&r), (STATUS_TRUNCATED, 0), "{r:?}");
    let expected = 500.0 * r[R_PROGRESS]
        + 300.0 * (1.0 - r[R_MIN_SPEED_INSIDE] / 3.0)
        + 200.0 * (r[R_MAX_NEAR_DWELL] / 128.0);
    assert!(
        (r[R_SCORE] - expected).abs() < 1.0,
        "shaped score {} vs expected {expected}: {r:?}",
        r[R_SCORE]
    );
    assert!(r[R_SCORE] > 500.0 * r[R_PROGRESS] + 250.0, "{r:?}");
}

#[test]
#[ignore = "real GPU"]
fn stop_shaping_tracks_near_stop_duration_without_relaxing_success() {
    let mut f = Fixture::new_with(straight(Lesson::Stop), 1280, true);
    let approach = stop_approach(6.0);
    let z = approach.last().unwrap().pos[1];
    f.run(approach);
    // 0.5 m/s is below 2*v_stop (0.6), but above the strict 0.3 m/s stop.
    f.run((0..64).map(|_| Step::at([93.0, z + 1.0], [0.0, 0.5]).throttle(0.0)));
    let r = f.record();
    assert_eq!(status(&r), (STATUS_RUNNING, 0), "{r:?}");
    assert_eq!(
        r[R_DWELL], 0.0,
        "near-stop must not count as a successful stop"
    );
    assert_eq!(r[R_MAX_DWELL], 0.0);
    assert_eq!(r[R_NEAR_DWELL], 64.0);
    assert_eq!(r[R_MAX_NEAR_DWELL], 64.0);
}

#[test]
#[ignore = "real GPU"]
fn stop_shaping_pays_the_full_bonus_on_success_and_nothing_before_the_area() {
    let mut f = Fixture::new_with(straight(Lesson::Stop), 1280, true);
    let approach = stop_approach(6.0);
    let z = approach.last().unwrap().pos[1];
    f.run(approach);
    f.run((0..128).map(|_| Step::at([93.0, z + 2.0], [0.0, 0.0]).throttle(0.0)));
    let r = f.record();
    assert_eq!(status(&r), (STATUS_SUCCESS, 0), "{r:?}");
    let expected = 1000.0 + 500.0 * r[R_PROGRESS] + 300.0 + 200.0;
    assert!(
        (r[R_SCORE] - expected).abs() < 1.0,
        "{} vs {expected}: {r:?}",
        r[R_SCORE]
    );

    // Overshooting at speed never enters the slow regime: the shaping term is zero.
    let mut f = Fixture::new_with(straight(Lesson::Stop), 1280, true);
    f.run(straight_run([93.0, 114.0], [0.0, 1.0], 8.0, 400));
    let r = f.record();
    assert_eq!(status(&r), (STATUS_FAILED, 5), "{r:?}");
    assert!(
        r[R_MIN_SPEED_INSIDE] > 7.5 && r[R_MAX_DWELL] == 0.0,
        "{r:?}"
    );
    assert!(
        (r[R_SCORE] - (500.0 * r[R_PROGRESS] - 200.0)).abs() < 1.0,
        "{r:?}"
    );
}

// ── Braking envelope (docs/SCHOOL-STOP-ENVELOPE-20260923.md §4) ──────────────────────

const A_REF: f32 = STOP_ENVELOPE_A_REF_MPS2;
const STOP_TARGET_Z: f32 = 144.0;

fn v_allow(d: f32) -> f32 {
    (0.3f32 * 0.3 + 2.0 * A_REF * (d - CONTRACT.area_radius_m).max(0.0)).sqrt()
}

/// Ramp to `cruise`, then track `v_allow(d) - 0.05` inside the braking zone (the curve
/// drops by a_ref/64 ≈ 0.03 m/s per tick near v_allow) down to a creep, and hold still
/// 2 m before the centre for `hold` ticks.
fn envelope_approach(cruise: f32, hold: usize) -> Vec<Step> {
    let mut steps = Vec::new();
    let mut z = 114.0f32;
    let mut speed = 0.0f32;
    let r_zone = CONTRACT.area_radius_m + cruise * cruise / (2.0 * A_REF);
    while z < STOP_TARGET_Z - 2.0 {
        let d = STOP_TARGET_Z - z;
        speed = if d <= r_zone {
            speed.min(v_allow(d) - 0.05).max(0.25)
        } else {
            (speed + 0.2).min(cruise)
        };
        z += speed / 64.0;
        steps.push(Step::at([93.0, z], [0.0, speed]).throttle(0.0));
    }
    let z_hold = z;
    steps.extend((0..hold).map(|_| Step::at([93.0, z_hold], [0.0, 0.0]).throttle(0.0)));
    steps
}

/// Ramp to 3 m/s and park at `stop_z` with the engine idle until the horizon.
fn park_before_goal(stop_z: f32) -> Vec<Step> {
    let mut steps = Vec::new();
    let mut z = 114.0f32;
    let mut speed = 0.0f32;
    while z < stop_z {
        speed = (speed + 0.2).min(3.0);
        z += speed / 64.0;
        steps.push(Step::at([93.0, z], [0.0, speed]).throttle(0.0));
    }
    steps.extend((0..300).map(|_| Step::at([93.0, stop_z], [0.0, 0.0]).throttle(0.0)));
    steps
}

fn old_shaping(r: &[f32; SCHOOL_STRIDE_F32]) -> f32 {
    300.0 * (1.0 - r[R_MIN_SPEED_INSIDE] / 3.0).clamp(0.0, 1.0)
        + 200.0 * (r[R_MAX_NEAR_DWELL] / 128.0).clamp(0.0, 1.0)
}

fn envelope_term(r: &[f32; SCHOOL_STRIDE_F32]) -> f32 {
    let conform = if r[R_ENV_ZONE_TICKS] > 0.0 {
        r[R_ENV_CONFORM_TICKS] / r[R_ENV_ZONE_TICKS]
    } else {
        0.0
    };
    200.0 * conform.clamp(0.0, 1.0)
        - 100.0 * (r[R_ENV_EXCESS_INTEGRAL] / STOP_ENVELOPE_EXCESS_CLAMP_M).clamp(0.0, 1.0)
}

#[test]
#[ignore = "real GPU"]
fn stop_envelope_pays_a_decelerating_approach_that_stops() {
    // §4 fixture 1: follows the curve, strict success, conformance near 1.
    let mut f = Fixture::new_with_flags(straight(Lesson::Stop), 2048, true, true);
    f.run(envelope_approach(3.0, 128));
    let r = f.record();
    assert_eq!(status(&r), (STATUS_SUCCESS, 0), "{r:?}");
    assert!(r[R_ENV_ZONE_TICKS] > 100.0, "{r:?}");
    let conform = r[R_ENV_CONFORM_TICKS] / r[R_ENV_ZONE_TICKS];
    assert!(conform >= 0.95, "conform {conform}: {r:?}");
    assert!(r[R_ENV_EXCESS_INTEGRAL] < 0.05, "{r:?}");
    assert!((r[R_ENV_ENTRY_SPEED] - 3.0).abs() < 0.2, "{r:?}");
    assert!(r[R_ENV_MIN_SPEED_APPROACH] <= 0.3 && r[R_ENV_DIST_AT_MIN_SPEED] <= 4.0, "{r:?}");
    let expected = 1000.0 + 500.0 * r[R_PROGRESS] + old_shaping(&r) + envelope_term(&r);
    assert!((r[R_SCORE] - expected).abs() < 1.0, "{} vs {expected}: {r:?}", r[R_SCORE]);
    assert!(envelope_term(&r) > 190.0, "{r:?}");
}

#[test]
#[ignore = "real GPU"]
fn stop_envelope_never_pays_parking_outside_the_zone_more_than_entering() {
    // §4 fixture 2: braking far and parking earns nothing from the envelope; parking in
    // the zone but outside the area earns conformance yet stays below a creeping entry.
    let mut far = Fixture::new_with_flags(straight(Lesson::Stop), 1280, true, true);
    far.run(park_before_goal(STOP_TARGET_Z - 8.0));
    while far.record()[R_STATUS] == STATUS_RUNNING as f32 {
        far.run([Step::at([93.0, STOP_TARGET_Z - 8.0], [0.0, 0.0]).throttle(0.0)]);
    }
    let r_far = far.record();
    assert_eq!(status(&r_far), (STATUS_TRUNCATED, 0), "{r_far:?}");
    assert_eq!(r_far[R_ENV_ZONE_TICKS], 0.0, "{r_far:?}");
    assert!((r_far[R_SCORE] - 500.0 * r_far[R_PROGRESS]).abs() < 1.0, "{r_far:?}");

    let mut near = Fixture::new_with_flags(straight(Lesson::Stop), 1280, true, true);
    near.run(park_before_goal(STOP_TARGET_Z - 5.0));
    while near.record()[R_STATUS] == STATUS_RUNNING as f32 {
        near.run([Step::at([93.0, STOP_TARGET_Z - 5.0], [0.0, 0.0]).throttle(0.0)]);
    }
    let r_near = near.record();
    assert_eq!(status(&r_near), (STATUS_TRUNCATED, 0), "{r_near:?}");
    assert!(r_near[R_ENV_ZONE_TICKS] > 250.0 && r_near[R_DWELL] == 0.0, "{r_near:?}");
    assert!(envelope_term(&r_near) > 150.0, "{r_near:?}");

    let mut creep = Fixture::new_with_flags(straight(Lesson::Stop), 1280, true, true);
    let approach = stop_approach(6.0);
    let z = approach.last().unwrap().pos[1];
    creep.run(approach);
    creep.run((0..64).map(|_| Step::at([93.0, z + 1.0], [0.0, 0.5]).throttle(0.0)));
    while creep.record()[R_STATUS] == STATUS_RUNNING as f32 {
        creep.run([Step::at([93.0, z + 1.0], [0.0, 0.5]).throttle(0.0)]);
    }
    let r_creep = creep.record();
    assert!(r_creep[R_SCORE] > r_near[R_SCORE] + 200.0, "{r_creep:?} vs {r_near:?}");
    assert!(r_near[R_SCORE] > r_far[R_SCORE], "{r_near:?} vs {r_far:?}");
}

#[test]
#[ignore = "real GPU"]
fn stop_envelope_charges_a_fast_overshoot_and_keeps_progress() {
    // §4 fixture 3.
    let mut f = Fixture::new_with_flags(straight(Lesson::Stop), 1280, true, true);
    f.run(straight_run([93.0, 114.0], [0.0, 1.0], 8.0, 400));
    let r = f.record();
    assert_eq!(status(&r), (STATUS_FAILED, 5), "{r:?}");
    assert_eq!(r[R_ENV_CONFORM_TICKS], 0.0, "{r:?}");
    assert_eq!(r[R_ENV_EXCESS_INTEGRAL], STOP_ENVELOPE_EXCESS_CLAMP_M, "{r:?}");
    assert!((r[R_SCORE] - (500.0 * r[R_PROGRESS] - 200.0 - 100.0)).abs() < 1.0, "{r:?}");
}

#[test]
#[ignore = "real GPU"]
fn stop_envelope_is_blind_to_the_pedal_command() {
    // §4 fixture 4: the same pose stream with the throttle held gives the same record,
    // except the propelled-tick counter which reads the applied throttle by design.
    let mut idle = Fixture::new_with_flags(straight(Lesson::Stop), 2048, true, true);
    idle.run(envelope_approach(3.0, 128));
    let mut pressed = Fixture::new_with_flags(straight(Lesson::Stop), 2048, true, true);
    pressed.run(envelope_approach(3.0, 128).into_iter().map(|s| s.throttle(1.0)));
    let (a, b) = (idle.record(), pressed.record());
    for slot in 0..SCHOOL_STRIDE_F32 {
        if slot == R_PROPELLED {
            continue;
        }
        assert_eq!(a[slot].to_bits(), b[slot].to_bits(), "slot {slot}: {a:?} vs {b:?}");
    }
    assert_eq!(status(&a), (STATUS_SUCCESS, 0));
}

#[test]
#[ignore = "real GPU"]
fn stop_envelope_off_keeps_the_historical_score_and_still_records() {
    // §4 fixture 5: flag off reproduces the v2 formulas; slots 48..53 are still populated.
    let stream = envelope_approach(3.0, 128);
    let mut shaped = Fixture::new_with_flags(straight(Lesson::Stop), 2048, true, false);
    shaped.run(stream.clone());
    let r = shaped.record();
    assert_eq!(status(&r), (STATUS_SUCCESS, 0), "{r:?}");
    let expected = 1000.0 + 500.0 * r[R_PROGRESS] + old_shaping(&r);
    assert!((r[R_SCORE] - expected).abs() < 1.0, "{} vs {expected}: {r:?}", r[R_SCORE]);
    assert!(r[R_ENV_ZONE_TICKS] > 100.0 && r[R_ENV_CONFORM_TICKS] > 100.0, "{r:?}");

    let mut plain = Fixture::new_with_flags(straight(Lesson::Stop), 2048, false, false);
    plain.run(stream);
    let r = plain.record();
    let expected = 1000.0 + 500.0 * r[R_PROGRESS];
    assert!((r[R_SCORE] - expected).abs() < 1.0, "{} vs {expected}: {r:?}", r[R_SCORE]);
}
