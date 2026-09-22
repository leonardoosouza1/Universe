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
    let (exit, exit_dir, wrong) = if lesson == Lesson::TurnNear { (near.0, near.1, far.0) } else { (far.0, far.1, near.0) };
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

#[derive(Clone, Copy)]
struct Step {
    pos: [f32; 2],
    vel: [f32; 2],
    heading: [f32; 2],
    mask: f32,
    throttle: f32,
    alive: f32,
    integrity: f32,
}

impl Step {
    fn at(pos: [f32; 2], vel: [f32; 2]) -> Self {
        let speed = (vel[0] * vel[0] + vel[1] * vel[1]).sqrt();
        let heading = if speed > 1e-6 { [vel[0] / speed, vel[1] / speed] } else { [0.0, 1.0] };
        Self { pos, vel, heading, mask: 0.0, throttle: 0.5, alive: 1.0, integrity: 1.0 }
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
}

impl Fixture {
    fn new(geometry: LessonGeometry, horizon: u32) -> Self {
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
        assert_ne!(adapter.get_info().device_type, wgpu::DeviceType::Cpu, "CPU fallback forbidden");
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
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let input = make(PHYSICS_INPUT_STRIDE);
        let output = make(PHYSICS_OUTPUT_STRIDE);
        let body = make(BODY_STRIDE);
        let school = SchoolGpu::new(&device, 1, CONTRACT, horizon, true);
        school.reset(&queue, &[geometry]).unwrap();
        Self { device, queue, school, input, output, body, tick: 0 }
    }

    fn step(&mut self, s: Step) {
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
        self.queue.write_buffer(&self.input, 0, bytemuck::cast_slice(&input));
        self.queue.write_buffer(&self.output, 0, bytemuck::cast_slice(&output));
        // Body slots the judge owns are rewritten by the kernel; everything else is fixed.
        self.queue.write_buffer(&self.body, 0, bytemuck::cast_slice(&body));
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.school
            .encode(&self.device, &mut encoder, &self.input, &self.output, &self.body, 1, self.tick)
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

fn status(r: &[f32; SCHOOL_STRIDE_F32]) -> (u32, u32) {
    (r[R_STATUS] as u32, r[R_FAIL_CODE] as u32)
}

/// Drive straight from the start at `speed` for `ticks`, one 1/64 s step each.
fn straight_run(start: [f32; 2], dir: [f32; 2], speed: f32, ticks: usize) -> Vec<Step> {
    (0..ticks)
        .map(|t| {
            let d = speed * (t as f32 + 1.0) / 64.0;
            Step::at([start[0] + dir[0] * d, start[1] + dir[1] * d], [dir[0] * speed, dir[1] * speed])
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
    f.run(straight_run([93.0, 114.0], [0.0, 1.0], 5.0, 400).into_iter().map(|s| s.throttle(0.0)));
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
    f.run([Step::at([93.0, 120.0], [0.0, 5.0]), Step::at([100.0, 121.0], [5.0, 0.0])]);
    assert_eq!(status(&f.record()), (STATUS_FAILED, 3));

    let mut f = Fixture::new(straight(Lesson::Move), 1280);
    f.run([Step::at([93.0, 120.0], [0.0, 5.0]), Step::at([93.0, 155.0], [0.0, 5.0]).throttle(0.0)]);
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
    assert!((r[R_PROGRESS] - (100.0 * 2.0 / 64.0) / 30.0).abs() < 0.02, "{r:?}");
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
        let pos = [last[0] + (corner[0] - last[0]) * a, last[1] + (corner[1] - last[1]) * a];
        let v = [(corner[0] - last[0]) / d * 4.0, (corner[1] - last[1]) / d * 4.0];
        steps.push(Step::at(pos, v));
    }
    for t in 1..=200 {
        let d = 4.0 * t as f32 / 64.0;
        steps.push(Step::at([corner[0] + exit_dir[0] * d, corner[1] + exit_dir[1] * d], [exit_dir[0] * 4.0, exit_dir[1] * 4.0]));
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

        let other = turn(if lesson == Lesson::TurnNear { Lesson::TurnFar } else { Lesson::TurnNear });
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
    assert!((body[70] - 96.0).abs() < 1e-4 && (body[71] - 288.0).abs() < 1e-4, "{:?}", &body[70..72]);
    f.run([Step::at([93.0, 281.0], [0.0, 5.0])]);
    let body = f.body();
    assert!((body[70] - 108.0).abs() < 1e-4 && (body[71] - 285.0).abs() < 1e-4, "{:?}", &body[70..72]);
    assert!((body[80] - 15.0).abs() < 1e-4 && (body[81] - 4.0).abs() < 1e-4);
}
