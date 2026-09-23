//! GPU judge for the organism school (Stage 2, `docs/SCHOOL-LESSONS-20260922.md`).
//!
//! One 32-float record per being, resident on the device, written only by
//! `school.wgsl::judge` after the taxi post pass. The host declares the lesson
//! geometry once per generation (`reset`) and reads the records back only at the
//! generation boundary (`download`). No controller, pilot or steering assistance
//! lives here: the judge publishes the lesson sub-goal to the same body slots the
//! taxi task uses (65, 70, 71, 80, 81, 90) and nothing else the brain can read.
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

pub const SCHOOL_ID: &str = "organism-school-v1";
pub const SCHOOL_SCHEMA_VERSION: u32 = 2;
pub const SCHOOL_STRIDE_F32: usize = 48;
pub const PHYSICS_INPUT_STRIDE: usize = 92;
pub const PHYSICS_OUTPUT_STRIDE: usize = 56;
pub const BODY_STRIDE: usize = 128;

// Record slots; the shader mirrors these constants.
pub const R_LESSON: usize = 0;
pub const R_PHASE: usize = 1;
pub const R_STATUS: usize = 2;
pub const R_FAIL_CODE: usize = 3;
pub const R_START: usize = 4;
pub const R_FORWARD: usize = 6;
pub const R_TARGET: usize = 8;
pub const R_EXIT: usize = 10;
pub const R_EXIT_DIR: usize = 12;
pub const R_BEST_REMAINING: usize = 14;
pub const R_DWELL: usize = 15;
pub const R_PROPELLED: usize = 16;
pub const R_ELAPSED: usize = 17;
pub const R_MAX_SPEED: usize = 18;
pub const R_CONTACT_TICKS: usize = 19;
pub const R_COLLISION_J: usize = 20;
pub const R_SCORE: usize = 21;
pub const R_OUTCOME_TICK: usize = 22;
pub const R_MAX_LAT: usize = 23;
pub const R_MAX_ALONG: usize = 24;
pub const R_ENERGY_START: usize = 25;
pub const R_DISTANCE_START: usize = 26;
pub const R_INITIAL_REMAINING: usize = 27;
pub const R_PROGRESS: usize = 28;
pub const R_ENTRY_SPEED: usize = 29;
pub const R_WRONG_EXIT: usize = 30;
/// Energy (J) and odometer (m) captured by `close`, so the judged span is separable from the episode.
pub const R_ENERGY_OUTCOME: usize = 32;
pub const R_DISTANCE_OUTCOME: usize = 33;
/// Stop shaping (docs/SCHOOL-STOP-SHAPING-20260922.md): slowest speed inside the stop area
/// (1e9 until it is entered), strict stop dwell and near-stop dwell; recorded always,
/// scored only with `stop_shaping`.
pub const R_MIN_SPEED_INSIDE: usize = 34;
pub const R_MAX_DWELL: usize = 35;
pub const R_NEAR_DWELL: usize = 36;
pub const R_MAX_NEAR_DWELL: usize = 37;
/// Service-event evidence appended after the frozen v1 range 0..39.
pub const R_ACCEPTED_PICKUP_TICK: usize = 40;
pub const R_ACCEPTED_DROPOFF_TICK: usize = 41;
pub const R_PICKUP_REQUESTS: usize = 42;
pub const R_DROPOFF_REQUESTS: usize = 43;
pub const R_REFUSAL_CODE: usize = 44;
pub const R_PASSENGER_ID: usize = 45;
pub const R_DELIVERY_START: usize = 46;
pub const R_DELIVERY_DELTA: usize = 47;

pub const STATUS_RUNNING: u32 = 0;
pub const STATUS_SUCCESS: u32 = 1;
pub const STATUS_FAILED: u32 = 2;
pub const STATUS_TRUNCATED: u32 = 3;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lesson {
    Move = 1,
    Stop = 2,
    TurnNear = 3,
    TurnFar = 4,
    ApproachPassenger = 5,
    Pickup = 6,
    NavigateWithPassenger = 7,
    Delivery = 8,
}

impl Lesson {
    /// Curriculum order: motor control first, then canonical taxi service phases.
    pub const ALL: [Lesson; 8] = [
        Lesson::Move,
        Lesson::Stop,
        Lesson::TurnNear,
        Lesson::TurnFar,
        Lesson::ApproachPassenger,
        Lesson::Pickup,
        Lesson::NavigateWithPassenger,
        Lesson::Delivery,
    ];

    pub fn parse(flag: &str) -> Option<Self> {
        match flag {
            "move" => Some(Self::Move),
            "stop" => Some(Self::Stop),
            "turn_near" => Some(Self::TurnNear),
            "turn_far" => Some(Self::TurnFar),
            "approach_passenger" => Some(Self::ApproachPassenger),
            "pickup" => Some(Self::Pickup),
            "navigate_with_passenger" => Some(Self::NavigateWithPassenger),
            "delivery" => Some(Self::Delivery),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Move => "move",
            Self::Stop => "stop",
            Self::TurnNear => "turn_near",
            Self::TurnFar => "turn_far",
            Self::ApproachPassenger => "approach_passenger",
            Self::Pickup => "pickup",
            Self::NavigateWithPassenger => "navigate_with_passenger",
            Self::Delivery => "delivery",
        }
    }
    pub fn id(self) -> u32 {
        self as u32
    }
    pub fn is_turn(self) -> bool {
        matches!(self, Self::TurnNear | Self::TurnFar)
    }
    pub fn is_service(self) -> bool {
        matches!(
            self,
            Self::ApproachPassenger | Self::Pickup | Self::NavigateWithPassenger | Self::Delivery
        )
    }
    pub fn starts_onboard(self) -> bool {
        self == Self::NavigateWithPassenger
    }
    /// Distance from the start to the junction centre along the lane flow.
    pub fn start_before_junction_m(self) -> f32 {
        if self.is_turn() { 37.5 } else { 60.0 }
    }
}

pub fn status_name(status: u32) -> &'static str {
    match status {
        STATUS_RUNNING => "running",
        STATUS_SUCCESS => "success",
        STATUS_FAILED => "failed",
        STATUS_TRUNCATED => "truncated",
        _ => "unknown",
    }
}

pub fn fail_code_name(code: u32) -> &'static str {
    match code {
        0 => "none",
        1 => "contact",
        2 => "death",
        3 => "corridor",
        4 => "born_inside_target",
        5 => "overshoot",
        6 => "drove_through_junction",
        7 => "wrong_exit",
        8 => "no_approach_speed",
        9 => "no_propelled_movement",
        10 => "premature_or_moving_pickup",
        11 => "passenger_lost_before_delivery",
        12 => "dropoff_outside_destination",
        13 => "wrong_passenger_id",
        14 => "service_goal_not_stopped",
        15 => "pickup_outside_passenger_radius",
        _ => "unknown",
    }
}

/// Declared tolerances, exported verbatim into the run manifest.
#[derive(Clone, Copy, Debug)]
pub struct LessonContract {
    pub goal_radius_m: f32,
    pub area_radius_m: f32,
    pub v_min_mps: f32,
    pub v_stop_mps: f32,
    pub hold_ticks: u32,
    pub corridor_half_m: f32,
    pub overshoot_along_m: f32,
    pub propelled_min_ticks: u32,
    pub box_half_m: f32,
    pub exit_overshoot_m: f32,
    pub align_min_cos: f32,
    pub gate_radius_m: f32,
    pub move_distance_m: f32,
    pub exit_gate_distance_m: f32,
}

impl Default for LessonContract {
    fn default() -> Self {
        Self {
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
        }
    }
}

impl LessonContract {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "goal_radius_m": self.goal_radius_m, "area_radius_m": self.area_radius_m,
            "v_min_mps": self.v_min_mps, "v_stop_mps": self.v_stop_mps, "hold_ticks": self.hold_ticks,
            "corridor_half_m": self.corridor_half_m, "overshoot_along_m": self.overshoot_along_m,
            "propelled_min_ticks": self.propelled_min_ticks, "box_half_m": self.box_half_m,
            "exit_overshoot_m": self.exit_overshoot_m, "align_min_cos": self.align_min_cos,
            "gate_radius_m": self.gate_radius_m, "move_distance_m": self.move_distance_m,
            "exit_gate_distance_m": self.exit_gate_distance_m,
        })
    }
}

/// Static geometry of one being's lesson, in world metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LessonGeometry {
    pub lesson: Lesson,
    pub start: [f32; 2],
    pub forward: [f32; 2],
    /// Move target, stop area centre, or junction centre for a turn.
    pub target: [f32; 2],
    /// Exit gate for a turn; equals `target` otherwise.
    pub exit: [f32; 2],
    /// Outbound lane direction at the exit gate; equals `forward` otherwise.
    pub exit_dir: [f32; 2],
    /// The other outbound gate of the same junction; `[0,0]` when not a turn.
    pub wrong_exit: [f32; 2],
    pub depot_distance_m: f32,
}

impl LessonGeometry {
    pub fn record(&self) -> [f32; SCHOOL_STRIDE_F32] {
        let mut r = [0.0f32; SCHOOL_STRIDE_F32];
        r[R_LESSON] = self.lesson.id() as f32;
        r[R_START] = self.start[0];
        r[R_START + 1] = self.start[1];
        r[R_FORWARD] = self.forward[0];
        r[R_FORWARD + 1] = self.forward[1];
        r[R_TARGET] = self.target[0];
        r[R_TARGET + 1] = self.target[1];
        r[R_EXIT] = self.exit[0];
        r[R_EXIT + 1] = self.exit[1];
        r[R_EXIT_DIR] = self.exit_dir[0];
        r[R_EXIT_DIR + 1] = self.exit_dir[1];
        r[R_WRONG_EXIT] = self.wrong_exit[0];
        r[R_WRONG_EXIT + 1] = self.wrong_exit[1];
        r[R_ACCEPTED_PICKUP_TICK] = -1.0;
        r[R_ACCEPTED_DROPOFF_TICK] = -1.0;
        r
    }
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "lesson": self.lesson.name(), "start": self.start, "forward": self.forward,
            "target": self.target, "exit": self.exit, "exit_dir": self.exit_dir,
            "wrong_exit": self.wrong_exit, "depot_distance_m": self.depot_distance_m,
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n: u32,
    tick: u32,
    horizon: u32,
    flags: u32,
    goal_radius: f32,
    area_radius: f32,
    v_min: f32,
    v_stop: f32,
    hold_ticks: f32,
    corridor_half: f32,
    overshoot_along: f32,
    propelled_min: f32,
    box_half: f32,
    exit_overshoot: f32,
    align_min: f32,
    gate_radius: f32,
}

pub struct SchoolGpu {
    pub buf_state: wgpu::Buffer,
    buf_snapshot: wgpu::Buffer,
    pub capacity: usize,
    pub contract: LessonContract,
    pub horizon_ticks: u32,
    /// Bench mode: taxi–taxi contacts off, passengers off.
    pub bench: bool,
    /// Stop lesson: the score also pays for slowing inside the area and holding (flags bit 1).
    pub stop_shaping: bool,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl SchoolGpu {
    pub fn new(
        device: &wgpu::Device,
        capacity: usize,
        contract: LessonContract,
        horizon_ticks: u32,
        bench: bool,
        stop_shaping: bool,
    ) -> Self {
        assert!(capacity > 0, "school capacity");
        assert!(horizon_ticks > 0, "school horizon");
        let bytes = (capacity * SCHOOL_STRIDE_F32 * 4) as u64;
        let buf_state = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("organism-school-state"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_snapshot = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("organism-school-snapshot"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("organism-school-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, true),
                storage(2, true),
                storage(3, false),
                storage(4, false),
                storage(5, true),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("organism-school"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("school.wgsl"))),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("organism-school-judge"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("organism-school-judge-layout"),
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &module,
            entry_point: Some("judge"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            buf_state,
            buf_snapshot,
            capacity,
            contract,
            horizon_ticks,
            bench,
            stop_shaping,
            layout,
            pipeline,
        }
    }

    /// Declare every being's lesson geometry and clear all counters. Called once
    /// per generation before the first judged tick, never mid-episode.
    pub fn reset(
        &self,
        queue: &wgpu::Queue,
        lessons: &[LessonGeometry],
    ) -> Result<(), &'static str> {
        if lessons.is_empty() || lessons.len() > self.capacity {
            return Err("school: lesson count exceeds capacity");
        }
        let mut rows = vec![0.0f32; self.capacity * SCHOOL_STRIDE_F32];
        for (slot, lesson) in lessons.iter().enumerate() {
            let record = lesson.record();
            if record.iter().any(|v| !v.is_finite()) {
                return Err("school: nonfinite lesson geometry");
            }
            rows[slot * SCHOOL_STRIDE_F32..(slot + 1) * SCHOOL_STRIDE_F32].copy_from_slice(&record);
        }
        queue.write_buffer(&self.buf_state, 0, bytemuck::cast_slice(&rows));
        Ok(())
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        physics_input: &wgpu::Buffer,
        physics_output: &wgpu::Buffer,
        body: &wgpu::Buffer,
        intent: &wgpu::Buffer,
        n: usize,
        tick: u32,
    ) -> Result<(), &'static str> {
        if n == 0 || n > self.capacity {
            return Err("school: invalid count");
        }
        for (buffer, stride) in [
            (physics_input, PHYSICS_INPUT_STRIDE),
            (physics_output, PHYSICS_OUTPUT_STRIDE),
            (body, BODY_STRIDE),
            (intent, 96),
        ] {
            if !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
                || buffer.size() < n as u64 * stride as u64 * 4
            {
                return Err("school: invalid resident buffer");
            }
        }
        let c = &self.contract;
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("organism-school-params"),
            contents: bytemuck::bytes_of(&Params {
                n: n as u32,
                tick,
                horizon: self.horizon_ticks,
                flags: self.bench as u32 | (self.stop_shaping as u32) << 1,
                goal_radius: c.goal_radius_m,
                area_radius: c.area_radius_m,
                v_min: c.v_min_mps,
                v_stop: c.v_stop_mps,
                hold_ticks: c.hold_ticks as f32,
                corridor_half: c.corridor_half_m,
                overshoot_along: c.overshoot_along_m,
                propelled_min: c.propelled_min_ticks as f32,
                box_half: c.box_half_m,
                exit_overshoot: c.exit_overshoot_m,
                align_min: c.align_min_cos,
                gate_radius: c.gate_radius_m,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("organism-school-judge"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: physics_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: physics_output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: body.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buf_state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: intent.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("organism-school-judge"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        Ok(())
    }

    /// Generation-boundary readback of every record. Not an ordinary-tick path.
    pub fn download(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        n: usize,
    ) -> Result<Vec<[f32; SCHOOL_STRIDE_F32]>, &'static str> {
        if n == 0 || n > self.capacity {
            return Err("school: invalid count");
        }
        let bytes = (n * SCHOOL_STRIDE_F32 * 4) as u64;
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.buf_state, 0, &self.buf_snapshot, 0, bytes);
        queue.submit(Some(encoder.finish()));
        let slice = self.buf_snapshot.slice(0..bytes);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        if !matches!(receiver.recv(), Ok(Ok(()))) {
            return Err("school: snapshot mapping failed");
        }
        let mapped = slice.get_mapped_range();
        let values: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
        drop(mapped);
        self.buf_snapshot.unmap();
        if values.iter().any(|v| !v.is_finite()) {
            return Err("school: nonfinite record");
        }
        Ok(values
            .chunks_exact(SCHOOL_STRIDE_F32)
            .map(|c| {
                let mut r = [0.0f32; SCHOOL_STRIDE_F32];
                r.copy_from_slice(c);
                r
            })
            .collect())
    }
}

/// Host-side view of one record for exports.
pub fn record_json(
    r: &[f32; SCHOOL_STRIDE_F32],
    energy_now_j: f32,
    distance_now_m: f32,
) -> serde_json::Value {
    let status = r[R_STATUS] as u32;
    let code = r[R_FAIL_CODE] as u32;
    // A record still running has no outcome snapshot; fall back to the live body.
    let closed = status != STATUS_RUNNING;
    let energy_at_outcome = if closed {
        r[R_ENERGY_OUTCOME]
    } else {
        energy_now_j
    };
    let distance_at_outcome = if closed {
        r[R_DISTANCE_OUTCOME]
    } else {
        distance_now_m
    };
    serde_json::json!({
        "lesson": r[R_LESSON] as u32, "phase": r[R_PHASE] as u32,
        "status": status_name(status), "status_code": status,
        "fail": fail_code_name(code), "fail_code": code,
        "success": status == STATUS_SUCCESS,
        "progress": r[R_PROGRESS], "best_remaining_m": r[R_BEST_REMAINING],
        "initial_remaining_m": r[R_INITIAL_REMAINING],
        "dwell_ticks": r[R_DWELL], "propelled_ticks": r[R_PROPELLED],
        "judged_ticks": r[R_ELAPSED], "outcome_tick": r[R_OUTCOME_TICK],
        "max_speed_mps": r[R_MAX_SPEED], "entry_speed_mps": r[R_ENTRY_SPEED],
        "max_lateral_m": r[R_MAX_LAT], "max_along_m": r[R_MAX_ALONG],
        "contact_ticks": r[R_CONTACT_TICKS], "collision_energy_j": r[R_COLLISION_J],
        "energy_spent_j": r[R_ENERGY_START] - energy_at_outcome,
        "distance_m": distance_at_outcome - r[R_DISTANCE_START],
        "energy_episode_j": r[R_ENERGY_START] - energy_now_j,
        "distance_episode_m": distance_now_m - r[R_DISTANCE_START],
        "score": r[R_SCORE],
        "min_speed_inside_mps": if r[R_MIN_SPEED_INSIDE] < 1.0e8 { serde_json::json!(r[R_MIN_SPEED_INSIDE]) } else { serde_json::Value::Null },
        "max_dwell_ticks": r[R_MAX_DWELL],
        "near_stop_dwell_ticks": r[R_NEAR_DWELL],
        "max_near_stop_dwell_ticks": r[R_MAX_NEAR_DWELL],
        "accepted_pickup_tick": if r[R_ACCEPTED_PICKUP_TICK] >= 0.0 { serde_json::json!(r[R_ACCEPTED_PICKUP_TICK]) } else { serde_json::Value::Null },
        "accepted_dropoff_tick": if r[R_ACCEPTED_DROPOFF_TICK] >= 0.0 { serde_json::json!(r[R_ACCEPTED_DROPOFF_TICK]) } else { serde_json::Value::Null },
        "pickup_requests": r[R_PICKUP_REQUESTS], "dropoff_requests": r[R_DROPOFF_REQUESTS],
        "service_refusal_code": r[R_REFUSAL_CODE], "passenger_id": r[R_PASSENGER_ID],
        "delivery_start": r[R_DELIVERY_START], "delivery_delta": r[R_DELIVERY_DELTA],
    })
}
