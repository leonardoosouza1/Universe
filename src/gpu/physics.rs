//! wgpu host for the vehicle physics kernel — three kernels chained in the same CommandEncoder.
//!
//! # Why three kernels?
//!
//! The original shader (`physics.wgsl`, 756 lines) caused a SIGSEGV in the ACO/NIR compiler of
//! the RADV NAVI22 driver (Mesa). Bisection (`fisica_bisect`) isolated the problem:
//!
//!   MINIMO         → OK  (copy only)
//!   FASE_SUSP      → OK  (suspension + var<private> arrays)
//!   FASE_POWER_A   → OK  (engine + RPM + drivetrain switch)
//!   FASE_POWER_B   → OK  (tyres WITHOUT Pacejka — only cos/sin + omega)
//!   FASE_POWER_C   → CRASH (adds atan/atan2/sin + Pacejka computation)
//!
//! Earlier attempt (2 kernels): physics_b.wgsl combined tyres+collision+integration and
//! still crashed — Pacejka alone (FASE_POWER_C) already exceeds the ACO's capacity.
//!
//! Final solution (3 kernels):
//!   Kernel A (physics_a.wgsl): suspension + engine + aero  → writes MidState (128B/car)
//!   Kernel B (physics_b.wgsl): Pacejka loop (4 wheels)     → reads MidState, writes MidState2 (64B/car)
//!   Kernel C (physics_c.wgsl): collision + integration     → reads MidState + MidState2, writes output
//!
//! Each kernel uses ≤ 50 simultaneous local f32s — below the ACO RDNA2 crash threshold.
//! The three dispatches stay in the same CommandEncoder — no host↔GPU round-trip between them.
//!
//! # Overhead
//!
//! Two extra dispatches per tick (64 threads, 1 workgroup for ≤60 cars). Measured as
//! negligible compared to the CPU↔GPU round-trip latency of the readback.
//!
//! # Declared tolerances (CPU vs GPU)
//!
//! Per single step (dt ≈ 1/60 s):
//! - Force (single-pass float): ≤ 2 ULP or relative error ≤ 1e-4
//! - Slip ratio / slip angle: ≤ 2 ULP (dominated by atan, IEEE-754 identical)
//! - Drift accumulated over N ticks: measured and declared in gpu_validate --fisica

use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

#[cfg(test)]
#[path = "physics_contact_tests.rs"]
mod physics_contact_tests;

// ============================================================
// === DATA STRUCTS (same as before)                         ===
// ============================================================

/// State of one wheel, packed into 2 vec4 (8 floats).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct WheelGpu {
    pub part_a: [f32; 4], // [omega, slip_ratio, slip_angle, normal_load]
    pub part_b: [f32; 4], // [tire_temp, susp_comp, prev_comp, padding]
}

impl WheelGpu {
    pub fn from_cpu(
        omega: f32,
        slip_ratio: f32,
        slip_angle: f32,
        normal_load: f32,
        tire_temp: f32,
        susp_comp: f32,
        prev_comp: f32,
    ) -> Self {
        Self {
            part_a: [omega, slip_ratio, slip_angle, normal_load],
            part_b: [tire_temp, susp_comp, prev_comp, 0.0],
        }
    }
}

/// Bitmask of the configuration flags.
pub fn pack_config_flags(
    stall: bool,
    thermal: bool,
    aero_drag: bool,
    downforce: bool,
    tcs: bool,
    abs: bool,
) -> u32 {
    (stall as u32)
        | ((thermal as u32) << 1)
        | ((aero_drag as u32) << 2)
        | ((downforce as u32) << 3)
        | ((tcs as u32) << 4)
        | ((abs as u32) << 5)
}

/// Optional per-step wall-contact telemetry; bits 0..5 are existing physics flags.
pub const CONTACT_DIAGNOSTICS_FLAG: u32 = 1 << 6;
/// Static-friction rest constraint. Off for legacy race runs; organism-taxi-v1 sets it
/// unconditionally (src/bin/gpu_evolve/organism_tick.rs:27), with no CLI switch.
pub const STATIC_REST_FLAG: u32 = 1 << 9;
/// Power-conjugate wheel/body force transform. Legacy race physics leaves this
/// clear; resident taxi physics declares the corrected frame in its manifest.
pub const POWER_CONSISTENT_STEERING_FLAG: u32 = 1 << 10;
/// Opt-in implicit wheel angular step; preserves the existing tyre force law.
pub const IMPLICIT_WHEEL_FLAG: u32 = 1 << 11;
/// Opt-in immediate TCS cut with a 100 ms torque recovery timescale.
pub const BOUNDED_TCS_FLAG: u32 = 1 << 12;
/// Versioned resident stall, paid starter and gearbox contract. Input83 is starter authorization.
pub const POWERTRAIN_V2_FLAG: u32 = 1 << 13;
pub const STABLE_WHEEL_FLAGS: u32 = IMPLICIT_WHEEL_FLAG | BOUNDED_TCS_FLAG;

/// World/runtime properties inherited at ecological birth, not body genes or
/// acquired motion/controller state. Keep morphology bit 7 with the newborn.
pub fn inherit_resident_profile(parent: &PhysicsStateGpu, newborn: &mut PhysicsStateGpu) {
    const PROFILE_MASK: u32 = CONTACT_DIAGNOSTICS_FLAG
        | (1 << 8)
        | STATIC_REST_FLAG
        | POWER_CONSISTENT_STEERING_FLAG
        | STABLE_WHEEL_FLAGS
        | POWERTRAIN_V2_FLAG;
    newborn.config_flags =
        (newborn.config_flags & !PROFILE_MASK) | (parent.config_flags & PROFILE_MASK);
    if parent.config_flags & POWERTRAIN_V2_FLAG != 0 {
        newborn.current_gear = 2;
        newborn.current_rpm = 800.0;
        newborn.is_stalled = 0;
        newborn.starter_timer = 0.0;
        newborn.shift_timer = 0.0;
        newborn._pad_wind = 0.0;
    }
    newborn.dt = parent.dt;
    newborn.engine_temp = 20.0;
    newborn.oil_temp = 20.0;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactDiagnostics {
    pub contact_mask: u32,
    pub impact_speed_mps: f32,
    pub impulse_ns: f32,
    pub energy_loss_j: f32,
}

/// Position extracted from `PhysicsStateGpu`, for distance comparison.
///
/// Deduplicated (E3): an identical copy existed in `gpu_evolve.rs`, `gpu_loop.rs` and
/// `cortex_evolve.rs` — the three bins compared position the same way.
#[inline]
pub fn pos_of(p: &PhysicsStateGpu) -> [f32; 3] {
    p.pos
}

/// Builds the initial `PhysicsStateGpu` for a being at position `start`, facing `fwd`.
///
/// Deduplicated (E3 — REFACTOR_PLAN §2.3): THREE byte-identical copies of this
/// function existed (`gpu_evolve.rs`, `gpu_loop.rs`, `cortex_evolve.rs`), confirmed
/// field by field before the unification — none had diverged (v0=0.0 in all three: "if
/// the beings move, the movement is the BRAIN'S DECISION", not injected velocity).
pub fn make_physics_state(start: [f32; 3], fwd: [f32; 3]) -> PhysicsStateGpu {
    let normal_load = 1350.0 * 9.81 / 4.0;
    // Tyre temperature is dimensionless throughout the GPU kernels:
    // 0 = cold, 0.5 = optimum, 1 = hot. The old 60.0 forced grip to its 0.70 floor.
    let wheel = WheelGpu::from_cpu(0.0, 0.0, 0.0, normal_load, 0.5, 0.0, 0.0);
    let lat = [fwd[2], 0.0, -fwd[0]];
    // v0 = 0 m/s: no injected velocity. Movement = the brain's decision.
    let v0 = 0.0_f32;
    PhysicsStateGpu {
        vel: [fwd[0] * v0, 0.0, fwd[2] * v0],
        av_y: 0.0,
        wheels: [
            wheel.part_a,
            wheel.part_b,
            wheel.part_a,
            wheel.part_b,
            wheel.part_a,
            wheel.part_b,
            wheel.part_a,
            wheel.part_b,
        ],
        pos: [start[0], start[1].max(0.5), start[2]],
        _pad_pos: 0.0,
        fwd,
        _pad_fwd: 0.0,
        lat,
        _pad_lat: 0.0,
        max_torque: BASE_ENGINE_TORQUE_NM,
        current_rpm: 2000.0,
        max_rpm: 8500.0,
        throttle: 0.0,
        brake_cmd: 0.0,
        steer: 0.0,
        drive_torque_out: 0.0,
        current_gear: 3, // 2nd gear (convention: 0=reverse, 1=neutral, 2=1st ...)
        shift_timer: 0.0,
        is_stalled: 0,
        starter_timer: 0.0,
        flywheel_inertia: 0.15,
        engine_temp: 90.0,
        oil_temp: 85.0,
        mass: 1350.0,
        drag_coeff: 0.32,
        wheel_base: 2.65,
        cg_height: 0.45,
        track_width: 1.55,
        lsd_lock_ratio: 0.3,
        brake_torque_max: 4000.0,
        drivetrain: 1,
        mu_peak: 1.6,
        cornering_stiffness: 55000.0,
        wheel_radius: 0.33,
        wheel_inertia: 1.5,
        optimal_temp: 0.5,
        susp_stiffness: 45000.0,
        susp_damping: 3000.0,
        susp_rest: 0.25,
        susp_max_travel: 0.12,
        _pad_before_wind: 0.0,
        wind: [0.0; 3],
        _pad_wind: 0.0,
        terrain_grip: 1.0,
        env_grip: 1.0,
        aquaplaning: 1.0,
        config_flags: pack_config_flags(true, true, true, true, true, true),
        dt: 1.0 / 64.0,
        _pad_dt: 0.0,
    }
}

/// Historical engine capacity, also the reference for fuel load scaling.
pub const BASE_ENGINE_TORQUE_NM: f32 = 620.0;

/// Stride of the `input: array<CarInput>` array in WGSL.
/// SizeOf(PhysicsStateGpu)=360, AlignOf=16 → stride=368.
pub const INPUT_WGSL_STRIDE: usize = 368;

fn serialize_input(state: &[PhysicsStateGpu]) -> Vec<u8> {
    let n = state.len();
    let size = std::mem::size_of::<PhysicsStateGpu>(); // 360
    let mut buf = vec![0u8; n * INPUT_WGSL_STRIDE];
    for (i, e) in state.iter().enumerate() {
        let dst = &mut buf[i * INPUT_WGSL_STRIDE..i * INPUT_WGSL_STRIDE + size];
        dst.copy_from_slice(bytemuck::bytes_of(e));
    }
    buf
}

/// Full state of one car for a GPU step.
#[repr(C)]
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize, Pod, Zeroable)]
pub struct PhysicsStateGpu {
    pub vel: [f32; 3],
    pub av_y: f32,
    pub wheels: [[f32; 4]; 8],
    pub pos: [f32; 3],
    pub _pad_pos: f32,
    pub fwd: [f32; 3],
    pub _pad_fwd: f32,
    pub lat: [f32; 3],
    pub _pad_lat: f32,
    pub max_torque: f32,
    pub current_rpm: f32,
    pub max_rpm: f32,
    pub throttle: f32,
    pub brake_cmd: f32,
    pub steer: f32,
    pub drive_torque_out: f32,
    pub current_gear: u32,
    pub shift_timer: f32,
    pub is_stalled: u32,
    pub starter_timer: f32,
    pub flywheel_inertia: f32,
    pub engine_temp: f32,
    pub oil_temp: f32,
    pub mass: f32,
    pub drag_coeff: f32,
    pub wheel_base: f32,
    pub cg_height: f32,
    pub track_width: f32,
    pub lsd_lock_ratio: f32,
    pub brake_torque_max: f32,
    pub drivetrain: u32,
    pub mu_peak: f32,
    pub cornering_stiffness: f32,
    pub wheel_radius: f32,
    pub wheel_inertia: f32,
    pub optimal_temp: f32,
    pub susp_stiffness: f32,
    pub susp_damping: f32,
    pub susp_rest: f32,
    pub susp_max_travel: f32,
    pub _pad_before_wind: f32,
    pub wind: [f32; 3],
    pub _pad_wind: f32,
    pub terrain_grip: f32,
    pub env_grip: f32,
    pub aquaplaning: f32,
    pub config_flags: u32,
    pub dt: f32,
    pub _pad_dt: f32,
}

impl PhysicsStateGpu {
    /// Validate the actual command and body before any GPU upload.
    pub fn validate_contract(&self) -> Result<(), crate::contracts::ContractError> {
        use crate::contracts::{TemperatureC, VehicleCommand, finite, positive};
        VehicleCommand::new(self.throttle, self.steer, self.brake_cmd)?;
        if self.is_stalled > 1 {
            return Err(crate::contracts::ContractError {
                field: "physics.input.is_stalled",
                expected: "0 or 1",
            });
        }
        if self.current_gear > 6 {
            return Err(crate::contracts::ContractError {
                field: "physics.input.gear",
                expected: "raw gear code 0..6",
            });
        }
        if self.config_flags & POWERTRAIN_V2_FLAG != 0
            && (self._pad_wind != 0.0 && self._pad_wind != 1.0)
        {
            return Err(crate::contracts::ContractError {
                field: "physics.input.starter_authorized",
                expected: "exact 0 or 1",
            });
        }
        TemperatureC::new(self.engine_temp)?;
        TemperatureC::new(self.oil_temp)?;
        for (field, value) in [
            ("mass_kg", self.mass),
            ("dt_seconds", self.dt),
            ("max_rpm", self.max_rpm),
            ("wheel_radius_m", self.wheel_radius),
            ("wheel_inertia", self.wheel_inertia),
            ("flywheel_inertia", self.flywheel_inertia),
            ("wheel_base_m", self.wheel_base),
            ("track_width_m", self.track_width),
        ] {
            positive(value, field)?;
        }
        for (field, value) in [
            ("yaw_rate_rad_s", self.av_y),
            ("max_torque_nm", self.max_torque),
            ("brake_torque_nm", self.brake_torque_max),
            ("current_rpm", self.current_rpm),
            ("drive_torque_out", self.drive_torque_out),
            ("shift_timer", self.shift_timer),
            ("starter_timer", self.starter_timer),
            ("drag_coeff", self.drag_coeff),
            ("cg_height", self.cg_height),
            ("lsd_lock_ratio", self.lsd_lock_ratio),
            ("mu_peak", self.mu_peak),
            ("cornering_stiffness", self.cornering_stiffness),
            ("optimal_temp", self.optimal_temp),
            ("susp_stiffness", self.susp_stiffness),
            ("susp_damping", self.susp_damping),
            ("susp_rest", self.susp_rest),
            ("susp_max_travel", self.susp_max_travel),
            ("cooling_scale", self._pad_before_wind),
            ("terrain_grip", self.terrain_grip),
            ("env_grip", self.env_grip),
            ("aquaplaning", self.aquaplaning),
        ] {
            finite(value, field)?;
        }
        for (field, values) in [
            ("velocity_mps", &self.vel),
            ("position_m", &self.pos),
            ("forward", &self.fwd),
            ("lateral", &self.lat),
            ("wind_mps", &self.wind),
        ] {
            for &value in values {
                finite(value, field)?;
            }
        }
        for wheel in &self.wheels {
            for &value in wheel {
                finite(value, "wheel")?;
            }
        }
        Ok(())
    }

    /// Decode the four diagnostic values returned in output padding by Kernel C.
    pub fn contact_diagnostics(&self) -> ContactDiagnostics {
        ContactDiagnostics {
            contact_mask: self._pad_pos.max(0.0) as u32,
            impact_speed_mps: self._pad_fwd,
            impulse_ns: self._pad_lat,
            energy_loss_j: self._pad_dt,
        }
    }
}

/// Per-car output after kernel B.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PhysicsOutputGpu {
    pub vel: [f32; 3],
    pub av_y: f32,
    pub wheels: [[f32; 4]; 8],
    pub pos: [f32; 3],
    pub _pad_pos: f32,
    pub fwd: [f32; 3],
    pub _pad_fwd: f32,
    pub lat: [f32; 3],
    pub _pad_lat: f32,
    pub current_rpm: f32,
    pub drive_torque_out: f32,
    pub shift_timer: f32,
    pub is_stalled: u32,
    pub starter_timer: f32,
    pub engine_temp: f32,
    pub oil_temp: f32,
    pub _pad_eng: f32,
}

impl PhysicsOutputGpu {
    /// Validate the complete readback before committing any car state.
    pub fn validate_contract(&self) -> Result<(), crate::contracts::ContractError> {
        use crate::contracts::{ContractError, TemperatureC, finite};
        for vector in [&self.vel, &self.pos, &self.fwd, &self.lat] {
            for &value in vector {
                finite(value, "physics.output.vector")?;
            }
        }
        for wheel in &self.wheels {
            for &value in wheel {
                finite(value, "physics.output.wheel")?;
            }
        }
        for value in [
            self.av_y,
            self.current_rpm,
            self.drive_torque_out,
            self.shift_timer,
            self.starter_timer,
            self._pad_pos,
            self._pad_fwd,
            self._pad_lat,
            self._pad_eng,
        ] {
            finite(value, "physics.output.scalar")?;
        }
        TemperatureC::new(self.engine_temp)?;
        TemperatureC::new(self.oil_temp)?;
        if self.is_stalled > 1 {
            return Err(ContractError {
                field: "physics.output.is_stalled",
                expected: "0 or 1",
            });
        }
        Ok(())
    }
}

/// Intermediate buffer between Kernel A and Kernel B. 128 bytes/car, stride=128.
/// Must exactly mirror MidState in physics_a.wgsl, physics_b.wgsl and physics_c.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MidState {
    normal_loads: [f32; 4], // 16
    susp_comps: [f32; 4],   // 16
    prev_comps: [f32; 4],   // 16
    current_rpm: f32,       //  4
    drive_torque: f32,      //  4
    drive_torque_out: f32,  //  4
    shift_timer: f32,       //  4
    is_stalled: u32,        //  4
    starter_timer: f32,     //  4
    engine_temp: f32,       //  4
    oil_temp: f32,          //  4
    omegas: [f32; 4],       // 16
    forces: [f32; 3],       // 12
    _pad_f: f32,            //  4
    torques_y: f32,         //  4
    yaw_inertia: f32,       //  4
    tcs_fraction: f32,      //  4: acquired TCS gain, kernel A owns it
    tcs_initialized: u32,   //  4: reset marker
                            // Total: 128 bytes — multiple of 16, stride = 128
}

/// Intermediate buffer between Kernel B and Kernel C. 64 bytes/car, stride=64.
/// Must exactly mirror MidState2 in physics_b.wgsl and physics_c.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MidState2 {
    forces: [f32; 3], // 12  — total forces (gravity + suspension + aero + tyres)
    torques_y: f32,   //  4  — accumulated yaw torque
    omegas: [f32; 4], // 16  — post-braking omegas
    slip_r: [f32; 4], // 16  — slip ratios
    slip_a: [f32; 4], // 16  — slip angles
                      // Total: 64 bytes — multiple of 16, stride = 64
}

/// Global world parameters (uniform buffer).
///
/// D5 (ESCADA_DIFICULDADE.md — climate) appended 4 fields at the end: `clima_mu`,
/// `vento_x`, `vento_z`, `clima_freq`. Safe append — the live kernels
/// (`physics_a/b/c.wgsl`) declare a shorter `WorldParams` (16 bytes) and
/// `min_binding_size: None` (see `uniform_entry` in `core.rs`) allows a buffer
/// larger than the binding declared in the shader: they keep reading only the
/// original 16 bytes, byte-identical to the behaviour before this commit.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WorldParams {
    grid_size: u32,
    n_beings: u32,
    cell_size: f32,
    _pad: f32,
    /// D5 — global grip multiplier: 1.0 = dry (default, bit-identical to the
    /// pre-D5 behaviour), 0.7 = rain. Consumed in `physics_power_c.wgsl`
    /// (the only kernel of the POWER trio that computes tyre force).
    clima_mu: f32,
    /// D5 — constant lateral wind (N) applied to the body, world X axis. Default 0.
    vento_x: f32,
    /// D5 — constant lateral wind (N) applied to the body, world Z axis. Default 0.
    vento_z: f32,
    /// D5 — spatial frequency of the μ variation. 0 = spatially uniform μ
    /// (default). >0 = deterministic slippery stretches by position.
    clima_freq: f32,
}

// ============================================================
// === PERSISTENT GPU CONTEXT                               ===
// ============================================================

/// wgpu context for the triple physics kernel. Persists between ticks as a Bevy resource.
/// Created once per track; recreated if the track changes (detected via signature).
///
/// Three pipelines compiled once; per tick only write_buffer + 3 dispatches + readback.
pub struct PhysicsGpu {
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    /// Kernel A pipeline (suspension + engine + aero)
    pipeline_a: wgpu::ComputePipeline,
    layout_a: wgpu::BindGroupLayout,
    /// Kernel B pipeline (Pacejka tyres — isolated to reduce register pressure)
    pipeline_b: wgpu::ComputePipeline,
    layout_b: wgpu::BindGroupLayout,
    /// Kernel C pipeline (collision + integration)
    pipeline_c: wgpu::ComputePipeline,
    layout_c: wgpu::BindGroupLayout,
    /// Voxel grid (static per track)
    buf_grid: wgpu::Buffer,
    /// Bilinear heightmap (static per track)
    buf_hmap: wgpu::Buffer,
    /// Input buffer (written by the host every tick)
    buf_input: wgpu::Buffer,
    /// Intermediate buffer A→B: suspension + engine + partial forces (128B/car)
    buf_mid: wgpu::Buffer,
    /// Intermediate buffer B→C: tyre forces + omegas + slip (64B/car)
    buf_mid2: wgpu::Buffer,
    /// Kernel C output buffer
    buf_output: wgpu::Buffer,
    /// Readback buffer (MAP_READ | COPY_DST)
    buf_readback: wgpu::Buffer,
    pub capacity: usize,
    pub adapter: String,
    grid_size: u32,
    cell_size: f32,
    /// D5 — climate: see `set_clima`. Default 1.0 = dry, no effect (bit-identical).
    clima_mu: f32,
    /// D5 — constant lateral wind (N), world X axis. Default 0.0.
    vento_x: f32,
    /// D5 — constant lateral wind (N), world Z axis. Default 0.0.
    vento_z: f32,
    /// D5 — spatial frequency of μ. Default 0.0 = uniform.
    clima_freq: f32,
}

impl PhysicsGpu {
    /// Opens a GPU device and compiles the two kernels.
    /// Returns `None` if there is no usable GPU.
    /// Opens its OWN device. Kept for the equivalence tests, which run the physics
    /// in isolation. In the simulation loop use `new_shared` — see why there.
    pub fn new(
        grid: &[u8],
        hmap: &[f32],
        grid_size: usize,
        cell_size: f32,
        capacity: usize,
    ) -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;
        let name = adapter.get_info().name.clone();

        // Kernel B uses 6 bindings (params, grid, hmap, input, mid, output).
        // downlevel_defaults() allows max_storage_buffers_per_shader_stage=4 — insufficient.
        // We raise it to 8 (same as BrainGpu in brain.rs).
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_storage_buffers_per_shader_stage = 8;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("iara-physics"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;
        Self::build(
            std::sync::Arc::new(device),
            std::sync::Arc::new(queue),
            name,
            grid,
            hmap,
            grid_size,
            cell_size,
            capacity,
        )
    }

    /// Builds on top of an ALREADY OPEN device, shared with retina/perception/brain.
    ///
    /// # Why this exists
    ///
    /// Physics was born with its own device to isolate a crash of the ACO compiler in
    /// the RADV driver. That crash was solved another way — splitting the shader into
    /// three kernels (`physics.wgsl` + `physics_b.wgsl` + `physics_c.wgsl`) —, so the
    /// isolation stopped being necessary and started to HURT: with two `wgpu::Device`
    /// on the same GPU, the loop hung on the first tick (`gpu_loop` got stuck in warmup,
    /// no error and no output). A single device solves it, and it is also what allows
    /// chaining the four dispatches in a single `CommandEncoder` with no readback
    /// between them — which is the point of the all-GPU loop.
    pub fn new_shared(
        device: std::sync::Arc<wgpu::Device>,
        queue: std::sync::Arc<wgpu::Queue>,
        adapter_name: String,
        grid: &[u8],
        hmap: &[f32],
        grid_size: usize,
        cell_size: f32,
        capacity: usize,
    ) -> Option<Self> {
        Self::build(
            device,
            queue,
            adapter_name,
            grid,
            hmap,
            grid_size,
            cell_size,
            capacity,
        )
    }

    fn build(
        device: std::sync::Arc<wgpu::Device>,
        queue: std::sync::Arc<wgpu::Queue>,
        name: String,
        grid: &[u8],
        hmap: &[f32],
        grid_size: usize,
        cell_size: f32,
        capacity: usize,
    ) -> Option<Self> {
        // Kernel A: 5 bindings (params, grid, hmap, input, mid)
        let mod_a = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("physics-a"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("physics_a.wgsl"))),
        });
        let layout_a = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("physics-layout-a"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),  // grid
                storage_entry(2, true),  // hmap
                storage_entry(3, true),  // input
                storage_entry(4, false), // mid (write)
            ],
        });
        let pipe_layout_a = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout_a],
            push_constant_ranges: &[],
        });
        let pipeline_a = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("physics-pipeline-a"),
            layout: Some(&pipe_layout_a),
            module: &mod_a,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Kernel B: 6 bindings (params, grid, hmap, input, mid, mid2)
        // Only the Pacejka loop — no collision and no integration, to stay below the
        // ACO/RDNA2 register limit.
        let mod_b = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("physics-b"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("physics_b.wgsl"))),
        });
        let layout_b = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("physics-layout-b"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true), // grid  (declared but unused in B — keeps the layout uniform)
                storage_entry(2, true), // hmap  (same)
                storage_entry(3, true), // input
                storage_entry(4, true), // mid   (read — MidState from A)
                storage_entry(5, false), // mid2  (write — MidState2 for C)
            ],
        });
        let pipe_layout_b = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout_b],
            push_constant_ranges: &[],
        });
        let pipeline_b = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("physics-pipeline-b"),
            layout: Some(&pipe_layout_b),
            module: &mod_b,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Kernel C: 7 bindings (params, grid, hmap, input, mid, mid2, output)
        let mod_c = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("physics-c"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("physics_c.wgsl"))),
        });
        let layout_c = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("physics-layout-c"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),  // grid
                storage_entry(2, true),  // hmap
                storage_entry(3, true),  // input
                storage_entry(4, true),  // mid  (read — MidState from A)
                storage_entry(5, true),  // mid2 (read — MidState2 from B)
                storage_entry(6, false), // output (write)
            ],
        });
        let pipe_layout_c = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout_c],
            push_constant_ranges: &[],
        });
        let pipeline_c = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("physics-pipeline-c"),
            layout: Some(&pipe_layout_c),
            module: &mod_c,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Static buffers per track.
        // COPY_DST (besides STORAGE): allows `reload_world()` to swap the track via
        // `queue.write_buffer` without recreating buffer/bind-group/pipeline — see the
        // same reasoning in `gpu::perception::PerceptionGpu::reload_world` (P1, track
        // rotation — docs/PLANO_EVOLUCAO.md). Valid because `grid_size`/`cell_size`
        // are constant by construction (`VoxelWorld::new`, world.rs), so the size in
        // bytes never changes between tracks.
        let grid_u32: Vec<u32> = grid.iter().map(|v| *v as u32).collect();
        let buf_grid = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("physics-grid"),
            contents: bytemuck::cast_slice(&grid_u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let buf_hmap = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("physics-hmap"),
            contents: bytemuck::cast_slice(hmap),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let cap = capacity.max(64);
        let bytes_input = (cap * INPUT_WGSL_STRIDE) as u64;
        let bytes_mid = (cap * std::mem::size_of::<MidState>()) as u64;
        let bytes_mid2 = (cap * std::mem::size_of::<MidState2>()) as u64;
        let bytes_output = (cap * std::mem::size_of::<PhysicsOutputGpu>()) as u64;

        let buf_input = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("physics-input"),
            size: bytes_input,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // buf_mid: written by A, read by B and C
        let buf_mid = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("physics-mid"),
            size: bytes_mid,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // buf_mid2: written by B, read by C
        let buf_mid2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("physics-mid2"),
            size: bytes_mid2,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let buf_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("physics-output"),
            size: bytes_output,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("physics-readback"),
            size: bytes_output,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Some(Self {
            device,
            queue,
            pipeline_a,
            layout_a,
            pipeline_b,
            layout_b,
            pipeline_c,
            layout_c,
            buf_grid,
            buf_hmap,
            buf_input,
            buf_mid,
            buf_mid2,
            buf_output,
            buf_readback,
            capacity: cap,
            adapter: name,
            grid_size: grid_size as u32,
            cell_size,
            // D5 — defaults = pre-climate behaviour, bit-identical.
            clima_mu: 1.0,
            vento_x: 0.0,
            vento_z: 0.0,
            clima_freq: 0.0,
        })
    }

    /// D5 (ESCADA_DIFICULDADE.md) — adjusts climate: global/per-stretch grip (μ) +
    /// constant lateral wind. Written into `WorldParams` (uniform) on every `step()`.
    ///
    /// - `mu`: grip multiplier. 1.0 = dry (default, no effect). 0.7 = rain.
    /// - `vento_x`/`vento_z`: constant lateral force (N) on the body, world X/Z axes.
    ///   Default 0.0 = no wind.
    /// - `freq`: spatial frequency of the μ variation. 0.0 (default) = uniform μ
    ///   (`clima_mu` applied equally over the whole track). >0.0 = μ varies by position
    ///   (deterministic slippery stretches — see `mu_local_at` in the shader).
    ///
    /// Calling with `(1.0, 0.0, 0.0, 0.0)` restores the default — bit-identical to the
    /// behaviour before this method existed (never called = already the constructor's
    /// default).
    pub fn set_clima(&mut self, mu: f32, vento_x: f32, vento_z: f32, freq: f32) {
        self.clima_mu = mu;
        self.vento_x = vento_x;
        self.vento_z = vento_z;
        self.clima_freq = freq;
    }

    /// Swaps grid/heightmap for a NEW track without recreating buffer, bind group or
    /// pipeline — only `queue.write_buffer` (P1: track rotation per generation,
    /// docs/PLANO_EVOLUCAO.md). `device`/`queue` are already struct fields, so it
    /// does not need to receive them as parameters.
    ///
    /// `grid`/`hmap` must have the same size this `PhysicsGpu` was built with —
    /// true whenever they come from `VoxelWorld::new`, whose `size` is constant
    /// regardless of the seed (world.rs). The `debug_assert!` documents that
    /// invariant; in release the cost is zero.
    pub fn reload_world(&self, grid: &[u8], hmap: &[f32]) {
        debug_assert_eq!(
            grid.len(),
            (self.grid_size * self.grid_size) as usize,
            "reload_world: new grid has a different size than the allocated buffer"
        );
        debug_assert_eq!(
            hmap.len(),
            (self.grid_size * self.grid_size) as usize,
            "reload_world: new heightmap has a different size than the allocated buffer"
        );
        let grid_u32: Vec<u32> = grid.iter().map(|v| *v as u32).collect();
        self.queue
            .write_buffer(&self.buf_grid, 0, bytemuck::cast_slice(&grid_u32));
        self.queue
            .write_buffer(&self.buf_hmap, 0, bytemuck::cast_slice(hmap));
    }

    /// Resident 368-byte-stride input storage. Initialize once with `upload_resident_state`,
    /// then let GPU adapters update this buffer between dispatches.
    pub fn input_buffer(&self) -> &wgpu::Buffer {
        &self.buf_input
    }

    /// Resident 224-byte-stride kernel-C output; safe for subsequent compute passes.
    pub fn output_buffer(&self) -> &wgpu::Buffer {
        &self.buf_output
    }

    /// Initialization/reset only. Rejects invalid records before uploading any bytes.
    pub fn upload_resident_state(&self, state: &[PhysicsStateGpu]) -> Result<(), &'static str> {
        if state.len() > self.capacity {
            return Err("resident physics population exceeds capacity");
        }
        if state.iter().any(|body| body.validate_contract().is_err()) {
            return Err("invalid resident physics state");
        }
        if !state.is_empty() {
            self.queue
                .write_buffer(&self.buf_input, 0, &serialize_input(state));
            self.queue.write_buffer(
                &self.buf_mid,
                0,
                &vec![0u8; state.len() * std::mem::size_of::<MidState>()],
            );
        }
        Ok(())
    }

    /// Explicit observation/checkpoint boundary only; no physics dispatch occurs.
    /// Returns the complete resident input record, including commands/capacities.
    pub fn read_resident_state(&self, n: usize) -> Result<Vec<PhysicsStateGpu>, &'static str> {
        if n > self.capacity {
            return Err("resident physics sample capacity");
        }
        if n == 0 {
            return Ok(Vec::new());
        }
        let bytes = (n * INPUT_WGSL_STRIDE) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resident physics sample"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.buf_input, 0, &staging, 0, bytes);
        self.queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| "resident physics map callback")?
            .map_err(|_| "resident physics map failed")?;
        let map = staging.slice(..).get_mapped_range();
        let result = map
            .chunks_exact(INPUT_WGSL_STRIDE)
            .map(|record| {
                let body: PhysicsStateGpu =
                    bytemuck::pod_read_unaligned(&record[..std::mem::size_of::<PhysicsStateGpu>()]);
                body.validate_contract()
                    .map_err(|_| "invalid resident physics sample")?;
                Ok(body)
            })
            .collect();
        drop(map);
        staging.unmap();
        result
    }

    pub fn resident_snapshot_buffers(&self) -> Vec<crate::gpu::state_snapshot::NamedBuffer<'_>> {
        vec![
            ("physics.input", &self.buf_input),
            ("physics.output", &self.buf_output),
            ("physics.mid", &self.buf_mid), // includes the acquired TCS gain
        ]
    }

    /// Birth/reset of one slot without rewriting surviving resident records.
    pub fn upload_resident_slot(
        &self,
        slot: usize,
        state: &PhysicsStateGpu,
    ) -> Result<(), &'static str> {
        if slot >= self.capacity || state.validate_contract().is_err() {
            return Err("invalid resident physics birth slot");
        }
        self.queue.write_buffer(
            &self.buf_input,
            (slot * INPUT_WGSL_STRIDE) as u64,
            &serialize_input(std::slice::from_ref(state)),
        );
        self.queue.write_buffer(
            &self.buf_mid,
            (slot * std::mem::size_of::<MidState>()) as u64,
            bytemuck::bytes_of(&MidState::zeroed()),
        );
        Ok(())
    }

    /// Encode the existing A/B/C kernels into the caller's encoder, without upload,
    /// submission, mapping or polling. `input` must use the padded WGSL stride and
    /// the same device; resident writers own value validation. Intermediate/output
    /// buffers belong to this instance, so dispatches sharing it must be ordered.
    pub fn encode_step(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        n: usize,
    ) -> Result<(), &'static str> {
        if n > self.capacity
            || n > u32::MAX as usize
            || input.size() < (n as u64) * (INPUT_WGSL_STRIDE as u64)
            || !input.usage().contains(wgpu::BufferUsages::STORAGE)
        {
            return Err("invalid resident physics input shape/usage");
        }
        if n == 0 {
            return Ok(());
        }
        let params = WorldParams {
            grid_size: self.grid_size,
            n_beings: n as u32,
            cell_size: self.cell_size,
            _pad: 0.0,
            clima_mu: self.clima_mu,
            vento_x: self.vento_x,
            vento_z: self.vento_z,
            clima_freq: self.clima_freq,
        };
        let buf_params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("physics-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Kernel A bind group: params, grid, hmap, input, mid
        let group_a = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout_a,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buf_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_hmap.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buf_mid.as_entire_binding(),
                },
            ],
        });

        // Kernel B bind group: params, grid, hmap, input, mid, mid2
        let group_b = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout_b,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buf_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_hmap.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buf_mid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.buf_mid2.as_entire_binding(),
                },
            ],
        });

        // Kernel C bind group: params, grid, hmap, input, mid, mid2, output
        let group_c = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout_c,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buf_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_hmap.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buf_mid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.buf_mid2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.buf_output.as_entire_binding(),
                },
            ],
        });

        let wg = (n as u32).div_ceil(64);
        {
            // Kernel A: suspension + engine + aero → writes buf_mid
            let mut p = encoder.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.pipeline_a);
            p.set_bind_group(0, &group_a, &[]);
            p.dispatch_workgroups(wg, 1, 1);
        }
        {
            // Kernel B: Pacejka loop → reads buf_mid, writes buf_mid2
            // Ending pass_a starts a global memory barrier: buf_mid visible to B.
            let mut p = encoder.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.pipeline_b);
            p.set_bind_group(0, &group_b, &[]);
            p.dispatch_workgroups(wg, 1, 1);
        }
        {
            // Kernel C: collision + integration → reads buf_mid + buf_mid2, writes buf_output
            // Ending pass_b starts a barrier: buf_mid2 visible to C.
            let mut p = encoder.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.pipeline_c);
            p.set_bind_group(0, &group_c, &[]);
            p.dispatch_workgroups(wg, 1, 1);
        }
        Ok(())
    }

    /// Runs one physics step for all cars in parallel on the GPU.
    ///
    /// Dispatches Kernel A, B and C in the same CommandEncoder (no round-trip).
    /// Returns `false` if the number of cars exceeds the capacity.
    pub fn step(&self, state: &mut [PhysicsStateGpu]) -> bool {
        self.step_with_passes(state, |_, _, _| {}, |_, _, _| {})
    }

    /// Extend the production physics encoder without intermediate GPU readback.
    /// The pre pass runs after input upload; the post pass runs after kernel C.
    pub fn step_with_passes(
        &self,
        state: &mut [PhysicsStateGpu],
        pre: impl FnOnce(&wgpu::Device, &mut wgpu::CommandEncoder, &wgpu::Buffer),
        post: impl FnOnce(&wgpu::Device, &mut wgpu::CommandEncoder, &wgpu::Buffer),
    ) -> bool {
        let n = state.len();
        if n == 0 {
            return true;
        }
        if n > self.capacity {
            return false;
        }

        for (slot, body) in state.iter().enumerate() {
            if let Err(error) = body.validate_contract() {
                eprintln!("[physics] rejected input slot {slot}: {error}");
                return false;
            }
        }

        let input_bytes = serialize_input(state);
        self.queue.write_buffer(&self.buf_input, 0, &input_bytes);
        let bytes_output = (n * std::mem::size_of::<PhysicsOutputGpu>()) as u64;
        let mut enc = self.device.create_command_encoder(&Default::default());
        pre(&self.device, &mut enc, &self.buf_input);
        if self.encode_step(&mut enc, &self.buf_input, n).is_err() {
            return false;
        }
        post(&self.device, &mut enc, &self.buf_output);
        enc.copy_buffer_to_buffer(&self.buf_output, 0, &self.buf_readback, 0, bytes_output);
        self.queue.submit(Some(enc.finish()));

        // Synchronous read — same pattern as RetinaGpu
        let slice = self.buf_readback.slice(0..bytes_output);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        if !matches!(rx.recv(), Ok(Ok(()))) {
            eprintln!("[physics] GPU readback mapping failed");
            return false;
        }

        {
            let data = slice.get_mapped_range();
            let outputs: &[PhysicsOutputGpu] = bytemuck::cast_slice(&data);
            if let Some((slot, error)) = outputs[..n]
                .iter()
                .enumerate()
                .find_map(|(slot, output)| output.validate_contract().err().map(|e| (slot, e)))
            {
                eprintln!("[physics] rejected output slot {slot}: {error}");
                drop(data);
                self.buf_readback.unmap();
                return false;
            }
            for (i, s) in outputs[..n].iter().enumerate() {
                apply_output(&mut state[i], s);
            }
        }
        self.buf_readback.unmap();
        true
    }

    /// Diagnostics: runs one step and returns the RAW output structs.
    pub fn step_raw(&self, state: &[PhysicsStateGpu]) -> Vec<PhysicsOutputGpu> {
        let n = state.len();
        if n == 0 || n > self.capacity {
            return vec![];
        }
        for (slot, body) in state.iter().enumerate() {
            if let Err(error) = body.validate_contract() {
                eprintln!("[physics] rejected raw input slot {slot}: {error}");
                return vec![];
            }
        }
        let params = WorldParams {
            grid_size: self.grid_size,
            n_beings: n as u32,
            cell_size: self.cell_size,
            _pad: 0.0,
            clima_mu: self.clima_mu,
            vento_x: self.vento_x,
            vento_z: self.vento_z,
            clima_freq: self.clima_freq,
        };
        let buf_params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("physics-params-raw"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let input_bytes = serialize_input(state);
        self.queue.write_buffer(&self.buf_input, 0, &input_bytes);
        let group_a = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout_a,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buf_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_hmap.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buf_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buf_mid.as_entire_binding(),
                },
            ],
        });
        let group_b = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout_b,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buf_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_hmap.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buf_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buf_mid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.buf_mid2.as_entire_binding(),
                },
            ],
        });
        let group_c = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout_c,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buf_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_hmap.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buf_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buf_mid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.buf_mid2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.buf_output.as_entire_binding(),
                },
            ],
        });
        let bytes_output = (n * std::mem::size_of::<PhysicsOutputGpu>()) as u64;
        let wg = (n as u32).div_ceil(64);
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.pipeline_a);
            p.set_bind_group(0, &group_a, &[]);
            p.dispatch_workgroups(wg, 1, 1);
        }
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.pipeline_b);
            p.set_bind_group(0, &group_b, &[]);
            p.dispatch_workgroups(wg, 1, 1);
        }
        {
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.pipeline_c);
            p.set_bind_group(0, &group_c, &[]);
            p.dispatch_workgroups(wg, 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.buf_output, 0, &self.buf_readback, 0, bytes_output);
        self.queue.submit(Some(enc.finish()));
        let slice = self.buf_readback.slice(0..bytes_output);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        if !matches!(rx.recv(), Ok(Ok(()))) {
            eprintln!("[physics] GPU readback mapping failed");
            return vec![];
        }
        let result = {
            let data = slice.get_mapped_range();
            let outputs: &[PhysicsOutputGpu] = bytemuck::cast_slice(&data);
            if outputs[..n]
                .iter()
                .any(|output| output.validate_contract().is_err())
            {
                eprintln!("[physics] rejected invalid raw output");
                vec![]
            } else {
                outputs[..n].to_vec()
            }
        };
        self.buf_readback.unmap();
        result
    }
}

/// Applies the Kernel B output back onto the PhysicsStateGpu.
fn apply_output(e: &mut PhysicsStateGpu, s: &PhysicsOutputGpu) {
    e.vel = s.vel;
    e.av_y = s.av_y;
    e.pos = s.pos;
    e.fwd = s.fwd;
    e.lat = s.lat;
    e.wheels = s.wheels;
    e.current_rpm = s.current_rpm;
    e.drive_torque_out = s.drive_torque_out;
    e.shift_timer = s.shift_timer;
    e.is_stalled = s.is_stalled;
    e.starter_timer = s.starter_timer;
    e.engine_temp = s.engine_temp;
    e.oil_temp = s.oil_temp;
    // Output padding carries contact diagnostics when bit 6 was requested.
    e._pad_pos = s._pad_pos;
    e._pad_fwd = s._pad_fwd;
    e._pad_lat = s._pad_lat;
    e._pad_dt = s._pad_eng;
}

// ============================================================
// === BIND GROUP HELPERS                                   ===
// ============================================================
// Deduplicated into src/gpu/core.rs (E3) — single canonical source. There were 14
// identical copies of these two functions scattered across src/gpu/*.rs and src/bin/*.rs.
use crate::gpu::core::{storage_entry, uniform_entry};

// ============================================================
// === BEVY RESOURCE                                        ===
// ============================================================

/// Bevy resource that keeps the GPU physics context between ticks.
///
/// Follows the `RetinaGpuRes` / `BrainGpuRes` pattern: `ctx: Option<PhysicsGpu>`, recreated
/// if the track changes (detected via world signature).
///
/// The caller must check `IARA_FISICA_CPU=1` (env var) to force CPU,
/// exactly as the retina uses `IARA_RETINA_CPU=1` — allows a controlled A/B.
#[derive(bevy::prelude::Resource, Default)]
pub struct PhysicsGpuRes {
    pub ctx: Option<PhysicsGpu>,
    /// World signature — detects track swaps and resets buffers.
    pub signature: u64,
}

impl PhysicsGpuRes {
    /// Returns a valid context or None (no GPU, falls back to CPU).
    ///
    /// Receives world data to create the static buffers. Recreates if the track changes.
    pub fn get_or_create(
        &mut self,
        world: &crate::world::VoxelWorld,
        signature: u64,
        capacity: usize,
    ) -> Option<&PhysicsGpu> {
        // GPU physics has its own device — does not share with the retina (devices are incompatible).
        // This is an accepted design cost: the overhead of two devices is small compared to the
        // parallelism gain (GPU physics runs while motor.rs computes channels on the CPU).
        let needs_update = self.ctx.is_none()
            || self.signature != signature
            || self
                .ctx
                .as_ref()
                .map(|c| c.capacity < capacity)
                .unwrap_or(true);
        if needs_update {
            // Converts VoxelType → u8 for the GPU grid buffer.
            let grid_u8: Vec<u8> = world.grid.iter().map(|v| *v as u8).collect();
            self.ctx = PhysicsGpu::new(
                &grid_u8,
                &world.height_map,
                world.size,
                world.cell_size,
                capacity.max(64),
            );
            self.signature = signature;
            if self.ctx.is_some() {
                println!(
                    "🏎 physics on GPU: {} beings (3 kernels physics_a/b/c.wgsl)",
                    capacity
                );
            } else {
                println!("⚠️  GPU physics failed — physics stays on the CPU");
            }
        }
        self.ctx.as_ref()
    }
}

#[cfg(test)]
mod state_tests {
    use super::*;

    #[test]
    fn body_contract_rejects_invalid_transport_fields() {
        let valid = make_physics_state([0.0, 0.5, 0.0], [1.0, 0.0, 0.0]);
        assert!(valid.validate_contract().is_ok());
        let mut invalid = valid;
        invalid.brake_cmd = 1.1;
        assert!(invalid.validate_contract().is_err());
        invalid = valid;
        invalid.mass = 0.0;
        assert!(invalid.validate_contract().is_err());
        invalid = valid;
        invalid.wheels[7][3] = f32::NAN;
        assert!(invalid.validate_contract().is_err());
    }

    #[test]
    fn output_contract_rejects_corrupt_readback() {
        let mut output = PhysicsOutputGpu::zeroed();
        assert!(output.validate_contract().is_ok());
        output.vel[2] = f32::INFINITY;
        assert!(output.validate_contract().is_err());
        output.vel[2] = 0.0;
        output.is_stalled = 2;
        assert!(output.validate_contract().is_err());
    }

    #[test]
    fn tyre_temperature_starts_at_normalised_optimum() {
        let state = make_physics_state([0.0, 0.5, 0.0], [1.0, 0.0, 0.0]);

        assert_eq!(state.optimal_temp.to_bits(), 0.5_f32.to_bits());
        for index in [1_usize, 3, 5, 7] {
            assert_eq!(
                state.wheels[index][0].to_bits(),
                state.optimal_temp.to_bits(),
                "wheel {} must use the same normalised temperature unit as the tyre kernel",
                index / 2
            );
        }
    }

    #[test]
    fn static_rest_shader_contract_parses_without_simulation() {
        let source = include_str!("physics_c.wgsl");
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
    }

    #[test]
    #[ignore = "exclusive real GPU fixture; never use a CPU adapter"]
    fn gpu_static_rest_matrix_real_adapter() {
        fn horizontal_speed(state: &PhysicsStateGpu) -> f32 {
            state.vel[0].hypot(state.vel[2])
        }
        fn run(gpu: &PhysicsGpu, mut state: PhysicsStateGpu, ticks: usize) -> PhysicsStateGpu {
            for _ in 0..ticks {
                assert!(gpu.step(std::slice::from_mut(&mut state)));
            }
            state
        }

        let size = 16usize;
        let grid = vec![0u8; size * size];
        let hmap = vec![0.0f32; size * size];
        let gpu = PhysicsGpu::new(&grid, &hmap, size, 1.0, 1).expect("real GPU adapter required");
        let adapter = gpu.adapter.to_ascii_lowercase();
        assert!(
            !adapter.contains("llvmpipe")
                && !adapter.contains("swiftshader")
                && !adapter.contains("software"),
            "CPU/software adapter forbidden: {}",
            gpu.adapter
        );
        eprintln!("[gpu-static-rest] adapter={}", gpu.adapter);

        let base = || {
            let mut state = make_physics_state([4.0, 0.5, 4.0], [1.0, 0.0, 0.0]);
            state.vel = [0.0376, 0.0, 0.0];
            state.brake_cmd = 1.0;
            state.throttle = 0.0;
            state.config_flags =
                pack_config_flags(true, true, true, true, true, true) | STATIC_REST_FLAG;
            state
        };

        let rest_on = run(&gpu, base(), 8);
        assert!(
            horizontal_speed(&rest_on) < 0.005,
            "rest constraint did not hold: {:?}",
            rest_on.vel
        );

        let mut rest_off_input = base();
        rest_off_input.config_flags &= !STATIC_REST_FLAG;
        let rest_off = run(&gpu, rest_off_input, 8);
        assert!(
            horizontal_speed(&rest_off) > 0.005,
            "flag-off path unexpectedly locked: {:?}",
            rest_off.vel
        );

        let mut no_brake = base();
        no_brake.brake_cmd = 0.0;
        let no_brake_out = run(&gpu, no_brake, 8);
        assert!(
            horizontal_speed(&no_brake_out) > 0.005,
            "brake-off path was frozen"
        );

        let mut weak_brake = base();
        weak_brake.brake_torque_max = 0.001;
        let weak_out = run(&gpu, weak_brake, 8);
        assert!(
            horizontal_speed(&weak_out) > 0.005,
            "inadequate brake was frozen"
        );

        let mut no_grip = base();
        no_grip.mu_peak = 0.0;
        let no_grip_out = run(&gpu, no_grip, 8);
        assert!(
            horizontal_speed(&no_grip_out) > 0.005,
            "zero-friction path was frozen"
        );

        let mut fast = base();
        fast.vel = [5.0, 0.0, 0.0];
        let fast_out = run(&gpu, fast, 1);
        assert!(horizontal_speed(&fast_out) > 1.0, "fast body was frozen");

        let mut airborne = base();
        airborne.pos[1] = 5.0;
        let airborne_out = run(&gpu, airborne, 1);
        assert!(
            horizontal_speed(&airborne_out) > 0.005,
            "airborne body was frozen"
        );
    }
}

#[cfg(test)]
mod contracts_layout_tests;
