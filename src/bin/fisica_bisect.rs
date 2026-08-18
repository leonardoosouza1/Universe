//! Bisection of the segfault in the RADV driver when compiling physics.wgsl.
//!
//! Method: minimal shader (copy only) → add phases one by one → the phase that crashes is the culprit.
//!
//! Usage:
//!   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json env -u DISPLAY ./target/release/fisica_bisect

use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

// Layout identical to PhysicsStateGpu — copied here so the binary is self-contained.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CarInput {
    vel: [f32; 3],
    av_y: f32,
    wheels: [[f32; 4]; 8],
    pos: [f32; 3],
    _pad_pos: f32,
    fwd: [f32; 3],
    _pad_fwd: f32,
    lat: [f32; 3],
    _pad_lat: f32,
    max_torque: f32,
    current_rpm: f32,
    max_rpm: f32,
    throttle: f32,
    brake_cmd: f32,
    steer: f32,
    drive_torque_out: f32,
    current_gear: u32,
    shift_timer: f32,
    is_stalled: u32,
    starter_timer: f32,
    flywheel_inertia: f32,
    engine_temp: f32,
    oil_temp: f32,
    mass: f32,
    drag_coeff: f32,
    wheel_base: f32,
    cg_height: f32,
    track_width: f32,
    lsd_lock_ratio: f32,
    brake_torque_max: f32,
    drivetrain: u32,
    mu_peak: f32,
    cornering_stiffness: f32,
    wheel_radius: f32,
    wheel_inertia: f32,
    optimal_temp: f32,
    susp_stiffness: f32,
    susp_damping: f32,
    susp_rest: f32,
    susp_max_travel: f32,
    _pad_before_wind: f32,
    wind: [f32; 3],
    _pad_wind: f32,
    terrain_grip: f32,
    env_grip: f32,
    aquaplaning: f32,
    config_flags: u32,
    dt: f32,
    _pad_dt: f32,
}

// stride WGSL: roundUp(16, 360) = 368
const STRIDE: usize = 368;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct CarOutput {
    vel: [f32; 3],
    av_y: f32,
    wheels: [[f32; 4]; 8],
    pos: [f32; 3],
    _pad_pos: f32,
    fwd: [f32; 3],
    _pad_fwd: f32,
    lat: [f32; 3],
    _pad_lat: f32,
    current_rpm: f32,
    drive_torque_out: f32,
    shift_timer: f32,
    is_stalled: u32,
    starter_timer: f32,
    engine_temp: f32,
    oil_temp: f32,
    _pad_eng: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WorldParams {
    grid_size: u32,
    n_beings: u32,
    cell_size: f32,
    _pad: f32,
}

// Creates a realistic car state for testing
fn make_car() -> CarInput {
    let mut c = CarInput::zeroed();
    c.vel = [20.0, 0.0, 0.0];
    c.av_y = 0.1;
    c.pos = [48.0, 0.5, 48.0];
    c.fwd = [1.0, 0.0, 0.0];
    c.lat = [0.0, 0.0, 1.0];
    c.max_torque = 620.0;
    c.current_rpm = 4000.0;
    c.max_rpm = 8500.0;
    c.throttle = 0.5;
    c.current_gear = 3;
    c.flywheel_inertia = 0.15;
    c.engine_temp = 90.0;
    c.oil_temp = 85.0;
    c.mass = 1350.0;
    c.drag_coeff = 0.32;
    c.wheel_base = 2.65;
    c.cg_height = 0.45;
    c.track_width = 1.55;
    c.lsd_lock_ratio = 0.3;
    c.brake_torque_max = 4000.0;
    c.drivetrain = 1;
    c.mu_peak = 1.6;
    c.cornering_stiffness = 55000.0;
    c.wheel_radius = 0.33;
    c.wheel_inertia = 1.5;
    c.optimal_temp = 0.5;
    c.susp_stiffness = 45000.0;
    c.susp_damping = 3000.0;
    c.susp_rest = 0.25;
    c.susp_max_travel = 0.12;
    c.terrain_grip = 1.0;
    c.env_grip = 1.0;
    c.aquaplaning = 1.0;
    c.config_flags = 0b111111; // all flags
    c.dt = 1.0 / 60.0;
    let omega = 20.0_f32 / 0.33;
    for w in c.wheels.iter_mut() {
        w[0] = omega;
    }
    c
}

// Serializes with stride=368
fn serialize(cars: &[CarInput]) -> Vec<u8> {
    let sz = std::mem::size_of::<CarInput>();
    assert_eq!(sz, 360, "Wrong size: {sz}");
    let mut buf = vec![0u8; cars.len() * STRIDE];
    for (i, c) in cars.iter().enumerate() {
        let dst = &mut buf[i * STRIDE..i * STRIDE + sz];
        dst.copy_from_slice(bytemuck::bytes_of(c));
    }
    buf
}

/// Tries to compile and run a WGSL shader. Returns true if it passed, false if it crashed.
/// Runs in a child process to isolate SIGSEGV.
fn test_shader(name: &str, wgsl: &'static str) -> bool {
    println!("  [BISECT] testing shader: {name}");
    let ok = unsafe {
        let pid = libc::fork();
        if pid == 0 {
            // Child: tries to use the shader
            match run_shader(wgsl) {
                Ok(_) => libc::_exit(0),
                Err(e) => {
                    eprintln!("    error: {e}");
                    libc::_exit(1);
                }
            }
        }
        let mut st = 0;
        libc::waitpid(pid, &mut st, 0);
        libc::WIFEXITED(st) && libc::WEXITSTATUS(st) == 0
    };
    if ok {
        println!("  [BISECT] {name}: OK");
    } else {
        println!("  [BISECT] {name}: CRASH (segfault or error)");
    }
    ok
}

fn run_shader(wgsl: &str) -> Result<Vec<CarOutput>, String> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or("no adapter")?;

    println!("    adapter: {}", adapter.get_info().name);

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("bisect"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .map_err(|e| format!("request_device: {e}"))?;

    // Compiles the shader — this is the step that can crash
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bisect-shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl)),
    });

    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            uniform_entry(0),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, false),
        ],
    });
    let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("bisect-pipeline"),
        layout: Some(&pipe_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Test data: 4 cars, flat 10x10 grid
    let n = 4usize;
    let cars: Vec<CarInput> = (0..n).map(|_| make_car()).collect();
    let grid_u32: Vec<u32> = vec![1u32; 10 * 10]; // Road = 1
    let hmap: Vec<f32> = vec![0.0f32; 10 * 10];

    let buf_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&WorldParams {
            grid_size: 10,
            n_beings: n as u32,
            cell_size: 1.0,
            _pad: 0.0,
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let buf_grid = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&grid_u32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_hmap = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&hmap),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let input_bytes = serialize(&cars);
    let buf_input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &input_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let out_sz = n * std::mem::size_of::<CarOutput>();
    let buf_output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: out_sz as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let buf_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: out_sz as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_grid.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_hmap.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buf_input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buf_output.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&buf_output, 0, &buf_readback, 0, out_sz as u64);
    queue.submit(Some(enc.finish()));

    let slice = buf_readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    let _ = rx.recv();

    let result = {
        let data = slice.get_mapped_range();
        bytemuck::cast_slice::<u8, CarOutput>(&data)[..n].to_vec()
    };
    buf_readback.unmap();
    Ok(result)
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

fn main() {
    // Force the Radeon ICD to avoid conflicts with other ICDs (intel_hasvk, lvp, etc.)
    if std::env::var("VK_ICD_FILENAMES").is_err() {
        unsafe {
            std::env::set_var(
                "VK_ICD_FILENAMES",
                "/usr/share/vulkan/icd.d/radeon_icd.json",
            );
        }
    }

    // First tests the CURRENT SOLUTION: 3 kernels (A, B tyres, C integration)
    println!("=== TRIPLE-KERNEL SOLUTION (physics_a + physics_b + physics_c) ===");
    println!();

    let ok_a = test_shader_a("KERNEL_A (suspension + engine + aero)");
    if !ok_a {
        println!("  KERNEL_A failed — additional diagnosis needed.");
    } else {
        let ok_b = test_shader_b("KERNEL_B (Pacejka tyres) + KERNEL_C (integration)");
        if ok_b {
            println!();
            println!("SUCCESS: kernels A, B and C compile and run without crashing.");
            println!("The triple-kernel solution works on this GPU/driver.");
            return;
        } else {
            println!();
            println!("KERNEL_B/C FAILED — investigate the bisect below.");
        }
    }

    // Diagnostic bisect (only if the dual-kernel solution failed)
    println!();
    println!("=== PHYSICS SHADER BISECTION (diagnostics) ===");
    println!();

    // Step 1: minimal shader (copy only) with the same bind group layout
    let ok_minimal = test_shader(
        "MINIMO (copy only)",
        include_str!("../gpu/physics_minimal.wgsl"),
    );

    if !ok_minimal {
        println!();
        println!("RESULT: the MINIMAL shader already crashes.");
        println!(
            "Cause: the problem is in the bind group layout, the structs, or the buffer sizes."
        );
        println!("It is not the compute code — it is the declaration of the structs or bindings.");
        return;
    }

    // Step 2: shader with the suspension phase added
    let ok_susp = test_shader(
        "FASE_SUSP (suspension + normal load)",
        include_str!("../gpu/physics_phase_susp.wgsl"),
    );

    if !ok_susp {
        println!();
        println!("RESULT: crash in FASE_SUSP (suspension + normal load).");
        println!(
            "Probable cause: var<private> arrays of size 4 (wheel_offsets, normal_loads, etc.)"
        );
        println!("or heightmap access with dynamic indexing.");
        return;
    }

    // Step 2b: engine phase WITHOUT tyres (to isolate the sub-cause)
    let ok_power_a = test_shader(
        "FASE_POWER_A (engine+RPM without tyres)",
        include_str!("../gpu/physics_power_a.wgsl"),
    );
    if !ok_power_a {
        println!();
        println!("RESULT: crash in FASE_POWER_A (engine without tyres).");
        println!(
            "Probable cause: many var<private> arrays at once (omegas/tire_temps/slip_ratios/slip_angles + nloads/scomp)."
        );
        println!("Register pressure: ~14 arrays × 4 floats = 56 local f32s besides the rest.");
        return;
    }

    // Step 2c: tyre loop WITHOUT Pacejka (only cos/sin + omega integration)
    let ok_power_b = test_shader(
        "FASE_POWER_B (tyres without Pacejka)",
        include_str!("../gpu/physics_power_b.wgsl"),
    );
    if !ok_power_b {
        println!();
        println!("RESULT: crash in FASE_POWER_B (tyre loop without Pacejka).");
        println!("Cause: merely adding the tyre loop with additional arrays explodes");
        println!("register pressure — the ACO/NIR compiler cannot spill.");
        println!("Solution: split into multiple kernels (one per phase).");
        return;
    }

    // Step 2d: isolated Pacejka (engine + tyres with atan/atan2/sin + Kamm — WITHOUT aero)
    let ok_power_c = test_shader(
        "FASE_POWER_C (Pacejka + Kamm without aero)",
        include_str!("../gpu/physics_power_c.wgsl"),
    );
    if !ok_power_c {
        println!();
        println!("RESULT: crash in FASE_POWER_C (Pacejka without aero).");
        println!(
            "Confirmed cause: the additional floats of the Pacejka computation (f_long/f_lat/sa/sr/vr/comb/scale)"
        );
        println!("added to all the other local arrays exceed the ACO's register capacity.");
        println!(
            "The ACO RDNA2 compiler does not spill to local memory automatically in this case."
        );
        return;
    }

    // Step 3: FULL powertrain phase (with tyres + Pacejka + aero)
    let ok_power = test_shader(
        "FASE_POWER (powertrain + tyres + Pacejka + aero)",
        include_str!("../gpu/physics_phase_power.wgsl"),
    );

    if !ok_power {
        println!();
        println!("RESULT: crash in FASE_POWER (powertrain + tyres).");
        println!("Probable cause: switch statement with drivetrain, big 4-wheel loop,");
        println!("or the Pacejka computation (nested atan/sin).");
        return;
    }

    // Step 4: collision + integration
    let ok_integ = test_shader(
        "FASE_INTEG (collision + integration)",
        include_str!("../gpu/physics_phase_integ.wgsl"),
    );

    if !ok_integ {
        println!();
        println!("RESULT: crash in FASE_INTEG (collision + integration).");
        println!(
            "Probable cause: 4-probe-point loop with voxel_at(), or the yaw damping computation."
        );
        return;
    }

    // If we got here, the full shader should work — but we know it does not.
    // This would be a compilation crash in the final link/optimization, not in an isolated phase.
    println!();
    println!("RESULT: all isolated phases run. The crash happens only with the FULL shader.");
    println!("Probable cause: the RADV ACO/NIR compiler does not support the shader's total size.");
    println!("Solution: split into multiple kernels (one per phase).");

    // Tests the two kernels A and B of the dual-kernel solution
    println!();
    println!("=== TESTING DUAL-KERNEL SOLUTION (physics_a.wgsl + physics_b.wgsl) ===");
    println!();

    let ok_a = test_shader_a("KERNEL_A (suspension + engine + aero)");
    if !ok_a {
        println!("  KERNEL_A failed — the kernel A shader crashes the driver.");
        return;
    }

    let ok_b = test_shader_b("KERNEL_B (tyres + collision + integration)");
    if !ok_b {
        println!("  KERNEL_B failed — even with kernel A separate, kernel B crashes.");
        return;
    }

    println!();
    println!("SUCCESS: both kernels A and B compile without crashing.");
    println!("The dual-kernel solution works on this GPU/driver.");
}

/// Tests Kernel A (physics_a.wgsl), which has 5 bindings: params, grid, hmap, input, mid.
/// The 'mid' buffer here has the MidState layout (128 bytes).
fn test_shader_a(name: &str) -> bool {
    println!("  [BISECT] testing shader: {name}");
    let ok = unsafe {
        let pid = libc::fork();
        if pid == 0 {
            match run_shader_a() {
                Ok(_) => libc::_exit(0),
                Err(e) => {
                    eprintln!("    error: {e}");
                    libc::_exit(1);
                }
            }
        }
        let mut st = 0;
        libc::waitpid(pid, &mut st, 0);
        libc::WIFEXITED(st) && libc::WEXITSTATUS(st) == 0
    };
    if ok {
        println!("  [BISECT] {name}: OK");
    } else {
        println!("  [BISECT] {name}: CRASH (segfault or error)");
    }
    ok
}

/// Tests Kernel B (physics_b.wgsl), which has 6 bindings: params, grid, hmap, input, mid, output.
fn test_shader_b(name: &str) -> bool {
    println!("  [BISECT] testing shader: {name}");
    let ok = unsafe {
        let pid = libc::fork();
        if pid == 0 {
            match run_shader_b() {
                Ok(_) => libc::_exit(0),
                Err(e) => {
                    eprintln!("    error: {e}");
                    libc::_exit(1);
                }
            }
        }
        let mut st = 0;
        libc::waitpid(pid, &mut st, 0);
        libc::WIFEXITED(st) && libc::WEXITSTATUS(st) == 0
    };
    if ok {
        println!("  [BISECT] {name}: OK");
    } else {
        println!("  [BISECT] {name}: CRASH (segfault or error)");
    }
    ok
}

/// MidState espelhando o struct WGSL — 128 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MidState {
    normal_loads: [f32; 4],
    susp_comps: [f32; 4],
    prev_comps: [f32; 4],
    current_rpm: f32,
    drive_torque: f32,
    drive_torque_out: f32,
    shift_timer: f32,
    is_stalled: u32,
    starter_timer: f32,
    engine_temp: f32,
    oil_temp: f32,
    omegas: [f32; 4],
    forces: [f32; 3],
    _pad_f: f32,
    torques_y: f32,
    yaw_inertia: f32,
    _pad2: [f32; 2],
}

fn run_shader_a() -> Result<Vec<MidState>, String> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or("no adapter")?;
    println!("    adapter: {}", adapter.get_info().name);
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("bisect-a"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .map_err(|e| format!("request_device: {e}"))?;

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("physics-a"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "../gpu/physics_a.wgsl"
        ))),
    });

    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            uniform_entry(0),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, false), // mid (read_write)
        ],
    });
    let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipe_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let n = 4usize;
    let cars: Vec<CarInput> = (0..n).map(|_| make_car()).collect();
    let grid_u32: Vec<u32> = vec![1u32; 10 * 10];
    let hmap: Vec<f32> = vec![0.0f32; 10 * 10];

    let buf_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&WorldParams {
            grid_size: 10,
            n_beings: n as u32,
            cell_size: 1.0,
            _pad: 0.0,
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let buf_grid = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&grid_u32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_hmap = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&hmap),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let input_bytes = serialize(&cars);
    let buf_input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &input_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let mid_sz = (n * std::mem::size_of::<MidState>()) as u64;
    let buf_mid = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: mid_sz,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let buf_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: mid_sz,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_grid.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_hmap.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buf_input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buf_mid.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&buf_mid, 0, &buf_readback, 0, mid_sz);
    queue.submit(Some(enc.finish()));

    let slice = buf_readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    let _ = rx.recv();

    let result = {
        let data = slice.get_mapped_range();
        bytemuck::cast_slice::<u8, MidState>(&data)[..n].to_vec()
    };
    buf_readback.unmap();
    println!(
        "    Kernel A OK — mid[0].current_rpm={:.1}",
        result[0].current_rpm
    );
    Ok(result)
}

fn run_shader_b() -> Result<Vec<CarOutput>, String> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or("no adapter")?;
    println!("    adapter: {}", adapter.get_info().name);

    let mut lim = wgpu::Limits::downlevel_defaults();
    lim.max_storage_buffers_per_shader_stage = 8;
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("bisect-b"),
            required_features: wgpu::Features::empty(),
            required_limits: lim,
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .map_err(|e| format!("request_device: {e}"))?;

    let mod_a = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("physics-a"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "../gpu/physics_a.wgsl"
        ))),
    });
    let mod_b = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("physics-b"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
            "../gpu/physics_b.wgsl"
        ))),
    });

    let layout_a = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            uniform_entry(0),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, false),
        ],
    });
    let layout_b = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            uniform_entry(0),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, true),
            storage_entry(5, false),
        ],
    });

    let pipe_a = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&layout_a],
                push_constant_ranges: &[],
            }),
        ),
        module: &mod_a,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let pipe_b = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&layout_b],
                push_constant_ranges: &[],
            }),
        ),
        module: &mod_b,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let n = 4usize;
    let cars: Vec<CarInput> = (0..n).map(|_| make_car()).collect();
    let grid_u32: Vec<u32> = vec![1u32; 10 * 10];
    let hmap: Vec<f32> = vec![0.0f32; 10 * 10];

    let buf_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&WorldParams {
            grid_size: 10,
            n_beings: n as u32,
            cell_size: 1.0,
            _pad: 0.0,
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let buf_grid = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&grid_u32),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_hmap = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&hmap),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let input_bytes = serialize(&cars);
    let buf_input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &input_bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let mid_sz = (n * std::mem::size_of::<MidState>()) as u64;
    let buf_mid = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: mid_sz,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let out_sz = (n * std::mem::size_of::<CarOutput>()) as u64;
    let buf_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: out_sz,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let buf_rb = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: out_sz,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout_a,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_grid.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_hmap.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buf_input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buf_mid.as_entire_binding(),
            },
        ],
    });
    let group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout_b,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_grid.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_hmap.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buf_input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buf_mid.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: buf_out.as_entire_binding(),
            },
        ],
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut p = enc.begin_compute_pass(&Default::default());
        p.set_pipeline(&pipe_a);
        p.set_bind_group(0, &group_a, &[]);
        p.dispatch_workgroups(1, 1, 1);
    }
    {
        let mut p = enc.begin_compute_pass(&Default::default());
        p.set_pipeline(&pipe_b);
        p.set_bind_group(0, &group_b, &[]);
        p.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&buf_out, 0, &buf_rb, 0, out_sz);
    queue.submit(Some(enc.finish()));

    let slice = buf_rb.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    let _ = rx.recv();

    let result = {
        let data = slice.get_mapped_range();
        bytemuck::cast_slice::<u8, CarOutput>(&data)[..n].to_vec()
    };
    buf_rb.unmap();
    println!(
        "    Kernel B OK — out[0].vel=[{:.3},{:.3},{:.3}]",
        result[0].vel[0], result[0].vel[1], result[0].vel[2]
    );
    Ok(result)
}
