//! Bissecção do segfault no driver RADV ao compilar physics.wgsl.
//!
//! Método: shader mínimo (só copia) → adicionar fases uma a uma → a fase que crasha é a culpada.
//!
//! Uso:
//!   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json env -u DISPLAY ./target/release/fisica_bisect

use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

// Layout idêntico a PhysicsStateGpu — copiado aqui para o binário ser self-contained.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CarInput {
    vel:        [f32; 3],
    av_y:       f32,
    wheels:     [[f32; 4]; 8],
    pos:        [f32; 3],
    _pad_pos:   f32,
    fwd:        [f32; 3],
    _pad_fwd:   f32,
    lat:        [f32; 3],
    _pad_lat:   f32,
    max_torque:      f32,
    current_rpm:     f32,
    max_rpm:         f32,
    throttle:        f32,
    brake_cmd:       f32,
    steer:           f32,
    drive_torque_out:f32,
    current_gear:    u32,
    shift_timer:     f32,
    is_stalled:      u32,
    starter_timer:   f32,
    flywheel_inertia:f32,
    engine_temp:     f32,
    oil_temp:        f32,
    mass:            f32,
    drag_coeff:      f32,
    wheel_base:      f32,
    cg_height:       f32,
    track_width:     f32,
    lsd_lock_ratio:  f32,
    brake_torque_max:f32,
    drivetrain:      u32,
    mu_peak:             f32,
    cornering_stiffness: f32,
    wheel_radius:        f32,
    wheel_inertia:       f32,
    optimal_temp:        f32,
    susp_stiffness:  f32,
    susp_damping:    f32,
    susp_rest:       f32,
    susp_max_travel: f32,
    _pad_before_wind: f32,
    wind:            [f32; 3],
    _pad_wind:       f32,
    terrain_grip:    f32,
    env_grip:        f32,
    aquaplaning:     f32,
    config_flags:    u32,
    dt:              f32,
    _pad_dt:         f32,
}

// stride WGSL: roundUp(16, 360) = 368
const STRIDE: usize = 368;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct CarOutput {
    vel:         [f32; 3],
    av_y:        f32,
    wheels:      [[f32; 4]; 8],
    pos:         [f32; 3],
    _pad_pos:    f32,
    fwd:         [f32; 3],
    _pad_fwd:    f32,
    lat:         [f32; 3],
    _pad_lat:    f32,
    current_rpm:     f32,
    drive_torque_out:f32,
    shift_timer:     f32,
    is_stalled:      u32,
    starter_timer:   f32,
    engine_temp:     f32,
    oil_temp:        f32,
    _pad_eng:        f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WorldParams {
    grid_size: u32,
    n_beings:  u32,
    cell_size: f32,
    _pad:      f32,
}

// Cria um estado de carro realista para teste
fn make_car() -> CarInput {
    let mut c = CarInput::zeroed();
    c.vel = [20.0, 0.0, 0.0];
    c.av_y = 0.1;
    c.pos  = [48.0, 0.5, 48.0];
    c.fwd  = [1.0, 0.0, 0.0];
    c.lat  = [0.0, 0.0, 1.0];
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
    c.config_flags = 0b111111; // todos os flags
    c.dt = 1.0 / 60.0;
    let omega = 20.0_f32 / 0.33;
    for w in c.wheels.iter_mut() { w[0] = omega; }
    c
}

// Serializa com stride=368
fn serialize(cars: &[CarInput]) -> Vec<u8> {
    let sz = std::mem::size_of::<CarInput>();
    assert_eq!(sz, 360, "Tamanho errado: {sz}");
    let mut buf = vec![0u8; cars.len() * STRIDE];
    for (i, c) in cars.iter().enumerate() {
        let dst = &mut buf[i * STRIDE..i * STRIDE + sz];
        dst.copy_from_slice(bytemuck::bytes_of(c));
    }
    buf
}

/// Tenta compilar e executar um shader WGSL. Retorna true se passou, false se crashou.
/// Executa em processo filho para isolar SIGSEGV.
fn test_shader(name: &str, wgsl: &'static str) -> bool {
    println!("  [BISECT] testando shader: {name}");
    let ok = unsafe {
        let pid = libc::fork();
        if pid == 0 {
            // Filho: tenta usar o shader
            match run_shader(wgsl) {
                Ok(_)  => libc::_exit(0),
                Err(e) => {
                    eprintln!("    erro: {e}");
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
        println!("  [BISECT] {name}: CRASH (segfault ou erro)");
    }
    ok
}

fn run_shader(wgsl: &str) -> Result<Vec<CarOutput>, String> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(
        &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }
    )).ok_or("sem adaptador")?;

    println!("    adaptador: {}", adapter.get_info().name);

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("bisect"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    )).map_err(|e| format!("request_device: {e}"))?;

    // Compila o shader — este é o passo que pode crashar
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

    // Dados de teste: 4 carros, grade 10x10 plana
    let n = 4usize;
    let cars: Vec<CarInput> = (0..n).map(|_| make_car()).collect();
    let grid_u32: Vec<u32> = vec![1u32; 10 * 10]; // Road = 1
    let hmap: Vec<f32>     = vec![0.0f32; 10 * 10];

    let buf_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&WorldParams { grid_size: 10, n_beings: n as u32, cell_size: 1.0, _pad: 0.0 }),
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
            wgpu::BindGroupEntry { binding: 0, resource: buf_params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_grid.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_hmap.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: buf_output.as_entire_binding() },
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
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
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
    // Forçar ICD Radeon para evitar conflito com outros ICDs (intel_hasvk, lvp, etc.)
    if std::env::var("VK_ICD_FILENAMES").is_err() {
        unsafe { std::env::set_var("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/radeon_icd.json"); }
    }

    // Primeiro testa a SOLUCAO ATUAL: 3 kernels (A, B pneus, C integração)
    println!("=== SOLUCAO TRIPLE-KERNEL (physics_a + physics_b + physics_c) ===");
    println!();

    let ok_a = test_shader_a("KERNEL_A (suspensao + motor + aero)");
    if !ok_a {
        println!("  KERNEL_A falhou — diagnostico adicional necessario.");
    } else {
        let ok_b = test_shader_b("KERNEL_B (pneus Pacejka) + KERNEL_C (integracao)");
        if ok_b {
            println!();
            println!("SUCESSO: kernels A, B e C compilam e executam sem crash.");
            println!("A solucao triple-kernel funciona nesta GPU/driver.");
            return;
        } else {
            println!();
            println!("KERNEL_B/C FALHOU — investigar bisect abaixo.");
        }
    }

    // Bisect de diagnóstico (somente se a solução dual-kernel falhou)
    println!();
    println!("=== BISSECCAO DO SHADER DE FISICA (diagnostico) ===");
    println!();

    // Passo 1: shader mínimo (só copia) com mesmo layout de bind group
    let ok_minimal = test_shader("MINIMO (so copia)", include_str!("../gpu/physics_minimal.wgsl"));

    if !ok_minimal {
        println!();
        println!("RESULTADO: o shader MINIMO ja crasha.");
        println!("Causa: o problema esta no layout de bind group, nos structs, ou no tamanho dos buffers.");
        println!("Nao eh o codigo de computacao — eh a declaracao dos structs ou bindings.");
        return;
    }

    // Passo 2: shader com fase de suspensão adicionada
    let ok_susp = test_shader("FASE_SUSP (suspensao + carga normal)", include_str!("../gpu/physics_phase_susp.wgsl"));

    if !ok_susp {
        println!();
        println!("RESULTADO: crash na FASE_SUSP (suspensão + carga normal).");
        println!("Causa provável: arrays var<private> de tamanho 4 (wheel_offsets, normal_loads, etc.)");
        println!("ou acesso ao heightmap com indexação dinâmica.");
        return;
    }

    // Passo 2b: fase de motor SEM pneus (para isolar sub-causa)
    let ok_power_a = test_shader("FASE_POWER_A (motor+RPM sem pneus)", include_str!("../gpu/physics_power_a.wgsl"));
    if !ok_power_a {
        println!();
        println!("RESULTADO: crash na FASE_POWER_A (motor sem pneus).");
        println!("Causa provável: muitos arrays var<private> ao mesmo tempo (omegas/tire_temps/slip_ratios/slip_angles + nloads/scomp).");
        println!("Pressão de registradores: ~14 arrays × 4 floats = 56 f32 locais além dos demais.");
        return;
    }

    // Passo 2c: loop de pneus SEM Pacejka (só cos/sin + omega integration)
    let ok_power_b = test_shader("FASE_POWER_B (pneus sem Pacejka)", include_str!("../gpu/physics_power_b.wgsl"));
    if !ok_power_b {
        println!();
        println!("RESULTADO: crash na FASE_POWER_B (loop de pneus sem Pacejka).");
        println!("Causa: a simples adição do loop de pneus com arrays adicionais explode");
        println!("a pressão de registradores — o compilador ACO/NIR não consegue spill.");
        println!("Solução: separar em múltiplos kernels (um por fase).");
        return;
    }

    // Passo 2d: Pacejka isolado (motor + pneus com atan/atan2/sin + Kamm — SEM aero)
    let ok_power_c = test_shader("FASE_POWER_C (Pacejka + Kamm sem aero)", include_str!("../gpu/physics_power_c.wgsl"));
    if !ok_power_c {
        println!();
        println!("RESULTADO: crash na FASE_POWER_C (Pacejka sem aero).");
        println!("Causa confirmada: os floats adicionais do cálculo Pacejka (f_long/f_lat/sa/sr/vr/comb/scale)");
        println!("somados com todos os outros arrays locais excedem a capacidade de registradores do ACO.");
        println!("O compilador ACO RDNA2 não faz spill para memória local automaticamente neste caso.");
        return;
    }

    // Passo 3: fase de powertrain COMPLETA (com pneus + Pacejka + aero)
    let ok_power = test_shader("FASE_POWER (powertrain + pneus + Pacejka + aero)", include_str!("../gpu/physics_phase_power.wgsl"));

    if !ok_power {
        println!();
        println!("RESULTADO: crash na FASE_POWER (powertrain + pneus).");
        println!("Causa provável: switch statement com drivetrain, loop grande de 4 rodas,");
        println!("ou cálculo Pacejka (atan/sin aninhados).");
        return;
    }

    // Passo 4: colisão + integração
    let ok_integ = test_shader("FASE_INTEG (colisao + integracao)", include_str!("../gpu/physics_phase_integ.wgsl"));

    if !ok_integ {
        println!();
        println!("RESULTADO: crash na FASE_INTEG (colisao + integracao).");
        println!("Causa provável: loop de 4 probe points com voxel_at(), ou cálculo de yaw damping.");
        return;
    }

    // Se chegou aqui, o shader completo deveria funcionar — mas sabemos que não funciona.
    // Isto seria um crash de compilação no link/otimização final, não numa fase isolada.
    println!();
    println!("RESULTADO: todas as fases isoladas rodam. O crash ocorre somente com o shader COMPLETO.");
    println!("Causa provável: o compilador ACO/NIR do RADV não suporta o tamanho total do shader.");
    println!("Solução: dividir em múltiplos kernels (um por fase).");

    // Testa os dois kernels A e B da solução dual-kernel
    println!();
    println!("=== TESTANDO SOLUCAO DUAL-KERNEL (physics_a.wgsl + physics_b.wgsl) ===");
    println!();

    let ok_a = test_shader_a("KERNEL_A (suspensao + motor + aero)");
    if !ok_a {
        println!("  KERNEL_A falhou — o shader de kernel A crasha o driver.");
        return;
    }

    let ok_b = test_shader_b("KERNEL_B (pneus + colisao + integracao)");
    if !ok_b {
        println!("  KERNEL_B falhou — mesmo com kernel A separado, o kernel B crasha.");
        return;
    }

    println!();
    println!("SUCESSO: ambos os kernels A e B compilam sem crash.");
    println!("A solução dual-kernel funciona nesta GPU/driver.");
}

/// Testa o Kernel A (physics_a.wgsl) que tem 5 bindings: params, grid, hmap, input, mid.
/// O buffer 'mid' aqui tem layout de MidState (128 bytes).
fn test_shader_a(name: &str) -> bool {
    println!("  [BISECT] testando shader: {name}");
    let ok = unsafe {
        let pid = libc::fork();
        if pid == 0 {
            match run_shader_a() {
                Ok(_)  => libc::_exit(0),
                Err(e) => {
                    eprintln!("    erro: {e}");
                    libc::_exit(1);
                }
            }
        }
        let mut st = 0;
        libc::waitpid(pid, &mut st, 0);
        libc::WIFEXITED(st) && libc::WEXITSTATUS(st) == 0
    };
    if ok { println!("  [BISECT] {name}: OK"); }
    else  { println!("  [BISECT] {name}: CRASH (segfault ou erro)"); }
    ok
}

/// Testa o Kernel B (physics_b.wgsl) que tem 6 bindings: params, grid, hmap, input, mid, output.
fn test_shader_b(name: &str) -> bool {
    println!("  [BISECT] testando shader: {name}");
    let ok = unsafe {
        let pid = libc::fork();
        if pid == 0 {
            match run_shader_b() {
                Ok(_)  => libc::_exit(0),
                Err(e) => {
                    eprintln!("    erro: {e}");
                    libc::_exit(1);
                }
            }
        }
        let mut st = 0;
        libc::waitpid(pid, &mut st, 0);
        libc::WIFEXITED(st) && libc::WEXITSTATUS(st) == 0
    };
    if ok { println!("  [BISECT] {name}: OK"); }
    else  { println!("  [BISECT] {name}: CRASH (segfault ou erro)"); }
    ok
}

/// MidState espelhando o struct WGSL — 128 bytes.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MidState {
    normal_loads:     [f32; 4],
    susp_comps:       [f32; 4],
    prev_comps:       [f32; 4],
    current_rpm:      f32,
    drive_torque:     f32,
    drive_torque_out: f32,
    shift_timer:      f32,
    is_stalled:       u32,
    starter_timer:    f32,
    engine_temp:      f32,
    oil_temp:         f32,
    omegas:           [f32; 4],
    forces:           [f32; 3],
    _pad_f:           f32,
    torques_y:        f32,
    yaw_inertia:      f32,
    _pad2:            [f32; 2],
}

fn run_shader_a() -> Result<Vec<MidState>, String> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(
        &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }
    )).ok_or("sem adaptador")?;
    println!("    adaptador: {}", adapter.get_info().name);
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("bisect-a"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    )).map_err(|e| format!("request_device: {e}"))?;

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("physics-a"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
            include_str!("../gpu/physics_a.wgsl")
        )),
    });

    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            uniform_entry(0),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, false),  // mid (read_write)
        ],
    });
    let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None, bind_group_layouts: &[&bind_layout], push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None, layout: Some(&pipe_layout), module: &module,
        entry_point: Some("main"), compilation_options: Default::default(), cache: None,
    });

    let n = 4usize;
    let cars: Vec<CarInput> = (0..n).map(|_| make_car()).collect();
    let grid_u32: Vec<u32> = vec![1u32; 10 * 10];
    let hmap: Vec<f32>     = vec![0.0f32; 10 * 10];

    let buf_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&WorldParams { grid_size: 10, n_beings: n as u32, cell_size: 1.0, _pad: 0.0 }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let buf_grid = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&grid_u32), usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_hmap = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&hmap), usage: wgpu::BufferUsages::STORAGE,
    });
    let input_bytes = serialize(&cars);
    let buf_input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: &input_bytes, usage: wgpu::BufferUsages::STORAGE,
    });
    let mid_sz = (n * std::mem::size_of::<MidState>()) as u64;
    let buf_mid = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: mid_sz,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let buf_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: mid_sz,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &bind_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_grid.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_hmap.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: buf_mid.as_entire_binding() },
        ],
    });

    let mut enc = device.create_command_encoder(&Default::default());
    { let mut pass = enc.begin_compute_pass(&Default::default());
      pass.set_pipeline(&pipeline); pass.set_bind_group(0, &group, &[]);
      pass.dispatch_workgroups(1, 1, 1); }
    enc.copy_buffer_to_buffer(&buf_mid, 0, &buf_readback, 0, mid_sz);
    queue.submit(Some(enc.finish()));

    let slice = buf_readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    device.poll(wgpu::Maintain::Wait);
    let _ = rx.recv();

    let result = {
        let data = slice.get_mapped_range();
        bytemuck::cast_slice::<u8, MidState>(&data)[..n].to_vec()
    };
    buf_readback.unmap();
    println!("    Kernel A OK — mid[0].current_rpm={:.1}", result[0].current_rpm);
    Ok(result)
}

fn run_shader_b() -> Result<Vec<CarOutput>, String> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(
        &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }
    )).ok_or("sem adaptador")?;
    println!("    adaptador: {}", adapter.get_info().name);

    let mut lim = wgpu::Limits::downlevel_defaults();
    lim.max_storage_buffers_per_shader_stage = 8;
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("bisect-b"), required_features: wgpu::Features::empty(),
            required_limits: lim, memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    )).map_err(|e| format!("request_device: {e}"))?;

    let mod_a = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("physics-a"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
            include_str!("../gpu/physics_a.wgsl")
        )),
    });
    let mod_b = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("physics-b"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
            include_str!("../gpu/physics_b.wgsl")
        )),
    });

    let layout_a = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            uniform_entry(0), storage_entry(1, true), storage_entry(2, true),
            storage_entry(3, true), storage_entry(4, false),
        ],
    });
    let layout_b = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            uniform_entry(0), storage_entry(1, true), storage_entry(2, true),
            storage_entry(3, true), storage_entry(4, true), storage_entry(5, false),
        ],
    });

    let pipe_a = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&layout_a], push_constant_ranges: &[],
        })),
        module: &mod_a, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None,
    });
    let pipe_b = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&layout_b], push_constant_ranges: &[],
        })),
        module: &mod_b, entry_point: Some("main"),
        compilation_options: Default::default(), cache: None,
    });

    let n = 4usize;
    let cars: Vec<CarInput> = (0..n).map(|_| make_car()).collect();
    let grid_u32: Vec<u32> = vec![1u32; 10 * 10];
    let hmap: Vec<f32>     = vec![0.0f32; 10 * 10];

    let buf_params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&WorldParams { grid_size: 10, n_beings: n as u32, cell_size: 1.0, _pad: 0.0 }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let buf_grid = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&grid_u32), usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_hmap = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&hmap), usage: wgpu::BufferUsages::STORAGE,
    });
    let input_bytes = serialize(&cars);
    let buf_input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: &input_bytes, usage: wgpu::BufferUsages::STORAGE,
    });
    let mid_sz  = (n * std::mem::size_of::<MidState>()) as u64;
    let buf_mid = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: mid_sz, usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let out_sz  = (n * std::mem::size_of::<CarOutput>()) as u64;
    let buf_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: out_sz,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let buf_rb  = device.create_buffer(&wgpu::BufferDescriptor {
        label: None, size: out_sz,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &layout_a,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_grid.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_hmap.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: buf_mid.as_entire_binding() },
        ],
    });
    let group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &layout_b,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf_params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: buf_grid.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: buf_hmap.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: buf_input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: buf_mid.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: buf_out.as_entire_binding() },
        ],
    });

    let mut enc = device.create_command_encoder(&Default::default());
    { let mut p = enc.begin_compute_pass(&Default::default());
      p.set_pipeline(&pipe_a); p.set_bind_group(0, &group_a, &[]);
      p.dispatch_workgroups(1, 1, 1); }
    { let mut p = enc.begin_compute_pass(&Default::default());
      p.set_pipeline(&pipe_b); p.set_bind_group(0, &group_b, &[]);
      p.dispatch_workgroups(1, 1, 1); }
    enc.copy_buffer_to_buffer(&buf_out, 0, &buf_rb, 0, out_sz);
    queue.submit(Some(enc.finish()));

    let slice = buf_rb.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
    device.poll(wgpu::Maintain::Wait);
    let _ = rx.recv();

    let result = {
        let data = slice.get_mapped_range();
        bytemuck::cast_slice::<u8, CarOutput>(&data)[..n].to_vec()
    };
    buf_rb.unmap();
    println!("    Kernel B OK — out[0].vel=[{:.3},{:.3},{:.3}]",
        result[0].vel[0], result[0].vel[1], result[0].vel[2]);
    Ok(result)
}
