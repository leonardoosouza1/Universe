//! Host wgpu para o kernel de física veicular — três kernels encadeados no mesmo CommandEncoder.
//!
//! # Por que três kernels?
//!
//! O shader original (`physics.wgsl`, 756 linhas) causava SIGSEGV no compilador ACO/NIR do
//! driver RADV NAVI22 (Mesa). A bissecção (`fisica_bisect`) isolou o problema:
//!
//!   MINIMO         → OK  (só copia)
//!   FASE_SUSP      → OK  (suspensão + arrays var<private>)
//!   FASE_POWER_A   → OK  (motor + RPM + switch drivetrain)
//!   FASE_POWER_B   → OK  (pneus SEM Pacejka — só cos/sin + omega)
//!   FASE_POWER_C   → CRASH (adiciona atan/atan2/sin + cálculo Pacejka)
//!
//! Tentativa anterior (2 kernels): physics_b.wgsl combinava pneus+colisão+integração e
//! ainda crashava — o Pacejka sozinho (FASE_POWER_C) já excede a capacidade do ACO.
//!
//! Solução final (3 kernels):
//!   Kernel A (physics_a.wgsl): suspensão + motor + aero    → escreve MidState (128B/carro)
//!   Kernel B (physics_b.wgsl): loop Pacejka (4 rodas)      → lê MidState, escreve MidState2 (64B/carro)
//!   Kernel C (physics_c.wgsl): colisão + integração        → lê MidState + MidState2, escreve saída
//!
//! Cada kernel usa ≤ 50 f32 locais simultâneos — abaixo do limiar de crash do ACO RDNA2.
//! Os três dispatches ficam no mesmo CommandEncoder — sem round-trip host↔GPU entre eles.
//!
//! # Overhead
//!
//! Dois dispatches extras por tick (64 threads, 1 workgroup para ≤60 carros). Medido como
//! insignificante comparado com a latência de round-trip CPU↔GPU do readback.
//!
//! # Tolerâncias declaradas (CPU vs GPU)
//!
//! Por passo único (dt ≈ 1/60 s):
//! - Força (float de passada simples): ≤ 2 ULP ou erro relativo ≤ 1e-4
//! - Slip ratio / slip angle: ≤ 2 ULP (dominado por atan, idêntico IEEE-754)
//! - Deriva acumulada por N ticks: medida e declarada em gpu_validate --fisica

use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

// ============================================================
// === STRUCTS DE DADOS (mesmos que antes)                   ===
// ============================================================

/// Estado de uma roda, empacotado em 2 vec4 (8 floats).
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

/// Bitmask dos flags de configuração.
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

/// Posição extraída de `PhysicsStateGpu`, para comparação de distância.
///
/// Deduplicado (E3): existia idêntica em `gpu_evolve.rs`, `gpu_loop.rs` e
/// `cortex_evolve.rs` — os três bins comparavam posição do mesmo jeito.
#[inline]
pub fn pos_of(p: &PhysicsStateGpu) -> [f32; 3] {
    p.pos
}

/// Constrói o `PhysicsStateGpu` inicial para um ser na posição `start`, olhando `fwd`.
///
/// Deduplicado (E3 — REFACTOR_PLAN §2.3): existiam TRÊS cópias byte-idênticas desta
/// função (`gpu_evolve.rs`, `gpu_loop.rs`, `cortex_evolve.rs`), confirmadas campo a
/// campo antes da unificação — nenhuma tinha divergido (v0=0.0 nos três: "se os seres
/// se moverem, o movimento é DECISÃO DO CÉREBRO", não velocidade injetada).
pub fn make_physics_state(start: [f32; 3], fwd: [f32; 3]) -> PhysicsStateGpu {
    let normal_load = 1350.0 * 9.81 / 4.0;
    let wheel = WheelGpu::from_cpu(0.0, 0.0, 0.0, normal_load, 60.0, 0.0, 0.0);
    let lat = [fwd[2], 0.0, -fwd[0]];
    // v0 = 0 m/s: sem velocidade injetada. Movimento = decisão do cérebro.
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
        max_torque: 620.0,
        current_rpm: 2000.0,
        max_rpm: 8500.0,
        throttle: 0.0,
        brake_cmd: 0.0,
        steer: 0.0,
        drive_torque_out: 0.0,
        current_gear: 3, // 2ª marcha (convenção: 0=ré, 1=neutro, 2=1ª ...)
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

/// Stride do array `input: array<CarInput>` no WGSL.
/// SizeOf(PhysicsStateGpu)=360, AlignOf=16 → stride=368.
pub(crate) const INPUT_WGSL_STRIDE: usize = 368;

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

/// Estado completo de um carro para um passo de GPU.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
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

/// Saída por carro após o kernel B.
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

/// Buffer intermediário entre Kernel A e Kernel B. 128 bytes/carro, stride=128.
/// Deve espelhar exatamente MidState em physics_a.wgsl, physics_b.wgsl e physics_c.wgsl.
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
    _pad2: [f32; 2],        //  8
                            // Total: 128 bytes — múltiplo de 16, stride = 128
}

/// Buffer intermediário entre Kernel B e Kernel C. 64 bytes/carro, stride=64.
/// Deve espelhar exatamente MidState2 em physics_b.wgsl e physics_c.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MidState2 {
    forces: [f32; 3], // 12  — forças totais (gravidade + suspensão + aero + pneus)
    torques_y: f32,   //  4  — torque yaw acumulado
    omegas: [f32; 4], // 16  — omegas pós-frenagem
    slip_r: [f32; 4], // 16  — slip ratios
    slip_a: [f32; 4], // 16  — slip angles
                      // Total: 64 bytes — múltiplo de 16, stride = 64
}

/// Parâmetros globais do mundo (uniform buffer).
///
/// D5 (ESCADA_DIFICULDADE.md — clima) acrescentou 4 campos ao final: `clima_mu`,
/// `vento_x`, `vento_z`, `clima_freq`. Apêndice seguro — os kernels vivos
/// (`physics_a/b/c.wgsl`) declaram um `WorldParams` mais curto (16 bytes) e
/// `min_binding_size: None` (ver `uniform_entry` em `core.rs`) permite buffer
/// maior que o binding declarado no shader: eles continuam lendo só os 16
/// bytes originais, byte-idêntico ao comportamento anterior a este commit.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WorldParams {
    grid_size: u32,
    n_beings: u32,
    cell_size: f32,
    _pad: f32,
    /// D5 — multiplicador global de grip: 1.0 = seco (default, bit-idêntico ao
    /// comportamento pré-D5), 0.7 = chuva. Consumido em `physics_power_c.wgsl`
    /// (único kernel do trio POWER que computa força de pneu).
    clima_mu: f32,
    /// D5 — vento lateral constante (N) aplicado ao corpo, eixo mundo X. Default 0.
    vento_x: f32,
    /// D5 — vento lateral constante (N) aplicado ao corpo, eixo mundo Z. Default 0.
    vento_z: f32,
    /// D5 — frequência espacial da variação de μ. 0 = μ espacialmente uniforme
    /// (default). >0 = trechos escorregadios determinísticos por posição.
    clima_freq: f32,
}

// ============================================================
// === CONTEXTO GPU PERSISTENTE                             ===
// ============================================================

/// Contexto wgpu para o kernel de física triplo. Persiste entre ticks como recurso Bevy.
/// Criado uma vez por pista; recriado se a pista mudar (detectado via assinatura).
///
/// Três pipelines compilados uma vez; por tick apenas write_buffer + 3 dispatches + readback.
pub struct PhysicsGpu {
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    /// Pipeline do Kernel A (suspensão + motor + aero)
    pipeline_a: wgpu::ComputePipeline,
    layout_a: wgpu::BindGroupLayout,
    /// Pipeline do Kernel B (pneus Pacejka — isolado para reduzir pressão de registradores)
    pipeline_b: wgpu::ComputePipeline,
    layout_b: wgpu::BindGroupLayout,
    /// Pipeline do Kernel C (colisão + integração)
    pipeline_c: wgpu::ComputePipeline,
    layout_c: wgpu::BindGroupLayout,
    /// Grade de voxels (estática por pista)
    buf_grid: wgpu::Buffer,
    /// Heightmap bilinear (estático por pista)
    buf_hmap: wgpu::Buffer,
    /// Buffer de entrada (escrito pelo host a cada tick)
    buf_input: wgpu::Buffer,
    /// Buffer intermediário A→B: suspensão + motor + forças parciais (128B/carro)
    buf_mid: wgpu::Buffer,
    /// Buffer intermediário B→C: forças de pneu + omegas + slip (64B/carro)
    buf_mid2: wgpu::Buffer,
    /// Buffer de saída do Kernel C
    buf_output: wgpu::Buffer,
    /// Buffer de readback (MAP_READ | COPY_DST)
    buf_readback: wgpu::Buffer,
    pub capacity: usize,
    pub adapter: String,
    grid_size: u32,
    cell_size: f32,
    /// D5 — clima: ver `set_clima`. Default 1.0 = seco, sem efeito (bit-idêntico).
    clima_mu: f32,
    /// D5 — vento lateral constante (N), eixo mundo X. Default 0.0.
    vento_x: f32,
    /// D5 — vento lateral constante (N), eixo mundo Z. Default 0.0.
    vento_z: f32,
    /// D5 — frequência espacial de μ. Default 0.0 = uniforme.
    clima_freq: f32,
}

impl PhysicsGpu {
    /// Abre dispositivo GPU e compila os dois kernels.
    /// Devolve `None` se não houver GPU utilizável.
    /// Abre um dispositivo PRÓPRIO. Mantido para os testes de equivalência, que rodam a
    /// física isolada. No laço de simulação use `new_shared` — ver o porquê lá.
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

        // Kernel B usa 6 bindings (params, grid, hmap, input, mid, output).
        // downlevel_defaults() permite max_storage_buffers_per_shader_stage=4 — insuficiente.
        // Elevamos para 8 (igual ao BrainGpu em brain.rs).
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

    /// Constrói sobre um dispositivo JÁ ABERTO, compartilhado com retina/percepção/cérebro.
    ///
    /// # Por que isto existe
    ///
    /// A física nasceu com dispositivo próprio para isolar um crash do compilador ACO do
    /// driver RADV. Esse crash foi resolvido de outro jeito — dividindo o shader em três
    /// kernels (`physics.wgsl` + `physics_b.wgsl` + `physics_c.wgsl`) —, então o isolamento
    /// deixou de ser necessário e passou a ATRAPALHAR: com dois `wgpu::Device` na mesma GPU,
    /// o laço travava no primeiro tick (`gpu_loop` ficava preso no aquecimento, sem erro e
    /// sem saída). Um dispositivo só resolve, e é também o que permite encadear os quatro
    /// dispatches num único `CommandEncoder` sem readback entre eles — que é o ponto do
    /// laço todo-GPU.
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
                storage_entry(4, false), // mid (escrita)
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
        // Apenas o loop Pacejka — sem colisão e sem integração para ficar abaixo do limite
        // de registradores ACO/RDNA2.
        let mod_b = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("physics-b"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("physics_b.wgsl"))),
        });
        let layout_b = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("physics-layout-b"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true), // grid  (declarado mas não usado em B — mantém layout uniforme)
                storage_entry(2, true), // hmap  (idem)
                storage_entry(3, true), // input
                storage_entry(4, true), // mid   (leitura — MidState de A)
                storage_entry(5, false), // mid2  (escrita — MidState2 para C)
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
                storage_entry(4, true),  // mid  (leitura — MidState de A)
                storage_entry(5, true),  // mid2 (leitura — MidState2 de B)
                storage_entry(6, false), // output (escrita)
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

        // Buffers estáticos por pista.
        // COPY_DST (além de STORAGE): permite `reload_world()` trocar a pista via
        // `queue.write_buffer` sem recriar buffer/bind-group/pipeline — ver o mesmo
        // raciocínio em `gpu::perception::PerceptionGpu::reload_world` (P1, rotação
        // de pistas — docs/PLANO_EVOLUCAO.md). Válido porque `grid_size`/`cell_size`
        // são constantes por construção (`VoxelWorld::new`, world.rs), então o
        // tamanho em bytes nunca muda entre pistas.
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
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // buf_mid: escrito por A, lido por B e C
        let buf_mid = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("physics-mid"),
            size: bytes_mid,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        // buf_mid2: escrito por B, lido por C
        let buf_mid2 = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("physics-mid2"),
            size: bytes_mid2,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let buf_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("physics-output"),
            size: bytes_output,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
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
            // D5 — defaults = comportamento pré-clima, bit-idêntico.
            clima_mu: 1.0,
            vento_x: 0.0,
            vento_z: 0.0,
            clima_freq: 0.0,
        })
    }

    /// D5 (ESCADA_DIFICULDADE.md) — ajusta clima: aderência (μ) global/por-trecho +
    /// vento lateral constante. Escrito no `WorldParams` (uniform) a cada `step()`.
    ///
    /// - `mu`: multiplicador de grip. 1.0 = seco (default, sem efeito). 0.7 = chuva.
    /// - `vento_x`/`vento_z`: força lateral constante (N) no corpo, eixos mundo X/Z.
    ///   Default 0.0 = sem vento.
    /// - `freq`: frequência espacial da variação de μ. 0.0 (default) = μ uniforme
    ///   (`clima_mu` aplicado igual em toda a pista). >0.0 = μ varia por posição
    ///   (trechos escorregadios determinísticos — ver `mu_local_at` no shader).
    ///
    /// Chamar com `(1.0, 0.0, 0.0, 0.0)` restaura o default — bit-idêntico ao
    /// comportamento anterior à existência deste método (nunca chamado = já é
    /// o default do construtor).
    pub fn set_clima(&mut self, mu: f32, vento_x: f32, vento_z: f32, freq: f32) {
        self.clima_mu = mu;
        self.vento_x = vento_x;
        self.vento_z = vento_z;
        self.clima_freq = freq;
    }

    /// Troca grade/heightmap para uma pista NOVA sem recriar buffer, bind group nem
    /// pipeline — só `queue.write_buffer` (P1: rotação de pistas por geração,
    /// docs/PLANO_EVOLUCAO.md). `device`/`queue` já são campos da struct, por isso
    /// não precisa recebê-los como parâmetro.
    ///
    /// `grid`/`hmap` têm de ter o mesmo tamanho com que este `PhysicsGpu` foi
    /// construído — verdade sempre que vêm de `VoxelWorld::new`, cujo `size` é
    /// constante independente da seed (world.rs). O `debug_assert!` documenta essa
    /// invariante; em release o custo é zero.
    pub fn reload_world(&self, grid: &[u8], hmap: &[f32]) {
        debug_assert_eq!(
            grid.len(),
            (self.grid_size * self.grid_size) as usize,
            "reload_world: grade nova tem tamanho diferente do buffer alocado"
        );
        debug_assert_eq!(
            hmap.len(),
            (self.grid_size * self.grid_size) as usize,
            "reload_world: heightmap novo tem tamanho diferente do buffer alocado"
        );
        let grid_u32: Vec<u32> = grid.iter().map(|v| *v as u32).collect();
        self.queue
            .write_buffer(&self.buf_grid, 0, bytemuck::cast_slice(&grid_u32));
        self.queue
            .write_buffer(&self.buf_hmap, 0, bytemuck::cast_slice(hmap));
    }

    /// Executa um passo de física para todos os carros em paralelo na GPU.
    ///
    /// Despacha Kernel A, B e C no mesmo CommandEncoder (sem round-trip).
    /// Retorna `false` se o número de carros exceder a capacidade.
    pub fn step(&self, state: &mut [PhysicsStateGpu]) -> bool {
        let n = state.len();
        if n == 0 {
            return true;
        }
        if n > self.capacity {
            return false;
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

        // Sobe estado dos carros com stride WGSL correto (368 bytes/carro)
        let input_bytes = serialize_input(state);
        self.queue.write_buffer(&self.buf_input, 0, &input_bytes);

        // Bind group do Kernel A: params, grid, hmap, input, mid
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

        // Bind group do Kernel B: params, grid, hmap, input, mid, mid2
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

        // Bind group do Kernel C: params, grid, hmap, input, mid, mid2, output
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
            // Kernel A: suspensão + motor + aero → escreve buf_mid
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.pipeline_a);
            p.set_bind_group(0, &group_a, &[]);
            p.dispatch_workgroups(wg, 1, 1);
        }
        {
            // Kernel B: loop Pacejka → lê buf_mid, escreve buf_mid2
            // Encerrar pass_a inicia barreira de memória global: buf_mid visível para B.
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.pipeline_b);
            p.set_bind_group(0, &group_b, &[]);
            p.dispatch_workgroups(wg, 1, 1);
        }
        {
            // Kernel C: colisão + integração → lê buf_mid + buf_mid2, escreve buf_output
            // Encerrar pass_b inicia barreira: buf_mid2 visível para C.
            let mut p = enc.begin_compute_pass(&Default::default());
            p.set_pipeline(&self.pipeline_c);
            p.set_bind_group(0, &group_c, &[]);
            p.dispatch_workgroups(wg, 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.buf_output, 0, &self.buf_readback, 0, bytes_output);
        self.queue.submit(Some(enc.finish()));

        // Leitura síncrona — igual ao padrão do RetinaGpu
        let slice = self.buf_readback.slice(0..bytes_output);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();

        {
            let data = slice.get_mapped_range();
            let outputs: &[PhysicsOutputGpu] = bytemuck::cast_slice(&data);
            for (i, s) in outputs[..n].iter().enumerate() {
                apply_output(&mut state[i], s);
            }
        }
        self.buf_readback.unmap();
        true
    }

    /// Diagnóstico: executa um passo e devolve os structs de saída RAW.
    pub fn step_raw(&self, state: &[PhysicsStateGpu]) -> Vec<PhysicsOutputGpu> {
        let n = state.len();
        if n == 0 || n > self.capacity {
            return vec![];
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
        let _ = rx.recv();
        let result = {
            let data = slice.get_mapped_range();
            let outputs: &[PhysicsOutputGpu] = bytemuck::cast_slice(&data);
            outputs[..n].to_vec()
        };
        self.buf_readback.unmap();
        result
    }
}

/// Aplica a saída do Kernel B de volta no PhysicsStateGpu.
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
}

// ============================================================
// === HELPERS DE BIND GROUP                                ===
// ============================================================
// Deduplicado para src/gpu/core.rs (E3) — fonte canônica única. Havia 14 cópias
// idênticas destas duas funções espalhadas por src/gpu/*.rs e src/bin/*.rs.
use crate::gpu::core::{storage_entry, uniform_entry};

// ============================================================
// === RECURSO BEVY                                         ===
// ============================================================

/// Recurso Bevy que mantém o contexto de física GPU entre ticks.
///
/// Segue o padrão de `RetinaGpuRes` / `BrainGpuRes`: `ctx: Option<PhysicsGpu>`, recriado se
/// a pista mudar (detectado via assinatura world).
///
/// O caller deve verificar `IARA_FISICA_CPU=1` (env var) para forçar CPU,
/// exatamente como retina usa `IARA_RETINA_CPU=1` — permite A/B controlado.
#[derive(bevy::prelude::Resource, Default)]
pub struct PhysicsGpuRes {
    pub ctx: Option<PhysicsGpu>,
    /// Assinatura do mundo — detecta troca de pista e reinicia buffers.
    pub signature: u64,
}

impl PhysicsGpuRes {
    /// Retorna contexto válido ou None (sem GPU, cai na CPU).
    ///
    /// Recebe dados de mundo para criar os buffers estáticos. Recria se a pista mudar.
    pub fn get_or_create(
        &mut self,
        world: &crate::world::VoxelWorld,
        signature: u64,
        capacity: usize,
    ) -> Option<&PhysicsGpu> {
        // Fisica GPU tem device próprio — não compartilha com retina (devices são incompatíveis).
        // Isso é um custo de design aceito: o overhead de dois devices é pequeno comparado ao
        // ganho de paralelismo (física GPU roda enquanto motor.rs calcula canais na CPU).
        let needs_update = self.ctx.is_none()
            || self.signature != signature
            || self
                .ctx
                .as_ref()
                .map(|c| c.capacity < capacity)
                .unwrap_or(true);
        if needs_update {
            // Converte VoxelType → u8 para o buffer de grade da GPU.
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
                    "🏎 física na GPU: {} seres (3 kernels physics_a/b/c.wgsl)",
                    capacity
                );
            } else {
                println!("⚠️  física GPU falhou — física continua na CPU");
            }
        }
        self.ctx.as_ref()
    }
}
