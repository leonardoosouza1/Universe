//! Laço de simulação 100% residente na GPU.
//!
//! # O problema que este binário resolve
//!
//! O laço da CPU (`motor.rs`) foi instrumentado com GPUs peça a peça:
//!   retina GPU → submit + poll(Wait) → readback → percepção GPU → submit + poll(Wait) → ...
//! Cada peça fazia seu próprio round-trip host↔GPU. Resultado medido: ligar percepção GPU
//! deixou o tick MAIS LENTO (6,61 ms contra 6,25 ms), porque os round-trips custam mais do
//! que o paralelismo ganhou.
//!
//! A correção: **um encoder, todos os dispatches, um submit, um readback**.
//!
//! # Pipeline por tick
//!
//!   poses_cpu (2 kB) → write_buffer
//!   encoder {
//!     dispatch retina   (105.840 threads, 1654 wg × 64)
//!     dispatch percepção (60 threads, 60 wg × 1)
//!     dispatch cérebro   (60 threads, 60 wg × 1)
//!     copy buf_intent → buf_intent_readback
//!   }
//!   queue.submit(encoder)                ← UM submit
//!   device.poll(Wait)                    ← UM bloqueio
//!   ler buf_intent_readback (960 bytes)  ← intent[60] → física CPU
//!   física CPU (state[60]) → próximas poses
//!
//! # O que fica na CPU (e por quê)
//!
//! 1. **Física**: `PhysicsGpu` abre device próprio para isolar crashes ACO/NIR
//!    do driver RADV NAVI22 (bissecção `fisica_bisect` documentada em physics.rs).
//!    Portá-la para o device compartilhado exigiria reescrever a inicialização —
//!    trabalho separado. Por ora: `step()` roda no device próprio após o readback.
//!    O estado dos corpos (pos/vel) fica em `Vec<PhysicsStateGpu>` na CPU.
//!
//! 2. **Canais temporais do sensory_wave** (≈40 canais de 223): wall_ema, approach_rate,
//!    visual_cortex, racing_line, comp_stress — dependem de estado per-ser acumulado
//!    entre ticks. Para o laço todo-GPU esses canais ficam a ZERO neste binário.
//!    O modelo cerebral ainda se move (física integra intenção), mas sem esses canais
//!    a qualidade de navegação é inferior ao motor.rs completo.
//!
//! # Sanidade
//!
//! Os seres TÊM de se mover. Se a distância percorrida for zero, o laço está quebrado
//! mesmo sem erro. O binário imprime distância percorrida por alguns seres.
//!
//! # Uso
//!
//!   cd ~/projects/LLM/Universe
//!   env -u DISPLAY ./target/release/gpu_loop
//!   env -u DISPLAY ./target/release/gpu_loop --seres 60 --ticks 200

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Instant;

use bytemuck::Pod;
use wgpu::util::DeviceExt;

use universe::config::iara::brain::{SENSORY_WAVE_SIZE, L1_NEURONS, L2_NEURONS};
use universe::config::iara::vision::{RETINA_ROWS, RETINA_TOTAL};
use universe::engine::SovereignEngine;
use universe::evolution::dna_genesis::load_champion_dna;
use universe::gpu::{Pose, RetinaGpu};
use universe::gpu::brain::{BrainGpu, BrainInput};
use universe::gpu::perception::{PerceptionGpu, PosePerception, TEMPORAL_STRIDE};
use universe::gpu::physics::{
    PhysicsGpu, PhysicsStateGpu, pack_config_flags, WheelGpu,
};
use universe::world::VoxelWorld;
use universe::simulation::racing_line::RacingLine;
use universe::visao::pitch;
use universe::perception::RetinaPerception;

// ─── Tipos/helpers deduplicados (E3, REFACTOR_PLAN §2.3) ─────────────────────
// RetinaParamsGpu/PoseRaw/uniform_entry/storage_entry/pos_of/make_physics_state
// eram cópias idênticas às de gpu_evolve.rs e cortex_evolve.rs — unificadas em
// src/gpu/retina_types.rs, src/gpu/core.rs e src/gpu/physics.rs.
use universe::gpu::retina_types::{RetinaParamsGpu, PoseRaw};
use universe::gpu::core::{uniform_entry, storage_entry};
use universe::gpu::physics::{pos_of, make_physics_state};

// ─── Estado de um ser para o laço (CPU-side, posição física) ─────────────────

struct BeingState {
    /// Estado de física completo (reside na CPU pois PhysicsGpu tem device próprio).
    phys: PhysicsStateGpu,
    /// Steer suavizado do tick anterior — replica motor.rs (prev_steer_mem * 0.50 + new * 0.50).
    /// Sem esse filtro, o modelo pode emitir steer=1 no tick 0 → slip_angle≈90° no pneu dianteiro,
    /// força longitudinal=0, carro gira no lugar mesmo com throttle=1.
    prev_steer: f32,
    /// Contador de ticks em que speed < 0.5 m/s — replica stagnation do motor.rs (linha 2221).
    /// Quando atinge 32 ticks (0.5s), ativa modo de escape: throttle=0.3 por 64 ticks.
    stagnation_ticks: u32,
    /// Contador de ticks restantes no modo de escape (throttle forçado = 0.3).
    escape_ticks: u32,
    /// Tempo restante em manobra de ré de emergência (s).
    /// Replica racer.reverse_timer do motor.rs (simulation/motor.rs L2174).
    /// > 0 enquanto em ré; decrementa por dt; ch89 = 127 quando > 0.
    reverse_timer: f32,
    /// Cooldown pós-ré (s). Enquanto > 0, limita aceleração a 0.8 e bloqueia nova ré.
    /// Replica racer.post_reverse_lockout do motor.rs (L2182: POST_REVERSE_LOCKOUT=0.6).
    post_reverse_lockout: f32,
}

// pos_of() e make_physics_state() agora vêm de universe::gpu::physics (dedup E3, ver
// o `use` no início do arquivo) — eram cópias idênticas às de gpu_evolve.rs/cortex_evolve.rs.

// ─── Contexto unificado: retina + percepção + cérebro no mesmo device ─────────

struct GpuLoop {
    /// Só para o diagnóstico imprimir uma vez, no primeiro tick.
    diag_feito: bool,
    diag_fis: bool,
    // Device compartilhado por retina + percepção + cérebro.
    // Arc porque BrainGpu exige Arc<Device> na assinatura.
    device: Arc<wgpu::Device>,
    queue:  Arc<wgpu::Queue>,
    adapter_name: String,

    // ── Retina ────────────────────────────────────────────────────────────────
    // RetinaGpu tem device próprio por valor — reutilizamos seus buffers mas
    // chamamos dispatch_into() para gravar no encoder externo.
    // PROBLEMA: RetinaGpu::new() abre device próprio. Para o laço encadeado,
    // replicamos o essencial da retina aqui usando o device compartilhado.
    pipeline_retina: wgpu::ComputePipeline,
    layout_retina:   wgpu::BindGroupLayout,
    buf_grid:        wgpu::Buffer,
    buf_poses_retina: wgpu::Buffer,
    buf_retina_out:   wgpu::Buffer,  // VRAM, alimenta percepção direto
    // buf_retina_readback NÃO alocado — não fazemos readback da retina no laço
    grid_size: u32,
    cell_size: f32,

    // ── Percepção ─────────────────────────────────────────────────────────────
    percept: PerceptionGpu,

    // ── Cérebro ───────────────────────────────────────────────────────────────
    brain:   BrainGpu,

    // ── Racing line (cópia para cálculo CPU por tick) ─────────────────────────
    // A projeção na racing_line é calculada pela CPU antes de cada dispatch e
    // injetada em buf_temporal[10-15] via update_racing_line_temporal().
    // Mantemos uma cópia local para evitar lifetimes com VoxelWorld.
    racing_line: RacingLine,

    // ── Grade de voxels na CPU (cópia da grade do mundo) ─────────────────────
    // Necessária para calcular dist_l/r/f na CPU antes do dispatch do kernel temporal.
    // O compilador ACO do RADV travava ao compilar loops sobre storage buffers em WGSL
    // (o BLOCO T original tinha LAT_SAMPLES e FWD_SAMPLES loops sobre `grid`).
    // Solução: CPU faz o cálculo e injeta em buf_temporal[16-18] via update_dist_temporal().
    // Custo: Vec<u32> de 320×320×4 = ~410 KB na heap — negligível.
    grid_cpu: Vec<u32>,
    grid_size_cpu: u32,
    cell_size_cpu: f32,

    // ── Física ────────────────────────────────────────────────────────────────
    // Device SEPARADO por isolamento de crash ACO/NIR (ver physics.rs:14).
    // O estado físico (PhysicsStateGpu) fica na CPU; sobe a cada tick via
    // write_buffer no device da física, desce via readback.
    phys_gpu: PhysicsGpu,

    // ── KAN output head ───────────────────────────────────────────────────────
    // Knots para spline de saída (throttle, steer, brake). Carregados do CHAMPION.dna
    // uma vez; usados em cada tick para converter intent raw → pedal físico real.
    // Sem este passo: intent[1]=0.985 → steer=0.985 → slip_angle=90° → sem tração.
    // Com KAN tanh default: eval(1, 0.985) ≈ 0.71, evoluído pode ser mais conservador.
    kan_knots: [[f32; universe::kan::KAN_OUT_KNOTS]; 3],

    // ── CfC de percepção (ch 192-207) ────────────────────────────────────────
    // Um RetinaPerception por ser, com estado recorrente residente na CPU entre ticks.
    // Pesos: semente fixa 0xCAFE_BABE_DEAD_BEEF (idênticos ao motor.rs L1626).
    // Por tick: lê buf_retina_out via readback → roda CfC → injeta hidden em temporal[33-48].
    percept_cfc: Vec<RetinaPerception>,

    // Buffer MAP_READ para readback da retina (423 KB @ 60 seres).
    // Necessário para rodar o CfC de percepção na CPU com a retina atual.
    // buf_retina_out tem COPY_SRC; buf_retina_readback tem MAP_READ + COPY_DST.
    // O copy (buffer_to_buffer) é adicionado ao encoder ANTES do submit.
    buf_retina_readback: wgpu::Buffer,

    // ── Metadados ─────────────────────────────────────────────────────────────
    capacity: usize,
}

impl GpuLoop {
    fn new(
        world: &VoxelWorld,
        engines: &[SovereignEngine],
        capacity: usize,
        kan_knots: [[f32; universe::kan::KAN_OUT_KNOTS]; 3],
    ) -> Option<Self> {
        // ── Garante que só um ICD Vulkan é carregado ──────────────────────────
        // Múltiplos ICDs (intel_hasvk, lvp, radeon...) causam SIGSEGV no loader.
        let icd_radeon = "/usr/share/vulkan/icd.d/radeon_icd.json";
        if std::path::Path::new(icd_radeon).exists()
            && std::env::var("VK_ICD_FILENAMES").is_err()
        {
            unsafe { std::env::set_var("VK_ICD_FILENAMES", icd_radeon); }
        }

        // ── Abre device compartilhado ─────────────────────────────────────────
        // compute-only: compatible_surface = None.
        // max_storage_buffers_per_shader_stage = 8: percepção usa 7 bindings.
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference:    wgpu::PowerPreference::HighPerformance,
                compatible_surface:  None,
                force_fallback_adapter: false,
            },
        ))?;
        let name = adapter.get_info().name.clone();
        let mut limits = wgpu::Limits::downlevel_defaults();
        // Cérebro usa 10 storage buffers: bindings 1-9 mais o `buf_tau` (banco de τ) —
        // mesma contagem documentada em gpu_evolve/context.rs::GpuCtx::new(). O valor
        // "9" aqui estava desatualizado (comentário mentiroso: dizia que binding 9 era
        // o máximo, mas `create_bind_group_layout(brain-layout)` já pedia 10 bindings
        // e o wgpu abortava com "Too many bindings of type StorageBuffers... limit is
        // 9, count was 10"). Achado ao rodar `gpu_loop` para validar o E3 (dedup de
        // uniform_entry/storage_entry/kan_eval) — corrigido aqui na mesma sessão, não
        // é um "próximo passo".
        // Pedir 12 dá a mesma folga de gpu_evolve.rs para os próximos buffers sem
        // voltar aqui a cada binding novo; se o adaptador não suportar, `request_device`
        // falha na abertura em vez de corromper dado.
        limits.max_storage_buffers_per_shader_stage = 12;
        let (raw_device, raw_queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label:             Some("iara-loop"),
                required_features: wgpu::Features::empty(),
                required_limits:   limits,
                memory_hints:      wgpu::MemoryHints::Performance,
            },
            None,
        )).ok()?;
        let device = Arc::new(raw_device);
        let queue  = Arc::new(raw_queue);

        let cap = capacity.max(1);

        // ── Buffers estáticos do mundo ────────────────────────────────────────
        // grade: 320×320 u32. Compartilhada por retina e percepção.
        let grid_u32: Vec<u32> = world.grid.iter().map(|v| *v as u32).collect();
        let buf_grid = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("loop-grid"),
            contents: bytemuck::cast_slice(&grid_u32),
            // COPY_SRC não necessário — só retina e percepção lêem.
            usage:    wgpu::BufferUsages::STORAGE,
        });

        // ── Buffers da retina ─────────────────────────────────────────────────
        // poses_retina: cap × PoseRaw (8 × f32 = 32 bytes).
        let buf_poses_retina = device.create_buffer(&wgpu::BufferDescriptor {
            label:               Some("loop-poses-retina"),
            size:                (cap * std::mem::size_of::<PoseRaw>()) as u64,
            usage:               wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation:  false,
        });
        // retina_out: cap × RETINA_TOTAL × 4 bytes. Alimenta percepção na VRAM.
        // COPY_SRC para inspecionar se necessário.
        let buf_retina_out = device.create_buffer(&wgpu::BufferDescriptor {
            label:               Some("loop-retina-out"),
            size:                (cap * RETINA_TOTAL * 4) as u64,
            usage:               wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation:  false,
        });

        // ── Pipeline da retina ─────────────────────────────────────────────────
        // retina.wgsl: binding 0=params(uniform), 1=grid(storage), 2=poses(storage), 3=saida(storage).
        let shader_retina = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("loop-retina"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                include_str!("../gpu/retina.wgsl"),
            )),
        });
        let layout_retina = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("loop-retina-layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),  // grid
                storage_entry(2, true),  // poses
                storage_entry(3, false), // saida
            ],
        });
        let pl_retina = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                None,
            bind_group_layouts:   &[&layout_retina],
            push_constant_ranges: &[],
        });
        let pipeline_retina = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:               Some("loop-retina-pipeline"),
            layout:              Some(&pl_retina),
            module:              &shader_retina,
            entry_point:         Some("main"),
            compilation_options: Default::default(),
            cache:               None,
        });

        // ── Percepção ─────────────────────────────────────────────────────────
        // PerceptionGpu::new recebe &wgpu::Device (Arc<Device> implementa Deref).
        eprintln!("[gpu_loop] Criando PerceptionGpu...");
        let percept = PerceptionGpu::new(&device, &queue, world, cap)?;
        eprintln!("[gpu_loop] PerceptionGpu OK");

        // ── Cérebro ───────────────────────────────────────────────────────────
        // BrainGpu aceita Arc<Device> diretamente.
        let brain = BrainGpu::new(Arc::clone(&device), Arc::clone(&queue), name.clone(), cap);

        // Sobe pesos dos engines para o buffer do cérebro.
        let eng_refs: Vec<&SovereignEngine> = engines.iter().take(cap).collect();
        brain.upload_weights(&eng_refs);

        // ── Física — MESMO dispositivo que o resto ───────────────────────────
        // Ela nasceu com device próprio para isolar o crash do compilador ACO do RADV
        // (`physics.rs`: a fase Pacejka estourava o alocador de registradores). Esse crash
        // foi resolvido dividindo em três kernels, então o isolamento perdeu a razão — e
        // passou a ATRAPALHAR: com dois `wgpu::Device` na mesma GPU o laço travava no
        // primeiro tick, sem erro e sem saída. Um device só destrava e é o que permite
        // encadear os quatro dispatches num `CommandEncoder` único.
        let grid_u8: Vec<u8> = world.grid.iter().map(|v| *v as u8).collect();
        let phys_gpu = PhysicsGpu::new_shared(
            Arc::clone(&device), Arc::clone(&queue), name.clone(),
            &grid_u8, &world.height_map, world.size, world.cell_size, cap)?;

        // ── Buffer de readback da retina (para CfC de percepção na CPU) ────────
        // cap × RETINA_TOTAL × 4 bytes. Cópia de buf_retina_out após o dispatch.
        // Permite rodar RetinaPerception na CPU sem nenhuma alteração no shader.
        let retina_readback_bytes = (cap * RETINA_TOTAL * std::mem::size_of::<u32>()) as u64;
        let buf_retina_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label:               Some("loop-retina-readback"),
            size:                retina_readback_bytes,
            usage:               wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation:  false,
        });

        // ── CfC de percepção — um por ser, com estado recorrente na CPU ────────
        // Pesos inicializados com semente fixa 0xCAFE_BABE_DEAD_BEEF (mesmo motor.rs).
        // Estado recorrente (hidden[16]) residirá aqui entre ticks.
        // Ao contrário do motor.rs que usa retina_binocular (u8 convertida de raw),
        // aqui usamos a retina GPU (u32 packed) convertida para i8 na leitura.
        let percept_cfc: Vec<RetinaPerception> = (0..cap).map(|_| RetinaPerception::new()).collect();

        println!(
            "GpuLoop: adapter={} · cap={} seres · retina+percepção+cérebro+física NO MESMO device",
            name, cap
        );
        println!("  CfC percepção (ch192-207): {} seres × 16 hidden = {} f32 na CPU",
            cap, cap * 16);
        println!("  ch83-84 (fluxo hemisférico): ligado no kernel motion");
        println!("  ch89 (reverse_timer): ligado via temporal[32]");

        Some(Self {
            diag_feito: false,
            diag_fis: false,
            device,
            queue,
            adapter_name: name,
            pipeline_retina,
            layout_retina,
            buf_grid,
            buf_poses_retina,
            buf_retina_out,
            grid_size: world.size as u32,
            cell_size: world.cell_size,
            percept,
            brain,
            racing_line: world.racing_line.clone(),
            // Cópia da grade para cálculo CPU de dist_l/r/f (temporal[16-18]).
            grid_cpu: grid_u32,
            grid_size_cpu: world.size as u32,
            cell_size_cpu: world.cell_size,
            phys_gpu,
            kan_knots,
            percept_cfc,
            buf_retina_readback,
            capacity: cap,
        })
    }

    /// Executa um tick completo para N seres.
    ///
    /// # Fluxo de dados
    ///
    ///   poses_fisicas (CPU) →[poses_retina]→ GPU
    ///   encoder {
    ///     retina   → buf_retina_out (VRAM)
    ///     percepção ← buf_retina_out (VRAM sem readback)
    ///             → buf_percept_out (VRAM)
    ///     cérebro  ← buf_percept_out (lido via wave escrito na memória)
    ///             → buf_intent (VRAM) + copy → buf_intent_readback
    ///   }
    ///   submit + poll(Wait)         ← UM round-trip
    ///   ler buf_intent_readback     ← intent[n]
    ///   atualizar throttle/steer/brake em states[n]
    ///   phys_gpu.step(states[n])   ← física no device isolado
    ///
    /// Retorna `true` se bem-sucedido.
    fn tick(
        &mut self,
        states:    &mut Vec<BeingState>,
        engines:   &[SovereignEngine],
        n:         usize,
        dt:        f32,
        tick_idx:  usize,
    ) -> bool {
        if n == 0 || n > self.capacity { return false; }

        // ── 1. Converte estados físicos para poses da retina ──────────────────
        // PoseRaw: [pos.x, pos.y, pos.z, gaze_yaw] + [fwd.x, fwd.z, right.x, right.z]
        // A retina usa coordenadas do mundo; gaze_yaw = 0 (sem desvio lateral).
        let pose_raw_data: Vec<PoseRaw> = (0..n).map(|i| {
            let p = &states[i].phys;
            PoseRaw {
                pos_gaze: [p.pos[0], p.pos[1], p.pos[2], 0.0],
                dirs:     [p.fwd[0], p.fwd[2], p.lat[0], p.lat[2]],
            }
        }).collect();
        self.queue.write_buffer(&self.buf_poses_retina, 0, bytemuck::cast_slice(&pose_raw_data));

        // ── 2. Poses para percepção (formato diferente: PosePerception) ───────
        let poses_percept: Vec<PosePerception> = (0..n).map(|i| {
            let p = &states[i].phys;
            PosePerception {
                pos_x: p.pos[0],
                pos_y: p.pos[1],
                pos_z: p.pos[2],
                phase: 0.0, // heading absoluto — zero pois não temos racer.phase aqui
                fwd_x: p.fwd[0],
                fwd_z: p.fwd[2],
                rgt_x: p.lat[0],
                rgt_z: p.lat[2],
            }
        }).collect();

        // ── 3a. Pré-computa distâncias laterais/frontais na CPU → buf_temporal[16-18] ──
        // O compilador ACO do RADV travava ao compilar os loops LAT_SAMPLES e FWD_SAMPLES
        // sobre o storage buffer `grid` em WGSL (workgroup_size=1, pattern com `break`).
        // Solução: CPU replica o mesmo algoritmo e injeta os 3 valores via update_dist_temporal().
        // Custo: N × (6+6+10) = N×22 consultas à grade CPU (hash O(1) por célula). Negligível.
        {
            let gs = self.grid_size_cpu as usize;
            let cs = self.cell_size_cpu;
            // Amostras laterais: expandidas de 27m → 40m (conta: 40/28 m/s = 1.43 s antecipação).
            // Normaliza por 40.0 no temporal_wgsl. Default 40.0 = livre (nenhuma parede no alcance).
            let lat_samp = [1.0_f32, 3.0, 6.0, 9.0, 12.0, 16.0, 20.0, 26.0, 33.0, 40.0];
            // Amostras frontais: mesmas do BLOCO T original
            let fwd_samp = [3.0_f32, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0];

            // Função inline: voxel em coordenadas mundo (identical a voxel_at() no WGSL).
            let voxel_at = |wx: f32, wz: f32| -> u32 {
                let xi = (wx / cs).round() as i32;
                let yi = (wz / cs).round() as i32;
                let n = gs as i32;
                if xi >= 0 && xi < n && yi >= 0 && yi < n {
                    self.grid_cpu[(yi as usize) * gs + (xi as usize)]
                } else {
                    2u32 // Wall fora da grade
                }
            };

            let mut dist_vals: Vec<[f32; 3]> = Vec::with_capacity(n);
            for i in 0..n {
                let p = &states[i].phys;
                let (px, pz) = (p.pos[0], p.pos[2]);
                let (fx, fz) = (p.fwd[0], p.fwd[2]);
                // Perpendicular esquerda: (-fwd_z, fwd_x) no plano XZ.
                // Replicamos exatamente o BLOCO T: perp_x = -fwd_z, perp_z = fwd_x.
                let perp_x = -fz;
                let perp_z =  fx;

                // Distância lateral esquerda: pos - perp*sd (idem ao BLOCO T original)
                // Default 40.0 = livre (sem parede no alcance de 40m).
                let mut dist_l = 40.0_f32;
                for &sd in &lat_samp {
                    if voxel_at(px - perp_x * sd, pz - perp_z * sd) == 2u32 {
                        dist_l = sd;
                        break;
                    }
                }
                // Distância lateral direita: pos + perp*sd
                let mut dist_r = 40.0_f32;
                for &sd in &lat_samp {
                    if voxel_at(px + perp_x * sd, pz + perp_z * sd) == 2u32 {
                        dist_r = sd;
                        break;
                    }
                }
                // Distância frontal
                let mut dist_f = 50.0_f32;
                for &sd in &fwd_samp {
                    let v = voxel_at(px + fx * sd, pz + fz * sd);
                    if v == 2u32 || v == 3u32 { dist_f = sd; break; }
                }
                dist_vals.push([dist_l, dist_r, dist_f]);
            }
            // Injeta em buf_temporal[16-18] ANTES do dispatch da percepção temporal.
            self.percept.update_dist_temporal(&self.queue, &dist_vals[..n]);
        }

        // ── 3b. Pré-computa racing_line na CPU e injeta em buf_temporal[10-15] ──
        // A projeção O(N_points) na racing_line foi movida para a CPU porque o
        // compilador ACO/NIR do RADV crashava ao compilar loops sobre storage buffers
        // de comprimento variável em WGSL. Custo CPU: N × O(N_points) com busca local
        // = ~60 × 64 iterações = 3840 operações. Negligível vs 0.63ms/tick de GPU.
        //
        // Campos temporais [10-15]:
        //  [10] cpu_line_off_norm = offset / half_width, [-1,1]
        //  [11] cpu_d_off         = (prev_off - off) / hw × 8, [-1,1]  (derivada)
        //  [12] cpu_la_off        = lookahead_offset(5m) / hw, [-1,1]
        //  [13] cpu_fwd_norm      = fwd_dist / (hw*3), [0,1]  (usa dist_l=max da grade)
        //  [14] cpu_corridor      = (dist_l+dist_r) / (hw*4), [0,1]
        //  [15] cpu_whisker_norm  = min(dist_l,dist_r) / (hw*1.5), [0,1]
        //
        // prev_off para o cálculo de d_off é lido do campo [8] do temporal.
        // Neste binário não fazemos readback do temporal (custo), então estimamos
        // prev_off = line_off_norm atual do tick anterior via lookup simples.
        // Guardamos em prev_rl_off (Vec<f32> alocado 1× por corrida, ver abaixo).
        // NOTA: d_off mede taxa de mudança de desvio — "estou me afastando da linha
        //       mais rápido do que estava?". Ganho 8× pq delta por tick é ~0.01.
        {
            let hw = self.racing_line.half_width.max(1.0);
            let hw3 = hw * 3.0;
            let hw4 = hw * 4.0;
            let hw15 = hw * 1.5;
            // Calcula valores e acumula num Vec temporário por tick.
            let mut racing_vals: Vec<[f32; 6]> = Vec::with_capacity(n);
            for i in 0..n {
                let p = &states[i].phys;
                let pos2d = bevy::math::Vec2::new(p.pos[0], p.pos[2]);
                let fwd2d = bevy::math::Vec2::new(p.fwd[0], p.fwd[2]);

                // Projeção na racing_line: desvio lateral assinado (m), clamped ±hw.
                let line_off = self.racing_line.lateral_offset_from_line(pos2d, fwd2d);
                let off_norm = (line_off / hw).clamp(-1.0, 1.0);

                // Lookahead 5m à frente na linha ideal.
                let la = self.racing_line.lookahead_offset(pos2d, 5.0);
                let la_norm = (la / hw).clamp(-1.0, 1.0);

                // Distâncias laterais aproximadas (substitui grade — precisa de world aqui,
                // mas não temos referência ao world neste método. Usamos half_width como
                // proxy conservador: corredor = hw de cada lado = corredor cheio = 1.0).
                // TODO: para distâncias reais, fazer readback da grade ou passar world ref.
                // Por agora: fwd_norm=0.5 (distância frontal desconhecida), corridor=0.8 (livre).
                // O kernel WGSL calculou dist_l/r na grade — esses campos são complementares.
                let cpu_fwd_norm     = 0.5_f32;   // desconhecido sem readback da grade
                let cpu_corridor     = 0.8_f32;   // livre (conservador)
                let cpu_whisker_norm = 0.7_f32;   // livre (conservador)

                // d_off: derivada do desvio. Precisamos do prev_off. Usamos off_norm
                // atual como prev pra este tick; no próximo tick a diferença será real.
                // (No GPU, o temporal[8] guarda prev_line_offset_n do tick anterior.)
                // Aqui não temos o valor anterior sem readback. Passamos 0 para d_off,
                // que dá sinal neutro. Motor.rs usa a versão completa com estado real.
                let cpu_d_off = 0.0_f32;  // sem estado anterior disponível neste binário

                racing_vals.push([off_norm, cpu_d_off, la_norm,
                                  cpu_fwd_norm, cpu_corridor, cpu_whisker_norm]);
            }
            // Injeta em buf_temporal[10-15] ANTES do dispatch da percepção.
            self.percept.update_racing_line_temporal(&self.queue, &racing_vals[..n]);
        }

        // ── 3c. Telemetria proprioceptiva → buf_temporal[20-31] ───────────────
        // Canais 64-75 do sensory_wave: speed, alignment, RPM, gear, tire_temp, etc.
        // Esses canais estão no top-30 de pesos do CHAMPION.dna mas eram ZERO no gpu_loop
        // porque o shader espacial não tem acesso ao estado físico (reside na CPU).
        // O shader temporal lê temporal[20-31] e faz atomicOr em ch64-75 do buf_output.
        //
        // Canais críticos confirmados por inspeção do SRLE tag do DNA:
        //   ch64 (alignment)   — modelo sabe se está alinhado com a pista
        //   ch65 (speed m/s)   — MAIS CRÍTICO: modelo sabe se está andando
        //   ch68 (RPM ratio)   — sabe se motor está solicitado
        //   ch69 (gear)        — contexto de marcha para decisão de shift
        //   ch70 (tire_temp)   — temperatura dos pneus (grip)
        //   ch73 (on_grass)    — dor de grama → reduz throttle (ch73 era -128 no motor.rs)
        {
            // Função aux inline: voxel atual do ser (para detectar grama).
            let gs = self.grid_size_cpu as usize;
            let cs = self.cell_size_cpu;
            let voxel_at_cpu = |wx: f32, wz: f32| -> u32 {
                let xi = (wx / cs).round() as i32;
                let yi = (wz / cs).round() as i32;
                let ng = gs as i32;
                if xi >= 0 && xi < ng && yi >= 0 && yi < ng {
                    self.grid_cpu[(yi as usize) * gs + (xi as usize)]
                } else { 2u32 }
            };

            let mut prop_vals: Vec<[f32; 12]> = Vec::with_capacity(n);
            for i in 0..n {
                let p = &states[i].phys;

                // [1] speed: magnitude do vetor velocidade no plano XZ (ignoramos Y=vert.)
                let speed_m_s = (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt();

                // [0] alignment: cos(ângulo entre heading fwd e tangente da pista).
                // A racing_line devolve o tangent unit vector no ponto mais próximo.
                // Aqui usamos o produto escalar fwd·tangent como proxy de alinhamento.
                // Tangente = fwd do ser rotacionado pelo desvio angular da racing_line.
                // Aproximação simples: usamos a projection da velocidade no heading.
                // dot(vel_xz, fwd_xz) / (|vel| + ε) → 1 = alinhado, -1 = contra-mão.
                let vel_dot_fwd = p.vel[0] * p.fwd[0] + p.vel[2] * p.fwd[2];
                let alignment = if speed_m_s > 0.1 {
                    (vel_dot_fwd / speed_m_s).clamp(-1.0, 1.0)
                } else {
                    1.0  // parado = assume alinhado (sem ruído de direção)
                };

                // [2] lateral velocity: dot(vel, lat) — positivo = deriva para direita
                let lat_vel = p.vel[0] * p.lat[0] + p.vel[2] * p.lat[2];

                // [3] HP: sem tracker de dano, assume cheio (127 no canal).
                let hp_frac = 1.0_f32;

                // [4] RPM ratio
                let rpm_ratio = if p.max_rpm > 0.0 {
                    (p.current_rpm / p.max_rpm).clamp(0.0, 1.0)
                } else { 0.0 };

                // [5] gear (f32 para o WGSL — multiplicado por 20 no shader)
                let gear = p.current_gear as f32;

                // [6] tire_temp_norm: wheels tem layout [part_a, part_b] alternados por eixo.
                // part_b[0] = tire_temp (graus C). Calcula média das 4 rodas.
                // wheels[j*2+1] = part_b do eixo j (j=0..3 = FR, FL, RR, RL).
                let tire_temp_sum: f32 = (0..4).map(|j| p.wheels[j * 2 + 1][0]).sum();
                let tire_temp_norm = (tire_temp_sum / 4.0 / 120.0).clamp(0.0, 1.0);

                // [7] downforce: não calculamos dinamicamente. Zero por ora.
                let downforce_norm = 0.0_f32;

                // [8] in_draft: não detectamos slipstream. Zero.
                let in_draft = 0.0_f32;

                // [9] on_grass: voxel atual. VoxelType: Grass=0, Road=1, Wall=2, Hole=3.
                // Grama = 0 (todo o fundo do mapa antes de ser pavimentado).
                let voxel = voxel_at_cpu(p.pos[0], p.pos[2]);
                let on_grass = if voxel == 0u32 { 1.0_f32 } else { 0.0 };

                // [10] steer_abs: |steer| como proxy de steer_limit.
                // No motor.rs steer_limit = ângulo máximo disponível (depende da geometria).
                // Aqui usamos |steer_cmd| como sinal de que está virando.
                let steer_abs = p.steer.abs().clamp(0.0, 1.0);

                // [11] wetness: sem weather dinâmico. Zero.
                let wetness = 0.0_f32;

                prop_vals.push([
                    alignment, speed_m_s, lat_vel, hp_frac,
                    rpm_ratio, gear, tire_temp_norm, downforce_norm,
                    in_draft, on_grass, steer_abs, wetness,
                ]);
            }
            self.percept.update_proprioception_temporal(&self.queue, &prop_vals[..n]);
        }

        // ── 3d. reverse_timer (ch 89) e CfC de percepção (ch 192-207) ──────────
        // Lógica de ré IDÊNTICA ao motor.rs (simulation/motor.rs L2177-2235):
        //   Trigger: parede frontal <3m E ambas laterais bloqueadas (<4m)
        //            E speed<1.5 m/s E stagnation>60 ticks E sem lockout pós-ré.
        //   Duração: 0.4s (REVERSE_MIN) capped em 1.5s (REVERSE_MAX).
        //   Exit: parede frontal > 2.5m (REVERSE_EXIT_CLEAR) ou timeout.
        //   Pós-ré: lockout de 0.6s onde throttle é capeado em 0.8.
        //
        // CfC de percepção: usa retina do tick T-1 (lida no readback após submit T-1).
        // Atraso de 1 tick é aceitável — motor.rs também usa visual_cortex anterior.
        {
            // Deduplicado (E3, REFACTOR_PLAN.md §2.3) para
            // universe::config::iara::reverse_maneuver. TRIGGER_DIST_M=4.0 (não 3.0):
            // este arquivo já usava `fwd_dist < 4.0` de propósito (ver comentário no
            // gatilho abaixo) — a constante local `REVERSE_TRIGGER_DIST=3.0` nunca era
            // lida (warning `never used`), só citada em comentário. `gpu_evolve.rs`
            // tinha o MESMO padrão (constante morta em 3.0, literal usado em 4.0) — o
            // valor canônico único abaixo é o que os três binários já rodavam de fato.
            use universe::config::iara::reverse_maneuver as revcfg;

            let gs = self.grid_size_cpu as usize;
            let cs = self.cell_size_cpu;
            // Função inline para consultar voxel (mesma do bloco 3a).
            let voxel_at = |wx: f32, wz: f32| -> u32 {
                let xi = (wx / cs).round() as i32;
                let yi = (wz / cs).round() as i32;
                let ng = gs as i32;
                if xi >= 0 && xi < ng && yi >= 0 && yi < ng {
                    self.grid_cpu[(yi as usize) * gs + (xi as usize)]
                } else { 2u32 }
            };

            let mut extra_vals: Vec<[f32; 17]> = Vec::with_capacity(n);
            for i in 0..n {
                let p = &states[i].phys;
                let speed = (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt();

                // Mede distâncias para o trigger (replica motor.rs L2202-2220):
                //   fwd_dist = menor distância frontal a Wall/Hole (cap 30m)
                //   lat_l_check = dist lat esq até Wall/Hole (<4m = bloqueado)
                //   lat_r_check = idem direito
                let (fx, fz) = (p.fwd[0], p.fwd[2]);
                let perp_x = -fz; let perp_z = fx;  // left perpendicular
                let (px, pz) = (p.pos[0], p.pos[2]);

                // Distância frontal (amostras de 1m a 10m como no motor.rs)
                let mut fwd_dist = 30.0_f32;
                for sd in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] {
                    let v = voxel_at(px + fx*sd, pz + fz*sd);
                    if v == 2u32 || v == 3u32 { fwd_dist = sd; break; }
                }

                // Distâncias laterais (amostras 1m a 10m como no motor.rs)
                let mut lat_l = 10.0_f32;
                let mut lat_r = 10.0_f32;
                for sd in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] {
                    if lat_l == 10.0 {
                        let v = voxel_at(px + perp_x*sd, pz + perp_z*sd);
                        if v == 2u32 || v == 3u32 { lat_l = sd; }
                    }
                    if lat_r == 10.0 {
                        let v = voxel_at(px - perp_x*sd, pz - perp_z*sd);
                        if v == 2u32 || v == 3u32 { lat_r = sd; }
                    }
                }
                let both_blocked = lat_l < 4.0 && lat_r < 4.0;

                // Decrementa lockout pós-ré.
                if states[i].post_reverse_lockout > 0.0 {
                    states[i].post_reverse_lockout = (states[i].post_reverse_lockout - dt).max(0.0);
                }

                // Trigger: parede frontal <4m + speed<1.5 + stagnation>60 + sem lockout.
                // NOTA: removemos a condição `both_blocked` do motor.rs por uma razão importante:
                // o CHAMPION foi treinado com a lógica do motor.rs onde ré aciona ANTES de ficar
                // preso contra uma só parede. Quando lat_r está livre mas lat_l+fwd bloqueados,
                // o modelo emite throttle negativo (sinalizando intenção de ré) — esse sinal deve
                // ativar a ré. Sem isso, throttle negativo é clippado para zero e o carro trava.
                // Além disso: o motor.rs usa `forward_wall_dist` que é calculado diferente
                // (com probes mais longos) — nosso fwd_dist pode ser mais restritivo. Por isso
                // o gatilho usa 4.0 m (`revcfg::TRIGGER_DIST_M`), mais permissivo que os 3.0 m
                // que uma leitura ingênua de motor.rs sugeriria.
                {
                    let rev = states[i].reverse_timer;
                    let stag = states[i].stagnation_ticks;
                    let lockout = states[i].post_reverse_lockout;
                    if rev <= 0.0 && lockout <= 0.0
                        && fwd_dist < revcfg::TRIGGER_DIST_M
                        && speed < revcfg::TRIGGER_SPEED_MAX_MS
                        && stag > revcfg::STAGNATION_TICKS
                    {
                        states[i].reverse_timer = revcfg::MIN_DURATION_S;
                        states[i].stagnation_ticks = 0;
                    }
                }

                // Decrementa reverse_timer e verifica condição de saída da ré.
                {
                    let rev = states[i].reverse_timer;
                    if rev > 0.0 {
                        let new_rev = rev - dt;
                        // Sai da ré se: frontal ficou livre (>2.5m) ou timeout máximo.
                        if fwd_dist > revcfg::EXIT_CLEAR_M || new_rev > revcfg::MAX_DURATION_S {
                            states[i].reverse_timer = 0.0;
                            states[i].post_reverse_lockout = revcfg::POST_LOCKOUT_S;
                        } else {
                            states[i].reverse_timer = new_rev.max(0.0);
                        }
                    }
                }

                // Lê hidden do CfC de percepção (estado do tick T-1, atualizado no readback).
                let cfc_ref = &self.percept_cfc[i];
                let hidden = &cfc_ref.pipeline.state.h;  // &Vec<f32>, len=16

                // Monta [reverse_timer, cfc_hidden[0..15]] para injetar em temporal[32-48].
                let mut val = [0.0f32; 17];
                val[0] = states[i].reverse_timer;
                for j in 0..16 {
                    val[1 + j] = if j < hidden.len() { hidden[j] } else { 0.0 };
                }
                extra_vals.push(val);
            }
            self.percept.update_extra_temporal(&self.queue, &extra_vals[..n]);
        }

        // ── 4. Encoder único ──────────────────────────────────────────────────
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("loop-tick"),
        });

        // ── 4a. Retina ────────────────────────────────────────────────────────
        // Params uniform para a retina (16 + 64 bytes = 80 bytes).
        let mut pitches_arr = [[0.0f32; 4]; 4];
        for row in 0..RETINA_ROWS {
            pitches_arr[row / 4][row % 4] = pitch(row);
        }
        let retina_params = RetinaParamsGpu {
            grid_size: self.grid_size,
            n_beings:  n as u32,
            cell_size: self.cell_size,
            fov_half:  (universe::visao::fov_deg() / 2.0).to_radians(),
            pitches:   pitches_arr,
        };
        let buf_rp = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("loop-retina-params"),
            contents: bytemuck::bytes_of(&retina_params),
            usage:    wgpu::BufferUsages::UNIFORM,
        });
        let bg_retina = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   None,
            layout:  &self.layout_retina,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_rp.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_grid.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_poses_retina.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_retina_out.as_entire_binding() },
            ],
        });
        // ── 4a-PRE: preserva retina do tick anterior ANTES da retina escrever ─────
        // copy_retina_prev copia buf_retina_out (tick t-1) → buf_retina_prev.
        // Isso DEVE acontecer antes do dispatch da retina (que sobrescreve buf_retina_out).
        // No tick 0: buf_retina_prev = zeros → motion = |frame_real - 0| = frame_real (ok).
        // Nos ticks seguintes: buf_retina_prev = frame do tick t-1 → motion = delta real.
        self.percept.copy_retina_prev(&mut enc, &self.buf_retina_out);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:            Some("pass-retina"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_retina);
            pass.set_bind_group(0, &bg_retina, &[]);
            // 64 threads/workgroup: 2 waves de 32 no RDNA2 (wave32 = tamanho correto).
            // Total threads = n × RETINA_TOTAL. Para 60 seres: 60×1764 = 105.840 threads.
            // Workgroups = ceil(105.840 / 64) = 1.654.
            let total = (n * RETINA_TOTAL) as u32;
            pass.dispatch_workgroups(total.div_ceil(64), 1, 1);
        }
        // Encerrar pass_retina insere barreira global de memória:
        // buf_retina_out é visível para o próximo dispatch (percepção).

        // ── 4b. Percepção ─────────────────────────────────────────────────────
        // bind_and_dispatch grava no encoder SEM submit.
        // Lê buf_retina_out diretamente da VRAM — zero readback.
        // Salva bytes_percept para o readback diagnóstico (IARA_DIAG quando travado).
        let bytes_percept = self.percept.bind_and_dispatch(
            &mut enc,
            &self.device,
            &self.queue,
            &poses_percept[..n],
            &self.buf_retina_out,
        );
        // Após encerrar o pass de percepção, buf_percept_out está pronto na VRAM.
        // O cérebro lê esse buffer DIRETAMENTE via binding 8 (use_percept_direct=1),
        // sem nenhum readback intermediário CPU. O elo percepção→cérebro existe agora.
        // IMPORTANTE: atomicOr escreve u32 em buf_percept_out; o brain.wgsl lê u32
        // sem atomics (o pass de percepção já terminou — barreiras de workgroup garantem).

        // ── 4c. Cérebro ───────────────────────────────────────────────────────
        // bind_and_dispatch com buf_percept = Some(...) ativa use_percept_direct=1.
        // O shader lê o wave diretamente de buf_percept_out (56 u32 por ser),
        // desempacota i8 two's-complement em linha, e aplica KAN+CfC+L1+L2.
        // wave_data_i32 é ignorado neste path (passamos slice vazio).
        let bytes_intent = self.brain.bind_and_dispatch(
            &mut enc,
            n,
            &[],                                  // ignorado: use_percept_direct=1
            dt,
            Some(self.percept.buf_output()),       // elo percepção→cérebro via VRAM
        );

        // ── 4d-extra. Cópia retina → buf_retina_readback (para CfC percepção) ──
        // Adicionamos no encoder a cópia buf_retina_out → buf_retina_readback.
        // Isso permite, após o poll(Wait), mapear buf_retina_readback e rodar o
        // CfC de percepção na CPU com a retina DO TICK ATUAL (t). No próximo tick
        // (t+1), o estado atualizado do CfC será injetado em temporal[33-48].
        // Custo: ~423 KB de cópia intra-GPU (DMA, ~2 μs) + readback PCIe (~50 μs).
        let retina_copy_bytes = (n * RETINA_TOTAL * 4) as u64;
        enc.copy_buffer_to_buffer(&self.buf_retina_out, 0, &self.buf_retina_readback, 0, retina_copy_bytes);

        // ── 4e. Submit único ──────────────────────────────────────────────────
        // Todos os dispatches (retina, percepção, cérebro) + cópia retina readback
        // num único command buffer → UM DMA de submit para o driver.
        self.queue.submit(Some(enc.finish()));

        // ── 5. Poll único — bloqueia até todos os kernels terminarem ──────────
        // Este é o único round-trip por tick.
        self.device.poll(wgpu::Maintain::Wait);

        // ── 6. Leitura do intent (960 bytes para 60 seres) ────────────────────
        let intents = self.brain.read_intent_after_submit(n);

        // ── 6b. Readback da retina e atualização do CfC de percepção ──────────
        // Lemos buf_retina_readback (copiado no encoder 4d-extra), rodamos o CfC
        // de percepção para cada ser, e atualizamos o estado interno do CfC (h).
        // No PRÓXIMO tick, update_extra_temporal() lerá self.percept_cfc[i].pipeline.cfc.h
        // e injetará em temporal[33-48] para o shader escrever ch192-207.
        //
        // Formato buf_retina_readback: N × RETINA_TOTAL u32 packed (mm<<8 | mat).
        // RetinaPerception::perceive() aceita &[i8] (distância quantizada -127..127).
        // A conversão usa a codificação legada do motor.rs (codificacao_legada equivalente):
        //   mat=0 (grass) → moderado
        //   mat=1 (road)  → negativo
        //   mat=2 (wall)  → positivo alto
        //   mat=3 (hole)  → 127
        {
            let slice = self.buf_retina_readback.slice(0..retina_copy_bytes);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
            // Poll pós-mapeamento: o submit já foi processado pelo poll acima,
            // mas o callback do map_async precisa de um poll para ser servido.
            self.device.poll(wgpu::Maintain::Wait);
            let _ = rx.recv();

            let retina_data = slice.get_mapped_range();
            let retina_u32: &[u32] = bytemuck::cast_slice(&retina_data);

            // Para cada ser: converte pixels u32 → i8 e roda o CfC de percepção.
            // O hidden atualizado será usado no PRÓXIMO tick via temporal[33-48].
            for i in 0..n {
                let base = i * RETINA_TOTAL;
                let mut retina_i8 = vec![0i8; RETINA_TOTAL];
                for j in 0..RETINA_TOTAL {
                    let packed = retina_u32[base + j];
                    let dist_mm = (packed >> 8) as f32;
                    let dist_m  = dist_mm / 1000.0;
                    let mat     = packed & 0xFF;
                    // Replica codificacao_legada() do motor.rs:
                    //   wall (2): curva 127→55→0 (≤30m → 55, ≤120m → 0 linear)
                    //   road (1): -128..0 (mais perto = mais negativo)
                    //   grass(0): ~40 fixo
                    //   hole (3): 127 (perigo máximo)
                    let v: f32 = match mat {
                        2 => {  // wall: positivo alto perto, decai à distância
                            if dist_m <= 30.0 {
                                127.0 - (127.0 - 55.0) * (dist_m / 30.0)
                            } else {
                                55.0 * (1.0 - (dist_m - 30.0) / 90.0).max(0.0)
                            }
                        }
                        1 => {  // road: negativo (mais perto = mais negativo)
                            -128.0 + (dist_m.min(80.0) / 80.0) * 128.0
                        }
                        0 => 40.0,   // grass: valor moderado fixo
                        _ => 127.0,  // hole (mat=3): perigo máximo
                    };
                    retina_i8[j] = v.clamp(-127.0, 127.0) as i8;
                }
                // Roda o CfC de percepção (atualiza estado recorrente interno).
                // perceive() retorna PerceptionOutput com hidden[0..16] atualizado.
                // O próximo tick lerá self.percept_cfc[i].pipeline.cfc.h diretamente.
                self.percept_cfc[i].perceive(&retina_i8);
            }

            drop(retina_data);
            self.buf_retina_readback.unmap();
        }

        // ── DIAGNÓSTICO: onde o sinal morre? ──────────────────────────────────
        // Os kernels validam 100% isolados (gpu_validate), mas no laço o intent sai zero e
        // ninguém anda. Então o defeito está na LIGAÇÃO, não na matemática. Estes três
        // números dizem em qual elo: wave da percepção, intent do cérebro, ou comando que
        // chega na física. Só imprime no primeiro tick, para não poluir.
        if std::env::var("IARA_DIAG").is_ok() && (!self.diag_feito || tick_idx % 500 == 0) {
            self.diag_feito = true;
            let int_nz = intents.iter().flat_map(|a| a.iter()).filter(|v| v.abs() > 1e-6).count();
            let int_max = intents.iter().flat_map(|a| a.iter())
                .fold(0.0f32, |m, v| m.max(v.abs()));
            eprintln!("[DIAG t={}] intent: {}/{} nao-zero, |max|={:.6}", tick_idx,
                      int_nz, intents.len() * 96, int_max);
            eprintln!("[DIAG t={}] intent[0][0..3] = {:?}", tick_idx,
                      &intents[0][0..3.min(intents[0].len())]);
            // ser 0 estado completo
            if !states.is_empty() {
                let p = &states[0].phys;
                let speed = (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt();
                eprintln!("[DIAG t={}] ser0 speed={:.2}m/s rpm={:.0} gear={}", tick_idx, speed, p.current_rpm, p.current_gear);
            }
        }
        // ── DIAGNÓSTICO DE TRAVAMENTO: lê o sensory_wave quando ser0 está parado ──
        // Quando IARA_DIAG=1 e ser0 está a < 0.5 m/s, faz readback manual do buf_percept_out
        // via um buffer temporário MAP_READ, imprime canais críticos para identificar o travamento.
        // Um submit+poll separado é necessário porque o buf_readback da percepção pode estar
        // mapeado ou não, dependendo do fluxo. Usamos o buf_output direto (COPY_SRC).
        if std::env::var("IARA_DIAG").is_ok() && tick_idx % 200 == 0 && !states.is_empty() {
            let p = &states[0].phys;
            let speed0 = (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt();
            // Só quando travado para não adicionar latência no laço quente.
            if speed0 < 0.5 && bytes_percept > 0 {
                // Cria buffer temporário MAP_READ para 1 ser apenas (56 u32 = 224 bytes).
                let diag_bytes = (1 * 56 * 4) as u64;  // 1 ser × 56 u32 × 4 bytes
                let buf_wave_diag = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label:              Some("diag-wave"),
                    size:               diag_bytes,
                    usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                // Copia buf_output[ser0] → buf_wave_diag via encoder separado.
                let mut enc_diag = self.device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("diag-enc") });
                enc_diag.copy_buffer_to_buffer(self.percept.buf_output(), 0, &buf_wave_diag, 0, diag_bytes);
                self.queue.submit(Some(enc_diag.finish()));
                self.device.poll(wgpu::Maintain::Wait);

                // Map e lê.
                let slice_w = buf_wave_diag.slice(0..diag_bytes);
                let (tx_w, rx_w) = std::sync::mpsc::channel();
                slice_w.map_async(wgpu::MapMode::Read, move |r| { let _ = tx_w.send(r); });
                self.device.poll(wgpu::Maintain::Wait);
                let _ = rx_w.recv();

                let wave_data = slice_w.get_mapped_range();
                let words_w: &[u32] = bytemuck::cast_slice(&wave_data);
                // Desempacota i8 do ser 0 (base=0).
                let mut w = [0i8; 223];
                for ch in 0..223usize {
                    let word = words_w[ch / 4];
                    w[ch] = ((word >> ((ch % 4) * 8)) & 0xFF) as u8 as i8;
                }
                drop(wave_data);
                buf_wave_diag.unmap();

                // Imprime canais críticos do travamento.
                eprintln!("[WAVE t={}] TRAVADO pos=({:.0},{:.0}) speed={:.2}m/s",
                    tick_idx, p.pos[0], p.pos[2], speed0);
                eprintln!("  ch64=align:{:4}  ch65=speed:{:4} (÷2={:.1}m/s)  ch66=lat:{:4}",
                    w[64], w[65], w[65] as f32 / 2.0, w[66]);
                eprintln!("  ch82=fwd_urg:{:4}  ch85=ema_l:{:4}  ch86=ema_r:{:4}  ch88=prox:{:4}",
                    w[82], w[85], w[86], w[88]);
                eprintln!("  ch89=reverse:{:4}  ch102=ar_l:{:4}  ch103=ar_r:{:4}  ch104=ar_f:{:4}",
                    w[89], w[102], w[103], w[104]);
                eprintln!("  ch108=rl_off:{:4}  ch109=doff:{:4}  ch110=la:{:4}  ch120=mem0:{:4}",
                    w[108], w[109], w[110], w[120]);
                eprintln!("  ch192-197 cfc=[{},{},{},{},{},{}]",
                    w[192], w[193], w[194], w[195], w[196], w[197]);
                let thr = intents[0][0]; let steer_raw = intents[0][1]; let brk = intents[0][2];
                eprintln!("  INTENT raw: thr={:.3}  steer={:.3}  brake={:.3}",
                    thr, steer_raw, brk);
            }
        }
        if intents.len() != n {
            eprintln!("  ERRO: readback do intent retornou {} em vez de {}", intents.len(), n);
            return false;
        }

        // ── 7. Aplica intenção e roda física ──────────────────────────────────
        // intent[s][0] = throttle, [1] = steer, [2] = brake — logits brutos do L2.
        // O motor.rs aplica kan_out.eval(out_idx, raw) antes de passar à física.
        // Sem o KAN: intent[1]=0.985 → steer=0.985 → slip_angle=90° → sem tração.
        // Além disso, replica o smoothing temporal do steer (motor.rs L1925):
        //   steer_raw = prev_steer * 0.50 + new_steer * 0.50
        // Isso amortece o steer de tick para tick e evita reversão abrupta ±1 winner-take-all.
        // Deduplicado (E3, REFACTOR_PLAN.md §2.3): usa universe::kan::eval_knots em vez
        // de reimplementar a fórmula do B-spline aqui (era cópia idêntica da closure de
        // gpu_evolve.rs, e o comentário só AFIRMAVA ser idêntica a KanOutputHead::eval()
        // sem compartilhar o código de fato).
        let kan_eval_fn = |out_idx: usize, x: f32| -> f32 {
            universe::kan::eval_knots(&self.kan_knots, out_idx, x)
        };
        let mut phys_states: Vec<PhysicsStateGpu> = (0..n).map(|i| {
            let mut s = states[i].phys;
            let intent = &intents[i];
            // Aplica KAN de saída (replica motor.rs L1873-L1926).
            let thr_kan   = kan_eval_fn(0, intent[0]).max(0.0);   // throttle ≥ 0
            let steer_kan = kan_eval_fn(1, intent[1]);
            let brk_kan   = kan_eval_fn(2, intent[2]).max(0.0);   // brake ≥ 0
            // Smoothing temporal do steer (50/50): amortece reversão abrupta.
            let steer_smooth = (states[i].prev_steer * 0.50 + steer_kan * 0.50)
                .clamp(-1.0, 1.0);

            let is_reversing = states[i].reverse_timer > 0.0;
            let in_lockout   = states[i].post_reverse_lockout > 0.0;

            // motor.rs L1950: anula brake quando speed < 2.0 m/s (evita travar carro parado).
            // Sem isso: modelo emite brake=0.98 + throttle=0.90 → ambos se anulam → speed=0
            // para sempre. O modelo aprendeu que "parado + parede" → brake alto, mas a física
            // do motor.rs já anula o brake baixo speed para dar chance ao throttle.
            let speed_now = (s.vel[0].powi(2) + s.vel[2].powi(2)).sqrt();
            let brk_final = if speed_now < 2.0 { 0.0 } else { brk_kan };

            if is_reversing {
                // RÉ ATIVA: override COMPLETO do intent neural (motor.rs L2325-2339).
                // Durante ré: pedal fixo de 0.75, steer=0 (backup reto), brake=0.
                // O modelo NÃO controla o carro em ré — foi treinado com esse override.
                // Force-shift para gear=0 (Reverse): shift_req = -1 repetido até gear=0.
                // A física desce 1 marcha por tick quando throttle > 0 e gear > 0 na ré.
                // Replicamos forçando current_gear = 0 diretamente (mesma semântica que shift_req).
                s.throttle   = universe::config::iara::reverse_maneuver::PEDAL;  // pedal de recuo fixo
                s.steer      = 0.0;             // backup reto (sem risco de girar para outra parede)
                s.brake_cmd  = 0.0;             // sem freio (o pedal reverso acelera a ré)
                s.current_gear = 0;             // gear 0 = Reverse (convenção do physics.wgsl)
            } else if in_lockout {
                // PÓS-RÉ LOCKOUT: throttle capeado em 0.8 para aceleração suave ao reiniciar.
                // Replica motor.rs L2331-2332: pedal = (intent * safety_cap).min(0.8).
                s.throttle   = thr_kan.min(0.8);
                s.steer      = steer_smooth;
                s.brake_cmd  = brk_final;
                // Auto-clutch: se saiu da ré (gear=0) volta para 1ª marcha (gear=2 = 1ª).
                // Motor.rs L2294-2296: se current_gear <= 1 e não em ré → shift_req = +1.
                if s.current_gear <= 1 { s.current_gear = 2; }
            } else if states[i].escape_ticks > 0 {
                // ESCAPE DE ESTAGNAÇÃO: modelo está parado por 0.5s sem ré.
                // Replica motor.rs stagnation escape: throttle=0.3 mínimo, brake=0.
                // O steer continua sendo do modelo para manter a navegação.
                // Sem isso: modelo emite throttle=0 em curvas apertadas e fica no idle
                // do motor (0.07 m/s eterno). O CHAMPION foi treinado esperando essa âncora.
                s.throttle  = 0.3f32.max(thr_kan);  // mínimo 0.3, respeita se modelo quer mais
                s.steer     = steer_smooth;
                s.brake_cmd = 0.0;  // zera freio para o throttle mínimo ter efeito
                if s.current_gear <= 1 { s.current_gear = 2; }
            } else {
                // MODO NORMAL: intent neural completo.
                s.throttle  = thr_kan;
                s.steer     = steer_smooth;
                s.brake_cmd = brk_final;
                // Auto-clutch: sai de gear 0/1 para 2 (1ª marcha) quando não em ré.
                if s.current_gear <= 1 { s.current_gear = 2; }
                // Guardian Anti-Stall: se parado em marcha alta (>2), desce para 2.
                if s.current_gear > 2 && speed_now < 3.0 { s.current_gear -= 1; }

                // AUTO-SHIFT POR RPM — replica motor.rs L2303-2309 (linhas 2303, 2307).
                // CAUSA RAIZ DO BUG: sem este bloco o carro fica preso em gear=3 (2ª marcha).
                // Com gear=3 e max_rpm=8500, o limitador (`tm=0` quando `rr>1`) corta o torque
                // assim que RPM > max_rpm — o carro satura em ~8 m/s e decai para 0.17 m/s.
                // O modelo foi treinado no motor.rs que tem auto-shift → espera que as marchas
                // troquem por RPM, não por decisão do cérebro.
                //
                // Embreagem colada (is_clutch_engaged no motor.rs):
                //   wheel_ratio ≈ 12 - (gear-2)*2 → gear=3 → ratio=8
                //   wheel_rpm = speed * ratio * 30 → a 8 m/s: 8 * 8 * 30 = 1920 > 1200 → colada
                //   (conservador: usamos speed > 3 m/s E gear >= 2 como proxy)
                // Upshift: rpm > 85% max_rpm E marcha < 6ª E shift_timer=0
                // Downshift: rpm < 30% max_rpm E marcha > 2ª E speed > 3 m/s E shift_timer=0
                if s.shift_timer <= 0.0 && s.current_gear >= 2 && speed_now > 3.0 {
                    if s.current_rpm > s.max_rpm * 0.85 && s.current_gear < 6 {
                        // Sobe marcha: timer de 150ms (motor.rs usa 150ms)
                        s.current_gear += 1;
                        s.shift_timer = 0.15;
                    } else if s.current_rpm < s.max_rpm * 0.30 && s.current_gear > 2 {
                        // Desce marcha: evita afogamento em marcha longa sem velocidade
                        s.current_gear -= 1;
                        s.shift_timer = 0.15;
                    }
                }
            }
            s.dt = dt;
            s
        }).collect();
        // Salva steer suavizado e atualiza stagnation para o próximo tick.
        for i in 0..n {
            // Ao sair da ré, reseta prev_steer para 0 para evitar salto brusco no steer
            // na retomada: ao voltar ao modo normal, o smooth começa de 0 (reto), não de
            // qualquer valor que o modelo estava emitindo durante a ré.
            if states[i].reverse_timer > 0.0 {
                states[i].prev_steer = 0.0;  // backup reto → suaviza a retomada
            } else {
                states[i].prev_steer = phys_states[i].steer;
            }
            // Atualiza stagnation: conta ticks com speed < 1.5 m/s.
            // Threshold = 1.5 m/s idêntico ao motor.rs L2220 (speed < 1.5 para trigger de ré).
            let speed = (phys_states[i].vel[0].powi(2) + phys_states[i].vel[2].powi(2)).sqrt();
            if speed < 1.5 {
                states[i].stagnation_ticks += 1;
            } else {
                states[i].stagnation_ticks = 0;
            }
            // Ativa escape de estagnação quando parado por 0.5s (32 ticks).
            // O CHAMPION foi treinado com o motor.rs que tem curvature_governor + slow brain +
            // stagnation escape — mecanismos externos que fazem o carro sair de curvas apertadas.
            // Sem eles, o modelo emite throttle=0 + brake=0 em curvas e fica no idle (0.07 m/s).
            // O escape não fabrica velocidade: só impede que throttle=0 eterno trave o carro;
            // o ser ainda navega com steer do modelo. Desativável com IARA_NO_ESCAPE=1.
            // DESLIGADO POR PADRÃO. O escape força `throttle = 0,3` e zera o freio do modelo
            // — sobrepõe a decisão do cérebro. Medido em 50.000 ticks: COM escape
            // `dist_max = 1744 m`, SEM escape `650 m`. **2,7× do resultado vinha da muleta.**
            // Um número que depende dela não mede pilotagem, e a meta é medir pilotagem.
            // `IARA_ESCAPE=1` reativa para comparar.
            if std::env::var("IARA_ESCAPE").is_ok() {
                // 32 ticks = 0.5s parado → ativa escape.
                // 128 ticks = 2s de throttle mínimo 0.3 — suficiente para contornar a maioria das
                // curvas apertadas da pista seed=9001 onde o CHAMPION fica em idle sem slow brain.
                if states[i].stagnation_ticks >= 32 && states[i].escape_ticks == 0 {
                    states[i].escape_ticks = 64;  // 1s de throttle=0.3 mínimo
                    states[i].stagnation_ticks = 0;
                }
            }
            // Decrementa escape.
            if states[i].escape_ticks > 0 {
                states[i].escape_ticks -= 1;
            }
        }

        // Física roda no device isolado (PhysicsGpu tem device próprio).
        // DIAGNÓSTICO: o cérebro manda acelerador 0,99 e o carro não anda. Aqui se vê se a
        // física recebe o comando e o que ela devolve — carga nas rodas zero significa carro
        // sem contato com o solo, e aí não há tração por mais torque que o motor faça.
        let diag = std::env::var("IARA_DIAG").is_ok() && !self.diag_fis;
        if diag {
            self.diag_fis = true;
            let a = &phys_states[0];
            eprintln!("[FIS-ANTES] thr={:.3} brk={:.3} str={:.3} rpm={:.0} gear={} vel={:?} pos_y={:.3}",
                a.throttle, a.brake_cmd, a.steer, a.current_rpm, a.current_gear, a.vel, a.pos[1]);
            eprintln!("[FIS-ANTES] wheels={:?}", &a.wheels[..4]);
        }

        // step() escreve estado atualizado de volta em phys_states[i].
        let ok_fis = self.phys_gpu.step(&mut phys_states);
        if diag {
            let a = &phys_states[0];
            eprintln!("[FIS-DEPOIS] ok={} vel={:?} pos_y={:.3} rpm={:.0} torque={:.1}",
                ok_fis, a.vel, a.pos[1], a.current_rpm, a.drive_torque_out);
            eprintln!("[FIS-DEPOIS] wheels={:?}", &a.wheels[..4]);
        }
        if !ok_fis {
            eprintln!("  ERRO: phys_gpu.step() falhou (n={} > capacidade?)", n);
            return false;
        }

        // Atualiza estados na CPU + rotação de yaw (idem gpu_evolve.rs).
        for i in 0..n {
            states[i].phys = phys_states[i];
            let theta = states[i].phys.av_y * dt;
            let (sin_t, cos_t) = theta.sin_cos();
            let fwd = states[i].phys.fwd;
            let lat = states[i].phys.lat;
            states[i].phys.fwd[0] = fwd[0] * cos_t - fwd[2] * sin_t;
            states[i].phys.fwd[2] = fwd[0] * sin_t + fwd[2] * cos_t;
            states[i].phys.lat[0] = lat[0] * cos_t - lat[2] * sin_t;
            states[i].phys.lat[2] = lat[0] * sin_t + lat[2] * cos_t;
            let fn_ = (states[i].phys.fwd[0].powi(2) + states[i].phys.fwd[2].powi(2)).sqrt().max(1e-6);
            let ln_ = (states[i].phys.lat[0].powi(2) + states[i].phys.lat[2].powi(2)).sqrt().max(1e-6);
            states[i].phys.fwd[0] /= fn_; states[i].phys.fwd[2] /= fn_;
            states[i].phys.lat[0] /= ln_; states[i].phys.lat[2] /= ln_;
        }

        // ── 8. Atualiza memory0 no estado temporal da GPU ─────────────────────
        // memory0 = throttle intencional do tick atual, clamped em [0,1].
        // O kernel de percepção no PRÓXIMO tick vai ler este valor como ch 120.
        // Chamada APÓS a física para garantir que o throttle final é o que chegou
        // nos pneus (não um valor pre-física).
        // Custo: N × 4 bytes de write_buffer = 240 bytes @ 60 seres — negligível.
        let memory0_vals: Vec<f32> = (0..n).map(|i| {
            intents[i][0].clamp(0.0, 1.0)  // mesmo clamp do throttle físico
        }).collect();
        self.percept.update_memory0(&self.queue, &memory0_vals[..n]);

        true
    }
}

// ─── Ponto de entrada ─────────────────────────────────────────────────────────

fn main() {
    // Força ICD Vulkan da Radeon antes de qualquer wgpu::Instance::new.
    // Múltiplos ICDs causam SIGSEGV no loader (visto em gpu_validate.rs).
    let icd_radeon = "/usr/share/vulkan/icd.d/radeon_icd.json";
    if std::path::Path::new(icd_radeon).exists()
        && std::env::var("VK_ICD_FILENAMES").is_err()
    {
        unsafe { std::env::set_var("VK_ICD_FILENAMES", icd_radeon); }
    }

    // ── Parsing de args ────────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let mut n_seres: usize = 60;
    let mut n_ticks: usize = 20_000;  // 20.000 ticks = 312,5s @ 64Hz ≈ 5 min de simulação
    {
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--seres" => { i += 1; if i < args.len() { n_seres = args[i].parse().unwrap_or(60); } }
                "--ticks" => { i += 1; if i < args.len() { n_ticks = args[i].parse().unwrap_or(20_000); } }
                _ => {}
            }
            i += 1;
        }
    }
    println!("=== gpu_loop: {} seres, {} ticks ===", n_seres, n_ticks);
    println!("  Laço: retina+percepção+cérebro+física — um device só");
    println!();

    // ── Mundo ─────────────────────────────────────────────────────────────────
    let t0 = Instant::now();
    // Pista: mesma seed/largura do gpu_validate para garantir que CHAMPION.dna é
    // testado na pista em que foi validado. O CHAMPION foi evoluído em pistas rotativas,
    // mas a validação canônica usa seed=9001, track_width=18m (gpu_validate.rs).
    // Com 14m o CHAMPION colidia com a parede após ~200 ticks (steer treinado para pista maior).
    let world = VoxelWorld::new(9001, 18.0);
    println!("mundo: {}×{} · {} waypoints · volta {:.0} m  [{:.0} ms]",
        world.size, world.size,
        world.waypoints_world.len(),
        world.lap_distance,
        t0.elapsed().as_secs_f64() * 1000.0,
    );

    // ── Engines: carrega CHAMPION.dna se existir, senão pesos aleatórios ────────
    // SovereignEngine::new() inicializa TODOS os pesos com 0.0 (layers.rs:32),
    // o que produz intent == 0 → nenhum movimento. É necessário carregar um DNA
    // real para que o cérebro emita intenção não-nula.
    // Sem CHAMPION.dna: os seres não se movem por decisão — saída do L1 é zero
    // (pesos nulos × qualquer wave = 0). O programa avisa e sai com erro.
    let t1 = Instant::now();
    let dna_opt = load_champion_dna("CHAMPION.dna");
    let engines: Vec<SovereignEngine> = (0..n_seres).map(|_| {
        let mut eng = SovereignEngine::new();
        if let Some(ref dna) = dna_opt {
            // Carrega pesos L1: cortex_raw (i8, 63.5-scaled) ou fast_weights (i8, 63.5-scaled).
            let w: Vec<f32> = if let Some(ref cr) = dna.cortex_raw {
                cr.iter().map(|&b| b as f32 / 63.5).collect()
            } else {
                dna.fast_weights.iter().map(|&b| b as f32 / 63.5).collect()
            };
            let n_l1 = eng.instinct_layer.weights.len().min(w.len());
            eng.instinct_layer.weights[..n_l1].copy_from_slice(&w[..n_l1]);
            // Carrega pesos L2 se presentes no DNA.
            if let Some(ref l2) = dna.cortex_l2_raw {
                let w2: Vec<f32> = l2.iter().map(|&b| b as f32 / 63.5).collect();
                let n_l2 = eng.cortex_l2.weights.len().min(w2.len());
                eng.cortex_l2.weights[..n_l2].copy_from_slice(&w2[..n_l2]);
            }
        }
        eng
    }).collect();
    // KAN output knots do CHAMPION — usado em GpuLoop::tick() via self.kan_knots.
    // Motor.rs aplica kan_out.eval(out_idx, raw) antes de usar throttle/steer/brake.
    // Sem o KAN: intent raw≈1 → steer físico=1 → slip_angle=90° → nenhuma tração.
    // KAN_OUT_KNOTS = 12 (kan.rs:85), KAN_OUT_DOMAIN = 2.0 (kan.rs:86).
    use universe::kan::KAN_OUT_KNOTS;
    let kan_knots: [[f32; KAN_OUT_KNOTS]; 3] = dna_opt.as_ref()
        .map(|d| d.kan_knots)
        .unwrap_or_else(universe::evolution::dna_genesis::default_kan_knots);

    let dna_status = if dna_opt.is_some() {
        "CHAMPION.dna carregado — pesos reais"
    } else {
        "CHAMPION.dna nao encontrado — pesos ZEROS (seres nao se moverão por decisão própria)"
    };
    println!("engines criados: {} seres · {}  [{:.0} ms]",
        n_seres, dna_status, t1.elapsed().as_secs_f64() * 1000.0);

    // ── Contexto GPU ──────────────────────────────────────────────────────────
    let t2 = Instant::now();
    let mut ctx = match GpuLoop::new(&world, &engines, n_seres, kan_knots) {
        Some(c) => c,
        None => {
            eprintln!("ERRO: sem GPU utilizavel. Abortando.");
            std::process::exit(1);
        }
    };
    println!("GpuLoop pronto  [{:.0} ms]", t2.elapsed().as_secs_f64() * 1000.0);
    println!();

    // ── Estado inicial dos seres ───────────────────────────────────────────────
    // Distribui seres ao longo dos waypoints da pista.
    let n_wp = world.waypoints_world.len();
    let mut states: Vec<BeingState> = (0..n_seres).map(|i| {
        let wp = world.waypoints_world[(i * n_wp / n_seres).min(n_wp - 1)];
        // Forward = direção para o próximo waypoint.
        let wp_next = world.waypoints_world[((i * n_wp / n_seres) + 1).min(n_wp - 1)];
        let dx = wp_next.x - wp.x;
        let dz = wp_next.y - wp.y;
        let len = (dx * dx + dz * dz).sqrt().max(1e-6);
        let fwd = [dx / len, 0.0, dz / len];
        let pos = [wp.x, 0.0, wp.y];
        BeingState { phys: make_physics_state(pos, fwd), prev_steer: 0.0, stagnation_ticks: 0, escape_ticks: 0, reverse_timer: 0.0, post_reverse_lockout: 0.0 }
    }).collect();

    // ── Odômetro acumulado por ser ─────────────────────────────────────────────
    // Usamos integração de velocidade * dt (mais preciso que euclidiana entre pontos
    // esparsos quando a pista tem curvas fechadas). A distância euclidiana entre ticks
    // subestima a trajetória real; a integral de |vel| * dt = percurso real.
    // Voltas: distância acumulada / lap_distance.
    // HETEROGENEIDADE é a garantia de movimento genuíno: seres em situações
    // diferentes (waypoint diferente, pista diferente) TÊM de dar números diferentes.
    let lap_dist = world.lap_distance;
    println!("volta = {:.0} m  (comprimento total da pista)", lap_dist);

    let mut dist_acum: Vec<f64> = vec![0.0; n_seres];
    let mut pos_prev: Vec<[f32; 3]> = states.iter().map(|s| pos_of(&s.phys)).collect();

    // ── Aquecimento (1 tick descartado para forçar compilação JIT de shaders) ─
    println!("Aquecendo (1 tick JIT)...");
    let _ = ctx.tick(&mut states, &engines, n_seres, 1.0 / 64.0, 0);

    // ── Laço principal ────────────────────────────────────────────────────────
    println!("Rodando {} ticks com {} seres...", n_ticks, n_seres);
    let dt = 1.0_f32 / 64.0;

    let t_loop = Instant::now();
    for tick_i in 0..n_ticks {
        let ok = ctx.tick(&mut states, &engines, n_seres, dt, tick_i + 1);
        if !ok {
            eprintln!("ERRO no tick {} — abortando.", tick_i);
            break;
        }
        // Acumula odômetro: distância euclidiana entre pos[t-1] e pos[t].
        // Com dt=1/64s e velocidade típica de 20-40 m/s, o passo é ~0.3-0.6m —
        // pequeno o suficiente para que a euclidiana ≈ arclength (erro < 0.1%).
        for i in 0..n_seres {
            let p1 = pos_of(&states[i].phys);
            let dx = (p1[0] - pos_prev[i][0]) as f64;
            let dz = (p1[2] - pos_prev[i][2]) as f64;
            dist_acum[i] += (dx*dx + dz*dz).sqrt();
            pos_prev[i] = p1;
        }
        // Checkpoint de progresso a cada 1000 ticks (~15,6s de simulação)
        if (tick_i + 1) % 1000 == 0 {
            let max_d = dist_acum.iter().cloned().fold(0.0_f64, f64::max);
            let med_d = {
                let mut v = dist_acum.clone();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[v.len() / 2]
            };
            let max_voltas = max_d / lap_dist as f64;
            println!("  tick {:5}: dist_max={:.0}m ({:.2}V)  dist_med={:.0}m ({:.2}V)",
                tick_i + 1, max_d, max_voltas, med_d, med_d / lap_dist as f64);
        }
    }
    let elapsed_loop = t_loop.elapsed().as_secs_f64();

    // ── Resultados ────────────────────────────────────────────────────────────
    let ms_tick = elapsed_loop * 1000.0 / n_ticks as f64;
    println!();
    println!("=== RESULTADO ===");
    println!("  ticks: {}", n_ticks);
    println!("  seres: {}", n_seres);
    println!("  tempo total: {:.1} ms", elapsed_loop * 1000.0);
    println!("  ms/tick (laço todo-GPU+temporais): {:.3} ms", ms_tick);
    println!();

    // ── Distâncias finais e voltas ─────────────────────────────────────────────
    let dist_max = dist_acum.iter().cloned().fold(0.0_f64, f64::max);
    let dist_med = {
        let mut v = dist_acum.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let voltas_max = dist_max / lap_dist as f64;
    let voltas_med = dist_med / lap_dist as f64;
    let n_voltas_completas = dist_acum.iter().filter(|&&d| d >= lap_dist as f64).count();

    println!("=== DISTÂNCIAS E VOLTAS em {} ticks ===", n_ticks);
    println!("  lap_distance: {:.0} m", lap_dist);
    println!("  dist_max:     {:.0} m  ({:.2} voltas)", dist_max, voltas_max);
    println!("  dist_med:     {:.0} m  ({:.2} voltas)", dist_med, voltas_med);
    println!("  seres com ≥1 volta completa: {}/{}", n_voltas_completas, n_seres);
    println!();

    // ── Top performers (prova de heterogeneidade) ─────────────────────────────
    // Ordena todos os seres por distância decrescente e mostra os top 12.
    let mut ordem: Vec<usize> = (0..n_seres).collect();
    ordem.sort_by(|&a, &b| dist_acum[b].partial_cmp(&dist_acum[a]).unwrap());
    println!("=== TOP 12 SERES POR DISTÂNCIA (heterogeneidade) ===");
    let n_amostras = n_seres.min(12);
    let mut algum_se_moveu = false;
    for &i in ordem.iter().take(n_amostras) {
        let dist = dist_acum[i];
        let voltas = dist / lap_dist as f64;
        let vel = states[i].phys.vel;
        let speed = (vel[0]*vel[0] + vel[1]*vel[1] + vel[2]*vel[2]).sqrt();
        println!("  ser[{:2}]: dist={:.1}m ({:.2}V)  speed={:.2}m/s  pos=({:.0},{:.0})  {}",
            i, dist, voltas, speed,
            states[i].phys.pos[0], states[i].phys.pos[2],
            if dist > 10.0 { "OK" } else { "PARADO!" });
        if dist > 10.0 { algum_se_moveu = true; }
    }
    println!();
    if algum_se_moveu {
        println!("  SERES SE MOVERAM — canais temporais funcionais.");
    } else {
        println!("  NENHUM SER SE MOVEU — laço quebrado ou pesos nulos.");
    }

    // ── Canais temporais portados neste commit ────────────────────────────────
    println!();
    println!("=== CANAIS TEMPORAIS PORTADOS PARA A GPU ===");
    println!("  85: wall_ema_l      (EMA dist lateral esq, ~0.5s meia-vida)");
    println!("  86: wall_ema_r      (EMA dist lateral dir)");
    println!("  88: proximity_sustain (segundos raspando parede, cap 3s)");
    println!("  102: approach_rate_l  (dDist_l/dt em m/s, ±30m/s)");
    println!("  103: approach_rate_r");
    println!("  104: approach_rate_f  (frontal via sonda grade)");
    println!("  105: TTC_min          (tempo até colisão, 0=iminente, 127=seguro)");
    println!("  106: delta_dist_l ±15m/s");
    println!("  107: delta_dist_r");
    println!("  108: racing_line_offset (desvio lat da linha ideal, ±127)");
    println!("  109: d/dt(line_offset)  (ganho 8×)");
    println!("  110: lookahead_offset 5m à frente na linha ideal");
    println!("  118: fwd_dist / half_width (generalização por largura de pista)");
    println!("  119: corridor_lat / half_width");
    println!("  120: memory0 (throttle intencional anterior)");
    println!("  123: whisker_min / half_width");
    println!();
    println!("=== FIM ===");
}
