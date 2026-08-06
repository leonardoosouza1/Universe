//! Gerente de estado residente na GPU — ponto único de dispositivo compartilhado.
//!
//! # Por que este arquivo existe
//!
//! O módulo `gpu/mod.rs` (retina) abre seu próprio `wgpu::Device`. Quando física,
//! cérebro e percepção fizeram o mesmo cada um, teríamos 3-4 dispositivos abertos em
//! paralelo na mesma GPU, desperdiçando memória de command buffers e impossibilitando
//! compartilhar buffers entre subsistemas (a retina produziria pixels num buffer que o
//! cérebro não enxerga, exigindo readback+reupload). Este struct resolve isso: **um
//! dispositivo, todos os buffers residentes**.
//!
//! # Conta de bytes em trânsito por tick (60 seres, configuração nominal)
//!
//! Buffers que SOBEM da CPU uma vez por geração (estáticos por pista):
//!   grade     : 320×320 × 4 bytes (u32) = 409.600 bytes
//!   height_map: 320×320 × 4 bytes (f32) = 409.600 bytes
//!   banking   : 320×320 × 4 bytes (f32) = 409.600 bytes
//!   flow      : 320×320 × 8 bytes (Vec2 = [f32;2]) = 819.200 bytes
//!   subtotal estático = 2.048.000 bytes ≈ 2 MB
//!
//! Buffers que SOBEM uma vez por geração por depender dos pesos do ser (vivos):
//!   pesos instinct_layer (BitLinear 223→96, f32):
//!     223 × 96 × 4 = 85.632 bytes × 60 seres = 5.137.920 bytes
//!   pesos cortex_l2 (FloatLinear 96→48, f32):
//!     96 × 48 × 4 = 18.432 bytes × 60 = 1.105.920 bytes
//!   pesos cortex_l3 (FloatLinear 48→24, f32):
//!     48 × 24 × 4 = 4.608 bytes × 60 = 276.480 bytes
//!   subtotal pesos = 6.520.320 bytes ≈ 6,2 MB
//!
//! POR TICK — só o que MUDA entre ticks (o mínimo):
//!   SUBIDA → CPU→GPU:
//!     poses (posição+direção de cada ser vivo):
//!       60 × 32 bytes (PosGpu: [f32;8]) = 1.920 bytes
//!   DESCIDA → GPU→CPU:
//!     intenção motora (throttle, steer, brake × 60 seres):
//!       60 × 3 × 4 bytes = 720 bytes
//!   TOTAL POR TICK = 2.640 bytes ≈ 2,6 kB
//!
//! Estado CfC (60×96 f32 = 23.040 bytes) FICA residente na GPU entre ticks.
//! Só desce ao fim da geração para serializar a evolução.
//!
//! # Estado da retina por tick
//!
//! RETINA_TOTAL = 14×126 = 1.764 pixels × 4 bytes (u32) = 7.056 bytes × 60 seres = 423.360 bytes.
//! Esses bytes NUNCA descem para a CPU no caminho quente — ficam num buf_retina_saida
//! residente que o kernel do cérebro lê diretamente no passo seguinte.

use std::borrow::Cow;
use wgpu::util::DeviceExt;

use crate::config::iara::{
    brain::{SENSORY_WAVE_SIZE, L1_NEURONS, L2_NEURONS, L3_NEURONS},
    vision::RETINA_TOTAL,
};
use crate::world::VoxelWorld;

// ─── Capacidade máxima de seres suportada pelo CoreGpu. ──────────────────
// 60 é o nominal; alocamos 64 para alinhamento de 64 threads/workgroup e headroom.
const MAX_CAPACITY: usize = 64;

// ─── Tamanho da grade do mundo. Declarado aqui para a conta de bytes. ───────
// Lido de world.rs: `let size = 320`.
pub const GRID_SIZE: usize = 320;

// ─── Dimensões dos pesos, por ser. ─────────────────────────────────────────
// BitLinear 223→96: entrada é SENSORY_WAVE_SIZE (calculado = 223), saída L1=96.
// FloatLinear 96→48: entrada L1_NEURONS, saída L2_NEURONS.
// FloatLinear 48→24: entrada L2_NEURONS, saída L3_NEURONS.
const W1_TOTAL: usize = SENSORY_WAVE_SIZE * L1_NEURONS; // 223×96 = 21.408 f32
const W2_TOTAL: usize = L1_NEURONS * L2_NEURONS;         // 96×48  =  4.608 f32
const W3_TOTAL: usize = L2_NEURONS * L3_NEURONS;         // 48×24  =  1.152 f32

// CfC state (96 dimensões, uma fatia por ser, f32).
const CFC_DIM: usize = L1_NEURONS; // 96

// ─── Layout de postura por ser (GPU-side). ─────────────────────────────────
// 8 × f32 = 32 bytes, alinhado a 16 (requisito wgpu para storage buffer).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PosGpu {
    /// [pos.x, pos.y, pos.z, padding] + [fwd.x, fwd.z, right.x, right.z]
    pub pos_pad:  [f32; 4],
    pub dirs:     [f32; 4],
}

// ─── Intenção motora devolvida pela GPU por tick. ──────────────────────────
// 3 × f32 + padding = 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct IntentGpu {
    pub throttle: f32,
    pub steer:    f32,
    pub brake:    f32,
    pub _pad:     f32,
}

/// Contexto de GPU compartilhado por todos os kernels.
///
/// Mantém um único `wgpu::Device` aberto sem surface (compute-only). Todos os
/// buffers residentes ficam aqui — a retina, o cérebro e a física lêem e escrevem
/// nesses buffers via bind groups, sem precisar de readback intermediário.
///
/// # Ciclo de vida
///
/// ```text
/// CoreGpu::new(mundo, capacidade)   ← uma vez por pista
///     |
///     ├── sobe grade, height_map, banking, flow  (estáticos)
///     └── sobe pesos dos seres                   (uma vez por geração)
///
/// tick(poses, ...)                     ← a cada tick da simulação
///     |
///     ├── escreve poses (2 kB)
///     ├── dispatch percepção (ASSUMIDO: PercepcaoGpu via kernel externo)
///     ├── dispatch cérebro  (ASSUMIDO: CerebroGpu via kernel externo)
///     ├── dispatch física   (ASSUMIDO: FisicaGpu via kernel externo)
///     └── readback intenção motora (720 bytes)
///
/// baixar_estado_cfc()                  ← uma vez por geração, antes da evolução
/// ```
pub struct CoreGpu {
    pub device:  wgpu::Device,
    pub queue:   wgpu::Queue,
    pub adapter: String,

    // ── Buffers de mundo (estáticos por pista, sobem em `novo`) ─────────────
    /// grade[320×320] → u32 (VoxelType cast).
    pub buf_grid:      wgpu::Buffer,
    /// height_map[320×320] → f32 (metros).
    pub buf_height:     wgpu::Buffer,
    /// banking[320×320] → f32 (radianos).
    pub buf_banking:    wgpu::Buffer,
    /// flow[320×320] → [f32;2] (direção normalizada na pista).
    pub buf_flow:       wgpu::Buffer,

    // ── Buffers de pesos (uma vez por geração) ────────────────────────────
    /// Pesos concatenados: ser0:[W1|W2|W3|CfC_state], ser1:..., ...
    /// W1 = L1_NEURONS × SENSORY_WAVE_SIZE f32 (instinct_layer, BitLinear, pesos são f32).
    /// W2 = L2_NEURONS × L1_NEURONS f32 (cortex_l2, FloatLinear).
    /// W3 = L3_NEURONS × L2_NEURONS f32 (cortex_l3, FloatLinear).
    /// CfC_state = CFC_DIM f32 (estado temporal — atualizado pelo cérebro in-place).
    ///
    /// Stride por ser = (W1_TOTAL + W2_TOTAL + W3_TOTAL + CFC_DIM) × sizeof(f32).
    pub buf_weights:      wgpu::Buffer,

    // ── Buffers de estado por tick ────────────────────────────────────────
    /// poses[capacidade] → PosGpu (escrito pela CPU a cada tick, 2 kB para 60 seres).
    pub buf_poses:      wgpu::Buffer,
    /// retina_saida[capacidade × RETINA_TOTAL] → u32 (dist_mm<<8 | material).
    /// NUNCA desce para a CPU no caminho quente. O kernel do cérebro lê daqui.
    pub buf_retina_output: wgpu::Buffer,
    /// intencao[capacidade] → IntencaoGpu (desce para a CPU a cada tick, 720 bytes).
    pub buf_intent:   wgpu::Buffer,
    /// Staging de leitura para intencao (MAP_READ).
    pub buf_intent_readback: wgpu::Buffer,

    // ── Metadados ─────────────────────────────────────────────────────────
    pub capacity:    usize,
    pub grid_size:     u32,
    pub cell_size:     f32,

    // ── Pipelines compilados ──────────────────────────────────────────────
    // ASSUMIDO: os kernels de percepção, cérebro e física são compilados aqui
    // quando seus módulos WGSL estiverem prontos. Por ora, placeholders.
    //
    // ASSUMIDO: pipeline_retina reutiliza o shader de src/gpu/retina.wgsl.
    // ASSUMIDO: pipeline_percepcao virá de src/gpu/percepcao.wgsl (agente C).
    // ASSUMIDO: pipeline_cerebro virá de src/gpu/cerebro.wgsl (agente B).
    // ASSUMIDO: pipeline_fisica virá de src/gpu/fisica.wgsl (agente A).
    pub pipeline_retina: wgpu::ComputePipeline,
    pub layout_retina:   wgpu::BindGroupLayout,
}

// Stride de pesos por ser, em f32 (não em bytes).
fn weights_stride_f32() -> usize {
    W1_TOTAL + W2_TOTAL + W3_TOTAL + CFC_DIM
}

impl CoreGpu {
    /// Abre o dispositivo e sobe os dados estáticos do mundo.
    ///
    /// Devolve `None` quando não há GPU utilizável — a simulação cai na CPU sem
    /// travar. `capacidade` deve ser >= número de seres por geração (60 nominal).
    ///
    /// `min(capacidade, MAX_CAPACITY)` seres são alocados; exceder 64 seres
    /// exige aumentar MAX_CAPACITY e recompilar (tamanho de buffer fixo em
    /// tempo de compilação para WGSL).
    pub fn new(world: &VoxelWorld, capacity: usize) -> Option<Self> {
        // ── Abre dispositivo: compute-only, sem surface, sem janela. ─────────
        // `compatible_surface: None` é o ponto: pede adaptador para COMPUTE.
        // O Bevy já faz isso no módulo de retina (mod.rs) — replicamos o mesmo
        // padrão para não inventar API nova.
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..Default::default()
        });
        let adapter_obj = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference:    wgpu::PowerPreference::HighPerformance,
                compatible_surface:  None,
                force_fallback_adapter: false,
            },
        ))?;
        let name = adapter_obj.get_info().name.clone();
        let (device, queue) = pollster::block_on(adapter_obj.request_device(
            &wgpu::DeviceDescriptor {
                label:             Some("iara-core"),
                required_features: wgpu::Features::empty(),
                required_limits:   wgpu::Limits::downlevel_defaults(),
                memory_hints:      wgpu::MemoryHints::Performance,
            },
            None,
        )).ok()?;

        let cap = capacity.min(MAX_CAPACITY).max(1);

        // ── Buffers estáticos do mundo ──────────────────────────────────────
        // Subindo grade como u32 (VoxelType cast).
        // 320×320 × 4 = 409.600 bytes.
        let grid_u32: Vec<u32> = world.grid.iter().map(|v| *v as u32).collect();
        let buf_grid = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("core-grid"),
            contents: bytemuck::cast_slice(&grid_u32),
            usage:    wgpu::BufferUsages::STORAGE,
        });

        // height_map: 320×320 × 4 = 409.600 bytes.
        let buf_height = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("core-height"),
            contents: bytemuck::cast_slice(&world.height_map),
            usage:    wgpu::BufferUsages::STORAGE,
        });

        // banking: 320×320 × 4 = 409.600 bytes.
        let buf_banking = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("core-banking"),
            contents: bytemuck::cast_slice(&world.banking),
            usage:    wgpu::BufferUsages::STORAGE,
        });

        // flow: Vec<Vec2> → [f32;2] por célula → 320×320×8 = 819.200 bytes.
        // `Vec2` é bytemuck::Pod em bevy 0.16 (layout [f32; 2]).
        let flow_raw: Vec<[f32; 2]> = world.flow.iter().map(|v| [v.x, v.y]).collect();
        let buf_flow = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("core-flow"),
            contents: bytemuck::cast_slice(&flow_raw),
            usage:    wgpu::BufferUsages::STORAGE,
        });

        // ── Buffer de pesos (por ser) ───────────────────────────────────────
        // Aloca espaço para `cap` seres; preenchido por `carregar_pesos()` antes
        // da primeira geração. COPY_DST para escrita pela CPU entre gerações.
        // Stride por ser em bytes = weights_stride_f32() × 4.
        // Para 60 seres: (21408+4608+1152+96) × 4 × 60 = 27264 × 4 × 60 = 6.543.360 bytes ≈ 6,2 MB.
        let bytes_weights = (weights_stride_f32() * cap * std::mem::size_of::<f32>()) as u64;
        let buf_weights = device.create_buffer(&wgpu::BufferDescriptor {
            label:               Some("core-weights"),
            size:                bytes_weights,
            // COPY_SRC é necessário porque o estado recorrente (CfC) vive DENTRO deste
            // buffer e precisa ser lido de volta no fim da geração. Sem a flag, o wgpu
            // recusa o copy_buffer_to_buffer com erro de validação — foi o que aconteceu.
            usage:               wgpu::BufferUsages::STORAGE
                                 | wgpu::BufferUsages::COPY_DST
                                 | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation:  false,
        });

        // ── Buffers de tick ────────────────────────────────────────────────
        // poses: 60 × 32 bytes = 1.920 bytes.
        let bytes_poses = (cap * std::mem::size_of::<PosGpu>()) as u64;
        let buf_poses = device.create_buffer(&wgpu::BufferDescriptor {
            label:               Some("core-poses"),
            size:                bytes_poses,
            usage:               wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation:  false,
        });

        // retina_saida: 60 × 1764 × 4 = 423.360 bytes.
        // COPY_SRC para poder copiar para buf de leitura se precisar inspecionar.
        let bytes_retina = (cap * RETINA_TOTAL * std::mem::size_of::<u32>()) as u64;
        let buf_retina_output = device.create_buffer(&wgpu::BufferDescriptor {
            label:               Some("core-retina-output"),
            size:                bytes_retina,
            usage:               wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation:  false,
        });

        // intencao: 60 × 16 bytes = 960 bytes (IntencaoGpu tem padding).
        let bytes_intent = (cap * std::mem::size_of::<IntentGpu>()) as u64;
        let buf_intent = device.create_buffer(&wgpu::BufferDescriptor {
            label:               Some("core-intent"),
            size:                bytes_intent,
            usage:               wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation:  false,
        });
        let buf_intent_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label:               Some("core-intent-readback"),
            size:                bytes_intent,
            usage:               wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation:  false,
        });

        // ── Pipeline da retina (reutiliza retina.wgsl) ─────────────────────
        // ASSUMIDO: o shader retina.wgsl aceita exatamente o layout abaixo
        // (uniform params, storage grade read-only, storage poses read-only,
        // storage saida write-only). Este é o layout existente em mod.rs.
        let shader_retina = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("retina"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                include_str!("retina.wgsl"),
            )),
        });
        let layout_retina = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("retina-layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),  // grade
                storage_entry(2, true),  // poses
                storage_entry(3, false), // saida
            ],
        });
        let pipe_layout_retina = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                None,
            bind_group_layouts:   &[&layout_retina],
            push_constant_ranges: &[],
        });
        let pipeline_retina = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:               Some("retina-pipeline"),
            layout:              Some(&pipe_layout_retina),
            module:              &shader_retina,
            entry_point:         Some("main"),
            compilation_options: Default::default(),
            cache:               None,
        });

        Some(Self {
            device,
            queue,
            adapter: name,
            buf_grid,
            buf_height,
            buf_banking,
            buf_flow,
            buf_weights,
            buf_poses,
            buf_retina_output,
            buf_intent,
            buf_intent_readback,
            capacity: cap,
            grid_size:  world.size as u32,
            cell_size:  world.cell_size,
            pipeline_retina,
            layout_retina,
        })
    }

    /// Sobe os pesos de todos os seres para a GPU.
    ///
    /// Chamado uma vez por geração (antes do primeiro tick), quando a seleção/
    /// mutação já produziu os pesos da nova geração. `n_seres` deve ser
    /// `<= self.capacity`.
    ///
    /// Layout no buffer:
    ///   ser 0: [W1: W1_TOTAL f32] [W2: W2_TOTAL f32] [W3: W3_TOTAL f32] [CfC: CFC_DIM f32]
    ///   ser 1: idem, stride = weights_stride_f32() f32
    ///
    /// ASSUMIDO: o chamador (CerebroGpu ou orquestrador) extrai os pesos de
    /// `SovereignEngine::instinct_layer.weights`, `cortex_l2.weights`,
    /// `cortex_l3.weights` e inicializa o estado CfC como zeros.
    pub fn upload_weights(
        &self,
        weights_w1: &[f32], // [n_seres × W1_TOTAL]
        weights_w2: &[f32], // [n_seres × W2_TOTAL]
        weights_w3: &[f32], // [n_seres × W3_TOTAL]
        n_beings: usize,
    ) {
        let n = n_beings.min(self.capacity);
        let stride = weights_stride_f32();
        let mut interleaved = vec![0.0f32; stride * n];
        for s in 0..n {
            let base = s * stride;
            // W1
            let w1s = s * W1_TOTAL;
            interleaved[base..base + W1_TOTAL]
                .copy_from_slice(&weights_w1[w1s..w1s + W1_TOTAL]);
            // W2
            let w2s = s * W2_TOTAL;
            let o2 = base + W1_TOTAL;
            interleaved[o2..o2 + W2_TOTAL]
                .copy_from_slice(&weights_w2[w2s..w2s + W2_TOTAL]);
            // W3
            let w3s = s * W3_TOTAL;
            let o3 = o2 + W2_TOTAL;
            interleaved[o3..o3 + W3_TOTAL]
                .copy_from_slice(&weights_w3[w3s..w3s + W3_TOTAL]);
            // CfC state: zeros (reset a cada geração, como CfCCell::reset()).
            // bytes [o3+W3_TOTAL .. o3+W3_TOTAL+CFC_DIM] já são 0 por `vec![0.0]`.
        }
        self.queue.write_buffer(&self.buf_weights, 0, bytemuck::cast_slice(&interleaved));
    }

    /// Escreve as poses dos seres vivos e dispara os kernels em sequência.
    ///
    /// # Ordem de dispatch (uma única `CommandEncoder`, sem readbacks intermediários)
    ///
    ///   1. **retina** — lê grade + poses, escreve buf_retina_saida
    ///   2. **percepção** — lê grade/flow/height/banking + buf_retina_saida + poses,
    ///      escreve canais semânticos (ASSUMIDO: PercepcaoGpu via buf separado)
    ///   3. **cérebro** — lê pesos + retina + canais semânticos + estado CfC,
    ///      escreve buf_intencao + atualiza estado CfC in-place
    ///   4. **física** — lê intencao + estado do corpo (ASSUMIDO: FisicaGpu via buf separado)
    ///
    /// # Readback
    ///
    /// Só `buf_intencao` desce para a CPU (720 bytes nominais). O estado CfC
    /// só desce ao fim da geração via `baixar_estado_cfc()`.
    ///
    /// ASSUMIDO: percepção e física têm seus próprios bind groups, criados pelos
    /// agentes B, C e A. Este método só chama o dispatch da retina; os demais
    /// serão encadeados ao mesmo encoder na integração final (GPU_PLANO.md §D).
    ///
    /// # Parâmetros
    ///
    /// - `poses`: posturas dos seres vivos neste tick (len = n_vivos).
    /// - `params_retina`: bytes do `ParamsGpu` do shader de retina (uniforme).
    ///   ASSUMIDO: o chamador constrói `ParamsGpu` com `pitches`, `n_seres`,
    ///   `grid_size`, `cell_size` e `fov_half`, exatamente como em `RetinaGpu::render`.
    ///
    /// Retorna as intenções motoras lidas da GPU: `Vec` de comprimento `n_vivos`.
    pub fn tick(
        &self,
        poses:         &[PosGpu],
        params_retina: &[u8],  // bytemuck::bytes_of(&ParamsGpu{...})
        n_alive:       usize,
    ) -> Vec<IntentGpu> {
        let n = n_alive.min(self.capacity);
        if n == 0 {
            return Vec::new();
        }

        // 1. Sobe poses (1.920 bytes para 60 seres — único tráfego CPU→GPU por tick).
        self.queue.write_buffer(
            &self.buf_poses,
            0,
            bytemuck::cast_slice(&poses[..n]),
        );

        // 2. Constrói o uniform de params da retina como buffer temporário.
        //    Criamos por tick porque `n_seres` pode variar (mortes ao longo da geração).
        let buf_params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("core-params-retina"),
            contents: params_retina,
            usage:    wgpu::BufferUsages::UNIFORM,
        });

        // 3. Bind group da retina (mesmo layout de mod.rs::RetinaGpu).
        let bg_retina = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   None,
            layout:  &self.layout_retina,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_grid.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_poses.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_retina_output.as_entire_binding() },
            ],
        });

        // 4. CommandEncoder único — todos os dispatches sem readback intermediário.
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("core-tick"),
        });

        // ── Dispatch retina ──────────────────────────────────────────────────
        // 64 threads/workgroup: 2 waves de 32 no RDNA2 (wave32 é o size correto).
        // Total de threads = n × RETINA_TOTAL = 60 × 1764 = 105.840.
        // Workgroups = ceil(105840 / 64) = 1.654.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:                    Some("pass-retina"),
                timestamp_writes:         None,
            });
            pass.set_pipeline(&self.pipeline_retina);
            pass.set_bind_group(0, &bg_retina, &[]);
            let total = (n * RETINA_TOTAL) as u32;
            pass.dispatch_workgroups(total.div_ceil(64), 1, 1);
        }

        // ── ASSUMIDO: dispatch percepção ─────────────────────────────────────
        // PercepcaoGpu (agente C) expõe um método como:
        //   fn bind_e_dispatch(enc: &mut CommandEncoder, nucleo: &CoreGpu, n: usize)
        // que cria o bind group lendo buf_retina_saida + buf_grade + buf_flow + buf_height
        // e escreve num buf_percepcao_saida (canais semânticos 16-127 + 128-191).
        //
        // ASSUMIDO: buf_percepcao_saida é um campo de CoreGpu (a adicionar na integração)
        // com tamanho = n × SENSORY_WAVE_SIZE × sizeof(i8).
        //   Para 60 seres: 60 × 223 = 13.380 bytes.

        // ── ASSUMIDO: dispatch cérebro ───────────────────────────────────────
        // CerebroGpu (agente B) expõe:
        //   fn bind_e_dispatch(enc: &mut CommandEncoder, nucleo: &CoreGpu, n: usize)
        // que lê buf_pesos (pesos + estado CfC) + buf_percepcao_saida,
        // atualiza CfC state in-place no buf_pesos e escreve buf_intencao.
        //
        // ASSUMIDO: 1 workgroup por ser (60 workgroups), cada um processa seu
        // forward completo: KAN → CfC → BitLinear(L1) → FloatLinear(L2) → FloatLinear(L3).

        // ── ASSUMIDO: dispatch física ────────────────────────────────────────
        // FisicaGpu (agente A) expõe:
        //   fn bind_e_dispatch(enc: &mut CommandEncoder, nucleo: &CoreGpu, n: usize, dt: f32)
        // que lê buf_intencao + estado do corpo (suspensão, velocidade — buf a adicionar)
        // e integra o passo de física in-place.

        // ── Readback de intencao (único readback por tick, 720 bytes) ─────────
        let bytes_int = (n * std::mem::size_of::<IntentGpu>()) as u64;
        enc.copy_buffer_to_buffer(
            &self.buf_intent, 0,
            &self.buf_intent_readback, 0,
            bytes_int,
        );

        self.queue.submit(Some(enc.finish()));

        // ── Mapeamento síncrono — bloqueia até GPU terminar. ──────────────────
        // `device.poll(Wait)` drena a fila antes de mapear.
        // Em produção, isso limita a ~1 frame de pipeline. Para a primeira versão
        // (equivalência garantida) é aceitável; async mapping é a otimização seguinte.
        let slice = self.buf_intent_readback.slice(0..bytes_int);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        let vals: &[IntentGpu] = bytemuck::cast_slice(&data);
        let out: Vec<IntentGpu> = vals[..n].to_vec();
        drop(data);
        self.buf_intent_readback.unmap();
        out
    }

    /// Desce o estado CfC de todos os seres para a CPU.
    ///
    /// Chamado UMA VEZ por geração, antes da seleção evolutiva. O estado CfC é
    /// runtime-only (não persiste em .dna), mas precisamos das últimas ativações
    /// para telemetria e para o pipeline de evolução de neuromodulação.
    ///
    /// Retorna Vec<Vec<f32>> com `n_seres` fatias de `CFC_DIM` f32.
    ///
    /// # Custo
    ///
    /// Para 60 seres: 60 × 96 × 4 = 23.040 bytes de tráfego GPU→CPU.
    /// Acontece 0 vezes por tick durante a geração, 1 vez ao fim.
    pub fn download_cfc_state(&self, n_beings: usize) -> Vec<Vec<f32>> {
        let n = n_beings.min(self.capacity);
        let stride = weights_stride_f32();
        let cfc_offset = (W1_TOTAL + W2_TOTAL + W3_TOTAL) as u64;

        // Staging buffer para todo o bloco de pesos (precisamos só do CfC, mas
        // wgpu exige copiar blocos alinhados; copiamos o stride inteiro e extraímos).
        let bytes_total_weights = (stride * n * std::mem::size_of::<f32>()) as u64;
        let buf_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label:               Some("core-cfc-staging"),
            size:                bytes_total_weights,
            usage:               wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation:  false,
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(
            &self.buf_weights, 0,
            &buf_staging,    0,
            bytes_total_weights,
        );
        self.queue.submit(Some(enc.finish()));

        let slice = buf_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        let vals: &[f32] = bytemuck::cast_slice(&data);
        let mut out = Vec::with_capacity(n);
        for s in 0..n {
            let base_f32 = s * stride + cfc_offset as usize;
            out.push(vals[base_f32..base_f32 + CFC_DIM].to_vec());
        }
        drop(data);
        buf_staging.unmap();
        out
    }

    /// Expõe referência ao dispositivo para os kernels externos (fisica/cerebro/percepcao)
    /// criarem seus próprios bind groups e pipelines sem abrir novo dispositivo.
    ///
    /// ASSUMIDO: FisicaGpu, CerebroGpu, PercepcaoGpu recebem `&CoreGpu` e chamam
    /// `nucleo.device` e `nucleo.fila` diretamente — nunca criam `wgpu::Instance` própria.
    pub fn device(&self) -> &wgpu::Device { &self.device }
    pub fn queue(&self) -> &wgpu::Queue   { &self.queue }

    /// Constantes de dimensão expostas para que os kernels externos não precisem
    /// importar as constantes de config diretamente (evita acoplamento de crate).
    pub const fn retina_total()       -> usize { RETINA_TOTAL }
    pub const fn sensory_wave_size()  -> usize { SENSORY_WAVE_SIZE }
    pub const fn cfc_dim()            -> usize { CFC_DIM }
    pub const fn weights_stride()       -> usize { W1_TOTAL + W2_TOTAL + W3_TOTAL + CFC_DIM }
    pub const fn w1_total()           -> usize { W1_TOTAL }
    pub const fn w2_total()           -> usize { W2_TOTAL }
    pub const fn w3_total()           -> usize { W3_TOTAL }
}

// ── Helpers de bind group layout ────────────────────────────────────────────
// Copiados de mod.rs para não criar dependência circular (mod.rs é de outro agente).

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty:                wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size:   None,
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
            min_binding_size:   None,
        },
        count: None,
    }
}

// ── Resumo de bytes por tick (para referência rápida em code review) ─────────
//
// BUFFERS RESIDENTES (não saem da GPU durante a geração):
//   buf_grade        :   409.600 bytes  (320×320×4)
//   buf_height       :   409.600 bytes  (320×320×4)
//   buf_banking      :   409.600 bytes  (320×320×4)
//   buf_flow         :   819.200 bytes  (320×320×8)
//   buf_pesos        : 6.543.360 bytes  (stride=27264 f32 × 60 seres × 4)
//   buf_retina_saida :   423.360 bytes  (60 × 1764 × 4)
//   buf_intencao     :       960 bytes  (60 × 16)
//   TOTAL RESIDENTE  : 9.015.680 bytes ≈ 8,6 MB
//
// POR TICK (tráfego real CPU↔GPU):
//   SUBIDA  (write_buffer buf_poses)       :  1.920 bytes
//   DESCIDA (copy+map buf_intencao_leit)   :    960 bytes (inclui padding)
//   TOTAL POR TICK                         :  2.880 bytes ≈ 2,8 kB
//
// AO FIM DA GERAÇÃO (baixar_estado_cfc, 1×):
//   DESCIDA buf_pesos completo (para extrair CfC)  : 6.543.360 bytes
//   (otimização futura: copiar só a fatia CfC = 60×96×4 = 23.040 bytes
//    via offsets — requer wgpu::CommandEncoder::copy_buffer_to_buffer com
//    source_offset = ser * stride_bytes + offset_cfc_bytes)
