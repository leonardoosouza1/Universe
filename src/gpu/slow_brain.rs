// ── CÉREBRO LENTO NA GPU — host Rust ──────────────────────────────────────────
//
// Hospeda SlowBrain (CfC-64 → Dense1(64→16) → Dense2(16→3)) na GPU para que o
// caminho GPU tenha ambos os cérebros (rápido: brain.rs + lento: este arquivo).
//
// # Por que dois kernels em vez de um
//
// RADV/ACO trava em shaders grandes neste projeto (histórico de crashes silenciosos
// com kernels > ~200 linhas de corpo). O SlowBrain tem três etapas com barreiras
// entre elas (CfC → Dense1 → Dense2). Com 64 threads e workgroup único isso
// funcionaria em um kernel, mas ficaria > 150 linhas de corpo. Dividimos em
// dois kernels encadeados no mesmo CommandEncoder:
//   Kernel A (run_cfc):   atualiza estado CfC → salva s em buf_state
//   Kernel B (run_dense): lê s, recalcula tanh, Dense1 + Dense2 → output
// Um único submit para os dois.
//
// # Buffers residentes
//
//   buf_state:   [STATE_PER_SER × cap f32] — cfc_state, fatigue, counter,
//                tick_counter, output[3]. Inicializado em zeros.
//   buf_weights: [WEIGHTS_PER_SER × cap f32] — Dense1+Dense2, upload por spawn.
//   buf_input:   [INPUT_SIZE × cap i32] — input por tick (48 canais i8 sign-extended).
//   buf_output:  [3 × cap f32] — saída por tick, lida pelo host.
//   buf_readback: [3 × cap f32] — MAP_READ separado (wgpu exige).
//
// # Contagem de slots em STATE_PER_SER = 133 f32
//   [0..64)    cfc_state
//   [64..128)  cfc_fatigue
//   [128]      counter (u32 bitcast)
//   [129]      tick_counter (u32 bitcast)
//   [130..133) output[3]
//
// # Pesos em WEIGHTS_PER_SER = 1091 f32
//   [0..1024)    dense1.weights (64×16)
//   [1024..1040) dense1.bias   (16)
//   [1040..1088) dense2.weights (16×3)
//   [1088..1091) dense2.bias   (3)
//   Ordem idêntica a SlowBrain::all_weights() (engine.rs) — DNA round-trip correto.
//
// # map_async → poll → recv (ordem obrigatória)
//
// Deadlock documentado neste projeto: poll(Wait) ANTES de map_async não serve o
// callback. O poll tem de vir DEPOIS do pedido de mapeamento. Veja brain.rs:710
// para o histórico. Todo readback aqui segue a ordem: map_async → poll → recv.

use std::borrow::Cow;
use std::sync::Arc;
use wgpu::util::DeviceExt;

// ── DIMENSÕES ────────────────────────────────────────────────────────────────

const INPUT_SIZE:      usize = 48;
const CFC_SIZE:        usize = 64;
const D2_OUT:          usize = 3;
const OUTPUT_DIM:      usize = D2_OUT;

/// Slots de estado por ser (f32):
///   cfc_state[64] + cfc_fatigue[64] + counter(1) + tick_counter(1) + output[3] = 133
const STATE_PER_SER:   usize = CFC_SIZE + CFC_SIZE + 1 + 1 + D2_OUT;  // 133

/// Pesos por ser (f32). Ordem espelha SlowBrain::all_weights():
///   dense1.weights(1024) + dense1.bias(16) + dense2.weights(48) + dense2.bias(3) = 1091
const WEIGHTS_PER_SER: usize = 1024 + 16 + 48 + 3;  // 1091

// ── UNIFORM ───────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SlowBrainParams {
    n_beings: u32,
    dt:       f32,
    _pad0:    u32,
    _pad1:    u32,
}

// ── CONTEXTO PERSISTENTE ──────────────────────────────────────────────────────

/// Contexto GPU do SlowBrain. Vive durante toda a geração.
///
/// Invariante: todos os buffers suportam até `capacity` seres.
pub struct SlowBrainGpu {
    device:       Arc<wgpu::Device>,
    queue:        Arc<wgpu::Queue>,

    // Dois pipelines — dois kernels encadeados no mesmo CommandEncoder.
    pipeline_cfc:   wgpu::ComputePipeline,
    pipeline_dense: wgpu::ComputePipeline,
    layout:         wgpu::BindGroupLayout,

    // Buffers residentes
    buf_state:    wgpu::Buffer,   // f32 × (cap × STATE_PER_SER) — residente, zeros init
    buf_weights:  wgpu::Buffer,   // f32 × (cap × WEIGHTS_PER_SER) — upload por spawn
    buf_input:    wgpu::Buffer,   // i32 × (cap × INPUT_SIZE) — escrito por tick ou pela colheita
    // Pipeline de COLHEITA: preenche buf_input a partir da onda sensorial que já
    // está na GPU, evitando um readback por tick só para escolher 48 de 223 canais.
    pipeline_gather: wgpu::ComputePipeline,
    layout_gather:   wgpu::BindGroupLayout,
    buf_output:   wgpu::Buffer,   // f32 × (cap × OUTPUT_DIM) — lido por tick (COPY_SRC)
    buf_readback: wgpu::Buffer,   // f32 × (cap × OUTPUT_DIM) — MAP_READ

    /// Capacidade máxima (número de seres suportado com estes buffers).
    pub capacity: usize,
}

impl SlowBrainGpu {
    /// Cria o contexto compartilhando device/queue com o BrainGpu (retina/cérebro).
    ///
    /// Receber Arc<Device/Queue> evita abrir um segundo dispositivo Vulkan para o
    /// mesmo hardware — o mesmo padrão de brain.rs:128.
    pub fn new_shared(
        device:   Arc<wgpu::Device>,
        queue:    Arc<wgpu::Queue>,
        capacity: usize,
    ) -> Self {
        // ── Shader ─────────────────────────────────────────────────────────
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("slow-brain"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                include_str!("slow_brain.wgsl"),
            )),
        });

        // ── Bind group layout (um único layout para os dois kernels) ────────
        // Ambos os kernels usam o mesmo set de bindings — separar layouts seria
        // desnecessário e criaria dois bind groups por dispatch.
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("slow-brain-layout"),
            entries: &[
                uniform_entry(0),       // params
                storage_entry(1, true), // buf_input   (read)
                storage_entry(2, true), // buf_weights (read)
                storage_entry(3, false),// buf_state   (read_write, residente)
                storage_entry(4, false),// buf_output  (read_write)
            ],
        });

        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                None,
            bind_group_layouts:   &[&layout],
            push_constant_ranges: &[],
        });

        // Pipeline A: CfC
        let pipeline_cfc = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:               Some("slow-cfc-pipeline"),
            layout:              Some(&pipe_layout),
            module:              &module,
            entry_point:         Some("run_cfc"),
            compilation_options: Default::default(),
            cache:               None,
        });

        // Pipeline B: Dense1 + Dense2
        let pipeline_dense = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:               Some("slow-dense-pipeline"),
            layout:              Some(&pipe_layout),
            module:              &module,
            entry_point:         Some("run_dense"),
            compilation_options: Default::default(),
            cache:               None,
        });

        // ── Buffers ─────────────────────────────────────────────────────────

        // Estado residente: zeros = estado inicial correto (CfCCell::new zerados).
        let bytes_state = (capacity * STATE_PER_SER * 4) as u64;
        let state_zeros: Vec<u8> = vec![0u8; bytes_state as usize];
        let buf_state = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("slow-state"),
            contents: &state_zeros,
            usage:    wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST  // para reset em reset_being()
                    | wgpu::BufferUsages::COPY_SRC, // para diagnóstico (se necessário)
        });

        // Pesos: alocados mas não inicializados até upload_weights().
        let bytes_weights = (capacity * WEIGHTS_PER_SER * 4) as u64;
        let buf_weights = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("slow-weights"),
            size:               bytes_weights,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Input: i32 × INPUT_SIZE × cap — escrito a cada tick.
        let bytes_input = (capacity * INPUT_SIZE * 4) as u64;
        // Módulo e pipeline da colheita — shader separado para manter cada kernel
        // pequeno (RADV/ACO trava em shaders grandes neste projeto).
        let module_gather = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("slow-gather"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("slow_gather.wgsl"))),
        });
        let layout_gather = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("slow-gather-layout"),
            entries: &[uniform_entry(0), storage_entry(1, true), storage_entry(2, false)],
        });
        let pl_gather = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("slow-gather-pl"), bind_group_layouts: &[&layout_gather], push_constant_ranges: &[],
        });
        let pipeline_gather = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("slow-gather-pipe"), layout: Some(&pl_gather), module: &module_gather,
            entry_point: Some("gather"), compilation_options: Default::default(), cache: None,
        });

        let buf_input = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("slow-input"),
            size:               bytes_input,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Output: f32 × OUTPUT_DIM × cap — escrito pelo kernel B.
        // COPY_SRC para copiar para buf_readback no mesmo encoder.
        let bytes_output = (capacity * OUTPUT_DIM * 4) as u64;
        let buf_output = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("slow-output"),
            size:               bytes_output,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Readback: MAP_READ não pode combinar com STORAGE em wgpu.
        let buf_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("slow-readback"),
            size:               bytes_output,
            usage:              wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            pipeline_cfc,
            pipeline_dense,
            layout,
            buf_state,
            buf_weights,
            buf_input,
            pipeline_gather,
            layout_gather,
            buf_output,
            buf_readback,
            capacity,
        }
    }

    // ── UPLOAD DE PESOS ────────────────────────────────────────────────────────

    /// Sobe pesos de `n_beings` SlowBrains para a GPU.
    ///
    /// Chamar em spawn e após evolução. Ordem dos pesos: all_weights() de cada ser.
    /// Layout: ser0[1091 f32] ser1[1091 f32] ...
    ///
    /// `weights`: saída concatenada de SlowBrain::all_weights() para cada ser,
    /// total = n_beings × 1091 f32.
    pub fn upload_weights(&self, weights: &[f32], n_beings: usize) {
        let n = n_beings.min(self.capacity);
        let expected = n * WEIGHTS_PER_SER;
        let actual   = weights.len().min(expected);
        // Escrita direta: SoA já está na ordem correta se o caller concatenou
        // all_weights() de cada ser na ordem certa.
        self.queue.write_buffer(&self.buf_weights, 0,
            bytemuck::cast_slice(&weights[..actual]));
    }

    // ── DISPATCH (por tick) ────────────────────────────────────────────────────

    /// Grava os dois kernels num CommandEncoder externo (sem submit).
    ///
    /// # Contrato
    ///
    /// 1. Antes: subir input via `queue.write_buffer(&buf_input, ...)` ou usar
    ///    `upload_input()` (método conveniente abaixo).
    /// 2. Chamar este método para gravar os dois dispatches.
    /// 3. No mesmo encoder: `enc.copy_buffer_to_buffer(buf_output→buf_readback)`.
    /// 4. `queue.submit(enc.finish())`.
    /// 5. `device.poll(Wait)`.
    /// 6. `read_outputs(n_beings)` para ler o resultado.
    ///
    /// Retorna bytes_output para o copy subsequente.
    /// Preenche `buf_input` colhendo 48 canais da onda sensorial que já está na GPU.
    ///
    /// Alternativa a `upload_input`, para o caminho `gpu_evolve`, onde a onda nunca
    /// desce para a CPU. Deve ser encadeado ANTES de `dispatch` no mesmo encoder.
    pub fn dispatch_gather(
        &self,
        enc:      &mut wgpu::CommandEncoder,
        wave_buf: &wgpu::Buffer,
        n_beings: usize,
    ) {
        let n = n_beings.min(self.capacity);
        if n == 0 { return; }

        let params = SlowBrainParams { n_beings: n as u32, dt: 0.0, _pad0: 0, _pad1: 0 };
        let buf_params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("slow-gather-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &self.layout_gather,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wave_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_input.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("slow-gather-pass"), timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline_gather);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(n as u32, 1, 1);
    }

    pub fn dispatch(
        &self,
        enc:      &mut wgpu::CommandEncoder,
        n_beings: usize,
        dt:       f32,
    ) -> u64 {
        let n = n_beings.min(self.capacity);
        if n == 0 { return 0; }

        let params = SlowBrainParams { n_beings: n as u32, dt, _pad0: 0, _pad1: 0 };
        let buf_params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("slow-params"),
            contents: bytemuck::bytes_of(&params),
            usage:    wgpu::BufferUsages::UNIFORM,
        });

        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   None,
            layout:  &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_weights.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_state.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.buf_output.as_entire_binding() },
            ],
        });

        // Kernel A: CfC — 1 workgroup por ser, 64 threads por workgroup
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:            Some("pass-slow-cfc"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_cfc);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(n as u32, 1, 1);
        }

        // Kernel B: Dense — mesmo bind group, encadeado no mesmo encoder.
        // Não há barreira explícita necessária aqui porque os dois dispatches no
        // mesmo encoder já garantem a ordem de execução em wgpu/Vulkan.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:            Some("pass-slow-dense"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_dense);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(n as u32, 1, 1);
        }

        let bytes_output = (n * OUTPUT_DIM * 4) as u64;

        // Copia output para readback no mesmo encoder (DMA sem round-trip).
        enc.copy_buffer_to_buffer(
            &self.buf_output,   0,
            &self.buf_readback, 0,
            bytes_output,
        );

        bytes_output
    }

    // ── UPLOAD DE INPUT (conveniente) ─────────────────────────────────────────

    /// Converte input i8 de cada ser para i32 (sign-extended) e sobe para a GPU.
    ///
    /// `inputs`: slice de n_beings × 48 bytes (i8 como bytes, cast para i32 aqui).
    /// Chamar ANTES de `dispatch()` ou antes do submit do encoder externo.
    pub fn upload_input(&self, inputs: &[[i8; INPUT_SIZE]], n_beings: usize) {
        let n = n_beings.min(self.capacity).min(inputs.len());
        let mut data: Vec<i32> = Vec::with_capacity(n * INPUT_SIZE);
        for inp in inputs[..n].iter() {
            for &b in inp.iter() {
                data.push(b as i32);
            }
        }
        self.queue.write_buffer(&self.buf_input, 0, bytemuck::cast_slice(&data));
    }

    // ── LEITURA DOS OUTPUTS ────────────────────────────────────────────────────

    /// Lê os outputs do buf_readback após device.poll(Wait).
    ///
    /// DEVE ser chamado depois que o encoder que continha `dispatch()` foi submetido
    /// e `device.poll(Wait)` retornou. Segue a ordem obrigatória:
    ///   map_async → poll(Wait) → recv → get_mapped_range
    ///
    /// (poll ANTES de map_async é o deadlock silencioso documentado em brain.rs:710)
    pub fn read_outputs(&self, n_beings: usize) -> Vec<[f32; OUTPUT_DIM]> {
        let n = n_beings.min(self.capacity);
        if n == 0 { return Vec::new(); }

        let bytes = (n * OUTPUT_DIM * 4) as u64;
        let slice = self.buf_readback.slice(0..bytes);

        let (tx, rx) = std::sync::mpsc::channel();
        // map_async registra o callback — poll(Wait) abaixo vai servi-lo.
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });

        // poll DEPOIS do map_async: sem isso o callback nunca é servido.
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        let vals: &[f32] = bytemuck::cast_slice(&data);

        let mut result: Vec<[f32; OUTPUT_DIM]> = Vec::with_capacity(n);
        for s in 0..n {
            let base = s * OUTPUT_DIM;
            let mut out = [0.0f32; OUTPUT_DIM];
            out.copy_from_slice(&vals[base..base + OUTPUT_DIM]);
            result.push(out);
        }

        drop(data);
        self.buf_readback.unmap();
        result
    }

    // ── CONVENIÊNCIA: dispatch + submit + poll + read (tudo-em-um) ────────────

    /// Executa o ciclo completo: upload input → dispatch → submit → poll → read.
    ///
    /// Equivalente ao `BrainGpu::forward()` — submete seu próprio encoder.
    /// Use quando o SlowBrain é o único kernel a dispatchar neste tick.
    /// Para encadeamento com outros kernels, prefira `dispatch()` + submit externo.
    pub fn forward(
        &self,
        inputs:   &[[i8; INPUT_SIZE]],
        n_beings: usize,
        dt:       f32,
    ) -> Vec<[f32; OUTPUT_DIM]> {
        let n = n_beings.min(self.capacity).min(inputs.len());
        if n == 0 { return Vec::new(); }

        self.upload_input(inputs, n);

        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("slow-brain-forward"),
        });

        let _bytes = self.dispatch(&mut enc, n, dt);
        self.queue.submit(Some(enc.finish()));

        self.read_outputs(n)
    }

    // ── RESET DE ESTADO PARA UM SER ────────────────────────────────────────────

    /// Zera o estado CfC, a fadiga e os contadores do ser `idx`.
    ///
    /// Equivale a `CfCCell::reset()` + reinício do tick_counter + limpeza do output.
    /// Chamado em spawn/death — ser recém-nascido começa sem estado acumulado.
    pub fn reset_being(&self, idx: usize) {
        if idx >= self.capacity { return; }
        let offset = (idx * STATE_PER_SER * 4) as u64;
        let zeros: Vec<u8> = vec![0u8; STATE_PER_SER * 4];
        self.queue.write_buffer(&self.buf_state, offset, &zeros);
    }
}

// ── RECURSO BEVY ──────────────────────────────────────────────────────────────

/// Recurso Bevy que mantém o contexto do SlowBrain GPU entre ticks.
///
/// Padrão idêntico ao BrainGpuRes (brain.rs): `ctx: Option<SlowBrainGpu>`,
/// criado/recriado se a capacidade for insuficiente.
#[derive(bevy::prelude::Resource, Default)]
pub struct SlowBrainGpuRes {
    pub ctx: Option<SlowBrainGpu>,
}

impl SlowBrainGpuRes {
    /// Devolve contexto válido, criando se necessário. None = sem GPU utilizável.
    pub fn get_or_create(
        &mut self,
        device:   Arc<wgpu::Device>,
        queue:    Arc<wgpu::Queue>,
        capacity: usize,
    ) -> Option<&SlowBrainGpu> {
        let needs = self.ctx.is_none()
            || self.ctx.as_ref().map(|c| c.capacity < capacity).unwrap_or(true);
        if needs {
            self.ctx = Some(SlowBrainGpu::new_shared(device, queue, capacity.max(64)));
        }
        self.ctx.as_ref()
    }
}

// ── HELPERS ───────────────────────────────────────────────────────────────────

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty:                 wgpu::BufferBindingType::Uniform,
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

// ── COMO EXTRAIR PESOS DE SlowBrain (referência para o caller) ────────────────
//
// Para fazer upload_weights(weights, n):
//
//   let mut all_weights: Vec<f32> = Vec::new();
//   for engine in engines[..n].iter() {
//       all_weights.extend(engine.slow_brain.all_weights());
//   }
//   slow_brain_gpu.upload_weights(&all_weights, n);
//
// all_weights() retorna (engine.rs:125-132):
//   dense1.weights[1024] ++ dense1.bias[16] ++ dense2.weights[48] ++ dense2.bias[3]
// = 1091 f32, mesma ordem do buf_weights aqui.
//
// ── DIVERGÊNCIA GPU×CPU ESPERADA ─────────────────────────────────────────────
//
// - CfC (64 ops de exp/tanh/hash): ≤ 1 ULP por canal. Drift entre ticks acumulável
//   mas determinístico (hash idêntico ao cfc.rs).
// - Dense1 matmul (64 FMA): < 1e-5 por neurônio. Mesmo padrão que brain.wgsl L1.
// - Dense2 matmul (16 FMA): < 1e-6 por neurônio (16× menos termos que L1).
// - tanh saídas: ≤ 1 ULP.
//
// A verificação do tick_counter é determinística: o ser N roda no mesmo tick
// na GPU e na CPU se tick_interval=8 e os contadores começaram em 0.
//
// ── O QUE NÃO RODEI ─────────────────────────────────────────────────────────
//
// A GPU está ocupada (constraint da tarefa). Não executei nada na GPU.
// Compilação foi verificada pelo cargo build (ver saída no relatório final).
