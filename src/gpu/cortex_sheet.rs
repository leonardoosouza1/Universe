// ── FOLHA CORTICAL EQUIPOTENTE — HOST RUST ────────────────────────────────────
//
// Gerencia os 4 kernels da folha cortical emergente:
//
//   Kernel A (afference):   injeta sensory_wave com viés gaussiano + ondas espontâneas
//   Kernel B (hebb):        plasticidade hebbiana com normalização subtrativa
//   Kernel C (motor):       projeta ativações → [throttle, steer, brake]
//
// # Como plugar no gpu_loop
//
//   1. Adicionar `cortex: CortexSheet` ao struct `GpuLoop`.
//   2. Em `GpuLoop::new()`, chamar `CortexSheet::new(Arc::clone(&device), Arc::clone(&queue), cap)`.
//   3. No início de cada tick, chamar `cortex.upload_wave(&queue, &wave_data[..n])`.
//   4. Adicionar os dispatches ao encoder ANTES do readback:
//
//      cortex.dispatch_afference(&mut enc, n, tick);
//      cortex.dispatch_hebb(&mut enc, &queue, n);   // lê buf_act pós-afference
//      cortex.dispatch_motor(&mut enc, n);
//
//   5. Após submit+poll, chamar `cortex.read_motor(&device, &queue)` para obter
//      [throttle, steer, brake] por ser.
//
// # Anatomia dos buffers
//
//   buf_wave:  i32[cap × 223]          sensory_wave por ser (sobre-escrito por tick)
//   buf_tau:   f32[cap × 1024]         constantes de tempo, log-espaçadas, por unidade
//   buf_state: f32[cap × 1024]         estado EMA residente entre ticks
//   buf_act:   f32[cap × 1024]         ativação tanh(state), escrita a cada tick
//   buf_afw:   f32[cap × 1024 × 223]   pesos hebbians de aferência (lernt)
//   buf_mw:    f32[cap × 1024 × 3]     pesos motores (throttle/steer/brake)
//   buf_motor: f32[cap × 3]            saída motora [−1,1]
//
// # Tamanhos (cap=60)
//   buf_afw: 60 × 1024 × 223 × 4 = ~55 MB  (maior buffer — VRAM)
//   buf_tau: 60 × 1024 × 4 = ~246 KB
//   buf_act: 60 × 1024 × 4 = ~246 KB
//
// # Prova de atividade heterogênea
//   Ver `CortexSheet::read_act_stats`: retorna (min, max, mean, std) das 1024
//   ativações de um ser. Se min == max → folha degenerada. Heterogeneidade real
//   aparece como std > 0.01 e max-min > 0.1 após os primeiros ticks.

use std::borrow::Cow;
use std::sync::Arc;
use wgpu::util::DeviceExt;

// ── Constantes de layout ──────────────────────────────────────────────────────

/// Dimensão da folha cortical: 32 × 32 unidades.
pub const SHEET_W: usize = 32;
pub const SHEET_H: usize = 32;
pub const SHEET_N: usize = SHEET_W * SHEET_H; // 1024 unidades

/// Canais de entrada sensoriais.
const N_INPUTS: usize = 223;

/// Saídas motoras: throttle=0, steer=1, brake=2.
const N_MOTOR: usize = 3;

// τ mínimo e máximo do banco log-espaçado (em segundos).
// gamma ≈ 1.6 ms → delta ≈ 318 ms (doc §5.3)
const TAU_MIN: f32 = 0.0016;  // 1.6 ms — banda gamma
const TAU_MAX: f32 = 0.318;   // 318 ms — banda delta

// ── Uniform struct espelhando WGSL ────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsAfference {
    n_beings:  u32,
    tick:      u32,
    // fat_beta: força de inibição da fadiga adaptativa (mecanismo 1). 0.0=desligado.
    // Vem do campo CortexSheet::fat_beta, configurável via arg --fat-beta.
    fat_beta:  f32,
    // div_sigma: ganho da normalização divisiva (mecanismo 2). 0.0=desligado.
    // Vem do campo CortexSheet::div_sigma, configurável via arg --div-sigma.
    div_sigma: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsHebb {
    n_beings: u32,
    tick:     u32,
    act_sum:  f32,
    _pad0:    u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsMotor {
    n_beings: u32,
    tick:     u32,
    _pad0:    u32,
    _pad1:    u32,
}

// ── Helpers de bind group layout ─────────────────────────────────────────────

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

// ── Contexto da folha cortical ────────────────────────────────────────────────

/// Folha cortical equipotente: 1024 unidades por ser, organização emergente.
pub struct CortexSheet {
    device: Arc<wgpu::Device>,
    queue:  Arc<wgpu::Queue>,

    // Pipelines dos 3 kernels
    pipeline_aff:   wgpu::ComputePipeline,
    pipeline_hebb:  wgpu::ComputePipeline,
    pipeline_motor: wgpu::ComputePipeline,

    // Layouts de bind group
    layout_aff:   wgpu::BindGroupLayout,
    layout_hebb:  wgpu::BindGroupLayout,
    layout_motor: wgpu::BindGroupLayout,

    // Buffers persistentes
    buf_wave:  wgpu::Buffer,  // i32 × (cap × N_INPUTS) — sobre-escrito por tick
    buf_tau:   wgpu::Buffer,  // f32 × (cap × SHEET_N)  — constante após init
    buf_state: wgpu::Buffer,  // f32 × (cap × SHEET_N)  — estado EMA residente
    buf_act:   wgpu::Buffer,  // f32 × (cap × SHEET_N)  — ativação após kernel A
    buf_afw:   wgpu::Buffer,  // f32 × (cap × SHEET_N × N_INPUTS) — pesos hebbians
    buf_mw:    wgpu::Buffer,  // f32 × (cap × SHEET_N × N_MOTOR) — pesos motores
    buf_motor: wgpu::Buffer,  // f32 × (cap × N_MOTOR)  — saída motora
    // Fadiga adaptativa por unidade (spike-frequency adaptation — anti-saturação)
    // Cresce quando |act| é alto por muitos ticks; subtrai do net_input quebrando ponto fixo.
    buf_fat:   wgpu::Buffer,  // f32 × (cap × SHEET_N)  — estado lento de fadiga

    // Buffer de readback para leitura CPU do estado e saída motora
    buf_motor_rb: wgpu::Buffer,
    buf_act_rb:   wgpu::Buffer,

    /// Capacidade máxima de seres
    pub capacity: usize,

    /// fat_beta: força de inibição da fadiga adaptativa (mecanismo 1).
    /// 0.0 = desligado (folha SATURADA), 1.2 = valor original forte.
    /// Passado para o shader via uniform a cada dispatch_afference.
    pub fat_beta: f32,

    /// div_sigma: ganho da normalização divisiva por vizinhança (mecanismo 2).
    /// 0.0 = desligado, 2.0 = valor original forte.
    /// Passado para o shader via uniform a cada dispatch_afference.
    pub div_sigma: f32,

    /// Cache dos τ do ser 0 (última atualização via set_tau_cache).
    /// None = padrão homogêneo (antes do primeiro genoma carregado).
    /// Evita readback GPU→CPU a cada chamada de tau_bank().
    tau_cache: Option<Vec<f32>>,
}

impl CortexSheet {
    /// Cria nova folha cortical para `capacity` seres.
    ///
    /// Os τ são sorteados log-linearmente entre TAU_MIN e TAU_MAX, usando o
    /// índice da unidade como semente — reprodutível e sem rótulo de região.
    ///
    /// Os pesos buf_afw começam em 1.0 (prior gaussiano puro, sem aprendizado).
    /// Os pesos buf_mw começam com pequenos valores aleatórios (semente 0xBEEF).
    pub fn new(
        device: Arc<wgpu::Device>,
        queue:  Arc<wgpu::Queue>,
        capacity: usize,
    ) -> Self {
        let cap = capacity.max(1);

        // ── Compilar shaders ─────────────────────────────────────────────────
        // Cada kernel em arquivo separado — previne SIGSEGV do ACO em shaders >1050 linhas.
        let mod_aff = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("cortex-afference"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                include_str!("cortex_sheet.wgsl"),
            )),
        });
        let mod_hebb = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("cortex-hebb"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                include_str!("cortex_sheet_hebb.wgsl"),
            )),
        });
        let mod_motor = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("cortex-motor"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(
                include_str!("cortex_sheet_motor.wgsl"),
            )),
        });

        // ── Layout A (afference): uniform + wave(r) + afw(rw) + state(rw) + tau(r) + act(rw) + fat(rw) ──
        let layout_aff = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("cortex-aff-layout"),
            entries: &[
                uniform_entry(0),         // params
                storage_entry(1, true),   // buf_wave (read)
                storage_entry(2, false),  // buf_afw  (read_write — lido para prior)
                storage_entry(3, false),  // buf_state (read_write)
                storage_entry(4, true),   // buf_tau  (read)
                storage_entry(5, false),  // buf_act  (read_write — lê vizinhos t-1, escreve t)
                storage_entry(6, false),  // buf_fat  (read_write — fadiga adaptativa)
            ],
        });

        // ── Layout B (hebb): uniform + act(r) + wave(r) + afw(rw) ──────────
        let layout_hebb = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("cortex-hebb-layout"),
            entries: &[
                uniform_entry(0),         // params (inclui act_sum)
                storage_entry(1, true),   // buf_act  (read)
                storage_entry(2, true),   // buf_wave (read)
                storage_entry(3, false),  // buf_afw  (read_write)
            ],
        });

        // ── Layout C (motor): uniform + act(r) + mw(r) + motor(rw) ─────────
        let layout_motor = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("cortex-motor-layout"),
            entries: &[
                uniform_entry(0),        // params
                storage_entry(1, true),  // buf_act   (read)
                storage_entry(2, true),  // buf_mw    (read)
                storage_entry(3, false), // buf_motor (write)
            ],
        });

        // ── Pipelines ────────────────────────────────────────────────────────
        let pl_aff = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&layout_aff], push_constant_ranges: &[],
        });
        let pl_hebb = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&layout_hebb], push_constant_ranges: &[],
        });
        let pl_motor = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&layout_motor], push_constant_ranges: &[],
        });

        let pipeline_aff = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cortex-aff-pipeline"),
            layout: Some(&pl_aff),
            module: &mod_aff,
            entry_point: Some("afference"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_hebb = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cortex-hebb-pipeline"),
            layout: Some(&pl_hebb),
            module: &mod_hebb,
            entry_point: Some("hebb"),
            compilation_options: Default::default(),
            cache: None,
        });
        let pipeline_motor = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("cortex-motor-pipeline"),
            layout: Some(&pl_motor),
            module: &mod_motor,
            entry_point: Some("motor"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── Buffers ──────────────────────────────────────────────────────────

        // buf_wave: i32 por canal por ser (sobre-escrito por tick)
        let buf_wave = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cortex-wave"),
            size:  (cap * N_INPUTS * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // buf_tau: log-espaçado entre TAU_MIN e TAU_MAX, determinístico por índice.
        // MUTÁVEL (COPY_DST): o genoma pode injetar τ por parte via write_buffer,
        // permitindo divisão bipartite por cronograma neurogenético sem recompilar o shader.
        // τ_i = TAU_MIN × (TAU_MAX/TAU_MIN)^(i/(SHEET_N−1)) — valor padrão homogêneo.
        let tau_data = default_tau_homogeneous(cap);
        let buf_tau = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("cortex-tau"),
            contents: bytemuck::cast_slice(&tau_data),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // buf_state: zeros (estado EMA inicial)
        let state_zeros: Vec<f32> = vec![0.0; cap * SHEET_N];
        let buf_state = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("cortex-state"),
            contents: bytemuck::cast_slice(&state_zeros),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
        });

        // buf_act: zeros (ativação inicial)
        let buf_act = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cortex-act"),
            size:  (cap * SHEET_N * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                 | wgpu::BufferUsages::COPY_SRC
                 | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // buf_fat: fadiga adaptativa por unidade — inicializada em zeros.
        // Cresce com |act| sustentado; decai quando a unidade silencia.
        // Tamanho: cap × SHEET_N × 4 bytes (~246 KB @ cap=60)
        let fat_zeros: Vec<f32> = vec![0.0f32; cap * SHEET_N];
        let buf_fat = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("cortex-fat"),
            contents: bytemuck::cast_slice(&fat_zeros),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // buf_afw: inicializa em 1.0 (prior gaussiano puro — sem viés hebbiano inicial)
        // Tamanho: cap × SHEET_N × N_INPUTS × 4 bytes ≈ 55 MB @ cap=60
        let afw_ones: Vec<f32> = vec![1.0; cap * SHEET_N * N_INPUTS];
        let buf_afw = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("cortex-afw"),
            contents: bytemuck::cast_slice(&afw_ones),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // buf_mw: pesos motores inicializados com pseudo-ruído pequeno (hash de índice)
        // Sem zeros — zeros tornam o motor surdo ao estado inicial da folha.
        let mw_data: Vec<f32> = (0..cap)
            .flat_map(|ser| {
                (0..SHEET_N).flat_map(move |u| {
                    (0..N_MOTOR).map(move |m| {
                        // Hash de Knuth para gerar ±pequeno valor inicial
                        let h: u32 = ((ser as u32) * 1000003u32)
                            .wrapping_add((u as u32) * 2654435761u32)
                            .wrapping_add(m as u32 * 16777619u32);
                        // Mapeia [0, 2^32) → [−0.1, 0.1]
                        let norm = (h as f32 / u32::MAX as f32) - 0.5;
                        norm * 0.2  // amplitude ±0.1
                    })
                })
            })
            .collect();
        let buf_mw = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("cortex-mw"),
            contents: bytemuck::cast_slice(&mw_data),
            usage:    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // buf_motor: saída [throttle, steer, brake] por ser
        let buf_motor = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cortex-motor"),
            size:  (cap * N_MOTOR * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Readback buffers (MAP_READ separado — wgpu exige)
        let buf_motor_rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cortex-motor-rb"),
            size:  (cap * N_MOTOR * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_act_rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cortex-act-rb"),
            size:  (cap * SHEET_N * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        eprintln!(
            "[cortex_sheet] Folha 32×32={} unidades × {} seres | τ ∈ [{:.1}ms, {:.0}ms]",
            SHEET_N, cap, TAU_MIN * 1000.0, TAU_MAX * 1000.0
        );
        eprintln!(
            "[cortex_sheet] buf_afw: {} MB | buf_mw: {} KB",
            cap * SHEET_N * N_INPUTS * 4 / (1024 * 1024),
            cap * SHEET_N * N_MOTOR * 4 / 1024,
        );

        Self {
            device,
            queue,
            pipeline_aff,
            pipeline_hebb,
            pipeline_motor,
            layout_aff,
            layout_hebb,
            layout_motor,
            buf_wave,
            buf_tau,
            buf_state,
            buf_act,
            buf_afw,
            buf_mw,
            buf_motor,
            buf_motor_rb,
            buf_act_rb,
            buf_fat,
            capacity: cap,
            // Valores padrão: inibição FORTE (comportamento original).
            // Pode ser sobrescrito via CortexSheet::set_inhibition() antes do primeiro tick.
            fat_beta:  1.2,
            div_sigma: 2.0,
            tau_cache: None,  // preenchido por set_tau_cache() após primeiro genoma
        }
    }

    /// Configura os parâmetros de inibição anti-saturação em tempo de execução.
    ///
    /// Deve ser chamado ANTES do primeiro dispatch_afference de uma nova varredura.
    /// Os valores são passados para o shader via uniform a cada dispatch — sem recompilação.
    ///
    /// - fat_beta=0.0, div_sigma=0.0 → folha SATURADA (baseline da varredura).
    /// - fat_beta=1.2, div_sigma=2.0 → inibição FORTE (valor original medido).
    pub fn set_inhibition(&mut self, fat_beta: f32, div_sigma: f32) {
        self.fat_beta  = fat_beta;
        self.div_sigma = div_sigma;
    }

    // ── Upload da sensory_wave ────────────────────────────────────────────────

    /// Sobe o sensory_wave de `n` seres para a GPU.
    ///
    /// `waves`: slice de arrays i8, um por ser, N_INPUTS elementos cada.
    /// Converte i8 → i32 (sign-extended) para o shader.
    pub fn upload_wave(&self, waves: &[[i8; 223]], n: usize) {
        let n = n.min(self.capacity).min(waves.len());
        let data: Vec<i32> = waves[..n]
            .iter()
            .flat_map(|w| w.iter().map(|&b| b as i32))
            .collect();
        self.queue.write_buffer(&self.buf_wave, 0, bytemuck::cast_slice(&data));
    }

    // ── Dispatch dos kernels (adicionados ao encoder externo) ─────────────────

    /// Dispatch do Kernel A (aferência + EMA + ativação).
    /// Grava em buf_state e buf_act.
    /// dispatch: (SHEET_N/32, n_beings, 1) = (32, n, 1)
    ///
    /// fat_beta e div_sigma são lidos de self (configurados via set_inhibition).
    /// Isso permite varredura paramétrica sem recompilar o shader.
    pub fn dispatch_afference(&self, enc: &mut wgpu::CommandEncoder, n: usize, tick: u32) {
        let n = n.min(self.capacity) as u32;
        let params = ParamsAfference {
            n_beings:  n,
            tick,
            fat_beta:  self.fat_beta,
            div_sigma: self.div_sigma,
        };
        let buf_params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("cortex-aff-params"),
            contents: bytemuck::bytes_of(&params),
            usage:    wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   None,
            layout:  &self.layout_aff,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_wave.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_afw.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_state.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.buf_tau.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.buf_act.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.buf_fat.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cortex-aff"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline_aff);
        pass.set_bind_group(0, &bg, &[]);
        // (SHEET_N/32, n_beings, 1) = (32, n, 1)
        pass.dispatch_workgroups(SHEET_N as u32 / 32, n, 1);
    }

    /// Dispatch do Kernel B (plasticidade hebbiana).
    ///
    /// IMPORTANTE: deve ser chamado APÓS `dispatch_afference` ter sido adicionado
    /// ao mesmo encoder (ou ao encoder submetido anteriormente), pois lê buf_act.
    ///
    /// `act_sum`: Σ_j |act_j| do tick anterior (ou 0.0 no tick 0).
    /// Para o tick 0, act_sum=0 → only Hebb sem normalização (ok).
    pub fn dispatch_hebb(
        &self,
        enc: &mut wgpu::CommandEncoder,
        n: usize,
        tick: u32,
        act_sum: f32,
    ) {
        let n = n.min(self.capacity) as u32;
        let params = ParamsHebb { n_beings: n, tick, act_sum, _pad0: 0 };
        let buf_params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("cortex-hebb-params"),
            contents: bytemuck::bytes_of(&params),
            usage:    wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   None,
            layout:  &self.layout_hebb,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_act.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_wave.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_afw.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cortex-hebb"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline_hebb);
        pass.set_bind_group(0, &bg, &[]);
        // dispatch (SHEET_N/32, n_beings, 1) = (32, n, 1)
        pass.dispatch_workgroups(SHEET_N as u32 / 32, n, 1);
    }

    /// Dispatch do Kernel C (projeção motora).
    /// Lê buf_act, escreve buf_motor.
    pub fn dispatch_motor(&self, enc: &mut wgpu::CommandEncoder, n: usize, tick: u32) {
        let n = n.min(self.capacity) as u32;
        let params = ParamsMotor { n_beings: n, tick, _pad0: 0, _pad1: 0 };
        let buf_params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("cortex-motor-params"),
            contents: bytemuck::bytes_of(&params),
            usage:    wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   None,
            layout:  &self.layout_motor,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_act.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_mw.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_motor.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cortex-motor"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline_motor);
        pass.set_bind_group(0, &bg, &[]);
        // 1 workgroup por ser, workgroup_size(64)
        pass.dispatch_workgroups(n, 1, 1);
    }

    /// Adiciona cópias de readback ao encoder (para leitura CPU após submit).
    pub fn copy_for_readback(&self, enc: &mut wgpu::CommandEncoder, n: usize) {
        let n = n.min(self.capacity);
        // Motor output: n × 3 × 4 bytes
        enc.copy_buffer_to_buffer(
            &self.buf_motor, 0,
            &self.buf_motor_rb, 0,
            (n * N_MOTOR * 4) as u64,
        );
        // Act (para estatísticas): n × SHEET_N × 4 bytes
        enc.copy_buffer_to_buffer(
            &self.buf_act, 0,
            &self.buf_act_rb, 0,
            (n * SHEET_N * 4) as u64,
        );
    }

    // ── Leituras CPU (pós submit+poll) ───────────────────────────────────────

    /// Lê [throttle, steer, brake] de todos os seres após submit+poll.
    /// Retorna Vec de len n, cada elemento = [throttle, steer, brake].
    pub fn read_motor(&self, n: usize) -> Vec<[f32; 3]> {
        let n = n.min(self.capacity);
        let bytes = (n * N_MOTOR * 4) as u64;

        let slice = self.buf_motor_rb.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let result: Vec<[f32; 3]> = (0..n)
            .map(|i| [floats[i * 3], floats[i * 3 + 1], floats[i * 3 + 2]])
            .collect();
        drop(data);
        self.buf_motor_rb.unmap();
        result
    }

    /// Lê as ativações da folha para diagnóstico de heterogeneidade.
    ///
    /// Retorna estatísticas por ser: (min, max, mean, std) das 1024 ativações.
    /// Prova real: std > 0.01 e (max − min) > 0.1 após tick > 5.
    ///
    /// # Prova de não-fabricação
    /// Os valores são lidos diretamente do buf_act_rb (MAP_READ da GPU).
    /// Cada unidade tem τ diferente → decai em ritmo diferente → min ≠ max.
    pub fn read_act_stats(&self, n: usize) -> Vec<(f32, f32, f32, f32)> {
        let n = n.min(self.capacity);
        let bytes = (n * SHEET_N * 4) as u64;

        let slice = self.buf_act_rb.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);

        let mut stats = Vec::with_capacity(n);
        for ser in 0..n {
            let base = ser * SHEET_N;
            let slice_s = &floats[base..base + SHEET_N];

            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            let mut sum = 0.0_f32;
            for &v in slice_s {
                if v < mn { mn = v; }
                if v > mx { mx = v; }
                sum += v;
            }
            let mean = sum / SHEET_N as f32;

            let var: f32 = slice_s.iter()
                .map(|&v| (v - mean) * (v - mean))
                .sum::<f32>() / SHEET_N as f32;
            let std = var.sqrt();

            stats.push((mn, mx, mean, std));
        }

        drop(data);
        self.buf_act_rb.unmap();
        stats
    }

    /// Lê as ativações brutas (1024 valores por ser) após submit+poll.
    ///
    /// Retorna Vec<f32> com n × SHEET_N valores: [ser0_unit0, ..., ser0_unit1023, ser1_unit0, ...].
    ///
    /// Usado por cortex_map para calcular dominância de modalidade via diferença de resposta
    /// e para acumular atividade média por unidade ao longo dos ticks.
    pub fn read_act_raw(&self, n: usize) -> Vec<f32> {
        let n = n.min(self.capacity);
        let total = n * SHEET_N;
        let bytes = (total * 4) as u64;

        let slice = self.buf_act_rb.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let result = floats[..total].to_vec();
        drop(data);
        self.buf_act_rb.unmap();
        result
    }

    // ── Acesso aos buffers para carga de genoma ──────────────────────────────

    /// Referência ao buf_afw (pesos hebbians de aferência) para write_buffer externo.
    /// Layout: ser × SHEET_N × N_INPUTS f32. Usado pelo cortex_evolve para injetar
    /// pesos iniciais do genoma antes de cada geração.
    pub fn buf_afw_ref(&self) -> &wgpu::Buffer { &self.buf_afw }

    /// Referência ao buf_mw (pesos motores) para write_buffer externo.
    /// Layout: ser × SHEET_N × N_MOTOR f32.
    pub fn buf_mw_ref(&self) -> &wgpu::Buffer { &self.buf_mw }

    /// Referência ao buf_state (estado EMA) para reset entre gerações.
    /// Layout: ser × SHEET_N f32. Escrever zeros reseta a folha.
    pub fn buf_state_ref(&self) -> &wgpu::Buffer { &self.buf_state }

    /// Referência ao buf_fat (fadiga adaptativa) para reset entre gerações.
    /// Layout: ser × SHEET_N f32. Escrever zeros reseta a fadiga acumulada.
    /// DEVE ser resetado junto com buf_state para evitar que fadiga residual
    /// de uma geração penalize o início da próxima (viés de inicialização).
    pub fn buf_fat_ref(&self) -> &wgpu::Buffer { &self.buf_fat }

    /// Retorna o τ de cada unidade do ser 0 (para inspeção do banco de τ).
    ///
    /// Retorna a cópia em cache (gerada na CPU junto com os dados carregados por genoma).
    /// Se nenhum genoma foi carregado, retorna o padrão log-espaçado homogêneo.
    pub fn tau_bank(&self) -> Vec<f32> {
        if let Some(ref cache) = self.tau_cache {
            cache.clone()
        } else {
            default_tau_homogeneous(1)[..SHEET_N].to_vec()
        }
    }

    /// Referência ao buf_tau (constante de tempo) para write_buffer externo.
    /// Layout: ser × SHEET_N f32.
    /// Usado pelo cortex_evolve para injetar τ por parte conforme o cronograma do genoma.
    pub fn buf_tau_ref(&self) -> &wgpu::Buffer { &self.buf_tau }

    /// Atualiza o cache de τ do ser 0 (para que tau_bank() reflita o genoma carregado).
    ///
    /// Deve ser chamado sempre que buf_tau for atualizado via write_buffer.
    /// O cache evita readback GPU→CPU a cada inspeção.
    pub fn set_tau_cache(&mut self, taus_ser0: Vec<f32>) {
        self.tau_cache = Some(taus_ser0);
    }
}

// ── Funções auxiliares de cronograma bipartite ────────────────────────────────

/// Gera τ homogêneo log-espaçado (padrão sem genoma).
/// Retorna cap × SHEET_N valores f32.
pub fn default_tau_homogeneous(cap: usize) -> Vec<f32> {
    let log_ratio = (TAU_MAX / TAU_MIN).ln();
    (0..cap)
        .flat_map(|_ser| {
            (0..SHEET_N).map(|i| {
                let t = i as f32 / (SHEET_N as f32 - 1.0);
                TAU_MIN * (log_ratio * t).exp()
            })
        })
        .collect()
}

/// Calcula a divisão bipartite pelo cronograma neurogenético.
///
/// Modelo de Finlay & Darlington (1995): N_parte ∝ 2^((t_fim − t_início) × taxa).
/// Partes (ordem fixa de início/fim):
///   - cerebelo: início=0.35, fim=0.80  → faixa rápida (gamma/beta)
///   - córtex:   início=0.55, fim=1.00 + schedule_ext → faixa lenta (theta/delta)
///
/// `schedule_ext` é o único parâmetro evoluível (extensão da janela cortical).
/// Clampado em [0.0, 0.5] — valores maiores que 0.5 distorcem além do plausível.
///
/// Retorna: (n_cerebelo, n_cortex, taus_por_unidade[SHEET_N]).
///
/// Distribuição de τ:
///   - cerebelo (unidades 0..n_cerebelo-1): τ ∈ [TAU_MIN, TAU_MID]  (gamma+beta)
///   - córtex   (unidades n_cerebelo..SHEET_N-1): τ ∈ [TAU_MID, TAU_MAX] (theta+delta)
///
/// Teste de sanidade: razão córtex/cerebelo deve estar em [0.15, 0.40] (≈1:4 biológico).
/// A proporção EMERGE do schedule_ext, não é imposta.
pub fn compute_bipart_schedule(schedule_ext: f32) -> (usize, usize, Vec<f32>) {
    // Parâmetros do cronograma (Finlay & Darlington §2 do doc)
    // FAIXA ABERTA ATÉ 2,0. Estava travada em 0,5, o que limitava o córtex a ~30% e
    // impedia a evolução de sequer TESTAR córtex-pesado.
    //
    // Motivo para abrir: a razão 1:4 (80% cerebelo) é o fato conservado em MAMÍFEROS, e
    // mamífero tem centenas de músculos com controle motor fino. **Nosso corpo é um carro com
    // TRÊS atuadores** (acelerador, esterço, freio). O achado N1 deste ecossistema — "o corpo
    // dita a arquitetura", correlação +0,972 — prevê que a proporção ótima acompanha o corpo,
    // não a biologia de outro corpo.
    //
    // Medido: bipartite 1:4 ficou ABAIXO da folha homogênea (med10 110 m contra 164 m,
    // max10 727 m contra 886 m). Com 80% das unidades em τ curto (1,6–12 ms), sobra pouca
    // integração de trajetória longa — que é justamente o que quebrou a parede antes.
    let ext      = schedule_ext.clamp(0.0, 2.0_f32);
    let taxa: f32 = 4.0;   // expoente de proliferação (log2 da expansão por unidade de tempo)

    // Duração de cada parte no cronograma calibrado para razão 1:4 biológica:
    //   ext=0.0  →  córtex=20.0%, cerebelo=80.0%  (razão 1:4 exata)
    //   ext=0.1  →  córtex=24.8%, cerebelo=75.2%
    //   ext=0.3  →  córtex=36.5%, cerebelo=63.5%
    //   ext=0.5  →  córtex=50.0%, cerebelo=50.0%
    //
    // Fórmula: dur_cb e dur_cx satisfazem 2^(dur_cb−dur_cx)=4 quando ext=0.
    // → dur_cb = dur_cx + 2/taxa = 0.45 + 0.50 = 0.95
    let dur_cerebelo = 0.95_f32;                       // calibrado: cerebelo=80% com ext=0
    let dur_cortex   = 0.45_f32 + ext;                 // córtex=20% com ext=0

    // N relativo: 2^(dur × taxa) — "late equals large"
    let k_cerebelo = (dur_cerebelo * taxa).exp2();
    let k_cortex   = (dur_cortex   * taxa).exp2();
    let k_total    = k_cerebelo + k_cortex;

    // N absoluto: proporção × SHEET_N (mínimo 1 cada)
    let n_cerebelo = ((k_cerebelo / k_total) * SHEET_N as f32).round() as usize;
    let n_cerebelo = n_cerebelo.clamp(1, SHEET_N - 1);
    let n_cortex   = SHEET_N - n_cerebelo;

    // Distribuição de τ por parte:
    //   cerebelo → τ ∈ [TAU_MIN, TAU_MID]  (gamma/beta: 1.6 ms – 12 ms)
    //   córtex   → τ ∈ [TAU_MID, TAU_MAX]  (theta/delta: 12 ms – 318 ms)
    //
    // TAU_MID é o ponto de corte entre beta e alpha (≈12 ms = 1/(2π·13Hz))
    const TAU_MID: f32 = 0.012;   // 12 ms — fronteira beta/alpha (doc §5.3)

    let log_r_cb = (TAU_MID  / TAU_MIN).ln();
    let log_r_cx = (TAU_MAX  / TAU_MID).ln();

    let mut taus = vec![0.0_f32; SHEET_N];

    // Cerebelo: primeiras n_cerebelo unidades — τ curto (gamma/beta)
    for i in 0..n_cerebelo {
        let t = if n_cerebelo > 1 { i as f32 / (n_cerebelo as f32 - 1.0) } else { 0.0 };
        taus[i] = TAU_MIN * (log_r_cb * t).exp();
    }
    // Córtex: unidades restantes — τ longo (theta/delta)
    for i in 0..n_cortex {
        let t = if n_cortex > 1 { i as f32 / (n_cortex as f32 - 1.0) } else { 0.0 };
        taus[n_cerebelo + i] = TAU_MID * (log_r_cx * t).exp();
    }

    (n_cerebelo, n_cortex, taus)
}
