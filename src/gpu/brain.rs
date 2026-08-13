// ── CÉREBRO NA GPU — host Rust ────────────────────────────────────────────────
//
// Empacota os pesos de todos os seres em buffers SoA, despacha o kernel
// `cerebro.wgsl` e devolve `final_intent[96]` por ser.
//
// Contexto wgpu: reutilizamos o dispositivo que já foi aberto pelo `RetinaGpu`.
// O `CerebroGpu` aceebe um `Arc<wgpu::Device>` + `Arc<wgpu::Queue>` para não
// abrir um segundo dispositivo (GPU tem um único VkDevice lógico; abrir dois é
// ineficiente e pode falhar em Vulkan com certas versões de driver AMD).
//
// Buffers residentes entre ticks:
//   - buf_wave:   sobre-escrito por tick (sensory input muda a cada tick)
//   - buf_kan:    sobre-escrito em spawn/evolução (pesos evoluem só entre gens)
//   - buf_l1w:    idem
//   - buf_l2w:    idem
//   - buf_intent: resultado, lido pela CPU a cada tick
//   - buf_cfc:    estado recorrente, NUNCA baixa a CPU por tick
//                 (só em spawn: host inicializa zeros; em death: host lê pra log)
//
// # Por que Arc e não referências?
//
// `RetinaGpu` e `CerebroGpu` vivem como campos de recursos Bevy distintos.
// Bevy não permite dois recursos mutáveis do mesmo tipo, então os contextos GPU
// ficam em recursos separados. Arc permite compartilhar device/queue sem lifetime
// fight. Custo: 1 atomic increment por tick (negligível).

use std::borrow::Cow;
use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::engine::SovereignEngine;
use crate::config::iara::brain::{SENSORY_WAVE_SIZE, L1_NEURONS, L2_NEURONS};

// ── CONSTANTES DERIVADAS DAS DIMENSÕES DO CÉREBRO ────────────────────────────

/// Pesos L1 por ser: SENSORY_WAVE_SIZE × L1_NEURONS floats
const L1_WEIGHTS_PER_SER: usize = SENSORY_WAVE_SIZE * L1_NEURONS;  // 223 × 96 = 21.408

/// Pesos L2 por ser: L1_NEURONS × L2_NEURONS floats
const L2_WEIGHTS_PER_SER: usize = L1_NEURONS * L2_NEURONS;          // 96 × 48  = 4.608

/// Knots KAN por ser: SENSORY_WAVE_SIZE canais × 16 knots
const KAN_KNOTS: usize = 16;
const KAN_WEIGHTS_PER_SER: usize = SENSORY_WAVE_SIZE * KAN_KNOTS;   // 223 × 16 = 3.568

/// Fadiga de saída herdada: 0,85. Documentada em `BrainParams::out_beta`.
pub const OUT_BETA_PADRAO: f32 = 0.85;

/// Estado CfC por ser: 96 floats (state) + 1 u32 (counter, bitcast como f32) = 97 f32
const CFC_SLOTS_PER_SER: usize = L1_NEURONS + 1;  // 97

/// Fadiga adaptativa por ser: 192 floats.
/// Slots [0..96]  = fadiga de input do CfC (estado recorrente, já existia).
/// Slots [96..192] = fadiga de saída (output fatigue, quebra ponto fixo de final_intent).
/// Expandido de 96 → 192 para suportar o OUT_ALPHA/OUT_BETA em brain.wgsl.
const FATIGUE_PER_SER: usize = L1_NEURONS * 2;  // 192

/// Dimensão do final_intent (96 slots, engine.rs:472)
const INTENT_DIM: usize = L1_NEURONS;  // 96

// ── STRUCT REPR(C) PARA UNIFORM ───────────────────────────────────────────────

/// Parâmetros globais passados como uniform buffer.
///
/// `#[repr(C)]` garante layout previsível. `bytemuck::Pod` exige que todos os
/// campos sejam tipos primitivos sem padding implícito — aqui estamos no limite:
/// 4 campos de 4 bytes = 16 bytes, alinhado a 16 bytes (requisito WGSL uniform).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BrainParams {
    pub n_beings: u32,
    pub dt: f32,
    /// Quando 1: lê o sensory_wave do binding 8 (buf_percept, saída de percepcao.wgsl,
    /// 56 u32 por ser) em vez do binding 1 (buf_wave, i32 sign-extended da CPU).
    /// Elimina o round-trip percepção→CPU→GPU por tick. O gpu_validate passa 0.
    pub use_percept_direct: u32,
    /// Força da fadiga de SAÍDA (`OUT_BETA`), antes uma constante de shader.
    ///
    /// Em regime permanente `out_fat → |raw|`, então a saída vira `(1 - out_beta)·|raw|`:
    /// com 0,85 um comando SUSTENTADO sai a 15% da intenção, em ~1 s. Isso é adaptação a
    /// estímulo constante — ótimo num sensor (detecta novidade), veneno num comando motor,
    /// onde segurar o acelerador na reta é justamente o que se quer. E como o operador é
    /// aplicado DEPOIS do L2, nenhum peso do genoma escapa dele.
    ///
    /// Vira parâmetro para poder ser MEDIDO em teste pareado em vez de discutido.
    /// O padrão 0,85 é o histórico: sem passar nada, o comportamento é o de antes.
    pub out_beta: f32,
}

// ── INPUT POR SER ─────────────────────────────────────────────────────────────

/// O que o chamador entrega por tick, por ser.
pub struct BrainInput<'a> {
    /// Referência ao engine do ser (para ler pesos e estado CfC).
    pub engine: &'a SovereignEngine,
    /// sensory_wave[SENSORY_WAVE_SIZE] já calculado pelo motor.rs.
    pub wave: &'a [i8; SENSORY_WAVE_SIZE],
}

// ── CONTEXTO PERSISTENTE ──────────────────────────────────────────────────────

/// Contexto GPU do forward pass cerebral. Vive durante toda a geração.
///
/// Invariante: todos os buffers têm capacidade >= `capacidade` seres.
/// Se capacidade mudar (spawn de mais seres do que o previsto), recriar com
/// `BrainGpu::new`.
pub struct BrainGpu {
    device:   Arc<wgpu::Device>,
    queue:     Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    layout:   wgpu::BindGroupLayout,

    // Buffers persistentes (tamanho fixo durante a geração)
    buf_wave:   wgpu::Buffer,  // i32 × (cap × WAVE) — sobre-escrito por tick
    buf_kan:    wgpu::Buffer,  // f32 × (cap × KAN_WEIGHTS_PER_SER) — por geração
    buf_l1w:    wgpu::Buffer,  // f32 × (cap × L1_WEIGHTS_PER_SER)  — por geração
    buf_l2w:    wgpu::Buffer,  // f32 × (cap × L2_WEIGHTS_PER_SER)  — por geração
    buf_intent: wgpu::Buffer,  // f32 × (cap × INTENT_DIM)          — lido por tick
    buf_cfc:     wgpu::Buffer,  // f32 × (cap × CFC_SLOTS_PER_SER)   — residente
    /// Fadiga adaptativa (spike-frequency adaptation): estado lento por neurônio.
    /// Cresce quando |state[i]| é alto, satura quando o neurônio dispara forte por muito tempo.
    /// Inicializado em zeros; atualizado a cada tick pelo kernel brain.wgsl.
    buf_fatigue: wgpu::Buffer,  // f32 × (cap × FATIGUE_PER_SER)     — residente

    /// Banco de τ: taxa de decaimento do CfC por canal por ser, em s⁻¹ (d = 1/τ).
    ///
    /// Constante durante a geração, como os pesos: τ é herdado do genoma e não muda na
    /// vida do ser. Enviado por `upload_tau`. Nasce em `DECAY_DEFAULT` (τ = 2,0 s em todos
    /// os canais) para que quem não chamar `upload_tau` obtenha o comportamento antigo.
    buf_tau:    wgpu::Buffer,   // f32 × (cap × L1_NEURONS)          — por geração

    /// Buffer de debug temporário: exporta l1_raw + inputs CfC/KAN para ser 0.
    /// Layout: [l1_raw[0..95], cfc_in[0..95], kan_in[96..222]] = 319 f32
    buf_debug:  wgpu::Buffer,

    // Buffer de readback CPU (MAP_READ, tamanho = buf_intent)
    buf_readback: wgpu::Buffer,

    /// Buffer placeholder para binding 8 quando use_percept_direct == 0 (cpu path).
    /// O gpu_validate e forward() usam buf_wave (binding 1) e precisam de algo
    /// válido no binding 8 para que create_bind_group não falhe. Tamanho = 4 bytes.
    buf_percept_dummy: wgpu::Buffer,

    /// Força da fadiga de saída usada nos dispatches. Ver `BrainParams::out_beta`.
    pub out_beta: f32,

    /// Número máximo de seres suportado com estes buffers.
    pub capacity: usize,

    pub adapter: String,
}

impl BrainGpu {
    /// Cria novo contexto GPU. Recebe device/queue compartilhados com a retina.
    ///
    /// `capacidade`: número máximo de seres por dispatch. Dimensiona todos os
    /// buffers. Se o número de seres vivos superar isso, o caller deve recriar.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue:   Arc<wgpu::Queue>,
        adapter_name: String,
        capacity: usize,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brain"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("brain.wgsl"))),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brain-layout"),
            entries: &[
                uniform_entry(0),              // params (inclui use_percept_direct)
                storage_entry(1, true),         // buf_wave: i32 sign-extended (CPU path)
                storage_entry(2, true),         // buf_kan  (read)
                storage_entry(3, true),         // buf_l1w  (read)
                storage_entry(4, true),         // buf_l2w  (read)
                storage_entry(5, false),        // buf_intent (read_write)
                storage_entry(6, false),        // buf_cfc    (read_write, estado recorrente)
                storage_entry(7, false),        // buf_debug  (diagnóstico temporário)
                storage_entry(8, true),         // buf_percept: u32 de percepcao.wgsl (GPU path)
                storage_entry(9, false),        // buf_fatigue (read_write, fadiga adaptativa)
                storage_entry(10, true),        // buf_tau     (read, banco de τ por canal)
            ],
        });

        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("brain-pipeline"),
            layout: Some(&pipe_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── Buffers ─────────────────────────────────────────────────────────

        // Sensory wave: i32 por canal (sign-extended de i8, para evitar bit
        // manipulation no shader). 4 bytes × WAVE × cap.
        let bytes_wave = (capacity * SENSORY_WAVE_SIZE * 4) as u64;
        let buf_wave = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-wave"),
            size: bytes_wave,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // KAN: f32 por knot. Sobre-escrito em spawn/evolução, não por tick.
        let bytes_kan = (capacity * KAN_WEIGHTS_PER_SER * 4) as u64;
        let buf_kan = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-kan"),
            size: bytes_kan,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // L1 weights: f32. Maior buffer: 60 × 21.408 × 4 = ~5.1 MB.
        let bytes_l1w = (capacity * L1_WEIGHTS_PER_SER * 4) as u64;
        let buf_l1w = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-l1w"),
            size: bytes_l1w,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // L2 weights: f32. 60 × 4.608 × 4 = ~1.1 MB.
        let bytes_l2w = (capacity * L2_WEIGHTS_PER_SER * 4) as u64;
        let buf_l2w = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-l2w"),
            size: bytes_l2w,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Intent: f32 × 96 × cap. Saída do kernel.
        let bytes_intent = (capacity * INTENT_DIM * 4) as u64;
        let buf_intent = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-intent"),
            size: bytes_intent,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // CfC state: f32 × 97 × cap. Estado recorrente — inicializado em zeros.
        // Zeros são o estado inicial correto (CfCCell::new zeros state, counter=0).
        let bytes_cfc = (capacity * CFC_SLOTS_PER_SER * 4) as u64;
        let cfc_zeros: Vec<u8> = vec![0u8; bytes_cfc as usize];
        let buf_cfc = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brain-cfc"),
            contents: &cfc_zeros,
            usage: wgpu::BufferUsages::STORAGE
                 | wgpu::BufferUsages::COPY_SRC  // para leitura em spawn/death
                 | wgpu::BufferUsages::COPY_DST, // para reset em spawn
        });

        // Fadiga adaptativa: f32 × 96 × cap. Estado lento por neurônio CfC.
        // Inicializado em zeros: ser recém-nascido não tem fadiga acumulada.
        // Resetado junto com o CfC em spawn/death via reset_cfc.
        let bytes_fatigue = (capacity * FATIGUE_PER_SER * 4) as u64;
        let fatigue_zeros: Vec<u8> = vec![0u8; bytes_fatigue as usize];
        let buf_fatigue = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brain-fatigue"),
            contents: &fatigue_zeros,
            usage: wgpu::BufferUsages::STORAGE
                 | wgpu::BufferUsages::COPY_DST  // para reset em spawn
                 | wgpu::BufferUsages::COPY_SRC, // para diagnóstico de fadiga
        });

        // Banco de τ: f32 × 96 × cap. Nasce em DECAY_DEFAULT (τ = 2,0 s), que é o valor
        // único que o cérebro usava — assim o caminho que não chama `upload_tau` continua
        // idêntico ao de antes em vez de rodar com τ = 0 (decay = 1, memória infinita).
        let taus_init: Vec<f32> = vec![crate::cfc::DECAY_DEFAULT; capacity * L1_NEURONS];
        let buf_tau = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brain-tau"),
            contents: bytemuck::cast_slice(&taus_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Readback: MAP_READ separado (não pode combinar com STORAGE em wgpu).
        let buf_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-readback"),
            size: bytes_intent,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Debug: exporta l1_raw[96] + cfc_in[96] + kan_in[127] = 319 f32 para ser 0.
        // Tamanho fixo (não escalado com cap): só ser 0.
        let buf_debug = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-debug"),
            size: (319 * 4) as u64,  // 96 + 96 + 127 = 319 f32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Placeholder para binding 8 (buf_percept) quando use_percept_direct == 0.
        // O wgpu exige que todos os slots declarados no layout tenham um buffer no
        // bind group — mesmo que o shader não os leia. 4 bytes é o mínimo permitido.
        let buf_percept_dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-percept-dummy"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            pipeline,
            layout,
            buf_wave,
            buf_kan,
            buf_l1w,
            buf_l2w,
            buf_intent,
            buf_cfc,
            buf_fatigue,
            buf_tau,
            out_beta: OUT_BETA_PADRAO,
            buf_debug,
            buf_readback,
            buf_percept_dummy,
            capacity,
            adapter: adapter_name,
        }
    }

    // ── UPLOAD DE PESOS (por geração, não por tick) ────────────────────────

    /// Sobe pesos KAN, L1, L2 de todos os seres para a GPU.
    ///
    /// Chamar em spawn e depois de evolução (mutação de pesos). NÃO chamar por
    /// tick — pesos não mudam durante uma geração.
    ///
    /// Layout SoA: ser 0 inteiro, depois ser 1 inteiro, etc.
    pub fn upload_weights(&self, engines: &[&SovereignEngine]) {
        let n = engines.len().min(self.capacity);

        // ── KAN ─────────────────────────────────────────────────────────────
        // SensoryPreFilter: Vec<KanAxiomFilter>, cada um com spline_coeffs [f32; 16].
        // Layout: ser × WAVE × KAN_KNOTS (SoA por ser).
        let mut kan_data: Vec<f32> = Vec::with_capacity(n * KAN_WEIGHTS_PER_SER);
        for eng in engines[..n].iter() {
            for filter in eng.kan_filters.filters.iter().take(SENSORY_WAVE_SIZE) {
                kan_data.extend_from_slice(&filter.spline_coeffs);
            }
            // Padding se num filters < SENSORY_WAVE_SIZE (não deve ocorrer, mas defensivo)
            let missing = SENSORY_WAVE_SIZE.saturating_sub(eng.kan_filters.filters.len());
            for _ in 0..missing {
                // Coeficientes identity = 1.0 (sem deformação)
                for _ in 0..KAN_KNOTS {
                    kan_data.push(1.0);
                }
            }
        }
        self.queue.write_buffer(&self.buf_kan, 0, bytemuck::cast_slice(&kan_data));

        // ── L1 (instinct_layer: BitLinear WAVE→96) ───────────────────────
        // weights: Vec<f32>, row-major: weights[j * WAVE + i] = peso do neurônio j para input i
        let mut l1_data: Vec<f32> = Vec::with_capacity(n * L1_WEIGHTS_PER_SER);
        for eng in engines[..n].iter() {
            // Pesos estão em Vec<f32> com len = WAVE * L1_OUT (layers.rs:40)
            let w = &eng.instinct_layer.weights;
            debug_assert_eq!(w.len(), L1_WEIGHTS_PER_SER,
                "L1 weights size mismatch: {} vs {}", w.len(), L1_WEIGHTS_PER_SER);
            l1_data.extend_from_slice(w);
        }
        self.queue.write_buffer(&self.buf_l1w, 0, bytemuck::cast_slice(&l1_data));

        // ── L2 (cortex_l2: FloatLinear 96→48) ────────────────────────────
        // weights: Vec<f32>, row-major: weights[j * L1_OUT + i]
        let mut l2_data: Vec<f32> = Vec::with_capacity(n * L2_WEIGHTS_PER_SER);
        for eng in engines[..n].iter() {
            let w = &eng.cortex_l2.weights;
            debug_assert_eq!(w.len(), L2_WEIGHTS_PER_SER,
                "L2 weights size mismatch: {} vs {}", w.len(), L2_WEIGHTS_PER_SER);
            l2_data.extend_from_slice(w);
        }
        self.queue.write_buffer(&self.buf_l2w, 0, bytemuck::cast_slice(&l2_data));
    }

    /// Envia o banco de τ: `n × L1_NEURONS` taxas de decaimento em s⁻¹, em ordem de ser.
    ///
    /// Chamado uma vez por geração, junto com os pesos — τ é herdado, não muda por tick.
    /// Quem não chamar fica com o `DECAY_DEFAULT` da inicialização, que é o τ = 2,0 s
    /// uniforme de antes; assim o controle do experimento é o comportamento histórico.
    pub fn upload_tau(&self, taus: &[f32], n: usize) {
        assert!(n <= self.capacity, "n={n} excede capacidade {}", self.capacity);
        // Comprimento errado aqui seria um bug MUDO: o kernel leria τ do ser seguinte
        // (ou lixo do fim do buffer) e a condição "heterogênea" viraria ruído embaralhado
        // sem nenhum sintoma visível. Foi exatamente essa classe de erro — stride errado
        // na onda sensorial — que já embaralhou a entrada do cérebro lento neste projeto.
        assert_eq!(taus.len(), n * L1_NEURONS,
            "banco de τ tem {} valores, esperava {} ({n} seres × {L1_NEURONS} canais)",
            taus.len(), n * L1_NEURONS);
        self.queue.write_buffer(&self.buf_tau, 0, bytemuck::cast_slice(taus));
    }

    // ── DOWNLOAD DO ESTADO CFC (diagnóstico) ──────────────────────────────

    /// Lê buf_cfc da GPU e devolve o state[0..96] de cada ser.
    /// Usado pelo gpu_validate para comparar CfC pós-forward com a CPU.
    /// Custo: uma cópia de GPU→CPU (readback). Não usar em hot-path de tick.
    pub fn download_cfc_state(&self, n: usize) -> Vec<Vec<f32>> {
        let n = n.min(self.capacity);
        let bytes = (n * CFC_SLOTS_PER_SER * 4) as u64;

        // Buffer MAP_READ temporário para readback do CfC
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-cfc-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.buf_cfc, 0, &staging, 0, bytes);
        self.queue.submit(Some(enc.finish()));

        let slice = staging.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        let vals: &[f32] = bytemuck::cast_slice(&data);

        // Cada ser tem CFC_SLOTS_PER_SER=97 f32: state[0..96] + counter (bitcast f32)
        (0..n).map(|s| {
            let base = s * CFC_SLOTS_PER_SER;
            vals[base..base + 96].to_vec()  // só o state, sem o counter
        }).collect()
    }

    // ── DOWNLOAD DO BUFFER DE DEBUG (diagnóstico) ─────────────────────────

    /// Lê buf_debug da GPU: l1_raw[0..96], cfc_in[0..96], kan_in[96..222] para ser 0.
    /// Retorna (l1_raw, cfc_inputs, kan_inputs).
    /// Usado pelo gpu_validate para comparar L1 raw diretamente com a CPU.
    pub fn download_debug(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let bytes = (319 * 4) as u64;

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-debug-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.buf_debug, 0, &staging, 0, bytes);
        self.queue.submit(Some(enc.finish()));

        let slice = staging.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        let vals: &[f32] = bytemuck::cast_slice(&data);

        let l1_raw   = vals[0..96].to_vec();
        let cfc_in   = vals[96..192].to_vec();
        let kan_in   = vals[192..319].to_vec();
        (l1_raw, cfc_in, kan_in)
    }

    // ── DOWNLOAD DO BUFFER DE FADIGA (diagnóstico) ───────────────────────────

    /// Lê buf_fatigue da GPU para diagnóstico.
    /// Retorna (n × FATIGUE_PER_SER) floats:
    ///   slots [0..96]   por ser = fadiga input CfC
    ///   slots [96..192] por ser = fadiga output (out_fat para cada canal)
    pub fn download_fatigue(&self, n: usize) -> Vec<f32> {
        let n = n.min(self.capacity);
        let bytes = (n * FATIGUE_PER_SER * 4) as u64;

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-fatigue-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.buf_fatigue, 0, &staging, 0, bytes);
        self.queue.submit(Some(enc.finish()));

        let slice = staging.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        bytemuck::cast_slice::<u8, f32>(&data).to_vec()
    }

    // ── RESET DE ESTADO CFC PARA UM SER ───────────────────────────────────

    /// Reseta o estado CfC e a fadiga adaptativa de um ser específico para zeros.
    ///
    /// Equivalente a `CfCCell::reset()`. Escreve zeros nos 97 f32 do CfC e nos
    /// 192 f32 da fadiga do ser (96 input + 96 output), preservando o estado dos outros.
    /// Chamado em spawn/death — garante que ser recém-nascido começa sem fadiga.
    pub fn reset_cfc(&self, ser_idx: usize) {
        if ser_idx >= self.capacity { return; }
        // Reseta CfC state + counter (97 f32)
        let cfc_offset = (ser_idx * CFC_SLOTS_PER_SER * 4) as u64;
        let cfc_zeros: Vec<u8> = vec![0u8; CFC_SLOTS_PER_SER * 4];
        self.queue.write_buffer(&self.buf_cfc, cfc_offset, &cfc_zeros);
        // Reseta fadiga adaptativa: 192 f32 por ser (slots [0..96] input + [96..192] output).
        let fat_offset = (ser_idx * FATIGUE_PER_SER * 4) as u64;
        let fat_zeros: Vec<u8> = vec![0u8; FATIGUE_PER_SER * 4];
        self.queue.write_buffer(&self.buf_fatigue, fat_offset, &fat_zeros);
    }

    // ── FORWARD PASS (por tick) ───────────────────────────────────────────

    /// Executa o forward pass do cérebro de todos os seres na GPU.
    ///
    /// Entradas:
    ///   `inputs`: slice de `(engine, wave)` por ser, na mesma ordem do spawn.
    ///   `dt`: delta de tempo deste tick (seg).
    ///
    /// Saída:
    ///   Vec de `[f32; INTENT_DIM]`, um por ser, na mesma ordem.
    ///   Equivalente a `engine.final_intent` pós `execute_cycle`, excluindo
    ///   Trinity, axiom_policy e efeitos colaterais de metabolism.
    ///
    /// # Nota de equivalência
    ///
    /// O caller (motor.rs) deve:
    /// 1. Chamar este método para obter final_intent GPU.
    /// 2. Aplicar axiom_policy e Trinity (Gate+Reflex+Sentinel) na CPU,
    ///    exatamente como fazia antes — esses caminhos não mudaram.
    /// 3. NÃO chamar `engine.execute_cycle` para os seres que foram para a GPU
    ///    (senão o estado CfC seria atualizado duas vezes).
    pub fn forward(
        &self,
        inputs: &[BrainInput<'_>],
        dt: f32,
    ) -> Vec<[f32; INTENT_DIM]> {
        let n = inputs.len().min(self.capacity);
        if n == 0 { return Vec::new(); }

        // ── Upload da sensory_wave (muda a cada tick) ────────────────────
        // Converte i8 → i32 (sign-extend) para o shader leia sem bit-tricks.
        let mut wave_data: Vec<i32> = Vec::with_capacity(n * SENSORY_WAVE_SIZE);
        for input in inputs[..n].iter() {
            for &b in input.wave.iter() {
                wave_data.push(b as i32);
            }
        }
        self.queue.write_buffer(&self.buf_wave, 0, bytemuck::cast_slice(&wave_data));

        // ── Uniform params ───────────────────────────────────────────────
        // use_percept_direct = 0: lê do buf_wave (binding 1), path CPU.
        // O gpu_validate usa forward() — nunca deve passar 1 aqui.
        let params = BrainParams {
            n_beings: n as u32,
            dt,
            use_percept_direct: 0,
            out_beta: crate::gpu::brain::OUT_BETA_PADRAO,
        };
        let buf_params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brain-params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // ── Bind group ───────────────────────────────────────────────────
        // Binding 8 = buf_percept_dummy (placeholder, não lido com use_percept_direct=0).
        // Binding 9 = buf_fatigue (fadiga adaptativa, residente entre ticks).
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_wave.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_kan.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_l1w.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.buf_l2w.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.buf_intent.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.buf_cfc.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.buf_debug.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.buf_percept_dummy.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: self.buf_fatigue.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: self.buf_tau.as_entire_binding() },
            ],
        });

        // ── Dispatch ─────────────────────────────────────────────────────
        // 1 workgroup por ser. Workgroup size = 128 threads (WGSL).
        let bytes_intent = (n * INTENT_DIM * 4) as u64;
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            // Dispatch exato: 1 workgroup = 1 ser. Sem arredondamento necessário.
            pass.dispatch_workgroups(n as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.buf_intent, 0, &self.buf_readback, 0, bytes_intent);
        self.queue.submit(Some(enc.finish()));

        // ── Readback ─────────────────────────────────────────────────────
        let slice = self.buf_readback.slice(0..bytes_intent);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        // poll(Wait) bloqueia até o GPU terminar. Para uso em Bevy system isso é
        // aceitável porque o system já bloqueia a thread do scheduler de qualquer forma.
        // Alternativa async (não bloqueia a thread) exigiria reestruturar o system
        // para dois stages (submit / collect) — melhoria futura se necessário.
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        let vals: &[f32] = bytemuck::cast_slice(&data);

        let mut result: Vec<[f32; INTENT_DIM]> = Vec::with_capacity(n);
        for s in 0..n {
            let base = s * INTENT_DIM;
            let mut intent = [0.0f32; INTENT_DIM];
            intent.copy_from_slice(&vals[base..base + INTENT_DIM]);
            result.push(intent);
        }

        drop(data);
        self.buf_readback.unmap();
        result
    }

    // ── MÉTODOS DE ENCADEAMENTO (sem submit próprio) ──────────────────────────

    /// Prepara a sensory_wave e grava o dispatch do cérebro num CommandEncoder externo.
    ///
    /// # Por que existe
    ///
    /// `forward()` fazia encoder + submit + poll próprios, adicionando ~0.3–0.5 ms de
    /// round-trip por tick (medido: overhead de command buffer + sincronização de fila).
    /// Com `bind_and_dispatch`, retina → percepção → cérebro ficam num único encoder e
    /// um único submit, eliminando 2 round-trips intermediários por tick.
    ///
    /// # Contrato
    ///
    /// 1. Chamar `queue.write_buffer(&self.buf_wave, ...)` ANTES deste método (ou passar
    ///    `wave_data_i32` aqui — feito internamente).
    /// 2. Gravar no encoder externo com este método.
    /// 3. Copiar `buf_intent → buf_readback` no mesmo encoder via `enc.copy_buffer_to_buffer`.
    /// 4. `queue.submit(enc.finish())` + `device.poll(Wait)`.
    /// 5. Chamar `read_intent_after_submit(n)` para ler o resultado.
    ///
    /// Retorna `bytes_intent` para o copy subsequente.
    ///
    /// Modo CPU (wave_data_i32 fornecido, buf_percept = None):
    ///   Sobe wave via write_buffer. use_percept_direct = 0. Usado pelo gpu_validate.
    ///
    /// Modo GPU (wave_data_i32 ignorado, buf_percept = Some(&buf)):
    ///   O shader lê o wave direto do buf_percept (saída de percepcao.wgsl, 56 u32/ser).
    ///   use_percept_direct = 1. Usado pelo gpu_loop. Zero readback intermediário.
    pub fn bind_and_dispatch(
        &self,
        enc:          &mut wgpu::CommandEncoder,
        n:            usize,
        wave_data_i32: &[i32],               // usado APENAS quando buf_percept = None
        dt:           f32,
        buf_percept:  Option<&wgpu::Buffer>, // Some = GPU path (use_percept_direct=1)
    ) -> u64 {
        let n = n.min(self.capacity);
        if n == 0 { return 0; }

        // Decide o modo de leitura do wave.
        let use_percept_direct: u32;
        let percept_binding: &wgpu::Buffer;

        if let Some(pb) = buf_percept {
            // GPU path: buf_percept contém saída de percepcao.wgsl.
            // NÃO subimos wave_data_i32 — seria uma escrita inútil e custaria PCIe.
            use_percept_direct = 1;
            percept_binding = pb;
        } else {
            // CPU path: sobe wave_data_i32 via write_buffer (sem bloqueio até submit).
            // wave_data_i32 deve ter len = n × SENSORY_WAVE_SIZE.
            self.queue.write_buffer(&self.buf_wave, 0, bytemuck::cast_slice(wave_data_i32));
            use_percept_direct = 0;
            percept_binding = &self.buf_percept_dummy;
        };

        // Uniform: 16 bytes, custo negligível.
        let params = BrainParams {
            n_beings: n as u32,
            dt,
            use_percept_direct,
            out_beta: self.out_beta,
        };
        let buf_params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("brain-params-chain"),
            contents: bytemuck::bytes_of(&params),
            usage:    wgpu::BufferUsages::UNIFORM,
        });

        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_wave.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_kan.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_l1w.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.buf_l2w.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.buf_intent.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.buf_cfc.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.buf_debug.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: percept_binding.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: self.buf_fatigue.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: self.buf_tau.as_entire_binding() },
            ],
        });

        // 1 workgroup por ser — sem arredondamento.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:            Some("pass-brain"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(n as u32, 1, 1);
        }

        let bytes_intent = (n * INTENT_DIM * 4) as u64;

        // Copia no mesmo encoder — DMA sem round-trip.
        enc.copy_buffer_to_buffer(&self.buf_intent, 0, &self.buf_readback, 0, bytes_intent);

        bytes_intent
    }

    /// Lê o buf_readback do cérebro após device.poll(Wait) do encoder externo.
    ///
    /// Deve ser chamado APENAS depois que o CommandEncoder que continha `bind_and_dispatch`
    /// foi submetido e `device.poll(Wait)` retornou. Sem isso o buffer não estará mapeado.
    pub fn read_intent_after_submit(&self, n: usize) -> Vec<[f32; INTENT_DIM]> {
        let n = n.min(self.capacity);
        if n == 0 { return Vec::new(); }

        let bytes_intent = (n * INTENT_DIM * 4) as u64;
        let slice = self.buf_readback.slice(0..bytes_intent);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        // TRAVAMENTO CORRIGIDO: `map_async` registra um callback que só é SERVIDO por um
        // `poll` POSTERIOR. O comentário anterior dizia "poll já foi feito pelo caller,
        // recv() completa imediatamente" — e estava errado: o poll do chamador acontece
        // ANTES do map_async, então o callback nunca era servido e `recv()` bloqueava para
        // sempre. Era isso que prendia `gpu_loop` no tick de aquecimento, sem erro e sem
        // saída. O poll tem de vir DEPOIS do pedido de mapeamento.
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        let vals: &[f32] = bytemuck::cast_slice(&data);
        let mut result: Vec<[f32; INTENT_DIM]> = Vec::with_capacity(n);
        for s in 0..n {
            let base = s * INTENT_DIM;
            let mut intent = [0.0f32; INTENT_DIM];
            intent.copy_from_slice(&vals[base..base + INTENT_DIM]);
            result.push(intent);
        }
        drop(data);
        self.buf_readback.unmap();
        result
    }

    /// Referência ao buf_intent (VRAM) para outros kernels consumirem o output direto.
    pub fn buf_intent(&self) -> &wgpu::Buffer { &self.buf_intent }
}

// ── RECURSO BEVY ─────────────────────────────────────────────────────────────

/// Recurso Bevy que mantém o contexto cerebral GPU entre ticks.
///
/// Segue o padrão de `RetinaGpuRes`: `ctx: Option<BrainGpu>`, recriado se
/// a capacidade for insuficiente ou se o dispositivo for perdido.
///
/// O caller deve verificar `IARA_CEREBRO_CPU=1` (env var) para forçar CPU,
/// exatamente como a retina usa `IARA_RETINA_CPU=1` — permite A/B controlado.
#[derive(bevy::prelude::Resource, Default)]
pub struct BrainGpuRes {
    pub ctx: Option<BrainGpu>,
}

impl BrainGpuRes {
    /// Devolve contexto válido, criando se necessário.
    ///
    /// Recebe device/queue do `RetinaGpu` para reutilizar o mesmo dispositivo.
    /// `None` = sem GPU utilizável; o caller usa CPU.
    pub fn get_or_create(
        &mut self,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        name: String,
        capacity: usize,
    ) -> Option<&BrainGpu> {
        let needs_alloc = self.ctx.is_none()
            || self.ctx.as_ref().map(|c| c.capacity < capacity).unwrap_or(true);
        if needs_alloc {
            self.ctx = Some(BrainGpu::new(device, queue, name, capacity.max(64)));
            if let Some(c) = &self.ctx {
                println!("🧠 cérebro na GPU ({} seres)", c.capacity);
            }
        }
        self.ctx.as_ref()
    }
}

// ── HELPERS (mesmos de mod.rs, duplicados para não quebrar regra de não editar mod.rs) ─

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

// ── COMO TESTAR EQUIVALÊNCIA (sem rodar nada) ─────────────────────────────────
//
// O arnês de validação `src/bin/gpu_valida.rs` (agente D) executará:
//
// 1. Criação de um SovereignEngine com pesos aleatórios mas determinísticos
//    (semente fixada, igual na CPU e na GPU).
//
// 2. Vetor de inputs fixos (sensory_wave conhecido).
//
// 3. CPU: chama `engine.execute_cycle(wave, 0.0, DT)` com DT=1/64, depois
//    lê `engine.final_intent[0..96]`.
//
// 4. GPU: chama `cerebro.forward(&[BrainInput{engine, wave}], DT)`, obtém
//    `intent_gpu[0..96]`.
//
// 5. Compara par a par: `|cpu[i] - gpu[i]| <= 2e-5` (tolerância ≤ 2 ULP pra
//    f32 em [-1, 1], contrato do GPU_PLANO.md).
//    Para o estado CfC acumulado após N ticks: medir a deriva e declarar
//    (não assumir zero drift).
//
// 6. ATENÇÃO: desabilitar no teste o `metabolism.can_process()` e tudo que
//    modifica state fora do caminho neural puro (cortisol, dopamina, etc.),
//    para que a comparação seja justa (esses efeitos ficam na CPU e não entram
//    no kernel).
//
// 7. ATENÇÃO 2: `execute_cycle` incrementa `denoise_counter` via `cfc_state.update`.
//    Na GPU, o counter começa em 0 e é salvo em `buf_cfc`. Para o teste ser
//    comparável, inicializar ambos com counter = 0 e rodar exactamente N ticks.
//
// 8. DIVERGÊNCIAS ESPERADAS DOCUMENTADAS:
//    - tanh: ≤ 1 ULP (IEEE 754)
//    - exp:  ≤ 1 ULP
//    - Matmul L1 (WAVE=223 ops): acumulação f32 sequencial, mesma ordem.
//      Diferença esperada: < 1e-6 por neurônio (223 fma em f32, drift < 1e-5).
//    - Saída quantizada i8 pós-tanh: diferença ≤ 1 LSB = 1/127 ≈ 0.008.
//
// Este comentário substitui um arquivo de teste que o agente D vai implementar.
