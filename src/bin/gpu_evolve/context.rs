//! GpuCtx: contexto GPU do laço de evolução — retina + percepção + cérebro rápido +
//! cérebro lento + física, tudo num único device wgpu compartilhado.
//!
//! Espelha (estrutura, não código compartilhado) o `GpuLoop` de gpu_loop.rs — a
//! diferença real está em `tick()`: SEM muletas motoras (escape_ticks, curvature
//! governor). O cérebro decide tudo; o piloto fantasma/sensorial (pilots.rs) só entra
//! por flag de diagnóstico.

use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use wgpu::util::DeviceExt;

use universe::analysis::regioes::{ContextoAmostra, CurvaturaPista};
use universe::config::iara::vision::{RETINA_ROWS, RETINA_TOTAL};
use universe::engine::SovereignEngine;
use universe::fly_map::spiking::EyeMode;
use universe::gpu::brain::BrainGpu;
use universe::gpu::core::{storage_entry, uniform_entry};
use universe::gpu::fly_map::{FlyMapGpu, FlySpikingGpu};
use universe::gpu::perception::{PerceptionGpu, PosePerception};
use universe::gpu::physics::{PhysicsGpu, PhysicsStateGpu};
use universe::gpu::retina_digest::RetinaDigestGpu;
use universe::gpu::retina_types::{PoseRaw, RetinaParamsGpu};
use universe::gpu::slow_brain::SlowBrainGpu;
use universe::gpu::timing::GpuTimer;
use universe::kan::KAN_OUT_KNOTS;
use universe::perception::channels::Channel;
use universe::simulation::racing_line::RacingLine;
use universe::visao::pitch;
use universe::world::VoxelWorld;

use crate::audit::auditoria_acumular;
use crate::statics::*;
use crate::types::BeingState;

pub struct GpuCtx {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub pipeline_retina: wgpu::ComputePipeline,
    pub layout_retina: wgpu::BindGroupLayout,
    pub buf_grid: wgpu::Buffer,
    pub buf_poses_retina: wgpu::Buffer,
    pub buf_retina_out: wgpu::Buffer,
    pub grid_size: u32,
    pub cell_size: f32,
    pub percept: PerceptionGpu,
    pub brain: BrainGpu,
    /// LABORATÓRIO DE OBSERVAÇÃO (`--observar`): destino da gravação por tick do
    /// ser 0 — ver o bloco de captura em `tick()` para o layout (394 f32/linha).
    pub obs_writer: Option<std::io::BufWriter<std::fs::File>>,
    /// `--record-wave <path>`: raw 192-channel sensory-wave recording for being 0, one
    /// row per tick (`gpu::wave_record`, 195 f32/row). Records the SAME being
    /// `--observe` does, so the two files line up tick for tick — that alignment is
    /// what the Task 5 open-loop screen cross-checks against. Forces a per-tick
    /// readback of the whole wave buffer (`percept.read_after_submit`), so it must stay
    /// off by default; once Phase 1 (the open-loop brain-kernel screen) exists, using
    /// this flag together with it must force `K = 1` (one candidate replayed per pass,
    /// not a batch), because a batched replay could not also afford this readback.
    pub wave_writer: Option<std::io::BufWriter<std::fs::File>>,
    /// Optional compact per-neuron spiking trace (`--fly-spiking-trace`). The writer is
    /// deliberately owned by the context so the GPU snapshot is mapped and persisted at
    /// the same tick boundary as the command readback.
    pub spiking_trace_writer: Option<std::io::BufWriter<std::fs::File>>,
    pub spiking_trace_every: usize,
    pub spiking_trace_being: usize,
    /// E-REG1: custo de fiação por ser (constante durante a geração — pesos não
    /// mudam no meio). Preenchido pelo main a cada aplicação de genomas; drenado
    /// da vida por tick quando CUSTO_FIACAO > 0. Vazio = sem custo.
    pub fiacao_custo: Vec<f32>,
    /// PILOTO POR FÓRMULA: quando presente, o ser 0 é dirigido pelo controlador
    /// ridge (X = debug do cérebro pós-KAN/CfC, mesma fonte do --observar).
    pub piloto_formula: Option<crate::piloto_linear::PilotoLinear>,
    /// PILOTO v2 (não-linear). Mesma posição no tick, mesmo X.
    pub piloto_forja: Option<crate::piloto_forja::PilotoForja>,
    /// M4: pesos do maestro por ser (n×16), preenchidos pelo main por geração.
    pub gates_w: Vec<f32>,
    /// M4: último gate aplicado ao ser 0 (telemetria [neuromod] do main).
    pub ultimo_gate: [f32; 4],
    /// Metade LENTA do cérebro (System 2): CfC-64 → Dense → Dense, 1 tick a cada 8.
    /// Estava ausente de todo o caminho GPU — `grep slow src/gpu/` voltava vazio — o que
    /// significa que "portar o cérebro para a GPU" tinha na verdade portado só a metade
    /// rápida. As três saídas modulam agressão, teto de acelerador e viés de volante.
    pub slow: SlowBrainGpu,
    /// Amostras da saída do lento ao longo da geração, para MEDIR se ele merece existir.
    /// Uma linha por amostra, três colunas — alimenta `brain_capacity::participation_ratio`.
    pub slow_samples: Vec<Vec<f32>>,
    /// Última intenção neural, reusada nos ticks em que o cérebro não dispara.
    pub last_intents: Vec<[f32; 96]>,
    /// Ticks de física por decisão neural. 1 = decide todo tick (64 Hz).
    ///
    /// É a "velocidade de simulação" pelo lado do cérebro. O `motor.rs` tinha esse
    /// mecanismo adaptativo pelo cortisol e ele foi DESLIGADO com a justificativa
    /// registrada em motor.rs:1717 — "skip_n=2 em gen>=10 reduzia brain freq pra 30Hz e
    /// degradava DNA reload". Ou seja: já se sabia que existe um ótimo interior, e a
    /// resposta foi travar em 1 em vez de medir onde ele fica. Aqui vira parâmetro.
    pub brain_skip: u32,
    pub brain_tick: u32,
    pub racing_line: RacingLine,
    pub grid_cpu: Vec<u32>,
    pub grid_size_cpu: u32,
    pub cell_size_cpu: f32,
    pub phys_gpu: PhysicsGpu,
    /// CfC de percepção + min_wall_ahead + edge_row6/7 — kernel `retina_digest.wgsl`.
    /// Substitui o antigo readback de 340 KB/tick (retina inteira) + laço Rust
    /// (`percept_cfc: Vec<RetinaPerception>` rodando na CPU); ver doc do módulo
    /// `gpu::retina_digest` para o antes/depois completo.
    pub retina_digest: RetinaDigestGpu,
    pub capacity: usize,
    /// Per-kernel GPU timing (`--kernel-profile`, gated by `statics::perfil_kernel()`).
    /// `None` when the device lacks `TIMESTAMP_QUERY` (checked once at construction).
    /// Lives here rather than in `statics.rs` because it owns a `QuerySet` and two
    /// buffers — device-backed state, not a CLI knob (see the ownership table in
    /// `statics.rs` for why state and dose are kept apart).
    pub timer: Option<GpuTimer>,

    // ── Amostras de ativação (E4, `--ativacao`) ─────────────────────────────
    // Todas bounded por geração: `main.rs` faz `std::mem::take` no relatório de
    // fim-de-geração (mesmo padrão de `slow_samples`) — sem isso acumulariam sem
    // limite numa bateria longa (risco R6 do REFACTOR_PLAN.md).
    /// Estado CfC de L1 (96 valores), uma linha por (ser, tick amostrado).
    pub l1_samples: Vec<Vec<f32>>,
    /// `final_intent[0..48]` (saída pós-L2), uma linha por (ser, tick amostrado).
    /// Reaproveita o readback de `intents` que já acontece todo tick — custo extra
    /// zero além do `l1_samples` (ver comentário no ponto de amostragem em `tick()`).
    pub l2_samples: Vec<Vec<f32>>,
    /// (throttle, steer, brake) do "campeão" (ser 0 = elite herdado da geração
    /// anterior sem mutação — mesmo raciocínio do bloco `[elite]` em main.rs),
    /// amostrado no mesmo passo — usado para entropia de política.
    pub policy_samples: Vec<[f32; 3]>,
    /// Índices de ser com NaN/Inf detectado no estado L1 nesta geração (guarda de
    /// runtime, item 5 do E4). Populado em `tick()`, consumido/relatado em `main.rs`.
    pub nan_seres: std::collections::HashSet<usize>,

    // ── Regiões da pista × ativação (`--regioes`, src/analysis/regioes.rs) ─────
    // Nunca cruzávamos ativação com POSIÇÃO. Estes dois vetores andam em LOCKSTEP
    // com `l1_samples`/`l2_samples` — empilhados na MESMA iteração do mesmo loop,
    // então `regiao_l1[i]` é o contexto físico (curvatura/velocidade/dist_parede)
    // exato do (ser, tick) que produziu `l1_samples[i]`. É o que garante o
    // alinhamento por índice que `analysis::regioes::analisar` exige sem precisar
    // carregar tick/ser junto — mais barato e sem acoplar os dois módulos.
    /// Contexto alinhado a `l1_samples` (mesmo tamanho sempre, mesmo `continue` em NaN).
    pub regiao_l1: Vec<ContextoAmostra>,
    /// Contexto alinhado a `l2_samples` (mesmo tamanho sempre — L2 nunca pula por NaN).
    pub regiao_l2: Vec<ContextoAmostra>,
    /// κ(x,z) da pista atual — recalculado em `new()` e em `reload_world()` (rotação
    /// de pistas, `--pistas`/`--pista-held-out`) para nunca medir curvatura da pista
    /// errada depois de uma troca.
    track_curvatura: CurvaturaPista,

    // ── HOSTILE WORLD (BODY PHASE, 2026-08-17) ──────────────────────────────
    //
    // STATE, not dose. The dose knobs are in `statics.rs` and its ownership table
    // says why: a static is a knob the CLI turns once, and simulation state in a
    // process global is how one run contaminates the next inside the same process.
    // All four are inert at their defaults (empty vector / `None` / 0), so a run
    // without the new flags takes exactly the old code path.
    /// D7 energy stations of the CURRENT generation. Empty = `--resource-scarce` off.
    /// Placed by `main()` once per generation (same seed ⇒ same field for every being
    /// of the generation, the D2 comparability contract); drained and recharged here.
    pub stations: Vec<universe::resources::EnergyStation>,
    /// Dose of the station field, rebuilt by `main()` from the `--recurso-*` statics.
    pub station_cfg: universe::resources::StationConfig,
    /// Scripted predator of the CURRENT generation. `None` = `--predator` off.
    pub predator: Option<universe::predator::Predator>,
    /// Dose of the predator, rebuilt by `main()` from the `--predator*` statics.
    pub predator_params: universe::predator::PredatorParams,
    /// Tick index below which the predator does not drain — `--predator-grace`.
    pub predator_grace_ticks: usize,
    /// Per-being tyre-thermal parameters. Empty = `--tire-thermal` off. One entry per
    /// being because the optimum is heterogeneous; each entry is DC-normalised
    /// separately, so selection sees the behaviour and not the parameter.
    pub thermal: Vec<universe::tire_thermal::ThermalParams>,
    /// D8 per-being sensory subscription, life/tick. Same shape and lifetime as
    /// `fiacao_custo`: constant during a generation because the weights are. Empty = off.
    pub sensor_cost: Vec<f32>,

    // ── Telemetry of the hostile world, accumulated over one generation ──────
    // `main()` zeroes these at the start of every generation and prints them in the
    // `[recurso]` / `[predador]` / `[pneu]` lines. They are counters, not decisions:
    // nothing in the tick reads them back.
    /// Beings whose fuel reached zero this generation (each being counted once).
    pub tel_dry_beings: u32,
    /// Being-ticks spent inside the predator's drain radius.
    pub tel_predator_ticks: u64,
    /// Total life ticks the predator took this generation.
    pub tel_predator_life: f64,
    /// Fuel actually transferred by the stations this generation, fuel units.
    pub tel_station_fuel: f64,
    /// Sum over sampled being-ticks of the normalised tyre temperature, and the count —
    /// the mean is the honest way to report a per-tick state in one number.
    pub tel_tire_temp_sum: f64,
    /// Sum of SQUARED normalised temperature, for the standard deviation.
    ///
    /// The mean is the wrong canary for this mechanism and the sweep of 2026-08-26 showed
    /// why: `normalize_dc` exists precisely to hold the MEAN grip fixed across doses, so
    /// asking the mean temperature to move is asking the DC normaliser to fail. Measured,
    /// the mean barely budges (0.531 at g=0.5 against 0.488 at g=2.0) while the dose is a
    /// factor of four. What the dose controls is the SWING, so that is what has to be
    /// printed, or the canary passes a mechanism that is not integrating.
    pub tel_tire_temp_sq: f64,
    pub tel_tire_temp_n: u64,

    // ── C0 per-tick lap telemetry (PRD `docs/PRD-CEREBRO-NEUROMOD.md` §C0) ──
    // Three per-being vectors rewritten from scratch every `tick()`, which `main()`
    // reads right after the call and hands to `laps::LapTracker::on_tick`. They live
    // here, and not in `main()`, because the three values are produced at three
    // different points INSIDE this function (grass in the proprioception block, dry in
    // the fuel drain, damage in the hidden-damage block); recomputing them outside
    // would be a second copy of three formulas. Length `n` after every `tick()`.
    // Nothing in the tick reads them back — they are telemetry, not state.
    /// Being standing on a grass voxel this tick (the value behind `prop[9]`).
    pub lap_grass: Vec<bool>,
    /// Hidden damage actually added to the being this tick, after the 1.0 clamp.
    pub lap_damage: Vec<f32>,
    /// The being's tank reached zero this tick (the `tel_dry_beings` event).
    pub lap_dry: Vec<bool>,
    // ── C1 — SECTOR HIPPOCAMPUS (docs/PRD-CEREBRO-NEUROMOD.md §C1) ──────────
    //
    // STATE, not dose — same reasoning as the hostile-world block above, and the same
    // consequence: the three CLI knobs live in `statics.rs`, the 24 memory slots per
    // being live here. All three fields are inert at their defaults (empty vectors,
    // `None`), so a run without `--hippocampus-sector` takes exactly the old code path.
    /// One hippocampus per being, index-aligned with `states`. Empty = the organ was
    /// never switched on this run. Rebuilt (not carried) at the start of every
    /// generation by `main()`: memory that survives a generation boundary would make the
    /// champion's re-evaluation non-deterministic.
    pub hip_mem: Vec<universe::gpu::hippocampus::SectorMemory>,
    /// `n × N_HIP_GENES` hippocampus weights, filled once per generation by `main()` from
    /// the genomes — the same lifetime and the same reason as `gates_w`: the genes do not
    /// change inside a generation, so flattening them once per generation beats reaching
    /// into the genome on every tick.
    pub hip_genes: Vec<f32>,
    /// `--hippocampus-sector`: per-generation `hip_gen<N>.f32` recording of being 0, one row
    /// of `HIP_OBS_COLS` f32 per tick. A SEPARATE file from `obs.f32` on purpose —
    /// `obs.f32` is 394 fixed columns and several tools index into it by offset.
    pub hip_writer: Option<std::io::BufWriter<std::fs::File>>,
    /// FLY MAP (docs/FLY-MAP-BRAIN.md) — the Drosophila connectome, resident on this same
    /// device, stepped once per tick. `None` unless `--fly-map` was given, and `None` is
    /// the whole flag-off contract: no buffer is created, no kernel is recorded, no
    /// readback happens and the tick is byte-for-byte the one that predates this field.
    ///
    /// State, not dose: it owns seventeen device buffers, so it lives here and not in
    /// `statics.rs` — the same line the ownership table there draws for `GpuTimer`.
    pub fly: Option<FlyMapGpu>,
    /// Per-being steering adaptation for `--fly-readout reflex` (F4v). Empty and untouched
    /// unless `--fly-reflex-alpha`/`--fly-reflex-gain` are asked for.
    fly_reflex_adapt: Vec<universe::fly_map::ReflexAdaptation>,
    /// Event-driven LIF fly map. Mutually exclusive with `fly`: both models would
    /// otherwise consume the same connectome while only one command stream is applied.
    pub fly_spiking: Option<FlySpikingGpu>,
}

/// Contexto físico (curvatura/velocidade/dist_parede) de UM ser neste tick, para
/// pareamento 1:1 com uma amostra de ativação (`--regioes`). Função livre (não método,
/// não closure) de propósito: evita qualquer disputa de borrow com `self` no ponto de
/// chamada, que já mistura leitura de `states`/`phys_states` com `self.l1_samples.push`
/// no mesmo bloco — ver `tick()`.
/// Signed bearing of `rel` in the being's own basis, normalised to `[-1, 1]`.
///
/// `atan2(rel · lat, rel · fwd) / PI`, with `fwd`/`lat` the unit basis vectors the
/// physics state carries (`lat = [fwd.z, 0, -fwd.x]`, `gpu/physics.rs:105`). Positive
/// means "toward `+lat`" — stated as a basis rather than as "right", because no
/// `file:line` in this repository fixes the world's handedness, and a word that is
/// wrong reads exactly like a word that is right.
///
/// A zero vector answers 0.0, the same code the block uses for "no target". That
/// collision is deliberate and matches the wall probes: a target sitting exactly on
/// the being is not distinguishable from no target, and the paired proximity channel
/// is what separates them (`perception/channels.rs`, BODY_SLOTS doc).
///
/// Pinned by `bearing_sign_follows_the_lat_basis` — the sign of a bearing is invisible
/// to every aggregate statistic this project prints, so it gets a test, not a comment.
fn signed_bearing(rel: bevy::math::Vec2, fwd: bevy::math::Vec2, lat: bevy::math::Vec2) -> f32 {
    let f = rel.dot(fwd);
    let l = rel.dot(lat);
    if f == 0.0 && l == 0.0 {
        return 0.0;
    }
    l.atan2(f) / std::f32::consts::PI
}

fn contexto_regiao(
    track: &CurvaturaPista,
    p: &PhysicsStateGpu,
    dist_parede: f32,
) -> ContextoAmostra {
    let curvatura = track.curvatura_em(p.pos[0], p.pos[2]);
    let velocidade = (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt();
    ContextoAmostra {
        curvatura,
        velocidade,
        dist_parede,
    }
}

impl GpuCtx {
    pub fn new(world: &VoxelWorld, capacity: usize) -> Option<Self> {
        let icd_radeon = "/usr/share/vulkan/icd.d/radeon_icd.json";
        if std::path::Path::new(icd_radeon).exists() && std::env::var("VK_ICD_FILENAMES").is_err() {
            unsafe {
                std::env::set_var("VK_ICD_FILENAMES", icd_radeon);
            }
        }

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
        let mut limits = wgpu::Limits::downlevel_defaults();
        // Cérebro usa 10 storage buffers: bindings 1-9 mais o `buf_tau` (banco de τ).
        //
        // O teto é de configuração, não de silício — `downlevel_defaults()` pede 4 para
        // caber em GL antigo, e a RX 6750 XT anuncia bem mais. Pedir 12 dá folga para os
        // próximos dois buffers sem voltar aqui a cada binding novo; se algum adaptador
        // não suportar, `request_device` falha na abertura em vez de corromper dado.
        // M4 (buf_gates, binding 13) estourou os 12 — subido para 16 (RX 6750 XT
        // anuncia bem mais; medido: 12 era teto de config, não de silício).
        limits.max_storage_buffers_per_shader_stage = 16;
        let (raw_device, raw_queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("iara-evolve"),
                // TIMESTAMP_QUERY and TIMESTAMP_QUERY_INSIDE_ENCODERS are requested only when
                // the adapter offers them: asking for a feature the adapter lacks makes
                // request_device return None, and losing the whole device to gain a profiler
                // is a bad trade. `src/gpu/timing.rs`'s `GpuTimer::new` gates on the first bit
                // and returns None (silently, no timings) when it was not granted; the second
                // bit is checked separately inside `GpuTimer` (`can_mark_in_encoder`) because
                // `scope` does not need it but `mark_begin`/`mark_end` do — those call
                // `CommandEncoder::write_timestamp` directly, which is a different wgpu
                // feature from the pass-level `ComputePassTimestampWrites` `scope` uses.
                // Confirmed present on this adapter (RADV/RX 6750 XT) 2026-08-26: both bits
                // set in `adapter.features()`. This is the device `gpu_evolve` actually runs
                // the tick on; `src/gpu/mod.rs`'s `RetinaGpu` is a separate device used by the
                // Bevy/`gpu_validate` path only.
                required_features: adapter.features()
                    & (wgpu::Features::TIMESTAMP_QUERY
                        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;
        let device = Arc::new(raw_device);
        let queue = Arc::new(raw_queue);
        // 16 label slots: `tick()` uses at most 7 this session (retina, perception,
        // retina_digest, fly, brain, slow.gather, slow) plus headroom for the next kernel
        // someone adds without having to revisit this constant.
        let timer = GpuTimer::new(&device, 16);
        let cap = capacity.max(1);

        // COPY_DST: permite `reload_world()` trocar de pista via `write_buffer` sem
        // recriar buffer/bind-group/pipeline (P1, rotação de pistas — ver o mesmo
        // raciocínio em `gpu::perception::PerceptionGpu::reload_world`).
        let grid_u32: Vec<u32> = world.grid.iter().map(|v| *v as u32).collect();
        let buf_grid = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("evolve-grid"),
            contents: bytemuck::cast_slice(&grid_u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let buf_poses_retina = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("evolve-poses-retina"),
            size: (cap * std::mem::size_of::<PoseRaw>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_retina_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("evolve-retina-out"),
            size: (cap * RETINA_TOTAL * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let shader_retina = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("evolve-retina"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../../gpu/retina.wgsl"))),
        });
        let layout_retina = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("evolve-retina-layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
            ],
        });
        let pl_retina = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout_retina],
            push_constant_ranges: &[],
        });
        let pipeline_retina = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("evolve-retina-pipeline"),
            layout: Some(&pl_retina),
            module: &shader_retina,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let percept = PerceptionGpu::new(&device, &queue, world, cap)?;

        let brain = BrainGpu::new(Arc::clone(&device), Arc::clone(&queue), name.clone(), cap);
        let slow = SlowBrainGpu::new_shared(Arc::clone(&device), Arc::clone(&queue), cap);

        let grid_u8: Vec<u8> = world.grid.iter().map(|v| *v as u8).collect();
        let phys_gpu = PhysicsGpu::new_shared(
            Arc::clone(&device),
            Arc::clone(&queue),
            name.clone(),
            &grid_u8,
            &world.height_map,
            world.size,
            world.cell_size,
            cap,
        )?;

        let retina_digest = RetinaDigestGpu::new(&device, &queue, cap);

        // κ(x,z) da pista inicial — mesma fonte que `reload_world` usa depois.
        let track_curvatura = {
            let wps: Vec<(f32, f32)> = world.waypoints_world.iter().map(|p| (p.x, p.y)).collect();
            CurvaturaPista::medir(&wps)
        };

        eprintln!("[gpu_evolve] Adapter: {} · cap={}", name, cap);

        Some(Self {
            device,
            queue,
            pipeline_retina,
            layout_retina,
            buf_grid,
            buf_poses_retina,
            buf_retina_out,
            grid_size: world.size as u32,
            cell_size: world.cell_size,
            percept,
            brain,
            slow,
            obs_writer: None,
            wave_writer: None,
            spiking_trace_writer: None,
            spiking_trace_every: 64,
            spiking_trace_being: 0,
            fiacao_custo: Vec::new(),
            piloto_formula: None,
            piloto_forja: None,
            gates_w: Vec::new(),
            ultimo_gate: [1.0; 4],
            slow_samples: Vec::new(),
            last_intents: Vec::new(),
            brain_skip: 1,
            brain_tick: 0,
            racing_line: world.racing_line.clone(),
            grid_cpu: grid_u32,
            grid_size_cpu: world.size as u32,
            cell_size_cpu: world.cell_size,
            phys_gpu,
            retina_digest,
            capacity: cap,
            timer,
            l1_samples: Vec::new(),
            l2_samples: Vec::new(),
            policy_samples: Vec::new(),
            nan_seres: std::collections::HashSet::new(),
            regiao_l1: Vec::new(),
            regiao_l2: Vec::new(),
            track_curvatura,
            stations: Vec::new(),
            station_cfg: universe::resources::StationConfig::default(),
            predator: None,
            predator_params: universe::predator::PredatorParams::default(),
            predator_grace_ticks: 0,
            thermal: Vec::new(),
            sensor_cost: Vec::new(),
            tel_dry_beings: 0,
            tel_predator_ticks: 0,
            tel_predator_life: 0.0,
            tel_station_fuel: 0.0,
            tel_tire_temp_sum: 0.0,
            tel_tire_temp_sq: 0.0,
            tel_tire_temp_n: 0,
            lap_grass: Vec::new(),
            lap_damage: Vec::new(),
            lap_dry: Vec::new(),
            // C1: inert until `main()` sizes them, which it only does with
            // `--hippocampus-sector`. `hippocampus_bias` returns [0, 0, 0] on an empty bank.
            hip_mem: Vec::new(),
            hip_genes: Vec::new(),
            hip_writer: None,
            // FLY MAP: `main()` fills this only when `--fly-map` was passed.
            fly: None,
            fly_reflex_adapt: Vec::new(),
            fly_spiking: None,
        })
    }

    /// Enables the opt-in fly activation trace. The header is written once and each
    /// sampled record is appended only after the corresponding GPU readback completed.
    pub fn enable_spiking_trace(
        &mut self,
        path: &str,
        every: usize,
        being: usize,
    ) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        let n = self
            .fly_spiking
            .as_ref()
            .map(|fly| fly.neurons())
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "spiking trace requires an active FlySpikingGpu",
                )
            })?;
        if being >= self.capacity {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("trace being {being} is outside capacity {}", self.capacity),
            ));
        }
        universe::fly_map::spiking::write_trace_header(&mut writer, n, being)?;
        self.spiking_trace_writer = Some(writer);
        self.spiking_trace_every = every.max(1);
        self.spiking_trace_being = being;
        Ok(())
    }

    /// Zeroes the hostile-world telemetry counters. Called by `main()` at the start of
    /// every generation, for the same reason `audit::nova_geracao()` exists: a counter
    /// that accumulates across generations reports the run, and the question is always
    /// about THIS generation.
    pub fn reset_hostile_telemetry(&mut self) {
        self.tel_dry_beings = 0;
        self.tel_predator_ticks = 0;
        self.tel_predator_life = 0.0;
        self.tel_station_fuel = 0.0;
        self.tel_tire_temp_sum = 0.0;
        self.tel_tire_temp_sq = 0.0;
        self.tel_tire_temp_n = 0;
    }

    /// Stations that are drained below the useful threshold — the `[recurso]` number
    /// that says whether the field was actually contested.
    pub fn stations_exhausted(&self) -> usize {
        let floor = universe::resources::MIN_USEFUL_CHARGE_FRAC * self.station_cfg.capacity;
        self.stations.iter().filter(|s| s.charge < floor).count()
    }

    /// Carrega pesos dos engines na GPU.
    pub fn upload_engines(&self, engines: &[SovereignEngine]) {
        let refs: Vec<&SovereignEngine> = engines.iter().take(self.capacity).collect();
        self.brain.upload_weights(&refs);
        // Pesos do lento na mesma ordem de `SlowBrain::all_weights()`, para que a
        // evolução mute um vetor só e o kernel leia o layout que já espera.
        let mut slow_w: Vec<f32> = Vec::with_capacity(refs.len() * 1091);
        for e in &refs {
            slow_w.extend_from_slice(&e.slow_brain.all_weights());
        }
        self.slow.upload_weights(&slow_w, refs.len());
    }

    /// C1 — the hippocampus bias for being `i`, ALREADY multiplied by `HIP_AUTHORITY`.
    ///
    /// Returns `[0.0, 0.0, 0.0]` — bit for bit, not approximately — whenever the organ is
    /// not active for this run, when the bank has not been sized, or when the genes are
    /// all zero (`gpu::hippocampus::bias`, identity contract). Adding an exact `+0.0` to a
    /// finite command leaves its bits untouched, which is what keeps a run without
    /// `--hippocampus-sector` identical to a run of the binary that predates C1.
    ///
    /// (The one IEEE exception, recorded rather than hidden: `-0.0 + 0.0` is `+0.0`. To
    /// reach it, `kan_eval` and the slow brain's bias would both have to be exactly
    /// `-0.0` on the same tick, and the result would still compare equal and propagate
    /// identically through every later operation — the physics never divides by a command.)
    fn hippocampus_bias(&self, i: usize) -> [f32; universe::gpu::hippocampus::HIP_OUTPUTS] {
        use universe::gpu::hippocampus as hip;
        if !hippocampus_active() || i >= self.hip_mem.len() {
            return [0.0; hip::HIP_OUTPUTS];
        }
        let genes = match self
            .hip_genes
            .get(i * hip::N_HIP_GENES..(i + 1) * hip::N_HIP_GENES)
        {
            Some(g) => g,
            None => return [0.0; hip::HIP_OUTPUTS],
        };
        let mem = &self.hip_mem[i];
        let x = mem.read_gated(mem.read_sector(), hippocampus_erase());
        let mut b = hip::bias(genes, &x);
        for v in b.iter_mut() {
            *v *= hip::HIP_AUTHORITY;
        }
        b
    }

    /// Executa um tick do laço GPU.
    /// SEM muletas: sem escape_ticks, sem throttle forçado, sem curvature governor.
    /// O cérebro decide tudo.
    pub fn tick(
        &mut self,
        states: &mut Vec<BeingState>,
        n: usize,
        dt: f32,
        tick_idx: usize,
        kan_knots: &[[f32; KAN_OUT_KNOTS]; 3],
    ) -> bool {
        if n == 0 || n > self.capacity {
            return false;
        }

        // `--kernel-profile <N>`: this single check gates EVERY instrumentation side
        // effect in this function -- `begin_frame`, every `mark_begin`/`resolve` call,
        // this CPU `Instant` (which wraps the WHOLE function body, so `gaps` at the end
        // means what it says), and physics's own CPU-wall measurement further down. A
        // run with the flag off (`n_tick == 0`, the default) never calls
        // `write_timestamp`, never resolves the query set, and never starts a stray
        // `Instant` -- so a baseline taken with the flag off is not contaminated by
        // having the capability, only by actually using it. `mark_end` no-ops on
        // `None` by construction, so it does not need its own gate.
        let n_tick = perfil_kernel();
        // `tick_idx == 0` is the JIT warm-up tick (`main.rs`'s `let _ = ctx.tick(..., 0,
        // ...)`, before the scored loop starts at `tick_idx == 1`). Modulo alone would
        // always match it (`0 % n_tick == 0` for any `n_tick`), making the FIRST sample
        // of every profiling run the warm-up tick rather than a steady-state one —
        // exactly how a retired 4-sample figure went wrong once already (5.9x spread,
        // one of the four being an outlier this large). Excluded explicitly.
        let prints_this_tick = n_tick > 0 && tick_idx > 0 && (tick_idx as u32) % n_tick == 0;
        let trace_sample = self.fly_spiking.is_some()
            && self.spiking_trace_writer.is_some()
            && tick_idx > 0
            && tick_idx % self.spiking_trace_every.max(1) == 0;
        let tick_t0 = prints_this_tick.then(std::time::Instant::now);
        // CPU-side phase breakdown, only when profiling. The GPU kernels reconstruct
        // ~8.5% of the tick at 768 beings; the other ~84% is this side, and until now
        // it was one opaque `gaps` number. Optimising it without this split is the same
        // guess that made an ablation report the brain at 21% when the kernel timer
        // measures 1.9%.
        let mut cpu_phases: Vec<(&'static str, f32)> = Vec::new();
        let mut total_gpu_ms: f32 = 0.0;
        let mut physics_ms: f32 = 0.0;

        // Sub-profile of `rec.cpu_prep` (Task 1, 2026-08-27-tick-cpu-cost): eight
        // timers partition the WHOLE prep region contiguously, from here to the
        // `rec.cpu_prep` push right before the first GPU call, so nothing in between
        // is left unbracketed. `prep.*` rows are children of `rec.cpu_prep`, not
        // siblings -- the printing code below excludes `rec.cpu_prep` from the leaf
        // sum for the same reason it already excludes `cpu.record`.
        let t_prep_body = prints_this_tick.then(std::time::Instant::now);

        // Guarda de runtime (E4, REFACTOR_PLAN.md §E4 "NaN-check"): detecta posição
        // não-finita ANTES de submeter à GPU. Só em debug — em release (`cfg!` avaliado
        // em compile-time, custo zero no binário de produção) o guard real fica nas
        // amostras de ativação abaixo, que rodam mesmo com `--release` mas só quando
        // `--ativacao` está ligado.
        if cfg!(debug_assertions) {
            debug_assert!(
                states[..n]
                    .iter()
                    .flat_map(|s| s.phys.pos.iter())
                    .all(|v| v.is_finite()),
                "NaN/Inf detectado em posições antes do submit GPU (tick {tick_idx})"
            );
        }

        // ── HOSTILE WORLD (BODY PHASE, 2026-08-17) ────────────────────────────
        //
        // Tyre thermal, D7 stations and the predator, all CPU-side, all default-off.
        // They run HERE, at the top of the tick, for one reason: everything below —
        // perception, brain, physics — has to see the same world this block produced,
        // and the body block that carries these signals to the brain is packed a few
        // dozen lines down. Running them after the physics would feed the brain a
        // world one tick older than the one it is driving in, which is the class of
        // ordering bug the pilot-override marker test exists to prevent.
        //
        // The inputs they read (`wheels[..]` slip, positions) are the OUTPUT of last
        // tick's physics, so there is exactly one tick of lag and it is the same lag
        // the retina and the proprioception already carry.
        //
        // Every branch is gated on state that is empty/`None` at its default, so with
        // no new flag this block compiles to three predictable-not-taken tests: flag
        // off is bit-for-bit the old tick (CLAUDE.md rule 3).
        let body_on = body_channels_on();
        let mut body_vals: Vec<[f32; universe::perception::channels::BODY_K]> = if body_on {
            vec![[0.0; universe::perception::channels::BODY_K]; n]
        } else {
            Vec::new()
        };
        {
            use bevy::math::Vec2;
            let pos2 = |p: &PhysicsStateGpu| Vec2::new(p.pos[0], p.pos[2]);

            // 1. TYRE THERMAL (`--tire-thermal`) — see `universe::tire_thermal`.
            //
            // The grip factor is applied through `phys.env_grip`, which is per being
            // and multiplies `mu_base` in the LIVE kernel (`physics_b.wgsl:141`). The
            // kernel's own legacy temperature factor (`physics_b.wgsl:190`) is left
            // untouched: `wheels[1].x` starts at 60.0 and is compared against
            // `optimal_temp = 0.5`, so it is pinned at the constant 0.70 — identical in
            // both arms of the A/B, therefore it cancels. Writing a real temperature
            // into `wheels[1].x` would make the kernel apply grip a SECOND time.
            //
            // `mu` fed back is the one actually used (post-factor): friction heating is
            // proportional to friction power, so a tyre that lost grip also makes less
            // heat. That negative feedback plus the floor is what bounds the overheat
            // spiral (`tire_thermal::overheat_feedback_is_bounded_by_the_floor`).
            if !self.thermal.is_empty() {
                for i in 0..n {
                    let p = &states[i].phys;
                    let speed = (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt();
                    // Aggregate slip, averaged over the four wheels. `wheels[j*2]` is
                    // `[omega, slip_ratio, slip_angle, normal_load]`
                    // (`gpu/physics.rs:49`, written back at `physics_c.wgsl:294`).
                    let mut slip = 0.0f32;
                    for j in 0..4 {
                        slip += p.wheels[j * 2][1].abs() + p.wheels[j * 2][2].abs();
                    }
                    slip *= 0.25;
                    let tp = self.thermal[i.min(self.thermal.len() - 1)];
                    let mu_used = p.mu_peak * p.terrain_grip * p.env_grip;
                    let t = universe::tire_thermal::step_temp(
                        states[i].tire_temp,
                        slip,
                        speed,
                        mu_used,
                        dt,
                        &tp,
                    );
                    states[i].tire_temp = t;
                    states[i].phys.env_grip = universe::tire_thermal::grip_factor(t, &tp);
                    self.tel_tire_temp_sum += t as f64;
                    self.tel_tire_temp_sq += (t as f64) * (t as f64);
                    self.tel_tire_temp_n += 1;
                }
            }

            // 2. D7 STATIONS (`--resource-scarce`) — `universe::resources`.
            //
            // One `step` call per tick, never `try_drain` + `recharge_all` separately
            // (the module contract): `step` drains in index order and recharges once,
            // and doing it by hand would double-recharge or reorder the scarcity.
            if !self.stations.is_empty() {
                let tank = FUEL_TANK.load(Ordering::Relaxed);
                let beings: Vec<universe::resources::BeingFuel> = (0..n)
                    .map(|i| universe::resources::BeingFuel {
                        pos: pos2(&states[i].phys),
                        fuel: states[i].fuel as f32,
                        tank,
                        alive: states[i].life_left > 0.0,
                    })
                    .collect();
                let takes =
                    universe::resources::step(&mut self.stations, &beings, &self.station_cfg);
                for i in 0..n {
                    if takes[i] > 0.0 {
                        states[i].fuel = (states[i].fuel + takes[i] as f64).min(tank as f64);
                        self.tel_station_fuel += takes[i] as f64;
                    }
                }
            }

            // 3. PREDATOR (`--predator`) — `universe::predator`.
            //
            // Paid in LIFE, never in fitness: the Crossed-Currency Law, the same choice
            // the ghost already encodes (`fantasma.rs:9-18`). Charging it in distance
            // would teach the population not to move, which is the failure the wall cost
            // already produced and paid for (`types.rs`, `life_left`).
            //
            // The ramp is continuous and quadratic, `D * (1 - d/R)^2`. A step function
            // would turn mortality into a threshold artefact — the dose would look like
            // a coin flip instead of a curve, and the canary could not read it.
            if let Some(pred) = self.predator {
                let targets: Vec<Vec2> = (0..n).map(|i| pos2(&states[i].phys)).collect();
                let alive: Vec<bool> = (0..n).map(|i| states[i].life_left > 0.0).collect();
                let next = pred.step(&targets, &alive, dt, &self.predator_params);
                self.predator = Some(next);
                let r = PREDATOR_RADIUS_M.load(Ordering::Relaxed);
                let d_max = PREDATOR_DRAIN.load(Ordering::Relaxed);
                if tick_idx >= self.predator_grace_ticks && r > 0.0 && d_max > 0.0 {
                    for i in 0..n {
                        if states[i].life_left <= 0.0 {
                            continue;
                        }
                        let d = next.pos.distance(targets[i]);
                        if d < r {
                            let ramp = 1.0 - d / r;
                            let drain = (d_max * ramp * ramp) as f64;
                            states[i].life_left -= drain;
                            self.tel_predator_ticks += 1;
                            self.tel_predator_life += drain;
                        }
                    }
                }
            }

            // 4. BODY BLOCK slots 0-5 (the world half). Slots 6-7 are the body half and
            // are filled in the proprioception loop below, where their values already
            // exist — passing the SAME number keeps ch70/ch71 bit-identical when the
            // thermal model is off, which is what makes this flag's A/B readable.
            //
            // The host normalises and the shader only encodes, so every reference
            // distance here is a host-side dose: changing `--predator-perception` never
            // touches WGSL (`perception/channels.rs`, BODY BLOCK header).
            if body_on {
                let cap = self.station_cfg.capacity.max(1e-6);
                let tank = FUEL_TANK.load(Ordering::Relaxed);
                let pred_ref = PREDATOR_PERCEPT_M.load(Ordering::Relaxed).max(1e-3);
                for i in 0..n {
                    let p = &states[i].phys;
                    let me = pos2(p);
                    let fwd = Vec2::new(p.fwd[0], p.fwd[2]);
                    let lat = Vec2::new(p.lat[0], p.lat[2]);
                    let v = &mut body_vals[i];
                    if let Some(ns) = universe::resources::nearest_charged(&self.stations, cap, me)
                    {
                        v[0] = signed_bearing(self.stations[ns.index].pos - me, fwd, lat);
                        v[1] = 1.0 - (ns.dist_m / STATION_PROX_REF_M).min(1.0);
                        v[2] = ns.charge_frac;
                    }
                    // OwnFuel. With `--fuel` off the tank is 0 and there is no fuel
                    // constraint at all, so the honest constant is "full", not "empty":
                    // 0.0 would tell the brain it is about to die every tick of every
                    // run that does not use fuel.
                    v[3] = if tank > 0.0 {
                        (states[i].fuel as f32 / tank).clamp(0.0, 1.0)
                    } else {
                        1.0
                    };
                    if let Some(pred) = self.predator {
                        let rel = pred.pos - me;
                        v[4] = signed_bearing(rel, fwd, lat);
                        v[5] = 1.0 - (rel.length() / pred_ref).min(1.0);
                    }
                }
            }
        }
        if let Some(t) = t_prep_body {
            cpu_phases.push(("prep.body", t.elapsed().as_secs_f32() * 1000.0));
        }

        // ── Poses retina ──────────────────────────────────────────────────────
        let t_prep_retina_view = prints_this_tick.then(std::time::Instant::now);
        let pose_raw: Vec<PoseRaw> = (0..n)
            .map(|i| {
                let p = &states[i].phys;
                PoseRaw {
                    pos_gaze: [p.pos[0], p.pos[1], p.pos[2], 0.0],
                    dirs: [p.fwd[0], p.fwd[2], p.lat[0], p.lat[2]],
                }
            })
            .collect();
        self.queue
            .write_buffer(&self.buf_poses_retina, 0, bytemuck::cast_slice(&pose_raw));

        // ── Poses percepção ───────────────────────────────────────────────────
        let poses_percept: Vec<PosePerception> = (0..n)
            .map(|i| {
                let p = &states[i].phys;
                PosePerception {
                    pos_x: p.pos[0],
                    pos_y: p.pos[1],
                    pos_z: p.pos[2],
                    phase: 0.0,
                    fwd_x: p.fwd[0],
                    fwd_z: p.fwd[2],
                    rgt_x: p.lat[0],
                    rgt_z: p.lat[2],
                }
            })
            .collect();
        if let Some(t) = t_prep_retina_view {
            cpu_phases.push(("prep.retina_view", t.elapsed().as_secs_f32() * 1000.0));
        }

        // ── Distâncias laterais/frontais (CPU) → buf_temporal[16-18] ─────────
        // 'dists' fica acessível fora do bloco de distâncias para que o bloco de
        // racing_line possa usar dist_l/r/f sem recalcular (conserta ch118/119/123
        // que antes recebiam 0.5/0.8/0.7 hardcoded).
        let t_prep_dists = prints_this_tick.then(std::time::Instant::now);
        let tick_dists: Vec<[f32; 3]> = {
            let gs = self.grid_size_cpu as usize;
            let cs = self.cell_size_cpu;
            // Expandido de 15m → 40m para paridade com perception.wgsl (BUG 1 fix).
            // Conta: 40 m ÷ 28 m/s = 1.43 s de antecipação lateral.
            let lat_samp = [1.0_f32, 3.0, 6.0, 9.0, 12.0, 16.0, 20.0, 26.0, 33.0, 40.0];
            let fwd_samp = [3.0_f32, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0];
            let voxel_at = |wx: f32, wz: f32| -> u32 {
                let xi = (wx / cs).round() as i32;
                let yi = (wz / cs).round() as i32;
                let ng = gs as i32;
                if xi >= 0 && xi < ng && yi >= 0 && yi < ng {
                    self.grid_cpu[(yi as usize) * gs + (xi as usize)]
                } else {
                    2u32
                }
            };
            let mut dists: Vec<[f32; 3]> = Vec::with_capacity(n);
            for i in 0..n {
                let p = &states[i].phys;
                let (px, pz) = (p.pos[0], p.pos[2]);
                let (fx, fz) = (p.fwd[0], p.fwd[2]);
                let (perp_x, perp_z) = (-fz, fx);
                let mut dist_l = 40.0_f32; // default = livre (sem parede em 40m)
                for &sd in &lat_samp {
                    if voxel_at(px - perp_x * sd, pz - perp_z * sd) == 2 {
                        dist_l = sd;
                        break;
                    }
                }
                let mut dist_r = 40.0_f32;
                for &sd in &lat_samp {
                    if voxel_at(px + perp_x * sd, pz + perp_z * sd) == 2 {
                        dist_r = sd;
                        break;
                    }
                }
                let mut dist_f = 50.0_f32;
                for &sd in &fwd_samp {
                    let v = voxel_at(px + fx * sd, pz + fz * sd);
                    if v == 2 || v == 3 {
                        dist_f = sd;
                        break;
                    }
                }
                dists.push([dist_l, dist_r, dist_f]);
            }
            dists
        };
        // Grava distâncias no BeingState para o replay writer do laço externo.
        // Conserto 4 (2026-08-12): os campos dist_wall_l/r/f existiam no ReplayFrame
        // mas chegavam zerados porque a escrita usava ..Default::default(). O diagnóstico
        // de "por que o cérebro não freia" é impossível sem saber o que ele enxergava.
        for i in 0..n {
            states[i].dist_wall_l = tick_dists[i][0];
            states[i].dist_wall_r = tick_dists[i][1];
            states[i].dist_wall_f = tick_dists[i][2];
        }
        self.percept
            .update_dist_temporal(&self.queue, &tick_dists[..n]);
        if let Some(t) = t_prep_dists {
            cpu_phases.push(("prep.dists", t.elapsed().as_secs_f32() * 1000.0));
        }

        // ── Racing line (CPU) → buf_temporal[10-15] ───────────────────────────
        // cpu_fwd_norm / cpu_corridor / cpu_whisker_norm agora são calculados a partir
        // das distâncias reais (tick_dists) em vez de constantes 0.5/0.8/0.7 hardcoded.
        // Isso alimenta ch118 (dist frontal normalizada), ch119 (corredor lateral) e
        // ch123 (whisker = menor lateral) que antes chegavam constantes ao cérebro.
        //
        // rv[i] NÃO é ch=108+i (não é linear): rv[0..3] alimentam ch108,109,110 mas
        // rv[3..6] pulam para ch118,119,123 (perception_temporal.wgsl:271-277).
        // Travado contra o registro único `Channel`.
        const _: () = {
            assert!(
                Channel::LineOffsetRacingLine.index() == 108,
                "rv[0] = off_norm"
            );
            assert!(
                Channel::LineOffsetDeriv.index() == 109,
                "rv[1] = 0.0 (derivada, ainda não implementada)"
            );
            assert!(Channel::LineLookAheadRL.index() == 110, "rv[2] = la_norm");
            assert!(Channel::FwdDistNorm.index() == 118, "rv[3] = cpu_fwd_norm");
            assert!(Channel::CorridorNorm.index() == 119, "rv[4] = cpu_corridor");
            assert!(
                Channel::WhiskerNorm.index() == 123,
                "rv[5] = cpu_whisker_norm"
            );
        };
        let t_prep_racing_line = prints_this_tick.then(std::time::Instant::now);
        {
            let hw = self.racing_line.half_width.max(1.0);
            let mut rv: Vec<[f32; 6]> = Vec::with_capacity(n);
            for i in 0..n {
                let p = &states[i].phys;
                let pos2d = bevy::math::Vec2::new(p.pos[0], p.pos[2]);
                let fwd2d = bevy::math::Vec2::new(p.fwd[0], p.fwd[2]);
                let off_norm =
                    (self.racing_line.lateral_offset_from_line(pos2d, fwd2d) / hw).clamp(-1.0, 1.0);
                let la_norm = (self.racing_line.lookahead_offset(pos2d, 5.0) / hw).clamp(-1.0, 1.0);

                // Normalização idêntica à descrita em perception.rs:700-702:
                //   fwd_norm  = dist_f / (hw * 3.0)  — quanto espaço à frente em relação à pista
                //   corridor  = (dist_l + dist_r) / (hw * 4.0) — largura efetiva do corredor
                //   whisker   = min(dist_l, dist_r) / (hw * 1.5) — lado mais apertado
                let dist_l = tick_dists[i][0];
                let dist_r = tick_dists[i][1];
                let dist_f = tick_dists[i][2];
                let cpu_fwd_norm = (dist_f / (hw * 3.0)).clamp(0.0, 1.0);
                let cpu_corridor = ((dist_l + dist_r) / (hw * 4.0)).clamp(0.0, 1.0);
                let cpu_whisker_norm = (dist_l.min(dist_r) / (hw * 1.5)).clamp(0.0, 1.0);

                // Derivada do offset lateral (cpu_d_off, slot [1]): diferença em relação
                // ao tick anterior guardado em temporal[8] (prev_line_offset_norm).
                // Por ora usa 0.0 — o shader temporal lê prev_line_offset_norm do próprio
                // temporal[8] e poderia calcular internamente, mas o campo de leitura seria
                // o valor do TICK ANTERIOR (correto). O valor 0.0 é conservador e correto
                // para o tick 0; a derivada real emerge a partir do tick 1 via temporal[8].
                rv.push([
                    off_norm,
                    0.0,
                    la_norm,
                    cpu_fwd_norm,
                    cpu_corridor,
                    cpu_whisker_norm,
                ]);
            }
            self.percept
                .update_racing_line_temporal(&self.queue, &rv[..n]);
        }
        if let Some(t) = t_prep_racing_line {
            cpu_phases.push(("prep.racing_line", t.elapsed().as_secs_f32() * 1000.0));
        }

        // ── Propriocepção → buf_temporal[20-31] ───────────────────────────────
        // prop[i] alimenta ch(64+i) no shader (perception_temporal.wgsl:324-349).
        // Travado contra o registro único `Channel` — ver nota em "Extra2" abaixo
        // sobre por que isto pega troca de índice mas não reordenação do push.
        const _: () = {
            assert!(Channel::VelAlign.index() == 64, "prop[0] = align");
            assert!(Channel::SpeedFwd.index() == 65, "prop[1] = speed");
            assert!(Channel::VelLat.index() == 66, "prop[2] = lat_vel");
            assert!(Channel::VidaFrac.index() == 67, "prop[3] = vida_frac");
            assert!(Channel::RpmNorm.index() == 68, "prop[4] = rpm_r");
            assert!(Channel::Gear.index() == 69, "prop[5] = current_gear");
            assert!(Channel::TireTemp.index() == 70, "prop[6] = tire_temp");
            assert!(Channel::TempMotor.index() == 71, "prop[7] = temp_motor");
            assert!(
                Channel::RiscoCinetico.index() == 72,
                "prop[8] = risco_cinetico"
            );
            assert!(Channel::OnGrass.index() == 73, "prop[9] = on_grass");
            assert!(Channel::SteerAbs.index() == 74, "prop[10] = |steer|");
            assert!(Channel::Estagnacao.index() == 75, "prop[11] = estagnado");
        };
        let t_prep_prop = prints_this_tick.then(std::time::Instant::now);
        // C0 lap telemetry: cleared here, so a being that is never reached by one of the
        // three sites below reports the neutral value and not last tick's.
        self.lap_grass.clear();
        self.lap_grass.resize(n, false);
        self.lap_damage.clear();
        self.lap_damage.resize(n, 0.0);
        self.lap_dry.clear();
        self.lap_dry.resize(n, false);
        {
            let gs = self.grid_size_cpu as usize;
            let cs = self.cell_size_cpu;
            let voxel_at_cpu = |wx: f32, wz: f32| -> u32 {
                let xi = (wx / cs).round() as i32;
                let yi = (wz / cs).round() as i32;
                let ng = gs as i32;
                if xi >= 0 && xi < ng && yi >= 0 && yi < ng {
                    self.grid_cpu[(yi as usize) * gs + (xi as usize)]
                } else {
                    2u32
                }
            };
            let mut prop: Vec<[f32; 12]> = Vec::with_capacity(n);
            for i in 0..n {
                let p = &states[i].phys;
                let speed = (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt();
                let vel_dot = p.vel[0] * p.fwd[0] + p.vel[2] * p.fwd[2];
                let align = if speed > 0.1 {
                    (vel_dot / speed).clamp(-1.0, 1.0)
                } else {
                    1.0
                };
                let lat_vel = p.vel[0] * p.lat[0] + p.vel[2] * p.lat[2];
                let rpm_r = if p.max_rpm > 0.0 {
                    (p.current_rpm / p.max_rpm).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let tt: f32 = (0..4).map(|j| p.wheels[j * 2 + 1][0]).sum::<f32>() / 4.0 / 120.0;
                let voxel = voxel_at_cpu(p.pos[0], p.pos[2]);
                let on_grass = if voxel == 0 { 1.0_f32 } else { 0.0 };
                self.lap_grass[i] = voxel == 0; // C0: same voxel, no second lookup
                // ── PROPRIOCEPÇÃO: quatro slots eram CONSTANTES ────────────────
                //
                // `prop[3]=1.0` (HP sempre cheio), `prop[7]=0`, `prop[8]=0`, `prop[11]=0`
                // viravam os canais 67, 71, 72 e 75 cravados — 4 dos 12 proprioceptivos
                // que a auditoria de visão pegou saturados em 127. Canal constante é peso
                // morto em L1 e, pior, é um estado corporal que o cérebro NÃO sente.
                //
                // Isso importa além da limpeza: medimos que neuromodulação só tem teto de
                // ganho quando existe corpo ACOPLADO — estado interno lento que faz o ótimo
                // de comportamento variar. Sem acoplamento o ganho é zero. Os estados abaixo
                // já eram calculados e simplesmente não chegavam a lugar nenhum.
                //
                //   67 ← vida restante   (era HP fixo em 1,0): o ser morre sem saber que vai
                //   71 ← temperatura do motor: acima de 105 °C a potência cai
                //        (physics.wgsl:402) — muda o ótimo de acelerar, τ ≈ 500-1000 ticks
                //   72 ← RISCO CINÉTICO: velocidade × curvatura aparente da retina.
                //        Antes: temperatura do óleo (constante em pista plana, auditada como
                //        canal morto). O ser precisa aprender a FREAR antes da curva — para isso
                //        o sinal de "velocidade alta + curva fechada à frente" tem de chegar
                //        ao cérebro em um canal dedicado. Conserto 3 (2026-08-12).
                //        edge_row6 e edge_row7 medem densidade de borda horizontal linha-a-linha:
                //        a diferença entre elas sobe quando a perspectiva da curva aparece
                //        (borda superior vs inferior divergem). Multiplicado pela velocidade
                //        normalizada: alto = velocidade + curvatura próxima = freiar agora.
                //   75 ← estagnação: quantos ticks sem avançar
                let vida_frac =
                    (states[i].life_left as f32 / life_cap().max(1) as f32).clamp(0.0, 1.0);
                let temp_motor = ((p.engine_temp - 20.0) / 130.0).clamp(0.0, 1.0);
                // Risco cinético = velocidade × curvatura aparente.
                // curv_aparente: diferença de edge density entre linhas 6 e 7 da retina.
                // Quanto maior a diferença, mais a perspectiva da parede está "virando" —
                // sinal de curva à frente. Multiplicado por speed_norm para que seja ZERO
                // em linha reta rápida e ALTO em curva fechada + velocidade alta.
                let curv_aparente = (states[i].edge_row6 - states[i].edge_row7).abs();
                let risco_cinetico = (speed / 30.0).clamp(0.0, 1.0) * curv_aparente;
                let estagnado = (states[i].stagnation_ticks as f32 / 64.0).clamp(0.0, 1.0);
                prop.push([
                    align,
                    speed,
                    lat_vel,
                    vida_frac,
                    rpm_r,
                    p.current_gear as f32,
                    tt.clamp(0.0, 1.0),
                    temp_motor,
                    risco_cinetico,
                    on_grass,
                    p.steer.abs().clamp(0.0, 1.0),
                    estagnado,
                ]);
                // BODY BLOCK slots 6-7 — the body half, filled HERE because the values
                // exist here and a second copy of a formula is a second thing to keep
                // in sync (`src/CLAUDE.md` rule 1).
                //
                // Both are SOURCE MOVES, not new signals: with the thermal model off,
                // slot 6 carries the exact `tt` that prop[6] carries and slot 7 the
                // exact `temp_motor` that prop[7] carries, so ch70/ch71 are
                // bit-identical with the flag on. That is what lets `--interoception`
                // be A/B-ed as "the four genuinely new signals", instead of as "eight
                // channels changed at once".
                if body_on {
                    let v = &mut body_vals[i];
                    v[6] = if self.thermal.is_empty() {
                        tt.clamp(0.0, 1.0)
                    } else {
                        states[i].tire_temp.clamp(0.0, 1.0)
                    };
                    v[7] = temp_motor;
                }
            }
            self.percept
                .update_proprioception_temporal(&self.queue, &prop[..n]);
            // One upload per tick, AFTER every slot is filled. Skipping a tick is not
            // an error: `buf_temporal` is resident, so the shader would silently re-read
            // the previous tick's values — a working-but-stale signal
            // (`gpu/perception.rs::update_body_channels`, contract).
            if body_on {
                self.percept
                    .update_body_channels(&self.queue, &body_vals[..n]);
            }
        }
        if let Some(t) = t_prep_prop {
            cpu_phases.push(("prep.prop", t.elapsed().as_secs_f32() * 1000.0));
        }

        // ── Reverse timer + CfC percepção → buf_temporal[32-48] ──────────────
        {
            let t_prep_extra = prints_this_tick.then(std::time::Instant::now);
            // Deduplicado (E3, REFACTOR_PLAN.md §2.3) para
            // universe::config::iara::reverse_maneuver — ver doc do módulo: a constante
            // local `REVERSE_TRIGGER_DIST=3.0` daqui era MORTA (nunca lida; o gatilho
            // abaixo usava o literal `4.0` hardcoded), achado ao unificar.
            use universe::config::iara::reverse_maneuver as revcfg;

            let gs = self.grid_size_cpu as usize;
            let cs = self.cell_size_cpu;
            let voxel_at = |wx: f32, wz: f32| -> u32 {
                let xi = (wx / cs).round() as i32;
                let yi = (wz / cs).round() as i32;
                let ng = gs as i32;
                if xi >= 0 && xi < ng && yi >= 0 && yi < ng {
                    self.grid_cpu[(yi as usize) * gs + (xi as usize)]
                } else {
                    2u32
                }
            };

            let mut extra: Vec<[f32; 17]> = Vec::with_capacity(n);
            for i in 0..n {
                let p = &states[i].phys;
                let speed = (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt();
                let (fx, fz) = (p.fwd[0], p.fwd[2]);
                let (px, pz) = (p.pos[0], p.pos[2]);
                let (perp_x, perp_z) = (-fz, fx);
                let mut fwd_dist = 30.0_f32;
                for sd in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] {
                    let v = voxel_at(px + fx * sd, pz + fz * sd);
                    if v == 2 || v == 3 {
                        fwd_dist = sd;
                        break;
                    }
                }

                // ── PILOTO SENSORIAL (pilots.rs) ────────────────────────────────
                //
                // Leonardo: *"roda uma bateria onde não tem cérebro, o carro deve sobreviver
                // por sensores, ver distância das paredes em ambos os lados, sempre tentando
                // centralizar no centro da pista e fazer as curvas de forma correta."*
                //
                // O piloto fantasma usa a racing line — um oráculo que o ser NÃO possui,
                // então ele mede o corpo, não a informação. O sensorial usa só raios de
                // parede, que é exatamente o que a retina do ser enxerga: mede o TETO DA
                // INFORMAÇÃO SENSORIAL. Se um controlador de vinte linhas completa a volta
                // com esses sensores, os dados bastam e a culpa é do cérebro.
                if PILOTO_SENSOR.load(Ordering::Relaxed) != 0.0 {
                    let g = PILOTO_SENSOR.load(Ordering::Relaxed);
                    let (steer, thr, brk) = crate::pilots::piloto_sensorial(
                        g, px, pz, fx, fz, perp_x, perp_z, speed, fwd_dist, &voxel_at,
                    );
                    states[i].sensor_steer = steer;
                    states[i].sensor_thr = thr;
                    states[i].sensor_brk = brk;
                }

                // Fan-ray para o piloto fantasma (pilots.rs): 7 direções
                // (frente + ±15°/30°/45°). Calculado aqui onde `voxel_at` está vivo;
                // consumido no bloco do piloto. A curva aparece como parede diagonal
                // ANTES de aparecer frontalmente — o fan detecta a curva antes do frontal
                // puro.
                if PILOTO_GANHO.load(Ordering::Relaxed) != 0.0 {
                    states[i].phantom_d_min =
                        crate::pilots::fan_ray_d_min(px, pz, fx, fz, &voxel_at);
                }

                if states[i].post_reverse_lockout > 0.0 {
                    states[i].post_reverse_lockout = (states[i].post_reverse_lockout - dt).max(0.0);
                }

                // Trigger de ré: parede frontal <4m E speed baixa E estagnado E sem lockout.
                {
                    let rev = states[i].reverse_timer;
                    let stag = states[i].stagnation_ticks;
                    let lockout = states[i].post_reverse_lockout;
                    if rev <= 0.0
                        && lockout <= 0.0
                        && fwd_dist < revcfg::TRIGGER_DIST_M
                        && speed < revcfg::TRIGGER_SPEED_MAX_MS
                        && stag > revcfg::STAGNATION_TICKS
                    {
                        // Anti-encalhe ESCALONADO (ver doc em types.rs::re_falhas):
                        // re-encalhe em <5 s = a ré anterior não escapou → dobra a
                        // duração (até 3×) e o volante alterna de lado durante a ré.
                        if states[i].re_desde_saida < 5.0 {
                            states[i].re_falhas = states[i].re_falhas.saturating_add(1);
                        } else {
                            states[i].re_falhas = 0;
                        }
                        let escala = 1.0 + (states[i].re_falhas.min(4) as f32) * 0.75;
                        states[i].reverse_timer = revcfg::MIN_DURATION_S * escala;
                        states[i].stagnation_ticks = 0;
                    }
                    states[i].re_desde_saida += dt;
                }
                // Decremento e saída de ré.
                {
                    let rev = states[i].reverse_timer;
                    if rev > 0.0 {
                        let nr = rev - dt;
                        if fwd_dist > revcfg::EXIT_CLEAR_M || nr > revcfg::MAX_DURATION_S {
                            states[i].reverse_timer = 0.0;
                            states[i].re_desde_saida = 0.0;
                            states[i].post_reverse_lockout = revcfg::POST_LOCKOUT_S;
                        } else {
                            states[i].reverse_timer = nr.max(0.0);
                        }
                    }
                }

                // val[1..17] (cfc_hidden) fica em ZERO deliberadamente: ch192-207 agora
                // são escritos diretamente por `retina_digest.wgsl` (que roda no mesmo
                // encoder, lendo o estado do CfC direto da VRAM — sem passar pela CPU).
                // Este slot de `update_extra_temporal` continua existindo só porque a
                // função é COMPARTILHADA com gpu_loop.rs/gpu_report.rs (que ainda rodam
                // o CfC na CPU); em gpu_evolve, só o slot [0] (reverse_timer) é real.
                let mut val = [0.0f32; 17];
                val[0] = states[i].reverse_timer;
                extra.push(val);
            }
            self.percept.update_extra_temporal(&self.queue, &extra[..n]);
            if let Some(t) = t_prep_extra {
                cpu_phases.push(("prep.extra", t.elapsed().as_secs_f32() * 1000.0));
            }

            // ── Extra2: slip, dopamina, dor, stress, edge (slots 49-57) ───────
            //
            // Canais reprovados pela auditoria que têm dado disponível no host:
            //   ch87  = slip angle (cross2D fwd × vel_norm) — calculado aqui da física
            //   ch124 = dor de rodopio (|av_y|>1.5 && speed<2.0) — física
            //   ch125 = estresse de proximidade O(n²) — posições dos seres
            //   ch126 = joy_tick — calculado no odômetro, tick anterior (1 tick de atraso)
            //   ch127 = joy_streak — idem
            //   ch99  = edge density linha 6 — calculado no readback anterior (1 tick atraso)
            //   ch100 = edge density linha 7 — idem
            //   ch121 = steer_mem (phys.steer) — física
            //   ch122 = brake_mem (phys.brake_cmd) — física
            //
            // Os literais acima são travados em tempo de compilação contra o registro
            // único `Channel` (src/perception/channels.rs) — a primeira versão desse
            // registro tinha ch124-127 TROCADOS (Spinning/CompStress/JoyTick/JoyStreak
            // em ordem errada) e só foi pega porque a ordem aqui embaixo (extra2.push)
            // é a fonte real. Se alguém reordenar `extra2.push([...])` sem atualizar os
            // comentários `[N] → chXXX`, esses `assert!` continuam vendo o Channel certo
            // (eles não leem a ordem do array, só travam os literais desta lista de
            // comentários contra o enum) — a defesa real contra reordenar o push está
            // no teste de runtime `--auditar-visao`, que mede variância por índice.
            const _: () = {
                assert!(Channel::SlipAngle.index() == 87, "ch87 = slip_angle");
                assert!(Channel::JoyTick.index() == 126, "ch126 = joy_tick");
                assert!(Channel::JoyStreak.index() == 127, "ch127 = joy_streak");
                assert!(
                    Channel::Spinning.index() == 124,
                    "ch124 = is_spinning (dor de rodopio)"
                );
                assert!(
                    Channel::CompStress.index() == 125,
                    "ch125 = comp_stress (proximidade)"
                );
                assert!(Channel::EdgeRow6.index() == 99, "ch99  = edge_row6");
                assert!(Channel::EdgeRow7.index() == 100, "ch100 = edge_row7");
                assert!(Channel::SteerMem.index() == 121, "ch121 = steer_mem");
                assert!(Channel::BrakeMem.index() == 122, "ch122 = brake_mem");
            };
            let t_prep_extra2 = prints_this_tick.then(std::time::Instant::now);
            {
                // Posições para comp_stress O(n²): 2304 pares para n=48, irrelevante.
                let positions: Vec<[f32; 2]> = (0..n)
                    .map(|i| [states[i].phys.pos[0], states[i].phys.pos[2]])
                    .collect();

                let mut extra2: Vec<[f32; 9]> = Vec::with_capacity(n);
                for i in 0..n {
                    let p = &states[i].phys;
                    let speed = (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt();

                    // slip_signal: cross product 2D entre forward e vel normalize.
                    // Mesma fórmula de motor.rs:1607-1613. Positivo = drift à direita.
                    let slip_signal = if speed > 1.0 {
                        let inv_s = 1.0 / speed;
                        let vx = p.vel[0] * inv_s;
                        let vz = p.vel[2] * inv_s;
                        // f × v: fwd_x*vel_z - fwd_z*vel_x
                        p.fwd[0] * vz - p.fwd[2] * vx
                    } else {
                        0.0
                    };

                    // is_spinning: |av_y| > 1.5 rad/s E speed < 2.0 m/s (motor.rs:1383).
                    let is_spinning = if p.av_y.abs() > 1.5 && speed < 2.0 {
                        1.0f32
                    } else {
                        0.0
                    };

                    // comp_stress O(n²): distância ao ser mais próximo em raio 20 m à frente.
                    // motor.rs:1359-1378: 0=líder (ou ninguém perto), 0.5=meio pelotão, 1.0=alguém <20m.
                    //
                    // BUG encontrado e corrigido (2026-08-14, FASE CANAIS): a versão anterior
                    // usava `if i > 0 {0.5} else {0.0}` como piso — um artefato do ÍNDICE do
                    // ser (o ser 0 sempre partia de 0.0, todos os outros de 0.5), sem relação
                    // com proximidade real. Como a auditoria de visão rastreia UM ser vivo ao
                    // longo do tempo (tipicamente o de índice mais baixo), esse ser lia
                    // comp_stress preso em {0.0, 1.0} — nunca 0.5 — e o canal parecia
                    // CONSTANTE sempre que ninguém chegava a <20m dele (o caso comum numa
                    // população espalhada), mesmo com o dado corretamente "alimentado". Agora
                    // o piso 0.5 depende de proximidade FÍSICA (alguém em até 40m, à frente ou
                    // não) — igual para todos os seres, sem depender de posição no array.
                    let (px, pz) = (positions[i][0], positions[i][1]);
                    let (fx, fz) = (p.fwd[0], p.fwd[2]);
                    let mut has_near_ahead = false;
                    let mut min_dist_any: f32 = f32::MAX;
                    for j in 0..n {
                        if j == i {
                            continue;
                        }
                        let dx = positions[j][0] - px;
                        let dz = positions[j][1] - pz;
                        let dist = (dx * dx + dz * dz).sqrt();
                        if dist < min_dist_any {
                            min_dist_any = dist;
                        }
                        if dist < 20.0 {
                            let dot = (dx * fx + dz * fz) / dist.max(0.01);
                            if dot > 0.3 {
                                has_near_ahead = true;
                            }
                        }
                    }
                    // 0.0=ninguém em 40m, 0.5=alguém no pelotão (<40m, qualquer direção),
                    // 1.0=alguém <20m à frente no cone de visão.
                    let comp_stress = if has_near_ahead {
                        1.0f32
                    } else if min_dist_any < 40.0 {
                        0.5
                    } else {
                        0.0
                    };

                    extra2.push([
                        slip_signal.clamp(-1.0, 1.0), // [0] → ch87
                        states[i].joy_tick,           // [1] → ch126
                        states[i].joy_streak / 5.0,   // [2] → ch127 (normalizado)
                        is_spinning,                  // [3] → ch124
                        comp_stress,                  // [4] → ch125
                        states[i].edge_row6,          // [5] → ch99 (tick anterior)
                        states[i].edge_row7,          // [6] → ch100 (tick anterior)
                        p.steer.clamp(-1.0, 1.0),     // [7] → ch121
                        p.brake_cmd.clamp(0.0, 1.0),  // [8] → ch122
                    ]);
                }
                self.percept
                    .update_extra2_temporal(&self.queue, &extra2[..n]);
                if let Some(t) = t_prep_extra2 {
                    cpu_phases.push(("prep.extra2", t.elapsed().as_secs_f32() * 1000.0));
                }
            }
        }

        // ── Encoder único ─────────────────────────────────────────────────────
        // `prep.encoder_setup`: not a per-being pass -- it is the remainder of the
        // prep region (command encoder + retina uniform buffer creation) that runs
        // AFTER the last per-being loop and BEFORE `rec.cpu_prep` closes. Without it
        // the `prep.*` rows would fall short of `rec.cpu_prep` by exactly this much,
        // which is the unmeasured-remainder failure mode this task exists to close.
        let t_prep_encoder_setup = prints_this_tick.then(std::time::Instant::now);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("evolve-tick"),
            });
        if prints_this_tick {
            if let Some(t) = self.timer.as_mut() {
                t.begin_frame();
            }
        }

        // Retina.
        let mut pitches_arr = [[0.0f32; 4]; 4];
        for row in 0..RETINA_ROWS {
            pitches_arr[row / 4][row % 4] = pitch(row);
        }
        let retina_params = RetinaParamsGpu {
            grid_size: self.grid_size,
            n_beings: n as u32,
            cell_size: self.cell_size,
            fov_half: (universe::visao::fov_deg() / 2.0).to_radians(),
            pitches: pitches_arr,
        };
        let buf_rp = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("evolve-rp"),
                contents: bytemuck::bytes_of(&retina_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        if let Some(t) = t_prep_encoder_setup {
            cpu_phases.push(("prep.encoder_setup", t.elapsed().as_secs_f32() * 1000.0));
        }
        if let (Some(t0), true) = (tick_t0, prints_this_tick) {
            // Everything before the first GPU call: per-being Rust loops building the
            // sensory extras, plus their uploads. Split out from `cpu.record` because
            // "the CPU spends 7 ms recording commands" and "the CPU spends 7 ms
            // computing per-being inputs" call for opposite fixes.
            cpu_phases.push(("rec.cpu_prep", t0.elapsed().as_secs_f32() * 1000.0));
        }
        let t_rec_gpu = prints_this_tick.then(std::time::Instant::now);
        let bg_retina = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout_retina,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_rp.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buf_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_poses_retina.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buf_retina_out.as_entire_binding(),
                },
            ],
        });

        // Bracketed with `mark_begin`/`mark_end` rather than `scope`: `write_timestamp`
        // is called from here, where `&mut enc` is already in hand, instead of inside
        // `begin_compute_pass` — see `timing.rs` for why (this label also folds in
        // `copy_retina_prev`'s buffer copy, which runs just before the pass).
        let t_retina = if prints_this_tick {
            self.timer
                .as_mut()
                .and_then(|t| t.mark_begin(&mut enc, "retina"))
        } else {
            None
        };
        self.percept
            .copy_retina_prev(&mut enc, &self.buf_retina_out);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("evolve-retina-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_retina);
            pass.set_bind_group(0, &bg_retina, &[]);
            let total = (n * RETINA_TOTAL) as u32;
            pass.dispatch_workgroups(total.div_ceil(64), 1, 1);
        }
        if let Some(t) = self.timer.as_mut() {
            t.mark_end(&mut enc, t_retina);
        }

        // Percepção.
        let t_percept = if prints_this_tick {
            self.timer
                .as_mut()
                .and_then(|t| t.mark_begin(&mut enc, "perception"))
        } else {
            None
        };
        self.percept.bind_and_dispatch(
            &mut enc,
            &self.device,
            &self.queue,
            &poses_percept[..n],
            &self.buf_retina_out,
        );
        if let Some(t) = self.timer.as_mut() {
            t.mark_end(&mut enc, t_percept);
        }

        // CfC de percepção + min_wall_ahead + edge_row6/7 — TUDO na GPU agora.
        //
        // Roda DEPOIS de `percept.bind_and_dispatch` (que zera `buf_output` antes de
        // escrever os canais espaciais/temporais) porque este kernel escreve ch192-207
        // no MESMO `buf_output` via atomicOr — escrever antes do clear perderia esses
        // canais. Roda ANTES do cérebro, para que `brain.bind_and_dispatch` já veja
        // ch192-207 preenchidos. Substitui o antigo readback de 340 KB/tick (retina
        // inteira) + laço Rust — ver `gpu::retina_digest` para o antes/depois completo.
        let t_digest = if prints_this_tick {
            self.timer
                .as_mut()
                .and_then(|t| t.mark_begin(&mut enc, "retina_digest"))
        } else {
            None
        };
        let extra_bytes = self.retina_digest.bind_and_dispatch(
            &mut enc,
            &self.device,
            &self.queue,
            n,
            &self.buf_retina_out,
            self.percept.buf_output(),
        );
        if let Some(t) = self.timer.as_mut() {
            t.mark_end(&mut enc, t_digest);
        }

        // ── FLY MAP (docs/FLY-MAP-BRAIN.md §4) ─────────────────────────────────
        //
        // Runs HERE for the same reason the slow brain's gather does: the sensory wave it
        // reads (`percept.buf_output()`) is only complete after `perception` and
        // `retina_digest` have written it, and it has to be read before this encoder is
        // submitted. It neither reads nor writes anything the fast brain touches, so the
        // two are independent within the encoder; the order between them is not
        // load-bearing, unlike the three orderings listed in `src/gpu/CLAUDE.md`.
        //
        // With `--fly-map` absent this is `None` and nothing at all is recorded.
        if self.fly.is_some() || self.fly_spiking.is_some() {
            let t_fly = if prints_this_tick {
                self.timer
                    .as_mut()
                    .and_then(|t| t.mark_begin(&mut enc, "fly_map"))
            } else {
                None
            };
            if let Some(fly) = self.fly.as_ref() {
                fly.dispatch(
                    &mut enc,
                    n,
                    self.percept.buf_output(),
                    universe::fly_map::eye_on(),
                    &self.buf_retina_out,
                );
            }
            if let Some(fly) = self.fly_spiking.as_ref() {
                // The retina is already resident in `buf_retina_out`; K1/K2/K3 remain
                // in this encoder so no CPU round trip occurs between perception and
                // the spiking readout. Motion mode is available to callers that bind
                // temporal contrast; the CLI currently selects photoreceptor mode.
                if trace_sample {
                    fly.set_debug_being(self.spiking_trace_being);
                }
                fly.dispatch(
                    &mut enc,
                    n,
                    &self.buf_retina_out,
                    universe::fly_map::eye_on(),
                    EyeMode::Photoreceptor,
                    dt,
                    0,
                );
                if trace_sample {
                    fly.record_debug_snapshot(&mut enc, n);
                }
            }
            if let Some(t) = self.timer.as_mut() {
                t.mark_end(&mut enc, t_fly);
            }
        }

        // Cérebro rápido — só dispara a cada `brain_skip` ticks.
        //
        // A percepção e a física continuam a 64 Hz: o que muda é a frequência de DECISÃO,
        // não a do mundo. Nos ticks pulados o ser age com a última intenção, exatamente
        // como um organismo cuja via motora é mais lenta que a sensorial.
        self.brain_tick = self.brain_tick.wrapping_add(1);
        // Dispara também quando ainda NÃO existe intenção anterior utilizável.
        //
        // Sem esta condição, com skip>1 os primeiros ticks devolviam `intents` vazio, a
        // guarda `intents.len() != n` abortava o tick e a geração inteira morria: medido
        // com skip=4 → 0 m, 100% de mortalidade, 0,0 s. Sem a guarda, teria sido pior —
        // o número sairia bonitinho e eu concluiria que "skip alto faz o modelo morrer",
        // quando o que morre é a execução.
        let brain_fires = self.brain_skip <= 1
            || self.last_intents.len() != n
            || (self.brain_tick % self.brain_skip) == 0;
        if brain_fires {
            // No bracket recorded when `brain_fires` is false — an absent label reads
            // as "did not fire this tick" instead of a misleading 0.000 ms.
            let t_brain = if prints_this_tick {
                self.timer
                    .as_mut()
                    .and_then(|t| t.mark_begin(&mut enc, "brain"))
            } else {
                None
            };
            self.brain
                .bind_and_dispatch(&mut enc, n, &[], dt, Some(self.percept.buf_output()));
            if let Some(t) = self.timer.as_mut() {
                t.mark_end(&mut enc, t_brain);
            }
        }

        // Cérebro lento — colhe 48 canais da onda que já está na GPU e roda.
        // Encadeado no MESMO encoder: nenhum readback intermediário, nenhuma volta à CPU.
        let t_slow_gather = if prints_this_tick {
            self.timer
                .as_mut()
                .and_then(|t| t.mark_begin(&mut enc, "slow.gather"))
        } else {
            None
        };
        self.slow
            .dispatch_gather(&mut enc, self.percept.buf_output(), n);
        if let Some(t) = self.timer.as_mut() {
            t.mark_end(&mut enc, t_slow_gather);
        }
        let t_slow = if prints_this_tick {
            self.timer
                .as_mut()
                .and_then(|t| t.mark_begin(&mut enc, "slow"))
        } else {
            None
        };
        self.slow.dispatch(&mut enc, n, dt);
        if let Some(t) = self.timer.as_mut() {
            t.mark_end(&mut enc, t_slow);
        }

        // Resolve must be on the same encoder as the marks above (`timing.rs`), so it
        // happens right before this encoder's own submit -- gated on `prints_this_tick`
        // like every other instrumentation call above, so a non-printing tick neither
        // resolves stale slots left over from the last printing tick nor pays for the
        // `resolve_query_set` + buffer copy at all.
        if prints_this_tick {
            if let Some(t) = self.timer.as_mut() {
                t.resolve(&mut enc);
            }
        }

        // Submit único.
        if let Some(t) = t_rec_gpu {
            cpu_phases.push(("rec.gpu_record", t.elapsed().as_secs_f32() * 1000.0));
        }
        if let (Some(t0), true) = (tick_t0, prints_this_tick) {
            // Everything before the submit: write_buffer uploads, bind-group creation and
            // command recording. Measured 2026-08-27 because `gaps` had 5.8 ms nobody
            // could name, and an unnamed 23% of the tick is not a diagnosis.
            cpu_phases.push(("cpu.record", t0.elapsed().as_secs_f32() * 1000.0));
        }
        self.queue.submit(Some(enc.finish()));
        let t_poll = prints_this_tick.then(std::time::Instant::now);
        self.device.poll(wgpu::Maintain::Wait);
        if let Some(t) = t_poll {
            cpu_phases.push(("cpu.poll", t.elapsed().as_secs_f32() * 1000.0));
        }

        // `--kernel-profile <N>`: print the GPU-kernel table every N ticks. This covers
        // everything recorded into the single encoder above (retina through slow);
        // `physics.step` is NOT here — it opens its own encoder and does its own
        // synchronous submit+readback later in this function, so no timestamp written
        // from this encoder can reach inside it. It is measured separately, on the
        // CPU, right around that call — see the `[perfil] physics.step` print below.
        if prints_this_tick && self.timer.is_none() {
            // `--kernel-profile` was asked for but this device never got a `GpuTimer`
            // (no `Features::TIMESTAMP_QUERY` at construction, see `GpuCtx::new`) — say
            // so explicitly instead of printing nothing, which would look identical to
            // the flag being off.
            println!("  [perfil] no GpuTimer on this device (TIMESTAMP_QUERY not granted)");
        }
        if let Some(t) = self.timer.as_ref() {
            if prints_this_tick {
                let period = self.queue.get_timestamp_period();
                let rows = t.read_ms(&self.device, period);
                if rows.is_empty() {
                    // An empty `Vec` here can mean two different things, and printing
                    // "0.000 ms" for either would hide a real problem (the failure mode
                    // the task brief called out explicitly). Which one it is is NOT
                    // decidable from this branch alone -- `read_ms` already `eprintln!`s
                    // the mapping-failure case itself, so check stderr: a line there means
                    // the buffer map failed (device lost, already mapped, driver hiccup);
                    // no line there means `can_mark_in_encoder` is false (device granted
                    // `TIMESTAMP_QUERY` but not `TIMESTAMP_QUERY_INSIDE_ENCODERS`), so
                    // every `mark_begin` this tick returned `None` and nothing was ever
                    // pushed into `labels`.
                    println!("  [perfil] no GPU-timed labels this tick (see stderr for why)");
                } else {
                    let soma: f32 = rows.iter().map(|(_, ms)| ms).sum();
                    for (label, ms) in &rows {
                        println!("  [perfil] {label:<22} {ms:8.4} ms");
                    }
                    println!(
                        "  [perfil] {:<22} {soma:8.4} ms (sum of GPU kernels)",
                        "TOTAL"
                    );
                    total_gpu_ms = soma;
                }
            }
        }

        // Leitura do intent.
        // ── AUDITORIA DA VISÃO ────────────────────────────────────────────────
        // Acumula dados ao longo de TODA a corrida (um tick por chamada) para
        // poder medir variância temporal por canal. Reporta ao final da corrida.
        if AUDITAR.load(Ordering::Relaxed) {
            use universe::gpu::perception::WAVE_U32S;
            let bytes = (n * WAVE_U32S * 4) as u64;
            let ondas = self.percept.read_after_submit(&self.device, n, bytes);
            if ondas.len() == n {
                // Escolhe o primeiro ser VIVO para rastrear temporalmente.
                // Rastrear ser[0] fixo produz σ_temporal=0 se ele morrer cedo.
                let ser_vivo = states[..n]
                    .iter()
                    .position(|s| s.life_left > 0.0)
                    .unwrap_or(0);
                let n_vivos = states[..n].iter().filter(|s| s.life_left > 0.0).count();
                auditoria_acumular(&ondas, n, ser_vivo, n_vivos);
            }
        }

        let t_ri = prints_this_tick.then(std::time::Instant::now);
        let mut intents = if brain_fires {
            let v = self.brain.read_intent_after_submit(n);
            if v.len() == n {
                self.last_intents = v.clone();
            }
            v
        } else {
            // Reusa a decisão anterior. Se ainda não houve nenhuma (primeiro tick com
            // skip > 1), cai em zero — que é "não faz nada", não um comando inventado.
            self.last_intents.clone()
        };
        if let Some(t) = t_ri {
            cpu_phases.push(("cpu.read_intent", t.elapsed().as_secs_f32() * 1000.0));
        }
        let t_rs = prints_this_tick.then(std::time::Instant::now);
        let slow_out = self.slow.read_outputs(n);
        if let Some(t) = t_rs {
            cpu_phases.push(("cpu.read_slow", t.elapsed().as_secs_f32() * 1000.0));
        }
        // FLY MAP: 3 floats per being, read with the same `map_async` → `poll` → `recv`
        // discipline as the intent above. Empty vector when the map is off, and an empty
        // vector is what the override below tests before touching anything.
        let t_rf = prints_this_tick.then(std::time::Instant::now);
        let fly_cmd = if let Some(fly) = self.fly_spiking.as_ref() {
            let commands = fly.read_commands_after_submit(n);
            if trace_sample {
                if let Some(mut frame) = fly.read_debug_snapshot_after_submit() {
                    frame.commands = commands
                        .get(self.spiking_trace_being)
                        .copied()
                        .unwrap_or([0.0; universe::fly_map::COMMANDS]);
                    if let Some(writer) = self.spiking_trace_writer.as_mut() {
                        if let Err(e) =
                            universe::fly_map::spiking::write_trace_frame(writer, &frame)
                        {
                            eprintln!("[fly-spiking] trace write failed: {e}");
                            self.spiking_trace_writer = None;
                        }
                    }
                } else {
                    eprintln!("[fly-spiking] trace snapshot readback failed; disabling trace");
                    self.spiking_trace_writer = None;
                }
            }
            commands
        } else if let Some(fly) = self.fly.as_ref() {
            fly.read_commands_after_submit(n)
        } else {
            Vec::new()
        };
        if let Some(t) = t_rf {
            cpu_phases.push(("cpu.read_fly", t.elapsed().as_secs_f32() * 1000.0));
        }
        // ── M4: MAESTRO DE REGIÕES (ver REGIOES_CEREBRAIS.md, spec M4) ─────────
        // gate_b = 1 + 0,5·tanh(w·slow_out + b), atenuado pela RESERVA da região;
        // usar o gate longe de 1,0 GASTA reserva (burnout), que repõe com τ=30 s.
        if NEUROMOD_REGIOES.load(Ordering::Relaxed)
            && !self.gates_w.is_empty()
            && slow_out.len() == n
        {
            let dt_nm = 1.0f32 / 64.0;
            let mut flat: Vec<f32> = Vec::with_capacity(n * 4);
            for i in 0..n {
                let so = slow_out[i];
                let w = &self.gates_w[i * 16..(i + 1) * 16];
                for b in 0..4 {
                    let z = w[b * 4] * so[0]
                        + w[b * 4 + 1] * so[1]
                        + w[b * 4 + 2] * so[2]
                        + w[b * 4 + 3];
                    let g_raw = 1.0 + 0.5 * z.tanh();
                    let r = &mut states[i].nm_reserva[b];
                    let uso = (g_raw - 1.0).abs();
                    *r = (*r + ((1.0 - *r) / 30.0 - 0.5 * uso * *r) * dt_nm).clamp(0.0, 1.0);
                    flat.push(1.0 + (g_raw - 1.0) * *r);
                }
            }
            self.ultimo_gate = [flat[0], flat[1], flat[2], flat[3]];
            self.brain.upload_gates(&flat, n);
        }
        // Falha silenciosa é o risco aqui: se `read_outputs` devolver vazio, o consumo
        // cai em `unwrap_or([0;3])` e o lento fica inerte sem nenhum sinal de erro.
        {
            use std::sync::atomic::{AtomicBool, Ordering};
            static AVISADO: AtomicBool = AtomicBool::new(false);
            if slow_out.len() != n && !AVISADO.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "[lento] AVISO: read_outputs devolveu {} de {} seres — órgão inerte",
                    slow_out.len(),
                    n
                );
            }
        }
        // Amostra esparsa (o lento só recomputa a cada 8 ticks; amostrar mais é redundância
        // que infla artificialmente a razão de participação).
        if !slow_out.is_empty() && self.slow_samples.len() < 20_000 {
            for o in slow_out.iter().take(8) {
                self.slow_samples.push(vec![o[0], o[1], o[2]]);
            }
        }

        // ── Vida/parede/estagnação/combustível — dados de `retina_digest.wgsl` ────
        //
        // O CfC de percepção (ch192-207), `min_wall_ahead` e `edge_row6/7` agora são
        // calculados NA GPU (retina_digest.wgsl, dispatched acima, no mesmo encoder).
        // Aqui só descem 3 floats/ser (min_wall_ahead, edge_row6, edge_row7) em vez dos
        // 1764 pixels/ser que eram lidos e reprocessados em Rust — a equação em si
        // (legacy_encoding, col-pool, zone-pool, CfC) não muda, só migra de host para
        // kernel (ver `gpu::retina_digest` e `math::cfc_perception`).
        //
        // O laço de vida abaixo (wall_ticks/dreno, estagnação, combustível, -1 tick de
        // life_left) CONTINUA na CPU — só passa a consumir `min_wall_ahead` já pronto
        // em vez de recalculá-lo dos pixels.
        let mut t_dl_outer = None;
        {
            let t_rd = prints_this_tick.then(std::time::Instant::now);
            let extra = self
                .retina_digest
                .read_after_submit(&self.device, n, extra_bytes);
            if let Some(t) = t_rd {
                cpu_phases.push(("cpu.read_digest", t.elapsed().as_secs_f32() * 1000.0));
            }
            t_dl_outer = prints_this_tick.then(std::time::Instant::now);
            for i in 0..n {
                let min_wall_ahead = extra[i][0];
                // Um tick "colado" é um tick com parede a menos de WALL_NEAR_M à frente.
                // O limiar é maior que o carro (≈2 m) para pegar o encosto, não o roçar.
                // Só conta para quem ainda está vivo. Um ser sem vida congela com freio,
                // muitas vezes ENCOSTADO na parede, e continuaria contando como "colado"
                // até o fim da geração — a métrica mediria o próprio mecanismo de morte
                // em vez do comportamento. Foi o que inverteu o sinal na primeira medição
                // (dreno maior aparecia com MAIS tempo colado).
                if min_wall_ahead < WALL_NEAR_M && states[i].life_left > 0.0 {
                    states[i].wall_ticks += 1;
                    // Custo em TEMPO: cada tick colado consome vida extra. Evitar a
                    // parede compra oportunidade de andar, em vez de disputar com ela.
                    states[i].life_left -= WALL_LIFE_DRAIN.load(Ordering::Relaxed) as f64;
                }
                // FÍSICA 1 — custo de RPM alto (ver doc de CUSTO_RPM em statics.rs):
                // acima de 90% do giro, cada tick queima vida proporcional ao excesso.
                // Dá à marcha um MOTIVO além da velocidade: girar menos = viver mais.
                let custo_rpm = CUSTO_RPM.load(Ordering::Relaxed);
                if custo_rpm > 0.0 && states[i].life_left > 0.0 {
                    let rpm_frac = states[i].phys.current_rpm / states[i].phys.max_rpm.max(1.0);
                    if rpm_frac > 0.9 {
                        let excesso = ((rpm_frac - 0.9) / 0.1).min(1.5);
                        states[i].life_left -= (custo_rpm * excesso) as f64;
                    }
                }
                // E-REG1 — custo de fiação (ver CUSTO_FIACAO em statics.rs): dreno
                // constante por tick proporcional ao fio "longo" do genoma do ser.
                if !self.fiacao_custo.is_empty() && states[i].life_left > 0.0 {
                    states[i].life_left -=
                        self.fiacao_custo[i.min(self.fiacao_custo.len() - 1)] as f64;
                }
                // D8 — sensory subscription (`--sensor-cost`, see statics.rs).
                //
                // Same shape and same site as the wiring cost above, and deliberately
                // so: both are per-being constants for the whole generation because the
                // weights are, both are paid in LIFE (crossed currency — perception
                // protects the fitness, so it must not be charged in fitness), and both
                // are recomputed by `main()` where the genomes live. What differs is the
                // statistic: this one is a per-FAMILY sum, so pruning one family removes
                // a whole term instead of nudging a mean over 18 432 weights whose
                // inter-individual variance the CLT drives to zero (E-REG1's negative).
                if !self.sensor_cost.is_empty() && states[i].life_left > 0.0 {
                    states[i].life_left -=
                        self.sensor_cost[i.min(self.sensor_cost.len() - 1)] as f64;
                }
                // Custo de estagnação: parar sem avançar QUEIMA VIDA (Conserto 1, 2026-08-12).
                // Sintoma medido: ser para em 272 m com life=3010 ticks e fica ESTÁTICO até
                // hp=0 (tick 4020), sem tentar sair. Com life-earn, parar é de graça porque a
                // vida acumulada ao andar é gasta só 1/tick — o ser tem 47 s de "folga". Esse
                // dreno torna a paralisia fatal e força o cérebro a continuar tentando.
                // Threshold 128 ticks = 2 s: evita penalizar quem apenas reduz speed na curva
                // (esses têm stagnation=0 pois speed≥1.5 m/s). Análogo ao wall_drain.
                if states[i].stagnation_ticks > 128 && states[i].life_left > 0.0 {
                    states[i].life_left -= STAG_LIFE_DRAIN.load(Ordering::Relaxed) as f64;
                }
                // Combustível: consumo proporcional ao acelerador ao quadrado, como nas
                // branches que davam voltas. Só morde quando ligado (`--fuel`).
                if FUEL_ON.load(Ordering::Relaxed) && states[i].fuel > 0.0 {
                    let thr = states[i].phys.throttle.abs() as f64;
                    states[i].fuel -= (0.002 + thr * thr * 0.004) / 64.0;
                    if states[i].fuel <= 0.0 {
                        states[i].life_left = 0.0;
                        // `[recurso]` telemetry: tank-empty CROSSINGS, not ticks spent
                        // empty (the `fuel > 0.0` guard above is what makes it a
                        // crossing). In practice that is once per being per generation,
                        // because running dry sets `life_left = 0` and a dead being
                        // neither reaches a checkpoint (sector refill) nor drains a
                        // station (`alive: false` in the `BeingFuel` built above). If a
                        // future refill path breaks that, the label has to change too.
                        self.tel_dry_beings += 1;
                        self.lap_dry[i] = true; // C0: same crossing, per lap
                    }
                }
                if states[i].life_left > 0.0 {
                    states[i].life_left -= 1.0;
                    states[i].ticks_alive += 1;
                }
                // Edge density das linhas 6 e 7 da retina → ch99, ch100.
                // Guardado nos states para ser entregue ao temporal NO PRÓXIMO tick (1 tick
                // de atraso) — mesmo contrato de antes, só que o valor já vem pronto.
                states[i].edge_row6 = extra[i][1];
                states[i].edge_row7 = extra[i][2];
            }
        }

        if intents.len() != n {
            return false;
        }

        // ── PILOTO POR FÓRMULA ────────────────────────────────────────────────
        // Substitui a DECISÃO do ser 0 mantendo percepção+KAN+CfC vivos. Lê pelo
        // mesmo canal do `--observar` (download_debug) → X idêntico ao do solve.
        //
        // ATENÇÃO À POSIÇÃO — este bloco TEM de vir antes da montagem de
        // `phys_states`. Ele estava DEPOIS (logo após o `.collect()`), e nessa
        // ordem o override não chegava ao motor: `phys_states` já tinha sido
        // montado com o intent do CÉREBRO, e a única coisa que a fórmula ainda
        // tocava era `mem0` (canal de percepção 120), no fim do tick. Resultado:
        // as corridas de 2026-08-16 mediram o campeão com UM canal de memória
        // contaminado, não o clone dirigindo — os comandos aplicados eram
        // bit-a-bit iguais aos do campeão nos primeiros 576 ticks enquanto o
        // intent gravado já diferia em 0,60 desde o tick 0. Achado por auditoria
        // adversarial; o teste `piloto_entra_antes_da_fisica` trava a ordem.
        if self.piloto_forja.is_some() && brain_fires && !intents.is_empty() {
            let (_l1_raw, cfc_in, kan_in) = self.brain.download_debug();
            let mut x = [0.0f32; 192];
            x[..96].copy_from_slice(&cfc_in[..96]);
            let n_kan = kan_in.len().min(96);
            x[96..96 + n_kan].copy_from_slice(&kan_in[..n_kan]);
            if let Some(pf) = self.piloto_forja.as_mut() {
                if let Some(y) = pf.agir_com_portao(&x) {
                    let mut v = intents[0];
                    v[0] = y[0];
                    v[1] = y[1];
                    v[2] = y[2];
                    intents[0] = v;
                    self.last_intents[0] = v;
                }
            }
        }

        // MARCA-ORDEM-PILOTO (lida pelo teste piloto_entra_antes_da_fisica)
        if self.piloto_formula.is_some() && brain_fires && !intents.is_empty() {
            let (_l1_raw, cfc_in, kan_in) = self.brain.download_debug();
            let mut x = [0.0f32; 192];
            x[..96].copy_from_slice(&cfc_in[..96]);
            let n_kan = kan_in.len().min(96);
            x[96..96 + n_kan].copy_from_slice(&kan_in[..n_kan]);
            if let Some(pl) = self.piloto_formula.as_mut() {
                // Com portão (piloto_gated.json): fora do domínio conhecido o piloto
                // ABSTÉM-SE e a decisão daquele tick fica com o cérebro, que já rodou.
                // Sem referências no arquivo, `agir_com_portao` nunca abstém — o
                // comportamento do v1 fica idêntico (ver piloto_linear.rs).
                if let Some(y) = pl.agir_com_portao(&x) {
                    let mut v = intents[0];
                    v[0] = y[0];
                    v[1] = y[1];
                    v[2] = y[2];
                    intents[0] = v;
                    self.last_intents[0] = v;
                }
            }
        }

        // ── FLY BRAIN OVERRIDE (`--fly-brain`, docs/FLY-MAP-BRAIN.md §2 and "Reflex
        // readout") ─────────────────────────────────────────────────────────────
        //
        // The three commands read out of the connectome REPLACE the fast brain's raw
        // intents, HERE, before `kan_eval` — so everything downstream (KAN, the slow
        // brain's modulation, the hippocampus bias, the steering filter, the clamps, the
        // gearbox) is bit-for-bit the pipeline a normal run uses. That is what makes this
        // an arm and not a second simulator: only the source of the three numbers changes.
        //
        // Same shape as the full override of `PILOTO_GANHO` below, and placed before the
        // assembly of `phys_states` for the same reason the pilot block is: an override
        // recorded after it never reaches the motor, it only contaminates `mem0`.
        //
        // `last_intents` moves with `intents`, or a tick on which the brain does not fire
        // (`--brain-skip > 1`) would replay the fast brain's last decision instead of the
        // map's.
        //
        // `apply_reflex_host_override` is the host half of `--fly-readout reflex`
        // ("Reflex readout"): with `throttle_is_none = false` and `sign = 1.0` (the
        // defaults, or `--fly-readout genome`) it is a bit-exact no-op, so a run that
        // never asks for the reflex arm sees exactly the numbers it always did. It
        // applies to both command sources — the rate map (`--fly-brain`) and Astra's
        // event-driven LIF (`--fly-spiking`), which write into the same `fly_cmd`.
        if (fly_brain() || fly_spiking()) && fly_cmd.len() == n {
            let throttle_is_none =
                universe::fly_map::readout_reflex_on() && fly_reflex_throttle_is_none();
            let cruise = fly_cruise_throttle();
            let sign = fly_reflex_sign();
            let alpha = fly_reflex_alpha();
            let gain = fly_reflex_gain();
            if self.fly_reflex_adapt.len() < n {
                self.fly_reflex_adapt
                    .resize(n, universe::fly_map::ReflexAdaptation::default());
            }
            for i in 0..n {
                let mut v = intents[i];
                let cmd = universe::fly_map::apply_reflex_host_override(
                    fly_cmd[i],
                    throttle_is_none,
                    cruise,
                    sign,
                );
                v[0] = cmd[0];
                // `alpha == 0.0` and `gain == 1.0` (the defaults) make this the identity, so a
                // run that does not ask for adaptation keeps the previous numbers exactly.
                v[1] = self.fly_reflex_adapt[i].step(cmd[1], alpha, gain);
                v[2] = cmd[2];
                intents[i] = v;
                if self.last_intents.len() == n {
                    self.last_intents[i] = v;
                }
            }
        }

        // ── KAN eval inline (idêntico ao gpu_loop.rs) ─────────────────────────
        // Deduplicado (E3): usa universe::kan::eval_knots — mesma fórmula que era
        // reimplementada aqui e em gpu_loop.rs como closure local.
        let kan_eval =
            |out_idx: usize, x: f32| -> f32 { universe::kan::eval_knots(kan_knots, out_idx, x) };

        if let Some(t) = t_dl_outer {
            cpu_phases.push(("cpu.digest_loop", t.elapsed().as_secs_f32() * 1000.0));
        }
        let t_asm = prints_this_tick.then(std::time::Instant::now);
        // MARCA-ORDEM-FISICA (lida pelo teste piloto_entra_antes_da_fisica)
        let mut phys_states: Vec<PhysicsStateGpu> = (0..n)
            .map(|i| {
                let mut s = states[i].phys;
                let intent = &intents[i];
                // ── Modulação da metade LENTA (System 2) ──────────────────────────
                // Mesmos três papéis de `motor.rs`: agressão soma ao acelerador, o teto
                // limita o acelerador e o viés desloca o volante. As magnitudes (0,5 / 0,8
                // / 0,20) são as de lá — não é ajuste novo, é o mesmo contrato.
                //
                // Nota medida: `throttle_cap` esteve travado em 1,0 durante toda a vida do
                // projeto porque `slow[1]` nunca era positivo (init com 100% dos pesos
                // negativos). Só agora ele pode agir.
                // Sem vida: o ser para. Não é penalidade de pontos — é o fim do tempo dele.
                if states[i].life_left <= 0.0 {
                    s.throttle = 0.0;
                    s.brake_cmd = 1.0;
                    s.steer = 0.0;
                    s.dt = dt;
                    return s;
                }

                let slow = if slow_ablated() {
                    [0.0; 3]
                } else {
                    slow_out.get(i).copied().unwrap_or([0.0; 3])
                };
                let aggression = slow[0] * 0.5;
                let throttle_cap = 1.0 - slow[1].max(0.0) * 0.8;
                // Autoridade do lento sobre o volante, parametrizável.
                //
                // Era ±0,20 fixo — fraco demais para o problema medido: o volante ELEITO pela
                // seleção crava em +0,63, e ±0,20 não solta. O lento é a única parte do cérebro
                // com memória temporal (o estado que sabe "já estou na curva"). Se, com
                // autoridade real, ele soltar o volante, o teto era a arquitetura sufocando a
                // memória; se não soltar mesmo com autoridade, o teto é o cérebro pequeno.
                let steer_bias = slow[2] * SLOW_STEER_AUTH.load(Ordering::Relaxed);

                // ── C1: SECTOR HIPPOCAMPUS (docs/PRD-CEREBRO-NEUROMOD.md §C1) ────
                //
                // ORDER IS THE CONTRACT: KAN → slow → **hippocampus** → steering
                // smoothing → clamps. The PRD fixes it (§3, "merge") and so does the
                // mechanism: the bias is a COMMAND correction, so it has to land while
                // the command is still a raw number, before `alfa` filters the steering
                // and before the clamps decide what the motor may receive. Applied after
                // the slow brain because the slow half is the modulator of record here —
                // the hippocampus corrects the command the whole brain arrived at, it
                // does not compete with one stage of it.
                //
                // `hip_bias` already carries `HIP_AUTHORITY` (0.5) and is exactly
                // `[0, 0, 0]` when the organ is off, so these three lines are a no-op on
                // the default path — see `hippocampus_bias`.
                let hip_bias = self.hippocampus_bias(i);
                let thr_kan = (kan_eval(0, intent[0]).max(0.0) + aggression + hip_bias[0])
                    .clamp(0.0, throttle_cap);
                let steer_kan =
                    (kan_eval(1, intent[1]) + steer_bias + hip_bias[1]).clamp(-1.0, 1.0);
                let brk_kan = (kan_eval(2, intent[2]) + hip_bias[2]).max(0.0);
                // Alisamento do volante: `prev*(1-α) + novo*α`.
                //
                // Com α=0,50 o comando leva ~4 ticks para chegar a 90% de um alvo novo, o que
                // é histerese suficiente para SUSTENTAR um volante cravado mesmo que o cérebro
                // já esteja pedindo para soltar. Medido: o ser vira à direita em 74% dos
                // quadros (média +0,40) e roda em círculo dentro da pista, travando o progresso
                // em 265 m. Como parâmetro dá para separar culpa do cérebro de culpa do filtro.
                let alfa = STEER_ALPHA.load(Ordering::Relaxed);
                let steer_smooth =
                    (states[i].prev_steer * (1.0 - alfa) + steer_kan * alfa).clamp(-1.0, 1.0);
                let is_reversing = states[i].reverse_timer > 0.0;
                let in_lockout = states[i].post_reverse_lockout > 0.0;
                let speed_now = (s.vel[0].powi(2) + s.vel[2].powi(2)).sqrt();
                // Brake anulado abaixo de 2 m/s (idêntico ao motor.rs).
                let brk_final = if speed_now < 2.0 { 0.0 } else { brk_kan };

                if is_reversing {
                    // Ré ativa: override completo, pedal fixo 0.75, gear=0.
                    // Volante ALTERNA de lado a cada ré re-disparada (re_falhas ímpar =
                    // esquerda) — sair reto devolvia o carro ao MESMO canto (medido).
                    s.throttle = universe::config::iara::reverse_maneuver::PEDAL;
                    s.steer = if states[i].re_falhas == 0 {
                        0.0
                    } else if states[i].re_falhas % 2 == 1 {
                        -0.6
                    } else {
                        0.6
                    };
                    s.brake_cmd = 0.0;
                    s.current_gear = 0;
                } else if in_lockout {
                    // Pós-ré: throttle capeado em 0.8.
                    s.throttle = thr_kan.min(0.8);
                    s.steer = steer_smooth;
                    s.brake_cmd = brk_final;
                    if s.current_gear <= 1 {
                        s.current_gear = 2;
                    }
                } else {
                    // Modo normal: intent neural completo. SEM muletas.
                    s.throttle = thr_kan;
                    s.steer = steer_smooth;
                    s.brake_cmd = brk_final;
                    // PILOTO FANTASMA (pilots.rs): sobrepõe tudo. Não é muleta de produção —
                    // é instrumento de diagnóstico, ligado só por flag, para responder "o
                    // corpo consegue?".
                    let pg = PILOTO_GANHO.load(Ordering::Relaxed);
                    if pg != 0.0 {
                        let pos2d = bevy::math::Vec2::new(s.pos[0], s.pos[2]);
                        // Pure-pursuit controller: aponta para um ponto L metros à frente na
                        // racing line e corrige pelo heading, não pelo desvio lateral. Evita o
                        // bug de `lookahead_offset` que mistura erro longitudinal com lateral
                        // quando o carro está atrás do ponto de antecipação. L adaptativo: mais
                        // distância em alta velocidade para suavizar.
                        let l_lookahead = (speed_now * 2.5).clamp(20.0, 45.0);
                        let tgt = self.racing_line.lookahead_point(pos2d, l_lookahead);
                        let thr_base = PILOTO_THR.load(Ordering::Relaxed);
                        let (steer, thr, brk) = crate::pilots::piloto_fantasma_comando(
                            pg,
                            thr_base,
                            s.pos[0],
                            s.pos[2],
                            s.fwd[0],
                            s.fwd[2],
                            tgt.x,
                            tgt.y, // Vec2.y = world Z
                            speed_now,
                            states[i].phantom_d_min,
                        );
                        s.steer = steer;
                        s.throttle = thr;
                        s.brake_cmd = brk;
                    }
                    if s.current_gear <= 1 {
                        s.current_gear = 2;
                    }
                    if s.current_gear > 2 && speed_now < 3.0 {
                        s.current_gear -= 1;
                    }
                }
                // ── D4: DANO OCULTO (ver DANO_ESCALA em statics.rs) ───────────────
                // Acumula por giro alto e parede; degrada acelerador e direção. Aplicado
                // DEPOIS dos ramos de comando (afeta o que chega à física, inclusive na
                // ré) e ANTES do câmbio (rpm de leitura não muda no mesmo tick).
                let dano_escala = DANO_ESCALA.load(Ordering::Relaxed);
                if dano_escala > 0.0 && states[i].life_left > 0.0 {
                    let rpm_frac = states[i].phys.current_rpm / states[i].phys.max_rpm.max(1.0);
                    let mut inc = 0.0f32;
                    if rpm_frac > 0.9 {
                        inc += 0.00015 * ((rpm_frac - 0.9) / 0.1).min(1.5);
                    }
                    if states[i]
                        .dist_wall_f
                        .min(states[i].dist_wall_l.min(states[i].dist_wall_r))
                        < 3.0
                    {
                        inc += 0.0004;
                    }
                    // C0: the increment after the 1.0 clamp, which is the damage the
                    // being actually took — `inc * dano_escala` overstates it at the cap.
                    let damage_before = states[i].dano;
                    states[i].dano = (states[i].dano + inc * dano_escala).min(1.0);
                    self.lap_damage[i] = states[i].dano - damage_before;
                    let d = states[i].dano;
                    s.throttle = s.throttle * (1.0 - 0.5 * d);
                    s.steer = s.steer * (1.0 - 0.3 * d);
                }
                // ── CÂMBIO (paridade com motor.rs:2451-2494) ──────────────────────
                // A física deste pipeline SEMPRE teve caixa de 6 marchas (gear_ratio em
                // physics_power_*.wgsl move a força), mas nada aqui subia de marcha: só o
                // piso "≥2" e a queda <3 m/s acima. Teto físico da marcha 2:
                // v_max = (8500+250)rpm ÷ (3,8×3,7) × 0,33 m = 21,5 m/s — e o pico real
                // medido da população era 21,9 m/s, cravado no limitador. O Bevy resolve
                // com auto-shift por RPM + override neural; portado aqui atrás de
                // `--cambio auto|neural` (padrão travado até o A/B validar).
                let modo_cambio = CAMBIO_MODO.load(Ordering::Relaxed);
                if modo_cambio > 0 && !is_reversing {
                    let rpm = states[i].phys.current_rpm;
                    let max_rpm = states[i].phys.max_rpm.max(1.0);
                    // Embreagem colada ≈ marcha ≥2 e v > 3,5 m/s. Bevy usa wheel_rpm>1200,
                    // que em marcha 2 (ratio 12) é v > 3,33 — aproximação de mesma ordem.
                    let engaged = s.current_gear >= 2 && speed_now > 3.5;
                    states[i].shift_cooldown = (states[i].shift_cooldown - dt).max(0.0);
                    if states[i].shift_cooldown <= 0.0 {
                        let mut req: i32 = 0;
                        let teto = CAMBIO_TETO.load(Ordering::Relaxed);
                        if engaged && rpm > max_rpm * 0.85 && s.current_gear < teto {
                            req = 1; // auto-upshift (motor.rs:2476)
                        } else if engaged && rpm < max_rpm * 0.30 && s.current_gear > 2 {
                            req = -1; // auto-downshift (motor.rs:2480)
                        } else if modo_cambio == 2 && engaged {
                            // Override neural por BORDA (motor.rs:2484-2487): intent[3] sobe,
                            // intent[4] desce — as duas primeiras linhas cegas de L2 ganham função.
                            if intent[3] > 0.5
                                && states[i].shift_up_prev <= 0.5
                                && s.current_gear < teto
                            {
                                req = 1;
                            } else if intent[4] > 0.5
                                && states[i].shift_down_prev <= 0.5
                                && s.current_gear > 2
                            {
                                req = -1;
                            }
                        }
                        if req != 0 {
                            s.current_gear =
                                (s.current_gear as i32 + req).clamp(2, teto as i32) as u32;
                            states[i].shift_cooldown = 0.35;
                        }
                        // prev SÓ atualiza quando a troca era possível (cooldown zerado,
                        // fora da ré): borda que sobe DURANTE cooldown/ré fica "pendente"
                        // e dispara na primeira janela válida. Atualizar sempre engolia a
                        // borda (prev virava 1 antes da janela abrir) — achado da bateria
                        // de validação P1.9, dois sítios (cooldown e pós-ré).
                        states[i].shift_up_prev = intent[3];
                        states[i].shift_down_prev = intent[4];
                    }
                }
                // Telemetria de pico (prova de vida do câmbio — ver statics.rs).
                if speed_now > VEL_PICO.load(Ordering::Relaxed) {
                    VEL_PICO.store(speed_now, Ordering::Relaxed);
                }
                if s.current_gear > MARCHA_PICO.load(Ordering::Relaxed) {
                    MARCHA_PICO.store(s.current_gear, Ordering::Relaxed);
                }
                // ── OVERRIDE BRUTO, o último da cadeia ────────────────────────────
                // Vem depois de TODOS os ramos (normal, ré, lockout, piloto fantasma) porque a
                // pergunta que ele responde é sobre o CORPO: dado este comando exato chegando
                // à física, o carro vira? Qualquer coisa entre o valor e o `PhysicsStateGpu`
                // seria mais uma variável a suspeitar.
                // Piloto sensorial: sobrepõe o cérebro inteiro.
                if PILOTO_SENSOR.load(Ordering::Relaxed) != 0.0 {
                    s.steer = states[i].sensor_steer;
                    s.throttle = states[i].sensor_thr;
                    s.brake_cmd = states[i].sensor_brk;
                    if s.current_gear <= 1 {
                        s.current_gear = 2;
                    }
                }
                let ke = STEER_ESCALA.load(Ordering::Relaxed);
                if ke > 0.0 {
                    s.steer = (s.steer * ke).clamp(-1.0, 1.0);
                }
                let sf = STEER_FIXO.load(Ordering::Relaxed);
                if sf != 0.0 {
                    s.steer = sf.clamp(-1.0, 1.0);
                }
                let tf = THR_FIXO.load(Ordering::Relaxed);
                if tf != 0.0 {
                    s.throttle = tf.clamp(0.0, 1.0);
                    s.brake_cmd = 0.0;
                }
                s.dt = dt;
                s
            })
            .collect();

        // ── LABORATÓRIO DE OBSERVAÇÃO (`--observar`) ──────────────────────────
        // Grava a cadeia neural COMPLETA do ser 0 neste tick, em f32 cru:
        //   [tick, speed, rpm, gear, engine_temp, throttle, steer, brake, life, joy,
        //    onda_cfc(96) canais 0..96 pós-KAN+CfC (i8 como f32),
        //    onda_kan(96) canais 96..192 pós-KAN,
        //    l1_raw(96)   acumulador L1 pré-tanh,
        //    intent(96)   final_intent pós-tanh (0..48=L2, 48..96=bypass)] = 394 f32.
        // Só quando o cérebro disparou (senão a linha repetiria a decisão anterior).
        // O bloco do PILOTO POR FÓRMULA já rodou lá em cima (antes de `phys_states`),
        // então `intents[0]` aqui já é a decisão da fórmula — o obs.f32 grava o que
        // de fato foi ao motor.
        if self.obs_writer.is_some() && brain_fires {
            let (l1_raw, cfc_in, kan_in) = self.brain.download_debug();
            let p = &phys_states[0];
            let st = &states[0];
            let mut row: Vec<f32> = Vec::with_capacity(394);
            row.extend_from_slice(&[
                tick_idx as f32,
                (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt(),
                p.current_rpm,
                p.current_gear as f32,
                p.engine_temp,
                p.throttle,
                p.steer,
                p.brake_cmd,
                st.life_left as f32,
                st.joy_tick,
            ]);
            row.extend_from_slice(&cfc_in); // 96
            row.extend_from_slice(&kan_in[..96.min(kan_in.len())]); // 96
            row.extend_from_slice(&l1_raw); // 96
            row.extend_from_slice(&intents[0]); // 96
            debug_assert_eq!(row.len(), 394);
            use std::io::Write as _;
            if let Some(w) = self.obs_writer.as_mut() {
                let _ = w.write_all(bytemuck::cast_slice(&row));
            }
        }

        // ── C1: HIPPOCAMPUS OBSERVATION (`hip_gen<N>.f32`) ─────────────────────
        //
        // Being 0 only, every tick, `HIP_OBS_COLS` f32 per row:
        //   [tick, sector, mem[s](4), mem[s+1](4), bias(3)]
        // `obs.f32` is deliberately NOT touched: it is 394 fixed columns and tools index
        // into it by offset, so a 395th column would silently shift every one of them.
        //
        // Written EVERY tick, not only when the brain fires (unlike `obs.f32` above): the
        // memory and the bias exist on every tick regardless of `brain_skip`, and a file
        // sampled on the brain's schedule could not be read against a per-tick lap trace.
        //
        // The bias here is recomputed rather than carried out of the command closure —
        // 27 multiply-adds against the borrow gymnastics of threading a value out of a
        // `map()` that already borrows `self`. It is the same function, so the row cannot
        // drift from what the commands actually received — which also means the three
        // bias columns are POST-authority (already × `HIP_AUTHORITY` = 0.5), i.e. the
        // number that was added to the command, not the raw `tanh` output. Divide by 0.5
        // to recover the organ's own output.
        if self.hip_writer.is_some() && !self.hip_mem.is_empty() {
            use universe::gpu::hippocampus as hip;
            let sector = self.hip_mem[0].read_sector();
            let here = self.hip_mem[0].slot(sector);
            let next = self.hip_mem[0].slot(sector + 1);
            let b = self.hippocampus_bias(0);
            let mut row: Vec<f32> = Vec::with_capacity(hip::HIP_OBS_COLS);
            row.push(tick_idx as f32);
            row.push(sector as f32);
            row.extend_from_slice(&here);
            row.extend_from_slice(&next);
            row.extend_from_slice(&b);
            debug_assert_eq!(row.len(), hip::HIP_OBS_COLS);
            use std::io::Write as _;
            if let Some(w) = self.hip_writer.as_mut() {
                let _ = w.write_all(bytemuck::cast_slice(&row));
            }
        }

        // ── OPEN-LOOP WAVE RECORDING (`--record-wave`) ──────────────────────────
        // Records the raw 192-channel sensory wave for being 0, plus the three
        // commands the recording being actually issued (`intents[0][0..3]`, the same
        // raw pre-KAN values `obs.f32` stores at columns 298..301). Gated on
        // `brain_fires` for the same reason `--observe` is: on a skipped tick
        // `intents` is a copy of the last decision, not a fresh one, and recording it
        // again would duplicate a row instead of adding information — this also keeps
        // `wave.f32` in row-for-row lockstep with `obs.f32`, which is what lets Task 5
        // cross-check one against the other.
        //
        // Reuses `percept.read_after_submit` exactly as `--audit-vision` does above
        // (same function, same `map_async` → `poll` → `recv` path) rather than adding a
        // second readback route for the same buffer.
        if self.wave_writer.is_some() && brain_fires {
            use universe::gpu::perception::WAVE_U32S;
            let bytes = (n * WAVE_U32S * 4) as u64;
            let ondas = self.percept.read_after_submit(&self.device, n, bytes);
            if let Some(wave0) = ondas.first() {
                let wave_f32: Vec<f32> = wave0.iter().map(|&v| v as f32).collect();
                let intent = [intents[0][0], intents[0][1], intents[0][2]];
                if let Some(w) = self.wave_writer.as_mut() {
                    let _ = universe::gpu::wave_record::write_row(w, &wave_f32, &intent);
                }
            }
        }

        // ── AMOSTRAGEM DE MÉTRICAS DE ATIVAÇÃO (E4, `--ativacao`) ───────────────
        //
        // Só roda quando `ATIVACAO_ON` está ligado E o cérebro disparou neste tick
        // (senão `intents` é uma cópia da última decisão — amostrar de novo seria
        // amostra duplicada, não amostra nova) E o tick bate no período configurado
        // (`ATIVACAO_N`, padrão 64). Custo real quando ligado:
        //   - L2 (`intents[0..48]`): ZERO round-trip extra — `intents` já foi lido de
        //     volta da GPU alguns parágrafos acima, todo tick em que o cérebro dispara,
        //     independente desta flag. Só reaproveita o `Vec` que já existe em CPU.
        //   - L1 (`buf_cfc`): round-trip extra de verdade (`download_cfc_state` monta
        //     encoder+staging+map própria) — é o único custo real desta seção, e só
        //     ocorre 1 vez a cada `ATIVACAO_N` ticks.
        // Custo quando DESLIGADO (padrão): zero — nem o `if` de dentro roda.
        if ATIVACAO_ON.load(Ordering::Relaxed) && brain_fires {
            let stride = ATIVACAO_N.load(Ordering::Relaxed).max(1);
            if (tick_idx as u32) % stride == 0 {
                // Distância à parede mais próxima (min dos 3 raios já calculados no topo
                // do tick, gravados em `states[i].dist_wall_*`) — mesmo dado que alimenta
                // ch118/119/123, reaproveitado aqui sem raio novo.
                let dist_parede = |i: usize| {
                    states[i]
                        .dist_wall_l
                        .min(states[i].dist_wall_r)
                        .min(states[i].dist_wall_f)
                };
                if intents.len() == n {
                    for (i, row) in intents.iter().enumerate() {
                        self.l2_samples.push(row[0..48].to_vec());
                        // L2 nunca pula por NaN (só L1 tem essa guarda) — lockstep
                        // garantido: um push de contexto por push de amostra.
                        self.regiao_l2.push(contexto_regiao(
                            &self.track_curvatura,
                            &states[i].phys,
                            dist_parede(i),
                        ));
                    }
                }
                // Estado CfC de L1 — único readback extra desta seção.
                let cfc = self.brain.download_cfc_state(n);
                for (i, row) in cfc.iter().enumerate() {
                    if row.iter().any(|v| !v.is_finite()) {
                        eprintln!(
                            "[ativacao] ERRO: NaN/Inf no estado L1 (CfC) do ser {i} no tick {tick_idx} — amostra descartada."
                        );
                        self.nan_seres.insert(i);
                        continue;
                    }
                    self.l1_samples.push(row.clone());
                    // Mesmo `continue` acima pula ESTE push também — l1_samples e
                    // regiao_l1 crescem sempre juntos, nunca desalinham.
                    self.regiao_l1.push(contexto_regiao(
                        &self.track_curvatura,
                        &states[i].phys,
                        dist_parede(i),
                    ));
                }
                // Comandos motores do "campeão" (ser 0) — para entropia de política.
                // `phys_states[0]` já tem o comando FINAL (pós axiom_policy/piloto/
                // override), antes da física consumi-lo — é o que realmente foi
                // mandado ao corpo neste tick.
                if n > 0 {
                    let p = &phys_states[0];
                    self.policy_samples.push([p.throttle, p.steer, p.brake_cmd]);
                }
            }
        }

        // Atualiza steer e stagnation.
        for i in 0..n {
            if states[i].reverse_timer > 0.0 {
                states[i].prev_steer = 0.0;
            } else {
                states[i].prev_steer = phys_states[i].steer;
            }
            let speed = (phys_states[i].vel[0].powi(2) + phys_states[i].vel[2].powi(2)).sqrt();
            if speed < 1.5 {
                states[i].stagnation_ticks += 1;
            } else {
                states[i].stagnation_ticks = 0;
            }
        }

        // `PhysicsGpu::step` opens its OWN `CommandEncoder`, submits it, and blocks on
        // a synchronous `map_async` → `poll` → `recv` readback before returning — a
        // completely separate submission from the single per-tick encoder timed above,
        // on the same device. No GPU timestamp written from that other encoder can reach in
        // here, so this is measured on the CPU instead: `Instant` is the RIGHT tool
        // exactly because the call is synchronous — the wall time it reports IS the
        // real cost (three kernels' GPU execution plus the driver round-trip for the
        // blocking readback), not an approximation of it. Labelled distinctly in the
        // printout so a reader never mistakes it for a `GpuTimer` measurement. Reuses
        // the SAME `prints_this_tick` computed at the top of this function -- an
        // `Instant` is cheap enough that a second gate is not needed for overhead
        // reasons, but a single source of truth for "is this a printing tick" is what
        // keeps this row and the GPU rows above sampled on the SAME tick.
        let physics_t0 = prints_this_tick.then(std::time::Instant::now);
        if let Some(t) = t_asm {
            cpu_phases.push(("cpu.assemble_phys", t.elapsed().as_secs_f32() * 1000.0));
        }
        let ok = self.phys_gpu.step(&mut phys_states);
        if let Some(t0) = physics_t0 {
            physics_ms = t0.elapsed().as_secs_f32() * 1000.0;
            println!(
                "  [perfil] {:<22} {:8.4} ms (CPU wall, blocking submit)",
                "physics.step", physics_ms
            );
        }
        if !ok {
            return false;
        }
        for i in 0..n {
            states[i].phys = phys_states[i];
            // ── Rotação de yaw: aplica av_y * dt à orientação do ser ─────────
            // O kernel de física escreve `out.fwd = fwd` (sem rotacionar — esse
            // passo ficava no ECS Bevy). No modo headless fazemos manualmente:
            //   rotate_y(θ): x' = x·cos(θ) - z·sin(θ),  z' = x·sin(θ) + z·cos(θ)
            // Convenção: av_y > 0 = rotação anti-horária vista de cima (+Y) —
            // igual à CPU (physics.rs:760: transform.rotate_y(body.angular_velocity.y * dt)).
            let theta = states[i].phys.av_y * dt;
            let (sin_t, cos_t) = theta.sin_cos();
            let fwd = states[i].phys.fwd;
            let lat = states[i].phys.lat;
            states[i].phys.fwd[0] = fwd[0] * cos_t - fwd[2] * sin_t;
            states[i].phys.fwd[2] = fwd[0] * sin_t + fwd[2] * cos_t;
            states[i].phys.lat[0] = lat[0] * cos_t - lat[2] * sin_t;
            states[i].phys.lat[2] = lat[0] * sin_t + lat[2] * cos_t;
            // Normaliza para evitar deriva numérica acumulada.
            let fn_ = (states[i].phys.fwd[0].powi(2) + states[i].phys.fwd[2].powi(2))
                .sqrt()
                .max(1e-6);
            let ln_ = (states[i].phys.lat[0].powi(2) + states[i].phys.lat[2].powi(2))
                .sqrt()
                .max(1e-6);
            states[i].phys.fwd[0] /= fn_;
            states[i].phys.fwd[2] /= fn_;
            states[i].phys.lat[0] /= ln_;
            states[i].phys.lat[2] /= ln_;
        }

        // memory0 → temporal.
        let mem0: Vec<f32> = (0..n).map(|i| intents[i][0].clamp(0.0, 1.0)).collect();
        self.percept.update_memory0(&self.queue, &mem0[..n]);

        // `--kernel-profile <N>`: the deliverable this instrument exists to produce.
        // `TOTAL` and `physics.step` only cover what got a bracket or an `Instant`;
        // everything else in this function -- buffer/bind-group rebuilds, the
        // per-being Rust loops before and after `phys_gpu.step`, submission and
        // scheduling latency for the two device round trips -- is invisible to both.
        // Wrapping the WHOLE function body in one CPU `Instant` and subtracting what
        // was already measured turns that invisible remainder into a number instead
        // of a guess. Not clamped to zero: a negative value would mean `tick_t0` was
        // started after some of the work it is supposed to cover, which is a bug in
        // this instrument worth seeing, not worth hiding.
        if let Some(t0) = tick_t0 {
            let tick_ms = t0.elapsed().as_secs_f32() * 1000.0;
            let gaps_ms = tick_ms - total_gpu_ms - physics_ms;
            println!(
                "  [perfil] {:<22} {gaps_ms:8.4} ms (tick wall {tick_ms:.4} - TOTAL - physics.step)",
                "gaps"
            );
            // Breakdown of the `gaps` number above. Anything unaccounted for is Rust
            // work outside these five phases; print it explicitly rather than letting
            // the reader assume the phases sum to the gap.
            // `cpu.record` is the PARENT of `rec.cpu_prep` + `rec.gpu_record`; summing it
            // as a sibling double-counts and printed a -6.68 ms residual. Sum the leaves.
            // `rec.cpu_prep` is itself now a PARENT too (Task 1, 2026-08-27-tick-cpu-cost):
            // the `prep.*` rows sub-profile it, so it gets the same exclusion `cpu.record`
            // already has, for the same reason.
            let somado: f32 = cpu_phases
                .iter()
                .filter(|(l, _)| *l != "cpu.record" && *l != "rec.cpu_prep")
                .map(|(_, ms)| ms)
                .sum();
            for (label, ms) in &cpu_phases {
                let marca = if *label == "cpu.record" {
                    " (= rec.* above)"
                } else if *label == "rec.cpu_prep" {
                    " (= prep.* above)"
                } else {
                    ""
                };
                println!("  [perfil]   {label:<20} {ms:8.4} ms{marca}");
            }
            // The CPU phases decompose the TICK WALL, not `gaps`. The GPU kernels
            // execute INSIDE `cpu.poll` — that is what the CPU is blocked waiting for —
            // so `gaps = wall - TOTAL - physics` double-subtracts them and the phases
            // legitimately sum to more than it. Measured 2026-08-27: phases 20.96 ms
            // against a gaps of 19.96, the 1.0 ms difference being the kernel time
            // counted on both sides. Report against the wall.
            println!(
                "  [perfil]   {:<20} {:8.4} ms (wall {tick_ms:.4} - phases; kernels run INSIDE cpu.poll)",
                "cpu.unaccounted",
                tick_ms - somado - physics_ms
            );
        }

        true
    }

    /// Reseta CfC de percepção (estado recorrente) para todos os seres.
    /// Chamado entre gerações para que os filhos comecem sem memória.
    pub fn reset_cfc(&mut self, n: usize) {
        self.retina_digest.reset(&self.queue, n);
    }

    /// Troca a pista sem recriar device/pipelines/bind-groups — só reescreve os
    /// buffers estáticos (grade/fluxo/alturas/heightmap) via `write_buffer` (P1:
    /// rotação de pistas por geração + avaliação held-out, ver
    /// docs/PLANO_EVOLUCAO.md §P1).
    ///
    /// NÃO chamado em nenhum caminho quando `--pistas`/`--pista-held-out` estão
    /// ausentes — o binário sem essas flags nunca invoca este método, então o
    /// comportamento (e o A/B em andamento) fica bit-a-bit idêntico ao de antes
    /// desta mudança.
    ///
    /// Só é seguro porque `VoxelWorld::new` usa `size=320`/`cell_size=3.0`
    /// CONSTANTES independentes da seed (world.rs) — nenhum buffer estático muda de
    /// tamanho entre pistas, então dá pra sobrescrever em vez de realocar. Custo:
    /// ~400 KB de upload total (grid×3 + heights + hmap), sub-milissegundo — ver a
    /// estimativa em docs/PLANO_EVOLUCAO.md §P1 (irrelevante perto dos ~19,2 s/gen
    /// medidos no P0).
    pub fn reload_world(&mut self, world: &VoxelWorld) {
        debug_assert_eq!(
            world.size as u32, self.grid_size,
            "reload_world: pista nova tem grade de tamanho diferente — buffer não realoca"
        );
        debug_assert!(
            (world.cell_size - self.cell_size).abs() < 1e-6,
            "reload_world: pista nova tem cell_size diferente — buffer não realoca"
        );

        let grid_u32: Vec<u32> = world.grid.iter().map(|v| *v as u32).collect();
        self.queue
            .write_buffer(&self.buf_grid, 0, bytemuck::cast_slice(&grid_u32));

        self.percept.reload_world(&self.queue, world);

        let grid_u8: Vec<u8> = world.grid.iter().map(|v| *v as u8).collect();
        self.phys_gpu.reload_world(&grid_u8, &world.height_map);

        // Cópias CPU que ficariam presas na pista antiga se esquecidas aqui — ver
        // risco correspondente em docs/PLANO_EVOLUCAO.md §P1: `racing_line`
        // alimenta os canais sensoriais 108-110 (linha ideal), `grid_cpu` alimenta
        // o voxel-sob-o-ser (dopamina/joy_tick, colisão CPU-side dos raios de
        // parede em `tick()`).
        self.racing_line = world.racing_line.clone();
        self.grid_cpu = grid_u32;

        // κ(x,z) da pista NOVA — sem isto, depois de uma troca de pista (`--pistas`/
        // `--track-held-out`), `--regions` continuaria medindo curvatura da pista
        // ANTERIOR contra amostras de ativação da pista atual: números com aparência
        // válida, silenciosamente errados.
        let wps: Vec<(f32, f32)> = world.waypoints_world.iter().map(|p| (p.x, p.y)).collect();
        self.track_curvatura = CurvaturaPista::medir(&wps);
    }
}

#[cfg(test)]
mod testes_ordem {
    /// TRAVA DE ORDEM: o override do piloto TEM de acontecer antes da montagem de
    /// `phys_states`, senão o comando da fórmula não chega ao motor — ele fica só
    /// no `intents`, que depois disso só alimenta `mem0` (canal de percepção 120).
    ///
    /// Esse foi um erro REAL (2026-08-16): o bloco estava depois do `.collect()` e
    /// a bateria inteira mediu o campeão com um canal de memória contaminado,
    /// acreditando estar medindo o clone dirigindo. Achado por auditoria
    /// adversarial; a evidência que matou foi "comandos aplicados bit-a-bit
    /// idênticos aos do campeão nos 576 primeiros ticks enquanto o intent gravado
    /// já diferia em 0,60 desde o tick 0".
    ///
    /// O teste lê o PRÓPRIO fonte: é a única forma de travar ordem entre dois
    /// trechos do mesmo tick sem montar uma GPU no teste.
    #[test]
    fn piloto_entra_antes_da_fisica() {
        let fonte = include_str!("context.rs");
        // As marcas são montadas em RUNTIME: escritas literalmente aqui, o
        // `include_str!` do próprio teste casaria consigo mesmo (o teste passou a
        // contar 3 ocorrências de si próprio na primeira tentativa).
        let m_piloto = format!("MARCA-{}-PILOTO", "ORDEM");
        let m_phys = format!("MARCA-{}-FISICA", "ORDEM");
        let i_piloto = fonte.find(&m_piloto).expect("marca do piloto sumiu");
        let i_phys = fonte.find(&m_phys).expect("marca da física sumiu");
        assert!(
            i_piloto < i_phys,
            "REGRESSÃO: o override do piloto (byte {i_piloto}) está DEPOIS da \
             montagem de phys_states (byte {i_phys}) — a fórmula não chega ao motor."
        );
        // E só pode existir UM bloco de override (havia dois durante o conserto).
        assert_eq!(
            fonte.matches(&m_piloto).count(),
            1,
            "marca do piloto duplicada no código"
        );
    }
}

#[cfg(test)]
mod hostile_world_tests {
    use super::signed_bearing;
    use bevy::math::Vec2;

    /// `fwd`/`lat` exactly as `gpu::physics::make_physics_state` builds them:
    /// `lat = [fwd.z, 0, -fwd.x]`, projected onto the XZ plane.
    fn basis(fwd: Vec2) -> (Vec2, Vec2) {
        (fwd, Vec2::new(fwd.y, -fwd.x))
    }

    /// The bearing's sign follows the `lat` basis vector, and nothing else.
    ///
    /// Why a test and not a comment: a flipped bearing is invisible to every aggregate
    /// this binary prints. The being would steer away from the station it should reach
    /// and into the predator it should flee, the fitness would just be lower, and the
    /// conclusion on the page would be "the pressure does not select", which is exactly
    /// the sentence the whole BODY PHASE exists to test. The `-0.5` case is the load
    /// bearing one: `-lat` is where the doc's word "left" is either right or wrong, and
    /// the basis is the only thing checkable.
    #[test]
    fn bearing_sign_follows_the_lat_basis() {
        for fwd in [
            Vec2::new(1.0, 0.0),
            Vec2::new(0.0, 1.0),
            Vec2::new(-1.0, 0.0),
            Vec2::new(0.7071, 0.7071),
        ] {
            let (f, l) = basis(fwd);
            assert!(
                (f.length() - 1.0).abs() < 1e-4 && (l.length() - 1.0).abs() < 1e-4,
                "basis must stay unit-length"
            );
            assert!(f.dot(l).abs() < 1e-4, "basis must stay orthogonal");
            assert!(
                signed_bearing(f * 10.0, f, l).abs() < 1e-5,
                "straight ahead is 0"
            );
            assert!(
                (signed_bearing(l * 10.0, f, l) - 0.5).abs() < 1e-5,
                "toward +lat is +0.5 (a quarter turn = pi/2, over pi)"
            );
            assert!(
                (signed_bearing(-l * 10.0, f, l) + 0.5).abs() < 1e-5,
                "toward -lat is -0.5"
            );
            assert!(
                (signed_bearing(-f * 10.0, f, l).abs() - 1.0).abs() < 1e-5,
                "directly behind saturates at |1|"
            );
        }
    }

    /// The encoded range never leaves `[-1, 1]`, at any distance and any direction.
    ///
    /// The shader clamps, so an out-of-range host value does not error — it pins the
    /// channel at 127, which reads as a working sensor stuck on maximum. That failure
    /// mode is why the range is asserted on the host side, where it is still a bug.
    #[test]
    fn bearing_stays_inside_the_encodable_range() {
        let (f, l) = basis(Vec2::new(0.0, 1.0));
        for step in 0..360 {
            let a = (step as f32).to_radians();
            for d in [0.001f32, 1.0, 350.0] {
                let b = signed_bearing(Vec2::new(a.cos(), a.sin()) * d, f, l);
                assert!(b >= -1.0 && b <= 1.0, "bearing {b} outside [-1,1]");
            }
        }
        // A target exactly on top of the being answers the same 0.0 as "no target".
        // Deliberate, documented, and paired with a proximity channel that separates
        // the two — asserted so nobody "fixes" it into a NaN.
        assert_eq!(signed_bearing(Vec2::ZERO, f, l), 0.0);
    }

    /// `--tire-thermal 0` must be EXACTLY off, not approximately off.
    ///
    /// The dose scales the depth of the grip curve, so gain 0 puts the floor at 1.0 and
    /// the whole curve collapses onto the constant 1.0 — `env_grip` is then the same
    /// 1.0 the physics state is born with. A dose whose "off" is 0.999 would make the
    /// off arm of the A/B a different world from the no-flag run, and the paired
    /// comparison would be measuring the difference between two ON arms.
    #[test]
    fn thermal_gain_zero_is_exactly_neutral() {
        let p = crate::thermal_base_params(0.0);
        for i in 0..=100 {
            let t = i as f32 / 100.0;
            let g = universe::tire_thermal::grip_factor(t, &p);
            assert!(
                (g - 1.0).abs() < 1e-5,
                "gain 0 must give grip 1.0 at every temperature, got {g} at {t}"
            );
        }
    }

    /// Every dose keeps the DC gain at 1.0, so the arms differ in WHEN grip is
    /// available and never in how much of it there is on average.
    ///
    /// This is the confound the project already paid for: a curve whose mean is 0.879
    /// makes "thermal on" also mean "12% less grip everywhere", and the A/B then cannot
    /// tell the dynamic from the offset. The deep doses are the ones at risk, because
    /// the trim they need grows past a fixed cap — which is why `thermal_base_params`
    /// derives `max_factor` from the band instead of hardcoding it.
    #[test]
    fn every_thermal_dose_keeps_unit_dc_gain() {
        let band = universe::tire_thermal::reference_band();
        for gain in [0.5f32, 1.0, 1.5, 2.0, 3.0] {
            let p = crate::thermal_base_params(gain);
            let dc = universe::tire_thermal::dc_gain(&band, &p);
            assert!(
                (dc - 1.0).abs() < 1e-3,
                "dose {gain}: DC gain {dc} != 1.0 — the ON arm would be a grip nerf"
            );
            assert!(
                p.max_factor > p.gain_trim,
                "dose {gain}: cap {} would clamp the trim {} and the mean could not \
                 reach 1.0",
                p.max_factor,
                p.gain_trim
            );
        }
    }

    /// A deeper dose must be a WIDER SWING, not a lower average — the property that
    /// makes the dose sweep readable at all.
    #[test]
    fn deeper_thermal_dose_widens_the_swing() {
        let band = universe::tire_thermal::reference_band();
        let mut prev = 0.0f32;
        for gain in [0.5f32, 1.0, 1.5, 2.0] {
            let p = crate::thermal_base_params(gain);
            let lo = band
                .iter()
                .map(|&t| universe::tire_thermal::grip_factor(t, &p))
                .fold(f32::MAX, f32::min);
            let hi = band
                .iter()
                .map(|&t| universe::tire_thermal::grip_factor(t, &p))
                .fold(f32::MIN, f32::max);
            let swing = hi - lo;
            assert!(
                swing > prev,
                "dose {gain} swing {swing} is not wider than the previous {prev}"
            );
            prev = swing;
        }
    }
}
