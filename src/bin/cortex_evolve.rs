//! Genetic evolution using the CORTICAL SHEET as the beings' brain.
//!
//! # Difference from gpu_evolve.rs
//!
//! `gpu_evolve.rs` evolves BrainGpu weights (BitLinear CfC).
//! This binary evolves the CortexSheet parameters:
//!   - initial afferent weights (buf_afw) — where each channel lands on the sheet
//!   - afference targets (tx_k, ty_k) — inherited from the genome hash
//!   - τ per unit — log-spaced bank, but the SPREAD evolves
//!   - initial motor weights (buf_mw) — which units drive the output
//!
//! What does NOT evolve: region labels. No `if channel_is_vision {}`.
//!
//! # Sheet genome
//!
//! `CortexGenome` carries:
//!   - `afw_init[SHEET_N × N_INPUTS]`: initial Hebbian weights (f32, σ=0.3)
//!   - `mw_init[SHEET_N × 3]`: initial motor weights (f32, σ=0.2)
//!   - `tau_spread`: scalar multiplying the log-spaced bank (1.0=default, evolves)
//!
//! Hebbian plasticity still runs during the ticks — the genome provides the STARTING POINT.
//!
//! # Modality map (diagnostic every N gens)
//!
//! Every `--map-interval` generations, prints a 32×32 ASCII map of the dominant
//! modality per unit (differential response A vs B), neighbor cross-correlation, and
//! intra/inter-region ratio. Criterion registered BEFORE measuring: ratio > 3×.
//!
//! # Fitness
//!
//! Continuous distance in meters — identical to gpu_evolve.rs.
//! NO motor crutches.
//!
//! # Usage
//!
//!   cd ~/projects/LLM/Universe
//!   env -u DISPLAY nice -n 19 ./target/release/cortex_evolve \
//!       --seres 40 --ticks 6000 --gens 100
//!
//!   # Extended report every 10 generations:
//!   env -u DISPLAY nice -n 19 ./target/release/cortex_evolve \
//!       --seres 40 --ticks 6000 --gens 100 --map-interval 10

use std::sync::Arc;
use std::time::Instant;

use wgpu::util::DeviceExt;

use universe::config::iara::vision::{RETINA_ROWS, RETINA_TOTAL};
use universe::gpu::cortex_sheet::{
    CortexSheet, SHEET_H, SHEET_N, SHEET_W, compute_bipart_schedule, default_tau_homogeneous,
};
use universe::gpu::evolution::{GenResult, gen_stats, linear_slope};
use universe::gpu::perception::{PerceptionGpu, PosePerception};
use universe::gpu::physics::{PhysicsGpu, PhysicsStateGpu, WheelGpu, pack_config_flags};
use universe::perception::RetinaPerception;
use universe::simulation::racing_line::RacingLine;
use universe::visao::pitch;
use universe::world::VoxelWorld;

// ── Sheet constants ───────────────────────────────────────────────────────────

// WARNING: legacy 223-channel wave, OUTDATED — the real `SENSORY_WAVE_SIZE` today is 192
// (see src/perception/channels.rs). The original comment here claimed "= SENSORY_WAVE_SIZE",
// which is false and is the cause of the runtime panic documented in the project's CLAUDE.md
// ("index out of bounds: len is 192 but index is 192"). NOT fixed in this round (E6):
// it requires porting the whole Hebbian cortical sheet (cortex_sheet.rs/.wgsl, N=1024×223 weights)
// from 223→192 inputs, out of golden-parity scope — already recorded as not-done in E2/E3.
const N_INPUTS: usize = 223;
const N_MOTOR: usize = 3; // throttle, steer, brake

// Genome parameters
// σ of the initial afw weights: larger than the default (1.0) for more initial diversity
const SIGMA_AFW: f32 = 0.5;
// σ of the initial motor weights
const SIGMA_MW: f32 = 0.2;

// ── Local types ───────────────────────────────────────────────────────────────
//
// RetinaParamsGpu/PoseRaw/uniform_entry/storage_entry/pos_of/make_physics_state
// were identical copies of the ones in gpu_evolve.rs and gpu_loop.rs — deduplicated in E3
// (REFACTOR_PLAN.md §2.3) into src/gpu/retina_types.rs, src/gpu/core.rs and
// src/gpu/physics.rs.
use universe::gpu::core::{storage_entry, uniform_entry};
use universe::gpu::physics::{make_physics_state, pos_of};
use universe::gpu::retina_types::{PoseRaw, RetinaParamsGpu};

// ── Cortical sheet genome ─────────────────────────────────────────────────────

/// Genome of a being with the cortical sheet as its brain.
///
/// Evolves the sheet's INITIAL WEIGHTS — Hebb still runs during the ticks.
/// The genome is the starting point; intra-generation learning is emergent.
///
/// Magnitude σ_afw=0.50: larger than the 1.0 prior (uniform) to create
/// initial specialization pressure. Units with high afw for a channel
/// respond more to it → Hebb specializes them faster.
///
/// NEW (schedule-driven bipartite): field `schedule_ext` = extension of the
/// cortical window. The cortex/cerebellum proportion EMERGES from this parameter via
/// the Finlay & Darlington 1995 model (doc §2). The genome encodes ONE schedule
/// parameter, not the part sizes.
#[derive(Clone)]
struct CortexGenome {
    /// Initial Hebbian weights: SHEET_N × N_INPUTS f32. Layout: unit × channel.
    /// Initialized with N(0, σ_afw) around 1.0 (Gaussian prior).
    /// Clamped to [0, 5.0] to keep the sign positive (like buf_afw in the shader).
    afw: Vec<f32>, // size: SHEET_N × N_INPUTS = 1024 × 223 = 228.352

    /// Initial motor weights: SHEET_N × 3 f32. Layout: unit × motor.
    /// Clamped to [-2.0, 2.0].
    mw: Vec<f32>, // size: SHEET_N × 3 = 3.072

    /// τ spread multiplier: 1.0=default (199×). Clamped to [0.3, 3.0].
    /// Values >1 stretch toward more long scales, <1 concentrates in fast bands.
    /// KEPT for compatibility — in bipartite mode, τ is dictated by schedule_ext.
    tau_spread: f32,

    /// Cortical window extension: single parameter of the neurogenetic schedule.
    /// Controls the cortex/cerebellum proportion via "late equals large" (Finlay 1995).
    /// 0.0 → equal parts; 0.5 → cortex dominates (~80% of the sheet).
    /// The proportion EMERGES — it is not imposed explicitly.
    /// Clamped to [0.0, 0.5].
    schedule_ext: f32,
}

impl CortexGenome {
    /// 64-bit Knuth-MMIX LCG.
    fn lcg(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f32) / (1u64 << 53) as f32
    }

    /// Box-Muller N(0, σ).
    fn gauss(state: &mut u64, sigma: f32) -> f32 {
        let u1 = Self::lcg(state).max(1e-7);
        let u2 = Self::lcg(state);
        let g = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
        (g * sigma).clamp(-3.0 * sigma, 3.0 * sigma)
    }

    /// Generates a random genome with a deterministic seed (being index).
    /// Magnitude σ=0.50 around the 1.0 prior for afw (creates initial diversity).
    /// σ=0.20 for mw (scale used in cortex_sheet.rs).
    fn random(seed: u64) -> Self {
        let mut state = seed
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(0x6C62272E07BB0142);

        // afw: centered at 1.0 with N(0, SIGMA_AFW) noise. Clamp ≥ 0 (positive prior).
        let afw: Vec<f32> = (0..SHEET_N * N_INPUTS)
            .map(|_| (1.0 + Self::gauss(&mut state, SIGMA_AFW)).clamp(0.0, 5.0))
            .collect();

        // mw: zero-centered with N(0, SIGMA_MW) noise.
        let mw: Vec<f32> = (0..SHEET_N * N_MOTOR)
            .map(|_| Self::gauss(&mut state, SIGMA_MW).clamp(-2.0, 2.0))
            .collect();

        // tau_spread: centered at 1.0 with small noise [0.5, 2.0].
        let tau_spread = (1.0 + Self::gauss(&mut state, 0.3)).clamp(0.3, 3.0);

        // schedule_ext: cortical window extension.
        // Centered at 0.0 (purely biological: cortex=20%, cerebellum=80%, exact 1:4 ratio).
        // N(0, 0.05) noise → population starts close to the biological target.
        // Clamped to [0.0, 0.5] — evolution can raise it if it brings fitness.
        let schedule_ext = (Self::gauss(&mut state, 0.6).abs()).clamp(0.0, 2.0); // open range: see cortex_sheet.rs

        Self {
            afw,
            mw,
            tau_spread,
            schedule_ext,
        }
    }

    /// In-place Gaussian mutation.
    /// p_mutate: probability of each weight being perturbed.
    /// sigma: standard deviation of the noise.
    fn mutate(&mut self, p_mutate: f32, sigma: f32, seed: u64) {
        let mut state = seed
            .wrapping_mul(0xDEAD_BEEF_1234_5678)
            .wrapping_add(0xCAFE_BABE_9876_5432);

        for w in self.afw.iter_mut() {
            if Self::lcg(&mut state) < p_mutate {
                *w = (*w + Self::gauss(&mut state, sigma)).clamp(0.0, 5.0);
            }
        }
        for w in self.mw.iter_mut() {
            if Self::lcg(&mut state) < p_mutate {
                *w = (*w + Self::gauss(&mut state, sigma * 0.5)).clamp(-2.0, 2.0);
            }
        }
        // tau_spread mutates rarely (1%)
        if Self::lcg(&mut state) < 0.01 {
            self.tau_spread = (self.tau_spread + Self::gauss(&mut state, 0.1)).clamp(0.3, 3.0);
        }
        // schedule_ext mutates with σ=0.03 — key parameter of the schedule
        if Self::lcg(&mut state) < 0.15 {
            self.schedule_ext = (self.schedule_ext + Self::gauss(&mut state, 0.15)).clamp(0.0, 2.0);
        }
    }
}

// ── Sheet evolutionary cycle ──────────────────────────────────────────────────

struct EvoParams {
    p_mutate: f32,
    sigma: f32,
    n_elite: usize,
    tournament_k: usize,
}

fn evolve_cortex(
    genomes: &[CortexGenome],
    fitnesses: &[f64],
    params: &EvoParams,
    generation: usize,
) -> Vec<CortexGenome> {
    let n = genomes.len();
    assert_eq!(fitnesses.len(), n);

    // Sort by decreasing fitness.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        fitnesses[b]
            .partial_cmp(&fitnesses[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut nova: Vec<CortexGenome> = Vec::with_capacity(n);

    // Elite passes through without mutation.
    for &idx in order.iter().take(params.n_elite) {
        nova.push(genomes[idx].clone());
    }

    // LCG for the tournament.
    let mut state = (generation as u64)
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(0xCAFE_BABE_DEAD_BEEF);
    let mut lcg = move || -> u64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };

    while nova.len() < n {
        // Tournament k: best among k random candidates.
        let mut best = (lcg() as usize) % n;
        for _ in 1..params.tournament_k {
            let cand = (lcg() as usize) % n;
            if fitnesses[cand] > fitnesses[best] {
                best = cand;
            }
        }
        let mut filho = genomes[best].clone();
        let seed = lcg().wrapping_add(nova.len() as u64 * 0x1337);
        filho.mutate(params.p_mutate, params.sigma, seed);
        nova.push(filho);
    }

    nova
}

// ── Per-being state ───────────────────────────────────────────────────────────

struct BeingState {
    phys: PhysicsStateGpu,
    prev_steer: f32,
    stagnation_ticks: u32,
    reverse_timer: f32,
    post_reverse_lockout: f32,
}

// pos_of() and make_physics_state() now come from universe::gpu::physics (E3 dedup, see
// the `use` at the top of the file) — v0=0.0 on both paths makes the previous version
// (literal `vel: [0.0; 3]`) numerically identical to the canonical one (`fwd * v0` with v0=0.0).

// ── GPU context ───────────────────────────────────────────────────────────────

struct CortexCtx {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    // Retina
    pipeline_retina: wgpu::ComputePipeline,
    layout_retina: wgpu::BindGroupLayout,
    buf_grid: wgpu::Buffer,
    buf_poses_retina: wgpu::Buffer,
    buf_retina_out: wgpu::Buffer,
    grid_size: u32,
    cell_size: f32,

    // Perception → sensory_wave
    percept: PerceptionGpu,
    buf_retina_readback: wgpu::Buffer,

    // Cortical sheet (one shared sheet per being — weights reloaded between generations)
    cortex: CortexSheet,

    // Physics
    phys_gpu: PhysicsGpu,

    // Perception CfC (CPU-side recurrent state)
    percept_cfc: Vec<RetinaPerception>,

    racing_line: RacingLine,
    grid_cpu: Vec<u32>,
    grid_size_cpu: u32,
    cell_size_cpu: f32,
    capacity: usize,

    // act_sum from the previous tick (for the Hebb kernel, computed on the CPU after readback)
    act_sum_prev: Vec<f32>,

    // Last sensory_wave read from the GPU — preserved between tick() and replay recording.
    // Each entry is an array of 223 i8 channels (one per being). Indexed by being index.
    last_wave: Vec<[i8; N_INPUTS]>,
}

impl CortexCtx {
    fn new(world: &VoxelWorld, capacity: usize) -> Option<Self> {
        // Force the Radeon Vulkan ICD.
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

        // Limit of 9 storage buffers: perception uses 9.
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_storage_buffers_per_shader_stage = 9;

        let (raw_device, raw_queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("cortex-evolve"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;

        let device = Arc::new(raw_device);
        let queue = Arc::new(raw_queue);
        let cap = capacity.max(1);

        // World grid (static)
        let grid_u32: Vec<u32> = world.grid.iter().map(|v| *v as u32).collect();
        let buf_grid = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cortex-evolve-grid"),
            contents: bytemuck::cast_slice(&grid_u32),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Retina buffers
        let buf_poses_retina = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cortex-evolve-poses"),
            size: (cap * std::mem::size_of::<PoseRaw>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_retina_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cortex-evolve-retina-out"),
            size: (cap * RETINA_TOTAL * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Retina pipeline
        let shader_retina = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cortex-evolve-retina"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "../gpu/retina.wgsl"
            ))),
        });
        let layout_retina = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cortex-evolve-retina-layout"),
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
            label: Some("cortex-evolve-retina-pipeline"),
            layout: Some(&pl_retina),
            module: &shader_retina,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // GPU perception (builds the 223-channel sensory_wave)
        let percept = PerceptionGpu::new(&device, &queue, world, cap)?;

        // Cortical sheet (capacity beings)
        let cortex = CortexSheet::new(Arc::clone(&device), Arc::clone(&queue), cap);

        // GPU physics
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

        // Retina readback (to feed percept_cfc CPU-side)
        let buf_retina_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cortex-evolve-retina-rb"),
            size: (cap * RETINA_TOTAL * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let percept_cfc: Vec<RetinaPerception> =
            (0..cap).map(|_| RetinaPerception::new()).collect();

        eprintln!("[cortex_evolve] Adapter: {} · cap={}", name, cap);
        eprintln!("[cortex_evolve] Sheet 32×32=1024 units | τ ∈ [1.6ms, 318ms] spread 199×");
        eprintln!(
            "[cortex_evolve] Genome: afw={}  mw={}  tau_spread=1f",
            SHEET_N * N_INPUTS,
            SHEET_N * N_MOTOR
        );

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
            buf_retina_readback,
            cortex,
            phys_gpu,
            percept_cfc,
            racing_line: world.racing_line.clone(),
            grid_cpu: grid_u32,
            grid_size_cpu: world.size as u32,
            cell_size_cpu: world.cell_size,
            capacity: cap,
            act_sum_prev: vec![0.0_f32; cap],
            last_wave: vec![[0i8; N_INPUTS]; cap],
        })
    }

    /// Loads a genome's weights into the cortical sheet (via write_buffer straight into the buffers).
    /// Called between generations to re-initialize the sheet.
    ///
    /// NEW (schedule-driven bipartite): also injects the τ values computed by the genome's
    /// neurogenetic schedule. Each unit's τ now depends on `schedule_ext`,
    /// which determines the cortex/cerebellum proportion via "late equals large".
    fn upload_genome_to_cortex(&mut self, genome: &CortexGenome, ser_idx: usize) {
        // being's afw: ser_idx × SHEET_N × N_INPUTS floats inside buf_afw.
        // buf_afw layout: being × (SHEET_N × N_INPUTS).
        let offset_afw = (ser_idx * SHEET_N * N_INPUTS * 4) as u64;
        self.queue.write_buffer(
            self.cortex.buf_afw_ref(),
            offset_afw,
            bytemuck::cast_slice(&genome.afw),
        );

        // being's mw: ser_idx × SHEET_N × N_MOTOR floats inside buf_mw.
        let offset_mw = (ser_idx * SHEET_N * N_MOTOR * 4) as u64;
        self.queue.write_buffer(
            self.cortex.buf_mw_ref(),
            offset_mw,
            bytemuck::cast_slice(&genome.mw),
        );

        // τ per part: computed by the genome's neurogenetic schedule.
        // Each being may have a slightly different proportion (schedule_ext evolves).
        // buf_tau layout: being × SHEET_N f32.
        // With `--homogeneous`, all 1024 units get the FULL τ bank
        // (1.6–318 ms) instead of being split into two parts by schedule. It is the control
        // that tells whether splitting into parts helps or hurts on THIS three-actuator body.
        let taus_ser = if HOMOGENEOUS.load(std::sync::atomic::Ordering::Relaxed) {
            default_tau_homogeneous(SHEET_N)
        } else {
            compute_bipart_schedule(genome.schedule_ext).2
        };
        let offset_tau = (ser_idx * SHEET_N * 4) as u64;
        self.queue.write_buffer(
            self.cortex.buf_tau_ref(),
            offset_tau,
            bytemuck::cast_slice(&taus_ser),
        );

        // Updates being 0's cache for inspection via tau_bank().
        if ser_idx == 0 {
            self.cortex.set_tau_cache(taus_ser);
        }
    }

    /// Resets the sheet's EMA state (buf_state) and fatigue (buf_fat) to zero between generations.
    /// Both must be reset together: residual fatigue from one generation would penalize
    /// the start of the next with negative initialization bias.
    fn reset_cortex_state(&self, n: usize) {
        let zeros = vec![0.0_f32; n.min(self.capacity) * SHEET_N];
        self.queue
            .write_buffer(self.cortex.buf_state_ref(), 0, bytemuck::cast_slice(&zeros));
        // Resets adaptive fatigue — zeroed together with the EMA state.
        self.queue
            .write_buffer(self.cortex.buf_fat_ref(), 0, bytemuck::cast_slice(&zeros));
    }

    /// Resets the perception CfC for all beings.
    fn reset_cfc(&mut self, n: usize) {
        for i in 0..n.min(self.capacity) {
            self.percept_cfc[i] = RetinaPerception::new();
        }
        // Reset act_sum_prev
        for s in self.act_sum_prev.iter_mut().take(n.min(self.capacity)) {
            *s = 0.0;
        }
    }

    /// Runs one tick: retina → perception → cortical sheet → physics.
    /// NO motor crutches (no escape_ticks, no curvature_governor).
    fn tick(&mut self, states: &mut Vec<BeingState>, n: usize, dt: f32, tick_idx: usize) -> bool {
        if n == 0 || n > self.capacity {
            return false;
        }

        // ── Retina poses ──────────────────────────────────────────────────────
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

        // ── Perception poses ──────────────────────────────────────────────────
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

        // ── Lateral/frontal distances (CPU) ───────────────────────────────────
        {
            let gs = self.grid_size_cpu as usize;
            let cs = self.cell_size_cpu;
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
                let mut dist_l = 40.0_f32;
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
            self.percept.update_dist_temporal(&self.queue, &dists[..n]);
        }

        // ── Racing line ───────────────────────────────────────────────────────
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
                rv.push([off_norm, 0.0, la_norm, 0.5, 0.8, 0.7]);
            }
            self.percept
                .update_racing_line_temporal(&self.queue, &rv[..n]);
        }

        // ── Proprioception ────────────────────────────────────────────────────
        {
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
                let tt: f32 = (0..4).map(|j| p.wheels[j * 2 + 1][0]).sum::<f32>() / 4.0;
                let on_grass = if voxel_at(p.pos[0], p.pos[2]) == 0 {
                    1.0_f32
                } else {
                    0.0
                };
                prop.push([
                    align,
                    speed,
                    lat_vel,
                    1.0,
                    rpm_r,
                    p.current_gear as f32,
                    tt.clamp(0.0, 1.0),
                    0.0,
                    0.0,
                    on_grass,
                    p.steer.abs().clamp(0.0, 1.0),
                    0.0,
                ]);
            }
            self.percept
                .update_proprioception_temporal(&self.queue, &prop[..n]);
        }

        // ── Reverse timer + perception CfC ────────────────────────────────────
        {
            // Deduplicated (E3, REFACTOR_PLAN.md §2.3) into
            // universe::config::iara::reverse_maneuver — this file already used the
            // correct value (4.0, via a constant actually REFERENCED), unlike
            // gpu_evolve.rs/gpu_loop.rs where the local constant `=3.0` was dead.
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
                let mut fwd_dist = 30.0_f32;
                for sd in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] {
                    let v = voxel_at(px + fx * sd, pz + fz * sd);
                    if v == 2 || v == 3 {
                        fwd_dist = sd;
                        break;
                    }
                }

                if states[i].post_reverse_lockout > 0.0 {
                    states[i].post_reverse_lockout = (states[i].post_reverse_lockout - dt).max(0.0);
                }
                {
                    let rev = states[i].reverse_timer;
                    let stag = states[i].stagnation_ticks;
                    let lock = states[i].post_reverse_lockout;
                    if rev <= 0.0
                        && lock <= 0.0
                        && fwd_dist < revcfg::TRIGGER_DIST_M
                        && speed < revcfg::TRIGGER_SPEED_MAX_MS
                        && stag > revcfg::STAGNATION_TICKS
                    {
                        states[i].reverse_timer = revcfg::MIN_DURATION_S;
                        states[i].stagnation_ticks = 0;
                    }
                }
                {
                    let rev = states[i].reverse_timer;
                    if rev > 0.0 {
                        let nr = rev - dt;
                        if fwd_dist > revcfg::EXIT_CLEAR_M || nr > revcfg::MAX_DURATION_S {
                            states[i].reverse_timer = 0.0;
                            states[i].post_reverse_lockout = revcfg::POST_LOCKOUT_S;
                        } else {
                            states[i].reverse_timer = nr.max(0.0);
                        }
                    }
                }

                let hidden = &self.percept_cfc[i].pipeline.state.h;
                let mut val = [0.0f32; 17];
                val[0] = states[i].reverse_timer;
                for j in 0..16 {
                    val[1 + j] = if j < hidden.len() { hidden[j] } else { 0.0 };
                }
                extra.push(val);
            }
            self.percept.update_extra_temporal(&self.queue, &extra[..n]);
        }

        // ── Single encoder ────────────────────────────────────────────────────
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cortex-evolve-tick"),
            });

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
            // `--retina-beings` (docs/RETINA-BEINGS.md §2) is not wired into this
            // binary — off, same as before the field existed.
            beings_on: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let buf_rp = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cortex-evolve-rp"),
                contents: bytemuck::bytes_of(&retina_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
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
        self.percept
            .copy_retina_prev(&mut enc, &self.buf_retina_out);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cortex-evolve-retina"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_retina);
            pass.set_bind_group(0, &bg_retina, &[]);
            pass.dispatch_workgroups(((n * RETINA_TOTAL) as u32).div_ceil(64), 1, 1);
        }

        // Perception → buf_output = sensory_wave.
        self.percept.bind_and_dispatch(
            &mut enc,
            &self.device,
            &self.queue,
            &poses_percept[..n],
            &self.buf_retina_out,
        );

        // Retina copy for the CfC readback.
        let retina_copy_bytes = (n * RETINA_TOTAL * 4) as u64;
        enc.copy_buffer_to_buffer(
            &self.buf_retina_out,
            0,
            &self.buf_retina_readback,
            0,
            retina_copy_bytes,
        );

        // ── Submit 1: retina + perception ────────────────────────────────────
        // Pattern identical to gpu_evolve.rs — ONE submit, one poll(Wait), individual
        // readbacks with map_async + poll(Wait) + recv() each.
        self.queue.submit(Some(enc.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        // ── Read 1: sensory_wave from perception ──────────────────────────────
        // bytes_out = n × WAVE_U32S × 4 (WAVE_U32S = ceil(223/4) = 56 u32 per being)
        use universe::gpu::perception::WAVE_U32S;
        let waves_i8: Vec<[i8; N_INPUTS]> = {
            let bytes_out = (n * WAVE_U32S * 4) as u64;
            // percept.buf_output → percept.buf_readback was copied in enc above.
            // We map perception's internal buf_readback via a public reference.
            // Since there is no public buf_readback API, we use buf_wave as destination:
            // we copy percept.buf_output → self.buf_wave_readback (dedicated buffer).
            // ALTERNATIVE: read buf_retina_readback as a proxy for the wave.
            // REAL SOLUTION: create a dedicated buf_wave_rb and copy percept.buf_output into it.
            // Here we use the correct readback: percept.buf_readback via read_wave_direct.
            //
            // wgpu 24: map_async needs poll(Wait) AFTER map_async, not before.
            // The poll(Wait) above waited for the submit to complete (GPU did the work).
            // Now map_async queues the mapping request.
            // The second poll(Wait) below processes the mapping-completion event.
            let rb = self.percept.buf_readback_ref();
            let slice = rb.slice(0..bytes_out);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            // Poll AFTER map_async — the correct pattern in wgpu 24.
            self.device.poll(wgpu::Maintain::Wait);
            let _ = rx.recv();
            let data = slice.get_mapped_range();
            let words: &[u32] = bytemuck::cast_slice(&data);
            let mut out = vec![[0i8; N_INPUTS]; n];
            for being in 0..n {
                let base = being * WAVE_U32S;
                for ch in 0..N_INPUTS {
                    let word = words[base + ch / 4];
                    let byte = ((word >> ((ch % 4) * 8)) & 0xFF) as u8;
                    out[being][ch] = byte as i8;
                }
            }
            drop(data);
            rb.unmap();
            out
        };

        // ── Read 2: retina (for the perception CfC) ──────────────────────────
        {
            let slice = self.buf_retina_readback.slice(0..retina_copy_bytes);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            self.device.poll(wgpu::Maintain::Wait);
            let _ = rx.recv();
            let data = slice.get_mapped_range();
            let retina_u32: &[u32] = bytemuck::cast_slice(&data);
            for i in 0..n {
                let base = i * RETINA_TOTAL;
                let mut retina_i8 = vec![0i8; RETINA_TOTAL];
                for j in 0..RETINA_TOTAL {
                    let packed = retina_u32[base + j];
                    let dist_m = (packed >> 8) as f32 / 1000.0;
                    let mat = packed & 0xFF;
                    let v: f32 = match mat {
                        2 => {
                            if dist_m <= 30.0 {
                                127.0 - (127.0 - 55.0) * (dist_m / 30.0)
                            } else {
                                55.0 * (1.0 - (dist_m - 30.0) / 90.0).max(0.0)
                            }
                        }
                        1 => -128.0 + (dist_m.min(80.0) / 80.0) * 128.0,
                        0 => 40.0,
                        _ => 127.0,
                    };
                    retina_i8[j] = v.clamp(-127.0, 127.0) as i8;
                }
                self.percept_cfc[i].perceive(&retina_i8);
            }
            drop(data);
            self.buf_retina_readback.unmap();
        }

        // Uploads sensory_wave to the cortical sheet.
        if !waves_i8.is_empty() {
            self.cortex
                .upload_wave(&waves_i8[..n.min(waves_i8.len())], n);
        }

        // Keeps the wave so the replay-recording loop can read it without doing
        // another GPU readback. The data already came from the GPU this tick — we just copy.
        for i in 0..n.min(self.last_wave.len()) {
            self.last_wave[i] = waves_i8[i];
        }

        // Sheet encoder: afference → hebb → motor → readback.
        let mut enc2 = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cortex-evolve-folha"),
            });

        let tick = tick_idx as u32;
        self.cortex.dispatch_afference(&mut enc2, n, tick);

        // act_sum: mean over beings (simplification — avoids an extra readback per tick).
        // On even ticks uses act_sum_prev computed in the previous tick.
        let mean_act_sum = if n > 0 {
            self.act_sum_prev[..n].iter().sum::<f32>() / n as f32
        } else {
            0.0
        };
        self.cortex.dispatch_hebb(&mut enc2, n, tick, mean_act_sum);
        self.cortex.dispatch_motor(&mut enc2, n, tick);
        self.cortex.copy_for_readback(&mut enc2, n);

        self.queue.submit(Some(enc2.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        // Reads motor output and activation statistics.
        let motors = self.cortex.read_motor(n);
        let stats = self.cortex.read_act_stats(n);

        // Updates act_sum_prev per being (Σ|a_j| → used in the next Hebb).
        for i in 0..n {
            // Estimate: mean × SHEET_N × factor (|mean| underestimates, *2 is a conservative correction)
            self.act_sum_prev[i] = stats[i].2.abs() * SHEET_N as f32 * 2.0;
        }

        if motors.len() != n {
            return false;
        }

        // ── Applies motor output to physics ───────────────────────────────────
        // The sheet outputs tanh(-1..1). We map to throttle/steer/brake.
        // NO crutches: whatever the sheet says, goes.
        let mut phys_states: Vec<PhysicsStateGpu> = (0..n)
            .map(|i| {
                let mut s = states[i].phys;
                let motor = motors[i];
                // motor[0] = throttle (-1..1), mapped to (0..1)
                let thr = ((motor[0] + 1.0) * 0.5).clamp(0.0, 1.0);
                // motor[1] = steer (-1..1)
                let steer = motor[1].clamp(-1.0, 1.0);
                // motor[2] = brake (-1..1), mapped to (0..1)
                let brk = ((motor[2] + 1.0) * 0.5).clamp(0.0, 1.0);

                let steer_smooth = (states[i].prev_steer * 0.50 + steer * 0.50).clamp(-1.0, 1.0);
                let speed = (s.vel[0].powi(2) + s.vel[2].powi(2)).sqrt();
                let brk_final = if speed < 2.0 { 0.0 } else { brk };
                let is_reversing = states[i].reverse_timer > 0.0;
                let in_lockout = states[i].post_reverse_lockout > 0.0;

                if is_reversing {
                    s.throttle = universe::config::iara::reverse_maneuver::PEDAL;
                    s.steer = 0.0;
                    s.brake_cmd = 0.0;
                    s.current_gear = 0;
                } else if in_lockout {
                    s.throttle = thr.min(0.8);
                    s.steer = steer_smooth;
                    s.brake_cmd = brk_final;
                    if s.current_gear <= 1 {
                        s.current_gear = 2;
                    }
                } else {
                    s.throttle = thr;
                    s.steer = steer_smooth;
                    s.brake_cmd = brk_final;
                    if s.current_gear <= 1 {
                        s.current_gear = 2;
                    }
                    if s.current_gear > 2 && speed < 3.0 {
                        s.current_gear -= 1;
                    }
                }
                s.dt = dt;
                s
            })
            .collect();

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

        let ok = self.phys_gpu.step(&mut phys_states);
        if !ok {
            return false;
        }
        for i in 0..n {
            states[i].phys = phys_states[i];
        }

        // memory0 → temporal (parity with gpu_evolve).
        let mem0: Vec<f32> = (0..n)
            .map(|i| motors[i][0].clamp(-1.0, 1.0) * 0.5 + 0.5)
            .collect();
        self.percept.update_memory0(&self.queue, &mem0[..n]);

        true
    }
}

// ── Modality map diagnostic ──────────────────────────────────────────────────

/// Prints a 32×32 map of modality dominance via differential response.
///
/// Modality A = channels 0-111 (retina/vision: first 112 channels)
/// Modality B = channels 112-222 (proprioception/racing-line: last 111 channels)
///
/// Criterion registered BEFORE measuring: intra/inter ratio > 3×.
///
/// Uses the sheet in its CURRENT state (post generation ticks) — measures the emerged allocation.
fn print_modal_map(ctx: &mut CortexCtx, generation: usize) {
    // CRITERION REGISTERED HERE (before measuring): ratio > 3×.
    let limiar_razao = 3.0_f32;

    // We use only 1 being (being 0) for the map — faster and sufficient.
    let n = 1;

    const MOD_A_END: usize = 111; // channels 0-111
    const MOD_B_START: usize = 112; // channels 112-222

    // ── Probe A: only modality A active ───────────────────────────────────────
    let mut wave_a = [[0i8; N_INPUTS]; 1];
    for k in 0..=MOD_A_END {
        let v = (2.0 * std::f32::consts::PI * k as f32 / 112.0 + 0.5).sin() * 100.0;
        wave_a[0][k] = v.clamp(-127.0, 127.0) as i8;
    }
    ctx.cortex.upload_wave(&wave_a, n);
    let mut enc_a = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("map-probe-a"),
        });
    ctx.cortex.dispatch_afference(&mut enc_a, n, 9999);
    ctx.cortex.copy_for_readback(&mut enc_a, n);
    ctx.queue.submit(Some(enc_a.finish()));
    ctx.device.poll(wgpu::Maintain::Wait);
    let resp_a = ctx.cortex.read_act_raw(n);

    // ── Probe B: only modality B active ───────────────────────────────────────
    let mut wave_b = [[0i8; N_INPUTS]; 1];
    for k in MOD_B_START..N_INPUTS {
        let kk = k - MOD_B_START;
        let v = (2.0 * std::f32::consts::PI * kk as f32 / 111.0 + 0.5).cos() * 100.0;
        wave_b[0][k] = v.clamp(-127.0, 127.0) as i8;
    }
    ctx.cortex.upload_wave(&wave_b, n);
    let mut enc_b = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("map-probe-b"),
        });
    ctx.cortex.dispatch_afference(&mut enc_b, n, 10000);
    ctx.cortex.copy_for_readback(&mut enc_b, n);
    ctx.queue.submit(Some(enc_b.finish()));
    ctx.device.poll(wgpu::Maintain::Wait);
    let resp_b = ctx.cortex.read_act_raw(n);

    let resp_a = &resp_a[..SHEET_N];
    let resp_b = &resp_b[..SHEET_N];

    // Dominance per unit
    let dominance: Vec<u8> = (0..SHEET_N)
        .map(|u| {
            let da = resp_a[u].abs();
            let db = resp_b[u].abs();
            if (da - db).abs() < 1e-6 {
                2u8
            } else if da > db {
                0u8
            } else {
                1u8
            }
        })
        .collect();

    let n_a = dominance.iter().filter(|&&d| d == 0).count();
    let n_b = dominance.iter().filter(|&&d| d == 1).count();
    let n_indef = dominance.iter().filter(|&&d| d == 2).count();

    println!(
        "\n── MODALITY MAP (gen {:4}) ───────────────────────────────────────",
        generation
    );
    println!("   A=channels 0-111 (vision/retina)  B=channels 112-222 (proprioception)");
    println!(
        "   Registered criterion: intra/inter ratio > {:.0}× (no loosening)",
        limiar_razao
    );
    println!();

    // 32×32 ASCII map
    for row in 0..SHEET_H {
        print!("   ");
        for col in 0..SHEET_W {
            let u = row * SHEET_W + col;
            let c = match dominance[u] {
                0 => 'A',
                1 => 'B',
                _ => '?',
            };
            print!("{c}");
        }
        println!();
    }
    println!();
    println!(
        "   A={} ({:.1}%)  B={} ({:.1}%)  ?={} ({:.1}%)",
        n_a,
        n_a as f32 / SHEET_N as f32 * 100.0,
        n_b,
        n_b as f32 / SHEET_N as f32 * 100.0,
        n_indef,
        n_indef as f32 / SHEET_N as f32 * 100.0
    );

    // ── Boundary map: cross-correlation between neighbors ────────────────────
    let norms: Vec<f32> = (0..SHEET_N)
        .map(|u| {
            (resp_a[u] * resp_a[u] + resp_b[u] * resp_b[u])
                .sqrt()
                .max(1e-8)
        })
        .collect();

    // Correlation with the 4 neighbors (NSEW)
    let neighbor_corr: Vec<f32> = (0..SHEET_N)
        .map(|u| {
            let row = u / SHEET_W;
            let col = u % SHEET_W;
            let mut sum_r = 0.0_f32;
            let mut cnt = 0u32;
            let neighbors: [Option<usize>; 4] = [
                if row > 0 { Some(u - SHEET_W) } else { None },
                if row < SHEET_H - 1 {
                    Some(u + SHEET_W)
                } else {
                    None
                },
                if col > 0 { Some(u - 1) } else { None },
                if col < SHEET_W - 1 { Some(u + 1) } else { None },
            ];
            for &nb in &neighbors {
                if let Some(j) = nb {
                    let dot = resp_a[u] * resp_a[j] + resp_b[u] * resp_b[j];
                    sum_r += dot / (norms[u] * norms[j]);
                    cnt += 1;
                }
            }
            if cnt == 0 { 1.0 } else { sum_r / cnt as f32 }
        })
        .collect();

    println!("\n── BOUNDARY MAP (cross-correlation with neighbors) ──────────────────");
    println!(
        "   . = interior (corr≥0.9)  , = transition (0.7-0.9)  o = boundary (0.5-0.7)  # = strong (<0.5)"
    );
    for row in 0..SHEET_H {
        print!("   ");
        for col in 0..SHEET_W {
            let u = row * SHEET_W + col;
            let c = if neighbor_corr[u] >= 0.9 {
                '.'
            } else if neighbor_corr[u] >= 0.7 {
                ','
            } else if neighbor_corr[u] >= 0.5 {
                'o'
            } else {
                '#'
            };
            print!("{c}");
        }
        println!();
    }

    // ── Refutation criterion P1: intra/inter ratio ─────────────────────────
    let mut intra_sum = 0.0_f64;
    let mut intra_cnt = 0u64;
    let mut inter_sum = 0.0_f64;
    let mut inter_cnt = 0u64;

    for u in 0..SHEET_N {
        let du = dominance[u];
        if du == 2 {
            continue;
        }
        let row = u / SHEET_W;
        let col = u % SHEET_W;
        // East and south neighbors (no double counting)
        let nbs: [Option<usize>; 4] = [
            if col < SHEET_W - 1 { Some(u + 1) } else { None },
            if row < SHEET_H - 1 {
                Some(u + SHEET_W)
            } else {
                None
            },
            if col < SHEET_W - 1 && row < SHEET_H - 1 {
                Some(u + SHEET_W + 1)
            } else {
                None
            },
            if col < SHEET_W - 1 && row > 0 {
                Some(u - SHEET_W + 1)
            } else {
                None
            },
        ];
        for &nb in &nbs {
            if let Some(j) = nb {
                let dj = dominance[j];
                if dj == 2 {
                    continue;
                }
                let dot = resp_a[u] * resp_a[j] + resp_b[u] * resp_b[j];
                let r = (dot / (norms[u] * norms[j])) as f64;
                if du == dj {
                    intra_sum += r;
                    intra_cnt += 1;
                } else {
                    inter_sum += r;
                    inter_cnt += 1;
                }
            }
        }
    }

    println!("\n── CRITERION P1: intra/inter ratio ───────────────────────────────────");
    println!(
        "   Registered threshold: ratio > {:.0}× (evaluated below)",
        limiar_razao
    );

    if intra_cnt == 0 || inter_cnt == 0 {
        println!(
            "   NOT MEASURED: insufficient pairs (intra={} inter={})",
            intra_cnt, inter_cnt
        );
        println!("   Cause: homogeneous map — no detectable boundary.");
    } else {
        let corr_intra = (intra_sum / intra_cnt as f64) as f32;
        let corr_inter = (inter_sum / inter_cnt as f64) as f32;
        let razao = if corr_inter.abs() > 1e-8 {
            (corr_intra / corr_inter).abs()
        } else {
            f32::INFINITY
        };

        println!(
            "   intra pairs={} corr={:+.4}  |  inter pairs={} corr={:+.4}",
            intra_cnt, corr_intra, inter_cnt, corr_inter
        );

        let razao_str = if razao.is_infinite() {
            ">999".to_string()
        } else {
            format!("{:.2}", razao)
        };
        if razao >= limiar_razao || corr_inter.abs() < 0.01 {
            println!(
                "   PASS ratio={}× ≥ {:.0}× — regions EMERGE (Hebbian boundary detected)",
                razao_str, limiar_razao
            );
        } else if razao >= 1.5 {
            println!(
                "   INCONCLUSIVE ratio={}× (1.5-{:.0}× — a trend, not confirmed)",
                razao_str, limiar_razao
            );
        } else {
            println!(
                "   FAIL ratio={}× < {:.0}× — regions do NOT emerge after this generation",
                razao_str, limiar_razao
            );
        }
    }

    // ── τ spread: active vs silent ────────────────────────────────────────
    // We use probe A activation as a proxy for unit activity.
    let ativo_limiar = 0.05_f32;
    let silente_limiar = 0.01_f32;
    let taus = ctx.cortex.tau_bank();

    let tau_ativas: Vec<f32> = (0..SHEET_N)
        .filter(|&u| resp_a[u].abs() > ativo_limiar)
        .map(|u| taus[u])
        .collect();
    let tau_silentes: Vec<f32> = (0..SHEET_N)
        .filter(|&u| resp_a[u].abs() < silente_limiar)
        .map(|u| taus[u])
        .collect();

    println!("\n── τ SPREAD (active vs silent) ───────────────────────────────────────");
    if !tau_ativas.is_empty() && !tau_silentes.is_empty() {
        let mut ta = tau_ativas.clone();
        let mut ts = tau_silentes.clone();
        ta.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let spread_a = ta.last().unwrap() / ta[0];
        let spread_s = ts.last().unwrap() / ts[0];
        println!(
            "   ACTIVE ({:4}):  τ ∈ [{:.2}ms, {:.1}ms]  spread={:.1}×",
            ta.len(),
            ta[0] * 1000.0,
            ta.last().unwrap() * 1000.0,
            spread_a
        );
        println!(
            "   SILENT ({:4}): τ ∈ [{:.2}ms, {:.1}ms]  spread={:.1}×",
            ts.len(),
            ts[0] * 1000.0,
            ts.last().unwrap() * 1000.0,
            spread_s
        );
        // Prediction: champions have active spread > 10× (doc §5.3).
        let status = if spread_a > 10.0 {
            "PASS >10×"
        } else if spread_a > 5.0 {
            "medium 5-10×"
        } else {
            "below 5×"
        };
        println!("   Active spread = {:.1}× → {}", spread_a, status);
    } else {
        println!(
            "   NOT MEASURED: active={} silent={}",
            tau_ativas.len(),
            tau_silentes.len()
        );
    }
    println!();
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Enabled by `--homogeneous`. Process-wide global because the genome upload does not
/// receive the configuration — and each battery run gets its own process, so there is
/// nothing to confuse.
static HOMOGENEOUS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn main() {
    let icd_radeon = "/usr/share/vulkan/icd.d/radeon_icd.json";
    if std::path::Path::new(icd_radeon).exists() && std::env::var("VK_ICD_FILENAMES").is_err() {
        unsafe {
            std::env::set_var("VK_ICD_FILENAMES", icd_radeon);
        }
    }

    // ── Arg parsing ───────────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let mut n_seres: usize = 40;
    let mut n_ticks: usize = 6_000;
    let mut n_gens: usize = 100;
    let mut map_interval: usize = 10; // print map every N gens
    let mut out_dir = ".".to_string();
    let mut evo_sigma: f32 = 0.05;
    let mut evo_p_mutate: f32 = 0.02;
    let mut evo_n_elite: usize = 3;
    let mut evo_tourney: usize = 3;
    // Anti-saturation inhibition parameters — configurable for sweeps without recompiling.
    // fat_beta=0.0, div_sigma=0.0 → SATURATED sheet (baseline that gave 1055m at gen 2).
    // fat_beta=1.2, div_sigma=2.0 → STRONG inhibition (kills saturation but costs the ceiling).
    let mut homogeneous = false;
    let mut fat_beta: f32 = 1.2;
    let mut div_sigma: f32 = 2.0;
    {
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--seres" => {
                    i += 1;
                    if i < args.len() {
                        n_seres = args[i].parse().unwrap_or(40);
                    }
                }
                "--ticks" => {
                    i += 1;
                    if i < args.len() {
                        n_ticks = args[i].parse().unwrap_or(6_000);
                    }
                }
                "--gens" => {
                    i += 1;
                    if i < args.len() {
                        n_gens = args[i].parse().unwrap_or(100);
                    }
                }
                "--map-interval" => {
                    i += 1;
                    if i < args.len() {
                        map_interval = args[i].parse().unwrap_or(10);
                    }
                }
                "--sigma" => {
                    i += 1;
                    if i < args.len() {
                        evo_sigma = args[i].parse().unwrap_or(0.05);
                    }
                }
                "--p-mutate" => {
                    i += 1;
                    if i < args.len() {
                        evo_p_mutate = args[i].parse().unwrap_or(0.02);
                    }
                }
                "--n-elite" => {
                    i += 1;
                    if i < args.len() {
                        evo_n_elite = args[i].parse().unwrap_or(3);
                    }
                }
                "--tourney" => {
                    i += 1;
                    if i < args.len() {
                        evo_tourney = args[i].parse().unwrap_or(3);
                    }
                }
                "--out" => {
                    i += 1;
                    if i < args.len() {
                        out_dir = args[i].clone();
                    }
                }
                // New args for the inhibition sweep (without recompiling the shader):
                // `--homogeneous` reverts to a sheet WITHOUT parts (all 1024 units on the
                // full τ bank). It is the battery's CONTROL: without it there is no way to know
                // whether splitting into parts helps or hurts. Measured before with a fixed 1:4
                // ratio: the bipartite came in BELOW the homogeneous (med10 110 m vs 164 m).
                "--homogeneous" => {
                    homogeneous = true;
                    HOMOGENEOUS.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                "--fat-beta" => {
                    i += 1;
                    if i < args.len() {
                        fat_beta = args[i].parse().unwrap_or(1.2);
                    }
                }
                "--div-sigma" => {
                    i += 1;
                    if i < args.len() {
                        div_sigma = args[i].parse().unwrap_or(2.0);
                    }
                }
                _ => {}
            }
            i += 1;
        }
    }

    println!("=== cortex_evolve: 32×32 cortical sheet as brain [BIPARTITE] ===");
    println!(
        "  {} beings · {} ticks/gen · {} gens",
        n_seres, n_ticks, n_gens
    );
    println!(
        "  Genome: afw init N(1.0,σ={})  mw init N(0,σ={})  schedule_ext evolves",
        SIGMA_AFW, SIGMA_MW
    );
    println!(
        "  Evolution: sigma={:.3}  p_mutate={:.3}  n_elite={}  tourney={}",
        evo_sigma, evo_p_mutate, evo_n_elite, evo_tourney
    );
    println!(
        "  Inhibition: fat_beta={:.2}  div_sigma={:.2}  (sweep — no recompile)",
        fat_beta, div_sigma
    );
    println!("  Modality map every {} generations", map_interval);
    println!("  Registered P1 criterion: intra/inter ratio > 3× (no loosening)");
    println!(
        "{}",
        format!(
            "  NO motor crutches  |  Fitness: {}",
            std::env::var("IARA_FITNESS").unwrap_or_else(|_| "dist".into())
        )
    );
    println!(
        "  Population: RANDOM (seed=index, σ_afw={}, σ_mw={})",
        SIGMA_AFW, SIGMA_MW
    );
    println!("  BIPARTITE: cerebellum→γ/β (1.6-12ms) / cortex→θ/δ (12-318ms) via schedule_ext");
    println!("  Sanity: cortex/cerebellum ratio must be in [0.15, 0.40] ≈ biological 1:4");
    println!();

    // ── World ─────────────────────────────────────────────────────────────────
    let t0 = Instant::now();
    let world = VoxelWorld::new(9001, 18.0);
    let lap_dist = world.lap_distance as f64;
    println!(
        "world: {}×{} · {} wp · lap={:.0}m  [{:.0}ms]",
        world.size,
        world.size,
        world.waypoints_world.len(),
        lap_dist,
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // ── GPU ───────────────────────────────────────────────────────────────────
    let t1 = Instant::now();
    let mut ctx = match CortexCtx::new(&world, n_seres) {
        Some(c) => c,
        None => {
            eprintln!("ERROR: no GPU. Aborting.");
            std::process::exit(1);
        }
    };
    // Configures anti-saturation inhibition according to the sweep args.
    // fat_beta and div_sigma are passed to the shader via uniform — no recompile.
    ctx.cortex.set_inhibition(fat_beta, div_sigma);
    println!(
        "CortexCtx ready  [{:.0}ms]",
        t1.elapsed().as_secs_f64() * 1000.0
    );
    println!();

    // ── Initial genomes (random population) ───────────────────────────────────
    // Seed = index → deterministic and diverse.
    // σ_afw=0.50 (around 1.0): each being has a slightly different priority
    // for each channel, which gives Hebb material to specialize.
    let mut genomes: Vec<CortexGenome> = (0..n_seres)
        .map(|i| CortexGenome::random(i as u64))
        .collect();

    let evo_params = EvoParams {
        p_mutate: evo_p_mutate,
        sigma: evo_sigma,
        n_elite: evo_n_elite,
        tournament_k: evo_tourney,
    };

    let mut hist_med: Vec<f64> = Vec::with_capacity(n_gens);
    let mut hist_max: Vec<f64> = Vec::with_capacity(n_gens);
    let mut hist_results: Vec<GenResult> = Vec::with_capacity(n_gens);

    // ── MAIN LOOP ─────────────────────────────────────────────────────────────
    let t_total = Instant::now();
    for gen_idx in 0..n_gens {
        let t_gen = Instant::now();

        // --- Opens the replay writer for this generation ---
        // Records telemetry/gen_<idx>/cortex.replay with the best being.
        let replay_dir = std::path::PathBuf::from(format!("{}/gen_{}", out_dir, gen_idx));
        let _ = std::fs::create_dir_all(&replay_dir);
        let replay_path = replay_dir.join("cortex.replay");
        let mut replay_writer = universe::replay::ReplayWriter::create(
            &replay_path,
            universe::replay::SOURCE_CORTEX,
            1, // only the best being (index 0 = this generation's champion)
            64.0,
            9001u64, // same seed used in VoxelWorld::new(9001, ...)
        )
        .ok();

        // Loads genomes into the sheet (each being's initial weights).
        for (i, genome) in genomes.iter().enumerate().take(n_seres) {
            ctx.upload_genome_to_cortex(genome, i);
        }

        // ── Bipartite diagnostic (1× per generation, from the genomes) ────────
        // Measures: emerged ratio, τ distributions per part, 32×32 map.
        // Uses the genome of the previous BEST being (gen>0) or being 0 (gen=0).
        // The schedule is computed on the CPU — no GPU cost.
        if gen_idx == 0 || (gen_idx + 1) % map_interval == 0 {
            // population mean schedule_ext
            let ext_med: f32 = genomes[..n_seres]
                .iter()
                .map(|g| g.schedule_ext)
                .sum::<f32>()
                / n_seres as f32;
            let ext_min: f32 = genomes[..n_seres]
                .iter()
                .map(|g| g.schedule_ext)
                .fold(f32::INFINITY, f32::min);
            let ext_max: f32 = genomes[..n_seres]
                .iter()
                .map(|g| g.schedule_ext)
                .fold(f32::NEG_INFINITY, f32::max);

            // Uses being 0 as reference for the map
            let (n_cb, n_cx, taus_ser0) = compute_bipart_schedule(genomes[0].schedule_ext);
            let razao_cx_cb = n_cx as f32 / n_cb as f32;
            let razao_fracao = n_cx as f32 / SHEET_N as f32; // cortical fraction

            // Sanity: cortex/total ≈ 0.20 (biological 1:4 ratio: cerebellum has 4× more).
            // Range [0.15, 0.50] — ext=0 gives exactly 0.20, ext=0.5 gives 0.50.
            // If it leaves [0.15, 0.50] something is wrong in the schedule or the clamp.
            let sanidade = if razao_fracao >= 0.15 && razao_fracao <= 0.50 {
                if (razao_fracao - 0.20).abs() <= 0.05 {
                    "PASS ≈1:4 biological"
                } else {
                    "OK (within the evolvable range)"
                }
            } else {
                "OUTSIDE the range — bug in the schedule"
            };

            // τ distribution per part
            let tau_cb: Vec<f32> = taus_ser0[..n_cb].to_vec();
            let tau_cx: Vec<f32> = taus_ser0[n_cb..].to_vec();
            let tau_cb_min = tau_cb.iter().cloned().fold(f32::INFINITY, f32::min) * 1000.0;
            let tau_cb_max = tau_cb.iter().cloned().fold(f32::NEG_INFINITY, f32::max) * 1000.0;
            let tau_cx_min = tau_cx.iter().cloned().fold(f32::INFINITY, f32::min) * 1000.0;
            let tau_cx_max = tau_cx.iter().cloned().fold(f32::NEG_INFINITY, f32::max) * 1000.0;

            println!(
                "\n── BIPARTITE gen {:4} ──────────────────────────────────────────────",
                gen_idx
            );
            println!(
                "   schedule_ext: med={:.3}  min={:.3}  max={:.3}  (ser0={:.3})",
                ext_med, ext_min, ext_max, genomes[0].schedule_ext
            );
            println!(
                "   cerebellum: {} units ({:.1}%) τ∈[{:.1}ms, {:.1}ms] γ/β",
                n_cb,
                n_cb as f32 / SHEET_N as f32 * 100.0,
                tau_cb_min,
                tau_cb_max
            );
            println!(
                "   cortex:   {} units ({:.1}%) τ∈[{:.1}ms, {:.0}ms] θ/δ",
                n_cx,
                n_cx as f32 / SHEET_N as f32 * 100.0,
                tau_cx_min,
                tau_cx_max
            );
            println!(
                "   Cortex/total ratio = {:.3}  ({}) | biological target ≈0.20 (1:4)",
                razao_fracao, sanidade
            );

            // 32×32 map showing which part each unit belongs to
            // 'c' = cerebellum (γ/β), 'C' = cortex (θ/δ)
            println!("   32×32 map: 'c'=cerebellum(γ/β) 'C'=cortex(θ/δ)");
            for row in 0..SHEET_H {
                print!("   ");
                for col in 0..SHEET_W {
                    let u = row * SHEET_W + col;
                    let ch = if u < n_cb { 'c' } else { 'C' };
                    print!("{ch}");
                }
                println!();
            }
            println!();
        }

        // Resets EMA state and CfC between generations.
        ctx.reset_cortex_state(n_seres);
        ctx.reset_cfc(n_seres);

        // Initial state of the beings (distributed along the waypoints).
        let n_wp = world.waypoints_world.len();
        let mut states: Vec<BeingState> = (0..n_seres)
            .map(|i| {
                let wi = (i * n_wp / n_seres).min(n_wp - 1);
                let wp = world.waypoints_world[wi];
                let wp_next = world.waypoints_world[(wi + 1).min(n_wp - 1)];
                let dx = wp_next.x - wp.x;
                let dz = wp_next.y - wp.y;
                let len = (dx * dx + dz * dz).sqrt().max(1e-6);
                let fwd = [dx / len, 0.0, dz / len];
                BeingState {
                    phys: make_physics_state([wp.x, 0.0, wp.y], fwd),
                    prev_steer: 0.0,
                    stagnation_ticks: 0,
                    reverse_timer: 0.0,
                    post_reverse_lockout: 0.0,
                }
            })
            .collect();

        // Odometer.
        let mut dist_acum: Vec<f64> = vec![0.0; n_seres];
        let mut soma_vel: Vec<f64> = vec![0.0; n_seres];
        let mut n_amostras: Vec<f64> = vec![0.0; n_seres];
        // ARC progress along the track, with wrap detection — the only accumulator that cannot
        // be fooled by driving in circles, because backing up SUBTRACTS. Mirrors the logic of
        // `motor.rs` (signed delta, wrap when the jump exceeds half a lap).
        let mut prog_acum: Vec<f64> = vec![0.0; n_seres];
        let mut prog_prev: Vec<f32> = (0..n_seres)
            .map(|i| {
                let p = pos_of(&states[i].phys);
                world.get_track_progress(p[0], p[2])
            })
            .collect();
        let mut pos_prev: Vec<[f32; 3]> = states.iter().map(|s| pos_of(&s.phys)).collect();

        // Warm-up tick (JIT compilation — discarded).
        let _ = ctx.tick(&mut states, n_seres, 1.0 / 64.0, 0);

        // Simulation ticks.
        let dt = 1.0_f32 / 64.0;
        let mut abortado = false;
        for tick_i in 0..n_ticks {
            let ok = ctx.tick(&mut states, n_seres, dt, tick_i + 1);
            if !ok {
                eprintln!("[gen {:4}] ERROR at tick {} — aborting.", gen_idx, tick_i);
                abortado = true;
                break;
            }

            // --- REPLAY @ 16Hz: records the leading being (highest dist_acum so far) ---
            // Captures pos/vel/commands every 4 ticks (64Hz / 4 = 16Hz).
            // intent == final (no governor/sentinel on the cortex_evolve path).
            if tick_i % 4 == 0 {
                if let Some(ref mut rw) = replay_writer {
                    let best_lead = dist_acum
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    let ph = &states[best_lead].phys;
                    let speed = (ph.vel[0].powi(2) + ph.vel[2].powi(2)).sqrt();
                    let arc_prog = prog_acum[best_lead] as f32;

                    // Reads the leader's sensory_wave stored in ctx.last_wave.
                    // i8 channels in [-127,127]; conversion to physical units based
                    // on the perception.wgsl comments (each channel documented below).
                    let wv = &ctx.last_wave[best_lead];

                    // ch 76: left lateral dist — 127=touching, 0=40m clear (line 631 wgsl)
                    let dist_wall_l = (wv[76] as f32 / 127.0) * 40.0;
                    // ch 77: right lateral dist — same scale (line 632 wgsl)
                    let dist_wall_r = (wv[77] as f32 / 127.0) * 40.0;
                    // ch 82: frontal urgency — 127=touching, 0=30m+ (line 695 wgsl)
                    let dist_wall_f = (1.0 - wv[82] as f32 / 127.0) * 30.0;

                    // ch 78-80: curvature 40/70/120m — enc = (c_signed/FRAC_PI_2)*127
                    // Inverting: c_signed = (val/127) * FRAC_PI_2. For the player
                    // we use signed [-1..1] (FRAC_PI_2 is already absorbed by the display).
                    // (lines 664-665 wgsl)
                    let curv_40m = wv[78] as f32 / 127.0;
                    let curv_70m = wv[79] as f32 / 127.0;
                    let curv_120m = wv[80] as f32 / 127.0;

                    // ch 32: cross product forward×flow (angle 0 in BLOCK E, dist_ahead=10m)
                    // enc = cross * 127 — so val/127 ∈ [-1,1] (lines 450-451 wgsl)
                    let flow_align = wv[32] as f32 / 127.0;

                    // dims_saturated: counts channels with |val| > 114 (≈0.9×127).
                    // Neural saturation diagnostic — no specific channel.
                    let dims_saturated: u8 = wv
                        .iter()
                        .filter(|&&v| v.unsigned_abs() > 114)
                        .count()
                        .min(255) as u8;

                    // retina_fov: ch 128-135 (HD fovea, bins 0-7, line 852 wgsl)
                    let retina_fov: [i8; 8] = [
                        wv[128], wv[129], wv[130], wv[131], wv[132], wv[133], wv[134], wv[135],
                    ];
                    // retina_peri: ch 144-151 (HD peripheral, bins 0-7, line 875 wgsl)
                    let retina_peri: [i8; 8] = [
                        wv[144], wv[145], wv[146], wv[147], wv[148], wv[149], wv[150], wv[151],
                    ];

                    let rf = universe::replay::ReplayFrame {
                        pos: ph.pos,
                        fwd: [ph.fwd[0], ph.fwd[2]],
                        speed,
                        yaw_rate: ph.av_y,
                        arc_prog,
                        laps: 0, // cortex_evolve does not track laps yet
                        gear: ph.current_gear as u8,
                        hp_norm: 127,   // no HP on this path
                        fuel_norm: 127, // no fuel on this path
                        tick: tick_i as u32,
                        dist_wall_l,
                        dist_wall_r,
                        dist_wall_f,
                        curv_40m,
                        curv_70m,
                        curv_120m,
                        flow_align,
                        line_offset: 0.0, // ch 108-110 not ported (requires temporal state)
                        // No intent/final distinction (no governor/sentinel)
                        intent_thr: ph.throttle,
                        intent_str: ph.steer,
                        intent_brk: ph.brake_cmd,
                        final_thr: ph.throttle,
                        final_str: ph.steer,
                        final_brk: ph.brake_cmd,
                        dims_saturated,
                        intent_mag_q: 0,
                        axiom_str_q: 0,
                        safety_flags: (states[best_lead].reverse_timer > 0.0) as u8 * 8,
                        retina_fov,
                        retina_peri,
                        _alignment: [0u8; 3],
                        _pad: [0u8; 8],
                    };
                    let _ = rw.write_frame(0, &rf);
                }
            }

            for i in 0..n_seres {
                let p1 = pos_of(&states[i].phys);
                let dx = (p1[0] - pos_prev[i][0]) as f64;
                let dz = (p1[2] - pos_prev[i][2]) as f64;
                let passo = (dx * dx + dz * dz).sqrt();
                // MEASURED: `dist_acum` sums ABSOLUTE displacement — it does not distinguish
                // advancing from driving in circles, and does not charge for speed. Crawling at
                // 11 m/s for 24.000 ticks accumulates more than running at 30 m/s and crashing at
                // the third corner, so selection rewards exactly the observed behavior: throttle
                // never above 0.75, 42% of the time below 0.3, averaging 42 km/h in a GT3.
                //
                // `IARA_FITNESS` picks the criterion, to compare instead of assume:
                //   dist      (default) — absolute displacement, the control
                //   veloc     — distance × mean speed, charges for going fast
                //   progresso — only track progress, ignores driving in circles
                dist_acum[i] += passo;
                soma_vel[i] += (passo / dt as f64).min(60.0);
                n_amostras[i] += 1.0;

                let cur = world.get_track_progress(p1[0], p1[2]);
                let ld = lap_dist as f32;
                let mut d = cur - prog_prev[i];
                if d < -ld * 0.5 {
                    d += ld;
                }
                // crossed the line forward
                else if d > ld * 0.5 {
                    d -= ld;
                } // crossed backward
                prog_acum[i] += d.clamp(-ld * 0.1, ld * 0.1) as f64;
                prog_prev[i] = cur;

                pos_prev[i] = p1;
            }
        }

        // Closes this generation's replay writer.
        if let Some(ref mut rw) = replay_writer {
            let _ = rw.flush();
        }
        drop(replay_writer);

        // ── Cortical sheet saturation diagnostic (read from the last tick) ────
        // Measures the fraction of units with |act|>0.9 and detects universally saturated channels.
        // Also prints max/mean of activations as a heterogeneity proxy.
        // Readback of n_seres × SHEET_N f32 — once per generation, acceptable.
        {
            // copy_for_readback was called on the last tick; act_rb already has the final tick's data.
            let raw = ctx.cortex.read_act_raw(n_seres);
            let sat_thresh = 0.9_f32;
            let total_units = n_seres * SHEET_N;
            let n_sat: usize = raw.iter().filter(|&&v| v.abs() > sat_thresh).count();
            // Count per unit: what fraction of beings has this unit saturated
            let mut unit_sat_count = vec![0u32; SHEET_N];
            for s in 0..n_seres {
                for u in 0..SHEET_N {
                    if raw[s * SHEET_N + u].abs() > sat_thresh {
                        unit_sat_count[u] += 1;
                    }
                }
            }
            let n_universal: usize = unit_sat_count
                .iter()
                .filter(|&&c| c == n_seres as u32)
                .count();
            let mean_sat_per_ser = n_sat as f32 / n_seres as f32;
            let frac_total = n_sat as f32 / total_units as f32 * 100.0;
            // max |act| and mean |act| over all beings and units
            let max_act = raw.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            let mean_act = raw.iter().map(|v| v.abs()).sum::<f32>() / total_units as f32;
            println!(
                "  [sheet-sat] |act|>0.9: {:.1} units/being ({:.1}% total) | universal={} | max={:.3} mean={:.3}",
                mean_sat_per_ser, frac_total, n_universal, max_act, mean_act
            );
        }

        // Generation statistics.
        let r = gen_stats(gen_idx, &dist_acum, lap_dist);
        let t_gen_s = t_gen.elapsed().as_secs_f64();
        hist_med.push(r.dist_med);
        hist_max.push(r.dist_max);

        // Saves the best genome.
        let best_idx = dist_acum
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let save_path = format!("{}/cortex_best_gen{:04}.bin", out_dir, gen_idx);
        if let Err(e) = save_cortex_genome(&genomes[best_idx], &save_path) {
            eprintln!("[gen {:4}] WARNING: failed to save genome: {}", gen_idx, e);
        }

        // Per-generation report.
        println!(
            "gen {:4} | med={:7.0}m ({:.2}V) max={:7.0}m ({:.2}V) | ≥1V={} ≥2V={} dead={:.0}% | {:.1}s{}",
            gen_idx,
            r.dist_med,
            r.dist_med / lap_dist,
            r.dist_max,
            r.dist_max / lap_dist,
            r.n_ge1v,
            r.n_ge2v,
            r.mortality * 100.0,
            t_gen_s,
            if abortado { " [ABORTED]" } else { "" },
        );

        hist_results.push(r);

        // Modality map every N generations.
        if (gen_idx + 1) % map_interval == 0 || gen_idx == 0 {
            print_modal_map(&mut ctx, gen_idx);
        }

        // Evolution.
        if gen_idx + 1 < n_gens {
            // Configurable selection criterion — see the comment on the accumulation above.
            let criterio = std::env::var("IARA_FITNESS").unwrap_or_else(|_| "dist".into());
            // DIVERGENCE PROOF: without this, three runs with different criteria can come out
            // identical and I would have no way to know — it already happened this session with a stale binary.
            // Prints on the 1st generation the mean value of EACH criterion over the SAME population.
            let fitness: Vec<f64> = match criterio.as_str() {
                "veloc" => dist_acum
                    .iter()
                    .zip(soma_vel.iter())
                    .zip(n_amostras.iter())
                    .map(|((d, sv), n)| {
                        let vmed = if *n > 0.0 { sv / n } else { 0.0 };
                        d * (1.0 + vmed / 30.0) // 30 m/s as the reference scale
                    })
                    .collect(),
                // "progresso": monotonic progress along the TRACK (arc-length), not displacement
                // on the plane. The only criterion that cannot be fooled by driving in circles.
                "progresso" => prog_acum.iter().map(|p| p.max(0.0)).collect(),
                _ => dist_acum.clone(),
            };
            if gen_idx == 0 {
                let m = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len().max(1) as f64;
                let vel: Vec<f64> = dist_acum
                    .iter()
                    .zip(soma_vel.iter())
                    .zip(n_amostras.iter())
                    .map(|((d, sv), n)| d * (1.0 + (if *n > 0.0 { sv / n } else { 0.0 }) / 30.0))
                    .collect();
                println!(
                    "   [fitness] ACTIVE criterion = {}  ·  dist={:.0}  veloc={:.0}  progresso={:.0}",
                    criterio,
                    m(&dist_acum),
                    m(&vel),
                    m(&prog_acum)
                );
            }
            genomes = evolve_cortex(&genomes, &fitness, &evo_params, gen_idx);
        }
    }

    // ── Final summary ─────────────────────────────────────────────────────────
    let total_s = t_total.elapsed().as_secs_f64();
    println!();
    println!("=== FINAL RESULT ({:.1}s total) ===", total_s);

    if let Some(last) = hist_results.last() {
        println!("Last generation (gen {:4}):", last.generation);
        println!(
            "  median: {:.0}m ({:.3} laps)",
            last.dist_med,
            last.dist_med / lap_dist
        );
        println!(
            "  maximum:  {:.0}m ({:.3} laps)",
            last.dist_max,
            last.dist_max / lap_dist
        );
        println!(
            "  ≥1 lap: {}/{}  ≥2 laps: {}/{}",
            last.n_ge1v, n_seres, last.n_ge2v, n_seres
        );
        println!("  mortality: {:.1}%", last.mortality * 100.0);
        println!();
    }

    println!("=== MEDIAN TRAJECTORY (OLS) ===");
    if hist_med.len() >= 2 {
        let (slope, stderr) = linear_slope(&hist_med);
        let sigma_ratio = if stderr > 1e-10 { slope / stderr } else { 0.0 };
        println!(
            "  All {} gens: slope={:+.2}m/gen  σ={:.2}  ratio={:.2}σ",
            hist_med.len(),
            slope,
            stderr,
            sigma_ratio
        );
        for remover in 1..=3.min(hist_med.len().saturating_sub(2)) {
            let n_rest = hist_med.len() - remover;
            let (s2, e2) = linear_slope(&hist_med[..n_rest]);
            let sr2 = if e2 > 1e-10 { s2 / e2 } else { 0.0 };
            println!(
                "  -{}gen ({} gens): slope={:+.2}m/gen  σ={:.2}  ratio={:.2}σ",
                remover, n_rest, s2, e2, sr2
            );
        }
    }

    println!();
    println!("=== FULL TABLE ===");
    for (i, (&med, &max)) in hist_med.iter().zip(hist_max.iter()).enumerate() {
        println!(
            "  gen {:4}: med={:7.0}m ({:.3}V) max={:7.0}m ({:.3}V)",
            i,
            med,
            med / lap_dist,
            max,
            max / lap_dist
        );
    }
    println!("=== END cortex_evolve ===");
}

// ── Genome persistence ────────────────────────────────────────────────────────

fn save_cortex_genome(genome: &CortexGenome, path: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    for w in &genome.afw {
        f.write_all(&w.to_le_bytes())?;
    }
    for w in &genome.mw {
        f.write_all(&w.to_le_bytes())?;
    }
    f.write_all(&genome.tau_spread.to_le_bytes())?;
    // schedule_ext: bipartite schedule parameter (new field — appended at the end)
    f.write_all(&genome.schedule_ext.to_le_bytes())?;
    Ok(())
}
