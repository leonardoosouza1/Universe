//! Manager of GPU-resident state — the single point where the device is shared.
//!
//! # Why this file exists
//!
//! The `gpu/mod.rs` module (retina) opens its own `wgpu::Device`. Once physics, brain and
//! perception each did the same, we would have 3-4 devices open in parallel on the same
//! GPU, wasting command-buffer memory and making it impossible to share buffers between
//! subsystems (the retina would produce pixels into a buffer the brain cannot see, forcing
//! a readback + re-upload). This struct solves that: **one device, all buffers resident**.
//!
//! # Byte budget in flight per tick (60 beings, nominal configuration)
//!
//! > **Stale figure, kept on purpose.** Every line below that involves the first layer was
//! > computed with a sensory wave of 223 channels. `SENSORY_WAVE_SIZE` is *computed* in
//! > `config.rs:309` and is **192** today, so the real W1 figures are smaller than the ones
//! > recorded here: `192 × 96 = 18,432 f32` (73,728 bytes per being, 4,423,680 for 60),
//! > weights subtotal 5,806,080 bytes ≈ 5.5 MB, stride 24,288 f32, buf_weights 5,829,120
//! > bytes, resident total 8,301,440 bytes ≈ 7.9 MB. The numbers below are the original
//! > measurement and are left intact; read the constant, never this comment.
//!
//! Buffers UPLOADED from the CPU once per generation (static per track):
//!   grid      : 320×320 × 4 bytes (u32) = 409,600 bytes
//!   height_map: 320×320 × 4 bytes (f32) = 409,600 bytes
//!   banking   : 320×320 × 4 bytes (f32) = 409,600 bytes
//!   flow      : 320×320 × 8 bytes (Vec2 = [f32;2]) = 819,200 bytes
//!   static subtotal = 2,048,000 bytes ≈ 2 MB
//!
//! Buffers UPLOADED once per generation because they depend on the being's weights (living):
//!   instinct_layer weights (BitLinear 223→96, f32):
//!     223 × 96 × 4 = 85,632 bytes × 60 beings = 5,137,920 bytes
//!   cortex_l2 weights (FloatLinear 96→48, f32):
//!     96 × 48 × 4 = 18,432 bytes × 60 = 1,105,920 bytes
//!   cortex_l3 weights (FloatLinear 48→24, f32):
//!     48 × 24 × 4 = 4,608 bytes × 60 = 276,480 bytes
//!   weights subtotal = 6,520,320 bytes ≈ 6.2 MB
//!
//! PER TICK — only what CHANGES between ticks (the minimum):
//!   UPLOAD → CPU→GPU:
//!     poses (position + heading of every living being):
//!       60 × 32 bytes (PosGpu: [f32;8]) = 1,920 bytes
//!   DOWNLOAD → GPU→CPU:
//!     motor intent (throttle, steer, brake × 60 beings):
//!       60 × 3 × 4 bytes = 720 bytes
//!   TOTAL PER TICK = 2,640 bytes ≈ 2.6 kB
//!
//! The CfC state (60×96 f32 = 23,040 bytes) STAYS resident on the GPU between ticks.
//! It only comes down at the end of the generation, to serialise evolution.
//!
//! # Retina state per tick
//!
//! RETINA_TOTAL = 14×126 = 1,764 pixels × 4 bytes (u32) = 7,056 bytes × 60 beings =
//! 423,360 bytes. Those bytes NEVER come down to the CPU on the hot path — they sit in a
//! resident buf_retina_output that the brain kernel reads directly in the next step.

use std::borrow::Cow;
use wgpu::util::DeviceExt;

use crate::config::iara::{
    brain::{L1_NEURONS, L2_NEURONS, L3_NEURONS, SENSORY_WAVE_SIZE},
    vision::RETINA_TOTAL,
};
use crate::world::VoxelWorld;

// ─── Maximum number of beings CoreGpu supports. ──────────────────────────
// 60 is nominal; we allocate 64 for alignment with 64 threads/workgroup, plus headroom.
const MAX_CAPACITY: usize = 64;

// ─── World grid size. Declared here for the byte budget above. ─────────────
// Taken from world.rs: `GRID_CELLS: f32 = 320.0` (world.rs:53), used as `size` in
// `VoxelWorld::new` (world.rs:84).
pub const GRID_SIZE: usize = 320;

// ─── Weight dimensions, per being. ─────────────────────────────────────────
// BitLinear 223→96: the input is SENSORY_WAVE_SIZE, the output L1=96. The "223" is the
// value the constant had when this was written; it is COMPUTED in `config.rs:309` and is
// 192 today, so the real product is 192×96 = 18,432 f32. The code reads the constant, so
// only the comment was ever wrong.
// FloatLinear 96→48: input L1_NEURONS, output L2_NEURONS.
// FloatLinear 48→24: input L2_NEURONS, output L3_NEURONS.
const W1_TOTAL: usize = SENSORY_WAVE_SIZE * L1_NEURONS; // 223×96 = 21,408 f32 (see note)
const W2_TOTAL: usize = L1_NEURONS * L2_NEURONS; // 96×48  =  4,608 f32
const W3_TOTAL: usize = L2_NEURONS * L3_NEURONS; // 48×24  =  1,152 f32

// CfC state (96 dimensions, one slice per being, f32).
const CFC_DIM: usize = L1_NEURONS; // 96

// ─── Per-being pose layout (GPU side). ─────────────────────────────────────
// 8 × f32 = 32 bytes, aligned to 16 (wgpu requirement for a storage buffer).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PosGpu {
    /// [pos.x, pos.y, pos.z, padding] + [fwd.x, fwd.z, right.x, right.z]
    pub pos_pad: [f32; 4],
    pub dirs: [f32; 4],
}

// ─── Motor intent returned by the GPU each tick. ───────────────────────────
// 3 × f32 + padding = 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct IntentGpu {
    pub throttle: f32,
    pub steer: f32,
    pub brake: f32,
    pub _pad: f32,
}

/// GPU context shared by every kernel.
///
/// Keeps a single `wgpu::Device` open with no surface (compute-only). Every resident buffer
/// lives here — the retina, the brain and physics read and write those buffers through bind
/// groups, with no intermediate readback.
///
/// # Life cycle
///
/// ```text
/// CoreGpu::new(world, capacity)    ← once per track
///     |
///     ├── uploads grid, height_map, banking, flow  (static)
///     └── uploads the beings' weights              (once per generation)
///
/// tick(poses, ...)                     ← every simulation tick
///     |
///     ├── writes poses (2 kB)
///     ├── perception dispatch (ASSUMED: PerceptionGpu via an external kernel)
///     ├── brain dispatch      (ASSUMED: BrainGpu via an external kernel)
///     ├── physics dispatch    (ASSUMED: PhysicsGpu via an external kernel)
///     └── motor intent readback (720 bytes)
///
/// download_cfc_state()                 ← once per generation, before evolution
/// ```
pub struct CoreGpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: String,

    // ── World buffers (static per track, uploaded in `new`) ────────────────
    /// grid[320×320] → u32 (VoxelType cast).
    pub buf_grid: wgpu::Buffer,
    /// height_map[320×320] → f32 (metres).
    pub buf_height: wgpu::Buffer,
    /// banking[320×320] → f32 (radians).
    pub buf_banking: wgpu::Buffer,
    /// flow[320×320] → [f32;2] (normalised direction along the track).
    pub buf_flow: wgpu::Buffer,

    // ── Weight buffers (once per generation) ──────────────────────────────
    /// Concatenated weights: being0:[W1|W2|W3|CfC_state], being1:..., ...
    /// W1 = L1_NEURONS × SENSORY_WAVE_SIZE f32 (instinct_layer, BitLinear; weights are f32).
    /// W2 = L2_NEURONS × L1_NEURONS f32 (cortex_l2, FloatLinear).
    /// W3 = L3_NEURONS × L2_NEURONS f32 (cortex_l3, FloatLinear).
    /// CfC_state = CFC_DIM f32 (temporal state — updated in place by the brain).
    ///
    /// Stride per being = (W1_TOTAL + W2_TOTAL + W3_TOTAL + CFC_DIM) × sizeof(f32).
    pub buf_weights: wgpu::Buffer,

    // ── Per-tick state buffers ────────────────────────────────────────────
    /// poses[capacity] → PosGpu (written by the CPU each tick, 2 kB for 60 beings).
    pub buf_poses: wgpu::Buffer,
    /// retina_output[capacity × RETINA_TOTAL] → u32 (dist_mm<<8 | material).
    /// NEVER comes down to the CPU on the hot path. The brain kernel reads it from here.
    pub buf_retina_output: wgpu::Buffer,
    /// intent[capacity] → IntentGpu (comes down to the CPU each tick, 720 bytes).
    pub buf_intent: wgpu::Buffer,
    /// Read staging for intent (MAP_READ).
    pub buf_intent_readback: wgpu::Buffer,

    // ── Metadata ──────────────────────────────────────────────────────────
    pub capacity: usize,
    pub grid_size: u32,
    pub cell_size: f32,

    // ── Compiled pipelines ────────────────────────────────────────────────
    // ASSUMED: the perception, brain and physics kernels get compiled here once their
    // WGSL modules are ready. For now, placeholders.
    //
    // ASSUMED: pipeline_retina reuses the shader in src/gpu/retina.wgsl.
    // ASSUMED: the perception pipeline will come from src/gpu/perception.wgsl (agent C).
    // ASSUMED: the brain pipeline will come from src/gpu/brain.wgsl (agent B).
    // ASSUMED: the physics pipeline will come from src/gpu/physics.wgsl (agent A).
    pub pipeline_retina: wgpu::ComputePipeline,
    pub layout_retina: wgpu::BindGroupLayout,
}

// Weight stride per being, in f32 (not in bytes).
fn weights_stride_f32() -> usize {
    W1_TOTAL + W2_TOTAL + W3_TOTAL + CFC_DIM
}

impl CoreGpu {
    /// Opens the device and uploads the world's static data.
    ///
    /// Returns `None` when there is no usable GPU — the simulation falls back to the CPU
    /// instead of hanging. `capacity` must be >= the number of beings per generation
    /// (60 nominal).
    ///
    /// `min(capacity, MAX_CAPACITY)` beings are allocated; going past 64 beings requires
    /// raising MAX_CAPACITY and recompiling (buffer size is fixed at compile time for WGSL).
    pub fn new(world: &VoxelWorld, capacity: usize) -> Option<Self> {
        // ── Open the device: compute-only, no surface, no window. ────────────
        // `compatible_surface: None` is the point: it asks for a COMPUTE adapter.
        // Bevy already does this in the retina module (mod.rs) — we replicate the
        // same pattern rather than inventing a new API.
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..Default::default()
        });
        let adapter_obj =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))?;
        let name = adapter_obj.get_info().name.clone();
        let (device, queue) = pollster::block_on(adapter_obj.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("iara-core"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;

        let cap = capacity.min(MAX_CAPACITY).max(1);

        // ── Static world buffers ────────────────────────────────────────────
        // Uploading the grid as u32 (VoxelType cast).
        // 320×320 × 4 = 409,600 bytes.
        let grid_u32: Vec<u32> = world.grid.iter().map(|v| *v as u32).collect();
        let buf_grid = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("core-grid"),
            contents: bytemuck::cast_slice(&grid_u32),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // height_map: 320×320 × 4 = 409,600 bytes.
        let buf_height = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("core-height"),
            contents: bytemuck::cast_slice(&world.height_map),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // banking: 320×320 × 4 = 409,600 bytes.
        let buf_banking = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("core-banking"),
            contents: bytemuck::cast_slice(&world.banking),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // flow: Vec<Vec2> → [f32;2] per cell → 320×320×8 = 819,200 bytes.
        // `Vec2` is bytemuck::Pod in bevy 0.16 (layout [f32; 2]).
        let flow_raw: Vec<[f32; 2]> = world.flow.iter().map(|v| [v.x, v.y]).collect();
        let buf_flow = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("core-flow"),
            contents: bytemuck::cast_slice(&flow_raw),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // ── Weight buffer (per being) ───────────────────────────────────────
        // Allocates room for `cap` beings; filled by `upload_weights()` before the first
        // generation. COPY_DST so the CPU can write it between generations.
        // Stride per being in bytes = weights_stride_f32() × 4.
        // For 60 beings: (21408+4608+1152+96) × 4 × 60 = 27264 × 4 × 60 = 6,543,360 bytes
        // ≈ 6.2 MB — computed with the old 223-channel wave; with SENSORY_WAVE_SIZE = 192
        // it is 24288 × 4 × 60 = 5,829,120 bytes ≈ 5.6 MB.
        let bytes_weights = (weights_stride_f32() * cap * std::mem::size_of::<f32>()) as u64;
        let buf_weights = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("core-weights"),
            size: bytes_weights,
            // COPY_SRC is required because the recurrent state (CfC) lives INSIDE this
            // buffer and has to be read back at the end of the generation. Without the
            // flag, wgpu rejects the copy_buffer_to_buffer with a validation error — which
            // is exactly what happened.
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // ── Per-tick buffers ───────────────────────────────────────────────
        // poses: 60 × 32 bytes = 1,920 bytes.
        let bytes_poses = (cap * std::mem::size_of::<PosGpu>()) as u64;
        let buf_poses = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("core-poses"),
            size: bytes_poses,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // retina_output: 60 × 1764 × 4 = 423,360 bytes.
        // COPY_SRC so it can be copied into a readback buffer when inspection is needed.
        let bytes_retina = (cap * RETINA_TOTAL * std::mem::size_of::<u32>()) as u64;
        let buf_retina_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("core-retina-output"),
            size: bytes_retina,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // intent: 60 × 16 bytes = 960 bytes (IntentGpu carries padding).
        let bytes_intent = (cap * std::mem::size_of::<IntentGpu>()) as u64;
        let buf_intent = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("core-intent"),
            size: bytes_intent,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_intent_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("core-intent-readback"),
            size: bytes_intent,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Retina pipeline (reuses retina.wgsl) ───────────────────────────
        // ASSUMED: the retina.wgsl shader accepts exactly the layout below
        // (uniform params, read-only storage grid, read-only storage poses,
        // write-only storage output). This is the layout already in mod.rs.
        let shader_retina = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("retina"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(crate::gpu::RETINA_SHADER)),
        });
        let layout_retina = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("retina-layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),  // grid
                storage_entry(2, true),  // poses
                storage_entry(3, false), // output
            ],
        });
        let pipe_layout_retina = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout_retina],
            push_constant_ranges: &[],
        });
        let pipeline_retina = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("retina-pipeline"),
            layout: Some(&pipe_layout_retina),
            module: &shader_retina,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
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
            grid_size: world.size as u32,
            cell_size: world.cell_size,
            pipeline_retina,
            layout_retina,
        })
    }

    /// Uploads every being's weights to the GPU.
    ///
    /// Called once per generation (before the first tick), once selection/mutation has
    /// produced the new generation's weights. `n_beings` must be `<= self.capacity`.
    ///
    /// Buffer layout:
    ///   being 0: [W1: W1_TOTAL f32] [W2: W2_TOTAL f32] [W3: W3_TOTAL f32] [CfC: CFC_DIM f32]
    ///   being 1: same, stride = weights_stride_f32() f32
    ///
    /// ASSUMED: the caller (BrainGpu or the orchestrator) extracts the weights from
    /// `SovereignEngine::instinct_layer.weights`, `cortex_l2.weights`, `cortex_l3.weights`
    /// and initialises the CfC state to zeros.
    pub fn upload_weights(
        &self,
        weights_w1: &[f32], // [n_beings × W1_TOTAL]
        weights_w2: &[f32], // [n_beings × W2_TOTAL]
        weights_w3: &[f32], // [n_beings × W3_TOTAL]
        n_beings: usize,
    ) {
        let n = n_beings.min(self.capacity);
        let stride = weights_stride_f32();
        let mut interleaved = vec![0.0f32; stride * n];
        for s in 0..n {
            let base = s * stride;
            // W1
            let w1s = s * W1_TOTAL;
            interleaved[base..base + W1_TOTAL].copy_from_slice(&weights_w1[w1s..w1s + W1_TOTAL]);
            // W2
            let w2s = s * W2_TOTAL;
            let o2 = base + W1_TOTAL;
            interleaved[o2..o2 + W2_TOTAL].copy_from_slice(&weights_w2[w2s..w2s + W2_TOTAL]);
            // W3
            let w3s = s * W3_TOTAL;
            let o3 = o2 + W2_TOTAL;
            interleaved[o3..o3 + W3_TOTAL].copy_from_slice(&weights_w3[w3s..w3s + W3_TOTAL]);
            // CfC state: zeros (reset every generation, like CfCCell::reset()).
            // bytes [o3+W3_TOTAL .. o3+W3_TOTAL+CFC_DIM] are already 0 from `vec![0.0]`.
        }
        self.queue
            .write_buffer(&self.buf_weights, 0, bytemuck::cast_slice(&interleaved));
    }

    /// Writes the living beings' poses and fires the kernels in sequence.
    ///
    /// # Dispatch order (a single `CommandEncoder`, no intermediate readbacks)
    ///
    ///   1. **retina** — reads grid + poses, writes buf_retina_output
    ///   2. **perception** — reads grid/flow/height/banking + buf_retina_output + poses,
    ///      writes the semantic channels (ASSUMED: PerceptionGpu via a separate buffer)
    ///   3. **brain** — reads weights + retina + semantic channels + CfC state,
    ///      writes buf_intent and updates the CfC state in place
    ///   4. **physics** — reads intent + body state (ASSUMED: PhysicsGpu via a separate
    ///      buffer)
    ///
    /// # Readback
    ///
    /// Only `buf_intent` comes down to the CPU (720 bytes nominal). The CfC state only
    /// comes down at the end of the generation, via `download_cfc_state()`.
    ///
    /// ASSUMED: perception and physics have their own bind groups, created by agents B, C
    /// and A. This method only issues the retina dispatch; the others get chained into the
    /// same encoder in the final integration (`docs/arquivo/GPU_PLANO.md` §D).
    ///
    /// # Parameters
    ///
    /// - `poses`: poses of the beings alive this tick (len = n_alive).
    /// - `params_retina`: the bytes of the retina shader's `ParamsGpu` (uniform).
    ///   ASSUMED: the caller builds `ParamsGpu` with `pitches`, `n_beings`, `grid_size`,
    ///   `cell_size` and `fov_half`, exactly as in `RetinaGpu::render`.
    ///
    /// Returns the motor intents read back from the GPU: a `Vec` of length `n_alive`.
    pub fn tick(
        &self,
        poses: &[PosGpu],
        params_retina: &[u8], // bytemuck::bytes_of(&ParamsGpu{...})
        n_alive: usize,
    ) -> Vec<IntentGpu> {
        let n = n_alive.min(self.capacity);
        if n == 0 {
            return Vec::new();
        }

        // 1. Upload poses (1,920 bytes for 60 beings — the only CPU→GPU traffic per tick).
        self.queue
            .write_buffer(&self.buf_poses, 0, bytemuck::cast_slice(&poses[..n]));

        // 2. Build the retina params uniform as a temporary buffer.
        //    We create it per tick because `n_beings` can vary (deaths during a generation).
        let buf_params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("core-params-retina"),
                contents: params_retina,
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // 3. Retina bind group (same layout as mod.rs::RetinaGpu).
        let bg_retina = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout_retina,
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
                    resource: self.buf_poses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buf_retina_output.as_entire_binding(),
                },
            ],
        });

        // 4. A single CommandEncoder — every dispatch with no intermediate readback.
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("core-tick"),
            });

        // ── Retina dispatch ──────────────────────────────────────────────────
        // 64 threads/workgroup: 2 waves of 32 on RDNA2 (wave32 is the correct size).
        // Total threads = n × RETINA_TOTAL = 60 × 1764 = 105,840.
        // Workgroups = ceil(105840 / 64) = 1,654.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass-retina"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_retina);
            pass.set_bind_group(0, &bg_retina, &[]);
            let total = (n * RETINA_TOTAL) as u32;
            pass.dispatch_workgroups(total.div_ceil(64), 1, 1);
        }

        // ── ASSUMED: perception dispatch ─────────────────────────────────────
        // PerceptionGpu (agent C) exposes a method like:
        //   fn bind_and_dispatch(enc: &mut CommandEncoder, core: &CoreGpu, n: usize)
        // which builds the bind group reading buf_retina_output + buf_grid + buf_flow +
        // buf_height and writes into a perception output buffer (semantic channels 16-127
        // + 128-191).
        //
        // ASSUMED: that perception output buffer is a CoreGpu field (to be added during
        // integration) sized n × SENSORY_WAVE_SIZE × sizeof(i8).
        //   For 60 beings: 60 × 223 = 13,380 bytes (with today's SENSORY_WAVE_SIZE = 192,
        //   60 × 192 = 11,520 bytes).

        // ── ASSUMED: brain dispatch ──────────────────────────────────────────
        // BrainGpu (agent B) exposes:
        //   fn bind_and_dispatch(enc: &mut CommandEncoder, core: &CoreGpu, n: usize)
        // which reads buf_weights (weights + CfC state) + the perception output buffer,
        // updates the CfC state in place inside buf_weights and writes buf_intent.
        //
        // ASSUMED: 1 workgroup per being (60 workgroups), each running its full forward
        // pass: KAN → CfC → BitLinear(L1) → FloatLinear(L2) → FloatLinear(L3).

        // ── ASSUMED: physics dispatch ────────────────────────────────────────
        // PhysicsGpu (agent A) exposes:
        //   fn bind_and_dispatch(enc: &mut CommandEncoder, core: &CoreGpu, n: usize, dt: f32)
        // which reads buf_intent + the body state (suspension, velocity — buffer to be
        // added) and integrates the physics step in place.

        // ── Intent readback (the only readback per tick, 720 bytes) ───────────
        let bytes_int = (n * std::mem::size_of::<IntentGpu>()) as u64;
        enc.copy_buffer_to_buffer(&self.buf_intent, 0, &self.buf_intent_readback, 0, bytes_int);

        self.queue.submit(Some(enc.finish()));

        // ── Synchronous mapping — blocks until the GPU finishes. ──────────────
        // `device.poll(Wait)` drains the queue before mapping.
        // In production this caps pipelining at ~1 frame. For the first version
        // (guaranteed equivalence) that is acceptable; async mapping is the next
        // optimisation.
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

    /// Brings every being's CfC state down to the CPU.
    ///
    /// Called ONCE per generation, before evolutionary selection. The CfC state is
    /// runtime-only (it does not persist in .dna), but we need the last activations for
    /// telemetry and for the neuromodulation evolution pipeline.
    ///
    /// Returns a Vec<Vec<f32>> with `n_beings` slices of `CFC_DIM` f32.
    ///
    /// # Cost
    ///
    /// For 60 beings: 60 × 96 × 4 = 23,040 bytes of GPU→CPU traffic.
    /// It happens 0 times per tick during the generation, and once at the end.
    pub fn download_cfc_state(&self, n_beings: usize) -> Vec<Vec<f32>> {
        let n = n_beings.min(self.capacity);
        let stride = weights_stride_f32();
        let cfc_offset = (W1_TOTAL + W2_TOTAL + W3_TOTAL) as u64;

        // Staging buffer for the whole weight block (we only need the CfC, but wgpu
        // requires copying aligned blocks; we copy the entire stride and extract).
        let bytes_total_weights = (stride * n * std::mem::size_of::<f32>()) as u64;
        let buf_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("core-cfc-staging"),
            size: bytes_total_weights,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.buf_weights, 0, &buf_staging, 0, bytes_total_weights);
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

    /// Exposes a reference to the device so the external kernels (physics/brain/perception)
    /// can build their own bind groups and pipelines without opening a new device.
    ///
    /// ASSUMED: PhysicsGpu, BrainGpu and PerceptionGpu receive `&CoreGpu` and use
    /// `core.device` and `core.queue` directly — they never create their own
    /// `wgpu::Instance`.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Dimension constants exposed so external kernels do not have to import the config
    /// constants directly (avoids crate coupling).
    pub const fn retina_total() -> usize {
        RETINA_TOTAL
    }
    pub const fn sensory_wave_size() -> usize {
        SENSORY_WAVE_SIZE
    }
    pub const fn cfc_dim() -> usize {
        CFC_DIM
    }
    pub const fn weights_stride() -> usize {
        W1_TOTAL + W2_TOTAL + W3_TOTAL + CFC_DIM
    }
    pub const fn w1_total() -> usize {
        W1_TOTAL
    }
    pub const fn w2_total() -> usize {
        W2_TOTAL
    }
    pub const fn w3_total() -> usize {
        W3_TOTAL
    }
}

// ── Bind group layout helpers ───────────────────────────────────────────────
// The single canonical source (E3 / REFACTOR_PLAN §2.3 "Deduplication"). Before this
// unification there were 14 identical copies scattered across src/gpu/*.rs and
// src/bin/*.rs — each one fixed or not depending on whoever touched it last.
// `pub` so the binaries (gpu_evolve, gpu_loop, cortex_evolve) and the rest of the
// src/gpu/ modules import them instead of duplicating them.

pub fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

pub fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

// ── Per-tick byte summary (quick reference during code review) ───────────────
//
// The buf_weights line and the resident total below were computed with the old
// 223-channel sensory wave; with SENSORY_WAVE_SIZE = 192 (`config.rs:309`) they are
// 5,829,120 bytes and 8,301,440 bytes ≈ 7.9 MB. The original figures are kept.
//
// RESIDENT BUFFERS (they never leave the GPU during a generation):
//   buf_grid          :   409,600 bytes  (320×320×4)
//   buf_height        :   409,600 bytes  (320×320×4)
//   buf_banking       :   409,600 bytes  (320×320×4)
//   buf_flow          :   819,200 bytes  (320×320×8)
//   buf_weights       : 6,543,360 bytes  (stride=27264 f32 × 60 beings × 4)
//   buf_retina_output :   423,360 bytes  (60 × 1764 × 4)
//   buf_intent        :       960 bytes  (60 × 16)
//   RESIDENT TOTAL    : 9,015,680 bytes ≈ 8.6 MB
//
// PER TICK (real CPU↔GPU traffic):
//   UPLOAD   (write_buffer buf_poses)          :  1,920 bytes
//   DOWNLOAD (copy+map buf_intent_readback)    :    960 bytes (padding included)
//   TOTAL PER TICK                             :  2,880 bytes ≈ 2.8 kB
//
// AT THE END OF A GENERATION (download_cfc_state, 1×):
//   DOWNLOAD of the full buf_weights (to extract the CfC) : 6,543,360 bytes
//   (future optimisation: copy only the CfC slice = 60×96×4 = 23,040 bytes
//    via offsets — needs wgpu::CommandEncoder::copy_buffer_to_buffer with
//    source_offset = being * stride_bytes + cfc_offset_bytes)
