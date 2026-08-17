// ── SLOW BRAIN ON THE GPU — Rust host ─────────────────────────────────────────
//
// Hosts SlowBrain (CfC-64 → Dense1(64→16) → Dense2(16→3)) on the GPU so the
// GPU path has both brains (fast: brain.rs + slow: this file).
//
// # Why two kernels instead of one
//
// RADV/ACO crashes on large shaders in this project (history of silent crashes
// with kernels > ~200 body lines). SlowBrain has three stages with barriers
// between them (CfC → Dense1 → Dense2). With 64 threads and a single workgroup this
// would work in one kernel, but it would exceed 150 body lines. We split it into
// two kernels chained in the same CommandEncoder:
//   Kernel A (run_cfc):   updates the CfC state → saves s in buf_state
//   Kernel B (run_dense): reads s, recomputes tanh, Dense1 + Dense2 → output
// A single submit for both.
//
// # Resident buffers
//
//   buf_state:   [STATE_PER_SER × cap f32] — cfc_state, fatigue, counter,
//                tick_counter, output[3]. Initialized to zeros.
//   buf_weights: [WEIGHTS_PER_SER × cap f32] — Dense1+Dense2, uploaded per spawn.
//   buf_input:   [INPUT_SIZE × cap i32] — input per tick (48 sign-extended i8 channels).
//   buf_output:  [3 × cap f32] — output per tick, read by the host.
//   buf_readback: [3 × cap f32] — separate MAP_READ (wgpu requires it).
//
// # Slot count in STATE_PER_SER = 133 f32
//   [0..64)    cfc_state
//   [64..128)  cfc_fatigue
//   [128]      counter (u32 bitcast)
//   [129]      tick_counter (u32 bitcast)
//   [130..133) output[3]
//
// # Weights in WEIGHTS_PER_SER = 1091 f32
//   [0..1024)    dense1.weights (64×16)
//   [1024..1040) dense1.bias   (16)
//   [1040..1088) dense2.weights (16×3)
//   [1088..1091) dense2.bias   (3)
//   Same order as SlowBrain::all_weights() (engine.rs) — correct DNA round-trip.
//
// # map_async → poll → recv (mandatory order)
//
// Deadlock documented in this project: poll(Wait) BEFORE map_async does not serve the
// callback. The poll must come AFTER the mapping request. See brain.rs:710
// for the history. Every readback here follows the order: map_async → poll → recv.

use std::borrow::Cow;
use std::sync::Arc;
use wgpu::util::DeviceExt;

// ── DIMENSIONS ───────────────────────────────────────────────────────────────

const INPUT_SIZE: usize = 48;
const CFC_SIZE: usize = 64;
const D2_OUT: usize = 3;
const OUTPUT_DIM: usize = D2_OUT;

/// State slots per being (f32):
///   cfc_state[64] + cfc_fatigue[64] + counter(1) + tick_counter(1) + output[3] = 133
const STATE_PER_SER: usize = CFC_SIZE + CFC_SIZE + 1 + 1 + D2_OUT; // 133

/// Weights per being (f32). Order mirrors SlowBrain::all_weights():
///   dense1.weights(1024) + dense1.bias(16) + dense2.weights(48) + dense2.bias(3) = 1091
const WEIGHTS_PER_SER: usize = 1024 + 16 + 48 + 3; // 1091

// ── UNIFORM ───────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SlowBrainParams {
    n_beings: u32,
    dt: f32,
    _pad0: u32,
    _pad1: u32,
}

// ── PERSISTENT CONTEXT ────────────────────────────────────────────────────────

/// SlowBrain GPU context. Lives for the whole generation.
///
/// Invariant: every buffer supports up to `capacity` beings.
pub struct SlowBrainGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    // Two pipelines — two kernels chained in the same CommandEncoder.
    pipeline_cfc: wgpu::ComputePipeline,
    pipeline_dense: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,

    // Resident buffers
    buf_state: wgpu::Buffer, // f32 × (cap × STATE_PER_SER) — resident, zeros init
    buf_weights: wgpu::Buffer, // f32 × (cap × WEIGHTS_PER_SER) — uploaded per spawn
    buf_input: wgpu::Buffer, // i32 × (cap × INPUT_SIZE) — written per tick or by the gather
    // GATHER pipeline: fills buf_input from the sensory wave that is already
    // on the GPU, avoiding a readback per tick just to pick 48 of 223 channels.
    pipeline_gather: wgpu::ComputePipeline,
    layout_gather: wgpu::BindGroupLayout,
    buf_output: wgpu::Buffer, // f32 × (cap × OUTPUT_DIM) — read per tick (COPY_SRC)
    buf_readback: wgpu::Buffer, // f32 × (cap × OUTPUT_DIM) — MAP_READ

    /// Maximum capacity (number of beings supported with these buffers).
    pub capacity: usize,
}

impl SlowBrainGpu {
    /// Creates the context sharing device/queue with BrainGpu (retina/brain).
    ///
    /// Receiving Arc<Device/Queue> avoids opening a second Vulkan device for the
    /// same hardware — the same pattern as brain.rs:128.
    pub fn new_shared(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, capacity: usize) -> Self {
        // ── Shader ─────────────────────────────────────────────────────────
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("slow-brain"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("slow_brain.wgsl"))),
        });

        // ── Bind group layout (a single layout for both kernels) ────────────
        // Both kernels use the same set of bindings — separate layouts would be
        // unnecessary and would create two bind groups per dispatch.
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("slow-brain-layout"),
            entries: &[
                uniform_entry(0),        // params
                storage_entry(1, true),  // buf_input   (read)
                storage_entry(2, true),  // buf_weights (read)
                storage_entry(3, false), // buf_state   (read_write, resident)
                storage_entry(4, false), // buf_output  (read_write)
            ],
        });

        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        // Pipeline A: CfC
        let pipeline_cfc = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("slow-cfc-pipeline"),
            layout: Some(&pipe_layout),
            module: &module,
            entry_point: Some("run_cfc"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Pipeline B: Dense1 + Dense2
        let pipeline_dense = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("slow-dense-pipeline"),
            layout: Some(&pipe_layout),
            module: &module,
            entry_point: Some("run_dense"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── Buffers ─────────────────────────────────────────────────────────

        // Resident state: zeros = correct initial state (CfCCell::new zeroed).
        let bytes_state = (capacity * STATE_PER_SER * 4) as u64;
        let state_zeros: Vec<u8> = vec![0u8; bytes_state as usize];
        let buf_state = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("slow-state"),
            contents: &state_zeros,
            usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST  // for reset in reset_being()
                    | wgpu::BufferUsages::COPY_SRC, // for diagnostics (if needed)
        });

        // Weights: allocated but not initialized until upload_weights().
        let bytes_weights = (capacity * WEIGHTS_PER_SER * 4) as u64;
        let buf_weights = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("slow-weights"),
            size: bytes_weights,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Input: i32 × INPUT_SIZE × cap — written every tick.
        let bytes_input = (capacity * INPUT_SIZE * 4) as u64;
        // Gather module and pipeline — separate shader to keep each kernel
        // small (RADV/ACO crashes on large shaders in this project).
        let module_gather = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("slow-gather"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("slow_gather.wgsl"))),
        });
        let layout_gather = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("slow-gather-layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, false),
            ],
        });
        let pl_gather = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("slow-gather-pl"),
            bind_group_layouts: &[&layout_gather],
            push_constant_ranges: &[],
        });
        let pipeline_gather = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("slow-gather-pipe"),
            layout: Some(&pl_gather),
            module: &module_gather,
            entry_point: Some("gather"),
            compilation_options: Default::default(),
            cache: None,
        });

        let buf_input = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("slow-input"),
            size: bytes_input,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Output: f32 × OUTPUT_DIM × cap — written by kernel B.
        // COPY_SRC to copy into buf_readback in the same encoder.
        let bytes_output = (capacity * OUTPUT_DIM * 4) as u64;
        let buf_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("slow-output"),
            size: bytes_output,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Readback: MAP_READ cannot be combined with STORAGE in wgpu.
        let buf_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("slow-readback"),
            size: bytes_output,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
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

    // ── WEIGHT UPLOAD ──────────────────────────────────────────────────────────

    /// Uploads the weights of `n_beings` SlowBrains to the GPU.
    ///
    /// Call on spawn and after evolution. Weight order: all_weights() of each being.
    /// Layout: being0[1091 f32] being1[1091 f32] ...
    ///
    /// `weights`: concatenated output of SlowBrain::all_weights() for each being,
    /// total = n_beings × 1091 f32.
    pub fn upload_weights(&self, weights: &[f32], n_beings: usize) {
        let n = n_beings.min(self.capacity);
        let expected = n * WEIGHTS_PER_SER;
        let actual = weights.len().min(expected);
        // Direct write: the SoA is already in the correct order if the caller
        // concatenated all_weights() of each being in the right order.
        self.queue.write_buffer(
            &self.buf_weights,
            0,
            bytemuck::cast_slice(&weights[..actual]),
        );
    }

    // ── DISPATCH (per tick) ────────────────────────────────────────────────────

    /// Records the two kernels into an external CommandEncoder (no submit).
    ///
    /// # Contract
    ///
    /// 1. Before: upload the input via `queue.write_buffer(&buf_input, ...)` or use
    ///    `upload_input()` (convenience method below).
    /// 2. Call this method to record both dispatches.
    /// 3. In the same encoder: `enc.copy_buffer_to_buffer(buf_output→buf_readback)`.
    /// 4. `queue.submit(enc.finish())`.
    /// 5. `device.poll(Wait)`.
    /// 6. `read_outputs(n_beings)` to read the result.
    ///
    /// Returns bytes_output for the subsequent copy.
    /// Fills `buf_input` by gathering 48 channels of the sensory wave already on the GPU.
    ///
    /// Alternative to `upload_input`, for the `gpu_evolve` path, where the wave never
    /// comes down to the CPU. Must be chained BEFORE `dispatch` in the same encoder.
    pub fn dispatch_gather(
        &self,
        enc: &mut wgpu::CommandEncoder,
        wave_buf: &wgpu::Buffer,
        n_beings: usize,
    ) {
        let n = n_beings.min(self.capacity);
        if n == 0 {
            return;
        }

        let params = SlowBrainParams {
            n_beings: n as u32,
            dt: 0.0,
            _pad0: 0,
            _pad1: 0,
        };
        let buf_params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("slow-gather-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout_gather,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wave_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_input.as_entire_binding(),
                },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("slow-gather-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline_gather);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(n as u32, 1, 1);
    }

    pub fn dispatch(&self, enc: &mut wgpu::CommandEncoder, n_beings: usize, dt: f32) -> u64 {
        let n = n_beings.min(self.capacity);
        if n == 0 {
            return 0;
        }

        let params = SlowBrainParams {
            n_beings: n as u32,
            dt,
            _pad0: 0,
            _pad1: 0,
        };
        let buf_params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("slow-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buf_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buf_state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buf_output.as_entire_binding(),
                },
            ],
        });

        // Kernel A: CfC — 1 workgroup per being, 64 threads per workgroup
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass-slow-cfc"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_cfc);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(n as u32, 1, 1);
        }

        // Kernel B: Dense — same bind group, chained in the same encoder.
        // No explicit barrier is needed here because two dispatches in the
        // same encoder already guarantee execution order in wgpu/Vulkan.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass-slow-dense"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_dense);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(n as u32, 1, 1);
        }

        let bytes_output = (n * OUTPUT_DIM * 4) as u64;

        // Copies output to readback in the same encoder (DMA, no round-trip).
        enc.copy_buffer_to_buffer(&self.buf_output, 0, &self.buf_readback, 0, bytes_output);

        bytes_output
    }

    // ── INPUT UPLOAD (convenience) ────────────────────────────────────────────

    /// Converts each being's i8 input to i32 (sign-extended) and uploads to the GPU.
    ///
    /// `inputs`: slice of n_beings × 48 bytes (i8 as bytes, cast to i32 here).
    /// Call BEFORE `dispatch()` or before the external encoder's submit.
    pub fn upload_input(&self, inputs: &[[i8; INPUT_SIZE]], n_beings: usize) {
        let n = n_beings.min(self.capacity).min(inputs.len());
        let mut data: Vec<i32> = Vec::with_capacity(n * INPUT_SIZE);
        for inp in inputs[..n].iter() {
            for &b in inp.iter() {
                data.push(b as i32);
            }
        }
        self.queue
            .write_buffer(&self.buf_input, 0, bytemuck::cast_slice(&data));
    }

    // ── OUTPUT READ ────────────────────────────────────────────────────────────

    /// Reads the outputs from buf_readback after device.poll(Wait).
    ///
    /// MUST be called after the encoder that contained `dispatch()` was submitted
    /// and `device.poll(Wait)` returned. Follows the mandatory order:
    ///   map_async → poll(Wait) → recv → get_mapped_range
    ///
    /// (poll BEFORE map_async is the silent deadlock documented in brain.rs:710)
    pub fn read_outputs(&self, n_beings: usize) -> Vec<[f32; OUTPUT_DIM]> {
        let n = n_beings.min(self.capacity);
        if n == 0 {
            return Vec::new();
        }

        let bytes = (n * OUTPUT_DIM * 4) as u64;
        let slice = self.buf_readback.slice(0..bytes);

        let (tx, rx) = std::sync::mpsc::channel();
        // map_async registers the callback — poll(Wait) below will serve it.
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });

        // poll AFTER map_async: without this the callback is never served.
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

    // ── CONVENIENCE: dispatch + submit + poll + read (all-in-one) ─────────────

    /// Runs the full cycle: upload input → dispatch → submit → poll → read.
    ///
    /// Equivalent to `BrainGpu::forward()` — submits its own encoder.
    /// Use when the SlowBrain is the only kernel to dispatch this tick.
    /// For chaining with other kernels, prefer `dispatch()` + external submit.
    pub fn forward(
        &self,
        inputs: &[[i8; INPUT_SIZE]],
        n_beings: usize,
        dt: f32,
    ) -> Vec<[f32; OUTPUT_DIM]> {
        let n = n_beings.min(self.capacity).min(inputs.len());
        if n == 0 {
            return Vec::new();
        }

        self.upload_input(inputs, n);

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("slow-brain-forward"),
            });

        let _bytes = self.dispatch(&mut enc, n, dt);
        self.queue.submit(Some(enc.finish()));

        self.read_outputs(n)
    }

    // ── STATE RESET FOR ONE BEING ──────────────────────────────────────────────

    /// Zeroes the CfC state, the fatigue and the counters of being `idx`.
    ///
    /// Equivalent to `CfCCell::reset()` + restarting tick_counter + clearing the output.
    /// Called on spawn/death — a newborn being starts with no accumulated state.
    pub fn reset_being(&self, idx: usize) {
        if idx >= self.capacity {
            return;
        }
        let offset = (idx * STATE_PER_SER * 4) as u64;
        let zeros: Vec<u8> = vec![0u8; STATE_PER_SER * 4];
        self.queue.write_buffer(&self.buf_state, offset, &zeros);
    }
}

// ── BEVY RESOURCE ─────────────────────────────────────────────────────────────

/// Bevy resource keeping the SlowBrain GPU context between ticks.
///
/// Identical pattern to BrainGpuRes (brain.rs): `ctx: Option<SlowBrainGpu>`,
/// created/recreated if the capacity is insufficient.
#[derive(bevy::prelude::Resource, Default)]
pub struct SlowBrainGpuRes {
    pub ctx: Option<SlowBrainGpu>,
}

impl SlowBrainGpuRes {
    /// Returns a valid context, creating one if needed. None = no usable GPU.
    pub fn get_or_create(
        &mut self,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        capacity: usize,
    ) -> Option<&SlowBrainGpu> {
        let needs = self.ctx.is_none()
            || self
                .ctx
                .as_ref()
                .map(|c| c.capacity < capacity)
                .unwrap_or(true);
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

// ── HOW TO EXTRACT SlowBrain WEIGHTS (reference for the caller) ───────────────
//
// To call upload_weights(weights, n):
//
//   let mut all_weights: Vec<f32> = Vec::new();
//   for engine in engines[..n].iter() {
//       all_weights.extend(engine.slow_brain.all_weights());
//   }
//   slow_brain_gpu.upload_weights(&all_weights, n);
//
// all_weights() returns (engine.rs:125-132):
//   dense1.weights[1024] ++ dense1.bias[16] ++ dense2.weights[48] ++ dense2.bias[3]
// = 1091 f32, same order as buf_weights here.
//
// ── EXPECTED GPU×CPU DIVERGENCE ──────────────────────────────────────────────
//
// - CfC (64 exp/tanh/hash ops): ≤ 1 ULP per channel. Drift between ticks can
//   accumulate but is deterministic (hash identical to cfc.rs).
// - Dense1 matmul (64 FMA): < 1e-5 per neuron. Same pattern as brain.wgsl L1.
// - Dense2 matmul (16 FMA): < 1e-6 per neuron (16× fewer terms than L1).
// - tanh outputs: ≤ 1 ULP.
//
// The tick_counter verification is deterministic: being N runs on the same tick
// on the GPU and on the CPU if tick_interval=8 and the counters started at 0.
//
// ── WHAT I DID NOT RUN ──────────────────────────────────────────────────────
//
// The GPU was busy (task constraint). Nothing was executed on the GPU.
// Compilation was verified by cargo build (see output in the final report).
