// ── EQUIPOTENT CORTICAL SHEET — RUST HOST ─────────────────────────────────────
//
// Manages the 4 kernels of the emergent cortical sheet:
//
//   Kernel A (afference):   injects sensory_wave with gaussian bias + spontaneous waves
//   Kernel B (hebb):        hebbian plasticity with subtractive normalization
//   Kernel C (motor):       projects activations → [throttle, steer, brake]
//
// # How to plug into gpu_loop
//
//   1. Add `cortex: CortexSheet` to the `GpuLoop` struct.
//   2. In `GpuLoop::new()`, call `CortexSheet::new(Arc::clone(&device), Arc::clone(&queue), cap)`.
//   3. At the start of every tick, call `cortex.upload_wave(&queue, &wave_data[..n])`.
//   4. Add the dispatches to the encoder BEFORE the readback:
//
//      cortex.dispatch_afference(&mut enc, n, tick);
//      cortex.dispatch_hebb(&mut enc, &queue, n);   // reads buf_act post-afference
//      cortex.dispatch_motor(&mut enc, n);
//
//   5. After submit+poll, call `cortex.read_motor(&device, &queue)` to obtain
//      [throttle, steer, brake] per being.
//
// # Buffer anatomy
//
//   buf_wave:  i32[cap × 223]          sensory_wave per being (overwritten every tick)
//   buf_tau:   f32[cap × 1024]         time constants, log-spaced, per unit
//   buf_state: f32[cap × 1024]         EMA state resident between ticks
//   buf_act:   f32[cap × 1024]         tanh(state) activation, written every tick
//   buf_afw:   f32[cap × 1024 × 223]   hebbian afference weights (learnt)
//   buf_mw:    f32[cap × 1024 × 3]     motor weights (throttle/steer/brake)
//   buf_motor: f32[cap × 3]            motor output [−1,1]
//
// # Sizes (cap=60)
//   buf_afw: 60 × 1024 × 223 × 4 = ~55 MB  (largest buffer — VRAM)
//   buf_tau: 60 × 1024 × 4 = ~246 KB
//   buf_act: 60 × 1024 × 4 = ~246 KB
//
// # Proof of heterogeneous activity
//   See `CortexSheet::read_act_stats`: returns (min, max, mean, std) of the 1024
//   activations of one being. If min == max → degenerate sheet. Real heterogeneity
//   shows up as std > 0.01 and max-min > 0.1 after the first ticks.

use std::borrow::Cow;
use std::sync::Arc;
use wgpu::util::DeviceExt;

// ── Layout constants ──────────────────────────────────────────────────────────

/// Cortical sheet dimension: 32 × 32 units.
pub const SHEET_W: usize = 32;
pub const SHEET_H: usize = 32;
pub const SHEET_N: usize = SHEET_W * SHEET_H; // 1024 units

/// Sensory input channels.
const N_INPUTS: usize = 223;

/// Motor outputs: throttle=0, steer=1, brake=2.
const N_MOTOR: usize = 3;

// Minimum and maximum τ of the log-spaced bank (in seconds).
// gamma ≈ 1.6 ms → delta ≈ 318 ms (doc §5.3)
const TAU_MIN: f32 = 0.0016; // 1.6 ms — gamma band
// Clippy sees 0.318 and reports it as an approximation of `FRAC_1_PI` (0.31831).
// That is a numeric coincidence, not the meaning: this is 318 ms, the delta band,
// and replacing it with the constant would tie a physiological time constant to π.
#[allow(clippy::approx_constant)]
const TAU_MAX: f32 = 0.318; // 318 ms — delta band

// ── Uniform struct mirroring WGSL ─────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsAfference {
    n_beings: u32,
    tick: u32,
    // fat_beta: inhibition strength of the adaptive fatigue (mechanism 1). 0.0=off.
    // Comes from the CortexSheet::fat_beta field, configurable via the --fat-beta arg.
    fat_beta: f32,
    // div_sigma: gain of the divisive normalization (mechanism 2). 0.0=off.
    // Comes from the CortexSheet::div_sigma field, configurable via the --div-sigma arg.
    div_sigma: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsHebb {
    n_beings: u32,
    tick: u32,
    act_sum: f32,
    _pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ParamsMotor {
    n_beings: u32,
    tick: u32,
    _pad0: u32,
    _pad1: u32,
}

// ── Bind group layout helpers ────────────────────────────────────────────────

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

// ── Cortical sheet context ────────────────────────────────────────────────────

/// Equipotent cortical sheet: 1024 units per being, emergent organization.
pub struct CortexSheet {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    // Pipelines of the 3 kernels
    pipeline_aff: wgpu::ComputePipeline,
    pipeline_hebb: wgpu::ComputePipeline,
    pipeline_motor: wgpu::ComputePipeline,

    // Bind group layouts
    layout_aff: wgpu::BindGroupLayout,
    layout_hebb: wgpu::BindGroupLayout,
    layout_motor: wgpu::BindGroupLayout,

    // Persistent buffers
    buf_wave: wgpu::Buffer,  // i32 × (cap × N_INPUTS) — overwritten every tick
    buf_tau: wgpu::Buffer,   // f32 × (cap × SHEET_N)  — constant after init
    buf_state: wgpu::Buffer, // f32 × (cap × SHEET_N)  — resident EMA state
    buf_act: wgpu::Buffer,   // f32 × (cap × SHEET_N)  — activation after kernel A
    buf_afw: wgpu::Buffer,   // f32 × (cap × SHEET_N × N_INPUTS) — hebbian weights
    buf_mw: wgpu::Buffer,    // f32 × (cap × SHEET_N × N_MOTOR) — motor weights
    buf_motor: wgpu::Buffer, // f32 × (cap × N_MOTOR)  — motor output
    // Adaptive fatigue per unit (spike-frequency adaptation — anti-saturation)
    // Grows when |act| stays high for many ticks; subtracts from net_input, breaking the fixed point.
    buf_fat: wgpu::Buffer, // f32 × (cap × SHEET_N)  — slow fatigue state

    // Readback buffers for CPU reads of the state and the motor output
    buf_motor_rb: wgpu::Buffer,
    buf_act_rb: wgpu::Buffer,

    /// Maximum being capacity
    pub capacity: usize,

    /// fat_beta: inhibition strength of the adaptive fatigue (mechanism 1).
    /// 0.0 = off (SATURATED sheet), 1.2 = original strong value.
    /// Passed to the shader via uniform on every dispatch_afference.
    pub fat_beta: f32,

    /// div_sigma: gain of the divisive normalization over the neighborhood (mechanism 2).
    /// 0.0 = off, 2.0 = original strong value.
    /// Passed to the shader via uniform on every dispatch_afference.
    pub div_sigma: f32,

    /// Cache of being 0's τ (last updated via set_tau_cache).
    /// None = homogeneous default (before the first genome is loaded).
    /// Avoids a GPU→CPU readback on every tau_bank() call.
    tau_cache: Option<Vec<f32>>,
}

impl CortexSheet {
    /// Creates a new cortical sheet for `capacity` beings.
    ///
    /// The τ are drawn log-linearly between TAU_MIN and TAU_MAX, using the
    /// unit index as seed — reproducible and with no region label.
    ///
    /// The buf_afw weights start at 1.0 (pure gaussian prior, no learning).
    /// The buf_mw weights start with small random values (seed 0xBEEF).
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, capacity: usize) -> Self {
        let cap = capacity.max(1);

        // ── Compile shaders ──────────────────────────────────────────────────
        // Each kernel in its own file — prevents the ACO SIGSEGV in shaders >1050 lines.
        let mod_aff = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cortex-afference"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("cortex_sheet.wgsl"))),
        });
        let mod_hebb = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cortex-hebb"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("cortex_sheet_hebb.wgsl"))),
        });
        let mod_motor = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cortex-motor"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "cortex_sheet_motor.wgsl"
            ))),
        });

        // ── Layout A (afference): uniform + wave(r) + afw(rw) + state(rw) + tau(r) + act(rw) + fat(rw) ──
        let layout_aff = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cortex-aff-layout"),
            entries: &[
                uniform_entry(0),        // params
                storage_entry(1, true),  // buf_wave (read)
                storage_entry(2, false), // buf_afw  (read_write — read for the prior)
                storage_entry(3, false), // buf_state (read_write)
                storage_entry(4, true),  // buf_tau  (read)
                storage_entry(5, false), // buf_act  (read_write — reads neighbors t-1, writes t)
                storage_entry(6, false), // buf_fat  (read_write — adaptive fatigue)
            ],
        });

        // ── Layout B (hebb): uniform + act(r) + wave(r) + afw(rw) ──────────
        let layout_hebb = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cortex-hebb-layout"),
            entries: &[
                uniform_entry(0),        // params (includes act_sum)
                storage_entry(1, true),  // buf_act  (read)
                storage_entry(2, true),  // buf_wave (read)
                storage_entry(3, false), // buf_afw  (read_write)
            ],
        });

        // ── Layout C (motor): uniform + act(r) + mw(r) + motor(rw) ─────────
        let layout_motor = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cortex-motor-layout"),
            entries: &[
                uniform_entry(0),        // params
                storage_entry(1, true),  // buf_act   (read)
                storage_entry(2, true),  // buf_mw    (read)
                storage_entry(3, false), // buf_motor (write)
            ],
        });

        // ── Pipelines ────────────────────────────────────────────────────────
        let pl_aff = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout_aff],
            push_constant_ranges: &[],
        });
        let pl_hebb = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout_hebb],
            push_constant_ranges: &[],
        });
        let pl_motor = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout_motor],
            push_constant_ranges: &[],
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

        // buf_wave: i32 per channel per being (overwritten every tick)
        let buf_wave = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cortex-wave"),
            size: (cap * N_INPUTS * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // buf_tau: log-spaced between TAU_MIN and TAU_MAX, deterministic per index.
        // MUTABLE (COPY_DST): the genome can inject τ per part via write_buffer,
        // enabling a bipartite split by neurogenetic schedule without recompiling the shader.
        // τ_i = TAU_MIN × (TAU_MAX/TAU_MIN)^(i/(SHEET_N−1)) — homogeneous default value.
        let tau_data = default_tau_homogeneous(cap);
        let buf_tau = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cortex-tau"),
            contents: bytemuck::cast_slice(&tau_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // buf_state: zeros (initial EMA state)
        let state_zeros: Vec<f32> = vec![0.0; cap * SHEET_N];
        let buf_state = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cortex-state"),
            contents: bytemuck::cast_slice(&state_zeros),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        // buf_act: zeros (initial activation)
        let buf_act = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cortex-act"),
            size: (cap * SHEET_N * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // buf_fat: adaptive fatigue per unit — initialized to zeros.
        // Grows with sustained |act|; decays when the unit goes silent.
        // Size: cap × SHEET_N × 4 bytes (~246 KB @ cap=60)
        let fat_zeros: Vec<f32> = vec![0.0f32; cap * SHEET_N];
        let buf_fat = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cortex-fat"),
            contents: bytemuck::cast_slice(&fat_zeros),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // buf_afw: initialized to 1.0 (pure gaussian prior — no initial hebbian bias)
        // Size: cap × SHEET_N × N_INPUTS × 4 bytes ≈ 55 MB @ cap=60
        let afw_ones: Vec<f32> = vec![1.0; cap * SHEET_N * N_INPUTS];
        let buf_afw = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cortex-afw"),
            contents: bytemuck::cast_slice(&afw_ones),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // buf_mw: motor weights initialized with small pseudo-noise (index hash)
        // No zeros — zeros make the motor deaf to the sheet's initial state.
        let mw_data: Vec<f32> = (0..cap)
            .flat_map(|ser| {
                (0..SHEET_N).flat_map(move |u| {
                    (0..N_MOTOR).map(move |m| {
                        // Knuth hash to generate a ±small initial value
                        let h: u32 = ((ser as u32) * 1000003u32)
                            .wrapping_add((u as u32) * 2654435761u32)
                            .wrapping_add(m as u32 * 16777619u32);
                        // Maps [0, 2^32) → [−0.1, 0.1]
                        let norm = (h as f32 / u32::MAX as f32) - 0.5;
                        norm * 0.2 // amplitude ±0.1
                    })
                })
            })
            .collect();
        let buf_mw = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cortex-mw"),
            contents: bytemuck::cast_slice(&mw_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // buf_motor: [throttle, steer, brake] output per being
        let buf_motor = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cortex-motor"),
            size: (cap * N_MOTOR * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Readback buffers (separate MAP_READ — wgpu requires it)
        let buf_motor_rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cortex-motor-rb"),
            size: (cap * N_MOTOR * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_act_rb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cortex-act-rb"),
            size: (cap * SHEET_N * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        eprintln!(
            "[cortex_sheet] Sheet 32×32={} units × {} beings | τ ∈ [{:.1}ms, {:.0}ms]",
            SHEET_N,
            cap,
            TAU_MIN * 1000.0,
            TAU_MAX * 1000.0
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
            // Default values: STRONG inhibition (original behavior).
            // Can be overridden via CortexSheet::set_inhibition() before the first tick.
            fat_beta: 1.2,
            div_sigma: 2.0,
            tau_cache: None, // filled by set_tau_cache() after the first genome
        }
    }

    /// Configures the anti-saturation inhibition parameters at runtime.
    ///
    /// Must be called BEFORE the first dispatch_afference of a new sweep.
    /// The values are passed to the shader via uniform on every dispatch — no recompilation.
    ///
    /// - fat_beta=0.0, div_sigma=0.0 → SATURATED sheet (sweep baseline).
    /// - fat_beta=1.2, div_sigma=2.0 → STRONG inhibition (original measured value).
    pub fn set_inhibition(&mut self, fat_beta: f32, div_sigma: f32) {
        self.fat_beta = fat_beta;
        self.div_sigma = div_sigma;
    }

    // ── sensory_wave upload ───────────────────────────────────────────────────

    /// Uploads the sensory_wave of `n` beings to the GPU.
    ///
    /// `waves`: slice of i8 arrays, one per being, N_INPUTS elements each.
    /// Converts i8 → i32 (sign-extended) for the shader.
    pub fn upload_wave(&self, waves: &[[i8; 223]], n: usize) {
        let n = n.min(self.capacity).min(waves.len());
        let data: Vec<i32> = waves[..n]
            .iter()
            .flat_map(|w| w.iter().map(|&b| b as i32))
            .collect();
        self.queue
            .write_buffer(&self.buf_wave, 0, bytemuck::cast_slice(&data));
    }

    // ── Kernel dispatches (added to the external encoder) ─────────────────────

    /// Dispatch of Kernel A (afference + EMA + activation).
    /// Writes to buf_state and buf_act.
    /// dispatch: (SHEET_N/32, n_beings, 1) = (32, n, 1)
    ///
    /// fat_beta and div_sigma are read from self (configured via set_inhibition).
    /// This allows a parametric sweep without recompiling the shader.
    pub fn dispatch_afference(&self, enc: &mut wgpu::CommandEncoder, n: usize, tick: u32) {
        let n = n.min(self.capacity) as u32;
        let params = ParamsAfference {
            n_beings: n,
            tick,
            fat_beta: self.fat_beta,
            div_sigma: self.div_sigma,
        };
        let buf_params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cortex-aff-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout_aff,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buf_wave.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_afw.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buf_state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buf_tau.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.buf_act.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.buf_fat.as_entire_binding(),
                },
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

    /// Dispatch of Kernel B (hebbian plasticity).
    ///
    /// IMPORTANT: must be called AFTER `dispatch_afference` has been added
    /// to the same encoder (or to the previously submitted encoder), since it reads buf_act.
    ///
    /// `act_sum`: Σ_j |act_j| of the previous tick (or 0.0 at tick 0).
    /// For tick 0, act_sum=0 → Hebb only, no normalization (ok).
    pub fn dispatch_hebb(&self, enc: &mut wgpu::CommandEncoder, n: usize, tick: u32, act_sum: f32) {
        let n = n.min(self.capacity) as u32;
        let params = ParamsHebb {
            n_beings: n,
            tick,
            act_sum,
            _pad0: 0,
        };
        let buf_params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cortex-hebb-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout_hebb,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buf_act.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_wave.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buf_afw.as_entire_binding(),
                },
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

    /// Dispatch of Kernel C (motor projection).
    /// Reads buf_act, writes buf_motor.
    pub fn dispatch_motor(&self, enc: &mut wgpu::CommandEncoder, n: usize, tick: u32) {
        let n = n.min(self.capacity) as u32;
        let params = ParamsMotor {
            n_beings: n,
            tick,
            _pad0: 0,
            _pad1: 0,
        };
        let buf_params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cortex-motor-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout_motor,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buf_act.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_mw.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buf_motor.as_entire_binding(),
                },
            ],
        });
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cortex-motor"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline_motor);
        pass.set_bind_group(0, &bg, &[]);
        // 1 workgroup per being, workgroup_size(64)
        pass.dispatch_workgroups(n, 1, 1);
    }

    /// Adds readback copies to the encoder (for CPU reads after submit).
    pub fn copy_for_readback(&self, enc: &mut wgpu::CommandEncoder, n: usize) {
        let n = n.min(self.capacity);
        // Motor output: n × 3 × 4 bytes
        enc.copy_buffer_to_buffer(
            &self.buf_motor,
            0,
            &self.buf_motor_rb,
            0,
            (n * N_MOTOR * 4) as u64,
        );
        // Act (for statistics): n × SHEET_N × 4 bytes
        enc.copy_buffer_to_buffer(
            &self.buf_act,
            0,
            &self.buf_act_rb,
            0,
            (n * SHEET_N * 4) as u64,
        );
    }

    // ── CPU reads (post submit+poll) ─────────────────────────────────────────

    /// Reads [throttle, steer, brake] of every being after submit+poll.
    /// Returns a Vec of len n, each element = [throttle, steer, brake].
    pub fn read_motor(&self, n: usize) -> Vec<[f32; 3]> {
        let n = n.min(self.capacity);
        let bytes = (n * N_MOTOR * 4) as u64;

        let slice = self.buf_motor_rb.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
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

    /// Reads the sheet activations for heterogeneity diagnostics.
    ///
    /// Returns per-being statistics: (min, max, mean, std) of the 1024 activations.
    /// Real proof: std > 0.01 and (max − min) > 0.1 after tick > 5.
    ///
    /// # Non-fabrication proof
    /// The values are read directly from buf_act_rb (GPU MAP_READ).
    /// Each unit has a different τ → decays at a different rate → min ≠ max.
    pub fn read_act_stats(&self, n: usize) -> Vec<(f32, f32, f32, f32)> {
        let n = n.min(self.capacity);
        let bytes = (n * SHEET_N * 4) as u64;

        let slice = self.buf_act_rb.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
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
                if v < mn {
                    mn = v;
                }
                if v > mx {
                    mx = v;
                }
                sum += v;
            }
            let mean = sum / SHEET_N as f32;

            let var: f32 = slice_s
                .iter()
                .map(|&v| (v - mean) * (v - mean))
                .sum::<f32>()
                / SHEET_N as f32;
            let std = var.sqrt();

            stats.push((mn, mx, mean, std));
        }

        drop(data);
        self.buf_act_rb.unmap();
        stats
    }

    /// Reads the raw activations (1024 values per being) after submit+poll.
    ///
    /// Returns a Vec<f32> with n × SHEET_N values: [ser0_unit0, ..., ser0_unit1023, ser1_unit0, ...].
    ///
    /// Used by cortex_map to compute modality dominance via response difference
    /// and to accumulate mean activity per unit across ticks.
    pub fn read_act_raw(&self, n: usize) -> Vec<f32> {
        let n = n.min(self.capacity);
        let total = n * SHEET_N;
        let bytes = (total * 4) as u64;

        let slice = self.buf_act_rb.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let result = floats[..total].to_vec();
        drop(data);
        self.buf_act_rb.unmap();
        result
    }

    // ── Buffer access for genome loading ─────────────────────────────────────

    /// Reference to buf_afw (hebbian afference weights) for external write_buffer.
    /// Layout: being × SHEET_N × N_INPUTS f32. Used by cortex_evolve to inject
    /// the genome's initial weights before each generation.
    pub fn buf_afw_ref(&self) -> &wgpu::Buffer {
        &self.buf_afw
    }

    /// Reference to buf_mw (motor weights) for external write_buffer.
    /// Layout: being × SHEET_N × N_MOTOR f32.
    pub fn buf_mw_ref(&self) -> &wgpu::Buffer {
        &self.buf_mw
    }

    /// Reference to buf_state (EMA state) for reset between generations.
    /// Layout: being × SHEET_N f32. Writing zeros resets the sheet.
    pub fn buf_state_ref(&self) -> &wgpu::Buffer {
        &self.buf_state
    }

    /// Reference to buf_fat (adaptive fatigue) for reset between generations.
    /// Layout: being × SHEET_N f32. Writing zeros resets the accumulated fatigue.
    /// MUST be reset together with buf_state to prevent residual fatigue from
    /// one generation penalizing the start of the next (initialization bias).
    pub fn buf_fat_ref(&self) -> &wgpu::Buffer {
        &self.buf_fat
    }

    /// Returns the τ of each unit of being 0 (for τ bank inspection).
    ///
    /// Returns the cached copy (generated on the CPU together with the genome-loaded data).
    /// If no genome was loaded, returns the homogeneous log-spaced default.
    pub fn tau_bank(&self) -> Vec<f32> {
        if let Some(ref cache) = self.tau_cache {
            cache.clone()
        } else {
            default_tau_homogeneous(1)[..SHEET_N].to_vec()
        }
    }

    /// Reference to buf_tau (time constants) for external write_buffer.
    /// Layout: being × SHEET_N f32.
    /// Used by cortex_evolve to inject τ per part according to the genome's schedule.
    pub fn buf_tau_ref(&self) -> &wgpu::Buffer {
        &self.buf_tau
    }

    /// Updates the τ cache of being 0 (so tau_bank() reflects the loaded genome).
    ///
    /// Must be called whenever buf_tau is updated via write_buffer.
    /// The cache avoids a GPU→CPU readback on every inspection.
    pub fn set_tau_cache(&mut self, taus_ser0: Vec<f32>) {
        self.tau_cache = Some(taus_ser0);
    }
}

// ── Bipartite schedule helper functions ───────────────────────────────────────

/// Generates homogeneous log-spaced τ (default without a genome).
/// Returns cap × SHEET_N f32 values.
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

/// Computes the bipartite split by the neurogenetic schedule.
///
/// Finlay & Darlington (1995) model: N_part ∝ 2^((t_end − t_start) × rate).
/// Parts (fixed start/end order):
///   - cerebellum: start=0.35, end=0.80  → fast band (gamma/beta)
///   - cortex:     start=0.55, end=1.00 + schedule_ext → slow band (theta/delta)
///
/// `schedule_ext` is the only evolvable parameter (extension of the cortical window).
/// Clamped to [0.0, 0.5] — values above 0.5 distort beyond the plausible.
///
/// Returns: (n_cerebellum, n_cortex, taus_per_unit[SHEET_N]).
///
/// τ distribution:
///   - cerebellum (units 0..n_cerebellum-1): τ ∈ [TAU_MIN, TAU_MID]  (gamma+beta)
///   - cortex     (units n_cerebellum..SHEET_N-1): τ ∈ [TAU_MID, TAU_MAX] (theta+delta)
///
/// Sanity test: cortex/cerebellum ratio must be in [0.15, 0.40] (≈1:4 biological).
/// The proportion EMERGES from schedule_ext, it is not imposed.
pub fn compute_bipart_schedule(schedule_ext: f32) -> (usize, usize, Vec<f32>) {
    // Schedule parameters (Finlay & Darlington §2 of the doc)
    // RANGE OPENED UP TO 2.0. It was locked at 0.5, which limited the cortex to ~30% and
    // prevented evolution from even TESTING cortex-heavy configurations.
    //
    // Reason to open it: the 1:4 ratio (80% cerebellum) is the fact conserved in MAMMALS, and
    // a mammal has hundreds of muscles with fine motor control. **Our body is a car with
    // THREE actuators** (throttle, steering, brake). The N1 finding of this ecosystem — "the body
    // dictates the architecture", correlation +0.972 — predicts that the optimal proportion follows
    // the body, not the biology of another body.
    //
    // Measured: 1:4 bipartite came in BELOW the homogeneous sheet (med10 110 m vs 164 m,
    // max10 727 m vs 886 m). With 80% of the units at short τ (1.6–12 ms), little long-trajectory
    // integration is left — which is exactly what broke the wall before.
    let ext = schedule_ext.clamp(0.0, 2.0_f32);
    let taxa: f32 = 4.0; // proliferation exponent (log2 of the expansion per unit of time)

    // Duration of each part in the schedule calibrated for the 1:4 biological ratio:
    //   ext=0.0  →  cortex=20.0%, cerebellum=80.0%  (exact 1:4 ratio)
    //   ext=0.1  →  cortex=24.8%, cerebellum=75.2%
    //   ext=0.3  →  cortex=36.5%, cerebellum=63.5%
    //   ext=0.5  →  cortex=50.0%, cerebellum=50.0%
    //
    // Formula: dur_cb and dur_cx satisfy 2^(dur_cb−dur_cx)=4 when ext=0.
    // → dur_cb = dur_cx + 2/taxa = 0.45 + 0.50 = 0.95
    let dur_cerebelo = 0.95_f32; // calibrated: cerebellum=80% with ext=0
    let dur_cortex = 0.45_f32 + ext; // cortex=20% with ext=0

    // Relative N: 2^(dur × rate) — "late equals large"
    let k_cerebelo = (dur_cerebelo * taxa).exp2();
    let k_cortex = (dur_cortex * taxa).exp2();
    let k_total = k_cerebelo + k_cortex;

    // Absolute N: proportion × SHEET_N (minimum 1 each)
    let n_cerebelo = ((k_cerebelo / k_total) * SHEET_N as f32).round() as usize;
    let n_cerebelo = n_cerebelo.clamp(1, SHEET_N - 1);
    let n_cortex = SHEET_N - n_cerebelo;

    // τ distribution per part:
    //   cerebellum → τ ∈ [TAU_MIN, TAU_MID]  (gamma/beta: 1.6 ms – 12 ms)
    //   cortex     → τ ∈ [TAU_MID, TAU_MAX]  (theta/delta: 12 ms – 318 ms)
    //
    // TAU_MID is the cut point between beta and alpha (≈12 ms = 1/(2π·13Hz))
    const TAU_MID: f32 = 0.012; // 12 ms — beta/alpha boundary (doc §5.3)

    let log_r_cb = (TAU_MID / TAU_MIN).ln();
    let log_r_cx = (TAU_MAX / TAU_MID).ln();

    let mut taus = vec![0.0_f32; SHEET_N];

    // Cerebellum: first n_cerebelo units — short τ (gamma/beta)
    for i in 0..n_cerebelo {
        let t = if n_cerebelo > 1 {
            i as f32 / (n_cerebelo as f32 - 1.0)
        } else {
            0.0
        };
        taus[i] = TAU_MIN * (log_r_cb * t).exp();
    }
    // Cortex: remaining units — long τ (theta/delta)
    for i in 0..n_cortex {
        let t = if n_cortex > 1 {
            i as f32 / (n_cortex as f32 - 1.0)
        } else {
            0.0
        };
        taus[n_cerebelo + i] = TAU_MID * (log_r_cx * t).exp();
    }

    (n_cerebelo, n_cortex, taus)
}
