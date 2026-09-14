//! Host of `fly_map.wgsl`: the fixed connectome resident on the GPU, stepped every tick.
//!
//! The map never changes, so its seven buffers are uploaded once, in [`FlyMapGpu::new`],
//! and never touched again. What changes per generation is the SELECTION the genomes
//! carry ([`FlyMapGpu::upload_selection`]); what changes per tick is the state, which
//! stays on the device — only the three commands per being come back to the host.
//!
//! # Why the buffers are separate
//!
//! `indices` and `weights` are 60 MB each on the full map (15.1 M edges). The device is
//! created with `downlevel_defaults`, whose storage-binding cap is 128 MiB, so merging
//! them into one interleaved buffer would put a 120 MB binding one bad map away from the
//! limit — and a binding over the cap is a validation error at bind-group creation, in the
//! middle of a run. They stay separate, and [`FlyMapGpu::new`] refuses politely if any
//! single buffer would exceed the cap instead of letting wgpu panic later.
//!
//! # Device requirement
//!
//! `fly_step` binds 12 storage buffers, so the device must have been created with
//! `max_storage_buffers_per_shader_stage >= 12`. `gpu_evolve` already asks for 16
//! (`context.rs`), and the parity test asks for the same; `downlevel_defaults` alone
//! (4) is not enough and `create_bind_group_layout` says so.

use std::borrow::Cow;
use std::cell::Cell;
use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::fly_map::{COMMANDS, FlyMap, FlySelection, NONE, OUT_SLOTS};
use crate::gpu::core::{storage_entry, uniform_entry};
use crate::gpu::perception::WAVE_U32S;

/// Binding cap of `downlevel_defaults` — the ceiling every single buffer here respects.
const MAX_BINDING_BYTES: u64 = 128 * 1024 * 1024;

/// Default `tau_steps` of the rate model: `alpha = 1 / 4` (§2).
pub const DEFAULT_TAU_STEPS: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlyParams {
    n: u32,
    n_beings: u32,
    k_in: u32,
    n_types: u32,
    alpha: f32,
    wave_stride_u32: u32,
    none_id: u32,
    k_out: u32,
}

/// The connectome, resident on the GPU, plus the per-being selection and state.
pub struct FlyMapGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    pipeline_step: wgpu::ComputePipeline,
    pipeline_read: wgpu::ComputePipeline,
    layout_step: wgpu::BindGroupLayout,
    layout_read: wgpu::BindGroupLayout,

    // ── The map: uploaded once, read-only for the rest of the run ───────────
    buf_indptr: wgpu::Buffer,
    buf_indices: wgpu::Buffer,
    buf_weights: wgpu::Buffer,
    buf_type_of: wgpu::Buffer,
    buf_in_group: wgpu::Buffer,
    buf_out_indptr: wgpu::Buffer,
    buf_out_members: wgpu::Buffer,

    // ── The selection: uploaded once per generation ─────────────────────────
    buf_group_to_input: wgpu::Buffer,
    buf_out_pair: wgpu::Buffer,
    buf_gain_scale: wgpu::Buffer,
    buf_type_mask: wgpu::Buffer,

    // ── The state: resident, never read back ────────────────────────────────
    buf_x: wgpu::Buffer,
    buf_r: [wgpu::Buffer; 2],
    buf_cmd: wgpu::Buffer,
    buf_readback: wgpu::Buffer,

    n: usize,
    k_in: usize,
    k_out: usize,
    n_types: usize,
    capacity: usize,
    alpha: f32,
    /// Which half of the double buffer holds the CURRENT output. Flipped by every
    /// `dispatch`; a `Cell` so the tick can keep calling through a `&self`, like every
    /// other kernel host in this directory.
    phase: Cell<usize>,
}

impl FlyMapGpu {
    /// Uploads `map` and allocates room for `capacity` beings.
    ///
    /// `tau_steps` is the rate model's time constant in ticks (`alpha = 1 / tau_steps`);
    /// 0 is read as 1, because an alpha of infinity is not a configuration.
    ///
    /// Returns `None` when a buffer would exceed the 128 MiB binding cap — with the
    /// numbers printed, because the answer is "use the `--min-weight 3` export", not
    /// "buy a GPU".
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        map: &FlyMap,
        capacity: usize,
        tau_steps: u32,
    ) -> Option<Self> {
        let n = map.n();
        let cap = capacity.max(1);
        let state_bytes = (n * cap * 4) as u64;
        let edge_bytes = (map.m() * 4) as u64;
        if state_bytes > MAX_BINDING_BYTES || edge_bytes > MAX_BINDING_BYTES {
            eprintln!(
                "[fly-map] refusing to allocate: state {:.1} MiB per buffer (n={n} x B={cap}) \
                 and edges {:.1} MiB per buffer (m={}); the cap per storage binding is \
                 {} MiB. Use a smaller map (fly_export.py --min-weight 3 / --exclude-optic) \
                 or fewer beings.",
                state_bytes as f64 / (1024.0 * 1024.0),
                edge_bytes as f64 / (1024.0 * 1024.0),
                map.m(),
                MAX_BINDING_BYTES / (1024 * 1024)
            );
            return None;
        }

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fly-map"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("fly_map.wgsl"))),
        });

        // Two layouts, one per entry point, each carrying only what its kernel binds.
        // One shared layout would have to declare all 16 storage buffers, which is
        // exactly the device limit and leaves no room for the next binding.
        let layout_step = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fly-step-layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),   // indptr
                storage_entry(2, true),   // indices
                storage_entry(3, true),   // weights
                storage_entry(4, true),   // type_of
                storage_entry(5, true),   // in_group_of
                storage_entry(6, true),   // wave (PerceptionGpu::buf_output)
                storage_entry(7, true),   // group_to_input
                storage_entry(8, true),   // gain_scale
                storage_entry(9, true),   // type_mask
                storage_entry(10, true),  // r_old
                storage_entry(11, false), // r_new
                storage_entry(12, false), // x
            ],
        });
        let layout_read = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fly-read-layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(11, false), // r_new — the half `fly_step` just wrote
                storage_entry(13, true),  // out_indptr
                storage_entry(14, true),  // out_members
                storage_entry(15, true),  // out_pair
                storage_entry(16, false), // cmd
            ],
        });
        let pipeline_step = make_pipeline(&device, &module, &layout_step, "fly_step");
        let pipeline_read = make_pipeline(&device, &module, &layout_read, "fly_read");

        let ro = wgpu::BufferUsages::STORAGE;
        let rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let buf_indptr = init_u32(&device, "fly-indptr", &map.indptr, ro);
        let buf_indices = init_u32(&device, "fly-indices", &map.indices, ro);
        let buf_weights = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fly-weights"),
            contents: bytemuck::cast_slice(&map.weights),
            usage: ro,
        });
        let buf_type_of = init_u32(&device, "fly-type-of", &map.type_of, ro);
        let buf_in_group = init_u32(&device, "fly-in-group", &map.in_group_of, ro);
        let buf_out_indptr = init_u32(&device, "fly-out-indptr", &map.out_indptr, ro);
        let buf_out_members = init_u32(&device, "fly-out-members", &map.out_members, ro);

        // Selection tables: born INERT (every group unbound, mask on, gains 1), so a
        // dispatch that happens before the first `upload_selection` commands zero rather
        // than whatever the allocator left behind.
        let k_in = map.k_in.max(1);
        let k_out = map.k_out();
        let n_types = map.n_types.max(1);
        let buf_group_to_input = init_u32(&device, "fly-g2i", &vec![NONE; cap * k_in], rw);
        let buf_out_pair = init_u32(&device, "fly-out-pair", &vec![NONE; cap * OUT_SLOTS], rw);
        let buf_gain_scale = init_f32(&device, "fly-gain-scale", &vec![1.0f32; cap * 2], rw);
        let buf_type_mask = init_f32(&device, "fly-type-mask", &vec![1.0f32; cap * n_types], rw);

        let buf_x = zeroed(&device, "fly-x", state_bytes, rw);
        // COPY_SRC on both halves of `r`: `debug_read_state` copies whichever one the last
        // dispatch wrote, and wgpu rejects the copy at ENCODE time without the flag — the
        // failure lands in the parity test, not in the run, but it lands.
        let r_usage = rw | wgpu::BufferUsages::COPY_SRC;
        let buf_r = [
            zeroed(&device, "fly-r-a", state_bytes, r_usage),
            zeroed(&device, "fly-r-b", state_bytes, r_usage),
        ];
        let cmd_bytes = (cap * COMMANDS * 4) as u64;
        let buf_cmd = zeroed(
            &device,
            "fly-cmd",
            cmd_bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        );
        let buf_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fly-cmd-readback"),
            size: cmd_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Some(Self {
            device,
            queue,
            pipeline_step,
            pipeline_read,
            layout_step,
            layout_read,
            buf_indptr,
            buf_indices,
            buf_weights,
            buf_type_of,
            buf_in_group,
            buf_out_indptr,
            buf_out_members,
            buf_group_to_input,
            buf_out_pair,
            buf_gain_scale,
            buf_type_mask,
            buf_x,
            buf_r,
            buf_cmd,
            buf_readback,
            n,
            k_in,
            k_out,
            n_types,
            capacity: cap,
            alpha: 1.0 / tau_steps.max(1) as f32,
            phase: Cell::new(0),
        })
    }

    /// `alpha = 1 / tau_steps`, the value the kernel integrates with.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Beings these buffers can serve.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Uploads the per-being tables of §3. Once per generation, like `upload_gates`.
    ///
    /// A selection packed against a different layout is refused rather than uploaded
    /// half-way: the tables index each other, and a partial upload would mix two maps.
    pub fn upload_selection(&self, sel: &FlySelection, n_beings: usize) -> bool {
        let b = n_beings.min(self.capacity).min(sel.n_beings);
        if b == 0 {
            return false;
        }
        if sel.layout.k_in != self.k_in
            || sel.layout.n_types != self.n_types
            || sel.layout.k_out != self.k_out
        {
            eprintln!(
                "[fly-map] selection packed for k_in={} k_out={} n_types={} but the loaded \
                 map is k_in={} k_out={} n_types={} — not uploading.",
                sel.layout.k_in,
                sel.layout.k_out,
                sel.layout.n_types,
                self.k_in,
                self.k_out,
                self.n_types
            );
            return false;
        }
        self.queue.write_buffer(
            &self.buf_group_to_input,
            0,
            bytemuck::cast_slice(&sel.group_to_input[..b * self.k_in]),
        );
        self.queue.write_buffer(
            &self.buf_out_pair,
            0,
            bytemuck::cast_slice(&sel.out_pair[..b * OUT_SLOTS]),
        );
        self.queue.write_buffer(
            &self.buf_gain_scale,
            0,
            bytemuck::cast_slice(&sel.gain_scale[..b * 2]),
        );
        self.queue.write_buffer(
            &self.buf_type_mask,
            0,
            bytemuck::cast_slice(&sel.type_mask[..b * self.n_types]),
        );
        true
    }

    /// Zeroes `x` and both halves of `r` — a newborn generation starts at x = 0 (§2).
    ///
    /// Same reason `GpuCtx` resets `buf_cfc` between generations: recurrent state that
    /// crosses the boundary makes the champion's re-evaluation non-deterministic.
    pub fn reset_state(&self) {
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fly-reset"),
            });
        enc.clear_buffer(&self.buf_x, 0, None);
        enc.clear_buffer(&self.buf_r[0], 0, None);
        enc.clear_buffer(&self.buf_r[1], 0, None);
        enc.clear_buffer(&self.buf_cmd, 0, None);
        self.queue.submit(Some(enc.finish()));
        self.phase.set(0);
    }

    /// Records `fly_step` + `fly_read` into an existing encoder and queues the command
    /// copy — the same contract as `SlowBrainGpu::dispatch`: no submit, no readback here.
    ///
    /// `wave_buf` is `PerceptionGpu::buf_output()`, so this must be recorded AFTER the
    /// perception kernels have written the wave for this tick.
    pub fn dispatch(
        &self,
        enc: &mut wgpu::CommandEncoder,
        n_beings: usize,
        wave_buf: &wgpu::Buffer,
    ) {
        let b = n_beings.min(self.capacity);
        if b == 0 || self.n == 0 {
            return;
        }
        let params = FlyParams {
            n: self.n as u32,
            n_beings: b as u32,
            k_in: self.k_in as u32,
            n_types: self.n_types as u32,
            alpha: self.alpha,
            wave_stride_u32: WAVE_U32S as u32,
            none_id: NONE,
            k_out: self.k_out as u32,
        };
        let buf_params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fly-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let old = self.phase.get();
        let new = 1 - old;
        let bg_step = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fly-step-bg"),
            layout: &self.layout_step,
            entries: &[
                entry(0, &buf_params),
                entry(1, &self.buf_indptr),
                entry(2, &self.buf_indices),
                entry(3, &self.buf_weights),
                entry(4, &self.buf_type_of),
                entry(5, &self.buf_in_group),
                entry(6, wave_buf),
                entry(7, &self.buf_group_to_input),
                entry(8, &self.buf_gain_scale),
                entry(9, &self.buf_type_mask),
                entry(10, &self.buf_r[old]),
                entry(11, &self.buf_r[new]),
                entry(12, &self.buf_x),
            ],
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fly-step-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_step);
            pass.set_bind_group(0, &bg_step, &[]);
            pass.dispatch_workgroups((self.n as u32).div_ceil(128), 1, 1);
        }

        let bg_read = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fly-read-bg"),
            layout: &self.layout_read,
            entries: &[
                entry(0, &buf_params),
                entry(11, &self.buf_r[new]),
                entry(13, &self.buf_out_indptr),
                entry(14, &self.buf_out_members),
                entry(15, &self.buf_out_pair),
                entry(16, &self.buf_cmd),
            ],
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fly-read-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_read);
            pass.set_bind_group(0, &bg_read, &[]);
            pass.dispatch_workgroups(((b * COMMANDS) as u32).div_ceil(64), 1, 1);
        }

        enc.copy_buffer_to_buffer(
            &self.buf_cmd,
            0,
            &self.buf_readback,
            0,
            (b * COMMANDS * 4) as u64,
        );
        self.phase.set(new);
    }

    /// The three commands per being, after the encoder that held [`dispatch`] was
    /// submitted and `device.poll(Wait)` returned.
    ///
    /// Order is `map_async` → `poll` → `recv`, never `poll` first: that is the silent
    /// deadlock this project already paid for (`brain.rs`, `slow_brain.rs`).
    pub fn read_commands_after_submit(&self, n_beings: usize) -> Vec<[f32; COMMANDS]> {
        let b = n_beings.min(self.capacity);
        if b == 0 {
            return Vec::new();
        }
        let bytes = (b * COMMANDS * 4) as u64;
        let slice = self.buf_readback.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();
        let data = slice.get_mapped_range();
        let vals: &[f32] = bytemuck::cast_slice(&data);
        let mut out = Vec::with_capacity(b);
        for i in 0..b {
            let base = i * COMMANDS;
            let mut row = [0.0f32; COMMANDS];
            row.copy_from_slice(&vals[base..base + COMMANDS]);
            out.push(row);
        }
        drop(data);
        self.buf_readback.unmap();
        out
    }

    /// The activation buffer the last [`dispatch`] wrote — the one `fly_read` averaged,
    /// as `n × n_beings` floats, being-minor.
    ///
    /// Exposed for the parity test; nothing on the run path reads it. The first
    /// `n * n_beings` floats ARE the whole state, because the kernel strides by
    /// `n_beings` and not by `capacity` — which is also why a generation must not change
    /// `n_beings` half-way without a [`FlyMapGpu::reset_state`]: the stride would move
    /// under the state that is already there. `gpu_evolve` holds `n` fixed for a whole
    /// generation and resets between them.
    pub fn debug_read_state(&self, n_beings: usize) -> Vec<f32> {
        let b = n_beings.min(self.capacity);
        let bytes = (self.n * b * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fly-state-staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.buf_r[self.phase.get()], 0, &staging, 0, bytes);
        self.queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();
        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        out
    }
}

fn make_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    layout: &wgpu::BindGroupLayout,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("fly-map-pl"),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: Some(&pl),
        module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn entry<'a>(binding: u32, buf: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    }
}

fn init_u32(
    device: &wgpu::Device,
    label: &str,
    data: &[u32],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    // An empty slice is a valid map (a group with no members); wgpu rejects a zero-sized
    // buffer, so it becomes one padding word that nothing indexes.
    let one = [0u32];
    let src = if data.is_empty() { &one[..] } else { data };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(src),
        usage,
    })
}

fn init_f32(
    device: &wgpu::Device,
    label: &str,
    data: &[f32],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let one = [0.0f32];
    let src = if data.is_empty() { &one[..] } else { data };
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(src),
        usage,
    })
}

fn zeroed(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(4),
        usage,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
#[path = "parity_test.rs"]
mod parity_test;
