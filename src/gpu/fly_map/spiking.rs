//! GPU host for the event-driven spiking fly map.
//!
//! K1 propagates delayed spikes through the v2 presynaptic CSR using an atomic integer
//! synapse-count accumulator. K2 adds reproducible eye events without a host round trip.
//! K3 performs the leaky plus/minus readout.  The CPU contract lives in
//! `crate::fly_map::spiking`; this module owns only device state and command recording.

use std::borrow::Cow;
use std::cell::Cell;
use std::num::NonZeroU64;
use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::config::iara::vision::RETINA_TOTAL;
use crate::fly_map::spiking::{EyeMode, SpikingParams, SpikingTraceFrame};
use crate::fly_map::{FlyMap, FlySelection, COMMANDS, NONE, OUT_SLOTS};
use crate::gpu::core::{storage_entry, uniform_entry};

const MAX_SUBSTEPS: usize = 1024;
const UNIFORM_STRIDE: u64 = 256;
const MAX_BINDING_BYTES: u64 = 128 * 1024 * 1024;
const MAX_DISPATCH_X: u32 = 65_535;

/// Traffic counters for the spiking subsystem.
///
/// `queue.write_buffer` is a CPU→GPU upload through wgpu's staging path. A
/// `copy_buffer_to_buffer` stays inside the device, even when its destination is a
/// MAP_READ staging buffer; the latter becomes GPU→CPU traffic only when the host maps
/// it. Keeping these three classes separate makes `--kernel-profile` useful for finding
/// synchronization regressions instead of collapsing all time into one number.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FlySpikingTraffic {
    /// Bytes uploaded from host memory (selection, reset state and the batched uniforms).
    pub cpu_to_gpu_bytes: u64,
    /// Subset of [`cpu_to_gpu_bytes`] written while dispatching ticks (uniform schedules).
    /// Construction, selection and reset uploads are therefore visible separately.
    pub tick_cpu_to_gpu_bytes: u64,
    /// Bytes copied between GPU buffers (retina history and command staging).
    pub gpu_internal_copy_bytes: u64,
    /// Subset of [`gpu_internal_copy_bytes`] recorded while dispatching ticks.
    pub tick_gpu_internal_copy_bytes: u64,
    /// Bytes mapped back to host memory (commands, once per tick).
    pub gpu_to_cpu_bytes: u64,
    /// Subset of [`gpu_to_cpu_bytes`] mapped after a tick submit.
    pub tick_gpu_to_cpu_bytes: u64,
    /// Number of host uploads issued with `queue.write_buffer`.
    pub cpu_to_gpu_calls: u64,
    /// Number of tick uniform uploads. A healthy run has one per dispatched tick.
    pub tick_cpu_to_gpu_calls: u64,
    /// Number of host readback maps.
    pub gpu_to_cpu_calls: u64,
    /// Number of tick readback maps, including opt-in trace snapshots.
    pub tick_gpu_to_cpu_calls: u64,
    /// Number of ticks dispatched by this instance.
    pub ticks: u64,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SpikingUniform {
    n: u32,
    n_beings: u32,
    capacity: u32,
    n_words: u32,
    n_types: u32,
    delay_steps: u32,
    slot: u32,
    eye_on: u32,
    eye_mode: u32,
    step_lo: u32,
    step_hi: u32,
    seed_lo: u32,
    seed_hi: u32,
    dt_ms: f32,
    v0: f32,
    v_rst: f32,
    v_th: f32,
    decay_v: f32,
    decay_syn: f32,
    w_syn: f32,
    poisson_factor: f32,
    background_hz: f32,
    decay_read: f32,
    read_norm: f32,
    retina_stride: u32,
    refractory_steps: u32,
    _pad0: u32,
    _pad1: u32,
    debug_being: u32,
    integrate_groups_x: u32,
    _debug_pad1: u32,
    _debug_pad2: u32,
}

/// GPU-resident LIF state for one fixed connectome and a fixed population capacity.
pub struct FlySpikingGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline_propagate: wgpu::ComputePipeline,
    pipeline_integrate: wgpu::ComputePipeline,
    pipeline_readout: wgpu::ComputePipeline,
    pipeline_debug: wgpu::ComputePipeline,
    layout_propagate: wgpu::BindGroupLayout,
    layout_integrate: wgpu::BindGroupLayout,
    layout_readout: wgpu::BindGroupLayout,
    layout_debug: wgpu::BindGroupLayout,
    buf_params: wgpu::Buffer,
    buf_pre_indptr: wgpu::Buffer,
    buf_pre_indices: wgpu::Buffer,
    buf_pre_weight_units: wgpu::Buffer,
    buf_type_of: wgpu::Buffer,
    buf_type_mask: wgpu::Buffer,
    buf_gain_scale: wgpu::Buffer,
    buf_spike_ring: wgpu::Buffer,
    buf_syn_acc: wgpu::Buffer,
    buf_pixel_of_neuron: wgpu::Buffer,
    buf_no_refractory: wgpu::Buffer,
    buf_retina_prev: wgpu::Buffer,
    buf_v: wgpu::Buffer,
    buf_i_syn: wgpu::Buffer,
    buf_refractory: wgpu::Buffer,
    buf_spike_counts: wgpu::Buffer,
    buf_out_indptr: wgpu::Buffer,
    buf_out_members: wgpu::Buffer,
    buf_out_pair: wgpu::Buffer,
    buf_read_count: wgpu::Buffer,
    buf_command: wgpu::Buffer,
    buf_readback: wgpu::Buffer,
    buf_debug_v: wgpu::Buffer,
    buf_debug_counts: wgpu::Buffer,
    buf_debug_v_readback: wgpu::Buffer,
    buf_debug_counts_readback: wgpu::Buffer,
    n: usize,
    capacity: usize,
    n_types: usize,
    n_words: usize,
    delay_steps: usize,
    params: SpikingParams,
    step: Cell<u64>,
    traffic: Cell<FlySpikingTraffic>,
    debug_being: Cell<u32>,
}

impl FlySpikingGpu {
    /// Creates the spiking context. A v2 presynaptic CSR is required; silently rebuilding
    /// it from the postsynaptic CSR would turn construction into an O(m) surprise and make
    /// the map's file contract ambiguous.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        map: &FlyMap,
        capacity: usize,
        params: SpikingParams,
    ) -> Option<Self> {
        let pre = match map.pre_csr.as_ref() {
            Some(pre) => pre,
            None => {
                eprintln!("[fly-spiking] map has no presynaptic CSR (v2 flag bit 0)");
                return None;
            }
        };
        if map.n() == 0 || map.n_types == 0 {
            eprintln!("[fly-spiking] map must contain neurons and at least one cell type");
            return None;
        }
        let units: Vec<i32> = match pre
            .weights
            .iter()
            .map(|&w| {
                if !w.is_finite() || w.fract() != 0.0 || w < i32::MIN as f32 || w > i32::MAX as f32
                {
                    None
                } else {
                    Some(w as i32)
                }
            })
            .collect()
        {
            Some(v) => v,
            None => {
                eprintln!(
                    "[fly-spiking] presynaptic weights must be finite integer synapse counts"
                );
                return None;
            }
        };
        if pre.indptr.len() != map.n() + 1 || pre.indices.len() != map.m() || units.len() != map.m()
        {
            eprintln!("[fly-spiking] presynaptic CSR shape does not match the map");
            return None;
        }
        if pre.indptr.windows(2).any(|w| w[1] < w[0])
            || pre.indptr.last().copied().unwrap_or(0) as usize != map.m()
            || pre.indices.iter().any(|&post| post as usize >= map.n())
            || map.type_of.iter().any(|&ty| ty as usize >= map.n_types)
        {
            eprintln!("[fly-spiking] connectome indices/types are outside their declared ranges");
            return None;
        }
        if let Some(pixels) = &map.pixel_of_neuron {
            if pixels.len() != map.n()
                || pixels
                    .iter()
                    .any(|&pixel| pixel != NONE && pixel as usize >= RETINA_TOTAL)
            {
                eprintln!("[fly-spiking] pixel binding has an invalid length or retina index");
                return None;
            }
        }
        if let Some(bitmap) = &map.no_refractory {
            if bitmap.len() != map.n().div_ceil(32) {
                eprintln!("[fly-spiking] no-refractory bitmap has the wrong length");
                return None;
            }
        }
        if map.out_indptr.is_empty()
            || map.out_indptr.last().copied().unwrap_or(0) as usize != map.out_members.len()
            || map
                .out_members
                .iter()
                .any(|&neuron| neuron as usize >= map.n())
        {
            eprintln!("[fly-spiking] output groups are outside their declared ranges");
            return None;
        }
        // The accumulator is i32 because WGSL has portable integer atomics but no
        // portable f32 atomics. Bound the worst-case absolute one-substep delivery per
        // destination so a pathological map cannot wrap the accumulator and invert the
        // sign of a synaptic current.
        let mut post_abs = vec![0u64; map.n()];
        for (&post, &weight) in pre.indices.iter().zip(units.iter()) {
            let post = post as usize;
            let next = post_abs[post].saturating_add((weight as i64).unsigned_abs());
            if next > i32::MAX as u64 {
                eprintln!(
                    "[fly-spiking] absolute synapse delivery for neuron {post} exceeds i32 accumulator range"
                );
                return None;
            }
            post_abs[post] = next;
        }
        let cap = capacity.max(1);
        let n = map.n();
        let n_types = map.n_types.max(1);
        let n_words = cap.div_ceil(32);
        let delay_steps = params.delay_steps();
        let neuron_state_bytes = n.checked_mul(cap)?.checked_mul(4)? as u64;
        let ring_words = delay_steps.checked_mul(n)?.checked_mul(n_words)?;
        let ring_bytes = ring_words.checked_mul(4)? as u64;
        let retina_bytes = cap.checked_mul(RETINA_TOTAL)?.checked_mul(4)? as u64;
        if neuron_state_bytes > MAX_BINDING_BYTES || ring_bytes > MAX_BINDING_BYTES {
            eprintln!(
                "[fly-spiking] state/ring exceeds the per-binding cap: state={} MiB ring={} MiB",
                neuron_state_bytes / (1024 * 1024),
                ring_bytes / (1024 * 1024)
            );
            return None;
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fly-spiking"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("spiking.wgsl"))),
        });
        let layout_propagate = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fly-spiking-propagate-layout"),
            entries: &[
                dynamic_uniform(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, true),
                storage_entry(6, false),
                storage_entry(7, false),
            ],
        });
        let layout_integrate = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fly-spiking-integrate-layout"),
            entries: &[
                dynamic_uniform(0),
                storage_entry(5, true),
                storage_entry(6, false),
                storage_entry(7, false),
                storage_entry(8, true),
                storage_entry(9, true),
                storage_entry(10, true),
                storage_entry(11, true),
                storage_entry(12, true),
                storage_entry(13, false),
                storage_entry(14, false),
                storage_entry(15, false),
                storage_entry(16, false),
            ],
        });
        let layout_readout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fly-spiking-readout-layout"),
            entries: &[
                dynamic_uniform(0),
                // `spike_counts` is atomic and declared read_write in WGSL even though
                // K3 only loads it; the resource access class must still be writable.
                storage_entry(16, false),
                storage_entry(17, true),
                storage_entry(18, true),
                storage_entry(19, true),
                storage_entry(20, false),
                storage_entry(21, false),
            ],
        });
        let layout_debug = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fly-spiking-debug-layout"),
            entries: &[
                dynamic_uniform(0),
                storage_entry(13, false),
                storage_entry(16, false),
                storage_entry(22, false),
                storage_entry(23, false),
            ],
        });
        let pipeline_propagate = make_pipeline(&device, &shader, &layout_propagate, "propagate");
        let pipeline_integrate = make_pipeline(&device, &shader, &layout_integrate, "integrate");
        let pipeline_readout = make_pipeline(&device, &shader, &layout_readout, "readout");
        let pipeline_debug = make_pipeline(&device, &shader, &layout_debug, "debug_snapshot");

        let ro = wgpu::BufferUsages::STORAGE;
        let rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let buf_pre_indptr = init_u32(&device, "fly-spike-pre-indptr", &pre.indptr, ro);
        let buf_pre_indices = init_u32(&device, "fly-spike-pre-indices", &pre.indices, ro);
        let buf_pre_weight_units = init_i32(&device, "fly-spike-pre-weights", &units, ro);
        let buf_type_of = init_u32(&device, "fly-spike-type-of", &map.type_of, ro);
        let pixels = map.pixel_of_neuron.clone().unwrap_or_else(|| vec![NONE; n]);
        let buf_pixel_of_neuron = init_u32(&device, "fly-spike-pixels", &pixels, ro);
        let no_ref = map
            .no_refractory
            .clone()
            .unwrap_or_else(|| vec![0; n.div_ceil(32)]);
        let buf_no_refractory = init_u32(&device, "fly-spike-no-refractory", &no_ref, ro);
        let buf_out_indptr = init_u32(&device, "fly-spike-out-indptr", &map.out_indptr, ro);
        let buf_out_members = init_u32(&device, "fly-spike-out-members", &map.out_members, ro);

        let mask_bytes = (cap * n_types * 4) as u64;
        let buf_type_mask = zeroed(&device, "fly-spike-type-mask", mask_bytes, rw);
        let buf_gain_scale = init_f32(&device, "fly-spike-gain-scale", &vec![1.0; cap * 2], rw);
        let ring_usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let buf_spike_ring = zeroed(&device, "fly-spike-ring", ring_bytes, ring_usage);
        let buf_syn_acc = zeroed(
            &device,
            "fly-spike-accumulator",
            neuron_state_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let buf_retina_prev = zeroed(
            &device,
            "fly-spike-retina-prev",
            retina_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let buf_v = init_f32(&device, "fly-spike-v", &vec![params.v0; n * cap], rw);
        let buf_i_syn = zeroed(&device, "fly-spike-i-syn", neuron_state_bytes, rw);
        let buf_refractory = zeroed(&device, "fly-spike-refractory", neuron_state_bytes, rw);
        let buf_spike_counts = zeroed(
            &device,
            "fly-spike-counts",
            neuron_state_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let buf_out_pair = zeroed(
            &device,
            "fly-spike-out-pair",
            (cap * OUT_SLOTS * 4) as u64,
            rw,
        );
        let buf_read_count = zeroed(
            &device,
            "fly-spike-read-count",
            (cap * COMMANDS * 4) as u64,
            rw,
        );
        let buf_command = zeroed(
            &device,
            "fly-spike-command",
            (cap * COMMANDS * 4) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let buf_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fly-spike-command-readback"),
            size: (cap * COMMANDS * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_debug_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fly-spike-debug-v"),
            size: (n * 4).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_debug_counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fly-spike-debug-counts"),
            size: (n * 4).max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_debug_v_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fly-spike-debug-v-readback"),
            size: (n * 4).max(4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_debug_counts_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fly-spike-debug-counts-readback"),
            size: (n * 4).max(4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fly-spike-params"),
            size: UNIFORM_STRIDE * MAX_SUBSTEPS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // `create_buffer_init` performs the only host upload during construction. Keep
        // it in the same ledger as later queue writes so a run report can distinguish a
        // one-time map upload from per-tick synchronization.
        let initialization_bytes = (pre.indptr.len()
            + pre.indices.len()
            + units.len()
            + map.type_of.len()
            + pixels.len()
            + no_ref.len()
            + map.out_indptr.len()
            + map.out_members.len()
            + cap * 2
            + n * cap) as u64
            * 4;

        Some(Self {
            device,
            queue,
            pipeline_propagate,
            pipeline_integrate,
            pipeline_readout,
            pipeline_debug,
            layout_propagate,
            layout_integrate,
            layout_readout,
            layout_debug,
            buf_params,
            buf_pre_indptr,
            buf_pre_indices,
            buf_pre_weight_units,
            buf_type_of,
            buf_type_mask,
            buf_gain_scale,
            buf_spike_ring,
            buf_syn_acc,
            buf_pixel_of_neuron,
            buf_no_refractory,
            buf_retina_prev,
            buf_v,
            buf_i_syn,
            buf_refractory,
            buf_spike_counts,
            buf_out_indptr,
            buf_out_members,
            buf_out_pair,
            buf_read_count,
            buf_command,
            buf_readback,
            buf_debug_v,
            buf_debug_counts,
            buf_debug_v_readback,
            buf_debug_counts_readback,
            n,
            capacity: cap,
            n_types,
            n_words,
            delay_steps,
            params,
            step: Cell::new(0),
            traffic: Cell::new(FlySpikingTraffic {
                cpu_to_gpu_bytes: initialization_bytes,
                cpu_to_gpu_calls: 10,
                ..Default::default()
            }),
            debug_being: Cell::new(0),
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn neurons(&self) -> usize {
        self.n
    }

    /// Bytes and host/device operations observed since construction.
    pub fn traffic(&self) -> FlySpikingTraffic {
        self.traffic.get()
    }

    /// Selects the being copied by the optional activation trace. The value is carried
    /// in the same dynamic uniform slot as the substep schedule, so enabling tracing
    /// never adds a CPU write or a synchronization to an ordinary tick.
    pub fn set_debug_being(&self, being: usize) {
        self.debug_being
            .set(being.min(self.capacity.saturating_sub(1)) as u32);
    }

    fn add_traffic(&self, f: impl FnOnce(&mut FlySpikingTraffic)) {
        let mut t = self.traffic.get();
        f(&mut t);
        self.traffic.set(t);
    }

    pub fn upload_selection(&self, sel: &FlySelection, n_beings: usize) -> bool {
        let b = n_beings.min(self.capacity).min(sel.n_beings);
        if b == 0 || sel.layout.n_types != self.n_types {
            return false;
        }
        self.queue.write_buffer(
            &self.buf_type_mask,
            0,
            bytemuck::cast_slice(&sel.type_mask[..b * self.n_types]),
        );
        self.queue.write_buffer(
            &self.buf_gain_scale,
            0,
            bytemuck::cast_slice(&sel.gain_scale[..b * 2]),
        );
        self.queue.write_buffer(
            &self.buf_out_pair,
            0,
            bytemuck::cast_slice(&sel.out_pair[..b * OUT_SLOTS]),
        );
        self.add_traffic(|t| {
            t.cpu_to_gpu_bytes += ((b * self.n_types + b * 2 + b * OUT_SLOTS) * 4) as u64;
            t.cpu_to_gpu_calls += 3;
        });
        true
    }

    pub fn reset_state(&self) {
        let v = vec![self.params.v0; self.n * self.capacity];
        let zeros = vec![0u8; self.n * self.capacity * 4];
        self.queue
            .write_buffer(&self.buf_v, 0, bytemuck::cast_slice(&v));
        for buffer in [
            &self.buf_i_syn,
            &self.buf_refractory,
            &self.buf_spike_counts,
            &self.buf_syn_acc,
        ] {
            self.queue.write_buffer(buffer, 0, &zeros);
        }
        let ring = vec![0u8; self.delay_steps * self.n * self.n_words * 4];
        self.queue.write_buffer(&self.buf_spike_ring, 0, &ring);
        let retina = vec![0u8; self.capacity * RETINA_TOTAL * 4];
        self.queue.write_buffer(&self.buf_retina_prev, 0, &retina);
        let read = vec![0u8; self.capacity * COMMANDS * 4];
        self.queue.write_buffer(&self.buf_read_count, 0, &read);
        self.queue.write_buffer(&self.buf_command, 0, &read);
        self.add_traffic(|t| {
            t.cpu_to_gpu_bytes +=
                (v.len() * 4 + zeros.len() + ring.len() + retina.len() + read.len() * 2) as u64;
            // v + four zeroed state buffers + ring + retina history + read_count + command.
            t.cpu_to_gpu_calls += 10;
        });
        self.step.set(0);
    }

    /// Test/diagnostic hook for injecting a membrane potential without a readback. It is
    /// intentionally a full fixed-stride upload: callers cannot accidentally use a
    /// compact `n × n_beings` row and shift the recurrent state when capacity > beings.
    /// Production code only uses [`reset_state`].
    #[doc(hidden)]
    pub fn debug_set_v(&self, values: &[f32]) -> bool {
        if values.len() != self.n * self.capacity || values.iter().any(|v| !v.is_finite()) {
            return false;
        }
        self.queue
            .write_buffer(&self.buf_v, 0, bytemuck::cast_slice(values));
        true
    }

    /// Records all LIF substeps for one Universe tick. No substep readback occurs.
    pub fn dispatch(
        &self,
        enc: &mut wgpu::CommandEncoder,
        n_beings: usize,
        retina_cur: &wgpu::Buffer,
        eye_on: bool,
        eye_mode: EyeMode,
        tick_dt_s: f32,
        seed: u64,
    ) {
        let b = n_beings.min(self.capacity);
        if b == 0 {
            return;
        }
        let substeps = ((tick_dt_s.max(0.0) * 1000.0) / self.params.dt_ms)
            .round()
            .max(1.0) as usize;
        assert!(
            substeps <= MAX_SUBSTEPS,
            "spiking tick exceeds MAX_SUBSTEPS"
        );
        let decay_read = self.params.decay_read(tick_dt_s.max(0.0) * 1000.0);
        let (integrate_groups_x, integrate_groups_y) =
            integrate_dispatch_grid(self.n, self.capacity);
        let base_step = self.step.get();
        let bind_prop = self.bind_propagate();
        let bind_int = self.bind_integrate(retina_cur);
        // Upload the whole substep schedule in one queue write.  A write per 0.1 ms
        // substep made the host-side recording cost scale with 156 even though the
        // kernels themselves stayed asynchronous; the dynamic offsets below still
        // select each 256-byte slot without any per-substep CPU/GPU synchronization.
        let uniforms: Vec<SpikingUniform> = (0..substeps)
            .map(|local| {
                self.uniform(
                    base_step + local as u64,
                    b,
                    eye_on,
                    eye_mode,
                    decay_read,
                    seed,
                )
            })
            .collect();
        let mut packed_uniforms = vec![0u8; substeps * UNIFORM_STRIDE as usize];
        for (local, u) in uniforms.iter().enumerate() {
            let start = local * UNIFORM_STRIDE as usize;
            packed_uniforms[start..start + std::mem::size_of::<SpikingUniform>()]
                .copy_from_slice(bytemuck::bytes_of(u));
        }
        self.queue
            .write_buffer(&self.buf_params, 0, &packed_uniforms);
        self.add_traffic(|t| {
            t.cpu_to_gpu_bytes += packed_uniforms.len() as u64;
            t.cpu_to_gpu_calls += 1;
            t.tick_cpu_to_gpu_bytes += packed_uniforms.len() as u64;
            t.tick_cpu_to_gpu_calls += 1;
            t.ticks += 1;
        });
        enc.clear_buffer(
            &self.buf_spike_counts,
            0,
            Some((self.n * self.capacity * 4) as u64),
        );
        for local in 0..substeps {
            let absolute = base_step + local as u64;
            let offset = local as u64 * UNIFORM_STRIDE;
            let dynamic = [offset as u32];
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("fly-spike-propagate"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_propagate);
                pass.set_bind_group(0, &bind_prop, &dynamic);
                pass.dispatch_workgroups(((self.n * self.n_words) as u32).div_ceil(64), 1, 1);
            }
            let ring_offset =
                ((absolute as usize % self.delay_steps) * self.n * self.n_words * 4) as u64;
            enc.clear_buffer(
                &self.buf_spike_ring,
                ring_offset,
                Some((self.n * self.n_words * 4) as u64),
            );
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("fly-spike-integrate"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline_integrate);
                pass.set_bind_group(0, &bind_int, &dynamic);
                // Vulkan/WebGPU caps one dispatch dimension at 65,535 workgroups. A
                // full map at B=40 needs 86,650 integrate groups, so split the linear
                // state space across Y and carry the X stride in Params.
                pass.dispatch_workgroups(integrate_groups_x, integrate_groups_y, 1);
            }
        }

        let final_local = substeps - 1;
        let final_offset = (final_local as u64 * UNIFORM_STRIDE) as u32;
        let bind_read = self.bind_readout();
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fly-spike-readout"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_readout);
            pass.set_bind_group(0, &bind_read, &[final_offset]);
            pass.dispatch_workgroups(((b * COMMANDS) as u32).div_ceil(64), 1, 1);
        }
        let bytes = (b * COMMANDS * 4) as u64;
        enc.copy_buffer_to_buffer(&self.buf_command, 0, &self.buf_readback, 0, bytes);
        // Preserve the current retina only after all K2 reads for this tick have been
        // recorded. The copy is ordered after the substeps in the same encoder.
        enc.copy_buffer_to_buffer(
            retina_cur,
            0,
            &self.buf_retina_prev,
            0,
            (b * RETINA_TOTAL * 4) as u64,
        );
        self.add_traffic(|t| {
            t.gpu_internal_copy_bytes += bytes + (b * RETINA_TOTAL * 4) as u64;
            t.tick_gpu_internal_copy_bytes += bytes + (b * RETINA_TOTAL * 4) as u64;
        });
        self.step.set(base_step + substeps as u64);
    }

    fn uniform(
        &self,
        step: u64,
        n_beings: usize,
        eye_on: bool,
        eye_mode: EyeMode,
        decay_read: f32,
        seed: u64,
    ) -> SpikingUniform {
        SpikingUniform {
            n: self.n as u32,
            n_beings: n_beings as u32,
            capacity: self.capacity as u32,
            n_words: self.n_words as u32,
            n_types: self.n_types as u32,
            delay_steps: self.delay_steps as u32,
            slot: (step as usize % self.delay_steps) as u32,
            eye_on: eye_on as u32,
            eye_mode: matches!(eye_mode, EyeMode::Motion) as u32,
            step_lo: step as u32,
            step_hi: (step >> 32) as u32,
            seed_lo: seed as u32,
            seed_hi: (seed >> 32) as u32,
            dt_ms: self.params.dt_ms,
            v0: self.params.v0,
            v_rst: self.params.v_rst,
            v_th: self.params.v_th,
            decay_v: self.params.decay_v(),
            decay_syn: self.params.decay_syn(),
            w_syn: self.params.w_syn,
            poisson_factor: self.params.poisson_factor,
            background_hz: self.params.background_hz,
            decay_read,
            read_norm: 1.0,
            retina_stride: RETINA_TOTAL as u32,
            refractory_steps: self.params.refractory_steps(),
            _pad0: 0,
            _pad1: 0,
            debug_being: self.debug_being.get(),
            integrate_groups_x: integrate_dispatch_grid(self.n, self.capacity).0,
            _debug_pad1: 0,
            _debug_pad2: 0,
        }
    }

    fn bind_propagate(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fly-spike-propagate-bg"),
            layout: &self.layout_propagate,
            entries: &[
                uniform_resource(0, &self.buf_params),
                entry(1, &self.buf_pre_indptr),
                entry(2, &self.buf_pre_indices),
                entry(3, &self.buf_pre_weight_units),
                entry(4, &self.buf_type_of),
                entry(5, &self.buf_type_mask),
                entry(6, &self.buf_spike_ring),
                entry(7, &self.buf_syn_acc),
            ],
        })
    }

    fn bind_integrate(&self, retina_cur: &wgpu::Buffer) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fly-spike-integrate-bg"),
            layout: &self.layout_integrate,
            entries: &[
                uniform_resource(0, &self.buf_params),
                entry(5, &self.buf_type_mask),
                entry(6, &self.buf_spike_ring),
                entry(7, &self.buf_syn_acc),
                entry(8, &self.buf_gain_scale),
                entry(9, &self.buf_pixel_of_neuron),
                entry(10, &self.buf_no_refractory),
                entry(11, retina_cur),
                entry(12, &self.buf_retina_prev),
                entry(13, &self.buf_v),
                entry(14, &self.buf_i_syn),
                entry(15, &self.buf_refractory),
                entry(16, &self.buf_spike_counts),
            ],
        })
    }

    fn bind_readout(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fly-spike-readout-bg"),
            layout: &self.layout_readout,
            entries: &[
                uniform_resource(0, &self.buf_params),
                entry(16, &self.buf_spike_counts),
                entry(17, &self.buf_out_indptr),
                entry(18, &self.buf_out_members),
                entry(19, &self.buf_out_pair),
                entry(20, &self.buf_read_count),
                entry(21, &self.buf_command),
            ],
        })
    }

    fn bind_debug(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fly-spiking-debug-bg"),
            layout: &self.layout_debug,
            entries: &[
                uniform_resource(0, &self.buf_params),
                entry(13, &self.buf_v),
                entry(16, &self.buf_spike_counts),
                entry(22, &self.buf_debug_v),
                entry(23, &self.buf_debug_counts),
            ],
        })
    }

    /// Records a compact copy of one being's per-neuron voltage and spike count into the
    /// same command encoder as the simulation. The compute pass gathers the being-minor
    /// fixed-stride state before the two small MAP_READ copies, so a 60-being run copies
    /// `n` values instead of `n × capacity` values. This method is intentionally opt-in:
    /// it is a diagnostic readback and must never be present in a production battery.
    pub fn record_debug_snapshot(&self, enc: &mut wgpu::CommandEncoder, n_beings: usize) -> bool {
        let b = n_beings.min(self.capacity);
        if b == 0 || self.debug_being.get() as usize >= b {
            return false;
        }
        let bind = self.bind_debug();
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fly-spike-debug-snapshot"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline_debug);
        // The debug shader only consumes `n`, `capacity`, `n_beings` and `debug_being`;
        // slot zero contains the same values as every substep of the current dispatch.
        pass.set_bind_group(0, &bind, &[0]);
        pass.dispatch_workgroups((self.n as u32).div_ceil(64), 1, 1);
        drop(pass);
        let bytes = (self.n * 4) as u64;
        enc.copy_buffer_to_buffer(&self.buf_debug_v, 0, &self.buf_debug_v_readback, 0, bytes);
        enc.copy_buffer_to_buffer(
            &self.buf_debug_counts,
            0,
            &self.buf_debug_counts_readback,
            0,
            bytes,
        );
        self.add_traffic(|t| {
            t.gpu_internal_copy_bytes += bytes * 2;
            t.tick_gpu_internal_copy_bytes += bytes * 2;
        });
        true
    }

    /// Maps the compact activation copy recorded by [`record_debug_snapshot`].
    pub fn read_debug_snapshot_after_submit(&self) -> Option<SpikingTraceFrame> {
        let bytes = (self.n * 4) as u64;
        let v_slice = self.buf_debug_v_readback.slice(0..bytes);
        let (v_tx, v_rx) = std::sync::mpsc::channel();
        v_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = v_tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        if !matches!(v_rx.recv(), Ok(Ok(()))) {
            return None;
        }
        let v_data = v_slice.get_mapped_range();
        let v = bytemuck::cast_slice::<u8, f32>(&v_data).to_vec();
        drop(v_data);
        self.buf_debug_v_readback.unmap();

        let c_slice = self.buf_debug_counts_readback.slice(0..bytes);
        let (c_tx, c_rx) = std::sync::mpsc::channel();
        c_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = c_tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        if !matches!(c_rx.recv(), Ok(Ok(()))) {
            return None;
        }
        let c_data = c_slice.get_mapped_range();
        let spike_counts = bytemuck::cast_slice::<u8, u32>(&c_data).to_vec();
        drop(c_data);
        self.buf_debug_counts_readback.unmap();
        self.add_traffic(|t| {
            t.gpu_to_cpu_bytes += bytes * 2;
            t.gpu_to_cpu_calls += 2;
            t.tick_gpu_to_cpu_bytes += bytes * 2;
            t.tick_gpu_to_cpu_calls += 2;
        });
        Some(SpikingTraceFrame {
            step: self.step.get(),
            commands: [0.0; COMMANDS],
            v,
            spike_counts,
        })
    }

    pub fn read_commands_after_submit(&self, n_beings: usize) -> Vec<[f32; COMMANDS]> {
        let b = n_beings.min(self.capacity);
        let bytes = (b * COMMANDS * 4) as u64;
        if b == 0 {
            return Vec::new();
        }
        let slice = self.buf_readback.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();
        let data = slice.get_mapped_range();
        let values: &[f32] = bytemuck::cast_slice(&data);
        let out = (0..b)
            .map(|i| {
                let mut row = [0.0; COMMANDS];
                row.copy_from_slice(&values[i * COMMANDS..(i + 1) * COMMANDS]);
                row
            })
            .collect();
        drop(data);
        self.buf_readback.unmap();
        self.add_traffic(|t| {
            t.gpu_to_cpu_bytes += bytes;
            t.gpu_to_cpu_calls += 1;
            t.tick_gpu_to_cpu_bytes += bytes;
            t.tick_gpu_to_cpu_calls += 1;
        });
        out
    }
}

fn dynamic_uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    let mut e = uniform_entry(binding);
    if let wgpu::BindingType::Buffer {
        has_dynamic_offset, ..
    } = &mut e.ty
    {
        *has_dynamic_offset = true;
    }
    e
}

fn integrate_dispatch_grid(n: usize, capacity: usize) -> (u32, u32) {
    let groups = (n * capacity).div_ceil(64) as u32;
    let x = groups.min(MAX_DISPATCH_X).max(1);
    (x, groups.div_ceil(x))
}

fn make_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::BindGroupLayout,
    entry_point: &str,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("fly-spiking-pipeline-layout"),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry_point),
        layout: Some(&pipeline_layout),
        module: shader,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn uniform_resource(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: 0,
            // Bind only one struct even though the backing ring holds all substep
            // uniforms. This keeps the binding under downlevel's 16 KiB uniform
            // range while dynamic offsets select the 256-byte slot at dispatch time.
            size: NonZeroU64::new(std::mem::size_of::<SpikingUniform>() as u64),
        }),
    }
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

fn init_u32(
    device: &wgpu::Device,
    label: &str,
    data: &[u32],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage,
    })
}

fn init_i32(
    device: &wgpu::Device,
    label: &str,
    data: &[i32],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage,
    })
}

fn init_f32(
    device: &wgpu::Device,
    label: &str,
    data: &[f32],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage,
    })
}

#[cfg(test)]
mod dispatch_tests {
    use super::{integrate_dispatch_grid, SpikingUniform, MAX_DISPATCH_X};

    #[test]
    fn integrate_grid_covers_large_population_without_exceeding_x_limit() {
        let (x, y) = integrate_dispatch_grid(138_639, 40);
        assert_eq!(x, MAX_DISPATCH_X);
        assert_eq!(y, 2);
        assert!(x as u64 * y as u64 * 64 >= 138_639 * 40);

        let (small_x, small_y) = integrate_dispatch_grid(2, 1);
        assert_eq!((small_x, small_y), (1, 1));
        assert_eq!(std::mem::size_of::<SpikingUniform>(), 128);
    }
}

#[cfg(test)]
#[path = "spiking_parity_test.rs"]
mod spiking_parity_test;
