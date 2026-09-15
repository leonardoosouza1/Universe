//! Brain on the GPU — the Rust host side.
//!
//! Packs every being's weights into SoA buffers, dispatches the `brain.wgsl` kernel
//! and returns `final_intent[96]` per being.
//!
//! # wgpu context
//!
//! We reuse the device that `RetinaGpu` already opened. `BrainGpu` takes an
//! `Arc<wgpu::Device>` + `Arc<wgpu::Queue>` so it does not open a second device: a GPU
//! has a single logical `VkDevice`, and opening two is wasteful and can fail outright
//! under Vulkan with some AMD driver versions.
//!
//! # Buffers that stay resident between ticks
//!
//! | Buffer | Rewritten |
//! |---|---|
//! | `buf_wave` | every tick (the sensory input changes every tick) |
//! | `buf_kan` | on spawn / evolution (weights only change between generations) |
//! | `buf_l1w` | same |
//! | `buf_l2w` | same |
//! | `buf_intent` | result, read back by the CPU every tick |
//! | `buf_cfc` | recurrent state, NEVER downloaded to the CPU per tick (only on spawn, where the host writes zeros, and on death, where the host reads it for a log) |
//!
//! # Why `Arc` and not references?
//!
//! `RetinaGpu` and `BrainGpu` live as two distinct Bevy resources. Bevy does not allow
//! two mutable resources of the same type, so the GPU contexts have to sit in separate
//! resources. `Arc` shares device/queue without a lifetime fight. Cost: one atomic
//! increment per tick, which is negligible.

use std::borrow::Cow;
use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::config::iara::brain::{L1_NEURONS, L2_NEURONS, SENSORY_WAVE_SIZE};
use crate::engine::SovereignEngine;
use crate::gpu::evolution::Genome;

mod screen;

#[cfg(test)]
mod observation_tests;

// ── CONSTANTS DERIVED FROM THE BRAIN DIMENSIONS ──────────────────────────────

/// KAN knots per sensory channel (`kan.rs`: `spline_coeffs: [f32; 16]`).
const KAN_KNOTS: usize = 16;

/// L1 weights per being: `SENSORY_WAVE_SIZE × L1_NEURONS` floats = 192 × 96 = 18 432.
///
/// `SENSORY_WAVE_SIZE` is *computed* (`config.rs:309`), so the constant is right no
/// matter what the comment says. The comment that used to sit here still quoted the
/// pre-migration wave of 223 channels ("223 × 96 = 21 408"); `src/CLAUDE.md` cites this
/// very line as the example of a comment lying while the code is correct.
const L1_WEIGHTS_PER_BEING: usize = SENSORY_WAVE_SIZE * L1_NEURONS;

/// L2 weights per being: `L1_NEURONS × L2_NEURONS` floats = 96 × 48 = 4 608.
const L2_WEIGHTS_PER_BEING: usize = L1_NEURONS * L2_NEURONS;

/// KAN knots per being: `SENSORY_WAVE_SIZE` channels × 16 knots = 192 × 16 = 3 072.
/// (The old comment said "223 × 16 = 3 568", from the 223-channel wave.)
const KAN_WEIGHTS_PER_BEING: usize = SENSORY_WAVE_SIZE * KAN_KNOTS;

/// Inherited output fatigue: 0.85. Documented on `BrainParams::out_beta`.
pub const OUT_BETA_DEFAULT: f32 = 0.85;

/// Inherited output gain: 2.0. Documented on `BrainParams::out_gain`.
///
/// This was the hardcoded constant `2.0` in STAGE 2 of `brain.wgsl` (`l2_raw = acc *
/// (2.0/127.0)` and `raw_bp = w_l1r[tid] * 2.0`) and in `engine.rs` (`raw = val * 2.0`
/// in both loops). It became a parameter (P1.5, `docs/PLANO_EVOLUCAO.md`) for the A/B
/// against L2 saturation (48/48 neurons active in 97.4% of the ticks, policy entropy
/// 0.301 nats — below the 0.50 threshold). Anyone who does not pass `--output-gain`
/// keeps exactly this value.
pub const OUT_GAIN_DEFAULT: f32 = 2.0;

/// CfC state per being: 96 floats (state) + 1 u32 (counter, bitcast as f32) = 97 f32.
const CFC_SLOTS_PER_BEING: usize = L1_NEURONS + 1; // 97

const FAST_CFC_STATE_BYTES: u64 = (L1_NEURONS * std::mem::size_of::<f32>()) as u64;

fn fast_state_copy_range(
    capacity: usize,
    being: usize,
    destination_size: u64,
    destination_offset: u64,
) -> Result<(u64, u64), String> {
    if being >= capacity {
        return Err(format!(
            "being index {being} exceeds brain capacity {capacity}"
        ));
    }
    if destination_offset % 4 != 0 {
        return Err("destination offset must be 4-byte aligned".to_string());
    }
    let end = destination_offset
        .checked_add(FAST_CFC_STATE_BYTES)
        .ok_or_else(|| "destination range overflows".to_string())?;
    if end > destination_size {
        return Err(format!(
            "destination range [{destination_offset}, {end}) exceeds buffer size {destination_size}"
        ));
    }
    let source = (being as u64)
        .checked_mul(CFC_SLOTS_PER_BEING as u64)
        .and_then(|slots| slots.checked_mul(std::mem::size_of::<f32>() as u64))
        .ok_or_else(|| "source range overflows".to_string())?;
    Ok((source, FAST_CFC_STATE_BYTES))
}

/// P1.7 (`docs/PLANO_EVOLUCAO.md`) — asynchronous brain. Per-neuron metadata, SoA
/// layout per being, mirroring `buf_neuron_meta` in `brain.wgsl` byte for byte:
/// `[l1_alpha(96), l1_period(96), l1_threshold(96), l1_activation(96), l2_alpha(48),
/// l2_activation(48)]` = 480 f32 per being. See `NEURON_META_PER_BEING` in the shader.
pub const NEURON_META_PER_BEING: usize = L1_NEURONS * 4 + L2_NEURONS * 2; // 480

/// P1.7 — persistent state of the EMA memory (mechanism 1), one slot per L1+L2 neuron.
/// `[l1_mem(96), l2_mem(48)]` = 144 f32 per being. Mirrors `NEURON_MEM_PER_BEING` in
/// the shader.
pub const NEURON_MEM_PER_BEING: usize = L1_NEURONS + L2_NEURONS; // 144

/// Adaptive fatigue per being: 192 floats.
/// Slots `[0..96]`   = CfC input fatigue (recurrent state, present from the start).
/// Slots `[96..192]` = output fatigue, which breaks the `final_intent` fixed point.
/// Grown from 96 → 192 to support `OUT_ALPHA`/`OUT_BETA` in `brain.wgsl`.
const FATIGUE_PER_BEING: usize = L1_NEURONS * 2; // 192

/// Width of `final_intent` (96 slots, `engine.rs:472`).
const INTENT_DIM: usize = L1_NEURONS; // 96

// ── #[repr(C)] STRUCT FOR THE UNIFORM ────────────────────────────────────────

/// Global parameters passed in as a uniform buffer.
///
/// `#[repr(C)]` guarantees a predictable layout. `bytemuck::Pod` requires every field
/// to be a primitive with no implicit padding — this struct sits right on that edge:
/// 12 fields of 4 bytes = 48 bytes, a multiple of the 16-byte WGSL uniform alignment.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BrainParams {
    pub n_beings: u32,
    pub dt: f32,
    /// When 1: read the sensory wave from binding 8 (`buf_percept`, the output of
    /// `perception.wgsl`, 48 u32 per being) instead of binding 1 (`buf_wave`, i32
    /// sign-extended by the CPU). This removes the perception→CPU→GPU round trip per
    /// tick. `gpu_validate` passes 0.
    pub use_percept_direct: u32,
    /// Strength of the OUTPUT fatigue (`OUT_BETA`), previously a shader constant.
    ///
    /// In steady state `out_fat → |raw|`, so the output becomes `(1 - out_beta)·|raw|`:
    /// at 0.85 a SUSTAINED command comes out at 15% of the intent, within about 1 s.
    /// That is adaptation to a constant stimulus — excellent in a sensor (it detects
    /// novelty), poison in a motor command, where holding the throttle down on the
    /// straight is precisely what you want. And because the operator is applied AFTER
    /// L2, no genome weight escapes it.
    ///
    /// It became a parameter so it could be MEASURED in a paired test instead of
    /// argued about. The default 0.85 is the historical value: pass nothing and the
    /// behaviour is what it always was.
    pub out_beta: f32,
    /// Gain applied to `l2_raw` (through L2) and to the L1 bypass before the
    /// fatigue/tanh in STAGE 2. Used to be the hardcoded constant `2.0` in the shader.
    /// See `OUT_GAIN_DEFAULT` — the P1.5 A/B measures 2.0 vs 1.0 vs 0.5 against L2
    /// saturation.
    pub out_gain: f32,
    /// P1.5, second lever (default 0 = OFF): 1 normalises `l2_raw` by the standard
    /// deviation of the 48 `l2_raw` values of the SAME being before the gain/tanh — it
    /// attacks the SCALE of the L2 input (which mutation can grow across generations)
    /// rather than only the output gain. It is a `u32` because `bool` is not portably
    /// `bytemuck::Pod` across platforms; 0/1 is the contract.
    pub norm_l2_by_std: u32,
    /// P1.7 — bitmask, one bit per mechanism of the asynchronous brain: the control arm
    /// of the A/B, driven by the four `gpu_evolve` opt-out flags parsed at
    /// `src/bin/gpu_evolve/main.rs:586-601` (their names are still Portuguese; renaming a
    /// CLI flag breaks existing scripts, so it is deferred — see `NAMING.md`).
    /// This field used to be `_pad0`, dead padding.
    /// bit0 = memory, bit1 = period, bit2 = threshold, bit3 = heterogeneous activation.
    /// A set bit means the genome reading for that mechanism is IGNORED (treated as
    /// neutral), whatever mutation wrote there. See `brain.wgsl::Params` for the full
    /// per-bit semantics.
    pub mech_off_mask: u32,
    _pad1: u32,
    /// D3 (`docs/ESCADA_DIFICULDADE.md`) — sensor noise: standard deviation of the
    /// approximate Gaussian noise added to EVERY channel of the sensory wave, EVERY
    /// tick, in i8 channel units (-127..127). For example `sigma=6` ≈ 5% of the range
    /// (127 being the one-sided amplitude). Applied BEFORE the KAN, while the wave is
    /// read (see `brain.wgsl::apply_sensor_noise`). `0.0` (the default) means no noise
    /// and behaviour identical to the historical path.
    pub noise_sigma: f32,
    /// D3 — channel failure (sensor dropout): probability in `[0,1]` that the channel
    /// is ZEROED on this tick ("the sensor did not answer"), drawn per channel per tick
    /// through a deterministic hash (the same pattern as the CfC denoise, see
    /// `brain.wgsl::apply_sensor_noise`). `0.0` (the default) means no channel fails
    /// and behaviour identical to the historical path.
    pub dropout_prob: f32,
    /// Padding: keeps the uniform buffer a multiple of 16 bytes (12 fields × 4 bytes =
    /// 48, the same convention as the original struct, which was 8 × 4 = 32).
    /// Deterministically zero through `bytemuck::Zeroable`, never read by the shader.
    _pad2: u32,
    _pad3: u32,
}

// ── PER-BEING INPUT ──────────────────────────────────────────────────────────

/// What the caller hands over per tick, per being.
pub struct BrainInput<'a> {
    /// Reference to this being's engine (used to read weights and CfC state).
    pub engine: &'a SovereignEngine,
    /// `sensory_wave[SENSORY_WAVE_SIZE]`, already computed by `motor.rs`.
    pub wave: &'a [i8; SENSORY_WAVE_SIZE],
}

// ── PERSISTENT CONTEXT ───────────────────────────────────────────────────────

/// GPU context of the brain forward pass. It lives for a whole generation.
///
/// Invariant: every buffer has capacity >= `capacity` beings. If the capacity has to
/// change (more beings spawn than were budgeted for), rebuild it with `BrainGpu::new`.
pub struct BrainGpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,

    // Persistent buffers (fixed size for the whole generation)
    buf_wave: wgpu::Buffer,   // i32 × (cap × WAVE) — rewritten every tick
    buf_kan: wgpu::Buffer,    // f32 × (cap × KAN_WEIGHTS_PER_BEING) — per generation
    buf_l1w: wgpu::Buffer,    // f32 × (cap × L1_WEIGHTS_PER_BEING)  — per generation
    buf_l2w: wgpu::Buffer,    // f32 × (cap × L2_WEIGHTS_PER_BEING)  — per generation
    buf_intent: wgpu::Buffer, // f32 × (cap × INTENT_DIM)            — read every tick
    buf_cfc: wgpu::Buffer,    // f32 × (cap × CFC_SLOTS_PER_BEING)  — resident

    /// Adaptive fatigue (spike-frequency adaptation): slow per-neuron state.
    /// It grows while `|state[i]|` is high and saturates when a neuron fires hard for a
    /// long time. Initialised to zeros; updated every tick by the `brain.wgsl` kernel.
    buf_fatigue: wgpu::Buffer, // f32 × (cap × FATIGUE_PER_BEING)    — resident

    /// The τ bank: the CfC decay rate per channel per being, in s⁻¹ (d = 1/τ).
    ///
    /// Constant for the whole generation, like the weights: τ is inherited from the
    /// genome and does not change during a being's life. Uploaded by `upload_tau`. It
    /// is born at `DECAY_DEFAULT` (τ = 2.0 s on every channel) so that a caller who
    /// never calls `upload_tau` gets the old behaviour.
    buf_tau: wgpu::Buffer, // f32 × (cap × L1_NEURONS)           — per generation

    /// P1.7 — per-neuron metadata (alpha / period / threshold / activation), read-only,
    /// uploaded once per generation. See `NEURON_META_PER_BEING`.
    buf_neuron_meta: wgpu::Buffer, // f32 × (cap × NEURON_META_PER_BEING) — per generation

    /// P1.7 — persistent state of the per-neuron EMA memory (mechanism 1).
    /// FOLLOWS THE PATTERN of `buf_cfc`: read_write, resident, cleared on spawn and
    /// between generations through `reset_cfc` (the same method, extended).
    buf_neuron_mem: wgpu::Buffer, // f32 × (cap × NEURON_MEM_PER_BEING)  — resident

    /// M4: per-region gate, 4 f32 per being; 1.0 is neutral.
    buf_gates: wgpu::Buffer, // f32 × (cap × 4)

    /// Temporary debug buffer: exports `l1_raw` plus the CfC/KAN inputs of being 0.
    /// Layout: `[l1_raw[0..96], cfc_in[0..96], kan_in[96..223]]` = 319 f32. The last
    /// block is still sized for the 223-channel wave; with `SENSORY_WAVE_SIZE = 192`
    /// the shader only fills its first 96 entries (the guard `j < 127u` never binds,
    /// because only 96 threads reach it), so slots 288..319 hold whatever was there
    /// before. Harmless, but do not read past `192 + 96`.
    buf_debug: wgpu::Buffer,

    /// CPU readback buffer (`MAP_READ`, same size as `buf_intent`).
    buf_readback: wgpu::Buffer,

    /// Placeholder for binding 8 when `use_percept_direct == 0` (the CPU path).
    /// `gpu_validate` and `forward()` use `buf_wave` (binding 1) but still need
    /// something valid at binding 8 or `create_bind_group` fails. Size: 4 bytes.
    buf_percept_dummy: wgpu::Buffer,

    /// Output fatigue strength used by the dispatches. See `BrainParams::out_beta`.
    pub out_beta: f32,

    /// Output gain used by dispatches through `bind_and_dispatch`. See
    /// `BrainParams::out_gain` / `OUT_GAIN_DEFAULT`. Only `bind_and_dispatch` reads this
    /// field — `forward()` (used by `gpu_validate`) always hardcodes `OUT_GAIN_DEFAULT`,
    /// the same contract `out_beta` already had.
    pub out_gain: f32,

    /// Normalise `l2_raw` by the standard deviation of the being's 48 values, before
    /// the gain/tanh. See `BrainParams::norm_l2_by_std`. Defaults to `false` —
    /// identical to the historical behaviour (no normalisation).
    pub norm_l2_by_std: bool,

    /// D3 (`docs/ESCADA_DIFICULDADE.md`) — standard deviation of the sensor noise, in
    /// i8 channel units. See `BrainParams::noise_sigma`. Defaults to `0.0`, identical to
    /// the historical behaviour (no noise). Only `bind_and_dispatch` reads this field
    /// (the same contract as `out_beta`/`out_gain`); `forward()` always hardcodes `0.0`,
    /// since it is the CPU/GPU equivalence path.
    pub noise_sigma: f32,

    /// D3 — per-channel per-tick failure (dropout) probability. See
    /// `BrainParams::dropout_prob`. Defaults to `0.0`, identical to the historical
    /// behaviour (no channel fails). Same contract as `noise_sigma` above.
    pub dropout_prob: f32,

    /// P1.7 — per-mechanism control arm, set from the `gpu_evolve` opt-out flags
    /// (`src/bin/gpu_evolve/main.rs:977-980`). All four default to
    /// `false`: the mechanisms READ the genome normally (and the genome is born neutral
    /// — see `gpu::evolution::Genome`). They are combined in `mech_off_mask()` for the
    /// uniform. See the doc on `BrainParams::mech_off_mask`.
    pub mem_off: bool,
    pub period_off: bool,
    pub threshold_off: bool,
    pub activation_off: bool,

    /// Maximum number of beings these buffers support.
    pub capacity: usize,

    pub adapter: String,

    /// P1.8, finding "the sensory KAN is frozen at the identity" — `eng.kan_filters` is
    /// never touched by `apply_to_engine` or by mutation in `gpu_evolve` (it is not part
    /// of `Genome`), so EVERY being of EVERY generation uploads exactly the same 3 072
    /// identity coefficients (`[1.0; 16]` per channel). Re-uploading them every
    /// generation is wasted bandwidth — once this is `true` (after the first
    /// `upload_weights`), later calls skip the KAN block. Behaviour is identical: the
    /// only way this could go wrong is if `kan_filters` started varying per being or per
    /// generation, which happens nowhere in the code today (confirmed by grep:
    /// `apply_to_engine` only copies l1/l2/slow).
    kan_uploaded: std::sync::atomic::AtomicBool,
}

impl BrainGpu {
    /// Creates a new GPU context, taking the device/queue shared with the retina.
    ///
    /// `capacity` is the maximum number of beings per dispatch and sizes every buffer.
    /// If the number of living beings ever exceeds it, the caller must rebuild.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        adapter_name: String,
        capacity: usize,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brain"),
            // `brain.wgsl` stays at `src/gpu/brain.wgsl`, one level up from this file
            // (`src/gpu/brain/mod.rs`) — moving it alongside would also require
            // updating `tests/wgsl_validation.rs`'s hardcoded path and every doc
            // comment's bare `brain.wgsl` reference for no benefit; `include_str!`
            // crossing one directory level is the smaller, well-trodden change.
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../brain.wgsl"))),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("brain-layout"),
            entries: &[
                uniform_entry(0),         // params (includes use_percept_direct)
                storage_entry(1, true),   // buf_wave: i32 sign-extended (CPU path)
                storage_entry(2, true),   // buf_kan  (read)
                storage_entry(3, true),   // buf_l1w  (read)
                storage_entry(4, true),   // buf_l2w  (read)
                storage_entry(5, false),  // buf_intent (read_write)
                storage_entry(6, false),  // buf_cfc    (read_write, recurrent state)
                storage_entry(7, false),  // buf_debug  (temporary diagnostics)
                storage_entry(8, true),   // buf_percept: u32 from perception.wgsl (GPU path)
                storage_entry(9, false),  // buf_fatigue (read_write, adaptive fatigue)
                storage_entry(10, true),  // buf_tau     (read, per-channel τ bank)
                storage_entry(11, true), // buf_neuron_meta (read, P1.7 — alpha/period/threshold/activation)
                storage_entry(12, false), // buf_neuron_mem  (read_write, P1.7 — resident EMA memory)
                storage_entry(13, true),  // buf_gates (read, M4 — per-region gate, 4 f32/being)
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

        // Sensory wave: one i32 per channel (sign-extended from i8, to keep bit
        // manipulation out of the shader). 4 bytes × WAVE × cap.
        let bytes_wave = (capacity * SENSORY_WAVE_SIZE * 4) as u64;
        let buf_wave = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-wave"),
            size: bytes_wave,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // KAN: one f32 per knot. Rewritten on spawn/evolution, not per tick.
        let bytes_kan = (capacity * KAN_WEIGHTS_PER_BEING * 4) as u64;
        let buf_kan = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-kan"),
            size: bytes_kan,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // L1 weights: f32, the largest buffer — 60 × 18 432 × 4 ≈ 4.4 MB.
        let bytes_l1w = (capacity * L1_WEIGHTS_PER_BEING * 4) as u64;
        let buf_l1w = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-l1w"),
            size: bytes_l1w,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // L2 weights: f32. 60 × 4 608 × 4 ≈ 1.1 MB.
        let bytes_l2w = (capacity * L2_WEIGHTS_PER_BEING * 4) as u64;
        let buf_l2w = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-l2w"),
            size: bytes_l2w,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Intent: f32 × 96 × cap. The kernel's output.
        let bytes_intent = (capacity * INTENT_DIM * 4) as u64;
        let buf_intent = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-intent"),
            size: bytes_intent,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // CfC state: f32 × 97 × cap. Recurrent state, initialised to zeros — which is
        // the correct initial state (`CfCCell::new` zeroes the state and sets counter=0).
        let bytes_cfc = (capacity * CFC_SLOTS_PER_BEING * 4) as u64;
        let cfc_zeros: Vec<u8> = vec![0u8; bytes_cfc as usize];
        let buf_cfc = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brain-cfc"),
            contents: &cfc_zeros,
            usage: wgpu::BufferUsages::STORAGE
                 | wgpu::BufferUsages::COPY_SRC  // read back on spawn/death
                 | wgpu::BufferUsages::COPY_DST, // reset on spawn
        });

        // Adaptive fatigue: f32 × 192 × cap. Slow state per CfC neuron, initialised to
        // zeros — a newborn being has no accumulated fatigue. Cleared together with the
        // CfC on spawn/death through `reset_cfc`.
        let bytes_fatigue = (capacity * FATIGUE_PER_BEING * 4) as u64;
        let fatigue_zeros: Vec<u8> = vec![0u8; bytes_fatigue as usize];
        let buf_fatigue = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brain-fatigue"),
            contents: &fatigue_zeros,
            usage: wgpu::BufferUsages::STORAGE
                 | wgpu::BufferUsages::COPY_DST  // reset on spawn
                 | wgpu::BufferUsages::COPY_SRC, // fatigue diagnostics
        });

        // τ bank: f32 × 96 × cap. Born at DECAY_DEFAULT (τ = 2.0 s), the single value the
        // brain used to use — so a path that never calls `upload_tau` stays identical to
        // before instead of running with τ = 0 (decay = 1, i.e. infinite memory).
        let taus_init: Vec<f32> = vec![crate::cfc::DECAY_DEFAULT; capacity * L1_NEURONS];
        let buf_tau = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brain-tau"),
            contents: bytemuck::cast_slice(&taus_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // P1.7 — per-neuron metadata (alpha/period/threshold/activation). Zero-init: a
        // zero is already neutral for alpha/threshold/activation (0.0), and for the
        // period the shader reads 0 as "1" through `max(...,1u)` — see the comment in
        // the shader. Anyone who does not call `upload_neuron_meta` (for instance
        // `gpu_validate`/`forward()`, which also force `mech_off_mask` to switch the
        // mechanisms off entirely) keeps the historical behaviour.
        let neuron_meta_zeros: Vec<u8> = vec![0u8; capacity * NEURON_META_PER_BEING * 4];
        let buf_neuron_meta = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brain-neuron-meta"),
            contents: &neuron_meta_zeros,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // P1.7 — persistent state of the EMA memory (mechanism 1). Zeros FOLLOW THE
        // PATTERN of `buf_cfc`/`buf_fatigue` (the correct initial state: empty memory).
        let neuron_mem_zeros: Vec<u8> = vec![0u8; capacity * NEURON_MEM_PER_BEING * 4];
        let buf_neuron_mem = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brain-neuron-mem"),
            contents: &neuron_mem_zeros,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC, // per-being reset via write_buffer + future diagnostics
        });

        // M4: the gates start NEUTRAL (1.0) — with the flag off they are never rewritten.
        let buf_gates = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brain-gates"),
            contents: bytemuck::cast_slice(&vec![1.0f32; capacity * 4]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Readback: a separate MAP_READ buffer (wgpu forbids combining it with STORAGE).
        let buf_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-readback"),
            size: bytes_intent,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Debug: exports l1_raw[96] + cfc_in[96] + kan_in[127] = 319 f32 for being 0.
        // Fixed size (it does not scale with cap): being 0 only.
        let buf_debug = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brain-debug"),
            size: (319 * 4) as u64, // 96 + 96 + 127 = 319 f32
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Placeholder for binding 8 (buf_percept) when use_percept_direct == 0. wgpu
        // requires every slot declared in the layout to have a buffer in the bind group,
        // even one the shader never reads. 4 bytes is the smallest allowed.
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
            buf_neuron_meta,
            buf_neuron_mem,
            buf_gates,
            out_beta: OUT_BETA_DEFAULT,
            out_gain: OUT_GAIN_DEFAULT,
            norm_l2_by_std: false,
            noise_sigma: 0.0,
            dropout_prob: 0.0,
            mem_off: false,
            period_off: false,
            threshold_off: false,
            activation_off: false,
            buf_debug,
            buf_readback,
            buf_percept_dummy,
            capacity,
            adapter: adapter_name,
            kan_uploaded: std::sync::atomic::AtomicBool::new(false),
        }
    }

    // ── WEIGHT UPLOAD (per generation, not per tick) ───────────────────────

    /// Uploads every being's KAN, L1 and L2 weights to the GPU.
    ///
    /// Call this on spawn and after evolution (weight mutation). Do NOT call it per
    /// tick — weights do not change during a generation.
    ///
    /// SoA layout: all of being 0, then all of being 1, and so on.
    pub fn upload_weights(&self, engines: &[&SovereignEngine]) {
        let n = engines.len().min(self.capacity);

        // ── KAN ─────────────────────────────────────────────────────────────
        // SensoryPreFilter: Vec<KanAxiomFilter>, each holding spline_coeffs [f32; 16].
        // Layout: being × WAVE × KAN_KNOTS (SoA per being).
        //
        // P1.8, finding "the sensory KAN is frozen at the identity — adapt() is never
        // called and it is not part of gpu_evolve's Genome": `eng.kan_filters` is ALWAYS
        // `[1.0; 16]` per channel, for EVERY being, in EVERY generation (nothing mutates
        // it). Re-uploading the same 3072×n floats each generation is pure bandwidth, so
        // upload once (on the first `upload_weights` call) and skip afterwards. See the
        // doc on `kan_uploaded`.
        if !self
            .kan_uploaded
            .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            let mut kan_data: Vec<f32> = Vec::with_capacity(n * KAN_WEIGHTS_PER_BEING);
            for eng in engines[..n].iter() {
                for filter in eng.kan_filters.filters.iter().take(SENSORY_WAVE_SIZE) {
                    kan_data.extend_from_slice(&filter.spline_coeffs);
                }
                // Pad if there are fewer filters than SENSORY_WAVE_SIZE (should not
                // happen, but this is defensive).
                let missing = SENSORY_WAVE_SIZE.saturating_sub(eng.kan_filters.filters.len());
                for _ in 0..missing {
                    // Identity coefficients = 1.0 (no deformation).
                    for _ in 0..KAN_KNOTS {
                        kan_data.push(1.0);
                    }
                }
            }
            self.queue
                .write_buffer(&self.buf_kan, 0, bytemuck::cast_slice(&kan_data));
        }

        // ── L1 (instinct_layer: BitLinear WAVE→96) ───────────────────────
        // weights: Vec<f32>, row-major — weights[j * WAVE + i] is the weight of output
        // neuron j for input i.
        let mut l1_data: Vec<f32> = Vec::with_capacity(n * L1_WEIGHTS_PER_BEING);
        for eng in engines[..n].iter() {
            // The weights live in a Vec<f32> of len = WAVE * L1_OUT (layers.rs:40).
            let w = &eng.instinct_layer.weights;
            debug_assert_eq!(
                w.len(),
                L1_WEIGHTS_PER_BEING,
                "L1 weights size mismatch: {} vs {}",
                w.len(),
                L1_WEIGHTS_PER_BEING
            );
            l1_data.extend_from_slice(w);
        }
        self.queue
            .write_buffer(&self.buf_l1w, 0, bytemuck::cast_slice(&l1_data));

        // ── L2 (cortex_l2: FloatLinear 96→48) ────────────────────────────
        // weights: Vec<f32>, row-major — weights[j * L1_OUT + i].
        let mut l2_data: Vec<f32> = Vec::with_capacity(n * L2_WEIGHTS_PER_BEING);
        for eng in engines[..n].iter() {
            let w = &eng.cortex_l2.weights;
            debug_assert_eq!(
                w.len(),
                L2_WEIGHTS_PER_BEING,
                "L2 weights size mismatch: {} vs {}",
                w.len(),
                L2_WEIGHTS_PER_BEING
            );
            l2_data.extend_from_slice(w);
        }
        self.queue
            .write_buffer(&self.buf_l2w, 0, bytemuck::cast_slice(&l2_data));
    }

    /// Uploads the τ bank: `n × L1_NEURONS` decay rates in s⁻¹, in being order.
    ///
    /// Called once per generation, together with the weights — τ is inherited, it does
    /// not change per tick. A caller who skips it keeps the `DECAY_DEFAULT` from
    /// initialisation, which is the old uniform τ = 2.0 s; that way the experiment's
    /// control arm *is* the historical behaviour.
    pub fn upload_tau(&self, taus: &[f32], n: usize) {
        assert!(
            n <= self.capacity,
            "n={n} exceeds capacity {}",
            self.capacity
        );
        // A wrong length here would be a SILENT bug: the kernel would read the next
        // being's τ (or garbage past the end of the buffer) and the "heterogeneous"
        // condition would turn into shuffled noise with no visible symptom. That exact
        // class of error — a wrong stride on the sensory wave — has already scrambled
        // the slow brain's input in this project.
        assert_eq!(
            taus.len(),
            n * L1_NEURONS,
            "tau bank has {} values, expected {} ({n} beings × {L1_NEURONS} channels)",
            taus.len(),
            n * L1_NEURONS
        );
        self.queue
            .write_buffer(&self.buf_tau, 0, bytemuck::cast_slice(taus));
    }

    /// P1.7 — uploads the per-neuron metadata (alpha/period/threshold/activation) of
    /// `n` beings. Same contract as `upload_tau`: once per generation, SoA layout
    /// (`NEURON_META_PER_BEING` floats per being, see the field's comment).
    pub fn upload_neuron_meta(&self, meta: &[f32], n: usize) {
        assert!(
            n <= self.capacity,
            "n={n} exceeds capacity {}",
            self.capacity
        );
        assert_eq!(
            meta.len(),
            n * NEURON_META_PER_BEING,
            "neuron metadata (P1.7) has {} values, expected {} ({n} beings × {NEURON_META_PER_BEING})",
            meta.len(),
            n * NEURON_META_PER_BEING
        );
        self.queue
            .write_buffer(&self.buf_neuron_meta, 0, bytemuck::cast_slice(meta));
    }

    /// P1.7 — folds the four control flags into a single bitmask for the uniform.
    /// See the doc on `BrainParams::mech_off_mask` for what each bit means.
    fn mech_off_mask(&self) -> u32 {
        (self.mem_off as u32)
            | ((self.period_off as u32) << 1)
            | ((self.threshold_off as u32) << 2)
            | ((self.activation_off as u32) << 3)
    }

    /// D3 (`docs/ESCADA_DIFICULDADE.md`) — sets the sensor noise (σ) and the channel
    /// failure probability (dropout) for the NEXT dispatches.
    ///
    /// There is no separate "re-upload" buffer: `bind_and_dispatch` builds a fresh
    /// uniform buffer on every call out of `self.noise_sigma` / `self.dropout_prob` (the
    /// same contract as `out_beta`/`out_gain`/`norm_l2_by_std`, which are also public
    /// fields with no dedicated setter — see `gpu_evolve/main.rs: ctx.brain.out_beta =
    /// ...`). A method exists here because D3's two parameters always move together (the
    /// ladder tightens both as one rung).
    ///
    /// `sigma`: per-channel noise standard deviation, in i8 channel units (-127..127) —
    /// e.g. `6.0` ≈ 5% of the one-sided range (127). `0.0` switches it off.
    /// `dropout`: probability in `[0,1]` that a channel is zeroed on a tick. `0.0`
    /// switches it off.
    pub fn set_noise(&mut self, sigma: f32, dropout: f32) {
        self.noise_sigma = sigma;
        self.dropout_prob = dropout;
    }

    // ── CFC STATE DOWNLOAD (diagnostics) ───────────────────────────────────

    /// Reads `buf_cfc` back from the GPU and returns `state[0..96]` for each being.
    /// `gpu_validate` uses it to compare the post-forward CfC against the CPU.
    /// Cost: one GPU→CPU copy (readback). Do not use it in the per-tick hot path.
    pub fn download_cfc_state(&self, n: usize) -> Vec<Vec<f32>> {
        let n = n.min(self.capacity);
        let bytes = (n * CFC_SLOTS_PER_BEING * 4) as u64;

        // Temporary MAP_READ buffer for the CfC readback.
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
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        let vals: &[f32] = bytemuck::cast_slice(&data);

        // Each being holds CFC_SLOTS_PER_BEING=97 f32: state[0..96] + counter (bitcast).
        (0..n)
            .map(|s| {
                let base = s * CFC_SLOTS_PER_BEING;
                vals[base..base + 96].to_vec() // the state only, without the counter
            })
            .collect()
    }

    /// M4: uploads the per-region gates (4 f32 per being). Called once every 8 ticks by
    /// `tick()` when `--neuromod-regions` is on; never calling it leaves them neutral
    /// at 1.0.
    pub fn upload_gates(&self, gates: &[f32], n: usize) {
        debug_assert_eq!(gates.len(), n * 4);
        self.queue
            .write_buffer(&self.buf_gates, 0, bytemuck::cast_slice(gates));
    }

    // ── DEBUG BUFFER DOWNLOAD (diagnostics) ────────────────────────────────

    /// Reads `buf_debug` back from the GPU: `l1_raw[0..96]`, `cfc_in[0..96]` and
    /// `kan_in[96..223]` for being 0. Returns `(l1_raw, cfc_inputs, kan_inputs)`.
    /// `gpu_validate` uses it to compare L1 raw directly against the CPU.
    ///
    /// Only the first 96 entries of the third block are written on the current wave of
    /// 192 channels — see the `buf_debug` field doc.
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
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        let vals: &[f32] = bytemuck::cast_slice(&data);

        let l1_raw = vals[0..96].to_vec();
        let cfc_in = vals[96..192].to_vec();
        let kan_in = vals[192..319].to_vec();
        (l1_raw, cfc_in, kan_in)
    }

    // ── FATIGUE BUFFER DOWNLOAD (diagnostics) ──────────────────────────────

    /// Reads `buf_fatigue` back from the GPU for diagnostics.
    /// Returns `n × FATIGUE_PER_BEING` floats:
    ///   slots `[0..96]`   per being = CfC input fatigue
    ///   slots `[96..192]` per being = output fatigue (`out_fat` for each channel)
    pub fn download_fatigue(&self, n: usize) -> Vec<f32> {
        let n = n.min(self.capacity);
        let bytes = (n * FATIGUE_PER_BEING * 4) as u64;

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
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        bytemuck::cast_slice::<u8, f32>(&data).to_vec()
    }

    /// Copies the 96-float CfC L1 state for one being into a caller-owned GPU buffer.
    ///
    /// `buf_cfc` is laid out as `[state[0..96], counter]` per being, so the source
    /// stride is 97 f32 and this method deliberately copies only the first 96 f32.
    /// The counter is not an observation and is never exposed by this method. The
    /// destination must have `COPY_DST` usage and enough space at `destination_offset`.
    /// This records fast CfC state only; it is not a snapshot of P1.7 neuron memory,
    /// slow-brain state, or retina-perception state.
    pub fn copy_fast_state_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        destination: &wgpu::Buffer,
        being: usize,
        destination_offset: u64,
    ) -> Result<(), String> {
        let (source_offset, bytes) =
            fast_state_copy_range(self.capacity, being, destination.size(), destination_offset)?;
        encoder.copy_buffer_to_buffer(
            &self.buf_cfc,
            source_offset,
            destination,
            destination_offset,
            bytes,
        );
        Ok(())
    }

    /// Clears only the fast CfC L1 state for beings `[0, n)`, preserving each CfC
    /// counter and every other temporal mechanism (fatigue, P1.7 neuron EMA memory,
    /// slow brain and perception state). This is an intentionally partial memory
    /// ablation, suitable for `--erase-fast-memory`; it does not mean "erase all
    /// temporal learning".
    pub fn clear_fast_memory(&self, encoder: &mut wgpu::CommandEncoder, n: usize) {
        let bytes_per_being = (CFC_SLOTS_PER_BEING * std::mem::size_of::<f32>()) as u64;
        let n = n.min(self.capacity);
        for being in 0..n {
            let offset = being as u64 * bytes_per_being;
            encoder.clear_buffer(&self.buf_cfc, offset, Some(FAST_CFC_STATE_BYTES));
        }
    }

    // ── CFC STATE RESET FOR ONE BEING ──────────────────────────────────────

    /// Resets one being's CfC state, adaptive fatigue and per-neuron EMA memory (P1.7)
    /// to zeros.
    ///
    /// Equivalent to `CfCCell::reset()`. It writes zeros over the being's 97 CfC f32,
    /// its 192 fatigue f32 (96 input + 96 output) and its 144 per-neuron memory f32
    /// (mechanism 1, P1.7 — FOLLOWING THE PATTERN of this same method), leaving every
    /// other being untouched. Called on spawn/death: it guarantees a newborn starts with
    /// no fatigue AND no memory inherited from the slot's previous occupant. Without it
    /// the child would be born "remembering" what the parent last felt — the same
    /// state-leaking-across-generations bug that motivated resetting `buf_cfc` and
    /// `buf_fatigue` here.
    pub fn reset_cfc(&self, being_idx: usize) {
        if being_idx >= self.capacity {
            return;
        }
        // Reset the CfC state + counter (97 f32).
        let cfc_offset = (being_idx * CFC_SLOTS_PER_BEING * 4) as u64;
        let cfc_zeros: Vec<u8> = vec![0u8; CFC_SLOTS_PER_BEING * 4];
        self.queue
            .write_buffer(&self.buf_cfc, cfc_offset, &cfc_zeros);
        // Reset the adaptive fatigue: 192 f32 per being (slots [0..96] input,
        // [96..192] output).
        let fat_offset = (being_idx * FATIGUE_PER_BEING * 4) as u64;
        let fat_zeros: Vec<u8> = vec![0u8; FATIGUE_PER_BEING * 4];
        self.queue
            .write_buffer(&self.buf_fatigue, fat_offset, &fat_zeros);
        // P1.7 — reset the per-neuron EMA memory: 144 f32 per being (96 L1 + 48 L2).
        let mem_offset = (being_idx * NEURON_MEM_PER_BEING * 4) as u64;
        let mem_zeros: Vec<u8> = vec![0u8; NEURON_MEM_PER_BEING * 4];
        self.queue
            .write_buffer(&self.buf_neuron_mem, mem_offset, &mem_zeros);
    }

    // ── FORWARD PASS (per tick) ────────────────────────────────────────────

    /// Runs the brain forward pass for every being on the GPU.
    ///
    /// Inputs:
    ///   `inputs`: a slice of `(engine, wave)` per being, in spawn order.
    ///   `dt`: this tick's time delta, in seconds.
    ///
    /// Output:
    ///   one `[f32; INTENT_DIM]` per being, in the same order. Equivalent to
    ///   `engine.final_intent` after `execute_cycle`, excluding Trinity, `axiom_policy`
    ///   and the side effects of metabolism.
    ///
    /// # Equivalence note
    ///
    /// The caller (`motor.rs`) must:
    /// 1. Call this method to obtain the GPU `final_intent`.
    /// 2. Apply `axiom_policy` and Trinity (Gate + Reflex + Sentinel) on the CPU exactly
    ///    as before — those paths did not change.
    /// 3. NOT call `engine.execute_cycle` for the beings that went to the GPU, or the
    ///    CfC state would be updated twice.
    pub fn forward(&self, inputs: &[BrainInput<'_>], dt: f32) -> Vec<[f32; INTENT_DIM]> {
        let n = inputs.len().min(self.capacity);
        if n == 0 {
            return Vec::new();
        }

        // ── Sensory-wave upload (changes every tick) ─────────────────────
        // Converts i8 → i32 (sign-extend) so the shader reads it without bit tricks.
        let mut wave_data: Vec<i32> = Vec::with_capacity(n * SENSORY_WAVE_SIZE);
        for input in inputs[..n].iter() {
            for &b in input.wave.iter() {
                wave_data.push(b as i32);
            }
        }
        self.queue
            .write_buffer(&self.buf_wave, 0, bytemuck::cast_slice(&wave_data));

        // ── Uniform params ───────────────────────────────────────────────
        // use_percept_direct = 0: read from buf_wave (binding 1), the CPU path.
        // gpu_validate goes through forward() — it must never pass 1 here.
        let params = BrainParams {
            n_beings: n as u32,
            dt,
            use_percept_direct: 0,
            out_beta: crate::gpu::brain::OUT_BETA_DEFAULT,
            out_gain: crate::gpu::brain::OUT_GAIN_DEFAULT,
            norm_l2_by_std: 0,
            // forward() is the CPU/GPU EQUIVALENCE path (used by gpu_validate) — same
            // contract out_beta/out_gain/norm_l2_by_std already had here: ALWAYS the
            // historical behaviour, with P1.7's mechanisms switched off (0b1111), never
            // exposing the new parameters. `bind_and_dispatch` (production) is the one
            // that reads `self.mech_off_mask`.
            mech_off_mask: 0b1111,
            _pad1: 0,
            // D3: forward() never exposes sensor noise — same contract as above, always
            // 0.0 (the historical behaviour, no noise and no dropout).
            noise_sigma: 0.0,
            dropout_prob: 0.0,
            _pad2: 0,
            _pad3: 0,
        };
        let buf_params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("brain-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // ── Bind group ───────────────────────────────────────────────────
        // Binding 8 = buf_percept_dummy (placeholder, unread with use_percept_direct=0).
        // Binding 9 = buf_fatigue (adaptive fatigue, resident between ticks).
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
                    resource: self.buf_wave.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_kan.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buf_l1w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buf_l2w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.buf_intent.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.buf_cfc.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.buf_debug.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.buf_percept_dummy.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.buf_fatigue.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: self.buf_tau.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.buf_neuron_meta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: self.buf_neuron_mem.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: self.buf_gates.as_entire_binding(),
                },
            ],
        });

        // ── Dispatch ─────────────────────────────────────────────────────
        // One workgroup per being. Workgroup size = 128 threads (set in the WGSL).
        let bytes_intent = (n * INTENT_DIM * 4) as u64;
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            // Exact dispatch: 1 workgroup = 1 being. No rounding needed.
            pass.dispatch_workgroups(n as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.buf_intent, 0, &self.buf_readback, 0, bytes_intent);
        self.queue.submit(Some(enc.finish()));

        // ── Readback ─────────────────────────────────────────────────────
        let slice = self.buf_readback.slice(0..bytes_intent);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // poll(Wait) blocks until the GPU finishes. Inside a Bevy system that is
        // acceptable, because the system already blocks the scheduler thread anyway. An
        // async alternative (which would not block the thread) would mean restructuring
        // the system into two stages (submit / collect) — a future improvement if needed.
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

    // ── CHAINING METHODS (no submit of their own) ──────────────────────────────

    /// Prepares the sensory wave and records the brain dispatch into an external
    /// `CommandEncoder`.
    ///
    /// # Why it exists
    ///
    /// `forward()` created its own encoder, submit and poll, which added roughly
    /// 0.3–0.5 ms of round trip per tick (measured: command-buffer overhead plus queue
    /// synchronisation). With `bind_and_dispatch`, retina → perception → brain share a
    /// single encoder and a single submit, removing 2 intermediate round trips per tick.
    ///
    /// # Contract
    ///
    /// 1. Call `queue.write_buffer(&self.buf_wave, ...)` BEFORE this method (or pass
    ///    `wave_data_i32` here — it is done internally).
    /// 2. Record into the external encoder with this method.
    /// 3. Copy `buf_intent → buf_readback` in the same encoder via
    ///    `enc.copy_buffer_to_buffer`.
    /// 4. `queue.submit(enc.finish())` + `device.poll(Wait)`.
    /// 5. Call `read_intent_after_submit(n)` to read the result.
    ///
    /// Returns `bytes_intent` for the subsequent copy.
    ///
    /// CPU mode (`wave_data_i32` supplied, `buf_percept = None`): uploads the wave via
    ///   `write_buffer`, `use_percept_direct = 0`. Used by `gpu_validate`.
    ///
    /// GPU mode (`wave_data_i32` ignored, `buf_percept = Some(&buf)`): the shader reads
    ///   the wave straight out of `buf_percept` (the output of `perception.wgsl`,
    ///   48 u32 per being), `use_percept_direct = 1`. Used by `gpu_loop`, with no
    ///   intermediate readback at all.
    pub fn bind_and_dispatch(
        &self,
        enc: &mut wgpu::CommandEncoder,
        n: usize,
        wave_data_i32: &[i32], // used ONLY when buf_percept = None
        dt: f32,
        buf_percept: Option<&wgpu::Buffer>, // Some = GPU path (use_percept_direct=1)
    ) -> u64 {
        let n = n.min(self.capacity);
        if n == 0 {
            return 0;
        }

        // Decide how the wave is read.
        let use_percept_direct: u32;
        let percept_binding: &wgpu::Buffer;

        if let Some(pb) = buf_percept {
            // GPU path: buf_percept holds the output of perception.wgsl. We do NOT
            // upload wave_data_i32 — that would be a useless write and would cost PCIe.
            use_percept_direct = 1;
            percept_binding = pb;
        } else {
            // CPU path: upload wave_data_i32 via write_buffer (non-blocking until
            // submit). wave_data_i32 must have len = n × SENSORY_WAVE_SIZE.
            self.queue
                .write_buffer(&self.buf_wave, 0, bytemuck::cast_slice(wave_data_i32));
            use_percept_direct = 0;
            percept_binding = &self.buf_percept_dummy;
        };

        // Uniform: 48 bytes, negligible cost.
        let params = BrainParams {
            n_beings: n as u32,
            dt,
            use_percept_direct,
            out_beta: self.out_beta,
            out_gain: self.out_gain,
            norm_l2_by_std: if self.norm_l2_by_std { 1 } else { 0 },
            mech_off_mask: self.mech_off_mask(),
            _pad1: 0,
            // D3 (ESCADA_DIFICULDADE.md): sensor noise + channel dropout. Read from
            // `self.*` on every call — `set_noise` only updates those fields; the upload
            // happens here, on the next dispatch.
            noise_sigma: self.noise_sigma,
            dropout_prob: self.dropout_prob,
            _pad2: 0,
            _pad3: 0,
        };
        let buf_params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("brain-params-chain"),
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
                    resource: self.buf_wave.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_kan.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buf_l1w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buf_l2w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.buf_intent.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.buf_cfc.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.buf_debug.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: percept_binding.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: self.buf_fatigue.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: self.buf_tau.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.buf_neuron_meta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: self.buf_neuron_mem.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: self.buf_gates.as_entire_binding(),
                },
            ],
        });

        // One workgroup per being — no rounding.
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass-brain"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(n as u32, 1, 1);
        }

        let bytes_intent = (n * INTENT_DIM * 4) as u64;

        // Copy inside the same encoder — DMA with no round trip.
        enc.copy_buffer_to_buffer(&self.buf_intent, 0, &self.buf_readback, 0, bytes_intent);

        bytes_intent
    }

    /// Reads the brain's `buf_readback` after the external encoder's `device.poll(Wait)`.
    ///
    /// Call this ONLY once the `CommandEncoder` holding `bind_and_dispatch` has been
    /// submitted and `device.poll(Wait)` has returned. Otherwise the buffer is not
    /// mapped.
    pub fn read_intent_after_submit(&self, n: usize) -> Vec<[f32; INTENT_DIM]> {
        let n = n.min(self.capacity);
        if n == 0 {
            return Vec::new();
        }

        let bytes_intent = (n * INTENT_DIM * 4) as u64;
        let slice = self.buf_readback.slice(0..bytes_intent);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // FIXED HANG: `map_async` registers a callback that is only SERVED by a LATER
        // `poll`. The previous comment claimed "the caller already polled, so recv()
        // returns immediately" — and it was wrong: the caller's poll happens BEFORE the
        // map_async, so the callback was never served and `recv()` blocked forever. That
        // is what pinned `gpu_loop` on the warm-up tick, with no error and no output. The
        // poll has to come AFTER the map request.
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

    /// Reference to `buf_intent` (in VRAM), so other kernels can consume the output
    /// directly.
    pub fn buf_intent(&self) -> &wgpu::Buffer {
        &self.buf_intent
    }
}

// ── BEVY RESOURCE ────────────────────────────────────────────────────────────

/// Bevy resource that keeps the brain GPU context alive between ticks.
///
/// It follows the `RetinaGpuRes` pattern: `ctx: Option<BrainGpu>`, rebuilt when the
/// capacity is insufficient or the device is lost.
///
/// TODO: the original comment here stated that the caller should honour an environment
/// variable to force the CPU path, mirroring the retina. The retina's switch is real —
/// `IARA_RETINA_CPU` is read at `src/simulation/motor.rs:424` — but the brain's
/// equivalent exists nowhere in the tree (verified by grep), so there is currently no way
/// to run a controlled brain CPU/GPU A/B from the environment. Either implement the
/// switch or drop the claim; do not assume it works.
#[derive(bevy::prelude::Resource, Default)]
pub struct BrainGpuRes {
    pub ctx: Option<BrainGpu>,
}

impl BrainGpuRes {
    /// Returns a valid context, creating one if necessary.
    ///
    /// Takes device/queue from `RetinaGpu` so the same device is reused. `None` means
    /// there is no usable GPU and the caller should fall back to the CPU.
    pub fn get_or_create(
        &mut self,
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        name: String,
        capacity: usize,
    ) -> Option<&BrainGpu> {
        let needs_alloc = self.ctx.is_none()
            || self
                .ctx
                .as_ref()
                .map(|c| c.capacity < capacity)
                .unwrap_or(true);
        if needs_alloc {
            self.ctx = Some(BrainGpu::new(device, queue, name, capacity.max(64)));
            if let Some(c) = &self.ctx {
                println!("🧠 brain on the GPU ({} beings)", c.capacity);
            }
        }
        self.ctx.as_ref()
    }
}

// ── HELPERS (same as in mod.rs, duplicated so mod.rs does not have to be edited) ─

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

// ── HOW CPU/GPU EQUIVALENCE IS TESTED ────────────────────────────────────────
//
// The validation harness is `src/bin/gpu_validate.rs`. It does the following:
//
// 1. Builds a `SovereignEngine` with random but deterministic weights (fixed seed,
//    identical on the CPU and on the GPU).
//
// 2. Uses a fixed input vector (a known sensory wave).
//
// 3. CPU: calls `engine.execute_cycle(wave, 0.0, DT)` with DT = 1/64, then reads
//    `engine.final_intent[0..96]`.
//
// 4. GPU: calls `brain.forward(&[BrainInput{engine, wave}], DT)` and takes
//    `intent_gpu[0..96]`.
//
// 5. Compares element by element: `|cpu[i] - gpu[i]| <= 2e-5` (a tolerance of ≤ 2 ULP
//    for f32 in [-1, 1], the contract in `GPU_PLANO.md`). For the CfC state accumulated
//    over N ticks: measure the drift and state it — do not assume zero drift.
//
// 6. CAUTION: the test must disable `metabolism.can_process()` and everything else that
//    modifies state outside the pure neural path (cortisol, dopamine, and so on), so the
//    comparison is fair — those effects stay on the CPU and never enter the kernel.
//
// 7. CAUTION 2: `execute_cycle` increments `denoise_counter` through `cfc_state.update`.
//    On the GPU the counter starts at 0 and is stored in `buf_cfc`. For the test to be
//    comparable, initialise both with counter = 0 and run exactly N ticks.
//
// 8. DOCUMENTED EXPECTED DIVERGENCES:
//    - tanh: ≤ 1 ULP (IEEE 754)
//    - exp:  ≤ 1 ULP
//    - L1 matmul (WAVE = 192 operations): sequential f32 accumulation, in the same
//      order on both sides. Expected difference: < 1e-6 per neuron (192 f32 FMAs,
//      drift < 1e-5).
//    - i8-quantised output after tanh: difference ≤ 1 LSB = 1/127 ≈ 0.008.
