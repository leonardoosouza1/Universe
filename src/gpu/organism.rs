//! Resident `organism-taxi-v1` body state.
//!
//! This module owns the body-side writer for the new scenario. It is deliberately
//! independent of the legacy fuel/thermal writer: the caller must set
//! [`CONFIG_DISABLE_OLD_THERMAL`] when the new body owns those channels. The
//! equations are bounded model equations, not a calibrated biological claim.
//! Fatigue uses explicit model time scales: maximum normalized load accumulates
//! one fatigue unit in 120 seconds; unloaded rest removes one in 60 seconds.
//! Temperature follows the two-compartment energy balance without temperature
//! clipping. Observation/checkpoint validation rejects nonfinite/nonpositive K;
//! it must never silently replace a physical state with a plausible temperature.
//!
//! ABI (one being): `state` is 128 f32 (512 bytes), genes are 16 f32, physics
//! input is 92 f32 (the Rust 360-byte record is serialized in a 368-byte slot),
//! and physics output is 56 f32. The shader uses flat f32 indexing so padding
//! differences cannot silently change a field's position.
//!
//! State indices (the remaining indices are reserved telemetry scratch):
//! 0 energy_J, 1 water_kg, 2 material_kg, 3 age_s, 4 maturity, 5 fatigue,
//! 6 engine_wear, 7 tyre_wear, 8 integrity, 9 body_temperature_K, 10 damage,
//! 11 paid_repair_J, 12 paid_growth_J, 13 senescence, 14 last_power_W,
//! 15 last_heat_W; 16..27 immutable birth capacities; 28 schema version,
//! 29 alive, 30 config flags, 31 reserved; 32..41 effective capacities and
//! cost telemetry, 42 engine repair amount, 43 tyre repair amount, 44 repair
//! energy cost, 45 repair material cost, 46 resource limiter, 47..49 immutable
//! captured physical birth torque/brake/mass, 56 oil_temperature_K. Slots 50..63 are reserved for body observation. Slots 64..127 belong
//! to the taxi/world owner and are not interpreted by this module.
//!
//! Gene mapping (all input values are strictly `[-1, 1]`): engine, brake,
//! energy, water, material, cooling, repair rate, growth rate, mass trade-off,
//! thermal tolerance, leak resistance, senescence resistance, metabolic
//! efficiency, durability, repair efficiency, reserved.
//!
//! Model ledger (bounded engineering assumptions, not biological calibration): body
//! heat capacity is 12,000 J/K plus the thermal-margin adjustment, oil capacity is
//! 6,000 J/K, body↔oil conductance is 240 W/K, and oil↔ambient cooling uses the
//! existing cooling-capacity expression. Fatigue leaves 20% engine capacity at its
//! maximum; leak resistance is an inverse multiplier with baseline 1.0.

use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

pub const STATE_STRIDE_F32: usize = 128;
pub const GENE_STRIDE_F32: usize = 16;
pub const PHYSICS_INPUT_STRIDE_F32: usize = 92;
pub const PHYSICS_OUTPUT_STRIDE_F32: usize = 56;
pub const SCHEMA_VERSION: u32 = 1;
pub const CONFIG_DISABLE_OLD_THERMAL: u32 = 1 << 8;
pub const CONFIG_STATIC_REST: u32 = 1 << 9;

const WORKGROUP: u32 = 64;

// State slots. Keep these names stable: telemetry and later perception code
// consume the same resident buffer rather than reconstructing body state.
pub const S_ENERGY_J: usize = 0;
pub const S_WATER_KG: usize = 1;
pub const S_MATERIAL_KG: usize = 2;
pub const S_AGE_S: usize = 3;
pub const S_MATURITY: usize = 4;
pub const S_FATIGUE: usize = 5;
pub const S_ENGINE_WEAR: usize = 6;
pub const S_TYRE_WEAR: usize = 7;
pub const S_INTEGRITY: usize = 8;
pub const S_TEMPERATURE_K: usize = 9;
pub const S_DAMAGE: usize = 10;
pub const S_PAID_REPAIR_J: usize = 11;
pub const S_PAID_GROWTH_J: usize = 12;
pub const S_SENESCENCE: usize = 13;
pub const S_LAST_POWER_W: usize = 14;
pub const S_LAST_HEAT_W: usize = 15;
pub const S_BIRTH_ENGINE_CAP: usize = 16;
pub const S_BIRTH_BRAKE_CAP: usize = 17;
pub const S_BIRTH_ENERGY_CAP: usize = 18;
pub const S_BIRTH_WATER_CAP: usize = 19;
pub const S_BIRTH_MATERIAL_CAP: usize = 20;
pub const S_BIRTH_COOLING: usize = 21;
pub const S_BIRTH_REPAIR_RATE: usize = 22;
pub const S_BIRTH_GROWTH_RATE: usize = 23;
pub const S_BIRTH_MASS: usize = 24;
pub const S_BIRTH_THERMAL_MARGIN: usize = 25;
pub const S_BIRTH_LEAK_RESISTANCE: usize = 26;
pub const S_BIRTH_SENESCENCE_RESISTANCE: usize = 27;
/// Independent oil/engine thermal compartment, initialized at ambient at birth.
pub const S_OIL_TEMPERATURE_K: usize = 56;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct OrganismState {
    pub values: [f32; STATE_STRIDE_F32],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Params {
    n: u32,
    _pad: u32,
    dt: f32,
    config_flags: u32,
}

pub struct OrganismGpu {
    pub buf_state: wgpu::Buffer,
    pub buf_genes: wgpu::Buffer,
    pre_pipeline: wgpu::ComputePipeline,
    post_pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    buf_snapshot: wgpu::Buffer,
    pub capacity: usize,
    config_flags: u32,
}

impl OrganismGpu {
    /// Create resident buffers and compile both body phases on a shared device.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, capacity: usize) -> Self {
        let capacity = capacity.max(1);
        let _ = queue; // The first birth upload is explicit through `reset`.
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("organism-taxi-v1"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("organism.wgsl"))),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("organism-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(1, false),
                storage_entry(2, true),
                // Physics input is read/write: pre applies physical birth
                // capacities and the scenario flag before physics consumes it.
                storage_entry(3, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("organism-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pre_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("organism-pre"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("encode_pre"),
            compilation_options: Default::default(),
            cache: None,
        });
        let post_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("organism-post"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("encode_post"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bytes_state = (capacity * std::mem::size_of::<OrganismState>()) as u64;
        let bytes_genes = (capacity * GENE_STRIDE_F32 * std::mem::size_of::<f32>()) as u64;
        let buf_state = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("organism-state-128f32"),
            size: bytes_state,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_genes = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("organism-genes-16f32"),
            size: bytes_genes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_snapshot = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("organism-state-snapshot"),
            size: bytes_state,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buf_state,
            buf_genes,
            pre_pipeline,
            post_pipeline,
            layout,
            buf_snapshot,
            capacity,
            config_flags: 0,
        }
    }

    /// Select the scenario writer. Unknown bits are rejected instead of silently
    /// changing the legacy path.
    pub fn set_config_flags(&mut self, flags: u32) -> Result<(), &'static str> {
        if flags & !(CONFIG_DISABLE_OLD_THERMAL | CONFIG_STATIC_REST) != 0 {
            return Err("organism-taxi-v1: unknown config flag");
        }
        self.config_flags = flags;
        Ok(())
    }

    /// Start births with finite bounded DNA and immutable birth capacities.
    pub fn reset(
        &self,
        queue: &wgpu::Queue,
        genes: &[[f32; GENE_STRIDE_F32]],
    ) -> Result<(), &'static str> {
        if genes.len() > self.capacity {
            return Err("organism-taxi-v1: birth count exceeds capacity");
        }
        for genome in genes {
            if genome
                .iter()
                .any(|v| !v.is_finite() || !(-1.0..=1.0).contains(v))
            {
                return Err("organism-taxi-v1: gene outside [-1,1] or non-finite");
            }
        }
        let mut states = vec![OrganismState::zeroed(); self.capacity];
        let mut flat_genes = vec![0.0f32; self.capacity * GENE_STRIDE_F32];
        for (i, genome) in genes.iter().enumerate() {
            flat_genes[i * GENE_STRIDE_F32..(i + 1) * GENE_STRIDE_F32].copy_from_slice(genome);
            states[i] = birth_state(genome, self.config_flags)?;
        }
        queue.write_buffer(&self.buf_state, 0, bytemuck::cast_slice(&states));
        queue.write_buffer(&self.buf_genes, 0, bytemuck::cast_slice(&flat_genes));
        Ok(())
    }

    pub fn encode_pre(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        physics_input: &wgpu::Buffer,
        n: usize,
        dt: f32,
    ) -> Result<(), &'static str> {
        self.encode(device, encoder, &self.pre_pipeline, physics_input, n, dt)
    }

    pub fn encode_post(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        physics_output: &wgpu::Buffer,
        n: usize,
        dt: f32,
    ) -> Result<(), &'static str> {
        self.encode(device, encoder, &self.post_pipeline, physics_output, n, dt)
    }

    /// Copy a sampled resident state into the staging buffer; submit the same
    /// encoder as the tick, then call `read_snapshot` after completion.
    pub fn encode_snapshot(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        n: usize,
    ) -> Result<(), &'static str> {
        if n == 0 || n > self.capacity {
            return Err("organism-taxi-v1: invalid snapshot count");
        }
        encoder.copy_buffer_to_buffer(
            &self.buf_state,
            0,
            &self.buf_snapshot,
            0,
            (n * std::mem::size_of::<OrganismState>()) as u64,
        );
        Ok(())
    }

    pub fn read_snapshot(
        &self,
        device: &wgpu::Device,
        n: usize,
    ) -> Result<Vec<OrganismState>, &'static str> {
        if n == 0 || n > self.capacity {
            return Err("organism-taxi-v1: invalid snapshot count");
        }
        let bytes = (n * std::mem::size_of::<OrganismState>()) as u64;
        let slice = self.buf_snapshot.slice(0..bytes);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        if !matches!(receiver.recv(), Ok(Ok(()))) {
            return Err("organism-taxi-v1: snapshot mapping failed");
        }
        let mapped = slice.get_mapped_range();
        let values = bytemuck::cast_slice::<u8, OrganismState>(&mapped).to_vec();
        drop(mapped);
        self.buf_snapshot.unmap();
        for value in &values {
            validate_state(value)?;
        }
        Ok(values)
    }

    pub fn restore_bytes(&self, queue: &wgpu::Queue, bytes: &[u8]) -> Result<usize, &'static str> {
        let stride = std::mem::size_of::<OrganismState>();
        if bytes.is_empty()
            || bytes.len() % stride != 0
            || bytes.len() > self.buf_state.size() as usize
        {
            return Err("organism-taxi-v1: invalid state checkpoint bytes");
        }
        for value in bytemuck::cast_slice::<u8, OrganismState>(bytes) {
            validate_state(value)?;
        }
        queue.write_buffer(&self.buf_state, 0, bytes);
        Ok(bytes.len() / stride)
    }

    fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        physics: &wgpu::Buffer,
        n: usize,
        dt: f32,
    ) -> Result<(), &'static str> {
        if n > self.capacity || !dt.is_finite() || !(0.0..=1.0).contains(&dt) || dt == 0.0 {
            return Err("organism-taxi-v1: invalid n or dt");
        }
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("organism-step-params"),
            contents: bytemuck::bytes_of(&Params {
                n: n as u32,
                _pad: 0,
                dt,
                config_flags: self.config_flags,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("organism-bind-group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buf_state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_genes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: physics.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("organism-step"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP), 1, 1);
        Ok(())
    }
}

pub(crate) fn validate_state(state: &OrganismState) -> Result<(), &'static str> {
    if state.values.iter().any(|value| !value.is_finite()) {
        return Err("organism-taxi-v1: non-finite state checkpoint");
    }
    let v = &state.values;
    if v[28] != SCHEMA_VERSION as f32 || !(v[29] == 0.0 || v[29] == 1.0) {
        return Err("organism-taxi-v1: state schema or alive flag invalid");
    }
    if v[30] as u32 & !(CONFIG_DISABLE_OLD_THERMAL | CONFIG_STATIC_REST) != 0
        || v[30] < 0.0
        || v[30].fract() != 0.0
    {
        return Err("organism-taxi-v1: state config flags invalid");
    }
    for &index in &[S_ENERGY_J, S_WATER_KG, S_MATERIAL_KG] {
        if v[index] < 0.0 {
            return Err("organism-taxi-v1: negative resource in checkpoint");
        }
    }
    if v[S_ENERGY_J] > v[S_BIRTH_ENERGY_CAP]
        || v[S_WATER_KG] > v[S_BIRTH_WATER_CAP]
        || v[S_MATERIAL_KG] > v[S_BIRTH_MATERIAL_CAP]
        || v[S_TEMPERATURE_K] <= 0.0
        || v[S_OIL_TEMPERATURE_K] <= 0.0
    {
        return Err("organism-taxi-v1: resource, capacity, or temperature range invalid");
    }
    for &index in &[
        S_MATURITY,
        S_FATIGUE,
        S_ENGINE_WEAR,
        S_TYRE_WEAR,
        S_INTEGRITY,
        S_SENESCENCE,
    ] {
        if !(0.0..=1.0).contains(&v[index]) {
            return Err("organism-taxi-v1: normalized body state out of range");
        }
    }
    if v[S_BIRTH_ENERGY_CAP] <= 0.0 || v[S_BIRTH_WATER_CAP] <= 0.0 || v[S_BIRTH_MATERIAL_CAP] <= 0.0
    {
        return Err("organism-taxi-v1: invalid birth capacity");
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_abi_sizes_are_stable() {
        assert_eq!(std::mem::size_of::<OrganismState>(), STATE_STRIDE_F32 * 4);
        assert_eq!(std::mem::size_of::<Params>(), 16);
        assert_eq!(PHYSICS_INPUT_STRIDE_F32, 92);
        assert_eq!(PHYSICS_OUTPUT_STRIDE_F32, 56);
    }

    #[test]
    fn scenario_flag_is_the_high_legacy_boundary_bit() {
        assert_eq!(CONFIG_DISABLE_OLD_THERMAL, 1 << 8);
    }

    #[test]
    #[ignore = "requires a real shared GPU and a submitted physics pass"]
    fn gpu_body_invariants_require_real_gpu() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("organism invariant requires a real GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
                label: Some("organism-invariant"),
                ..Default::default()
            },
            None,
        ))
        .expect("organism invariant device");
        let mut body = OrganismGpu::new(&device, &queue, 1);
        body.set_config_flags(CONFIG_DISABLE_OLD_THERMAL).unwrap();
        body.reset(&queue, &[[0.0; GENE_STRIDE_F32]]).unwrap();
        let mut input = [0.0f32; PHYSICS_INPUT_STRIDE_F32];
        input[48] = 620.0;
        input[49] = 2_000.0;
        input[62] = 1_350.0;
        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("organism-test-input"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("organism-test-output"),
            size: (PHYSICS_OUTPUT_STRIDE_F32 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("organism-test"),
        });
        body.encode_pre(&device, &mut encoder, &input_buffer, 1, 1.0 / 64.0)
            .unwrap();
        body.encode_post(&device, &mut encoder, &output_buffer, 1, 1.0 / 64.0)
            .unwrap();
        body.encode_snapshot(&mut encoder, 1).unwrap();
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
        let state = body.read_snapshot(&device, 1).unwrap()[0];
        assert!(state.values[S_ENERGY_J] < 5_000_000.0);
        assert!((250.0..=500.0).contains(&state.values[S_TEMPERATURE_K]));
        assert_eq!(state.values[29], 1.0);
    }
}

/// Canonical newborn; acquired state never enters hereditary initialization.
pub fn birth_state(genome: &[f32; 16], flags: u32) -> Result<OrganismState, &'static str> {
    if genome
        .iter()
        .any(|v| !v.is_finite() || !(-1.0..=1.0).contains(v))
    {
        return Err("invalid organism DNA");
    }
    let mut state = OrganismState::zeroed();
    let v = &mut state.values;
    v[S_ENERGY_J] = 5_000_000.0;
    v[S_WATER_KG] = 10.0;
    v[S_MATERIAL_KG] = 50.0;
    v[S_INTEGRITY] = 1.0;
    v[S_TEMPERATURE_K] = 293.15;
    v[S_OIL_TEMPERATURE_K] = 293.15;
    v[S_BIRTH_ENGINE_CAP] = 1.0 + 0.2 * genome[0];
    v[S_BIRTH_BRAKE_CAP] = 1.0 + 0.2 * genome[1];
    v[S_BIRTH_ENERGY_CAP] = 5_000_000.0 * (1.0 + 0.25 * genome[2]);
    v[S_BIRTH_WATER_CAP] = 10.0 * (1.0 + 0.25 * genome[3]);
    v[S_BIRTH_MATERIAL_CAP] = 50.0 * (1.0 + 0.25 * genome[4]);
    v[S_BIRTH_COOLING] = 1.0 + 0.3 * genome[5];
    v[S_BIRTH_REPAIR_RATE] = 0.02 * (1.0 + genome[6]);
    v[S_BIRTH_GROWTH_RATE] = 0.02 * (1.0 + genome[7]);
    v[S_BIRTH_MASS] = 1.0 + 0.1 * genome[8];
    v[S_BIRTH_THERMAL_MARGIN] = 20.0 * (1.0 + genome[9]);
    v[S_BIRTH_LEAK_RESISTANCE] = 1.0 + 0.5 * genome[10];
    v[S_BIRTH_SENESCENCE_RESISTANCE] = 1.0 + 0.5 * genome[11];
    v[S_ENERGY_J] = v[S_BIRTH_ENERGY_CAP];
    v[S_WATER_KG] = v[S_BIRTH_WATER_CAP];
    v[S_MATERIAL_KG] = v[S_BIRTH_MATERIAL_CAP];
    // Capacity improvements pay construction mass and ongoing basal cost.
    v[S_BIRTH_MASS] += genome
        .iter()
        .take(16)
        .enumerate()
        .filter(|(i, _)| *i != 8 && *i != 15)
        .map(|(_, g)| 0.01 * *g)
        .sum::<f32>();
    v[28] = SCHEMA_VERSION as f32;
    v[29] = 1.0;
    v[30] = flags as f32;
    Ok(state)
}
