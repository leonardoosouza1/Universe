//! Bounded GPU-resident reference controller for `organism-taxi-v1`.
//!
//! This is a pilot for physical feasibility only. It is deliberately separate
//! from evolution: the caller must opt in and route its output through the
//! production `OrganismActionGpu` mapper. The controller never advances physics,
//! mutates taxi state, reads back a buffer, or injects a position.
//!
//! Inputs are the resident packed observation wave and the resident physics
//! input. The output is the existing 96-f32 intent ABI consumed by
//! `organism_action.wgsl`; therefore pickup, dropoff, and service requests still
//! obey the ordinary `taxi.wgsl` distance, speed, stock, and payment rules.

use std::borrow::Cow;

/// Stable identifier for manifests and explicit pilot opt-in.
pub const TAXI_WITNESS_ID: &str = "organism-taxi-reference-witness-v2";
const INTENT_STRIDE: usize = 96;
const PHYSICS_STRIDE: usize = 92;
const BODY_STRIDE: usize = 128;
const WAVE_WORDS: usize = 48;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    wave_words: u32,
    lesson: u32,
    _pad: u32,
}

/// GPU-only bounded controller. No method on this type submits a command buffer.
pub struct TaxiWitnessGpu {
    params: wgpu::Buffer,
    hold_ticks: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    pub capacity: usize,
    lesson: u32,
}

impl TaxiWitnessGpu {
    /// `lesson == 0` is production. Values 1..=8 mirror `school::Lesson`.
    pub fn new(device: &wgpu::Device, capacity: usize, lesson: u32) -> Self {
        assert!(lesson <= 8, "taxi witness lesson must be 0..=8");
        let capacity = capacity.max(1);
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("taxi-reference-witness-params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let hold_ticks = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("taxi-reference-witness-hold-ticks"),
            size: (capacity * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(TAXI_WITNESS_ID),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("taxi_witness.wgsl"))),
        });
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("taxi-reference-witness-layout"),
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
                storage(1, true),  // packed observation wave
                storage(2, true),  // resident physics input
                storage(3, true),  // resident body/taxi state
                storage(4, true),  // immutable route distance fields
                storage(5, true),  // immutable taxi station map
                storage(6, false), // production intent ABI
                storage(7, false), // consecutive stopped-at-goal ticks
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("taxi-reference-witness-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(TAXI_WITNESS_ID),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            params,
            hold_ticks,
            layout,
            pipeline,
            capacity,
            lesson,
        }
    }

    /// Encode one bounded controller pass. The caller must invoke this before
    /// `OrganismActionGpu::encode` in the same command encoder.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        packed_wave: &wgpu::Buffer,
        physics_input: &wgpu::Buffer,
        body_state: &wgpu::Buffer,
        routes: &wgpu::Buffer,
        taxi_world: &wgpu::Buffer,
        intent: &wgpu::Buffer,
        n: usize,
    ) -> Result<(), &'static str> {
        if n == 0 || n > self.capacity {
            return Err("taxi-witness: invalid count");
        }
        if packed_wave.size() < (n * WAVE_WORDS * 4) as u64
            || physics_input.size() < (n * PHYSICS_STRIDE * 4) as u64
            || body_state.size() < (n * BODY_STRIDE * 4) as u64
            || routes.size() < 16
            || taxi_world.size() < 4 * 4
            || intent.size() < (n * INTENT_STRIDE * 4) as u64
        {
            return Err("taxi-witness: buffer shape");
        }
        queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&Params {
                n: n as u32,
                wave_words: WAVE_WORDS as u32,
                lesson: self.lesson,
                _pad: 0,
            }),
        );
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("taxi-reference-witness-bind-group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: packed_wave.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: physics_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: body_state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: routes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: taxi_world.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: intent.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.hold_ticks.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("taxi-reference-witness"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn witness_shader_contract_parses_without_simulation() {
        let source = include_str!("taxi_witness.wgsl");
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
    }
}
