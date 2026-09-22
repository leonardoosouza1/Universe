//! Resident bounded driving telemetry accumulator.
//!
//! The 19 counters at body slots 104..122 mirror the diagnostics `Window`
//! ordering.  Slot 123 stores the previous contact mask as bits in an f32 and
//! is deliberately outside the rolling window.  The kernel is scheduled after
//! physics and before the organism post pass; it never performs a readback.

use std::borrow::Cow;

use wgpu::util::DeviceExt;

pub const PHYSICS_INPUT_STRIDE: usize = 92;
pub const PHYSICS_OUTPUT_STRIDE: usize = 56;
pub const BODY_STRIDE: usize = 128;
pub const WINDOW_START: usize = 104;
pub const WINDOW_END: usize = 123;
pub const PREVIOUS_CONTACT: usize = 123;
pub const WINDOW_TICKS: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    tick: u32,
}

pub struct OrganismTelemetryGpu {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    pub capacity: usize,
}

impl OrganismTelemetryGpu {
    pub fn new(device: &wgpu::Device, capacity: usize) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("organism-telemetry-layout"),
            entries: &[
                uniform(0),
                storage(1, true),
                storage(2, true),
                storage(3, false),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("organism-telemetry"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "organism_telemetry.wgsl"
            ))),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("organism-telemetry-pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("organism-telemetry-pipeline-layout"),
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            layout,
            pipeline,
            capacity: capacity.max(1),
        }
    }

    /// Accumulate one post-physics sample for every being. This is intentionally
    /// GPU-resident: callers submit the containing encoder with physics and body
    /// passes, then observe the window only at their existing checkpoint/readback
    /// boundary.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        physics_input: &wgpu::Buffer,
        physics_output: &wgpu::Buffer,
        body: &wgpu::Buffer,
        n: usize,
        tick: usize,
    ) -> Result<(), &'static str> {
        if n == 0 || n > self.capacity || tick > u32::MAX as usize {
            return Err("organism-telemetry: invalid count/tick");
        }
        for (buffer, stride) in [
            (physics_input, PHYSICS_INPUT_STRIDE),
            (physics_output, PHYSICS_OUTPUT_STRIDE),
            (body, BODY_STRIDE),
        ] {
            if !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
                || buffer.size() < n as u64 * stride as u64 * 4
            {
                return Err("organism-telemetry: invalid resident buffer");
            }
        }
        // A per-dispatch immutable uniform avoids a host queue write and keeps
        // this adapter usable inside a caller-owned CommandEncoder.
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("organism-telemetry-dispatch-params"),
            contents: bytemuck::bytes_of(&Params {
                n: n as u32,
                tick: tick as u32,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("organism-telemetry-bind-group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: physics_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: physics_output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: body.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("organism-telemetry"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        Ok(())
    }
}

fn uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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
    fn shader_validates_and_window_layout_is_stable() {
        let module = naga::front::wgsl::parse_str(include_str!("organism_telemetry.wgsl"))
            .expect("telemetry WGSL parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("telemetry WGSL validates");
        assert_eq!(WINDOW_END - WINDOW_START, 19);
        assert_eq!(PREVIOUS_CONTACT, 123);
    }

    #[test]
    #[ignore = "exclusive real GPU; resident telemetry fixture"]
    fn gpu_window_and_lifetime_fixture() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("real Vulkan adapter");
        assert_ne!(adapter.get_info().device_type, wgpu::DeviceType::Cpu);
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("organism-telemetry-fixture"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        ))
        .expect("GPU device");
        let telemetry = OrganismTelemetryGpu::new(&device, 1);
        let mut input = [0.0f32; PHYSICS_INPUT_STRIDE];
        input[36] = 1.0;
        input[51] = 0.6;
        input[52] = 0.4;
        input[55] = f32::from_bits(2);
        let mut output = [0.0f32; PHYSICS_OUTPUT_STRIDE];
        output[0] = 1.0;
        output[36] = 3.0;
        output[39] = f32::from_bits(1);
        output[43] = 5.0;
        output[47] = 4.0;
        output[55] = 12.5;
        output[4 + 1] = 0.25;
        output[4 + 2] = 0.1;
        let mut body = [0.0f32; BODY_STRIDE];
        body[29] = 1.0;
        body[103] = 2.0;
        let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("telemetry-input"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("telemetry-output"),
            contents: bytemuck::cast_slice(&output),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let body_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("telemetry-body"),
            contents: bytemuck::cast_slice(&body),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("telemetry-readback"),
            size: (BODY_STRIDE * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("telemetry-fixture"),
        });
        telemetry
            .encode(
                &device,
                &mut encoder,
                &input_buf,
                &output_buf,
                &body_buf,
                1,
                1,
            )
            .unwrap();
        encoder.copy_buffer_to_buffer(&body_buf, 0, &staging, 0, staging.size());
        queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range();
        let values: &[f32] = bytemuck::cast_slice(&mapped);
        assert_eq!(values[104], 1.0);
        assert_eq!(values[105], 1.0);
        assert_eq!(values[106], 1.0);
        assert_eq!(values[107], 1.0);
        assert_eq!(values[108], 1.0);
        assert_eq!(values[109], 1.0);
        assert_eq!(values[110], 1.0);
        assert_eq!(values[111], 0.0);
        assert_eq!(values[112], 0.0);
        assert_eq!(values[113], 1.0);
        assert_eq!(values[114], 1.0);
        assert_eq!(values[115], 2.0);
        assert_eq!(values[116], 12.5);
        assert_eq!(values[117], 4.0);
        assert_eq!(values[118], 5.0);
        assert_eq!(values[119], 0.25);
        assert_eq!(values[120], 0.1);
        assert_eq!(values[121], 0.6);
        assert_eq!(values[122], 0.4);
        assert_eq!(values[98], 2.0);
        assert_eq!(values[99], 1.0);
        assert_eq!(values[100], 12.5);
        assert_eq!(values[101], 1.0);
        assert_eq!(values[102], 1.0);
        assert_eq!(values[123].to_bits(), 1);
        drop(mapped);
        staging.unmap();
    }
}
