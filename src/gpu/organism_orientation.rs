//! GPU orientation integration for the organism scenario, after physics kernel C.
//! This replaces the host yaw update; callers must not apply both. Angular velocity
//! is already integrated by C and is read without modification here.
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    dt: f32,
    pad0: u32,
    pad1: u32,
}

pub struct OrganismOrientationGpu {
    pipeline: wgpu::ComputePipeline,
    feedback_pipeline: wgpu::ComputePipeline,
}
impl OrganismOrientationGpu {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("organism orientation"),
            source: wgpu::ShaderSource::Wgsl(include_str!("organism_orientation.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("organism orientation"),
            layout: None,
            module: &module,
            entry_point: Some("rotate"),
            compilation_options: Default::default(),
            cache: None,
        });
        let feedback_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("physics resident feedback"),
            layout: None,
            module: &module,
            entry_point: Some("feedback"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            pipeline,
            feedback_pipeline,
        }
    }
    /// Copy acquired kernel-C fields into the persistent 368-byte input record.
    /// Call after body post and rotation; no upload, submission or readback occurs.
    pub fn encode_feedback_to_input(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::Buffer,
        input: &wgpu::Buffer,
        n: usize,
    ) -> Result<(), &'static str> {
        if n > u32::MAX as usize || output.size() < n as u64 * 224 || input.size() < n as u64 * 368
        {
            return Err("invalid physics feedback shape");
        }
        if n == 0 {
            return Ok(());
        }
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&Params {
                n: n as u32,
                dt: 0.0,
                pad0: 0,
                pad1: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.feedback_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: input.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("physics resident feedback"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.feedback_pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        Ok(())
    }

    /// Encode after C, before physics readback or resident next-state copy.
    /// `dt` is the same timestep supplied to physics (not a field in CarOutput).
    pub fn encode_rotation_gpu(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::Buffer,
        n: usize,
        dt: f32,
    ) -> Result<(), &'static str> {
        if n > u32::MAX as usize || !dt.is_finite() || dt <= 0.0 || output.size() < (n as u64) * 224
        {
            return Err("invalid organism orientation shape/timestep");
        }
        if n == 0 {
            return Ok(());
        }
        let params = Params {
            n: n as u32,
            dt,
            pad0: 0,
            pad1: 0,
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("orientation params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("organism orientation"),
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
    use super::*;
    #[test]
    fn orientation_shader_validates() {
        let module =
            naga::front::wgsl::parse_str(include_str!("organism_orientation.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        assert_eq!(
            std::mem::size_of::<super::super::physics::PhysicsOutputGpu>(),
            224
        );
    }
    #[test]
    #[ignore = "analytic orientation fixture requires a real hardware Vulkan GPU"]
    fn gpu_orientation_quarter_turn_and_slot_bounds() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .unwrap();
        assert_ne!(
            adapter.get_info().device_type,
            wgpu::DeviceType::Cpu,
            "CPU adapter forbidden"
        );
        eprintln!("orientation fixture adapter: {:?}", adapter.get_info());
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits::downlevel_defaults(),
                ..Default::default()
            },
            None,
        ))
        .unwrap();
        let mut source = [[0.0f32; 56]; 3];
        for (i, row) in source.iter_mut().enumerate() {
            row[3] = if i == 0 {
                std::f32::consts::FRAC_PI_2
            } else {
                -std::f32::consts::FRAC_PI_2
            };
            row[40] = 2.0;
            row[46] = 3.0;
            row[41] = 0.25;
            row[45] = -0.5;
            row[39] = 5.0; // numeric contact mask: probes 0 and 2
            row[55] = 17.0;
        }
        let output = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&source),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let input_initial = [[777.0f32; 92]; 3];
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&input_initial),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let input_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 1104,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 672,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let kernel = OrganismOrientationGpu::new(&device);
        let mut encoder = device.create_command_encoder(&Default::default());
        kernel
            .encode_rotation_gpu(&device, &mut encoder, &output, 2, 1.0)
            .unwrap();
        kernel
            .encode_feedback_to_input(&device, &mut encoder, &output, &input, 2)
            .unwrap();
        encoder.copy_buffer_to_buffer(&input, 0, &input_staging, 0, 1104);
        encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, 672);
        queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let mapped = staging.slice(..).get_mapped_range();
        let rows: &[[f32; 56]] = bytemuck::cast_slice(&mapped);
        // Analytic ±90-degree rotations, independent of any CPU simulation/reference integrator.
        for (i, expected) in [[0.0, 1.0, -1.0, 0.0], [0.0, -1.0, 1.0, 0.0]]
            .iter()
            .enumerate()
        {
            for (slot, value) in [40, 42, 44, 46].into_iter().zip(expected) {
                assert!((rows[i][slot] - value).abs() < 0.00001);
            }
            for slot in 0..56 {
                if ![40, 42, 44, 46].contains(&slot) {
                    assert_eq!(rows[i][slot].to_bits(), source[i][slot].to_bits());
                }
            }
        }
        assert_eq!(rows[2], source[2]);
        let (tx, rx) = std::sync::mpsc::channel();
        input_staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let input_map = input_staging.slice(..).get_mapped_range();
        let inputs: &[[f32; 92]] = bytemuck::cast_slice(&input_map);
        for being in 0..2 {
            for slot in 0..92 {
                let from = if slot < 48 {
                    Some(slot)
                } else {
                    match slot {
                        49 => Some(48),
                        54 => Some(49),
                        56 => Some(50),
                        57 => Some(51),
                        58 => Some(52),
                        60 => Some(53),
                        61 => Some(54),
                        89 => Some(55),
                        _ => None,
                    }
                };
                // Slot 39 is the numeric contact mask on both sides (physics_c writes
                // f32(mask)); the old bit-pattern copy turned 1.0 into 1065353216.
                let expected = from.map(|o| rows[being][o]).unwrap_or(777.0);
                assert_eq!(
                    inputs[being][slot].to_bits(),
                    expected.to_bits(),
                    "feedback being {being} slot {slot}"
                );
            }
        }
        assert_eq!(inputs[2], input_initial[2]);
    }
}
