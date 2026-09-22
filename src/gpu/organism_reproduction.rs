//! Paid resident reproduction. A refused GPU transaction changes neither slot.
//! Child water is 25% of its expressed capacity, above the strict 20% growth
//! threshold; the parent pays every kilogram and retains at least 1 kg. Material
//! is 10% capacity plus paid construction: even minimum DNA capacity supplies
//! 3.75 kg, exceeding the 0.2 kg required for a full unit of growth.
use super::organism::OrganismState;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    parent: u32,
    child: u32,
    n: u32,
    pad: u32,
    energy: f32,
    cost: f32,
    reserve: f32,
    pad_water: f32,
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct BirthReceipt {
    pub values: [f32; 16],
}
impl BirthReceipt {
    pub fn accepted(&self) -> bool {
        self.values[0] == 1.0
    }
}

pub struct ReproductionGpu {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}
impl ReproductionGpu {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("paid reproduction"),
            source: wgpu::ShaderSource::Wgsl(include_str!("organism_reproduction.wgsl").into()),
        });
        let entries: Vec<_> = (0..4)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding == 2,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reproduction"),
            entries: &entries,
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("reproduction"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("reproduce"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { layout, pipeline }
    }
    /// Synchronous bounded scheduler boundary; all numerical debit/transfer occurs on GPU.
    pub fn transact(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        body: &wgpu::Buffer,
        n: usize,
        parent: usize,
        child: usize,
        template: &OrganismState,
        energy: f32,
        cost: f32,
        reserve: f32,
    ) -> Result<BirthReceipt, &'static str> {
        if n == 0
            || n > u32::MAX as usize
            || parent >= n
            || child >= n
            || body.size() < n as u64 * 512
            || ![energy, cost, reserve]
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0)
        {
            return Err("invalid reproduction request");
        }
        if !template.values.iter().all(|v| v.is_finite())
            || template.values[18] <= 0.0
            || template.values[19] <= 0.0
            || template.values[20] <= 0.0
            || template.values[24] <= 0.0
        {
            return Err("invalid newborn template");
        }
        let params = Params {
            parent: parent as u32,
            child: child as u32,
            n: n as u32,
            pad: 0,
            energy,
            cost,
            reserve,
            pad_water: 0.0,
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let fresh = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(template),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let receipt = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: body.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: fresh.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: receipt.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&receipt, 0, &staging, 0, 64);
        queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| "reproduction map callback")?
            .map_err(|_| "reproduction map")?;
        let mapped = staging.slice(..).get_mapped_range();
        let mut values = [0.0; 16];
        values.copy_from_slice(bytemuck::cast_slice(&mapped));
        drop(mapped);
        staging.unmap();
        Ok(BirthReceipt { values })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reproduction_shader_validates() {
        let module =
            naga::front::wgsl::parse_str(include_str!("organism_reproduction.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
    #[test]
    #[ignore = "scripted fixture requires a real hardware Vulkan GPU"]
    fn gpu_paid_reproduction_transaction() {
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
            "CPU adapters forbidden"
        );
        eprintln!("reproduction fixture adapter: {:?}", adapter.get_info());
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_limits: wgpu::Limits::downlevel_defaults(),
                ..Default::default()
            },
            None,
        ))
        .unwrap();
        let fresh = super::super::organism::birth_state(&[0.0; 16], 256).unwrap();
        let mut initial = [fresh; 3];
        initial[0].values[4] = 1.0;
        initial[0].values[64] = 42.0;
        initial[1].values[29] = 0.0;
        let mut body_model = super::super::organism::OrganismGpu::new(&device, &queue, 3);
        body_model.set_config_flags(256).unwrap();
        body_model.reset(&queue, &[[0.0; 16]; 3]).unwrap();
        let body = &body_model.buf_state;
        queue.write_buffer(body, 0, bytemuck::cast_slice(&initial));
        let gpu = ReproductionGpu::new(&device);
        let receipt = gpu
            .transact(
                &device,
                &queue,
                &body,
                3,
                0,
                1,
                &fresh,
                500_000.0,
                100_000.0,
                1_000_000.0,
            )
            .unwrap();
        assert!(receipt.accepted());
        let r = receipt.values;
        assert_eq!(r[1] - r[4], r[7] + r[11]);
        assert_eq!(r[2] - r[5], r[8]);
        assert_eq!(r[8], fresh.values[19] * 0.25);
        assert!(
            r[8] > fresh.values[19] * 0.2,
            "newborn must clear growth water gate"
        );
        assert!(r[5] >= 1.0, "parent water reserve must remain paid");
        assert!(
            r[9] >= 0.2,
            "newborn material can pay a full unit of growth"
        );
        assert_eq!(r[3] - r[6], r[9] + r[10]);
        assert!(!gpu
            .transact(
                &device,
                &queue,
                &body,
                3,
                0,
                1,
                &fresh,
                500_000.0,
                100_000.0,
                1_000_000.0
            )
            .unwrap()
            .accepted());
        // Refused duplicate leaves all slots bit-identical; survivor was never touched.
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 1536,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&body, 0, &staging, 0, 1536);
        queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let map = staging.slice(..).get_mapped_range();
        let states: &[OrganismState] = bytemuck::cast_slice(&map);
        assert_eq!(states[0].values[0], r[4]);
        assert_eq!(states[0].values[64], 42.0);
        assert_eq!(states[1].values[64], 0.0);
        assert_eq!(states[1].values[93], 1.0);
        assert_eq!(states[2].values, initial[2].values);
        drop(map);
        staging.unmap();
        // Follow the accepted child through real GPU physiology, with explicitly
        // stationary physics records. No service/cash grant or CPU model is used.
        let mut physical_input = vec![0.0f32; 3 * 92];
        let mut physical_output = vec![0.0f32; 3 * 56];
        for b in 0..3 {
            physical_input[b * 92 + 48] = 620.0;
            physical_input[b * 92 + 49] = 800.0;
            physical_input[b * 92 + 50] = 7000.0;
            physical_input[b * 92 + 55] = f32::from_bits(2);
            physical_input[b * 92 + 62] = 1350.0;
            physical_input[b * 92 + 68] = 4000.0;
            physical_output[b * 56 + 48] = 800.0;
        }
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&physical_input),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&physical_output),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        body_model
            .encode_pre(&device, &mut encoder, &input, 3, 1.0)
            .unwrap();
        body_model
            .encode_post(&device, &mut encoder, &output, 3, 1.0)
            .unwrap();
        body_model.encode_snapshot(&mut encoder, 3).unwrap();
        queue.submit(Some(encoder.finish()));
        let progressed = body_model.read_snapshot(&device, 3).unwrap();
        let child = &progressed[1].values;
        assert!(
            child[4] > 0.0,
            "paid newborn must grow without a cash/service bootstrap"
        );
        assert!(
            child[12] > 0.0 && child[2] < r[9],
            "growth must pay energy and material"
        );
        assert_eq!(child[64], 0.0, "growth must not mint currency");
        assert!((r[7] - child[0] - child[51] - child[52] - child[12] - child[11]).abs() < 0.1);
    }
}
