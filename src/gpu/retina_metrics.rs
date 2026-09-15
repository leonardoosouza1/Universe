//! Sampled retinal coverage and body proximity, reduced on the shared GPU device.
//! Readback is 32 bytes per being per sample, never the full retinal frame.

use super::core::{storage_entry, uniform_entry};
use crate::config::iara::vision::{RETINA_COLS, RETINA_TOTAL};
use wgpu::util::DeviceExt;

pub struct RetinaMetricsGpu {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    output: wgpu::Buffer,
    readback: wgpu::Buffer,
    capacity: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_validates() {
        let module = naga::front::wgsl::parse_str(include_str!("retina_metrics.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    #[ignore = "requires exclusive GPU access"]
    fn known_retinal_coverage_and_proximity() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        let metrics = RetinaMetricsGpu::new(&device, 3);
        let mut pixels = vec![0u32; 3 * RETINA_TOTAL];
        pixels[0] = (2500 << 8) | 4;
        pixels[RETINA_COLS - 1] = (1500 << 8) | 4;
        pixels[RETINA_TOTAL + RETINA_COLS - 1] = (1000 << 8) | 4;
        let retina = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&pixels),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let poses: [[f32; 8]; 3] = [
            [0., 1., 0., 0., 1., 0., 0., 1.],
            [1., 1., 0., 0., 1., 0., 0., 1.],
            [10., 1., 0., 0., 1., 0., 0., 1.],
        ];
        let poses = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&poses),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        metrics.dispatch(&device, &mut encoder, &retina, &poses, 3);
        queue.submit(Some(encoder.finish()));
        let rows = metrics.read(&device, 3);
        assert!((rows[0][0] - 2.0 / RETINA_TOTAL as f32).abs() < 1e-7);
        for (actual, expected) in rows[0][1..7].iter().zip([0., 1., 1., 1.5, 1., 1.]) {
            assert!((actual - expected).abs() < 1e-6);
        }
        for (actual, expected) in rows[1][1..7].iter().zip([1., 0., 1., 1., 1., 1.]) {
            assert!((actual - expected).abs() < 1e-6);
        }
        assert_eq!(&rows[2][0..7], &[0., 0., 0., 0., 1200., 9., 0.]);
    }
}

impl RetinaMetricsGpu {
    pub fn new(device: &wgpu::Device, capacity: usize) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("retina-metrics"),
            source: wgpu::ShaderSource::Wgsl(include_str!("retina_metrics.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("retina-metrics"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("retina-metrics"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffer = |label, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (capacity * 32) as u64,
                usage,
                mapped_at_creation: false,
            })
        };
        Self {
            pipeline,
            layout,
            capacity,
            output: buffer(
                "retina-metrics-output",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
            readback: buffer(
                "retina-metrics-readback",
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
        }
    }

    /// Records a reduction after the retina dispatch, into the same submission.
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        retina: &wgpu::Buffer,
        poses: &wgpu::Buffer,
        n: usize,
    ) {
        assert!(n <= self.capacity);
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("retina-metrics-params"),
            contents: bytemuck::cast_slice(&[n as u32, RETINA_TOTAL as u32, RETINA_COLS as u32, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: retina.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: poses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.output.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.output, 0, &self.readback, 0, (n * 32) as u64);
    }

    /// Call only after the encoder containing `dispatch` has been submitted.
    pub fn read(&self, device: &wgpu::Device, n: usize) -> Vec<[f32; 8]> {
        let slice = self.readback.slice(..(n * 32) as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |status| {
            let _ = tx.send(status);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("retinal metrics callback")
            .expect("retinal metrics readback");
        let data = slice.get_mapped_range();
        let result = bytemuck::cast_slice::<u8, [f32; 8]>(&data).to_vec();
        drop(data);
        self.readback.unmap();
        result
    }
}
