//! GPU feedback adapter; records consequences, never an action prescription.
use wgpu::util::DeviceExt;

pub struct OrganismFeedback {
    pipeline: wgpu::ComputePipeline,
}
impl OrganismFeedback {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("organism-feedback"),
            source: wgpu::ShaderSource::Wgsl(include_str!("organism_feedback.wgsl").into()),
        });
        Self {
            pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("organism-feedback"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            }),
        }
    }
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        body: &wgpu::Buffer,
        life: &wgpu::Buffer,
        n: usize,
        mode: u32,
    ) {
        // mode 0 legacy asymmetric energy term; mode 1 symmetric energy + paid-money cost;
        // mode 2 = mode 1 + same-key task progress (organism_feedback.wgsl header).
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("organism-feedback-count"),
            contents: bytemuck::cast_slice(&[n as u32, mode, 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: body.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: life.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn new_scenario_shaders_are_valid() {
        for source in [
            include_str!("organism.wgsl"),
            include_str!("organism_feedback.wgsl"),
            include_str!("taxi.wgsl"),
        ] {
            let module = naga::front::wgsl::parse_str(source).expect("scenario WGSL parses");
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .expect("scenario WGSL validates");
        }
    }
}
