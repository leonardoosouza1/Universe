use wgpu::util::DeviceExt;
#[test]
#[ignore = "exclusive real Vulkan GPU required; packing fixture, no CPU simulation"]
fn gpu_temporal_fuel_overwrites_spatial_byte() {
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
    assert_ne!(adapter.get_info().device_type, wgpu::DeviceType::Cpu);
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
        },
        None,
    ))
    .unwrap();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(include_str!("perception_temporal.wgsl").into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let buf = |data: &[u8], usage| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: data,
            usage,
        })
    };
    let params = buf(
        bytemuck::cast_slice(&[1u32, 1, 1f32.to_bits(), 0, 0, 0, 0, 0]),
        wgpu::BufferUsages::UNIFORM,
    );
    let mut previous = [0u32; 48];
    previous[5] = 0x1234567f;
    let output = buf(
        bytemuck::cast_slice(&previous),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let mut values = [0f32; 73];
    values[61] = 0.25;
    let temporal = buf(bytemuck::cast_slice(&values), wgpu::BufferUsages::STORAGE);
    let entries: Vec<_> = [(0, &params), (6, &output), (7, &temporal)]
        .into_iter()
        .map(|(binding, b)| wgpu::BindGroupEntry {
            binding,
            resource: b.as_entire_binding(),
        })
        .collect();
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 20, &staging, 0, 4);
    queue.submit(Some(encoder.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let mapped = staging.slice(..).get_mapped_range();
    let actual = bytemuck::cast_slice::<u8, u32>(&mapped)[0];
    assert_eq!(
        actual, 0x1234561f,
        "25% energy is31; wall127 must not contaminate it or adjacent bytes"
    );
}
