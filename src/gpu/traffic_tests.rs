//! Scripted GPU fixtures of geometry and analytic mechanics, not learned behavior.
use super::*;
#[test]
#[ignore = "exclusive hardware Vulkan fixture; no CPU fallback"]
fn gpu_traffic_crossings_and_pair_conservation() {
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
            label: Some("traffic analytic fixture"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
        },
        None,
    ))
    .unwrap();
    let traffic = TrafficGpu::new(
        &device,
        3,
        &[TrafficLight {
            position: [0., 0.],
            approach: [1., 0.],
            half_width: 2.,
            phase_offset_ticks: 0,
            green_ticks: 10,
            yellow_ticks: 2,
        }],
    )
    .unwrap();
    let buffer = |values: &[f32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(values),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
    };
    let mut inputs = vec![0.; 92 * 3];
    let mut outputs = vec![0.; 56 * 3];
    let mut bodies = vec![0.; 128 * 3];
    for b in 0..3 {
        inputs[b * 92 + 36] = -1.;
        inputs[b * 92 + 40] = 1.;
        inputs[b * 92 + 62] = 1000.;
        outputs[b * 56 + 36] = 1.;
        bodies[b * 128 + 29] = 1.;
        bodies[b * 128 + 64] = 10.;
    }
    let input = buffer(&inputs);
    let output = buffer(&outputs);
    let body = buffer(&bodies);
    let wave = buffer(&vec![0.; 48 * 3]);
    let cross = |tick| {
        let mut e = device.create_command_encoder(&Default::default());
        traffic
            .encode_senses(&device, &mut e, &input, &wave, 3, tick)
            .unwrap();
        traffic
            .encode_post(&device, &mut e, &output, &body, 3, tick)
            .unwrap();
        traffic
            .encode_post(&device, &mut e, &output, &body, 3, tick)
            .unwrap();
        queue.submit(Some(e.finish()));
    };
    cross(0);
    let rows = traffic.download(&device, &queue, 3);
    assert_eq!(rows[0][9], 1.);
    assert_eq!(rows[0][11], 1.);
    assert_eq!(rows[0][5], 1.);
    cross(12);
    let rows = traffic.download(&device, &queue, 3);
    assert_eq!(rows[0][8], 1.);
    assert_eq!(rows[0][11], 2.);
    assert_eq!(rows[0][5], -1.);
    assert_eq!(rows[0][14], 3., "one payment despite repeated post");
    // No crossing while parked beyond the stop line, including repeated post calls.
    inputs[36] = 1.;
    queue.write_buffer(&input, 0, bytemuck::cast_slice(&inputs));
    cross(13);
    let parked = traffic.download(&device, &queue, 3);
    assert_eq!(parked[0][11], 2.);
    assert_eq!(parked[0][14], 3.);
    inputs[36] = -1.;
    queue.write_buffer(&input, 0, bytemuck::cast_slice(&inputs));
    for tick in 14..18 {
        cross(tick);
    }
    assert_eq!(
        traffic.download(&device, &queue, 3)[0][14],
        10.,
        "fine cannot exceed cash; no negative balance"
    );
    // Two equal-mass approaching overlapping discs; third remains isolated.
    outputs.fill(0.);
    outputs[36] = -1.;
    outputs[0] = 1.;
    outputs[56 + 36] = 1.;
    outputs[56] = -1.;
    outputs[112 + 36] = 20.;
    queue.write_buffer(&output, 0, bytemuck::cast_slice(&outputs));
    let mut e = device.create_command_encoder(&Default::default());
    traffic
        .encode_contacts(&device, &mut e, &input, &output, &body, 3)
        .unwrap();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 3 * 224,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    e.copy_buffer_to_buffer(&output, 0, &staging, 0, 3 * 224);
    queue.submit(Some(e.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let mapped = staging.slice(..).get_mapped_range();
    let actual: &[f32] = bytemuck::cast_slice(&mapped);
    assert!((actual[0] + 0.05).abs() < 0.0001);
    assert!((actual[56] - 0.05).abs() < 0.0001);
    assert!((actual[0] + actual[56]).abs() < 0.0001, "pair momentum");
    let kinetic = 500. * (actual[0] * actual[0] + actual[56] * actual[56]);
    assert!(
        (kinetic + actual[55] + actual[111] - 1000.).abs() < 0.001,
        "dissipation accounted exactly once across pair"
    );
    assert!((actual[36] + 1.5).abs() < 0.0001);
    assert!((actual[92] - 1.5).abs() < 0.0001);
    assert_eq!(actual[112 + 36], 20.);
    assert_eq!(actual[112], 0.);
    assert_eq!(actual[112 + 55], 0.);
}
