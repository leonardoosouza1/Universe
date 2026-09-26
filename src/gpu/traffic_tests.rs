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
    drop(mapped);
    staging.unmap();
    // Repeat the same overlap with zero relative velocity: contact, no impact.
    outputs[0] = 0.0;
    outputs[56] = 0.0;
    queue.write_buffer(&output, 0, bytemuck::cast_slice(&outputs));
    let mut e = device.create_command_encoder(&Default::default());
    traffic.encode_contacts(&device, &mut e, &input, &output, &body, 3).unwrap();
    e.copy_buffer_to_buffer(&output, 0, &staging, 0, 3 * 224);
    queue.submit(Some(e.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let mapped = staging.slice(..).get_mapped_range();
    let actual: &[f32] = bytemuck::cast_slice(&mapped);
    assert_eq!(actual[39], 16.0);
    assert_eq!(actual[56 + 39], 16.0);
    assert_eq!(actual[55] + actual[56 + 55], 0.0);
    assert_eq!(actual[47] + actual[56 + 47], 0.0);
    assert_eq!(actual[112 + 39], 0.0);

}

/// Contact mask contract: physics_c writes the numeric value f32(mask); traffic
/// ORs bit 16 numerically; orientation feedback copies it numerically; telemetry
/// stores it numerically and counts one edge. Before this fixture the feedback
/// wrote f32(bitcast<u32>(1.0)) = 1065353216 into the physics record.
#[test]
#[ignore = "exclusive hardware Vulkan fixture; no CPU fallback"]
fn gpu_contact_mask_is_numeric_across_traffic_feedback_and_telemetry() {
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
            label: Some("contact mask fixture"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
        },
        None,
    ))
    .unwrap();
    let n = 3usize;
    let traffic = TrafficGpu::new(
        &device,
        n,
        &[TrafficLight { position: [500., 500.], approach: [1., 0.], half_width: 2., phase_offset_ticks: 0, green_ticks: 10, yellow_ticks: 2 }],
    )
    .unwrap();
    let buffer = |values: &[f32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(values),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        })
    };
    let read = |source: &wgpu::Buffer, floats: usize| -> Vec<f32> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (floats * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut e = device.create_command_encoder(&Default::default());
        e.copy_buffer_to_buffer(source, 0, &staging, 0, (floats * 4) as u64);
        queue.submit(Some(e.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let out = bytemuck::cast_slice::<u8, f32>(&staging.slice(..).get_mapped_range()).to_vec();
        staging.unmap();
        out
    };
    let mut inputs = vec![0.0f32; 92 * n];
    let mut outputs = vec![0.0f32; 56 * n];
    let mut bodies = vec![0.0f32; 128 * n];
    for b in 0..n {
        inputs[b * 92 + 62] = 1000.0;
        inputs[b * 92 + 40] = 1.0;
        bodies[b * 128 + 29] = 1.0;
    }
    // Approaching pair with probe masks already reported by physics_c: 1 and 5.
    outputs[36] = -1.0; outputs[0] = 1.0; outputs[39] = 1.0;
    outputs[56 + 36] = 1.0; outputs[56] = -1.0; outputs[56 + 39] = 5.0;
    outputs[112 + 36] = 20.0; outputs[112 + 39] = 8.0;
    let input = buffer(&inputs);
    let output = buffer(&outputs);
    let body = buffer(&bodies);
    let mut e = device.create_command_encoder(&Default::default());
    traffic.encode_contacts(&device, &mut e, &input, &output, &body, n).unwrap();
    queue.submit(Some(e.finish()));
    let after = read(&output, 56 * n);
    assert_eq!(after[39], 17.0, "1 | 16 numerically");
    assert_eq!(after[56 + 39], 21.0, "5 | 16 numerically");
    assert_eq!(after[112 + 39], 8.0, "isolated being keeps its probe mask");
    for b in 0..n {
        let mask = after[b * 56 + 39];
        assert!(mask >= 0.0 && mask.fract() == 0.0 && (mask as u32) & !31 == 0, "mask {mask} outside bits 0..4");
    }
    // Orientation feedback copies slot 39 numerically into the physics record.
    let orientation = crate::gpu::organism_orientation::OrganismOrientationGpu::new(&device);
    let mut e = device.create_command_encoder(&Default::default());
    orientation.encode_feedback_to_input(&device, &mut e, &output, &input, n).unwrap();
    queue.submit(Some(e.finish()));
    let fed = read(&input, 92 * n);
    assert_eq!(fed[39], 17.0);
    assert_eq!(fed[92 + 39], 21.0);
    assert_ne!(fed[39], 1065353216.0, "no bit pattern leaks into the record");
    let mut host: crate::gpu::physics::PhysicsStateGpu = bytemuck::Zeroable::zeroed();
    host._pad_pos = fed[39];
    assert_eq!(host.contact_diagnostics().contact_mask, 17);
    // Telemetry stores the numeric mask and counts exactly one contact edge.
    let telemetry = crate::gpu::organism_telemetry::OrganismTelemetryGpu::new(&device, n);
    // The window resets when tick % 64 == 1, so the second sample uses tick 2.
    let run_telemetry = |tick: usize| {
        let mut e = device.create_command_encoder(&Default::default());
        telemetry.encode(&device, &mut e, &input, &output, &body, n, tick).unwrap();
        queue.submit(Some(e.finish()));
        read(&body, 128 * n)
    };
    let first = run_telemetry(1);
    assert_eq!(first[123], 17.0, "body[123] numeric contact memory");
    assert_eq!(first[128 + 123], 21.0);
    assert_eq!(first[104 + 4], 1.0, "one contact edge counted from a zero memory");
    assert_eq!(first[104 + 2], 1.0, "one touching tick");
    let second = run_telemetry(2);
    assert_eq!(second[123], 17.0);
    assert_eq!(second[104 + 4], 1.0, "no second edge while contact persists");
    assert_eq!(second[104 + 2], 2.0, "touching ticks accumulate");
    assert_eq!(first[2 * 128 + 123], 8.0, "isolated being stores its probe mask");
}
