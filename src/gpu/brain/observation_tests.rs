use super::{
    brain_control_word, fast_state_copy_range, CFC_SLOTS_PER_BEING, FAST_CFC_STATE_BYTES,
    L1_NEURONS,
};

#[test]
fn packed_control_keeps_ablation_and_observation_independent() {
    assert_eq!(brain_control_word(false, None), 0);
    assert_eq!(brain_control_word(true, None), 1);
    assert_eq!(brain_control_word(false, Some(0)), 2);
    assert_eq!(brain_control_word(true, Some(7)), 17);
}

#[test]
fn fast_state_layout_excludes_counter_and_uses_97_slot_stride() {
    assert_eq!(L1_NEURONS, 96);
    assert_eq!(CFC_SLOTS_PER_BEING, 97);
    assert_eq!(FAST_CFC_STATE_BYTES, 96 * 4);
    assert_eq!(
        fast_state_copy_range(3, 0, FAST_CFC_STATE_BYTES, 0).unwrap(),
        (0, 384)
    );
    assert_eq!(
        fast_state_copy_range(3, 1, FAST_CFC_STATE_BYTES, 0).unwrap(),
        (97 * 4, 384)
    );
    assert_eq!(
        fast_state_copy_range(3, 2, 768, 384).unwrap(),
        (2 * 97 * 4, 384)
    );
}

#[test]
fn fast_state_copy_rejects_out_of_bounds_requests() {
    assert!(fast_state_copy_range(2, 2, 384, 0).is_err());
    assert!(fast_state_copy_range(2, 0, 383, 0).is_err());
    assert!(fast_state_copy_range(2, 0, 384, 2).is_err());
    assert!(fast_state_copy_range(2, 0, 384, u64::MAX - 3).is_err());
}

#[test]
#[ignore = "requires a real Vulkan GPU; no CPU numerical oracle"]
fn linear_wave_uses_resident_signed_sensors_and_service_outputs() {
    use std::sync::Arc;
    use wgpu::util::DeviceExt;
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {backends:wgpu::Backends::VULKAN,..Default::default()});
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference:wgpu::PowerPreference::HighPerformance,compatible_surface:None,force_fallback_adapter:false
    })).expect("real GPU required");
    assert_ne!(adapter.get_info().device_type,wgpu::DeviceType::Cpu);
    let (device,queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label:Some("linear-wave-fixture"),required_features:wgpu::Features::empty(),
        required_limits:adapter.limits(),memory_hints:wgpu::MemoryHints::MemoryUsage
    },None)).unwrap();
    let device=Arc::new(device); let queue=Arc::new(queue);
    let mut brain=super::BrainGpu::new(device.clone(),queue.clone(),adapter.get_info().name,2);
    brain.linear_wave=true;
    let mut weights=vec![0.0f32;2*96*192];
    // Analytic target: tanh(+/-1), with the same matrix row on both beings.
    weights[6*192]=1.0;
    weights[96*192+6*192]=1.0;
    queue.write_buffer(&brain.buf_l1w,0,bytemuck::cast_slice(&weights));
    let mut words=[0u32;96]; words[0]=127; words[48]=129; // signed +127 / -127
    let percept=device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:Some("resident-perception"),contents:bytemuck::cast_slice(&words),usage:wgpu::BufferUsages::STORAGE
    });
    let mut encoder=device.create_command_encoder(&Default::default());
    let size=brain.bind_and_dispatch(&mut encoder,2,&[],1.0/64.0,Some(&percept));
    encoder.copy_buffer_to_buffer(&brain.buf_intent,0,&brain.buf_readback,0,size);
    queue.submit(Some(encoder.finish()));
    let result=brain.read_intent_after_submit(2);
    assert!((result[0][6]-0.7615942).abs()<0.00002);
    assert!((result[1][6]+0.7615942).abs()<0.00002);
    for row in result { for (i,v) in row.iter().enumerate() { if i!=6 {assert_eq!(*v,0.0);} } }
}
