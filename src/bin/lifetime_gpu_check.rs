//! Small GPU-only fixture for the resident lifetime state.
//!
//! This binary deliberately has no CPU forward oracle.  It checks only contracts
//! observable from the real adapter: finite/bounded state, plasticity response,
//! reset isolation, the disabled control arm, and complete motor ablation.

use std::sync::Arc;

use universe::config::iara::brain::SENSORY_WAVE_SIZE;
use universe::engine::SovereignEngine;
use universe::gpu::brain::{AblationPlan, BrainGpu};
use universe::gpu::evolution::Genome;
use universe::gpu::lifetime::{LIFE_STATE_E_OFFSET, LIFE_STATE_P_OFFSET, LIFETIME_STRIDE};

const N: usize = 2;
const DT: f32 = 1.0 / 64.0;

fn main() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let Some(adapter) =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
    else {
        println!(r#"{{"status":"not_measured","reason":"no_adapter"}}"#);
        std::process::exit(2);
    };

    let info = adapter.get_info();
    let name_lower = info.name.to_ascii_lowercase();
    if info.device_type == wgpu::DeviceType::Cpu
        || name_lower.contains("lavapipe")
        || name_lower.contains("llvmpipe")
    {
        println!(
            r#"{{"status":"not_measured","reason":"unsupported_adapter","adapter":{}}}"#,
            serde_json::to_string(&info.name).unwrap()
        );
        std::process::exit(2);
    }

    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffers_per_shader_stage = 16;
    let Ok((raw_device, raw_queue)) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("lifetime-gpu-check"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    )) else {
        println!(r#"{{"status":"not_measured","reason":"request_device_failed"}}"#);
        std::process::exit(2);
    };
    let device = Arc::new(raw_device);
    let queue = Arc::new(raw_queue);
    let mut brain = BrainGpu::new(
        Arc::clone(&device),
        Arc::clone(&queue),
        info.name.clone(),
        N,
    );

    let genomes: Vec<_> = [0x51_u64, 0xA7_u64]
        .into_iter()
        .map(Genome::random)
        .collect();
    let mut engines: Vec<SovereignEngine> = (0..N).map(|_| SovereignEngine::new()).collect();
    for (genome, engine) in genomes.iter().zip(&mut engines) {
        genome.apply_to_engine(engine);
    }
    let engine_refs: Vec<_> = engines.iter().collect();
    brain.upload_weights(&engine_refs);
    let life: Vec<_> = genomes.iter().map(|genome| genome.lifetime).collect();

    let waves = |tick: usize| -> Vec<i32> {
        (0..N * SENSORY_WAVE_SIZE)
            .map(|i| (((i * 13 + tick * 7) % 127) as i32) - 63)
            .collect()
    };

    // Disabled lifetime mechanisms are the control arm.  Uploading non-neutral
    // genome data must not change this arm after the recurrent state is reset.
    brain.upload_lifetime(&life, 0);
    brain.reset_cfc(0);
    brain.reset_cfc(1);
    let control_before = dispatch(&mut brain, &device, &queue, &waves(0), 0);
    let mut mutated_life = life.clone();
    mutated_life[0].0[0] = (mutated_life[0].0[0] + 0.01).min(1.0);
    brain.upload_lifetime(&mutated_life, 0);
    brain.reset_cfc(0);
    brain.reset_cfc(1);
    let control_after = dispatch(&mut brain, &device, &queue, &waves(0), 0);
    let control_identity = max_intent_delta(&control_before.0, &control_after.0) <= 1e-6;

    brain.upload_lifetime(&life, 1);
    brain.reset_cfc(0);
    brain.reset_cfc(1);
    brain.set_lifetime_feedback(&[[1.0, 1.0, 1.0, 0.0]; N]);
    let mut learned = None;
    for tick in 0..64 {
        let sample = dispatch(&mut brain, &device, &queue, &waves(tick), tick as u32);
        learned = Some(sample);
    }
    let learned = learned.expect("learning loop has one sample");
    let p_bound = life
        .iter()
        .map(|genes| genes.0[2].clamp(0.001, 1.0))
        .fold(0.0, f32::max);
    let finite_bounded = learned.1.iter().all(|value| value.is_finite())
        && max_abs_range_slot(&learned.1, 0, LIFE_STATE_P_OFFSET, LIFE_STATE_E_OFFSET)
            <= p_bound + 1e-5
        && max_abs_range_slot(&learned.1, 1, LIFE_STATE_P_OFFSET, LIFE_STATE_E_OFFSET)
            <= p_bound + 1e-5;
    let p_nonzero = max_abs_range(&learned.1, LIFE_STATE_P_OFFSET, LIFE_STATE_E_OFFSET) > 1e-8;
    let e_nonzero = max_abs_range(&learned.1, LIFE_STATE_E_OFFSET, 324) > 1e-8;

    brain.upload_lifetime(&life, 1);
    brain.reset_cfc(0);
    brain.reset_cfc(1);
    brain.set_lifetime_feedback(&[[0.0, 1.0, 1.0, 0.0]; N]);
    let mut zero_reward = None;
    for tick in 0..64 {
        zero_reward = Some(dispatch(
            &mut brain,
            &device,
            &queue,
            &waves(tick),
            tick as u32,
        ));
    }
    let zero_reward = zero_reward.unwrap();
    let zero_p = max_abs_range(&zero_reward.1, LIFE_STATE_P_OFFSET, LIFE_STATE_E_OFFSET) <= 1e-8;

    // Read both slots immediately after resetting only slot 0.  The copy is
    // submitted after the queue write, so it observes the ordered reset.
    brain.upload_lifetime(&life, 1);
    brain.set_lifetime_feedback(&[[1.0, 1.0, 1.0, 0.0]; N]);
    brain.reset_cfc(0);
    brain.reset_cfc(1);
    for tick in 0..64 {
        let _ = dispatch(&mut brain, &device, &queue, &waves(tick), tick as u32);
    }
    let reset_before = read_lifetime_pair(&brain, &device, &queue);
    brain.reset_lifetime(0);
    let reset_pair = read_lifetime_pair(&brain, &device, &queue);
    let reset_one = max_abs_range_slot(&reset_pair, 0, LIFE_STATE_P_OFFSET, 324) <= 1e-8;
    let other_preserved =
        max_abs_delta_slot(&reset_before, &reset_pair, 1, LIFE_STATE_P_OFFSET, 324) <= 1e-8
            && max_abs_range_slot(&reset_before, 1, LIFE_STATE_P_OFFSET, 324) > 1e-8;

    // Ablating all L2 outputs must force the motor intent to zero even when
    // lifetime plasticity is enabled.
    brain.upload_lifetime(&life, 1);
    brain.reset_cfc(0);
    brain.reset_cfc(1);
    let targets = (0..48)
        .map(|i| format!("l2:{i}"))
        .collect::<Vec<_>>()
        .join(",");
    let plan = AblationPlan::parse(&targets, 0, 0, N).expect("valid complete L2 ablation");
    brain.set_ablation(&plan, N);
    brain.set_lifetime_feedback(&[[1.0, 1.0, 1.0, 0.0]; N]);
    let ablated = dispatch(&mut brain, &device, &queue, &waves(0), 0);
    let motor_ablation_zero = ablated.0[0][..3].iter().all(|value| value.abs() <= 1e-7);

    brain.clear_ablation();
    brain.upload_lifetime(&life, 2);
    for slot in 0..N {
        brain.reset_cfc(slot);
    }
    brain.set_lifetime_feedback(&[[0.0, 1.0, 1.0, 0.0]; N]);
    let juvenile = dispatch(&mut brain, &device, &queue, &waves(0), 100_000);
    let newborn_age = juvenile.1[375] < 0.01;
    let juvenile_mask = juvenile.0[0][13..48].iter().all(|v| *v == 0.0);
    let adult = dispatch(&mut brain, &device, &queue, &waves(1), 100_700);
    let mature_age = adult.1[375] == 1.0;
    let mut pruned = life.clone();
    for genes in &mut pruned {
        genes.0[7] = 0.0;
        genes.0[11] = 0.0;
    }
    brain.upload_lifetime(&pruned, 4);
    for slot in 0..N {
        brain.reset_cfc(slot);
    }
    brain.set_lifetime_feedback(&[[0.0, 1.0, 1.0, 0.0]; N]);
    let silent = dispatch(&mut brain, &device, &queue, &waves(0), 0);
    let pruned_silent = silent.0.iter().flatten().all(|v| *v == 0.0);
    println!(
        "{}",
        serde_json::json!({"newborn_age":newborn_age,"juvenile_mask":juvenile_mask,
        "mature_age":mature_age,"pruned_silent":pruned_silent})
    );
    let physics = universe::gpu::physics::PhysicsGpu::new_shared(
        Arc::clone(&device),
        Arc::clone(&queue),
        info.name.clone(),
        &vec![1u8; 64 * 64],
        &vec![0.0f32; 64 * 64],
        64,
        3.0,
        2,
    )
    .expect("shared real GPU physics");
    let base = universe::gpu::physics::make_physics_state([90.0, 0.5, 90.0], [1.0, 0.0, 0.0]);
    let mut bodies = vec![base, base];
    bodies[0]._pad_before_wind = -0.25;
    bodies[1]._pad_before_wind = 0.25;
    for body in &mut bodies {
        body.config_flags |= 1 << 7;
        body.throttle = 0.5;
    }
    for _ in 0..32 {
        assert!(physics.step(&mut bodies));
    }
    let thermal_order = bodies[1].engine_temp < bodies[0].engine_temp;
    println!(
        "{}",
        serde_json::json!({"thermal_order":thermal_order,
        "weak_cooling_c":bodies[0].engine_temp,"strong_cooling_c":bodies[1].engine_temp})
    );
    let all_ok = newborn_age
        && juvenile_mask
        && mature_age
        && pruned_silent
        && thermal_order
        && control_identity
        && finite_bounded
        && p_nonzero
        && e_nonzero
        && zero_p
        && reset_one
        && other_preserved
        && motor_ablation_zero;
    println!(
        r#"{{"status":"{}","adapter":{},"checks":{{"control_identity":{},"finite_bounded":{},"plasticity_p_nonzero":{},"plasticity_e_nonzero":{},"zero_reward_p_zero":{},"reset_one":{},"reset_other_preserved":{},"motor_ablation_zero":{}}}}}"#,
        if all_ok { "pass" } else { "fail" },
        serde_json::to_string(&info.name).unwrap(),
        control_identity,
        finite_bounded,
        p_nonzero,
        e_nonzero,
        zero_p,
        reset_one,
        other_preserved,
        motor_ablation_zero,
    );
    if !all_ok {
        std::process::exit(1);
    }
}

fn dispatch(
    brain: &mut BrainGpu,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    waves: &[i32],
    tick: u32,
) -> (Vec<[f32; 96]>, Vec<f32>) {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lifetime-check-staging"),
        size: (N * LIFETIME_STRIDE * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    brain.set_world_tick(tick);
    brain.bind_and_dispatch(&mut enc, N, waves, DT, None);
    for being in 0..N {
        brain
            .copy_lifetime_state_into(
                &mut enc,
                &staging,
                being,
                (being * LIFETIME_STRIDE * 4) as u64,
            )
            .expect("lifetime staging range");
    }
    queue.submit(Some(enc.finish()));
    let intents = brain.read_intent_after_submit(N);
    let state = map_f32(&staging, device);
    (intents, state)
}

fn read_lifetime_pair(brain: &BrainGpu, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<f32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lifetime-reset-staging"),
        size: (N * LIFETIME_STRIDE * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    for being in 0..N {
        brain
            .copy_lifetime_state_into(
                &mut enc,
                &staging,
                being,
                (being * LIFETIME_STRIDE * 4) as u64,
            )
            .unwrap();
    }
    queue.submit(Some(enc.finish()));
    map_f32(&staging, device)
}

fn map_f32(buffer: &wgpu::Buffer, device: &wgpu::Device) -> Vec<f32> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .expect("mapping callback")
        .expect("mapping succeeds");
    let values = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
    buffer.unmap();
    values
}

fn max_abs_range(values: &[f32], start: usize, end: usize) -> f32 {
    values[start..end]
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f32::max)
}

fn max_abs_range_slot(values: &[f32], slot: usize, start: usize, end: usize) -> f32 {
    let base = slot * LIFETIME_STRIDE;
    max_abs_range(values, base + start, base + end)
}

fn max_abs_delta_slot(before: &[f32], after: &[f32], slot: usize, start: usize, end: usize) -> f32 {
    let base = slot * LIFETIME_STRIDE;
    before[base + start..base + end]
        .iter()
        .zip(&after[base + start..base + end])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn max_intent_delta(before: &[[f32; 96]], after: &[[f32; 96]]) -> f32 {
    before
        .iter()
        .zip(after)
        .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
        .fold(0.0, f32::max)
}
