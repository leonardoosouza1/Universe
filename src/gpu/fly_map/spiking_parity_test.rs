//! GPU smoke/parity test for the event-driven LIF path.
//!
//! This deliberately uses a two-neuron map with one very strong synapse.  A spike is
//! injected into neuron 0, must cross the fixed delay ring, make neuron 1 fire, and
//! appear in the K3 command.  The test is ignored by default because it opens a real
//! adapter; the CPU reference tests remain part of the normal suite.

use std::sync::Arc;

use super::FlySpikingGpu;
use crate::fly_map::spiking::{EyeMode, SpikingParams, SpikingState};
use crate::fly_map::{FlyLayout, FlyMap, NONE, OUT_SLOTS, PresynapticCsr, pack_selection};

const N_TICKS: usize = 24;
const TOL: f32 = 2e-5;

fn tiny_map() -> FlyMap {
    FlyMap {
        type_of: vec![0, 0],
        in_group_of: vec![NONE, NONE],
        // post 0 has no inputs; post 1 receives source 0.
        indptr: vec![0, 0, 1],
        indices: vec![0],
        weights: vec![6000.0],
        out_indptr: vec![0, 1],
        out_members: vec![1],
        n_types: 1,
        k_in: 0,
        pre_csr: Some(PresynapticCsr {
            indptr: vec![0, 1, 1],
            indices: vec![1],
            weights: vec![6000.0],
        }),
        pixel_of_neuron: Some(vec![NONE, NONE]),
        no_refractory: None,
        // the reflex readout block (v3) is not what this test exercises: the spiking path
        // reads `out_pair` from the packed selection either way
        reflex_out: None,
    }
}

fn selection(map: &FlyMap) -> crate::fly_map::FlySelection {
    let layout = FlyLayout {
        k_in: map.k_in,
        k_out: map.k_out(),
        n_types: map.n_types,
    };
    let mut block = vec![-1.0; layout.block_len()];
    for command in 0..3 {
        block[crate::fly_map::OFF_OUT_PAIR + command * 2] = 0.0;
        block[crate::fly_map::OFF_OUT_PAIR + command * 2 + 1] = -1.0;
    }
    block[crate::fly_map::OFF_GAIN] = 1.0;
    block[crate::fly_map::OFF_SCALE] = 1.0;
    block[crate::fly_map::OFF_TYPE_MASK] = 1.0;
    pack_selection(&[block.as_slice()], layout, 1)
}

async fn open_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await?;
    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffers_per_shader_stage = 16;
    adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("fly-spiking-parity"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .ok()
}

#[test]
#[ignore = "opens a real GPU — run manually with --ignored fly_spiking_parity"]
fn fly_spiking_parity_gpu_matches_cpu_delay_and_readout() {
    let Some((device, queue)) = pollster::block_on(open_device()) else {
        eprintln!("[fly_spiking_parity] no GPU available — test skipped");
        return;
    };
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let map = tiny_map();
    let sel = selection(&map);
    let params = SpikingParams::default();
    let gpu = FlySpikingGpu::new(Arc::clone(&device), Arc::clone(&queue), &map, 1, params)
        .expect("tiny map fits");
    assert!(gpu.upload_selection(&sel, 1));
    gpu.reset_state();

    // Eye is disabled. The retina buffer still has the full binding size because WGSL
    // runtime arrays carry their declared storage range even when the branch is off.
    let retina = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fly-spiking-parity-retina"),
        size: (crate::config::iara::vision::RETINA_TOTAL * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut v = vec![params.v0; 2];
    v[0] = params.v_th + 1.0;
    assert!(gpu.debug_set_v(&v));

    let mut cpu = SpikingState::new(map.n(), 1, params);
    cpu.v[0] = params.v_th + 1.0;
    let dt_s = params.dt_ms * 1e-3;
    let dt_ms = params.dt_ms;
    let mut saw_command = false;
    for tick in 0..N_TICKS {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fly-spiking-parity-tick"),
        });
        let trace_requested = tick == 0;
        if trace_requested {
            gpu.set_debug_being(0);
        }
        gpu.dispatch(&mut enc, 1, &retina, false, EyeMode::Photoreceptor, dt_s, 0);
        if trace_requested {
            assert!(gpu.record_debug_snapshot(&mut enc, 1));
        }
        queue.submit(Some(enc.finish()));
        device.poll(wgpu::Maintain::Wait);
        let gpu_cmd = gpu.read_commands_after_submit(1);
        if trace_requested {
            let snapshot = gpu
                .read_debug_snapshot_after_submit()
                .expect("debug snapshot readback");
            assert_eq!(snapshot.v.len(), map.n());
            assert_eq!(snapshot.spike_counts.len(), map.n());
            assert!(
                snapshot.spike_counts[0] >= 1,
                "injected source spike was not traced"
            );
        }

        cpu.step(&map, &sel, 1, params, None, EyeMode::Photoreceptor, None, 0);
        let cpu_cmd = cpu.readout(&map, &sel, 1, params, dt_ms);
        for c in 0..OUT_SLOTS / 2 {
            assert!(
                (gpu_cmd[0][c] - cpu_cmd[0][c]).abs() < TOL,
                "tick {tick}, command {c}: GPU {} vs CPU {}",
                gpu_cmd[0][c],
                cpu_cmd[0][c]
            );
        }
        if cpu_cmd[0].iter().any(|v| v.abs() > 1e-6) {
            saw_command = true;
        }
    }
    assert!(
        saw_command,
        "the injected spike never reached the output group; parity was vacuous"
    );
}
