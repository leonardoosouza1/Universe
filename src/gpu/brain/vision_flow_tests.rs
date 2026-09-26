//! Real world pixels -> spatial/temporal perception -> native brain -> action.
//! Geometry is deterministic; no CPU ray caster or CPU neural implementation.
use super::*;
use crate::config::iara::vision::{RETINA_COLS, RETINA_TOTAL};
use crate::gpu::{
    core::{storage_entry, uniform_entry},
    perception::{PerceptionGpu, PosePerception},
    retina_types::{PoseRaw, RetinaParamsGpu},
};

#[test]
#[ignore = "real GPU only; run scripts/check_native_flow_gpu.py"]
fn retina_perception_native_action_contract() {
    run_retina_contract(1, 0.0, false);
}

#[test]
#[ignore = "real GPU only; run scripts/check_native_flow_gpu.py"]
fn centered_legacy_retina_preserves_sampling_and_matches_wall_faces() {
    // Isolate grid registration from the new road-view pitch/range profile.
    // The legacy distance encoding is horizontal wall range, not slant range.
    // An oblique pitch distinguishes this mode from physical road-view optics.
    run_retina_contract(2, -0.4, false);
    run_retina_contract(2, -0.4, true);
}

fn run_retina_contract(geometry_mode: u32, pitch: f32, rotate_wall: bool) {
    let mut f = Fixture::new();
    let mut world = crate::world::VoxelWorld::new(17, 28.0);
    world.grid.fill(crate::world::VoxelType::Road);
    // One vertical wall centred at x=120 m (cell40). Near observer x=114 m
    // sees the wall face at118.5 m: centre ray length approximately4.5 m.
    assert_eq!(world.cell_size, 3.0);
    for z in 0..world.size {
        let index = if rotate_wall {
            40 * world.size + z
        } else {
            z * world.size + 40
        };
        world.grid[index] = crate::world::VoxelType::Wall;
    }
    let poses: Vec<_> = (0..N)
        .map(|b| {
            let mirrored = geometry_mode == 2 && b >= 2;
            let x = if mirrored {
                if b % 2 == 0 { 126.0 } else { 180.0 }
            } else if b % 2 == 0 {
                114.0
            } else {
                60.0
            };
            let z = 96.0 + b as f32 * 24.0;
            let sign = if mirrored { -1.0 } else { 1.0 };
            PoseRaw {
                pos_gaze: [
                    if rotate_wall { z } else { x },
                    if geometry_mode == 2 { 80.0 } else { 1.7 },
                    if rotate_wall { x } else { z },
                    0.0,
                ],
                dirs: if rotate_wall {
                    [0.0, sign, sign, 0.0]
                } else {
                    [sign, 0.0, 0.0, -sign]
                },
            }
        })
        .collect();
    let params = RetinaParamsGpu {
        grid_size: world.size as u32,
        n_beings: N as u32,
        cell_size: world.cell_size,
        fov_half: 1.0,
        pitches: [[pitch; 4]; 4],
        // Force the old registration only in the gate's negative control.
        beings_on: 0,
        _pad0: if geometry_mode == 2
            && std::env::var("UNIVERSE_CONTRACT_FAULT").as_deref() == Ok("corner-retina-grid")
        {
            0
        } else {
            geometry_mode
        },
        _pad1: 0,
        _pad2: 0,
    };
    let grid: Vec<u32> = world.grid.iter().map(|v| *v as u32).collect();
    let upload = |label, bytes: &[u8], usage| {
        f.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage,
            })
    };
    let uniform = upload(
        "vision-contract-params",
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::UNIFORM,
    );
    let grid = upload(
        "vision-contract-grid",
        bytemuck::cast_slice(&grid),
        wgpu::BufferUsages::STORAGE,
    );
    let pose_buffer = upload(
        "vision-contract-poses",
        bytemuck::cast_slice(&poses),
        wgpu::BufferUsages::STORAGE,
    );
    let pixels = f.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vision-contract-pixels"),
        size: (N * RETINA_TOTAL * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let layout = f
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
            ],
        });
    let pipeline_layout = f
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
    let shader = f.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("production-retina-contract"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(crate::gpu::RETINA_SHADER)),
    });
    let pipeline = f
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
    let bindings: Vec<_> = [&uniform, &grid, &pose_buffer, &pixels]
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    let group = f.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &bindings,
    });
    let percept = PerceptionGpu::new(&f.device, &f.queue, &world, N).unwrap();
    let perceived_poses: Vec<_> = poses
        .iter()
        .map(|p| PosePerception {
            pos_x: p.pos_gaze[0],
            pos_y: p.pos_gaze[1],
            pos_z: p.pos_gaze[2],
            phase: 0.0,
            fwd_x: p.dirs[0],
            fwd_z: p.dirs[1],
            rgt_x: p.dirs[2],
            rgt_z: p.dirs[3],
        })
        .collect();
    f.reset(0.0);
    let cars: Vec<_> = poses
        .iter()
        .map(|p| {
            make_physics_state(
                [p.pos_gaze[0], 0.5, p.pos_gaze[2]],
                [p.dirs[0], 0.0, p.dirs[1]],
            )
        })
        .collect();
    f.physics.upload_resident_state(&cars).unwrap();
    let staging = f.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: pixels.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = f.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((N * RETINA_TOTAL).div_ceil(64) as u32, 1, 1);
    }
    let bytes = percept.bind_and_dispatch(&mut enc, &f.device, &f.queue, &perceived_poses, &pixels);
    enc.copy_buffer_to_buffer(&pixels, 0, &staging, 0, pixels.size());
    enc.copy_buffer_to_buffer(
        percept.buf_output(),
        0,
        percept.buf_readback_ref(),
        0,
        bytes,
    );
    f.queue.submit(Some(enc.finish()));
    f.brain.set_observed_being(Some(0));
    f.step_from(1, Some(percept.buf_output()));
    let (cars, stage, intent) = f.snapshot();
    let wave = percept.read_after_submit(&f.device, N, bytes);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    f.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let mapped = staging.slice(..).get_mapped_range();
    let values: &[u32] = bytemuck::cast_slice(&mapped);
    let near = values[RETINA_COLS / 2];
    let far = values[RETINA_TOTAL + RETINA_COLS / 2];
    assert_eq!(near & 255, 2, "retina must see the known wall material");
    assert_eq!(far & 255, 2);
    assert!(
        ((near >> 8) as f32 / 1000.0 - 4.5).abs() < 0.05,
        "physical retina distance must use metres and the wall face: {}m",
        (near >> 8) as f32 / 1000.0
    );
    assert!(((far >> 8) as f32 / 1000.0 - 58.5).abs() < 0.05);
    if geometry_mode == 2 {
        for (being, expected) in [(2, 4.5), (3, 58.5)] {
            let pixel = values[being * RETINA_TOTAL + RETINA_COLS / 2];
            assert_eq!(pixel & 255, 2);
            assert!(
                ((pixel >> 8) as f32 / 1000.0 - expected).abs() < 0.05,
                "mirrored wall range must agree: being={being}, pixel={pixel}"
            );
        }
    }
    assert!(
        wave[0][0] > wave[1][0],
        "near wall must produce stronger pooled visual input"
    );
    assert!(
        stage.l1_matmul[1] > 0.0,
        "retina->perception->CfC->L1 connection"
    );
    assert!(
        intent[0][1] > intent[1][1],
        "native brain must preserve the controlled visual ordering"
    );
    assert!(
        cars[0].steer > cars[1].steer,
        "visual difference must reach the applied motor output"
    );
    eprintln!(
        "CONTRACT PASS: wall distances={:.3}/{:.3}m, wave={}/{}, steering={}/{}",
        (near >> 8) as f32 / 1000.0,
        (far >> 8) as f32 / 1000.0,
        wave[0][0],
        wave[1][0],
        cars[0].steer,
        cars[1].steer
    );
}
