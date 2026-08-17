//! GPU simulation, with no graphics.
//!
//! # The architecture the owner asked for
//!
//! > *"I need it to run on the GPU with no graphics at all, because that was supposed to
//! > be in the foundation of the project: ways to visualize with Bevy and without, and the
//! > one without is not supposed to render Bevy — Bevy would be just a frontend that shows
//! > in a graphics engine what is happening"* (translated from the original request)
//!
//! In other words: the core is compute, and Bevy is an optional window over it. This module
//! is the start of that core — it talks to the GPU through `wgpu` (already in the tree via
//! Bevy) **with no surface, no swapchain and no window**, so it runs under
//! `env -u DISPLAY`, in cron, on a machine with no graphical session.
//!
//! # Why the retina came first
//!
//! Measured over 2440 PERF samples from that day's races:
//!
//! | component | share of the time |
//! |---|---|
//! | **retina** | **57.0%** |
//! | physics and the rest | 26.9% |
//! | brain | 14.3% |
//! | perception | 1.9% |
//!
//! The retina is 14 rows × 126 columns × N beings — with 60 beings, **105,840 DDA marches
//! per tick**, all independent. It is the most parallel work in the simulator and the most
//! expensive. The brain, which is what the thesis studies, costs a quarter of that.
//!
//! # The non-negotiable rule
//!
//! The kernel has to give the SAME result as the CPU, pixel by pixel. A GPU path that
//! silently answers differently would invalidate every experiment already measured — and
//! after a full day of uncovering constants that carried dead assumptions, that is exactly
//! the class of error that cannot get in. The original note here pointed at
//! `testes::gpu_igual_a_cpu`; that test no longer exists in the tree (checked 2026-08-17,
//! no match anywhere under `src/` or `tests/`). The equivalence check that does exist is
//! the `src/bin/gpu_validate.rs` binary — run it before and after touching this module.

// ── GPU simulation core, written in parallel by agents ────────────────────────
// Each module brings a WGSL kernel + its wgpu host. The rule is the same in all of them:
// equivalence with the CPU demonstrated before any claim of speed-up.
pub mod physics;
pub mod brain;
pub mod perception;
pub mod core;
pub mod evolution;
pub mod cortex_sheet;
pub mod slow_brain;
pub mod retina_types;
pub mod retina_digest;
pub mod genome_io;

use std::borrow::Cow;

use bevy::math::{Vec2, Vec3};
use wgpu::util::DeviceExt;

use crate::config::iara::vision::{RETINA_COLS, RETINA_ROWS, RETINA_TOTAL};
use crate::world::{VoxelType, VoxelWorld};

/// Bevy resource that keeps the GPU context alive between ticks.
///
/// Recreating the device and recompiling the shader once per tick would cancel out the
/// whole gain, so this lives in the application. The world signature is what makes a track
/// change detectable, so the grid can be re-uploaded — if that slipped through, the beings
/// would see the PREVIOUS track and nothing would flag it.
#[derive(bevy::prelude::Resource, Default)]
pub struct RetinaGpuRes {
    pub ctx: Option<RetinaGpu>,
    pub signature: u64,
}

impl RetinaGpuRes {
    /// Returns a context valid for this world, creating or recreating it when needed.
    /// `None` = no usable GPU; the caller falls back to the CPU instead of killing the race.
    pub fn get_or_create(&mut self, world: &VoxelWorld, capacity: usize) -> Option<&RetinaGpu> {
        let sig = world_signature(world);
        let needs_rebuild = self.ctx.is_none()
            || self.signature != sig
            || self.ctx.as_ref().map(|c| c.capacity < capacity).unwrap_or(true);
        if needs_rebuild {
            self.ctx = RetinaGpu::new(world, capacity.max(64));
            self.signature = sig;
            if let Some(c) = &self.ctx {
                println!("🖥  retina on the GPU: {} ({} beings)", c.adapter, capacity);
            } else {
                println!("⚠️  no usable GPU — the retina stays on the CPU");
            }
        }
        self.ctx.as_ref()
    }
}

/// Cheap and sufficient: size, lap length and a sampling of the grid.
/// Exposed as `pub` so that motor.rs can pass the same signature to the GPU perception
/// context without recomputing or duplicating the logic.
pub fn world_signature(world: &VoxelWorld) -> u64 {
    let mut h = world.size as u64 ^ ((world.lap_distance as u64) << 20);
    let step = (world.grid.len() / 512).max(1);
    for (i, v) in world.grid.iter().step_by(step).enumerate() {
        h = h.wrapping_mul(1099511628211).wrapping_add((*v as u64) ^ i as u64);
    }
    h
}

/// One retina pixel, in PHYSICAL quantities: what a lidar would hand back.
///
/// The old retina returned an `i8` produced by a hand-written salience curve that resolved
/// 4 useful levels for road and 3 for grass (it normalised by 1200 m while the ray was
/// capped at 33.3 m) and put walls at the top of the scale, so the consumer's `max`-pool
/// erased the road from any bin that contained a wall. The decision about what matters now
/// happens downstream of here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pixel {
    /// Distance to the first opaque surface, in metres.
    pub dist_m: f32,
    /// 0 = Grass, 1 = Road, 2 = Wall, 3 = Hole (same order as `VoxelType`).
    pub material: u8,
}

/// The OLD encoding, reconstructed from the physical quantities.
///
/// It exists for two reasons, both methodological: to prove the GPU kernel is equivalent to
/// the CPU pixel by pixel, and to allow comparing physical × encoded in an A/B instead of
/// assuming the new one is better. Assuming without measuring was exactly the mistake of
/// whoever wrote the 1200 m.
pub fn legacy_encoding(px: Pixel) -> i8 {
    let d = px.dist_m;
    let depth_fade = (d / 1200.0).clamp(0.0, 1.0);
    let v = match px.material {
        2 => {
            if d < 30.0 { 127.0 - ((d / 30.0).clamp(0.0, 1.0).sqrt() * 72.0) }
            else if d < 120.0 { 55.0 - (((d - 30.0) / 90.0).clamp(0.0, 1.0) * 55.0) }
            else if d < 1200.0 { 0.0 - (((d - 120.0) / 480.0).clamp(0.0, 1.0) * 25.0) }
            else { -25.0 - (((d - 1200.0) / 1200.0).clamp(0.0, 1.0) * 25.0) }
        }
        0 => 40.0 - (depth_fade * 40.0),
        1 => -128.0 + (depth_fade * 96.0),
        _ => 90.0 - ((d / 50.0).clamp(0.0, 1.0) * 30.0),
    };
    v.clamp(-128.0, 127.0) as i8
}

/// A being's pose at the instant of the dispatch: where it is and where it is looking.
#[derive(Clone, Copy, Debug)]
pub struct Pose {
    pub pos: Vec3,
    pub forward: Vec2,
    pub right: Vec2,
    pub gaze_yaw: f32,
}

// Deduplicated (E3): this was the fifth identical copy of `RetinaParamsGpu`/`PoseRaw`
// (the other four were in gpu_evolve.rs, gpu_loop.rs, cortex_evolve.rs and — earlier
// still — inside this very file under different names). Local alias so that every
// `ParamsGpu { .. }` literal below did not have to be rewritten.
use crate::gpu::retina_types::{RetinaParamsGpu as ParamsGpu, PoseRaw};

/// Persistent compute context. Building this costs (opens a device, compiles a shader), so
/// it lives for the whole race — recreating it per tick would cancel out the gain.
pub struct RetinaGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    buf_grid: wgpu::Buffer,
    grid_size: u32,
    cell_size: f32,
    pub capacity: usize,
    buf_poses: wgpu::Buffer,
    buf_output: wgpu::Buffer,
    buf_readback: wgpu::Buffer,
    pub adapter: String,
}

impl RetinaGpu {
    /// Opens the GPU and uploads the world (static per track — uploaded once, not per tick).
    ///
    /// Returns `None` when there is no usable GPU, so the caller can fall back to the CPU
    /// without killing the simulation. A run that lasts days cannot die for want of a device.
    pub fn new(world: &VoxelWorld, capacity: usize) -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..Default::default()
        });
        // `compatible_surface: None` is the point: we ask for a COMPUTE adapter, no window.
        let adapter_obj = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;
        let name = adapter_obj.get_info().name.clone();
        // perception.wgsl uses 6 storage buffers — above the downlevel default limit (4).
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_storage_buffers_per_shader_stage = 8;
        let (device, queue) = pollster::block_on(adapter_obj.request_device(&wgpu::DeviceDescriptor {
            label: Some("iara-compute"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
        }, None))
        .ok()?;

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("retina"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("retina.wgsl"))),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("retina-layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
            ],
        });
        let pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("retina-pipeline"),
            layout: Some(&pipe_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let grid_u32: Vec<u32> = world.grid.iter().map(|v| *v as u32).collect();
        let buf_grid = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("grid"),
            contents: bytemuck::cast_slice(&grid_u32),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bytes_poses = (capacity * std::mem::size_of::<PoseRaw>()) as u64;
        let bytes_output = (capacity * RETINA_TOTAL * 4) as u64;
        let buf_poses = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("poses"),
            size: bytes_poses,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: bytes_output,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes_output,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Some(Self {
            device,
            queue,
            pipeline,
            layout,
            buf_grid,
            grid_size: world.size as u32,
            cell_size: world.cell_size,
            capacity,
            buf_poses,
            buf_output,
            buf_readback,
            adapter: name,
        })
    }

    /// One retina per being, in a single dispatch. Returns PHYSICS: distance and material.
    pub fn render(&self, poses_in: &[Pose]) -> Vec<Vec<Pixel>> {
        let n = poses_in.len().min(self.capacity);
        if n == 0 {
            return Vec::new();
        }

        let mut pitches = [[0.0f32; 4]; 4];
        for row in 0..RETINA_ROWS {
            pitches[row / 4][row % 4] = crate::visao::pitch(row);
        }
        let params = ParamsGpu {
            grid_size: self.grid_size,
            n_beings: n as u32,
            cell_size: self.cell_size,
            fov_half: (crate::visao::fov_deg() / 2.0).to_radians(),
            pitches,
        };
        let buf_params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let pose_data: Vec<PoseRaw> = poses_in[..n]
            .iter()
            .map(|p| PoseRaw {
                pos_gaze: [p.pos.x, p.pos.y, p.pos.z, p.gaze_yaw],
                dirs: [p.forward.x, p.forward.y, p.right.x, p.right.y],
            })
            .collect();
        self.queue.write_buffer(&self.buf_poses, 0, bytemuck::cast_slice(&pose_data));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_grid.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_poses.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_output.as_entire_binding() },
            ],
        });

        let bytes = (n * RETINA_TOTAL * 4) as u64;
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // 64 threads per group: 2 waves of 32 on RDNA2, which is this GPU's wave size.
            let total = (n * RETINA_TOTAL) as u32;
            pass.dispatch_workgroups(total.div_ceil(64), 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.buf_output, 0, &self.buf_readback, 0, bytes);
        self.queue.submit(Some(enc.finish()));

        let slice = self.buf_readback.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        let vals: &[u32] = bytemuck::cast_slice(&data);
        let mut out = Vec::with_capacity(n);
        for s in 0..n {
            let mut r = Vec::with_capacity(RETINA_TOTAL);
            for p in 0..RETINA_TOTAL {
                let raw = vals[s * RETINA_TOTAL + p];
                r.push(Pixel {
                    dist_m: (raw >> 8) as f32 / 1000.0,
                    material: (raw & 0xFF) as u8,
                });
            }
            out.push(r);
        }
        drop(data);
        self.buf_readback.unmap();
        out
    }

    /// The retina's output buffer, so another kernel can consume it DIRECTLY in VRAM.
    ///
    /// This is the whole point of porting perception: without it, the 1764 pixels per being
    /// would travel back to the CPU only to be reduced to 223 channels and uploaded again.
    /// (223 was the channel count when this was written; `SENSORY_WAVE_SIZE` is computed in
    /// `config.rs:309` and is 192 today — read the constant, not this number.)
    pub fn output_buffer(&self) -> &wgpu::Buffer { &self.buf_output }

    /// The device and the queue, for the other kernels to SHARE instead of opening another.
    /// Opening a second VkDevice on the same GPU wastes memory and some drivers refuse it.
    pub fn device(&self) -> &wgpu::Device { &self.device }
    pub fn queue(&self) -> &wgpu::Queue { &self.queue }

    /// The retina translated into the old encoding — only for the equivalence proof and for
    /// the A/B against the physical encoding. Do not use it on a new code path.
    pub fn render_legacy(&self, poses_in: &[Pose]) -> Vec<Vec<i8>> {
        self.render(poses_in).into_iter()
            .map(|r| r.into_iter().map(legacy_encoding).collect())
            .collect()
    }

    /// Records the retina dispatch into an external `CommandEncoder` WITHOUT submitting or
    /// polling.
    ///
    /// # Why it exists
    ///
    /// `render_legacy()` created its own encoder + submit + poll(Wait), adding a
    /// synchronisation barrier. With `dispatch_into`, the retina and perception can be
    /// chained into the same encoder and submitted once.
    ///
    /// The result lands in `self.buf_output` (VRAM) and in `self.buf_readback` (via a copy
    /// in the same encoder), but the map can only happen AFTER the caller's submit + poll.
    ///
    /// # Expected flow
    ///
    /// ```text
    /// 1. Write poses via queue.write_buffer() BEFORE calling this method
    /// 2. dispatch_into(&mut enc, n, device)      ← retina dispatch
    /// 3. percept.bind_and_dispatch(&mut enc, ...) ← perception dispatch
    /// 4. queue.submit(enc.finish())              ← ONE submit
    /// 5. device.poll(Wait)                       ← ONE block
    /// 6. read_legacy_after_submit(n)             ← retina result
    /// 7. percept.read_after_submit(n, bytes_out) ← perception result
    /// ```
    ///
    /// Returns the N beings dispatched (so the caller can call
    /// `read_legacy_after_submit(N)`).
    pub fn dispatch_into(
        &self,
        enc:      &mut wgpu::CommandEncoder,
        poses_in: &[Pose],
    ) -> usize {
        let n = poses_in.len().min(self.capacity);
        if n == 0 { return 0; }

        // Prepare uniform + poses (write_buffer does not block — the data sits in the queue
        // until submit).
        let mut pitches = [[0.0f32; 4]; 4];
        for row in 0..RETINA_ROWS {
            pitches[row / 4][row % 4] = crate::visao::pitch(row);
        }
        let params = ParamsGpu {
            grid_size: self.grid_size,
            n_beings: n as u32,
            cell_size: self.cell_size,
            fov_half: (crate::visao::fov_deg() / 2.0).to_radians(),
            pitches,
        };
        let buf_params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("retina-params-chain"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let pose_data: Vec<PoseRaw> = poses_in[..n]
            .iter()
            .map(|p| PoseRaw {
                pos_gaze: [p.pos.x, p.pos.y, p.pos.z, p.gaze_yaw],
                dirs: [p.forward.x, p.forward.y, p.right.x, p.right.y],
            })
            .collect();
        // write_buffer: CPU→GPU staging, no blocking until submit.
        self.queue.write_buffer(&self.buf_poses, 0, bytemuck::cast_slice(&pose_data));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf_grid.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buf_poses.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buf_output.as_entire_binding() },
            ],
        });

        let bytes = (n * RETINA_TOTAL * 4) as u64;
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // 64 threads per group: 2 waves of 32 on RDNA2 (this GPU's wave size).
            let total = (n * RETINA_TOTAL) as u32;
            pass.dispatch_workgroups(total.div_ceil(64), 1, 1);
        }
        // Copy into readback in the same encoder — DMA with no round trip through the CPU.
        enc.copy_buffer_to_buffer(&self.buf_output, 0, &self.buf_readback, 0, bytes);

        n
    }

    /// Reads the retina's buf_readback and decodes it into legacy [i8; RETINA_TOTAL] × N.
    ///
    /// MUST be called after the `device.poll(Wait)` of the encoder that held `dispatch_into`.
    /// Without that poll the buffer is not available for mapping and this method will panic.
    pub fn read_legacy_after_submit(&self, n: usize) -> Vec<[i8; RETINA_TOTAL]> {
        if n == 0 { return Vec::new(); }
        let bytes = (n * RETINA_TOTAL * 4) as u64;
        let slice = self.buf_readback.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        // The caller already polled with Wait — the recv() should complete immediately.
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        let vals: &[u32] = bytemuck::cast_slice(&data);
        let mut out = vec![[0i8; RETINA_TOTAL]; n];
        for s in 0..n {
            for p in 0..RETINA_TOTAL {
                let raw = vals[s * RETINA_TOTAL + p];
                let px = Pixel {
                    dist_m: (raw >> 8) as f32 / 1000.0,
                    material: (raw & 0xFF) as u8,
                };
                out[s][p] = legacy_encoding(px);
            }
        }
        drop(data);
        self.buf_readback.unmap();
        out
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Only so the equivalence test does not have to depend on an internal detail.
pub const COLS: usize = RETINA_COLS;

/// Keeps the `VoxelType` import used even when no code path in this file names the type.
#[allow(dead_code)]
fn _types_used(_: VoxelType) {}
