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
pub mod brain;
// C3 of docs/PRD-CEREBRO-NEUROMOD.md — the three method rulers the Universe did not
// have: how many genes of a block are on (`dof`), how much a block moves the command
// (`contribution`), and whether a block can move it at all (`sensitivity`). All three
// are PURE and GPU-free on purpose: the sensitivity gate has to be runnable before
// anything is queued on the shared GPU.
pub mod contribution;
pub mod core;
pub mod cortex_sheet;
pub mod dof;
pub mod evolution;
// Host + kernels of the fly map (`docs/FLY-MAP-BRAIN.md`). The map itself, its file
// format and the CPU reference the kernels are compared against live in the GPU-free
// `crate::fly_map`.
pub mod fly_map;
pub mod genome_io;
pub mod hippocampus;
pub mod perception;
pub mod physics;
pub mod retina_digest;
pub mod retina_types;
pub mod screen_brain;
pub mod sensitivity;
pub mod slow_brain;
pub mod timing;
pub mod wave_record;

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
            || self
                .ctx
                .as_ref()
                .map(|c| c.capacity < capacity)
                .unwrap_or(true);
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
        h = h
            .wrapping_mul(1099511628211)
            .wrapping_add((*v as u64) ^ i as u64);
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
    /// 0 = Grass, 1 = Road, 2 = Wall, 3 = Hole (same order as `VoxelType`) — or
    /// [`BEING_MATERIAL`] (4), a code this kernel PRODUCES when `--retina-beings`
    /// intersects the ray with another being's body first. `VoxelType` (`src/world.rs`)
    /// still has only the four grid materials; `Being` is never written into `grid[]`.
    pub material: u8,
}

/// Material code the retina emits when the nearest thing the ray hits is another
/// being's body, not a grid voxel (`--retina-beings`, `docs/RETINA-BEINGS.md` §2).
///
/// Deliberately outside `VoxelType`'s 0-3 range: this is a code the RETINA produces,
/// never a voxel the world stores. Mirrored in `src/gpu/retina.wgsl`
/// (`const BEING_MATERIAL: u32 = 4u;`) and consumed by `legacy_encoding` below and by
/// `gpu/fly_map/fly_map.wgsl::eye_pixel_drive`'s CPU counterpart, `fly_map_eye.rs`.
pub const BEING_MATERIAL: u8 = 4;

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
            if d < 30.0 {
                127.0 - ((d / 30.0).clamp(0.0, 1.0).sqrt() * 72.0)
            } else if d < 120.0 {
                55.0 - (((d - 30.0) / 90.0).clamp(0.0, 1.0) * 55.0)
            } else if d < 1200.0 {
                0.0 - (((d - 120.0) / 480.0).clamp(0.0, 1.0) * 25.0)
            } else {
                -25.0 - (((d - 1200.0) / 1200.0).clamp(0.0, 1.0) * 25.0)
            }
        }
        0 => 40.0 - (depth_fade * 40.0),
        1 => -128.0 + (depth_fade * 96.0),
        3 => 90.0 - ((d / 50.0).clamp(0.0, 1.0) * 30.0),
        // Being (docs/RETINA-BEINGS.md §2): a code the RETINA produces, never a grid
        // voxel — flat -29 rather than a curve over distance, because the raw
        // `Pixel.dist_m` already carries the real distance; this function only
        // reconstructs the OLD salience-curve encoding for the legacy consumers
        // (perception.wgsl's i8 channels, the fly eye binding). -29 sits in the one gap
        // no other material's formula reaches AT ANY distance, not only at the same one:
        // Wall spans [-25, 127] (minimum at d=1200: -25.0 - clamp(0..1)*25.0 = -25.0),
        // Grass [0, 40], Hole [60, 90], Road [-128, -32] (maximum at d=1200:
        // -128.0 + 1.0*96.0 = -32.0) — so the open interval (-32, -25) is untouched by
        // all four, and -29 sits inside it.
        4 => -29.0,
        m => unreachable!(
            "legacy_encoding: material {m} is not Grass/Road/Wall/Hole/Being (0-4) — a \
             new retina material code needs its own branch here, not the old `_` default \
             (which used to silently encode anything unknown as Hole)"
        ),
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
use crate::gpu::retina_types::{PoseRaw, RetinaParamsGpu as ParamsGpu};

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
        let adapter_obj =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))?;
        let name = adapter_obj.get_info().name.clone();
        // perception.wgsl uses 6 storage buffers — above the downlevel default limit (4).
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_storage_buffers_per_shader_stage = 8;
        let (device, queue) = pollster::block_on(adapter_obj.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("iara-compute"),
                // TIMESTAMP_QUERY is requested only when the adapter offers it: asking for a
                // feature the adapter lacks makes request_device return None, and losing the
                // whole device to gain a profiler is a bad trade. `GpuTimer::new` returns None
                // when it was not granted, and the caller prints nothing rather than lying.
                required_features: adapter_obj.features() & wgpu::Features::TIMESTAMP_QUERY,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
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
    ///
    /// `--retina-beings` off: bit-for-bit the kernel this project had before
    /// `docs/RETINA-BEINGS.md` — see [`render_with_beings`] for the ON path.
    pub fn render(&self, poses_in: &[Pose]) -> Vec<Vec<Pixel>> {
        self.render_inner(poses_in, false)
    }

    /// [`render`], but with `--retina-beings` engaged: the kernel also intersects each ray
    /// with every OTHER being's body, a vertical cylinder of radius
    /// `config::iara::vision::BEING_RADIUS_M` in the (x, z) plane
    /// (`docs/RETINA-BEINGS.md` §2).
    ///
    /// Exists as a SEPARATE method, not a parameter on [`render`], so every existing
    /// caller's signature and behaviour stay untouched; `gpu_validate --retina` and the
    /// geometry tests are what call this one.
    pub fn render_with_beings(&self, poses_in: &[Pose]) -> Vec<Vec<Pixel>> {
        self.render_inner(poses_in, true)
    }

    fn render_inner(&self, poses_in: &[Pose], beings_on: bool) -> Vec<Vec<Pixel>> {
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
            beings_on: beings_on as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let buf_params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
        self.queue
            .write_buffer(&self.buf_poses, 0, bytemuck::cast_slice(&pose_data));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buf_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_poses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buf_output.as_entire_binding(),
                },
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
    pub fn output_buffer(&self) -> &wgpu::Buffer {
        &self.buf_output
    }

    /// The device and the queue, for the other kernels to SHARE instead of opening another.
    /// Opening a second VkDevice on the same GPU wastes memory and some drivers refuse it.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// The retina translated into the old encoding — only for the equivalence proof and for
    /// the A/B against the physical encoding. Do not use it on a new code path.
    pub fn render_legacy(&self, poses_in: &[Pose]) -> Vec<Vec<i8>> {
        self.render(poses_in)
            .into_iter()
            .map(|r| r.into_iter().map(legacy_encoding).collect())
            .collect()
    }

    /// [`render_legacy`] with `--retina-beings` engaged — see [`render_with_beings`]. What
    /// `gpu_validate --retina` compares against `World::render_binocular_retina`'s
    /// beings-on path.
    pub fn render_legacy_with_beings(&self, poses_in: &[Pose]) -> Vec<Vec<i8>> {
        self.render_with_beings(poses_in)
            .into_iter()
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
    pub fn dispatch_into(&self, enc: &mut wgpu::CommandEncoder, poses_in: &[Pose]) -> usize {
        let n = poses_in.len().min(self.capacity);
        if n == 0 {
            return 0;
        }

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
            // `--retina-beings` is not threaded through this chained-encoder path (used by
            // the Bevy visualizer's GPU chain) — off, same as this field's absence before
            // it existed. See `render_with_beings` for the path that does turn it on.
            beings_on: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let buf_params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
        self.queue
            .write_buffer(&self.buf_poses, 0, bytemuck::cast_slice(&pose_data));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buf_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_poses.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buf_output.as_entire_binding(),
                },
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
        if n == 0 {
            return Vec::new();
        }
        let bytes = (n * RETINA_TOTAL * 4) as u64;
        let slice = self.buf_readback.slice(0..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
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

#[cfg(test)]
mod being_tests {
    use super::*;

    /// [`BEING_MATERIAL`] must encode to a value none of the four grid materials ever
    /// reach, AT ANY distance — not only at the one distance a test happens to pick — so
    /// this checks the closed-form ranges directly instead of sampling a few points.
    /// `docs/RETINA-BEINGS.md` §2 only asks for "the same distance"; the -29 chosen in
    /// `legacy_encoding` clears the stronger bar for free (see the comment there).
    #[test]
    fn being_encoding_never_collides_with_a_grid_material() {
        let being = legacy_encoding(Pixel {
            dist_m: 42.0, // arbitrary — the encoding is flat over distance
            material: BEING_MATERIAL,
        });
        assert_eq!(being, -29);

        // Sample every material densely over the full 0..1200 m the kernel clamps to,
        // and assert none of them ever equals -29.
        let mut d = 0.0f32;
        while d <= 1200.0 {
            for material in [0u8, 1, 2, 3] {
                let v = legacy_encoding(Pixel {
                    dist_m: d,
                    material,
                });
                assert_ne!(
                    v, being,
                    "material {material} at {d} m encodes to {v}, colliding with Being"
                );
            }
            d += 0.5;
        }
    }

    /// A material code outside 0-4 is a bug (the retina kernel never produces one), not
    /// an environment failure — `legacy_encoding` must reject it loudly instead of
    /// silently falling back to the Hole formula, which is what the old `_ =>` arm did.
    #[test]
    #[should_panic(expected = "is not Grass/Road/Wall/Hole/Being")]
    fn legacy_encoding_rejects_an_unknown_material() {
        let _ = legacy_encoding(Pixel {
            dist_m: 10.0,
            material: 5,
        });
    }

    /// Guards the WGSL mirror the way `docs/STYLE.md` §4 asks when a shared constant
    /// cannot be `use`d across the Rust/WGSL boundary: if either literal changes without
    /// the other, this fails instead of the two silently drifting apart — the exact bug
    /// class `docs/STYLE.md` §4 documents for `WAVE_U32S`.
    #[test]
    fn being_radius_wgsl_mirror_matches_the_config_constant() {
        let expected = crate::config::iara::vision::BEING_RADIUS_M;
        assert_eq!(
            expected, 0.775,
            "update the WGSL literal below together with this constant"
        );
        let src = include_str!("retina.wgsl");
        assert!(
            src.contains("const BEING_RADIUS: f32 = 0.775;"),
            "retina.wgsl's BEING_RADIUS literal no longer matches \
             config::iara::vision::BEING_RADIUS_M = {expected}"
        );
    }
}

/// `docs/RETINA-BEINGS.md` §3.3: "the geometry is right, not just plausible" — beings at
/// known positions, on the real GPU kernel AND the CPU reference at once, so a bug only
/// one side has cannot hide behind the other looking right.
///
/// `#[ignore]`d for the same reason as `gpu/fly_map/parity_test.rs`: opens a real
/// `wgpu::Device`. Run explicitly:
/// `HSA_OVERRIDE_GFX_VERSION=10.3.0 ROCR_VISIBLE_DEVICES=0 cargo test --release --lib -- \
///   --ignored being_geometry`
#[cfg(test)]
mod being_geometry_gpu_tests {
    use super::*;

    /// `VoxelWorld::new`'s track is recentred near `(GRID_CELLS/2, GRID_CELLS/2)` in cell
    /// units, i.e. world (480, 480) m, and its widest recorded span is on the order of
    /// 330 m (`src/world.rs:1157`, "raio até ~110 cells" -- check-language: allow). A
    /// point near a grid corner —
    /// world (30, 30) m here, cell (10, 10) — is over 600 m from the track centre, well
    /// outside the Road/Wall band `VoxelWorld::new` paints
    /// (`grid[idx] = if dist_m < track_width/2 {Road} else if ... {Wall} else {Grass}`),
    /// so it is Grass in every direction for hundreds of metres: an open field to place
    /// synthetic beings in without a real track wall interfering.
    fn open_field_world() -> VoxelWorld {
        VoxelWorld::new(42, 10.0)
    }

    /// The observer: at the open-field point above, facing world +x
    /// (`forward = (1, 0)` in the (x, z) plane `render_binocular_retina` reads), gaze
    /// centred (no saccade).
    fn observer() -> Pose {
        Pose {
            pos: Vec3::new(30.0, 0.0, 30.0),
            forward: Vec2::new(1.0, 0.0),
            right: Vec2::new(0.0, 1.0),
            gaze_yaw: 0.0,
        }
    }

    fn cpu_retina(world: &VoxelWorld, eye: Pose, others: &[Vec3]) -> [i8; RETINA_TOTAL] {
        world.render_binocular_retina(
            eye.pos,
            Vec3::new(eye.forward.x, 0.0, eye.forward.y),
            Vec3::new(eye.right.x, 0.0, eye.right.y),
            eye.gaze_yaw,
            others,
        )
    }

    #[test]
    #[ignore = "opens a real GPU — run manually: cargo test --release --lib -- --ignored being_geometry"]
    fn being_directly_ahead_at_10m_appears_in_the_central_columns() {
        let Some(world_gpu) = RetinaGpu::new(&open_field_world(), 4) else {
            eprintln!("[being_geometry] no GPU available — test skipped");
            return;
        };
        let world = open_field_world();
        let eye = observer();
        let being_pos = Vec3::new(eye.pos.x + 10.0, 0.0, eye.pos.z);
        let poses = vec![
            eye,
            Pose {
                pos: being_pos,
                ..eye
            },
        ];

        let gpu_out = world_gpu.render_with_beings(&poses);
        // Row 0 is the fovea; normalized_i == 0 sits at column (RETINA_COLS - 1) / 2. A
        // small window around it, not one exact index, because the tangent warp puts the
        // true centre at a non-integer column.
        let row0 = &gpu_out[0][0..RETINA_COLS];
        let center = (RETINA_COLS - 1) / 2;
        let hit = (center - 3..=center + 3)
            .map(|c| row0[c])
            .find(|px| px.material == BEING_MATERIAL);
        let px = hit.unwrap_or_else(|| {
            panic!("no pixel in row 0's central columns saw the being 10 m ahead")
        });
        assert!(
            (px.dist_m - 10.0).abs() <= 3.0, // one cell (cell_size = 3.0 m)
            "GPU distance to the being = {} m, expected 10 m +/- one cell",
            px.dist_m
        );

        let cpu = cpu_retina(&world, eye, &[being_pos]);
        assert!(
            (center - 3..=center + 3).any(|c| cpu[c] == -29),
            "CPU reference saw no Being pixel in row 0's central columns either"
        );
    }

    #[test]
    #[ignore = "opens a real GPU — run manually: cargo test --release --lib -- --ignored being_geometry"]
    fn being_at_90_degrees_is_outside_the_120_degree_field_of_view() {
        let Some(world_gpu) = RetinaGpu::new(&open_field_world(), 4) else {
            eprintln!("[being_geometry] no GPU available — test skipped");
            return;
        };
        let world = open_field_world();
        let eye = observer();
        // Exactly along `right` — 90 deg off `forward` — 10 m out. The field of view is
        // 120 deg total (+/- 60 deg half), so no ray can point within 30 deg of this
        // bearing; at 10 m that is metres of lateral slack, far past `BEING_RADIUS_M`.
        let being_pos = Vec3::new(eye.pos.x, 0.0, eye.pos.z + 10.0);
        let poses = vec![
            eye,
            Pose {
                pos: being_pos,
                ..eye
            },
        ];

        let gpu_out = world_gpu.render_with_beings(&poses);
        assert!(
            gpu_out[0].iter().all(|px| px.material != BEING_MATERIAL),
            "a being at 90 deg, outside the 120 deg FOV, showed up in the GPU retina"
        );

        let cpu = cpu_retina(&world, eye, &[being_pos]);
        assert!(
            cpu.iter().all(|&v| v != -29),
            "a being at 90 deg, outside the 120 deg FOV, showed up in the CPU retina"
        );
    }

    #[test]
    #[ignore = "opens a real GPU — run manually: cargo test --release --lib -- --ignored being_geometry"]
    fn being_behind_a_wall_stays_invisible() {
        let world = open_field_world();
        let Some(world_gpu) = RetinaGpu::new(&world, 4) else {
            eprintln!("[being_geometry] no GPU available — test skipped");
            return;
        };
        // Near the grid's negative-x edge (`GRID_CELLS = 320`, cell_size = 3.0 m, so
        // world x = 0 is cell 0): `get_voxel_at`/`voxel_at` return Wall for any position
        // outside 0..320 cells, a deterministic "wall" independent of the procedural
        // track's own geometry (`src/world.rs::get_voxel_at`, mirrored in
        // `retina.wgsl::voxel_at`). Facing -x, the ray leaves the grid within about one
        // cell (~3 m); a being 50 m further out, on the SAME ray, sits far behind it.
        let eye = Pose {
            pos: Vec3::new(1.0, 0.0, 1.0),
            forward: Vec2::new(-1.0, 0.0),
            right: Vec2::new(0.0, 1.0),
            gaze_yaw: 0.0,
        };
        let being_pos = Vec3::new(eye.pos.x - 50.0, 0.0, eye.pos.z);
        let poses = vec![
            eye,
            Pose {
                pos: being_pos,
                ..eye
            },
        ];

        let gpu_out = world_gpu.render_with_beings(&poses);
        let row0 = &gpu_out[0][0..RETINA_COLS];
        let center = (RETINA_COLS - 1) / 2;
        for c in center - 3..=center + 3 {
            assert_ne!(
                row0[c].material, BEING_MATERIAL,
                "column {c}: a being 50 m behind the grid-edge wall was seen instead of \
                 the wall — the nearer DDA hit did not win"
            );
            assert!(
                row0[c].dist_m < 10.0,
                "column {c}: expected the near wall (~3 m), got {} m — the being's 50 m \
                 leaked through",
                row0[c].dist_m
            );
        }

        let cpu = cpu_retina(&world, eye, &[being_pos]);
        for c in center - 3..=center + 3 {
            assert_ne!(
                cpu[c], -29,
                "CPU reference: column {c} saw the being through the wall"
            );
        }
    }
}
