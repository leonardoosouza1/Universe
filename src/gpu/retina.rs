//! Standalone retinal compute renderer and explicit readback API.
//! The integrated evolution loop records its resident retina on the shared device.
use super::core::{storage_entry, uniform_entry};
use super::retina_encoding::{legacy_encoding, Pixel, Pose};
use crate::config::iara::vision::{RETINA_COLS, RETINA_ROWS, RETINA_TOTAL};
use crate::world::VoxelWorld;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

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
        limits.max_storage_buffers_per_shader_stage = 16;
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
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(crate::gpu::RETINA_SHADER)),
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

    /// Explicit optics for numerical fixtures; does not alter process-wide parameters.
    pub fn render_road_view(&self, poses: &[Pose], range_m: f32) -> Vec<Vec<Pixel>> {
        assert!(range_m.is_finite() && (20.0..=300.0).contains(&range_m));
        self.render_profile(poses, false, Some(range_m))
    }

    fn render_inner(&self, poses_in: &[Pose], beings_on: bool) -> Vec<Vec<Pixel>> {
        self.render_profile(poses_in, beings_on, crate::visao::road_view_range())
    }

    fn render_profile(
        &self,
        poses_in: &[Pose],
        beings_on: bool,
        road_range: Option<f32>,
    ) -> Vec<Vec<Pixel>> {
        let n = poses_in.len().min(self.capacity);
        if n == 0 {
            return Vec::new();
        }

        let mut pitches = [[0.0f32; 4]; 4];
        for row in 0..RETINA_ROWS {
            pitches[row / 4][row % 4] = road_range.map_or_else(
                || crate::visao::pitch(row),
                |r| crate::visao::road_view_pitch(r, row),
            );
        }
        let params = ParamsGpu {
            grid_size: self.grid_size,
            n_beings: n as u32,
            cell_size: self.cell_size,
            fov_half: (crate::visao::fov_deg() / 2.0).to_radians(),
            pitches,
            beings_on: beings_on as u32,
            _pad0: if road_range.is_some() { 1 } else { crate::visao::retina_geometry_mode() },
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
            _pad0: crate::visao::retina_geometry_mode(),
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
    /// Call after submitting the encoder that held `dispatch_into`.
    /// This method polls after requesting the mapping so its callback can complete.
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
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("retina mapping callback")
            .expect("retina mapping failed");

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
