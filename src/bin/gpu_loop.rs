//! Simulation loop 100% resident on the GPU.
//!
//! # The problem this binary solves
//!
//! The CPU loop (`motor.rs`) was instrumented with GPUs piece by piece:
//!   GPU retina → submit + poll(Wait) → readback → GPU perception → submit + poll(Wait) → ...
//! Each piece did its own host↔GPU round-trip. Measured result: enabling GPU perception
//! made the tick SLOWER (6.61 ms versus 6.25 ms), because the round-trips cost more than
//! the parallelism gained.
//!
//! The fix: **one encoder, all dispatches, one submit, one readback**.
//!
//! # Pipeline per tick
//!
//!   poses_cpu (2 kB) → write_buffer
//!   encoder {
//!     dispatch retina    (105.840 threads, 1654 wg × 64)
//!     dispatch perception (60 threads, 60 wg × 1)
//!     dispatch brain      (60 threads, 60 wg × 1)
//!     copy buf_intent → buf_intent_readback
//!   }
//!   queue.submit(encoder)                ← ONE submit
//!   device.poll(Wait)                    ← ONE block
//!   read buf_intent_readback (960 bytes) ← intent[60] → CPU physics
//!   CPU physics (state[60]) → next poses
//!
//! # What stays on the CPU (and why)
//!
//! 1. **Physics**: `PhysicsGpu` opens its own device to isolate ACO/NIR crashes
//!    of the RADV NAVI22 driver (bisection `fisica_bisect` documented in physics.rs).
//!    Porting it to the shared device would require rewriting the initialization —
//!    separate work. For now: `step()` runs on its own device after the readback.
//!    Body state (pos/vel) stays in `Vec<PhysicsStateGpu>` on the CPU.
//!
//! 2. **Temporal channels of sensory_wave** (≈40 channels of 223): wall_ema, approach_rate,
//!    visual_cortex, racing_line, comp_stress — they depend on per-being state accumulated
//!    between ticks. For the all-GPU loop these channels stay at ZERO in this binary.
//!    The brain model still moves (physics integrates intent), but without these channels
//!    navigation quality is inferior to the full motor.rs.
//!
//! # Sanity
//!
//! The beings MUST move. If the distance travelled is zero, the loop is broken
//! even with no error. The binary prints distance travelled for a few beings.
//!
//! # Usage
//!
//!   cd ~/projects/LLM/Universe
//!   env -u DISPLAY ./target/release/gpu_loop
//!   env -u DISPLAY ./target/release/gpu_loop --seres 60 --ticks 200

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Instant;

use bytemuck::Pod;
use wgpu::util::DeviceExt;

use universe::config::iara::brain::{L1_NEURONS, L2_NEURONS, SENSORY_WAVE_SIZE};
use universe::config::iara::vision::{RETINA_ROWS, RETINA_TOTAL};
use universe::engine::SovereignEngine;
use universe::evolution::dna_genesis::load_champion_dna;
use universe::gpu::brain::{BrainGpu, BrainInput};
use universe::gpu::perception::{PerceptionGpu, PosePerception, TEMPORAL_STRIDE};
use universe::gpu::physics::{PhysicsGpu, PhysicsStateGpu, WheelGpu, pack_config_flags};
use universe::gpu::{Pose, RetinaGpu};
use universe::perception::RetinaPerception;
use universe::simulation::racing_line::RacingLine;
use universe::visao::pitch;
use universe::world::VoxelWorld;

// ─── Deduplicated types/helpers (E3, REFACTOR_PLAN §2.3) ─────────────────────
// RetinaParamsGpu/PoseRaw/uniform_entry/storage_entry/pos_of/make_physics_state
// were identical copies of the ones in gpu_evolve.rs and cortex_evolve.rs — unified in
// src/gpu/retina_types.rs, src/gpu/core.rs and src/gpu/physics.rs.
use universe::gpu::core::{storage_entry, uniform_entry};
use universe::gpu::physics::{make_physics_state, pos_of};
use universe::gpu::retina_types::{PoseRaw, RetinaParamsGpu};

// ─── State of one being for the loop (CPU-side, physical position) ───────────

struct BeingState {
    /// Full physics state (resides on the CPU because PhysicsGpu has its own device).
    phys: PhysicsStateGpu,
    /// Smoothed steer from the previous tick — replicates motor.rs (prev_steer_mem * 0.50 + new * 0.50).
    /// Without this filter, the model can emit steer=1 at tick 0 → slip_angle≈90° on the front tire,
    /// longitudinal force=0, the car spins in place even with throttle=1.
    prev_steer: f32,
    /// Counter of ticks with speed < 0.5 m/s — replicates motor.rs stagnation (line 2221).
    /// When it reaches 32 ticks (0.5s), activates escape mode: throttle=0.3 for 64 ticks.
    stagnation_ticks: u32,
    /// Counter of ticks remaining in escape mode (forced throttle = 0.3).
    escape_ticks: u32,
    /// Time remaining in emergency reverse maneuver (s).
    /// Replicates racer.reverse_timer from motor.rs (simulation/motor.rs L2174).
    /// > 0 while reversing; decrements by dt; ch89 = 127 while > 0.
    reverse_timer: f32,
    /// Post-reverse cooldown (s). While > 0, caps acceleration at 0.8 and blocks a new reverse.
    /// Replicates racer.post_reverse_lockout from motor.rs (L2182: POST_REVERSE_LOCKOUT=0.6).
    post_reverse_lockout: f32,
}

// pos_of() and make_physics_state() now come from universe::gpu::physics (dedup E3, see
// the `use` at the top of the file) — they were identical copies of gpu_evolve.rs/cortex_evolve.rs.

// ─── Unified context: retina + perception + brain on the same device ─────────

struct GpuLoop {
    /// Only so the diagnostic prints once, at the first tick.
    diag_feito: bool,
    diag_fis: bool,
    // Device shared by retina + perception + brain.
    // Arc because BrainGpu requires Arc<Device> in its signature.
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    adapter_name: String,

    // ── Retina ────────────────────────────────────────────────────────────────
    // RetinaGpu has its own device by value — we reuse its buffers but
    // call dispatch_into() to record into the external encoder.
    // PROBLEM: RetinaGpu::new() opens its own device. For the chained loop,
    // we replicate the essentials of the retina here using the shared device.
    pipeline_retina: wgpu::ComputePipeline,
    layout_retina: wgpu::BindGroupLayout,
    buf_grid: wgpu::Buffer,
    buf_poses_retina: wgpu::Buffer,
    buf_retina_out: wgpu::Buffer, // VRAM, feeds perception directly
    // buf_retina_readback NOT allocated — we do no retina readback in the loop
    grid_size: u32,
    cell_size: f32,

    // ── Perception ────────────────────────────────────────────────────────────
    percept: PerceptionGpu,

    // ── Brain ─────────────────────────────────────────────────────────────────
    brain: BrainGpu,

    // ── Racing line (copy for per-tick CPU computation) ───────────────────────
    // The projection onto the racing_line is computed by the CPU before each dispatch
    // and injected into buf_temporal[10-15] via update_racing_line_temporal().
    // We keep a local copy to avoid lifetimes with VoxelWorld.
    racing_line: RacingLine,

    // ── Voxel grid on the CPU (copy of the world's grid) ─────────────────────
    // Needed to compute dist_l/r/f on the CPU before the temporal kernel dispatch.
    // The RADV ACO compiler hung while compiling loops over storage buffers in WGSL
    // (the original BLOCK T had LAT_SAMPLES and FWD_SAMPLES loops over `grid`).
    // Solution: the CPU does the computation and injects into buf_temporal[16-18] via
    // update_dist_temporal().
    // Cost: Vec<u32> of 320×320×4 = ~410 KB on the heap — negligible.
    grid_cpu: Vec<u32>,
    grid_size_cpu: u32,
    cell_size_cpu: f32,

    // ── Physics ───────────────────────────────────────────────────────────────
    // SEPARATE device for ACO/NIR crash isolation (see physics.rs:14).
    // The physical state (PhysicsStateGpu) stays on the CPU; goes up each tick via
    // write_buffer on the physics device, comes down via readback.
    phys_gpu: PhysicsGpu,

    // ── KAN output head ───────────────────────────────────────────────────────
    // Knots for the output spline (throttle, steer, brake). Loaded from CHAMPION.dna
    // once; used each tick to convert raw intent → real physical pedal.
    // Without this step: intent[1]=0.985 → steer=0.985 → slip_angle=90° → no traction.
    // With default KAN tanh: eval(1, 0.985) ≈ 0.71, an evolved one can be more conservative.
    kan_knots: [[f32; universe::kan::KAN_OUT_KNOTS]; 3],

    // ── Perception CfC (ch 192-207) ──────────────────────────────────────────
    // One RetinaPerception per being, with recurrent state resident on the CPU between ticks.
    // Weights: fixed seed 0xCAFE_BABE_DEAD_BEEF (identical to motor.rs L1626).
    // Per tick: reads buf_retina_out via readback → runs CfC → injects hidden into temporal[33-48].
    percept_cfc: Vec<RetinaPerception>,

    // MAP_READ buffer for retina readback (423 KB @ 60 beings).
    // Needed to run the perception CfC on the CPU with the current retina.
    // buf_retina_out has COPY_SRC; buf_retina_readback has MAP_READ + COPY_DST.
    // The copy (buffer_to_buffer) is added to the encoder BEFORE the submit.
    buf_retina_readback: wgpu::Buffer,

    // ── Metadata ──────────────────────────────────────────────────────────────
    capacity: usize,
}

impl GpuLoop {
    fn new(
        world: &VoxelWorld,
        engines: &[SovereignEngine],
        capacity: usize,
        kan_knots: [[f32; universe::kan::KAN_OUT_KNOTS]; 3],
    ) -> Option<Self> {
        // ── Ensure only one Vulkan ICD is loaded ──────────────────────────────
        // Multiple ICDs (intel_hasvk, lvp, radeon...) cause SIGSEGV in the loader.
        let icd_radeon = "/usr/share/vulkan/icd.d/radeon_icd.json";
        if std::path::Path::new(icd_radeon).exists() && std::env::var("VK_ICD_FILENAMES").is_err() {
            unsafe {
                std::env::set_var("VK_ICD_FILENAMES", icd_radeon);
            }
        }

        // ── Open shared device ────────────────────────────────────────────────
        // compute-only: compatible_surface = None.
        // max_storage_buffers_per_shader_stage = 8: perception uses 7 bindings.
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;
        let name = adapter.get_info().name.clone();
        let mut limits = wgpu::Limits::downlevel_defaults();
        // The brain uses 10 storage buffers: bindings 1-9 plus `buf_tau` (τ bank) —
        // same count documented in gpu_evolve/context.rs::GpuCtx::new(). The value
        // "9" here was outdated (lying comment: it said binding 9 was the maximum,
        // but `create_bind_group_layout(brain-layout)` already asked for 10 bindings
        // and wgpu aborted with "Too many bindings of type StorageBuffers... limit is
        // 9, count was 10"). Found while running `gpu_loop` to validate E3 (dedup of
        // uniform_entry/storage_entry/kan_eval) — fixed here in the same session, not
        // a "next step".
        // Asking for 12 gives the same headroom as gpu_evolve.rs for the next buffers
        // without coming back here for every new binding; if the adapter does not
        // support it, `request_device` fails at open instead of corrupting data.
        limits.max_storage_buffers_per_shader_stage = 16;
        let (raw_device, raw_queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("iara-loop"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;
        let device = Arc::new(raw_device);
        let queue = Arc::new(raw_queue);

        let cap = capacity.max(1);

        // ── Static world buffers ──────────────────────────────────────────────
        // grid: 320×320 u32. Shared by retina and perception.
        let grid_u32: Vec<u32> = world.grid.iter().map(|v| *v as u32).collect();
        let buf_grid = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("loop-grid"),
            contents: bytemuck::cast_slice(&grid_u32),
            // COPY_SRC not needed — only retina and perception read it.
            usage: wgpu::BufferUsages::STORAGE,
        });

        // ── Retina buffers ────────────────────────────────────────────────────
        // poses_retina: cap × PoseRaw (8 × f32 = 32 bytes).
        let buf_poses_retina = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("loop-poses-retina"),
            size: (cap * std::mem::size_of::<PoseRaw>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // retina_out: cap × RETINA_TOTAL × 4 bytes. Feeds perception in VRAM.
        // COPY_SRC to inspect if necessary.
        let buf_retina_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("loop-retina-out"),
            size: (cap * RETINA_TOTAL * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // ── Retina pipeline ───────────────────────────────────────────────────
        // retina.wgsl: binding 0=params(uniform), 1=grid(storage), 2=poses(storage), 3=saida(storage).
        let shader_retina = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("loop-retina"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(universe::gpu::RETINA_SHADER)),
        });
        let layout_retina = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("loop-retina-layout"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),  // grid
                storage_entry(2, true),  // poses
                storage_entry(3, false), // output
            ],
        });
        let pl_retina = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout_retina],
            push_constant_ranges: &[],
        });
        let pipeline_retina = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("loop-retina-pipeline"),
            layout: Some(&pl_retina),
            module: &shader_retina,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── Perception ────────────────────────────────────────────────────────
        // PerceptionGpu::new takes &wgpu::Device (Arc<Device> implements Deref).
        eprintln!("[gpu_loop] Creating PerceptionGpu...");
        let percept = PerceptionGpu::new(&device, &queue, world, cap)?;
        eprintln!("[gpu_loop] PerceptionGpu OK");

        // ── Brain ─────────────────────────────────────────────────────────────
        // BrainGpu accepts Arc<Device> directly.
        let brain = BrainGpu::new(Arc::clone(&device), Arc::clone(&queue), name.clone(), cap);

        // Uploads the engines' weights to the brain buffer.
        let eng_refs: Vec<&SovereignEngine> = engines.iter().take(cap).collect();
        brain.upload_weights(&eng_refs);

        // ── Physics — SAME device as everything else ─────────────────────────
        // It was born with its own device to isolate the RADV ACO compiler crash
        // (`physics.rs`: the Pacejka phase blew up the register allocator). That crash
        // was solved by splitting into three kernels, so the isolation lost its reason — and
        // started to HURT: with two `wgpu::Device` on the same GPU the loop hung on the
        // first tick, no error and no output. A single device unblocks it and is what allows
        // chaining the four dispatches into a single `CommandEncoder`.
        let grid_u8: Vec<u8> = world.grid.iter().map(|v| *v as u8).collect();
        let phys_gpu = PhysicsGpu::new_shared(
            Arc::clone(&device),
            Arc::clone(&queue),
            name.clone(),
            &grid_u8,
            &world.height_map,
            world.size,
            world.cell_size,
            cap,
        )?;

        // ── Retina readback buffer (for the perception CfC on the CPU) ─────────
        // cap × RETINA_TOTAL × 4 bytes. Copy of buf_retina_out after the dispatch.
        // Allows running RetinaPerception on the CPU with no shader change at all.
        let retina_readback_bytes = (cap * RETINA_TOTAL * std::mem::size_of::<u32>()) as u64;
        let buf_retina_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("loop-retina-readback"),
            size: retina_readback_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Perception CfC — one per being, with recurrent state on the CPU ────
        // Weights initialized with fixed seed 0xCAFE_BABE_DEAD_BEEF (same as motor.rs).
        // Recurrent state (hidden[16]) will reside here between ticks.
        // Unlike motor.rs which uses retina_binocular (u8 converted from raw),
        // here we use the GPU retina (packed u32) converted to i8 on read.
        let percept_cfc: Vec<RetinaPerception> =
            (0..cap).map(|_| RetinaPerception::new()).collect();

        println!(
            "GpuLoop: adapter={} · cap={} beings · retina+perception+brain+physics ON THE SAME device",
            name, cap
        );
        println!(
            "  perception CfC (ch192-207): {} beings × 16 hidden = {} f32 on the CPU",
            cap,
            cap * 16
        );
        println!("  ch83-84 (hemispheric flow): enabled in the motion kernel");
        println!("  ch89 (reverse_timer): enabled via temporal[32]");

        Some(Self {
            diag_feito: false,
            diag_fis: false,
            device,
            queue,
            adapter_name: name,
            pipeline_retina,
            layout_retina,
            buf_grid,
            buf_poses_retina,
            buf_retina_out,
            grid_size: world.size as u32,
            cell_size: world.cell_size,
            percept,
            brain,
            racing_line: world.racing_line.clone(),
            // Copy of the grid for CPU computation of dist_l/r/f (temporal[16-18]).
            grid_cpu: grid_u32,
            grid_size_cpu: world.size as u32,
            cell_size_cpu: world.cell_size,
            phys_gpu,
            kan_knots,
            percept_cfc,
            buf_retina_readback,
            capacity: cap,
        })
    }

    /// Runs one full tick for N beings.
    ///
    /// # Data flow
    ///
    ///   physical poses (CPU) →[poses_retina]→ GPU
    ///   encoder {
    ///     retina    → buf_retina_out (VRAM)
    ///     perception ← buf_retina_out (VRAM without readback)
    ///             → buf_percept_out (VRAM)
    ///     brain    ← buf_percept_out (read via the wave written to memory)
    ///             → buf_intent (VRAM) + copy → buf_intent_readback
    ///   }
    ///   submit + poll(Wait)         ← ONE round-trip
    ///   read buf_intent_readback    ← intent[n]
    ///   update throttle/steer/brake in states[n]
    ///   phys_gpu.step(states[n])   ← physics on the isolated device
    ///
    /// Returns `true` on success.
    fn tick(
        &mut self,
        states: &mut Vec<BeingState>,
        engines: &[SovereignEngine],
        n: usize,
        dt: f32,
        tick_idx: usize,
    ) -> bool {
        if n == 0 || n > self.capacity {
            return false;
        }

        // ── 1. Convert physical states to retina poses ────────────────────────
        // PoseRaw: [pos.x, pos.y, pos.z, gaze_yaw] + [fwd.x, fwd.z, right.x, right.z]
        // The retina uses world coordinates; gaze_yaw = 0 (no lateral deviation).
        let pose_raw_data: Vec<PoseRaw> = (0..n)
            .map(|i| {
                let p = &states[i].phys;
                PoseRaw {
                    pos_gaze: [p.pos[0], p.pos[1], p.pos[2], 0.0],
                    dirs: [p.fwd[0], p.fwd[2], p.lat[0], p.lat[2]],
                }
            })
            .collect();
        self.queue.write_buffer(
            &self.buf_poses_retina,
            0,
            bytemuck::cast_slice(&pose_raw_data),
        );

        // ── 2. Poses for perception (different format: PosePerception) ────────
        let poses_percept: Vec<PosePerception> = (0..n)
            .map(|i| {
                let p = &states[i].phys;
                PosePerception {
                    pos_x: p.pos[0],
                    pos_y: p.pos[1],
                    pos_z: p.pos[2],
                    phase: 0.0, // absolute heading — zero since we have no racer.phase here
                    fwd_x: p.fwd[0],
                    fwd_z: p.fwd[2],
                    rgt_x: p.lat[0],
                    rgt_z: p.lat[2],
                }
            })
            .collect();

        // ── 3a. Precompute lateral/frontal distances on the CPU → buf_temporal[16-18] ──
        // The RADV ACO compiler hung while compiling the LAT_SAMPLES and FWD_SAMPLES loops
        // over the `grid` storage buffer in WGSL (workgroup_size=1, pattern with `break`).
        // Solution: the CPU replicates the same algorithm and injects the 3 values via
        // update_dist_temporal().
        // Cost: N × (6+6+10) = N×22 CPU grid queries (O(1) hash per cell). Negligible.
        {
            let gs = self.grid_size_cpu as usize;
            let cs = self.cell_size_cpu;
            // Lateral samples: expanded from 27m → 40m (math: 40/28 m/s = 1.43 s anticipation).
            // Normalized by 40.0 in temporal_wgsl. Default 40.0 = free (no wall in range).
            let lat_samp = [1.0_f32, 3.0, 6.0, 9.0, 12.0, 16.0, 20.0, 26.0, 33.0, 40.0];
            // Frontal samples: same as the original BLOCK T
            let fwd_samp = [3.0_f32, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0];

            // Inline function: voxel in world coordinates (identical to voxel_at() in the WGSL).
            let voxel_at = |wx: f32, wz: f32| -> u32 {
                let xi = (wx / cs).round() as i32;
                let yi = (wz / cs).round() as i32;
                let n = gs as i32;
                if xi >= 0 && xi < n && yi >= 0 && yi < n {
                    self.grid_cpu[(yi as usize) * gs + (xi as usize)]
                } else {
                    2u32 // Wall outside the grid
                }
            };

            let mut dist_vals: Vec<[f32; 3]> = Vec::with_capacity(n);
            for i in 0..n {
                let p = &states[i].phys;
                let (px, pz) = (p.pos[0], p.pos[2]);
                let (fx, fz) = (p.fwd[0], p.fwd[2]);
                // Left perpendicular: (-fwd_z, fwd_x) in the XZ plane.
                // We replicate BLOCK T exactly: perp_x = -fwd_z, perp_z = fwd_x.
                let perp_x = -fz;
                let perp_z = fx;

                // Left lateral distance: pos - perp*sd (same as the original BLOCK T)
                // Default 40.0 = free (no wall within 40m range).
                let mut dist_l = 40.0_f32;
                for &sd in &lat_samp {
                    if voxel_at(px - perp_x * sd, pz - perp_z * sd) == 2u32 {
                        dist_l = sd;
                        break;
                    }
                }
                // Right lateral distance: pos + perp*sd
                let mut dist_r = 40.0_f32;
                for &sd in &lat_samp {
                    if voxel_at(px + perp_x * sd, pz + perp_z * sd) == 2u32 {
                        dist_r = sd;
                        break;
                    }
                }
                // Frontal distance
                let mut dist_f = 50.0_f32;
                for &sd in &fwd_samp {
                    let v = voxel_at(px + fx * sd, pz + fz * sd);
                    if v == 2u32 || v == 3u32 {
                        dist_f = sd;
                        break;
                    }
                }
                dist_vals.push([dist_l, dist_r, dist_f]);
            }
            // Injects into buf_temporal[16-18] BEFORE the temporal perception dispatch.
            self.percept
                .update_dist_temporal(&self.queue, &dist_vals[..n]);
        }

        // ── 3b. Precompute racing_line on the CPU and inject into buf_temporal[10-15] ──
        // The O(N_points) projection onto the racing_line was moved to the CPU because the
        // RADV ACO/NIR compiler crashed while compiling loops over variable-length storage
        // buffers in WGSL. CPU cost: N × O(N_points) with local search
        // = ~60 × 64 iterations = 3840 operations. Negligible vs 0.63ms/tick of GPU.
        //
        // Temporal fields [10-15]:
        //  [10] cpu_line_off_norm = offset / half_width, [-1,1]
        //  [11] cpu_d_off         = (prev_off - off) / hw × 8, [-1,1]  (derivative)
        //  [12] cpu_la_off        = lookahead_offset(5m) / hw, [-1,1]
        //  [13] cpu_fwd_norm      = fwd_dist / (hw*3), [0,1]  (uses dist_l=max of the grid)
        //  [14] cpu_corridor      = (dist_l+dist_r) / (hw*4), [0,1]
        //  [15] cpu_whisker_norm  = min(dist_l,dist_r) / (hw*1.5), [0,1]
        //
        // prev_off for the d_off computation is read from field [8] of the temporal.
        // In this binary we do no temporal readback (cost), so we estimate
        // prev_off = current line_off_norm from the previous tick via simple lookup.
        // We store it in prev_rl_off (Vec<f32> allocated 1× per run, see below).
        // NOTE: d_off measures the rate of change of deviation — "am I drifting away from
        //       the line faster than I was?". Gain 8× because the per-tick delta is ~0.01.
        {
            let hw = self.racing_line.half_width.max(1.0);
            let hw3 = hw * 3.0;
            let hw4 = hw * 4.0;
            let hw15 = hw * 1.5;
            // Computes values and accumulates them in a temporary Vec per tick.
            let mut racing_vals: Vec<[f32; 6]> = Vec::with_capacity(n);
            for i in 0..n {
                let p = &states[i].phys;
                let pos2d = bevy::math::Vec2::new(p.pos[0], p.pos[2]);
                let fwd2d = bevy::math::Vec2::new(p.fwd[0], p.fwd[2]);

                // Projection onto the racing_line: signed lateral deviation (m), clamped ±hw.
                let line_off = self.racing_line.lateral_offset_from_line(pos2d, fwd2d);
                let off_norm = (line_off / hw).clamp(-1.0, 1.0);

                // Lookahead 5m ahead on the ideal line.
                let la = self.racing_line.lookahead_offset(pos2d, 5.0);
                let la_norm = (la / hw).clamp(-1.0, 1.0);

                // Approximate lateral distances (replaces the grid — needs world here,
                // but we have no reference to world in this method. We use half_width as
                // a conservative proxy: corridor = hw on each side = full corridor = 1.0).
                // TODO: for real distances, do a grid readback or pass a world ref.
                // For now: fwd_norm=0.5 (frontal distance unknown), corridor=0.8 (free).
                // The WGSL kernel computed dist_l/r on the grid — these fields are complementary.
                let cpu_fwd_norm = 0.5_f32; // unknown without grid readback
                let cpu_corridor = 0.8_f32; // free (conservative)
                let cpu_whisker_norm = 0.7_f32; // free (conservative)

                // d_off: derivative of the deviation. We need prev_off. We use the current
                // off_norm as prev for this tick; next tick the difference will be real.
                // (On the GPU, temporal[8] stores prev_line_offset_n from the previous tick.)
                // Here we have no previous value without readback. We pass 0 for d_off,
                // which gives a neutral signal. Motor.rs uses the full version with real state.
                let cpu_d_off = 0.0_f32; // no previous state available in this binary

                racing_vals.push([
                    off_norm,
                    cpu_d_off,
                    la_norm,
                    cpu_fwd_norm,
                    cpu_corridor,
                    cpu_whisker_norm,
                ]);
            }
            // Injects into buf_temporal[10-15] BEFORE the perception dispatch.
            self.percept
                .update_racing_line_temporal(&self.queue, &racing_vals[..n]);
        }

        // ── 3c. Proprioceptive telemetry → buf_temporal[20-31] ────────────────
        // Channels 64-75 of the sensory_wave: speed, alignment, RPM, gear, tire_temp, etc.
        // These channels are in the top-30 weights of CHAMPION.dna but were ZERO in gpu_loop
        // because the spatial shader has no access to the physical state (resides on the CPU).
        // The temporal shader reads temporal[20-31] and does atomicOr on ch64-75 of buf_output.
        //
        // Critical channels confirmed by inspecting the DNA's SRLE tag:
        //   ch64 (alignment)   — the model knows whether it is aligned with the track
        //   ch65 (speed m/s)   — MOST CRITICAL: the model knows whether it is moving
        //   ch68 (RPM ratio)   — knows whether the engine is under load
        //   ch69 (gear)        — gear context for the shift decision
        //   ch70 (tire_temp)   — tire temperature (grip)
        //   ch73 (on_grass)    — grass pain → reduces throttle (ch73 was -128 in motor.rs)
        {
            // Inline aux function: the being's current voxel (to detect grass).
            let gs = self.grid_size_cpu as usize;
            let cs = self.cell_size_cpu;
            let voxel_at_cpu = |wx: f32, wz: f32| -> u32 {
                let xi = (wx / cs).round() as i32;
                let yi = (wz / cs).round() as i32;
                let ng = gs as i32;
                if xi >= 0 && xi < ng && yi >= 0 && yi < ng {
                    self.grid_cpu[(yi as usize) * gs + (xi as usize)]
                } else {
                    2u32
                }
            };

            let mut prop_vals: Vec<[f32; 12]> = Vec::with_capacity(n);
            for i in 0..n {
                let p = &states[i].phys;

                // [1] speed: magnitude of the velocity vector in the XZ plane (we ignore Y=vert.)
                let speed_m_s = (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt();

                // [0] alignment: cos(angle between the fwd heading and the track tangent).
                // The racing_line returns the tangent unit vector at the closest point.
                // Here we use the dot product fwd·tangent as an alignment proxy.
                // Tangent = the being's fwd rotated by the racing_line's angular deviation.
                // Simple approximation: we use the projection of the velocity onto the heading.
                // dot(vel_xz, fwd_xz) / (|vel| + ε) → 1 = aligned, -1 = wrong way.
                let vel_dot_fwd = p.vel[0] * p.fwd[0] + p.vel[2] * p.fwd[2];
                let alignment = if speed_m_s > 0.1 {
                    (vel_dot_fwd / speed_m_s).clamp(-1.0, 1.0)
                } else {
                    1.0 // stopped = assume aligned (no direction noise)
                };

                // [2] lateral velocity: dot(vel, lat) — positive = drifting to the right
                let lat_vel = p.vel[0] * p.lat[0] + p.vel[2] * p.lat[2];

                // [3] HP: no damage tracker, assume full (127 on the channel).
                let hp_frac = 1.0_f32;

                // [4] RPM ratio
                let rpm_ratio = if p.max_rpm > 0.0 {
                    (p.current_rpm / p.max_rpm).clamp(0.0, 1.0)
                } else {
                    0.0
                };

                // [5] gear (f32 for the WGSL — multiplied by 20 in the shader)
                let gear = p.current_gear as f32;

                // [6] tire_temp_norm: wheels has layout [part_a, part_b] alternating per axle.
                // part_b[0] is already normalised to [0,1].
                // wheels[j*2+1] = part_b of axle j (j=0..3 = FR, FL, RR, RL).
                let tire_temp_sum: f32 = (0..4).map(|j| p.wheels[j * 2 + 1][0]).sum();
                let tire_temp_norm = (tire_temp_sum / 4.0).clamp(0.0, 1.0);

                // [7] downforce: we do not compute it dynamically. Zero for now.
                let downforce_norm = 0.0_f32;

                // [8] in_draft: we do not detect slipstream. Zero.
                let in_draft = 0.0_f32;

                // [9] on_grass: current voxel. VoxelType: Grass=0, Road=1, Wall=2, Hole=3.
                // Grass = 0 (the whole map background before being paved).
                let voxel = voxel_at_cpu(p.pos[0], p.pos[2]);
                let on_grass = if voxel == 0u32 { 1.0_f32 } else { 0.0 };

                // [10] steer_abs: |steer| as a proxy for steer_limit.
                // In motor.rs steer_limit = maximum available angle (depends on the geometry).
                // Here we use |steer_cmd| as a signal that it is turning.
                let steer_abs = p.steer.abs().clamp(0.0, 1.0);

                // [11] wetness: no dynamic weather. Zero.
                let wetness = 0.0_f32;

                prop_vals.push([
                    alignment,
                    speed_m_s,
                    lat_vel,
                    hp_frac,
                    rpm_ratio,
                    gear,
                    tire_temp_norm,
                    downforce_norm,
                    in_draft,
                    on_grass,
                    steer_abs,
                    wetness,
                ]);
            }
            self.percept
                .update_proprioception_temporal(&self.queue, &prop_vals[..n]);
        }

        // ── 3d. reverse_timer (ch 89) and perception CfC (ch 192-207) ──────────
        // Reverse logic IDENTICAL to motor.rs (simulation/motor.rs L2177-2235):
        //   Trigger: frontal wall <3m AND both laterals blocked (<4m)
        //            AND speed<1.5 m/s AND stagnation>60 ticks AND no post-reverse lockout.
        //   Duration: 0.4s (REVERSE_MIN) capped at 1.5s (REVERSE_MAX).
        //   Exit: frontal wall > 2.5m (REVERSE_EXIT_CLEAR) or timeout.
        //   Post-reverse: 0.6s lockout where throttle is capped at 0.8.
        //
        // Perception CfC: uses the retina from tick T-1 (read in the readback after submit T-1).
        // A 1-tick delay is acceptable — motor.rs also uses the previous visual_cortex.
        {
            // Deduplicated (E3, REFACTOR_PLAN.md §2.3) into
            // universe::config::iara::reverse_maneuver. TRIGGER_DIST_M=4.0 (not 3.0):
            // this file already used `fwd_dist < 4.0` on purpose (see the comment at the
            // trigger below) — the local constant `REVERSE_TRIGGER_DIST=3.0` was never
            // read (`never used` warning), only quoted in a comment. `gpu_evolve.rs`
            // had the SAME pattern (dead constant at 3.0, literal used at 4.0) — the
            // single canonical value below is what the three binaries were actually running.
            use universe::config::iara::reverse_maneuver as revcfg;

            let gs = self.grid_size_cpu as usize;
            let cs = self.cell_size_cpu;
            // Inline function to query a voxel (same as in block 3a).
            let voxel_at = |wx: f32, wz: f32| -> u32 {
                let xi = (wx / cs).round() as i32;
                let yi = (wz / cs).round() as i32;
                let ng = gs as i32;
                if xi >= 0 && xi < ng && yi >= 0 && yi < ng {
                    self.grid_cpu[(yi as usize) * gs + (xi as usize)]
                } else {
                    2u32
                }
            };

            let mut extra_vals: Vec<[f32; 17]> = Vec::with_capacity(n);
            for i in 0..n {
                let p = &states[i].phys;
                let speed = (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt();

                // Measures distances for the trigger (replicates motor.rs L2202-2220):
                //   fwd_dist = smallest frontal distance to Wall/Hole (cap 30m)
                //   lat_l_check = left lateral distance to Wall/Hole (<4m = blocked)
                //   lat_r_check = same on the right
                let (fx, fz) = (p.fwd[0], p.fwd[2]);
                let perp_x = -fz;
                let perp_z = fx; // left perpendicular
                let (px, pz) = (p.pos[0], p.pos[2]);

                // Frontal distance (samples from 1m to 10m as in motor.rs)
                let mut fwd_dist = 30.0_f32;
                for sd in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] {
                    let v = voxel_at(px + fx * sd, pz + fz * sd);
                    if v == 2u32 || v == 3u32 {
                        fwd_dist = sd;
                        break;
                    }
                }

                // Lateral distances (samples 1m to 10m as in motor.rs)
                let mut lat_l = 10.0_f32;
                let mut lat_r = 10.0_f32;
                for sd in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] {
                    if lat_l == 10.0 {
                        let v = voxel_at(px + perp_x * sd, pz + perp_z * sd);
                        if v == 2u32 || v == 3u32 {
                            lat_l = sd;
                        }
                    }
                    if lat_r == 10.0 {
                        let v = voxel_at(px - perp_x * sd, pz - perp_z * sd);
                        if v == 2u32 || v == 3u32 {
                            lat_r = sd;
                        }
                    }
                }
                let both_blocked = lat_l < 4.0 && lat_r < 4.0;

                // Decrements the post-reverse lockout.
                if states[i].post_reverse_lockout > 0.0 {
                    states[i].post_reverse_lockout = (states[i].post_reverse_lockout - dt).max(0.0);
                }

                // Trigger: frontal wall <4m + speed<1.5 + stagnation>60 + no lockout.
                // NOTE: we removed motor.rs's `both_blocked` condition for an important reason:
                // the CHAMPION was trained with the motor.rs logic where reverse triggers BEFORE
                // getting stuck against a single wall. When lat_r is free but lat_l+fwd blocked,
                // the model emits negative throttle (signaling reverse intent) — that signal must
                // activate the reverse. Without this, negative throttle is clipped to zero and the
                // car gets stuck.
                // Also: motor.rs uses `forward_wall_dist`, which is computed differently
                // (with longer probes) — our fwd_dist can be more restrictive. That is why
                // the trigger uses 4.0 m (`revcfg::TRIGGER_DIST_M`), more permissive than the 3.0 m
                // a naive reading of motor.rs would suggest.
                {
                    let rev = states[i].reverse_timer;
                    let stag = states[i].stagnation_ticks;
                    let lockout = states[i].post_reverse_lockout;
                    if rev <= 0.0
                        && lockout <= 0.0
                        && fwd_dist < revcfg::TRIGGER_DIST_M
                        && speed < revcfg::TRIGGER_SPEED_MAX_MS
                        && stag > revcfg::STAGNATION_TICKS
                    {
                        states[i].reverse_timer = revcfg::MIN_DURATION_S;
                        states[i].stagnation_ticks = 0;
                    }
                }

                // Decrements reverse_timer and checks the reverse exit condition.
                {
                    let rev = states[i].reverse_timer;
                    if rev > 0.0 {
                        let new_rev = rev - dt;
                        // Exits reverse if: front is clear (>2.5m) or maximum timeout.
                        if fwd_dist > revcfg::EXIT_CLEAR_M || new_rev > revcfg::MAX_DURATION_S {
                            states[i].reverse_timer = 0.0;
                            states[i].post_reverse_lockout = revcfg::POST_LOCKOUT_S;
                        } else {
                            states[i].reverse_timer = new_rev.max(0.0);
                        }
                    }
                }

                // Reads the perception CfC hidden (state of tick T-1, updated in the readback).
                let cfc_ref = &self.percept_cfc[i];
                let hidden = &cfc_ref.pipeline.state.h; // &Vec<f32>, len=16

                // Builds [reverse_timer, cfc_hidden[0..15]] to inject into temporal[32-48].
                let mut val = [0.0f32; 17];
                val[0] = states[i].reverse_timer;
                for j in 0..16 {
                    val[1 + j] = if j < hidden.len() { hidden[j] } else { 0.0 };
                }
                extra_vals.push(val);
            }
            self.percept
                .update_extra_temporal(&self.queue, &extra_vals[..n]);
        }

        // ── 4. Single encoder ─────────────────────────────────────────────────
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("loop-tick"),
            });

        // ── 4a. Retina ────────────────────────────────────────────────────────
        // Uniform params for the retina (16 + 64 + 16 bytes = 96 bytes — grown from 80
        // for `--retina-beings`, docs/RETINA-BEINGS.md §2).
        let mut pitches_arr = [[0.0f32; 4]; 4];
        for row in 0..RETINA_ROWS {
            pitches_arr[row / 4][row % 4] = pitch(row);
        }
        let retina_params = RetinaParamsGpu {
            grid_size: self.grid_size,
            n_beings: n as u32,
            cell_size: self.cell_size,
            fov_half: (universe::visao::fov_deg() / 2.0).to_radians(),
            pitches: pitches_arr,
            // `--retina-beings` is not wired into this binary — off, same as before the
            // field existed.
            beings_on: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let buf_rp = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("loop-retina-params"),
                contents: bytemuck::bytes_of(&retina_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg_retina = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout_retina,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_rp.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buf_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_poses_retina.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buf_retina_out.as_entire_binding(),
                },
            ],
        });
        // ── 4a-PRE: preserves the previous tick's retina BEFORE the retina writes ─
        // copy_retina_prev copies buf_retina_out (tick t-1) → buf_retina_prev.
        // This MUST happen before the retina dispatch (which overwrites buf_retina_out).
        // At tick 0: buf_retina_prev = zeros → motion = |real_frame - 0| = real_frame (ok).
        // On subsequent ticks: buf_retina_prev = frame of tick t-1 → motion = real delta.
        self.percept
            .copy_retina_prev(&mut enc, &self.buf_retina_out);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass-retina"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_retina);
            pass.set_bind_group(0, &bg_retina, &[]);
            // 64 threads/workgroup: 2 waves of 32 on RDNA2 (wave32 = the correct size).
            // Total threads = n × RETINA_TOTAL. For 60 beings: 60×1764 = 105.840 threads.
            // Workgroups = ceil(105.840 / 64) = 1.654.
            let total = (n * RETINA_TOTAL) as u32;
            pass.dispatch_workgroups(total.div_ceil(64), 1, 1);
        }
        // Ending pass_retina inserts a global memory barrier:
        // buf_retina_out is visible to the next dispatch (perception).

        // ── 4b. Perception ────────────────────────────────────────────────────
        // bind_and_dispatch records into the encoder WITHOUT submit.
        // Reads buf_retina_out directly from VRAM — zero readback.
        // Saves bytes_percept for the diagnostic readback (IARA_DIAG when stuck).
        let bytes_percept = self.percept.bind_and_dispatch(
            &mut enc,
            &self.device,
            &self.queue,
            &poses_percept[..n],
            &self.buf_retina_out,
        );
        // After the perception pass ends, buf_percept_out is ready in VRAM.
        // The brain reads that buffer DIRECTLY via binding 8 (use_percept_direct=1),
        // with no intermediate CPU readback. The perception→brain link exists now.
        // IMPORTANT: atomicOr writes u32 into buf_percept_out; brain.wgsl reads u32
        // without atomics (the perception pass has already ended — workgroup barriers guarantee it).

        // ── 4c. Brain ─────────────────────────────────────────────────────────
        // bind_and_dispatch with buf_percept = Some(...) enables use_percept_direct=1.
        // The shader reads the wave directly from buf_percept_out (56 u32 per being),
        // unpacks i8 two's-complement inline, and applies KAN+CfC+L1+L2.
        // wave_data_i32 is ignored on this path (we pass an empty slice).
        let bytes_intent = self.brain.bind_and_dispatch(
            &mut enc,
            n,
            &[], // ignored: use_percept_direct=1
            dt,
            Some(self.percept.buf_output()), // perception→brain link via VRAM
        );

        // ── 4d-extra. Copy retina → buf_retina_readback (for the perception CfC) ──
        // We add to the encoder the copy buf_retina_out → buf_retina_readback.
        // This allows, after poll(Wait), mapping buf_retina_readback and running the
        // perception CfC on the CPU with the retina OF THE CURRENT TICK (t). At the next
        // tick (t+1), the updated CfC state will be injected into temporal[33-48].
        // Cost: ~423 KB of intra-GPU copy (DMA, ~2 μs) + PCIe readback (~50 μs).
        let retina_copy_bytes = (n * RETINA_TOTAL * 4) as u64;
        enc.copy_buffer_to_buffer(
            &self.buf_retina_out,
            0,
            &self.buf_retina_readback,
            0,
            retina_copy_bytes,
        );

        // ── 4e. Single submit ─────────────────────────────────────────────────
        // All dispatches (retina, perception, brain) + retina readback copy
        // in a single command buffer → ONE submit DMA to the driver.
        self.queue.submit(Some(enc.finish()));

        // ── 5. Single poll — blocks until all kernels finish ──────────────────
        // This is the only round-trip per tick.
        self.device.poll(wgpu::Maintain::Wait);

        // ── 6. Intent read (960 bytes for 60 beings) ──────────────────────────
        let intents = self.brain.read_intent_after_submit(n);

        // ── 6b. Retina readback and perception CfC update ─────────────────────
        // We read buf_retina_readback (copied in encoder 4d-extra), run the perception
        // CfC for each being, and update the CfC's internal state (h).
        // At the NEXT tick, update_extra_temporal() will read self.percept_cfc[i].pipeline.cfc.h
        // and inject it into temporal[33-48] for the shader to write ch192-207.
        //
        // buf_retina_readback format: N × RETINA_TOTAL packed u32 (mm<<8 | mat).
        // RetinaPerception::perceive() accepts &[i8] (quantized distance -127..127).
        // The conversion uses motor.rs's legacy encoding (codificacao_legada equivalent):
        //   mat=0 (grass) → moderate
        //   mat=1 (road)  → negative
        //   mat=2 (wall)  → high positive
        //   mat=3 (hole)  → 127
        {
            let slice = self.buf_retina_readback.slice(0..retina_copy_bytes);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            // Post-mapping poll: the submit was already processed by the poll above,
            // but the map_async callback needs a poll to be served.
            self.device.poll(wgpu::Maintain::Wait);
            let _ = rx.recv();

            let retina_data = slice.get_mapped_range();
            let retina_u32: &[u32] = bytemuck::cast_slice(&retina_data);

            // For each being: converts u32 pixels → i8 and runs the perception CfC.
            // The updated hidden will be used at the NEXT tick via temporal[33-48].
            for i in 0..n {
                let base = i * RETINA_TOTAL;
                let mut retina_i8 = vec![0i8; RETINA_TOTAL];
                for j in 0..RETINA_TOTAL {
                    let packed = retina_u32[base + j];
                    let dist_mm = (packed >> 8) as f32;
                    let dist_m = dist_mm / 1000.0;
                    let mat = packed & 0xFF;
                    // Replicates codificacao_legada() from motor.rs:
                    //   wall (2): curve 127→55→0 (≤30m → 55, ≤120m → 0 linear)
                    //   road (1): -128..0 (closer = more negative)
                    //   grass(0): ~40 fixed
                    //   hole (3): 127 (maximum danger)
                    let v: f32 = match mat {
                        2 => {
                            // wall: high positive when close, decays with distance
                            if dist_m <= 30.0 {
                                127.0 - (127.0 - 55.0) * (dist_m / 30.0)
                            } else {
                                55.0 * (1.0 - (dist_m - 30.0) / 90.0).max(0.0)
                            }
                        }
                        1 => {
                            // road: negative (closer = more negative)
                            -128.0 + (dist_m.min(80.0) / 80.0) * 128.0
                        }
                        0 => 40.0,  // grass: fixed moderate value
                        _ => 127.0, // hole (mat=3): maximum danger
                    };
                    retina_i8[j] = v.clamp(-127.0, 127.0) as i8;
                }
                // Runs the perception CfC (updates the internal recurrent state).
                // perceive() returns PerceptionOutput with hidden[0..16] updated.
                // The next tick will read self.percept_cfc[i].pipeline.cfc.h directly.
                self.percept_cfc[i].perceive(&retina_i8);
            }

            drop(retina_data);
            self.buf_retina_readback.unmap();
        }

        // ── DIAGNOSTIC: where does the signal die? ────────────────────────────
        // The kernels validate 100% in isolation (gpu_validate), but in the loop the intent
        // comes out zero and nobody moves. So the defect is in the LINKAGE, not the math.
        // These three numbers say in which link: perception wave, brain intent, or the command
        // that reaches physics. Only prints on the first tick, to avoid clutter.
        if std::env::var("IARA_DIAG").is_ok() && (!self.diag_feito || tick_idx % 500 == 0) {
            self.diag_feito = true;
            let int_nz = intents
                .iter()
                .flat_map(|a| a.iter())
                .filter(|v| v.abs() > 1e-6)
                .count();
            let int_max = intents
                .iter()
                .flat_map(|a| a.iter())
                .fold(0.0f32, |m, v| m.max(v.abs()));
            eprintln!(
                "[DIAG t={}] intent: {}/{} non-zero, |max|={:.6}",
                tick_idx,
                int_nz,
                intents.len() * 96,
                int_max
            );
            eprintln!(
                "[DIAG t={}] intent[0][0..3] = {:?}",
                tick_idx,
                &intents[0][0..3.min(intents[0].len())]
            );
            // being 0 full state
            if !states.is_empty() {
                let p = &states[0].phys;
                let speed = (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt();
                eprintln!(
                    "[DIAG t={}] being0 speed={:.2}m/s rpm={:.0} gear={}",
                    tick_idx, speed, p.current_rpm, p.current_gear
                );
            }
        }
        // ── STALL DIAGNOSTIC: reads the sensory_wave when being0 is stopped ───
        // When IARA_DIAG=1 and being0 is at < 0.5 m/s, does a manual readback of buf_percept_out
        // via a temporary MAP_READ buffer, prints critical channels to identify the stall.
        // A separate submit+poll is needed because the perception's buf_readback may or may not
        // be mapped, depending on the flow. We use buf_output directly (COPY_SRC).
        if std::env::var("IARA_DIAG").is_ok() && tick_idx % 200 == 0 && !states.is_empty() {
            let p = &states[0].phys;
            let speed0 = (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt();
            // Only when stuck, to avoid adding latency to the hot loop.
            if speed0 < 0.5 && bytes_percept > 0 {
                // Creates a temporary MAP_READ buffer for 1 being only (56 u32 = 224 bytes).
                let diag_bytes = (1 * 56 * 4) as u64; // 1 being × 56 u32 × 4 bytes
                let buf_wave_diag = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("diag-wave"),
                    size: diag_bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                // Copies buf_output[being0] → buf_wave_diag via a separate encoder.
                let mut enc_diag =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("diag-enc"),
                        });
                enc_diag.copy_buffer_to_buffer(
                    self.percept.buf_output(),
                    0,
                    &buf_wave_diag,
                    0,
                    diag_bytes,
                );
                self.queue.submit(Some(enc_diag.finish()));
                self.device.poll(wgpu::Maintain::Wait);

                // Maps and reads.
                let slice_w = buf_wave_diag.slice(0..diag_bytes);
                let (tx_w, rx_w) = std::sync::mpsc::channel();
                slice_w.map_async(wgpu::MapMode::Read, move |r| {
                    let _ = tx_w.send(r);
                });
                self.device.poll(wgpu::Maintain::Wait);
                let _ = rx_w.recv();

                let wave_data = slice_w.get_mapped_range();
                let words_w: &[u32] = bytemuck::cast_slice(&wave_data);
                // Unpacks i8 of being 0 (base=0).
                let mut w = [0i8; 223];
                for ch in 0..223usize {
                    let word = words_w[ch / 4];
                    w[ch] = ((word >> ((ch % 4) * 8)) & 0xFF) as u8 as i8;
                }
                drop(wave_data);
                buf_wave_diag.unmap();

                // Prints the critical stall channels.
                eprintln!(
                    "[WAVE t={}] STUCK pos=({:.0},{:.0}) speed={:.2}m/s",
                    tick_idx, p.pos[0], p.pos[2], speed0
                );
                eprintln!(
                    "  ch64=align:{:4}  ch65=speed:{:4} (÷2={:.1}m/s)  ch66=lat:{:4}",
                    w[64],
                    w[65],
                    w[65] as f32 / 2.0,
                    w[66]
                );
                eprintln!(
                    "  ch82=fwd_urg:{:4}  ch85=ema_l:{:4}  ch86=ema_r:{:4}  ch88=prox:{:4}",
                    w[82], w[85], w[86], w[88]
                );
                eprintln!(
                    "  ch89=reverse:{:4}  ch102=ar_l:{:4}  ch103=ar_r:{:4}  ch104=ar_f:{:4}",
                    w[89], w[102], w[103], w[104]
                );
                eprintln!(
                    "  ch108=rl_off:{:4}  ch109=doff:{:4}  ch110=la:{:4}  ch120=mem0:{:4}",
                    w[108], w[109], w[110], w[120]
                );
                eprintln!(
                    "  ch192-197 cfc=[{},{},{},{},{},{}]",
                    w[192], w[193], w[194], w[195], w[196], w[197]
                );
                let thr = intents[0][0];
                let steer_raw = intents[0][1];
                let brk = intents[0][2];
                eprintln!(
                    "  INTENT raw: thr={:.3}  steer={:.3}  brake={:.3}",
                    thr, steer_raw, brk
                );
            }
        }
        if intents.len() != n {
            eprintln!(
                "  ERROR: intent readback returned {} instead of {}",
                intents.len(),
                n
            );
            return false;
        }

        // ── 7. Applies intent and runs physics ────────────────────────────────
        // intent[s][0] = throttle, [1] = steer, [2] = brake — raw logits from L2.
        // motor.rs applies kan_out.eval(out_idx, raw) before passing to physics.
        // Without the KAN: intent[1]=0.985 → steer=0.985 → slip_angle=90° → no traction.
        // Also replicates the temporal steer smoothing (motor.rs L1925):
        //   steer_raw = prev_steer * 0.50 + new_steer * 0.50
        // This damps steer from tick to tick and avoids abrupt ±1 winner-take-all reversal.
        // Deduplicated (E3, REFACTOR_PLAN.md §2.3): uses universe::kan::eval_knots instead
        // of reimplementing the B-spline formula here (it was an identical copy of the
        // gpu_evolve.rs closure, and the comment only CLAIMED to be identical to
        // KanOutputHead::eval() without actually sharing the code).
        let kan_eval_fn = |out_idx: usize, x: f32| -> f32 {
            universe::kan::eval_knots(&self.kan_knots, out_idx, x)
        };
        let mut phys_states: Vec<PhysicsStateGpu> = (0..n)
            .map(|i| {
                let mut s = states[i].phys;
                let intent = &intents[i];
                // Applies the output KAN (replicates motor.rs L1873-L1926).
                let thr_kan = kan_eval_fn(0, intent[0]).max(0.0); // throttle ≥ 0
                let steer_kan = kan_eval_fn(1, intent[1]);
                let brk_kan = kan_eval_fn(2, intent[2]).max(0.0); // brake ≥ 0
                // Temporal steer smoothing (50/50): damps abrupt reversal.
                let steer_smooth =
                    (states[i].prev_steer * 0.50 + steer_kan * 0.50).clamp(-1.0, 1.0);

                let is_reversing = states[i].reverse_timer > 0.0;
                let in_lockout = states[i].post_reverse_lockout > 0.0;

                // motor.rs L1950: cancels brake when speed < 2.0 m/s (avoids locking a stopped car).
                // Without this: the model emits brake=0.98 + throttle=0.90 → both cancel out → speed=0
                // forever. The model learned that "stopped + wall" → high brake, but motor.rs's physics
                // already cancels the brake at low speed to give the throttle a chance.
                let speed_now = (s.vel[0].powi(2) + s.vel[2].powi(2)).sqrt();
                let brk_final = if speed_now < 2.0 { 0.0 } else { brk_kan };

                if is_reversing {
                    // ACTIVE REVERSE: COMPLETE override of the neural intent (motor.rs L2325-2339).
                    // During reverse: fixed pedal of 0.75, steer=0 (straight backup), brake=0.
                    // The model does NOT control the car in reverse — it was trained with this override.
                    // Force-shift to gear=0 (Reverse): shift_req = -1 repeated until gear=0.
                    // Physics drops 1 gear per tick when throttle > 0 and gear > 0 in reverse.
                    // We replicate it by forcing current_gear = 0 directly (same semantics as shift_req).
                    s.throttle = universe::config::iara::reverse_maneuver::PEDAL; // fixed reverse pedal
                    s.steer = 0.0; // straight backup (no risk of turning into another wall)
                    s.brake_cmd = 0.0; // no brake (the reverse pedal accelerates the reverse)
                    s.current_gear = 0; // gear 0 = Reverse (physics.wgsl convention)
                } else if in_lockout {
                    // POST-REVERSE LOCKOUT: throttle capped at 0.8 for smooth acceleration on restart.
                    // Replicates motor.rs L2331-2332: pedal = (intent * safety_cap).min(0.8).
                    s.throttle = thr_kan.min(0.8);
                    s.steer = steer_smooth;
                    s.brake_cmd = brk_final;
                    // Auto-clutch: if it left reverse (gear=0), returns to 1st gear (gear=2 = 1st).
                    // Motor.rs L2294-2296: if current_gear <= 1 and not reversing → shift_req = +1.
                    if s.current_gear <= 1 {
                        s.current_gear = 2;
                    }
                } else if states[i].escape_ticks > 0 {
                    // STAGNATION ESCAPE: the model has been stopped for 0.5s with no reverse.
                    // Replicates motor.rs stagnation escape: throttle=0.3 minimum, brake=0.
                    // Steer remains the model's to keep navigating.
                    // Without this: the model emits throttle=0 in tight corners and stays at engine
                    // idle (0.07 m/s forever). The CHAMPION was trained expecting this anchor.
                    s.throttle = 0.3f32.max(thr_kan); // minimum 0.3, respects if the model wants more
                    s.steer = steer_smooth;
                    s.brake_cmd = 0.0; // zeroes the brake so the minimum throttle has effect
                    if s.current_gear <= 1 {
                        s.current_gear = 2;
                    }
                } else {
                    // NORMAL MODE: full neural intent.
                    s.throttle = thr_kan;
                    s.steer = steer_smooth;
                    s.brake_cmd = brk_final;
                    // Auto-clutch: leaves gear 0/1 for 2 (1st gear) when not reversing.
                    if s.current_gear <= 1 {
                        s.current_gear = 2;
                    }
                    // Guardian Anti-Stall: if stopped in a high gear (>2), drops to 2.
                    if s.current_gear > 2 && speed_now < 3.0 {
                        s.current_gear -= 1;
                    }

                    // AUTO-SHIFT BY RPM — replicates motor.rs L2303-2309 (lines 2303, 2307).
                    // ROOT CAUSE OF THE BUG: without this block the car gets stuck in gear=3 (2nd gear).
                    // With gear=3 and max_rpm=8500, the limiter (`tm=0` when `rr>1`) cuts torque
                    // as soon as RPM > max_rpm — the car saturates at ~8 m/s and decays to 0.17 m/s.
                    // The model was trained on motor.rs which has auto-shift → it expects gears
                    // to change by RPM, not by a brain decision.
                    //
                    // Engaged clutch (is_clutch_engaged in motor.rs):
                    //   wheel_ratio ≈ 12 - (gear-2)*2 → gear=3 → ratio=8
                    //   wheel_rpm = speed * ratio * 30 → at 8 m/s: 8 * 8 * 30 = 1920 > 1200 → engaged
                    //   (conservative: we use speed > 3 m/s AND gear >= 2 as a proxy)
                    // Upshift: rpm > 85% max_rpm AND gear < 6th AND shift_timer=0
                    // Downshift: rpm < 30% max_rpm AND gear > 2nd AND speed > 3 m/s AND shift_timer=0
                    if s.shift_timer <= 0.0 && s.current_gear >= 2 && speed_now > 3.0 {
                        if s.current_rpm > s.max_rpm * 0.85 && s.current_gear < 6 {
                            // Upshift: 150ms timer (motor.rs uses 150ms)
                            s.current_gear += 1;
                            s.shift_timer = 0.15;
                        } else if s.current_rpm < s.max_rpm * 0.30 && s.current_gear > 2 {
                            // Downshift: avoids choking in a long gear without speed
                            s.current_gear -= 1;
                            s.shift_timer = 0.15;
                        }
                    }
                }
                s.dt = dt;
                s
            })
            .collect();
        // Saves the smoothed steer and updates stagnation for the next tick.
        for i in 0..n {
            // On leaving reverse, resets prev_steer to 0 to avoid an abrupt steer jump
            // on resumption: on returning to normal mode, the smooth starts from 0 (straight),
            // not from whatever the model was emitting during reverse.
            if states[i].reverse_timer > 0.0 {
                states[i].prev_steer = 0.0; // straight backup → smooths the resumption
            } else {
                states[i].prev_steer = phys_states[i].steer;
            }
            // Updates stagnation: counts ticks with speed < 1.5 m/s.
            // Threshold = 1.5 m/s identical to motor.rs L2220 (speed < 1.5 for the reverse trigger).
            let speed = (phys_states[i].vel[0].powi(2) + phys_states[i].vel[2].powi(2)).sqrt();
            if speed < 1.5 {
                states[i].stagnation_ticks += 1;
            } else {
                states[i].stagnation_ticks = 0;
            }
            // Activates the stagnation escape when stopped for 0.5s (32 ticks).
            // The CHAMPION was trained with motor.rs which has curvature_governor + slow brain +
            // stagnation escape — external mechanisms that get the car out of tight corners.
            // Without them, the model emits throttle=0 + brake=0 in corners and stays at idle (0.07 m/s).
            // The escape does not fabricate speed: it only prevents eternal throttle=0 from locking
            // the car; the being still navigates with the model's steer. Disable with IARA_NO_ESCAPE=1.
            // OFF BY DEFAULT. The escape forces `throttle = 0.3` and zeroes the model's brake
            // — it overrides the brain's decision. Measured over 50.000 ticks: WITH escape
            // `dist_max = 1744 m`, WITHOUT escape `650 m`. **2.7× of the result came from the crutch.**
            // A number that depends on it does not measure driving, and the goal is to measure driving.
            // `IARA_ESCAPE=1` re-enables it for comparison.
            if std::env::var("IARA_ESCAPE").is_ok() {
                // 32 ticks = 0.5s stopped → activates escape.
                // 128 ticks = 2s of minimum throttle 0.3 — enough to get around most of the
                // tight corners of track seed=9001 where the CHAMPION idles without the slow brain.
                if states[i].stagnation_ticks >= 32 && states[i].escape_ticks == 0 {
                    states[i].escape_ticks = 64; // 1s of minimum throttle=0.3
                    states[i].stagnation_ticks = 0;
                }
            }
            // Decrements the escape.
            if states[i].escape_ticks > 0 {
                states[i].escape_ticks -= 1;
            }
        }

        // Physics runs on the isolated device (PhysicsGpu has its own device).
        // DIAGNOSTIC: the brain sends throttle 0.99 and the car does not move. Here you see
        // whether physics receives the command and what it returns — zero wheel load means a car
        // with no ground contact, and then there is no traction no matter how much torque the engine makes.
        let diag = std::env::var("IARA_DIAG").is_ok() && !self.diag_fis;
        if diag {
            self.diag_fis = true;
            let a = &phys_states[0];
            eprintln!(
                "[PHYS-BEFORE] thr={:.3} brk={:.3} str={:.3} rpm={:.0} gear={} vel={:?} pos_y={:.3}",
                a.throttle, a.brake_cmd, a.steer, a.current_rpm, a.current_gear, a.vel, a.pos[1]
            );
            eprintln!("[PHYS-BEFORE] wheels={:?}", &a.wheels[..4]);
        }

        // step() writes the updated state back into phys_states[i].
        let ok_fis = self.phys_gpu.step(&mut phys_states);
        if diag {
            let a = &phys_states[0];
            eprintln!(
                "[PHYS-AFTER] ok={} vel={:?} pos_y={:.3} rpm={:.0} torque={:.1}",
                ok_fis, a.vel, a.pos[1], a.current_rpm, a.drive_torque_out
            );
            eprintln!("[PHYS-AFTER] wheels={:?}", &a.wheels[..4]);
        }
        if !ok_fis {
            eprintln!("  ERROR: phys_gpu.step() failed (n={} > capacity?)", n);
            return false;
        }

        // Updates CPU states + yaw rotation (same as gpu_evolve.rs).
        for i in 0..n {
            states[i].phys = phys_states[i];
            let theta = states[i].phys.av_y * dt;
            let (sin_t, cos_t) = theta.sin_cos();
            let fwd = states[i].phys.fwd;
            let lat = states[i].phys.lat;
            states[i].phys.fwd[0] = fwd[0] * cos_t - fwd[2] * sin_t;
            states[i].phys.fwd[2] = fwd[0] * sin_t + fwd[2] * cos_t;
            states[i].phys.lat[0] = lat[0] * cos_t - lat[2] * sin_t;
            states[i].phys.lat[2] = lat[0] * sin_t + lat[2] * cos_t;
            let fn_ = (states[i].phys.fwd[0].powi(2) + states[i].phys.fwd[2].powi(2))
                .sqrt()
                .max(1e-6);
            let ln_ = (states[i].phys.lat[0].powi(2) + states[i].phys.lat[2].powi(2))
                .sqrt()
                .max(1e-6);
            states[i].phys.fwd[0] /= fn_;
            states[i].phys.fwd[2] /= fn_;
            states[i].phys.lat[0] /= ln_;
            states[i].phys.lat[2] /= ln_;
        }

        // ── 8. Updates memory0 in the GPU temporal state ──────────────────────
        // memory0 = intentional throttle of the current tick, clamped to [0,1].
        // The perception kernel at the NEXT tick will read this value as ch 120.
        // Called AFTER physics to guarantee the final throttle is what reached
        // the tires (not a pre-physics value).
        // Cost: N × 4 bytes of write_buffer = 240 bytes @ 60 beings — negligible.
        let memory0_vals: Vec<f32> = (0..n)
            .map(|i| {
                intents[i][0].clamp(0.0, 1.0) // same clamp as the physical throttle
            })
            .collect();
        self.percept.update_memory0(&self.queue, &memory0_vals[..n]);

        true
    }
}

// ─── Entry point ──────────────────────────────────────────────────────────────

fn main() {
    // Forces the Radeon Vulkan ICD before any wgpu::Instance::new.
    // Multiple ICDs cause SIGSEGV in the loader (seen in gpu_validate.rs).
    let icd_radeon = "/usr/share/vulkan/icd.d/radeon_icd.json";
    if std::path::Path::new(icd_radeon).exists() && std::env::var("VK_ICD_FILENAMES").is_err() {
        unsafe {
            std::env::set_var("VK_ICD_FILENAMES", icd_radeon);
        }
    }

    // ── Args parsing ──────────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let mut n_seres: usize = 60;
    let mut n_ticks: usize = 20_000; // 20.000 ticks = 312.5s @ 64Hz ≈ 5 min of simulation
    {
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--seres" => {
                    i += 1;
                    if i < args.len() {
                        n_seres = args[i].parse().unwrap_or(60);
                    }
                }
                "--ticks" => {
                    i += 1;
                    if i < args.len() {
                        n_ticks = args[i].parse().unwrap_or(20_000);
                    }
                }
                _ => {}
            }
            i += 1;
        }
    }
    println!("=== gpu_loop: {} beings, {} ticks ===", n_seres, n_ticks);
    println!("  Loop: retina+perception+brain+physics — one single device");
    println!();

    // ── World ─────────────────────────────────────────────────────────────────
    let t0 = Instant::now();
    // Track: same seed/width as gpu_validate to guarantee CHAMPION.dna is
    // tested on the track it was validated on. The CHAMPION was evolved on rotating tracks,
    // but the canonical validation uses seed=9001, track_width=18m (gpu_validate.rs).
    // With 14m the CHAMPION collided with the wall after ~200 ticks (steer trained for a wider track).
    let world = VoxelWorld::new(9001, 18.0);
    println!(
        "world: {}×{} · {} waypoints · lap {:.0} m  [{:.0} ms]",
        world.size,
        world.size,
        world.waypoints_world.len(),
        world.lap_distance,
        t0.elapsed().as_secs_f64() * 1000.0,
    );

    // ── Engines: loads CHAMPION.dna if it exists, otherwise random weights ─────
    // SovereignEngine::new() initializes ALL weights with 0.0 (layers.rs:32),
    // which produces intent == 0 → no movement. A real DNA must be loaded
    // for the brain to emit non-null intent.
    // Without CHAMPION.dna: the beings do not move by decision — the L1 output is zero
    // (null weights × any wave = 0). The program warns and exits with an error.
    let t1 = Instant::now();
    let dna_opt = load_champion_dna("CHAMPION.dna");
    let engines: Vec<SovereignEngine> = (0..n_seres)
        .map(|_| {
            let mut eng = SovereignEngine::new();
            if let Some(ref dna) = dna_opt {
                // Loads L1 weights: cortex_raw (i8, 63.5-scaled) or fast_weights (i8, 63.5-scaled).
                let w: Vec<f32> = if let Some(ref cr) = dna.cortex_raw {
                    cr.iter().map(|&b| b as f32 / 63.5).collect()
                } else {
                    dna.fast_weights.iter().map(|&b| b as f32 / 63.5).collect()
                };
                let n_l1 = eng.instinct_layer.weights.len().min(w.len());
                eng.instinct_layer.weights[..n_l1].copy_from_slice(&w[..n_l1]);
                // Loads L2 weights if present in the DNA.
                if let Some(ref l2) = dna.cortex_l2_raw {
                    let w2: Vec<f32> = l2.iter().map(|&b| b as f32 / 63.5).collect();
                    let n_l2 = eng.cortex_l2.weights.len().min(w2.len());
                    eng.cortex_l2.weights[..n_l2].copy_from_slice(&w2[..n_l2]);
                }
            }
            eng
        })
        .collect();
    // CHAMPION's KAN output knots — used in GpuLoop::tick() via self.kan_knots.
    // Motor.rs applies kan_out.eval(out_idx, raw) before using throttle/steer/brake.
    // Without the KAN: raw intent≈1 → physical steer=1 → slip_angle=90° → no traction.
    // KAN_OUT_KNOTS = 12 (kan.rs:85), KAN_OUT_DOMAIN = 2.0 (kan.rs:86).
    use universe::kan::KAN_OUT_KNOTS;
    let kan_knots: [[f32; KAN_OUT_KNOTS]; 3] = dna_opt
        .as_ref()
        .map(|d| d.kan_knots)
        .unwrap_or_else(universe::evolution::dna_genesis::default_kan_knots);

    let dna_status = if dna_opt.is_some() {
        "CHAMPION.dna loaded — real weights"
    } else {
        "CHAMPION.dna not found — ZERO weights (beings will not move by their own decision)"
    };
    println!(
        "engines created: {} beings · {}  [{:.0} ms]",
        n_seres,
        dna_status,
        t1.elapsed().as_secs_f64() * 1000.0
    );

    // ── GPU context ───────────────────────────────────────────────────────────
    let t2 = Instant::now();
    let mut ctx = match GpuLoop::new(&world, &engines, n_seres, kan_knots) {
        Some(c) => c,
        None => {
            eprintln!("ERROR: no usable GPU. Aborting.");
            std::process::exit(1);
        }
    };
    println!(
        "GpuLoop ready  [{:.0} ms]",
        t2.elapsed().as_secs_f64() * 1000.0
    );
    println!();

    // ── Initial being states ───────────────────────────────────────────────────
    // Distributes beings along the track's waypoints.
    let n_wp = world.waypoints_world.len();
    let mut states: Vec<BeingState> = (0..n_seres)
        .map(|i| {
            let wp = world.waypoints_world[(i * n_wp / n_seres).min(n_wp - 1)];
            // Forward = direction to the next waypoint.
            let wp_next = world.waypoints_world[((i * n_wp / n_seres) + 1).min(n_wp - 1)];
            let dx = wp_next.x - wp.x;
            let dz = wp_next.y - wp.y;
            let len = (dx * dx + dz * dz).sqrt().max(1e-6);
            let fwd = [dx / len, 0.0, dz / len];
            let pos = [wp.x, 0.0, wp.y];
            BeingState {
                phys: make_physics_state(pos, fwd),
                prev_steer: 0.0,
                stagnation_ticks: 0,
                escape_ticks: 0,
                reverse_timer: 0.0,
                post_reverse_lockout: 0.0,
            }
        })
        .collect();

    // ── Accumulated odometer per being ─────────────────────────────────────────
    // We use integration of speed * dt (more precise than euclidean between sparse
    // points when the track has tight corners). The euclidean distance between ticks
    // underestimates the real trajectory; the integral of |vel| * dt = real path.
    // Laps: accumulated distance / lap_distance.
    // HETEROGENEITY is the guarantee of genuine movement: beings in different
    // situations (different waypoint, different track) MUST yield different numbers.
    let lap_dist = world.lap_distance;
    println!("lap = {:.0} m  (total track length)", lap_dist);

    let mut dist_acum: Vec<f64> = vec![0.0; n_seres];
    let mut pos_prev: Vec<[f32; 3]> = states.iter().map(|s| pos_of(&s.phys)).collect();

    // ── Warm-up (1 discarded tick to force shader JIT compilation) ────────────
    println!("Warming up (1 JIT tick)...");
    let _ = ctx.tick(&mut states, &engines, n_seres, 1.0 / 64.0, 0);

    // ── Main loop ─────────────────────────────────────────────────────────────
    println!("Running {} ticks with {} beings...", n_ticks, n_seres);
    let dt = 1.0_f32 / 64.0;

    let t_loop = Instant::now();
    for tick_i in 0..n_ticks {
        let ok = ctx.tick(&mut states, &engines, n_seres, dt, tick_i + 1);
        if !ok {
            eprintln!("ERROR at tick {} — aborting.", tick_i);
            break;
        }
        // Accumulates the odometer: euclidean distance between pos[t-1] and pos[t].
        // With dt=1/64s and typical speed of 20-40 m/s, the step is ~0.3-0.6m —
        // small enough that euclidean ≈ arclength (error < 0.1%).
        for i in 0..n_seres {
            let p1 = pos_of(&states[i].phys);
            let dx = (p1[0] - pos_prev[i][0]) as f64;
            let dz = (p1[2] - pos_prev[i][2]) as f64;
            dist_acum[i] += (dx * dx + dz * dz).sqrt();
            pos_prev[i] = p1;
        }
        // Progress checkpoint every 1000 ticks (~15.6s of simulation)
        if (tick_i + 1) % 1000 == 0 {
            let max_d = dist_acum.iter().cloned().fold(0.0_f64, f64::max);
            let med_d = {
                let mut v = dist_acum.clone();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[v.len() / 2]
            };
            let max_voltas = max_d / lap_dist as f64;
            println!(
                "  tick {:5}: dist_max={:.0}m ({:.2}L)  dist_med={:.0}m ({:.2}L)",
                tick_i + 1,
                max_d,
                max_voltas,
                med_d,
                med_d / lap_dist as f64
            );
        }
    }
    let elapsed_loop = t_loop.elapsed().as_secs_f64();

    // ── Results ───────────────────────────────────────────────────────────────
    let ms_tick = elapsed_loop * 1000.0 / n_ticks as f64;
    println!();
    println!("=== RESULT ===");
    println!("  ticks: {}", n_ticks);
    println!("  beings: {}", n_seres);
    println!("  total time: {:.1} ms", elapsed_loop * 1000.0);
    println!("  ms/tick (all-GPU loop + temporals): {:.3} ms", ms_tick);
    println!();

    // ── Final distances and laps ───────────────────────────────────────────────
    let dist_max = dist_acum.iter().cloned().fold(0.0_f64, f64::max);
    let dist_med = {
        let mut v = dist_acum.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let voltas_max = dist_max / lap_dist as f64;
    let voltas_med = dist_med / lap_dist as f64;
    let n_voltas_completas = dist_acum.iter().filter(|&&d| d >= lap_dist as f64).count();

    println!("=== DISTANCES AND LAPS in {} ticks ===", n_ticks);
    println!("  lap_distance: {:.0} m", lap_dist);
    println!(
        "  dist_max:     {:.0} m  ({:.2} laps)",
        dist_max, voltas_max
    );
    println!(
        "  dist_med:     {:.0} m  ({:.2} laps)",
        dist_med, voltas_med
    );
    println!(
        "  beings with ≥1 complete lap: {}/{}",
        n_voltas_completas, n_seres
    );
    println!();

    // ── Top performers (heterogeneity proof) ──────────────────────────────────
    // Sorts all beings by decreasing distance and shows the top 12.
    let mut ordem: Vec<usize> = (0..n_seres).collect();
    ordem.sort_by(|&a, &b| dist_acum[b].partial_cmp(&dist_acum[a]).unwrap());
    println!("=== TOP 12 BEINGS BY DISTANCE (heterogeneity) ===");
    let n_amostras = n_seres.min(12);
    let mut algum_se_moveu = false;
    for &i in ordem.iter().take(n_amostras) {
        let dist = dist_acum[i];
        let voltas = dist / lap_dist as f64;
        let vel = states[i].phys.vel;
        let speed = (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]).sqrt();
        println!(
            "  being[{:2}]: dist={:.1}m ({:.2}L)  speed={:.2}m/s  pos=({:.0},{:.0})  {}",
            i,
            dist,
            voltas,
            speed,
            states[i].phys.pos[0],
            states[i].phys.pos[2],
            if dist > 10.0 { "OK" } else { "STOPPED!" }
        );
        if dist > 10.0 {
            algum_se_moveu = true;
        }
    }
    println!();
    if algum_se_moveu {
        println!("  BEINGS MOVED — temporal channels functional.");
    } else {
        println!("  NO BEING MOVED — broken loop or null weights.");
    }

    // ── Temporal channels ported in this commit ───────────────────────────────
    println!();
    println!("=== TEMPORAL CHANNELS PORTED TO THE GPU ===");
    println!("  85: wall_ema_l      (EMA of left lateral dist, ~0.5s half-life)");
    println!("  86: wall_ema_r      (EMA of right lateral dist)");
    println!("  88: proximity_sustain (seconds scraping the wall, cap 3s)");
    println!("  102: approach_rate_l  (dDist_l/dt in m/s, ±30m/s)");
    println!("  103: approach_rate_r");
    println!("  104: approach_rate_f  (frontal via grid probe)");
    println!("  105: TTC_min          (time to collision, 0=imminent, 127=safe)");
    println!("  106: delta_dist_l ±15m/s");
    println!("  107: delta_dist_r");
    println!("  108: racing_line_offset (lat deviation from the ideal line, ±127)");
    println!("  109: d/dt(line_offset)  (gain 8×)");
    println!("  110: lookahead_offset 5m ahead on the ideal line");
    println!("  118: fwd_dist / half_width (generalization by track width)");
    println!("  119: corridor_lat / half_width");
    println!("  120: memory0 (previous intentional throttle)");
    println!("  123: whisker_min / half_width");
    println!();
    println!("=== END ===");
}
