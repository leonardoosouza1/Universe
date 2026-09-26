//! Resident GPU adapter for the existing native action mapper.
//!
//! It preserves `gpu_evolve/native_action.rs` ordering and constants. Hip bias is
//! intentionally zero for this first residency step; a later binding can provide
//! the already resident hippocampus output without changing the mapper contract.

use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
#[cfg(test)]
use wgpu::util::DeviceExt;

pub const INTENT_STRIDE: usize = 96;
pub const SLOW_STRIDE: usize = 3;
pub const HEAD_STRIDE: usize = 36;
pub const PHYSICS_STRIDE: usize = 92;
pub const KNOTS: usize = 12;
pub const FLAG_LINEAR_HEAD: u32 = 1;
pub const FLAG_SIGNED_THROTTLE: u32 = 2;
pub const FLAG_ALLOW_LOW_SPEED_BRAKE: u32 = 4;
/// Resolve the two antagonistic longitudinal outputs into one command.  Without
/// this, a controller can press throttle and brake together; in particular the
/// static-rest constraint can never engage while throttle remains positive.
pub const FLAG_EXCLUSIVE_PEDALS: u32 = 8;
/// Push the negative half of the fast brake logit into the exclusive net as extra
/// drive instead of clipping it, so the brake row has no dead zone under mutation
/// (docs/SCHOOL-STOP-CONTRACT-AB-20260924.md, signed-brake battery). Requires
/// `FLAG_EXCLUSIVE_PEDALS`; the slow inhibition path stays rectified.
pub const FLAG_SIGNED_BRAKE: u32 = 16;
/// Preserve positive brake authority instead of subtracting requested throttle.
/// Requires exclusive pedals and low-speed braking; incompatible with signed brake.
pub const FLAG_BRAKE_PRIORITY: u32 = 32;
pub const GEARBOX_OFF: u32 = 0;
pub const GEARBOX_AUTO: u32 = 1;
pub const GEARBOX_NEURAL: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ActionParams {
    pub n: u32,
    pub flags: u32,
    pub steering_alpha: f32,
    pub slow_steering_authority: f32,
    pub gearbox_mode: u32,
    pub gear_cap: u32,
    pub dt: f32,
    pub _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GearState {
    pub cooldown: f32,
    pub prev_up: f32,
    pub prev_down: f32,
    pub gear: u32,
}

pub struct OrganismActionGpu {
    pub buf_intent: wgpu::Buffer,
    pub buf_slow: wgpu::Buffer,
    pub buf_heads: wgpu::Buffer,
    pub buf_hip_bias: wgpu::Buffer,
    pub buf_gear_state: wgpu::Buffer,
    pub buf_previous_steer: wgpu::Buffer,
    /// 16 f32/being: after_head[3], after_modulation[3], pre_speed, filtered_steer,
    /// low_speed_brake_suppressed (0/1), reserved zeros[7]. Diagnostic only.
    pub buf_trace: wgpu::Buffer,
    params: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    pub capacity: usize,
}

impl OrganismActionGpu {
    pub fn new(device: &wgpu::Device, capacity: usize) -> Self {
        let capacity = capacity.max(1);
        let storage = |label: &str, bytes: u64, copy_dst: bool| {
            let mut usage = wgpu::BufferUsages::STORAGE;
            if copy_dst {
                usage |= wgpu::BufferUsages::COPY_DST;
            }
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage,
                mapped_at_creation: false,
            })
        };
        let buf_intent = storage(
            "organism-action-intent96",
            (capacity * INTENT_STRIDE * 4) as u64,
            true,
        );
        let buf_slow = storage(
            "organism-action-slow3",
            (capacity * SLOW_STRIDE * 4) as u64,
            true,
        );
        let buf_heads = storage(
            "organism-action-heads",
            (capacity * HEAD_STRIDE * 4) as u64,
            true,
        );
        let buf_hip_bias = storage("organism-action-hip-bias", (capacity * 3 * 4) as u64, true);
        let buf_gear_state = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("organism-action-gear-state"),
            size: (capacity * std::mem::size_of::<GearState>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_previous_steer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("organism-action-previous-steer"),
            size: (capacity * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_trace = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("organism action trace"),
            size: (capacity * 16 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("organism-action-params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("organism-action"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("organism_action.wgsl"))),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("organism-action-layout"),
            entries: &[
                uniform(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
                storage_entry(5, true),
                storage_entry(6, false),
                storage_entry(7, false),
                storage_entry(8, false),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("organism-action-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("organism-action-pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            buf_intent,
            buf_slow,
            buf_heads,
            buf_hip_bias,
            buf_gear_state,
            buf_previous_steer,
            buf_trace,
            params,
            bind_group_layout,
            pipeline,
            capacity,
        }
    }

    pub fn upload_heads(
        &self,
        queue: &wgpu::Queue,
        heads: &[[[f32; KNOTS]; 3]],
    ) -> Result<(), &'static str> {
        if heads.len() > self.capacity {
            return Err("organism-action: head count exceeds capacity");
        }
        if heads
            .iter()
            .flatten()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err("organism-action: non-finite head knot");
        }
        let flat: Vec<f32> = heads
            .iter()
            .flat_map(|h| h.iter().flat_map(|row| row.iter().copied()))
            .collect();
        queue.write_buffer(&self.buf_heads, 0, bytemuck::cast_slice(&flat));
        Ok(())
    }

    pub fn upload_hip_bias(
        &self,
        queue: &wgpu::Queue,
        bias: &[[f32; 3]],
    ) -> Result<(), &'static str> {
        if bias.len() > self.capacity {
            return Err("organism-action: hip bias count exceeds capacity");
        }
        if bias.iter().flatten().any(|value| !value.is_finite()) {
            return Err("organism-action: non-finite hip bias");
        }
        queue.write_buffer(&self.buf_hip_bias, 0, bytemuck::cast_slice(bias));
        Ok(())
    }

    pub fn upload_gear_state(
        &self,
        queue: &wgpu::Queue,
        states: &[GearState],
    ) -> Result<(), &'static str> {
        if states.len() > self.capacity {
            return Err("organism-action: gear state count exceeds capacity");
        }
        if states.iter().any(|s| {
            !s.cooldown.is_finite()
                || !s.prev_up.is_finite()
                || !s.prev_down.is_finite()
                || s.gear > 6
        }) {
            return Err("organism-action: invalid gear state");
        }
        queue.write_buffer(&self.buf_gear_state, 0, bytemuck::cast_slice(states));
        Ok(())
    }

    /// Reset a replaced slot only; never clear surviving controller state.
    pub fn reset_action_slot(
        &self,
        queue: &wgpu::Queue,
        slot: usize,
        gear: u32,
    ) -> Result<(), &'static str> {
        if slot >= self.capacity || gear > 6 {
            return Err("organism-action: invalid birth slot");
        }
        queue.write_buffer(
            &self.buf_trace,
            (slot * 16 * 4) as u64,
            bytemuck::cast_slice(&[0.0f32; 16]),
        );
        queue.write_buffer(
            &self.buf_gear_state,
            (slot * std::mem::size_of::<GearState>()) as u64,
            bytemuck::bytes_of(&GearState {
                gear,
                ..GearState::zeroed()
            }),
        );
        queue.write_buffer(
            &self.buf_previous_steer,
            (slot * 4) as u64,
            bytemuck::bytes_of(&0.0f32),
        );
        queue.write_buffer(
            &self.buf_intent,
            (slot * INTENT_STRIDE * 4) as u64,
            bytemuck::cast_slice(&[0.0f32; INTENT_STRIDE]),
        );
        queue.write_buffer(
            &self.buf_slow,
            (slot * SLOW_STRIDE * 4) as u64,
            bytemuck::cast_slice(&[0.0f32; SLOW_STRIDE]),
        );
        queue.write_buffer(
            &self.buf_hip_bias,
            (slot * 3 * 4) as u64,
            bytemuck::cast_slice(&[0.0f32; 3]),
        );
        Ok(())
    }

    /// Resident controller state that must travel with a runtime checkpoint.
    /// Intent/head/hip inputs are refreshed by the caller; these two buffers
    /// carry temporal state across ticks and therefore affect continuation.
    pub fn snapshot_buffers(&self) -> Vec<crate::gpu::state_snapshot::NamedBuffer<'_>> {
        vec![
            ("organism_action.previous_steer", &self.buf_previous_steer),
            ("organism_action.gear_state", &self.buf_gear_state),
        ]
    }

    pub fn encode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        physics_input: &wgpu::Buffer,
        n: usize,
        params: ActionParams,
    ) -> Result<(), &'static str> {
        if n == 0 || n > self.capacity {
            return Err("organism-action: invalid count");
        }
        if !params.steering_alpha.is_finite() || !params.slow_steering_authority.is_finite() {
            return Err("organism-action: non-finite parameters");
        }
        if !(0.0..=1.0).contains(&params.steering_alpha)
            || params.slow_steering_authority < 0.0
            || !params.dt.is_finite()
            || params.dt <= 0.0
            || params.gearbox_mode > GEARBOX_NEURAL
            || !(2..=6).contains(&params.gear_cap)
        {
            return Err("organism-action: invalid action parameters");
        }
        if params.flags
            & !(FLAG_LINEAR_HEAD
                | FLAG_SIGNED_THROTTLE
                | FLAG_ALLOW_LOW_SPEED_BRAKE
                | FLAG_EXCLUSIVE_PEDALS
                | FLAG_SIGNED_BRAKE
                | FLAG_BRAKE_PRIORITY)
            != 0
        {
            return Err("organism-action: unknown flags");
        }
        if params.flags & FLAG_SIGNED_BRAKE != 0 && params.flags & FLAG_EXCLUSIVE_PEDALS == 0 {
            return Err("organism-action: signed brake requires exclusive pedals");
        }
        if params.flags & FLAG_BRAKE_PRIORITY != 0
            && (params.flags & (FLAG_EXCLUSIVE_PEDALS | FLAG_ALLOW_LOW_SPEED_BRAKE)
                != (FLAG_EXCLUSIVE_PEDALS | FLAG_ALLOW_LOW_SPEED_BRAKE)
                || params.flags & FLAG_SIGNED_BRAKE != 0)
        {
            return Err(
                "organism-action: brake priority requires exclusive low-speed pedals and unsigned brake",
            );
        }
        queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&ActionParams {
                n: n as u32,
                ..params
            }),
        );
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("organism-action-bind-group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buf_intent.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_slow.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buf_heads.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buf_previous_steer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.buf_hip_bias.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.buf_gear_state.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: physics_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: self.buf_trace.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("organism-action"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        Ok(())
    }
}

fn uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shader_validates() {
        let module = naga::front::wgsl::parse_str(include_str!("organism_action.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    #[ignore = "exclusive real GPU; fixed resident command fixture"]
    fn gpu_fixed_signed_command_matches_mapper_contract() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("real Vulkan adapter");
        assert_ne!(
            adapter.get_info().device_type,
            wgpu::DeviceType::Cpu,
            "CPU fallback forbidden"
        );
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("organism-action-fixture"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 8,
                    ..wgpu::Limits::downlevel_defaults()
                },
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        ))
        .expect("real GPU device");
        let action = OrganismActionGpu::new(&device, 1);
        action.upload_heads(&queue, &[[[0.0; KNOTS]; 3]]).unwrap();
        action.upload_hip_bias(&queue, &[[0.0; 3]]).unwrap();
        action
            .upload_gear_state(
                &queue,
                &[GearState {
                    cooldown: 0.0,
                    prev_up: 0.0,
                    prev_down: 0.0,
                    gear: 2,
                }],
            )
            .unwrap();
        queue.write_buffer(
            &action.buf_intent,
            0,
            bytemuck::cast_slice(&[0.5f32, -0.4, 0.7]),
        );
        queue.write_buffer(
            &action.buf_slow,
            0,
            bytemuck::cast_slice(&[0.2f32, 0.5, -0.25]),
        );
        queue.write_buffer(&action.buf_previous_steer, 0, bytemuck::bytes_of(&0.1f32));
        let mut input = [0.0f32; PHYSICS_STRIDE];
        input[0] = 10.0;
        let physics = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("organism-action-physics"),
            contents: bytemuck::cast_slice(&input),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("organism-action-readback"),
            size: (PHYSICS_STRIDE * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("organism-action-fixture"),
        });
        action
            .encode(
                &device,
                &queue,
                &mut encoder,
                &physics,
                1,
                ActionParams {
                    n: 1,
                    flags: FLAG_LINEAR_HEAD | FLAG_SIGNED_THROTTLE,
                    steering_alpha: 0.5,
                    slow_steering_authority: 0.2,
                    gearbox_mode: GEARBOX_OFF,
                    gear_cap: 6,
                    dt: 1.0 / 64.0,
                    _pad: 0.0,
                },
            )
            .unwrap();
        encoder.copy_buffer_to_buffer(&physics, 0, &staging, 0, staging.size());
        queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range();
        let output: &[f32] = bytemuck::cast_slice(&mapped);
        assert_eq!(output[51].to_bits(), 0.6f32.to_bits());
        // Decimal -0.175 is rounded separately from the GPU fused arithmetic.
        assert!((output[53] + 0.175).abs() <= 2.0 * f32::EPSILON);
        assert_eq!(output[52].to_bits(), 1.0f32.to_bits());
        drop(mapped);
        staging.unmap();

        // The resident taxi exposes separate neural throttle/brake logits, but the
        // body has one longitudinal degree of freedom.  The scenario arbitration
        // subtracts the antagonists, so they can never fight in the applied command.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("organism-action-exclusive-pedals-fixture"),
        });
        action
            .encode(
                &device,
                &queue,
                &mut encoder,
                &physics,
                1,
                ActionParams {
                    n: 1,
                    flags: FLAG_LINEAR_HEAD
                        | FLAG_SIGNED_THROTTLE
                        | FLAG_ALLOW_LOW_SPEED_BRAKE
                        | FLAG_EXCLUSIVE_PEDALS,
                    steering_alpha: 0.5,
                    slow_steering_authority: 0.2,
                    gearbox_mode: GEARBOX_OFF,
                    gear_cap: 6,
                    dt: 1.0 / 64.0,
                    _pad: 0.0,
                },
            )
            .unwrap();
        encoder.copy_buffer_to_buffer(&physics, 0, &staging, 0, staging.size());
        queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range();
        let output: &[f32] = bytemuck::cast_slice(&mapped);
        assert_eq!(output[51].to_bits(), 0.0f32.to_bits());
        assert!(
            (output[52] - 0.4).abs() <= 2.0 * f32::EPSILON,
            "{}",
            output[52]
        );
        drop(mapped);
        staging.unmap();

        // Signed brake: a negative fast brake logit (-0.7) is pushed into the net as
        // drive. throttle_before = clamp(0.5 + 0.2*0.5, 0, 1) = 0.6, brake 0, so the
        // net is 1.3 and the applied throttle saturates at the cap (1.0). Without the
        // flag the same inputs give 0.6: the negative half was a dead zone.
        queue.write_buffer(
            &action.buf_intent,
            0,
            bytemuck::cast_slice(&[0.5f32, -0.4, -0.7]),
        );
        queue.write_buffer(
            &action.buf_slow,
            0,
            bytemuck::cast_slice(&[0.2f32, 0.0, -0.25]),
        );
        for (flags, expected_throttle) in [
            (
                FLAG_LINEAR_HEAD
                    | FLAG_SIGNED_THROTTLE
                    | FLAG_ALLOW_LOW_SPEED_BRAKE
                    | FLAG_EXCLUSIVE_PEDALS
                    | FLAG_SIGNED_BRAKE,
                1.0f32,
            ),
            (
                FLAG_LINEAR_HEAD
                    | FLAG_SIGNED_THROTTLE
                    | FLAG_ALLOW_LOW_SPEED_BRAKE
                    | FLAG_EXCLUSIVE_PEDALS,
                0.6f32,
            ),
        ] {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("organism-action-signed-brake-fixture"),
            });
            action
                .encode(
                    &device,
                    &queue,
                    &mut encoder,
                    &physics,
                    1,
                    ActionParams {
                        n: 1,
                        flags,
                        steering_alpha: 0.5,
                        slow_steering_authority: 0.2,
                        gearbox_mode: GEARBOX_OFF,
                        gear_cap: 6,
                        dt: 1.0 / 64.0,
                        _pad: 0.0,
                    },
                )
                .unwrap();
            encoder.copy_buffer_to_buffer(&physics, 0, &staging, 0, staging.size());
            queue.submit(Some(encoder.finish()));
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            let mapped = slice.get_mapped_range();
            let output: &[f32] = bytemuck::cast_slice(&mapped);
            assert!(
                (output[51] - expected_throttle).abs() <= 4.0 * f32::EPSILON,
                "flags {flags:#b}: throttle {} expected {expected_throttle}",
                output[51]
            );
            assert_eq!(output[52].to_bits(), 0.0f32.to_bits(), "flags {flags:#b}");
            drop(mapped);
            staging.unmap();
        }
        // The signed brake only exists inside the exclusive net.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("organism-action-signed-brake-rejected"),
        });
        assert_eq!(
            action
                .encode(
                    &device,
                    &queue,
                    &mut encoder,
                    &physics,
                    1,
                    ActionParams {
                        n: 1,
                        flags: FLAG_LINEAR_HEAD | FLAG_SIGNED_BRAKE,
                        steering_alpha: 0.5,
                        slow_steering_authority: 0.2,
                        gearbox_mode: GEARBOX_OFF,
                        gear_cap: 6,
                        dt: 1.0 / 64.0,
                        _pad: 0.0,
                    },
                )
                .err(),
            Some("organism-action: signed brake requires exclusive pedals")
        );

        // Explicit reverse is refused while moving, even when gearbox mode is active.
        input[0] = 1.0;
        queue.write_buffer(&physics, 0, bytemuck::cast_slice(&input));
        queue.write_buffer(
            &action.buf_intent,
            (8 * 4) as u64,
            bytemuck::bytes_of(&1.0f32),
        );
        action
            .upload_gear_state(
                &queue,
                &[GearState {
                    cooldown: 0.0,
                    prev_up: 0.0,
                    prev_down: 0.0,
                    gear: 2,
                }],
            )
            .unwrap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("organism-action-reverse-moving-fixture"),
        });
        action
            .encode(
                &device,
                &queue,
                &mut encoder,
                &physics,
                1,
                ActionParams {
                    n: 1,
                    flags: FLAG_LINEAR_HEAD | FLAG_SIGNED_THROTTLE,
                    steering_alpha: 0.5,
                    slow_steering_authority: 0.2,
                    gearbox_mode: GEARBOX_AUTO,
                    gear_cap: 6,
                    dt: 1.0 / 64.0,
                    _pad: 0.0,
                },
            )
            .unwrap();
        encoder.copy_buffer_to_buffer(&physics, 0, &staging, 0, staging.size());
        queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range();
        let output: &[f32] = bytemuck::cast_slice(&mapped);
        assert_eq!(output[55].to_bits(), 2u32);
        drop(mapped);
        staging.unmap();

        // The scenario-only gate keeps braking available while a taxi is nearly
        // stopped; the legacy path would suppress this at every speed below 2 m/s.
        input[0] = 0.1;
        queue.write_buffer(&physics, 0, bytemuck::cast_slice(&input));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("organism-action-low-speed-fixture"),
        });
        action
            .encode(
                &device,
                &queue,
                &mut encoder,
                &physics,
                1,
                ActionParams {
                    n: 1,
                    flags: FLAG_LINEAR_HEAD | FLAG_SIGNED_THROTTLE | FLAG_ALLOW_LOW_SPEED_BRAKE,
                    steering_alpha: 0.5,
                    slow_steering_authority: 0.2,
                    gearbox_mode: GEARBOX_OFF,
                    gear_cap: 6,
                    dt: 1.0 / 64.0,
                    _pad: 0.0,
                },
            )
            .unwrap();
        encoder.copy_buffer_to_buffer(&physics, 0, &staging, 0, staging.size());
        queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let mapped = slice.get_mapped_range();
        let output: &[f32] = bytemuck::cast_slice(&mapped);
        assert_eq!(output[52].to_bits(), 1.0f32.to_bits());
        drop(mapped);
        staging.unmap();
    }
}
