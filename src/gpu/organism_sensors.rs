//! Resident adapter into the existing 192-channel perception pipeline.
//!
//! `organism-resident-extras-v1` preserves track projection, arc lookaheads, probes,
//! slip, RPM, temperature and social proximity. Explicit scenario differences:
//! ch67=integrity*alive; ch75=stationary duration/1s (speed<0.5m/s);
//! reverse input=actual reverse gear (no host manoeuvre controller);
//! ch126=delivery/damage/paid-resource consequence since neural feedback baseline;
//! ch127=positive-consequence streak/5s. Temporal33..48 remain zero as in native192.
//! Legacy BODY_CHANNELS must be disabled: organism/taxi owns channels160..191.
//! Call before retina/digest/brain, so digest edges retain their historical 1-tick lag.
use crate::simulation::racing_line::RacingLine;
use wgpu::util::DeviceExt;
pub const OBSERVATION_EXTRAS_ID: &str = "organism-resident-extras-v1";
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    gs: u32,
    points: u32,
    line_offset: u32,
    horizons: u32,
    social: u32,
    continuous: u32,
    pad: u32,
    dt: f32,
    cell: f32,
    half_width: f32,
    total: f32,
}
pub struct OrganismSensorsGpu {
    pipelines: [wgpu::ComputePipeline; 3],
    layout: wgpu::BindGroupLayout,
    data: wgpu::Buffer,
    params: Params,
    capacity: usize,
}
impl OrganismSensorsGpu {
    /// Static geometry setup only. Horizons are the existing expressed per-genome distances.
    pub fn new(
        device: &wgpu::Device,
        grid: &[u32],
        grid_size: usize,
        cell_size: f32,
        line: &RacingLine,
        horizons: &[[f32; 3]],
        social: bool,
        continuous: bool,
    ) -> Result<Self, &'static str> {
        if grid_size == 0
            || grid_size > 16384
            || grid.len() != grid_size * grid_size
            || !cell_size.is_finite()
            || cell_size <= 0.0
            || horizons.is_empty()
            || horizons.len() > u32::MAX as usize
            || line.points.len() != line.arc.len()
            || line.points.len() != line.tangents.len()
            || !line.half_width.is_finite()
            || line.half_width < 0.0
            || !line.total_len.is_finite()
            || line.total_len < 0.0
            || line
                .points
                .iter()
                .chain(line.tangents.iter())
                .any(|v| !v.is_finite())
            || line.arc.iter().any(|v| !v.is_finite() || *v < 0.0)
            || line.arc.windows(2).any(|v| v[1] < v[0])
            || horizons
                .iter()
                .flatten()
                .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err("invalid resident sensor geometry");
        }
        if grid.iter().any(|v| *v > 255) {
            return Err("invalid voxel ID");
        }
        let mut data: Vec<f32> = grid.iter().map(|v| *v as f32).collect();
        let line_offset = data.len() as u32;
        for k in 0..line.points.len() {
            data.extend_from_slice(&[
                line.points[k].x,
                line.points[k].y,
                line.arc[k],
                line.tangents[k].x,
                line.tangents[k].y,
            ]);
        }
        let horizon_offset = data.len() as u32;
        for h in horizons {
            data.extend_from_slice(h);
        }
        let data = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("resident sensor static geometry"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(OBSERVATION_EXTRAS_ID),
            source: wgpu::ShaderSource::Wgsl(include_str!("organism_sensors.wgsl").into()),
        });
        let entries: Vec<_> = (0..9)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: matches!(binding, 1 | 3 | 4 | 8),
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resident sensor bridge layout"),
            entries: &entries,
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipelines = ["probes", "geometry", "bridge"].map(|entry| {
            #[cfg(test)]
            eprintln!("sensor compiling {entry}");
            let result = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
            #[cfg(test)]
            eprintln!("sensor compiled {entry}");
            result
        });
        Ok(Self {
            pipelines,
            layout,
            data,
            capacity: horizons.len(),
            params: Params {
                n: 0,
                gs: grid_size as u32,
                points: line.points.len() as u32,
                line_offset,
                horizons: horizon_offset,
                social: social as u32,
                continuous: continuous as u32,
                pad: 0,
                dt: 0.0,
                cell: cell_size,
                half_width: line.half_width,
                total: line.total_len,
            },
        })
    }
    /// Update only inherited horizon distances when one ecological slot is replaced.
    pub fn upload_horizon_slot(
        &self,
        queue: &wgpu::Queue,
        slot: usize,
        horizons: [f32; 3],
    ) -> Result<(), &'static str> {
        if slot >= self.capacity || horizons.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err("invalid horizon birth slot");
        }
        queue.write_buffer(
            &self.data,
            (self.params.horizons as usize + slot * 3) as u64 * 4,
            bytemuck::cast_slice(&horizons),
        );
        Ok(())
    }
    /// Buffers in order: physics input368B, organism512B, digest76B, temporal292B,
    /// retina pose32B and perception pose32B, all per-being strides.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        physical: &wgpu::Buffer,
        body: &wgpu::Buffer,
        digest: &wgpu::Buffer,
        temporal: &wgpu::Buffer,
        retina_pose: &wgpu::Buffer,
        perception_pose: &wgpu::Buffer,
        intent: &wgpu::Buffer,
        n: usize,
        dt: f32,
    ) -> Result<(), &'static str> {
        if n > self.capacity || !dt.is_finite() || dt <= 0.0 {
            return Err("invalid sensor count/timestep");
        }
        for (buffer, stride) in [
            (physical, 368),
            (body, 512),
            (digest, 76),
            (temporal, 292),
            (retina_pose, 32),
            (perception_pose, 32),
            (intent, 384),
        ] {
            if buffer.size() < n as u64 * stride
                || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            {
                return Err("invalid resident sensor buffer");
            }
        }
        if n == 0 {
            return Ok(());
        }
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&Params {
                n: n as u32,
                dt,
                ..self.params
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let buffers = [
            &uniform,
            physical,
            body,
            &self.data,
            digest,
            temporal,
            retina_pose,
            perception_pose,
            intent,
        ];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &entries,
        });
        for pipeline in &self.pipelines {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("organism resident sensor phase"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn organism_sensor_shader_validates() {
        let module = naga::front::wgsl::parse_str(include_str!("organism_sensors.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        assert_eq!(super::super::perception::TEMPORAL_STRIDE, 73);
        assert_eq!(std::mem::size_of::<super::Params>(), 48);
    }
}

#[cfg(test)]
#[test]
#[ignore = "analytic resident sensor fixture requires a real hardware Vulkan GPU"]
fn gpu_resident_sensor_geometry_fixture() {
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
    assert_ne!(
        adapter.get_info().device_type,
        wgpu::DeviceType::Cpu,
        "CPU adapter forbidden"
    );
    eprintln!("sensor fixture adapter: {:?}", adapter.get_info());
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits {
                max_storage_buffers_per_shader_stage: 8,
                ..wgpu::Limits::downlevel_defaults()
            },
            ..Default::default()
        },
        None,
    ))
    .unwrap();
    let v = bevy::math::Vec2::new;
    let line = RacingLine {
        points: vec![
            v(0.0, 0.0),
            v(20.0, 0.0),
            v(20.0, 20.0),
            v(0.0, 20.0),
            v(0.0, 0.0),
        ],
        tangents: vec![
            v(1.0, 0.0),
            v(0.0, 1.0),
            v(-1.0, 0.0),
            v(0.0, -1.0),
            v(1.0, 0.0),
        ],
        arc: vec![0.0, 20.0, 40.0, 60.0, 80.0],
        total_len: 80.0,
        half_width: 10.0,
    };
    let mut grid = vec![1u32; 32 * 32];
    for z in 0..32 {
        grid[z * 32 + 10] = 2;
    }
    let bridge = OrganismSensorsGpu::new(
        &device,
        &grid,
        32,
        1.0,
        &line,
        &[[5.0, 20.0, 40.0]],
        false,
        true,
    )
    .unwrap();
    let make = |values: &[f32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(values),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
    };
    let mut phys = [0.0f32; 92];
    phys[36] = 5.0;
    phys[38] = 5.0;
    phys[40] = 1.0;
    phys[46] = 1.0;
    phys[50] = 8000.0;
    phys[55] = f32::from_bits(2);
    phys[60] = 20.0;
    let physics = make(&phys);
    let birth = super::organism::birth_state(&[0.0; 16], 256).unwrap();
    let body = make(&birth.values);
    let mut digest_values = [0.0; 19];
    digest_values[1] = 0.25;
    digest_values[2] = 0.75;
    let digest = make(&digest_values);
    let mut temporal_values = [0.0; 73];
    temporal_values[0] = 0.375;
    let temporal = make(&temporal_values);
    let retina = make(&[0.0; 8]);
    let perception = make(&[0.0; 8]);
    let mut intent_values = [0.0; 96];
    intent_values[0] = 0.375;
    let intent = make(&intent_values);
    let mut encoder = device.create_command_encoder(&Default::default());
    bridge
        .encode(
            &device,
            &mut encoder,
            &physics,
            &body,
            &digest,
            &temporal,
            &retina,
            &perception,
            &intent,
            1,
            1.0 / 64.0,
        )
        .unwrap();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 356,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&temporal, 0, &staging, 0, 292);
    encoder.copy_buffer_to_buffer(&retina, 0, &staging, 292, 32);
    encoder.copy_buffer_to_buffer(&perception, 0, &staging, 324, 32);
    queue.submit(Some(encoder.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let mapped = staging.slice(..).get_mapped_range();
    let values: &[f32] = bytemuck::cast_slice(&mapped);
    for (slot, expected) in [
        (0, 0.375),
        (7, 0.375),
        (10, 0.5),
        (12, 0.5),
        (16, 5.5),
        (17, 26.5),
        (18, 4.5),
        (23, 1.0),
        (25, 2.0),
        (32, 0.0),
        (54, 0.25),
        (55, 0.75),
        (66, 0.0),
        (67, 1.0),
        (68, 1.0),
    ] {
        assert!(
            (values[slot] - expected).abs() < 0.0001,
            "slot {slot}: {} != {expected}",
            values[slot]
        );
    }
    assert_eq!(&values[73..81], &[5.0, 0.0, 5.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
    assert_eq!(&values[73..81], &values[81..89]);
}
