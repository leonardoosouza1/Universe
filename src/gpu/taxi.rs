//! GPU task/services adapter for `organism-taxi-v1`, owned by the existing GpuCtx.
//!
//! Station input: [x_m,z_m,energy_capacity_J,water_capacity_kg,material_capacity_kg,
//! energy_regen_J_s,water_regen_kg_s,material_regen_kg_s]. Stocks start full at world
//! birth; reset_world is an explicit episode reset, never an ordinary tick.
//! encode_birth consumes these stocks for initial reserves and construction, after
//! organism reset. Environmental genesis stocks are not evolved or earned resources.
//! Post order: physical/body transition → stocks → rotating service/passenger arbitration.
//! Transfers become available to body metabolism on the NEXT tick. Taxi is the only
//! external-resource writer; body is the only metabolic writer. No host policy runs here.
//!
//! Taxi state slots:64 money,65 onboard,66/67 destinationx/z,68 deliveries,69 refusal;
//! 70/71 originx/z,72 passengerID,73 fare,74 taskage_s,75 lastpaidID;
//! 76/77/78 receivedJ/kg/kg,79 actualcost,80/81 targetdelta_m,82 neareststation,
//! 83 serviceavailable,84 initialized,85/86 servicedelta_m,87..89 stockfractions,
//! 90 waitingavailable,91 constructionmaterial_kg,92 fundedbirthenergy_J,93 birthfundingmarker. Slots94..97 belong to runtime transition feedback; other slots reserved. IDs are exact f32 integers; issuer
//! stops at2^24 instead of reusing IDs. Refusal0=none,1=moving,2=distance,3=money,
//! 4=no stock/deficit,5=no passenger,6=dead,7=unfundedbirth. Requests are nativeintent5/6/7>0.5.
//!
//! Explicit observation schema `organism-taxi-observation-v1`: organism160..175,
//! taxi176..191 (see shader, which also publishes body160..175); these replace the legacy raw horizon for this scenario.
//! Positions/destination/nearestservice are disclosed map observations. They do not steer.
//! Observation v4 keeps the v3 packing but replaces task direction/distance with
//! a legal-road route-field gradient and compressed remaining route distance.
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

pub const TAXI_WORLD_HEADER_FLOATS: usize = 4;
pub const TAXI_STATION_STRIDE_FLOATS: usize = 16;
pub const S_MONEY: usize = 64;
pub const S_ONBOARD: usize = 65;
pub const S_DELIVERIES: usize = 68;
pub const S_REFUSAL: usize = 69;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n: u32,
    stations: u32,
    tick: u32,
    wave_words: u32,
    dt: f32,
    observation: u32,
    mode: u32,
    padding: u32,
}

/// Execution contract for the taxi adapter.
///
/// The discriminants are part of the Rust/WGSL ABI; keep them in sync with
/// `taxi.wgsl`. A bench owns its task state and therefore never receives
/// production passengers or consumes shared genesis stocks.
#[repr(u32)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TaxiMode {
    #[default]
    Production = 0,
    MotorBench = 1,
    ServiceBench = 2,
}

/// Taxi channel schema selected by `--navigation-observation`.
pub const NAVIGATION_OBSERVATION_V2: u32 = 2;
pub const NAVIGATION_OBSERVATION_V3: u32 = 3;
/// Route-gradient navigation: legal-road direction and remaining route distance.
pub const NAVIGATION_OBSERVATION_V4: u32 = 4;
/// Bearing to a speed-dependent preview waypoint on the same immutable route field.
pub const NAVIGATION_OBSERVATION_V5: u32 = 5;

pub struct TaxiGpu {
    pub buf_world: wgpu::Buffer,
    pub buf_routes: wgpu::Buffer,
    pub capacity: usize,
    pub station_count: usize,
    initial_world: Vec<f32>,
    observation_version: u32,
    mode: TaxiMode,
    post_layout: wgpu::BindGroupLayout,
    birth_layout: wgpu::BindGroupLayout,
    senses_layout: wgpu::BindGroupLayout,
    post_pipeline: wgpu::ComputePipeline,
    birth_pipeline: wgpu::ComputePipeline,
    senses_pipeline: wgpu::ComputePipeline,
    route_senses_pipeline: wgpu::ComputePipeline,
}
fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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
fn uniform() -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
impl TaxiGpu {
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        capacity: usize,
        stations: &[[f32; 8]],
    ) -> Self {
        assert!(
            capacity > 0 && capacity <= u32::MAX as usize,
            "taxi capacity"
        );
        assert!(
            stations.len() >= 2 && stations.len() <= 256,
            "taxi requires2..256stations"
        );
        assert!(
            stations
                .iter()
                .all(|s| s.iter().all(|v| v.is_finite()) && s[2..].iter().all(|v| *v >= 0.0)),
            "invalid taxi station config"
        );
        for (i, a) in stations.iter().enumerate() {
            for b in &stations[i + 1..] {
                let dx = f64::from(a[0]) - f64::from(b[0]);
                let dz = f64::from(a[1]) - f64::from(b[1]);
                assert!(
                    dx * dx + dz * dz > 64.0,
                    "taxi station interaction regions must not overlap"
                );
            }
        }
        let mut initial_world =
            vec![0.0; TAXI_WORLD_HEADER_FLOATS + stations.len() * TAXI_STATION_STRIDE_FLOATS];
        initial_world[0] = 1.0;
        for (i, station) in stations.iter().enumerate() {
            let start = TAXI_WORLD_HEADER_FLOATS + i * TAXI_STATION_STRIDE_FLOATS;
            initial_world[start..start + 8].copy_from_slice(station);
            initial_world[start + 8..start + 11].copy_from_slice(&station[2..5]);
        }
        let buf_world = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("organism-taxi-world"),
            contents: bytemuck::cast_slice(&initial_world),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("organism-taxi-services"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("taxi.wgsl"))),
        });
        let post_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("taxi-post-layout"),
            entries: &[
                uniform(),
                storage(1, false),
                storage(2, false),
                storage(3, true),
                storage(4, true),
                storage(6, true),
            ],
        });
        let birth_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("taxi-genesis-layout"),
            entries: &[uniform(), storage(1, false), storage(2, false)],
        });
        let senses_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("taxi-senses-layout"),
            entries: &[
                uniform(),
                storage(1, false),
                storage(2, false),
                storage(3, true),
                storage(5, false),
                storage(6, true),
            ],
        });
        let pipeline = |layout: &wgpu::BindGroupLayout, name: &str| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(name),
                bind_group_layouts: &[layout],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let post_pipeline = pipeline(&post_layout, "post");
        let birth_pipeline = pipeline(&birth_layout, "birth");
        let senses_pipeline = pipeline(&senses_layout, "senses");
        let route_senses_pipeline = pipeline(&senses_layout, "route_senses");
        let buf_routes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("taxi-no-road-field"), contents: bytemuck::cast_slice(&[0.0f32; 4]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        Self {
            buf_world,
            buf_routes,
            capacity,
            station_count: stations.len(),
            initial_world,
            observation_version: NAVIGATION_OBSERVATION_V2,
            mode: TaxiMode::Production,
            post_layout,
            birth_layout,
            senses_layout,
            post_pipeline,
            birth_pipeline,
            senses_pipeline,
            route_senses_pipeline,
        }
    }
    /// Select the taxi channel schema before the first tick; never mid-run.
    pub fn set_observation_version(&mut self, version: u32) {
        assert!(
            matches!(
                version,
                NAVIGATION_OBSERVATION_V2 | NAVIGATION_OBSERVATION_V3 | NAVIGATION_OBSERVATION_V4 | NAVIGATION_OBSERVATION_V5
            ),
            "unknown navigation observation version {version}"
        );
        self.observation_version = version;
    }
    pub fn observation_version(&self) -> u32 {
        self.observation_version
    }
    /// Select the world/task contract before birth; never change it mid-episode.
    pub fn set_mode(&mut self, mode: TaxiMode) {
        self.mode = mode;
    }
    pub fn mode(&self) -> TaxiMode {
        self.mode
    }
    /// Install the scenario's immutable road field once, never per tick.
    pub fn configure_routes(&mut self, device: &wgpu::Device, world: &crate::world::VoxelWorld, stations: &[[f32; 8]]) -> Result<(), &'static str> {
        if stations.len() != self.station_count { return Err("route station count mismatch"); }
        let packed = super::taxi_routes::pack_distance_fields(world, stations)?;
        self.buf_routes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("taxi-road-distance-fields"), contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        Ok(())
    }
    pub fn reset_world(&self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.buf_world,
            0,
            bytemuck::cast_slice(&self.initial_world),
        );
    }
    fn params(&self, device: &wgpu::Device, n: usize, dt: f32, tick: u32) -> wgpu::Buffer {
        assert!(n <= self.capacity, "taxi population exceeds capacity");
        assert!(dt.is_finite() && dt >= 0.0, "taxi invalid dt");
        assert_eq!(
            super::perception::WAVE_U32S,
            48,
            "taxi observation schema requires192packedchannels"
        );
        let p = Params {
            n: n as u32,
            stations: self.station_count as u32,
            tick,
            wave_words: super::perception::WAVE_U32S as u32,
            dt,
            observation: self.observation_version,
            mode: self.mode as u32,
            padding: 0,
        };
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("taxi-params"),
            contents: bytemuck::bytes_of(&p),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }
    /// Fund fresh organism-reset reserves from environmental stocks exactly once.
    /// Construction consumes material before reserves; unfunded slots remain dead.
    /// This is genesis, not a per-tick service or ecological reproduction operation.
    pub fn encode_birth(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        bodybuf: &wgpu::Buffer,
        n: usize,
    ) {
        if n == 0 {
            return;
        }
        assert!(
            bodybuf.size() >= (n * 128 * 4) as u64,
            "taxi birth body shape"
        );
        let params = self.params(device, n, 0.0, 0);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("taxi-paid-genesis"),
            layout: &self.birth_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bodybuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_world.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("taxi-paid-genesis"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.birth_pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    pub fn encode_post(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        physics_output: &wgpu::Buffer,
        bodybuf: &wgpu::Buffer,
        intentbuf: &wgpu::Buffer,
        n: usize,
        dt: f32,
        tick: u32,
    ) {
        if n == 0 {
            return;
        }
        assert!(
            physics_output.size() >= (n * 56 * 4) as u64,
            "taxi physics shape"
        );
        assert!(bodybuf.size() >= (n * 128 * 4) as u64, "taxi body shape");
        assert!(intentbuf.size() >= (n * 96 * 4) as u64, "taxi intent shape");
        let params = self.params(device, n, dt, tick);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("taxi-post"),
            layout: &self.post_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bodybuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_world.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: physics_output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: intentbuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry { binding: 6, resource: self.buf_routes.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("taxi-arbitration"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.post_pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    pub fn encode_senses(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        bodybuf: &wgpu::Buffer,
        physics_input: &wgpu::Buffer,
        packed_wave: &wgpu::Buffer,
        n: usize,
    ) {
        if n == 0 {
            return;
        }
        assert!(bodybuf.size() >= (n * 128 * 4) as u64, "taxi body shape");
        assert!(physics_input.size() >= (n * 92 * 4) as u64, "taxi physics input shape");
        assert!(
            packed_wave.size() >= (n * super::perception::WAVE_U32S * 4) as u64,
            "taxi packed sensory shape"
        );
        let params = self.params(device, n, 0.0, 0);
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("taxi-senses"),
            layout: &self.senses_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bodybuf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_world.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: physics_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: packed_wave.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.buf_routes.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("taxi-observation"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.senses_pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        drop(pass);
        if matches!(self.observation_version, NAVIGATION_OBSERVATION_V4 | NAVIGATION_OBSERVATION_V5) {
            let mut route = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("taxi-route-observation"), timestamp_writes: None,
            });
            route.set_pipeline(&self.route_senses_pipeline);
            route.set_bind_group(0, &group, &[]);
            route.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn taxi_shader_contract_parses_and_validates_without_simulation() {
        let source = include_str!("taxi.wgsl");
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
    }

    /// Scripted boundary fixture, NOT learned taxi competence or CPU dynamics.
    #[test]
    #[ignore = "exclusive real Vulkan GPU required; scripted controls, no CPU simulation"]
    fn taxi_gpu_scripted_stock_payment_once_and_moving_refusal() {
        use super::*;
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("real GPU adapter required; no CPU fallback");
        assert_ne!(
            adapter.get_info().device_type,
            wgpu::DeviceType::Cpu,
            "CPU adapter forbidden"
        );
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("taxi-scripted-contract"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        ))
        .expect("GPU device");
        let taxi = TaxiGpu::new(
            &device,
            &queue,
            1,
            &[
                [0.0, 0.0, 10000.0, 10.0, 10.0, 0.0, 0.0, 0.0],
                [100.0, 0.0, 10000.0, 10.0, 10.0, 0.0, 0.0, 0.0],
            ],
        );
        let buffer = |data: &[f32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scripted-taxi-fixture"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        };
        let mut state = [0.0f32; 128];
        state[18] = 20000.0;
        state[19] = 10.0;
        state[20] = 10.0;
        state[29] = 1.0;
        let body = buffer(&state);
        let mut pose = [0.0f32; 56];
        pose[0] = 1.0; // An explicit moving fixture must refuse both requests.
        let physics = buffer(&pose);
        let mut command = [0.0f32; 96];
        command[5] = 1.0;
        command[6] = 1.0;
        let intent = buffer(&command);
        let step = |tick| {
            let mut encoder = device.create_command_encoder(&Default::default());
            taxi.encode_post(
                &device,
                &mut encoder,
                &physics,
                &body,
                &intent,
                1,
                1.0,
                tick,
            );
            queue.submit(Some(encoder.finish()));
        };
        let read = |source: &wgpu::Buffer, floats: usize| -> Vec<f32> {
            let bytes = (floats * 4) as u64;
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("taxi-test-readback"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes);
            queue.submit(Some(encoder.finish()));
            let (tx, rx) = std::sync::mpsc::channel();
            staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    tx.send(result).unwrap();
                });
            device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            let mapped = staging.slice(..).get_mapped_range();
            let result = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
            drop(mapped);
            staging.unmap();
            result
        };
        step(0);
        let moving = read(&body, 128);
        assert_eq!(moving[S_REFUSAL], 1.0);
        assert_eq!(moving[S_ONBOARD], 0.0);
        assert_eq!(moving[0], 0.0);
        pose[0] = 0.0;
        queue.write_buffer(&physics, 0, bytemuck::cast_slice(&pose));
        step(1);
        let served = read(&body, 128);
        let world = read(&taxi.buf_world, 36);
        assert!(
            (served[0] - 10000.0).abs() < 0.001,
            "served E={} money={} refusal={} world={:?}",
            served[0],
            served[64],
            served[69],
            world
        );
        assert!((world[12] + served[0] - 10000.0).abs() < 0.001);
        assert!((world[13] + served[1] - 10.0).abs() < 0.0001);
        assert!((world[14] + served[2] - 10.0).abs() < 0.0001);
        assert_eq!(served[S_ONBOARD], 1.0);
        assert!((served[S_MONEY] - 22.9).abs() < 0.0001);
        assert_eq!(served[127], 100.0);
        // Scripted approach and moving-away poses test bounded progress only;
        // the taxi kernel cannot write positions or teleport a car.
        pose[36] = 50.0;
        queue.write_buffer(&physics, 0, bytemuck::cast_slice(&pose));
        command[5] = 0.0;
        command[6] = 0.0;
        command[7] = 0.0;
        queue.write_buffer(&intent, 0, bytemuck::cast_slice(&command));
        step(2);
        let approach = read(&body, 128);
        assert!(approach[127] > served[127] && approach[127] < 200.0);
        pose[36] = 180.0;
        queue.write_buffer(&physics, 0, bytemuck::cast_slice(&pose));
        step(3);
        let moving_away = read(&body, 128);
        assert_eq!(
            moving_away[127], approach[127],
            "phase progress keeps the best remaining distance; retreat cannot erase or farm it"
        );

        // A post-physics fixture is now at the destination. Dropoff clears the
        // transient potential while the completed delivery supplies 1000 points.
        pose[36] = served[66];
        pose[38] = served[67];
        queue.write_buffer(&physics, 0, bytemuck::cast_slice(&pose));
        command[5] = 0.0;
        command[6] = 0.0;
        command[7] = 1.0;
        queue.write_buffer(&intent, 0, bytemuck::cast_slice(&command));
        step(4);
        let paid = read(&body, 128);
        step(5);
        let repeated = read(&body, 128);
        assert_eq!(paid[S_DELIVERIES], 1.0);
        assert_eq!(paid[S_ONBOARD], 0.0);
        assert_eq!(paid[127], 0.0);
        assert!(1000.0 > approach[127]);
        assert!((paid[S_MONEY] - 34.9).abs() < 0.0001);
        assert_eq!(repeated[S_MONEY], paid[S_MONEY]);
        assert_eq!(repeated[S_DELIVERIES], 1.0);

        // Three founders share two finite stations. Construction leaves insufficient
        // material for founder2: refusal cannot leave its prefunded reset reserves alive.
        let genesis = TaxiGpu::new(
            &device,
            &queue,
            3,
            &[
                [0.0, 0.0, 10000.0, 10.0, 20.0, 0.0, 0.0, 0.0],
                [100.0, 0.0, 10000.0, 10.0, 20.0, 0.0, 0.0, 0.0],
            ],
        );
        let mut founders = vec![0.0f32; 3 * 128];
        for state in founders.chunks_exact_mut(128) {
            state[0] = 6000.0;
            state[1] = 4.0;
            state[2] = 5.0;
            state[18] = 6000.0;
            state[19] = 4.0;
            state[20] = 5.0;
            state[24] = 1.0;
            state[29] = 1.0;
        }
        let founders_buf = buffer(&founders);
        let fund = || {
            let mut encoder = device.create_command_encoder(&Default::default());
            genesis.encode_birth(&device, &mut encoder, &founders_buf, 3);
            queue.submit(Some(encoder.finish()));
        };
        fund();
        let funded = read(&founders_buf, 3 * 128);
        let stocks = read(&genesis.buf_world, 36);
        let mut received = [0.0; 3];
        let mut construction = 0.0;
        for state in funded.chunks_exact(128) {
            for r in 0..3 {
                received[r] += state[r];
            }
            construction += state[91];
            assert_eq!(state[93], 1.0);
            assert_eq!(state[92], state[0]);
        }
        assert_eq!(received, [12000.0, 8.0, 10.0]);
        assert_eq!(construction, 27.0);
        assert_eq!(stocks[12] + stocks[28] + received[0], 20000.0);
        assert_eq!(stocks[13] + stocks[29] + received[1], 20.0);
        assert_eq!(stocks[14] + stocks[30] + received[2] + construction, 40.0);
        assert_eq!(funded[2 * 128 + 29], 0.0);
        assert_eq!(&funded[2 * 128..2 * 128 + 3], &[0.0; 3]);
        assert_eq!(funded[2 * 128 + 69], 7.0);
        fund(); // A repeated birth call must not charge stocks or create reserves.
        assert_eq!(read(&founders_buf, 3 * 128), funded);
        assert_eq!(read(&genesis.buf_world, 36), stocks);

        // The school bench is a controlled motor-learning experiment: classmates with
        // mutated bodies must not change the elite's station-stock observation. Genesis
        // therefore preserves the world and funds every declared reserve independently.
        let mut school = TaxiGpu::new(
            &device,
            &queue,
            3,
            &[
                [0.0, 0.0, 10000.0, 10.0, 20.0, 0.0, 0.0, 0.0],
                [100.0, 0.0, 10000.0, 10.0, 20.0, 0.0, 0.0, 0.0],
            ],
        );
        school.set_mode(TaxiMode::MotorBench);
        let school_before = read(&school.buf_world, 36);
        let mut pupils = founders.clone();
        pupils[128] = 9000.0;
        pupils[128 + 1] = 9.0;
        pupils[128 + 2] = 19.0;
        let pupils_buf = buffer(&pupils);
        let mut encoder = device.create_command_encoder(&Default::default());
        school.encode_birth(&device, &mut encoder, &pupils_buf, 3);
        queue.submit(Some(encoder.finish()));
        let school_after = read(&school.buf_world, 36);
        let pupils_after = read(&pupils_buf, 3 * 128);
        assert_eq!(school_after, school_before, "school birth changed shared stocks");
        for (expected, pupil) in pupils.chunks_exact(128).zip(pupils_after.chunks_exact(128)) {
            assert_eq!(&pupil[..3], &expected[..3]);
            assert_eq!(pupil[29], 1.0);
            assert_eq!(pupil[93], 1.0);
        }
        // U-shaped road: the legal detour initially increases Euclidean distance.
        // GPU metric fixture; this is NOT the physical feasibility witness.
        let mut road = crate::world::VoxelWorld::new(0, 15.0);
        road.size = 7; road.cell_size = 10.0;
        road.grid = vec![crate::world::VoxelType::Wall; 49];
        for z in 1..=5 { road.grid[z*7+1] = crate::world::VoxelType::Road; road.grid[z*7+5] = crate::world::VoxelType::Road; }
        for x in 1..=5 { road.grid[5*7+x] = crate::world::VoxelType::Road; }
        let stations = [[10.,10.,10000.,10.,20.,0.,0.,0.], [50.,10.,10000.,10.,20.,0.,0.,0.]];
        let mut routed = TaxiGpu::new(&device, &queue, 3, &stations);
        routed.configure_routes(&device, &road, &stations).unwrap();
        let mut states = vec![0.0f32; 3*128];
        let mut poses = vec![0.0f32; 3*56];
        for (b, position) in [[10.,10.], [10.,20.], [20.,10.]].iter().enumerate() {
            let s=b*128; let o=b*56;
            states[s+29]=1.; states[s+65]=1.; states[s+66]=50.; states[s+67]=10.;
            states[s+70]=10.; states[s+71]=10.; states[s+84]=1.;
            // Same already-issued passenger and baseline route distance for all
            // three fixtures. Without this memory each body correctly starts its
            // progress baseline at its own current pose, so the comparison proves
            // nothing about movement along the U-shaped route.
            states[s+31]=1.; states[s+61]=120.; states[s+62]=120.;
            poses[o+36]=position[0]; poses[o+38]=position[1];
        }
        let states_buffer=buffer(&states); let poses_buffer=buffer(&poses);
        let no_requests=buffer(&vec![0.0f32; 3*96]);
        let mut encoder=device.create_command_encoder(&Default::default());
        routed.encode_post(&device, &mut encoder, &poses_buffer, &states_buffer, &no_requests, 3, 0., 0);
        queue.submit(Some(encoder.finish()));
        let evaluated=read(&states_buffer, 3*128);
        assert!((evaluated[127]-100.0).abs()<1e-4);
        assert!((evaluated[128+127]-(100.0+100.0/12.0)).abs()<1e-4, "legal detour must improve score");
        assert!((evaluated[256+127]-100.0).abs()<1e-4,"wall must not receive progress bonus: {}", evaluated[256+127]);

        // Cell convention on the left leg (x = 10): z = 15.3 m / 10 m = 1.53 cells ->
        // row 2 by round (110 m remaining), row 1 by floor (120 m). Exact tie
        // 15.0 -> row 2 (half away from zero, world.rs convention).
        let mut tie_poses = vec![0.0f32; 3*56];
        for (b, z) in [15.3f32, 14.7, 15.0].iter().enumerate() { tie_poses[b*56+36]=10.0; tie_poses[b*56+38]=*z; }
        let tie_buffer=buffer(&tie_poses);
        let tie_states=buffer(&states);
        let mut encoder=device.create_command_encoder(&Default::default());
        routed.encode_post(&device, &mut encoder, &tie_buffer, &tie_states, &no_requests, 3, 0., 0);
        queue.submit(Some(encoder.finish()));
        let tie=read(&tie_states, 3*128);
        // route_length origin(10,10)->(50,10) along the U = 120 m; score = 100 + 100*(1-remaining/120).
        let score = |remaining: f32| 100.0 + 100.0 * (1.0 - remaining / 120.0);
        assert!((tie[127]-score(110.0)).abs()<1e-4, "z=15.3 rounds to row 2: {}", tie[127]);
        assert!((tie[128+127]-score(120.0)).abs()<1e-4, "z=14.7 stays in row 1: {}", tie[128+127]);
        assert!((tie[256+127]-tie[127]).abs()<1e-4, "exact tie 15.0 rounds away from zero like 15.3");
    }

    #[test]
    #[ignore = "exclusive real GPU; navigation observation v2 saturation vs v3 body frame"]
    fn taxi_gpu_navigation_v3_keeps_direction_and_distance_beyond_100m() {
        use super::*;
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("Vulkan adapter");
        assert_ne!(adapter.get_info().device_type, wgpu::DeviceType::Cpu);
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("taxi navigation fixture"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        ))
        .expect("GPU device");
        let stations = [
            [0.0, 0.0, 10000.0, 10.0, 10.0, 0.0, 0.0, 0.0],
            [1000.0, 1000.0, 10000.0, 10.0, 10.0, 0.0, 0.0, 0.0],
        ];
        let buffer = |data: &[f32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("navigation-fixture"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            })
        };
        let read = |source: &wgpu::Buffer, words: usize| -> Vec<u32> {
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: None, size: (words * 4) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(source, 0, &staging, 0, (words * 4) as u64);
            queue.submit(Some(encoder.finish()));
            let (tx, rx) = std::sync::mpsc::channel();
            staging.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            let out = bytemuck::cast_slice::<u8, u32>(&staging.slice(..).get_mapped_range()).to_vec();
            staging.unmap();
            out
        };
        let byte = |word: u32, lane: usize| -> i32 {
            let raw = ((word >> (lane * 8)) & 255) as i32;
            if raw > 127 { raw - 256 } else { raw }
        };
        // Four beings, all at the origin heading +x with lateral +z; three targets
        // that v2 cannot tell apart and one 12 m task the pickup radius cares about.
        let targets = [[1000.0f32, 100.0], [100.0, 1000.0], [1000.0, -100.0], [12.0, 0.0]];
        let n = targets.len();
        let mut bodies = vec![0.0f32; n * 128];
        let mut physics = vec![0.0f32; n * 92];
        for (b, target) in targets.iter().enumerate() {
            let s = b * 128;
            bodies[s + 29] = 1.0; bodies[s + 65] = 1.0; bodies[s + 84] = 1.0;
            bodies[s + 80] = target[0]; bodies[s + 81] = target[1];
            bodies[s + 85] = 30.0; bodies[s + 86] = -40.0; // nearest service 50 m away, forward-right
            bodies[s + 87] = 0.9; bodies[s + 88] = 0.4; bodies[s + 89] = 0.7;
            bodies[s + 69] = 3.0; bodies[s + 73] = 20.0;
            physics[b * 92 + 40] = 1.0; physics[b * 92 + 46] = 1.0;
        }
        let body = buffer(&bodies);
        let input = buffer(&physics);
        let wave = buffer(&vec![0.0f32; n * 48]);
        let mut taxi = TaxiGpu::new(&device, &queue, n, &stations);
        let run = |taxi: &TaxiGpu| {
            let mut encoder = device.create_command_encoder(&Default::default());
            taxi.encode_senses(&device, &mut encoder, &body, &input, &wave, n);
            queue.submit(Some(encoder.finish()));
            read(&wave, n * 48)
        };
        let v2 = run(&taxi);
        for b in 0..3 {
            let w = v2[b * 48 + 44];
            assert_eq!((byte(w, 2), byte(w, 3).abs()), (127, 127), "v2 saturates being {b}");
        }
        assert_eq!(byte(v2[44], 2), byte(v2[48 + 44], 2), "v2: (1000,100) and (100,1000) collide");
        taxi.set_observation_version(NAVIGATION_OBSERVATION_V3);
        let v3 = run(&taxi);
        let channel = |b: usize, ch: usize| byte(v3[b * 48 + ch / 4], ch % 4);
        let expect = |value: f32| (value.clamp(-1.0, 1.0) * 127.0).round() as i32;
        for (b, target) in targets.iter().enumerate() {
            let d = target[0].hypot(target[1]);
            let near = |ch: usize, value: f32, what: &str| {
                let got = channel(b, ch);
                assert!((got - expect(value)).abs() <= 1, "being {b} {what}: byte {got} vs {}", expect(value));
            };
            near(178, target[0] / d, "forward cosine");
            near(179, target[1] / d, "lateral sine");
            near(183, d / (d + 100.0), "compressed distance");
            near(184, 0.6, "service forward");
            near(185, -0.8, "service lateral");
            near(186, 50.0 / 150.0, "service distance");
            near(187, 0.4, "min stock fraction");
            near(188, 0.5, "refusal 3/6");
            near(190, 0.2, "fare 20/100");
            assert_eq!(channel(b, 177), 127, "onboard unchanged");
        }
        let v3_dir: Vec<(i32, i32, i32)> = (0..3).map(|b| (channel(b, 178), channel(b, 179), channel(b, 183))).collect();
        assert!(v3_dir[0] != v3_dir[1] && v3_dir[0] != v3_dir[2] && v3_dir[1] != v3_dir[2], "v3 separates the three far targets: {v3_dir:?}");
        assert!((channel(3, 183) - expect(12.0 / 112.0)).abs() <= 1, "12 m task keeps ~14 quantisation steps: {}", channel(3, 183));
        // No task: direction 0, distance 1 (far), validity through 177/180.
        bodies[65] = 0.0; bodies[90] = 0.0;
        queue.write_buffer(&body, 0, bytemuck::cast_slice(&bodies));
        let idle = run(&taxi);
        assert_eq!((byte(idle[44], 2), byte(idle[44], 3), byte(idle[45], 3)), (0, 0, 127));
    }

    #[test]
    #[ignore = "exclusive real GPU; route-gradient navigation fixture"]
    fn taxi_gpu_navigation_v4_follows_legal_route_gradient() {
        let v3 = crate::organism_channels::channel_manifest(3);
        let v4 = crate::organism_channels::channel_manifest(4);
        assert_eq!(v4["schema"], "organism-city-observation-v4");
        for channel in 0..192 {
            assert_eq!(v3["channels"][channel] != v4["channels"][channel], matches!(channel,178|179|183));
        }
        for channel in [172usize,173,182,191] {
            assert_eq!(v4["channels"][channel]["producer"], "src/gpu/traffic.wgsl::senses");
        }
        use super::*;
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("real GPU adapter required; no CPU fallback");
        assert_ne!(adapter.get_info().device_type, wgpu::DeviceType::Cpu);
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("taxi v4 navigation fixture"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        ))
        .expect("GPU device");
        let stations = [
            [0.0, 0.0, 10000.0, 10.0, 10.0, 0.0, 0.0, 0.0],
            [30.0, 30.0, 10000.0, 10.0, 10.0, 0.0, 0.0, 0.0],
        ];
        let buffer = |data: &[f32]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("taxi v4 navigation fixture"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        };
        let size = 16usize;
        let cell = 3.0f32;
        let mut routes = vec![size as f32, cell, 2.0, 1.0];
        // A 3 m/cell legal-road field with an L-shaped route and a tied branch:
        // row 0 from x=0..10, column 10 to y=10, plus column 5 to y=5 and
        // row 5 to x=10. The direct diagonal crosses -1 off-road cells.
        for target in [(0i32, 0i32), (10i32, 10i32)] {
            for row in 0..size {
                for col in 0..size {
                    let legal = (row == 0 && col <= 10)
                        || (col == 10 && row <= 10)
                        || (col == 5 && row <= 5)
                        || (row == 5 && col >= 5 && col <= 10);
                    let distance = if legal {
                        if target == (0, 0) {
                            if row == 0 {
                                col as i32
                            } else if col == 10 {
                                row as i32 + 10
                            } else if col == 5 {
                                row as i32 + 5
                            } else {
                                col as i32 + 5
                            }
                        } else if row == 0 {
                            20 - col as i32
                        } else if col == 10 {
                            10 - row as i32
                        } else if col == 5 {
                            15 - row as i32
                        } else {
                            15 - col as i32
                        }
                    } else {
                        -1
                    };
                    routes.push(distance as f32 * cell);
                }
            }
        }
        let mut taxi = TaxiGpu::new(&device, &queue, 4, &stations);
        taxi.buf_routes = buffer(&routes);
        taxi.set_observation_version(NAVIGATION_OBSERVATION_V4);
        let mut bodies = vec![0.0f32; 4 * 128];
        let mut physics = vec![0.0f32; 4 * 92];
        for (being, position) in [(0.0, 0.0), (15.0, 15.0), (30.0, 27.0), (30.0, 30.0)]
            .into_iter()
            .enumerate()
        {
            let s = being * 128;
            bodies[s + 29] = 1.0;
            bodies[s + 65] = 1.0;
            bodies[s + 66] = 30.0;
            bodies[s + 67] = 30.0;
            bodies[s + 84] = 1.0;
            physics[being * 92 + 36] = position.0;
            physics[being * 92 + 38] = position.1;
            physics[being * 92 + 40] = 1.0;
            physics[being * 92 + 46] = 1.0;
        }
        let body = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("taxi v4 body"),
            contents: bytemuck::cast_slice(&bodies),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        let input = buffer(&physics);
        let wave = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("taxi v4 wave"),
            contents: bytemuck::cast_slice(&vec![0u32; 4 * 48]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let read = |source: &wgpu::Buffer, words: usize| -> Vec<u32> {
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (words * 4) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(source, 0, &staging, 0, (words * 4) as u64);
            queue.submit(Some(encoder.finish()));
            let (tx, rx) = std::sync::mpsc::channel();
            staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            let out =
                bytemuck::cast_slice::<u8, u32>(&staging.slice(..).get_mapped_range()).to_vec();
            staging.unmap();
            out
        };
        let run = || {
            let mut encoder = device.create_command_encoder(&Default::default());
            taxi.encode_senses(&device, &mut encoder, &body, &input, &wave, 4);
            queue.submit(Some(encoder.finish()));
            read(&wave, 4 * 48)
        };
        let output = run();
        let byte = |word: u32, lane: usize| -> i32 {
            let raw = ((word >> (lane * 8)) & 255) as i32;
            if raw > 127 { raw - 256 } else { raw }
        };
        let channel = |being: usize, channel: usize| {
            byte(output[being * 48 + channel / 4], channel % 4)
        };
        assert_eq!((channel(0, 178), channel(0, 179)), (127, 0));
        assert_eq!((channel(1, 178), channel(1, 179)), (127, 0), "forward tie wins");
        assert_eq!((channel(2, 178), channel(2, 179)), (0, 127), "near station point");
        assert_eq!((channel(3, 178), channel(3, 179), channel(3, 183)), (0, 0, 0));
        assert_eq!(channel(0, 183), 48);
        assert_eq!(channel(1, 183), 29);
        assert_eq!(channel(2, 183), 4);
        physics[36] = 5000.0;
        queue.write_buffer(&input, 0, bytemuck::cast_slice(&physics));
        let offroad = run();
        let offroad_channel = |channel: usize| {
            byte(offroad[channel / 4], channel % 4)
        };
        assert_eq!((offroad_channel(178), offroad_channel(179), offroad_channel(183)), (0, 0, 127));
        bodies[65] = 0.0;
        physics[36] = 0.0;
        queue.write_buffer(&body, 0, bytemuck::cast_slice(&bodies));
        queue.write_buffer(&input, 0, bytemuck::cast_slice(&physics));
        let idle = run();
        let idle_channel = |channel: usize| byte(idle[channel / 4], channel % 4);
        assert_eq!((idle_channel(178), idle_channel(179), idle_channel(183)), (0, 0, 127));
        assert_eq!(
            read(&body, bodies.len()),
            bytemuck::cast_slice::<f32, u32>(&bodies).to_vec(),
            "v4 senses must not mutate body state"
        );
        assert_eq!(
            read(&input, physics.len()),
            bytemuck::cast_slice::<f32, u32>(&physics).to_vec(),
            "v4 senses must not mutate physics input"
        );

        // Same road field and physical poses, with preview as the only change.
        for b in 0..4 {
            bodies[b*128+65]=1.0;
            physics[b*92+36]=18.0;physics[b*92+38]=0.0;
            physics[b*92+40]=1.0;physics[b*92+42]=0.0;
            physics[b*92+44]=0.0;physics[b*92+46]=1.0;
        }
        physics[92+40]=0.0;physics[92+42]=1.0;
        physics[92+44]=-1.0;physics[92+46]=0.0;
        physics[2*92+36]=30.0;physics[2*92+38]=27.0;
        physics[3*92+36]=5000.0;
        queue.write_buffer(&body,0,bytemuck::cast_slice(&bodies));
        queue.write_buffer(&input,0,bytemuck::cast_slice(&physics));
        let evaluate = |taxi: &TaxiGpu| {
            let mut encoder=device.create_command_encoder(&Default::default());
            taxi.encode_senses(&device,&mut encoder,&body,&input,&wave,4);
            queue.submit(Some(encoder.finish()));read(&wave,4*48)
        };
        let local=evaluate(&taxi);
        let preview_version=if std::env::var("UNIVERSE_CONTRACT_FAULT").as_deref()==Ok("no-navigation-preview") {
            NAVIGATION_OBSERVATION_V4
        } else {NAVIGATION_OBSERVATION_V5};
        taxi.set_observation_version(preview_version);
        let preview=evaluate(&taxi);
        let get=|row:&[u32],b:usize,ch:usize|byte(row[b*48+ch/4],ch%4);
        assert_eq!((get(&local,0,178),get(&local,0,179)),(127,0));
        assert_eq!((get(&preview,0,178),get(&preview,0,179)),(114,57),
            "preview must expose the turn before reaching the junction");
        assert_eq!((get(&preview,1,178),get(&preview,1,179)),(57,-114),"preview bearing uses body frame");
        assert_eq!((get(&preview,2,178),get(&preview,2,179)),(0,127),"near-station precision retained");
        assert_eq!((get(&preview,3,178),get(&preview,3,179),get(&preview,3,183)),(0,0,127));
        for b in 0..4 {for ch in 0..192 {if ch!=178 && ch!=179 {
            assert_eq!(get(&local,b,ch),get(&preview,b,ch),"preview modified unrelated channel {ch}");
        }}}
        physics[0]=20.0;
        queue.write_buffer(&input,0,bytemuck::cast_slice(&physics));
        let fast=evaluate(&taxi);
        assert_eq!((get(&fast,0,178),get(&fast,0,179)),(57,114),"faster approach needs longer preview");
        assert_eq!(read(&body,bodies.len()),bytemuck::cast_slice::<f32,u32>(&bodies).to_vec());
        assert_eq!(read(&input,physics.len()),bytemuck::cast_slice::<f32,u32>(&physics).to_vec(),"preview cannot write controls or pose");
        let v5=crate::organism_channels::channel_manifest(5);
        assert_eq!(v5["schema"],"organism-city-observation-v5");
        for ch in 0..192 {assert_eq!(v4["channels"][ch]!=v5["channels"][ch],matches!(ch,178|179));}
    }
}
