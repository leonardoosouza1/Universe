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
    padding: [u32; 3],
}

pub struct TaxiGpu {
    pub buf_world: wgpu::Buffer,
    pub buf_routes: wgpu::Buffer,
    pub capacity: usize,
    pub station_count: usize,
    initial_world: Vec<f32>,
    post_layout: wgpu::BindGroupLayout,
    birth_layout: wgpu::BindGroupLayout,
    senses_layout: wgpu::BindGroupLayout,
    post_pipeline: wgpu::ComputePipeline,
    birth_pipeline: wgpu::ComputePipeline,
    senses_pipeline: wgpu::ComputePipeline,
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
            entries: &[uniform(), storage(1, false), storage(5, false)],
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
            post_layout,
            birth_layout,
            senses_layout,
            post_pipeline,
            birth_pipeline,
            senses_pipeline,
        }
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
            padding: [0; 3],
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
        packed_wave: &wgpu::Buffer,
        n: usize,
    ) {
        if n == 0 {
            return;
        }
        assert!(bodybuf.size() >= (n * 128 * 4) as u64, "taxi body shape");
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
                    binding: 5,
                    resource: packed_wave.as_entire_binding(),
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
        assert!(moving_away[127] < approach[127]);

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
        assert_eq!(evaluated[256+127],100.0,"wall must not receive progress bonus");

    }
}
