//! Real-GPU fixture for the phase-aware task progress of `taxi.wgsl`
//! (docs/SCHOOL-INCENTIVES-20260922.md). The city, the stations and the road
//! distance fields are the production ones; poses are written by hand and only the
//! taxi post kernel runs. No CPU oracle of the kernel.
use crate::gpu::taxi::{TaxiGpu, TaxiMode, TAXI_STATION_STRIDE_FLOATS, TAXI_WORLD_HEADER_FLOATS};
use crate::organism_city::build_city;
use crate::world::{VoxelType, VoxelWorld};

const N: usize = 4;
const BODY: usize = 128;
const PHYSICS: usize = 56;
const INTENT: usize = 96;

struct Fixture {
    device: wgpu::Device,
    queue: wgpu::Queue,
    taxi: TaxiGpu,
    world: VoxelWorld,
    stations: Vec<[f32; 8]>,
    body: wgpu::Buffer,
    physics: wgpu::Buffer,
    intent: wgpu::Buffer,
    bodies: Vec<f32>,
    poses: Vec<f32>,
    intents: Vec<f32>,
    tick: u32,
}

#[test]
#[ignore = "needs the real GPU"]
fn taxi_mode_controls_passenger_issuance() {
    let waiting_id = |mode| {
        let mut fixture = Fixture::new();
        fixture.taxi.set_mode(mode);
        let (_, world) = fixture.step();
        world[TAXI_WORLD_HEADER_FLOATS + 11]
    };

    assert!(waiting_id(TaxiMode::Production) > 0.0);
    assert_eq!(waiting_id(TaxiMode::MotorBench), 0.0);
    assert_eq!(waiting_id(TaxiMode::ServiceBench), 0.0);
}

impl Fixture {
    fn new() -> Self {
        let _quiet = crate::world::silence_map_diag();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("real Vulkan GPU required");
        assert_ne!(adapter.get_info().device_type, wgpu::DeviceType::Cpu, "CPU fallback forbidden");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("taxi-progress-fixture"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        ))
        .expect("real GPU device");
        let mut world = VoxelWorld::new(7, 12.0);
        let stations = build_city(&mut world).stations;
        crate::organism_scenario::validate_city(&world, &stations).expect("valid city");
        let mut taxi = TaxiGpu::new(&device, &queue, N, &stations);
        taxi.configure_routes(&device, &world, &stations).expect("road routes");
        taxi.reset_world(&queue);
        let make = |floats: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("taxi-progress-fixture"),
                size: (floats * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let mut bodies = vec![0.0f32; N * BODY];
        for b in 0..N {
            let s = b * BODY;
            bodies[s] = 1.0e6;
            bodies[s + 18] = 1.0e6;
            bodies[s + 8] = 1.0;
            bodies[s + 29] = 1.0;
        }
        Self {
            body: make(N * BODY),
            physics: make(N * PHYSICS),
            intent: make(N * INTENT),
            device,
            queue,
            taxi,
            world,
            stations,
            bodies,
            poses: vec![0.0; N * PHYSICS],
            intents: vec![0.0; N * INTENT],
            tick: 0,
        }
    }

    fn place(&mut self, b: usize, pos: [f32; 2]) {
        self.poses[b * PHYSICS + 36] = pos[0];
        self.poses[b * PHYSICS + 38] = pos[1];
    }

    fn request(&mut self, b: usize, channel: usize, on: bool) {
        self.intents[b * INTENT + channel] = if on { 1.0 } else { 0.0 };
    }

    /// One post-kernel step; returns the body records and the world buffer.
    fn step(&mut self) -> (Vec<f32>, Vec<f32>) {
        self.queue.write_buffer(&self.body, 0, bytemuck::cast_slice(&self.bodies));
        self.queue.write_buffer(&self.physics, 0, bytemuck::cast_slice(&self.poses));
        self.queue.write_buffer(&self.intent, 0, bytemuck::cast_slice(&self.intents));
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.taxi.encode_post(&self.device, &mut encoder, &self.physics, &self.body, &self.intent, N, 1.0 / 64.0, self.tick);
        let world_floats = TAXI_WORLD_HEADER_FLOATS + self.stations.len() * TAXI_STATION_STRIDE_FLOATS;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("taxi-progress-staging"),
            size: ((N * BODY + world_floats) * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&self.body, 0, &staging, 0, (N * BODY * 4) as u64);
        encoder.copy_buffer_to_buffer(&self.taxi.buf_world, 0, &staging, (N * BODY * 4) as u64, (world_floats * 4) as u64);
        self.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let out: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        drop(slice);
        staging.unmap();
        self.tick += 1;
        // The kernel owns the task slots; carry its body writes into the next step.
        self.bodies.copy_from_slice(&out[..N * BODY]);
        (out[..N * BODY].to_vec(), out[N * BODY..].to_vec())
    }

    fn is_road(&self, x: f32, z: f32) -> bool {
        let cell = self.world.cell_size;
        let size = self.world.size as i32;
        let col = (x / cell).round() as i32;
        let row = (z / cell).round() as i32;
        col >= 0 && row >= 0 && col < size && row < size && self.world.grid[(row * size + col) as usize] == VoxelType::Road
    }

    /// A straight road segment of `length_m` leaving the station along one axis.
    fn road_direction_from(&self, station: [f32; 2], length_m: f32) -> [f32; 2] {
        for dir in [[1.0f32, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]] {
            let mut d = 0.0;
            let mut ok = true;
            while d <= length_m {
                if !self.is_road(station[0] + dir[0] * d, station[1] + dir[1] * d) {
                    ok = false;
                    break;
                }
                d += self.world.cell_size;
            }
            if ok {
                return dir;
            }
        }
        panic!("no straight road of {length_m} m leaves the station");
    }
}

fn fitness_of(body: &[f32]) -> f64 {
    let mut state = crate::gpu::organism::OrganismState { values: [0.0; crate::gpu::organism::STATE_STRIDE_F32] };
    state.values.copy_from_slice(body);
    crate::organism_scenario::fitness(&state)
}

#[test]
#[ignore = "needs the real GPU"]
fn approach_pays_for_closed_road_distance_only_once() {
    let mut f = Fixture::new();
    let station = [f.stations[0][0], f.stations[0][1]];
    let dir = f.road_direction_from(station, 42.0);
    let far = [station[0] + dir[0] * 36.0, station[1] + dir[1] * 36.0];
    let mid = [station[0] + dir[0] * 18.0, station[1] + dir[1] * 18.0];
    let near = [station[0] + dir[0] * 3.0, station[1] + dir[1] * 3.0];
    // An off-road cell beside the street (streets are 15 m wide, 192 m apart).
    let grass = [30.0f32, -30.0, 45.0, -45.0]
        .iter()
        .flat_map(|side| [[far[0] + dir[1] * side, far[1] + dir[0] * side], [mid[0] + dir[1] * side, mid[1] + dir[0] * side]])
        .find(|p| !f.is_road(p[0], p[1]))
        .expect("an off-road cell beside the street");
    f.place(0, far);
    f.place(1, far);
    f.place(2, near);
    f.place(3, grass);
    let (body, world) = f.step();
    let id = world[TAXI_WORLD_HEADER_FLOATS + 11];
    assert!(id > 0.0, "station 0 issues a passenger at tick 0");
    for b in [0usize, 1, 2] {
        assert_eq!(body[b * BODY + 31], id * 2.0, "waiting key for being {b}");
        assert_eq!(body[b * BODY + 127], 0.0, "no credit at the first sight, being {b}");
    }
    // Road distances are BFS sums of 3 m cells, so the pose-to-cell rounding is ±1 cell.
    assert!((body[62] - 36.0).abs() <= 3.0 + 1e-3, "initial route distance {}", body[62]);
    assert!(body[2 * BODY + 62] >= 0.0 && body[2 * BODY + 62] < 8.0, "near start {}", body[2 * BODY + 62]);
    assert_eq!(body[3 * BODY + 62], -1.0, "unreachable start records -1");
    let initial = body[62];

    f.place(0, mid);
    let (body, _) = f.step();
    let expected = 100.0 * (1.0 - body[61] / initial);
    assert!((body[127] - expected).abs() < 1e-3 && (40.0..=60.0).contains(&body[127]), "halfway pays ~50: {} (best {} of {})", body[127], body[61], initial);
    assert_eq!(body[BODY + 127], 0.0, "idle being earns nothing");
    assert_eq!(body[2 * BODY + 127], 0.0, "born within 8 m earns nothing");
    let halfway = body[127];
    let best = body[61];

    // Driving back does not lose the credit, and does not add any either.
    f.place(0, far);
    let (body, _) = f.step();
    assert_eq!(body[127], halfway, "oscillation keeps the credit: {}", body[127]);
    assert_eq!(body[61], best, "best remaining is kept: {}", body[61]);

    // The unreachable start never pays for this key even once back on the road.
    f.place(3, mid);
    let (body, _) = f.step();
    assert_eq!(body[3 * BODY + 127], 0.0, "unreachable start earns nothing for this key");
    assert_eq!(body[3 * BODY + 62], -1.0);
}

#[test]
#[ignore = "needs the real GPU"]
fn delivery_outranks_approach_outranks_idle() {
    let mut f = Fixture::new();
    let station = [f.stations[0][0], f.stations[0][1]];
    let dir = f.road_direction_from(station, 42.0);
    let far = [station[0] + dir[0] * 36.0, station[1] + dir[1] * 36.0];
    let mid = [station[0] + dir[0] * 18.0, station[1] + dir[1] * 18.0];
    for b in 0..N {
        f.place(b, far);
    }
    let (_, world) = f.step();
    let destination_index = world[TAXI_WORLD_HEADER_FLOATS + 12] as usize;
    let destination = [f.stations[destination_index][0], f.stations[destination_index][1]];
    assert_ne!(destination_index, 0);

    f.place(0, mid);
    let (body, _) = f.step();
    let approach_fitness = fitness_of(&body[..BODY]);
    let idle_fitness = fitness_of(&body[BODY..2 * BODY]);
    assert!((40.0..=60.0).contains(&body[127]), "halfway pays ~50: {}", body[127]);
    let approach_credit = body[127];

    // Pickup: at the station, stationary, pickup requested.
    f.place(0, station);
    f.request(0, 6, true);
    let (body, _) = f.step();
    f.request(0, 6, false);
    assert_eq!(body[65], 1.0, "onboard after pickup");
    assert_eq!(body[31], body[72] * 2.0 + 1.0, "carrying key");
    assert!((body[127] - 100.0).abs() < 1e-3, "carrying pays the phase bonus only: {}", body[127]);
    assert!(body[62] > 8.0, "route to the destination {}", body[62]);
    let carrying_fitness = fitness_of(&body[..BODY]);

    // Arrive: remaining 0 pays the full approach; then drop off.
    f.place(0, destination);
    let (body, _) = f.step();
    assert!((body[127] - 200.0).abs() < 1e-3, "at the destination: {}", body[127]);
    f.request(0, 7, true);
    let (body, _) = f.step();
    assert_eq!(body[68], 1.0, "delivered");
    assert_eq!(body[65], 0.0);
    let delivered_fitness = fitness_of(&body[..BODY]);
    assert!(body[127] < 100.0, "no carrying bonus after the drop-off: {}", body[127]);

    assert!(delivered_fitness > carrying_fitness, "{delivered_fitness} vs {carrying_fitness}");
    assert!(carrying_fitness > approach_fitness, "{carrying_fitness} vs {approach_fitness}");
    assert!(approach_fitness > idle_fitness, "{approach_fitness} vs {idle_fitness}");
    assert!((approach_fitness - idle_fitness - approach_credit as f64).abs() < 1e-3);
}
