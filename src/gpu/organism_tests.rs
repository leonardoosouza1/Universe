//! Real-GPU body boundary fixtures. No CPU physics, forward pass, or reference model.
//! Explicit stationary physics records isolate body budgets from driving skill.
use crate::gpu::organism::*;

struct Fixture {
    device: wgpu::Device,
    queue: wgpu::Queue,
    body: OrganismGpu,
    input: wgpu::Buffer,
    output: wgpu::Buffer,
}
impl Fixture {
    fn new() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("real Vulkan GPU required");
        assert_ne!(
            adapter.get_info().device_type,
            wgpu::DeviceType::Cpu,
            "CPU fallback forbidden"
        );
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("organism-body-budget-fixtures"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        ))
        .expect("real GPU device");
        let mut body = OrganismGpu::new(&device, &queue, 1);
        body.set_config_flags(CONFIG_DISABLE_OLD_THERMAL).unwrap();
        let make_buffer = |floats: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("explicit-physics-fixture"),
                size: (floats * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let input = make_buffer(PHYSICS_INPUT_STRIDE_F32);
        let output = make_buffer(PHYSICS_OUTPUT_STRIDE_F32);
        Self {
            device,
            queue,
            body,
            input,
            output,
        }
    }
    fn reset(&self, genes: [f32; 16]) -> OrganismState {
        self.body.reset(&self.queue, &[genes]).unwrap();
        let mut input = [0.0f32; PHYSICS_INPUT_STRIDE_F32];
        input[48] = 620.0;
        input[49] = 800.0;
        input[50] = 7000.0;
        input[55] = f32::from_bits(2); // encoded u32 first gear
        input[62] = 1350.0;
        input[68] = 4000.0;
        input[87] = f32::from_bits(CONFIG_DISABLE_OLD_THERMAL);
        self.queue
            .write_buffer(&self.input, 0, bytemuck::cast_slice(&input));
        let mut output = [0.0f32; PHYSICS_OUTPUT_STRIDE_F32];
        output[48] = 800.0;
        output[53] = 20.0;
        output[54] = 20.0;
        self.queue
            .write_buffer(&self.output, 0, bytemuck::cast_slice(&output));
        self.snapshot()
    }
    fn write(&self, state: &OrganismState) {
        self.queue
            .write_buffer(&self.body.buf_state, 0, bytemuck::bytes_of(state));
    }
    fn snapshot(&self) -> OrganismState {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.body.encode_snapshot(&mut encoder, 1).unwrap();
        self.queue.submit(Some(encoder.finish()));
        self.body.read_snapshot(&self.device, 1).unwrap()[0]
    }
    fn step(&self, dt: f32) -> OrganismState {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.body
            .encode_pre(&self.device, &mut encoder, &self.input, 1, dt)
            .unwrap();
        self.body
            .encode_post(&self.device, &mut encoder, &self.output, 1, dt)
            .unwrap();
        self.body.encode_snapshot(&mut encoder, 1).unwrap();
        self.queue.submit(Some(encoder.finish()));
        self.body.read_snapshot(&self.device, 1).unwrap()[0]
    }
    fn read_input(&self) -> Vec<f32> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dead-input-readback"),
            size: self.input.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.input, 0, &staging, 0, self.input.size());
        self.queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = staging.slice(..).get_mapped_range();
        let result = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }
}

#[test]
#[ignore = "exclusive real GPU; six scripted physiology cases, no learned-policy claim"]
fn gpu_body_paid_repair_growth_cooling_age_death_and_birth_contracts() {
    let fixture = Fixture::new();

    // Negative capacity genes must be valid at birth, not rejected only at first readback.
    let born = fixture.reset([-1.0; 16]);
    assert_eq!(born.values[S_ENERGY_J], born.values[S_BIRTH_ENERGY_CAP]);
    assert_eq!(born.values[S_WATER_KG], born.values[S_BIRTH_WATER_CAP]);
    assert_eq!(
        born.values[S_MATERIAL_KG],
        born.values[S_BIRTH_MATERIAL_CAP]
    );
    assert_eq!(born.values[S_TEMPERATURE_K], 293.15);
    assert_eq!(born.values[S_AGE_S], 0.0);

    // An old, resting body repairs only with material; repair never erases age.
    let mut old = fixture.reset([0.0; 16]);
    old.values[S_ENERGY_J] = 10000.0;
    old.values[S_AGE_S] = 7200.0;
    old.values[S_MATURITY] = 1.0;
    old.values[S_ENGINE_WEAR] = 0.5;
    old.values[S_TYRE_WEAR] = 0.5;
    old.values[S_MATERIAL_KG] = 0.0;
    fixture.write(&old);
    let starved = fixture.step(1.0);
    assert_eq!(starved.values[S_ENGINE_WEAR], 0.5);
    assert_eq!(starved.values[S_TYRE_WEAR], 0.5);
    assert_eq!(starved.values[44], 0.0);
    assert_eq!(starved.values[S_AGE_S], 7201.0);
    old.values[S_MATERIAL_KG] = 1.0;
    fixture.write(&old);
    let repaired = fixture.step(1.0);
    assert!(repaired.values[S_ENGINE_WEAR] < 0.5 && repaired.values[S_TYRE_WEAR] < 0.5);
    assert!(repaired.values[44] > 0.0 && repaired.values[45] > 0.0);
    assert_eq!(repaired.values[S_AGE_S], 7201.0);
    assert!(
        (10000.0
            - repaired.values[S_ENERGY_J]
            - repaired.values[51]
            - repaired.values[52]
            - repaired.values[44])
            .abs()
            < 0.002
    );
    assert!((1.0 - repaired.values[S_MATERIAL_KG] - repaired.values[45]).abs() < 0.00001);
    assert_eq!(repaired.values[63], 0.0);

    // Simultaneous growth and repair cannot spend the same joule/material twice.
    let mut growing = fixture.reset([0.0; 16]);
    growing.values[S_ENERGY_J] = 1000.0;
    growing.values[S_MATERIAL_KG] = 1.0;
    growing.values[S_MATURITY] = 0.5;
    growing.values[S_ENGINE_WEAR] = 0.5;
    growing.values[S_TYRE_WEAR] = 0.5;
    fixture.write(&growing);
    let paid = fixture.step(1.0);
    let growth_paid = paid.values[S_PAID_GROWTH_J] - growing.values[S_PAID_GROWTH_J];
    assert!(growth_paid > 0.0 && paid.values[44] > 0.0);
    let energy_spent = paid.values[51] + paid.values[52] + paid.values[44] + growth_paid;
    assert!((1000.0 - paid.values[S_ENERGY_J] - energy_spent).abs() < 0.002);
    let growth_fraction = paid.values[S_MATURITY] - growing.values[S_MATURITY];
    assert!(
        (1.0 - paid.values[S_MATERIAL_KG] - paid.values[45] - growth_fraction * 0.2).abs()
            < 0.00001
    );
    assert!(paid.values[S_ENERGY_J] >= 0.0 && paid.values[S_MATERIAL_KG] >= 0.0);
    assert_eq!(paid.values[63], 0.0);

    // A budget smaller than basal demand cannot be reused by growth or repair.
    let mut depleted = fixture.reset([0.0; 16]);
    depleted.values[S_ENERGY_J] = 5.0;
    depleted.values[S_MATURITY] = 0.5;
    depleted.values[S_ENGINE_WEAR] = 0.5;
    fixture.write(&depleted);
    let exhausted = fixture.step(1.0);
    assert_eq!(exhausted.values[S_ENERGY_J], 0.0);
    assert_eq!(exhausted.values[51], 5.0);
    assert_eq!(exhausted.values[S_PAID_GROWTH_J], 0.0);
    assert_eq!(exhausted.values[44], 0.0);
    assert_eq!(exhausted.values[S_ENGINE_WEAR], 0.5);
    assert_eq!(exhausted.values[29], 0.0);

    // Compare GPU time subdivisions, not a CPU thermal integrator. Zero torque,
    // full water and disabled growth/repair isolate cooling plus constant basal heat.
    let mut genes = [0.0; 16];
    genes[6] = -1.0;
    genes[7] = -1.0;
    let initial = fixture.reset(genes);
    let mut warm = initial;
    warm.values[S_TEMPERATURE_K] = 350.0;
    warm.values[S_MATURITY] = 1.0;
    let mut temperatures = Vec::new();
    for (dt, steps) in [(1.0, 2), (0.5, 4), (0.25, 8), (0.125, 16)] {
        fixture.write(&warm);
        let mut last = warm;
        for _ in 0..steps {
            last = fixture.step(dt);
        }
        assert!(last.values[S_TEMPERATURE_K] < 350.0);
        assert!(last.values[S_TEMPERATURE_K] > 293.15);
        assert_eq!(last.values[S_AGE_S], 2.0);
        temperatures.push(last.values[S_TEMPERATURE_K]);
    }
    // The implicit two-compartment update is first-order in dt. Compare adjacent
    // GPU refinements rather than asserting equality against a CPU thermal oracle:
    // each difference should shrink by roughly two when dt is halved. The final
    // residual below 0.01 K is an explicit f32 acceptance bound for this fixture.
    let d01 = (temperatures[0] - temperatures[1]).abs();
    let d12 = (temperatures[1] - temperatures[2]).abs();
    let d23 = (temperatures[2] - temperatures[3]).abs();
    assert!(
        d01 > d12 && d12 > d23,
        "cooling not converging: {temperatures:?}"
    );
    assert!(
        (1.5..=2.5).contains(&(d01 / d12)),
        "first refinement ratio: {temperatures:?}"
    );
    assert!(
        (1.5..=2.5).contains(&(d12 / d23)),
        "second refinement ratio: {temperatures:?}"
    );
    assert!(
        d23 < 0.01,
        "f32 convergence residual too large: {temperatures:?}"
    );

    // Dead bodies do not consume or age; pre-phase explicitly cuts power and brakes.
    let mut dead = fixture.reset([0.0; 16]);
    dead.values[29] = 0.0;
    dead.values[S_AGE_S] = 7200.0;
    fixture.write(&dead);
    fixture
        .queue
        .write_buffer(&fixture.input, (51 * 4) as u64, bytemuck::bytes_of(&1.0f32));
    fixture.queue.write_buffer(
        &fixture.input,
        (54 * 4) as u64,
        bytemuck::bytes_of(&100.0f32),
    );
    let remains = fixture.step(1.0);
    let input = fixture.read_input();
    assert_eq!(input[48], 0.0);
    assert_eq!(input[51], 0.0);
    assert_eq!(input[54], 0.0);
    assert_eq!(input[52], 1.0);
    assert_eq!(remains.values[57], 0.0);
    assert_eq!(remains.values[59], 1.0);
    assert_eq!(remains.values[60], 1.0);
    for slot in (0..128).filter(|i| !(57..=60).contains(i)) {
        assert_eq!(
            remains.values[slot], dead.values[slot],
            "dead resource/state slot {slot}"
        );
    }
}

#[test]
#[ignore = "exclusive real GPU; verifies the two-temperature resident balance"]
fn gpu_body_two_temperature_exchange_and_energy_budget() {
    let fixture = Fixture::new();
    let mut genes = [0.0; 16];
    genes[6] = -1.0; // no repair demand
    genes[7] = -1.0; // no growth demand
    let mut state = fixture.reset(genes);
    state.values[S_MATURITY] = 1.0;
    state.values[S_TEMPERATURE_K] = 360.0;
    state.values[S_OIL_TEMPERATURE_K] = 300.0;
    state.values[S_ENERGY_J] = 100_000.0;
    fixture.write(&state);
    let after = fixture.step(0.25);

    assert!(after.values[S_TEMPERATURE_K].is_finite());
    assert!(after.values[S_OIL_TEMPERATURE_K].is_finite());
    assert!(after.values[S_OIL_TEMPERATURE_K] > 300.0);
    assert!(after.values[S_TEMPERATURE_K] < 360.0);
    assert!((after.values[S_TEMPERATURE_K] - after.values[S_OIL_TEMPERATURE_K]).abs() < 60.0);
    // The thermal exchange does not mint or destroy body budget: only the
    // explicitly reported basal/motion payments leave the energy compartment.
    let spent = after.values[51]
        + after.values[52]
        + after.values[S_PAID_REPAIR_J]
        + (after.values[S_PAID_GROWTH_J] - state.values[S_PAID_GROWTH_J]);
    assert!((state.values[S_ENERGY_J] - after.values[S_ENERGY_J] - spent).abs() < 0.01);
}

#[test]
#[ignore = "exclusive real GPU; fixed full-load/rest time constants and unclipped thermal balance"]
fn gpu_body_fatigue_time_constants_and_unclipped_temperature() {
    let fixture = Fixture::new();
    let mut genes = [0.0; 16];
    genes[6] = -1.0;
    genes[7] = -1.0;
    let mut state = fixture.reset(genes);
    state.values[S_MATURITY] = 1.0;
    fixture.write(&state);
    // Fixed physical-output fixture, not a CPU engine: 100 Nm crank torque at
    // 1000 RPM exceeds the 10kW normalized chemical-load threshold.
    fixture.queue.write_buffer(
        &fixture.output,
        48 * 4,
        bytemuck::cast_slice(&[1000.0f32, 1406.0]),
    );
    let mut after = state;
    for _ in 0..8 {
        after = fixture.step(0.25);
    }
    assert!(
        (after.values[S_FATIGUE] - 2.0 / 120.0).abs() < 0.000001,
        "two-second load fatigue {}",
        after.values[S_FATIGUE]
    );
    assert!(
        after.values[S_FATIGUE] < 0.02,
        "full load must not exhaust body in two seconds"
    );
    let mut resting = fixture.reset(genes);
    resting.values[S_FATIGUE] = 0.5;
    fixture.write(&resting);
    let recovered = fixture.step(1.0);
    assert!((recovered.values[S_FATIGUE] - (0.5 - 1.0 / 60.0)).abs() < 0.000001);

    for initial_temperature in [230.0f32, 650.0] {
        let mut state = fixture.reset(genes);
        state.values[S_TEMPERATURE_K] = initial_temperature;
        state.values[S_OIL_TEMPERATURE_K] = initial_temperature;
        // Dry compartment isolates thermal energy balance from latent heat; post
        // marks it dead after this single paid transition, as required by water=0.
        state.values[S_WATER_KG] = 0.0;
        fixture.write(&state);
        let after = fixture.step(0.25);
        assert!(after.values[S_TEMPERATURE_K].is_finite() && after.values[S_TEMPERATURE_K] > 0.0);
        assert!(
            after.values[S_OIL_TEMPERATURE_K].is_finite()
                && after.values[S_OIL_TEMPERATURE_K] > 0.0
        );
        if initial_temperature < 250.0 {
            assert!(after.values[S_TEMPERATURE_K] < 250.0);
        } else {
            assert!(after.values[S_TEMPERATURE_K] > 600.0);
        }
        // Sum of compartment changes must equal paid heat minus heat to ambient.
        // This is a conservation identity over GPU results, not a CPU integrator.
        let body_capacity = 12000.0 + state.values[S_BIRTH_THERMAL_MARGIN] * 100.0;
        let delta = body_capacity * (after.values[S_TEMPERATURE_K] - initial_temperature)
            + 6000.0 * (after.values[S_OIL_TEMPERATURE_K] - initial_temperature);
        let cooling = 120.0 * after.values[37] * 0.25;
        let net =
            after.values[15] * 0.25 - cooling * (after.values[S_OIL_TEMPERATURE_K] - 293.15) * 0.25;
        assert!(
            (delta - net).abs() < 3.0,
            "unclipped thermal balance at {initial_temperature}K: delta={delta}, net={net}"
        );
    }
}

#[test]
fn organism_temperature_validation_requires_positive_finite_kelvin() {
    let mut state = birth_state(&[0.0; 16], CONFIG_DISABLE_OLD_THERMAL).unwrap();
    state.values[S_TEMPERATURE_K] = 650.0;
    state.values[S_OIL_TEMPERATURE_K] = 230.0;
    assert!(validate_state(&state).is_ok());
    for invalid in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let mut invalid_state = state;
        invalid_state.values[S_OIL_TEMPERATURE_K] = invalid;
        assert!(validate_state(&invalid_state).is_err());
        invalid_state = state;
        invalid_state.values[S_TEMPERATURE_K] = invalid;
        assert!(validate_state(&invalid_state).is_err());
    }
}

#[test]
#[ignore = "exclusive real GPU; resident hot/cold resource stress at production dt"]
fn gpu_body_resident_water_heat_and_fuel_stress() {
    let fixture = Fixture::new();
    let mut genes = [0.0; 16];
    genes[6] = -1.0; // Isolate resource/thermal budgets from repair and growth.
    genes[7] = -1.0;
    let mut outcomes = Vec::new();
    for kelvin in [313.15, 393.15] {
        let mut initial = fixture.reset(genes);
        initial.values[S_MATURITY] = 1.0;
        initial.values[S_TEMPERATURE_K] = kelvin;
        initial.values[S_OIL_TEMPERATURE_K] = kelvin;
        fixture.write(&initial);
        // A controlled mechanical load isolates physiology. This is not a
        // driving policy or a CPU motor model. The shader owns all transitions.
        fixture.queue.write_buffer(&fixture.output, 48 * 4,
            bytemuck::cast_slice(&[1000.0f32, 1406.0]));
        let mut encoder = fixture.device.create_command_encoder(&Default::default());
        for _ in 0..512 {
            fixture.body.encode_pre(&fixture.device, &mut encoder, &fixture.input, 1, 1.0 / 64.0).unwrap();
            fixture.body.encode_post(&fixture.device, &mut encoder, &fixture.output, 1, 1.0 / 64.0).unwrap();
        }
        fixture.body.encode_snapshot(&mut encoder, 1).unwrap();
        fixture.queue.submit(Some(encoder.finish()));
        let after = fixture.body.read_snapshot(&fixture.device, 1).unwrap()[0];
        assert_eq!(after.values[S_AGE_S], 8.0);
        assert_eq!(after.values[29], 1.0);
        assert!(after.values[S_ENERGY_J] < initial.values[S_ENERGY_J] - 100_000.0,
            "mechanical work must debit chemical energy at production dt");
        assert!(after.values[S_ENGINE_WEAR] > initial.values[S_ENGINE_WEAR]);
        assert!(after.values[S_FATIGUE] > 0.05);
        assert!(after.values.iter().all(|v| v.is_finite()));
        outcomes.push(after);
    }
    // The current model is a closed coolant circuit: healthy cold bodies do
    // not drink continuously. Boiling/damage consume water; ordinary motion
    // consumes chemical energy. Do not invent water loss just to animate a bar.
    assert_eq!(outcomes[0].values[S_WATER_KG], outcomes[0].values[S_BIRTH_WATER_CAP]);
    assert!(outcomes[1].values[S_WATER_KG] < outcomes[0].values[S_WATER_KG] - 0.01,
        "hot body must consume coolant under sustained load");
    assert!(outcomes[1].values[S_TEMPERATURE_K] < 393.15);
    eprintln!("RESOURCE STRESS PASS: 8s resident dt=1/64, cold/hot water={}/{}, energy={}/{}, hot temperature={}K",
        outcomes[0].values[S_WATER_KG], outcomes[1].values[S_WATER_KG],
        outcomes[0].values[S_ENERGY_J], outcomes[1].values[S_ENERGY_J], outcomes[1].values[S_TEMPERATURE_K]);
}

#[test]
#[ignore = "exclusive real GPU; actual A/B/C physics with extreme genes and scarce energy"]
fn gpu_body_extreme_genes_reserve_basal_before_motion() {
    let Fixture {
        device,
        queue,
        body,
        ..
    } = Fixture::new();
    let device = std::sync::Arc::new(device);
    let queue = std::sync::Arc::new(queue);
    let physics = crate::gpu::physics::PhysicsGpu::new_shared(
        device.clone(),
        queue.clone(),
        "fixture shared real adapter".into(),
        &vec![1u8; 32 * 32],
        &vec![0.0; 32 * 32],
        32,
        1.0,
        1,
    )
    .unwrap();
    let dt = 1.0 / 64.0;
    let mut observed_paid_motion = false;
    for genes in [[-1.0; 16], [1.0; 16]] {
        for energy in [1.0f32, 1000.0] {
            body.reset(&queue, &[genes]).unwrap();
            let mut birth = birth_state(&genes, CONFIG_DISABLE_OLD_THERMAL).unwrap();
            birth.values[S_ENERGY_J] = energy;
            birth.values[S_MATURITY] = 1.0;
            queue.write_buffer(&body.buf_state, 0, bytemuck::bytes_of(&birth));
            let mut state =
                crate::gpu::physics::make_physics_state([16.0, 0.5, 16.0], [1.0, 0.0, 0.0]);
            state.dt = dt;
            state.throttle = 1.0;
            state.max_rpm = 7000.0;
            state.current_rpm = 7000.0;
            state.current_gear = 2;
            physics.upload_resident_state(&[state]).unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            body.encode_pre(&device, &mut encoder, physics.input_buffer(), 1, dt)
                .unwrap();
            physics
                .encode_step(&mut encoder, physics.input_buffer(), 1)
                .unwrap();
            body.encode_post(&device, &mut encoder, physics.output_buffer(), 1, dt)
                .unwrap();
            body.encode_snapshot(&mut encoder, 1).unwrap();
            queue.submit(Some(encoder.finish()));
            let after = body.read_snapshot(&device, 1).unwrap()[0];
            assert!(
                after.values[63] < 0.001,
                "unpaid motion with energy={energy}, mass={} debt={}",
                birth.values[S_BIRTH_MASS],
                after.values[63]
            );
            let spent = after.values[51]
                + after.values[52]
                + after.values[S_PAID_GROWTH_J]
                + after.values[S_PAID_REPAIR_J];
            assert!((energy - after.values[S_ENERGY_J] - spent).abs() < 0.001);
            if energy == 1.0 {
                assert_eq!(after.values[52], 0.0);
                assert_eq!(after.values[51], 1.0);
            }
            observed_paid_motion |= after.values[52] > 0.0;
        }
    }
    assert!(
        observed_paid_motion,
        "fixture must exercise powered motion, not only idle bodies"
    );
}
