//! Real-GPU fixture for the online feedback adapter (docs/SCHOOL-INCENTIVES-20260922.md).
//! Only the feedback kernel runs, on hand-written body states; the check is the sign
//! of a loss/recovery energy cycle in each reward mode and the paid-money cost.
use crate::gpu::organism_feedback::OrganismFeedback;
use crate::gpu::lifetime::{LIFE_AUX_MONEY_OFFSET, LIFE_AUX_PROGRESS_OFFSET, LIFE_AUX_TASK_KEY_OFFSET, LIFE_FEEDBACK_OFFSET, LIFETIME_STRIDE};

const N: usize = 2;
const BODY: usize = 128;

struct Fixture {
    device: wgpu::Device,
    queue: wgpu::Queue,
    feedback: OrganismFeedback,
    body: wgpu::Buffer,
    life: wgpu::Buffer,
    values: Vec<f32>,
}

impl Fixture {
    fn new() -> Self {
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
                label: Some("organism-feedback-fixture"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        ))
        .expect("real GPU device");
        let make = |floats: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("feedback-fixture"),
                size: (floats * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let body = make(N * BODY);
        let life = make(N * LIFETIME_STRIDE);
        queue.write_buffer(&life, 0, bytemuck::cast_slice(&vec![0.0f32; N * LIFETIME_STRIDE]));
        let mut values = vec![0.0f32; N * BODY];
        for b in 0..N {
            let s = b * BODY;
            values[s] = 1000.0; // energy J
            values[s + 18] = 1000.0; // energy capacity
            values[s + 8] = 1.0; // integrity
            values[s + 29] = 1.0; // alive
            values[s + 64] = 25.0; // money
        }
        let feedback = OrganismFeedback::new(&device);
        Self { device, queue, feedback, body, life, values }
    }

    fn encode(&mut self, mode: u32) -> Vec<f32> {
        self.queue.write_buffer(&self.body, 0, bytemuck::cast_slice(&self.values));
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.feedback.encode(&self.device, &mut encoder, &self.body, &self.life, N, mode);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("feedback-staging"),
            size: ((N * LIFETIME_STRIDE + N * BODY) * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&self.life, 0, &staging, 0, (N * LIFETIME_STRIDE * 4) as u64);
        encoder.copy_buffer_to_buffer(&self.body, 0, &staging, (N * LIFETIME_STRIDE * 4) as u64, (N * BODY * 4) as u64);
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
        // The kernel rewrote the adapter baselines (94..97); keep them for the next step.
        self.values.copy_from_slice(&out[N * LIFETIME_STRIDE..]);
        out[..N * LIFETIME_STRIDE].to_vec()
    }
}

fn reward(life: &[f32], b: usize) -> f32 {
    life[b * LIFETIME_STRIDE + LIFE_FEEDBACK_OFFSET]
}

/// One loss of q followed by one recovery of q. Legacy pays +0.19 q/capacity for the
/// cycle; symmetric pays exactly zero, so refuelling loops are not rewarded.
fn energy_cycle(mode: u32) -> (f32, f32, f32) {
    let mut f = Fixture::new();
    let armed = f.encode(mode);
    assert_eq!(reward(&armed, 0), 0.0, "first decision is unarmed");
    assert_eq!(f.values[97], 1.0, "adapter armed after the first pass");
    f.values[0] = 500.0;
    let loss = reward(&f.encode(mode), 0);
    f.values[0] = 1000.0;
    let gain = reward(&f.encode(mode), 0);
    (loss, gain, loss + gain)
}

#[test]
#[ignore = "needs the real GPU"]
fn legacy_energy_cycle_is_rewarded() {
    let (loss, gain, cycle) = energy_cycle(0);
    assert!((loss - -0.005).abs() < 1e-6, "loss {loss}");
    assert!((gain - 0.1).abs() < 1e-6, "gain {gain}");
    assert!((cycle - 0.095).abs() < 1e-6, "legacy cycle {cycle}");
}

#[test]
#[ignore = "needs the real GPU"]
fn symmetric_energy_cycle_nets_zero() {
    let (loss, gain, cycle) = energy_cycle(1);
    assert!((loss - -0.005).abs() < 1e-6, "loss {loss}");
    assert!((gain - 0.005).abs() < 1e-6, "gain {gain}");
    assert!(cycle.abs() < 1e-6, "symmetric cycle {cycle}");
}

#[test]
#[ignore = "needs the real GPU"]
fn symmetric_charges_paid_money_and_ignores_income() {
    let mut f = Fixture::new();
    let armed = f.encode(1);
    assert_eq!(armed[LIFE_AUX_MONEY_OFFSET], 25.0, "money baseline recorded at arming");
    // Being 0 pays 5 for a service; being 1 earns a 10 fare.
    f.values[64] = 20.0;
    f.values[BODY + 64] = 35.0;
    let life = f.encode(1);
    assert!((reward(&life, 0) - -0.002).abs() < 1e-6, "paid cost {}", reward(&life, 0));
    assert_eq!(reward(&life, 1), 0.0, "income is not an online reward: {}", reward(&life, 1));
    assert_eq!(life[LIFE_AUX_MONEY_OFFSET], 20.0);
    assert_eq!(life[LIFETIME_STRIDE + LIFE_AUX_MONEY_OFFSET], 35.0);
    // Legacy ignores money entirely.
    f.values[64] = 10.0;
    let legacy = f.encode(0);
    assert_eq!(reward(&legacy, 0), 0.0);
}

/// Progress mode pays only for a rise of body 127 on an unchanged task key.
#[test]
#[ignore = "needs the real GPU"]
fn progress_mode_pays_same_key_gains_only() {
    let mut f = Fixture::new();
    f.values[31] = 2.0; // waiting passenger 1
    f.values[127] = 0.0;
    let armed = f.encode(2);
    assert_eq!(armed[LIFE_AUX_PROGRESS_OFFSET], 0.0);
    assert_eq!(armed[LIFE_AUX_TASK_KEY_OFFSET], 2.0);
    f.values[127] = 50.0; // closed half the road distance
    let life = f.encode(2);
    assert!((reward(&life, 0) - 0.5).abs() < 1e-6, "approach gain {}", reward(&life, 0));
    assert_eq!(reward(&life, 1), 0.0, "no task, no progress: {}", reward(&life, 1));
    f.values[127] = 50.0; // parked: no rise
    let life = f.encode(2);
    assert_eq!(reward(&life, 0), 0.0, "parking pays nothing: {}", reward(&life, 0));
    f.values[31] = 3.0; // pickup: key changes, 127 jumps to the carrying bonus
    f.values[127] = 100.0;
    let life = f.encode(2);
    assert_eq!(reward(&life, 0), 0.0, "a key change pays nothing: {}", reward(&life, 0));
    f.values[127] = 130.0; // toward the destination on the same key
    let life = f.encode(2);
    assert!((reward(&life, 0) - 0.3).abs() < 1e-6, "carrying gain {}", reward(&life, 0));
    f.values[31] = 0.0; // drop-off: no task; the fall of 127 is not a loss
    f.values[127] = 0.0;
    f.values[68] = 1.0;
    let life = f.encode(2);
    assert!((reward(&life, 0) - 1.0).abs() < 1e-6, "delivery {}", reward(&life, 0));
    // Symmetric mode ignores progress entirely.
    f.values[31] = 4.0;
    f.encode(1);
    f.values[127] = 80.0;
    let life = f.encode(1);
    assert_eq!(reward(&life, 0), 0.0, "symmetric ignores progress: {}", reward(&life, 0));
}

#[test]
#[ignore = "needs the real GPU"]
fn delivery_and_damage_terms_are_shared_by_both_modes() {
    for mode in [0u32, 1u32, 2u32] {
        let mut f = Fixture::new();
        f.encode(mode);
        f.values[68] = 1.0; // one delivery
        f.values[BODY + 8] = 0.75; // 25% integrity lost
        let life = f.encode(mode);
        assert!((reward(&life, 0) - 1.0).abs() < 1e-6, "mode {mode} delivery {}", reward(&life, 0));
        assert!((reward(&life, 1) - -0.5).abs() < 1e-6, "mode {mode} damage {}", reward(&life, 1));
    }
}
