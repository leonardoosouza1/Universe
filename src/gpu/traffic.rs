//! Deterministic directional stop lines; no control override. Red crossing pays a cash-bounded fine of 3 currency units.
//! This declared city law is a currency sink, accounted in state14 (paid fines).
//! Call senses after taxi senses and before brain; post after physical integration,
//! with the SAME absolute tick. Phase convention: green first, yellow next, red second
//! half of cycle; opposite approaches share offset, cross traffic differs by half-cycle.
//! Observation schema organism-city-observation-v2 replaces 172/173 repaired wear,
//! 182 delivery count and duplicate alive191: presence, distance/100m, phase(-1/0/+1),
//! time until change/cycle. Raw body/taxi quantities remain unchanged and observable.
//! Presence=0 means no relevant light within 100m. Direction relevance uses heading
//! alignment >0.5 and lateral distance within stop-line halfwidth.
//! Checkpoint buf_state verbatim; static light metadata must enter source/config hash.
use wgpu::util::DeviceExt;
pub const TRAFFIC_STATE_BYTES: u64 = 64;
pub const TRAFFIC_OBSERVATION_ID: &str = "organism-city-observation-v2";
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TrafficLight {
    pub position: [f32; 2],
    pub approach: [f32; 2],
    pub half_width: f32,
    pub phase_offset_ticks: u32,
    pub green_ticks: u32,
    pub yellow_ticks: u32,
}
pub struct TrafficGpu {
    pub buf_state: wgpu::Buffer,
    pub capacity: usize,
    contact_snapshot: wgpu::Buffer,
    contact_nearest: wgpu::Buffer,
    pairs: wgpu::ComputePipeline,
    contacts: wgpu::ComputePipeline,
    lights: wgpu::Buffer,
    light_count: u32,
    senses: wgpu::ComputePipeline,
    post: wgpu::ComputePipeline,
}
impl TrafficGpu {
    pub fn new(
        device: &wgpu::Device,
        capacity: usize,
        lights: &[TrafficLight],
    ) -> Result<Self, &'static str> {
        if capacity == 0 || capacity > u32::MAX as usize || lights.is_empty() || lights.len() > 4096
        {
            return Err("invalid traffic capacity/count");
        }
        for l in lights {
            let norm = l.approach[0] * l.approach[0] + l.approach[1] * l.approach[1];
            if !l
                .position
                .iter()
                .chain(l.approach.iter())
                .all(|x| x.is_finite())
                || !l.half_width.is_finite()
                || l.half_width <= 0.0
                || (norm - 1.0).abs() > 0.0001
                || l.green_ticks == 0
                || l.yellow_ticks == 0
                || l.green_ticks as u64 + l.yellow_ticks as u64 > u32::MAX as u64 / 4
            {
                return Err("invalid traffic light");
            }
        }
        let lights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("traffic static stop lines"),
            contents: bytemuck::cast_slice(lights),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let buf_state = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("traffic acquired state"),
            size: capacity as u64 * TRAFFIC_STATE_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("directional traffic"),
            source: wgpu::ShaderSource::Wgsl(include_str!("traffic.wgsl").into()),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let contact_snapshot = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("traffic contact snapshot"),
            size: capacity as u64 * 224,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let contact_nearest = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("traffic contact pairs"),
            size: capacity as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let contacts_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("traffic contacts"),
            source: wgpu::ShaderSource::Wgsl(include_str!("traffic_contacts.wgsl").into()),
        });
        let contact_pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &contacts_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Ok(Self {
            buf_state,
            capacity,
            contact_snapshot,
            contact_nearest,
            pairs: contact_pipeline("pairs"),
            contacts: contact_pipeline("contacts"),
            lights: lights_buffer,
            light_count: lights.len() as u32,
            senses: pipeline("senses"),
            post: pipeline("post"),
        })
    }
    pub fn reset_slot(&self, queue: &wgpu::Queue, slot: usize) {
        assert!(slot < self.capacity);
        queue.write_buffer(
            &self.buf_state,
            slot as u64 * TRAFFIC_STATE_BYTES,
            &[0u8; 64],
        );
    }
    fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        physical: &wgpu::Buffer,
        wave: Option<&wgpu::Buffer>,
        body: Option<&wgpu::Buffer>,
        n: usize,
        tick: u32,
    ) -> Result<(), &'static str> {
        let stride = if wave.is_some() { 368 } else { 224 };
        if n > self.capacity
            || physical.size() < n as u64 * stride
            || wave.is_some_and(|w| w.size() < n as u64 * 192)
            || body.is_some_and(|b| b.size() < n as u64 * 512)
        {
            return Err("invalid traffic buffer shape");
        }
        if n == 0 {
            return Ok(());
        }
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("traffic tick"),
            contents: bytemuck::cast_slice(&[n as u32, self.light_count, tick, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let pipeline = if wave.is_some() {
            &self.senses
        } else {
            &self.post
        };
        let mut entries = vec![];
        for (binding, buffer) in [&params, &self.lights, &self.buf_state, physical]
            .iter()
            .enumerate()
        {
            entries.push(wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            });
        }
        if let Some(w) = wave {
            entries.push(wgpu::BindGroupEntry {
                binding: 4,
                resource: w.as_entire_binding(),
            });
        }
        if let Some(b) = body {
            entries.push(wgpu::BindGroupEntry {
                binding: 5,
                resource: b.as_entire_binding(),
            });
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("traffic"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("traffic"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        Ok(())
    }
    pub fn encode_senses(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        wave: &wgpu::Buffer,
        n: usize,
        tick: u32,
    ) -> Result<(), &'static str> {
        self.encode(device, encoder, input, Some(wave), None, n, tick)
    }
    pub fn encode_post(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::Buffer,
        body: &wgpu::Buffer,
        n: usize,
        tick: u32,
    ) -> Result<(), &'static str> {
        self.encode(device, encoder, output, None, Some(body), n, tick)
    }
    /// City-v2 live-body disc contacts. O(n²) pair search; no claim of 4000-body
    /// throughput. Radius1.5m is explicit model geometry, not the vehicle rectangle.
    /// Dead/unborn slots do not collide. Snapshot is scratch, not checkpoint state.
    /// Encode after A/B/C, before orientation, telemetry and body post.
    pub fn encode_contacts(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        body: &wgpu::Buffer,
        n: usize,
    ) -> Result<(), &'static str> {
        if n > self.capacity
            || input.size() < n as u64 * 368
            || output.size() < n as u64 * 224
            || body.size() < n as u64 * 512
        {
            return Err("invalid contact shape");
        }
        if n == 0 {
            return Ok(());
        }
        encoder.copy_buffer_to_buffer(output, 0, &self.contact_snapshot, 0, n as u64 * 224);
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("contact params"),
            contents: bytemuck::cast_slice(&[n as u32, 0, 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        for (pipeline, bindings) in [
            (
                &self.pairs,
                vec![
                    (0, &params),
                    (1, &self.contact_snapshot),
                    (3, body),
                    (5, &self.contact_nearest),
                ],
            ),
            (
                &self.contacts,
                vec![
                    (0, &params),
                    (1, &self.contact_snapshot),
                    (2, input),
                    (4, output),
                    (5, &self.contact_nearest),
                ],
            ),
        ] {
            let entries: Vec<_> = bindings
                .into_iter()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("traffic contacts"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("traffic contacts"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
        }
        Ok(())
    }
    /// Sampled observer readback only; no ordinary-tick map. Rows: startXZ,initialized,
    /// nearest_light_id(1-based;0 absent),distance_m,phase,time_left/cycle,reserved,
    /// red_count,green_count,yellow_count,total,last_light_id,last_tick,paid_fines,reserved.
    /// Export conversion to f32 loses integer precision above 2^24; raw checkpoint retains u32.
    pub fn download(&self, device: &wgpu::Device, queue: &wgpu::Queue, n: usize) -> Vec<[f32; 16]> {
        assert!(n <= self.capacity);
        if n == 0 {
            return vec![];
        }
        let bytes = n as u64 * TRAFFIC_STATE_BYTES;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("traffic observer"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.buf_state, 0, &staging, 0, bytes);
        queue.submit(Some(encoder.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("traffic map callback")
            .expect("traffic map");
        let mapped = staging.slice(..).get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&mapped);
        let rows = words
            .chunks_exact(16)
            .map(|row| {
                std::array::from_fn(|i| {
                    if i < 2 || (4..8).contains(&i) || i == 14 {
                        f32::from_bits(row[i])
                    } else {
                        row[i] as f32
                    }
                })
            })
            .collect();
        drop(mapped);
        staging.unmap();
        rows
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn traffic_shader_validates() {
        for source in [
            include_str!("traffic.wgsl"),
            include_str!("traffic_contacts.wgsl"),
        ] {
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            assert_eq!(std::mem::size_of::<TrafficLight>(), 32);
        }
    }
}

#[cfg(test)]
#[path = "traffic_tests.rs"]
mod gpu_tests;
