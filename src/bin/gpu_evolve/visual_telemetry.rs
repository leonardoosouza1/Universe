//! Optional CSV sink for reduced retinal diagnostics and the commands actually applied.
//! Set UNIVERSE_RETINA_TELEMETRY to a new CSV path. Sampling is fixed at 64 ticks.
use crate::types::BeingState;
use std::io::{BufWriter, Write};
use universe::gpu::{physics::PhysicsStateGpu, retina_metrics::RetinaMetricsGpu};

pub struct VisualTelemetry {
    pub gpu: RetinaMetricsGpu,
    writer: BufWriter<std::fs::File>,
    tick: u64,
    episode: u64,
}

impl VisualTelemetry {
    pub fn from_env(device: &wgpu::Device, capacity: usize) -> Option<Self> {
        let path = std::env::var_os("UNIVERSE_RETINA_TELEMETRY")?;
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .expect("retinal telemetry must use a new writable file");
        let mut writer = BufWriter::new(file);
        writeln!(writer, "episode,tick,being,alive,coverage,asymmetry,left_pixels,right_pixels,visible_distance,body_distance,overlap_count,raw_steer,applied_steer,throttle,brake,speed,x,z,forward_x,forward_z")
            .expect("retinal telemetry header");
        Some(Self {
            gpu: RetinaMetricsGpu::new(device, capacity),
            writer,
            tick: 0,
            episode: 0,
        })
    }

    pub fn sample_tick(&mut self) -> Option<u64> {
        let tick = self.tick;
        self.tick += 1;
        (tick % 64 == 0).then_some(tick)
    }

    pub fn reset(&mut self) {
        self.episode += 1;
        self.tick = 0;
    }

    pub fn record(
        &mut self,
        device: &wgpu::Device,
        tick: u64,
        alive: &[bool],
        states: &[BeingState],
        commands: &[PhysicsStateGpu],
        intents: &[[f32; 96]],
    ) {
        let metrics = self.gpu.read(device, states.len());
        for (i, m) in metrics.iter().enumerate() {
            let cmd = &commands[i];
            let speed = states[i].phys.vel[0].hypot(states[i].phys.vel[2]);
            writeln!(
                self.writer,
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                self.episode,
                tick,
                i,
                u8::from(alive[i]),
                m[0],
                m[1],
                m[2],
                m[3],
                m[4],
                m[5],
                m[6],
                intents[i][1],
                cmd.steer,
                cmd.throttle,
                cmd.brake_cmd,
                speed,
                states[i].phys.pos[0],
                states[i].phys.pos[2],
                states[i].phys.fwd[0],
                states[i].phys.fwd[2]
            )
            .expect("retinal telemetry row");
        }
        self.writer.flush().expect("retinal telemetry flush");
    }
}
