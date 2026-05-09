use std::fs::{self, File};
use std::io::{BufWriter, Write};
use serde_json::json;
use crate::config::iara::brain::SENSORY_WAVE_SIZE;

/// Fixed-size replay frame. #[repr(C)] for zero-copy binary write.
/// Size: 272 bytes (fits in 256 + 16 padding, numpy will handle)
#[repr(C)]
pub struct ReplayFrame {
    pub frame_tick: u32,
    pub generation: u16,    // renamed from 'gen' (reserved keyword)
    pub rank: u8,
    pub flags: u8,          // bit0=is_dead
    pub pos_x: f32,
    pub pos_z: f32,
    pub heading: f32,       // yaw from quaternion
    pub speed_ms: f32,
    pub throttle: f32,
    pub steer: f32,
    pub brake: f32,
    pub hp: f32,
    pub distance_m: f32,
    pub track_progress: f32,
    pub total_laps: u16,
    pub racer_id: u16,
    pub joy_tick: f32,
    pub sensory_wave: [i8; SENSORY_WAVE_SIZE],
    pub _reserved: [u8; 8],
}

pub struct TelemetryWriter {
    pub run_id: String,
    gen_writer: Option<BufWriter<File>>,
    replay_writer: Option<BufWriter<File>>,
    debug_writer: Option<BufWriter<File>>,
}

impl TelemetryWriter {
    pub fn new(run_id: String, lab: &crate::lab::LabParams) -> Self {
        let dir = format!("telemetry/{}", run_id);
        let _ = fs::create_dir_all(&dir);

        // run_meta.json with feature flags + lab experiment metadata
        let features: Vec<&str> = vec![
            #[cfg(feature = "survival_pressure")]
            "survival_pressure",
            #[cfg(feature = "simple_elitism")]
            "simple_elitism",
            #[cfg(feature = "trinity_swarm")]
            "trinity_swarm",
        ];
        let meta = json!({
            "run_id": run_id.clone(),
            "features": features,
            "experiment_name": lab.experiment_name,
            "variant_id": lab.variant_id
        });
        let _ = std::fs::write(format!("{}/run_meta.json", dir), meta.to_string());

        // generations.ndjson
        let gen_file = File::options()
            .create(true)
            .append(true)
            .open(format!("{}/generations.ndjson", dir))
            .ok();
        let gen_writer = gen_file.map(|f| BufWriter::new(f));

        // replay.bin
        let replay_file = File::options()
            .create(true)
            .append(true)
            .open(format!("{}/replay.bin", dir))
            .ok();
        let replay_writer = replay_file.map(|f| BufWriter::with_capacity(256 * 256, f));

        // debug.ndjson (spawn/early behavior diagnostics)
        let debug_file = File::options()
            .create(true)
            .append(true)
            .open(format!("{}/debug.ndjson", dir))
            .ok();
        let debug_writer = debug_file.map(|f| BufWriter::new(f));

        Self {
            run_id,
            gen_writer,
            replay_writer,
            debug_writer,
        }
    }

    pub fn write_gen(&mut self, data: serde_json::Value) {
        if let Some(w) = &mut self.gen_writer {
            let _ = writeln!(w, "{}", data);
        }
    }

    pub fn write_frame(&mut self, frame: &ReplayFrame) {
        if let Some(w) = &mut self.replay_writer {
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    frame as *const ReplayFrame as *const u8,
                    std::mem::size_of::<ReplayFrame>(),
                )
            };
            let _ = w.write_all(bytes);
        }
    }

    pub fn write_debug(&mut self, data: serde_json::Value) {
        if let Some(w) = &mut self.debug_writer {
            let _ = writeln!(w, "{}", data);
        }
    }
}

pub fn gen_run_id() -> String {
    let unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{}", unix)
}

pub fn name_to_id(name: &str) -> u16 {
    let mut h: u32 = 2166136261;
    for b in name.bytes() {
        h = h.wrapping_mul(16777619) ^ b as u32;
    }
    h as u16
}

/// Save racer's brain trace to JSON for frame-by-frame vision debugging.
pub fn save_brain_trace(run_id: &str, racer_name: &str, trace: &[crate::simulation::racer::BrainSnapshot]) {
    if trace.is_empty() { return; }
    let mut frames = Vec::new();
    for (i, snap) in trace.iter().enumerate() {
        frames.push(json!({
            "frame": i,
            "t": snap.t,
            "speed_kmh": snap.speed_kmh,
            "wall_prox_l": snap.wall_prox_l,
            "wall_prox_r": snap.wall_prox_r,
            "ttc_inv": snap.ttc_inv,
            "curv_25": snap.curv_25,
            "raw_throttle": snap.raw_throttle,
            "raw_steer": snap.raw_steer,
            "raw_brake": snap.raw_brake,
            "physical_steer": snap.physical_steer,
            "steer_limit": snap.steer_limit,
            "risk_damp": snap.risk_damp,
            "top_channels": snap.top_channels.iter().map(|(idx, val)| json!({"ch": idx, "val": val})).collect::<Vec<_>>(),
            "retina_fov": snap.retina_fov,
            "sensory_compact": snap.sensory_compact.to_vec()
        }));
    }
    let output = serde_json::to_string_pretty(&frames).unwrap_or_default();
    // Save to debug/ subdir for easy access; telemetry uses timestamp, we use run_id (gen_N)
    let _ = std::fs::create_dir_all("debug");
    let _ = std::fs::write(format!("debug/brain_trace_gen_{}.json", run_id.replace("gen_", "")), output);
}
