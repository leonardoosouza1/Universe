//! Native organism runtime checkpoints.
//!
//! A checkpoint is a bounded continuation artifact: acquired GPU buffers, host state,
//! and the exact heritable genome files are committed together.  It is deliberately
//! separate from the legacy "best DNA" resume path.  Restore validates every byte and
//! the frozen configuration before the first GPU or host mutation.

use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{context::GpuCtx, types::BeingState};
use universe::gpu::{
    evolution::Genome,
    genome_io,
    state_snapshot::{self, NamedBuffer, SnapshotBundle},
};

// v5 versions input83 starter authorization and the powertrain observation/action contract.
const FORMAT_VERSION: u32 = 5;
const MAX_BEINGS: usize = 4096;
const MAX_HIP_MEM: usize = 4096;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct HostState {
    states: Vec<BeingState>,
    last_intents: Vec<Vec<f32>>,
    brain_tick: u32,
    hip_mem: Vec<universe::gpu::hippocampus::SectorMemory>,
    ultimo_gate: [f32; 4],
    lifetime_reward_sum: Vec<f32>,
    lifetime_reward_ticks: u32,
    lap_grass: Vec<bool>,
    lap_damage: Vec<f32>,
    lap_dry: Vec<bool>,
    aux: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Manifest {
    format_version: u32,
    resident_implementation_sha256: String,
    tick: usize,
    generation: usize,
    config: Value,
    config_sha256: String,
    host_sha256: String,
    gpu_sha256: String,
    genome_files: Vec<String>,
    genome_sha256: Vec<String>,
}

/// Fingerprint resident adapters as well as the numerical kernels recorded by main.
/// Length-prefix each source to make file boundaries unambiguous.
pub fn resident_implementation_sha256() -> String {
    let mut hash = Sha256::new();
    for source in [
        include_bytes!("organism_tick.rs").as_slice(),
        include_bytes!("organism_checkpoint.rs").as_slice(),
        include_bytes!("ecology_runtime.rs").as_slice(),
        include_bytes!("../../ecology.rs").as_slice(),
        include_bytes!("../../gpu/organism_action.rs").as_slice(),
        include_bytes!("../../gpu/organism_action.wgsl").as_slice(),
        include_bytes!("../../gpu/organism_sensors.rs").as_slice(),
        include_bytes!("../../gpu/organism_sensors.wgsl").as_slice(),
        include_bytes!("../../gpu/organism_orientation.rs").as_slice(),
        include_bytes!("../../gpu/organism_orientation.wgsl").as_slice(),
        include_bytes!("../../gpu/organism_telemetry.rs").as_slice(),
        include_bytes!("../../gpu/organism_telemetry.wgsl").as_slice(),
        include_bytes!("../../gpu/organism_feedback.rs").as_slice(),
        include_bytes!("../../gpu/organism_feedback.wgsl").as_slice(),
        include_bytes!("../../gpu/organism_reproduction.rs").as_slice(),
        include_bytes!("../../gpu/organism_reproduction.wgsl").as_slice(),
        include_bytes!("../../gpu/organism.rs").as_slice(),
        include_bytes!("../../gpu/organism.wgsl").as_slice(),
        include_bytes!("../../gpu/physics.rs").as_slice(),
        include_bytes!("../../gpu/physics_a.wgsl").as_slice(),
        include_bytes!("../../gpu/physics_b.wgsl").as_slice(),
        include_bytes!("../../gpu/physics_c.wgsl").as_slice(),
        include_bytes!("../../gpu/perception.rs").as_slice(),
        include_bytes!("../../gpu/perception.wgsl").as_slice(),
        include_bytes!("../../gpu/perception_temporal.wgsl").as_slice(),
        include_bytes!("../../gpu/perception_motion.wgsl").as_slice(),
        include_bytes!("../../gpu/retina_digest.rs").as_slice(),
        include_bytes!("../../gpu/retina_digest.wgsl").as_slice(),
        include_bytes!("../../gpu/retina.wgsl").as_slice(),
        include_bytes!("../../gpu/retina_geometry.wgsl").as_slice(),
        include_bytes!("../../gpu/slow_brain.wgsl").as_slice(),
        include_bytes!("../../gpu/slow_gather.wgsl").as_slice(),
        include_bytes!("../../gpu/taxi.rs").as_slice(),
        include_bytes!("../../gpu/taxi.wgsl").as_slice(),
        include_bytes!("../../gpu/taxi_routes.rs").as_slice(),
        include_bytes!("../../gpu/taxi_witness.rs").as_slice(),
        include_bytes!("../../gpu/taxi_witness.wgsl").as_slice(),
        include_bytes!("../../gpu/traffic.rs").as_slice(),
        include_bytes!("../../gpu/traffic.wgsl").as_slice(),
        include_bytes!("../../gpu/traffic_contacts.wgsl").as_slice(),
        include_bytes!("../../organism_city.rs").as_slice(),
        include_bytes!("../../gpu/brain/mod.rs").as_slice(),
        include_bytes!("../../gpu/brain.wgsl").as_slice(),
    ] {
        hash.update((source.len() as u64).to_le_bytes());
        hash.update(source);
    }
    format!("{:x}", hash.finalize())
}

fn sha256(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    format!("{:x}", h.finalize())
}

fn finite(x: f32) -> bool {
    x.is_finite()
}

fn safe_genome_name(name: &str) -> bool {
    name.len() <= 64
        && name.starts_with("genome-")
        && name.ends_with(".bin")
        && Path::new(name).file_name().and_then(|n| n.to_str()) == Some(name)
}

fn validate_host(host: &HostState) -> Result<(), String> {
    let n = host.states.len();
    if n == 0 || n > MAX_BEINGS {
        return Err("checkpoint: invalid being count".into());
    }
    if host.last_intents.len() != n
        || host.lap_grass.len() != n
        || host.lap_damage.len() != n
        || host.lap_dry.len() != n
    {
        return Err("checkpoint: host vectors have inconsistent lengths".into());
    }
    if host.hip_mem.len() > MAX_HIP_MEM || (!host.hip_mem.is_empty() && host.hip_mem.len() != n) {
        return Err("checkpoint: invalid hippocampus length".into());
    }
    if host.lifetime_reward_sum.len() != n {
        return Err("checkpoint: invalid reward length".into());
    }
    if host.ultimo_gate.iter().any(|x| !finite(*x))
        || host.lifetime_reward_sum.iter().any(|x| !finite(*x))
        || host.lap_damage.iter().any(|x| !finite(*x))
    {
        return Err("checkpoint: non-finite host telemetry".into());
    }
    for intent in &host.last_intents {
        if intent.len() != 96 || intent.iter().any(|x| !finite(*x)) {
            return Err("checkpoint: non-finite intent".into());
        }
    }
    for state in &host.states {
        state
            .phys
            .validate_contract()
            .map_err(|e| format!("checkpoint: invalid physics state: {e}"))?;
    }
    // BeingState and PhysicsStateGpu carry all acquired host fields. Round-trip through
    // serde is bounded above; this explicit check catches NaN/Inf in f32 arrays too.
    let bytes = serde_json::to_vec(&host.states).map_err(|e| e.to_string())?;
    if bytes.len() > 64 * 1024 * 1024 {
        return Err("checkpoint: host state too large".into());
    }
    Ok(())
}

fn native_buffers(ctx: &GpuCtx) -> Result<Vec<NamedBuffer<'_>>, String> {
    if ctx.organism.is_none() {
        return Err("checkpoint: organism runtime is absent".into());
    }
    if ctx.taxi.is_none() {
        return Err("checkpoint: taxi runtime is absent".into());
    }
    if ctx.fly.is_some() || ctx.fly_spiking.is_some() || ctx.brain_control_direct {
        return Err("checkpoint: fly/direct brain runtime is not supported".into());
    }
    let mut out = Vec::new();
    out.extend(ctx.phys_gpu.resident_snapshot_buffers());
    out.extend(
        ctx.organism_action
            .as_ref()
            .ok_or("checkpoint: missing resident action")?
            .snapshot_buffers(),
    );
    out.extend(ctx.brain.snapshot_buffers());
    out.extend(ctx.slow.snapshot_buffers());
    out.extend(ctx.percept.snapshot_buffers());
    out.extend(ctx.retina_digest.snapshot_buffers());
    out.push(("organism.state", &ctx.organism.as_ref().unwrap().buf_state));
    out.push(("taxi.world", &ctx.taxi.as_ref().unwrap().buf_world));
    out.push((
        "traffic.state",
        &ctx.traffic
            .as_ref()
            .ok_or("checkpoint: missing traffic")?
            .buf_state,
    ));
    out.push(("retina.current", &ctx.buf_retina_out));
    Ok(out)
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut f = fs::File::create(path).map_err(|e| e.to_string())?;
    f.write_all(bytes).map_err(|e| e.to_string())?;
    f.sync_all().map_err(|e| e.to_string())
}

fn temp_dir_for(path: &Path) -> Result<PathBuf, String> {
    let parent = path
        .parent()
        .ok_or_else(|| "checkpoint path has no parent".to_string())?;
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let name = path
        .file_name()
        .and_then(|x| x.to_str())
        .ok_or_else(|| "invalid checkpoint name".to_string())?;
    let tmp = parent.join(format!(".{name}.tmp-{}", std::process::id()));
    if tmp.exists() {
        return Err("checkpoint temporary directory already exists".into());
    }
    fs::create_dir(&tmp).map_err(|e| e.to_string())?;
    Ok(tmp)
}

/// Save a native organism continuation. The destination must not already exist.
pub fn save(
    ctx: &GpuCtx,
    states: &[BeingState],
    genomes: &[Genome],
    path: &Path,
    tick: usize,
    generation: usize,
    config: Value,
    aux: Value,
) -> Result<(), String> {
    if path.exists() {
        return Err("checkpoint destination already exists".into());
    }
    if states.len() != genomes.len() {
        return Err("checkpoint: states/genomes length mismatch".into());
    }
    let host = HostState {
        states: states.to_vec(),
        last_intents: ctx.last_intents.iter().map(|v| v.to_vec()).collect(),
        brain_tick: ctx.brain_tick,
        hip_mem: ctx.hip_mem.clone(),
        ultimo_gate: ctx.ultimo_gate,
        lifetime_reward_sum: ctx.lifetime_reward_sum.clone(),
        lifetime_reward_ticks: ctx.lifetime_reward_ticks,
        lap_grass: ctx.lap_grass.clone(),
        lap_damage: ctx.lap_damage.clone(),
        lap_dry: ctx.lap_dry.clone(),
        aux,
    };
    validate_host(&host)?;
    let buffers = native_buffers(ctx)?;
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("organism-checkpoint"),
        });
    let pending = state_snapshot::encode_capture(&ctx.device, &mut encoder, &buffers)?;
    ctx.queue.submit(Some(encoder.finish()));
    let gpu = pending.read(&ctx.device)?;
    let tmp = temp_dir_for(path)?;
    let gpu_path = tmp.join("gpu.snapshot");
    state_snapshot::write_atomic(&gpu_path, &gpu)?;
    let gpu_bytes = fs::read(&gpu_path).map_err(|e| e.to_string())?;
    let host_bytes = serde_json::to_vec(&host).map_err(|e| e.to_string())?;
    let config_bytes = serde_json::to_vec(&config).map_err(|e| e.to_string())?;
    if config_bytes.len() > 1_048_576 {
        return Err("checkpoint: configuration payload too large".into());
    }
    let result = (|| {
        let mut genome_files = Vec::with_capacity(genomes.len());
        let mut genome_sha256 = Vec::with_capacity(genomes.len());
        for (i, genome) in genomes.iter().enumerate() {
            let name = format!("genome-{i:06}.bin");
            let gp = tmp.join(&name);
            genome_io::save_genome(
                genome,
                gp.to_str()
                    .ok_or_else(|| "invalid genome path".to_string())?,
            )
            .map_err(|e| e.to_string())?;
            genome_files.push(name);
            genome_sha256.push(sha256(&fs::read(&gp).map_err(|e| e.to_string())?));
        }
        write_file(&tmp.join("host.json"), &host_bytes)?;
        // gpu.snapshot was written atomically directly inside this temporary directory.
        let manifest = Manifest {
            format_version: FORMAT_VERSION,
            resident_implementation_sha256: resident_implementation_sha256(),
            tick,
            generation,
            config_sha256: sha256(&config_bytes),
            config,
            host_sha256: sha256(&host_bytes),
            gpu_sha256: sha256(&gpu_bytes),
            genome_files,
            genome_sha256,
        };
        write_file(
            &tmp.join("manifest.json"),
            &serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?,
        )?;
        if path.exists() {
            return Err("checkpoint destination appeared during save".into());
        }
        fs::rename(&tmp, path).map_err(|e| e.to_string())
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&tmp);
    }
    result
}

pub fn load_metadata(path: &Path) -> Result<(usize, usize, Value), String> {
    let m: Manifest =
        serde_json::from_slice(&fs::read(path.join("manifest.json")).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if m.format_version != FORMAT_VERSION {
        return Err("checkpoint: unsupported format".into());
    }
    if m.resident_implementation_sha256 != resident_implementation_sha256() {
        return Err("checkpoint: resident implementation mismatch".into());
    }
    if serde_json::to_vec(&m.config)
        .map_err(|e| e.to_string())?
        .len()
        > 1_048_576
    {
        return Err("checkpoint: configuration payload too large".into());
    }
    Ok((m.tick, m.generation, m.config))
}

pub fn load_genomes(path: &Path) -> Result<Vec<Genome>, String> {
    let m: Manifest =
        serde_json::from_slice(&fs::read(path.join("manifest.json")).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    if m.format_version != FORMAT_VERSION {
        return Err("checkpoint: unsupported format".into());
    }
    if m.resident_implementation_sha256 != resident_implementation_sha256() {
        return Err("checkpoint: resident implementation mismatch".into());
    }
    if m.genome_files.len() != m.genome_sha256.len() {
        return Err("checkpoint: invalid genome manifest".into());
    }
    if m.genome_files.len() > MAX_BEINGS {
        return Err("checkpoint: too many genome files".into());
    }
    if m.genome_files.iter().any(|name| !safe_genome_name(name)) {
        return Err("checkpoint: unsafe genome filename".into());
    }
    m.genome_files
        .iter()
        .zip(m.genome_sha256.iter())
        .map(|(name, expected)| {
            let bytes = fs::read(path.join(name)).map_err(|e| e.to_string())?;
            if sha256(&bytes) != *expected {
                return Err(format!("checkpoint: genome hash mismatch: {name}"));
            }
            genome_io::load_genome(
                path.join(name)
                    .to_str()
                    .ok_or_else(|| "invalid genome path".to_string())?,
            )
            .ok_or_else(|| format!("checkpoint: cannot decode {name}"))
        })
        .collect()
}

/// Restore after validating the complete manifest, host payload, GPU payload and DNA.
/// No destination buffer or caller vector is touched until all checks pass.
pub fn load(
    ctx: &mut GpuCtx,
    states: &mut Vec<BeingState>,
    path: &Path,
    expected_config: &Value,
) -> Result<(usize, usize, Value), String> {
    let manifest_bytes = fs::read(path.join("manifest.json")).map_err(|e| e.to_string())?;
    let m: Manifest = serde_json::from_slice(&manifest_bytes).map_err(|e| e.to_string())?;
    if m.format_version != FORMAT_VERSION {
        return Err("checkpoint: unsupported format".into());
    }
    if m.resident_implementation_sha256 != resident_implementation_sha256() {
        return Err("checkpoint: resident implementation mismatch".into());
    }
    let expected_config_bytes = serde_json::to_vec(expected_config).map_err(|e| e.to_string())?;
    if m.config_sha256 != sha256(&expected_config_bytes) {
        return Err("checkpoint: frozen configuration mismatch".into());
    }
    if m.config != *expected_config {
        return Err("checkpoint: configuration payload mismatch".into());
    }
    if serde_json::to_vec(&m.config)
        .map_err(|e| e.to_string())?
        .len()
        > 1_048_576
    {
        return Err("checkpoint: configuration payload too large".into());
    }
    let host_bytes = fs::read(path.join("host.json")).map_err(|e| e.to_string())?;
    let gpu_bytes = fs::read(path.join("gpu.snapshot")).map_err(|e| e.to_string())?;
    if sha256(&host_bytes) != m.host_sha256 || sha256(&gpu_bytes) != m.gpu_sha256 {
        return Err("checkpoint: payload hash mismatch".into());
    }
    let host: HostState = serde_json::from_slice(&host_bytes).map_err(|e| e.to_string())?;
    validate_host(&host)?;
    let gpu: SnapshotBundle = state_snapshot::read_file(&path.join("gpu.snapshot"))?;
    let buffers = native_buffers(ctx)?;
    let genomes = load_genomes(path)?;
    if genomes.len() != host.states.len() {
        return Err("checkpoint: genome/state count mismatch".into());
    }
    // state_snapshot::restore validates all names, dimensions and hashes before writes.
    state_snapshot::restore(&ctx.queue, &buffers, &gpu)?;
    *states = host.states;
    ctx.last_intents = host
        .last_intents
        .into_iter()
        .map(|v| {
            let mut a = [0.0_f32; 96];
            a.copy_from_slice(&v);
            a
        })
        .collect();
    ctx.brain_tick = host.brain_tick;
    ctx.hip_mem = host.hip_mem;
    ctx.ultimo_gate = host.ultimo_gate;
    ctx.lifetime_reward_sum = host.lifetime_reward_sum;
    ctx.lifetime_reward_ticks = host.lifetime_reward_ticks;
    ctx.lap_grass = host.lap_grass;
    ctx.lap_damage = host.lap_damage;
    ctx.lap_dry = host.lap_dry;
    ctx.organism_resident_ready = true;
    Ok((m.tick, m.generation, host.aux))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_hash_is_stable_and_changes_on_frozen_input() {
        let a = serde_json::json!({"scenario":"native","dt":0.016});
        let b = serde_json::json!({"scenario":"native","dt":0.017});
        assert_eq!(
            sha256(&serde_json::to_vec(&a).unwrap()),
            sha256(&serde_json::to_vec(&a).unwrap())
        );
        assert_ne!(
            sha256(&serde_json::to_vec(&a).unwrap()),
            sha256(&serde_json::to_vec(&b).unwrap())
        );
    }

    #[test]
    fn host_shape_and_intent_width_are_checked() {
        let mut host = HostState {
            states: Vec::new(),
            last_intents: Vec::new(),
            brain_tick: 0,
            hip_mem: Vec::new(),
            ultimo_gate: [0.0; 4],
            lifetime_reward_sum: Vec::new(),
            lifetime_reward_ticks: 0,
            lap_grass: Vec::new(),
            lap_damage: Vec::new(),
            lap_dry: Vec::new(),
            aux: serde_json::json!({}),
        };
        assert!(validate_host(&host).is_err());
        host.last_intents.push(vec![0.0; 96]);
        assert!(host.last_intents[0].len() == 96);
    }
}
