use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

const OBS_SCHEMA: &str = "vila-survival-observation/v3";
const DECISION_SCHEMA: &str = "vila-motor-decision/v1";
const MAX_OBSERVATION_AGE: u64 = 20;
const COMMAND_TTL: u64 = 40;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    schema: String,
    observation: Observation,
    candidates: Vec<Candidate>,
    candidate_count: usize,
    safe_count: usize,
    duration_ms: f64,
}

#[derive(Deserialize)]
struct Observation {
    episode: String,
    villager_uuid: String,
    v: usize,
    tick: u64,
    health: f64,
    on_ground: bool,
    sleeping: bool,
    aware: bool,
    exclusive_lease: bool,
    stationary_ticks: u64,
}

#[derive(Clone, Deserialize)]
struct Candidate {
    id: String,
    vx: f64,
    vz: f64,
    safe: bool,
    reject_reason: Option<String>,
}

#[derive(Deserialize)]
struct Episode {
    id: String,
    ai_pilot_enabled: bool,
}

#[derive(Deserialize)]
struct Live {
    tick: u64,
    villagers: Vec<LiveVillager>,
}

#[derive(Deserialize)]
struct LiveVillager {
    v: usize,
    uuid: String,
    alive: u8,
    sleeping: u8,
    on_ground: u8,
}

#[derive(Serialize)]
struct Decision<'a> {
    schema: &'static str,
    episode: &'a str,
    villager_uuid: &'a str,
    observation_v: usize,
    observation_tick: u64,
    live_tick: u64,
    seq: u64,
    candidate_id: &'a str,
    policy: &'static str,
    stationary_ticks: u64,
}

#[derive(Serialize)]
struct Command<'a> {
    seq: u64,
    episode: &'a str,
    villager_uuid: &'a str,
    issued_tick: u64,
    expires_tick: u64,
    action: &'static str,
    vx: f64,
    vz: f64,
    source_schema: &'static str,
    source_v: usize,
    source_tick: u64,
}

fn cardinal_velocity(candidate: &Candidate) -> bool {
    let expected = match candidate.id.as_str() {
        "north" => (0.0, -0.2),
        "south" => (0.0, 0.2),
        "east" => (0.2, 0.0),
        "west" => (-0.2, 0.0),
        _ => return false,
    };
    candidate.vx.is_finite()
        && candidate.vz.is_finite()
        && (candidate.vx - expected.0).abs() < 1e-9
        && (candidate.vz - expected.1).abs() < 1e-9
}

fn choose<'a>(
    snapshot: &'a Snapshot,
    live: &Live,
    episode: &Episode,
) -> Result<&'a Candidate, String> {
    let o = &snapshot.observation;
    if snapshot.schema != OBS_SCHEMA
        || snapshot.duration_ms < 0.0
        || !snapshot.duration_ms.is_finite()
        || o.episode != episode.id
        || !episode.ai_pilot_enabled
        || !o.exclusive_lease
        || o.aware
        || !o.on_ground
        || o.sleeping
        || !o.health.is_finite()
        || o.health <= 0.0
    {
        return Err("observation_or_ownership_invalid".into());
    }
    if o.v > 100_000
        || o.villager_uuid.len() != 36
        || o.episode.is_empty()
        || live.tick < o.tick
        || live.tick - o.tick > MAX_OBSERVATION_AGE
    {
        return Err("identity_or_observation_stale".into());
    }
    let actor = live
        .villagers
        .iter()
        .find(|v| v.uuid == o.villager_uuid && v.v == o.v)
        .ok_or("live_villager_missing")?;
    if actor.alive != 1 || actor.sleeping != 0 || actor.on_ground != 1 {
        return Err("live_villager_unavailable".into());
    }
    if snapshot.candidate_count != snapshot.candidates.len()
        || snapshot.candidate_count != 4
        || snapshot.safe_count != snapshot.candidates.iter().filter(|c| c.safe).count()
    {
        return Err("candidate_count_invalid".into());
    }
    let mut ids = std::collections::BTreeSet::new();
    for c in &snapshot.candidates {
        if !cardinal_velocity(c)
            || !ids.insert(c.id.as_str())
            || (c.safe && c.reject_reason.is_some())
            || (!c.safe && c.reject_reason.is_none())
        {
            return Err("candidate_contract_invalid".into());
        }
    }
    // Deliberately a fixed baseline until a task-level reward is measured.
    // This code proves policy ownership and command transport, not learning.
    snapshot
        .candidates
        .iter()
        .find(|c| c.id == "north" && c.safe)
        .or_else(|| snapshot.candidates.iter().find(|c| c.safe))
        .ok_or_else(|| "no_safe_candidate".into())
}

fn atomic_json<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let parent = path.parent().ok_or("output_without_parent")?;
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, serde_json::to_vec(value).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    fs::rename(tmp, path).map_err(|e| e.to_string())
}

fn run(args: &[String]) -> Result<(), String> {
    if args.len() != 7 || args[0] != "--decide-once" {
        return Err("usage: --decide-once OBS LIVE EPISODE DECISION PILOT SEQ".into());
    }
    let snapshot: Snapshot =
        serde_json::from_slice(&fs::read(&args[1]).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let live: Live = serde_json::from_slice(&fs::read(&args[2]).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let episode: Episode = serde_json::from_slice(&fs::read(&args[3]).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let seq: u64 = args[6].parse().map_err(|_| "invalid_seq")?;
    if seq == 0 || live.tick.checked_add(COMMAND_TTL).is_none() {
        return Err("invalid_seq_or_time".into());
    }
    let candidate = choose(&snapshot, &live, &episode)?;
    let o = &snapshot.observation;
    let decision = Decision {
        schema: DECISION_SCHEMA,
        episode: &o.episode,
        villager_uuid: &o.villager_uuid,
        observation_v: o.v,
        observation_tick: o.tick,
        live_tick: live.tick,
        seq,
        candidate_id: &candidate.id,
        policy: "fixed-safe-baseline",
        stationary_ticks: o.stationary_ticks,
    };
    let command = Command {
        seq,
        episode: &o.episode,
        villager_uuid: &o.villager_uuid,
        issued_tick: live.tick,
        expires_tick: live.tick + COMMAND_TTL,
        action: "nudge",
        vx: candidate.vx,
        vz: candidate.vz,
        source_schema: OBS_SCHEMA,
        source_v: o.v,
        source_tick: o.tick,
    };
    let pilot_path = Path::new(&args[5]);
    if pilot_path.exists() {
        let prior: serde_json::Value =
            serde_json::from_slice(&fs::read(pilot_path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        if prior
            .get("seq")
            .and_then(|v| v.as_u64())
            .unwrap_or(u64::MAX)
            >= seq
        {
            return Err("sequence_replay".into());
        }
    }
    atomic_json(Path::new(&args[4]), &decision)?;
    atomic_json(pilot_path, &command)?;
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(error) = run(&args) {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_cardinal_units_only() {
        let good = Candidate {
            id: "north".into(),
            vx: 0.0,
            vz: -0.2,
            safe: true,
            reject_reason: None,
        };
        assert!(cardinal_velocity(&good));
        assert!(!cardinal_velocity(&Candidate { vz: -0.4, ..good }));
    }

    #[test]
    fn rejects_stale_and_nonexclusive_snapshots() {
        let raw = r#"{"schema":"vila-survival-observation/v3","observation":{"episode":"e","villager_uuid":"12345678-1234-1234-1234-123456789012","v":10,"tick":100,"health":20.0,"on_ground":true,"sleeping":false,"aware":false,"exclusive_lease":true,"stationary_ticks":20},"candidates":[{"id":"north","vx":0.0,"vz":-0.2,"safe":true,"reject_reason":null},{"id":"south","vx":0.0,"vz":0.2,"safe":false,"reject_reason":"blocked"},{"id":"east","vx":0.2,"vz":0.0,"safe":false,"reject_reason":"blocked"},{"id":"west","vx":-0.2,"vz":0.0,"safe":false,"reject_reason":"blocked"}],"candidate_count":4,"safe_count":1,"duration_ms":1.0}"#;
        let mut snapshot: Snapshot = serde_json::from_str(raw).unwrap();
        let episode = Episode {
            id: "e".into(),
            ai_pilot_enabled: true,
        };
        let mut live = Live {
            tick: 100,
            villagers: vec![LiveVillager {
                v: 10,
                uuid: "12345678-1234-1234-1234-123456789012".into(),
                alive: 1,
                sleeping: 0,
                on_ground: 1,
            }],
        };
        assert_eq!(choose(&snapshot, &live, &episode).unwrap().id, "north");
        live.tick = 121;
        assert!(choose(&snapshot, &live, &episode).is_err());
        live.tick = 100;
        snapshot.observation.exclusive_lease = false;
        assert!(choose(&snapshot, &live, &episode).is_err());
    }
}
