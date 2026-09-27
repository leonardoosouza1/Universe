use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{self, Read, Write};
use std::path::Path;
use std::time::{Duration, SystemTime};

const SCHEMA: &str = "vila-goal-policy/v2";
const PAPER_SCHEMA: &str = "vila-bed-candidates/v3";
const MIN_STATIONARY_HOME_TICKS: u64 = 160;
const FEATURES: [&str; 5] = [
    "bias",
    "negative_distance",
    "negative_path_length",
    "negative_crowding",
    "night",
];

#[derive(Clone, Debug, Deserialize, Serialize)]
struct State {
    schema: String,
    features: Vec<String>,
    weights: Vec<f64>,
    updates: u64,
    candidate_stats: BTreeMap<String, CandidateStats>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CandidateStats {
    visits: u32,
    mean_reward: f64,
}

impl Default for State {
    fn default() -> Self {
        Self {
            schema: SCHEMA.into(),
            features: FEATURES.iter().map(|s| (*s).into()).collect(),
            weights: vec![0.0; FEATURES.len()],
            updates: 0,
            candidate_stats: BTreeMap::new(),
        }
    }
}

impl State {
    fn validate(&self) -> Result<(), String> {
        if self.schema != SCHEMA
            || self.features.iter().map(String::as_str).collect::<Vec<_>>() != FEATURES
        {
            return Err("policy schema or feature order mismatch".into());
        }
        if self.weights.len() != FEATURES.len()
            || !self
                .weights
                .iter()
                .all(|w| w.is_finite() && w.abs() <= 20.0)
        {
            return Err("invalid policy weights".into());
        }
        if self.candidate_stats.len() > 128
            || self.candidate_stats.iter().any(|(id, stats)| {
                id.is_empty()
                    || id.len() > 128
                    || stats.visits == 0
                    || stats.visits > 10_000
                    || !stats.mean_reward.is_finite()
                    || stats.mean_reward.abs() > 20.0
            })
        {
            return Err("invalid candidate statistics".into());
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct Observation {
    episode: String,
    villager_uuid: String,
    tick: u64,
    is_night: bool,
    has_home: bool,
    sleeping: bool,
    stationary_home_ticks: u64,
}

#[derive(Debug, Deserialize)]
struct Candidate {
    id: String,
    x: f64,
    y: f64,
    z: f64,
    distance: f64,
    path_points: u32,
    crowding: u32,
    path_complete: bool,
    bed_available: bool,
}

#[derive(Debug, Deserialize)]
struct HomeApproach {
    x: f64,
    y: f64,
    z: f64,
    distance: f64,
    path_points: u32,
}

#[derive(Debug, Deserialize)]
struct PaperSnapshot {
    schema: String,
    observation: Observation,
    candidates: Vec<Candidate>,
    candidate_count: usize,
    home_route_status: String,
    home_approach: Option<HomeApproach>,
}

#[derive(Debug, Deserialize)]
struct EpisodeHeader {
    id: String,
    ai_pilot_enabled: bool,
}

#[derive(Debug, Deserialize)]
struct LiveSnapshot {
    tick: u64,
    villagers: Vec<LiveVillager>,
}

#[derive(Debug, Deserialize)]
struct LiveVillager {
    uuid: String,
    alive: u8,
    sleeping: u8,
}

#[derive(Serialize)]
struct PilotCommand<'a> {
    seq: u64,
    episode: &'a str,
    villager_uuid: &'a str,
    issued_tick: u64,
    expires_tick: u64,
    action: &'a str,
    x: f64,
    y: f64,
    z: f64,
    speed: f64,
}

#[derive(Serialize)]
struct DispatchRecord<'a> {
    schema: &'static str,
    seq: u64,
    live_tick: u64,
    state: &'a State,
    decision: &'a Decision,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Decision {
    schema: String,
    goal: String,
    action: String,
    episode: String,
    villager_uuid: String,
    tick: u64,
    candidate_id: String,
    x: f64,
    y: f64,
    z: f64,
    features: Vec<f64>,
    expected_reward: f64,
    policy_updates: u64,
}

#[derive(Debug, Deserialize)]
struct Outcome {
    episode: String,
    villager_uuid: String,
    decision_tick: u64,
    action_tick: u64,
    end_tick: u64,
    candidate_id: String,
    arrived: bool,
    home_acquired: bool,
    slept: bool,
    progress_ratio: f64,
    health_lost: f64,
    rescued: bool,
}

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Request {
    Init,
    Decide {
        state: State,
        observation: Observation,
        candidates: Vec<Candidate>,
    },
    DecidePaper {
        state: State,
        snapshot: PaperSnapshot,
    },
    Learn {
        state: State,
        decision: Decision,
        outcome: Outcome,
    },
}

#[derive(Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Response {
    Init { state: State },
    Decide { decision: Option<Decision> },
    Learn { state: State, reward: f64 },
}

fn features(observation: &Observation, candidate: &Candidate) -> Option<Vec<f64>> {
    if !candidate.path_complete
        || !candidate.bed_available
        || candidate.id.is_empty()
        || ![candidate.x, candidate.y, candidate.z, candidate.distance]
            .iter()
            .all(|v| v.is_finite())
        || !(0.0..=16.0).contains(&candidate.distance)
        || candidate.path_points == 0
        || candidate.path_points > 64
        || candidate.crowding > 20
    {
        return None;
    }
    Some(vec![
        1.0,
        -candidate.distance / 16.0,
        -(candidate.path_points as f64) / 64.0,
        -(candidate.crowding as f64) / 20.0,
        f64::from(observation.is_night),
    ])
}

fn decide(
    state: &State,
    observation: Observation,
    candidates: Vec<Candidate>,
) -> Result<Option<Decision>, String> {
    state.validate()?;
    if observation.episode.is_empty()
        || observation.villager_uuid.is_empty()
        || candidates.len() > 128
    {
        return Err("invalid observation or too many candidates".into());
    }
    if observation.has_home || !observation.is_night || observation.sleeping {
        return Ok(None);
    }
    let mut best: Option<(Candidate, Vec<f64>, f64)> = None;
    for candidate in candidates {
        let Some(input) = features(&observation, &candidate) else {
            continue;
        };
        let learned_bed_value = state
            .candidate_stats
            .get(&candidate.id)
            .map_or(0.0, |s| s.mean_reward);
        let score: f64 = state
            .weights
            .iter()
            .zip(&input)
            .map(|(w, x)| w * x)
            .sum::<f64>()
            + learned_bed_value;
        let replace = match &best {
            None => true,
            Some((previous, _, previous_score)) => {
                score > *previous_score + 1e-9
                    || ((score - *previous_score).abs() <= 1e-9
                        && (candidate.distance < previous.distance
                            || (candidate.distance == previous.distance
                                && candidate.id < previous.id)))
            }
        };
        if replace {
            best = Some((candidate, input, score));
        }
    }
    Ok(best.map(|(candidate, input, score)| Decision {
        schema: SCHEMA.into(),
        goal: "claim_bed".into(),
        action: "walk_to".into(),
        episode: observation.episode,
        villager_uuid: observation.villager_uuid,
        tick: observation.tick,
        candidate_id: candidate.id,
        x: candidate.x,
        y: candidate.y,
        z: candidate.z,
        features: input,
        expected_reward: score,
        policy_updates: state.updates,
    }))
}

fn decide_paper(state: &State, snapshot: PaperSnapshot) -> Result<Option<Decision>, String> {
    if snapshot.schema != PAPER_SCHEMA {
        return Err("paper observation schema mismatch".into());
    }
    if snapshot.candidate_count != snapshot.candidates.len()
        || snapshot.candidate_count > 128
        || (snapshot.home_route_status != "reachable" && snapshot.home_approach.is_some())
    {
        return Err("inconsistent paper candidate envelope".into());
    }
    state.validate()?;
    let observation = snapshot.observation;
    if observation.episode.is_empty() || observation.villager_uuid.is_empty() {
        return Err("invalid paper observation identity".into());
    }
    if !observation.is_night || observation.sleeping {
        return Ok(None);
    }
    if !observation.has_home {
        return decide(state, observation, snapshot.candidates);
    }
    if observation.stationary_home_ticks < MIN_STATIONARY_HOME_TICKS {
        return Ok(None);
    }
    if snapshot.home_route_status != "reachable" {
        return Ok(None);
    }
    let Some(home) = snapshot.home_approach else {
        return Err("reachable HOME missing approach".into());
    };
    let candidate = Candidate {
        id: "home_route".into(),
        x: home.x,
        y: home.y,
        z: home.z,
        distance: home.distance,
        path_points: home.path_points,
        crowding: 0,
        path_complete: true,
        bed_available: true,
    };
    let Some(input) = features(&observation, &candidate) else {
        return Err("invalid HOME approach".into());
    };
    let score: f64 = state
        .weights
        .iter()
        .zip(&input)
        .map(|(w, x)| w * x)
        .sum::<f64>()
        + state
            .candidate_stats
            .get("home_route")
            .map_or(0.0, |s| s.mean_reward);
    Ok(Some(Decision {
        schema: SCHEMA.into(),
        goal: "return_home".into(),
        action: "walk_to".into(),
        episode: observation.episode,
        villager_uuid: observation.villager_uuid,
        tick: observation.tick,
        candidate_id: candidate.id,
        x: candidate.x,
        y: candidate.y,
        z: candidate.z,
        features: input,
        expected_reward: score,
        policy_updates: state.updates,
    }))
}

fn learn(mut state: State, decision: Decision, outcome: Outcome) -> Result<(State, f64), String> {
    state.validate()?;
    if decision.schema != SCHEMA
        || decision.action != "walk_to"
        || (decision.goal != "claim_bed" && decision.goal != "return_home")
        || (decision.candidate_id == "home_route") != (decision.goal == "return_home")
        || ![decision.x, decision.y, decision.z]
            .iter()
            .all(|v| v.is_finite())
        || decision.features.len() != FEATURES.len()
        || !decision
            .features
            .iter()
            .all(|x| x.is_finite() && x.abs() <= 1.0)
        || !decision.expected_reward.is_finite()
        || decision.policy_updates != state.updates
        || decision.episode != outcome.episode
        || decision.villager_uuid != outcome.villager_uuid
        || decision.tick != outcome.decision_tick
        || outcome.action_tick < outcome.decision_tick
        || outcome.action_tick - outcome.decision_tick > 40
        || decision.candidate_id != outcome.candidate_id
        || outcome.end_tick <= outcome.action_tick
        || outcome.end_tick - outcome.action_tick > 400
        || !outcome.progress_ratio.is_finite()
        || !(-1.0..=1.0).contains(&outcome.progress_ratio)
        || !outcome.health_lost.is_finite()
        || !(0.0..=20.0).contains(&outcome.health_lost)
    {
        return Err("invalid or mismatched action outcome".into());
    }
    let reward = 5.0 * f64::from(outcome.home_acquired)
        + 5.0 * f64::from(outcome.slept)
        + f64::from(outcome.arrived)
        + 0.5 * outcome.progress_ratio
        - outcome.health_lost
        - 3.0 * f64::from(outcome.rescued)
        - 0.2 * (outcome.end_tick - outcome.action_tick) as f64 / 200.0;
    let prediction: f64 = state
        .weights
        .iter()
        .zip(&decision.features)
        .map(|(w, x)| w * x)
        .sum::<f64>()
        + state
            .candidate_stats
            .get(&decision.candidate_id)
            .map_or(0.0, |s| s.mean_reward);
    if (decision.expected_reward - prediction).abs() > 1e-8 {
        return Err("decision does not match current policy state".into());
    }
    let error = (reward - prediction).clamp(-20.0, 20.0);
    for (weight, input) in state.weights.iter_mut().zip(&decision.features) {
        *weight = (*weight + 0.1 * error * input).clamp(-20.0, 20.0);
    }
    if !state.candidate_stats.contains_key(&decision.candidate_id)
        && state.candidate_stats.len() >= 128
    {
        return Err("candidate statistics capacity reached".into());
    }
    let stats = state
        .candidate_stats
        .entry(decision.candidate_id)
        .or_insert(CandidateStats {
            visits: 0,
            mean_reward: 0.0,
        });
    stats.visits = (stats.visits + 1).min(10_000);
    stats.mean_reward += (reward - stats.mean_reward) / f64::from(stats.visits);
    state.updates += 1;
    Ok((state, reward))
}

fn run(request: Request) -> Result<Response, String> {
    match request {
        Request::Init => Ok(Response::Init {
            state: State::default(),
        }),
        Request::Decide {
            state,
            observation,
            candidates,
        } => Ok(Response::Decide {
            decision: decide(&state, observation, candidates)?,
        }),
        Request::DecidePaper { state, snapshot } => Ok(Response::Decide {
            decision: decide_paper(&state, snapshot)?,
        }),
        Request::Learn {
            state,
            decision,
            outcome,
        } => {
            let (state, reward) = learn(state, decision, outcome)?;
            Ok(Response::Learn { state, reward })
        }
    }
}

fn read_bounded(path: &Path) -> Result<Vec<u8>, String> {
    let bytes = fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    if bytes.len() > 1_000_000 {
        return Err("input file exceeds 1 MB".into());
    }
    Ok(bytes)
}

fn require_recent(path: &Path) -> Result<(), String> {
    let modified = fs::metadata(path)
        .and_then(|meta| meta.modified())
        .map_err(|e| e.to_string())?;
    if SystemTime::now()
        .duration_since(modified)
        .map_err(|_| "input timestamp is in the future")?
        > Duration::from_secs(10)
    {
        return Err("stale policy input file".into());
    }
    Ok(())
}

fn publish_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path.parent().ok_or("output path has no parent")?;
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|e| e.to_string())?;
    let result = (|| {
        file.write_all(bytes).map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        fs::hard_link(&temporary, path).map_err(|e| e.to_string())?;
        fs::File::open(parent)
            .and_then(|dir| dir.sync_all())
            .map_err(|e| e.to_string())
    })();
    let _ = fs::remove_file(&temporary);
    result
}

fn dispatch_once(args: &[String]) -> Result<serde_json::Value, String> {
    if args.len() != 8 || args[0] != "--dispatch-once" {
        return Err("usage: --dispatch-once OBS LIVE EPISODE STATE DECISION PILOT SEQ".into());
    }
    let observation_path = Path::new(&args[1]);
    let live_path = Path::new(&args[2]);
    let episode_path = Path::new(&args[3]);
    let state_path = Path::new(&args[4]);
    let decision_path = Path::new(&args[5]);
    let pilot_path = Path::new(&args[6]);
    let seq: u64 = args[7].parse().map_err(|_| "invalid sequence")?;
    if live_path.parent() != episode_path.parent()
        || decision_path == pilot_path
        || decision_path.exists()
        || pilot_path.exists()
    {
        return Err("inconsistent paths or dispatch already exists".into());
    }
    require_recent(observation_path)?;
    require_recent(live_path)?;
    let snapshot: PaperSnapshot =
        serde_json::from_slice(&read_bounded(observation_path)?).map_err(|e| e.to_string())?;
    let live: LiveSnapshot =
        serde_json::from_slice(&read_bounded(live_path)?).map_err(|e| e.to_string())?;
    let header: EpisodeHeader =
        serde_json::from_slice(&read_bounded(episode_path)?).map_err(|e| e.to_string())?;
    if !header.ai_pilot_enabled || header.id != snapshot.observation.episode {
        return Err("pilot disabled or episode mismatch".into());
    }
    if live.tick < snapshot.observation.tick || live.tick - snapshot.observation.tick > 20 {
        return Err("stale or future policy observation".into());
    }
    let selected = live
        .villagers
        .iter()
        .find(|v| v.uuid == snapshot.observation.villager_uuid)
        .ok_or("selected villager missing from live snapshot")?;
    if selected.alive != 1 || selected.sleeping != 0 || snapshot.observation.sleeping {
        return Err("selected villager unavailable or sleeping".into());
    }
    let state: State = if state_path.exists() {
        serde_json::from_slice(&read_bounded(state_path)?).map_err(|e| e.to_string())?
    } else {
        State::default()
    };
    let Some(decision) = decide_paper(&state, snapshot)? else {
        return Ok(serde_json::json!({"status":"no_decision"}));
    };
    let expires_tick = live.tick.checked_add(40).ok_or("tick overflow")?;
    let command = PilotCommand {
        seq,
        episode: &decision.episode,
        villager_uuid: &decision.villager_uuid,
        issued_tick: live.tick,
        expires_tick,
        action: &decision.action,
        x: decision.x,
        y: decision.y,
        z: decision.z,
        speed: 0.6,
    };
    let record = DispatchRecord {
        schema: "vila-pilot-dispatch/v1",
        seq,
        live_tick: live.tick,
        state: &state,
        decision: &decision,
    };
    publish_new(
        decision_path,
        &serde_json::to_vec(&record).map_err(|e| e.to_string())?,
    )?;
    publish_new(
        pilot_path,
        &serde_json::to_vec(&command).map_err(|e| e.to_string())?,
    )?;
    Ok(serde_json::json!({
        "status":"dispatched", "seq":seq, "episode":decision.episode,
        "villager_uuid":decision.villager_uuid, "decision_tick":decision.tick,
        "issued_tick":live.tick, "expires_tick":expires_tick, "goal":decision.goal
    }))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !args.is_empty() {
        match dispatch_once(&args) {
            Ok(response) => println!("{response}"),
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(2);
            }
        }
        return;
    }
    let mut input = String::new();
    if io::stdin()
        .take(1_000_001)
        .read_to_string(&mut input)
        .is_err()
        || input.len() > 1_000_000
    {
        eprintln!("invalid policy request size");
        std::process::exit(2);
    }
    let result = serde_json::from_str::<Request>(&input)
        .map_err(|e| e.to_string())
        .and_then(run);
    match result {
        Ok(response) => println!(
            "{}",
            serde_json::to_string(&response).expect("serializable response")
        ),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs() -> Observation {
        Observation {
            episode: "test".into(),
            villager_uuid: "uuid".into(),
            tick: 100,
            is_night: true,
            has_home: false,
            sleeping: false,
            stationary_home_ticks: 0,
        }
    }
    fn candidate(id: &str, distance: f64) -> Candidate {
        Candidate {
            id: id.into(),
            x: 1.0,
            y: 80.0,
            z: 2.0,
            distance,
            path_points: 5,
            crowding: 0,
            path_complete: true,
            bed_available: true,
        }
    }
    #[test]
    fn chooses_nearest_valid_candidate_before_feedback() {
        let mut blocked = candidate("blocked", 1.0);
        blocked.path_complete = false;
        let decision = decide(
            &State::default(),
            obs(),
            vec![candidate("far", 8.0), blocked, candidate("near", 3.0)],
        )
        .unwrap()
        .unwrap();
        assert_eq!(decision.candidate_id, "near");
        assert!(decision.features.iter().all(|x| x.is_finite()));
    }
    #[test]
    fn feedback_updates_state_and_rejects_replay() {
        let state = State::default();
        let decision = decide(&state, obs(), vec![candidate("bed", 3.0)])
            .unwrap()
            .unwrap();
        let outcome = Outcome {
            episode: "test".into(),
            villager_uuid: "uuid".into(),
            decision_tick: 100,
            action_tick: 120,
            end_tick: 300,
            candidate_id: "bed".into(),
            arrived: true,
            home_acquired: true,
            slept: false,
            progress_ratio: 0.8,
            health_lost: 0.0,
            rescued: false,
        };
        let (updated, reward) = learn(state, decision.clone(), outcome).unwrap();
        assert!(reward > 5.0);
        assert_eq!(updated.updates, 1);
        let replay = Outcome {
            episode: "test".into(),
            villager_uuid: "uuid".into(),
            decision_tick: 100,
            action_tick: 120,
            end_tick: 300,
            candidate_id: "bed".into(),
            arrived: true,
            home_acquired: true,
            slept: false,
            progress_ratio: 0.8,
            health_lost: 0.0,
            rescued: false,
        };
        assert!(learn(updated, decision, replay).is_err());
    }
    #[test]
    fn home_owner_has_no_action() {
        let mut observation = obs();
        observation.has_home = true;
        assert!(
            decide(&State::default(), observation, vec![candidate("bed", 3.0)])
                .unwrap()
                .is_none()
        );
    }
    #[test]
    fn bad_bed_feedback_changes_next_choice() {
        let initial = State::default();
        let first = decide(
            &initial,
            obs(),
            vec![candidate("near", 3.0), candidate("far", 4.0)],
        )
        .unwrap()
        .unwrap();
        assert_eq!(first.candidate_id, "near");
        let outcome = Outcome {
            episode: "test".into(),
            villager_uuid: "uuid".into(),
            decision_tick: 100,
            action_tick: 120,
            end_tick: 300,
            candidate_id: "near".into(),
            arrived: false,
            home_acquired: false,
            slept: false,
            progress_ratio: 0.0,
            health_lost: 2.0,
            rescued: true,
        };
        let (learned, _) = learn(initial, first, outcome).unwrap();
        let second = decide(
            &learned,
            obs(),
            vec![candidate("near", 3.0), candidate("far", 4.0)],
        )
        .unwrap()
        .unwrap();
        assert_eq!(second.candidate_id, "far");
    }
    #[test]
    fn does_not_seek_bed_during_day() {
        let mut observation = obs();
        observation.is_night = false;
        assert!(
            decide(&State::default(), observation, vec![candidate("bed", 3.0)])
                .unwrap()
                .is_none()
        );
    }
    #[test]
    fn paper_home_route_is_a_distinct_goal() {
        let mut observation = obs();
        observation.has_home = true;
        observation.stationary_home_ticks = MIN_STATIONARY_HOME_TICKS;
        let snapshot = PaperSnapshot {
            schema: PAPER_SCHEMA.into(),
            observation,
            candidates: vec![],
            candidate_count: 0,
            home_route_status: "reachable".into(),
            home_approach: Some(HomeApproach {
                x: -532.5,
                y: 86.0,
                z: -412.5,
                distance: 7.6,
                path_points: 12,
            }),
        };
        let decision = decide_paper(&State::default(), snapshot).unwrap().unwrap();
        assert_eq!(decision.goal, "return_home");
        assert_eq!(decision.action, "walk_to");
        assert_eq!(decision.candidate_id, "home_route");
        assert_eq!(decision.x, -532.5);
    }
    #[test]
    fn paper_home_route_requires_a_complete_path() {
        let mut observation = obs();
        observation.has_home = true;
        observation.stationary_home_ticks = MIN_STATIONARY_HOME_TICKS;
        let snapshot = PaperSnapshot {
            schema: PAPER_SCHEMA.into(),
            observation,
            candidates: vec![],
            candidate_count: 0,
            home_route_status: "no_complete_path".into(),
            home_approach: None,
        };
        assert!(decide_paper(&State::default(), snapshot).unwrap().is_none());
    }
    #[test]
    fn paper_does_not_wake_a_sleeping_owner() {
        let mut observation = obs();
        observation.has_home = true;
        observation.sleeping = true;
        observation.stationary_home_ticks = MIN_STATIONARY_HOME_TICKS;
        let snapshot = PaperSnapshot {
            schema: PAPER_SCHEMA.into(),
            observation,
            candidates: vec![],
            candidate_count: 0,
            home_route_status: "reachable".into(),
            home_approach: Some(HomeApproach {
                x: -532.5,
                y: 86.0,
                z: -412.5,
                distance: 7.6,
                path_points: 12,
            }),
        };
        assert!(decide_paper(&State::default(), snapshot).unwrap().is_none());
    }
    #[test]
    fn paper_home_route_waits_for_stationary_evidence() {
        for ticks in [0, MIN_STATIONARY_HOME_TICKS - 1] {
            let mut observation = obs();
            observation.has_home = true;
            observation.stationary_home_ticks = ticks;
            let snapshot = PaperSnapshot {
                schema: PAPER_SCHEMA.into(),
                observation,
                candidates: vec![],
                candidate_count: 0,
                home_route_status: "reachable".into(),
                home_approach: Some(HomeApproach {
                    x: -527.5,
                    y: 88.0,
                    z: -371.5,
                    distance: 3.0,
                    path_points: 12,
                }),
            };
            assert!(decide_paper(&State::default(), snapshot).unwrap().is_none());
        }
    }
    #[test]
    fn dispatch_writes_matched_command_once() {
        let root = std::env::temp_dir().join(format!(
            "vila-policy-dispatch-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let episode_dir = root.join("episode");
        let control_dir = root.join("control");
        fs::create_dir_all(&episode_dir).unwrap();
        fs::create_dir_all(&control_dir).unwrap();
        let observation = control_dir.join("observation.json");
        let live = episode_dir.join("live.json");
        let episode = episode_dir.join("episode.json");
        let state = control_dir.join("state.json");
        let decision = control_dir.join("decision.json");
        let pilot = control_dir.join("pilot.json");
        fs::write(
            &observation,
            serde_json::json!({
                "schema":PAPER_SCHEMA,
                "observation":{"episode":"ep","villager_uuid":"uuid","tick":100,
                    "is_night":true,"has_home":true,"sleeping":false,
                    "stationary_home_ticks":160},
                "candidates":[],"candidate_count":0,"home_route_status":"reachable",
                "home_approach":{"x":1.5,"y":80.0,"z":2.5,"distance":3.0,"path_points":4}
            })
            .to_string(),
        )
        .unwrap();
        fs::write(
            &live,
            r#"{"tick":100,"villagers":[{"uuid":"uuid","alive":1,"sleeping":0}]}"#,
        )
        .unwrap();
        fs::write(&episode, r#"{"id":"ep","ai_pilot_enabled":true}"#).unwrap();
        let args = vec![
            "--dispatch-once".into(),
            observation.display().to_string(),
            live.display().to_string(),
            episode.display().to_string(),
            state.display().to_string(),
            decision.display().to_string(),
            pilot.display().to_string(),
            "1".into(),
        ];
        let output = dispatch_once(&args).unwrap();
        assert_eq!(output["status"], "dispatched");
        let command: serde_json::Value =
            serde_json::from_slice(&fs::read(&pilot).unwrap()).unwrap();
        let record: serde_json::Value =
            serde_json::from_slice(&fs::read(&decision).unwrap()).unwrap();
        assert_eq!(command["episode"], record["decision"]["episode"]);
        assert_eq!(
            command["villager_uuid"],
            record["decision"]["villager_uuid"]
        );
        assert_eq!(command["seq"], record["seq"]);
        assert_eq!(command["issued_tick"], 100);
        assert!(dispatch_once(&args).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
