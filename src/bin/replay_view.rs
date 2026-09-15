//! Export the existing binary replay as a scene with an optional measured brain trace.

use std::path::PathBuf;
use universe::replay::{PlayerOptions, ReplayReader, generate_html_player_with_options};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut replay = None;
    let mut html = None;
    let mut brain = None;
    let mut brain3d = None;
    let mut diagnostics = None;
    let mut native_genome = None;
    let mut native_being = None;
    let mut reconstruct_terrain = false;
    let mut width = 18.0_f32;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" {
            println!(
                "replay_view --replay FILE --html FILE [--width METRES] [--brain JSON] [--brain3d JSON] [--diagnostics JSON] [--native-genome CHECKPOINT --native-being INDEX] [--reconstruct-terrain]"
            );
            return Ok(());
        }
        if flag == "--reconstruct-terrain" {
            reconstruct_terrain = true;
            continue;
        }
        let value = args
            .next()
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--replay" => replay = Some(PathBuf::from(value)),
            "--html" => html = Some(PathBuf::from(value)),
            "--brain" => brain = Some(PathBuf::from(value)),
            "--brain3d" => brain3d = Some(PathBuf::from(value)),
            "--diagnostics" => diagnostics = Some(PathBuf::from(value)),
            "--native-genome" => native_genome = Some(PathBuf::from(value)),
            "--native-being" => native_being = Some(value.parse::<usize>()?),
            "--width" => width = value.parse()?,
            _ => return Err(format!("unknown argument: {flag}").into()),
        }
    }
    let replay = replay.ok_or("--replay is required")?;
    let html = html.ok_or("--html is required")?;
    let mut reader = ReplayReader::open(&replay)?;
    let mut options = PlayerOptions::from_replay(&replay, &reader.header, width)?;
    if let Some(path) = brain {
        let data: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        if !data["nodes"].is_array()
            || !data["edges"].is_array()
            || !data["trace"]["frames"].is_array()
        {
            return Err("brain JSON must contain nodes, edges and measured trace frames".into());
        }
        let being = data["trace"]["being"]
            .as_u64()
            .ok_or("missing trace being")?;
        if being >= reader.header.n_beings as u64 {
            return Err("trace being is outside the replay population".into());
        }
        options.brain = Some(data);
    }
    if let Some(path) = brain3d {
        options.brain3d = Some(serde_json::from_slice(&std::fs::read(path)?)?);
    }
    if let Some(path) = diagnostics {
        options.diagnostics = Some(serde_json::from_slice(&std::fs::read(path)?)?);
    }
    match (native_genome, native_being) {
        (Some(path), Some(being)) => {
            if being >= reader.header.n_beings as usize {
                return Err("--native-being is outside the replay population".into());
            }
            options.attach_native_checkpoint(&path, being)?;
        }
        (Some(_), None) => {
            return Err("--native-genome requires --native-being; checkpoint ownership cannot be inferred from a replay".into());
        }
        (None, Some(_)) => return Err("--native-being requires --native-genome".into()),
        (None, None) => {}
    }
    let width = options.metadata["track_width"]
        .as_f64()
        .ok_or("missing track width")? as f32;
    if reconstruct_terrain && options.metadata.get("terrain").is_some() {
        return Err("--reconstruct-terrain refuses to overwrite existing terrain metadata".into());
    }
    let previous_no_holes = universe::world::NO_HOLES.load(std::sync::atomic::Ordering::Relaxed);
    if reconstruct_terrain {
        // Legacy recordings may be reconstructed only from an explicit no-holes setting;
        // guessing this changes the hazards while preserving the same replay frames.
        let no_holes = options.metadata["no_holes"]
            .as_bool()
            .ok_or("--reconstruct-terrain requires metadata.no_holes boolean")?;
        universe::world::NO_HOLES.store(no_holes, std::sync::atomic::Ordering::Relaxed);
    }
    let world = {
        let _quiet = universe::world::silence_map_diag();
        universe::world::VoxelWorld::new(reader.header.world_seed, width)
    };
    universe::world::NO_HOLES.store(previous_no_holes, std::sync::atomic::Ordering::Relaxed);
    let track: Vec<[f32; 2]> = world.waypoints_world.iter().map(|p| [p.x, p.y]).collect();
    options.metadata["lap_distance"] = serde_json::json!(world.lap_distance);
    if reconstruct_terrain {
        options.metadata["terrain"] = universe::replay::terrain::snapshot(&world);
        options.metadata["terrain_provenance"] =
            serde_json::json!("reconstructed_from_seed_and_recorded_settings");
    }
    let frames = reader.all_frames();
    let labels: Vec<String> = (0..frames.len())
        .map(|id| format!("Being {id:02}"))
        .collect();
    let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
    if frames.iter().all(Vec::is_empty) {
        return Err("replay contains no frames".into());
    }
    generate_html_player_with_options(&track, &frames, &labels, &html, &options)?;
    println!(
        "{} · {} beings · world {} · {} Hz",
        html.display(),
        frames.len(),
        reader.header.world_seed,
        reader.header.tick_rate
    );
    Ok(())
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("replay_view: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
