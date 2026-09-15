//! Self-contained visual map for a `flymap` connectome.
//!
//! The map is static and lives in the browser; an optional `FSPKTR01` trace adds the
//! voltage and spike activity of one selected being over time. No GPU, server or external
//! JavaScript is needed. This is deliberately a diagnostic viewer: sampling edges and
//! compacting one being's state are explicit so opening a 15M-edge map does not freeze the
//! browser by trying to draw every edge.
//!
//! ```text
//! fly_map_view --map /path/to/map.flymap --html /media/llm-data/scratch/fly.html
//! fly_map_view --map /path/to/map.flymap --trace /media/llm-data/scratch/fly.fspike \
//!     --html /media/llm-data/scratch/fly.html
//! Optional alignment uses `--trace-steps-per-tick 156` (the default for the current
//! writer) and `--telemetry-episode` (default: the last episode in the CSV).
//! ```

use serde_json::json;
use std::path::PathBuf;
mod fly_map_view_support;
use fly_map_view_support::{TelemetryRow, load_csv};
use universe::fly_map::spiking::{SpikingTrace, read_trace_sampled};
use universe::fly_map::{FlyMap, NONE};

const MAX_EDGES: usize = 12_000;
const MAX_NODES: usize = 6_000;
const MAX_FLOW_TYPES: usize = 48;
const MAX_TRACE_FRAMES: usize = 512;

fn usage() -> ! {
    eprintln!(
        "usage: fly_map_view --map <file.flymap> [--trace <file.fspike>] [--telemetry <file.csv>] [--telemetry-episode <n>] [--trace-steps-per-tick <n>] [--html <file.html>] [--json <file.json>]"
    );
    std::process::exit(2)
}

fn parse_u64_arg(args: &[String], index: usize, flag: &str) -> u64 {
    args.get(index)
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| {
            eprintln!("{flag} requires an unsigned integer");
            std::process::exit(2);
        })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut map_path: Option<PathBuf> = None;
    let mut trace_path: Option<PathBuf> = None;
    let mut html_path: Option<PathBuf> = None;
    let mut json_path: Option<PathBuf> = None;
    let mut telemetry_path: Option<PathBuf> = None;
    let mut telemetry_episode: Option<u64> = None;
    let mut trace_steps_per_tick: u64 = 156;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--map" => {
                i += 1;
                map_path = args.get(i).map(PathBuf::from);
            }
            "--trace" => {
                i += 1;
                trace_path = args.get(i).map(PathBuf::from);
            }
            "--html" => {
                i += 1;
                html_path = args.get(i).map(PathBuf::from);
            }
            "--json" => {
                i += 1;
                json_path = Some(args.get(i).map(PathBuf::from).unwrap_or_else(|| usage()));
            }
            "--telemetry" => {
                i += 1;
                telemetry_path = args.get(i).map(PathBuf::from);
            }
            "--telemetry-episode" => {
                i += 1;
                telemetry_episode = Some(parse_u64_arg(&args, i, "--telemetry-episode"));
            }
            "--trace-steps-per-tick" => {
                i += 1;
                trace_steps_per_tick = parse_u64_arg(&args, i, "--trace-steps-per-tick");
                if trace_steps_per_tick == 0 {
                    eprintln!("--trace-steps-per-tick must be greater than zero");
                    std::process::exit(2);
                }
            }
            _ => usage(),
        }
        i += 1;
    }
    let map_path = map_path.unwrap_or_else(|| usage());
    let map = FlyMap::load(map_path.to_str().unwrap_or_default()).unwrap_or_else(|e| {
        eprintln!("failed to read {}: {e}", map_path.display());
        std::process::exit(1);
    });
    let html_path = html_path.unwrap_or_else(|| {
        let mut p = map_path.clone();
        p.set_extension("html");
        p
    });

    let (node_sample, edge_sample, flow_type_ids, type_flow) = summarize_map(&map);
    let mut telemetry = match telemetry_path.as_deref() {
        Some(path) => load_csv(path).unwrap_or_else(|message| {
            eprintln!("failed to read telemetry: {message}");
            std::process::exit(1);
        }),
        None => Vec::new(),
    };
    let selected_episode =
        telemetry_episode.or_else(|| telemetry.iter().map(|row| row.episode).max());
    if let Some(episode) = selected_episode {
        telemetry.retain(|row| row.episode == episode);
        if telemetry.is_empty() {
            eprintln!("telemetry episode {episode} was not found");
            std::process::exit(1);
        }
    }
    telemetry.sort_by_key(|row| (row.being, row.tick));
    if let Err(message) = validate_telemetry(&telemetry) {
        eprintln!("invalid telemetry: {message}");
        std::process::exit(1);
    }

    let output_groups: Vec<Vec<u32>> = (0..map.k_out())
        .map(|g| {
            let start = map.out_indptr[g] as usize;
            let end = map.out_indptr[g + 1] as usize;
            map.out_members[start..end].to_vec()
        })
        .collect();

    let trace = trace_path.as_ref().map(|path| {
        read_trace_sampled(path.to_str().unwrap_or_default(), MAX_TRACE_FRAMES).unwrap_or_else(
            |e| {
                eprintln!("failed to read trace {}: {e}", path.display());
                std::process::exit(1);
            },
        )
    });
    if let Some(trace) = &trace {
        if trace.n != map.n() {
            eprintln!(
                "trace/map neuron mismatch: trace has {}, map has {}",
                trace.n,
                map.n()
            );
            std::process::exit(1);
        }
        if let Err(message) = validate_trace_alignment(trace, &telemetry, trace_steps_per_tick) {
            eprintln!("invalid trace alignment: {message}");
            std::process::exit(1);
        }
    }
    let trace_json = match &trace {
        None => json!(null),
        Some(trace) => {
            json!({
                "being": trace.being,
                "frames": trace.frames.iter().map(|frame| json!({
                    "step": frame.step,
                    "commands": frame.commands,
                    "v": frame.v,
                    "spikes": frame.spike_counts,
                })).collect::<Vec<_>>(),
            })
        }
    };

    let data = json!({
        "n": map.n(),
        "m": map.m(),
        "nTypes": map.n_types,
        "kIn": map.k_in,
        "kOut": map.k_out(),
        "hasPreCsr": map.pre_csr.is_some(),
        "hasPixels": map.pixel_of_neuron.is_some(),
        "eyeBound": map.pixel_of_neuron.as_ref().map_or(0, |p| p.iter().filter(|&&x| x != NONE).count()),
        "typeFlow": type_flow,
        "flowTypeIds": flow_type_ids,
        "nodes": node_sample,
        "edges": edge_sample,
        "groups": output_groups,
        "trace": trace_json,
        "tracePath": trace_path.as_ref().and_then(|path| std::fs::canonicalize(path).ok()).map(|path| path.to_string_lossy().into_owned()),
        "telemetry": telemetry.iter().map(|row| json!({
            "episode": row.episode, "tick": row.tick, "being": row.being, "alive": row.alive,
            "coverage": row.coverage, "asymmetry": row.asymmetry, "leftPixels": row.left_pixels,
            "rightPixels": row.right_pixels, "visibleDistance": row.visible_distance,
            "bodyDistance": row.body_distance, "overlapCount": row.overlap_count,
            "rawSteer": row.raw_steer, "appliedSteer": row.applied_steer,
            "throttle": row.throttle, "brake": row.brake, "speed": row.speed,
            "x": row.x, "z": row.z, "forwardX": row.forward_x, "forwardZ": row.forward_z,
        })).collect::<Vec<_>>(),
        "telemetryEpisode": selected_episode,
        "traceStepsPerTick": trace_steps_per_tick,
    });
    if let Some(path) = json_path {
        if let Err(error) = std::fs::write(&path, data.to_string()) {
            eprintln!("could not write {}: {error}", path.display());
            std::process::exit(1);
        }
    }
    let html = render_html(&data.to_string());
    if let Err(e) = std::fs::write(&html_path, html) {
        eprintln!("could not write {}: {e}", html_path.display());
        std::process::exit(1);
    }
    println!(
        "→ {} · {} neurons · {} edges ({} drawn) · trace {}",
        html_path.display(),
        map.n(),
        map.m(),
        edge_sample.len(),
        if trace.is_some() { "loaded" } else { "off" }
    );
}

/// Summarises the connectome in bounded memory while keeping graph edges drawable.
///
/// Sampling nodes and edges independently made almost every sampled edge disappear in
/// the browser because one or both endpoints were absent. The first pass now selects
/// endpoints from deterministic edge samples; the second retains only edges whose two
/// real endpoints are in that node set. The type-flow matrix still counts the complete
/// map for the most connected types.
fn summarize_map(map: &FlyMap) -> (Vec<[u32; 4]>, Vec<[u32; 3]>, Vec<u32>, Vec<u32>) {
    let edge_stride = map.m().div_ceil(MAX_EDGES).max(1);
    let mut incidence = vec![0u64; map.n_types];
    let mut selected = vec![false; map.n()];
    let mut selected_count = 0usize;
    for_each_edge(map, |edge_idx, src, dst, _weight| {
        let src_type = map.type_of[src as usize] as usize;
        let dst_type = map.type_of[dst as usize] as usize;
        incidence[src_type] = incidence[src_type].saturating_add(1);
        incidence[dst_type] = incidence[dst_type].saturating_add(1);
        if edge_idx % edge_stride != 0 {
            return;
        }
        let needed = usize::from(!selected[src as usize]) + usize::from(!selected[dst as usize]);
        if selected_count + needed <= MAX_NODES {
            if !selected[src as usize] {
                selected[src as usize] = true;
                selected_count += 1;
            }
            if !selected[dst as usize] {
                selected[dst as usize] = true;
                selected_count += 1;
            }
        }
    });

    if selected_count < MAX_NODES.min(map.n()) {
        let node_stride = map.n().div_ceil(MAX_NODES).max(1);
        for neuron in (0..map.n()).step_by(node_stride) {
            if !selected[neuron] {
                selected[neuron] = true;
                selected_count += 1;
                if selected_count == MAX_NODES.min(map.n()) {
                    break;
                }
            }
        }
    }

    let mut type_ids: Vec<usize> = (0..map.n_types).collect();
    type_ids.sort_unstable_by(|&a, &b| incidence[b].cmp(&incidence[a]).then_with(|| a.cmp(&b)));
    type_ids.truncate(MAX_FLOW_TYPES);
    type_ids.sort_unstable();
    let mut type_slot = vec![usize::MAX; map.n_types];
    for (slot, &type_id) in type_ids.iter().enumerate() {
        type_slot[type_id] = slot;
    }
    let width = type_ids.len();
    let mut flow = vec![0u32; width * width];
    let mut edge_sample = Vec::with_capacity(MAX_EDGES.min(map.m()));
    for_each_edge(map, |_, src, dst, weight| {
        let a = type_slot[map.type_of[src as usize] as usize];
        let b = type_slot[map.type_of[dst as usize] as usize];
        if a != usize::MAX && b != usize::MAX {
            flow[a * width + b] = flow[a * width + b].saturating_add(1);
        }
        if edge_sample.len() < MAX_EDGES && selected[src as usize] && selected[dst as usize] {
            edge_sample.push([src, dst, weight.to_bits()]);
        }
    });
    let nodes = selected
        .iter()
        .enumerate()
        .filter(|(_, keep)| **keep)
        .map(|(neuron, _)| {
            [
                neuron as u32,
                map.type_of[neuron],
                map.pixel_of_neuron
                    .as_ref()
                    .and_then(|pixels| pixels.get(neuron))
                    .copied()
                    .unwrap_or(NONE),
                output_group(map, neuron),
            ]
        })
        .collect();
    (
        nodes,
        edge_sample,
        type_ids.into_iter().map(|id| id as u32).collect(),
        flow,
    )
}

fn for_each_edge(map: &FlyMap, mut f: impl FnMut(usize, u32, u32, f32)) {
    if let Some(pre) = &map.pre_csr {
        let mut edge_idx = 0;
        for src in 0..map.n() {
            for edge in pre.indptr[src] as usize..pre.indptr[src + 1] as usize {
                f(edge_idx, src as u32, pre.indices[edge], pre.weights[edge]);
                edge_idx += 1;
            }
        }
    } else {
        let mut edge_idx = 0;
        for dst in 0..map.n() {
            for edge in map.indptr[dst] as usize..map.indptr[dst + 1] as usize {
                f(edge_idx, map.indices[edge], dst as u32, map.weights[edge]);
                edge_idx += 1;
            }
        }
    }
}

fn output_group(map: &FlyMap, neuron: usize) -> u32 {
    for group in 0..map.k_out() {
        let start = map.out_indptr[group] as usize;
        let end = map.out_indptr[group + 1] as usize;
        if map.out_members[start..end].contains(&(neuron as u32)) {
            return group as u32;
        }
    }
    NONE
}

fn trace_step_to_tick(step: u64, steps_per_tick: u64) -> Result<u64, String> {
    if steps_per_tick == 0 {
        return Err("steps-per-tick must be greater than zero".to_owned());
    }
    if step < steps_per_tick {
        return Err(format!(
            "trace step {step} is before the first complete world tick"
        ));
    }
    if step % steps_per_tick != 0 {
        return Err(format!(
            "trace step {step} is not divisible by {steps_per_tick} steps per tick"
        ));
    }
    Ok(step / steps_per_tick - 1)
}

fn validate_telemetry(rows: &[TelemetryRow]) -> Result<(), String> {
    for pair in rows.windows(2) {
        if pair[0].being == pair[1].being && pair[0].tick == pair[1].tick {
            return Err(format!(
                "duplicate row for being {} at tick {}",
                pair[0].being, pair[0].tick
            ));
        }
    }
    Ok(())
}

fn validate_trace_alignment(
    trace: &SpikingTrace,
    telemetry: &[TelemetryRow],
    steps_per_tick: u64,
) -> Result<(), String> {
    let mut previous_tick = None;
    for frame in &trace.frames {
        let tick = trace_step_to_tick(frame.step, steps_per_tick)?;
        if previous_tick.is_some_and(|previous| tick <= previous) {
            return Err(format!(
                "trace tick {tick} follows tick {}; the trace reset or repeated a frame",
                previous_tick.unwrap()
            ));
        }
        if !telemetry.is_empty()
            && !telemetry
                .iter()
                .any(|row| row.being == trace.being && row.tick == tick)
        {
            return Err(format!(
                "no telemetry row for trace being {} at mapped tick {tick}",
                trace.being
            ));
        }
        previous_tick = Some(tick);
    }
    Ok(())
}

fn render_html(data: &str) -> String {
    include_str!("fly_map_view_support/viewer.html")
        .replace("__DATA__", data)
        .replace("__MODEL__", include_str!("fly_map_view_support/model.js"))
        .replace("__SCRIPT__", include_str!("fly_map_view_support/viewer.js"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use universe::fly_map::spiking::SpikingTraceFrame;

    fn frame(step: u64) -> SpikingTraceFrame {
        SpikingTraceFrame {
            step,
            commands: [0.0; 3],
            v: vec![0.0],
            spike_counts: vec![0],
        }
    }

    #[test]
    fn maps_complete_lif_steps_to_the_world_tick_that_produced_them() {
        assert_eq!(trace_step_to_tick(156, 156).unwrap(), 0);
        assert_eq!(trace_step_to_tick(10_140, 156).unwrap(), 64);
        assert!(trace_step_to_tick(155, 156).is_err());
        assert!(trace_step_to_tick(157, 156).is_err());
    }

    #[test]
    fn rejects_trace_resets_instead_of_rendering_a_false_alignment() {
        let trace = SpikingTrace {
            n: 1,
            being: 0,
            frames: vec![frame(312), frame(156)],
        };
        let error = validate_trace_alignment(&trace, &[], 156).unwrap_err();
        assert!(error.contains("reset or repeated"));
    }

    #[test]
    fn rejects_trace_and_telemetry_from_different_sample_ticks() {
        let header = "episode,tick,being,alive,coverage,asymmetry,left_pixels,right_pixels,visible_distance,body_distance,overlap_count,raw_steer,applied_steer,throttle,brake,speed";
        let csv = format!("{header}\n1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0\n");
        let telemetry = fly_map_view_support::parse_csv(&csv).unwrap();
        let trace = SpikingTrace {
            n: 1,
            being: 0,
            frames: vec![frame(10_140)],
        };
        let error = validate_trace_alignment(&trace, &telemetry, 156).unwrap_err();
        assert!(error.contains("mapped tick 64"));
    }

    #[test]
    fn generated_html_embeds_data_and_the_external_source_assets() {
        let html = render_html("{\"n\":1}");
        assert!(html.contains("window.FLY_MAP_DATA = {\"n\":1}"));
        assert!(html.contains("function createModel"));
        assert!(html.contains("function traceStepToTick"));
        assert!(!html.contains("__MODEL__"));
        assert!(!html.contains("__SCRIPT__"));
    }
}
