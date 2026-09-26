//! Standalone scene replay export with explicit measurement availability.

use super::{ReplayFrame, ReplayHeader, SOURCE_MOTOR};
use serde_json::{Value, json};
use std::io;
use std::path::{Component, Path};

#[path = "player_brain.rs"]
mod brain;
#[path = "player_diagnostics.rs"]
mod diagnostics;
#[path = "player_native.rs"]
mod native;

pub struct PlayerOptions {
    pub metadata: Value,
    pub brain: Option<Value>,
    pub brain3d: Option<Value>,
    pub native_brain: Option<Value>,
    pub diagnostics: Option<Value>,
}

impl Default for PlayerOptions {
    fn default() -> Self {
        Self {
            metadata: json!({
                "schema_version": 2, "tick_rate": 64.0, "track_width": 18.0,
                "source": "legacy replay", "life_semantics": "unavailable",
                "fuel_enabled": null, "raw_intents_recorded": false,
                "retina_recorded": false, "neural_summary_recorded": false,
                "wall_probes_recorded": false,
                "laps_recorded": false, "frame_phase": "unspecified",
                "sensor_phase": "unspecified", "arc_semantics": "recorded_distance_m"
            }),
            brain: None,
            brain3d: None,
            native_brain: None,
            diagnostics: None,
        }
    }
}

impl PlayerOptions {
    /// Sidecars describe facts absent from the compatible 128-byte frame format.
    pub fn from_replay(path: &Path, header: &ReplayHeader, width: f32) -> io::Result<Self> {
        if !width.is_finite()
            || width <= 0.0
            || !header.tick_rate.is_finite()
            || header.tick_rate <= 0.0
        {
            return Err(invalid(
                "track width and tick rate must be finite and positive",
            ));
        }
        let mut result = Self::default();
        let meta = result.metadata.as_object_mut().unwrap();
        meta.insert("tick_rate".into(), json!(header.tick_rate));
        meta.insert("track_width".into(), json!(width));
        meta.insert("world_seed".into(), json!(header.world_seed));
        if header.source == SOURCE_MOTOR {
            meta.extend(
                json!({"source":"motor", "life_semantics":"health",
                "fuel_enabled":true, "raw_intents_recorded":true,
                "retina_recorded":true, "neural_summary_recorded":true,
                "wall_probes_recorded":true,
                "laps_recorded":true})
                .as_object()
                .unwrap()
                .clone(),
            );
        }
        let sidecar = path.with_extension("meta.json");
        if sidecar.exists() {
            let values: Value =
                serde_json::from_slice(&std::fs::read(sidecar)?).map_err(invalid)?;
            let values = values
                .as_object()
                .ok_or_else(|| invalid("replay metadata must be an object"))?;
            if values.get("schema_version").and_then(Value::as_u64) != Some(1)
                || values.get("world_seed").and_then(Value::as_u64) != Some(header.world_seed)
                || values.get("tick_rate").and_then(Value::as_f64) != Some(header.tick_rate as f64)
            {
                return Err(invalid(
                    "replay metadata version, seed or clock does not match its header",
                ));
            }
            meta.extend(values.clone());
            // Early schema-1 recordings named the measured source before adding
            // a dedicated availability flag; preserve that explicit provenance.
            if !values.contains_key("wall_probes_recorded")
                && values.get("sensor_source").and_then(Value::as_str)
                    == Some("captured_cpu_queries")
            {
                meta.insert("wall_probes_recorded".into(), json!(true));
            }
        }
        meta.insert("schema_version".into(), json!(2));
        Ok(result)
    }

    /// Attach the exact native checkpoint topology to one explicitly identified replay
    /// being. Checkpoint bytes must match the identity recorded beside the replay.
    pub fn attach_native_checkpoint(&mut self, path: &Path, trace_being: usize) -> io::Result<()> {
        native::validate_association(&self.metadata)?;
        let expected_hash = self.metadata["checkpoint_sha256"].as_str().ok_or_else(|| {
            invalid(
                "native topology requires metadata.checkpoint_sha256; checkpoint ownership cannot be inferred",
            )
        })?;
        let slow_kinematics = self.metadata["slow_kinematics"]
            .as_str()
            .unwrap_or("legacy");
        let mut payload =
            native::from_checkpoint(path, trace_being, expected_hash, slow_kinematics)?;
        apply_motor_head_provenance(&mut payload, path, &self.metadata)?;
        self.native_brain = Some(payload);
        Ok(())
    }

    /// Auto-select the generation-specific genome the diagnostics writer saved beside
    /// the selected observation, when the caller passed no explicit `--native-genome`.
    /// Returns `true` when attached and `false` when no native diagnostics are present.
    /// Once native provenance is claimed, a selected row with a missing/mismatched
    /// genome is an error rather than a silent fallback.
    pub fn attach_observed_genome(&mut self, replay_path: &Path) -> io::Result<bool> {
        let Some(diagnostics) = &self.diagnostics else {
            return Ok(false);
        };
        let Some(selected_being) = diagnostics["meta"]["selected_being"].as_u64() else {
            return Ok(false);
        };
        if self.metadata["brain_model"].as_str() != Some("native") {
            // An observation may accompany a non-native replay for its charts, but
            // cannot establish native-controller provenance for that replay.
            return Ok(false);
        }
        let Some(replay_generation) = self.metadata["generation"].as_u64() else {
            return Err(invalid(
                "observed native genome requires replay metadata.generation",
            ));
        };
        let observation_schema = diagnostics["meta"]["observation_schema"].as_u64();
        if observation_schema != Some(2) {
            return Err(invalid(format!(
                "observed native genome requires diagnostics observation_schema=2, got {observation_schema:?}"
            )));
        }
        let observations = diagnostics["observations"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        let same_being: Vec<&Value> = observations
            .iter()
            .filter(|row| row["being"].as_u64() == Some(selected_being))
            .collect();
        if same_being.is_empty() {
            // The selected being was recorded, but nothing in this diagnostics
            // report is about it -- most likely a diagnostics report from an
            // unrelated run. Nothing to auto-detect from; not an error.
            return Ok(false);
        }
        let row = same_being
            .iter()
            .find(|row| {
                row["genome_file"].as_str().is_some_and(|name| !name.is_empty())
                    && row["generation"].as_u64() == Some(replay_generation)
            })
            .ok_or_else(|| {
                invalid(format!(
                    "no observation for being {selected_being} names a genome_file matching replay generation {replay_generation}; pass --native-genome explicitly"
                ))
        })?;
        let genome_file = row["genome_file"].as_str().unwrap();
        let replay_dir = replay_path.parent().unwrap_or_else(|| Path::new("."));
        let replay_root = std::fs::canonicalize(replay_dir).map_err(|e| {
            invalid(format!(
                "observed genome {genome_file} declared by the observation is missing: replay directory {} cannot be resolved ({e})",
                replay_dir.display()
            ))
        })?;
        let slow_kinematics = self.metadata["slow_kinematics"]
            .as_str()
            .unwrap_or("legacy");
        let mut payloads = serde_json::Map::new();
        for candidate in same_being
            .iter()
            .filter(|r| r["generation"].as_u64() == Some(replay_generation))
        {
            let name = candidate["genome_file"]
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| invalid("observation is missing its genome_file"))?;
            if payloads.contains_key(name) {
                continue;
            }
            let relative = Path::new(name);
            if relative.components().count() != 1
                || !relative
                    .components()
                    .all(|c| matches!(c, Component::Normal(_)))
            {
                return Err(invalid(format!(
                    "observed genome path must be a single relative file name: {name:?}"
                )));
            }
            let resolved = std::fs::canonicalize(replay_dir.join(relative)).map_err(|e| {
                invalid(format!(
                    "observed genome {name} declared by the observation is missing at {} ({e})",
                    replay_dir.join(relative).display()
                ))
            })?;
            if resolved.parent() != Some(replay_root.as_path()) {
                return Err(invalid(
                    "observed genome resolves outside the replay directory",
                ));
            }
            let mut payload =
                native::from_observed_genome(&resolved, selected_being as usize, slow_kinematics)?;
            apply_motor_head_provenance(&mut payload, &resolved, &self.metadata)?;
            payloads.insert(name.to_owned(), payload);
        }
        let mut initial = payloads
            .get(genome_file)
            .expect("selected observed genome was loaded")
            .clone();
        initial["observedGenomes"] = Value::Object(payloads);
        self.native_brain = Some(initial);
        Ok(true)
    }
}

/// The genotype alone cannot identify whether the historical shared head was used.
fn apply_motor_head_provenance(
    payload: &mut Value,
    path: &Path,
    metadata: &Value,
) -> io::Result<()> {
    let inherited = flag(metadata, "body_evolution_enabled")
        || flag(metadata, "ecology")
        || metadata["lifetime_flags"].as_u64().unwrap_or(0) != 0;
    if inherited {
        let genome = crate::gpu::genome_io::load_genome(
            path.to_str()
                .ok_or_else(|| invalid("non-UTF8 genome path"))?,
        )
        .ok_or_else(|| invalid("cannot load inherited motor head"))?;
        payload["fast"]["outputKnots"] = json!(genome.kan_knots);
        payload["fast"]["outputKnotsSource"] =
            json!("inherited per-being spline; inactive when recorded motor_head is linear");
    }
    Ok(())
}

fn invalid(message: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

fn measured(value: f32) -> io::Result<f32> {
    value
        .is_finite()
        .then_some(value)
        .ok_or_else(|| invalid("nonfinite replay measurement"))
}

fn flag(meta: &Value, name: &str) -> bool {
    meta[name].as_bool() == Some(true)
}

fn frame_json(f: &ReplayFrame, meta: &Value) -> io::Result<Value> {
    for value in f.pos.into_iter().chain(f.fwd).chain([
        f.speed,
        f.yaw_rate,
        f.arc_prog,
        f.dist_wall_l,
        f.dist_wall_r,
        f.dist_wall_f,
        f.final_thr,
        f.final_str,
        f.final_brk,
    ]) {
        measured(value)?;
    }
    let raw = flag(meta, "raw_intents_recorded");
    if raw {
        measured(f.intent_thr)?;
        measured(f.intent_str)?;
        measured(f.intent_brk)?;
    }
    let neural = flag(meta, "neural_summary_recorded");
    let retina = flag(meta, "retina_recorded");
    Ok(json!({
        "t": f.tick, "x": f.pos[0], "y": f.pos[2], "height": f.pos[1],
        "fx": f.fwd[0], "fy": f.fwd[1], "spd": f.speed, "yawRate": f.yaw_rate,
        "gear": f.gear, "life": meta["life_semantics"].as_str()
            .filter(|kind| matches!(*kind, "health" | "remaining_tick_budget" | "alive_flag"))
            .map(|_| f.hp_norm as f32 / 127.0),
        "fuel": flag(meta,"fuel_enabled").then_some(f.fuel_norm as f32 / 127.0),
        "thr": f.final_thr, "str": f.final_str, "brk": f.final_brk,
        "ithr": raw.then_some(f.intent_thr), "istr": raw.then_some(f.intent_str), "ibrk": raw.then_some(f.intent_brk),
        "cl":flag(meta,"wall_probes_recorded").then_some(f.dist_wall_l),
        "cr":flag(meta,"wall_probes_recorded").then_some(f.dist_wall_r),
        "cf":flag(meta,"wall_probes_recorded").then_some(f.dist_wall_f),
        "c40":neural.then_some(f.curv_40m), "c70":neural.then_some(f.curv_70m), "c120":neural.then_some(f.curv_120m),
        "fl":neural.then_some(f.flow_align), "lo":neural.then_some(f.line_offset),
        "ds":neural.then_some(f.dims_saturated), "sf":neural.then_some(f.safety_flags),
        "intentMag":neural.then_some(f.intent_mag_q as f32/127.0),
        "axiomStrength":neural.then_some(f.axiom_str_q as f32/127.0),
        "fov":retina.then_some(f.retina_fov), "peri":retina.then_some(f.retina_peri),
        "laps":flag(meta,"laps_recorded").then_some(f.laps), "arc": f.arc_prog,
    }))
}

/// Serializes strings safely inside an inline script without HTML termination.
fn script_json(value: &Value) -> String {
    value
        .to_string()
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('&', "\\u0026")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029")
}

/// The public compatibility entry point preserves the old call signature.
pub fn generate_html_player(
    track: &[[f32; 2]],
    frames: &[Vec<ReplayFrame>],
    labels: &[&str],
    output: &Path,
) -> io::Result<()> {
    generate_html_player_with_options(track, frames, labels, output, &PlayerOptions::default())
}

pub fn generate_html_player_with_options(
    track: &[[f32; 2]],
    frames: &[Vec<ReplayFrame>],
    labels: &[&str],
    output: &Path,
    options: &PlayerOptions,
) -> io::Result<()> {
    if track.len() < 2 || track.iter().flatten().any(|v| !v.is_finite()) {
        return Err(invalid("replay requires a finite world-space track"));
    }
    let width = options.metadata["track_width"].as_f64().unwrap_or(0.0);
    let hz = options.metadata["tick_rate"].as_f64().unwrap_or(0.0);
    if width <= 0.0 || hz <= 0.0 {
        return Err(invalid("invalid replay dimensions or clock"));
    }
    if options.native_brain.is_some() && options.brain3d.is_some() {
        return Err(invalid(
            "a replay cannot attach native topology and fly anatomy at the same time",
        ));
    }
    if options.metadata["brain_model"].as_str() == Some("native") && options.brain3d.is_some() {
        return Err(invalid(
            "fly anatomical data cannot be attached to a native-controller replay",
        ));
    }
    if options
        .metadata
        .get("brain_model")
        .and_then(Value::as_str)
        .is_some_and(|model| model != "native")
        && options.native_brain.is_some()
    {
        return Err(invalid(
            "native topology cannot be attached to a non-native controller replay",
        ));
    }
    if let Some(data) = &options.brain {
        brain::validate(data, frames, &options.metadata)?;
    }
    let mut beings = Vec::with_capacity(frames.len());
    for (id, sequence) in frames.iter().enumerate() {
        if sequence.windows(2).any(|pair| pair[1].tick <= pair[0].tick) {
            return Err(invalid("replay ticks must increase within each being"));
        }
        let values = sequence
            .iter()
            .map(|f| frame_json(f, &options.metadata))
            .collect::<io::Result<Vec<_>>>()?;
        beings.push(
            json!({"id":id,"label":labels.get(id).copied().unwrap_or("Being"),"frames":values}),
        );
    }
    if let Some(data) = &options.diagnostics {
        diagnostics::validate(data, frames, &options.metadata)?;
    }
    if let Some(data) = &options.brain3d {
        let trace = json!({"n":data["n"],"tracePath":data["tracePath"],
            "traceStepsPerTick":data["traceStepsPerTick"],
            "trace":{"being":data["traceBeing"],"frames":data["frames"]}});
        brain::validate(&trace, frames, &options.metadata)?;
    }
    if let Some(data) = &options.native_brain {
        let being = data["traceBeing"]
            .as_u64()
            .ok_or_else(|| invalid("native brain is missing traceBeing"))?
            as usize;
        if being >= frames.len() {
            return Err(invalid(
                "native brain traceBeing is outside the replay population",
            ));
        }
    }
    let data = format!(
        "<script>window.REPLAY_TRACK={};window.REPLAY_BEINGS={};window.REPLAY_META={};window.REPLAY_BRAIN={};window.REPLAY_BRAIN_3D={};window.REPLAY_NATIVE_BRAIN={};window.DRIVING_DATA={};</script>",
        script_json(&json!(track)),
        script_json(&json!(beings)),
        script_json(&options.metadata),
        script_json(options.brain.as_ref().unwrap_or(&Value::Null)),
        script_json(options.brain3d.as_ref().unwrap_or(&Value::Null)),
        script_json(options.native_brain.as_ref().unwrap_or(&Value::Null)),
        script_json(options.diagnostics.as_ref().unwrap_or(&Value::Null))
    );
    let html = include_str!("../assets/replay_player.html")
        .replace(
            "__REPLAY_STYLE__",
            include_str!("../assets/replay_player/style.css"),
        )
        .replace(
            "__REPLAY_MODEL__",
            include_str!("../assets/replay_player/model.js"),
        )
        .replace(
            "__REPLAY_VIEW__",
            include_str!("../assets/replay_player/view.js"),
        )
        .replace("__BRAIN_3D__", include_str!("../assets/brain3d/brain3d.js"))
        .replace(
            "__NATIVE_BRAIN_3D__",
            include_str!("../assets/replay_player/native_brain.js"),
        )
        .replace(
            "__REPLAY_STUDY__",
            concat!(
                include_str!("../assets/replay_player/vision_projection.js"),
                "\n",
                include_str!("../assets/replay_player/study.js")
            ),
        )
        .replace(
            "__STUDY_STYLE__",
            include_str!("../assets/replay_player/study.css"),
        )
        .replacen("</head>", &format!("{data}</head>"), 1);
    std::fs::write(output, html)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_coordinates_keep_aspect_and_missing_measurements_stay_missing() {
        let f = ReplayFrame {
            pos: [10.25, 1.0, 400.5],
            fwd: [0.6, 0.8],
            fuel_norm: 127,
            ..Default::default()
        };
        let value = frame_json(&f, &PlayerOptions::default().metadata).unwrap();
        assert_eq!(value["x"], 10.25);
        assert_eq!(value["y"], 400.5);
        assert!(value["fuel"].is_null());
        assert!(value["ithr"].is_null());
        assert!(value["fov"].is_null());
        assert!(value["laps"].is_null());
    }

    #[test]
    fn actual_fuel_and_commands_are_exported_when_recorded() {
        let f = ReplayFrame {
            fuel_norm: 64,
            intent_str: -0.7,
            ..Default::default()
        };
        let value = frame_json(
            &f,
            &json!({"fuel_enabled":true,"raw_intents_recorded":true}),
        )
        .unwrap();
        assert!((value["fuel"].as_f64().unwrap() - 64.0 / 127.0).abs() < 1e-6);
        assert!((value["istr"].as_f64().unwrap() + 0.7).abs() < 1e-6);
    }

    #[test]
    fn bad_numeric_data_is_rejected_and_script_labels_cannot_break_out() {
        let f = ReplayFrame {
            speed: f32::NAN,
            ..Default::default()
        };
        assert!(frame_json(&f, &json!({})).is_err());
        let encoded = script_json(&json!("</script><img onerror='bad'>"));
        assert!(!encoded.contains('<'));
        assert!(!encoded.contains('>'));
    }

    #[test]
    fn attach_observed_genome_is_a_silent_noop_without_diagnostics_or_selected_being() {
        let mut options = PlayerOptions::default();
        assert!(
            !options
                .attach_observed_genome(Path::new("/nonexistent/whatever.replay"))
                .unwrap()
        );
        options.diagnostics = Some(json!({"meta": {}, "observations": []}));
        assert!(
            !options
                .attach_observed_genome(Path::new("/nonexistent/whatever.replay"))
                .unwrap()
        );
    }

    #[test]
    fn attach_observed_genome_is_a_silent_noop_for_a_replay_that_never_claimed_native() {
        // Default metadata has no checkpoint_mix/generation/tau_reseeded, so
        // validate_association fails -- this must not become a hard error for a
        // replay that never asked to be treated as a native checkpoint clone.
        let mut options = PlayerOptions::default();
        options.diagnostics = Some(json!({"meta":{"selected_being":0},
            "observations":[{"being":0,"generation":0,"genome_file":"g.bin"}]}));
        assert!(
            !options
                .attach_observed_genome(Path::new("/nonexistent/whatever.replay"))
                .unwrap()
        );
    }

    #[test]
    fn attach_observed_genome_errors_when_the_named_file_is_missing() {
        let mut options = PlayerOptions::default();
        options.metadata = json!({"brain_model":"native","checkpoint_mix":1.0,"generation":0,"tau_reseeded":false});
        options.diagnostics = Some(json!({"meta":{"selected_being":0,"observation_schema":2},
            "observations":[{"being":0,"generation":0,"genome_file":"missing.bin"}]}));
        let error = options
            .attach_observed_genome(Path::new("/nonexistent/dir/gen0000.replay"))
            .unwrap_err();
        assert!(error.to_string().contains("missing"), "{error}");
    }

    #[test]
    fn attach_observed_genome_errors_when_no_observation_names_a_genome_for_this_generation() {
        let mut options = PlayerOptions::default();
        options.metadata = json!({"brain_model":"native","checkpoint_mix":1.0,"generation":0,"tau_reseeded":false});
        options.diagnostics = Some(json!({"meta":{"selected_being":0,"observation_schema":2},
            "observations":[{"being":0,"generation":1,"genome_file":"g.bin"}]}));
        let error = options
            .attach_observed_genome(Path::new("/nonexistent/whatever.replay"))
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("genome_file matching replay generation"),
            "{error}"
        );
    }

    #[test]
    fn attach_observed_genome_attaches_the_exact_file_the_matching_observation_names() {
        use crate::gpu::evolution::Genome;
        use crate::gpu::genome_io::save_genome;
        let mut original = Genome::random(4242);
        original.randomize_tau(11);
        let dir = std::env::temp_dir().join("universe_attach_observed_genome_ok");
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let genome_path = dir.join("observed_genome_gen0000.bin");
        save_genome(&original, genome_path.to_str().unwrap()).expect("wrote genome");
        let replay_path = dir.join("gen0000.replay");
        let mut options = PlayerOptions::default();
        options.metadata = json!({"brain_model":"native","checkpoint_mix":1.0,"generation":0,
            "tau_reseeded":false,"slow_kinematics":"canonical"});
        // The diagnostics report may carry rows for other beings/generations too;
        // only the one matching selected_being and the replay's generation counts.
        options.diagnostics = Some(
            json!({"meta":{"selected_being":3,"observation_schema":2},"observations":[
                {"being":2,"generation":0,"genome_file":"observed_genome_gen0000.bin"},
                {"being":3,"generation":1,"genome_file":"wrong_generation.bin"},
                {"being":3,"generation":0,"genome_file":"observed_genome_gen0000.bin"}
            ]}),
        );
        let attached = options
            .attach_observed_genome(&replay_path)
            .expect("attach");
        assert!(attached);
        let native = options.native_brain.as_ref().expect("native_brain set");
        assert_eq!(native["traceBeing"], 3);
        assert_eq!(native["checkpoint"], "observed_genome_gen0000.bin");
        assert!(
            native["association"]
                .as_str()
                .unwrap()
                .contains("no external checkpoint_sha256 to cross-check")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn attach_observed_genome_rejects_wrong_observation_schema_and_path_traversal() {
        let mut options = PlayerOptions::default();
        options.metadata = json!({"brain_model":"native","generation":1});
        options.diagnostics = Some(json!({"meta":{"selected_being":4,"observation_schema":1},
            "observations":[{"being":4,"generation":1,"genome_file":"../outside.bin"}]}));
        let error = options
            .attach_observed_genome(Path::new("/tmp/gen0001.replay"))
            .unwrap_err();
        assert!(error.to_string().contains("observation_schema"), "{error}");

        options.diagnostics = Some(json!({"meta":{"selected_being":4,"observation_schema":2},
            "observations":[{"being":4,"generation":1,"genome_file":"../outside.bin"}]}));
        let error = options
            .attach_observed_genome(Path::new("/tmp/gen0001.replay"))
            .unwrap_err();
        assert!(
            error.to_string().contains("single relative file name"),
            "{error}"
        );
    }
}
