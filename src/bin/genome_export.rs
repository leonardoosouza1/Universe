//! lab-ui PRD S2 — exports a `.bin` genome (`universe::gpu::genome_io`, UGEN v2/v3/v4) as
//! `weights.json`: one entry per genome block, in the exact order `genome_io.rs` reads
//! and writes them (the one place the layout is documented as canonical -- see that
//! module's header, "DO NOT reorder Genome's fields or the write order").
//!
//! This binary does NOT reimplement the genome parser: it calls
//! `universe::gpu::genome_io::load_genome`, the same function every other reader in this
//! repository uses (`genoma_view`, `genoma_dump`, `replay::player_native`).
//!
//! Usage:
//!   genome_export <genome.bin> <out.json>
//!
//! CPU-only: opens no GPU device, reads one file, writes one file.

use serde_json::json;
use sha2::{Digest, Sha256};
use std::io::Write as _;

use universe::config::iara::brain::{L1_NEURONS, L2_NEURONS, SENSORY_WAVE_SIZE};
use universe::gpu::evolution::Genome;
use universe::gpu::genome_io::load_genome;
use universe::kan::{KAN_OUT_KNOTS, KanAxiomFilter};

const SENSORY_KAN_KNOTS: usize = 16;

const SLOW_CFC: usize = 64;
const SLOW_HIDDEN: usize = 16;
const SLOW_OUTPUTS: usize = 3;
const SLOW_D1_WEIGHTS: usize = SLOW_CFC * SLOW_HIDDEN; // 1024
const SLOW_D1_BIAS: usize = SLOW_HIDDEN; // 16
const SLOW_D2_WEIGHTS: usize = SLOW_HIDDEN * SLOW_OUTPUTS; // 48
const SLOW_D2_BIAS: usize = SLOW_OUTPUTS; // 3

struct Stage {
    name: &'static str,
    shape: Vec<usize>,
    dtype: &'static str,
    scale: f64,
    matrix: Vec<f32>,
    note: &'static str,
}

fn stage_flow_metadata(
    name: &str,
) -> (
    Option<&'static str>,
    Option<&'static str>,
    Option<&'static str>,
) {
    match name {
        "sensory_kan" => (
            Some("wave"),
            Some("sensory_kan"),
            Some("elementwise_spline"),
        ),
        "l1" => (Some("l1_input"), Some("l1_matmul"), Some("dense")),
        "l2" => (Some("l1_to_l2"), Some("l2_matmul"), Some("dense")),
        "motor_head" => (
            Some("raw_intent"),
            Some("after_head"),
            Some("elementwise_spline"),
        ),
        _ => (None, None, None),
    }
}

/// Builds the ordered stage list from a loaded genome. Pure function (no I/O) so both
/// `main()` and the unit tests below exercise the exact same slicing logic -- a genome
/// layout bug here would otherwise only be caught by eyeballing weights.json output.
fn build_stages(genome: &Genome) -> Vec<Stage> {
    // Genome-layout invariants this export depends on. If any of these fire, the genome
    // layout changed and this exporter (and DEVIATIONS.md) must be updated together --
    // exactly the rule genome_io.rs itself states for save_genome/load_genome.
    assert_eq!(
        genome.l1.len(),
        L1_NEURONS * SENSORY_WAVE_SIZE,
        "L1 weight count changed"
    );
    assert_eq!(
        genome.l2.len(),
        L2_NEURONS * L1_NEURONS,
        "L2 weight count changed"
    );
    let n_hip = genome.hip.len();
    assert_eq!(
        genome.slow.len(),
        SLOW_D1_WEIGHTS + SLOW_D1_BIAS + SLOW_D2_WEIGHTS + SLOW_D2_BIAS,
        "slow-brain weight count changed"
    );

    let d1_w = genome.slow[0..SLOW_D1_WEIGHTS].to_vec();
    let d1_b = genome.slow[SLOW_D1_WEIGHTS..SLOW_D1_WEIGHTS + SLOW_D1_BIAS].to_vec();
    let d2_begin = SLOW_D1_WEIGHTS + SLOW_D1_BIAS;
    let d2_w = genome.slow[d2_begin..d2_begin + SLOW_D2_WEIGHTS].to_vec();
    let d2_b = genome.slow[d2_begin + SLOW_D2_WEIGHTS..].to_vec();
    let kan_flat: Vec<f32> = genome
        .kan_knots
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    let sensory_kan_coefficients = KanAxiomFilter::new().spline_coeffs;
    let sensory_kan: Vec<f32> = (0..SENSORY_WAVE_SIZE)
        .flat_map(|_| sensory_kan_coefficients)
        .collect();
    // gpu_evolve passes this runtime-global head to every being.  Genome.kan_knots is
    // retained below as stored DNA, but it is not the head that produced the replay.
    let runtime_motor_head = universe::evolution::dna_genesis::default_kan_knots();
    let motor_head: Vec<f32> = runtime_motor_head.iter().flatten().copied().collect();
    let (alpha_l1, alpha_l2) = genome.neuron_alpha.split_at(L1_NEURONS);
    let (act_l1, act_l2) = genome.neuron_activation.split_at(L1_NEURONS);

    vec![
        Stage {
            name: "sensory_kan",
            shape: vec![SENSORY_WAVE_SIZE, SENSORY_KAN_KNOTS],
            dtype: "f32",
            scale: 1.0,
            matrix: sensory_kan,
            note: "Runtime sensory KAN coefficients, one 16-knot spline per wave channel. The production filter is currently fixed to identity (all coefficients 1.0), shared by every being and absent from Genome; exported here because these are the exact runtime values used by the replay.",
        },
        Stage {
            name: "l1",
            shape: vec![L1_NEURONS, SENSORY_WAVE_SIZE],
            dtype: "f32",
            scale: 1.0,
            matrix: genome.l1.clone(),
            note: "BitLinear L1, wave[192] -> 96 neurons (perception -> KAN -> CfC -> L1 -> L2 -> motor). Row-major [neuron][input_channel]: weights[neuron*192+channel] (src/layers.rs BitLinear::forward). Input channels 0..96 are the CfC-filtered retina (see 'tau'), 96..192 bypass the CfC.",
        },
        Stage {
            name: "l2",
            shape: vec![L2_NEURONS, L1_NEURONS],
            dtype: "f32",
            scale: 1.0,
            matrix: genome.l2.clone(),
            note: "FloatLinear L2, L1-output[96] -> 48. Row-major [neuron][input]: weights[neuron*96+input] (src/float_layers.rs FloatLinear::forward). Only L2 outputs 0..48 feed the motor chain; L1 outputs 48..96 bypass L2 directly (src/engine.rs execute_cycle, final_intent assembly).",
        },
        Stage {
            name: "motor_head",
            shape: vec![3, KAN_OUT_KNOTS],
            dtype: "f32",
            scale: 1.0,
            matrix: motor_head,
            note: "Runtime output KAN head used by gpu_evolve for this replay: one 12-knot spline per [throttle,steer,brake] channel. It is an element-wise nonlinear transfer raw_intent[i] -> after_head[i], not a dense 3x3 matrix.",
        },
        Stage {
            name: "kan_knots",
            shape: vec![3, KAN_OUT_KNOTS],
            dtype: "f32",
            scale: 1.0,
            matrix: kan_flat,
            note: "Stored motor spline DNA. Consult motor_head_source and runtime_motor_head to distinguish inheritance from the applied head; linear mode does not apply these knots.",
        },
        Stage {
            name: "slow_d1_weights",
            shape: vec![SLOW_HIDDEN, SLOW_CFC],
            dtype: "f32",
            scale: 1.0,
            matrix: d1_w,
            note: "Slow-brain Dense1, CfC(64) -> 16, row-major [out][in] (src/engine.rs SlowBrain, src/dense_layer.rs DenseLayer::forward: out[j] = sum_i w[j*64+i]*in[i] + bias[j]). Followed by ReLU.",
        },
        Stage {
            name: "slow_d1_bias",
            shape: vec![SLOW_HIDDEN],
            dtype: "f32",
            scale: 1.0,
            matrix: d1_b,
            note: "Slow-brain Dense1 bias, 16 values.",
        },
        Stage {
            name: "slow_d2_weights",
            shape: vec![SLOW_OUTPUTS, SLOW_HIDDEN],
            dtype: "f32",
            scale: 1.0,
            matrix: d2_w,
            note: "Slow-brain Dense2, 16 -> 3 [throttle,steer,brake], row-major [out][in]. Followed by tanh -> SlowBrain.output, read by the motor mapper as 'slow_output'.",
        },
        Stage {
            name: "slow_d2_bias",
            shape: vec![SLOW_OUTPUTS],
            dtype: "f32",
            scale: 1.0,
            matrix: d2_b,
            note: "Slow-brain Dense2 bias, 3 values.",
        },
        Stage {
            name: "tau",
            shape: vec![L1_NEURONS],
            dtype: "f32",
            scale: 1.0,
            matrix: genome.tau.clone(),
            note: "NOT a weight matrix: per-channel CfC decay rate d (s^-1), tau=1/d (src/cfc.rs CfCCell.decay_rates). Filters wave channels 0..96 before they reach L1; this is the genome's PARAMETER for that filter, not the runtime CfC state (the runtime state is neural.trace's 'cfc' block, produced per tick, not stored in the genome).",
        },
        Stage {
            name: "sensors",
            shape: vec![genome.sensors.len()],
            dtype: "f32",
            scale: 1.0,
            matrix: genome.sensors.clone(),
            note: "NOT a weight matrix: heritable sensor-bank genes (SENSORS PHASE step 1). Semantics and ranges: src/gpu/evolution.rs SENSOR_DEFAULTS/SENSOR_RANGES, in the same gene order.",
        },
        Stage {
            name: "neuron_alpha_l1",
            shape: vec![L1_NEURONS],
            dtype: "f32",
            scale: 1.0,
            matrix: alpha_l1.to_vec(),
            note: "NOT a weight matrix: P1.7 per-L1-neuron EMA memory coefficient. Disabled genome-wide when --no-neuron-memory was passed for this run (see manifest.json 'disabled_flags').",
        },
        Stage {
            name: "neuron_alpha_l2",
            shape: vec![L2_NEURONS],
            dtype: "f32",
            scale: 1.0,
            matrix: alpha_l2.to_vec(),
            note: "NOT a weight matrix: P1.7 per-L2-neuron EMA memory coefficient. Same on/off flag as neuron_alpha_l1.",
        },
        Stage {
            name: "neuron_period_l1",
            shape: vec![L1_NEURONS],
            dtype: "f32",
            scale: 1.0,
            matrix: genome.neuron_period_l1.clone(),
            note: "NOT a weight matrix: P1.7 per-L1-neuron asynchronous firing period. Disabled genome-wide when --no-neuron-threshold was passed (see manifest.json 'disabled_flags').",
        },
        Stage {
            name: "neuron_threshold_l1",
            shape: vec![L1_NEURONS],
            dtype: "f32",
            scale: 1.0,
            matrix: genome.neuron_threshold_l1.clone(),
            note: "NOT a weight matrix: P1.7 per-L1-neuron firing threshold. Same on/off flag as neuron_period_l1.",
        },
        Stage {
            name: "neuron_activation_l1",
            shape: vec![L1_NEURONS],
            dtype: "f32",
            scale: 1.0,
            matrix: act_l1.to_vec(),
            note: "NOT a weight matrix: P1.7 per-L1-neuron heterogeneous activation identifier. Disabled genome-wide when --no-heterogeneous-activation was passed (see manifest.json 'disabled_flags').",
        },
        Stage {
            name: "neuron_activation_l2",
            shape: vec![L2_NEURONS],
            dtype: "f32",
            scale: 1.0,
            matrix: act_l2.to_vec(),
            note: "NOT a weight matrix: P1.7 per-L2-neuron heterogeneous activation identifier. Same on/off flag as neuron_activation_l1.",
        },
        Stage {
            name: "gates_w",
            shape: vec![genome.gates_w.len()],
            dtype: "f32",
            scale: 1.0,
            matrix: genome.gates_w.clone(),
            note: "NOT a weight matrix: M4 conductor per-region gate genes, 16 values, all-zero means neutral (gate multiplier 1.0 everywhere, no --neuromod-regions effect).",
        },
        Stage {
            name: "hip",
            shape: vec![n_hip],
            dtype: "f32",
            scale: 1.0,
            matrix: genome.hip.clone(),
            note: "NOT a weight matrix: C1 sector-hippocampus genes (src/gpu/hippocampus.rs). All-zero means no hippocampus (bias [0,0,0], i.e. no effect on after_modulation).",
        },
        Stage {
            name: "fly",
            shape: vec![genome.fly.len()],
            dtype: "f32",
            scale: 1.0,
            matrix: genome.fly.clone(),
            note: "NOT a weight matrix: FLY MAP connectome-selection genes (docs/FLY-MAP-BRAIN.md), length 200+n_types when the run used --fly-map, empty otherwise. Opaque without the run's own fly-map file -- this export does not attempt to interpret it.",
        },
    ]
}

fn stages_to_json(stages: &[Stage]) -> (Vec<&'static str>, Vec<serde_json::Value>) {
    let brain_flow: Vec<&str> = stages.iter().map(|s| s.name).collect();
    let stages_json: Vec<serde_json::Value> = stages
        .iter()
        .map(|s| {
            let (input, output, connectivity) = stage_flow_metadata(s.name);
            json!({
                "name": s.name,
                "shape": s.shape,
                "dtype": s.dtype,
                "scale": s.scale,
                "matrix": s.matrix,
                "note": s.note,
                "input": input,
                "output": output,
                "connectivity": connectivity,
            })
        })
        .collect();
    (brain_flow, stages_json)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 && args.len() != 4 {
        eprintln!("usage: genome_export <genome.bin> <out.json> [generation.meta.json]");
        std::process::exit(2);
    }
    let genome_path = &args[1];
    let out_path = &args[2];

    let bytes = match std::fs::read(genome_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("genome_export: cannot read {genome_path}: {e}");
            std::process::exit(1);
        }
    };
    let genome_sha256 = {
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        format!("{:x}", hasher.finalize())
    };

    let genome = match load_genome(genome_path) {
        Some(g) => g,
        None => {
            eprintln!(
                "genome_export: {genome_path} did not parse as a UGEN genome (too short, or truncated -- see universe::gpu::genome_io)"
            );
            std::process::exit(1);
        }
    };

    let metadata: serde_json::Value = if let Some(path) = args.get(3) {
        serde_json::from_slice(&std::fs::read(path).expect("read generation metadata"))
            .expect("parse generation metadata")
    } else {
        serde_json::Value::Null
    };
    let inherited_head = metadata["body_evolution_enabled"] == true
        || metadata["ecology"] == true
        || metadata["lifetime_flags"].as_u64().unwrap_or(0) != 0;
    let mut stages = build_stages(&genome);
    if inherited_head {
        let head = stages.iter_mut().find(|s| s.name == "motor_head").unwrap();
        head.matrix = genome.kan_knots.iter().flatten().copied().collect();
        head.note = "Inherited per-being motor spline; inactive when runtime_motor_head is linear.";
    }
    let (brain_flow, stages_json) = stages_to_json(&stages);

    let out = json!({
        "schema_version": 2,
        "source_genome": genome_path,
        "runtime_motor_head": metadata["motor_head"],
        "motor_head_source": if inherited_head { "inherited" } else { "historical_shared_default" },
        "hereditary": {"body": genome.body.0, "organs": genome.organs.0, "lifetime": genome.lifetime.0},
        "genome_sha256": genome_sha256,
        "brain_flow": brain_flow,
        "brain_flow_note": "Exported block order is not kernel dispatch order. Sensory KAN is runtime identity; l1/l2 are inherited matrices. motor_head_source identifies shared versus inherited knots, and runtime_motor_head identifies whether the spline was applied. Acquired lifetime weights are recorded separately in neural.trace.",
        "dimensions": {
            "wave": SENSORY_WAVE_SIZE,
            "sensory_kan": SENSORY_WAVE_SIZE,
            "l1_input": SENSORY_WAVE_SIZE,
            "l1": L1_NEURONS,
            "l2": L2_NEURONS,
            "raw_intent": 3,
            "after_head": 3
        },
        "stages": stages_json,
    });

    let mut f = match std::fs::File::create(out_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("genome_export: cannot create {out_path}: {e}");
            std::process::exit(1);
        }
    };
    let text = serde_json::to_string_pretty(&out).expect("serialize weights.json");
    f.write_all(text.as_bytes()).expect("write weights.json");
    println!(
        "genome_export: wrote {out_path} ({} stages, genome sha256 {genome_sha256})",
        stages_json.len()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Genome::random` gives every block its production length (see
    /// `src/gpu/evolution.rs::Genome::random`), so this exercises the exact slicing this
    /// binary depends on without needing a `.bin` file on disk.
    #[test]
    fn stage_shapes_match_genome_field_lengths() {
        let genome = Genome::random(12345);
        let stages = build_stages(&genome);

        let by_name = |name: &str| {
            stages
                .iter()
                .find(|s| s.name == name)
                .unwrap_or_else(|| panic!("missing stage {name:?}"))
        };

        assert_eq!(by_name("l1").matrix.len(), genome.l1.len());
        assert_eq!(
            by_name("l1").shape.iter().product::<usize>(),
            genome.l1.len()
        );
        assert_eq!(by_name("l2").matrix.len(), genome.l2.len());
        assert_eq!(
            by_name("l2").shape.iter().product::<usize>(),
            genome.l2.len()
        );
        assert_eq!(by_name("kan_knots").matrix.len(), 3 * KAN_OUT_KNOTS);
        assert_eq!(
            by_name("sensory_kan").shape,
            [SENSORY_WAVE_SIZE, SENSORY_KAN_KNOTS]
        );
        assert_eq!(
            by_name("sensory_kan").matrix.len(),
            SENSORY_WAVE_SIZE * SENSORY_KAN_KNOTS
        );
        assert!(
            by_name("sensory_kan")
                .matrix
                .iter()
                .all(|value| *value == 1.0)
        );
        assert_eq!(by_name("motor_head").shape, [3, KAN_OUT_KNOTS]);
        assert_eq!(by_name("motor_head").matrix.len(), 3 * KAN_OUT_KNOTS);
        assert_eq!(
            stage_flow_metadata("sensory_kan"),
            (
                Some("wave"),
                Some("sensory_kan"),
                Some("elementwise_spline")
            )
        );
        assert_eq!(
            stage_flow_metadata("l1"),
            (Some("l1_input"), Some("l1_matmul"), Some("dense"))
        );
        assert_eq!(
            stage_flow_metadata("motor_head"),
            (
                Some("raw_intent"),
                Some("after_head"),
                Some("elementwise_spline")
            )
        );
        assert_eq!(by_name("tau").matrix, genome.tau);
        assert_eq!(by_name("sensors").matrix, genome.sensors);
        assert_eq!(by_name("gates_w").matrix, genome.gates_w);
        assert_eq!(by_name("hip").matrix, genome.hip);
        assert_eq!(by_name("fly").matrix, genome.fly);

        // Slow brain: the four sub-blocks reassemble to exactly genome.slow, in order.
        let mut slow_reassembled = Vec::new();
        slow_reassembled.extend_from_slice(&by_name("slow_d1_weights").matrix);
        slow_reassembled.extend_from_slice(&by_name("slow_d1_bias").matrix);
        slow_reassembled.extend_from_slice(&by_name("slow_d2_weights").matrix);
        slow_reassembled.extend_from_slice(&by_name("slow_d2_bias").matrix);
        assert_eq!(slow_reassembled, genome.slow);

        // neuron_alpha / neuron_activation: the l1+l2 split reassembles to the original.
        let mut alpha_reassembled = by_name("neuron_alpha_l1").matrix.clone();
        alpha_reassembled.extend_from_slice(&by_name("neuron_alpha_l2").matrix);
        assert_eq!(alpha_reassembled, genome.neuron_alpha);
        let mut act_reassembled = by_name("neuron_activation_l1").matrix.clone();
        act_reassembled.extend_from_slice(&by_name("neuron_activation_l2").matrix);
        assert_eq!(act_reassembled, genome.neuron_activation);

        assert_eq!(by_name("neuron_period_l1").matrix, genome.neuron_period_l1);
        assert_eq!(
            by_name("neuron_threshold_l1").matrix,
            genome.neuron_threshold_l1
        );
    }

    #[test]
    fn brain_flow_names_every_stage_once_in_order() {
        let genome = Genome::random(7);
        let stages = build_stages(&genome);
        let (brain_flow, stages_json) = stages_to_json(&stages);
        assert_eq!(brain_flow.len(), stages.len());
        assert_eq!(stages_json.len(), stages.len());
        assert_eq!(brain_flow[0], "sensory_kan");
        assert_eq!(brain_flow[1], "l1");
        assert_eq!(brain_flow[2], "l2");
        assert_eq!(brain_flow.last(), Some(&"fly"));
        let unique: std::collections::HashSet<_> = brain_flow.iter().collect();
        assert_eq!(
            unique.len(),
            brain_flow.len(),
            "a stage name repeats in brain_flow"
        );
    }

    /// `genome_sha256` in weights.json must equal the file's own sha256, so a consumer
    /// can tell whether a trace's weights.json still matches the genome.bin it names.
    #[test]
    fn genome_sha256_matches_a_known_vector() {
        let mut hasher = Sha256::new();
        hasher.update(b"abc");
        let digest = format!("{:x}", hasher.finalize());
        assert_eq!(
            digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
