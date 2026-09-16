//! lab-ui PRD S2 — exports a `.bin` genome (`universe::gpu::genome_io`, UGEN v2) as
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
use universe::kan::KAN_OUT_KNOTS;

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

/// Builds the ordered stage list from a loaded genome. Pure function (no I/O) so both
/// `main()` and the unit tests below exercise the exact same slicing logic -- a genome
/// layout bug here would otherwise only be caught by eyeballing weights.json output.
fn build_stages(genome: &Genome) -> Vec<Stage> {
    // Genome-layout invariants this export depends on. If any of these fire, the genome
    // layout changed and this exporter (and DEVIATIONS.md) must be updated together --
    // exactly the rule genome_io.rs itself states for save_genome/load_genome.
    assert_eq!(genome.l1.len(), L1_NEURONS * SENSORY_WAVE_SIZE, "L1 weight count changed");
    assert_eq!(genome.l2.len(), L2_NEURONS * L1_NEURONS, "L2 weight count changed");
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
    let kan_flat: Vec<f32> = genome.kan_knots.iter().flat_map(|row| row.iter().copied()).collect();
    let (alpha_l1, alpha_l2) = genome.neuron_alpha.split_at(L1_NEURONS);
    let (act_l1, act_l2) = genome.neuron_activation.split_at(L1_NEURONS);

    vec![
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
            name: "kan_knots",
            shape: vec![3, KAN_OUT_KNOTS],
            dtype: "f32",
            scale: 1.0,
            matrix: kan_flat,
            note: "KAN output head: one spline of 12 knots per motor channel [throttle,steer,brake]. universe::kan::eval_knots(knots, out_idx, x) clamps x to +/-KAN_OUT_DOMAIN, maps to a knot index and linearly interpolates. Applied to raw_intent[0..3] to produce after_head (src/bin/gpu_evolve/native_action.rs map_native_action, MotorHead::Spline).",
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
            json!({
                "name": s.name,
                "shape": s.shape,
                "dtype": s.dtype,
                "scale": s.scale,
                "matrix": s.matrix,
                "note": s.note,
            })
        })
        .collect();
    (brain_flow, stages_json)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: genome_export <genome.bin> <out.json>");
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
            eprintln!("genome_export: {genome_path} did not parse as a UGEN genome (too short, or truncated -- see universe::gpu::genome_io)");
            std::process::exit(1);
        }
    };

    let stages = build_stages(&genome);
    let (brain_flow, stages_json) = stages_to_json(&stages);

    let out = json!({
        "schema_version": 1,
        "source_genome": genome_path,
        "genome_sha256": genome_sha256,
        "brain_flow": brain_flow,
        "brain_flow_note": "Order = universe::gpu::genome_io.rs's own on-disk block order (the one place the genome layout is documented as canonical; DO NOT reorder). This is a GENOME layout order, not a per-tick KERNEL DISPATCH order: at runtime tau feeds the CfC before L1 runs, kan_knots is applied on the host AFTER L1+L2 (to raw_intent, producing after_head), and slow_* runs as a parallel GPU kernel pair (slow_gather.wgsl + slow_brain.wgsl) alongside brain.wgsl, not after it. See ARCHITECTURE.md section 2 for the actual per-tick dispatch order.",
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
    println!("genome_export: wrote {out_path} ({} stages, genome sha256 {genome_sha256})", stages_json.len());
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

        let by_name = |name: &str| stages.iter().find(|s| s.name == name).unwrap_or_else(|| panic!("missing stage {name:?}"));

        assert_eq!(by_name("l1").matrix.len(), genome.l1.len());
        assert_eq!(by_name("l1").shape.iter().product::<usize>(), genome.l1.len());
        assert_eq!(by_name("l2").matrix.len(), genome.l2.len());
        assert_eq!(by_name("l2").shape.iter().product::<usize>(), genome.l2.len());
        assert_eq!(by_name("kan_knots").matrix.len(), 3 * KAN_OUT_KNOTS);
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
        assert_eq!(by_name("neuron_threshold_l1").matrix, genome.neuron_threshold_l1);
    }

    #[test]
    fn brain_flow_names_every_stage_once_in_order() {
        let genome = Genome::random(7);
        let stages = build_stages(&genome);
        let (brain_flow, stages_json) = stages_to_json(&stages);
        assert_eq!(brain_flow.len(), stages.len());
        assert_eq!(stages_json.len(), stages.len());
        assert_eq!(brain_flow[0], "l1");
        assert_eq!(brain_flow[1], "l2");
        assert_eq!(brain_flow.last(), Some(&"fly"));
        let unique: std::collections::HashSet<_> = brain_flow.iter().collect();
        assert_eq!(unique.len(), brain_flow.len(), "a stage name repeats in brain_flow");
    }

    /// `genome_sha256` in weights.json must equal the file's own sha256, so a consumer
    /// can tell whether a trace's weights.json still matches the genome.bin it names.
    #[test]
    fn genome_sha256_matches_a_known_vector() {
        let mut hasher = Sha256::new();
        hasher.update(b"abc");
        let digest = format!("{:x}", hasher.finalize());
        assert_eq!(digest, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }
}
