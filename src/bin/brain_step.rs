//! lab-ui PRD S2 — CPU-only reference forward for the fast brain (L1/L2), for the
//! Counterfactual panel (PRD §6): given a genome and a recorded (wave, CfC-state) pair
//! from a trace-v1 sample, recompute every stage through to the raw motor intent, with
//! optional per-stage/index ablation ("what if this neuron were zero").
//!
//! # Why this stops at `raw_intent` and does not continue to `after_head`/
//! `after_modulation`/`final_command`
//!
//! Those three stages are produced by `map_native_action`
//! (`src/bin/gpu_evolve/native_action.rs`), which is a pure, already-CPU, already-tested
//! function -- exactly the kind of code this task says to call, not reimplement. But it
//! lives in the `gpu_evolve` BINARY crate's own module tree, not in the `universe`
//! LIBRARY crate, so a separate binary cannot `use` it without either (a) moving it into
//! the library -- a real API change to a file with its own tests and call sites, not done
//! here without sign-off -- or (b) copy-pasting its logic, which is exactly what "do NOT
//! reimplement the brain in a second place" forbids. See `src/lab_api/DEVIATIONS.md` /
//! `Universe/docs/` note for S2. `raw_intent` is also literally the PRD's own acceptance
//! target ("the reference forward's motor intent must match the recorded `raw_intent`
//! block"), so stopping there is not a scope cut against the spec, it is the spec.
//!
//! # What IS reused, verbatim, from the same modules the parity tests use
//!
//! - `universe::layers::BitLinear::forward` -- L1, the exact struct `gpu_validate`
//!   compares against `brain.wgsl` (see `src/gpu/brain/mod.rs`'s "HOW CPU/GPU EQUIVALENCE
//!   IS TESTED" section) and the same one `src/engine.rs`'s `SovereignEngine.instinct_layer`
//!   uses.
//! - `universe::float_layers::FloatLinear::forward` -- L2, ditto (`SovereignEngine.cortex_l2`).
//! - `universe::cfc::CfCCell::quantize_state_i8` -- the exact `(tanh(state)*127).clamp(-127,127)`
//!   line `CfCCell::update` uses to turn its resident f32 state into the i8 L1 reads,
//!   extracted (this task) into its own function so this binary calls the SAME formula
//!   instead of a second copy of it. Used here for the CfC state (this binary is GIVEN
//!   the state recorded in `neural.trace`'s 'cfc' block, so it does not call
//!   `CfCCell::update` itself -- there is no wave-to-state history available offline to
//!   replay the recurrence, only the already-computed state) AND for L1's own
//!   raw-to-i8 activation step, which is the identical formula
//!   (`src/engine.rs::execute_cycle`'s `l1_act[i] = (v.tanh()*127).clamp(-127,127) as i8`
//!   repeats it inline for the same reason CfC's did -- both are one line of pure
//!   elementwise math, not a second brain).
//!
//! # Known, documented approximation: no persistent output fatigue
//!
//! `src/engine.rs::execute_cycle`'s final combination step
//! (`self.out_fatigue[j] += OUT_ALPHA*(|raw|-out_fatigue[j])*dt; adapted = sign(raw)*max(0,
//! |raw|-OUT_BETA*out_fatigue[j])`) depends on `out_fatigue`, state accumulated over many
//! prior ticks that a single offline sample does not carry. This binary is a STATELESS,
//! single-tick function -- it uses `out_fatigue=0` for every call, which makes the
//! combination step `tanh(raw*2.0)` exactly (threshold `OUT_BETA*0=0`, so
//! `sign(raw)*max(0,|raw|-0) = raw`). Any mismatch against a recorded `raw_intent` that
//! this causes is reported, not hidden -- see the `--batch` acceptance report.
//!
//! CPU-only: opens no GPU device.
//!
//! # Usage
//!   brain_step <genome.bin> <input.json> <out.json>
//!   brain_step --batch <genome.bin> <samples.json> <out.json>
//!
//! `input.json`: `{"wave":[192 i8 as int],"cfc_state":[96 f32],"dt":0.015625,
//!   "ablate":[{"stage":"l1","index":5}, ...]}` (dt unused by the stateless forward below,
//!   accepted for forward-compatibility and echoed back).
//! `samples.json`: `[{"tick":N,"wave":[...],"cfc_state":[...],"raw_intent_recorded":[t,s,b]}, ...]`

use serde::Deserialize;
use serde_json::json;
use std::io::Write as _;

use universe::cfc::CfCCell;
use universe::config::iara::brain::{L1_NEURONS, L2_NEURONS, SENSORY_WAVE_SIZE};
use universe::float_layers::FloatLinear;
use universe::gpu::genome_io::load_genome;
use universe::layers::BitLinear;

#[derive(Deserialize)]
struct Ablation {
    stage: String,
    index: usize,
}

#[derive(Deserialize)]
struct StepInput {
    wave: Vec<i32>,
    cfc_state: Vec<f32>,
    #[serde(default)]
    dt: Option<f32>,
    #[serde(default)]
    ablate: Vec<Ablation>,
}

#[derive(Deserialize)]
struct Sample {
    #[serde(default)]
    tick: Option<i64>,
    wave: Vec<i32>,
    cfc_state: Vec<f32>,
    #[serde(default)]
    raw_intent_recorded: Option<[f32; 3]>,
    #[serde(default)]
    ablate: Vec<Ablation>,
}

fn zero_at(v: &mut [f32], ablate: &[Ablation], stage: &str) {
    for a in ablate {
        if a.stage == stage {
            if let Some(slot) = v.get_mut(a.index) {
                *slot = 0.0;
            } else {
                eprintln!(
                    "brain_step: WARNING ablation index {} out of range for stage {stage:?} (len {})",
                    a.index,
                    v.len()
                );
            }
        }
    }
}

struct StepOutput {
    wave_bypass: Vec<f32>,       // wave[96..192] as recorded (i8, shown as f32 for uniformity)
    cfc_quantized: Vec<f32>,     // CfCCell::quantize_state_i8(cfc_state[i]), shown as f32
    l1_raw: Vec<f32>,            // BitLinear(l1_input) output, pre-activation
    l1_act: Vec<f32>,            // CfCCell::quantize_state_i8(l1_raw[i]), shown as f32 (this is L2's input)
    l2_raw: Vec<f32>,            // FloatLinear(l1_act) output
    final_intent: Vec<f32>,      // 96-dim combination (fatigue=0 approximation)
    raw_intent: [f32; 3],        // final_intent[0..3]
}

/// The reference forward. Composes BitLinear::forward, FloatLinear::forward and
/// CfCCell::quantize_state_i8 -- see the module doc for exactly what is reused and what
/// is a documented approximation (no persistent output fatigue).
fn forward(l1: &BitLinear, l2: &FloatLinear, wave: &[i32], cfc_state: &[f32], ablate: &[Ablation]) -> StepOutput {
    assert_eq!(wave.len(), SENSORY_WAVE_SIZE, "wave must have SENSORY_WAVE_SIZE={SENSORY_WAVE_SIZE} entries");
    assert_eq!(cfc_state.len(), L1_NEURONS, "cfc_state must have L1_NEURONS={L1_NEURONS} entries (the CfC hidden size)");

    let mut cfc_state = cfc_state.to_vec();
    zero_at(&mut cfc_state, ablate, "cfc");

    // wave[0..96] is NOT read directly: L1's first 96 input channels are the CfC-filtered
    // retina (see genome_export's 'l1' stage note and 'tau' stage note). wave[96..192]
    // bypasses the CfC and feeds L1 directly, unchanged.
    let mut wave_bypass: Vec<f32> = wave[L1_NEURONS..SENSORY_WAVE_SIZE].iter().map(|&v| v as f32).collect();
    zero_at(&mut wave_bypass, ablate, "wave");

    let cfc_quantized: Vec<f32> = cfc_state.iter().map(|&s| CfCCell::quantize_state_i8(s) as f32).collect();

    let mut l1_input_i8 = vec![0i8; SENSORY_WAVE_SIZE];
    for i in 0..L1_NEURONS {
        l1_input_i8[i] = cfc_quantized[i] as i8;
    }
    for (k, &v) in wave_bypass.iter().enumerate() {
        l1_input_i8[L1_NEURONS + k] = v as i8;
    }

    let mut l1_raw = l1.forward(&l1_input_i8);
    zero_at(&mut l1_raw, ablate, "l1");

    let l1_act_i8: Vec<i8> = l1_raw.iter().map(|&v| CfCCell::quantize_state_i8(v)).collect();
    let l1_act: Vec<f32> = l1_act_i8.iter().map(|&v| v as f32).collect();

    let mut l2_raw = l2.forward(&l1_act_i8);
    zero_at(&mut l2_raw, ablate, "l2");

    // Combination step, src/engine.rs execute_cycle's final_intent assembly, with
    // out_fatigue=0 (see module doc "Known, documented approximation").
    let mut final_intent = vec![0.0f32; L1_NEURONS];
    for j in 0..L2_NEURONS {
        final_intent[j] = (l2_raw[j] * 2.0).tanh();
    }
    for k in 0..(L1_NEURONS - L2_NEURONS) {
        final_intent[L2_NEURONS + k] = (l1_raw[L2_NEURONS + k] * 2.0).tanh();
    }
    zero_at(&mut final_intent, ablate, "intent");

    let raw_intent = [final_intent[0], final_intent[1], final_intent[2]];

    StepOutput {
        wave_bypass,
        cfc_quantized,
        l1_raw,
        l1_act,
        l2_raw,
        final_intent,
        raw_intent,
    }
}

fn load_l1_l2(genome_path: &str) -> (BitLinear, FloatLinear, String) {
    let bytes = std::fs::read(genome_path).unwrap_or_else(|e| {
        eprintln!("brain_step: cannot read {genome_path}: {e}");
        std::process::exit(1);
    });
    let genome_sha256 = {
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(&bytes);
        format!("{:x}", h.finalize())
    };
    let genome = load_genome(genome_path).unwrap_or_else(|| {
        eprintln!("brain_step: {genome_path} did not parse as a UGEN genome");
        std::process::exit(1);
    });
    let mut l1 = BitLinear::new(SENSORY_WAVE_SIZE, L1_NEURONS);
    l1.weights = genome.l1;
    let mut l2 = FloatLinear::new(L1_NEURONS, L2_NEURONS);
    l2.weights = genome.l2;
    (l1, l2, genome_sha256)
}

fn output_json(o: &StepOutput) -> serde_json::Value {
    json!({
        "wave_bypass": o.wave_bypass,
        "cfc_quantized_i8": o.cfc_quantized,
        "l1_raw": o.l1_raw,
        "l1_act_i8": o.l1_act,
        "l2_raw": o.l2_raw,
        "final_intent": o.final_intent,
        "raw_intent": o.raw_intent,
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 5 && args[1] == "--batch" {
        run_batch(&args[2], &args[3], &args[4]);
        return;
    }
    if args.len() != 4 {
        eprintln!("usage: brain_step <genome.bin> <input.json> <out.json>");
        eprintln!("       brain_step --batch <genome.bin> <samples.json> <out.json>");
        std::process::exit(2);
    }
    let (l1, l2, genome_sha256) = load_l1_l2(&args[1]);
    let input_text = std::fs::read_to_string(&args[2]).unwrap_or_else(|e| {
        eprintln!("brain_step: cannot read {}: {e}", args[2]);
        std::process::exit(1);
    });
    let input: StepInput = serde_json::from_str(&input_text).unwrap_or_else(|e| {
        eprintln!("brain_step: {} is not a valid input.json: {e}", args[2]);
        std::process::exit(1);
    });
    let out = forward(&l1, &l2, &input.wave, &input.cfc_state, &input.ablate);
    let mut v = output_json(&out);
    v["genome_sha256"] = json!(genome_sha256);
    v["dt_echo"] = json!(input.dt);
    v["ablate_echo"] = json!(input.ablate.iter().map(|a| json!({"stage": a.stage, "index": a.index})).collect::<Vec<_>>());
    write_json(&args[3], &v);
    println!("brain_step: wrote {} (raw_intent = {:?})", args[3], out.raw_intent);
}

fn run_batch(genome_path: &str, samples_path: &str, out_path: &str) {
    let (l1, l2, genome_sha256) = load_l1_l2(genome_path);
    let samples_text = std::fs::read_to_string(samples_path).unwrap_or_else(|e| {
        eprintln!("brain_step: cannot read {samples_path}: {e}");
        std::process::exit(1);
    });
    let samples: Vec<Sample> = serde_json::from_str(&samples_text).unwrap_or_else(|e| {
        eprintln!("brain_step: {samples_path} is not a valid samples.json: {e}");
        std::process::exit(1);
    });

    let mut per_sample = Vec::with_capacity(samples.len());
    let mut max_abs_err = 0.0f32;
    let mut max_abs_err_sample: Option<i64> = None;
    let mut n_compared = 0usize;

    for (i, s) in samples.iter().enumerate() {
        let out = forward(&l1, &l2, &s.wave, &s.cfc_state, &s.ablate);
        let mut entry = output_json(&out);
        entry["tick"] = json!(s.tick);
        if let Some(recorded) = s.raw_intent_recorded {
            let errs = [
                (out.raw_intent[0] - recorded[0]).abs(),
                (out.raw_intent[1] - recorded[1]).abs(),
                (out.raw_intent[2] - recorded[2]).abs(),
            ];
            let sample_max = errs.iter().cloned().fold(0.0f32, f32::max);
            entry["raw_intent_recorded"] = json!(recorded);
            entry["abs_error"] = json!(errs);
            entry["max_abs_error"] = json!(sample_max);
            n_compared += 1;
            if sample_max > max_abs_err {
                max_abs_err = sample_max;
                max_abs_err_sample = s.tick.or(Some(i as i64));
            }
        }
        per_sample.push(entry);
    }

    let report = json!({
        "genome": genome_path,
        "genome_sha256": genome_sha256,
        "n_samples": samples.len(),
        "n_compared_against_recorded_raw_intent": n_compared,
        "max_abs_error": max_abs_err,
        "max_abs_error_at_tick_or_index": max_abs_err_sample,
        "note": "max_abs_error is the largest |computed - recorded| over any of the 3 raw_intent channels, over all compared samples. See brain_step.rs module doc for the two documented approximations (no persistent output fatigue; being/genome identity not verified by this tool -- see contract/DEVIATIONS.md #2).",
        "samples": per_sample,
    });
    write_json(out_path, &report);
    println!(
        "brain_step --batch: wrote {out_path} ({} samples, {} compared, max_abs_error={max_abs_err})",
        samples.len(),
        n_compared
    );
}

fn write_json(path: &str, v: &serde_json::Value) {
    let mut f = std::fs::File::create(path).unwrap_or_else(|e| {
        eprintln!("brain_step: cannot create {path}: {e}");
        std::process::exit(1);
    });
    let text = serde_json::to_string_pretty(v).expect("serialize brain_step output");
    f.write_all(text.as_bytes()).expect("write brain_step output");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_l1_l2() -> (BitLinear, FloatLinear) {
        let l1 = BitLinear::new(SENSORY_WAVE_SIZE, L1_NEURONS);
        let l2 = FloatLinear::new(L1_NEURONS, L2_NEURONS);
        (l1, l2)
    }

    #[test]
    fn all_zero_genome_and_input_gives_zero_raw_intent() {
        let (l1, l2) = zero_l1_l2();
        let wave = vec![0i32; SENSORY_WAVE_SIZE];
        let cfc_state = vec![0.0f32; L1_NEURONS];
        let out = forward(&l1, &l2, &wave, &cfc_state, &[]);
        assert_eq!(out.raw_intent, [0.0, 0.0, 0.0]);
        assert!(out.final_intent.iter().all(|&v| v == 0.0));
        assert_eq!(out.l1_raw.len(), L1_NEURONS);
        assert_eq!(out.l2_raw.len(), L2_NEURONS);
    }

    #[test]
    fn quantize_state_i8_matches_tanh_saturation() {
        // A large positive state saturates to +127, a large negative one to -127 --
        // the same clamp CfCCell::update applies to its own output.
        assert_eq!(CfCCell::quantize_state_i8(10.0), 127);
        assert_eq!(CfCCell::quantize_state_i8(-10.0), -127);
        assert_eq!(CfCCell::quantize_state_i8(0.0), 0);
    }

    #[test]
    fn a_single_l1_weight_reaches_raw_intent_channel_0() {
        // final_intent[0] is on the L2 side (channel 0 < L2_NEURONS), so the path under
        // test is: wave bypass channel 96 -> L1 neuron 0 -> L2 neuron 0 -> raw_intent[0].
        let (mut l1, mut l2) = zero_l1_l2();
        // L1 neuron 0's weight on bypass channel wave[96] (l1 input index L1_NEURONS+0).
        l1.weights[0 * SENSORY_WAVE_SIZE + L1_NEURONS] = 1.0;
        // L2 neuron 0's weight on L1 neuron 0's activation.
        l2.weights[0 * L1_NEURONS + 0] = 1.0;

        let mut wave = vec![0i32; SENSORY_WAVE_SIZE];
        wave[L1_NEURONS] = 127; // maximum bypass input
        let cfc_state = vec![0.0f32; L1_NEURONS];

        let out = forward(&l1, &l2, &wave, &cfc_state, &[]);
        // l1_raw[0] = weights[0*192+96] * wave[96] / 127 = 1.0 * 127/127 = 1.0
        assert!((out.l1_raw[0] - 1.0).abs() < 1e-5, "l1_raw[0]={}", out.l1_raw[0]);
        assert!(out.raw_intent[0] > 0.0, "raw_intent[0] should be positive, got {}", out.raw_intent[0]);
        // Every other raw_intent channel stays exactly 0 -- no other weight is nonzero.
        assert_eq!(out.raw_intent[1], 0.0);
        assert_eq!(out.raw_intent[2], 0.0);
    }

    #[test]
    fn ablating_the_only_active_l1_neuron_zeroes_its_effect() {
        let (mut l1, mut l2) = zero_l1_l2();
        l1.weights[0 * SENSORY_WAVE_SIZE + L1_NEURONS] = 1.0;
        l2.weights[0 * L1_NEURONS + 0] = 1.0;
        let mut wave = vec![0i32; SENSORY_WAVE_SIZE];
        wave[L1_NEURONS] = 127;
        let cfc_state = vec![0.0f32; L1_NEURONS];

        let unablated = forward(&l1, &l2, &wave, &cfc_state, &[]);
        assert!(unablated.raw_intent[0] > 0.0);

        let ablate = vec![Ablation { stage: "l1".to_string(), index: 0 }];
        let ablated = forward(&l1, &l2, &wave, &cfc_state, &ablate);
        assert_eq!(ablated.raw_intent, [0.0, 0.0, 0.0], "zeroing the only active L1 neuron must zero every downstream output");
    }

    #[test]
    fn ablating_a_cfc_channel_removes_it_from_the_l1_input() {
        let (mut l1, l2) = zero_l1_l2();
        // L1 neuron 0's weight on CfC-fed input channel 5 (not a bypass channel).
        l1.weights[0 * SENSORY_WAVE_SIZE + 5] = 1.0;
        let wave = vec![0i32; SENSORY_WAVE_SIZE];
        let mut cfc_state = vec![0.0f32; L1_NEURONS];
        cfc_state[5] = 5.0; // saturates quantize_state_i8 to +127

        let unablated = forward(&l1, &l2, &wave, &cfc_state, &[]);
        assert!(unablated.l1_raw[0] > 0.0, "l1_raw[0]={}", unablated.l1_raw[0]);

        let ablate = vec![Ablation { stage: "cfc".to_string(), index: 5 }];
        let ablated = forward(&l1, &l2, &wave, &cfc_state, &ablate);
        assert_eq!(ablated.l1_raw[0], 0.0);
    }

    #[test]
    fn out_of_range_ablation_index_warns_and_does_not_panic() {
        let (l1, l2) = zero_l1_l2();
        let wave = vec![0i32; SENSORY_WAVE_SIZE];
        let cfc_state = vec![0.0f32; L1_NEURONS];
        let ablate = vec![Ablation { stage: "l1".to_string(), index: 999_999 }];
        // Must not panic -- zero_at() checks bounds with get_mut and only warns.
        let out = forward(&l1, &l2, &wave, &cfc_state, &ablate);
        assert_eq!(out.raw_intent, [0.0, 0.0, 0.0]);
    }
}
