//! Honest, compact description of the native controller stored in a GPU-evolve checkpoint.
//!
//! Coordinates are deliberately left to the browser: they are a schematic functional
//! layout, not anatomy. The matrices below are the exact loaded checkpoint parameters.
//! Runtime gates, fatigue and pruning can change effective signal flow, so the payload
//! calls them parameters rather than observed propagation.

use serde_json::{Value, json};
use std::io;
use std::path::Path;

use crate::config::iara::brain::{L1_NEURONS, L2_NEURONS, SENSORY_WAVE_SIZE};
use crate::gpu::evolution::{N_L1_WEIGHTS, N_L2_WEIGHTS};
use crate::gpu::genome_io::load_genome;

const SLOW_INPUTS: usize = 48;
const SLOW_CFC: usize = 64;
const SLOW_HIDDEN: usize = 16;
const SLOW_OUTPUTS: usize = 3;
const SLOW_D1_WEIGHTS: usize = SLOW_CFC * SLOW_HIDDEN;
const SLOW_D1_BIAS: usize = SLOW_HIDDEN;
const SLOW_D2_WEIGHTS: usize = SLOW_HIDDEN * SLOW_OUTPUTS;
const SLOW_D2_BIAS: usize = SLOW_OUTPUTS;
const SLOW_WEIGHTS: usize = SLOW_D1_WEIGHTS + SLOW_D1_BIAS + SLOW_D2_WEIGHTS + SLOW_D2_BIAS;

/// Exact `CHANNEL_MAP` from `gpu/slow_gather.wgsl`. Negative entries are real zero
/// padding: those CPU-only body/memory signals do not exist in `gpu_evolve`.
const SLOW_GATHER_CHANNEL_MAP: [i32; SLOW_INPUTS] = [
    -1, -1, -1, -1, 33, -1, -1, 32, -1, -1, -1, -1, -1, -1, 36, 37, 38, 39, 40, 41, 42, 43, -1, -1,
    -1, -1, -1, 78, 79, -1, 80, 82, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
    140, 141, 142, 143,
];

fn slow_gather_map(mode: &str) -> io::Result<([i32; SLOW_INPUTS], &'static str)> {
    let mut map = SLOW_GATHER_CHANNEL_MAP;
    match mode {
        "legacy" => Ok((map, "historical mapping: slots 4/7 read wave 33/32")),
        "canonical" => {
            map[4] = 65;
            map[7] = 64;
            Ok((map, "canonical kinematics: slots 4/7 read wave 65/64"))
        }
        _ => Err(invalid(format!(
            "unknown metadata.slow_kinematics value {mode:?}; expected legacy or canonical"
        ))),
    }
}

fn invalid(message: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

fn finite(values: &[f32], name: &str) -> io::Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(invalid(format!(
            "native brain {name} contains nonfinite values"
        )))
    }
}

pub(super) fn validate_association(metadata: &Value) -> io::Result<()> {
    if metadata
        .get("brain_model")
        .and_then(Value::as_str)
        .is_some_and(|model| model != "native")
    {
        return Err(invalid(
            "native topology cannot be attached to a non-native controller replay",
        ));
    }
    if metadata.get("checkpoint_mix").and_then(Value::as_f64) != Some(1.0) {
        return Err(invalid(
            "native topology requires metadata.checkpoint_mix=1; a checkpoint hash alone does not associate evolved beings with that genome",
        ));
    }
    if metadata.get("generation").and_then(Value::as_u64) != Some(0) {
        return Err(invalid(
            "native topology requires the checkpoint-clone generation 0; later evolved generations need their own recorded genome identity",
        ));
    }
    if metadata.get("tau_reseeded").and_then(Value::as_bool) == Some(true) {
        return Err(invalid(
            "native topology cannot use checkpoint decay rates when metadata says tau was reseeded",
        ));
    }
    Ok(())
}

fn sha256(bytes: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut data = bytes.to_vec();
    let bit_len = (data.len() as u64) * 8;
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&bit_len.to_be_bytes());
    let mut h = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    for chunk in data.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes(word.try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, value) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *slot = slot.wrapping_add(value);
        }
    }
    h.iter().map(|value| format!("{value:08x}")).collect()
}

pub(super) fn from_checkpoint(
    path: &Path,
    trace_being: usize,
    expected_hash: &str,
    slow_kinematics: &str,
) -> io::Result<Value> {
    let bytes = std::fs::read(path)?;
    let actual_hash = sha256(&bytes);
    if expected_hash.len() != 64
        || !expected_hash.bytes().all(|byte| byte.is_ascii_hexdigit())
        || !actual_hash.eq_ignore_ascii_case(expected_hash)
    {
        return Err(invalid(format!(
            "native genome hash {actual_hash} does not match replay checkpoint_sha256 {expected_hash}"
        )));
    }
    let path_text = path
        .to_str()
        .ok_or_else(|| invalid("native genome path is not valid UTF-8"))?;
    let genome = load_genome(path_text)
        .ok_or_else(|| invalid(format!("could not load native genome {}", path.display())))?;
    if genome.l1.len() != N_L1_WEIGHTS || genome.l2.len() != N_L2_WEIGHTS {
        return Err(invalid(
            "native genome fast matrices have unexpected dimensions",
        ));
    }
    if genome.tau.len() != L1_NEURONS || genome.slow.len() != SLOW_WEIGHTS {
        return Err(invalid(
            "native genome recurrent/slow blocks have unexpected dimensions",
        ));
    }
    finite(&genome.l1, "L1 weights")?;
    finite(&genome.l2, "L2 weights")?;
    finite(&genome.tau, "CfC decay bank")?;
    finite(&genome.slow, "slow weights")?;
    if genome.neuron_alpha.len() != L1_NEURONS + L2_NEURONS
        || genome.neuron_period_l1.len() != L1_NEURONS
        || genome.neuron_threshold_l1.len() != L1_NEURONS
        || genome.neuron_activation.len() != L1_NEURONS + L2_NEURONS
    {
        return Err(invalid(
            "native genome neuron-mechanism blocks have unexpected dimensions",
        ));
    }
    finite(&genome.neuron_alpha, "neuron memory coefficients")?;
    finite(&genome.neuron_period_l1, "L1 periods")?;
    finite(&genome.neuron_threshold_l1, "L1 thresholds")?;
    finite(&genome.neuron_activation, "neuron activation identifiers")?;

    let d1_weights = &genome.slow[..SLOW_D1_WEIGHTS];
    let d1_bias = &genome.slow[SLOW_D1_WEIGHTS..SLOW_D1_WEIGHTS + SLOW_D1_BIAS];
    let d2_begin = SLOW_D1_WEIGHTS + SLOW_D1_BIAS;
    let d2_weights = &genome.slow[d2_begin..d2_begin + SLOW_D2_WEIGHTS];
    let d2_bias = &genome.slow[d2_begin + SLOW_D2_WEIGHTS..];
    let runtime_knots = crate::evolution::dna_genesis::default_kan_knots();
    let sensory_knots = crate::kan::KanAxiomFilter::new().spline_coeffs;
    let (slow_map, slow_map_description) = slow_gather_map(slow_kinematics)?;
    let checkpoint_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("checkpoint");

    Ok(json!({
        "schemaVersion": 1,
        "kind": "native-controller",
        "checkpoint": checkpoint_name,
        "checkpointSha256": actual_hash,
        "traceBeing": trace_being,
        "association": "checkpoint bytes verified against replay metadata.checkpoint_sha256; traceBeing explicitly selected by the exporter",
        "coordinateSemantics": "schematic functional layout; positions are not anatomical measurements",
        "weightSemantics": "exact loaded checkpoint parameters; a stored edge is not proof of effective signal flow",
        "activitySemantics": "sampled measurements only; no propagation or interpolation",
        "architecture": {
            "wave": SENSORY_WAVE_SIZE,
            "sensoryKan": SENSORY_WAVE_SIZE,
            "inputCfc": L1_NEURONS,
            "l1": L1_NEURONS,
            "l2": L2_NEURONS,
            "motorHead": 3,
            "slowInput": SLOW_INPUTS,
            "slowCfc": SLOW_CFC,
            "slowHidden": SLOW_HIDDEN,
            "slowOutput": SLOW_OUTPUTS
        },
        "fast": {
            "l1Weights": genome.l1,
            "l2Weights": genome.l2,
            "tau": genome.tau,
            "sensoryKnots": sensory_knots,
            "sensoryKnotsSource": "fixed runtime identity coefficients shared by all 192 sensory channels; not a genome field",
            "l1Alpha": &genome.neuron_alpha[..L1_NEURONS],
            "l1Period": genome.neuron_period_l1,
            "l1Threshold": genome.neuron_threshold_l1,
            "l1Activation": &genome.neuron_activation[..L1_NEURONS],
            "l2Alpha": &genome.neuron_alpha[L1_NEURONS..],
            "l2Activation": &genome.neuron_activation[L1_NEURONS..],
            "neuronMechanismSemantics": "checkpoint parameters; run control flags can force memory, period, threshold or activation mechanisms off",
            "outputKnots": runtime_knots,
            "outputKnotsSource": "gpu_evolve runtime default; checkpoint Genome.kan_knots is stored but inactive on this production path",
            "inputPath": "wave[0..192) -> fixed sensory KAN; channels [0..96) -> CfC -> L1 inputs [0..96), channels [96..192) bypass CfC -> L1 inputs [96..192)",
            "runtimeCaveat": "input filtering, recurrent/output fatigue, asynchronous neuron settings and optional regional gates affect effective flow"
        },
        "slow": {
            "gatherMap": slow_map.as_slice(),
            "d1Weights": d1_weights,
            "d1Bias": d1_bias,
            "d2Weights": d2_weights,
            "d2Bias": d2_bias,
            "tickInterval": 8,
            "kinematicsMode": slow_kinematics,
            "kinematicsDescription": slow_map_description,
            "mappingSource": "gpu/slow_gather.wgsl CHANNEL_MAP; -1 means an unavailable source is actually zero-filled",
            "mappingWarning": if slow_kinematics == "legacy" { "This is the executed historical channel-index mapping. Its old semantic comments are stale: current wave channels 32/33 are curvature-related, not alignment/speed (the current alignment/speed channels are 64/65). The viewer preserves the executed mapping and does not silently correct it." } else { "The explicit canonical run mode changes only slow-input slots 4/7 to the current speed/alignment channels 65/64; every other slot preserves CHANNEL_MAP." }
        },
        "measured": {
            "wave": "observation.wave[192], i8 sensor code",
            "inputCfc": "observation.fast_state[96], dimensionless recurrent state",
            "rawIntent": "observation.body.motor_trace.raw_intent[3] or legacy body.raw_motor[3], dimensionless pre-output-KAN intent",
            "afterHead": "observation.body.motor_trace.after_head[3] when recorded, dimensionless",
            "slowOutput": "observation.body.motor_trace.slow_output[3] or legacy body.slow[3], dimensionless",
            "unavailable": ["L1 activation", "L2 dimensions 3..47", "slow CfC state", "slow hidden activation"]
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slow_gather_map_matches_the_gpu_shape_and_declared_live_sources() {
        assert_eq!(SLOW_GATHER_CHANNEL_MAP.len(), 48);
        assert_eq!(
            &SLOW_GATHER_CHANNEL_MAP[32..],
            &(128..144).collect::<Vec<_>>()
        );
        assert!(SLOW_GATHER_CHANNEL_MAP.iter().all(|&source| source < 192));
        let (legacy, _) = slow_gather_map("legacy").unwrap();
        let (canonical, _) = slow_gather_map("canonical").unwrap();
        assert_eq!((legacy[4], legacy[7]), (33, 32));
        assert_eq!((canonical[4], canonical[7]), (65, 64));
        assert!(slow_gather_map("guess").is_err());
    }

    #[test]
    fn sha256_matches_the_standard_vector() {
        assert_eq!(
            sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn association_requires_an_unmodified_checkpoint_clone_population() {
        let exact = json!({"brain_model":"native","checkpoint_mix":1.0,"generation":0,"tau_reseeded":false});
        assert!(validate_association(&exact).is_ok());
        for changed in [
            json!({"brain_model":"fly_rate","checkpoint_mix":1.0,"generation":0}),
            json!({"brain_model":"native","checkpoint_mix":0.5,"generation":0}),
            json!({"brain_model":"native","checkpoint_mix":1.0,"generation":1}),
            json!({"brain_model":"native","checkpoint_mix":1.0,"generation":0,"tau_reseeded":true}),
        ] {
            assert!(validate_association(&changed).is_err());
        }
    }
}
