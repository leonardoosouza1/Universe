//! Validate a measured neural trace before attaching it to a body's replay.

use super::{invalid, ReplayFrame};
use serde_json::Value;
use std::io;

pub(super) fn validate(data: &Value, bodies: &[Vec<ReplayFrame>], meta: &Value) -> io::Result<()> {
    let path = meta["brain_trace_path"]
        .as_str()
        .ok_or_else(|| invalid("replay does not identify a recorded brain trace"))?;
    if data["tracePath"].as_str() != Some(path) {
        return Err(invalid(
            "brain trace path does not match the replay recording",
        ));
    }
    if meta["raw_intents_recorded"].as_bool() != Some(true) {
        return Err(invalid(
            "raw commands are required to verify brain/body alignment",
        ));
    }
    let n = data["n"]
        .as_u64()
        .filter(|n| *n > 0)
        .ok_or_else(|| invalid("missing brain neuron count"))? as usize;
    let being = data["trace"]["being"]
        .as_u64()
        .ok_or_else(|| invalid("missing trace being"))? as usize;
    let body = bodies
        .get(being)
        .ok_or_else(|| invalid("trace being is outside replay population"))?;
    let stride = data["traceStepsPerTick"]
        .as_u64()
        .filter(|n| *n > 0)
        .ok_or_else(|| invalid("missing neural clock"))?;
    let frames = data["trace"]["frames"]
        .as_array()
        .filter(|v| !v.is_empty())
        .ok_or_else(|| invalid("empty brain trace"))?;
    let mut previous = 0;
    for frame in frames {
        let step = frame["step"]
            .as_u64()
            .ok_or_else(|| invalid("invalid neural step"))?;
        if step <= previous || step % stride != 0 {
            return Err(invalid("nonmonotonic or misaligned neural clock"));
        }
        previous = step;
        let tick = u32::try_from(step / stride - 1).map_err(invalid)?;
        let index = body
            .binary_search_by_key(&tick, |f| f.tick)
            .map_err(|_| invalid(format!("no body frame for neural tick {tick}")))?;
        for key in ["v", "spikes"] {
            let values = frame[key]
                .as_array()
                .ok_or_else(|| invalid("missing neuron state"))?;
            if values.len() != n
                || values
                    .iter()
                    .any(|v| v.as_f64().is_none_or(|v| !v.is_finite()))
            {
                return Err(invalid("neuron state shape or finite-value mismatch"));
            }
        }
        let commands = frame["commands"]
            .as_array()
            .filter(|v| v.len() == 3)
            .ok_or_else(|| invalid("missing trace motor commands"))?;
        let recorded = &body[index];
        for (actual, expected) in commands.iter().zip([
            recorded.intent_thr,
            recorded.intent_str,
            recorded.intent_brk,
        ]) {
            let actual = actual
                .as_f64()
                .ok_or_else(|| invalid("invalid trace command"))?;
            if !actual.is_finite()
                || (actual - expected as f64).abs() > 1e-5 * (expected.abs() as f64).max(1.0)
            {
                return Err(invalid(format!(
                    "neural/body motor mismatch at tick {tick}"
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn cross_run_or_shifted_activity_is_rejected() {
        let meta = json!({"brain_trace_path":"/recorded/brain", "raw_intents_recorded":true});
        let body = vec![vec![ReplayFrame {
            tick: 64,
            intent_str: 0.5,
            ..Default::default()
        }]];
        let mut data = json!({"tracePath":"/recorded/brain", "n":1,"traceStepsPerTick":156,
            "trace":{"being":0,"frames":[{"step":10140,"v":[-52.0],"spikes":[0],"commands":[0.0,0.5,0.0]}]}});
        assert!(validate(&data, &body, &meta).is_ok());
        data["tracePath"] = json!("/other/brain");
        assert!(validate(&data, &body, &meta).is_err());
        data["tracePath"] = json!("/recorded/brain");
        data["trace"]["frames"][0]["commands"][1] = json!(0.4);
        assert!(validate(&data, &body, &meta).is_err());
        data["trace"]["frames"][0]["commands"][1] = json!(0.5);
        data["trace"]["frames"][0]["step"] = json!(9984);
        assert!(validate(&data, &body, &meta).is_err());
    }
}
