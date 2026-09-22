//! Reject sensory recordings that do not belong to the replay's body and clock.
use super::{invalid, ReplayFrame};
use serde_json::Value;
use std::io;

pub(super) fn validate(data: &Value, bodies: &[Vec<ReplayFrame>], meta: &Value) -> io::Result<()> {
    if data["meta"]["tick_rate"].as_f64() != meta["tick_rate"].as_f64() {
        return Err(invalid("driving observation clock does not match replay"));
    }
    let generation = meta["generation"].as_u64().unwrap_or(0);
    let rows = data["observations"]
        .as_array()
        .ok_or_else(|| invalid("missing driving observations"))?;
    let mut matched = 0;
    for row in rows
        .iter()
        .filter(|r| r["generation"].as_u64() == Some(generation))
    {
        let being = row["being"]
            .as_u64()
            .ok_or_else(|| invalid("invalid observation being"))? as usize;
        let tick = row["tick"]
            .as_u64()
            .and_then(|t| u32::try_from(t).ok())
            .ok_or_else(|| invalid("invalid observation tick"))?;
        let frames = bodies
            .get(being)
            .ok_or_else(|| invalid("observation being outside replay"))?;
        // Observation cadence may exceed the replay cadence; never compare unlike times.
        let Ok(i) = frames.binary_search_by_key(&tick, |f| f.tick) else {
            continue;
        };
        let f = &frames[i];
        for (name, actual) in [("x", f.pos[0]), ("z", f.pos[2])] {
            let expected = row["body"][name]
                .as_f64()
                .filter(|v| v.is_finite())
                .ok_or_else(|| invalid("invalid observed position"))?;
            if (expected - actual as f64).abs() > 1e-4 {
                return Err(invalid(format!(
                    "driving observation position mismatch at tick {tick}"
                )));
            }
        }
        matched += 1;
    }
    if matched == 0 {
        return Err(invalid(
            "no driving observation aligns with this replay generation",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn rejects_wrong_body_and_accepts_matching_generation() {
        let frames = vec![vec![ReplayFrame {
            tick: 16,
            pos: [10.0, 0.0, 20.0],
            ..Default::default()
        }]];
        let mut data = json!({"meta":{"tick_rate":64},"observations":[
            {"being":0,"tick":16,"generation":3,"body":{"x":10.0,"z":20.0}}
        ]});
        let meta = json!({"tick_rate":64,"generation":3});
        assert!(validate(&data, &frames, &meta).is_ok());
        data["observations"][0]["body"]["x"] = json!(11.0);
        assert!(validate(&data, &frames, &meta).is_err());
        assert!(validate(&data, &frames, &json!({"tick_rate":64,"generation":0})).is_err());
    }
}
