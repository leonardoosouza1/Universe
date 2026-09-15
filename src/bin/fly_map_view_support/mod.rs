//! Parser and bounded model for synchronized evolution telemetry.
//!
//! The viewer receives measured rows from `gpu_evolve`; missing columns remain
//! unavailable instead of being reconstructed from the connectome trace.

use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub struct TelemetryRow {
    pub episode: u64,
    pub tick: u64,
    pub being: usize,
    pub alive: bool,
    pub coverage: f32,
    pub asymmetry: f32,
    pub left_pixels: f32,
    pub right_pixels: f32,
    pub visible_distance: f32,
    pub body_distance: f32,
    pub overlap_count: u32,
    pub raw_steer: f32,
    pub applied_steer: f32,
    pub throttle: f32,
    pub brake: f32,
    pub speed: f32,
    pub x: Option<f32>,
    pub z: Option<f32>,
    pub forward_x: Option<f32>,
    pub forward_z: Option<f32>,
}

pub fn load_csv(path: &Path) -> Result<Vec<TelemetryRow>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse_csv(&text)
}

pub fn parse_csv(text: &str) -> Result<Vec<TelemetryRow>, String> {
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header = lines.next().ok_or("telemetry CSV is empty")?;
    let columns: Vec<&str> = header.split(',').map(str::trim).collect();
    let col = |name: &str| {
        columns
            .iter()
            .position(|value| *value == name)
            .ok_or_else(|| format!("telemetry CSV is missing column {name}"))
    };
    let required = [
        "episode",
        "tick",
        "being",
        "alive",
        "coverage",
        "asymmetry",
        "left_pixels",
        "right_pixels",
        "visible_distance",
        "body_distance",
        "overlap_count",
        "raw_steer",
        "applied_steer",
        "throttle",
        "brake",
        "speed",
    ];
    let indices: Vec<usize> = required
        .iter()
        .map(|name| col(name))
        .collect::<Result<_, _>>()?;
    let optional = |name: &str| columns.iter().position(|value| *value == name);
    let parse = |fields: &[&str], idx: usize, name: &str| {
        let value = fields
            .get(idx)
            .ok_or_else(|| format!("row is missing {name}"))?
            .parse::<f32>()
            .map_err(|e| format!("invalid {name}: {e}"))?;
        value
            .is_finite()
            .then_some(value)
            .ok_or_else(|| format!("non-finite {name}"))
    };
    let parse_u64 = |fields: &[&str], idx: usize, name: &str| {
        fields
            .get(idx)
            .ok_or_else(|| format!("row is missing {name}"))?
            .parse::<u64>()
            .map_err(|e| format!("invalid {name}: {e}"))
    };
    let mut rows = Vec::new();
    for (line_no, line) in lines.enumerate() {
        let fields: Vec<&str> = line.split(',').map(str::trim).collect();
        let alive_text = fields
            .get(indices[3])
            .ok_or_else(|| format!("row {} is truncated", line_no + 2))?;
        let alive = if *alive_text == "1" || alive_text.eq_ignore_ascii_case("true") {
            true
        } else if *alive_text == "0" || alive_text.eq_ignore_ascii_case("false") {
            false
        } else {
            return Err(format!(
                "invalid alive on row {}: expected 0/1 or true/false",
                line_no + 2
            ));
        };
        let opt = |name: &str| -> Result<Option<f32>, String> {
            let Some(idx) = optional(name) else {
                return Ok(None);
            };
            let value = fields
                .get(idx)
                .ok_or_else(|| format!("row is missing {name}"))?
                .parse::<f32>()
                .map_err(|e| format!("invalid {name}: {e}"))?;
            value
                .is_finite()
                .then_some(Some(value))
                .ok_or_else(|| format!("non-finite {name}"))
        };
        rows.push(TelemetryRow {
            episode: parse_u64(&fields, indices[0], "episode")?,
            tick: parse_u64(&fields, indices[1], "tick")?,
            being: parse_u64(&fields, indices[2], "being")? as usize,
            alive,
            coverage: parse(&fields, indices[4], "coverage")?,
            asymmetry: parse(&fields, indices[5], "asymmetry")?,
            left_pixels: parse(&fields, indices[6], "left_pixels")?,
            right_pixels: parse(&fields, indices[7], "right_pixels")?,
            visible_distance: parse(&fields, indices[8], "visible_distance")?,
            body_distance: parse(&fields, indices[9], "body_distance")?,
            overlap_count: parse_u64(&fields, indices[10], "overlap_count")? as u32,
            raw_steer: parse(&fields, indices[11], "raw_steer")?,
            applied_steer: parse(&fields, indices[12], "applied_steer")?,
            throttle: parse(&fields, indices[13], "throttle")?,
            brake: parse(&fields, indices[14], "brake")?,
            speed: parse(&fields, indices[15], "speed")?,
            x: opt("x")?,
            z: opt("z")?,
            forward_x: opt("forward_x")?,
            forward_z: opt("forward_z")?,
        });
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_required_and_optional_columns() {
        let csv = "episode,tick,being,alive,coverage,asymmetry,left_pixels,right_pixels,visible_distance,body_distance,overlap_count,raw_steer,applied_steer,throttle,brake,speed,x,z,forward_x,forward_z\n1,64,2,1,.5,-.2,3,4,12,2,1,.1,.2,.8,0,7,10,20,0,1\n";
        let row = parse_csv(csv).unwrap().pop().unwrap();
        assert_eq!(row.being, 2);
        assert_eq!(row.x, Some(10.0));
        assert!(row.alive);
    }
    #[test]
    fn rejects_missing_required_column() {
        assert!(
            parse_csv("tick,being\n1,0\n")
                .unwrap_err()
                .contains("episode")
        );
    }

    #[test]
    fn rejects_non_finite_optional_coordinates() {
        let csv = "episode,tick,being,alive,coverage,asymmetry,left_pixels,right_pixels,visible_distance,body_distance,overlap_count,raw_steer,applied_steer,throttle,brake,speed,x\n1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,NaN\n";
        assert!(parse_csv(csv).unwrap_err().contains("non-finite x"));
    }

    #[test]
    fn parses_boolean_values_case_insensitively() {
        let header = "episode,tick,being,alive,coverage,asymmetry,left_pixels,right_pixels,visible_distance,body_distance,overlap_count,raw_steer,applied_steer,throttle,brake,speed";
        let csv = format!("{header}\n1,0,0,True,0,0,0,0,0,0,0,0,0,0,0,0\n");
        assert!(parse_csv(&csv).unwrap()[0].alive);
    }
}
