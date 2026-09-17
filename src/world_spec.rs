//! `WorldSpec` — a seeded, parametric world request (`--world`, `--world-file`).
//!
//! Motivation (owner's words, 2026-09-17): "a system of different worlds, generated
//! randomly from seeds like Minecraft, so we can test thousands of environments;
//! something that takes seeds + parameters and builds a world." Today's tracks
//! (`world::track_waypoints_for_seed`) are all easy: see
//! `docs/WORLD-GENERATOR.md` for exactly which knobs exist and which of them this
//! module can actually move.
//!
//! Pure and GPU-free on purpose, like `gpu_evolve/pistas.rs` — parsing and
//! fingerprinting are testable without hardware.

/// One world request: a required `seed` plus optional difficulty knobs. Any field left
/// `None` means "today's hardcoded value" — see `docs/WORLD-GENERATOR.md` for what each
/// default resolves to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldSpec {
    pub seed: u64,
    /// Desired minimum curve radius, metres. Smaller = tighter curves = harder braking.
    pub radius: Option<f32>,
    /// Desired number of curve vertices in the generated circuit.
    pub curves: Option<usize>,
    /// Track width, metres. Same knob as today's global `--width`.
    pub width: Option<f32>,
    /// Hole density in `[0, 1]`: fraction of the sharpest turns that get a hazard.
    pub holes: Option<f32>,
    /// Desired lap length, metres.
    pub length: Option<f32>,
}

impl WorldSpec {
    /// A spec with only `seed` set must drive the world exactly like `--tracks
    /// <seed>` did before this module existed — see `world::track_waypoints_for_spec`.
    pub fn is_seed_only(&self) -> bool {
        self.radius.is_none()
            && self.curves.is_none()
            && self.width.is_none()
            && self.holes.is_none()
            && self.length.is_none()
    }

    /// Distinguishes two specs for the track-rotation reload check
    /// (`gpu_evolve/main.rs`'s `world_key_atual`) even when they share a `seed` but
    /// differ in some other field — a raw `seed` comparison would wrongly treat those
    /// as "no change" and skip the reload. FNV-1a over every field's bit pattern.
    pub fn fingerprint(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        let mix = |h: u64, x: u64| -> u64 { (h ^ x).wrapping_mul(0x100000001b3) };
        h = mix(h, self.seed);
        h = mix(h, self.radius.map_or(u64::MAX, |v| v.to_bits() as u64));
        h = mix(h, self.curves.map_or(u64::MAX, |v| v as u64));
        h = mix(h, self.width.map_or(u64::MAX, |v| v.to_bits() as u64));
        h = mix(h, self.holes.map_or(u64::MAX, |v| v.to_bits() as u64));
        h = mix(h, self.length.map_or(u64::MAX, |v| v.to_bits() as u64));
        h
    }
}

/// Parses one `--world` value: `key=value` pairs separated by commas, e.g.
/// `seed=8005,radius=9,curves=16,width=18,holes=0.4,length=3500`. `seed` is
/// required; unknown keys or unparsable values are an error (unlike `--tracks`,
/// which silently drops bad tokens — a spec is a deliberate experiment request, so
/// a typo here should fail loudly instead of quietly falling back to defaults).
pub fn parse(s: &str) -> Result<WorldSpec, String> {
    let mut seed: Option<u64> = None;
    let mut radius: Option<f32> = None;
    let mut curves: Option<usize> = None;
    let mut width: Option<f32> = None;
    let mut holes: Option<f32> = None;
    let mut length: Option<f32> = None;

    for pair in s.split(',') {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        let (key, value) = pair
            .split_once('=')
            .ok_or_else(|| format!("--world: '{pair}' is not key=value"))?;
        let key = key.trim();
        let value = value.trim();
        match key {
            "seed" => {
                seed = Some(
                    value
                        .parse::<u64>()
                        .map_err(|e| format!("--world: bad seed '{value}': {e}"))?,
                )
            }
            "radius" => {
                radius = Some(
                    value
                        .parse::<f32>()
                        .map_err(|e| format!("--world: bad radius '{value}': {e}"))?,
                )
            }
            "curves" => {
                curves = Some(
                    value
                        .parse::<usize>()
                        .map_err(|e| format!("--world: bad curves '{value}': {e}"))?,
                )
            }
            "width" => {
                width = Some(
                    value
                        .parse::<f32>()
                        .map_err(|e| format!("--world: bad width '{value}': {e}"))?,
                )
            }
            "holes" => {
                let v = value
                    .parse::<f32>()
                    .map_err(|e| format!("--world: bad holes '{value}': {e}"))?;
                if !(0.0..=1.0).contains(&v) {
                    return Err(format!("--world: holes={v} must be in [0, 1]"));
                }
                holes = Some(v);
            }
            "length" => {
                length = Some(
                    value
                        .parse::<f32>()
                        .map_err(|e| format!("--world: bad length '{value}': {e}"))?,
                )
            }
            other => return Err(format!("--world: unknown key '{other}'")),
        }
    }

    let seed = seed.ok_or_else(|| format!("--world '{s}': missing required key 'seed'"))?;
    Ok(WorldSpec {
        seed,
        radius,
        curves,
        width,
        holes,
        length,
    })
}

/// Parses `--world-file <path.json>`: a JSON array of objects with the same keys as
/// `parse` (`seed` required, the rest optional numbers), e.g.
/// `[{"seed": 8005, "radius": 9}, {"seed": 8100}]`. No `serde` dependency: this repo's
/// only other JSON writer (genome reports) also hand-rolls its encoding, so a small
/// hand-rolled reader here matches the existing style instead of adding a crate for
/// one call site.
pub fn parse_world_file(path: &std::path::Path) -> Result<Vec<WorldSpec>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("--world-file {}: {e}", path.display()))?;
    parse_world_file_str(&text)
}

fn parse_world_file_str(text: &str) -> Result<Vec<WorldSpec>, String> {
    let objects = split_top_level_objects(text)?;
    objects.iter().map(|o| parse_json_object(o)).collect()
}

/// Splits a `[ {...}, {...} ]` JSON array into its object substrings, tracking
/// brace depth so a comma inside a nested value never miscounts as a separator.
fn split_top_level_objects(text: &str) -> Result<Vec<String>, String> {
    let t = text.trim();
    let inner = t
        .strip_prefix('[')
        .and_then(|r| r.strip_suffix(']'))
        .ok_or_else(|| "--world-file: expected a top-level JSON array".to_string())?;
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (i, c) in inner.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => depth -= 1,
            ',' if depth == 0 => {
                let obj = inner[start..i].trim();
                if !obj.is_empty() {
                    out.push(obj.to_string());
                }
                start = i + 1;
            }
            _ => {}
        }
    }
    let tail = inner[start..].trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }
    Ok(out)
}

/// Parses one `{"seed": 8005, "radius": 9}` object into a `WorldSpec`. Deliberately
/// tiny: numeric values and `key: value` pairs only, no nesting, no strings — the
/// same key set as `parse()`.
fn parse_json_object(obj: &str) -> Result<WorldSpec, String> {
    let inner = obj
        .trim()
        .strip_prefix('{')
        .and_then(|r| r.strip_suffix('}'))
        .ok_or_else(|| format!("--world-file: '{obj}' is not a JSON object"))?;
    let csv: String = inner
        .split(',')
        .map(|kv| {
            let (k, v) = kv.split_once(':').unwrap_or(("", kv));
            format!("{}={}", k.trim().trim_matches('"'), v.trim())
        })
        .collect::<Vec<_>>()
        .join(",");
    parse(&csv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_seed_only() {
        let s = parse("seed=8005").unwrap();
        assert_eq!(s.seed, 8005);
        assert!(s.is_seed_only());
    }

    #[test]
    fn parses_every_key() {
        let s = parse("seed=8005,radius=9,curves=16,width=18,holes=0.4,length=3500").unwrap();
        assert_eq!(s.seed, 8005);
        assert_eq!(s.radius, Some(9.0));
        assert_eq!(s.curves, Some(16));
        assert_eq!(s.width, Some(18.0));
        assert_eq!(s.holes, Some(0.4));
        assert_eq!(s.length, Some(3500.0));
        assert!(!s.is_seed_only());
    }

    #[test]
    fn missing_seed_is_an_error() {
        assert!(parse("radius=9").is_err());
    }

    #[test]
    fn unknown_key_is_an_error() {
        assert!(parse("seed=1,bogus=2").is_err());
    }

    #[test]
    fn malformed_pair_is_an_error() {
        assert!(parse("seed").is_err());
    }

    #[test]
    fn holes_out_of_range_is_an_error() {
        assert!(parse("seed=1,holes=1.5").is_err());
    }

    #[test]
    fn whitespace_around_pairs_is_tolerated() {
        let s = parse(" seed = 8005 , radius = 9 ").unwrap();
        assert_eq!(s.seed, 8005);
        assert_eq!(s.radius, Some(9.0));
    }

    #[test]
    fn fingerprint_differs_when_any_field_differs() {
        let a = parse("seed=8005,radius=9").unwrap();
        let b = parse("seed=8005,radius=10").unwrap();
        let c = parse("seed=8005,radius=9").unwrap();
        assert_ne!(a.fingerprint(), b.fingerprint());
        assert_eq!(a.fingerprint(), c.fingerprint());
    }

    #[test]
    fn world_file_parses_array_of_specs() {
        let json = r#"[{"seed": 8005, "radius": 9}, {"seed": 8100}]"#;
        let specs = parse_world_file_str(json).unwrap();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].seed, 8005);
        assert_eq!(specs[0].radius, Some(9.0));
        assert!(specs[1].is_seed_only());
    }

    #[test]
    fn world_file_rejects_non_array() {
        assert!(parse_world_file_str(r#"{"seed": 1}"#).is_err());
    }
}
