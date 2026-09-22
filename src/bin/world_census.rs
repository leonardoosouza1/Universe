//! WORLD CENSUS — validity of thousands of seeded worlds, one TSV line each.
//!
//! # Why it exists
//!
//! Owner's request (2026-09-17): "a system of different worlds, generated randomly
//! from seeds like Minecraft, so we can test thousands of environments." `pista_check`
//! (`src/bin/pista_check.rs`) answers "is THIS one seed drivable?" one track at a
//! time, with a full metre-by-metre printout — useful for one track, unusable for
//! thousands. This binary runs the same validity check `pista_check` runs, CPU-side,
//! over a whole seed range, and prints one line per world.
//!
//! # Usage
//!
//!   world_census --seeds 8100..8299 [radius=9,width=20,curves=..,holes=..,length=..] [--out path.tsv]
//!
//! The bracketed part is the same `key=value` list `--world`/`WorldSpec` takes (see
//! `src/world_spec.rs`), minus `seed` — one is generated per seed in the range and
//! merged with these shared parameters. Omit it to census the plain `--tracks <seed>`
//! path (`world::track_waypoints_for_seed`) instead of `WorldSpec`.

use universe::world::VoxelWorld;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut seeds: Option<(u64, u64)> = None;
    let mut shared = String::new();
    let mut out_path: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--seeds" => {
                i += 1;
                if i < args.len() {
                    seeds = Some(parse_range(&args[i]).unwrap_or_else(|e| {
                        eprintln!("--seeds {}: {e}", args[i]);
                        std::process::exit(2);
                    }));
                }
            }
            "--out" => {
                i += 1;
                if i < args.len() {
                    out_path = Some(args[i].clone());
                }
            }
            other if other.contains('=') => shared = other.to_string(),
            other => {
                eprintln!("world_census: unknown argument '{other}'");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let (lo, hi) = seeds.unwrap_or_else(|| {
        eprintln!("world_census: --seeds A..B is required");
        std::process::exit(2);
    });

    let header = "seed\tvalid\tmin_radius_m\tn_curves\tlength_m\twidth_m\tholes\tdefects";
    let mut lines = vec![header.to_string()];
    for seed in lo..=hi {
        lines.push(census_one(seed, &shared));
    }

    let body = lines.join("\n");
    match out_path {
        Some(path) => {
            std::fs::write(&path, format!("{body}\n"))
                .unwrap_or_else(|e| panic!("world_census: failed to write {path}: {e}"));
            eprintln!("world_census: {} worlds written to {path}", hi - lo + 1);
        }
        None => println!("{body}"),
    }
}

/// Parses `"8100..8299"` into an inclusive `(lo, hi)`.
fn parse_range(s: &str) -> Result<(u64, u64), String> {
    let (a, b) = s
        .split_once("..")
        .ok_or_else(|| "expected A..B".to_string())?;
    let lo: u64 = a
        .trim()
        .parse()
        .map_err(|e| format!("bad lower bound: {e}"))?;
    let hi: u64 = b
        .trim()
        .parse()
        .map_err(|e| format!("bad upper bound: {e}"))?;
    if hi < lo {
        return Err(format!("upper bound {hi} < lower bound {lo}"));
    }
    Ok((lo, hi))
}

/// Builds one world (spec-driven if `shared` is non-empty, else the plain
/// `--tracks <seed>` path) and runs the same axis-walk `pista_check` uses to decide
/// drivability. Returns one TSV line.
fn census_one(seed: u64, shared: &str) -> String {
    let spec = if shared.is_empty() {
        None
    } else {
        let with_seed = format!("seed={seed},{shared}");
        match universe::world_spec::parse(&with_seed) {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("[world_census] seed={seed}: {e}");
                return format!("{seed}\tERROR\t\t\t\t\t\t{e}");
            }
        }
    };

    let world = {
        let _q = universe::world::silence_map_diag();
        match &spec {
            Some(s) => VoxelWorld::new_from_spec(s),
            None => VoxelWorld::new(seed, 28.0),
        }
    };
    let width = spec.as_ref().and_then(|s| s.width).unwrap_or(28.0);

    let wps_cells = spec
        .as_ref()
        .map(universe::world::track_waypoints_for_spec)
        .unwrap_or_else(|| universe::world::track_waypoints_for_seed(seed));
    let d = universe::pista_gen::medir(&wps_cells, width, 40.0);

    // Numeric density, not the word "default": when `holes` is unset we still apply
    // an effective density, we just get there via the seed-derived 1-2 coin flip
    // (`track_holes_for_seed`) instead of `WorldSpec.holes` — report what actually
    // got placed relative to the curve count `medir` just measured, so every row is
    // comparable on the same axis.
    let cell_size = 3.0;
    let holes_placed = match spec.as_ref().and_then(|s| s.holes) {
        Some(density) => {
            universe::world::track_holes_with_density(&wps_cells, width / cell_size, density).len()
        }
        None => universe::world::track_holes_for_seed(seed, &wps_cells, width / cell_size).len(),
    };
    let holes_report = if d.n_curvas > 0 {
        format!("{:.3}", holes_placed as f32 / d.n_curvas as f32)
    } else {
        "0.000".to_string()
    };

    let defects = axis_defect_count(&world, width, world.lap_distance.min(3000.0));
    let valid = defects == 0;

    format!(
        "{seed}\t{valid}\t{:.1}\t{}\t{:.0}\t{width:.0}\t{holes_report}\t{defects}",
        d.r_min, d.n_curvas, d.comprimento
    )
}

/// Same axis walk `pista_check` does (`src/bin/pista_check.rs`): every metre along
/// the centreline, count points where the axis is off asphalt or the free width is
/// less than half the declared width. Duplicated rather than imported because
/// `pista_check` computes it inline in `main()`, not as a reusable function — pulling
/// it out cleanly is a separate refactor, not part of this generator's scope.
fn axis_defect_count(world: &VoxelWorld, width: f32, until_m: f32) -> usize {
    use bevy::math::Vec2;
    use universe::world::VoxelType;

    let wps = &world.waypoints_world;
    let n = wps.len();
    if n < 2 {
        return 1;
    }
    let mut arco = 0.0f32;
    let mut wp = 0usize;
    let mut t = 0.0f32;
    let mut problemas = 0usize;
    while arco < until_m && wp < n - 1 {
        let a = wps[wp];
        let b = wps[wp + 1];
        let seg = (b - a).length().max(0.001);
        let p = a + (b - a) * (t / seg);
        let dir_seg = (b - a) / seg;
        let nrm = Vec2::new(-dir_seg.y, dir_seg.x);

        let mat = world.material_at(p);
        let livre = |sinal: f32| -> f32 {
            let mut d = 0.0f32;
            while d < 300.0 {
                let q = p + nrm * (d * sinal);
                if !matches!(world.material_at(q), VoxelType::Road) {
                    return d;
                }
                d += 0.5;
            }
            300.0
        };
        let esq = livre(1.0);
        let dir = livre(-1.0);
        if !matches!(mat, VoxelType::Road) || esq + dir < width * 0.5 {
            problemas += 1;
        }

        arco += 1.0;
        t += 1.0;
        if t >= seg {
            t -= seg;
            wp += 1;
        }
    }
    problemas
}
