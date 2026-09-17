//! TRACK VALIDATION — is the track drivable, or is the map broken?
//!
//! # Why it exists
//!
//! Leonardo: *"either you got it wrong, or the map is bugged for the model. Seriously,
//! something is wrong. The model can do everything — it turns, sees, thinks — but it does
//! not take the curve."*
//!
//! Every controller tested stops between 255 and 271 m: the evolved brain, a geometric
//! pilot with the racing line, a sensory pilot using wall rays, and the same pilot
//! with the signal inverted. Four independent controllers stopping at the same point is not
//! a control coincidence — it is a property of the world.
//!
//! This binary simulates nothing. It WALKS along the track axis meter by meter and asks
//! the same `VoxelWorld` the simulator uses: what is here? Does the track continue? What is the
//! free width? If there is a wall across the middle of the road at 270 m, the
//! investigation is over.
//!
//! # Usage
//!
//!   pista_check [--width 28] [--ate 600] [--seed 9001]
//!   pista_check --world seed=8005,radius=9,width=18,holes=0.4,length=3500 [--ate 600]

use bevy::math::Vec2;
use universe::world::{VoxelType, VoxelWorld};
use universe::world_spec::WorldSpec;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut width = 28.0f32;
    let mut ate = 600.0f32;
    let mut seed = 9001u64;
    let mut spec: Option<WorldSpec> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--width" => {
                i += 1;
                if i < args.len() {
                    width = args[i].parse().unwrap_or(28.0);
                }
            }
            "--ate" => {
                i += 1;
                if i < args.len() {
                    ate = args[i].parse().unwrap_or(600.0);
                }
            }
            "--seed" => {
                i += 1;
                if i < args.len() {
                    seed = args[i].parse().unwrap_or(9001);
                }
            }
            // Same `WorldSpec` syntax as `gpu_evolve --world` (universe::world_spec).
            // A seed-only spec is byte-identical to `--seed <seed>` above — see
            // docs/WORLD-GENERATOR.md.
            "--world" => {
                i += 1;
                if i < args.len() {
                    match universe::world_spec::parse(&args[i]) {
                        Ok(s) => spec = Some(s),
                        Err(e) => {
                            eprintln!("{e}");
                            std::process::exit(2);
                        }
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }

    let world = {
        let _q = universe::world::silence_map_diag();
        match &spec {
            Some(s) => VoxelWorld::new_from_spec(s),
            None => VoxelWorld::new(seed, width),
        }
    };
    // For `--world`, report the spec's own width/holes/seed instead of the `--width`
    // default and the coin-flip hole count — those are what the requested track
    // actually got built with (see `VoxelWorld::new_from_spec`, `world.rs`).
    let (report_seed, report_width) = match &spec {
        Some(s) => (s.seed, s.width.unwrap_or(28.0)),
        None => (seed, width),
    };
    // Shadow `width` with the value the grid was actually built at, so the defect
    // checks below (`esq + dir < width * 0.5`) use the spec's real width too.
    let width = report_width;
    let wps = &world.waypoints_world;
    let n = wps.len();

    println!(
        "TRACK seed {report_seed} · declared width {report_width:.0} m · lap {:.0} m · \
         {n} waypoints",
        world.lap_distance
    );
    if let Some(s) = &spec {
        println!(
            "  world spec: radius={} curves={} holes={} length={}\n",
            s.radius.map_or("default".to_string(), |v| format!("{v:.1}m")),
            s.curves.map_or("default".to_string(), |v| v.to_string()),
            s.holes.map_or("default (1-2)".to_string(), |v| format!("{v:.2}")),
            s.length.map_or("default".to_string(), |v| format!("{v:.0}m")),
        );
    } else {
        println!();
    }

    // Walks along the AXIS, interpolating between waypoints, and measures at each meter:
    //   · the material under the axis (it must be Road; anything else is a hole in the layout)
    //   · the free width on each side until a wall is found
    //   · the local curve radius
    let mut arco = 0.0f32;
    let mut wp = 0usize;
    let mut t = 0.0f32;
    let mut problemas = 0usize;

    println!(
        "{:>6} {:>8} {:>7} {:>7} {:>7}  {}",
        "arc", "material", "left", "right", "radius", ""
    );
    println!("{}", "-".repeat(62));

    while arco < ate && wp < n - 1 {
        let a = wps[wp];
        let b = wps[wp + 1];
        let seg = (b - a).length().max(0.001);
        let p = a + (b - a) * (t / seg);

        // Normal to the axis, to measure the free width on both sides.
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

        // Local radius from the angle between neighboring segments.
        let raio = if wp > 0 && wp + 1 < n {
            let v1 = wps[wp] - wps[wp - 1];
            let v2 = wps[wp + 1] - wps[wp];
            let ang = (v1.x * v2.y - v1.y * v2.x).atan2(v1.dot(v2)).abs();
            if ang > 1e-4 {
                (0.5 * (v1.length() + v2.length())) / ang
            } else {
                99999.0
            }
        } else {
            99999.0
        };

        // Only prints what matters: every 10 m, and ALWAYS when there is a defect.
        let defeito = !matches!(mat, VoxelType::Road) || esq + dir < width * 0.5;
        if defeito {
            problemas += 1;
        }
        if arco as u32 % 10 == 0 || defeito {
            let marca = if !matches!(mat, VoxelType::Road) {
                "  ← AXIS OFF THE TRACK"
            } else if esq + dir < width * 0.5 {
                "  ← BOTTLENECK"
            } else {
                ""
            };
            println!(
                "{:6.0} {:>8} {:6.1}m {:6.1}m {:6.0}m{}",
                arco,
                format!("{:?}", mat),
                esq,
                dir,
                raio.min(99999.0),
                marca
            );
        }

        arco += 1.0;
        t += 1.0;
        if t >= seg {
            t -= seg;
            wp += 1;
        }
    }

    // ── WHAT THE BEING ACTUALLY PERCEIVES ─────────────────────────────────
    //
    // Leonardo: *"it has to perceive the curve, its vision has to literally see it."*
    //
    // Fidelity is not enough. A perfect sensor of a CONSTANT value delivers zero bits of
    // information, and that is how I mistook correlation 1.000 for "they can see". Here I
    // reproduce the frontal ray exactly as `gpu_evolve.rs` computes it — probes from 1 to
    // 10 m, default 30.0 when nothing is found — and report how many times it SATURATES.
    {
        let mut arco = 0.0f32;
        let mut wp = 0usize;
        let mut t = 0.0f32;
        let mut sat = 0usize;
        let mut tot = 0usize;
        let mut vals: Vec<f32> = Vec::new();
        while arco < ate && wp < n - 1 {
            let a = wps[wp];
            let b = wps[wp + 1];
            let seg = (b - a).length().max(0.001);
            let p = a + (b - a) * (t / seg);
            let f = (b - a) / seg;
            // The RIGHT SENSOR. The previous version measured the 1 to 10 m probes, which are the
            // REVERSE-GEAR trigger in `gpu_evolve.rs:573` — not vision. The being's vision
            // is in `perception.wgsl:373`: eight samples up to 60 m. Measuring the wrong sensor
            // and concluding "the being is blind" is exactly the mistake this file exists to
            // prevent from happening.
            let mut fwd = 60.0f32;
            for sd in [3.0f32, 8.0, 15.0, 25.0, 35.0, 45.0, 52.0, 60.0] {
                if !matches!(world.material_at(p + f * sd), VoxelType::Road) {
                    fwd = sd;
                    break;
                }
            }
            if fwd >= 60.0 {
                sat += 1;
            }
            vals.push(fwd);
            tot += 1;
            arco += 1.0;
            t += 1.0;
            if t >= seg {
                t -= seg;
                wp += 1;
            }
        }
        let media = vals.iter().sum::<f32>() / tot.max(1) as f32;
        let dp = (vals.iter().map(|v| (v - media).powi(2)).sum::<f32>() / tot.max(1) as f32).sqrt();
        println!("\n── WHAT THE FRONTAL VISION DELIVERS (car on the axis, perception.wgsl) ──");
        println!(
            "  saturated at 60.0 m: {:.1}% of the points ({sat} of {tot})",
            100.0 * sat as f32 / tot.max(1) as f32
        );
        println!("  mean {media:.1} m · standard deviation {dp:.2} m");
        if dp < 0.5 {
            println!(
                "  → NULL VARIANCE: the channel is constant. The being is blind to the curve through\n                      \x20    this sensor, however faithful it may be."
            );
        }
    }

    println!("\n{}", "-".repeat(62));
    if problemas == 0 {
        println!(
            "No defects: the axis is on asphalt and the free width is coherent\n\
                  across the whole examined stretch. The track is drivable — the problem is not the map."
        );
    } else {
        println!(
            "{problemas} points with defects in the examined stretch.\n\
                  Axis off the track or a bottleneck means that NO controller\n\
                  would pass — the control investigation was chasing the wrong thing."
        );
    }
}
