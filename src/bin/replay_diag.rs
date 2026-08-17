//! BEHAVIOUR DIAGNOSIS from the replay — without rendering anything.
//!
//! # Why it exists
//!
//! Leonardo, looking at the freshly fixed map: *"now that you have the html, you can even
//! debug, you can use the data that feeds the html, no need to render and validate,
//! just read and compare: whether it leaves the track, crashes, reverses."*
//!
//! He is right, and it changes the cycle. Until now I judged behaviour by generation
//! averages (median, maximum, mortality), which say HOW MUCH but never WHERE nor WHY. The
//! replay stores position and speed per frame, and the track geometry is reconstructible
//! from the `world_seed` already in the header. Crossing the two yields the per-frame
//! truth: at which point of the layout it left, how long it spent on grass, where it stopped.
//!
//! That is how the last two bugs surfaced — the map that was the being's own trail,
//! and the odometer fitness that did not require following the track. Neither appeared in
//! any average.
//!
//! # Usage
//!
//!   replay_diag --replay telemetry/<race>/gen0011.replay [--width 28]

use bevy::math::Vec2;
use std::path::PathBuf;
use universe::replay::ReplayReader;
use universe::world::{VoxelType, VoxelWorld}; // same type `world.rs` uses (via bevy::prelude)

/// One contiguous off-track episode.
struct Excursion {
    start_tick: u32,
    end_tick: u32,
    at_progress: f32,
    max_depth: f32,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut path: Option<PathBuf> = None;
    let mut width = 18.0f32;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--replay" => {
                i += 1;
                if i < args.len() {
                    path = Some(PathBuf::from(&args[i]));
                }
            }
            "--width" => {
                i += 1;
                if i < args.len() {
                    width = args[i].parse().unwrap_or(18.0);
                }
            }
            _ => {}
        }
        i += 1;
    }
    let Some(path) = path else {
        eprintln!("usage: replay_diag --replay <file.replay> [--width 28]");
        std::process::exit(2);
    };

    let mut reader = match ReplayReader::open(&path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to open {}: {e}", path.display());
            std::process::exit(1);
        }
    };
    let seed = reader.header.world_seed;
    let frames = reader.all_frames();
    let Some(fr) = frames.first() else {
        eprintln!("replay has no frames");
        std::process::exit(1);
    };
    if fr.is_empty() {
        eprintln!("replay has no frames");
        std::process::exit(1);
    }

    // The world constructor dumps 129 waypoints to stdout. Useful once, noise on
    // every call — silence it while building.
    let world = {
        let _mudo = universe::world::silence_map_diag();
        VoxelWorld::new(seed, width)
    };
    let lap = world.lap_distance;

    println!("=== DIAGNOSIS: {} ===", path.display());
    println!(
        "world seed {seed} · track {width:.0} m · lap {lap:.0} m · {} frames\n",
        fr.len()
    );

    // ── Per-frame sweep ─────────────────────────────────────────────────────
    let mut on_road = 0usize;
    let mut on_grass = 0usize;
    let mut on_wall = 0usize;
    let mut reversing = 0usize;
    let mut stopped = 0usize;
    let mut excursions: Vec<Excursion> = Vec::new();
    let mut cur: Option<Excursion> = None;
    // Monotonic NET progress, not amplitude. `max - min` counts a being that backs up
    // across the line (14m→2900m) as '99.5% of the lap' — false. Same bug already
    // fixed in replay_obs; replay_diag still had the old version.
    let mut prog_liq = 0.0f32;
    let mut prog_pico = 0.0f32;
    let mut prog_ant = world.get_track_progress(fr[0].pos[0], fr[0].pos[2]);
    let mut speed_sum = 0.0f64;
    let mut speed_max = 0.0f32;

    for f in fr {
        let p = Vec2::new(f.pos[0], f.pos[2]);
        let mat = world.material_at(p);
        let prog = world.get_track_progress(p.x, p.y);
        let mut d = prog - prog_ant;
        if d < -lap * 0.5 {
            d += lap;
        } else if d > lap * 0.5 {
            d -= lap;
        }
        prog_liq += d.clamp(-lap * 0.1, lap * 0.1);
        prog_pico = prog_pico.max(prog_liq);
        prog_ant = prog;
        speed_sum += f.speed as f64;
        speed_max = speed_max.max(f.speed);

        // Reverse gear: velocity projected on the forward direction is negative.
        // `fwd` in the frame is (x, z), already normalized.
        if f.gear == 0 {
            reversing += 1;
        }
        if f.speed < 0.5 {
            stopped += 1;
        }

        let fora = match mat {
            VoxelType::Road => {
                on_road += 1;
                false
            }
            VoxelType::Wall => {
                on_wall += 1;
                true
            }
            _ => {
                on_grass += 1;
                true
            }
        };

        // Excursion depth: distance to the track centerline minus the half-width.
        let depth = distance_to_centerline(&world, p) - width / 2.0;

        if fora {
            match cur.as_mut() {
                Some(e) => {
                    e.end_tick = f.tick;
                    e.max_depth = e.max_depth.max(depth);
                }
                None => {
                    cur = Some(Excursion {
                        start_tick: f.tick,
                        end_tick: f.tick,
                        at_progress: prog,
                        max_depth: depth,
                    })
                }
            }
        } else if let Some(e) = cur.take() {
            excursions.push(e);
        }
    }
    if let Some(e) = cur.take() {
        excursions.push(e);
    }

    let n = fr.len() as f32;
    let pct = |c: usize| 100.0 * c as f32 / n;

    println!("── WHERE THE BEING SPENT ITS TIME ──");
    println!("  on track:   {:5.1}%  ({} frames)", pct(on_road), on_road);
    println!(
        "  on grass:   {:5.1}%  ({} frames)",
        pct(on_grass),
        on_grass
    );
    println!("  in wall:    {:5.1}%  ({} frames)", pct(on_wall), on_wall);
    println!("  stopped (<0.5 m/s): {:5.1}%", pct(stopped));
    println!("  reverse gear:       {:5.1}%", pct(reversing));

    println!("\n── PROGRESS AROUND THE CIRCUIT ──");
    let span = prog_pico.max(0.0);
    println!(
        "  progress reached: {:.0} m of {:.0} m ({:.1}% of the lap)",
        span,
        lap,
        100.0 * span / lap
    );
    println!(
        "  mean speed {:.2} m/s ({:.1} km/h) · max {:.2} m/s ({:.1} km/h)",
        speed_sum / fr.len() as f64,
        speed_sum / fr.len() as f64 * 3.6,
        speed_max,
        speed_max * 3.6
    );

    // ── Does the being TRY to turn? ────────────────────────────────────────
    // They all stop at the same point of the layout (~265 m, where the curve goes from 10° to 18°).
    // This separates the two explanations: steering at zero = not commanding a turn; steering at
    // full lock without the car rotating = commands it and the body does not obey.
    {
        let mut str_abs: Vec<f32> = fr.iter().map(|f| f.final_str.abs()).collect();
        str_abs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = str_abs[str_abs.len() / 2];
        let p90 = str_abs[(str_abs.len() * 9 / 10).min(str_abs.len() - 1)];
        let saturado = str_abs.iter().filter(|&&v| v > 0.9).count();
        let quieto = str_abs.iter().filter(|&&v| v < 0.05).count();
        let mut thr: Vec<f32> = fr.iter().map(|f| f.final_thr).collect();
        thr.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("\n── COMMANDS ──");
        println!(
            "  steering |x|: median {:.2} · p90 {:.2} · at full lock {:.0}% · near zero {:.0}%",
            med,
            p90,
            pct(saturado),
            pct(quieto)
        );
        println!("  throttle: median {:.2}", thr[thr.len() / 2]);
        // The SIGN decides between "corrects to both sides" and "lives crooked".
        // Steering pinned to one side + constant throttle = circle, and a circle inside a
        // wide track counts neither as leaving the track nor as crashing: it vanishes from
        // every aggregate metric and only shows in the progress that stops climbing.
        let assinado: Vec<f32> = fr.iter().map(|f| f.final_str).collect();
        let media_ass = assinado.iter().sum::<f32>() / assinado.len() as f32;
        let esq = assinado.iter().filter(|&&v| v < -0.05).count();
        let dir = assinado.iter().filter(|&&v| v > 0.05).count();
        println!(
            "  signed steering: mean {:+.2} · left {:.0}% · right {:.0}%",
            media_ass,
            pct(esq),
            pct(dir)
        );
        if media_ass.abs() > 0.3 && (pct(esq) > 80.0 || pct(dir) > 80.0) {
            println!("  ⇒ STEERING PINNED to one side only: the being drives in a CIRCLE, not a curve.");
            println!("    A circle inside a wide track neither leaves the layout nor crashes —");
            println!("    it vanishes from every aggregate metric and only shows as stalled progress.");
        }

        if med < 0.05 && p90 < 0.2 {
            println!("  ⇒ the steering stays STILL: the brain is not commanding a turn.");
        } else if pct(saturado) > 30.0 {
            println!(
                "  ⇒ steering at full lock {:.0}% of the time: commands a turn and it does not resolve.",
                pct(saturado)
            );
        }
    }

    // ── Does the steering RESPOND to the track? ────────────────────────────
    //
    // Removing the smoothing (`--steer-alpha 1.0`) changed nothing: 265 m and +0.40 mean in
    // both cases. So the constant command comes from the brain, not the filter. What remains is
    // separating "does not see the curve" from "sees it and does not use it": if the steering
    // does not correlate with the layout curvature ahead, it is not driving — it is holding.
    {
        let mut cur: Vec<f32> = Vec::with_capacity(fr.len());
        let mut st: Vec<f32> = Vec::with_capacity(fr.len());
        for f in fr {
            let p = Vec2::new(f.pos[0], f.pos[2]);
            let k = curvatura_a_frente(&world, p, 25.0);
            if k.is_finite() {
                cur.push(k);
                st.push(f.final_str);
            }
        }
        if cur.len() > 30 {
            let r = correlacao(&cur, &st);
            let var_st = variancia(&st);
            println!("\n── DOES THE STEERING FOLLOW THE TRACK? ──");
            println!("  correlation steering × curvature 25 m ahead: {:+.3}", r);
            println!("  steering variance: {:.4}", var_st);
            if r.abs() < 0.15 {
                println!("  ⇒ does NOT follow: the command is practically independent of the layout.");
                println!("    The being is not driving, it is holding the wheel in one position.");
            } else if r > 0.3 {
                println!(
                    "  ⇒ follows the curvature — the problem is amplitude or timing, not reading."
                );
            }
        }
    }

    println!("\n── TRACK EXCURSIONS ({}) ──", excursions.len());
    if excursions.is_empty() {
        println!("  none — stayed on the layout the whole time.");
    } else {
        // The FIRST excursion is the one that matters: it is where the competence ends. The
        // following ones already happen with the being lost, so they describe the drift, not the failure.
        for (k, e) in excursions.iter().take(6).enumerate() {
            let dur = (e.end_tick - e.start_tick) as f32 / 64.0;
            println!(
                "  {}. tick {:>6}–{:<6} ({:5.1}s) · at {:.0} m of the lap ({:.0}%) · up to {:.1} m off",
                k + 1,
                e.start_tick,
                e.end_tick,
                dur,
                e.at_progress,
                100.0 * e.at_progress / lap,
                e.max_depth.max(0.0)
            );
        }
        if excursions.len() > 6 {
            println!("  … and {} more", excursions.len() - 6);
        }

        let e = &excursions[0];
        println!(
            "\n  The FIRST excursion is the diagnosis: the being drove well up to {:.0} m",
            e.at_progress
        );
        println!(
            "  ({:.0}% of the lap) and left the layout there.",
            100.0 * e.at_progress / lap
        );
        descrever_curva(&world, e.at_progress);
    }

    // Short verdict, so the tables do not need interpreting every time.
    println!("\n── VERDICT ──");
    if pct(on_road) < 50.0 {
        println!("  Spends more time OFF the track than on it. The objective is not requiring");
        println!("  following the layout, or it cannot — look at the first excursion above.");
    } else if pct(stopped) > 30.0 {
        println!(
            "  Stays stopped {:.0}% of the time. The bottleneck is MOVING, not navigating.",
            pct(stopped)
        );
    } else if span < lap * 0.5 {
        println!(
            "  Stays on track but only covers {:.0}% of the lap: gets stuck at a specific point.",
            100.0 * span / lap
        );
    } else {
        println!("  Follows the layout and covers a good part of the lap.");
    }
}

/// Distance to the track centerline, via the nearest waypoint.
fn distance_to_centerline(world: &VoxelWorld, p: Vec2) -> f32 {
    world
        .waypoints_world
        .iter()
        .map(|w| w.distance(p))
        .fold(f32::MAX, f32::min)
}

/// Describes the curvature at the point where the being left — straight and tight curve fail
/// for different reasons, and the table alone does not distinguish them.
fn descrever_curva(world: &VoxelWorld, at_progress: f32) {
    let wps = &world.waypoints_world;
    if wps.len() < 3 {
        return;
    }
    // Finds the waypoint whose accumulated progress is closest to the exit point.
    let mut acc = 0.0f32;
    let mut idx = 0usize;
    for i in 0..wps.len() - 1 {
        let seg = wps[i].distance(wps[i + 1]);
        if acc + seg >= at_progress {
            idx = i;
            break;
        }
        acc += seg;
    }
    let a = wps[idx.saturating_sub(1)];
    let b = wps[idx];
    let c = wps[(idx + 1).min(wps.len() - 1)];
    let v1 = (b - a).normalize_or_zero();
    let v2 = (c - b).normalize_or_zero();
    let ang = v1.dot(v2).clamp(-1.0, 1.0).acos().to_degrees();
    let tipo = if ang > 45.0 {
        "TIGHT CURVE"
    } else if ang > 15.0 {
        "curve"
    } else {
        "straight"
    };
    println!("  Geometry there: {tipo} ({ang:.0}° between segments, waypoint {idx}).");
}

/// Signed curvature of the layout `ahead` meters in front of point `p`.
///
/// Positive = the track turns right. Serves as the "correct answer" against which to compare
/// the steering command, without depending on any of the brain's sensory channels.
fn curvatura_a_frente(world: &VoxelWorld, p: Vec2, ahead: f32) -> f32 {
    let wps = &world.waypoints_world;
    if wps.len() < 4 {
        return f32::NAN;
    }
    let mut i0 = 0usize;
    let mut best = f32::MAX;
    for (i, w) in wps.iter().enumerate() {
        let d = w.distance(p);
        if d < best {
            best = d;
            i0 = i;
        }
    }
    // Walks the list until accumulating `ahead` meters.
    let mut acc = 0.0;
    let mut i = i0;
    while acc < ahead && i + 1 < wps.len() {
        acc += wps[i].distance(wps[i + 1]);
        i += 1;
    }
    if i + 1 >= wps.len() {
        return f32::NAN;
    }
    let a = wps[i.saturating_sub(1)];
    let b = wps[i];
    let c = wps[i + 1];
    let v1 = (b - a).normalize_or_zero();
    let v2 = (c - b).normalize_or_zero();
    // The 2D cross product gives the DIRECTION of the curve.
    v1.x * v2.y - v1.y * v2.x
}

fn variancia(v: &[f32]) -> f32 {
    let m = v.iter().sum::<f32>() / v.len() as f32;
    v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32
}

fn correlacao(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len()) as f32;
    let ma = a.iter().sum::<f32>() / n;
    let mb = b.iter().sum::<f32>() / n;
    let cov: f32 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let sa: f32 = a.iter().map(|x| (x - ma).powi(2)).sum::<f32>().sqrt();
    let sb: f32 = b.iter().map(|y| (y - mb).powi(2)).sum::<f32>().sqrt();
    if sa * sb < 1e-12 {
        0.0
    } else {
        cov / (sa * sb)
    }
}
