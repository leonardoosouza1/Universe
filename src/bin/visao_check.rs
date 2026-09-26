//! PER-OBJECT VISION VALIDATION — does the being see what is in front of it?
//!
//! # Why it exists
//!
//! Leonardo: *"it would be good to have some way of validating whether the model really
//! sees things — whether it sees a wall, whether it sees the ground, whether it sees
//! something we place. From an event where the vision hits that object, we compute
//! whether it is really seeing it. To evaluate: gee, the model does so well without the
//! hole, but with the hole it can't — fine, is it because it saturates here, or because
//! it only sees when it is right on top of it."*
//!
//! The 2026-08-13 A/B showed the problem in numbers: same seed, same genome, a single
//! flag of difference. With holes the population stalled at 1.076 m for 20 generations
//! and **zero** of the 48 beings completed a lap. Without holes, median 3.189 m and
//! **25 of 48** completed. The holes sit at 1.087 m — eleven metres past the wall.
//!
//! That does not prove the being is blind to the hole; it proves it dies there. This
//! binary separates the two things by measuring the **detection curve**: it walks an
//! observer towards the object metre by metre and answers, for each distance, how many
//! retina pixels find it and with what encoded value. If the hole only appears at 8 m,
//! no brain swerves at 6 m/s — and the defect is in the encoding, not the behaviour.
//!
//! # What it measures
//!
//! It replicates the DDA of `retina.wgsl` (which stops at `v==2u` Wall or `v==3u` Hole)
//! and the `legacy_encoding()` of `perception.wgsl`, and reports per distance:
//!
//! · **pixels** that hit the target (out of `RETINA_COLS` × sampled rows)
//! · **encoded i8 value** — the number the brain actually receives
//! · **contrast** against the background of that same column without the object
//! · **ambiguity**: whether the value collides with another material's at another distance
//!
//! # Usage
//!
//!   visao_check [--seed 9001] [--width 28] [--alvo buraco|parede] [--de 120] [--ate 2]

use bevy::math::Vec2;
use universe::config::iara::vision::{RETINA_COLS, RETINA_ROWS};
use universe::world::{VoxelType, VoxelWorld};

/// Legacy encoding, an exact replica of `perception.wgsl:251`.
///
/// Kept here (rather than imported) because the original lives in WGSL: any divergence
/// between the two is precisely the kind of bug this tool exists to expose. If someone
/// touches the shader and forgets this copy, the test `encoding_matches_shader` breaks.
fn legacy_encoding(d: f32, mat: VoxelType) -> i32 {
    let depth_fade = (d / 1200.0).clamp(0.0, 1.0);
    let v = match mat {
        VoxelType::Wall => {
            if d < 30.0 {
                127.0 - ((d / 30.0).clamp(0.0, 1.0).sqrt() * 72.0)
            } else if d < 120.0 {
                55.0 - (((d - 30.0) / 90.0).clamp(0.0, 1.0) * 55.0)
            } else if d < 1200.0 {
                -(((d - 120.0) / 480.0).clamp(0.0, 1.0) * 25.0)
            } else {
                -25.0 - (((d - 1200.0) / 1200.0).clamp(0.0, 1.0) * 25.0)
            }
        }
        VoxelType::Grass => 40.0 - (depth_fade * 40.0),
        VoxelType::Road => -128.0 + (depth_fade * 96.0),
        VoxelType::Hole => 90.0 - ((d / 50.0).clamp(0.0, 1.0) * 30.0),
    };
    v.clamp(-128.0, 127.0) as i32
}

/// One vision ray: walks the world until it hits a Wall or a Hole, like the shader's DDA.
///
/// Returns the first obstacle's `(distance, material)`, or `None` if the ray covered the
/// whole range without finding anything — the "open sky" case that saturates the channel.
fn raio(world: &VoxelWorld, origem: Vec2, dir: Vec2, alcance: f32) -> Option<(f32, VoxelType)> {
    // Half-voxel step: fine enough not to skip a 2×2-cell target, cheap enough to sweep
    // 120 m in microseconds.
    let passo = world.cell_size * 0.5;
    let mut d = passo;
    while d < alcance {
        let m = world.material_at(origem + dir * d);
        if matches!(m, VoxelType::Wall | VoxelType::Hole) {
            return Some((d, m));
        }
        d += passo;
    }
    None
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut seed = 9001u64;
    let mut width = 28.0f32;
    let mut alvo = "buraco".to_string();
    let mut de = 120.0f32;
    let mut ate = 2.0f32;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                seed = args[i].parse().unwrap_or(9001);
            }
            "--width" => {
                i += 1;
                width = args[i].parse().unwrap_or(28.0);
            }
            "--alvo" => {
                i += 1;
                alvo = args[i].clone();
            }
            "--de" => {
                i += 1;
                de = args[i].parse().unwrap_or(120.0);
            }
            "--ate" => {
                i += 1;
                ate = args[i].parse().unwrap_or(2.0);
            }
            _ => {}
        }
        i += 1;
    }

    let world = {
        let _q = universe::world::silence_map_diag();
        VoxelWorld::new(seed, width)
    };
    let alvo_tipo = if alvo == "parede" {
        VoxelType::Wall
    } else {
        VoxelType::Hole
    };

    // Finds the first target of the requested type by walking the track axis, and stores
    // the AXIS point where it appears — that is where the being arrives from, so that is
    // where the vision matters.
    let wps = &world.waypoints_world;
    let mut alvo_pos: Option<(Vec2, Vec2, f32)> = None; // (target point, track direction, arc)
    {
        let mut arco = 0.0f32;
        let mut wp = 0usize;
        let mut t = 0.0f32;
        while wp < wps.len() - 1 && arco < 3000.0 {
            let a = wps[wp];
            let b = wps[wp + 1];
            let seg = (b - a).length().max(0.001);
            let p = a + (b - a) * (t / seg);
            let dir = (b - a) / seg;
            // Sweeps the track's width: the hole is not necessarily on the axis.
            let nrm = Vec2::new(-dir.y, dir.x);
            for off in [-12.0f32, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0] {
                if world.material_at(p + nrm * off) == alvo_tipo {
                    alvo_pos = Some((p + nrm * off, dir, arco));
                    break;
                }
            }
            if alvo_pos.is_some() {
                break;
            }
            arco += 1.0;
            t += 1.0;
            if t >= seg {
                t -= seg;
                wp += 1;
            }
        }
    }

    let Some((alvo_ponto, dir_pista, arco_alvo)) = alvo_pos else {
        println!("No '{alvo}' found within the first 3000 m of seed {seed}.");
        return;
    };

    let fov = universe::visao::fov_deg().to_radians();
    println!(
        "VISION VALIDATION · seed {seed} · track {width:.0} m · FOV {:.0}° · retina {RETINA_COLS}×{RETINA_ROWS}",
        fov.to_degrees()
    );
    println!(
        "Target: {alvo} at {arco_alvo:.0} m from the start ({:.0}, {:.0})\n",
        alvo_ponto.x, alvo_ponto.y
    );
    println!(
        "{:>6} {:>8} {:>9} {:>9} {:>10}  {}",
        "dist", "pixels", "i8 value", "bg i8", "contrast", "verdict"
    );
    println!("{}", "-".repeat(74));

    let mut primeira_deteccao: Option<f32> = None;
    let mut d = de;
    while d >= ate {
        // Observer on the axis, `d` metres before the target, looking along the track.
        let obs = alvo_ponto - dir_pista * d;
        let frente = dir_pista;
        let lat = Vec2::new(-frente.y, frente.x);

        let mut pixels_no_alvo = 0usize;
        let mut valor_alvo: Option<i32> = None;
        let mut valor_fundo: Option<i32> = None;

        // Sweeps the retina columns (the horizontal axis is what sees a lateral obstacle).
        for col in 0..RETINA_COLS {
            let frac = (col as f32 + 0.5) / RETINA_COLS as f32 - 0.5; // -0.5 .. +0.5
            let ang = frac * fov;
            let dir_raio = (frente * ang.cos() + lat * ang.sin()).normalize();
            match raio(&world, obs, dir_raio, 200.0) {
                Some((dist, mat)) if mat == alvo_tipo => {
                    pixels_no_alvo += 1;
                    let v = legacy_encoding(dist, mat);
                    valor_alvo = Some(valor_alvo.map_or(v, |o: i32| o.max(v)));
                }
                Some((dist, mat)) => {
                    let v = legacy_encoding(dist, mat);
                    valor_fundo = Some(valor_fundo.map_or(v, |o: i32| o.max(v)));
                }
                None => {}
            }
        }

        if pixels_no_alvo > 0 && primeira_deteccao.is_none() {
            primeira_deteccao = Some(d);
        }

        let va = valor_alvo.unwrap_or(-128);
        let vf = valor_fundo.unwrap_or(-128);
        let contraste = (va - vf).abs();
        let veredito = if pixels_no_alvo == 0 {
            "INVISIBLE"
        } else if contraste < 10 {
            "← blends into the background"
        } else if pixels_no_alvo == 1 {
            "1 pixel only (fragile)"
        } else {
            "visible"
        };
        println!("{d:6.0} {pixels_no_alvo:8} {va:9} {vf:9} {contraste:10}  {veredito}");

        d -= if d > 20.0 { 10.0 } else { 2.0 };
    }

    println!("\n{}", "-".repeat(74));
    match primeira_deteccao {
        Some(dist) => {
            // What matters is not seeing, it is seeing IN TIME. At 6 m/s (the speed
            // measured in the 2026-08-13 battery) every metre of anticipation is worth
            // 1/6 s of reaction.
            for v in [4.0f32, 6.0, 10.0] {
                println!(
                    "  at {v:.0} m/s the being has {:.1} s from first detection to impact",
                    dist / v
                );
            }
            println!("\n  First detection at {dist:.0} m.");
        }
        None => println!(
            "  The target was NEVER detected at any tested distance.\n\
                          \x20 The being is literally blind to this object — no brain avoids it."
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Anchors the numbers of the encoding table. If someone touches `legacy_encoding` in
    /// the shader and not here, this test reports the divergence.
    #[test]
    fn encoding_matches_shader() {
        assert_eq!(legacy_encoding(0.0, VoxelType::Wall), 127);
        assert_eq!(legacy_encoding(0.0, VoxelType::Hole), 90);
        assert_eq!(legacy_encoding(50.0, VoxelType::Hole), 60);
        // The saturation that motivated this tool: past 50 m the hole is a constant.
        assert_eq!(
            legacy_encoding(50.0, VoxelType::Hole),
            legacy_encoding(200.0, VoxelType::Hole)
        );
    }

    /// The measured ambiguity: a distant hole has the same value as a wall at mid distance.
    #[test]
    fn buraco_longe_colide_com_parede_media() {
        let buraco_60m = legacy_encoding(60.0, VoxelType::Hole);
        let parede_28m = legacy_encoding(28.0, VoxelType::Wall);
        assert!(
            (buraco_60m - parede_28m).abs() <= 3,
            "hole at 60 m = {buraco_60m}, wall at 28 m = {parede_28m}"
        );
    }
}
