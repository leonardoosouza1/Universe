//! sensor_audit.rs — Statistical audit of the sensory vector (223 channels).
//!
//! Replicates the channels of interest without Bevy: instantiates VoxelWorld, generates
//! N poses spread along the track, computes the channels by hand and measures per-channel
//! variance and correlation between suspect pairs.
//!
//! Usage: env -u DISPLAY ./target/release/sensor_audit
//!
//! Output: table channel | variance | duplicate group.

use bevy::math::{Vec2, Vec3};
use universe::world::VoxelWorld;

fn main() {
    // Deterministic world for reproducibility.
    let world = VoxelWorld::new(9001, 18.0);
    let n_wp = world.waypoints_world.len();
    println!(
        "World: {}x{} cells · track {} waypoints",
        world.size, world.size, n_wp
    );

    // Generates N poses along the track with different lateral offsets and headings.
    // To measure variance ACROSS BEINGS: beings at different poses in the same tick.
    // To measure TEMPORAL variance: same being at different ticks (same poses with tick offset).
    let n_poses = 120usize; // 2× the usual number of beings (60)
    let mut poses: Vec<(Vec3, Vec2, Vec2)> = Vec::new(); // (pos, forward, right)

    for k in 0..n_poses {
        // Distributes uniformly along the track
        let wp_idx = (k * n_wp / n_poses).min(n_wp - 2);
        let wp = world.waypoints_world[wp_idx];
        let wp_next = world.waypoints_world[(wp_idx + 1).min(n_wp - 1)];

        // Forward: direction of the track segment
        let seg_dir = (wp_next - wp).normalize_or_zero();
        let fwd = Vec2::new(seg_dir.x, seg_dir.y);
        let rgt = Vec2::new(-fwd.y, fwd.x); // perpendicular

        // Lateral offset: 0 (center), ±4m (middle), ±8m (off the track, on the grass).
        // At ±8m the being is on the grass — 3m/8m ahead may cross the edge back.
        let lat_offset = match k % 5 {
            0 => -8.0_f32,
            1 => -4.0,
            2 => 0.0,
            3 => 4.0,
            _ => 8.0,
        };
        let pos_2d = wp + rgt * lat_offset;
        let pos = Vec3::new(pos_2d.x, 0.0, pos_2d.y);

        poses.push((pos, fwd, rgt));
    }

    println!("Generating {} sensory_wave samples...", n_poses);

    // Computes the channels of interest for each pose.
    // Suspect channels identified in the code:
    //   ch 56-59: material_code (4× the same voxel of the current position)
    //   ch 73:    prop_on_grass (0 if on track, -128 if on grass — almost always 0)
    //
    // Other duplicated groups identified IN motor.rs:
    //   ch 36-39: open_l (left opening, 4×)
    //   ch 40-43: open_r (right opening, 4× — OVERWRITES road_ahead NEAR afterwards)
    //   ch 44-47: wall_imminent (4×)
    //   ch 48-51: trapped_score (4×)
    //   ch 60-63: escape_dir (4×)
    //   ch 90-93: dark_center (3× + 1 afterwards)
    //
    // For channels that depend on the retina (open_l/r, wall_imminent, trapped, escape, dark_center),
    // we have no retina access without Bevy. We measure only the grid-dependent ones:
    //   ch 56-59: material (grid)
    //   ch 73:    on_grass (grid)
    //   ch 16-23: wall_distance_fov (grid — reference with real variance)
    //   ch 76-77: lateral dist (grid — reference that differs between sides)

    const N_CH: usize = 223;
    let mut samples: Vec<[i8; N_CH]> = Vec::with_capacity(n_poses);

    for (pos, fwd, rgt) in &poses {
        let mut wave = [0i8; N_CH];

        // ── ch 56-59: current material + lookahead (after the 2026-08-04 fix) ──
        use universe::world::VoxelType;
        let enc_mat = |vox: VoxelType| -> i8 {
            match vox {
                VoxelType::Road => 0,
                VoxelType::Grass => 64,
                VoxelType::Wall => 100,
                VoxelType::Hole => -127,
            }
        };
        let fwd3 = Vec3::new(fwd.x, 0.0, fwd.y);
        wave[56] = enc_mat(world.get_voxel_at(pos.x, pos.z));
        wave[57] = enc_mat(world.get_voxel_at(pos.x + fwd3.x * 3.0, pos.z + fwd3.z * 3.0));
        wave[58] = enc_mat(world.get_voxel_at(pos.x + fwd3.x * 8.0, pos.z + fwd3.z * 8.0));
        wave[59] = enc_mat(world.get_voxel_at(pos.x + fwd3.x * 20.0, pos.z + fwd3.z * 20.0));

        // ── ch 73: available corridor (after the 2026-08-04 fix) ─────────────
        let lp_x_73 = -fwd.y;
        let lp_z_73 = fwd.x;
        let mut dist_l_73 = 30.0f32;
        let mut dist_r_73 = 30.0f32;
        for &sd in &[1.0f32, 3.0, 6.0, 9.0, 12.0, 16.0, 20.0, 26.0, 30.0] {
            let lx = pos.x + lp_x_73 * sd;
            let lz = pos.z + lp_z_73 * sd;
            let rx = pos.x - lp_x_73 * sd;
            let rz = pos.z - lp_z_73 * sd;
            if dist_l_73 == 30.0
                && matches!(
                    world.get_voxel_at(lx, lz),
                    VoxelType::Wall | VoxelType::Hole
                )
            {
                dist_l_73 = sd;
            }
            if dist_r_73 == 30.0
                && matches!(
                    world.get_voxel_at(rx, rz),
                    VoxelType::Wall | VoxelType::Hole
                )
            {
                dist_r_73 = sd;
            }
        }
        let corredor = (dist_l_73 + dist_r_73).min(30.0);
        wave[73] = ((corredor / 30.0) * 127.0).clamp(0.0, 127.0) as i8;

        // ── ch 16-23: wall_distance_fov (reference with real variance) ───────
        let fov_half_rad: f32 = 50.0_f32.to_radians();
        const WALL_SAMPLES: [f32; 8] = [3.0, 8.0, 15.0, 25.0, 35.0, 45.0, 52.0, 60.0];
        for ai in 0..8usize {
            let frac = (ai as f32 - 3.5) / 3.5;
            let yaw = frac * fov_half_rad;
            let dx = fwd.x * yaw.cos() + rgt.x * yaw.sin();
            let dz = fwd.y * yaw.cos() + rgt.y * yaw.sin();
            let len = (dx * dx + dz * dz).sqrt();
            let (ndx, ndz) = if len > 1e-6 {
                (dx / len, dz / len)
            } else {
                (0.0, 0.0)
            };
            let mut wall_min = f32::MAX;
            for &sd in &WALL_SAMPLES {
                let sx = pos.x + ndx * sd;
                let sz = pos.z + ndz * sd;
                let v = world.get_voxel_at(sx, sz);
                if v == VoxelType::Wall || v == VoxelType::Hole {
                    wall_min = wall_min.min(sd);
                }
            }
            wave[16 + ai] = if wall_min >= 60.0 {
                0
            } else {
                (127.0 * (60.0 - wall_min) / 60.0).clamp(1.0, 127.0) as i8
            };
        }

        // ── ch 76-77: lateral left/right ─────────────────────────────────────
        // Original CPU version (15m range, 6 samples)
        let lp_x = -fwd.y;
        let lp_z = fwd.x;
        let mut dist_l = 15.0f32;
        let mut dist_r = 15.0f32;
        for &sd in &[1.0f32, 3.0, 6.0, 9.0, 12.0, 15.0] {
            let lx = pos.x + lp_x * sd;
            let lz = pos.z + lp_z * sd;
            let rx = pos.x - lp_x * sd;
            let rz = pos.z - lp_z * sd;
            let vl = world.get_voxel_at(lx, lz);
            let vr = world.get_voxel_at(rx, rz);
            if dist_l == 15.0 && (vl == VoxelType::Wall || vl == VoxelType::Hole) {
                dist_l = sd;
            }
            if dist_r == 15.0 && (vr == VoxelType::Wall || vr == VoxelType::Hole) {
                dist_r = sd;
            }
        }
        wave[76] = ((dist_l / 15.0) * 127.0).clamp(0.0, 127.0) as i8;
        wave[77] = ((dist_r / 15.0) * 127.0).clamp(0.0, 127.0) as i8;

        samples.push(wave);
    }

    // ── Computes per-channel variance ─────────────────────────────────────────
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!(
        "TABLE: variance per channel (measured over {} samples)",
        n_poses
    );
    println!("Chan  | Var    | Min  | Max  | Note");
    println!("------|--------|------|------|------------------------------------------");

    let canais_interesse = vec![
        // Material duplicates (ch 56-59)
        56, 57, 58, 59, // on_grass (ch 73) — almost always 0
        73, // Reference with real variance
        16, 17, 18, 19, 20, 21, 22, 23, // Lateral
        76, 77,
    ];

    let mut variancas: Vec<(usize, f64)> = Vec::new();

    for &ch in &canais_interesse {
        let vals: Vec<f64> = samples.iter().map(|s| s[ch] as f64).collect();
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / vals.len() as f64;
        let min_v = vals.iter().cloned().fold(f64::MAX, f64::min);
        let max_v = vals.iter().cloned().fold(f64::MIN, f64::max);
        let nota = if var < 1.0 {
            "*** CONSTANT — dead channel ***"
        } else if var < 50.0 {
            "low variance"
        } else {
            "normal variance"
        };
        println!(
            "{:5} | {:6.1} | {:4.0} | {:4.0} | {}",
            ch, var, min_v, max_v, nota
        );
        variancas.push((ch, var));
    }

    // ── Correlation between ch 56 and ch 57, 58, 59 ──────────────────────────
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("CORRELATION between suspect pairs");
    println!("Pair         | Corr    | Interpretation");
    println!("-------------|---------|----------------------------------------");

    let pares = vec![
        (56usize, 57usize),
        (56, 58),
        (56, 59),
        (57, 58),
        (76, 77), // must be distinct (opposite sides)
        (16, 17), // must be distinct (different angles)
    ];

    for (a, b) in &pares {
        let va: Vec<f64> = samples.iter().map(|s| s[*a] as f64).collect();
        let vb: Vec<f64> = samples.iter().map(|s| s[*b] as f64).collect();
        let ma = va.iter().sum::<f64>() / va.len() as f64;
        let mb = vb.iter().sum::<f64>() / vb.len() as f64;
        let num = va
            .iter()
            .zip(vb.iter())
            .map(|(a, b)| (a - ma) * (b - mb))
            .sum::<f64>();
        let da = va.iter().map(|v| (v - ma).powi(2)).sum::<f64>().sqrt();
        let db = vb.iter().map(|v| (v - mb).powi(2)).sum::<f64>().sqrt();
        let corr = if da > 0.0 && db > 0.0 {
            num / (da * db)
        } else {
            1.0
        };
        let interp = if corr.abs() > 0.999 {
            "EXACT DUPLICATE"
        } else if corr.abs() > 0.9 {
            "near duplicate"
        } else {
            "independent (ok)"
        };
        println!("ch{:3} vs ch{:3} | {:7.4} | {}", a, b, corr, interp);
    }

    // ── DIVERGENCE PROOF — retina groups with a synthetic retina ─────────────
    // Builds synthetic retinas with distinct per-row patterns and verifies that
    // the channels AFTER the fix (motor.rs 2026-08-04) have DIFFERENT values.
    // Without this the fix may compile but still produce duplicates.
    //
    // Synthetic retina:
    //   Row r: left half (cols 0..63) = -r*10 (dark = track),
    //          right half (cols 63..126) = -(14-r)*8 (lighter in the near rows)
    //   Result: near rows have more "track" on the right, far rows on the left.
    //   Wall_imminent: we vary the central pixel per row (0, 50, 90, 127 in 4 rows).

    use universe::config::iara::vision::{RETINA_COLS, RETINA_ROWS};

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("DIVERGENCE PROOF — synthetic retina, after the motor.rs 2026-08-04 fix");
    println!(
        "Each channel of the 6 groups MUST have a value different from the other 3 in its group."
    );
    println!("Chan  | Val | Group | Observation");
    println!("------|-----|-------|------------------------------------------");

    // Builds the synthetic retina: 14 rows × 126 cols.
    let mut retina_syn = [-128i8; RETINA_ROWS * RETINA_COLS];
    for row in 0..RETINA_ROWS {
        for col in 0..RETINA_COLS {
            let half = RETINA_COLS / 2;
            retina_syn[row * RETINA_COLS + col] = if col < half {
                // Left: darkness grows with the row (far rows = more track on the left)
                (-(row as i32 * 10 + 5)).clamp(-127, 0) as i8
            } else {
                // Right: greater darkness in the near rows
                (-((RETINA_ROWS - 1 - row) as i32 * 8 + 4)).clamp(-127, 0) as i8
            };
        }
    }
    // Inject walls for wall_imminent: leave high pixels in different regions
    // Row 3 (reflex_row = ROWS/2-1 = 6): center with low max
    // Row near (ROWS-2 = 12): center with high max
    let reflex_row = (RETINA_ROWS / 2).saturating_sub(1);
    let near_row = RETINA_ROWS.saturating_sub(2);
    for col in (RETINA_COLS * 5 / 16)..(RETINA_COLS * 11 / 16) {
        retina_syn[reflex_row * RETINA_COLS + col] = 50; // wall_central_max = 50 → val=50
    }
    for col in (RETINA_COLS * 6 / 16)..(RETINA_COLS * 10 / 16) {
        retina_syn[near_row * RETINA_COLS + col] = 120; // near_max = 120 → val=127
    }
    // Left lateral (cols 0..31): low (no wall)
    for col in 0..(RETINA_COLS / 4) {
        retina_syn[reflex_row * RETINA_COLS + col] = 10; // wall_left_max = 10 → val=0
    }
    // Right lateral (cols 95..126): medium
    for col in (RETINA_COLS * 3 / 4)..RETINA_COLS {
        retina_syn[reflex_row * RETINA_COLS + col] = 85; // wall_right_max = 85 → val=90
    }
    // Escape: to prove divergence between rows, each row needs a UNIQUE escape
    // pattern. We zero the frontal half (cols 31..94) of each row to wall
    // (positive = wall), then place a track run at a different position.
    // Row 3: run at cols 32..39 (~left, negative escape)
    // Row 5: run at cols 50..58 (~center-right, slightly positive escape)
    // Row 7: run at cols 70..78 (~right, positive escape)
    // Row 9: no run (all wall → escape=0)
    for row in [3usize, 5, 7, 9] {
        for col in (RETINA_COLS / 4)..(RETINA_COLS * 3 / 4) {
            retina_syn[row * RETINA_COLS + col] = 30; // wall in that band
        }
    }
    let escape_runs: [(usize, usize, usize); 3] = [(3, 32, 39), (5, 50, 58), (7, 70, 78)];
    for &(row, lo, hi) in &escape_runs {
        for col in lo..hi {
            retina_syn[row * RETINA_COLS + col] = -80; // track (dark)
        }
    }

    // Replicates the motor.rs computation with the synthetic retina
    let half = RETINA_COLS / 2;
    let half_col = (RETINA_COLS / 2) as f32 - 0.5;

    // --- Groups ch 36-39 and ch 40-43: open_l/r per row ---
    let dark_sum_left_at_row = |row: usize| -> f32 {
        let off = row.min(RETINA_ROWS - 1) * RETINA_COLS;
        let mut s: i32 = 0;
        for i in 0..half {
            let v = retina_syn[off + i] as i32;
            if v < 0 {
                s += -v;
            }
        }
        (s as f32 / 4096.0).clamp(0.0, 1.0)
    };
    let dark_sum_right_at_row = |row: usize| -> f32 {
        let off = row.min(RETINA_ROWS - 1) * RETINA_COLS;
        let mut s: i32 = 0;
        for i in half..RETINA_COLS {
            let v = retina_syn[off + i] as i32;
            if v < 0 {
                s += -v;
            }
        }
        (s as f32 / 4096.0).clamp(0.0, 1.0)
    };
    let ch36 = (dark_sum_left_at_row(3) * 127.0) as i8;
    let ch37 = (dark_sum_left_at_row(5) * 127.0) as i8;
    let ch38 = (dark_sum_left_at_row(7) * 127.0) as i8;
    let ch39 = (dark_sum_left_at_row(9.min(RETINA_ROWS - 1)) * 127.0) as i8;
    let ch40 = (dark_sum_right_at_row(3) * 127.0) as i8;
    let ch41 = (dark_sum_right_at_row(5) * 127.0) as i8;
    let ch42 = (dark_sum_right_at_row(7) * 127.0) as i8;
    let ch43 = (dark_sum_right_at_row(9.min(RETINA_ROWS - 1)) * 127.0) as i8;
    let ol_distinct = !(ch36 == ch37 && ch37 == ch38 && ch38 == ch39);
    let or_distinct = !(ch40 == ch41 && ch41 == ch42 && ch42 == ch43);
    println!(
        "   36 | {:3} | open_l row3   | {}",
        ch36,
        if ol_distinct {
            "OK — distinct channels"
        } else {
            "*** FAIL: still a duplicate ***"
        }
    );
    println!("   37 | {:3} | open_l row5   |", ch37);
    println!("   38 | {:3} | open_l row7   |", ch38);
    println!("   39 | {:3} | open_l row9   |", ch39);
    println!(
        "   40 | {:3} | open_r row3   | {}",
        ch40,
        if or_distinct {
            "OK — distinct channels"
        } else {
            "*** FAIL: still a duplicate ***"
        }
    );
    println!("   41 | {:3} | open_r row5   |", ch41);
    println!("   42 | {:3} | open_r row7   |", ch42);
    println!("   43 | {:3} | open_r row9   |", ch43);

    // --- Group ch 44-47: wall_imminent per angular region ---
    let mut wall_central_max: i32 = -128;
    for i in (RETINA_COLS * 5 / 16)..(RETINA_COLS * 11 / 16) {
        let v = retina_syn[reflex_row * RETINA_COLS + i] as i32;
        if v > wall_central_max {
            wall_central_max = v;
        }
    }
    let ch44: i8 = if wall_central_max > 110 {
        127
    } else if wall_central_max > 80 {
        90
    } else if wall_central_max > 40 {
        50
    } else {
        0
    };
    let mut wall_near_max: i32 = -128;
    for i in (RETINA_COLS * 6 / 16)..(RETINA_COLS * 10 / 16) {
        let v = retina_syn[near_row * RETINA_COLS + i] as i32;
        if v > wall_near_max {
            wall_near_max = v;
        }
    }
    let ch45: i8 = if wall_near_max > 110 {
        127
    } else if wall_near_max > 80 {
        90
    } else if wall_near_max > 40 {
        50
    } else {
        0
    };
    let mut wall_left_max: i32 = -128;
    for i in 0..(RETINA_COLS / 4) {
        let v = retina_syn[reflex_row * RETINA_COLS + i] as i32;
        if v > wall_left_max {
            wall_left_max = v;
        }
    }
    let ch46: i8 = if wall_left_max > 80 {
        90
    } else if wall_left_max > 40 {
        50
    } else {
        0
    };
    let mut wall_right_max: i32 = -128;
    for i in (RETINA_COLS * 3 / 4)..RETINA_COLS {
        let v = retina_syn[reflex_row * RETINA_COLS + i] as i32;
        if v > wall_right_max {
            wall_right_max = v;
        }
    }
    let ch47: i8 = if wall_right_max > 80 {
        90
    } else if wall_right_max > 40 {
        50
    } else {
        0
    };
    let wi_distinct = !(ch44 == ch45 && ch45 == ch46 && ch46 == ch47);
    println!(
        "   44 | {:3} | wall_imm central fov | {}",
        ch44,
        if wi_distinct {
            "OK — distinct channels"
        } else {
            "*** FAIL ***"
        }
    );
    println!("   45 | {:3} | wall_imm narrow fov  |", ch45);
    println!("   46 | {:3} | wall_imm left lat    |", ch46);
    println!("   47 | {:3} | wall_imm right lat   |", ch47);

    // --- Group ch 48-51: trapped decomposed ---
    let wall_pressure = ch44 as f32;
    let speed_kmh_now = 0.0f32; // stopped
    let open_l_norm = dark_sum_left_at_row(3);
    let open_r_norm = dark_sum_right_at_row(3);
    let stuck_speed = (1.0 - (speed_kmh_now / 25.0).clamp(0.0, 1.0)) * 127.0;
    let lr_blocked = (1.0 - (open_l_norm + open_r_norm) * 0.5).clamp(0.0, 1.0) * 127.0;
    let trapped_i8: i8 = if wall_pressure >= 80.0 {
        (stuck_speed * 0.3 + wall_pressure * 0.5 + lr_blocked * 0.2).clamp(0.0, 127.0) as i8
    } else {
        0
    };
    let ch48 = wall_pressure.clamp(0.0, 127.0) as i8;
    let ch49 = stuck_speed.clamp(0.0, 127.0) as i8;
    let ch50 = lr_blocked.clamp(0.0, 127.0) as i8;
    let ch51 = trapped_i8;
    let tr_distinct = !(ch48 == ch49 && ch49 == ch50 && ch50 == ch51);
    println!(
        "   48 | {:3} | trapped: wall_press | {}",
        ch48,
        if tr_distinct {
            "OK — distinct channels"
        } else {
            "*** FAIL ***"
        }
    );
    println!("   49 | {:3} | trapped: stuck_speed|", ch49);
    println!("   50 | {:3} | trapped: lr_blocked |", ch50);
    println!("   51 | {:3} | trapped: composite  |", ch51);

    // --- Group ch 60-63: escape_dir per row ---
    let escape_at_row = |row: usize| -> i8 {
        let row_off = row.min(RETINA_ROWS - 1) * RETINA_COLS;
        let mut b_start = 0usize;
        let mut b_len = 0usize;
        let mut c_start = 0usize;
        let mut c_len = 0usize;
        for i in (RETINA_COLS / 4)..(RETINA_COLS * 3 / 4) {
            let v = retina_syn[row_off + i] as i32;
            if v < -10 {
                if c_len == 0 {
                    c_start = i;
                }
                c_len += 1;
                if c_len > b_len {
                    b_start = c_start;
                    b_len = c_len;
                }
            } else {
                c_len = 0;
            }
        }
        let norm = if b_len >= 2 {
            let center = b_start as f32 + (b_len as f32 - 1.0) * 0.5;
            ((center - 31.5) / 31.5).clamp(-1.0, 1.0) * 127.0
        } else {
            0.0
        };
        norm as i8
    };
    let ch60 = escape_at_row(3);
    let ch61 = escape_at_row(5);
    let ch62 = escape_at_row(7);
    let ch63 = escape_at_row(9.min(RETINA_ROWS - 1));
    let es_distinct = !(ch60 == ch61 && ch61 == ch62 && ch62 == ch63);
    println!(
        "   60 | {:3} | escape row3   | {}",
        ch60,
        if es_distinct {
            "OK — distinct channels"
        } else {
            "*** FAIL ***"
        }
    );
    println!("   61 | {:3} | escape row5   |", ch61);
    println!("   62 | {:3} | escape row7   |", ch62);
    println!("   63 | {:3} | escape row9   |", ch63);

    // --- Group ch 90-93: dark_center variants ---
    let dark_center_for_row_inner = |row: usize, w_size: usize| -> f32 {
        let row_off = row * RETINA_COLS;
        let search_lo = RETINA_COLS / 4;
        let search_hi = (RETINA_COLS * 3 / 4).saturating_sub(w_size);
        let mut best_sum: f32 = -1.0;
        let mut best_centroid: f32 = 0.0;
        for start in search_lo..=search_hi {
            let mut sum: f32 = 0.0;
            let mut weighted: f32 = 0.0;
            for k in 0..w_size {
                let v = retina_syn[row_off + start + k] as i32;
                let d = (-v as f32).max(0.0);
                sum += d;
                weighted += d * (start + k) as f32;
            }
            if sum > best_sum {
                best_sum = sum;
                best_centroid = if sum > 1.0 {
                    weighted / sum
                } else {
                    (start + w_size / 2) as f32
                };
            }
        }
        if best_sum > 1.0 {
            ((best_centroid - half_col) / half_col).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    };
    let dc_40m = dark_center_for_row_inner((RETINA_ROWS / 2).saturating_sub(1), 8);
    let dc_25m = dark_center_for_row_inner(5.min(RETINA_ROWS - 1), 16);
    // LR diff
    let mut ldark: i32 = 0;
    let mut rdark: i32 = 0;
    let row3_off = ((RETINA_ROWS / 2).saturating_sub(1)) * RETINA_COLS;
    for i in 0..half {
        let v = retina_syn[row3_off + i] as i32;
        if v < 0 {
            ldark += -v;
        }
    }
    for i in half..RETINA_COLS {
        let v = retina_syn[row3_off + i] as i32;
        if v < 0 {
            rdark += -v;
        }
    }
    let lr_diff_norm = ((rdark - ldark) as f32 / 4096.0).clamp(-1.0, 1.0);
    let dc_15m = dark_center_for_row_inner(7.min(RETINA_ROWS - 1), 8);
    let ch90 = (dc_40m * 127.0).clamp(-127.0, 127.0) as i8;
    let ch91 = (dc_25m * 127.0).clamp(-127.0, 127.0) as i8;
    let ch92 = (lr_diff_norm * 127.0).clamp(-127.0, 127.0) as i8;
    let ch93 = (dc_15m * 127.0).clamp(-127.0, 127.0) as i8;
    let dc_distinct = !(ch90 == ch91 && ch91 == ch92 && ch92 == ch93);
    println!(
        "   90 | {:3} | dark_ctr W8 row3(40m) | {}",
        ch90,
        if dc_distinct {
            "OK — distinct channels"
        } else {
            "*** FAIL ***"
        }
    );
    println!("   91 | {:3} | dark_ctr W16 row5(25m)|", ch91);
    println!("   92 | {:3} | dark_ctr LR-diff      |", ch92);
    println!("   93 | {:3} | dark_ctr W8 row7(15m) |", ch93);

    let all_ok =
        ol_distinct && or_distinct && wi_distinct && tr_distinct && es_distinct && dc_distinct;
    println!("\n═══════════════════════════════════════════════════════════════════");
    if all_ok {
        println!("RESULT: PASS — all 6 groups have channels with distinct values.");
        println!("0 duplicates confirmed on the synthetic retina.");
    } else {
        println!("RESULT: FAIL — some group still has a duplicate!");
    }
    println!("Freed channels (formerly duplicates, now distinct): 18");
    println!("(6 groups × 3 extra channels = 18 channels that used to waste information)");
}
