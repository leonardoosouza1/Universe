//! Unit tests for `hip_check`. NO GPU, no file system: every fixture is a synthetic f32
//! stream built by the same formula `src/bin/gpu_evolve/context.rs:2222-2240` writes, so
//! the tests pin the CONTRACT (column layout, post-authority factor, corruption caught)
//! and not a recorded file that may not be on this machine.
//!
//! Split out of `main.rs` to keep both files under the 600-line cap of `docs/STYLE.md` §2,
//! the same way `src/bin/gpu_evolve/` splits `tests.rs` and `laps_tests.rs`.

use super::*;

/// 29 distinct non-zero genes, so a wrongly sliced `W` cannot pass by accident — an
/// all-ones `W` would make every output row identical (rule 2 of this project's
/// CLAUDE.md: identical numbers where they should differ are an alarm).
fn synthetic_genes() -> Vec<f32> {
    (0..hip::N_HIP_GENES)
        .map(|k| 0.11 + 0.037 * k as f32)
        .collect()
}

/// One row the way `context.rs` writes it, with the bias filled by the SAME formula
/// the run applies (post-authority).
fn push_row(out: &mut Vec<f32>, genes: &[f32], tick: usize, sector: usize, mem: &[f32; 8]) {
    let mut x = [0.0f32; hip::HIP_READ_LEN];
    x[..8].copy_from_slice(mem);
    x[hip::HIP_READ_LEN - 1] = 1.0;
    let mut b = hip::bias(genes, &x);
    for v in b.iter_mut() {
        *v *= hip::HIP_AUTHORITY;
    }
    out.push(tick as f32);
    out.push(sector as f32);
    out.extend_from_slice(mem);
    out.extend_from_slice(&b);
}

/// `n` ticks walking the sectors in order, with a memory that never repeats a value.
fn synthetic_stream(genes: &[f32], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * hip::HIP_OBS_COLS);
    for t in 0..n {
        let mut mem = [0.0f32; 8];
        for (k, v) in mem.iter_mut().enumerate() {
            *v = ((t + k) as f32 * 0.017).sin();
        }
        push_row(&mut out, genes, t, t % hip::N_SECTORS, &mem);
    }
    out
}

#[test]
fn recomputation_matches_a_synthetic_recording_exactly() {
    let genes = synthetic_genes();
    let rows = parse_rows(&synthetic_stream(&genes, 200)).expect("well-formed stream");
    assert_eq!(rows.len(), 200);
    let rep = check(&genes, &rows);
    assert_eq!(rep.max_diff, 0.0, "same function, same order: must be exact");
    assert!(rep.max_abs_bias > 0.0, "the fixture must exercise the organ");
    assert_eq!(rep.rows, 200);
}

/// The gate has to be able to FAIL — a check that only ever passes proves nothing.
#[test]
fn a_corrupted_bias_value_is_caught() {
    let genes = synthetic_genes();
    let mut stream = synthetic_stream(&genes, 50);
    stream[30 * hip::HIP_OBS_COLS + COL_BIAS + 1] += 0.25; // steer of row 30
    let rep = check(&genes, &parse_rows(&stream).expect("well-formed, only wrong"));
    assert!(
        (rep.max_diff - 0.25).abs() < 1e-6,
        "expected the injected 0.25, got {}",
        rep.max_diff
    );
    assert_eq!(rep.worst_row, 30);
    assert!(rep.max_diff > DEFAULT_TOLERANCE, "must trip the gate");
}

/// Forgetting `HIP_AUTHORITY` is the plausible silent bug: a uniform 2× error.
#[test]
fn the_recorded_bias_is_post_authority() {
    let genes = synthetic_genes();
    let rows = parse_rows(&synthetic_stream(&genes, 10)).unwrap();
    let raw = hip::bias(&genes, &rows[3].read());
    for o in 0..hip::HIP_OUTPUTS {
        assert!(
            (rows[3].bias[o] - raw[o] * hip::HIP_AUTHORITY).abs() < 1e-7,
            "column {o} is not post-authority"
        );
    }
}

/// `gain == 0` must reproduce an all-zero recording bit for bit — the birth-neutral
/// contract that u0c generations 0, 1, 2 and 4 rely on.
#[test]
fn zero_gain_genome_reproduces_an_all_zero_recording() {
    let mut genes = synthetic_genes();
    genes[hip::GENE_READ_GAIN] = 0.0;
    let rows = parse_rows(&synthetic_stream(&genes, 40)).unwrap();
    assert_eq!(peak(&rows.iter().flat_map(|r| r.bias).collect::<Vec<_>>()), 0.0);
    assert_eq!(check(&genes, &rows).max_diff, 0.0);
}

#[test]
fn a_ragged_file_is_rejected() {
    let mut stream = synthetic_stream(&synthetic_genes(), 5);
    stream.pop(); // a write cut in the middle of the last row
    assert_eq!(
        parse_rows(&stream),
        Err(RowError::Ragged {
            floats: 5 * hip::HIP_OBS_COLS - 1
        })
    );
}

#[test]
fn an_impossible_sector_or_a_nan_is_rejected() {
    let genes = synthetic_genes();
    let mut stream = synthetic_stream(&genes, 5);
    stream[2 * hip::HIP_OBS_COLS + COL_SECTOR] = 99.0;
    assert_eq!(
        parse_rows(&stream),
        Err(RowError::BadSector {
            row: 2,
            value: 99.0
        })
    );
    let mut stream = synthetic_stream(&genes, 5);
    stream[hip::HIP_OBS_COLS + COL_MEM] = f32::NAN;
    assert_eq!(
        parse_rows(&stream),
        Err(RowError::NotFinite {
            row: 1,
            col: COL_MEM
        })
    );
}

/// The U1 split: memory zero on lap 1 and 0.5 on lap 2 must land on the right side of
/// the line, for every sector.
#[test]
fn sector_stats_split_the_first_pass_from_the_later_ones() {
    let genes = synthetic_genes();
    let mut stream = Vec::new();
    for lap in 0..2 {
        for s in 0..hip::N_SECTORS {
            let mem = [if lap == 0 { 0.0 } else { 0.5 }; 8];
            push_row(&mut stream, &genes, lap, s, &mem);
            push_row(&mut stream, &genes, lap, s, &mem);
        }
    }
    let rows = parse_rows(&stream).unwrap();
    for (s, st) in sector_stats(&rows).iter().enumerate() {
        assert_eq!((st.passes, st.ticks[0], st.ticks[1]), (2, 2, 2), "sector {s}");
        assert_eq!(st.here_sum[0], 0.0, "sector {s} first pass must be empty");
        assert_eq!(st.next_sum[0], 0.0, "sector {s} lookahead must be empty too");
        assert!(
            (mean(st.here_sum[1], st.ticks[1]) - 0.5).abs() < 1e-6,
            "sector {s} later pass must carry the memory"
        );
    }
    assert_eq!(check(&genes, &rows).max_diff, 0.0);
}

#[test]
fn the_generation_number_is_read_from_the_file_name() {
    assert_eq!(generation_in_name("/a/b/hip_gen0003.f32"), Some(3));
    assert_eq!(generation_in_name("evolve_best_gen0079.bin"), Some(79));
    assert_eq!(generation_in_name("champion.bin"), None);
}
