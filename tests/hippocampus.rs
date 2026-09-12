//! C1 acceptance tests — sector hippocampus (`docs/PRD-CEREBRO-NEUROMOD.md` §C1).
//!
//! No GPU, no device, no I/O: `gpu::hippocampus` is a pure module, which is the point of
//! it being a module at all. The two tests that need the genome on disk
//! (`genoma_sobrevive_ida_e_volta_no_disco` and its sibling below)
//! live in `src/bin/gpu_evolve/tests.rs`, next to the serializer they guard.
//!
//! Every test here prints the numbers it asserts on. A green test that shows no number
//! cannot tell a reader whether the mechanism moved by 1e-9 or by 0.4.

use universe::gpu::hippocampus::{
    ERASED_READ, HIP_OUTPUTS, HIP_READ_LEN, MEM_PER_SECTOR, N_HIP_GENES, N_SECTORS, SectorMemory,
    bias, read_gain, write_rate,
};

/// Deterministic pseudo-random floats in [-1, 1] — a local LCG rather than a dependency,
/// and one of `rand`'s own constants so the sequence is not pathological.
fn lcg_fill(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// (a) An all-zero genome gives a bias of exactly `[0, 0, 0]`, for any memory content.
///
/// This is THE birth-neutral contract (the P1.7 block of `evolve()`): generation 0 of
/// the `--hipocampo-setor` arm has to be the old brain, bit for bit, or the paired A/B is
/// comparing two different brains from the first tick. Asserting on the BITS and not just
/// on `== 0.0` matters because `-0.0 == 0.0` is true in Rust while the bits differ, and a
/// negative zero reaching the commands would be a silent difference between the arms.
#[test]
fn zero_genes_give_exactly_zero_bias_for_any_memory() {
    let genes = [0.0f32; N_HIP_GENES];
    let mut worst = 0u32;
    for trial in 0..100 {
        let vals = lcg_fill(0xC1_0000 + trial as u64, HIP_READ_LEN);
        let mut x = [0.0f32; HIP_READ_LEN];
        x.copy_from_slice(&vals);
        x[HIP_READ_LEN - 1] = 1.0; // the bias column is always 1.0 in a real read
        let b = bias(&genes, &x);
        for (o, v) in b.iter().enumerate() {
            assert_eq!(
                v.to_bits(),
                0u32,
                "trial {trial}, output {o}: bias {v} (bits {:#010x}) is not +0.0 — the \
                 birth-neutral contract is broken and generation 0 of the two arms would differ",
                v.to_bits()
            );
            worst = worst.max(v.to_bits());
        }
    }
    println!("(a) 100 random memories, all-zero genes: worst |bias| bit pattern = {worst:#010x}");
    assert_eq!(worst, 0);
}

/// (a2) The same contract through the real read path, including an already-written memory.
#[test]
fn zero_genes_stay_neutral_after_the_memory_is_written() {
    let genes = [0.0f32; N_HIP_GENES];
    let mut m = SectorMemory::new();
    for s in 0..N_SECTORS {
        m.write(s, &[0.9, 0.4, 0.3, -0.7], 1.0);
    }
    for s in 0..N_SECTORS {
        let b = bias(&genes, &m.read(s));
        assert_eq!(b, [0.0; HIP_OUTPUTS], "sector {s}");
    }
    println!("(a2) 24 written sectors, all-zero genes: bias [0, 0, 0] on every one");
}

/// (b) With real weights the organ MOVES all three commands, and harder as the gain grows.
///
/// The failure this guards is the one `neuromod-lab/src/testes_sentidos.py` would have
/// caught in 30 s and did not exist here (PRD §C3): a new input wired in so that it
/// cannot change the output. A block that is born neutral AND stays neutral is
/// indistinguishable from a block that is not wired at all, and the battery would spend
/// GPU hours measuring nothing.
#[test]
fn nonzero_weights_move_every_output_and_scale_with_the_gain() {
    let mut m = SectorMemory::new();
    m.write(5, &[0.80, 0.25, 0.10, 0.40], 1.0);
    m.write(6, &[0.35, 0.60, 0.05, -0.30], 1.0);
    let x = m.read(5);

    // W at ±0.5, as the PRD's acceptance criterion asks — but with a DIFFERENT sign
    // pattern per output row. The obvious version (`k % 2`) makes row 1 the exact
    // negative of row 0 and row 2 identical to it, so the three |bias| values come out
    // equal to the last digit: identical numbers where they should differ, which this
    // project treats as an alarm and not as a confirmation (CLAUDE.md rule 2). Here they
    // would have been an artefact of the fixture rather than a bug, which is worse — a
    // fixture that cannot tell three independent outputs from one output printed
    // three times.
    let mut genes = [0.0f32; N_HIP_GENES];
    for o in 0..HIP_OUTPUTS {
        for k in 0..HIP_READ_LEN {
            genes[o * HIP_READ_LEN + k] = if (k + o) % 3 == 0 { 0.5 } else { -0.5 };
        }
    }

    let gains = [0.25f32, 0.5, 1.0, 2.0];
    let mut prev = [0.0f32; HIP_OUTPUTS];
    for (step, &g) in gains.iter().enumerate() {
        genes[N_HIP_GENES - 1] = g;
        let b = bias(&genes, &x);
        println!(
            "(b) gene_gain={g:4.2} -> gain={:+.4}  bias = [{:+.6}, {:+.6}, {:+.6}]",
            read_gain(&genes),
            b[0],
            b[1],
            b[2]
        );
        for o in 0..HIP_OUTPUTS {
            assert!(
                b[o].abs() > 1e-3,
                "output {o} did not move: |bias| = {:.3e} <= 1e-3 — the block is wired \
                 but cannot change the command",
                b[o].abs()
            );
            if step > 0 {
                assert!(
                    b[o].abs() > prev[o].abs(),
                    "output {o}: |bias| {:.6} did not grow over the previous gain's {:.6}",
                    b[o].abs(),
                    prev[o].abs()
                );
            }
        }
        prev = b;
    }
}

/// (c) Three writes into sector 5 with α = 0.5 land on the closed-form value.
///
/// `mem ← mem + α(x − mem)` from 0 with a constant `x` gives `x·(1 − (1−α)^n)`, so after
/// three writes at α = 0.5 the slot holds exactly 7/8 of `x`. Checking the arithmetic and
/// not merely "it changed" is what catches an EMA written the wrong way round (`α·mem +
/// (1−α)·x`), which converges to the same place and gets there at the wrong speed.
#[test]
fn three_writes_at_alpha_one_half_hit_the_closed_form() {
    let mut m = SectorMemory::new();
    let x = [0.8f32, 0.4, 0.2, -0.6];
    for _ in 0..3 {
        m.write(5, &x, 0.5);
    }
    let got = m.slot(5);
    let expected: Vec<f32> = x.iter().map(|v| v * 7.0 / 8.0).collect();
    println!(
        "(c) x = {x:?}\n    after 3 writes at alpha=0.5: {got:?}\n    closed form 7/8·x:    {expected:?}"
    );
    for k in 0..MEM_PER_SECTOR {
        assert!(
            (got[k] - expected[k]).abs() < 1e-6,
            "component {k}: {} vs closed form {}",
            got[k],
            expected[k]
        );
    }
    // Untouched sectors stay untouched — a write that leaked into its neighbours would
    // make `read`'s "this sector and the next" two copies of the same thing.
    assert_eq!(m.slot(4), [0.0; MEM_PER_SECTOR]);
    assert_eq!(m.slot(6), [0.0; MEM_PER_SECTOR]);
}

/// (c2) A read is this sector, then the next, then the constant 1.0 — and it wraps.
#[test]
fn read_concatenates_this_sector_the_next_and_the_bias_column() {
    let mut m = SectorMemory::new();
    m.write(23, &[1.0, 2.0, 3.0, 4.0], 1.0);
    m.write(0, &[5.0, 6.0, 7.0, 8.0], 1.0);
    let x = m.read(23);
    println!("(c2) read(23) = {x:?}");
    assert_eq!(
        x,
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0],
        "sector 23's neighbour is sector 0 — the grid is a ring"
    );
}

/// (f) `--hipocampo-apagar` zeroes the memory and LEAVES the resting bias standing.
///
/// The distinction is the whole experiment. Erasing the memory asks "does the content
/// pay?"; the resting bias `gain·tanh(W[:, 8])` is what the being evolved independently
/// of any episode, and removing it too would be ablating the organ — a different
/// question, which `--sem-hipocampo` answers. A test that only checked "the bias changed"  // check-language: allow -- quoting the flag name verbatim
/// would pass on either implementation.
#[test]
fn erased_read_zeroes_the_memory_and_keeps_the_resting_bias() {
    let mut m = SectorMemory::new();
    m.write(7, &[0.9, 0.5, 0.2, 0.6], 1.0);
    m.write(8, &[0.3, 0.7, 0.1, -0.4], 1.0);

    let erased = m.read_gated(7, true);
    println!("(f) read_gated(7, erased=true) = {erased:?}");
    assert_eq!(erased, ERASED_READ);
    assert_eq!(
        erased[HIP_READ_LEN - 1],
        1.0,
        "the bias column survives erasure"
    );
    assert!(
        erased[..HIP_READ_LEN - 1].iter().all(|&v| v == 0.0),
        "every memory component must read as zero"
    );
    assert_eq!(
        m.read_gated(7, false),
        m.read(7),
        "erased=false is the plain read"
    );

    // The resting bias: only the bias column has weight, so erasing changes nothing.
    let mut genes = [0.0f32; N_HIP_GENES];
    for o in 0..HIP_OUTPUTS {
        genes[o * HIP_READ_LEN + (HIP_READ_LEN - 1)] = 0.4 + 0.1 * o as f32;
    }
    genes[N_HIP_GENES - 1] = 1.0;
    let live = bias(&genes, &m.read(7));
    let dark = bias(&genes, &erased);
    println!("(f) resting-only genes: live = {live:?}, erased = {dark:?}");
    assert_eq!(
        live, dark,
        "with only the bias column weighted, erasure is a no-op"
    );
    assert!(
        dark.iter().all(|v| v.abs() > 1e-3),
        "the resting bias must survive --hipocampo-apagar: {dark:?}"
    );

    // And with episodic weights, erasure DOES change the command — otherwise the
    // ablation would be measuring nothing.
    let mut episodic = [0.0f32; N_HIP_GENES];
    for (k, g) in episodic
        .iter_mut()
        .take(HIP_OUTPUTS * HIP_READ_LEN)
        .enumerate()
    {
        *g = if k % (HIP_READ_LEN) == HIP_READ_LEN - 1 {
            0.0
        } else {
            0.6
        };
    }
    episodic[N_HIP_GENES - 1] = 1.0;
    let a = bias(&episodic, &m.read(7));
    let b = bias(&episodic, &erased);
    println!("(f) episodic-only genes: live = {a:?}, erased = {b:?}");
    for o in 0..HIP_OUTPUTS {
        assert!(
            (a[o] - b[o]).abs() > 1e-3,
            "output {o}: erasing the memory changed the bias by only {:.3e}",
            (a[o] - b[o]).abs()
        );
        assert_eq!(
            b[o], 0.0,
            "with no bias-column weight, an erased read biases nothing"
        );
    }
}

/// The write path end to end: accumulate ticks, cross, and check what landed in the slot.
///
/// `dt_improvement` is 0 on the first passage and positive on a faster second one. Getting
/// the sign backwards here would hand selection a reward for being slow, and nothing
/// downstream would notice — the number is plausible either way.
#[test]
fn crossing_closes_the_sector_just_left_with_the_measured_summary() {
    let mut m = SectorMemory::new();
    // 100 ticks in sector 0 at 20 m/s, 25 of them on grass, no damage.
    for t in 0..100 {
        m.accumulate(20.0, t < 25, 0.0);
    }
    let (closed, x) = m.cross(1, 1.0).expect("first crossing writes sector 0");
    println!(
        "(w) first passage: closed sector {closed}, x = [{:.4}, {:.4}, {:.4}, {:.4}]",
        x[0], x[1], x[2], x[3]
    );
    assert_eq!(
        closed, 0,
        "the sector WRITTEN is the one just left, not the one entered"
    );
    assert!((x[0] - 0.5).abs() < 1e-6, "20/40 m/s = 0.5, got {}", x[0]);
    assert!(
        (x[1] - 0.25).abs() < 1e-6,
        "25/100 ticks on grass, got {}",
        x[1]
    );
    assert_eq!(x[2], 0.0, "no damage taken");
    assert_eq!(
        x[3], 0.0,
        "first passage has no previous time to improve on"
    );

    // Go round the ring back to sector 0 and traverse it faster: 80 ticks instead of 100.
    for s in 1..N_SECTORS as u32 {
        for _ in 0..10 {
            m.accumulate(20.0, false, 0.0);
        }
        m.cross((s + 1) % N_SECTORS as u32, 1.0);
    }
    for t in 0..80 {
        m.accumulate(30.0, false, if t < 40 { 0.10 } else { 0.30 });
    }
    let (closed2, x2) = m
        .cross(1, 1.0)
        .expect("second crossing writes sector 0 again");
    println!(
        "(w) second passage: closed sector {closed2}, x = [{:.4}, {:.4}, {:.4}, {:.4}]",
        x2[0], x2[1], x2[2], x2[3]
    );
    assert_eq!(closed2, 0);
    assert!(
        (x2[0] - 0.75).abs() < 1e-6,
        "30/40 m/s = 0.75, got {}",
        x2[0]
    );
    assert!(
        (x2[3] - 0.2).abs() < 1e-6,
        "(100 - 80)/100 = +0.20 improvement, got {}",
        x2[3]
    );
    assert!(
        x2[2] > 0.0,
        "damage taken inside the sector has to show up: {}",
        x2[2]
    );

    // A crossing with nothing accumulated writes nothing — otherwise a being sitting on a
    // sector line would overwrite a real memory with an empty one every tick.
    assert!(
        m.cross(2, 1.0).is_none(),
        "empty accumulator must not write"
    );
}

/// The two gene→parameter maps hit their documented defaults exactly.
#[test]
fn birth_genes_give_alpha_one_half_and_gain_zero() {
    let genes = [0.0f32; N_HIP_GENES];
    println!(
        "(g) write_rate(0) = {} (bits {:#010x}), read_gain(0) = {} (bits {:#010x})",
        write_rate(&genes),
        write_rate(&genes).to_bits(),
        read_gain(&genes),
        read_gain(&genes).to_bits()
    );
    assert_eq!(write_rate(&genes), 0.5);
    assert_eq!(read_gain(&genes).to_bits(), 0);
}
