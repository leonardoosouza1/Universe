//! `--memory` — how much of the hippocampus command is the MEMORY, and how much is the
//! constant offset the genome was born with. NO GPU.
//!
//! # Why it exists
//!
//! `--setores` reports, per sector and per pass, the mean of the LARGEST of the three
//! `|bias|` components. That statistic answers "did the organ output anything?" and it
//! cannot answer "did the memory CONTENT change the command?", for two reasons:
//!
//! 1. It takes a maximum ACROSS components. A genome whose brake row holds a large
//!    constant offset (the u1v champions hold ~0.17 on brake) reports that offset in
//!    every row of the table, first pass and later pass alike, and a memory-driven swing
//!    of 0.08 on throttle never reaches the maximum. The number stays flat and the
//!    mechanism looks dead.
//! 2. The read carries a constant 1.0 in its last column (`hip::HIP_READ_LEN`), so
//!    `|bias|` is bounded BELOW by the resting bias `gain · tanh(W[:, 8])` even with every
//!    memory slot at zero. Comparing `|bias|` between passes compares that floor plus a
//!    perturbation with the same floor plus another one.
//!
//! This report removes the floor and never mixes components:
//!
//! ```text
//! Δb(t) = b(x_t) − b(x_empty),   x_empty = [0, 0, 0, 0, 0, 0, 0, 0, 1]
//! ```
//!
//! `Δb` is exactly the part of the command that the content of the 8 memory columns is
//! responsible for, component by component, and it is zero by construction on a row whose
//! memory is still empty.
//!
//! # The one statistic that is a tautology, kept because it says so
//!
//! `var(Δb)/var(b)` is **1 by construction**: `Δb` differs from `b` by a CONSTANT vector
//! and variance is shift invariant. It is printed because the whole variance of the
//! command really is memory-driven — but a reader must not read 1.000 as a measurement.
//! The measurement is `mean |Δb|` against `mean |b|` in the table above it: that ratio is
//! where a dominant constant offset shows up, and it is the comparison `--setores` could
//! not make.

use universe::gpu::hippocampus as hip;

use crate::{HipRow, gene_line, load_hip_genes, load_rows, resting_bias};

/// Command channels, in the output order of `hip::bias` and of `kan_eval`.
const COMPONENTS: [&str; hip::HIP_OUTPUTS] = ["throttle", "steer", "brake"];

/// For every row, `0` when it belongs to the FIRST contiguous pass through its sector and
/// `1` when it belongs to any later pass.
///
/// This is the single source of the pass rule: `crate::sector_stats` splits its columns by
/// it and [`report`] splits its means by it, so the two reports can never disagree about
/// what a first pass is. A being that wobbles back over a boundary and returns counts as a
/// second pass through the sector, which is what the read actually saw — the slot was
/// already written when it came back.
pub fn pass_kind(rows: &[HipRow]) -> Vec<usize> {
    let mut passes = [0usize; hip::N_SECTORS];
    let mut prev: Option<usize> = None;
    rows.iter()
        .map(|row| {
            let s = row.sector;
            if prev != Some(s) {
                passes[s] += 1;
            }
            prev = Some(s);
            usize::from(passes[s] > 1)
        })
        .collect()
}

/// True when at least one of the 8 memory columns is non-zero.
///
/// The test is exact equality against zero, not a threshold: the slots start at a written
/// `0.0` and only an EMA write moves them, so "has the being ever stored anything readable
/// here?" is a bit-level question. A threshold would silently drop the smallest real
/// memories, which are the ones on the first lap.
fn has_memory(row: &HipRow) -> bool {
    row.mem.iter().any(|v| *v != 0.0)
}

/// `b(x)` as the run applied it: the organ's output times [`hip::HIP_AUTHORITY`], so it is
/// directly comparable with the recorded bias columns.
fn command(genes: &[f32], x: &[f32; hip::HIP_READ_LEN]) -> [f32; hip::HIP_OUTPUTS] {
    let mut b = hip::bias(genes, x);
    for v in b.iter_mut() {
        *v *= hip::HIP_AUTHORITY;
    }
    b
}

/// `Δb = b(x_t) − b(x_empty)`, component by component.
fn delta(
    b: &[f32; hip::HIP_OUTPUTS],
    resting: &[f32; hip::HIP_OUTPUTS],
) -> [f32; hip::HIP_OUTPUTS] {
    let mut d = [0.0f32; hip::HIP_OUTPUTS];
    for (o, v) in d.iter_mut().enumerate() {
        *v = b[o] - resting[o];
    }
    d
}

/// Running mean and population variance of one channel (Welford).
///
/// The naive `sumsq/n − mean²` would subtract two numbers of order 700 to produce one of
/// order 0.01 on these files — the textbook case where the subtraction loses the digits
/// the answer is made of, and the case where a ratio that must print as 1.000000 would
/// print as 0.9997 and be mistaken for a finding. Welford never forms that difference.
#[derive(Clone, Copy, Debug, Default)]
struct Moments {
    n: u64,
    mean: f64,
    m2: f64,
}

impl Moments {
    fn push(&mut self, x: f64) {
        self.n += 1;
        let d = x - self.mean;
        self.mean += d / self.n as f64;
        self.m2 += d * (x - self.mean);
    }

    /// Population variance (divided by `n`), 0.0 when nothing was pushed.
    fn var(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.m2 / self.n as f64
        }
    }
}

/// The verdict of `--memory`, per command component.
///
/// Every `[_; hip::HIP_OUTPUTS]` field is indexed by [`COMPONENTS`]: throttle, steer,
/// brake. Nothing in this struct is ever reduced across components — that reduction is
/// the flaw this report exists to fix.
#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryReport {
    pub rows: usize,
    /// Rows with at least one non-zero memory column — the denominator of
    /// [`Self::mean_abs_delta`], since a row with an empty memory has `Δb = 0` exactly and
    /// averaging those in measures how much of the file is lap 1, not how strong the
    /// organ is.
    pub rows_with_memory: usize,
    /// `b(x_empty)`, post-authority: the command floor the genome carries from tick 0.
    pub resting: [f32; hip::HIP_OUTPUTS],
    /// `max |b(formula) − b(recorded)|` over every row and component. This report replays
    /// the formula, so a non-zero value here means the rows below describe a being the
    /// telemetry did not record (see the elite-offset rule in `main.rs`'s header).
    pub max_formula_gap: f32,
    /// Mean `|Δb|` over the rows with memory present.
    pub mean_abs_delta: [f64; hip::HIP_OUTPUTS],
    /// Max `|Δb|` over ALL rows, and the row it happened on.
    pub max_abs_delta: [f64; hip::HIP_OUTPUTS],
    pub worst_row: [usize; hip::HIP_OUTPUTS],
    /// Mean and max `|b|` over ALL rows — the reference the memory term is compared with.
    pub mean_abs_bias: [f64; hip::HIP_OUTPUTS],
    pub max_abs_bias: [f64; hip::HIP_OUTPUTS],
    pub sd_bias: [f64; hip::HIP_OUTPUTS],
    /// Population variance of `b` and of `Δb` over all rows. See this module's header:
    /// the ratio is 1 by construction.
    pub var_bias: [f64; hip::HIP_OUTPUTS],
    pub var_delta: [f64; hip::HIP_OUTPUTS],
    /// `[first pass, later passes]` row counts, all rows and memory-present rows.
    pub pass_rows: [usize; 2],
    pub pass_rows_with_memory: [usize; 2],
    /// Mean `|Δb|` over the memory-present rows of each pass class. The same denominator
    /// as [`Self::mean_abs_delta`], so the two classes decompose it exactly.
    pub mean_abs_delta_pass: [[f64; hip::HIP_OUTPUTS]; 2],
}

impl MemoryReport {
    /// `var(Δb)/var(b)`, or `None` for a channel whose command never moved — a constant
    /// channel has `var(b) = 0` and the fraction is `0/0`, which is not 1 and not 0.
    pub fn var_ratio(&self, o: usize) -> Option<f64> {
        (self.var_bias[o] > 0.0).then(|| self.var_delta[o] / self.var_bias[o])
    }
}

/// Replays the formula over every row and splits the command into its memory term and its
/// resting offset.
pub fn report(genes: &[f32], rows: &[HipRow]) -> MemoryReport {
    let resting = resting_bias(genes);
    let kinds = pass_kind(rows);
    let mut rep = MemoryReport {
        rows: rows.len(),
        resting,
        ..Default::default()
    };
    let mut abs_delta = [Moments::default(); hip::HIP_OUTPUTS];
    let mut abs_delta_pass = [[Moments::default(); hip::HIP_OUTPUTS]; 2];
    let mut abs_bias = [Moments::default(); hip::HIP_OUTPUTS];
    let mut signed_bias = [Moments::default(); hip::HIP_OUTPUTS];
    let mut signed_delta = [Moments::default(); hip::HIP_OUTPUTS];

    for (i, (row, &k)) in rows.iter().zip(&kinds).enumerate() {
        let b = command(genes, &row.read());
        let d = delta(&b, &resting);
        let has = has_memory(row);
        rep.pass_rows[k] += 1;
        rep.rows_with_memory += usize::from(has);
        rep.pass_rows_with_memory[k] += usize::from(has);
        for o in 0..hip::HIP_OUTPUTS {
            rep.max_formula_gap = rep.max_formula_gap.max((b[o] - row.bias[o]).abs());
            let ad = d[o].abs() as f64;
            if ad > rep.max_abs_delta[o] {
                rep.max_abs_delta[o] = ad;
                rep.worst_row[o] = i;
            }
            rep.max_abs_bias[o] = rep.max_abs_bias[o].max(b[o].abs() as f64);
            abs_bias[o].push(b[o].abs() as f64);
            signed_bias[o].push(b[o] as f64);
            signed_delta[o].push(d[o] as f64);
            if has {
                abs_delta[o].push(ad);
                abs_delta_pass[k][o].push(ad);
            }
        }
    }

    for o in 0..hip::HIP_OUTPUTS {
        rep.mean_abs_delta[o] = abs_delta[o].mean;
        rep.mean_abs_bias[o] = abs_bias[o].mean;
        rep.var_bias[o] = signed_bias[o].var();
        rep.var_delta[o] = signed_delta[o].var();
        rep.sd_bias[o] = rep.var_bias[o].sqrt();
        for k in 0..2 {
            rep.mean_abs_delta_pass[k][o] = abs_delta_pass[k][o].mean;
        }
    }
    rep
}

/// `--memory`: the per-component memory term. Needs the genome, because `x_empty` and
/// therefore the resting bias are a property of `W`, not of the telemetry.
pub fn run(genome_path: &str, hip_path: &str) -> ! {
    let genes = load_hip_genes(genome_path);
    let rows = load_rows(hip_path);
    let rep = report(&genes, &rows);

    println!("hip_check --memory — memory term vs resting offset, CPU only");
    println!("  genome    {genome_path}");
    println!("  telemetry {hip_path}");
    println!("  genes     {}", gene_line(&genes));
    let pct = 100.0 * rep.rows_with_memory as f64 / rep.rows.max(1) as f64;
    println!(
        "  rows      {}   with any memory present {} ({pct:.1}%)",
        rep.rows, rep.rows_with_memory
    );
    println!(
        "  max |b(formula) - b(recorded)| = {:.3e}   (this report replays the formula; a \
         value above ~1e-5 means the genome is not the being the telemetry recorded)",
        rep.max_formula_gap
    );
    println!(
        "\n  delta_b(t) = b(x_t) - b(x_empty) with x_empty = [0,0,0,0,0,0,0,0,1]: the part \
         of the command the memory CONTENT is responsible for, per component, with the \
         resting offset the genome carries from tick 0 subtracted out. mean|delta_b| is \
         over the rows that have a memory ({} of {}); max|delta_b| and mean|b| are over \
         all rows.",
        rep.rows_with_memory, rep.rows
    );
    println!();
    println!(
        "  {:>9} {:>12} {:>12} {:>9} {:>12} {:>12} {:>12}",
        "component", "mean|delta|", "max|delta|", "at row", "mean|b|", "sd(b)", "resting b"
    );
    for (o, name) in COMPONENTS.iter().enumerate() {
        println!(
            "  {:>9} {:>12.6} {:>12.6} {:>9} {:>12.6} {:>12.6} {:>12.6}",
            name,
            rep.mean_abs_delta[o],
            rep.max_abs_delta[o],
            rep.worst_row[o],
            rep.mean_abs_bias[o],
            rep.sd_bias[o],
            rep.resting[o]
        );
    }

    println!(
        "\n  mean|delta_b| by pass, over the memory-present rows of each class: '1st' is \
         the first contiguous pass through a sector, where mem[s] is still empty by \
         construction and only the lookahead mem[s+1] can carry anything."
    );
    println!(
        "  {:>9} {:>12} {:>12} {:>9}",
        "component", "1st pass", "later", "later/1st"
    );
    for (o, name) in COMPONENTS.iter().enumerate() {
        let (first, later) = (rep.mean_abs_delta_pass[0][o], rep.mean_abs_delta_pass[1][o]);
        let ratio = if first > 0.0 {
            format!("{:.2}", later / first)
        } else {
            "n/a".to_string()
        };
        println!("  {name:>9} {first:>12.6} {later:>12.6} {ratio:>9}");
    }
    println!(
        "  rows: 1st pass {} ({} with memory) · later {} ({} with memory)",
        rep.pass_rows[0],
        rep.pass_rows_with_memory[0],
        rep.pass_rows[1],
        rep.pass_rows_with_memory[1]
    );

    print!("\n  var(delta_b)/var(b) over all rows:");
    for (o, name) in COMPONENTS.iter().enumerate() {
        match rep.var_ratio(o) {
            Some(r) => print!("   {name} {r:.6}"),
            None => print!("   {name} n/a (constant channel)"),
        }
    }
    println!();
    println!(
        "  NOTE: that ratio is 1 BY CONSTRUCTION — delta_b differs from b by the constant \
         resting vector and variance is shift invariant. It is printed because all of the \
         command's variance really is memory-driven, but the measurement is mean|delta_b| \
         against mean|b| in the first table: THAT is where a dominant constant offset \
         shows up, and it is the comparison --setores cannot make (it maxes over the three \
         components, so a large constant brake offset hides a smaller throttle swing)."
    );
    std::process::exit(0);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One row the way `context.rs` writes it, with the bias column filled by the SAME
    /// formula the run applies (post-authority), so `max_formula_gap` is 0 and every
    /// number below is the formula's and not a fixture's.
    fn make_row(
        genes: &[f32],
        tick: usize,
        sector: usize,
        mem: [f32; 2 * hip::MEM_PER_SECTOR],
    ) -> HipRow {
        let mut row = HipRow {
            tick: tick as f32,
            sector,
            mem,
            bias: [0.0; hip::HIP_OUTPUTS],
        };
        row.bias = command(genes, &row.read());
        row
    }

    /// 29 distinct non-zero genes: an all-equal `W` would make a wrongly sliced row
    /// invisible, and identical numbers where they should differ are this project's alarm.
    fn varied_genes() -> Vec<f32> {
        (0..hip::N_HIP_GENES)
            .map(|k| 0.11 + 0.037 * k as f32)
            .collect()
    }

    /// Two laps of every sector, with a memory that never repeats a value.
    fn varied_rows(genes: &[f32]) -> Vec<HipRow> {
        (0..2 * hip::N_SECTORS)
            .map(|t| {
                let mut mem = [0.0f32; 2 * hip::MEM_PER_SECTOR];
                for (k, v) in mem.iter_mut().enumerate() {
                    *v = ((t + k) as f32 * 0.017).sin();
                }
                make_row(genes, t, t % hip::N_SECTORS, mem)
            })
            .collect()
    }

    /// Zero the constant column of `W` and the resting bias is exactly zero, so the memory
    /// term IS the whole command: `Δb == b` bit for bit, row by row and component by
    /// component. This is the calibration of the subtraction — if the wrong column were
    /// being treated as the bias column, this test would fail while the others still
    /// passed.
    #[test]
    fn with_a_zero_constant_column_the_delta_is_the_whole_command() {
        let mut genes = varied_genes();
        for o in 0..hip::HIP_OUTPUTS {
            genes[o * hip::HIP_READ_LEN + hip::HIP_READ_LEN - 1] = 0.0;
        }
        let resting = resting_bias(&genes);
        assert_eq!(resting, [0.0; hip::HIP_OUTPUTS], "resting must vanish");

        let rows = varied_rows(&genes);
        for (i, row) in rows.iter().enumerate() {
            let b = command(&genes, &row.read());
            assert_eq!(
                delta(&b, &resting),
                b,
                "row {i}: delta must equal b exactly"
            );
        }
        let rep = report(&genes, &rows);
        assert_eq!(
            rep.rows_with_memory, rep.rows,
            "the fixture must carry memory"
        );
        assert_eq!(rep.max_formula_gap, 0.0);
        for o in 0..hip::HIP_OUTPUTS {
            assert_eq!(rep.mean_abs_delta[o], rep.mean_abs_bias[o], "mean {o}");
            assert_eq!(rep.max_abs_delta[o], rep.max_abs_bias[o], "max {o}");
            assert!(rep.max_abs_delta[o] > 0.0, "component {o} must move");
        }
    }

    /// Zero every column of `W` EXCEPT the constant one and the command no longer depends
    /// on the memory at all: `Δb == 0` on every row, while `|b|` stays large. That is the
    /// case `--setores` cannot distinguish from a working organ, and the one this report
    /// has to call zero.
    #[test]
    fn with_only_the_constant_column_the_delta_is_zero_everywhere() {
        let mut genes = varied_genes();
        for o in 0..hip::HIP_OUTPUTS {
            for k in 0..hip::HIP_READ_LEN - 1 {
                genes[o * hip::HIP_READ_LEN + k] = 0.0;
            }
        }
        let resting = resting_bias(&genes);
        let rows = varied_rows(&genes);
        for (i, row) in rows.iter().enumerate() {
            let b = command(&genes, &row.read());
            assert_eq!(b, resting, "row {i}: b must be the resting bias");
            assert_eq!(delta(&b, &resting), [0.0; hip::HIP_OUTPUTS], "row {i}");
        }
        let rep = report(&genes, &rows);
        assert_eq!(rep.max_formula_gap, 0.0);
        for o in 0..hip::HIP_OUTPUTS {
            assert_eq!(rep.mean_abs_delta[o], 0.0, "mean {o}");
            assert_eq!(rep.max_abs_delta[o], 0.0, "max {o}");
            assert_eq!(rep.var_delta[o], 0.0, "var {o}");
            assert_eq!(rep.mean_abs_delta_pass[0][o], 0.0, "1st pass {o}");
            assert_eq!(rep.mean_abs_delta_pass[1][o], 0.0, "later pass {o}");
            assert!(
                rep.mean_abs_bias[o] > 0.01,
                "component {o} must still command"
            );
            // A constant channel has var(b) = 0, so the fraction is 0/0 and not 1.
            assert_eq!(rep.var_ratio(o), None, "component {o} never moved");
        }
    }

    /// Throttle reads `mem[s][0]` with weight 2, steer is dead, brake is a constant 0.75
    /// on the bias column, gain gene 1.0. Every number below is the closed form
    /// `0.5 · tanh(1) · tanh(W·x)` written out, so the test pins the arithmetic and not
    /// just its shape.
    ///
    /// Rows: sectors 0, 0, 1, 0, 0, with `mem[s][0]` = 0.5, 0.5, 0 (empty row), 0.25,
    /// 0.25. The two 0.25 rows are a LATER pass through sector 0, so the pass split has to
    /// separate them from the 0.5 rows.
    #[test]
    fn a_memory_driven_throttle_and_a_constant_brake_give_the_closed_form() {
        /// `0.5 · tanh(1) · tanh(2 · 0.5)` — throttle on the first-pass rows.
        const A: f64 = 0.290_012_836_456_298_83;
        /// `0.5 · tanh(1) · tanh(2 · 0.25)` — throttle on the later-pass rows.
        const B: f64 = 0.175_972_878_932_952_88;
        /// `0.5 · tanh(1) · tanh(0.75)` — the brake offset, on every row.
        const BRAKE: f64 = 0.241_862_893_104_553_22;
        const EPS: f64 = 1e-6;

        let mut genes = vec![0.0f32; hip::N_HIP_GENES];
        genes[0] = 2.0; // throttle row, weight on mem[s][0]
        genes[2 * hip::HIP_READ_LEN + hip::HIP_READ_LEN - 1] = 0.75; // brake, constant col
        genes[hip::GENE_READ_GAIN] = 1.0;

        let mut mem_half = [0.0f32; 2 * hip::MEM_PER_SECTOR];
        mem_half[0] = 0.5;
        let mut mem_quarter = [0.0f32; 2 * hip::MEM_PER_SECTOR];
        mem_quarter[0] = 0.25;
        let empty = [0.0f32; 2 * hip::MEM_PER_SECTOR];
        let rows = vec![
            make_row(&genes, 0, 0, mem_half),
            make_row(&genes, 1, 0, mem_half),
            make_row(&genes, 2, 1, empty),
            make_row(&genes, 3, 0, mem_quarter),
            make_row(&genes, 4, 0, mem_quarter),
        ];
        assert_eq!(
            pass_kind(&rows),
            vec![0, 0, 0, 1, 1],
            "the pass split itself"
        );

        let rep = report(&genes, &rows);
        assert_eq!(rep.max_formula_gap, 0.0);
        assert_eq!(rep.rows, 5);
        assert_eq!(rep.rows_with_memory, 4, "the sector-1 row is empty");
        assert_eq!(rep.pass_rows, [3, 2]);
        assert_eq!(rep.pass_rows_with_memory, [2, 2]);

        assert!((rep.resting[2] as f64 - BRAKE).abs() < EPS, "resting brake");
        assert_eq!(rep.resting[0], 0.0, "throttle has no constant column");
        assert_eq!(rep.resting[1], 0.0, "steer is dead");

        // Throttle: mean over the 4 memory rows is (A + B)/2, the peak is A on row 0.
        assert!(
            (rep.mean_abs_delta[0] - 0.5 * (A + B)).abs() < EPS,
            "mean|delta| throttle = {}",
            rep.mean_abs_delta[0]
        );
        assert!((rep.max_abs_delta[0] - A).abs() < EPS);
        assert_eq!(rep.worst_row[0], 0);
        assert!((rep.mean_abs_delta_pass[0][0] - A).abs() < EPS, "1st pass");
        assert!(
            (rep.mean_abs_delta_pass[1][0] - B).abs() < EPS,
            "later pass"
        );
        // mean |b| dilutes the same rows with the empty one: (2A + 0 + 2B)/5.
        assert!((rep.mean_abs_bias[0] - 0.4 * (A + B)).abs() < EPS);
        // Variance is shift invariant, so the fraction is 1 even here.
        assert!((rep.var_ratio(0).expect("throttle moved") - 1.0).abs() < 1e-12);

        // Steer commands nothing; brake commands a constant, which is the case the old
        // report scored as activity.
        for o in [1usize, 2] {
            assert_eq!(rep.mean_abs_delta[o], 0.0, "component {o} delta");
            assert_eq!(rep.max_abs_delta[o], 0.0, "component {o} peak");
            assert_eq!(rep.var_ratio(o), None, "component {o} is constant");
        }
        assert_eq!(rep.mean_abs_bias[1], 0.0, "steer is silent");
        assert!(
            (rep.mean_abs_bias[2] - BRAKE).abs() < EPS,
            "brake is the offset"
        );
    }
}
