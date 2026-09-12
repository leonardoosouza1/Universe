//! The matrix the gate prints, and the verdict that decides each row.
//!
//! Split out of `sensitivity/mod.rs` (600-line cap, `docs/STYLE.md` §2): the parent is
//! about WHICH genes to perturb and how; this file is about how the answer is classified
//! and rendered. `run_gate` fills these types in; `src/bin/sensibilidade.rs` prints them.

use super::{MOVE_THRESHOLD, Reach};
use crate::gpu::contribution::OUTPUT_NAMES;


/// The verdict for one group. `Blind` and `OffPathMoved` are the two that fail the gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// On-path and it moved an output. What a wired block looks like.
    Moved,
    /// On-path and it moved nothing. The failure this gate exists to catch.
    Blind,
    /// Declared off-path and silent, as declared.
    OffPathSilent,
    /// Declared off-path and yet it moved an output — the declaration in [`groups`] is
    /// wrong. Fails, because every other row is read against that same list.
    OffPathMoved,
    /// The block is absent from this genome (legacy file, optional block never written).
    /// Nothing to perturb; excluded from the gate.
    Absent,
}

/// One row of the printed matrix.
#[derive(Clone, Debug)]
pub struct GroupRow {
    pub name: &'static str,
    pub reach: Reach,
    pub n_genes: usize,
    /// Genes that are non-zero BEFORE any perturbation — "is this block switched on in
    /// this genome at all", next to the sensitivity answer "could it be".
    pub n_nonzero: usize,
    /// Per output, the largest mean `|Δ|` over every (δ, sign) arm tried.
    pub peak_abs_delta: [f32; 3],
    /// Largest fraction of ticks on which a command changed SIGN, over outputs and arms.
    /// The column that separates "moved by a rounding error" from "turned the other
    /// way": a block can be under the mean threshold on every output and still flip the
    /// steering wheel across zero on a third of the ticks.
    pub peak_sign_flip_frac: f32,
    /// Largest fraction of ticks on which the DOMINANT command changed identity — the
    /// lab's own `argmax` ruler (`testes_sentidos.py`), kept for comparability.
    pub peak_argmax_flip_frac: f32,
    /// Non-finite ticks summed over the arms — see `BlockContribution::nonfinite_ticks`.
    pub nonfinite_ticks: usize,
    pub verdict: Verdict,
}

/// The verdict for one group, from the three facts that decide it. Extracted from
/// [`run_gate`] so every branch — including the two that fail — is testable without
/// having to manufacture a genome that produces it.
pub fn classify(n_genes: usize, reach: Reach, moved: bool) -> Verdict {
    match (n_genes, reach, moved) {
        (0, _, _) => Verdict::Absent,
        (_, Reach::Runner, true) => Verdict::Moved,
        (_, Reach::Runner, false) => Verdict::Blind,
        (_, Reach::OffPath(_), false) => Verdict::OffPathSilent,
        (_, Reach::OffPath(_), true) => Verdict::OffPathMoved,
    }
}

/// The whole matrix plus the gate line.
#[derive(Clone, Debug)]
pub struct SensitivityReport {
    pub ticks: usize,
    pub deltas: Vec<f32>,
    pub rows: Vec<GroupRow>,
}

impl SensitivityReport {
    /// Groups that fail the gate, with the reason folded into the verdict.
    pub fn failures(&self) -> Vec<&GroupRow> {
        self.rows
            .iter()
            .filter(|r| matches!(r.verdict, Verdict::Blind | Verdict::OffPathMoved))
            .collect()
    }

    pub fn passed(&self) -> bool {
        self.failures().is_empty()
    }
}

impl std::fmt::Display for SensitivityReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let deltas: Vec<String> = self.deltas.iter().map(|d| format!("{d:+.2}")).collect();
        writeln!(
            f,
            "sensitivity matrix — {} ticks, deltas ±[{}], move threshold {:.0e}",
            self.ticks,
            deltas.join(", "),
            MOVE_THRESHOLD
        )?;
        writeln!(
            f,
            "{:<20} {:<9} {:>7} {:>8} {:>11} {:>11} {:>11} {:>6} {:>7}  {}",
            "group",
            "reach",
            "genes",
            "nonzero",
            format!("|d {}|", OUTPUT_NAMES[0]),
            format!("|d {}|", OUTPUT_NAMES[1]),
            format!("|d {}|", OUTPUT_NAMES[2]),
            "sign",
            "argmax",
            "verdict"
        )?;
        for r in &self.rows {
            let reach = match r.reach {
                Reach::Runner => "runner",
                Reach::OffPath(_) => "off-path",
            };
            let verdict = match r.verdict {
                Verdict::Moved => "moved",
                Verdict::Blind => "BLIND",
                Verdict::OffPathSilent => "silent (expected)",
                Verdict::OffPathMoved => "UNEXPECTED SIGNAL",
                Verdict::Absent => "absent",
            };
            writeln!(
                f,
                "{:<20} {:<9} {:>7} {:>8} {:>11.3e} {:>11.3e} {:>11.3e} {:>6.2} {:>7.2}  {}{}",
                r.name,
                reach,
                r.n_genes,
                r.n_nonzero,
                r.peak_abs_delta[0],
                r.peak_abs_delta[1],
                r.peak_abs_delta[2],
                r.peak_sign_flip_frac,
                r.peak_argmax_flip_frac,
                verdict,
                if r.nonfinite_ticks > 0 {
                    format!(" (+{} non-finite ticks)", r.nonfinite_ticks)
                } else {
                    String::new()
                }
            )?;
        }
        for r in &self.rows {
            if let Reach::OffPath(why) = r.reach {
                writeln!(f, "  off-path {:<20} {}", r.name, why)?;
            }
        }
        Ok(())
    }
}
