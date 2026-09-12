//! `dof` — how many genes of a block are actually switched ON.
//!
//! One of the three rulers C3 of `docs/PRD-CEREBRO-NEUROMOD.md` brings over from the
//! neuromod lab. The question it answers is the one an evolution log cannot: *did
//! selection ever turn the new block on, or is it still sitting at its neutral birth
//! value?* Without it, a null result ("the hippocampus arm did not beat the control")
//! is unreadable — a block evolution never lit up and a block that lit up and did not
//! pay look identical in the fitness column.
//!
//! The counting rule is the lab's `gene_min`: a gene counts when `|w| > threshold`,
//! `threshold` defaulting to [`DEFAULT_GENE_THRESHOLD`] (0.5). It is deliberately a
//! HARD threshold and not a norm: the question is how many independent knobs are away
//! from neutral, not how much total mass the block carries. A block of 27 genes all at
//! 0.4 has a real L2 norm and zero degrees of freedom under this ruler, which is the
//! honest reading — none of those genes alone changes anything a mutation of the same
//! size would not undo.
//!
//! # This file is PURE
//!
//! No GPU, no I/O, no atomics. It reads a [`Genome`] and returns numbers. The callers
//! (the generation log and `laps_gen<N>.jsonl`, PRD §C3) live in `gpu_evolve` and are
//! wired at merge time, not here.
//!
//! # Blocks covered today
//!
//! `slow`, `tau` and `gates_w` — the three heritable blocks that exist in `Genome` on
//! this branch (`src/gpu/evolution.rs:93-182`). The PRD also asks for `dof_hip` and
//! `dof_tal`; those blocks do not exist yet (C1 and C2 add them) and this file does not
//! invent fields — see the `TODO(C1)`/`TODO(C2)` in [`dof_by_block`].

use crate::gpu::evolution::Genome;

/// The lab's `gene_min`: a gene is ON when `|w|` is strictly above this.
///
/// 0.5 is not tuned here — it is the value `neuromod-lab` counts with, kept identical so
/// the two labs' `dof` columns mean the same thing. Callers that need another cut pass
/// it explicitly; nothing in this file reads a global.
pub const DEFAULT_GENE_THRESHOLD: f32 = 0.5;

/// Number of genes in `genes` whose magnitude is strictly above `threshold`.
///
/// `NaN` never counts: `f32`'s comparison returns `false` for any NaN operand, so a
/// block that has blown up reads as `dof = 0` rather than as a full block. That is the
/// conservative direction — a NaN block is broken, not "maximally used", and a `dof`
/// that silently inflated on NaN would hide exactly the failure it should surface. A
/// caller that needs to tell "neutral" from "broken" apart checks finiteness itself.
///
/// Infinities DO count (`|inf| > threshold` is true), same reasoning inverted: an
/// infinite gene is unmistakably away from neutral.
pub fn dof(genes: &[f32], threshold: f32) -> usize {
    genes.iter().filter(|w| w.abs() > threshold).count()
}

/// `dof` per heritable block of the genome, in a fixed order, so two runs' lines can be
/// diffed column by column.
///
/// The order is the genome's own disk order (`genome_io.rs`'s block list), not
/// alphabetical: `slow`, `tau`, `gates_w`. Blocks that are absent from a loaded genome
/// (legacy file, optional block missing) come back as length 0 and therefore `dof = 0`,
/// which is the truth — that genome really does not carry the block.
///
/// # A caveat that belongs to `tau` alone
///
/// `tau` is not a weight. It holds the CfC decay rate per channel in s⁻¹
/// (`Genome::tau`, `evolution.rs:111-117`), born uniform at `cfc::DECAY_DEFAULT` = 0.5
/// (τ = 2 s). Under the 0.5 cut, a newborn τ bank scores `dof = 0` — which is the
/// behaviour the PRD's acceptance criterion asks for ("`dof` = 0 in generation 0 of both
/// arms") — but the *meaning* of a counted τ gene is "this channel's memory is shorter
/// than 2 s", not "this weight is large". Read the `tau` column as a spread indicator,
/// never as a weight magnitude.
///
/// TODO(C1): add `("hip", dof(&genome.hip, threshold))` when C1 lands the hippocampus
/// block on `Genome`. TODO(C2): the same for `tal`. Neither field exists on this branch
/// and this function does not invent them — a column of zeros for a block that does not
/// exist would read exactly like a block that exists and is switched off.
pub fn dof_by_block(genome: &Genome, threshold: f32) -> Vec<(&'static str, usize)> {
    vec![
        ("slow", dof(&genome.slow, threshold)),
        ("tau", dof(&genome.tau, threshold)),
        ("gates_w", dof(&genome.gates_w, threshold)),
    ]
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_GENE_THRESHOLD, dof, dof_by_block};
    use crate::gpu::evolution::{Genome, N_CFC_CHANNELS};

    /// A genome with every block empty except the ones a test fills in. Built by hand
    /// instead of via `Genome::random` so no test here depends on an RNG or on a default
    /// that might move — the point of `dof` is to count what is THERE.
    fn empty_genome() -> Genome {
        Genome {
            l1: Vec::new(),
            l2: Vec::new(),
            kan_knots: [[0.0; crate::kan::KAN_OUT_KNOTS]; 3],
            slow: Vec::new(),
            tau: Vec::new(),
            sensors: Vec::new(),
            neuron_alpha: Vec::new(),
            neuron_period_l1: Vec::new(),
            neuron_threshold_l1: Vec::new(),
            neuron_activation: Vec::new(),
            gates_w: Vec::new(),
            hip: Vec::new(),
        }
    }

    #[test]
    fn counts_only_genes_strictly_above_the_threshold() {
        // 0.5 itself must NOT count: the rule is `>`, not `>=`. A τ bank born at exactly
        // DECAY_DEFAULT (0.5) has to read as zero degrees of freedom, otherwise every
        // generation-0 genome would report a fully lit block.
        let genes = [0.0, 0.5, -0.5, 0.500_001, -0.500_001, 2.0, -3.0];
        assert_eq!(
            dof(&genes, 0.5),
            4,
            "0.5 and -0.5 are on the cut, not above it"
        );
    }

    #[test]
    fn an_empty_block_has_zero_degrees_of_freedom() {
        assert_eq!(dof(&[], DEFAULT_GENE_THRESHOLD), 0);
    }

    #[test]
    fn a_nan_gene_does_not_count_as_used() {
        // Documented direction: a blown-up block reads as 0, never as "fully used".
        let genes = [f32::NAN, f32::NAN, 9.0];
        assert_eq!(dof(&genes, DEFAULT_GENE_THRESHOLD), 1);
    }

    #[test]
    fn an_infinite_gene_counts_as_used() {
        let genes = [f32::INFINITY, f32::NEG_INFINITY, 0.0];
        assert_eq!(dof(&genes, DEFAULT_GENE_THRESHOLD), 2);
    }

    #[test]
    fn the_threshold_is_a_parameter_not_a_constant() {
        let genes = [0.1, 0.2, 0.3, 0.4];
        assert_eq!(dof(&genes, DEFAULT_GENE_THRESHOLD), 0);
        assert_eq!(dof(&genes, 0.15), 3);
        assert_eq!(dof(&genes, 0.0), 4);
    }

    #[test]
    fn a_newborn_tau_bank_reports_zero_degrees_of_freedom() {
        // The PRD's generation-0 gate: the arms must be indistinguishable by `dof`.
        let mut g = empty_genome();
        g.tau = vec![crate::cfc::DECAY_DEFAULT; N_CFC_CHANNELS];
        let by_block = dof_by_block(&g, DEFAULT_GENE_THRESHOLD);
        assert_eq!(by_block, vec![("slow", 0), ("tau", 0), ("gates_w", 0)]);
    }

    #[test]
    fn blocks_come_back_in_disk_order_with_their_own_counts() {
        let mut g = empty_genome();
        g.slow = vec![0.9, -0.9, 0.1];
        g.tau = vec![0.5, 4.0, 0.5];
        g.gates_w = vec![0.0; 16];
        g.gates_w[3] = -0.777;
        let by_block = dof_by_block(&g, DEFAULT_GENE_THRESHOLD);
        assert_eq!(
            by_block,
            vec![("slow", 2), ("tau", 1), ("gates_w", 1)],
            "order is the genome's disk order, counts are per block"
        );
    }

    #[test]
    fn a_genome_missing_an_optional_block_reports_zero_for_it() {
        // A legacy `.bin` with no M4 block loads with `gates_w` empty; `dof` must say 0
        // rather than panic or borrow another block's count.
        let mut g = empty_genome();
        g.slow = vec![3.0; 5];
        let by_block = dof_by_block(&g, DEFAULT_GENE_THRESHOLD);
        assert_eq!(by_block[2], ("gates_w", 0));
    }
}
