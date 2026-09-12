//! How much a brain block actually contributes to the MOTOR COMMAND.
//!
//! The second of C3's three rulers (`docs/PRD-CEREBRO-NEUROMOD.md` §C3). `dof` says how
//! many genes of a block are switched on; this says whether switching them on reaches
//! the steering wheel. The two answer different questions and a block can pass one and
//! fail the other — a fully lit block whose effect the output clamps away contributes
//! nothing, and that is a result, not a bug.
//!
//! Input: two open-loop traces of the same run, one WITH the block and one WITHOUT it
//! (ablated, frozen, or perturbed — this file does not care how the second trace was
//! produced, only that it is the same ticks in the same order). Output: the mean `|Δ|`
//! per command and the fraction of ticks on which the DECISION changed, not just the
//! number.
//!
//! # Why two different "the decision changed" fractions
//!
//! `|Δ| = 0.004` on the steering wheel can be irrelevant (the wheel was at 0.6 and went
//! to 0.604) or decisive (the wheel was at 0.000 and went to 0.004 — the car now turns
//! the other way). A mean absolute delta cannot tell those apart, which is exactly how a
//! block gets called "negligible" on the strength of a small number. So:
//!
//! - [`BlockContribution::sign_flip_frac`] — per command, the fraction of ticks where
//!   the sign changed. On this motor triple it is `steer` that carries the meaning
//!   (left versus right); `throttle` and `brake` are rectified downstream
//!   (`gpu_evolve/context.rs:1862-1868` takes `.max(0.0)` on both), so a sign flip there
//!   is a flip between "pedal" and "no pedal".
//! - [`BlockContribution::argmax_flip_frac`] — the lab's own ruler (`testes_sentidos.py`
//!   compares `argmax` over the action scores): which of the three commands is the
//!   largest. Kept for comparability with the lab's matrix. On a triple where one entry
//!   is signed and two are rectified it is the weaker of the two fractions; read
//!   `sign_flip_frac` first.
//!
//! # This file is PURE
//!
//! No GPU, no I/O, no globals. It takes two slices and returns numbers.

/// Command names in the order the brain writes them — `final_intent[0..3)`, the same
/// three slots `gpu_evolve/context.rs` records into a `UWAV` row
/// (`gpu/wave_record.rs`). One source of truth for every table this module prints.
pub const OUTPUT_NAMES: [&str; 3] = ["throttle", "steer", "brake"];

/// What one block contributed to the command, over a whole trace.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockContribution {
    /// Ticks compared. Equal to the length of both input traces — they are required to
    /// match, so this is never a silently truncated count.
    pub ticks: usize,
    /// Mean `|with − without|` per command, over the ticks where both sides were finite.
    /// `0.0` for a command whose every tick was non-finite (read `nonfinite_ticks`
    /// before believing a zero).
    pub mean_abs_delta: [f32; 3],
    /// Largest `|with − without|` per command, same finiteness rule. A block can have a
    /// tiny mean and a large max — it acts rarely and hard, which is what a reflex looks
    /// like and what a mean alone erases.
    pub max_abs_delta: [f32; 3],
    /// Fraction of ticks on which the command's sign changed, per command. Sign is
    /// `-1 / 0 / +1` with an EXACT zero (see the module doc): `0.0 → 0.3` counts as a
    /// change, `0.3 → 0.6` does not.
    pub sign_flip_frac: [f32; 3],
    /// Fraction of ticks on which the largest of the three commands changed identity.
    pub argmax_flip_frac: f32,
    /// Ticks where at least one of the six values was NaN or infinite. Those ticks are
    /// excluded from the means and maxima and counted here instead — a block that drives
    /// the brain to NaN is emphatically not "contributing zero", and a mean that quietly
    /// swallowed the NaN would say it was.
    pub nonfinite_ticks: usize,
}

impl BlockContribution {
    /// The largest mean contribution across the three commands — the single number a
    /// gate compares against a threshold ("did this block move ANY output?").
    pub fn peak_mean_abs_delta(&self) -> f32 {
        self.mean_abs_delta
            .iter()
            .copied()
            .fold(0.0f32, |a, b| if b > a { b } else { a })
    }

    /// True when the block moved at least one command's mean above `threshold`, or made
    /// the brain non-finite on at least one tick.
    ///
    /// The non-finite clause is deliberate and is the reason this is a method and not an
    /// inline comparison at the call site: a block whose perturbation blows the brain up
    /// has demonstrably reached the output, and a gate that reported it as "blind"
    /// because the mean was computed over zero finite ticks would send the reader
    /// looking for a wiring bug that is not there.
    pub fn moved_any_output(&self, threshold: f32) -> bool {
        self.nonfinite_ticks > 0 || self.peak_mean_abs_delta() > threshold
    }
}

/// Sign as `-1 / 0 / +1`, with an exact zero.
///
/// `f32::signum` is wrong for this: it returns `+1.0` for `+0.0` and `-1.0` for `-0.0`,
/// so a command sitting at rest would be reported as "positive" and a trace that never
/// leaves zero would show sign flips between `+0.0` and `-0.0`. NaN maps to 0 so that a
/// NaN tick — already excluded from the means — cannot also fabricate a flip.
fn sign3(v: f32) -> i8 {
    if v > 0.0 {
        1
    } else if v < 0.0 {
        -1
    } else {
        0
    }
}

/// Index of the largest of the three commands. Ties and NaN both resolve to the LOWEST
/// index, which keeps the function total: any two identical traces give identical
/// argmaxes and therefore a flip fraction of exactly zero, with no NaN-dependent noise.
fn argmax3(v: [f32; 3]) -> usize {
    let mut best = 0usize;
    for i in 1..3 {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

/// Contribution of one block, comparing the trace WITH it against the trace WITHOUT it.
///
/// Both slices are `[throttle, steer, brake]` per tick, in tick order. They must have
/// the same length and be non-empty:
///
/// - **different lengths** is rejected, not truncated to the shorter one. Two traces of
///   different lengths are two different runs (one diverged, crashed, or was recorded
///   under another tick budget); silently comparing the common prefix is how a block
///   gets a contribution number computed against a run that stopped early — the same
///   failure `wave_record::read_all` rejects a partial row for.
/// - **empty** is rejected because the alternative is returning all zeros, which reads
///   exactly like "this block contributes nothing" — the single most expensive way to be
///   wrong here.
pub fn contribution(
    with_block: &[[f32; 3]],
    without_block: &[[f32; 3]],
) -> Result<BlockContribution, String> {
    if with_block.len() != without_block.len() {
        return Err(format!(
            "trace length mismatch: with={} ticks, without={} ticks. These are two \
             different runs, not two arms of the same one — re-record the shorter side \
             rather than comparing the common prefix.",
            with_block.len(),
            without_block.len()
        ));
    }
    if with_block.is_empty() {
        return Err(
            "empty traces: nothing to compare (an all-zero contribution would \
                    be indistinguishable from a block that truly does nothing)"
                .to_string(),
        );
    }

    let ticks = with_block.len();
    let mut sum_abs = [0.0f64; 3];
    let mut finite_ticks = [0usize; 3];
    let mut max_abs = [0.0f32; 3];
    let mut sign_flips = [0usize; 3];
    let mut argmax_flips = 0usize;
    let mut nonfinite_ticks = 0usize;

    for (a, b) in with_block.iter().zip(without_block.iter()) {
        let tick_finite = a.iter().chain(b.iter()).all(|v| v.is_finite());
        if !tick_finite {
            nonfinite_ticks += 1;
            continue;
        }
        for k in 0..3 {
            let d = (a[k] - b[k]).abs();
            sum_abs[k] += d as f64;
            finite_ticks[k] += 1;
            if d > max_abs[k] {
                max_abs[k] = d;
            }
            if sign3(a[k]) != sign3(b[k]) {
                sign_flips[k] += 1;
            }
        }
        if argmax3(*a) != argmax3(*b) {
            argmax_flips += 1;
        }
    }

    let mut mean_abs_delta = [0.0f32; 3];
    let mut sign_flip_frac = [0.0f32; 3];
    for k in 0..3 {
        if finite_ticks[k] > 0 {
            mean_abs_delta[k] = (sum_abs[k] / finite_ticks[k] as f64) as f32;
            sign_flip_frac[k] = sign_flips[k] as f32 / finite_ticks[k] as f32;
        }
    }
    // Denominator of the argmax fraction is the number of COMPARABLE ticks, not `ticks`
    // — otherwise a trace that went non-finite halfway through would report a fraction
    // diluted by ticks nobody could compare.
    let comparable = ticks - nonfinite_ticks;
    let argmax_flip_frac = if comparable > 0 {
        argmax_flips as f32 / comparable as f32
    } else {
        0.0
    };

    Ok(BlockContribution {
        ticks,
        mean_abs_delta,
        max_abs_delta: max_abs,
        sign_flip_frac,
        argmax_flip_frac,
        nonfinite_ticks,
    })
}

#[cfg(test)]
mod tests {
    use super::{argmax3, contribution, sign3};

    #[test]
    fn two_identical_traces_contribute_exactly_nothing() {
        let t = vec![[0.5, -0.2, 0.0], [0.7, 0.1, 0.3], [-0.1, 0.0, 0.9]];
        let c = contribution(&t, &t).expect("same length, non-empty");
        assert_eq!(c.ticks, 3);
        assert_eq!(c.mean_abs_delta, [0.0, 0.0, 0.0]);
        assert_eq!(c.max_abs_delta, [0.0, 0.0, 0.0]);
        assert_eq!(c.sign_flip_frac, [0.0, 0.0, 0.0]);
        assert_eq!(c.argmax_flip_frac, 0.0);
        assert_eq!(c.nonfinite_ticks, 0);
        assert!(!c.moved_any_output(1e-4));
    }

    #[test]
    fn the_mean_and_the_max_are_computed_per_output() {
        // steer moves by 0.2 then 0.6 (mean 0.4, max 0.6); throttle by a constant 0.1;
        // brake not at all.
        let with_block = vec![[0.5, 0.4, 0.2], [0.5, 0.8, 0.2]];
        let without = vec![[0.4, 0.2, 0.2], [0.4, 0.2, 0.2]];
        let c = contribution(&with_block, &without).unwrap();
        assert!((c.mean_abs_delta[0] - 0.1).abs() < 1e-6);
        assert!((c.mean_abs_delta[1] - 0.4).abs() < 1e-6);
        assert_eq!(c.mean_abs_delta[2], 0.0);
        assert!((c.max_abs_delta[1] - 0.6).abs() < 1e-6);
        assert!((c.peak_mean_abs_delta() - 0.4).abs() < 1e-6);
    }

    #[test]
    fn a_sign_flip_is_counted_even_when_the_delta_is_tiny() {
        // The case the mean alone cannot see: the wheel crosses zero by 0.002 and the
        // car now turns the other way. 2 of 4 ticks flip.
        let with_block = vec![
            [0.0, 0.001, 0.0],
            [0.0, 0.001, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.5, 0.0],
        ];
        let without = vec![
            [0.0, -0.001, 0.0],
            [0.0, -0.001, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.5, 0.0],
        ];
        let c = contribution(&with_block, &without).unwrap();
        // Two ticks move by 0.002 and two by 0.0 — mean 0.001, which any "is this
        // block negligible?" eyeball test would wave through.
        assert!(
            (c.mean_abs_delta[1] - 0.001).abs() < 1e-6,
            "the mean is tiny, as designed: {}",
            c.mean_abs_delta[1]
        );
        assert!(
            (c.sign_flip_frac[1] - 0.5).abs() < 1e-6,
            "half the ticks changed which way the car turns"
        );
    }

    #[test]
    fn a_command_resting_at_zero_never_counts_as_a_sign_flip() {
        // Guards the reason `sign3` exists instead of `f32::signum`: signum(+0.0)=+1 and
        // signum(-0.0)=-1 would report a flip between two zeros.
        assert_eq!(sign3(0.0), 0);
        assert_eq!(sign3(-0.0), 0);
        let with_block = vec![[0.0, 0.0, 0.0]; 5];
        let without = vec![[-0.0, -0.0, -0.0]; 5];
        let c = contribution(&with_block, &without).unwrap();
        assert_eq!(c.sign_flip_frac, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn the_argmax_fraction_counts_ticks_where_the_dominant_command_changed() {
        assert_eq!(argmax3([0.1, 0.9, 0.2]), 1);
        assert_eq!(
            argmax3([0.9, 0.9, 0.2]),
            0,
            "ties resolve to the lowest index"
        );
        let with_block = vec![[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]];
        let without = vec![[0.9, 0.1, 0.0], [0.9, 0.1, 0.0]];
        let c = contribution(&with_block, &without).unwrap();
        assert!((c.argmax_flip_frac - 0.5).abs() < 1e-6);
    }

    #[test]
    fn a_non_finite_tick_is_excluded_from_the_means_and_reported_on_its_own() {
        let with_block = vec![[0.5, f32::NAN, 0.0], [0.5, 0.4, 0.0]];
        let without = vec![[0.4, 0.2, 0.0], [0.4, 0.2, 0.0]];
        let c = contribution(&with_block, &without).unwrap();
        assert_eq!(c.nonfinite_ticks, 1);
        // Only the second tick is averaged: throttle 0.1, steer 0.2.
        assert!((c.mean_abs_delta[0] - 0.1).abs() < 1e-6);
        assert!((c.mean_abs_delta[1] - 0.2).abs() < 1e-6);
        assert!(c.mean_abs_delta.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn a_block_that_only_produces_non_finite_ticks_still_counts_as_having_moved() {
        let with_block = vec![[f32::INFINITY, f32::NAN, 0.0]; 3];
        let without = vec![[0.4, 0.2, 0.0]; 3];
        let c = contribution(&with_block, &without).unwrap();
        assert_eq!(c.nonfinite_ticks, 3);
        assert_eq!(
            c.peak_mean_abs_delta(),
            0.0,
            "no comparable tick to average"
        );
        assert!(
            c.moved_any_output(1e-4),
            "a block that blows the brain up has reached the output"
        );
    }

    #[test]
    fn traces_of_different_lengths_are_rejected_not_truncated() {
        let a = vec![[0.0, 0.0, 0.0]; 3];
        let b = vec![[0.0, 0.0, 0.0]; 2];
        let err = contribution(&a, &b).expect_err("must not compare a common prefix");
        assert!(
            err.contains("length mismatch"),
            "message names the cause: {err}"
        );
    }

    #[test]
    fn empty_traces_are_rejected_rather_than_reported_as_a_null_contribution() {
        let err = contribution(&[], &[]).expect_err("empty is a caller bug, not a null");
        assert!(err.contains("empty"), "message names the cause: {err}");
    }
}
