//! The SENSITIVITY GATE: can each gene block still move the steering wheel?
//!
//! Third of C3's rulers (`docs/PRD-CEREBRO-NEUROMOD.md` §C3) and a direct port of the
//! lab's `neuromod-lab/src/testes_sentidos.py:1-3`: *perturb only the genes of one
//! block, replay, and check whether the output changed*. In the lab it took 30 s and
//! would have caught finding 84 — sixteen body senses that were invisible to the
//! Linear architecture's steering. A block that cannot move the output is not a weak
//! mechanism, it is an unwired one, and no number of GPU generations will tell the two
//! apart: both look like "the new arm did not beat the control".
//!
//! # It runs on the CPU, and that decides what it can see
//!
//! The gate has to be runnable before anything is queued on the shared GPU (PRD §0.4),
//! so the replay here uses `SovereignEngine` (`src/engine.rs:195-260`), the CPU brain
//! this repository already has. It does NOT reimplement the brain kernel. That
//! prohibition is not style: `src/bin/screen.rs`'s own header records that this project
//! already shipped two disjoint brains once (`cortex_evolve` ended with zero references
//! to `SovereignEngine`), and a CPU replica of `brain.wgsl` written for this gate would
//! be the third.
//!
//! The price is that `SovereignEngine` is not `BrainGpu`. Several genome blocks are read
//! only by `brain.wgsl` or by the host loop inside the `gpu_evolve` binary, and in an
//! open-loop CPU replay they are structurally unable to move anything. Calling those
//! "blind" would be crying wolf, so every group carries an explicit [`Reach`]:
//!
//! - [`Reach::Runner`] — this replay reads the block. These are GATED: not moving the
//!   output fails the run.
//! - [`Reach::OffPath`] — this replay provably cannot read the block, with the reason
//!   and the evidence recorded in [`groups`]. Not moving is expected and does not fail.
//!
//! The `OffPath` declaration is not a free pass: it is FALSIFIABLE. An off-path group
//! that *does* move an output fails the run just as loudly, because then the list is
//! wrong and the reader needs to know before trusting any other row.
//!
//! # Why a group can be legitimately blind here
//!
//! Three distinct reasons, all of them real on this branch:
//!
//! 1. **The block lives in the GPU kernel.** The four P1.7 mechanisms
//!    (`neuron_alpha`, `neuron_period_l1`, `neuron_threshold_l1`, `neuron_activation`)
//!    exist only in `brain.wgsl`; `SovereignEngine` has no per-neuron memory, period,
//!    threshold or activation id to receive them.
//! 2. **The block lives in the host loop of a BINARY.** The output KAN, the slow
//!    brain's three modulations and the M4 conductor are applied in
//!    `gpu_evolve/context.rs:1600-1624` and `:1862-1907` — binary code the library
//!    cannot call. `Genome::apply_to_engine` loads `slow` into the engine
//!    (`evolution.rs:697-699`) but `execute_cycle` never ticks it, so the weights are
//!    present and mute.
//! 3. **The block reads state the recording does not carry.** A recorded `UWAV` wave is
//!    192 sensory channels and nothing else. A block whose input is an internal state
//!    the recording never captured has nothing to read. `sensors` is today's example
//!    (its genes change `perception.wgsl`, which an open-loop replay does not run); the
//!    hippocampus C1 adds will be the next one, since its input is a per-sector memory
//!    written by lap events that no wave row contains.
//!
//! That third case is the one to watch, because it is the only one a *synthetic input*
//! can fix. When C1 lands `Genome::hip`, this gate keeps it honest by feeding the
//! hippocampus a fixed non-zero memory instead of the zero a replay would produce — the
//! `--mem-sintetico` flag of `src/bin/sensitivity.rs`. The flag is NOT implemented on
//! this branch: there is no `hip` block to inject into, and a flag that silently does
//! nothing is worse than a documented absence. See that binary's header.
//!
//! # This module is pure
//!
//! No GPU, no file I/O, no globals written. It takes a genome and a wave and returns a
//! report. The binary is a CLI over it, and the tests call it directly.

mod replay;
mod report;

pub use replay::{replay_open_loop, synthetic_wave};
pub use report::{classify, GroupRow, SensitivityReport, Verdict};

use crate::config::iara::brain::SENSORY_WAVE_SIZE;
use crate::gpu::contribution::{contribution, BlockContribution};
use crate::gpu::evolution::Genome;

/// Mean `|Δ|` on a command above which a block counts as having moved it. The PRD's
/// figure (§C3 acceptance: "moves at least one output above 1e-4").
pub const MOVE_THRESHOLD: f32 = 1e-4;

/// Perturbation sizes the gate sweeps by default, largest first: 0.5 is the lab's
/// `gene_min` (a gene at 0.5 is exactly the one `dof` counts), 0.1 is the small-signal
/// check — a block that only answers to 0.5 and not to 0.1 is saturating somewhere, and
/// the two columns side by side say so.
pub const DEFAULT_DELTAS: [f32; 2] = [0.5, 0.1];

/// Whether this replay can read a block at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reach {
    /// The CPU replay in [`replay_open_loop`] reads these genes. Gated.
    Runner,
    /// The replay provably cannot read these genes; the string is the reason, and it is
    /// printed next to the row so nobody has to take it on faith.
    OffPath(&'static str),
}

/// One gene block and its reach.
#[derive(Clone, Copy, Debug)]
pub struct GeneGroup {
    /// The `Genome` field name, spelled exactly as the struct spells it.
    pub name: &'static str,
    pub reach: Reach,
}

/// Every heritable block of `Genome` (`src/gpu/evolution.rs:93-182`), in the genome's
/// own disk order, with the reason each one is or is not readable by the CPU replay.
///
/// `l1` and `l2` are in the list as POSITIVE CONTROLS, not because the PRD asks for
/// them: they are the two blocks that must move the output under any working harness.
/// If they ever read "blind", the harness is broken and every other row on the matrix is
/// meaningless — which is precisely the failure a gate made only of new blocks cannot
/// detect.
///
/// TODO(C1): when the hippocampus block lands, add
/// `GeneGroup { name: "hip", reach: Reach::Runner }` here AND the matching call into
/// `gpu::hippocampus` inside [`replay_open_loop`] — a group added here without the
/// runner hook reports `blind` and fails, which is the intended direction (the gate
/// refuses to certify a block it does not actually exercise).
/// TODO(C2): the same for `tal`.
pub fn groups() -> Vec<GeneGroup> {
    vec![
        GeneGroup {
            name: "l1",
            reach: Reach::Runner,
        },
        GeneGroup {
            name: "l2",
            reach: Reach::Runner,
        },
        GeneGroup {
            name: "kan_knots",
            reach: Reach::OffPath(
                "output KAN is evaluated by the host loop (context.rs:1862-1866); \
                 apply_to_engine says so itself (evolution.rs:700)",
            ),
        },
        GeneGroup {
            name: "slow",
            reach: Reach::OffPath(
                "loaded into the engine (evolution.rs:697-699) but execute_cycle never \
                 ticks the slow brain; its 3 outputs reach the command in context.rs:1855",
            ),
        },
        GeneGroup {
            name: "tau",
            reach: Reach::Runner,
        },
        GeneGroup {
            name: "sensors",
            reach: Reach::OffPath(
                "perception genes read by perception.wgsl; an open-loop replay consumes \
                 an already-recorded wave, so no perception step runs",
            ),
        },
        GeneGroup {
            name: "neuron_alpha",
            reach: Reach::OffPath("P1.7 mechanism 1 exists only in brain.wgsl"),
        },
        GeneGroup {
            name: "neuron_period_l1",
            reach: Reach::OffPath("P1.7 mechanism 2 (period) exists only in brain.wgsl"),
        },
        GeneGroup {
            name: "neuron_threshold_l1",
            reach: Reach::OffPath("P1.7 mechanism 2 (threshold) exists only in brain.wgsl"),
        },
        GeneGroup {
            name: "neuron_activation",
            reach: Reach::OffPath("P1.7 mechanism 3 exists only in brain.wgsl"),
        },
        GeneGroup {
            name: "gates_w",
            reach: Reach::OffPath(
                "M4 conductor: computed on the host from the slow brain (context.rs:1600-1624) \
                 and applied inside brain.wgsl:871",
            ),
        },
    ]
}

/// The block's genes, flattened. `kan_knots` is an array of arrays and is copied into a
/// 36-value vector; every other block is a `Vec<f32>` already.
fn block_genes(g: &Genome, name: &str) -> Option<Vec<f32>> {
    Some(match name {
        "l1" => g.l1.clone(),
        "l2" => g.l2.clone(),
        "kan_knots" => g.kan_knots.iter().flatten().copied().collect(),
        "slow" => g.slow.clone(),
        "tau" => g.tau.clone(),
        "sensors" => g.sensors.clone(),
        "neuron_alpha" => g.neuron_alpha.clone(),
        "neuron_period_l1" => g.neuron_period_l1.clone(),
        "neuron_threshold_l1" => g.neuron_threshold_l1.clone(),
        "neuron_activation" => g.neuron_activation.clone(),
        "gates_w" => g.gates_w.clone(),
        _ => return None,
    })
}

/// Adds `delta` to every gene of one block, leaving the rest of the genome untouched.
/// Returns `None` for a name that is not a block — a typo in [`groups`] must not be
/// silently interpreted as "this block contributes nothing".
///
/// Additive and uniform on purpose. A multiplicative perturbation is invisible on a
/// block born at zero (`gates_w`, `neuron_alpha`, and the `hip` block C1 will add all
/// start neutral at 0.0), and those are exactly the blocks this gate exists for.
pub fn perturbed(g: &Genome, name: &str, delta: f32) -> Option<Genome> {
    let mut out = g.clone();
    {
        let slice: &mut [f32] = match name {
            "l1" => &mut out.l1,
            "l2" => &mut out.l2,
            "kan_knots" => {
                for row in out.kan_knots.iter_mut() {
                    for k in row.iter_mut() {
                        *k += delta;
                    }
                }
                return Some(out);
            }
            "slow" => &mut out.slow,
            "tau" => &mut out.tau,
            "sensors" => &mut out.sensors,
            "neuron_alpha" => &mut out.neuron_alpha,
            "neuron_period_l1" => &mut out.neuron_period_l1,
            "neuron_threshold_l1" => &mut out.neuron_threshold_l1,
            "neuron_activation" => &mut out.neuron_activation,
            "gates_w" => &mut out.gates_w,
            _ => return None,
        };
        for w in slice.iter_mut() {
            *w += delta;
        }
    }
    Some(out)
}

/// Runs the gate: for every group, perturb only that block by ±δ for each δ, replay, and
/// compare against the unperturbed replay of the same wave.
///
/// `waves` is `ticks × SENSORY_WAVE_SIZE` f32 (see [`replay_open_loop`]). Returns an
/// error rather than an empty report when there is nothing to replay — a matrix of zeros
/// from an empty wave reads exactly like a brain with no wiring at all.
pub fn run_gate(
    genome: &Genome,
    waves: &[f32],
    dt: f32,
    deltas: &[f32],
) -> Result<SensitivityReport, String> {
    if waves.len() < SENSORY_WAVE_SIZE {
        return Err(format!(
            "wave has {} values, fewer than one {SENSORY_WAVE_SIZE}-channel tick",
            waves.len()
        ));
    }
    if waves.len() % SENSORY_WAVE_SIZE != 0 {
        return Err(format!(
            "wave has {} values, not a whole number of {SENSORY_WAVE_SIZE}-channel ticks",
            waves.len()
        ));
    }
    if deltas.is_empty() {
        return Err("no perturbation sizes given: nothing would be tested".to_string());
    }

    let ticks = waves.len() / SENSORY_WAVE_SIZE;
    let baseline = replay_open_loop(genome, waves, dt);

    let mut rows = Vec::new();
    for group in groups() {
        let genes = block_genes(genome, group.name)
            .ok_or_else(|| format!("group '{}' is not a Genome block", group.name))?;
        let n_genes = genes.len();
        let n_nonzero = genes.iter().filter(|w| **w != 0.0).count();

        let mut peak = [0.0f32; 3];
        let mut peak_sign = 0.0f32;
        let mut peak_argmax = 0.0f32;
        let mut nonfinite = 0usize;
        if n_genes > 0 {
            for &d in deltas {
                for signed in [d, -d] {
                    let probe = perturbed(genome, group.name, signed)
                        .ok_or_else(|| format!("group '{}' is not a Genome block", group.name))?;
                    let trace = replay_open_loop(&probe, waves, dt);
                    let c: BlockContribution = contribution(&trace, &baseline)?;
                    for (k, slot) in peak.iter_mut().enumerate() {
                        if c.mean_abs_delta[k] > *slot {
                            *slot = c.mean_abs_delta[k];
                        }
                        if c.sign_flip_frac[k] > peak_sign {
                            peak_sign = c.sign_flip_frac[k];
                        }
                    }
                    if c.argmax_flip_frac > peak_argmax {
                        peak_argmax = c.argmax_flip_frac;
                    }
                    nonfinite += c.nonfinite_ticks;
                }
            }
        }

        // A single sign flip counts as movement even when every mean stays under the
        // threshold. That is the whole argument of `contribution`'s module doc: a block
        // that pushes the wheel across zero by 0.002 changed which way the car turns,
        // and a gate reading only means would call it blind.
        let moved = nonfinite > 0 || peak_sign > 0.0 || peak.iter().any(|v| *v > MOVE_THRESHOLD);
        let verdict = classify(n_genes, group.reach, moved);
        rows.push(GroupRow {
            name: group.name,
            reach: group.reach,
            n_genes,
            n_nonzero,
            peak_abs_delta: peak,
            peak_sign_flip_frac: peak_sign,
            peak_argmax_flip_frac: peak_argmax,
            nonfinite_ticks: nonfinite,
            verdict,
        });
    }

    Ok(SensitivityReport {
        ticks,
        deltas: deltas.to_vec(),
        rows,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        classify, groups, perturbed, run_gate, synthetic_wave, Reach, Verdict, DEFAULT_DELTAS,
        SENSORY_WAVE_SIZE,
    };
    use crate::gpu::evolution::Genome;

    fn fixture_genome() -> Genome {
        // `random` is seeded, so this genome is the same on every machine and every run.
        Genome::random(0xC3_5E_1751)
    }

    // The replay's own guards (determinism, genome dependence, wave shape) live in
    // `replay.rs`; what is tested here is the gate built on top of it.

    #[test]
    fn perturbing_a_block_leaves_every_other_block_untouched() {
        let g = fixture_genome();
        let p = perturbed(&g, "tau", 0.25).expect("tau is a block");
        assert_eq!(p.l1, g.l1);
        assert_eq!(p.l2, g.l2);
        assert_eq!(p.slow, g.slow);
        for (a, b) in p.tau.iter().zip(g.tau.iter()) {
            assert!((a - b - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn an_unknown_block_name_is_rejected_rather_than_ignored() {
        let g = fixture_genome();
        assert!(
            perturbed(&g, "hip", 0.5).is_none(),
            "hip does not exist yet"
        );
    }

    #[test]
    fn the_matrix_has_one_row_per_declared_group() {
        let g = fixture_genome();
        let w = synthetic_wave(16);
        let r = run_gate(&g, &w, 1.0 / 64.0, &DEFAULT_DELTAS).expect("wave is well formed");
        assert_eq!(r.rows.len(), groups().len());
        assert_eq!(r.ticks, 16);
        let names: Vec<&str> = r.rows.iter().map(|x| x.name).collect();
        assert_eq!(names[0], "l1", "rows follow the genome's disk order");
        assert_eq!(*names.last().unwrap(), "gates_w");
    }

    #[test]
    fn the_positive_controls_move_the_output() {
        // If L1 or L2 ever read blind, the harness is broken and no other row means
        // anything — this is the test that says so first.
        let g = fixture_genome();
        let w = synthetic_wave(16);
        let r = run_gate(&g, &w, 1.0 / 64.0, &[0.5]).unwrap();
        for name in ["l1", "l2"] {
            let row = r.rows.iter().find(|x| x.name == name).unwrap();
            assert_eq!(row.verdict, Verdict::Moved, "{name} must reach the command");
        }
    }

    #[test]
    fn a_block_the_replay_cannot_read_is_reported_off_path_not_blind() {
        let g = fixture_genome();
        let w = synthetic_wave(16);
        let r = run_gate(&g, &w, 1.0 / 64.0, &[0.5]).unwrap();
        let row = r.rows.iter().find(|x| x.name == "slow").unwrap();
        assert!(matches!(row.reach, Reach::OffPath(_)));
        assert_ne!(
            row.verdict,
            Verdict::Blind,
            "off-path is not a gate failure"
        );
    }

    #[test]
    fn every_verdict_branch_is_reachable_including_the_two_that_fail() {
        // A gate whose failure branches are never exercised is a gate that has never
        // been shown to fail. `classify` is extracted precisely so they can be.
        let off = Reach::OffPath("reason");
        assert_eq!(classify(0, Reach::Runner, false), Verdict::Absent);
        assert_eq!(classify(0, off, true), Verdict::Absent);
        assert_eq!(classify(9, Reach::Runner, true), Verdict::Moved);
        assert_eq!(classify(9, Reach::Runner, false), Verdict::Blind);
        assert_eq!(classify(9, off, false), Verdict::OffPathSilent);
        assert_eq!(classify(9, off, true), Verdict::OffPathMoved);
    }

    #[test]
    fn a_null_perturbation_certifies_nothing() {
        // End-to-end proof that the gate CAN go red, without lying about any block's
        // reach: δ = 0 leaves every genome untouched, so both traces are bit-identical
        // and every on-path group has to come back BLIND. A gate that stayed green here
        // would be green by construction.
        let g = fixture_genome();
        let w = synthetic_wave(16);
        let r = run_gate(&g, &w, 1.0 / 64.0, &[0.0]).expect("well-formed wave");
        assert!(!r.passed(), "a null perturbation must not certify:\n{r}");
        let blind: Vec<&str> = r.failures().iter().map(|x| x.name).collect();
        assert_eq!(
            blind,
            vec!["l1", "l2", "tau"],
            "exactly the on-path groups go blind; off-path rows stay silent"
        );
        assert!(r.failures().iter().all(|x| x.verdict == Verdict::Blind));
    }

    #[test]
    fn an_all_zero_wave_does_not_silence_this_brain() {
        // Measured while building the gate, and recorded here because it is a trap for
        // the next person writing a negative control: an all-zero wave is NOT a silent
        // brain. `CfCCell` injects deterministic denoising noise into its state on every
        // tick (`cfc.rs:79-81,176-190`, `denoise_mag` = 0.10), so `filtered_retina` is
        // non-zero even with no input, L1 still has something to weigh, and perturbing
        // it still moves the command. Use δ = 0 for a null control, not a null wave.
        let g = fixture_genome();
        let w = vec![0.0f32; 16 * SENSORY_WAVE_SIZE];
        let r = run_gate(&g, &w, 1.0 / 64.0, &[0.5]).expect("a zero wave is well formed");
        let l1 = r.rows.iter().find(|x| x.name == "l1").unwrap();
        assert_eq!(
            l1.verdict,
            Verdict::Moved,
            "denoising noise keeps L1 audible"
        );
    }

    #[test]
    fn a_wave_that_is_not_a_whole_number_of_ticks_is_rejected() {
        let g = fixture_genome();
        let mut w = synthetic_wave(4);
        w.pop();
        let err = run_gate(&g, &w, 1.0 / 64.0, &DEFAULT_DELTAS).expect_err("partial tick");
        assert!(err.contains("whole number"), "{err}");
    }

    #[test]
    fn an_empty_wave_is_rejected_rather_than_reported_as_a_matrix_of_zeros() {
        let g = fixture_genome();
        let err = run_gate(&g, &[], 1.0 / 64.0, &DEFAULT_DELTAS).expect_err("empty wave");
        assert!(err.contains("fewer than one"), "{err}");
    }
}
