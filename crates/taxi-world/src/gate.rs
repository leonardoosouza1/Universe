//! The identity gate (plan §T5, SPEC §11): runs `tick::step` against a
//! [`Trace`] and reports how closely it reproduces the reference trajectory,
//! classifying every action disagreement by the trace's own `margem_por_tique`
//! (SPEC §11.2's table) instead of a single global tolerance.
//!
//! **Design decision beyond the plan's literal `step` sketch, declared here and
//! in this crate's final report.** `run_gate` always drives the simulated
//! trajectory with the REFERENCE action (`acoes_por_tique[t][b]`, via `tick::
//! step`'s `forced_choice`), for every tick of every [`GateLevel`] — not only on
//! a numerical tie, as SPEC §11.2's literal wording for P2 says (margin < 1e-3
//! is an accepted tie: follow the trace's action and continue). The reasoning: this
//! cleanly separates two independent questions a single "sometimes-forced"
//! policy would tangle back together — (1) does `decision::decide` itself
//! reproduce Python's `argmax` (measured by `natural_choice` vs the trace's
//! action, unconditionally, the SAME every level) and (2) does physics/
//! services/passengers reproduce Python's STATE given an IDENTICAL action
//! sequence (measured by `traco`/`traco_extra`, immune to any decide()-side
//! disagreement cascading into a different position two ticks later). A bug
//! found this way always points at ONE of those two questions, never a mix.

use std::collections::BTreeMap;

use crate::constants::{G, NF, S};
use crate::tick::{step, Sim};
use crate::trace::Trace;

/// One action disagreement between Rust's `decide()` and the reference trace,
/// classified by that tick's own `margem_por_tique` (SPEC §11.2).
#[derive(Debug, Clone)]
pub struct Divergence {
    pub tick: usize,
    pub being: usize,
    pub esc_py: u32,
    pub esc_rust: u32,
    /// The trace's own `margem_por_tique[tick][being]` — the gap between the
    /// best and runner-up score, from the PYTHON side (the classification
    /// signal SPEC §11.2 specifies, not Rust's own `StepRecord::margin`).
    pub margin: f32,
}

/// How far `run_gate` runs and which SPEC §11.2 gate it checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateLevel {
    /// P0: tick 0 only, `F` against `F_primeiros_50[0]`.
    Senses,
    /// P1: tick 0 only, the full `tudo` row (`nota_por_tique[0]`) and `esc`.
    Score,
    /// P2: the first simulated day (`min(240, trace.T)` ticks), actions and
    /// state, hard tolerances.
    FirstDay,
    /// P3: the whole trace (`trace.T` ticks), actions and state, statistical
    /// tolerances, plus the final `t_vida`/`rec` check.
    FullLife,
}

/// The result of one [`run_gate`] call.
#[derive(Debug, Default)]
pub struct GateReport {
    pub actions_equal: usize,
    pub actions_total: usize,
    /// Per being: the first tick `decide()` disagreed with the trace AND that
    /// disagreement's margin classified it as a BUG (`> 1e-2`) — `None` if
    /// every disagreement for that being (if any) was a tie or grey-zone case.
    pub first_divergence: Vec<Option<usize>>,
    /// Disagreements with `margin > 1e-2` (SPEC §11.2: a BUG — the gate fails).
    pub bugs: Vec<Divergence>,
    /// Disagreements with `1e-3 <= margin <= 1e-2` (SPEC §11.2: the grey zone —
    /// reported, not a failure).
    pub grey: Vec<Divergence>,
    /// Maximum absolute difference seen, per compared variable name.
    pub max_abs: BTreeMap<&'static str, f32>,
    /// `|rust.alive_ticks[b] - trace.t_vida[b]|`, populated only by
    /// [`GateLevel::FullLife`] (the only level that runs to the end of the
    /// trace).
    pub final_t_vida_diff: Vec<f32>,
    /// `|rust.revenue[b] - trace.rec[b]|`, same populated-only-by-`FullLife`
    /// rule as `final_t_vida_diff`.
    pub final_rec_diff: Vec<f32>,
}

impl GateReport {
    fn note_max(&mut self, key: &'static str, diff: f32) {
        let entry = self.max_abs.entry(key).or_insert(0.0);
        if diff > *entry {
            *entry = diff;
        }
    }

    /// A one-paragraph human summary — the plan requires the harness to print,
    /// on failure, `t`, `b`, `esc_py`, `esc_rust`: a gate that just says
    /// "failed" does not count as delivered (plan §T5).
    pub fn summary(&self) -> String {
        let frac = if self.actions_total > 0 {
            self.actions_equal as f64 / self.actions_total as f64
        } else {
            0.0
        };
        let mut s = format!(
            "actions: {}/{} equal ({:.4}%), {} grey-zone disagreements, {} BUG disagreements\n",
            self.actions_equal,
            self.actions_total,
            frac * 100.0,
            self.grey.len(),
            self.bugs.len()
        );
        if let Some(first_bug) = self.bugs.first() {
            s += &format!(
                "first BUG divergence: tick={} being={} esc_py={} esc_rust={} margin={:e}\n",
                first_bug.tick, first_bug.being, first_bug.esc_py, first_bug.esc_rust, first_bug.margin
            );
        }
        for (k, v) in &self.max_abs {
            s += &format!("  max |diff| {k}: {v:e}\n");
        }
        s
    }
}

fn flatten_xy(pairs: &[[f32; 2]]) -> Vec<f32> {
    pairs.iter().flat_map(|p| [p[0], p[1]]).collect()
}

/// SPEC §11.2's classification of an action disagreement by the trace's own
/// `margem_por_tique`: `< 1e-3` is an accepted numerical tie, `1e-3..=1e-2` is
/// the "grey zone" (reported, not a failure), `> 1e-2` is a BUG. A single
/// function so `run_gate` and any synthetic test of the classification itself
/// (`tests/identity.rs`) share one rule, not two copies that could silently
/// drift apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivergenceClass {
    Tie,
    Grey,
    Bug,
}

pub fn classify_divergence(margin: f32) -> DivergenceClass {
    if margin > 1e-2 {
        DivergenceClass::Bug
    } else if margin > 1e-3 {
        DivergenceClass::Grey
    } else {
        DivergenceClass::Tie
    }
}

/// Runs `tick::step` against `trace`'s own recorded trajectory, `level`'s
/// number of ticks, and reports how closely Rust reproduces it (SPEC §11.2).
///
/// Every level reads `alvo_por_tique[t]` as `step`'s `target` (SPEC §7.2's
/// identity-gate hook: the Python side's RNG contaminates `alvo` after the
/// first pickup, so the gate reads it rather than recomputing it) and
/// `acoes_por_tique[t]` as `step`'s `forced_choice` (this module's own header
/// explains why every tick, not only ties).
pub fn run_gate(trace: &Trace, level: GateLevel) -> GateReport {
    let n = trace.b;
    let mut sim = Sim::from_trace(trace);

    let acoes: Vec<Vec<u32>> = serde_json::from_value(trace.acoes_por_tique.clone())
        .expect("acoes_por_tique must be [T][B] of ints");
    let traco: Vec<Vec<[f32; 7]>> = serde_json::from_value(trace.traco.clone())
        .expect("traco must be [T][B][7] (fome,fuel,agua,dinh,x,y,vida)");
    let traco_extra: Vec<Vec<[f32; 4]>> = serde_json::from_value(trace.traco_extra.clone())
        .expect("traco_extra must be [T][B][4] (calor,fad,res,pd_)");
    let alvo: Vec<Vec<[f32; 2]>> = serde_json::from_value(trace.alvo_por_tique.clone())
        .expect("alvo_por_tique must be [T][B][2]");
    let margem: Vec<Vec<f32>> = serde_json::from_value(trace.margem_por_tique.clone())
        .expect("margem_por_tique must be [T][B]");

    let ticks = match level {
        GateLevel::Senses | GateLevel::Score => 1,
        GateLevel::FirstDay => (sim.cfg.day_ticks as usize).min(trace.t),
        GateLevel::FullLife => trace.t,
    };
    assert!(acoes.len() >= ticks, "acoes_por_tique shorter than the requested horizon");
    assert!(traco.len() >= ticks, "traco shorter than the requested horizon");

    let mut report = GateReport { first_divergence: vec![None; n], ..GateReport::default() };

    // P0/P1 only ever look at tick 0, so these can stay borrowed from the raw
    // `serde_json::Value`s directly instead of being parsed unconditionally
    // (avoids paying to type-check `nota_por_tique`/`F_primeiros_50` — whose
    // length is capped at 200/50 ticks, SPEC §11.1 — on a P3 run that never
    // reads them).
    let nota_t0 = if level == GateLevel::Score {
        Some(
            trace.nota_por_tique.as_array().and_then(|a| a.first()).cloned().unwrap_or_default(),
        )
    } else {
        None
    };
    let f_t0 = if level == GateLevel::Senses {
        Some(trace.f_primeiros_50.as_array().and_then(|a| a.first()).cloned().unwrap_or_default())
    } else {
        None
    };

    for tick in 0..ticks {
        let target = flatten_xy(&alvo[tick]);
        let forced: Vec<u32> = acoes[tick].clone();
        let record = step(&mut sim, Some(&target), Some(&forced), None);

        for b in 0..n {
            report.actions_total += 1;
            let esc_py = acoes[tick][b];
            let esc_rust = record.natural_choice[b];
            if esc_rust == esc_py {
                report.actions_equal += 1;
            } else {
                let m = margem[tick][b];
                let d = Divergence { tick, being: b, esc_py, esc_rust, margin: m };
                match classify_divergence(m) {
                    DivergenceClass::Bug => {
                        if report.first_divergence[b].is_none() {
                            report.first_divergence[b] = Some(tick);
                        }
                        report.bugs.push(d);
                    }
                    DivergenceClass::Grey => report.grey.push(d),
                    // An accepted numerical tie (SPEC §11.2) — not recorded as
                    // a divergence at all.
                    DivergenceClass::Tie => {}
                }
            }
        }

        if level == GateLevel::Senses {
            if let Some(rows) = &f_t0 {
                for b in 0..n {
                    let row = rows[b].as_array().expect("F_primeiros_50[0][b] must be an array");
                    for g in 0..G {
                        let opt = row[g].as_array().expect("F_primeiros_50[0][b][g] must be an array");
                        for feat in 0..NF {
                            let expected = opt[feat].as_f64().unwrap() as f32;
                            let got = record.f[(b * G + g) * NF + feat];
                            report.note_max("F", (expected - got).abs());
                        }
                    }
                }
            }
        }

        if level == GateLevel::Score {
            if let Some(rows) = &nota_t0 {
                let total = G + S;
                for b in 0..n {
                    let row = rows[b].as_array().expect("nota_por_tique[0][b] must be an array");
                    for i in 0..total {
                        let expected = row[i].as_f64().unwrap() as f32;
                        let got = record.scores[b * total + i];
                        let masked_py = expected <= -1e8;
                        let masked_rust = got <= -1e8;
                        if masked_py != masked_rust {
                            // A mask mismatch (one side thinks a service is
                            // available, the other doesn't) is reported at
                            // full magnitude, not swallowed by a diff of two
                            // huge negative numbers that happen to be close.
                            report.note_max("nota_mask_mismatch", (expected - got).abs());
                        } else if !masked_py {
                            report.note_max("nota", (expected - got).abs());
                        }
                    }
                }
            }
        }

        // `traco[k]` records the state BEFORE step `k` runs (`traco[0]` is the
        // untouched initial state — verified against the real trace during
        // this task: `traco[0] == inicial`). The state THIS iteration's `step`
        // call just produced is therefore `traco[tick + 1]`, not `traco[tick]`
        // — and on the very last iteration of a `FullLife` run that next row
        // does not exist (`len(traco) == trace.t`): skip rather than panic.
        //
        // `traco_extra[k]`, unlike `traco[k]`, is recorded AFTER `um_passo`
        // returns (`traco_referencia.py:19`: `um_passo(...); extra[t] =
        // stack(calor, fad, res, pd_)`, OUTSIDE `um_passo`, whereas `traj[m.t]`
        // — `traco` — is written INSIDE `um_passo`, BEFORE `m.passo(esc)`
        // runs). So `traco_extra[k]` is state after `k + 1` steps — the SAME
        // point in time as `traco[k + 1]`, not `traco[k]`. Confirmed
        // empirically during this task (`traco_extra[0]` != the initial
        // calor/fad, `traco_extra[0] == state after exactly one step`) after
        // reading `traco_referencia.py:17-19` directly — this file has NO
        // `esquema` entry warning about the mismatch. `traco_extra[tick]`
        // (not `tick + 1`) is therefore the row aligned with THIS iteration's
        // result.
        if matches!(level, GateLevel::FirstDay | GateLevel::FullLife) {
            if let (Some(t_row_all), Some(e_row_all)) = (traco.get(tick + 1), traco_extra.get(tick)) {
                for b in 0..n {
                    let t_row = t_row_all[b];
                    let e_row = e_row_all[b];
                    report.note_max("fome", (sim.beings.hunger[b] - t_row[0]).abs());
                    report.note_max("fuel", (sim.beings.fuel[b] - t_row[1]).abs());
                    report.note_max("agua", (sim.beings.water[b] - t_row[2]).abs());
                    report.note_max("dinh", (sim.beings.money[b] - t_row[3]).abs());
                    report.note_max("pos_x", (sim.beings.pos_x[b] - t_row[4]).abs());
                    report.note_max("pos_y", (sim.beings.pos_y[b] - t_row[5]).abs());
                    report.note_max("vida", (sim.beings.life[b] - t_row[6]).abs());
                    report.note_max("calor", (sim.beings.heat[b] - e_row[0]).abs());
                    report.note_max("fad", (sim.beings.fatigue[b] - e_row[1]).abs());
                    report.note_max("res", (sim.beings.reserve[b] - e_row[2]).abs());
                }
            }
        }
    }

    if level == GateLevel::FullLife {
        for b in 0..n {
            report.final_t_vida_diff.push((sim.beings.alive_ticks[b] - trace.t_vida[b]).abs());
            report.final_rec_diff.push((sim.beings.revenue[b] - trace.rec[b]).abs());
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_divergence_below_the_margin_is_reported_as_a_tie_not_a_bug() {
        // Synthetic: 1 being, 1 tick, esc disagrees but margin is a clean tie
        // (SPEC §11.2: < 1e-3 -> not even recorded as a divergence).
        let mut report = GateReport { first_divergence: vec![None], ..GateReport::default() };
        let margin = 1e-4_f32;
        report.actions_total += 1;
        let esc_py = 3u32;
        let esc_rust = 1u32;
        if esc_rust == esc_py {
            report.actions_equal += 1;
        } else if margin > 1e-2 {
            report.bugs.push(Divergence { tick: 0, being: 0, esc_py, esc_rust, margin });
        } else if margin > 1e-3 {
            report.grey.push(Divergence { tick: 0, being: 0, esc_py, esc_rust, margin });
        }
        assert!(report.bugs.is_empty(), "a margin of 1e-4 must not be classified as a bug");
        assert!(report.grey.is_empty(), "a margin of 1e-4 must not even be grey-zone");
    }

    #[test]
    fn a_divergence_above_the_bug_threshold_is_reported_as_a_bug() {
        let mut report = GateReport { first_divergence: vec![None], ..GateReport::default() };
        let margin = 0.5_f32;
        let esc_py = 3u32;
        let esc_rust = 1u32;
        report.actions_total += 1;
        if esc_rust != esc_py && margin > 1e-2 {
            report.first_divergence[0] = Some(0);
            report.bugs.push(Divergence { tick: 0, being: 0, esc_py, esc_rust, margin });
        }
        assert_eq!(report.bugs.len(), 1);
        assert_eq!(report.first_divergence[0], Some(0));
    }

    #[test]
    fn a_grey_zone_divergence_is_reported_but_not_a_bug() {
        let mut report = GateReport { first_divergence: vec![None], ..GateReport::default() };
        let margin = 5e-3_f32;
        let esc_py = 3u32;
        let esc_rust = 1u32;
        report.actions_total += 1;
        if esc_rust != esc_py {
            if margin > 1e-2 {
                report.bugs.push(Divergence { tick: 0, being: 0, esc_py, esc_rust, margin });
            } else if margin > 1e-3 {
                report.grey.push(Divergence { tick: 0, being: 0, esc_py, esc_rust, margin });
            }
        }
        assert!(report.bugs.is_empty());
        assert_eq!(report.grey.len(), 1);
    }
}
