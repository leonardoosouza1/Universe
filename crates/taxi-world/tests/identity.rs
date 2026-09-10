//! T5's identity gate acceptance tests (plan §T5 "Testes de aceite", SPEC §11.2).
//!
//! The tiny-fixture test always runs; the four gates P0-P3 are `#[ignore]`d
//! (this repo's convention for a test needing an external resource,
//! `CLAUDE.md:17-24`) and need `TAXI_TRACO_REF` pointed at a real
//! `traco_ref.json` (or its JSON-sanitized copy — `nevoa: Infinity` is not
//! valid JSON, see this crate's final report). Run them with:
//!
//! ```text
//! TAXI_TRACO_REF=/path/to/traco_ref.json CARGO_BUILD_JOBS=4 \
//!   cargo test -p taxi-world --release -- --ignored
//! ```
//!
//! **History (2026-09-10) — two bugs found and fixed by running against two
//! trace generations, kept here because both were invisible without the
//! second trace.**
//!
//! 1. The first `traco_ref.json` (`e81`, and its sanitized copy) was missing
//!    `mapa.fase_preco` (`neuromod-lab/src/campo.py:94,124,154`: `gerar_mapa`
//!    DOES compute a real, per-district random `fase_preco`, but that export
//!    predated the field being added to the JSON — `Trace::to_world` already
//!    documented the gap and defaulted it to zero, `src/trace.rs`). Against
//!    THAT trace, P0/P1 passed (`t=0` never needs a day-price adjustment) but
//!    P2/P3 showed state divergence isolated ENTIRELY to price-derived
//!    variables (`dinh`/`fuel`/`res` wrong, `pos`/`fome`/`agua`/`vida`/`calor`/
//!    `fad` bit-exact) — diagnosed by hand-computing one being's fare with
//!    Rust's own formula (it matched Rust's own output exactly), so the
//!    mismatch against Python had to be an input, not a formula.
//! 2. Once a corrected trace (with real, non-zero `fase_preco`) arrived, P0/P1
//!    THEMSELVES started failing (`max|ΔF|` jumped from 2.4e-7 to 0.94) — a
//!    second, genuine bug this crate had: `tick::Sim::new` seeded `price_mul`
//!    by calling `physics::update_daily_prices` at `t=0`, which computes
//!    `pmul0·(1 + amp·sin(2π·(0 + fase_preco)))` — NOT `pmul0` itself unless
//!    `fase_preco = 0`. Python's own `s.pmul` starts as `s.pmul0.clone()`
//!    verbatim (`campo.py:149`); the sine adjustment (SPEC §3.1 item 2) only
//!    ever runs inside `passo()`, i.e. from the FIRST `step()` call onward,
//!    never before it. With the first trace's `fase_preco = 0` this bug was
//!    invisible (the wrong formula degenerates to the right answer at phase
//!    zero) — fixed in `tick.rs` (`Sim::new` now copies `price_mul_base`
//!    directly).
//!
//! With both fixed, P0-P3 all pass against the corrected trace: 100% action
//! agreement at every level, state within f32 rounding
//! (`dinh` 3.8e-6, `fuel` 7.5e-9, `vida` 6.9e-3), and `t_vida`/`rec` bit-exact
//! for all 16 beings over the full 2400 ticks (see this crate's final report
//! for the numbers).

use std::path::{Path, PathBuf};

use taxi_world::gate::{classify_divergence, run_gate, DivergenceClass, GateLevel};
use taxi_world::Trace;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_trace.json")
}

fn real_trace_path() -> String {
    std::env::var("TAXI_TRACO_REF")
        .expect("set TAXI_TRACO_REF to the path of a real traco_ref.json (sanitized: no literal Infinity)")
}

#[test]
fn a_divergence_below_the_margin_is_reported_as_a_tie_not_a_bug() {
    // Synthetic margins spanning the three SPEC §11.2 bands, independent of any
    // trace file — this is `gate::classify_divergence` itself, not `run_gate`'s
    // wiring (that wiring is exercised by the tiny-fixture and #[ignore]d tests
    // below).
    assert_eq!(classify_divergence(1e-4), DivergenceClass::Tie);
    assert_eq!(classify_divergence(9.99e-4), DivergenceClass::Tie);
    assert_eq!(classify_divergence(5e-3), DivergenceClass::Grey);
    assert_eq!(classify_divergence(1e-2), DivergenceClass::Grey, "the boundary itself is not yet a bug (SPEC: > 1e-2)");
    assert_eq!(classify_divergence(1e-2 + 1e-6), DivergenceClass::Bug);
    assert_eq!(classify_divergence(0.5), DivergenceClass::Bug);
}

/// Always runs — no external resource needed. Runs the tiny hand-built fixture
/// (`tests/fixtures/tiny_trace.json`, 2 beings, 3 ticks) through `tick::step`
/// and checks the result reproduces the fixture's OWN recorded trajectory
/// (`traco`/`traco_extra`/`acoes_por_tique`/`margem_por_tique`) to a tight
/// tolerance — this is a wiring/regression test (does `step` compose T2-T4 in
/// the right order), not a correctness gate against Python (P0-P3, below,
/// are that). The fixture's derived per-tick fields were regenerated FROM this
/// same `tick.rs` implementation once it existed (see this crate's final
/// report on why the fixture as T1 left it could not have passed any such
/// test), so the tolerance here can be tight — both sides are the same code.
#[test]
fn the_tiny_fixture_reproduces_its_own_trace() {
    let trace = Trace::load(&fixture_path()).expect("tests/fixtures/tiny_trace.json must parse");
    let report = run_gate(&trace, GateLevel::FullLife);

    println!("{}", report.summary());

    assert_eq!(
        report.actions_equal, report.actions_total,
        "the tiny fixture's own recorded actions must match decide()'s natural choice at every tick"
    );
    assert!(report.bugs.is_empty(), "no BUG-classified action divergence: {:?}", report.bugs);
    for (k, v) in &report.max_abs {
        assert!(*v < 1e-3, "tiny fixture: max |diff| for {k} was {v}, expected < 1e-3");
    }
    for (b, d) in report.final_t_vida_diff.iter().enumerate() {
        assert!(*d < 1e-3, "being {b}: t_vida diff {d}");
    }
    for (b, d) in report.final_rec_diff.iter().enumerate() {
        assert!(*d < 1e-3, "being {b}: rec diff {d}");
    }
}

/// P0 (SPEC §11.2): tick 0's `F` against `F_primeiros_50[0]`, tolerance `1e-4`.
#[test]
#[ignore = "needs the real traco_ref.json (T0); point TAXI_TRACO_REF at it"]
fn senses_match_the_python_reference_on_the_first_tick() {
    let trace = Trace::load(Path::new(&real_trace_path())).expect("real trace must parse");
    let report = run_gate(&trace, GateLevel::Senses);
    println!("P0 (senses):\n{}", report.summary());
    let max_f = *report.max_abs.get("F").expect("F must have been compared");
    assert!(max_f <= 1e-4, "P0: max|deltaF| = {max_f:e}, SPEC §11.2 requires <= 1e-4");
}

/// P1 (SPEC §11.2): tick 0's full `tudo` row against `nota_por_tique[0]`
/// (tolerance `1e-3` on the unmasked entries, and the mask itself — which
/// service options are `-1e9` — must agree exactly), plus `esc` identical for
/// every being.
#[test]
#[ignore = "needs the real traco_ref.json (T0); point TAXI_TRACO_REF at it"]
fn scores_and_the_first_choice_match_the_python_reference() {
    let trace = Trace::load(Path::new(&real_trace_path())).expect("real trace must parse");
    let report = run_gate(&trace, GateLevel::Score);
    println!("P1 (score):\n{}", report.summary());
    assert!(
        report.max_abs.get("nota_mask_mismatch").is_none(),
        "P1: at least one service's availability mask disagreed with Python (sv boolean must be identical)"
    );
    let max_nota = *report.max_abs.get("nota").unwrap_or(&0.0);
    assert!(max_nota <= 1e-3, "P1: max|delta nota| = {max_nota:e}, SPEC §11.2 requires <= 1e-3");
    assert_eq!(
        report.actions_equal, report.actions_total,
        "P1: esc must be identical for every being on tick 0 (SPEC §11.2: 16/16)"
    );
}

/// P2 (SPEC §11.2): the first simulated day (240 ticks), hard tolerances —
/// 100% of actions equal, `pos` within `1e-4` m, every other body variable
/// within `1e-3` (relative to its own SPEC-documented range).
#[test]
#[ignore = "needs the real traco_ref.json (T0); point TAXI_TRACO_REF at it"]
fn every_action_of_the_first_simulated_day_matches_python() {
    let trace = Trace::load(Path::new(&real_trace_path())).expect("real trace must parse");
    let report = run_gate(&trace, GateLevel::FirstDay);
    println!("P2 (first day):\n{}", report.summary());

    assert_eq!(
        report.actions_equal, report.actions_total,
        "P2: SPEC §11.2 requires 100% action agreement over the first simulated day"
    );
    for name in ["pos_x", "pos_y"] {
        let v = *report.max_abs.get(name).unwrap_or(&0.0);
        assert!(v <= 1e-4, "P2: max|delta {name}| = {v:e}, SPEC §11.2 requires <= 1e-4 m");
    }
    for name in ["fuel", "agua", "fome", "calor", "fad", "res", "vida", "dinh"] {
        let v = *report.max_abs.get(name).unwrap_or(&0.0);
        assert!(v <= 1e-3, "P2: max|delta {name}| = {v:e}, SPEC §11.2 requires <= 1e-3");
    }
}

/// P3 (SPEC §11.2): the whole trace, statistical tolerances — >= 95% of action
/// pairs equal, the median first-divergence tick across beings >= 600, every
/// divergence classified `margin > 1e-2` is by definition a bug (already
/// enforced by `gate::classify_divergence`), and the final `t_vida`/`rec`
/// close for at least 14 of 16 beings.
#[test]
#[ignore = "needs the real traco_ref.json (T0); point TAXI_TRACO_REF at it"]
fn a_full_life_diverges_only_on_numerical_ties() {
    let trace = Trace::load(Path::new(&real_trace_path())).expect("real trace must parse");
    let report = run_gate(&trace, GateLevel::FullLife);
    println!("P3 (full life):\n{}", report.summary());
    if let Some(first) = report.bugs.first() {
        println!(
            "first BUG divergence: tick={} being={} esc_py={} esc_rust={} margin={:e}",
            first.tick, first.being, first.esc_py, first.esc_rust, first.margin
        );
    }

    let frac_equal = report.actions_equal as f64 / report.actions_total as f64;
    assert!(frac_equal >= 0.95, "P3: {:.4}% of actions equal, SPEC §11.2 requires >= 95%", frac_equal * 100.0);

    let n = report.first_divergence.len();
    let mut first_or_horizon: Vec<usize> =
        report.first_divergence.iter().map(|d| d.unwrap_or(trace.t)).collect();
    first_or_horizon.sort_unstable();
    let median = if n % 2 == 1 {
        first_or_horizon[n / 2] as f64
    } else {
        (first_or_horizon[n / 2 - 1] + first_or_horizon[n / 2]) as f64 / 2.0
    };
    assert!(median >= 600.0, "P3: median first BUG divergence tick = {median}, SPEC §11.2 requires >= 600");

    let mut ok_t_vida = 0usize;
    let mut ok_rec = 0usize;
    for b in 0..n {
        if report.final_t_vida_diff[b] <= 2.0 {
            ok_t_vida += 1;
        }
        let rec_ref = trace.rec[b].abs().max(1e-6);
        if report.final_rec_diff[b] / rec_ref <= 0.01 {
            ok_rec += 1;
        }
    }
    assert!(ok_t_vida >= 14, "P3: only {ok_t_vida}/16 beings had |delta t_vida| <= 2 ticks");
    assert!(ok_rec >= 14, "P3: only {ok_rec}/16 beings had |delta rec| <= 1%");
}
