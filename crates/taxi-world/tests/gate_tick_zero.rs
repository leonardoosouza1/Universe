//! T3's identity-gate rehearsal (SPEC §11, plan §T3 "Testes de aceite" brief):
//! `senses::observe`'s `F` at tick 0, compared column-by-column against the real
//! `traco_ref.json`'s `F_primeiros_50[0]` — "the gate worth gold" (this task's
//! brief). An integration test (not `senses.rs`'s own `#[cfg(test)] mod tests`)
//! purely to keep `senses.rs` under `docs/STYLE.md`'s 600-line cap; it exercises
//! only `senses.rs`'s PUBLIC surface (`taxi_world::senses::{observe, SensesInput,
//! NearestCache}`), the same surface T5 will wire against T2's `Beings`.
//!
//! Ignored by default: needs the real trace (T0's Python generator). Run with:
//! `TAXI_TRACO_REF=/path/to/traco_ref.json nice -n 10 env CARGO_BUILD_JOBS=4
//! cargo test -p taxi-world -- --ignored`.

use taxi_world::senses::{NearestCache, SensesInput, observe};
use taxi_world::{Config, G, NF, NPX, Trace};

/// Mirrors `senses::compute_target` (SPEC §4 `alvo()`, `campo.py:256-259`) —
/// duplicated here, not imported, because that function is deliberately private
/// to `senses.rs` (it is an internal fallback for `observe`'s `target: None`
/// path, not part of the module's public interface). Used only for a sharper
/// failure message: if `alvo` itself is wrong, every `d_alvo` column fails, and
/// this pinpoints that before the big per-feature loop below does.
fn expected_target(px: &[f32], pos_x: f32, pos_y: f32, carrying: f32, dest: [f32; 2]) -> [f32; 2] {
    let mut best_d = f32::INFINITY;
    let mut best = [0.0f32; 2];
    for slot in 0..NPX {
        let x = px[slot * 2];
        let y = px[slot * 2 + 1];
        let d = ((x - pos_x).powi(2) + (y - pos_y).powi(2)).sqrt();
        if d < best_d {
            best_d = d;
            best = [x, y];
        }
    }
    if carrying >= 0.0 { dest } else { best }
}

#[test]
#[ignore = "needs the real traco_ref.json (T0); point TAXI_TRACO_REF at it"]
fn the_first_tick_f_matches_the_python_reference_trace() {
    let path = std::env::var("TAXI_TRACO_REF")
        .expect("set TAXI_TRACO_REF to the path of a real traco_ref.json");
    let trace = Trace::load(std::path::Path::new(&path)).expect("real trace must parse");
    let world = trace.to_world();
    let n = trace.b;

    // t=0 initial state, read from `inicial` (SPEC §1: v0 reads, does not
    // generate). Fields NOT carried by the trace (`pd_`, `dest`, `t_ult`,
    // `cons`) use their documented `Campo.__init__` values (SPEC §1 table,
    // `campo.py:158-166`) — a genuine gap in `traco_ref.json`, declared in this
    // task's final report, not silently patched over.
    let pos_x: Vec<f32> = trace.inicial.pos.iter().map(|p| p[0]).collect();
    let pos_y: Vec<f32> = trace.inicial.pos.iter().map(|p| p[1]).collect();
    let carrying = vec![-1.0f32; n]; // campo.py:164 `s.pd_ = full(-1.)`
    let dest_x = vec![0.0f32; n]; // campo.py:164 `s.dest = zeros`
    let dest_y = vec![0.0f32; n];
    let since_pickup = vec![0.0f32; n]; // campo.py:166 `s.t_ult = zeros`
    let consumption = vec![0.15f32 * 2.0; n]; // campo.py:159 `fuel_passo*2`
    let map_idx = vec![0u32; n]; // single map (SPEC §11.2: "mapa carregado")
    let px: Vec<f32> = trace
        .inicial
        .px
        .iter()
        .flat_map(|slots| slots.iter().flat_map(|p| [p[0], p[1]]))
        .collect();

    let input = SensesInput {
        n,
        pos_x: &pos_x,
        pos_y: &pos_y,
        fuel: &trace.inicial.fuel,
        water: &trace.inicial.agua,
        hunger: &trace.inicial.fome,
        fatigue: &trace.inicial.fad,
        heat: &trace.inicial.calor,
        life: &trace.inicial.vida, // check-language: allow (field names of traco_ref.json)
        money: &trace.inicial.dinh,
        reserve: &trace.inicial.res,
        consumption: &consumption,
        carrying: &carrying,
        dest_x: &dest_x,
        dest_y: &dest_y,
        px: &px,
        home_district: &trace.inicial.casa,
        phase: &trace.inicial.fase,
        since_pickup: &since_pickup,
        map_idx: &map_idx,
    };

    // alvo() sanity check first — a sharper failure signal than a wall of F
    // mismatches if this is what actually broke.
    let alvo_ref = trace
        .alvo_por_tique
        .as_array()
        .expect("alvo_por_tique array")[0]
        .as_array()
        .expect("tick 0 row");
    for b in 0..n {
        let expected = alvo_ref[b].as_array().expect("per-being alvo pair");
        let ex = expected[0].as_f64().unwrap() as f32;
        let ey = expected[1].as_f64().unwrap() as f32;
        let got = expected_target(
            &px[b * NPX * 2..(b + 1) * NPX * 2],
            pos_x[b],
            pos_y[b],
            carrying[b],
            [dest_x[b], dest_y[b]],
        );
        assert!(
            (ex - got[0]).abs() < 1e-4 && (ey - got[1]).abs() < 1e-4,
            "alvo mismatch being {b}: expected [{ex}, {ey}], got {got:?}"
        );
    }

    let cfg = Config::default();
    let worlds = [world];
    // price_mul at tick 0 == the map's own base pmul: `update_daily_prices`
    // (T2) only runs inside `passo()`, AFTER the tick-0 `obs()` this test
    // reproduces (SPEC §3 `um_passo`: obs() before passo(); `campo.py:149`'s
    // `s.pmul = s.pmul0.clone()` is what the very first `obs()` still sees).
    let price_mul = worlds[0].price_mul_base.clone();

    let mut out_f = vec![0.0f32; n * G * NF];
    let mut nearest = NearestCache::default();
    observe(
        &input,
        &worlds,
        &price_mul,
        &cfg,
        0,
        None,
        &mut out_f,
        &mut nearest,
    );

    let f_ref = trace
        .f_primeiros_50
        .as_array()
        .expect("F_primeiros_50 array")[0]
        .as_array()
        .expect("tick 0 row");

    let mut max_abs_diff = 0.0f32;
    for b in 0..n {
        let row = f_ref[b].as_array().expect("per-being row");
        for g in 0..G {
            let opt = row[g].as_array().expect("per-option row");
            for feat in 0..NF {
                let expected = opt[feat].as_f64().unwrap() as f32;
                let got = out_f[(b * G + g) * NF + feat];
                let diff = (expected - got).abs();
                max_abs_diff = max_abs_diff.max(diff);
                assert!(
                    diff < 1e-4,
                    "being {b} option {g} feature {feat} ({}): expected {expected}, got {got}, diff {diff}",
                    trace.nomes_f.get(feat).map(String::as_str).unwrap_or("?")
                );
            }
        }
    }
    println!("max |diff| across all beings/options/features at tick 0: {max_abs_diff:e}");
}
