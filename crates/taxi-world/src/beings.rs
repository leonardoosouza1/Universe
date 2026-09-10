//! The being state, Structure-of-Arrays (SPEC §1, plan §3.1) — one `Vec<f32>` (or
//! `Vec<u32>`) per variable, length `n`, instead of an array of a `Being` struct.
//! `physics.rs` (this same task, T2) is the first reader/writer; `senses.rs` (T3),
//! `services.rs`/`passengers.rs` (T4) and `tick.rs` (T5) read and mutate it later.
//! Kept SoA for the same reason the plan gives (plan §3.1): the v1 WGSL port reads
//! these `Vec`s as `storage` buffers directly, and an array-of-structs would need the
//! same hand-rolled 16-byte-aligned serialization `src/gpu/physics.rs:174-238`
//! already pays for the car.
//!
//! T1's `trace.rs` deliberately does not build a `Beings` (its own doc comment on
//! `Trace` explains why: that file is not allowed to create `beings.rs`). This module
//! owns the one constructor the plan promises instead: [`Beings::from_trace`], a
//! direct, mechanical read of `Trace::inicial` (SPEC §1's "no v0 the Rust does not
//! GENERATE these values, it READS them from `traco_ref.json`") plus the handful of
//! per-being fields SPEC §1 says are deterministic constants at `t=0` and therefore
//! never appear in the JSON at all (`pd_ = -1`, `dest = emb = (0,0)`, `orig_d = 0`,
//! `t_ult = 0`, `metab = 1.0`, `cons = fuel_passo·2`, `tpar = 0`).

use crate::constants::{Config, NPX};
use crate::trace::Trace;

/// One taxi world's worth of beings, Structure-of-Arrays (plan §3.1). Every `Vec`
/// below has length `n`, except `px`/`pxd` (`n * NPX * 2`, layout `[being][slot][xy]`)
/// and `px_ok` (`n * NPX`, layout `[being][slot]`) — SPEC §7's `NPX = 12` private
/// passenger slots per being.
#[derive(Debug, Clone, PartialEq)]
pub struct Beings {
    pub n: usize,

    // --- position and body (SPEC §1) ---
    pub pos_x: Vec<f32>,
    pub pos_y: Vec<f32>,
    /// Fuel in the tank, `[0, 100]` (`fuel`).
    pub fuel: Vec<f32>,
    /// Water, `[0, 100]` (`agua`).
    pub water: Vec<f32>,
    /// Hunger, `(-inf, 100]` (`fome`) — a debt, not a bounded gauge: the SPEC's own
    /// range column has no floor.
    pub hunger: Vec<f32>,
    /// Fatigue, `[0, 100]` (`fad`).
    pub fatigue: Vec<f32>,
    /// Heat, `[25, 120]` (`calor`).
    pub heat: Vec<f32>,
    /// Life, `[0, 100]` (`vida`) — `<= 0` is dead.
    pub life: Vec<f32>,
    /// Money/debt, `[-debt_limit, inf)` (`dinh`).
    pub money: Vec<f32>,
    /// Reserve tank, `[0, RES_MAX]` (`res`).
    pub reserve: Vec<f32>,
    /// Ticks left parked — pane/exhaustion/overheat/rest all write this, and it
    /// gates `livre` at the START of every tick (`tpar`, SPEC §13.8).
    pub stopped_ticks: Vec<f32>,
    /// Metabolism multiplier, `>= 1` (`metab`) — always `1.0` in v0 (`custo_gene =
    /// 0`, no non-Linear brain spends genes against it, SPEC §1).
    pub metabolism: Vec<f32>,
    /// Measured fuel consumption, litres per metre, EMA (`cons`).
    pub consumption: Vec<f32>,
    /// Home district id (`casa`).
    pub home_district: Vec<u32>,
    /// This being's map's price/day phase offset, ticks (`fase`) — SPEC §1: per MAP,
    /// not per being, but stored per being here because SoA has no separate
    /// per-map table in v0 (`World` itself carries no `fase`; `Trace::to_world`
    /// does not either).
    pub phase: Vec<f32>,

    // --- passenger (SPEC §7) ---
    /// `-1.0` = empty, `+1.0` = carrying a passenger (`pd_`) — `f32`, not `bool`:
    /// physics reads it as a signed multiplier (`peso = 1 + 0.15·(pd_ >= 0) + ...`).
    pub carrying: Vec<f32>,
    /// Current passenger's destination (`dest`), valid only while `carrying >= 0`.
    pub dest_x: Vec<f32>,
    pub dest_y: Vec<f32>,
    /// Where the current passenger boarded (`emb`), valid only while `carrying >= 0`.
    pub board_x: Vec<f32>,
    pub board_y: Vec<f32>,
    /// District the current passenger boarded in (`orig_d`) — the fare uses THIS
    /// district's tax/price, never the drop-off's (SPEC §7.4).
    pub origin_district: Vec<u32>,
    /// Ticks since the last pickup (`t_ult`), reset to `0` on pickup.
    pub since_pickup: Vec<f32>,
    /// Passenger-slot positions, `n * NPX * 2`, `[being][slot][xy]` (`px`).
    pub px: Vec<f32>,
    /// Passenger-slot destinations, same layout (`pxd`).
    pub pxd: Vec<f32>,
    /// Passenger-slot active flag, `n * NPX`, `1` = active (`px_ok`) — `u32`, not
    /// `bool`: plan §3.1, WGSL has no `bool` in a storage buffer.
    pub px_ok: Vec<u32>,

    // --- counters (SPEC §1, all start at 0) ---
    /// Ticks lived so far, `+= vivo` every tick (`t_vida`) — the fitness numerator.
    pub alive_ticks: Vec<f32>,
    /// Passengers delivered (`ent`).
    pub deliveries: Vec<f32>,
    /// Fares earned (`rec`) — the fitness's other term.
    pub revenue: Vec<f32>,
    /// Every breakdown event: pane, exhaustion, overheat (`avar`).
    pub breakdowns: Vec<f32>,
    /// Cumulative damage taken (`dano_soma`).
    pub damage_total: Vec<f32>,
    /// Fuel-station visits with `litros > 0` (`n_posto`).
    pub refuels: Vec<f32>,
    /// Total litres bought (`litros`).
    pub liters: Vec<f32>,
    /// Meals bought (`n_come`).
    pub meals: Vec<f32>,
    /// Repairs bought (`n_oficina`).
    pub repairs: Vec<f32>,
    /// Water purchases (`n_agua`).
    pub drinks: Vec<f32>,
    /// Breakdowns specifically caused by an empty tank (`av_fuel`) — a subset of
    /// `breakdowns`, which also counts exhaustion and overheating.
    pub faults: Vec<f32>,

    // --- batch layout (plan §3.1, SPEC §10: b = p·(NC·R) + c·R + r) ---
    /// Which `World` (of the batch's `w: &[World]`) this being lives on.
    pub map_idx: Vec<u32>,
    /// Which genome (of a population) drives this being.
    pub genome_idx: Vec<u32>,
}

impl Beings {
    /// Builds the starting state for every being in `trace.inicial` (SPEC §1, plan
    /// §T1's note on `Trace::initial`). All `n = trace.b` beings land on map index
    /// `0` and genome index `0` — this loads ONE map's initial state, not a
    /// multi-map/multi-genome evolution batch (SPEC §10's `b = p·(NC·R) + c·R + r`
    /// layout is T6's `evolution.rs` problem, not this constructor's).
    ///
    /// `cfg` supplies `fuel_per_tick`: the one deterministic-but-not-fixed initial
    /// value SPEC §1 lists as a formula rather than a literal — `cons = fuel_passo·2`
    /// (`campo.py:159`) — so a `Config` with a non-default `fuel_per_tick` still
    /// gets the correct starting `consumption`.
    pub fn from_trace(trace: &Trace, cfg: &Config) -> Beings {
        let n = trace.b;
        let init = &trace.inicial;

        let mut pos_x = Vec::with_capacity(n);
        let mut pos_y = Vec::with_capacity(n);
        for p in &init.pos {
            pos_x.push(p[0]);
            pos_y.push(p[1]);
        }

        let mut px = vec![0.0f32; n * NPX * 2];
        let mut pxd = vec![0.0f32; n * NPX * 2];
        let mut px_ok = vec![0u32; n * NPX];
        for (being, slots) in init.px.iter().enumerate() {
            for (slot, p) in slots.iter().enumerate() {
                px[(being * NPX + slot) * 2] = p[0];
                px[(being * NPX + slot) * 2 + 1] = p[1];
            }
        }
        for (being, slots) in init.pxd.iter().enumerate() {
            for (slot, p) in slots.iter().enumerate() {
                pxd[(being * NPX + slot) * 2] = p[0];
                pxd[(being * NPX + slot) * 2 + 1] = p[1];
            }
        }
        for (being, slots) in init.px_ok.iter().enumerate() {
            for (slot, ok) in slots.iter().enumerate() {
                px_ok[being * NPX + slot] = *ok as u32;
            }
        }

        Beings {
            n,
            pos_x,
            pos_y,
            fuel: init.fuel.clone(),
            water: init.agua.clone(),
            hunger: init.fome.clone(),
            fatigue: init.fad.clone(),
            heat: init.calor.clone(),
            life: init.vida.clone(),
            money: init.dinh.clone(),
            reserve: init.res.clone(),
            stopped_ticks: vec![0.0; n],
            metabolism: vec![1.0; n],
            consumption: vec![cfg.fuel_per_tick * 2.0; n],
            home_district: init.casa.clone(),
            phase: init.fase.clone(),
            carrying: vec![-1.0; n],
            dest_x: vec![0.0; n],
            dest_y: vec![0.0; n],
            board_x: vec![0.0; n],
            board_y: vec![0.0; n],
            origin_district: vec![0; n],
            since_pickup: vec![0.0; n],
            px,
            pxd,
            px_ok,
            alive_ticks: vec![0.0; n],
            deliveries: vec![0.0; n],
            revenue: vec![0.0; n],
            breakdowns: vec![0.0; n],
            damage_total: vec![0.0; n],
            refuels: vec![0.0; n],
            liters: vec![0.0; n],
            meals: vec![0.0; n],
            repairs: vec![0.0; n],
            drinks: vec![0.0; n],
            faults: vec![0.0; n],
            map_idx: vec![0; n],
            genome_idx: vec![0; n],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::Config;
    use std::path::PathBuf;

    fn fixture_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny_trace.json")
    }

    fn load_fixture() -> Trace {
        Trace::load(&fixture_path()).expect("tests/fixtures/tiny_trace.json must parse")
    }

    #[test]
    fn from_trace_copies_every_documented_initial_field() {
        let trace = load_fixture();
        let cfg = Config::default();
        let beings = Beings::from_trace(&trace, &cfg);

        assert_eq!(beings.n, trace.b);
        for i in 0..trace.b {
            assert_eq!(beings.pos_x[i], trace.inicial.pos[i][0]);
            assert_eq!(beings.pos_y[i], trace.inicial.pos[i][1]);
            assert_eq!(beings.fuel[i], trace.inicial.fuel[i]);
            assert_eq!(beings.water[i], trace.inicial.agua[i]);
            assert_eq!(beings.hunger[i], trace.inicial.fome[i]);
            assert_eq!(beings.fatigue[i], trace.inicial.fad[i]);
            assert_eq!(beings.heat[i], trace.inicial.calor[i]);
            assert_eq!(beings.life[i], trace.inicial.vida[i]);
            assert_eq!(beings.money[i], trace.inicial.dinh[i]);
            assert_eq!(beings.reserve[i], trace.inicial.res[i]);
            assert_eq!(beings.home_district[i], trace.inicial.casa[i]);
            assert_eq!(beings.phase[i], trace.inicial.fase[i]);
        }
    }

    #[test]
    fn from_trace_sets_the_deterministic_defaults_the_json_never_carries() {
        // SPEC §1: pd_, dest, emb, orig_d, t_ult, metab, cons, tpar are all fixed
        // formulas of Campo.__init__, never read from disk.
        let trace = load_fixture();
        let cfg = Config::default();
        let beings = Beings::from_trace(&trace, &cfg);

        for i in 0..trace.b {
            assert_eq!(beings.carrying[i], -1.0, "pd_ starts at -1 (campo.py:164)");
            assert_eq!(beings.dest_x[i], 0.0);
            assert_eq!(beings.dest_y[i], 0.0);
            assert_eq!(beings.board_x[i], 0.0);
            assert_eq!(beings.board_y[i], 0.0);
            assert_eq!(beings.origin_district[i], 0);
            assert_eq!(beings.since_pickup[i], 0.0);
            assert_eq!(beings.metabolism[i], 1.0);
            assert_eq!(
                beings.consumption[i],
                0.30,
                "cons = fuel_passo*2 = 0.15*2 (campo.py:159)"
            );
            assert_eq!(beings.stopped_ticks[i], 0.0);
            assert_eq!(beings.map_idx[i], 0);
            assert_eq!(beings.genome_idx[i], 0);
        }
    }

    #[test]
    fn from_trace_flattens_passenger_slots_in_being_slot_xy_order() {
        let trace = load_fixture();
        let cfg = Config::default();
        let beings = Beings::from_trace(&trace, &cfg);

        for (being, slots) in trace.inicial.px.iter().enumerate() {
            for (slot, p) in slots.iter().enumerate() {
                let base = (being * NPX + slot) * 2;
                assert_eq!(beings.px[base], p[0]);
                assert_eq!(beings.px[base + 1], p[1]);
            }
        }
        for (being, slots) in trace.inicial.px_ok.iter().enumerate() {
            for (slot, ok) in slots.iter().enumerate() {
                let expected = if *ok { 1 } else { 0 };
                assert_eq!(beings.px_ok[being * NPX + slot], expected);
            }
        }
    }

    #[test]
    fn from_trace_uses_the_given_config_fuel_per_tick_for_the_initial_consumption() {
        let trace = load_fixture();
        let mut cfg = Config::default();
        cfg.fuel_per_tick = 0.5;
        let beings = Beings::from_trace(&trace, &cfg);
        for i in 0..trace.b {
            assert_eq!(beings.consumption[i], 1.0, "cons = fuel_passo*2 = 0.5*2");
        }
    }
}
