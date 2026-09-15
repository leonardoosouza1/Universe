//! `observe()` — the sense vector `F` (SPEC §4, `_obs_lento`, `campo.py:301-340`).
//!
//! SPEC = `/home/leonardo/projects/LLM/neuromod-lab/docs/71-spec-mundo-taxi.md`.
//! Plan = `docs/plano-mundo-taxi.md`, task T3. `F` has `G=9` rows (one per
//! movement option) of `NF=29` columns each: the first 10 (`NOMES_OP`, SPEC §4.1)
//! vary by option; the next 19 (`NOMES_CORPO`, SPEC §4.2) are identical across all
//! 9 options (SPEC §4.3, finding E58: since `decide()` in `decision.rs` subtracts
//! the mean of the 9 options right after, NO body weight can ever change which
//! option is chosen — only the service bias/weights can).
//!
//! This module does NOT depend on T2's `Beings` (`beings.rs`, written in parallel
//! by another implementer) — `SensesInput` below is this task's own minimal,
//! borrowed read interface, built to the plan's brief ("defina uma struct de
//! entrada própria... documentada, para que T5 ligue os dois sem redesenho").
//!
//! **Not touched by this module, on purpose:** `physics.rs`, `passengers.rs`
//! (T2/T4's files).

use crate::constants::{Config, DIRS, G, N_TYPES, NF, NPX};
use crate::world::World;

/// Width split of `F`'s `NF=29` columns (SPEC §4): 10 per-option columns
/// (`NOMES_OP`) + 19 identical-across-options body columns (`NOMES_CORPO`, with
/// `NM_RISCO=1, NM_PONTO=1` — SPEC §0).
///
/// **Divergence from the plan, declared.** The plan's own T3 interface block
/// (`docs/plano-mundo-taxi.md` §T3) writes `pub const NF: usize = N_OP +
/// N_BODY;`. `crates/taxi-world/src/constants.rs` (T1, merged first) already
/// declares `pub const NF: usize = 29` as, in its own doc comment, "the crate's
/// single source of truth" — and explicitly asks whoever writes `senses.rs` to
/// reuse `taxi_world::NF` rather than keep a second definition (two `pub const
/// NF` in the same crate, both re-exported from `lib.rs`, would not compile).
/// This module keeps `N_OP`/`N_BODY` exactly as the plan asks and ties them to
/// `crate::constants::NF` with a compile-time array-length check instead of
/// redeclaring `NF`.
pub const N_OP: usize = 10;
pub const N_BODY: usize = 19;
// Compile-time proof that N_OP + N_BODY == crate::constants::NF: a mismatch here
// is an array-length type error at build time, not a runtime surprise.
const _PROOF_N_OP_PLUS_N_BODY_EQUALS_NF: [(); N_OP + N_BODY] = [(); NF];

/// A finite stand-in for "this service type has zero points anywhere on the
/// map" (`world.rs`'s `nearest_services` returns `f32::INFINITY` for that case —
/// see that function's own doc comment). SPEC's Python side never hits a
/// genuinely-per-map-empty type this way (its masking is a batch-padding
/// artefact, not a real "no such service on this map" state, SPEC §2.1 vs
/// `campo.py:123-127`), but a hand-built Rust test fixture can. `1e9` mirrors the
/// SPEC's OWN masking constant (`campo.py:123-127`) rather than inventing a new
/// number, and staying finite keeps a genome weight of exactly `0.0` well-defined
/// (`0.0 * 1e9 = 0.0`, whereas `0.0 * f32::INFINITY = NaN` would poison the whole
/// dot product in `decision.rs`).
const ABSENT_TYPE_DISTANCE_SENTINEL: f32 = 1e9;

/// SPEC §4.1's option-column loop order (`campo.py:322`, `s.PERM_F =
/// [0,1,2,4,3]`): columns 1..5 (`d_posto, d_comida, d_agua, d_oficina,
/// d_descanso`) read `TIPOS` (posto, comida, agua, descanso, oficina — SPEC §0)
/// in THIS order, not `TIPOS`' own order — oficina (`TIPOS[4]`) is written
/// before descanso (`TIPOS[3]`). Getting this backwards is the exact bug the
/// plan's `option_features_follow_the_zero_one_two_four_three_order` acceptance
/// test exists to catch.
const OPTION_DISTANCE_TYPE_ORDER: [usize; 5] = [0, 1, 2, 4, 3];

/// Per-being, per-service-type nearest point and its distance from the being's
/// CURRENT (unprojected) position (SPEC §2.3 `_perto_exata`). Computed once per
/// tick by `observe()` and reused, UNCHANGED, by T2's `physics.rs` (a breakdown
/// tow targets `nearest.points` for the fuel type) and T4's `services.rs`
/// (service-radius/need reads `nearest.dist`) for the SAME tick — plan §T2/§T4:
/// "`_ptos`/`_dist` do `obs()` DESTE tique". Flat SoA, `TIPOS` order (posto,
/// comida, agua, descanso, oficina — SPEC §0), length `n * N_TYPES`.
#[derive(Debug, Clone, Default)]
pub struct NearestCache {
    pub points: Vec<[f32; 2]>,
    pub dist: Vec<f32>,
}

impl NearestCache {
    fn resize(&mut self, n: usize) {
        self.points.resize(n * N_TYPES, [0.0, 0.0]);
        self.dist.resize(n * N_TYPES, 0.0);
    }
}

/// Minimal, borrowed read interface onto a being population's state —
/// independent of T2's `beings.rs` `Beings` type by this task's hard rule (T2
/// writes that file in parallel; this crate's brief forbids creating it here).
/// Every field is a flat SoA slice of length `n`, except `px` (`n * NPX * 2`,
/// `[being][slot][xy]`, SPEC §1 `px`).
///
/// **T5 wiring note.** Every field name below matches the plan's own `Beings`
/// field names (plan §3.1 "Estado — SoA por ser") exactly, so that once T2's
/// `Beings` exists, building a `SensesInput` is a mechanical borrow —
/// `pos_x: &beings.pos_x`, `carrying: &beings.carrying`, `since_pickup:
/// &beings.since_pickup`, etc. — never a rename or a redesign.
pub struct SensesInput<'a> {
    pub n: usize,
    pub pos_x: &'a [f32],
    pub pos_y: &'a [f32],
    pub fuel: &'a [f32],
    pub water: &'a [f32],
    pub hunger: &'a [f32],
    pub fatigue: &'a [f32],
    pub heat: &'a [f32],
    pub life: &'a [f32],
    pub money: &'a [f32],
    pub reserve: &'a [f32],
    /// Fuel burned per metre, EMA (SPEC §1 `cons`; SPEC §4.2 `e_consumo`,
    /// `e_autonomia`).
    pub consumption: &'a [f32],
    /// `-1.0` = no passenger, `+1.0` = carrying one (SPEC §1 `pd_`).
    pub carrying: &'a [f32],
    pub dest_x: &'a [f32],
    pub dest_y: &'a [f32],
    /// `n * NPX * 2`, `[being][slot][xy]` (SPEC §1 `px`). Read by `alvo()`
    /// (`compute_target`, below) when `observe`'s `target` argument is `None`.
    pub px: &'a [f32],
    pub home_district: &'a [u32],
    /// Per-map day phase, already broadcast to every being on that map
    /// (SPEC §1 `fase`).
    pub phase: &'a [f32],
    /// Ticks since the last pickup (SPEC §1 `t_ult`), read by `e_espera`.
    pub since_pickup: &'a [f32],
    /// Which `World` (index into the `w: &[World]` slice `observe` receives)
    /// this being is on (SPEC §3.1 "Índice do lote", `map_idx[b] = c`).
    pub map_idx: &'a [u32],
}

/// Per-map running offset into a flat `price_mul` slice (SPEC §3.1 "Índice do
/// lote" applied to T2's per-map `physics::update_daily_prices`, which takes ONE
/// `&World` at a time): map `m`'s `K` district prices occupy `price_mul[offset(m)
/// .. offset(m) + w[m].tax.len()]`. Exposed so T5's driver lays its
/// `price_mul: Vec<f32>` out the same way it calls `update_daily_prices` once per
/// map.
///
/// **Interface note, not in the plan's own T3 block.** The plan's T3 signature
/// (`docs/plano-mundo-taxi.md` §T3) just says `price_mul: &[f32]` without stating
/// its layout across multiple maps; T2's `update_daily_prices(w_price_mul: &mut
/// [f32], w: &World, ...)` takes a single `World`, which only makes sense if the
/// caller slices a larger buffer per map. This function is that slicing
/// convention, made explicit and reusable rather than left for T5 to invent.
pub fn map_price_offsets(w: &[World]) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(w.len());
    let mut acc = 0usize;
    for world in w {
        offsets.push(acc);
        acc += world.tax.len();
    }
    offsets
}

/// SPEC §4 `_obs_lento` (`campo.py:301-340`): fills `out_f` (`F`, `n * G * NF`,
/// `[being][option][feature]`) and `nearest` (SPEC §2.3 `_perto_exata`, reused
/// this same tick by T2/T4).
///
/// `t` is the tick counter BEFORE the increment `passo()` performs as its own
/// first line (SPEC §3 `um_passo`: `obs()` "usa t"; T2's `step_physics` instead
/// receives `t` AFTER the increment — same tick, two different counter values,
/// by design, SPEC §3 says so explicitly).
///
/// `target`, when `Some`, is `alvo_por_tique[t]` straight from the reference
/// trace — the identity-gate hook (SPEC §7.2, plan §T3 "Interface"), flat length
/// `n * 2`, `[being][xy]`. `alvo()` reads a passenger slot ignoring `px_ok` (SPEC
/// §4 "Quirk que deve ser reproduzido" — see `compute_target` below), exactly the
/// kind of RNG-adjacent state where a float mismatch could otherwise diverge the
/// whole trajectory. `None` computes `alvo()` from `input.px`/`input.carrying`/
/// `input.dest_*` instead — the path a real battery (T6) uses.
///
/// `out_f` must already be sized `n * G * NF`.
#[allow(clippy::too_many_arguments)] // the arity is the plan's own T3 interface (docs/plano-mundo-taxi.md §T3)
pub fn observe(
    input: &SensesInput,
    w: &[World],
    price_mul: &[f32],
    cfg: &Config,
    t: i64,
    target: Option<&[f32]>,
    out_f: &mut [f32],
    nearest: &mut NearestCache,
) {
    let n = input.n;
    assert_eq!(out_f.len(), n * G * NF, "out_f must be sized n*G*NF");
    nearest.resize(n);

    let price_offsets = map_price_offsets(w);
    // esc_ = 1/vel (SPEC §4, campo.py:302): `vel` here is the CONSTANT
    // `cfg.day_speed`, not the night-adjusted `v` computed per being below —
    // distances are normalized so terrain never "steals the wheel" (SPEC §4.1
    // comment `[G3]`), independent of what time of day it currently is.
    let scale = 1.0 / cfg.day_speed;

    for b in 0..n {
        let world = &w[input.map_idx[b] as usize];
        let px_b = input.pos_x[b];
        let py_b = input.pos_y[b];

        // Nearest service of each type, from the CURRENT (unprojected)
        // position — computed once per being, reused below AND by T2/T4 later
        // this same tick.
        let (points, dists) = world.nearest_services(px_b, py_b);
        for tpe in 0..N_TYPES {
            nearest.points[b * N_TYPES + tpe] = points[tpe];
            nearest.dist[b * N_TYPES + tpe] = dists[tpe];
        }

        let hora = hour_of(t, input.phase[b], cfg);
        let is_night = hora >= 22.0 || hora < 6.0;
        let v = if is_night {
            cfg.night_speed
        } else {
            cfg.day_speed
        };

        let alvo = match target {
            Some(slice) => [slice[b * 2], slice[b * 2 + 1]],
            None => compute_target(input, b),
        };

        for g in 0..G {
            let pnx = (px_b + v * DIRS[g][0]).clamp(0.0, world.side);
            let pny = (py_b + v * DIRS[g][1]).clamp(0.0, world.side);
            let base = (b * G + g) * NF;

            // col 0: d_alvo
            out_f[base] = distance(pnx, pny, alvo[0], alvo[1]) * scale;

            // cols 1..5: d_posto, d_comida, d_agua, d_oficina, d_descanso
            // (SPEC §4.1 — TIPOS order via OPTION_DISTANCE_TYPE_ORDER, not
            // NOMES_OP's own column order).
            for (k, &tpe) in OPTION_DISTANCE_TYPE_ORDER.iter().enumerate() {
                let d = nearest.dist[b * N_TYPES + tpe];
                let raw = if d.is_finite() {
                    let p = nearest.points[b * N_TYPES + tpe];
                    distance(pnx, pny, p[0], p[1])
                } else {
                    ABSENT_TYPE_DISTANCE_SENTINEL
                };
                out_f[base + 1 + k] = raw.min(cfg.fog_distance) * scale;
            }

            // col 6: custo_frente
            out_f[base + 6] = world.terrain_cost_at(pnx, pny);
            // cols 7/8/9: preco_frente, aliq_frente, casa_frente
            let district = world.district_at(pnx, pny);
            let pm = price_mul[price_offsets[input.map_idx[b] as usize] + district as usize];
            out_f[base + 7] = pm;
            out_f[base + 8] = world.tax[district as usize];
            out_f[base + 9] = if district == input.home_district[b] {
                1.0
            } else {
                0.0
            };
        }

        // Body columns (10..28), identical across all 9 options (SPEC §4.2/4.3,
        // finding E58).
        let h_rad = hora * std::f32::consts::TAU / 24.0;
        // `_dist_v[:,0]` (SPEC §4.2): the FOG-PERCEIVED distance to the nearest
        // fuel station, from the being's ACTUAL (unprojected) position — a
        // DIFFERENT fog treatment than the OP columns' plain `.min(nevoa)`
        // above (SPEC §4 `_obs_lento`: `_dist_v` uses the `k_cego` blind-factor
        // formula, only the OP-column distances use a bare clamp).
        let dist_v0 = perceived_distance(nearest.dist[b * N_TYPES], cfg);
        let cons = input.consumption[b];
        let auton = (input.fuel[b] / cons.max(1e-3) - dist_v0) / 100.0;
        let risco = (dist_v0 + cfg.risk_margin - input.fuel[b] / cons.max(1e-3)) / 100.0;
        let home_district_here = world.district_at(px_b, py_b);

        let body: [f32; N_BODY] = [
            input.fuel[b] / 100.0,
            input.water[b] / 100.0,
            input.hunger[b] / 100.0,
            input.fatigue[b] / 100.0,
            input.heat[b] / 120.0,
            input.life[b] / 100.0,
            (input.money[b] / 150.0).clamp(-1.0, 1.0),
            input.reserve[b] / cfg.reserve_max,
            if input.carrying[b] >= 0.0 { 1.0 } else { 0.0 },
            h_rad.sin(),
            h_rad.cos(),
            if is_night { 1.0 } else { 0.0 },
            world.terrain_cost_at(px_b, py_b),
            1.0,
            cons / (cfg.fuel_per_tick * 3.0),
            auton.clamp(-2.0, 4.0),
            risco.clamp(0.0, 3.0),
            world.demand[home_district_here as usize] / 2.0,
            (input.since_pickup[b] / 100.0).min(2.0),
        ];

        for g in 0..G {
            let base = (b * G + g) * NF + N_OP;
            out_f[base..base + N_BODY].copy_from_slice(&body);
        }
    }
}

fn distance(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let dx = ax - bx;
    let dy = ay - by;
    (dx * dx + dy * dy).sqrt()
}

/// `hora() = ((t + fase) mod dia) / dia * 24` (SPEC §3.2, `campo.py:243`).
fn hour_of(t: i64, phase: f32, cfg: &Config) -> f32 {
    let day = cfg.day_ticks;
    ((t as f32 + phase) % day) / day * 24.0
}

/// `_dist_v` (SPEC §4 `_obs_lento`, `campo.py:308`): `dist` if within
/// `fog_distance`, else `blind_factor * fog_distance` (a PESSIMISTIC "I can't
/// see it, so my body assumes it's far" value — SPEC's own comment on `_dist_v`:
/// fog is a perception limit, not a teleport). Inert in v0 (`fog_distance` is
/// `+inf` by default, so the `else` branch never triggers), kept general because
/// `Config::fog_distance` is itself a real, non-hardcoded field.
fn perceived_distance(dist: f32, cfg: &Config) -> f32 {
    if dist <= cfg.fog_distance {
        dist
    } else {
        cfg.blind_factor * cfg.fog_distance
    }
}

/// SPEC §4 `alvo()` (`campo.py:256-259`): the destination if carrying a
/// passenger, else the nearest passenger slot — **ignoring `px_ok`** (SPEC §4
/// "Quirk que deve ser reproduzido": a being that has used every slot still
/// "sees" a phantom passenger in `d_alvo`; SPEC is explicit this must NOT be
/// fixed in v0, it would break the identity gate). Ties broken by first
/// occurrence, matching `torch.argmin`.
///
/// `pub(crate)` (T5 reconciliation, task brief's "what to do" list): `tick.rs`'s
/// `StepRecord.alvo` needs to report exactly the alvo `observe()` used this
/// tick, and `observe()` itself does not return it — re-deriving it with a
/// second, hand-copied implementation in `tick.rs` would be the same
/// single-source-of-truth risk item (d) of the plan's reconciliation list flags
/// for `apply_refuel`, so `tick.rs` calls this function directly instead.
pub(crate) fn compute_target(input: &SensesInput, b: usize) -> [f32; 2] {
    let mut best_d = f32::INFINITY;
    let mut best = [0.0f32; 2];
    for slot in 0..NPX {
        let idx = (b * NPX + slot) * 2;
        let px = input.px[idx];
        let py = input.px[idx + 1];
        let d = distance(input.pos_x[b], input.pos_y[b], px, py);
        if d < best_d {
            best_d = d;
            best = [px, py];
        }
    }
    if input.carrying[b] >= 0.0 {
        [input.dest_x[b], input.dest_y[b]]
    } else {
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{GRID, TEX};

    fn base_world() -> World {
        World {
            side: 20.0,
            cost_tex: vec![4.0; TEX * TEX],
            district_field: vec![0; GRID * GRID],
            centers: vec![[10.0, 10.0]],
            tax: vec![0.23],
            price_mul_base: vec![1.0], // unused: price_mul is a separate observe() argument
            rent: vec![0.0],
            demand: vec![0.86],
            price_phase: vec![0.0],
            services: [vec![], vec![], vec![], vec![], vec![]],
        }
    }

    /// A single hand-computable being: pos = world center, mid-range body
    /// values, `hora() == 6.0` exactly (clean `sin`/`cos`, and just past the
    /// night/day boundary).
    struct Fixture {
        pos_x: [f32; 1],
        pos_y: [f32; 1],
        fuel: [f32; 1],
        water: [f32; 1],
        hunger: [f32; 1],
        fatigue: [f32; 1],
        heat: [f32; 1],
        life: [f32; 1],
        money: [f32; 1],
        reserve: [f32; 1],
        consumption: [f32; 1],
        carrying: [f32; 1],
        dest_x: [f32; 1],
        dest_y: [f32; 1],
        px: [f32; NPX * 2],
        home_district: [u32; 1],
        phase: [f32; 1],
        since_pickup: [f32; 1],
        map_idx: [u32; 1],
    }

    impl Fixture {
        fn new() -> Self {
            Fixture {
                pos_x: [10.0],
                pos_y: [10.0],
                fuel: [50.0],
                water: [40.0],
                hunger: [20.0],
                fatigue: [15.0],
                heat: [60.0],
                life: [80.0],
                money: [75.0],
                reserve: [30.0],
                consumption: [0.3],
                carrying: [-1.0],
                dest_x: [0.0],
                dest_y: [0.0],
                px: [0.0; NPX * 2],
                home_district: [0],
                phase: [60.0], // hora(t=0) = (0+60 mod 240)/240*24 = 6.0
                since_pickup: [150.0],
                map_idx: [0],
            }
        }

        fn input(&self) -> SensesInput<'_> {
            SensesInput {
                n: 1,
                pos_x: &self.pos_x,
                pos_y: &self.pos_y,
                fuel: &self.fuel,
                water: &self.water,
                hunger: &self.hunger,
                fatigue: &self.fatigue,
                heat: &self.heat,
                life: &self.life,
                money: &self.money,
                reserve: &self.reserve,
                consumption: &self.consumption,
                carrying: &self.carrying,
                dest_x: &self.dest_x,
                dest_y: &self.dest_y,
                px: &self.px,
                home_district: &self.home_district,
                phase: &self.phase,
                since_pickup: &self.since_pickup,
                map_idx: &self.map_idx,
            }
        }
    }

    #[test]
    fn each_of_the_twenty_nine_columns_matches_its_formula() {
        let mut world = base_world();
        world.services[0] = vec![[10.0, 13.0]]; // posto,    dist 3 from (10,10)
        world.services[1] = vec![[14.0, 10.0]]; // comida,   dist 4
        world.services[2] = vec![[10.0, 5.0]]; // agua,     dist 5
        world.services[3] = vec![[10.0, 17.0]]; // descanso, dist 7
        world.services[4] = vec![[4.0, 10.0]]; // oficina,  dist 6

        let fx = Fixture::new();
        let worlds = [world];
        let price_mul = [0.77f32];
        let target = [10.0f32, 20.0f32]; // alvo, dist 10 from (10,10)

        let mut out_f = vec![0.0f32; G * NF];
        let mut nearest = NearestCache::default();
        observe(
            &fx.input(),
            &worlds,
            &price_mul,
            &Config::default(),
            0,
            Some(&target),
            &mut out_f,
            &mut nearest,
        );

        // g=8 ("stay"): pn[8] == pos exactly, so every option column is
        // hand-computable without projecting a position first.
        let base = 8 * NF;
        let expected: [(usize, &str, f32); 29] = [
            (0, "d_alvo", 10.0),
            (1, "d_posto", 3.0),
            (2, "d_comida", 4.0),
            (3, "d_agua", 5.0),
            (4, "d_oficina", 6.0),
            (5, "d_descanso", 7.0),
            (6, "custo_frente", 4.0),
            (7, "preco_frente", 0.77),
            (8, "aliq_frente", 0.23),
            (9, "casa_frente", 1.0),
            (10, "e_fuel", 0.5),
            (11, "e_agua", 0.4),
            (12, "e_fome", 0.2),
            (13, "e_fadiga", 0.15),
            (14, "e_calor", 0.5),
            (15, "e_vida", 0.8),
            (16, "e_dinheiro", 0.5),
            (17, "e_reserva", 0.25),
            (18, "e_tem_pax", 0.0),
            (19, "h_sin", 1.0),
            (20, "h_cos", 0.0),
            (21, "e_noite", 0.0),
            (22, "custo_aqui", 4.0),
            (23, "bias", 1.0),
            (24, "e_consumo", 0.3 / 0.45),
            (25, "e_autonomia", (50.0f32 / 0.3 - 3.0) / 100.0),
            (26, "e_risco", 0.0), // (3+30-166.667)/100 clamps to [0,3]
            (27, "e_demanda", 0.43),
            (28, "e_espera", 1.5),
        ];
        for (col, name, exp) in expected {
            let got = out_f[base + col];
            assert!(
                (got - exp).abs() < 1e-3,
                "column {col} ({name}): expected {exp}, got {got}"
            );
        }
    }

    #[test]
    fn option_features_follow_the_zero_one_two_four_three_order() {
        let mut world = base_world();
        world.services[0] = vec![[10.0, 13.0]]; // posto,    dist 3
        world.services[1] = vec![[14.0, 10.0]]; // comida,   dist 4
        world.services[2] = vec![[10.0, 5.0]]; // agua,     dist 5
        world.services[3] = vec![[10.0, 17.0]]; // descanso, dist 7
        world.services[4] = vec![[4.0, 10.0]]; // oficina,  dist 6

        let fx = Fixture::new();
        let worlds = [world];
        let price_mul = [1.0f32];
        let target = [10.0f32, 10.0f32]; // irrelevant here: only cols 1..5 checked

        let mut out_f = vec![0.0f32; G * NF];
        let mut nearest = NearestCache::default();
        observe(
            &fx.input(),
            &worlds,
            &price_mul,
            &Config::default(),
            0,
            Some(&target),
            &mut out_f,
            &mut nearest,
        );

        let base = 8 * NF; // "stay"
        assert!((out_f[base + 1] - 3.0).abs() < 1e-4, "d_posto");
        assert!((out_f[base + 2] - 4.0).abs() < 1e-4, "d_comida");
        assert!((out_f[base + 3] - 5.0).abs() < 1e-4, "d_agua");
        assert!(
            (out_f[base + 4] - 6.0).abs() < 1e-4,
            "d_oficina must read TIPOS[4] (dist 6), not TIPOS[3] (SPEC §4.1 permutation)"
        );
        assert!(
            (out_f[base + 5] - 7.0).abs() < 1e-4,
            "d_descanso must read TIPOS[3] (dist 7), not TIPOS[4] (SPEC §4.1 permutation)"
        );
    }

    #[test]
    fn body_features_are_identical_across_the_nine_options() {
        let world = base_world();
        let fx = Fixture::new();
        let worlds = [world];
        let price_mul = [1.0f32];
        let target = [12.0f32, 9.0f32];

        let mut out_f = vec![0.0f32; G * NF];
        let mut nearest = NearestCache::default();
        observe(
            &fx.input(),
            &worlds,
            &price_mul,
            &Config::default(),
            0,
            Some(&target),
            &mut out_f,
            &mut nearest,
        );

        for g in 1..G {
            for col in N_OP..NF {
                assert_eq!(
                    out_f[g * NF + col],
                    out_f[col],
                    "body column {col} must be identical between option 0 and option {g}"
                );
            }
        }
    }

    // The identity gate against the real `traco_ref.json` (SPEC §11, "o portão
    // que vale ouro") lives in `tests/gate_tick_zero.rs`, NOT here — it is
    // `#[ignore]`d and only exercises `observe`/`SensesInput`/`NearestCache`'s
    // PUBLIC surface, and moving it out is what keeps this file under
    // `docs/STYLE.md`'s 600-line cap. Run it with `TAXI_TRACO_REF=/path/to/
    // traco_ref.json nice -n 10 env CARGO_BUILD_JOBS=4 cargo test -p taxi-world
    // -- --ignored`.
}
