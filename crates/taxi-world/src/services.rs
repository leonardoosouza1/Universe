//! Service availability, cost and need (SPEC §6) — the `serv_ab`/`serv_need`/`custos`
//! trio (`campo.py:356-382`), plus the one piece of service-purchase ARITHMETIC
//! (refuel overflow into the reserve tank) this task's own acceptance test list
//! names explicitly (`overflow_beyond_a_hundred_liters_goes_to_the_reserve_minus_fifteen_percent`,
//! plan §T4). See `docs/plano-mundo-taxi.md` §T4 and
//! `/home/leonardo/projects/LLM/neuromod-lab/docs/71-spec-mundo-taxi.md` §6.
//!
//! # Why this module does not take `Beings` or T3's `NearestCache`
//!
//! T4's brief is explicit: T2 (`beings.rs`) and T3 (`senses.rs`) are being written in
//! parallel and must not be created here. This module instead takes its own
//! documented, flat-slice input structs ([`ServiceQuery`], [`Nearest`]) — the same
//! per-being layout the plan's `Beings` sketch (plan §3.1) already uses for these
//! fields (`fuel`, `water`, `hunger`, `fatigue`, `life`, `money`, `reserve`,
//! `home_district`, `phase`, `map_idx`), with `pos` as `&[[f32; 2]]` per this task's
//! brief. T5 wires the real `Beings`/`NearestCache` fields into these slices at the
//! call site — no conversion, just borrowing (`&beings.fuel[..]`, etc. — the one
//! exception is `Beings`' own `pos_x`/`pos_y` split, which T5 zips into `[f32; 2]`
//! pairs; see this crate's final report for that noted divergence).
//!
//! # `service_state` doubles as BOTH of `serv_ab`'s two call sites
//!
//! The SPEC calls `serv_ab`/`serv_need` TWICE per tick with different position
//! state: once before movement (`um_passo` step 4, scoring) and once after movement
//! (`passo` item 22, application) — reusing the SAME `_dist` cached at `obs()` time
//! in both calls, but a DIFFERENT `distrito(pos)` (SPEC §3.1 item 22, "armadilha 6").
//! `service_state` reproduces both call sites by construction: `nearest.dist` is
//! always the frozen old-position distance (the caller passes `observe()`'s output
//! unchanged), while the district/price-multiplier lookup always reads whatever
//! `q.pos` currently holds — pre-movement positions for the scoring call, post-
//! movement positions for the application call. The caller (T5's `tick.rs`) decides
//! which `pos`/`t` to pass at each call site; this function does not special-case it.

use crate::constants::{Config, N_SERVICES, N_TYPES};
use crate::world::World;

/// The 5 service types (SPEC §0 `TIPOS`), in the SAME order as `World::services`
/// and `Nearest`'s `n * N_TYPES` layout — the discriminants ARE the `TIPOS` index,
/// so `kind as usize` indexes both directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceKind {
    FuelStation = 0, // posto
    Food = 1,        // comida
    Water = 2,       // agua
    Rest = 3,        // descanso
    Repair = 4,      // oficina
}

/// `ACOES` (SPEC §6, `campo.py:16-17`): 5 fuel quantities, 2 water quantities, then
/// food/rest/repair (quantity unused — those three are flat-rate). Index `a` here IS
/// `esc - 9` (SPEC §5): `esc ∈ [9, 18]` selects `ACTIONS[esc - 9]`.
pub const ACTIONS: [(ServiceKind, f32); N_SERVICES] = [
    (ServiceKind::FuelStation, 25.0),
    (ServiceKind::FuelStation, 50.0),
    (ServiceKind::FuelStation, 75.0),
    (ServiceKind::FuelStation, 100.0),
    (ServiceKind::FuelStation, 150.0),
    (ServiceKind::Water, 50.0),
    (ServiceKind::Water, 100.0),
    (ServiceKind::Food, 0.0),
    (ServiceKind::Rest, 0.0),
    (ServiceKind::Repair, 0.0),
];

/// Borrowed view of the "nearest service" cache `observe()` (T3's `senses.rs`,
/// not yet written) produces once per tick from `World::nearest_services`
/// (SPEC §2.3 `_perto_exata`). Plan §T3 names the owning struct `NearestCache`
/// with owned `Vec` fields; this module borrows two flat slices instead of
/// depending on that not-yet-existing type — T5 passes `&cache.points[..]` /
/// `&cache.dist[..]` at the call site.
#[derive(Debug, Clone, Copy)]
pub struct Nearest<'a> {
    /// `n * N_TYPES`, `TIPOS` order, unused by `service_state` (kept for symmetry
    /// with T3's `NearestCache` and because a future caller may want it alongside
    /// `dist` without a second lookup).
    pub points: &'a [[f32; 2]],
    /// `n * N_TYPES`, `TIPOS` order — the distance from the OLD (pre-movement)
    /// position to the nearest point of each type, frozen at `observe()` time
    /// (SPEC §3.1 item 22).
    pub dist: &'a [f32],
}

/// The per-being body fields `service_state` reads (SPEC §6.1, §6.2). Field names
/// match the plan's `Beings` sketch (plan §3.1) one for one, so T5's wiring is a
/// direct borrow, not a redesign.
pub struct ServiceQuery<'a> {
    /// `n`, CURRENT position — see this module's doc comment on why the caller
    /// controls whether that means pre- or post-movement.
    pub pos: &'a [[f32; 2]],
    pub fuel: &'a [f32],
    pub water: &'a [f32],
    pub hunger: &'a [f32],
    pub fatigue: &'a [f32],
    pub life: &'a [f32],
    pub money: &'a [f32],
    pub reserve: &'a [f32],
    /// `casa` (SPEC §1) — the district a being calls home; the rest-stop cost is
    /// waived there (SPEC §6, `casa?`).
    pub home_district: &'a [u32],
    /// `fase` (SPEC §1) — per-map simulation phase offset, broadcast to every
    /// being sharing that map, needed for `hora()`/`noite_()` (SPEC §3.2).
    pub phase: &'a [f32],
    /// Which `World` (and which segment of `price_mul`) each being is on.
    pub map_idx: &'a [u32],
}

/// SPEC §6.1 (`serv_ab`) and §6.2 (`serv_need`/`custos`), combined into one pass
/// per being. Writes three `n * N_SERVICES` arrays:
///
/// - `out_avail[i*10+a]` = `sv_[a]` (SPEC §6.2's second return value): proximity
///   AND affordability, `1.0`/`0.0`. This is what masks the decision
///   (`ss[a] = -1e9` where this is `<= 0`, SPEC §5) and what
///   `a_service_you_cannot_afford_is_not_offered` checks directly.
/// - `out_cost[i*10+a]` = `cs[a]`, the money cost, independent of affordability.
/// - `out_need[i*10+a]` = `svn[a]` = `out_avail[i*10+a] * nec[a]` (SPEC §6.2) — the
///   scoring numerator T3's `decide()` multiplies by the genome's service weights.
///
/// `price_mul` is the current day's per-district multiplier (SPEC §3.1 item 2,
/// T2's `update_daily_prices`), laid out as one segment per `World` in `w`, each
/// segment `w[k].tax.len()` districts long, in `w`'s own order — the natural
/// flattening of T2's documented per-world `update_daily_prices(w_price_mul: &mut
/// [f32], w: &World, ...)` signature (plan §T2), which this module's
/// [`price_mul_at`] mirrors.
pub fn service_state(
    q: &ServiceQuery,
    w: &[World],
    price_mul: &[f32],
    nearest: Nearest,
    cfg: &Config,
    t: i64,
    out_avail: &mut [f32],
    out_cost: &mut [f32],
    out_need: &mut [f32],
) {
    let n = q.pos.len();
    assert_eq!(nearest.dist.len(), n * N_TYPES, "nearest.dist must be n * N_TYPES");
    assert_eq!(out_avail.len(), n * N_SERVICES, "out_avail must be n * N_SERVICES");
    assert_eq!(out_cost.len(), n * N_SERVICES, "out_cost must be n * N_SERVICES");
    assert_eq!(out_need.len(), n * N_SERVICES, "out_need must be n * N_SERVICES");

    for i in 0..n {
        let world = &w[q.map_idx[i] as usize];
        let district = world.district_at(q.pos[i][0], q.pos[i][1]);
        let pm = price_mul_at(price_mul, w, q.map_idx[i], district);
        let at_home = district == q.home_district[i];
        let night = is_night(t, q.phase[i], cfg);

        // `perto[t] = _dist[t] <= raio`, for t in TIPOS order (SPEC §6.1).
        let mut perto = [false; N_TYPES];
        for ty in 0..N_TYPES {
            perto[ty] = nearest.dist[i * N_TYPES + ty] <= cfg.service_radius_m;
        }
        // `loja_no_posto`: comida is updated FIRST, agua reads the ALREADY-updated
        // comida (SPEC §6.1 — the order is spelled out because a future edit to
        // `comida`'s rule would silently change `agua` too if this were reordered).
        if cfg.shop_at_station != 0.0 {
            perto[ServiceKind::Food as usize] |= perto[ServiceKind::FuelStation as usize];
            perto[ServiceKind::Water as usize] |=
                perto[ServiceKind::FuelStation as usize] | perto[ServiceKind::Food as usize];
        }

        let cabe = 100.0 - q.fuel[i];
        let sobra_res = (cfg.reserve_max - q.reserve[i]).max(0.0);

        for (a, (kind, qty)) in ACTIONS.iter().enumerate() {
            let mut sv = perto[*kind as usize];
            if *kind == ServiceKind::FuelStation {
                sv &= !night; // the fuel station closes at night (SPEC §6.1)
            }

            let cost = match kind {
                ServiceKind::FuelStation => cfg.fuel_price_per_l * qty * pm,
                ServiceKind::Water => cfg.water_price_per_l * qty * pm,
                ServiceKind::Food => cfg.food_price * pm,
                ServiceKind::Rest => {
                    if at_home {
                        0.0
                    } else {
                        cfg.rest_price * pm
                    }
                }
                ServiceKind::Repair => cfg.repair_price * pm,
            };
            let paga = q.money[i] >= cost;

            let need = match kind {
                // nec[a] = min(q, cabe)/100 + max(min(q-cabe, sobra_res), 0)/100*0.5
                ServiceKind::FuelStation => {
                    let tank_term = qty.min(cabe) / 100.0;
                    let reserve_term = (qty - cabe).min(sobra_res).max(0.0) / 100.0 * 0.5;
                    tank_term + reserve_term
                }
                ServiceKind::Water => qty.min(100.0 - q.water[i]) / 100.0,
                ServiceKind::Food => q.hunger[i] / 100.0,
                ServiceKind::Rest => q.fatigue[i] / 100.0,
                ServiceKind::Repair => 1.0 - q.life[i] / 100.0,
            };

            let sv_f = if sv { 1.0 } else { 0.0 };
            let paga_f = if paga { 1.0 } else { 0.0 };
            let avail = sv_f * paga_f;
            let idx = i * N_SERVICES + a;
            out_avail[idx] = avail;
            out_cost[idx] = cost;
            out_need[idx] = avail * need;
        }
    }
}

/// The effect of buying `liters` of fuel (SPEC §3.1 item 24, `campo.py:428-431`):
/// returns `(new_fuel, new_reserve)`. A tank over 100 L spills the overflow into
/// the reserve tank, MINUS a storage loss (`cfg.reserve_storage_loss_frac`,
/// `PERDA_ARMAZ = 0.15`), capped at `cfg.reserve_max`.
///
/// Not part of the plan's named T4 interface (`service_state`/`step_passengers`
/// only) — added because the plan's own T4 acceptance-test list names
/// `overflow_beyond_a_hundred_liters_goes_to_the_reserve_minus_fifteen_percent`,
/// and that formula (the `1 - PERDA_ARMAZ` loss) exists nowhere in `serv_need`'s
/// `nec` calculation — only in the state-mutation SPEC assigns to `passo()`'s
/// service-application block. See the final report for this divergence.
pub fn apply_refuel(fuel: f32, reserve: f32, liters: f32, cfg: &Config) -> (f32, f32) {
    let topped_up = fuel + liters;
    let overflow = (topped_up - 100.0).max(0.0);
    let new_fuel = topped_up.min(100.0);
    let new_reserve = (reserve + overflow * (1.0 - cfg.reserve_storage_loss_frac)).min(cfg.reserve_max);
    (new_fuel, new_reserve)
}

/// Looks up `district`'s current price multiplier for the world at `map_idx`, in
/// the per-world-concatenated `price_mul` buffer this module and
/// [`crate::passengers::step_passengers`] both read (see `service_state`'s doc
/// comment). `pub(crate)` — shared with `passengers.rs`'s fare calculation, not
/// part of the plan's public T4 interface.
pub(crate) fn price_mul_at(price_mul: &[f32], w: &[World], map_idx: u32, district: u32) -> f32 {
    let offset: usize = w[..map_idx as usize].iter().map(|world| world.tax.len()).sum();
    price_mul[offset + district as usize]
}

/// `hora() = ((t + fase) mod dia) / dia * 24` (SPEC §3.2, `campo.py:243`).
fn hour(t: i64, phase: f32, cfg: &Config) -> f32 {
    ((t as f32 + phase) % cfg.day_ticks) / cfg.day_ticks * 24.0
}

/// `noite_() = (hora >= 22) | (hora < 6)` (SPEC §3.2, `campo.py:254`, no regime in
/// v0). Duplicated here rather than imported from T2's `physics.rs` (not yet
/// written, and this task's hard rule forbids depending on it) — a one-line
/// formula transcribed identically from the same SPEC citation carries no
/// divergence risk.
fn is_night(t: i64, phase: f32, cfg: &Config) -> bool {
    let h = hour(t, phase, cfg);
    !(6.0..22.0).contains(&h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{GRID, TEX};

    /// A minimal `World` with `k` districts, everything else uniform — same
    /// pattern as `world.rs`'s own `tiny_world()` test helper.
    fn tiny_world(k: usize) -> World {
        World {
            side: 100.0,
            cost_tex: vec![1.0; TEX * TEX],
            district_field: vec![0; GRID * GRID],
            centers: vec![[0.0, 0.0]; k],
            tax: vec![0.0; k],
            price_mul_base: vec![1.0; k],
            rent: vec![0.0; k],
            demand: vec![0.0; k],
            price_phase: vec![0.0; k],
            services: [vec![], vec![], vec![], vec![], vec![]],
        }
    }

    /// Owned per-being body data for one test being — a `ServiceQuery` just
    /// borrows these fields, so tests can tweak one before calling `.query()`.
    struct Body {
        pos: [[f32; 2]; 1],
        fuel: [f32; 1],
        water: [f32; 1],
        hunger: [f32; 1],
        fatigue: [f32; 1],
        life: [f32; 1],
        money: [f32; 1],
        reserve: [f32; 1],
        home_district: [u32; 1],
        phase: [f32; 1],
        map_idx: [u32; 1],
        dist: [f32; N_TYPES],
        points: [[f32; 2]; N_TYPES],
    }

    impl Body {
        fn new(near: [bool; N_TYPES]) -> Self {
            Body {
                pos: [[0.0, 0.0]],
                fuel: [60.0],
                water: [60.0],
                hunger: [30.0],
                fatigue: [20.0],
                life: [100.0],
                money: [1000.0],
                reserve: [100.0],
                home_district: [0],
                phase: [120.0], // hour = 12, daytime
                map_idx: [0],
                dist: std::array::from_fn(|ty| if near[ty] { 1.0 } else { 999.0 }),
                points: [[0.0, 0.0]; N_TYPES],
            }
        }

        fn query(&self) -> ServiceQuery<'_> {
            ServiceQuery {
                pos: &self.pos,
                fuel: &self.fuel,
                water: &self.water,
                hunger: &self.hunger,
                fatigue: &self.fatigue,
                life: &self.life,
                money: &self.money,
                reserve: &self.reserve,
                home_district: &self.home_district,
                phase: &self.phase,
                map_idx: &self.map_idx,
            }
        }

        fn nearest(&self) -> Nearest<'_> {
            Nearest { points: &self.points, dist: &self.dist }
        }
    }

    #[test]
    fn the_shop_at_the_station_sells_food_and_water_in_that_order() {
        let w = tiny_world(1);
        let price_mul = [1.0];
        // (near_posto, near_comida) -> water must end up available whenever
        // EITHER is near (SPEC §6.1: water inherits from both the fuel station
        // and the already-widened food availability).
        for &(near_posto, near_comida) in &[(false, false), (false, true), (true, false), (true, true)] {
            let mut near = [false; N_TYPES];
            near[ServiceKind::FuelStation as usize] = near_posto;
            near[ServiceKind::Food as usize] = near_comida;
            let body = Body::new(near);
            let mut avail = [0.0; N_SERVICES];
            let mut cost = [0.0; N_SERVICES];
            let mut need = [0.0; N_SERVICES];
            service_state(
                &body.query(), &[w.clone()], &price_mul, body.nearest(), &Config::default(), 0,
                &mut avail, &mut cost, &mut need,
            );

            let expected = near_posto || near_comida;
            assert_eq!(
                avail[5] > 0.0 || avail[6] > 0.0,
                expected,
                "posto={near_posto} comida={near_comida}: water availability wrong"
            );
            assert_eq!(avail[7] > 0.0, expected, "posto={near_posto} comida={near_comida}: food availability wrong");
        }
    }

    #[test]
    fn the_fuel_station_is_closed_at_night() {
        let w = tiny_world(1);
        let price_mul = [1.0];
        let mut near = [false; N_TYPES];
        near[ServiceKind::FuelStation as usize] = true;
        near[ServiceKind::Food as usize] = true;
        let mut body = Body::new(near);
        body.phase = [230.0]; // hour = 23
        let mut avail = [0.0; N_SERVICES];
        let mut cost = [0.0; N_SERVICES];
        let mut need = [0.0; N_SERVICES];
        service_state(
            &body.query(), &[w], &price_mul, body.nearest(), &Config::default(), 0,
            &mut avail, &mut cost, &mut need,
        );

        for a in 0..5 {
            assert_eq!(avail[a], 0.0, "fuel action {a} must be closed at night");
        }
        assert!(avail[7] > 0.0, "food must be unaffected by night");
    }

    #[test]
    fn sleeping_at_home_is_free() {
        let w = tiny_world(1);
        let price_mul = [1.0];
        let mut near = [false; N_TYPES];
        near[ServiceKind::Rest as usize] = true;
        let mut body = Body::new(near);
        body.home_district = [0]; // district_at(0,0) == 0
        let mut avail = [0.0; N_SERVICES];
        let mut cost = [0.0; N_SERVICES];
        let mut need = [0.0; N_SERVICES];
        service_state(
            &body.query(), &[w], &price_mul, body.nearest(), &Config::default(), 0,
            &mut avail, &mut cost, &mut need,
        );

        assert_eq!(cost[8], 0.0, "resting at home must cost nothing");
    }

    #[test]
    fn the_refuel_need_splits_between_tank_and_reserve() {
        let w = tiny_world(1);
        let price_mul = [1.0];
        let mut near = [false; N_TYPES];
        near[ServiceKind::FuelStation as usize] = true;
        let mut body = Body::new(near);
        body.fuel = [60.0];
        body.reserve = [100.0];
        let mut avail = [0.0; N_SERVICES];
        let mut cost = [0.0; N_SERVICES];
        let mut need = [0.0; N_SERVICES];
        service_state(
            &body.query(), &[w], &price_mul, body.nearest(), &Config::default(), 0,
            &mut avail, &mut cost, &mut need,
        );

        // fuel=60 (cabe=40), res=100 (sobra_res=20), RES_MAX=120 (default).
        let expected = [0.25, 0.45, 0.50, 0.50, 0.50];
        for a in 0..5 {
            assert!(
                (need[a] - expected[a]).abs() < 1e-5,
                "action {a}: expected nec {}, got {}",
                expected[a],
                need[a]
            );
        }
    }

    #[test]
    fn a_service_you_cannot_afford_is_not_offered() {
        let w = tiny_world(1);
        let price_mul = [1.0];
        let mut near = [false; N_TYPES];
        near[ServiceKind::Food as usize] = true;
        let mut body = Body::new(near);
        body.money = [0.0];
        let mut avail = [0.0; N_SERVICES];
        let mut cost = [0.0; N_SERVICES];
        let mut need = [0.0; N_SERVICES];
        service_state(
            &body.query(), &[w], &price_mul, body.nearest(), &Config::default(), 0,
            &mut avail, &mut cost, &mut need,
        );

        assert!(cost[7] > 0.0, "food must have a nonzero price");
        assert_eq!(avail[7], 0.0, "with no money, food must not be offered");
    }

    #[test]
    fn overflow_beyond_a_hundred_liters_goes_to_the_reserve_minus_fifteen_percent() {
        let cfg = Config::default();
        let (new_fuel, new_reserve) = apply_refuel(90.0, 0.0, 25.0, &cfg);
        assert_eq!(new_fuel, 100.0, "the tank caps at 100 L");
        // overflow = (90 + 25) - 100 = 15; 15 * (1 - 0.15) = 12.75.
        assert!((new_reserve - 12.75).abs() < 1e-5, "got {new_reserve}");
    }

    #[test]
    fn refuel_below_the_tank_cap_never_touches_the_reserve() {
        let cfg = Config::default();
        let (new_fuel, new_reserve) = apply_refuel(10.0, 5.0, 25.0, &cfg);
        assert_eq!(new_fuel, 35.0);
        assert_eq!(new_reserve, 5.0, "no overflow, reserve must be untouched");
    }

    #[test]
    fn price_mul_lookup_offsets_by_world_index() {
        let w0 = tiny_world(2);
        let w1 = tiny_world(3);
        let worlds = [w0, w1];
        // world 0 occupies price_mul[0..2], world 1 occupies price_mul[2..5].
        let price_mul = [10.0, 11.0, 20.0, 21.0, 22.0];
        assert_eq!(price_mul_at(&price_mul, &worlds, 0, 0), 10.0);
        assert_eq!(price_mul_at(&price_mul, &worlds, 0, 1), 11.0);
        assert_eq!(price_mul_at(&price_mul, &worlds, 1, 0), 20.0);
        assert_eq!(price_mul_at(&price_mul, &worlds, 1, 2), 22.0);
    }
}
