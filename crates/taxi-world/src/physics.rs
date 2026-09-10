//! The per-tick physics: SPEC §3.1 items 1..28 and 32..35 ("tudo menos
//! passageiros" — items 29..31, boarding/respawn/delivery, are T4's
//! `passengers.rs`), plus the three small per-tick utilities every other task also
//! calls: [`is_night`], [`hour`] and [`update_daily_prices`].
//!
//! Ground truth is `campo.py:384-467` (`Campo.passo`), read line by line — the SPEC
//! table (`71-spec-mundo-taxi.md` §3.1) is a transcription, and this file follows the
//! CODE where the two ever disagree (none found while writing this). Every operation
//! below is numbered exactly as the SPEC table numbers it, in the same order, on
//! purpose: floating-point addition is not associative, so a step done early or late
//! changes the bits, and the identity gate (T5, SPEC §11) compares bits.
//!
//! Acceptance tests live in `tests/physics.rs` (plan §T2: "`crates/taxi-world/tests/
//! physics.rs` ou `#[cfg(test)]` internos"), not inline here — with them inline this
//! file would run past the 600-line cap (`docs/STYLE.md` §2).
//!
//! **What this file deliberately does NOT do:**
//! - Item 22 (`sv, pm = serv_ab()`) — `service_avail`/`service_cost` arrive as
//!   parameters, already computed by T4's `services.rs`, because in the source they
//!   are computed from the being's position AFTER this tick's movement (items 1-21)
//!   but the plan's T2 interface (plan §T2) hands them in rather than having physics
//!   reach into T4. Wiring the two in the right order (services state depends on the
//!   NEW position; T2 cannot compute it standalone) is T5's `tick.rs` job.
//! - Items 29-31 (passenger boarding, respawn, delivery) — T4's `passengers.rs`,
//!   explicitly out of scope (plan §T2 "Não tocar").  // check-language: allow -- quoting the plan verbatim
//! - Item 1 (`t += 1`) — the caller passes `t` already incremented (plan §T2's own
//!   doc: "`t` é o contador APÓS o incremento"); this crate has no global tick  // check-language: allow -- quoting the plan verbatim
//!   counter of its own to increment (T5's `tick.rs` owns that).

use crate::beings::Beings;
use crate::constants::{Config, DIRS, G, N_SERVICES, N_TYPES};
use crate::world::World;

/// Stand-in for T3's `senses::NearestCache` (plan §T3 "Interface") — nearest
/// service point/distance per being, per service type, `TIPOS` order
/// (`N_TYPES = 5`: posto, comida, agua, descanso, oficina).
///
/// `step_physics` needs only `points[.., 0]` (nearest fuel station), for the
/// breakdown tow (item 19). T3's `senses.rs` does not exist yet, so this crate has
// check-language: off -- the next three lines quote the plan verbatim
/// no single shared home for the type; the plan explicitly tells T2 to use "dublês
/// triviais" for now (plan §T2: "Enquanto T3/T4 não existem, o implementador usa
/// dublês triviais... os testes desta tarefa não dependem deles para nada além de
/// assinatura").
// check-language: on
///
/// **Ownership note for whoever lands T3 next.** This must become the ONE
/// `NearestCache` the crate has — either `senses.rs` re-exports this
/// (`pub use crate::physics::NearestCache;`) or the type moves to a shared module
/// both files import. Two structurally-identical-but-distinct `NearestCache` types
/// would each compile fine alone and then collide the moment `senses::observe` and
/// `physics::step_physics` are called from the same `tick.rs` (T5) with
/// incompatible types — the exact multi-implementer collision plan §4 exists to
/// avoid, so resolve it by picking one, not by keeping both.
#[derive(Debug, Clone)]
pub struct NearestCache {
    /// Nearest point of each type, `n * N_TYPES`, `TIPOS` order.
    pub points: Vec<[f32; 2]>,
    /// Distance to that point, `n * N_TYPES`, same layout as `points`.
    pub dist: Vec<f32>,
}

/// `hora() = ((t + fase) mod dia) / dia * 24` (SPEC §3.2, `campo.py:243`). Called
/// with the PRE-increment `t` by `senses::observe` (T3) and the POST-increment `t`
/// by `step_physics` — both callers pass whichever `t` the SPEC's own two call
// check-language: off -- the next two lines quote the SPEC verbatim
/// sites use (SPEC §13, armadilha 1: "duas chamadas de `noite_()` por tique com
/// valores possivelmente diferentes").
// check-language: on
pub fn hour(t: i64, phase: f32, cfg: &Config) -> f32 {
    ((t as f32 + phase) % cfg.day_ticks) / cfg.day_ticks * 24.0
}

/// `noite_() = (hora >= 22) | (hora < 6)` (`campo.py:254`). v0 has no `regime`
/// (seasons), so only the plain day/night branch applies (SPEC §3.2).
pub fn is_night(t: i64, phase: f32, cfg: &Config) -> bool {
    let h = hour(t, phase, cfg);
    h >= 22.0 || h < 6.0
}

/// Item 2 (SPEC §3.1): the daily price sine wave, one call per MAP per tick — `pmul
/// = pmul0·(1 + amp_preco·sin(2π·(dia/periodo_preco + fase_preco)))`
/// (`campo.py:387-388`). `dia = floor(t / cfg.dia)` is the same integer for every
/// map (`t` is the single global tick counter, SPEC §1), so this only needs to run
/// once per map, not once per being — unlike `step_physics`.
///
/// `w_price_mul` must have the same length as `w.price_mul_base`/`w.price_phase`
/// (one entry per district `k`); it is overwritten in place, not accumulated.
pub fn update_daily_prices(w_price_mul: &mut [f32], w: &World, cfg: &Config, t: i64) {
    let day = (t as f32 / cfg.day_ticks).floor();
    for k in 0..w_price_mul.len() {
        let phase = w.price_phase[k];
        let angle = 2.0 * std::f32::consts::PI * (day / cfg.day_price_period + phase);
        w_price_mul[k] = w.price_mul_base[k] * (1.0 + cfg.day_price_amplitude * angle.sin());
    }
}

/// SPEC §3.1 items 1..28 and 32..35 (`campo.py:384-467`, minus the passenger block
/// and minus item 22 — see this module's header). One call advances every being in
/// `b` by one tick.
///
/// - `t`: the tick counter AFTER this tick's increment (plan §T2 interface).
/// - `choice[i]`: `esc` for being `i`, already decided by T3's `decide()` — `0..=8`
///   moves (`8` = stay), `9..=18` uses service `esc - 9`.
/// - `nearest`: T3's `_ptos`/`_dist` from THIS tick's `obs()` (pre-movement
///   position) — only `nearest.points[i*N_TYPES + 0]` (nearest fuel station) is
///   read, for the breakdown tow (item 19).
/// - `service_avail`/`service_cost`: T4's `sv`/`cs` (`n * N_SERVICES` each,
///   `ACOES` order) — see this module's header for why physics receives these
///   instead of computing them.
pub fn step_physics(
    b: &mut Beings,
    w: &[World],
    cfg: &Config,
    t: i64,
    choice: &[u32],
    nearest: &NearestCache,
    service_avail: &[f32],
    service_cost: &[f32],
) {
    for i in 0..b.n {
        let world = &w[b.map_idx[i] as usize];

        // 3: livre reads the OLD stopped_ticks, THEN it is decremented (SPEC
        // §13.8/armadilha 8: get this order backwards and a being with
        // stopped_ticks == 1 wrongly becomes free THIS tick instead of next).
        let vivo = b.life[i] > 0.0;
        let livre = b.stopped_ticks[i] <= 0.0 && vivo;
        b.stopped_ticks[i] = (b.stopped_ticks[i] - 1.0).max(0.0);

        // 4
        let nv = is_night(t, b.phase[i], cfg);
        let v = if nv { cfg.night_speed } else { cfg.day_speed };

        // 5-6
        let esc = choice[i];
        let mut mov = livre && esc < G as u32 && esc != G as u32 - 1;
        let dir = DIRS[esc.min(G as u32 - 1) as usize];

        // 7
        let cu = world.terrain_cost_at(b.pos_x[i], b.pos_y[i]);
        let carrying_term = if b.carrying[i] >= 0.0 { 1.0 } else { 0.0 };
        let peso = 1.0 + 0.15 * carrying_term + 0.002 * b.reserve[i];

        // 8
        let pane = mov && b.fuel[i] < cfg.fuel_per_tick * cu * peso;
        mov = mov && !pane;
        let mov_f = if mov { 1.0 } else { 0.0 };

        // 9
        b.pos_x[i] = (b.pos_x[i] + mov_f * v * dir[0]).clamp(0.0, world.side);
        b.pos_y[i] = (b.pos_y[i] + mov_f * v * dir[1]).clamp(0.0, world.side);

        // 10
        let marcha = livre && !mov;
        let marcha_f = if marcha { 1.0 } else { 0.0 };

        // 11
        b.fuel[i] = (b.fuel[i] - cfg.fuel_per_tick * cu * peso * mov_f
            - cfg.fuel_per_tick * 0.33 * marcha_f)
            .max(0.0);

        // 12
        if mov {
            b.consumption[i] +=
                (cfg.fuel_per_tick * cu * peso / v.max(1e-3) - b.consumption[i]) * 0.05;
        }

        // 13
        b.heat[i] = (b.heat[i] + cfg.heat_rise * cu * mov_f - cfg.heat_fall * (1.0 - mov_f))
            .clamp(25.0, 120.0);

        // 14 (v0: 'agua' is not in custo_canais, so no metab factor here — SPEC §3.1#14)
        let vivo_f = if vivo { 1.0 } else { 0.0 };
        let hot_extra = if b.heat[i] > 60.0 { 1.0 } else { 0.0 };
        b.water[i] =
            (b.water[i] - cfg.water_per_tick * vivo_f - cfg.water_heat_per_tick * hot_extra * vivo_f)
                .max(0.0);

        // 15
        b.hunger[i] = (b.hunger[i] + cfg.hunger_per_tick * b.metabolism[i] * vivo_f).min(100.0);
        b.fatigue[i] =
            (b.fatigue[i] + cfg.fatigue_per_tick * mov_f - 0.1 * (1.0 - mov_f)).clamp(0.0, 100.0);

        // 16
        let exausto = livre && b.fatigue[i] >= 95.0;
        if exausto {
            b.stopped_ticks[i] = 12.0;
            b.fatigue[i] = 40.0;
            b.breakdowns[i] += 1.0;
        }

        // 17
        b.reserve[i] = (b.reserve[i] * (1.0 - cfg.reserve_decay_per_tick)).max(0.0);

        // 18
        if pane {
            b.breakdowns[i] += 1.0;
            b.faults[i] += 1.0;
            b.stopped_ticks[i] = 18.0;
        }

        // 19: the tow uses _ptos[0] from THIS tick's obs() (pre-movement position),
        // not a fresh lookup from the post-movement position (SPEC §3.1#19).
        if pane {
            let station = nearest.points[i * N_TYPES];
            b.pos_x[i] = station[0];
            b.pos_y[i] = station[1];
            b.fuel[i] = cfg.breakdown_fuel_refill;
            b.money[i] -= cfg.tow_cost;
        }

        // 20
        let pane_f = if pane { 1.0 } else { 0.0 };
        let kcal = b.reserve[i].min(cfg.breakdown_kcal);
        b.reserve[i] -= kcal * pane_f;
        let mut dn = cfg.damage_breakdown * pane_f + (cfg.breakdown_kcal - kcal) * pane_f;

        // 21: `heat` is reassigned to 60 BEFORE the D_RODAR term reads it — a being
        // that just overheated does not also pay the driving-while-hot penalty the
        // same tick (SPEC §3.1#21, armadilha 7).
        let qt = livre && b.heat[i] > cfg.heat_max;
        if qt {
            b.breakdowns[i] += 1.0;
            b.stopped_ticks[i] = 10.0;
        }
        b.heat[i] = if qt { 60.0 } else { b.heat[i] };
        let qt_f = if qt { 1.0 } else { 0.0 };
        let driving_hot = mov && b.heat[i] > 80.0;
        dn += cfg.damage_hot * qt_f + cfg.damage_driving_hot * if driving_hot { 1.0 } else { 0.0 };

        // 22 skipped on purpose — see this module's header.

        // 23: `service_avail`/`service_cost` are `sv`/`cs`; `paga` (can afford it)
        // is computed here, exactly as `campo.py:424` does inside `passo()` itself
        // (not inside `serv_ab`/`serv_need`, which only produce `sv`/`cs`).
        let pode = livre && esc >= 9;
        let k = if esc >= 9 { (esc - 9).min(N_SERVICES as u32 - 1) } else { 0 };
        let mut usa = [0.0f32; N_SERVICES];
        let mut spend = 0.0f32;
        for a in 0..N_SERVICES {
            let sv = service_avail[i * N_SERVICES + a];
            let onehot = if a as u32 == k { 1.0 } else { 0.0 };
            let pode_f = if pode { 1.0 } else { 0.0 };
            let cs = service_cost[i * N_SERVICES + a];
            let paga = if b.money[i] >= cs { 1.0 } else { 0.0 };
            let u = onehot * pode_f * sv * paga;
            usa[a] = u;
            spend += u * cs;
        }
        b.money[i] -= spend;

        // 24
        let litros =
            usa[0] * 25.0 + usa[1] * 50.0 + usa[2] * 75.0 + usa[3] * 100.0 + usa[4] * 150.0;
        let novo = b.fuel[i] + litros;
        let sobra = (novo - 100.0).max(0.0);
        b.fuel[i] = novo.min(100.0);
        b.reserve[i] = (b.reserve[i] + sobra * (1.0 - cfg.reserve_storage_loss_frac))
            .min(cfg.reserve_max);
        if litros > 0.0 {
            b.refuels[i] += 1.0;
        }
        b.liters[i] += litros;

        // 25
        b.water[i] = (b.water[i] + usa[5] * 50.0 + usa[6] * 100.0).min(100.0);
        if usa[5] + usa[6] > 0.0 {
            b.drinks[i] += 1.0;
        }

        // 26
        if usa[7] > 0.0 {
            b.hunger[i] = (b.hunger[i] - cfg.meal_relief).max(0.0);
            b.meals[i] += 1.0;
        }

        // 27
        if usa[8] > 0.0 {
            b.fatigue[i] = b.fatigue[i].min(40.0);
            b.stopped_ticks[i] = 12.0;
        }

        // 28
        if usa[9] > 0.0 {
            b.life[i] = (b.life[i] + cfg.repair_heal).min(100.0);
            b.heat[i] = 25.0;
            b.repairs[i] += 1.0;
        }

        // 29-31 (passengers) skipped on purpose — see this module's header.

        // 32: charged once per simulated day, `vivo` here is the SAME flag
        // captured at item 3 (pre-tick life), not recomputed after item 28's
        // possible repair (SPEC §3.1#32, `campo.py:459`).
        let midnight = (t as f32 + b.phase[i]) % cfg.day_ticks < 1.0;
        if midnight && vivo {
            b.money[i] -= cfg.rent * world.rent[b.home_district[i] as usize];
        }

        // 33
        dn += cfg.hunger_death_rate
            * ((b.hunger[i] - cfg.hunger_critical) / (100.0 - cfg.hunger_critical)).clamp(0.0, 1.0)
            + cfg.thirst_death_rate
                * ((cfg.thirst_critical - b.water[i]) / cfg.thirst_critical).clamp(0.0, 1.0);

        // 34
        b.money[i] = b.money[i].max(-cfg.debt_limit);
        let debt_frac = (-b.money[i] / cfg.debt_limit).clamp(0.0, 1.0);
        dn += 2.0 * debt_frac * debt_frac;

        // 35
        b.life[i] = (b.life[i] - dn * vivo_f).max(0.0);
        b.damage_total[i] += dn * vivo_f;
        b.alive_ticks[i] += vivo_f;
    }
}
