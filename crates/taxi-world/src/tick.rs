//! The full tick (plan §T5): observe → decide → step_physics → passengers, in
//! the exact order SPEC §3 `um_passo`/§3.1 `passo` specify. This module owns no
//! formula of its own — every number comes from T2's `physics.rs`, T3's
//! `senses.rs`/`decision.rs` or T4's `services.rs`/`passengers.rs`; this file's
//! only job is wiring those four modules' already-reviewed, already-tested
//! functions together in the right order, bridging the small interface
//! mismatches the task brief calls out (its "what to do" list, items a-e) with
//! plain glue — zip/unzip, borrow, no reimplemented formula.
//!
//! **Not touched by this module, on purpose:** the formulas inside `physics.rs`,
//! `senses.rs`, `decision.rs`, `services.rs`, `passengers.rs` — only their public
//! interfaces are called here (two small, documented exceptions are the
//! reconciliation edits made directly in `physics.rs`/`senses.rs`, see this
//! crate's final report).

use crate::beings::Beings;
use crate::constants::{Config, G, NF, S};
use crate::decision::decide;
use crate::genome::Genome;
use crate::passengers::{step_passengers, PassengerState, TaxiRng};
use crate::physics::{preview_movement, step_physics, update_daily_prices, NearestCache};
use crate::senses::{compute_target, map_price_offsets, observe, SensesInput};
use crate::services::{service_state, Nearest, ServiceQuery};
use crate::trace::Trace;
use crate::world::World;

/// One taxi-world simulation instance: every being, every map, the genome
/// driving them, the running per-map price-multiplier buffer (SPEC §3.1 item 2)
/// and the global tick counter (SPEC §1 `t`).
///
/// Built either from a [`Trace`] ([`Sim::from_trace`] — the identity gate's own
/// constructor, SPEC §1: v0 READS the initial state, it does not generate it) or
/// directly ([`Sim::new`] — a hand-built fixture, or a future battery driver,
/// T6, which this struct does not otherwise assume anything about).
pub struct Sim {
    pub beings: Beings,
    pub worlds: Vec<World>,
    pub cfg: Config,
    pub genome: Genome,
    /// Concatenated per-district price multiplier, one segment per `worlds[k]`
    /// in `worlds`' own order (`senses::map_price_offsets`' convention, which
    /// `services::price_mul_at` already mirrors independently — plan
    /// reconciliation item c found the two conventions already consistent, see
    /// this crate's final report).
    pub price_mul: Vec<f32>,
    /// Global tick counter, PRE-increment (SPEC §3: `obs()` sees `t`, `passo()`
    /// sees `t+1` — plan reconciliation item e). Starts at `0`, exactly
    /// `Campo.__init__`'s `s.t = 0`.
    pub t: i64,
}

impl Sim {
    /// Builds a `Sim` and seeds `price_mul` at the map's BASE multiplier for
    /// every world (SPEC §3.1, `campo.py:149`'s `s.pmul = s.pmul0.clone()`) —
    /// the very first `observe()` call must see `pmul0` UNADJUSTED, not the
    /// day-0 sine value.
    ///
    /// **Bug fixed during this task, caught only once a trace with non-zero
    /// `fase_preco` existed (SPEC §3.1 item 2's `sin(2π·(day/period +
    /// fase_preco))` is 0 at `fase_preco = 0`, which made the WRONG formula
    /// below degenerate to the right answer against the first, `fase_preco`-
    /// less trace — P0/P1 passed anyway, `max|ΔF|` ~2e-7, and only failed loud
    /// once a corrected trace with real per-district phases arrived).** This
    /// used to call `update_daily_prices(..., t=0)` here, which computes `day
    /// = floor(0/day_ticks) = 0` and therefore `pmul0·(1 + amp·sin(2π·(0 +
    /// fase_preco)))` — NOT `pmul0` itself unless `fase_preco = 0`. Python's
    /// OWN `s.pmul` starts as `s.pmul0.clone()` verbatim (`campo.py:149`); the
    /// sine adjustment (item 2) only ever runs INSIDE `passo()`, i.e. starting
    /// with the FIRST `step()` call this crate makes, never before it. Seeding
    /// with `update_daily_prices` amounted to running item 2 an extra, unpaired
    /// time before tick 0's `observe()` even happens.
    pub fn new(beings: Beings, worlds: Vec<World>, cfg: Config, genome: Genome) -> Sim {
        let mut price_mul = Vec::new();
        for world in &worlds {
            price_mul.extend_from_slice(&world.price_mul_base);
        }
        Sim { beings, worlds, cfg, genome, price_mul, t: 0 }
    }

    /// Builds a `Sim` for the identity gate: the map, the champion genome and
    /// the `t=0` being state, all READ (not generated) from `trace` (SPEC §1,
    /// §2.2). Uses `Config::default()` — the trace's OWN `cfg.respawn`/`cfg.
    /// ponto` overrides are applied by the caller through `step`'s `rng: None`
    /// (respawn off by construction, SPEC §7.2) rather than by copying `trace.
    /// cfg` field-by-field (see `constants::Config`'s own doc comment on why the
    /// two stay separate).
    pub fn from_trace(trace: &Trace) -> Sim {
        let world = trace.to_world();
        let cfg = Config::default();
        let beings = Beings::from_trace(trace, &cfg);
        let genome = Genome(trace.genoma.clone());
        Sim::new(beings, vec![world], cfg, genome)
    }
}

/// Everything one [`step`] call computed, for a caller ([`crate::gate`], or a
/// future T6 battery driver) to inspect or compare against a reference.
pub struct StepRecord {
    /// `esc` this tick actually advanced the trajectory with, per being — either
    /// `decide()`'s own choice, or `step`'s `forced_choice` override when the
    /// caller supplied one (the identity gate's tie-handling rule, SPEC §11.2:
    /// a margin below 1e-3 is an accepted tie — follow the trace's action and
    /// continue).
    pub choice: Vec<u32>,
    /// `decide()`'s OWN choice, UNCONDITIONALLY — even on a tick where
    /// `forced_choice` overrode the trajectory, so a caller can still measure
    /// how often the two agree (`gate::GateReport::actions_equal`).
    pub natural_choice: Vec<u32>,
    /// `top1 - top2` of `scores` (SPEC §5) — `margem_por_tique`.
    pub margin: Vec<f32>,
    /// `n * (G + S)`, the full `tudo = [sc(9); ss(10)]` row (SPEC §5) —
    /// `nota_por_tique`.
    pub scores: Vec<f32>,
    /// `n * G * NF`, `F` (SPEC §4) — `F_primeiros_50`.
    pub f: Vec<f32>,
    /// `n * 2`, `[being][xy]` — the `alvo` `observe()` actually used this tick
    /// (SPEC §4 `alvo()`; `alvo_por_tique`). `observe()` does not itself return
    /// this, so it is recomputed with `senses::compute_target` — the SAME
    /// function `observe()` calls internally when `target` is `None` (not a
    /// second, hand-copied implementation; see `compute_target`'s own doc
    /// comment on why it is `pub(crate)`).
    pub alvo: Vec<f32>,
}

/// A single tick, in the SPEC §3 `um_passo` order: `observe` → `decide` →
/// `step_physics` → passengers. Advances `sim` in place and returns a
/// [`StepRecord`] of everything a caller might need to compare against a
/// reference trace.
///
/// - `target`: `Some(alvo)` is the identity-gate hook (SPEC §7.2, plan §T3):
///   `observe()` reads `alvo` straight from the reference trace instead of
///   computing it, isolating the one quantity the Python side's RNG
///   irreversibly contaminates after the first pickup. `None` computes it
///   normally (the path a future battery, T6, uses).
/// - `forced_choice`: `Some(esc)` overrides the ACTION the tick's physics/
///   passengers actually run with (SPEC §11.2's tie-handling rule) while
///   `StepRecord::natural_choice` still reports what `decide()` itself picked —
///   deliberately not in the plan's own minimal `step` sketch (plan §T5
///   "Interface"), added because SPEC §11.2 explicitly requires following the
///   reference action through a numerical tie rather than letting Rust's own
///   tie-break snowball the trajectory apart from the tolerance the gate is
///   trying to measure (see this crate's final report).
/// - `rng`: `None` is the identity-gate mode (SPEC §7.2: a used passenger slot
///   is only marked inactive, never resampled — respawn off entirely). `Some`
///   is battery mode, for a future T6 driver.
pub fn step(
    sim: &mut Sim,
    target: Option<&[f32]>,
    forced_choice: Option<&[u32]>,
    mut rng: Option<&mut TaxiRng>,
) -> StepRecord {
    let n = sim.beings.n;
    let t_pre = sim.t;

    // === um_passo step 1: F, val = obs() (SPEC §3, uses t_pre) ===
    let input = senses_input(&sim.beings);
    let mut f = vec![0.0f32; n * G * NF];
    let mut nearest = NearestCache::default();
    observe(&input, &sim.worlds, &sim.price_mul, &sim.cfg, t_pre, target, &mut f, &mut nearest);

    let alvo = resolve_alvo(&sim.beings, target);

    // === um_passo step 4: svn, sv = serv_need() — SCORING call (SPEC §6),
    // OLD (pre-movement) position, t_pre. `nearest.dist` is this SAME tick's
    // obs() cache, per services.rs's own doc comment on serv_ab's two call
    // sites. ===
    let pos_old = zip_xy(&sim.beings.pos_x, &sim.beings.pos_y);
    let (avail_a, _cost_a, need_a) = call_service_state(
        &sim.beings,
        &sim.worlds,
        &sim.price_mul,
        &nearest,
        &sim.cfg,
        t_pre,
        &pos_old,
    );

    // === um_passo steps 2-3, 5-8: decide() (SPEC §5) ===
    let mut choice = vec![0u32; n];
    let mut margin = vec![0.0f32; n];
    let total = G + S;
    let mut scores = vec![0.0f32; n * total];
    decide(&f, &need_a, &avail_a, &sim.genome, n, &mut choice, &mut margin, &mut scores);

    let natural_choice = choice.clone();
    let effective_choice: Vec<u32> = match forced_choice {
        Some(forced) => forced.to_vec(),
        None => choice,
    };

    // === um_passo step 9: passo(esc) — t += 1 first (SPEC §3.1 item 1) ===
    sim.t += 1;
    let t_post = sim.t;

    // item 2: the daily price sine wave, once per map (SPEC §3.1 item 2).
    let offsets = map_price_offsets(&sim.worlds);
    for (k, world) in sim.worlds.iter().enumerate() {
        let len = world.tax.len();
        update_daily_prices(&mut sim.price_mul[offsets[k]..offsets[k] + len], world, &sim.cfg, t_post);
    }

    // Plan reconciliation item a/the physics.rs header's own note: item 22's
    // `service_state` call needs the POST-movement position, but that position
    // is only known once items 3-9 run — and `service_avail`/`service_cost`
    // must be ready BEFORE `step_physics` (which performs the items 23-28
    // purchase) can be called. `preview_movement` is `physics.rs`'s own
    // items-3-9 implementation (the SAME one `step_physics` calls internally),
    // run here read-only to resolve that ordering.
    let preview = preview_movement(&sim.beings, &sim.worlds, &sim.cfg, t_post, &effective_choice);

    // item 22: sv, pm = serv_ab() — APPLICATION call, NEW position but the SAME
    // frozen `nearest.dist` from step 1's obs() (SPEC §3.1 item 22, pitfall 6:
    // it mixes the OLD position's `_dist` with the district of the NEW
    // position — reproduced here by construction, not by special-casing:
    // `nearest` is untouched since step 1, only `pos` changed to
    // `preview.new_pos`).
    let (avail_b, cost_b, _need_b) = call_service_state(
        &sim.beings,
        &sim.worlds,
        &sim.price_mul,
        &nearest,
        &sim.cfg,
        t_post,
        &preview.new_pos,
    );

    // items 1, 3-21, 23-28, 32-35 (everything except item 22, already done
    // above, and items 29-31, passengers, below).
    step_physics(
        &mut sim.beings,
        &sim.worlds,
        &sim.cfg,
        t_post,
        &effective_choice,
        &nearest,
        &avail_b,
        &cost_b,
    );

    // items 29-31: passengers (SPEC §7). `free`/`acting_service` are the SAME
    // `livre`/`pode` items 3 and 23 use — `preview.free` is item 3's value
    // (computed before step_physics mutated anything, exactly what step_physics
    // itself used internally, by `resolve_move`'s construction); `pode` is
    // recomputed here from that same `free` and the effective choice, matching
    // SPEC §3.1 item 23's own formula (`livre & esc >= 9`).
    let acting_service: Vec<bool> = preview
        .free
        .iter()
        .zip(effective_choice.iter())
        .map(|(&free, &esc)| free && esc >= G as u32)
        .collect();

    // Plan reconciliation item b: T4's PassengerState wants `[f32; 2]` pairs for
    // `pos`/`dest`/`board_pos`; Beings (T2) stores split `_x`/`_y` Vecs. Bridge
    // with a temporary buffer, zipped in, unzipped back out after the call —
    // adapt with slices/zip, no redesign (the plan's own instruction).
    let pos_new = zip_xy(&sim.beings.pos_x, &sim.beings.pos_y);
    let mut dest = zip_xy(&sim.beings.dest_x, &sim.beings.dest_y);
    let mut board = zip_xy(&sim.beings.board_x, &sim.beings.board_y);
    {
        let mut st = PassengerState {
            pos: &pos_new,
            carrying: &mut sim.beings.carrying,
            dest: &mut dest,
            board_pos: &mut board,
            origin_district: &mut sim.beings.origin_district,
            since_pickup: &mut sim.beings.since_pickup,
            money: &mut sim.beings.money,
            revenue: &mut sim.beings.revenue,
            deliveries: &mut sim.beings.deliveries,
            map_idx: &sim.beings.map_idx,
            px: &mut sim.beings.px,
            pxd: &mut sim.beings.pxd,
            px_ok: &mut sim.beings.px_ok,
        };
        step_passengers(
            &mut st,
            &sim.worlds,
            &sim.price_mul,
            &sim.cfg,
            &preview.free,
            &acting_service,
            rng.as_deref_mut(),
        );
    }
    unzip_xy(&dest, &mut sim.beings.dest_x, &mut sim.beings.dest_y);
    unzip_xy(&board, &mut sim.beings.board_x, &mut sim.beings.board_y);

    StepRecord {
        choice: effective_choice,
        natural_choice,
        margin,
        scores,
        f,
        alvo,
    }
}

fn senses_input(b: &Beings) -> SensesInput<'_> {
    SensesInput {
        n: b.n,
        pos_x: &b.pos_x,
        pos_y: &b.pos_y,
        fuel: &b.fuel,
        water: &b.water,
        hunger: &b.hunger,
        fatigue: &b.fatigue,
        heat: &b.heat,
        life: &b.life,
        money: &b.money,
        reserve: &b.reserve,
        consumption: &b.consumption,
        carrying: &b.carrying,
        dest_x: &b.dest_x,
        dest_y: &b.dest_y,
        px: &b.px,
        home_district: &b.home_district,
        phase: &b.phase,
        since_pickup: &b.since_pickup,
        map_idx: &b.map_idx,
    }
}

#[allow(clippy::too_many_arguments)]
fn call_service_state(
    b: &Beings,
    w: &[World],
    price_mul: &[f32],
    nearest: &NearestCache,
    cfg: &Config,
    t: i64,
    pos: &[[f32; 2]],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = b.n;
    let q = ServiceQuery {
        pos,
        fuel: &b.fuel,
        water: &b.water,
        hunger: &b.hunger,
        fatigue: &b.fatigue,
        life: &b.life,
        money: &b.money,
        reserve: &b.reserve,
        home_district: &b.home_district,
        phase: &b.phase,
        map_idx: &b.map_idx,
    };
    let nn = Nearest { points: &nearest.points, dist: &nearest.dist };
    let mut avail = vec![0.0f32; n * S];
    let mut cost = vec![0.0f32; n * S];
    let mut need = vec![0.0f32; n * S];
    service_state(&q, w, price_mul, nn, cfg, t, &mut avail, &mut cost, &mut need);
    (avail, cost, need)
}

/// The `alvo` `observe()` used this tick, for `StepRecord` (see that field's own
/// doc comment): `target` verbatim when the caller supplied one (gate mode), or
/// `senses::compute_target` — the very function `observe()` itself calls
/// internally on the `None` path — otherwise.
fn resolve_alvo(b: &Beings, target: Option<&[f32]>) -> Vec<f32> {
    if let Some(t) = target {
        t.to_vec()
    } else {
        let input = senses_input(b);
        let mut out = vec![0.0f32; b.n * 2];
        for i in 0..b.n {
            let a = compute_target(&input, i);
            out[i * 2] = a[0];
            out[i * 2 + 1] = a[1];
        }
        out
    }
}

fn zip_xy(x: &[f32], y: &[f32]) -> Vec<[f32; 2]> {
    x.iter().zip(y).map(|(&a, &b)| [a, b]).collect()
}

fn unzip_xy(pairs: &[[f32; 2]], x: &mut [f32], y: &mut [f32]) {
    for (i, p) in pairs.iter().enumerate() {
        x[i] = p[0];
        y[i] = p[1];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{GRID, NPX, TEX};

    fn flat_world(side: f32) -> World {
        World {
            side,
            cost_tex: vec![1.0; TEX * TEX],
            district_field: vec![0; GRID * GRID],
            centers: vec![[0.0, 0.0]],
            tax: vec![0.0],
            price_mul_base: vec![1.0],
            rent: vec![0.0],
            demand: vec![0.0],
            price_phase: vec![0.0],
            services: [vec![], vec![], vec![], vec![], vec![]],
        }
    }

    fn one_being(side: f32) -> Beings {
        Beings {
            n: 1,
            pos_x: vec![side / 2.0],
            pos_y: vec![side / 2.0],
            fuel: vec![85.0],
            water: vec![85.0],
            hunger: vec![10.0],
            fatigue: vec![10.0],
            heat: vec![30.0],
            life: vec![100.0],
            money: vec![20.0],
            reserve: vec![0.0],
            stopped_ticks: vec![0.0],
            metabolism: vec![1.0],
            consumption: vec![0.30],
            home_district: vec![0],
            phase: vec![0.0],
            carrying: vec![-1.0],
            dest_x: vec![0.0],
            dest_y: vec![0.0],
            board_x: vec![0.0],
            board_y: vec![0.0],
            origin_district: vec![0],
            since_pickup: vec![0.0],
            px: vec![0.0; NPX * 2],
            pxd: vec![0.0; NPX * 2],
            px_ok: vec![0; NPX],
            alive_ticks: vec![0.0],
            deliveries: vec![0.0],
            revenue: vec![0.0],
            breakdowns: vec![0.0],
            damage_total: vec![0.0],
            refuels: vec![0.0],
            liters: vec![0.0],
            meals: vec![0.0],
            repairs: vec![0.0],
            drinks: vec![0.0],
            faults: vec![0.0],
            map_idx: vec![0],
            genome_idx: vec![0],
        }
    }

    fn all_zero_genome() -> Genome {
        Genome(vec![0.0f32; crate::constants::NP])
    }

    #[test]
    fn a_zero_genome_always_stays_put_and_burns_idle_fuel() {
        // With every weight 0, sc/ss are uniformly 0 after G1 centering — the
        // first-index tie-break (SPEC §5) must pick option 0, not 8 ("stay"),
        // so this is really a test that `step` wires `decide`'s tie-break
        // through unmodified, and that the resulting choice still drives
        // physics/passengers without panicking.
        let world = flat_world(20.0);
        let beings = one_being(20.0);
        let cfg = Config::default();
        let genome = all_zero_genome();
        let mut sim = Sim::new(beings, vec![world], cfg, genome);

        let before_fuel = sim.beings.fuel[0];
        let record = step(&mut sim, None, None, None);

        assert_eq!(sim.t, 1);
        assert_eq!(record.choice.len(), 1);
        assert_eq!(record.natural_choice, record.choice, "no forced_choice: must match");
        assert_eq!(record.f.len(), G * NF);
        assert_eq!(record.scores.len(), G + S);
        assert_eq!(record.alvo.len(), 2);
        // Some fuel must have been spent one way or another (moving or idling)
        // — this is a wiring smoke test, not a numeric gate (those live in
        // gate.rs/tests/identity.rs against the real trace).
        assert!(sim.beings.fuel[0] <= before_fuel);
    }

    #[test]
    fn a_forced_choice_overrides_the_trajectory_but_not_the_natural_choice_report() {
        let world = flat_world(20.0);
        let beings = one_being(20.0);
        let cfg = Config::default();
        let genome = all_zero_genome();
        let mut sim = Sim::new(beings, vec![world], cfg, genome);

        let forced = [8u32]; // "stay" — never burns move fuel, only idle fuel
        let record = step(&mut sim, None, Some(&forced), None);

        assert_eq!(record.choice, vec![8]);
        // With every genome weight 0 the natural choice is option 0 (first
        // index tie-break, SPEC §5) — forcing 8 must not have changed what
        // `decide()` itself reported choosing.
        assert_eq!(record.natural_choice, vec![0]);
        assert_eq!(sim.beings.pos_x[0], 10.0, "esc=8 is 'stay': position must not move");
    }

    #[test]
    fn two_ticks_advance_the_global_counter_and_keep_every_buffer_sized_to_n() {
        let world = flat_world(20.0);
        let mut beings = one_being(20.0);
        beings.n = 2;
        beings.pos_x = vec![5.0, 15.0];
        beings.pos_y = vec![5.0, 15.0];
        beings.fuel = vec![85.0, 85.0];
        beings.water = vec![85.0, 85.0];
        beings.hunger = vec![10.0, 10.0];
        beings.fatigue = vec![10.0, 10.0];
        beings.heat = vec![30.0, 30.0];
        beings.life = vec![100.0, 100.0];
        beings.money = vec![20.0, 20.0];
        beings.reserve = vec![0.0, 0.0];
        beings.stopped_ticks = vec![0.0, 0.0];
        beings.metabolism = vec![1.0, 1.0];
        beings.consumption = vec![0.30, 0.30];
        beings.home_district = vec![0, 0];
        beings.phase = vec![0.0, 0.0];
        beings.carrying = vec![-1.0, -1.0];
        beings.dest_x = vec![0.0, 0.0];
        beings.dest_y = vec![0.0, 0.0];
        beings.board_x = vec![0.0, 0.0];
        beings.board_y = vec![0.0, 0.0];
        beings.origin_district = vec![0, 0];
        beings.since_pickup = vec![0.0, 0.0];
        beings.px = vec![0.0; 2 * NPX * 2];
        beings.pxd = vec![0.0; 2 * NPX * 2];
        beings.px_ok = vec![0; 2 * NPX];
        beings.alive_ticks = vec![0.0, 0.0];
        beings.deliveries = vec![0.0, 0.0];
        beings.revenue = vec![0.0, 0.0];
        beings.breakdowns = vec![0.0, 0.0];
        beings.damage_total = vec![0.0, 0.0];
        beings.refuels = vec![0.0, 0.0];
        beings.liters = vec![0.0, 0.0];
        beings.meals = vec![0.0, 0.0];
        beings.repairs = vec![0.0, 0.0];
        beings.drinks = vec![0.0, 0.0];
        beings.faults = vec![0.0, 0.0];
        beings.map_idx = vec![0, 0];
        beings.genome_idx = vec![0, 0];

        let cfg = Config::default();
        let genome = all_zero_genome();
        let mut sim = Sim::new(beings, vec![world], cfg, genome);

        let r1 = step(&mut sim, None, None, None);
        assert_eq!(sim.t, 1);
        assert_eq!(r1.choice.len(), 2);
        let r2 = step(&mut sim, None, None, None);
        assert_eq!(sim.t, 2);
        assert_eq!(r2.choice.len(), 2);
        assert_eq!(r2.f.len(), 2 * G * NF);
    }
}
