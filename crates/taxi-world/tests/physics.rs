//! Acceptance tests for T2 (plan §T2 "Testes de aceite"), against `physics.rs`'s
//! `step_physics` (`campo.py:384-467`, SPEC §3.1). An integration test (not
//! `#[cfg(test)]` inside `physics.rs` itself) so the implementation file stays under
//! the 600-line cap (`docs/STYLE.md` §2) with the formula comments intact — see that
//! file's own header for exactly which SPEC items it implements and which three it
//! deliberately skips (22, 29-31).

use taxi_world::beings::Beings;
use taxi_world::constants::{Config, GRID, NPX, N_SERVICES, N_TYPES, TEX};
use taxi_world::physics::{step_physics, NearestCache};
use taxi_world::World;

/// A flat, uniform `TEX*TEX`/`GRID*GRID` world (terrain cost `1.0` everywhere, one
/// district) — enough for every physics test below, none of which probes map
/// geometry (that is `world.rs`'s job, T1, covered by its own tests).
fn flat_world(side: f32, district_rent: f32) -> World {
    World {
        side,
        cost_tex: vec![1.0; TEX * TEX],
        district_field: vec![0; GRID * GRID],
        centers: vec![[0.0, 0.0]],
        tax: vec![0.0],
        price_mul_base: vec![1.0],
        rent: vec![district_rent],
        demand: vec![0.0],
        price_phase: vec![0.0],
        services: [vec![], vec![], vec![], vec![], vec![]],
    }
}

/// One being, every field given a safe, inert default so a test only has to
/// override what it actually probes.
fn one_being() -> Beings {
    Beings {
        n: 1,
        pos_x: vec![100.0],
        pos_y: vec![100.0],
        fuel: vec![50.0],
        water: vec![80.0],
        hunger: vec![10.0],
        fatigue: vec![10.0],
        heat: vec![30.0],
        life: vec![100.0],
        money: vec![0.0],
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

fn no_services(n: usize) -> (Vec<f32>, Vec<f32>) {
    (vec![0.0; n * N_SERVICES], vec![0.0; n * N_SERVICES])
}

fn no_nearest(n: usize) -> NearestCache {
    NearestCache {
        points: vec![[0.0, 0.0]; n * N_TYPES],
        dist: vec![f32::INFINITY; n * N_TYPES],
    }
}

#[test]
fn staying_put_still_burns_idle_fuel() {
    let mut beings = one_being();
    let world = flat_world(200.0, 1.0);
    let cfg = Config::default();
    let (avail, cost) = no_services(1);
    let nearest = no_nearest(1);

    step_physics(&mut beings, &[world], &cfg, 1, &[8], &nearest, &avail, &cost);

    let expected = 50.0 - cfg.fuel_per_tick * 0.33;
    assert!(
        (beings.fuel[0] - expected).abs() < 1e-6,
        "esc=8 (stay) must still burn fuel_per_tick*0.33 of idle fuel, not zero: \
         got {}, expected {expected}",
        beings.fuel[0]
    );
    assert_ne!(beings.fuel[0], 50.0, "idle marcha must cost fuel");
}

#[test]
fn a_dry_tank_triggers_a_tow_to_the_nearest_station() {
    let mut beings = one_being();
    beings.fuel[0] = 0.1; // below fuel_per_tick*cu*peso = 0.15*1.0*1.0 = 0.15
    beings.reserve[0] = 0.0; // kcal = min(0, 20) = 0 -> dn = 6 + (20-0) = 26
    let world = flat_world(200.0, 1.0);
    let cfg = Config::default();
    let (avail, cost) = no_services(1);
    let mut nearest = no_nearest(1);
    nearest.points[0] = [42.0, 7.0]; // posto (TIPOS index 0)

    step_physics(&mut beings, &[world], &cfg, 1, &[0], &nearest, &avail, &cost);

    assert_eq!(beings.pos_x[0], 42.0, "pane must teleport to the nearest station");
    assert_eq!(beings.pos_y[0], 7.0);
    assert_eq!(beings.fuel[0], 15.0, "FUEL_POS_PANE = 15");
    assert_eq!(beings.stopped_ticks[0], 18.0);
    assert_eq!(beings.breakdowns[0], 1.0);
    assert_eq!(beings.faults[0], 1.0);
    let expected_life = 100.0 - 26.0;
    assert!(
        (beings.life[0] - expected_life).abs() < 1e-4,
        "dn = D_PANE(6) + (KCAL_PANE(20) - kcal(0)) = 26: got life {}, expected {expected_life}",
        beings.life[0]
    );
}

#[test]
fn overheating_resets_heat_and_skips_the_driving_penalty_that_tick() {
    let mut beings = one_being();
    beings.heat[0] = 95.0; // + calor_sobe*cu*mov = 95.15 > CALOR_MAX(92)
    beings.fuel[0] = 50.0; // plenty, no pane
    let world = flat_world(200.0, 1.0);
    let cfg = Config::default();
    let (avail, cost) = no_services(1);
    let nearest = no_nearest(1);

    step_physics(&mut beings, &[world], &cfg, 1, &[0], &nearest, &avail, &cost);

    assert_eq!(beings.heat[0], 60.0, "overheat must reset heat to 60");
    // D_RODAR would apply only if mov && heat > 80 -- heat is 60 post-reset, so
    // only D_QUENTE (6.0) is charged, not D_QUENTE + D_RODAR (6.15).
    let expected_life = 100.0 - cfg.damage_hot;
    assert!(
        (beings.life[0] - expected_life).abs() < 1e-4,
        "only D_QUENTE should apply this tick: got life {}, expected {expected_life}",
        beings.life[0]
    );
}

#[test]
fn exhaustion_parks_the_taxi_for_twelve_ticks_and_halves_fatigue() {
    let mut beings = one_being();
    beings.fatigue[0] = 94.9; // + fad_passo*mov = 95.05 >= 95
    let world = flat_world(200.0, 1.0);
    let cfg = Config::default();
    let (avail, cost) = no_services(1);
    let nearest = no_nearest(1);

    step_physics(&mut beings, &[world], &cfg, 1, &[0], &nearest, &avail, &cost);

    assert_eq!(beings.stopped_ticks[0], 12.0);
    assert_eq!(beings.fatigue[0], 40.0, "exhaustion resets fatigue to 40, not half");
    assert_eq!(beings.breakdowns[0], 1.0);
}

#[test]
fn hunger_thirst_and_debt_damage_match_the_three_formulas() {
    let world = flat_world(200.0, 1.0);
    let cfg = Config::default();
    let (avail, cost) = no_services(1);
    let nearest = no_nearest(1);

    // Hunger, saturated at fome = 100.
    let mut hungry = one_being();
    hungry.hunger[0] = 100.0;
    step_physics(&mut hungry, &[world.clone()], &cfg, 1, &[8], &nearest, &avail, &cost);
    let expected = cfg.hunger_death_rate
        * ((100.0 - cfg.hunger_critical) / (100.0 - cfg.hunger_critical)).clamp(0.0, 1.0);
    assert!(
        (hungry.life[0] - (100.0 - expected)).abs() < 1e-5,
        "fome=100 must cost exactly hunger_death_rate: got {}, expected loss {expected}",
        hungry.life[0]
    );

    // Thirst, at the floor agua = 0.
    let mut thirsty = one_being();
    thirsty.water[0] = 0.0;
    step_physics(&mut thirsty, &[world.clone()], &cfg, 1, &[8], &nearest, &avail, &cost);
    let expected = cfg.thirst_death_rate
        * ((cfg.thirst_critical - 0.0) / cfg.thirst_critical).clamp(0.0, 1.0);
    assert!(
        (thirsty.life[0] - (100.0 - expected)).abs() < 1e-5,
        "agua=0 must cost exactly thirst_death_rate: got {}, expected loss {expected}",
        thirsty.life[0]
    );

    // Debt, exactly at the limit dinh = -60.
    let mut indebted = one_being();
    indebted.money[0] = -cfg.debt_limit;
    step_physics(&mut indebted, &[world.clone()], &cfg, 1, &[8], &nearest, &avail, &cost);
    let expected = 2.0f32; // 2.0 * clamp(60/60,0,1)^2 = 2.0
    assert!(
        (indebted.life[0] - (100.0 - expected)).abs() < 1e-5,
        "dinh=-60 must cost exactly 2.0 (the debt term saturated): got {}, expected loss {expected}",
        indebted.life[0]
    );
}

#[test]
fn rent_is_charged_once_per_simulated_day() {
    let mut beings = one_being();
    beings.money[0] = 1000.0; // plenty, never hits the debt clamp
    beings.hunger[0] = 10.0;
    beings.water[0] = 85.0;
    beings.life[0] = 100.0;
    let world = flat_world(200.0, 4.0); // district rent = 4.0
    let cfg = Config::default();
    let (avail, cost) = no_services(1);
    let nearest = no_nearest(1);

    let mut charges = 0;
    let mut total_delta = 0.0f32;
    for t in 1..=480i64 {
        let before = beings.money[0];
        // esc=8 (stay): never moves, so pane can never trigger, isolating rent as
        // the only channel that can move `money`.
        step_physics(&mut beings, &[world.clone()], &cfg, t, &[8], &nearest, &avail, &cost);
        let delta = beings.money[0] - before;
        if delta != 0.0 {
            charges += 1;
            total_delta += delta;
        }
    }

    assert!(
        beings.life[0] > 0.0,
        "the being must survive all 480 ticks for this count to be meaningful"
    );
    assert_eq!(charges, 2, "480 ticks / 240-tick day = exactly 2 midnights");
    let expected_total = -2.0 * cfg.rent * 4.0;
    assert!(
        (total_delta - expected_total).abs() < 1e-3,
        "got total rent {total_delta}, expected {expected_total}"
    );
}

#[test]
fn the_stopped_counter_is_decremented_before_being_overwritten() {
    // SPEC §13.8 (armadilha 8): `livre` is read from stopped_ticks BEFORE the
    // decrement, not after. stopped_ticks = 1 is the edge case that tells the two
    // apart: the OLD value (1, > 0) must keep the being NOT free this tick, even
    // though the decremented value (0) would say otherwise.
    let mut beings = one_being();
    beings.stopped_ticks[0] = 1.0;
    let start_x = beings.pos_x[0];
    let start_fuel = beings.fuel[0];
    let world = flat_world(200.0, 1.0);
    let cfg = Config::default();
    let (avail, cost) = no_services(1);
    let nearest = no_nearest(1);

    // esc=0: would move (and burn moving fuel) if the being were wrongly free.
    step_physics(&mut beings, &[world], &cfg, 1, &[0], &nearest, &avail, &cost);

    assert_eq!(beings.stopped_ticks[0], 0.0, "1 decrements to 0");
    assert_eq!(beings.pos_x[0], start_x, "not livre this tick -- must not move");
    assert_eq!(
        beings.fuel[0], start_fuel,
        "not livre -- not even marcha (idle) fuel applies, since marcha also requires livre"
    );
}

#[test]
fn reserve_is_only_spent_by_the_beings_that_actually_broke_down() {
    // Two beings sharing the tick: being 0 breaks down (pane), being 1 does not.
    // Passive decay (item 17) applies to both; the KCAL_PANE spend (item 20) must
    // apply to being 0 ONLY.
    let world = flat_world(200.0, 1.0);
    let cfg = Config::default();

    let mut beings = Beings {
        n: 2,
        pos_x: vec![100.0, 100.0],
        pos_y: vec![100.0, 100.0],
        fuel: vec![0.1, 50.0], // being 0 pans, being 1 has plenty
        water: vec![80.0, 80.0],
        hunger: vec![10.0, 10.0],
        fatigue: vec![10.0, 10.0],
        heat: vec![30.0, 30.0],
        life: vec![100.0, 100.0],
        money: vec![0.0, 0.0],
        reserve: vec![40.0, 40.0],
        stopped_ticks: vec![0.0, 0.0],
        metabolism: vec![1.0, 1.0],
        consumption: vec![0.30, 0.30],
        home_district: vec![0, 0],
        phase: vec![0.0, 0.0],
        carrying: vec![-1.0, -1.0],
        dest_x: vec![0.0, 0.0],
        dest_y: vec![0.0, 0.0],
        board_x: vec![0.0, 0.0],
        board_y: vec![0.0, 0.0],
        origin_district: vec![0, 0],
        since_pickup: vec![0.0, 0.0],
        px: vec![0.0; NPX * 2 * 2],
        pxd: vec![0.0; NPX * 2 * 2],
        px_ok: vec![0; NPX * 2],
        alive_ticks: vec![0.0, 0.0],
        deliveries: vec![0.0, 0.0],
        revenue: vec![0.0, 0.0],
        breakdowns: vec![0.0, 0.0],
        damage_total: vec![0.0, 0.0],
        refuels: vec![0.0, 0.0],
        liters: vec![0.0, 0.0],
        meals: vec![0.0, 0.0],
        repairs: vec![0.0, 0.0],
        drinks: vec![0.0, 0.0],
        faults: vec![0.0, 0.0],
        map_idx: vec![0, 0],
        genome_idx: vec![0, 0],
    };
    let (avail, cost) = no_services(2);
    let nearest = no_nearest(2);

    step_physics(&mut beings, &[world], &cfg, 1, &[0, 0], &nearest, &avail, &cost);

    let passive_only = 40.0 * (1.0 - cfg.reserve_decay_per_tick);
    assert!(
        (beings.reserve[1] - passive_only).abs() < 1e-6,
        "being 1 (no breakdown) must only lose passive decay: got {}, expected {passive_only}",
        beings.reserve[1]
    );
    let kcal = passive_only.min(cfg.breakdown_kcal);
    let expected_0 = passive_only - kcal;
    assert!(
        (beings.reserve[0] - expected_0).abs() < 1e-6,
        "being 0 (broke down) must ALSO lose the KCAL_PANE spend: got {}, expected {expected_0}",
        beings.reserve[0]
    );
    assert!(
        beings.reserve[0] < beings.reserve[1],
        "the panned being must have spent more reserve"
    );
}
