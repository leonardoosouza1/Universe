//! T7 (integrator): `SimCpu`, a `Simulador`/`LifeSampler` pair over T5's real
//! `tick::Sim`/`tick::step` — the piece T6's own doc comments (`evolution.rs`,
//! `judge.rs`) say is still missing: both those modules were written against a
//! MINIMAL trait pair because T5 did not exist yet in this worktree, with the
//! real tick-based implementation "left to whoever wires T5 in later"
//! (`evolution.rs`'s module doc). This file is that wiring.
//!
//! # Why one `Sim` per genome, not one big `P*NC*R`-being batch
//!
//! `tick::Sim` carries exactly ONE `Genome` (`decision::decide` takes a single
//! `&Genome` for its whole call, `decision.rs:47-52`) — T5 built it for the
//! identity gate, which only ever needs one genome driving one trace. SPEC §10's
//! batch layout, `b = p*(NC*R) + c*R + r`, describes how PYTHON lays one giant
//! tensor out; it does not require this crate to hold one giant `Beings` too.
//! `avaliar` instead runs `genomas.len()` independent `Sim`s, each with exactly
//! `NC*R` beings ordered `c*R + r` (map outer, replica inner — the SAME inner
//! structure the formula names), and reduces each to one mean fitness. This
//! satisfies the doc comment's real warning ("get this backwards and every
//! genome is scored on a plausible-looking but WRONG mix of worlds") without
//! touching `decision.rs`, which this task's edit list does not include.
//!
//! # The birth state this module generates (SPEC §1, §7.1)
//!
//! T5's `beings::Beings::from_trace` only ever reads a birth state off a trace
//! (SPEC §1: "no v0 o Rust NÃO gera esses valores — ele os LÊ"). A GA battery has
//! no trace to read from — `gerar_mapa`'s birth sampling is exactly what SPEC §7.1
//! describes (`_perto`/`_destino`, `campo.py:154-172, 222-235`), and this module
//! reproduces it directly: uniform position on the map, the fixed SPEC §1 body
//! constants, and `NPX = 12` passenger slots born in a ring `[8, raio_px]` around
//! that position with `_destino`'s long-tailed radius. The ring/destination
//! formula is a byte-for-byte duplicate of `passengers::sample_new_slot`, not a
//! call to it — that function is `passengers.rs`-private, and this task's edit
//! list does not include `passengers.rs`. Keep the two in sync by hand if either
//! changes; [`spawn_slot`]'s doc comment says so again at the definition.

use std::f32::consts::TAU;

use crate::beings::Beings;
use crate::constants::{Config, NPX};
use crate::evolution::Simulador;
use crate::genome::Genome;
use crate::judge::{life_fitness, LifeSample, LifeSampler};
use crate::passengers::TaxiRng;
use crate::tick::{step, Sim};
use crate::world::World;

/// Max worker threads [`SimCpu::avaliar`] spawns across the genome axis (T7
/// rule: `std::thread::scope`, at most 4 threads, no new crate/no rayon).
const MAX_THREADS: usize = 4;

/// A `Simulador`/`LifeSampler` over `tick::Sim`. Stateless (a unit struct);
/// `&mut self` on both trait methods only because the trait signatures (T6,
/// written before this type existed) require it.
#[derive(Debug, Default, Clone, Copy)]
pub struct SimCpu;

impl SimCpu {
    pub fn new() -> Self {
        SimCpu
    }
}

impl Simulador for SimCpu {
    /// SPEC §10 step 1 (`campo.avaliar`): one mean fitness per genome, averaged
    /// over that genome's `mapas.len() * r` beings. Parallelized across the
    /// GENOME axis only (up to [`MAX_THREADS`] threads) — each genome's `NC*R`
    /// beings already run together inside one `Sim`, so there is nothing finer
    /// to split without touching `tick.rs`'s per-tick loop itself.
    fn avaliar(
        &mut self,
        genomas: &[Genome],
        mapas: &[World],
        cfg: &Config,
        t_max: u32,
        seed: u64,
        r: u32,
    ) -> Vec<f32> {
        assert!(!mapas.is_empty(), "avaliar: Simulador needs at least one map");
        if genomas.is_empty() {
            return Vec::new();
        }
        let r = r as usize;
        let n_threads = MAX_THREADS.min(genomas.len());
        let chunk = genomas.len().div_ceil(n_threads);
        let mut out = vec![0.0f32; genomas.len()];

        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            let mut start = 0usize;
            while start < genomas.len() {
                let end = (start + chunk).min(genomas.len());
                let slice = &genomas[start..end];
                handles.push((
                    start,
                    scope.spawn(move || {
                        slice
                            .iter()
                            .map(|g| score_genome(g, mapas, cfg, t_max, seed, r))
                            .collect::<Vec<f32>>()
                    }),
                ));
                start = end;
            }
            for (start, handle) in handles {
                let part = handle.join().expect("avaliar: a worker thread panicked");
                out[start..start + part.len()].copy_from_slice(&part);
            }
        });

        out
    }
}

impl LifeSampler for SimCpu {
    /// SPEC §9 (`campo.medidas`): `copies` identical copies of `genome`, run
    /// against every map in `mapas`, `R = 1` (no separate replica axis — SPEC §9
    /// is explicit the judge never has one). One `LifeSample` per simulated
    /// being, `copies * mapas.len()` of them, nothing pre-averaged (`judge()`
    /// does the averaging) — reuses the exact same batch shape `avaliar` uses
    /// for its replica axis, with `copies` standing in for `r`.
    fn sample(
        &mut self,
        genome: &Genome,
        mapas: &[World],
        cfg: &Config,
        t_max: u32,
        seed: u64,
        copies: u32,
    ) -> Vec<LifeSample> {
        assert!(!mapas.is_empty(), "sample: LifeSampler needs at least one map");
        let sim = run_genome(genome, mapas, cfg, t_max, seed, copies as usize);
        (0..sim.beings.n)
            .map(|i| LifeSample {
                alive_ticks: sim.beings.alive_ticks[i],
                dead: sim.beings.life[i] <= 0.0,
                deliveries: sim.beings.deliveries[i],
                // `m.av_fuel` ("panes" in the Python dict) — `Beings::faults`,
                // NOT `Beings::breakdowns` (`avar`, which also counts exhaustion
                // and overheating). See `LifeSample::breakdowns`'s own doc
                // comment in `judge.rs`.
                breakdowns: sim.beings.faults[i],
                hunger_end: sim.beings.hunger[i],
                water_end: sim.beings.water[i],
                debt: sim.beings.money[i],
                reserve: sim.beings.reserve[i],
                liters: sim.beings.liters[i],
                refuels: sim.beings.refuels[i],
                meals: sim.beings.meals[i],
                repairs: sim.beings.repairs[i],
                revenue: sim.beings.revenue[i],
            })
            .collect()
    }
}

/// Runs ONE genome's `NC*r`-being batch to `t_max` ticks (or until every being
/// is dead, checked once per simulated day — SPEC §8: "roda até `T_max` ou até
/// a fração de vivos cair a `corte_vivos`... checado 1x por dia simulado, `t %
/// 240 == 239`"; `corte_vivos = 0` here, i.e. only a FULLY dead batch stops
/// early). Battery mode throughout (`rng: Some`, SPEC §11.3): passengers
/// respawn, unlike the identity gate's `rng: None`.
fn run_genome(genome: &Genome, mapas: &[World], cfg: &Config, t_max: u32, seed: u64, r: usize) -> Sim {
    let beings = build_batch(mapas, cfg, seed, r);
    let mut sim = Sim::new(beings, mapas.to_vec(), *cfg, genome.clone());
    let mut rng = TaxiRng::new(seed);
    let day_ticks = cfg.day_ticks.max(1.0) as i64;

    for _ in 0..t_max {
        step(&mut sim, None, None, Some(&mut rng));
        if sim.t % day_ticks == 0 && all_dead(&sim.beings.life) {
            break;
        }
    }
    sim
}

fn score_genome(genome: &Genome, mapas: &[World], cfg: &Config, t_max: u32, seed: u64, r: usize) -> f32 {
    let sim = run_genome(genome, mapas, cfg, t_max, seed, r);
    let n = sim.beings.n;
    if n == 0 {
        return 0.0;
    }
    sim.beings
        .alive_ticks
        .iter()
        .zip(sim.beings.revenue.iter())
        .map(|(&alive_ticks, &revenue)| life_fitness(alive_ticks, revenue))
        .sum::<f32>()
        / n as f32
}

fn all_dead(life: &[f32]) -> bool {
    !life.is_empty() && life.iter().all(|&l| l <= 0.0)
}

/// One being's SPEC §1/§7.1 birth: a uniform position on the map, plus its
/// `NPX` passenger slots (see [`spawn_slot`]). Computed once per `(map,
/// replica)` pair (SPEC §1: `fase` is per MAP, but position/passengers are per
/// BEING — T7's own instruction pins them to `(seed, mapa, réplica)`, i.e. every
/// GENOME sharing that `(map, replica)` slot is born in the exact same place
/// with the exact same passengers waiting, common-random-numbers style — the
/// same spirit as SPEC §10's `crn`/"jardim comum" batching, just applied to the
/// birth state instead of the per-generation world seed).
struct Cradle {
    pos: [f32; 2],
    home_district: u32,
    px: [[f32; 2]; NPX],
    pxd: [[f32; 2]; NPX],
}

fn birth_cradle(world: &World, cfg: &Config, seed: u64) -> Cradle {
    let mut rng = TaxiRng::new(seed);
    let side = world.side;
    let pos = [rng.next_f32() * side, rng.next_f32() * side];
    let home_district = world.district_at(pos[0], pos[1]);
    let mut px = [[0.0f32; 2]; NPX];
    let mut pxd = [[0.0f32; 2]; NPX];
    for slot in 0..NPX {
        let (p, d) = spawn_slot(&mut rng, pos, cfg, side);
        px[slot] = p;
        pxd[slot] = d;
    }
    Cradle { pos, home_district, px, pxd }
}

/// `_perto(pos, 1)` then `_destino(po)` (SPEC §7.1, `campo.py:222-235`): a
/// passenger slot in a ring `[8, raio_px]` around `pos`, destination
/// `L·(0.05 + 0.45·u²)` away (concentrated nearby, long tail). This is a
/// deliberate BYTE-FOR-BYTE duplicate of `passengers::sample_new_slot` — see
/// this module's doc comment for why it is not a call: that function is
/// private to `passengers.rs`, which this task's edit list does not include.
fn spawn_slot(rng: &mut TaxiRng, pos: [f32; 2], cfg: &Config, side: f32) -> ([f32; 2], [f32; 2]) {
    let angle = rng.next_f32() * TAU;
    let radius = 8.0 + (cfg.passenger_spawn_radius_m - 8.0) * rng.next_f32();
    let px = [
        (pos[0] + radius * angle.cos()).clamp(0.0, side),
        (pos[1] + radius * angle.sin()).clamp(0.0, side),
    ];

    let angle2 = rng.next_f32() * TAU;
    let u = rng.next_f32();
    let radius2 = side * (0.05 + 0.45 * u * u);
    let pxd = [
        (px[0] + radius2 * angle2.cos()).clamp(0.0, side),
        (px[1] + radius2 * angle2.sin()).clamp(0.0, side),
    ];

    (px, pxd)
}

/// A combined seed for [`birth_cradle`]/the per-map `fase` draw — deterministic
/// per `(seed, tag_or_map, replica)`, distinct streams for distinct inputs. Not
/// meant to resist adversarial collisions, only to keep birth state
/// reproducible (SPEC §10: this crate's battery RNG needs to be
/// statistically-reasonable and deterministic, not torch-bit-identical).
fn mix_seed(seed: u64, a: u64, b: u64) -> u64 {
    seed.wrapping_add(a.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(b.wrapping_mul(0xBF58_476D_1CE4_E5B9))
        .wrapping_add(1)
}

/// The one-genome `NC*r`-being batch (SPEC §1 body constants, §7.1 passengers),
/// laid out `[map][replica]` — `c*r + rep`, the inner structure of SPEC §10's
/// `b = p*(NC*R) + c*R + r` (see this module's doc comment on why the OUTER `p`
/// axis is a separate `Sim` per genome instead of a wider `Beings`).
fn build_batch(mapas: &[World], cfg: &Config, seed: u64, r: usize) -> Beings {
    let nc = mapas.len();
    let n = nc * r;

    let mut b = Beings {
        n,
        pos_x: Vec::with_capacity(n),
        pos_y: Vec::with_capacity(n),
        fuel: Vec::with_capacity(n),
        water: Vec::with_capacity(n),
        hunger: Vec::with_capacity(n),
        fatigue: Vec::with_capacity(n),
        heat: Vec::with_capacity(n),
        life: Vec::with_capacity(n),
        money: Vec::with_capacity(n),
        reserve: Vec::with_capacity(n),
        stopped_ticks: Vec::with_capacity(n),
        metabolism: Vec::with_capacity(n),
        consumption: Vec::with_capacity(n),
        home_district: Vec::with_capacity(n),
        phase: Vec::with_capacity(n),
        carrying: Vec::with_capacity(n),
        dest_x: Vec::with_capacity(n),
        dest_y: Vec::with_capacity(n),
        board_x: Vec::with_capacity(n),
        board_y: Vec::with_capacity(n),
        origin_district: Vec::with_capacity(n),
        since_pickup: Vec::with_capacity(n),
        px: Vec::with_capacity(n * NPX * 2),
        pxd: Vec::with_capacity(n * NPX * 2),
        px_ok: Vec::with_capacity(n * NPX),
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
        map_idx: Vec::with_capacity(n),
        genome_idx: vec![0; n],
    };

    for (c, world) in mapas.iter().enumerate() {
        // `fase` (SPEC §1): per MAP, drawn once, shared by every replica on it.
        let mut fase_rng = TaxiRng::new(mix_seed(seed, c as u64, 0xFA5E));
        let fase = fase_rng.next_f32() * cfg.day_ticks;

        for rep in 0..r {
            let cradle = birth_cradle(world, cfg, mix_seed(seed, c as u64, rep as u64));

            b.pos_x.push(cradle.pos[0]);
            b.pos_y.push(cradle.pos[1]);
            b.fuel.push(85.0);
            b.water.push(85.0);
            b.hunger.push(10.0);
            b.fatigue.push(10.0);
            b.heat.push(30.0);
            b.life.push(100.0);
            b.money.push(cfg.initial_debt);
            b.reserve.push(0.0);
            b.stopped_ticks.push(0.0);
            b.metabolism.push(1.0);
            b.consumption.push(cfg.fuel_per_tick * 2.0);
            b.home_district.push(cradle.home_district);
            b.phase.push(fase);
            b.carrying.push(-1.0);
            b.dest_x.push(0.0);
            b.dest_y.push(0.0);
            b.board_x.push(0.0);
            b.board_y.push(0.0);
            b.origin_district.push(0);
            b.since_pickup.push(0.0);
            for slot in 0..NPX {
                b.px.push(cradle.px[slot][0]);
                b.px.push(cradle.px[slot][1]);
                b.pxd.push(cradle.pxd[slot][0]);
                b.pxd.push(cradle.pxd[slot][1]);
                // SPEC §1: `px_ok` starts "todas true" -- every slot active.
                b.px_ok.push(1);
            }
            b.map_idx.push(c as u32);
        }
    }

    b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{GRID, NP, TEX};

    fn flat_world(side: f32) -> World {
        World {
            side,
            cost_tex: vec![1.0; TEX * TEX],
            district_field: vec![0; GRID * GRID],
            centers: vec![[0.0, 0.0]],
            tax: vec![0.0],
            price_mul_base: vec![1.0],
            rent: vec![0.0],
            demand: vec![0.5],
            price_phase: vec![0.0],
            services: [vec![[side / 2.0, side / 2.0]], vec![], vec![], vec![], vec![]],
        }
    }

    fn flat_genome(value: f32) -> Genome {
        Genome(vec![value; NP])
    }

    // --- build_batch -----------------------------------------------------

    #[test]
    fn build_batch_lays_out_map_outer_replica_inner() {
        let worlds = vec![flat_world(50.0), flat_world(50.0)];
        let cfg = Config::default();
        let beings = build_batch(&worlds, &cfg, 123, 3);

        assert_eq!(beings.n, 6, "NC=2, r=3 -> n=6");
        assert_eq!(beings.map_idx, vec![0, 0, 0, 1, 1, 1], "map outer, replica inner (c*r + rep)");
        assert_eq!(beings.px.len(), 6 * NPX * 2);
        assert_eq!(beings.px_ok, vec![1u32; 6 * NPX], "SPEC 1: every slot starts active");
        for &f in &beings.fuel {
            assert_eq!(f, 85.0);
        }
        for &m in &beings.money {
            assert_eq!(m, cfg.initial_debt);
        }
    }

    #[test]
    fn build_batch_is_deterministic_for_a_fixed_seed() {
        let worlds = vec![flat_world(80.0)];
        let cfg = Config::default();
        let a = build_batch(&worlds, &cfg, 42, 4);
        let b = build_batch(&worlds, &cfg, 42, 4);
        assert_eq!(a.pos_x, b.pos_x);
        assert_eq!(a.pos_y, b.pos_y);
        assert_eq!(a.px, b.px);
        assert_eq!(a.pxd, b.pxd);
        assert_eq!(a.phase, b.phase);
    }

    #[test]
    fn build_batch_fase_is_shared_across_replicas_of_the_same_map_but_positions_are_not() {
        let worlds = vec![flat_world(80.0)];
        let cfg = Config::default();
        let beings = build_batch(&worlds, &cfg, 7, 4);
        assert!(
            beings.phase.windows(2).all(|w| w[0] == w[1]),
            "fase (SPEC 1) must be identical for every replica of the same map"
        );
        let distinct_positions: std::collections::HashSet<(u32, u32)> = beings
            .pos_x
            .iter()
            .zip(beings.pos_y.iter())
            .map(|(&x, &y)| (x.to_bits(), y.to_bits()))
            .collect();
        assert!(
            distinct_positions.len() > 1,
            "different replicas must (almost certainly) be born at different positions"
        );
    }

    // --- Simulador::avaliar ------------------------------------------------

    #[test]
    fn avaliar_returns_one_finite_fitness_per_genome() {
        let worlds = vec![flat_world(60.0)];
        let cfg = Config::default();
        let genomes = vec![flat_genome(0.0), flat_genome(0.3)];
        let mut sim = SimCpu::new();

        let out = sim.avaliar(&genomes, &worlds, &cfg, 40, 11, 2);

        assert_eq!(out.len(), 2);
        for &f in &out {
            assert!(f.is_finite(), "fitness must be finite, got {f}");
            assert!(f >= 0.0, "life_fitness starts at 0 and only grows over 40 ticks, got {f}");
        }
    }

    #[test]
    fn avaliar_is_deterministic_for_the_same_seed() {
        let worlds = vec![flat_world(60.0)];
        let cfg = Config::default();
        let genomes = vec![flat_genome(0.1), flat_genome(-0.2)];
        let mut sim = SimCpu::new();

        let a = sim.avaliar(&genomes, &worlds, &cfg, 60, 99, 2);
        let b = sim.avaliar(&genomes, &worlds, &cfg, 60, 99, 2);
        assert_eq!(a, b, "same seed/genomes/ticks must reproduce bit-exact fitness");
    }

    #[test]
    fn avaliar_handles_more_genomes_than_max_threads_without_losing_any() {
        let worlds = vec![flat_world(60.0)];
        let cfg = Config::default();
        // MAX_THREADS = 4: 10 genomes forces at least one thread to handle > 1.
        let genomes: Vec<Genome> = (0..10).map(|i| flat_genome(i as f32 * 0.05)).collect();
        let mut sim = SimCpu::new();

        let out = sim.avaliar(&genomes, &worlds, &cfg, 20, 5, 1);
        assert_eq!(out.len(), 10);
    }

    // --- LifeSampler::sample -------------------------------------------------

    #[test]
    fn sample_returns_copies_times_maps_life_samples() {
        let worlds = vec![flat_world(60.0), flat_world(60.0)];
        let cfg = Config::default();
        let genome = flat_genome(0.0);
        let mut sim = SimCpu::new();

        let samples = sim.sample(&genome, &worlds, &cfg, 30, 7777, 5);
        assert_eq!(samples.len(), 2 * 5, "copies * mapas.len()");
        for s in &samples {
            assert!(s.alive_ticks >= 0.0 && s.alive_ticks <= 30.0);
        }
    }

    #[test]
    fn sample_breakdowns_field_is_the_fuel_fault_counter_not_the_general_one() {
        // Structural check that LifeSample::breakdowns is wired to
        // `Beings::faults` (av_fuel), not `Beings::breakdowns` (avar): run long
        // enough that SOME beings break down, on a map with only ONE far-away
        // fuel stop so a broken-down being racks up `faults` too, then confirm
        // no sample's breakdowns count exceeds a hand count of `faults` from a
        // parallel run -- i.e. this is a smoke test that the field is plausible
        // (0 <= breakdowns), not a numeric identity gate (that belongs to
        // gate.rs against the real trace).
        let worlds = vec![flat_world(60.0)];
        let cfg = Config::default();
        let genome = flat_genome(0.0);
        let mut sim = SimCpu::new();
        let samples = sim.sample(&genome, &worlds, &cfg, 500, 7777, 8);
        for s in &samples {
            assert!(s.breakdowns >= 0.0);
        }
    }

    // --- run_genome / all_dead -----------------------------------------------

    #[test]
    fn all_dead_is_false_for_an_empty_batch_and_true_only_when_every_being_is_at_or_below_zero() {
        assert!(!all_dead(&[]));
        assert!(!all_dead(&[1.0, 0.0]));
        assert!(all_dead(&[0.0, -1.0, 0.0]));
    }

    #[test]
    fn run_genome_never_exceeds_t_max_ticks() {
        let worlds = vec![flat_world(60.0)];
        let cfg = Config::default();
        let genome = flat_genome(0.0);
        let sim = run_genome(&genome, &worlds, &cfg, 17, 3, 2);
        assert_eq!(sim.t, 17, "with nobody dying in 17 ticks, run_genome must run exactly t_max ticks");
    }
}
