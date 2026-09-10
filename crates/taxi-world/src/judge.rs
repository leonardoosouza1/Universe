//! The "vida" fitness ruler and the judge (SPEC §8 "Morte, t_vida, réguas" and §9
//! "Juiz", `campo.py:583-591` `_fit`'s `'vida'` branch and `campo.py:612-625`
//! `medidas`) — plan §T6 "judge.rs". See `docs/plano-mundo-taxi.md` §T6.
//!
//! # Why `judge` needs a second trait, not just `evolution::Simulador`
//!
//! `Simulador::avaliar` (`evolution.rs`) only returns ONE mean fitness per genome —
//! exactly what the GA loop needs, and exactly what `campo.avaliar` returns. The
//! judge (`campo.medidas`, `campo.py:612-625`) is a DIFFERENT Python function: it
//! runs 128 identical copies of ONE genome and reports a whole dict of per-being
//! fields (deaths, deliveries, hunger/water/debt at the end, breakdown counts,
//! refuel counts, the `t_vida` distribution...) that a bare fitness scalar cannot
//! reconstruct. So this module declares its OWN minimal trait, [`LifeSampler`],
//! for the one thing `judge` needs beyond `Simulador`: raw per-being outcomes for a
//! single genome. A real integration (T5) implements BOTH traits on the same
//! concrete simulator type. This is the same kind of divergence from the plan's
//! literal `judge(g, worlds, cfg, ticks) -> Measures` signature as `evolve`'s —
//! see this crate's final report.
//!
//! # `dof` and `metab` need no simulation at all
//!
//! `dof_de` (`campo.py:565-570`) counts genes "beyond the spine" (`Wb.shape[1] >
//! Linear.NP`); this crate's [`Genome`] is always exactly `NP` weights with no
//! extra per-architecture genes, so `dof_de` always takes the
//! `Wb.shape[1] <= Linear.NP` branch and returns `0`. `metab` (`campo.py:540`) is
//! only ever touched when `cfg.gene_cost > 0.`, and with `dof` structurally `0` the
//! product is `0` regardless — `metab` stays `1.0`. Both are therefore pure,
//! simulation-free constants here (`DOF_V0`, `METABOLISM_V0`), not fields
//! [`LifeSample`] carries.

use std::collections::BTreeMap;

use serde_json::{json, Value};

use crate::constants::Config;
use crate::evolution::{Champion, GaParams};
use crate::genome::Genome;
use crate::world::World;

/// SPEC §9: "128 cópias do MESMO genoma, semente do mundo 7777, R = 1".
pub const JUDGE_SEED: u64 = 7777;
/// SPEC §9: "128 cópias".
pub const JUDGE_COPIES: u32 = 128;
/// `dof_de` on this crate's fixed-length `Genome` (SPEC §9, see this module's doc
/// comment): always 0, there is no "beyond the spine" gene dimension in v0.
pub const DOF_V0: f32 = 0.0;
/// `metab` with `dof == 0` (see this module's doc comment): always 1.0 in v0.
pub const METABOLISM_V0: f32 = 1.0;
/// `w_trab` (SPEC §8's "vida" formula). `Config` (T1, `constants.rs`) deliberately
/// leaves `fitness`/`w_trab` out — plan §T2's own note: "belong to the fitness
/// formula in T6's judge.rs" — so this is this crate's one place that owns it.
pub const W_TRAB: f32 = 1.0;

/// The "vida" ruler (SPEC §8): `fit = t_vida/240 + w_trab * rec/300`. The 240 and
/// 300 are literal SPEC constants, not `cfg.day_ticks` — `campo.py:591` writes them
/// as bare numbers, not `cfg`-derived. This is the ONLY fitness formula this crate
/// implements ("Só a régua vida", plan §T6); a real `Simulador`/`LifeSampler`
/// wired to T5's `tick::step` should use this SAME function so `evolve`'s and
/// `judge`'s notion of "fit" never drifts apart.
pub fn life_fitness(alive_ticks: f32, revenue: f32) -> f32 {
    alive_ticks / 240.0 + W_TRAB * revenue / 300.0
}

/// One being's raw outcome at the end of a judge run (SPEC §9, `campo.py:612-621`
/// `medidas`'s per-`m` fields, before they are averaged across the batch).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LifeSample {
    /// `m.t_vida` — ticks alive out of `t_max` (SPEC §8: `t_vida += vivo` each tick).
    pub alive_ticks: f32,
    /// `m.vida <= 0`.
    pub dead: bool,
    /// `m.ent` — an ARRIVAL counter (`campo.py:160,457`: `s.ent += cheg.float()`
    /// on every completed drop-off), i.e. **deliveries**, not entropy.
    pub deliveries: f32,
    /// `m.av_fuel` — breakdown count (`panes` in the Python dict).
    pub breakdowns: f32,
    /// `m.fome` at the end of the run.
    pub hunger_end: f32,
    /// `m.agua` at the end of the run.
    pub water_end: f32,
    /// `m.dinh` at the end of the run.
    pub debt: f32,
    /// `m.res` at the end of the run.
    pub reserve: f32,
    /// `m.litros` — total litres bought over the run.
    pub liters: f32,
    /// `m.n_posto` — refuel-stop count.
    pub refuels: f32,
    /// `m.n_come` — meal count.
    pub meals: f32,
    /// `m.n_oficina` — repair count.
    pub repairs: f32,
    /// `m.rec` — fare revenue collected.
    pub revenue: f32,
}

/// Runs `copies` identical copies of ONE genome and returns one [`LifeSample`] per
/// simulated being (`copies * mapas.len()` of them — SPEC §9's `R = 1`, so there is
/// no separate replica axis here). A real implementation is `campo.medidas`'s Rust
/// twin: build `copies` copies of `genome`, run them against every map in `mapas`
/// for `t_max` ticks seeded from `seed`, and report each being's raw outcome
/// (nothing pre-averaged — [`judge`] does the averaging).
pub trait LifeSampler {
    fn sample(
        &mut self,
        genome: &Genome,
        mapas: &[World],
        cfg: &Config,
        t_max: u32,
        seed: u64,
        copies: u32,
    ) -> Vec<LifeSample>;
}

/// `t_vida` (SPEC §9's report block): mean/median/p10/p90 in SIMULATED DAYS
/// (`t_vida/240`, a literal constant — see [`life_fitness`]'s doc comment), plus
/// `ceiling_frac`: the fraction of beings that hit the horizon (`t_vida >= T`).
/// **`ceiling_frac > 0.05` invalidates the result** (SPEC §9): it means the
/// horizon, not the being, capped the measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LifeStats {
    pub mean: f32,
    pub median: f32,
    pub p10: f32,
    pub p90: f32,
    pub ceiling_frac: f32,
}

/// The judge's report for one genome (SPEC §9, `campo.medidas`'s dict, restricted
/// to the fields the plan/SPEC list as "relevantes ao v0"). See
/// [`measures_to_json`] for the Python-JSON-schema mapping.
#[derive(Debug, Clone, PartialEq)]
pub struct Measures {
    pub fit: f32,
    pub deaths: f32,
    pub deliveries: f32,
    pub breakdowns: f32,
    pub hunger_end: f32,
    pub water_end: f32,
    pub debt: f32,
    pub reserve: f32,
    pub liters_per_refuel: f32,
    pub refuels: f32,
    pub meals: f32,
    pub repairs: f32,
    pub dof: f32,
    pub metabolism: f32,
    pub revenue: f32,
    pub life: LifeStats,
}

/// SPEC §9: 128 copies of `genome`, world seed `JUDGE_SEED = 7777`, `R = 1`, no
/// jardim comum. **Never shares the evolution seed** (`777000 + 13*it`) — the judge
/// always runs on `JUDGE_SEED`, independent of which generation produced `genome`.
pub fn judge(
    sampler: &mut dyn LifeSampler,
    genome: &Genome,
    worlds: &[World],
    cfg: &Config,
    ticks: usize,
) -> Measures {
    let samples = sampler.sample(genome, worlds, cfg, ticks as u32, JUDGE_SEED, JUDGE_COPIES);
    assert!(!samples.is_empty(), "judge: LifeSampler::sample returned zero beings");
    let n = samples.len() as f32;

    let mean_f = |get: fn(&LifeSample) -> f32| samples.iter().map(get).sum::<f32>() / n;
    let mean_bool = |get: fn(&LifeSample) -> bool| {
        samples.iter().filter(|s| get(s)).count() as f32 / n
    };

    let fit = samples
        .iter()
        .map(|s| life_fitness(s.alive_ticks, s.revenue))
        .sum::<f32>()
        / n;

    // `(m.litros/m.n_posto.clamp(min=1)).mean()` (campo.py:617): a PER-BEING ratio,
    // averaged — not mean(litros)/mean(n_posto).
    let liters_per_refuel = samples
        .iter()
        .map(|s| s.liters / s.refuels.max(1.0))
        .sum::<f32>()
        / n;

    let life = life_stats(&samples, ticks as f32);

    Measures {
        fit,
        deaths: mean_bool(|s| s.dead),
        deliveries: mean_f(|s| s.deliveries),
        breakdowns: mean_f(|s| s.breakdowns),
        hunger_end: mean_f(|s| s.hunger_end),
        water_end: mean_f(|s| s.water_end),
        debt: mean_f(|s| s.debt),
        reserve: mean_f(|s| s.reserve),
        liters_per_refuel,
        refuels: mean_f(|s| s.refuels),
        meals: mean_f(|s| s.meals),
        repairs: mean_f(|s| s.repairs),
        dof: DOF_V0,
        metabolism: METABOLISM_V0,
        revenue: mean_f(|s| s.revenue),
        life,
    }
}

fn life_stats(samples: &[LifeSample], t_max: f32) -> LifeStats {
    let n = samples.len() as f32;
    let mut days: Vec<f32> = samples.iter().map(|s| s.alive_ticks / 240.0).collect();
    let mean = days.iter().sum::<f32>() / n;
    days.sort_by(|a, b| a.partial_cmp(b).expect("alive_ticks must not be NaN"));
    let median = quantile(&days, 0.5);
    let p10 = quantile(&days, 0.10);
    let p90 = quantile(&days, 0.90);
    let ceiling_frac = samples.iter().filter(|s| s.alive_ticks >= t_max).count() as f32 / n;
    LifeStats { mean, median, p10, p90, ceiling_frac }
}

/// Linear-interpolated quantile (matches `torch.quantile`'s default `linear`
/// method) over an ALREADY ASCENDING-SORTED slice.
fn quantile(sorted_ascending: &[f32], q: f32) -> f32 {
    let n = sorted_ascending.len();
    if n == 1 {
        return sorted_ascending[0];
    }
    let idx = q * (n as f32 - 1.0);
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        return sorted_ascending[lo];
    }
    let frac = idx - lo as f32;
    sorted_ascending[lo] + (sorted_ascending[hi] - sorted_ascending[lo]) * frac
}

fn round3(x: f32) -> f32 {
    (x * 1e3).round() / 1e3
}

fn round5(x: f32) -> f32 {
    (x * 1e5).round() / 1e5
}

/// Builds the `res['bracos'][rotulo]` entry `e43_campo.py:73-97` writes, so
/// `scripts/analise_campo.py` (which only reads `d['bracos']`) can read a Rust
/// judge run with NO changes on the Python side. `measures` and `champions` are
/// keyed by evolution seed exactly like `evolve`'s return value — pass `judge`'s
/// output per seed's champion straight through.
///
/// # What this function does NOT emit
///
/// `e43_campo.py`'s `med` dict also carries pure BATTERY-HARNESS bookkeeping that
/// has no equivalent in this crate's `evolve`/`judge` (`segundos`, `otim`,
/// `ativ_centrada`, `jardim_comum`, `corte_vivos`, `obs_rapido`, `sigma0`,
/// `sigma_espinha`, `gene_min`, `ser`), and the whole OUTER `res` object
/// (`nomes_f`, `cenario`, `cenario_hash`, `dispositivo`, `argv`, ..., `bracos`)
/// that wraps one-or-more `med` entries. Both belong to a future `taxi_evolve`
/// binary (plan §T6 lists `src/bin/taxi_evolve.rs`; this task's own file list does
/// NOT include a `bin/`, so it is out of scope here) — see this crate's final
/// report.
pub fn measures_to_json(
    measures: &BTreeMap<u64, Measures>,
    champions: &BTreeMap<u64, Champion>,
    p: &GaParams,
) -> Value {
    let seeds = &p.seeds;
    let n = seeds.len() as f32;
    assert!(n > 0.0, "measures_to_json: GaParams::seeds must not be empty");

    let get = |sd: u64| measures.get(&sd).expect("measures_to_json: missing seed in `measures`");

    let mut obj = serde_json::Map::new();

    // campo.medidas()'s scalar fields (campo.py:615-616), averaged across
    // evolution seeds (e43_campo.py:90-92) with a `_por_semente` sibling
    // (e43_campo.py:94-95).
    let scalars: [(&str, fn(&Measures) -> f32); 15] = [
        ("fit", |m| m.fit),
        ("mortos", |m| m.deaths),
        ("ent", |m| m.deliveries),
        ("panes", |m| m.breakdowns),
        ("fome_fim", |m| m.hunger_end),
        ("agua_fim", |m| m.water_end),
        ("dinh", |m| m.debt),
        ("res", |m| m.reserve),
        ("litros_por_abast", |m| m.liters_per_refuel),
        ("n_posto", |m| m.refuels),
        ("n_come", |m| m.meals),
        ("oficina", |m| m.repairs),
        ("dof", |m| m.dof),
        ("metab", |m| m.metabolism),
        ("rec", |m| m.revenue),
    ];
    for (key, extract) in scalars {
        let per_seed: Vec<f32> = seeds.iter().map(|&sd| round3(extract(get(sd)))).collect();
        let avg = round3(per_seed.iter().sum::<f32>() / n);
        obj.insert(key.to_string(), json!(avg));
        obj.insert(format!("{key}_por_semente"), json!(per_seed));
    }

    // t_vida: a nested dict AVERAGED ACROSS SEEDS (campo.py:619-620, then
    // e43_campo.py's dict-branch averaging, NOT one entry per seed), plus the
    // flattened scalar mirrors campo.py:621 also writes (note `frac_teto` has NO
    // `t_vida_` prefix on the flattened key, matching campo.py exactly).
    let life_field = |sd: u64, get_life: fn(&LifeStats) -> f32| get_life(&get(sd).life);
    let mean_life = |get_life: fn(&LifeStats) -> f32| -> f32 {
        seeds.iter().map(|&sd| life_field(sd, get_life)).sum::<f32>() / n
    };
    let life_entries: [(&str, fn(&LifeStats) -> f32); 5] = [
        ("media", |l| l.mean),
        ("mediana", |l| l.median),
        ("p10", |l| l.p10),
        ("p90", |l| l.p90),
        ("frac_teto", |l| l.ceiling_frac),
    ];
    let mut t_vida_obj = serde_json::Map::new();
    for (key, get_life) in life_entries {
        t_vida_obj.insert(key.to_string(), json!(round3(mean_life(get_life))));
    }
    obj.insert("t_vida".to_string(), Value::Object(t_vida_obj));
    for (short, get_life) in life_entries {
        let flat_key = if short == "frac_teto" {
            "frac_teto".to_string()
        } else {
            format!("t_vida_{short}")
        };
        let per_seed: Vec<f32> = seeds.iter().map(|&sd| round3(life_field(sd, get_life))).collect();
        let avg = round3(per_seed.iter().sum::<f32>() / n);
        obj.insert(flat_key.clone(), json!(avg));
        obj.insert(format!("{flat_key}_por_semente"), json!(per_seed));
    }

    // Champion weights (campo.py's med['W']), rounded to 5 places like
    // `e43_campo.py:95`.
    let mut w_obj = serde_json::Map::new();
    let mut ger_obj = serde_json::Map::new();
    for &sd in seeds {
        let champ = champions
            .get(&sd)
            .expect("measures_to_json: missing seed in `champions`");
        let weights: Vec<f32> = champ.genome.0.iter().map(|&w| round5(w)).collect();
        w_obj.insert(sd.to_string(), json!(weights));
        ger_obj.insert(sd.to_string(), json!(champ.hist.len()));
    }
    obj.insert("W".to_string(), Value::Object(w_obj));
    obj.insert("ger_efetiva".to_string(), Value::Object(ger_obj));

    obj.insert("seeds".to_string(), json!(seeds));
    obj.insert("sigma_mut".to_string(), json!(p.sigma_mut));
    // Only mode this crate implements (SPEC §10 default) — see this module's and
    // `evolution.rs`'s doc comments.
    obj.insert("crn".to_string(), json!("rotativo"));
    obj.insert("campeao".to_string(), json!("ultimo"));

    Value::Object(obj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::NP;

    fn dummy_genome() -> Genome {
        Genome(vec![0.0; NP])
    }

    fn sample(alive_ticks: f32, revenue: f32) -> LifeSample {
        LifeSample {
            alive_ticks,
            dead: true,
            deliveries: 0.0,
            breakdowns: 0.0,
            hunger_end: 0.0,
            water_end: 0.0,
            debt: 0.0,
            reserve: 0.0,
            liters: 0.0,
            refuels: 0.0,
            meals: 0.0,
            repairs: 0.0,
            revenue,
        }
    }

    struct FakeSampler {
        samples: Vec<LifeSample>,
        last_seed: Option<u64>,
    }

    impl LifeSampler for FakeSampler {
        fn sample(
            &mut self,
            _genome: &Genome,
            _mapas: &[World],
            _cfg: &Config,
            _t_max: u32,
            seed: u64,
            _copies: u32,
        ) -> Vec<LifeSample> {
            self.last_seed = Some(seed);
            self.samples.clone()
        }
    }

    #[test]
    fn a_life_that_hits_the_horizon_is_flagged_by_the_ceiling_fraction() {
        let ticks = 1000usize;
        // 10 beings hit the horizon exactly (t_max), 90 die well before it.
        let mut samples: Vec<LifeSample> = (0..10).map(|_| sample(ticks as f32, 0.0)).collect();
        samples.extend((0..90).map(|_| sample(100.0, 0.0)));
        let mut sampler = FakeSampler { samples, last_seed: None };

        let measures = judge(&mut sampler, &dummy_genome(), &[], &Config::default(), ticks);

        assert!(
            measures.life.ceiling_frac > 0.05,
            "10/100 = 0.10 must trip the SPEC §9 invalidation threshold, got {}",
            measures.life.ceiling_frac
        );
    }

    #[test]
    fn the_judge_never_shares_the_evolution_seed() {
        let samples = vec![sample(500.0, 10.0); 4];
        let mut sampler = FakeSampler { samples, last_seed: None };

        let _ = judge(&mut sampler, &dummy_genome(), &[], &Config::default(), 2400);

        assert_eq!(
            sampler.last_seed,
            Some(JUDGE_SEED),
            "judge must always sample with the fixed world seed 7777, never an evolution seed like 777000+13*it"
        );
        assert_ne!(JUDGE_SEED, 777_000, "sanity: the two seed spaces must not collide by accident");
    }

    #[test]
    fn life_fitness_matches_the_spec_eight_formula() {
        // fit = t_vida/240 + w_trab*rec/300, w_trab = 1.0.
        assert_eq!(life_fitness(240.0, 0.0), 1.0);
        assert_eq!(life_fitness(0.0, 300.0), 1.0);
        assert!((life_fitness(120.0, 150.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn measures_to_json_carries_the_python_dict_keys_the_analysis_script_reads() {
        let mut measures = BTreeMap::new();
        let mut champions = BTreeMap::new();
        for sd in [0u64, 1u64] {
            let m = Measures {
                fit: 10.0 + sd as f32,
                deaths: 0.5,
                deliveries: 3.0,
                breakdowns: 1.0,
                hunger_end: 20.0,
                water_end: 30.0,
                debt: -5.0,
                reserve: 40.0,
                liters_per_refuel: 60.0,
                refuels: 2.0,
                meals: 1.0,
                repairs: 0.0,
                dof: DOF_V0,
                metabolism: METABOLISM_V0,
                revenue: 100.0,
                life: LifeStats { mean: 5.0, median: 5.0, p10: 1.0, p90: 9.0, ceiling_frac: 0.0 },
            };
            measures.insert(sd, m);
            champions.insert(
                sd,
                Champion { genome: dummy_genome(), hist: vec![1.0, 2.0, 3.0] },
            );
        }
        let params = GaParams { seeds: vec![0, 1], ..GaParams::default() };

        let json = measures_to_json(&measures, &champions, &params);
        let obj = json.as_object().expect("measures_to_json must return a JSON object");

        for key in [
            "fit", "fit_por_semente", "mortos", "ent", "panes", "fome_fim", "agua_fim", "dinh",
            "res", "litros_por_abast", "n_posto", "n_come", "oficina", "dof", "metab", "rec",
            "t_vida", "t_vida_media", "t_vida_media_por_semente", "frac_teto", "W",
            "ger_efetiva", "seeds", "sigma_mut", "crn", "campeao",
        ] {
            assert!(obj.contains_key(key), "missing key `{key}` the Python analysis script reads");
        }

        assert_eq!(obj["fit"], json!(10.5));
        assert_eq!(obj["fit_por_semente"], json!([10.0, 11.0]));
        assert_eq!(obj["t_vida"]["media"], json!(5.0));
        assert_eq!(obj["W"]["0"], json!(vec![0.0f32; NP]));
        assert_eq!(obj["ger_efetiva"]["0"], json!(3));
    }
}
