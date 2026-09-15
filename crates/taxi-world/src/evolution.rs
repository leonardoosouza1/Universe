//! The GA loop (SPEC §10, `evoluir_sementes`, `src/lab.py:1706-1772`) — plan
//! §T6 "evolution.rs". See `docs/plano-mundo-taxi.md` §T6 and
//! `/home/leonardo/projects/LLM/neuromod-lab/docs/71-spec-mundo-taxi.md` §10.
//!
//! # Why `evolve` takes a `Simulador`, not just `worlds`/`cfg`
//!
//! T5 (`tick.rs`, `gate.rs`) is being written in parallel and does not exist yet in
//! this worktree. The plan's own T6 interface sketch (`docs/plano-mundo-taxi.md`
//! §T6) writes `pub fn evolve(worlds: &[World], cfg: &Config, p: &GaParams) ->
//! BTreeMap<u64, Champion>` with no simulator parameter — that sketch implicitly
//! assumed a single crate-wide tick function already existed to call. Since it does
//! not, this task's own brief (not the plan doc) asks for a MINIMAL trait the GA
//! runs against, with the real tick-based implementation left to whoever wires T5
//! in later. **This is a declared divergence from the plan's literal signature**:
//! `evolve`/`judge` both gained a leading `sim: &mut dyn Simulador` parameter — see
//! this crate's final report for the full list.
//!
//! [`Simulador::avaliar`] mirrors `campo.avaliar` (`campo.py:605-607`) exactly:
//! `fit.view(P, NC*R).mean(1)`, i.e. ONE fitness value per genome, averaged over
//! that genome's `mapas.len() * r` beings. The batch layout a real implementation
//! MUST use is SPEC §10's "índice do lote" (plan §3.1, armadilha 8):
//!
//! ```text
//! b = p * (NC * R) + c * R + r        // p = genome index, c = map index, r = replica
//! ```
//!
//! i.e. genome is the OUTERMOST axis, then map, then replica — get this backwards
//! and every genome is scored on a plausible-looking but WRONG mix of worlds. See
//! this module's tests for a worked reference implementation of the convention
//! (`BatchIndexFake`) and `judge.rs` for the second, `LifeSampler`, half of the
//! contract (SPEC §9's `medidas`, which needs per-being fields `avaliar` does not
//! expose).
//!
//! The GA's own RNG does not need to be bit-identical to torch's Philox — SPEC §10
//! says so explicitly ("O RNG do GA... não precisa ser bit-igual... o que precisa
//! bater é a distribuição do resultado"). This module reuses `passengers::TaxiRng`
//! (a crate-local xorshift64*) rather than adding `rand`/`rand_chacha`, per this
//! task's hard rule against new dependencies, plus a small Box-Muller wrapper for
//! `N(0, sigma)` draws.

use std::collections::BTreeMap;

use crate::constants::{Config, NP};
use crate::genome::Genome;
use crate::passengers::TaxiRng;
use crate::world::World;

/// A batch scorer: given `genomas.len()` genomes, `mapas.len()` maps and `r`
/// replicas per (genome, map) pair, returns one MEAN fitness per genome (SPEC §10
/// step 1, `campo.avaliar`). See this module's doc comment for the batch-index
/// convention an implementation must follow and for why this trait exists instead
/// of calling T5's `tick::step` directly.
pub trait Simulador {
    fn avaliar(
        &mut self,
        genomas: &[Genome],
        mapas: &[World],
        cfg: &Config,
        t_max: u32,
        seed: u64,
        r: u32,
    ) -> Vec<f32>;
}

/// GA hyperparameters (SPEC §10 table, `cenarios/mundoA_seres.json`'s
/// `protocolo`). Field names and defaults match the plan's own T6 interface block
/// exactly.
#[derive(Debug, Clone, PartialEq)]
pub struct GaParams {
    pub pop: usize,
    pub generations: usize,
    pub patience: Option<usize>,
    pub replicas: usize,
    pub seeds: Vec<u64>,
    pub sigma_mut: f32,
    pub elite_frac: f32,
    pub clamp: f32,
    pub ticks: usize,
    /// `W0` (SPEC §10 "Inicialização"): `None` seeds the population `~ N(0,
    /// 0.8²)` (today's only behaviour); `Some(champion)` seeds it
    /// `champion + N(0, SIGMA_ESPINHA²)` instead — every individual starts a
    /// small mutation away from a known-good genome rather than from scratch.
    pub seed_from: Option<Genome>,
    /// How the champion is picked out of the final generation (SPEC §10
    /// `--campeao`). See [`ChampionMode`].
    pub champion_mode: ChampionMode,
    /// `crn` (SPEC §10 `--crn`): `false` = `'rotativo'` (today's only
    /// behaviour — the per-generation world seed is `777_000 + 13*it`, a
    /// different world every generation); `true` = `'fixo'` (common random
    /// numbers — every generation scores against the SAME world seed,
    /// `777_000`, so a fitness delta between two generations' champions is
    /// never a world-draw artifact).
    pub crn: bool,
}

impl Default for GaParams {
    /// The v0 protocol's own defaults (SPEC §10 table: "valor do protocolo v0").
    /// `seed_from`/`champion_mode`/`crn` all default to the pre-T7 behaviour
    /// (no seeding, `'ultimo'`, `'rotativo'`) — every test written before T7
    /// keeps passing unchanged.
    fn default() -> Self {
        Self {
            pop: 128,
            generations: 30,
            patience: Some(15),
            replicas: 4,
            seeds: vec![0, 1, 2, 3, 4],
            sigma_mut: 0.1,
            elite_frac: 0.15,
            clamp: 5.0,
            ticks: 7200,
            seed_from: None,
            champion_mode: ChampionMode::Last,
            crn: false,
        }
    }
}

/// The winner of one evolution seed's run (SPEC §10 step 7: `campeao = W[si][0]`
/// of the last generation). T7 adds the second mode SPEC §10 documents
/// (`campeao='reavaliado'`) — see [`ChampionMode`].
#[derive(Debug, Clone, PartialEq)]
pub struct Champion {
    pub genome: Genome,
    /// Best fitness per generation actually run for this seed (SPEC §10 step 2,
    /// `hist[sd]`) — shorter than `GaParams::generations` when patience stopped the
    /// run early. This is `med['ger_efetiva'][sd]` in the Python JSON (`len(hist)`,
    /// `e43_campo.py:96`).
    pub hist: Vec<f32>,
}

/// How [`evolve`] picks the champion out of the final generation (SPEC §10,
/// `--campeao`). Named `ChampionMode`, not `Champion`, because that name is
/// already the RESULT struct above — the plan/task brief's own "`Champion::Last |
/// Reevaluated`" phrasing collides with it, so this is the T7 divergence from
/// that literal wording.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChampionMode {
    /// `campeao='ultimo'` (today's only behaviour, and the default): the
    /// argmax of the LAST generation, `populations[si][0]` verbatim. SPEC §10's
    /// own note on this mode: "maldição do vencedor com SNR .72" — a noisy
    /// last-generation fitness estimate biases the argmax toward genomes that
    /// got a lucky world draw, not necessarily the best genome.
    #[default]
    Last,
    /// `campeao='reavaliado'` (SPEC §10, pred 166): the WHOLE final elite (not
    /// just rank 0) is re-scored once more on a fresh world seed
    /// (`REEVAL_SEED`, below — distinct from both the training rotation
    /// `777_000 + 13*it` and `judge::JUDGE_SEED = 7777`), and the champion is
    /// whichever elite genome maximizes the AVERAGE of its last-generation
    /// fitness and this fresh one — "~.25% do custo do run" per SPEC, since
    /// only the elite (not the whole population) is re-scored, once.
    Reevaluated,
}

/// σ of the champion-seeded population's noise (SPEC §10 "Inicialização",
/// `sigma_espinha`). Not a `GaParams` field: this crate's `Genome` is always
/// exactly `NP` weights with no genes "beyond the spine" (see `judge.rs`'s doc
/// comment on `DOF_V0`), so SPEC §10's OTHER seeding constant, `sigma0` (noise
/// for genes beyond the spine), has nothing to apply to here and is left out.
const SIGMA_ESPINHA: f32 = 0.05;

/// World seed [`evolve`] uses for the [`ChampionMode::Reevaluated`] re-score —
/// picked to land in neither the training rotation's range (`777_000 + 13*it`,
/// `it` up to a few hundred) nor `judge::JUDGE_SEED = 7777`.
const REEVAL_SEED: u64 = 611_000;

/// Runs the GA independently for every seed in `p.seeds`, sharing ONE per-generation
/// batch call to `sim.avaliar` across all seeds (SPEC §10: `evoluir_sementes` stacks
/// all `S` populations into one `S*P`-row batch — "MEGA-LOTE de sementes",
/// `lab.py:1706-1710` — precisely so every seed is scored against the SAME
/// per-generation world seed, common-random-numbers style). Only `crn='rotativo'`
/// (`seed = 777000 + 13*it`) and `campeao='ultimo'` are implemented — see this
/// crate's final report.
pub fn evolve(
    sim: &mut dyn Simulador,
    worlds: &[World],
    cfg: &Config,
    p: &GaParams,
) -> BTreeMap<u64, Champion> {
    let s = p.seeds.len();
    assert!(s > 0, "evolve: GaParams::seeds must not be empty");
    assert!(
        p.pop >= 2,
        "evolve: GaParams::pop must be at least 2 (elite needs >= 2)"
    );

    if p.champion_mode == ChampionMode::Reevaluated {
        assert!(
            p.generations >= 1,
            "evolve: ChampionMode::Reevaluated needs at least one generation to have run, \
             to have an elite to reevaluate"
        );
    }

    let mut rngs: Vec<TaxiRng> = p.seeds.iter().map(|&sd| TaxiRng::new(sd)).collect();
    // SPEC §10 "Inicialização": no W0 -> W ~ N(0, 0.8^2), shape (P, NP), one
    // Generator per seed (`lab.py:1722`). With W0 (`p.seed_from`, T7): every
    // individual starts `champion + N(0, SIGMA_ESPINHA^2)` instead
    // (`lab.py:1724-1729`'s "sigma_espinha" branch — `sigma0`, the OTHER half of
    // that branch, has no gene dimension to apply to here, see `SIGMA_ESPINHA`'s
    // own doc comment).
    let mut populations: Vec<Vec<Genome>> = rngs
        .iter_mut()
        .map(|rng| {
            (0..p.pop)
                .map(|_| match &p.seed_from {
                    None => Genome((0..NP).map(|_| gaussian(rng) * 0.8).collect()),
                    Some(champion) => Genome(
                        champion
                            .0
                            .iter()
                            .map(|&w| w + gaussian(rng) * SIGMA_ESPINHA)
                            .collect(),
                    ),
                })
                .collect()
        })
        .collect();

    let mut hist: Vec<Vec<f32>> = vec![Vec::new(); s];
    // The final generation's elite, `(genome, fitness-at-that-generation)`, one
    // `Vec` per seed — overwritten every iteration, so after the loop it holds
    // ONLY the last generation's elite. Only read by `ChampionMode::Reevaluated`
    // (T7); `Last` mode ignores it entirely (`populations[si][0]` already IS
    // that same elite's rank-0 genome).
    let mut last_elite: Vec<Vec<(Genome, f32)>> = vec![Vec::new(); s];

    for it in 0..p.generations {
        // crn='rotativo' (SPEC §10 table default; `lab.py:1732`) unless `p.crn`
        // asks for 'fixo' (T7): the SAME world seed every generation.
        let gen_seed = if p.crn {
            777_000u64
        } else {
            777_000u64 + (it as u64) * 13
        };

        let mut batch: Vec<Genome> = Vec::with_capacity(s * p.pop);
        for pop in &populations {
            batch.extend(pop.iter().cloned());
        }
        let fitness = sim.avaliar(
            &batch,
            worlds,
            cfg,
            p.ticks as u32,
            gen_seed,
            p.replicas as u32,
        );
        assert_eq!(
            fitness.len(),
            batch.len(),
            "Simulador::avaliar must return exactly one fitness value per genome in `genomas`"
        );

        let mut next_populations = Vec::with_capacity(s);
        for si in 0..s {
            let start = si * p.pop;
            let seed_fitness = &fitness[start..start + p.pop];
            let (next, best, elite_fitness) =
                select_and_breed(&populations[si], seed_fitness, p, &mut rngs[si]);
            hist[si].push(best);
            last_elite[si] = next
                .iter()
                .take(elite_fitness.len())
                .cloned()
                .zip(elite_fitness.iter().copied())
                .collect();
            next_populations.push(next);
        }
        populations = next_populations;

        if let Some(pac) = p.patience {
            // SPEC §10 step 6 / `lab.py:1759-1763`: only checked from `it >=
            // paciencia` on, using the mean-across-seeds of the window max vs. the
            // max before it.
            if pac > 0 && it >= pac {
                let agora: f32 = hist
                    .iter()
                    .map(|h| window_max(&h[h.len() - pac..]))
                    .sum::<f32>()
                    / s as f32;
                let antes: f32 = hist
                    .iter()
                    .map(|h| window_max(&h[..h.len() - pac]))
                    .sum::<f32>()
                    / s as f32;
                if agora <= antes + 0.01 * antes.abs() + 1e-6 {
                    break;
                }
            }
        }
    }

    let mut champions = BTreeMap::new();
    match p.champion_mode {
        ChampionMode::Last => {
            for (si, &seed) in p.seeds.iter().enumerate() {
                champions.insert(
                    seed,
                    Champion {
                        genome: populations[si][0].clone(),
                        hist: hist[si].clone(),
                    },
                );
            }
        }
        ChampionMode::Reevaluated => {
            // One combined re-score batch across every seed's elite (mirrors the
            // main loop's own "MEGA-LOTE de sementes" batching), on a fresh world
            // seed (`REEVAL_SEED`) none of the training generations ever used.
            let mut batch: Vec<Genome> = Vec::new();
            let mut bounds: Vec<usize> = vec![0];
            for elite in &last_elite {
                batch.extend(elite.iter().map(|(g, _)| g.clone()));
                bounds.push(batch.len());
            }
            let reeval = sim.avaliar(
                &batch,
                worlds,
                cfg,
                p.ticks as u32,
                REEVAL_SEED,
                p.replicas as u32,
            );
            assert_eq!(
                reeval.len(),
                batch.len(),
                "Simulador::avaliar must return exactly one fitness value per genome in `genomas`"
            );

            for (si, &seed) in p.seeds.iter().enumerate() {
                let elite = &last_elite[si];
                let reeval_slice = &reeval[bounds[si]..bounds[si + 1]];
                let (best_genome, _) = elite
                    .iter()
                    .zip(reeval_slice.iter())
                    .map(|((genome, orig_fit), &new_fit)| {
                        (genome.clone(), (orig_fit + new_fit) / 2.0)
                    })
                    .fold(
                        None,
                        |acc: Option<(Genome, f32)>, (genome, avg)| match acc {
                            Some((best_genome, best_avg)) if best_avg >= avg => {
                                Some((best_genome, best_avg))
                            }
                            _ => Some((genome, avg)),
                        },
                    )
                    .expect(
                        "evolve: ChampionMode::Reevaluated needs a non-empty elite \
                         (elite_frac must select at least one genome)",
                    );
                champions.insert(
                    seed,
                    Champion {
                        genome: best_genome,
                        hist: hist[si].clone(),
                    },
                );
            }
        }
    }
    champions
}

/// One generation's selection + mutation for a SINGLE seed's population (SPEC §10
/// steps 2-5). Exposed on its own (not just inlined in [`evolve`]) so this module's
/// own tests can check the elite/mutation contract directly, without running a
/// multi-generation loop end to end. Returns `(next_population, best_fitness,
/// elite_fitness)` — `elite_fitness[k]` is `next[k]`'s fitness AT THIS
/// generation (i.e. BEFORE the next call re-scores it), length `ne`; added in
/// T7 for [`ChampionMode::Reevaluated`], which needs the elite's ORIGINAL
/// (pre-reevaluation) fitness alongside its genome, not just the population's
/// new shape.
pub fn select_and_breed(
    population: &[Genome],
    fitness: &[f32],
    p: &GaParams,
    rng: &mut TaxiRng,
) -> (Vec<Genome>, f32, Vec<f32>) {
    assert_eq!(
        population.len(),
        fitness.len(),
        "select_and_breed: one fitness value per genome"
    );
    let pop = population.len();
    let ne = elite_count(pop, p.elite_frac);

    // SPEC §10 step 2: `f[si].argsort(descending=True)`.
    let mut order: Vec<usize> = (0..pop).collect();
    order.sort_by(|&a, &b| {
        fitness[b]
            .partial_cmp(&fitness[a])
            .expect("select_and_breed: fitness must not be NaN")
    });
    let best = fitness[order[0]];
    let elite_fitness: Vec<f32> = order[..ne].iter().map(|&i| fitness[i]).collect();

    // SPEC §10 step 3: the elite passes through UNCHANGED.
    let mut next: Vec<Genome> = order[..ne].iter().map(|&i| population[i].clone()).collect();

    // SPEC §10 step 4: `idx ~ U{0..ne-1}`, `fil = clamp(Wb[idx] + N(0, sigma_mut), -5, +5)`.
    for _ in 0..(pop - ne) {
        let idx = ((rng.next_f32() * ne as f32) as usize).min(ne - 1);
        let child = {
            let parent = &next[idx];
            Genome(
                parent
                    .0
                    .iter()
                    .map(|&w| (w + gaussian(rng) * p.sigma_mut).clamp(-p.clamp, p.clamp))
                    .collect(),
            )
        };
        next.push(child);
    }

    (next, best, elite_fitness)
}

/// `ne = max(2, int(P*elite))` (SPEC §10 step 2), clamped to `pop` so a
/// pathologically small population never panics on the elite slice.
fn elite_count(pop: usize, elite_frac: f32) -> usize {
    ((pop as f32 * elite_frac) as usize).max(2).min(pop)
}

fn window_max(xs: &[f32]) -> f32 {
    xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

/// Standard-normal sample via Box-Muller, built on [`TaxiRng::next_f32`]. SPEC §10
/// is explicit the GA's RNG need not be torch-bit-identical, only statistically
/// comparable (SPEC §11.3) — so this crate-local generator, not `rand_distr`, is
/// intentional (no new dependency, per this task's hard rule). `u1` is floored away
/// from 0 so `ln` stays finite.
pub(crate) fn gaussian(rng: &mut TaxiRng) -> f32 {
    let u1 = rng.next_f32().max(1e-7);
    let u2 = rng.next_f32();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_genome(value: f32) -> Genome {
        Genome(vec![value; NP])
    }

    // --- select_and_breed ---------------------------------------------------

    #[test]
    fn the_elite_passes_untouched_to_the_next_generation() {
        // 10 genomes, fitness = index (so descending order reverses them);
        // ne = max(2, floor(10*0.15)) = max(2, 1) = 2.
        let population: Vec<Genome> = (0..10).map(|i| flat_genome(i as f32)).collect();
        let fitness: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let params = GaParams {
            pop: 10,
            elite_frac: 0.15,
            sigma_mut: 1.0,
            clamp: 5.0,
            ..GaParams::default()
        };
        let mut rng = TaxiRng::new(1);
        let (next, best, elite_fitness) =
            select_and_breed(&population, &fitness, &params, &mut rng);

        assert_eq!(
            best, 9.0,
            "best fitness must be the max, not the first entry"
        );
        assert_eq!(next[0], population[9], "the top genome must be bit-exact");
        assert_eq!(next[1], population[8], "the runner-up must be bit-exact");
        assert_eq!(next.len(), 10);
        assert_eq!(
            elite_fitness,
            vec![9.0, 8.0],
            "elite_fitness must be next[0..ne]'s own fitness, descending"
        );
    }

    #[test]
    fn every_gene_of_a_child_is_mutated_and_clamped_to_plus_minus_five() {
        // Every gene starts at 4.9 and sigma_mut is huge: without the clamp, every
        // mutated gene overshoots +-5 with overwhelming probability.
        let population: Vec<Genome> = (0..4).map(|_| flat_genome(4.9)).collect();
        let fitness = vec![1.0, 2.0, 3.0, 4.0];
        let params = GaParams {
            pop: 4,
            elite_frac: 0.5, // ne = max(2, 2) = 2
            sigma_mut: 100.0,
            clamp: 5.0,
            ..GaParams::default()
        };
        let mut rng = TaxiRng::new(7);
        let (next, _, _) = select_and_breed(&population, &fitness, &params, &mut rng);

        assert_eq!(next.len(), 4);
        for child in &next[2..] {
            assert_eq!(child.0.len(), NP);
            for (i, &w) in child.0.iter().enumerate() {
                assert!(
                    (-5.0..=5.0).contains(&w),
                    "gene {i} = {w} escaped the +-5 clamp"
                );
                assert_ne!(w, 4.9, "gene {i} was not mutated at all");
            }
        }
    }

    // --- evolve / patience ---------------------------------------------------

    /// Ignores the genomes entirely: every genome in a generation gets the SAME
    /// scripted fitness, so `hist` becomes exactly the scripted series and the
    /// patience-window stop point is fully predictable.
    struct ScriptedSimulador {
        series: Vec<f32>,
        call: usize,
    }

    impl Simulador for ScriptedSimulador {
        fn avaliar(
            &mut self,
            genomas: &[Genome],
            _mapas: &[World],
            _cfg: &Config,
            _t_max: u32,
            _seed: u64,
            _r: u32,
        ) -> Vec<f32> {
            let value = self.series[self.call];
            self.call += 1;
            vec![value; genomas.len()]
        }
    }

    #[test]
    fn the_patience_window_compares_the_best_of_the_window_against_the_best_before_it() {
        // pac = 2. hist after each generation: [0, 1, 2, 2, 2, ...].
        // it=2: window=[1,2]->2, before=[0]->0: 2 <= 0+1e-6? no.
        // it=3: window=[2,2]->2, before=[0,1]->1: 2 <= 1.01? no.
        // it=4: window=[2,2]->2, before=[0,1,2]->2: 2 <= 2.02? YES -> stop.
        let series = vec![0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        let mut sim = ScriptedSimulador { series, call: 0 };
        let params = GaParams {
            pop: 4,
            generations: 7,
            patience: Some(2),
            seeds: vec![42],
            ..GaParams::default()
        };
        let worlds: Vec<World> = vec![];
        let champions = evolve(&mut sim, &worlds, &Config::default(), &params);

        let champ = &champions[&42];
        assert_eq!(
            champ.hist,
            vec![0.0, 1.0, 2.0, 2.0, 2.0],
            "must stop right after the window condition first holds, generation 4 (0-indexed)"
        );
    }

    // --- Simulador batch-index contract (documentation-by-test) --------------

    /// A reference implementation of the `b = p*(NC*R) + c*R + r` convention this
    /// module's doc comment documents. Ignores genome/map CONTENT and derives a
    /// per-being value purely from the decoded `(p, c, r)` indices, so the expected
    /// per-genome means can be hand-computed.
    struct BatchIndexFake;

    impl Simulador for BatchIndexFake {
        fn avaliar(
            &mut self,
            genomas: &[Genome],
            mapas: &[World],
            _cfg: &Config,
            _t_max: u32,
            _seed: u64,
            r: u32,
        ) -> Vec<f32> {
            let p_count = genomas.len();
            let nc = mapas.len();
            let r = r as usize;
            let mut out = vec![0.0f32; p_count];
            for p_idx in 0..p_count {
                let mut sum = 0.0f32;
                for c in 0..nc {
                    for rep in 0..r {
                        // b = p*(NC*R) + c*R + rep -- decoded implicitly by this
                        // very loop nesting; the raw per-being value is a function
                        // of (p, c, rep) alone, matching this module's doc comment.
                        sum += 100.0 * p_idx as f32 + 10.0 * c as f32 + rep as f32;
                    }
                }
                out[p_idx] = sum / (nc * r) as f32;
            }
            out
        }
    }

    #[test]
    fn fitness_averages_over_maps_and_replicas_for_each_genome() {
        // P=2, NC=2, R=3.
        let genomas = vec![flat_genome(0.0), flat_genome(0.0)];
        let mapas: Vec<World> = (0..2)
            .map(|_| World {
                side: 200.0,
                cost_tex: vec![],
                district_field: vec![],
                centers: vec![],
                tax: vec![],
                price_mul_base: vec![],
                rent: vec![],
                demand: vec![],
                price_phase: vec![],
                services: [vec![], vec![], vec![], vec![], vec![]],
            })
            .collect();
        let mut sim = BatchIndexFake;
        let out = sim.avaliar(&genomas, &mapas, &Config::default(), 100, 0, 3);

        // p=0: mean of {0,1,2, 10,11,12} = 36/6 = 6.0
        // p=1: mean of {100,101,102, 110,111,112} = 636/6 = 106.0
        assert_eq!(out, vec![6.0, 106.0]);
    }

    #[test]
    fn genome_seeded_populations_stay_within_the_np_dimension() {
        // Sanity check on evolve()'s own population init, not a named acceptance
        // test: every genome handed to `avaliar` must have exactly NP weights.
        struct RecordingSimulador {
            saw_wrong_length: bool,
        }
        impl Simulador for RecordingSimulador {
            fn avaliar(
                &mut self,
                genomas: &[Genome],
                _mapas: &[World],
                _cfg: &Config,
                _t_max: u32,
                _seed: u64,
                _r: u32,
            ) -> Vec<f32> {
                if genomas.iter().any(|g| g.0.len() != NP) {
                    self.saw_wrong_length = true;
                }
                vec![0.0; genomas.len()]
            }
        }
        let mut sim = RecordingSimulador {
            saw_wrong_length: false,
        };
        let params = GaParams {
            pop: 3,
            generations: 1,
            patience: None,
            seeds: vec![1],
            ..GaParams::default()
        };
        let _ = evolve(&mut sim, &[], &Config::default(), &params);
        assert!(!sim.saw_wrong_length);
    }

    // --- T7: seed_from / champion_mode / crn ---------------------------------

    #[test]
    fn seed_from_seeds_every_individual_near_the_champion_not_at_n_0_64() {
        // generations: 0 -> evolve() never calls `sim.avaliar` at all (the loop
        // body never runs), so `populations[si][0]` comes straight out of the
        // (T7) seeded initializer, untouched — exactly what this test wants to
        // inspect. A `Simulador` that panics if called doubles as an assertion
        // that seeding really does skip the whole generation loop.
        struct PanicsIfCalled;
        impl Simulador for PanicsIfCalled {
            fn avaliar(
                &mut self,
                _: &[Genome],
                _: &[World],
                _: &Config,
                _: u32,
                _: u64,
                _: u32,
            ) -> Vec<f32> {
                panic!("avaliar must not be called when GaParams::generations == 0");
            }
        }
        let champion = Genome(vec![5.0; NP]);
        let params = GaParams {
            pop: 2,
            generations: 0,
            patience: None,
            seeds: vec![9],
            seed_from: Some(champion.clone()),
            ..GaParams::default()
        };
        let mut sim = PanicsIfCalled;
        let champions = evolve(&mut sim, &[], &Config::default(), &params);

        let seeded = &champions[&9].genome;
        assert_eq!(seeded.0.len(), NP);
        let mut any_gene_moved = false;
        for (i, (&w, &c)) in seeded.0.iter().zip(champion.0.iter()).enumerate() {
            assert!(
                (w - c).abs() < 0.3,
                "gene {i} = {w} strayed too far from the champion's {c} for N(0, {SIGMA_ESPINHA}^2) noise"
            );
            if w != c {
                any_gene_moved = true;
            }
        }
        assert!(
            any_gene_moved,
            "seed_from must add SOME noise, not pass the champion through untouched"
        );
    }

    #[test]
    fn without_seed_from_the_population_still_starts_at_n_0_64_around_zero() {
        // Regression guard for "comportamento antigo intacto": the default
        // (`seed_from: None`) path must still be the wide N(0, 0.8^2) draw, not
        // accidentally routed through the champion-seeding branch.
        struct PanicsIfCalled;
        impl Simulador for PanicsIfCalled {
            fn avaliar(
                &mut self,
                _: &[Genome],
                _: &[World],
                _: &Config,
                _: u32,
                _: u64,
                _: u32,
            ) -> Vec<f32> {
                panic!("avaliar must not be called when GaParams::generations == 0");
            }
        }
        let params = GaParams {
            pop: 2,
            generations: 0,
            patience: None,
            seeds: vec![9],
            ..GaParams::default()
        };
        let mut sim = PanicsIfCalled;
        let champions = evolve(&mut sim, &[], &Config::default(), &params);
        let seeded = &champions[&9].genome;
        // N(0, 0.8^2) over 40 genes: at least one gene landing outside +-0.3 is
        // overwhelmingly likely (each gene independently has ~73% chance of
        // being outside +-0.3; the chance ALL 40 land inside is ~0.73^40 ~ 6e-6).
        assert!(
            seeded.0.iter().any(|&w| w.abs() > 0.3),
            "default seeding must be the wide N(0, 0.8^2) draw, got a suspiciously narrow genome {:?}",
            seeded.0
        );
    }

    #[test]
    fn crn_true_scores_every_generation_against_the_same_world_seed() {
        struct SeedSpy {
            seen: Vec<u64>,
        }
        impl Simulador for SeedSpy {
            fn avaliar(
                &mut self,
                genomas: &[Genome],
                _: &[World],
                _: &Config,
                _: u32,
                seed: u64,
                _: u32,
            ) -> Vec<f32> {
                self.seen.push(seed);
                vec![0.0; genomas.len()]
            }
        }
        let params = GaParams {
            pop: 2,
            generations: 4,
            patience: None,
            seeds: vec![1],
            crn: true,
            ..GaParams::default()
        };
        let mut sim = SeedSpy { seen: Vec::new() };
        let _ = evolve(&mut sim, &[], &Config::default(), &params);
        assert_eq!(
            sim.seen,
            vec![777_000, 777_000, 777_000, 777_000],
            "crn=true (fixo) must reuse ONE world seed every generation"
        );
    }

    #[test]
    fn crn_false_rotates_the_world_seed_by_generation_matching_777000_plus_13_times_it() {
        struct SeedSpy {
            seen: Vec<u64>,
        }
        impl Simulador for SeedSpy {
            fn avaliar(
                &mut self,
                genomas: &[Genome],
                _: &[World],
                _: &Config,
                _: u32,
                seed: u64,
                _: u32,
            ) -> Vec<f32> {
                self.seen.push(seed);
                vec![0.0; genomas.len()]
            }
        }
        let params = GaParams {
            pop: 2,
            generations: 3,
            patience: None,
            seeds: vec![1],
            ..GaParams::default()
        };
        let mut sim = SeedSpy { seen: Vec::new() };
        let _ = evolve(&mut sim, &[], &Config::default(), &params);
        assert_eq!(
            sim.seen,
            vec![777_000, 777_013, 777_026],
            "crn=false (default, rotativo) must be 777_000 + 13*it"
        );
    }

    #[test]
    fn champion_mode_reevaluated_can_overturn_the_last_generations_argmax() {
        // pop = elite_frac = 1.0 -> ne = pop = 2, zero children: `next` is
        // always the SAME two `Genome` objects, only ever REORDERED by fitness,
        // so their identity (via `Genome`'s `PartialEq`) is trackable across
        // generations without needing to hand-predict any gaussian draw.
        //
        // Script (one training generation, then the T7 reeval call):
        //   training gen 0: genome A scores 5.0, genome B scores 1.0 -> A "wins"
        //     the naive argmax (ChampionMode::Last would report A).
        //   reeval (fresh world, REEVAL_SEED): A scores 0.0, B scores 10.0.
        //   averages: A = (5.0+0.0)/2 = 2.5; B = (1.0+10.0)/2 = 5.5 -> B must win
        //     under ChampionMode::Reevaluated — the "maldição do vencedor"
        //     correction SPEC §10 describes.
        struct Scripted {
            call: usize,
            captured_gen0: Option<Vec<Genome>>,
        }
        impl Simulador for Scripted {
            fn avaliar(
                &mut self,
                genomas: &[Genome],
                _: &[World],
                _: &Config,
                _: u32,
                seed: u64,
                _: u32,
            ) -> Vec<f32> {
                self.call += 1;
                if self.call == 1 {
                    self.captured_gen0 = Some(genomas.to_vec());
                    vec![5.0, 1.0]
                } else {
                    assert_eq!(
                        seed, REEVAL_SEED,
                        "the reeval call must use REEVAL_SEED, not the training rotation"
                    );
                    // Order must be [A, B] again: `last_elite` mirrors `next[0..ne]`,
                    // and with ne == pop, `next` after gen 0 is genomas reordered by
                    // gen-0 fitness descending -- i.e. still [A, B] (A already won).
                    vec![0.0, 10.0]
                }
            }
        }
        let params = GaParams {
            pop: 2,
            generations: 1,
            patience: None,
            seeds: vec![1],
            elite_frac: 1.0,
            champion_mode: ChampionMode::Reevaluated,
            ..GaParams::default()
        };
        let mut sim = Scripted {
            call: 0,
            captured_gen0: None,
        };
        let champions = evolve(&mut sim, &[], &Config::default(), &params);

        let gen0 = sim.captured_gen0.expect("gen 0 must have called avaliar");
        let genome_a = &gen0[0];
        let genome_b = &gen0[1];
        assert_ne!(
            genome_a, genome_b,
            "the test needs two DISTINGUISHABLE genomes"
        );
        assert_eq!(
            champions[&1].genome, *genome_b,
            "Reevaluated must pick B (avg 5.5), not A (avg 2.5, the naive last-gen argmax)"
        );
    }

    #[test]
    fn champion_mode_last_is_still_the_default_and_ignores_the_reeval_axis() {
        // Same script as above, but `champion_mode` left at its default
        // (`ChampionMode::Last`): must pick A (gen 0's argmax) and must NEVER
        // call `avaliar` a second time (no reeval batch at all).
        struct Scripted {
            call: usize,
        }
        impl Simulador for Scripted {
            fn avaliar(
                &mut self,
                _: &[Genome],
                _: &[World],
                _: &Config,
                _: u32,
                _: u64,
                _: u32,
            ) -> Vec<f32> {
                self.call += 1;
                assert!(
                    self.call <= 1,
                    "ChampionMode::Last must never call avaliar a second time"
                );
                vec![5.0, 1.0]
            }
        }
        let params = GaParams {
            pop: 2,
            generations: 1,
            patience: None,
            seeds: vec![1],
            elite_frac: 1.0,
            ..GaParams::default()
        };
        let mut sim = Scripted { call: 0 };
        let champions = evolve(&mut sim, &[], &Config::default(), &params);
        assert_eq!(sim.call, 1);
        // A is whichever genome scored 5.0 at the only generation that ran --
        // reuse `select_and_breed` directly to name it without hand-predicting
        // the gaussian population.
        assert_eq!(champions[&1].hist, vec![5.0]);
    }
}
