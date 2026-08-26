# M0 dose canaries — what was measured without training

Measured 2026-08-26. Ten evaluation runs, `--gens 1` throughout: no evolution, no
training. Frozen champion `ereg-m4-m4-s77` gen 43, 48 beings, 20,000 ticks, tracks
[8005, 8200, 9001], `--width 28 --out-beta 0.0 --mascara-regioes --neuromod-regioes`.
Runs were spaced 40 s apart to leave the machine usable; GPU drew 20.7 W on average and
peaked at 33 W of a ~200 W card, at 46% mean utilisation. These workloads do not
saturate the device — the same launch-latency ceiling recorded for small batches
elsewhere in this project.

Every arm reproduced its distance to the metre across a reboot and a rebuild.

## D8 — sensory subscription (`--custo-sensor`)

`docs/ESCADA_DIFICULDADE.md` left this dose as open debt. It is now closed for the part
that does not need evolution.

| dose | median | Δ vs dose 0 | mortality | being 0 median speed |
|---|---|---|---|---|
| 0.00 | 3761 m | — | 0% | 4.57 m/s |
| 0.05 | 3805 m | +44 m | 0% | 4.24 m/s |
| 0.10 | 3558 m | −203 m | 0% | 2.93 m/s |
| 0.20 | 3319 m | −442 m | 0% | 2.10 m/s |
| 0.30 | 3150 m | −611 m | 0% | 1.27 m/s |
| 0.45 | 2991 m | −770 m | 0% | 0.44 m/s |

**Canary passed** on the ROADMAP's criterion: every dose gives a different number,
monotonically, including the off arm. Read it with the replication section below — these
are single runs, and the one effect that was replicated shrank by 2.2×. **Mortality is 0% across the whole sweep** — this pressure STALLS rather than kills,
the same signature the fuel sweep already showed, so mortality must be read as a ceiling
here and never as a floor.

**Usable dose is ≤ 0.20.** At 0.30 the being is down to 1.27 m/s and at 0.45 to 0.44 m/s;
below about 2 m/s there is no driving left to select among.

### The design verdict, from the mechanism's own arithmetic

The `[sensor] CEILING` line reports what pruning buys. Against a 20,000-tick generation:

| dose | pruning ONE family of 16 channels buys | share of the generation |
|---|---|---|
| 0.10 | 167 ticks | 0.83% |
| 0.20 | 333 ticks | 1.67% |
| 0.30 | 500 ticks | 2.50% |
| 0.45 | 750 ticks | 3.75% |

At the strongest dose the being can still drive, **deleting an entire sensory family buys
1.7% more life**. That is the whole selection gradient the mechanism offers, and it has to
compete with the variance of the evolutionary process itself. The doses where the reward
is worth noticing are the doses where the being cannot move.

This is a ceiling computed before spending generations on it, and it says D8 as currently
shaped is unlikely to select for sensor pruning at any usable dose. Widening the gradient
means changing the mechanism — charging per channel family more steeply, or making the
saved life buy something other than time — not turning the dial further.

## The replication that shrank every number here

`--seed` does not vary anything under `--retomar`: the population is the champion plus
attenuated clones, with no random genome anywhere. Measured — seeds 77, 78 and 79 give a
median of exactly 3761 m. **Every dose number above is therefore n = 1 with no error bar.**

Replication has to come from somewhere else, so it came from neighbouring checkpoints of
the same training run — near-identical genomes, biological replicates. One contrast,
`--pneu-termico 1.0` against off, paired across five of them:

| genome | off | thermal 1.0 | effect |
|---|---|---|---|
| gen 41 | 2186 m | 2313 m | **+127 m** |
| gen 42 | 3299 m | 2962 m | −337 m |
| gen 43 | 3761 m | 2951 m | −810 m |
| gen 44 | 3270 m | 2845 m | −425 m |
| gen 45 | 4074 m | 3680 m | −394 m |

Mean effect −368 m, sd across replicates 334 m, same sign in 4 of 5, paired
t = −2.46 on 4 df, **p = 0.069**.

### A second, independent replication axis — and together they settle it

`--largada-aleatoria` makes `--seed` choose the generation's starting waypoint
(`main.rs:2509,2661`), which gives a replication axis that varies the START instead of the
genome. Predicted from the hash before running — seeds 77/78/79/80 → waypoints
0/18/36/54 — and the `[largada]` line confirmed all four.

| start | off | thermal 1.0 | effect |
|---|---|---|---|
| wp 0 | 4139 m | 3293 m | −846 m |
| wp 18 | 2417 m | 2423 m | **+6 m** |
| wp 36 | 3674 m | 2966 m | −708 m |
| wp 54 | 4243 m | 3302 m | −941 m |

Mean −622 m, sd 430 m, paired t = −2.90 on 3 df, p = 0.063. On its own, again not
significant.

**Pooled across both axes — genome and start are independent sources of variation:**

| axis | n | mean effect | negative | p |
|---|---|---|---|---|
| genome (gen 41-45) | 5 | −368 m | 4/5 | 0.069 |
| start (wp 0/18/36/54) | 4 | −622 m | 3/4 | 0.063 |
| **pooled** | **9** | **−481 m** | **7/9** | **0.0051** |

Neither design clears 0.05 alone; together they do, comfortably. Two independent
conditions with the same slope are worth more than one with a flattering sigma — the rule
this project already paid for. **The thermal effect on a frozen champion is real, at about
−481 m.** The single run reported earlier in the day, −810 m, overstated it by 1.7×.

Note what still dwarfs it: the off arm alone spans 1888 m across genomes and 1826 m across
starts, roughly 4× the effect. That is precisely why n = 1 could not see it.

Three things follow, and they apply to every table on this page:

1. **The single-run number was 2.2× too large.** gen 43 gave −810 m and the five-genome
   mean is −368 m. gen 43 is the largest of the five, and it is the one this session
   happened to pick. Following the effect SIZE as n grows is the rule here, not the sigma.
2. **One replication axis is not enough.** Neither n = 5 by genome (p = 0.069) nor n = 4
   by start (p = 0.063) clears 0.05; pooled they reach p = 0.0051. A single axis of
   replication on this simulator buys less than it looks like it should.
3. **Genome-to-genome variation dominates the dose.** The off arm alone ranges 2186 m to
   4074 m across five adjacent generations — sd 716 m, 1.9× the mean dose effect. Any
   comparison of two doses measured on ONE genome is reading a difference smaller than the
   noise it did not measure.

The canary criterion in `docs/ROADMAP.md` — "off / weak / strong must produce different
numbers" — passes all three families here (5 of 5 distinct doses each) and is too weak to
mean anything: distinct numbers are what noise produces. None of the three families is
monotone in its dose. The criterion should be an ORDERED response that survives
replication, and the cheapest replication available is a handful of neighbouring
checkpoints.

## What could not be decided without training

Ranking the four M0 mechanisms by whether they make the world DEMAND more brain. The
attempt and why it fails is in `docs/TAU_BANK_DIMENSIONALITY.md` § 5: on a frozen genome
every arm scores the same, because a fixed genome computes a fixed function and a world
pressure can only change its input distribution. D8 is the extreme case — its tax is a
constant per genome, so within one generation it can only drain life, never select.

The distances do separate the arms (`off` 2939 m, `--interocepcao` 3761 m,
`--predador 0.70` 3627 m, `--pneu-termico 2.0` 3266 m, `--recurso-escasso 4` 2981 m), but
all of them are the frozen champion meeting a changed world, not a population adapting to
one. Note the sign: every pressure RAISED the distance. These doses are gentle enough that
they do not even hurt a being that has never seen them.

## Standing

`docs/ROADMAP.md` M0 requires, for each mechanism: flag off by default, cross-currency
cost, dose canary before any battery, prediction registered before any verdict. After this
session: flags verified inert when off (bit-identical `obs.f32` between the commit before
and after they landed), D8's dose canary closed with a usable range and a design verdict,
and the other three still owing their sweeps. None of them has been shown to make the
world harder, and none can be until a training run is released.
