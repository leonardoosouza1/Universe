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

**Canary passed**: every dose gives a different number, monotonically, including the off
arm. **Mortality is 0% across the whole sweep** — this pressure STALLS rather than kills,
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
