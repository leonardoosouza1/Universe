# The reloaded champion does not drive

Measured 2026-08-25. Reproducible, five configurations, no training run involved.

## The fact

`ereg-m4-m4-s77` reached, at generation 43, `med=8551m (3.598V) max=9870m` over 48 beings
(`run.log`). Reloading that generation's own checkpoint and evaluating it produces:

| beings | tracks | `--same-start` | best in run | **being 0 (the exact elite)** |
|---|---|---|---|---|
| 8 | 8005 | no | 42 m | **122 m** |
| 8 | 8200 | no | 56 m | — |
| 8 | 8005,8200,9001 | no | 42 m | **122 m** |
| 48 | 8005,8200,9001 | no | 1684 m | **118 m** |
| 48 | 8005,8200,9001 | yes | 1429 m | **124 m** |

Being 0 is the elite, byte for byte: `main.rs:1904` does `pop.push(best.clone())` with no
mutation, and the loop order is "N ticks → evaluate → select → mutate" (`main.rs:5`), so
generation 0 evaluates the population exactly as assembled.

It covers 118-124 m every time, against the 9870 m the same bytes were credited with.
Meanwhile a sibling in the same run — an *attenuated clone* of that elite, every weight
multiplied by 0.85 and nothing else changed (`main.rs:1938-1946`) — reaches 1684 m.
**Same brain, 0.85 of the gain, 14× the distance.**

## What it is not

* Not a regression from the hostile-world commit. The binary built at `8afc088` (before
  it) and the binary at `75ca9eb` produce **bit-identical** `obs.f32` on the same command
  line, same SHA256 `819aeabb6ca1180d…`. That doubles as the no-op check that commit owed.
* Not a serialiser field drop, the failure this repository has had twice before.
  `genome_io.rs` writes every field the `Genome` struct declares — `l1`, `l2`, `slow`,
  `tau`, `sensors`, `kan_knots`, `neuron_alpha`, `neuron_period_l1`,
  `neuron_threshold_l1`, `neuron_activation`, `gates_w`. Verified field by field.
* Not the track (42 m on 8005 vs 56 m on 8200), not population size for being 0
  (122 m at 8 beings, 118 m at 48), not `--same-start` (124 m with it).

## What it probably is, and what would decide it

Population size moves the BEST being by 40× (42 m → 1684 m) while leaving being 0 flat.
That is the signature of a spawn-dependent evaluation, which this project has already paid
for once — `project_fitness_media_a_largada`: "each being was born at a different
waypoint, so fitness averages the PLACE, not the brain." The champion's recorded 9870 m
may be the number of a lucky spawn rather than of a genome, in which case nothing is
broken except the belief that a saved `.bin` carries a reproducible skill.

Against that: `--same-start` did not rescue being 0. If every being starts at the same
place, the 47 identical attenuated clones should also score identically — and they do not
(median 0 m, max 1429 m). Something still differentiates beings that share a genome and a
starting point, and finding it is the next step.

Two experiments, both cheap, neither a training run:

1. Evaluate the same genome N times in the SAME slot and in DIFFERENT slots. If distance
   depends on the slot index, the evaluation is not a function of the genome and every A/B
   in this repository that compares reloaded champions is measuring slot assignment.
2. Re-evaluate the champion at the exact generation and world seed of its own record
   (`gen_offset 43`, the ladder state of that run), not at generation 0. If the number
   comes back, the checkpoint is fine and the resume path is failing to restore the world.

## Why this is filed before it is solved

It invalidates measurements that were already taken. `docs/TAU_BANK_DIMENSIONALITY.md`
sections 2-5 were computed from `--observar` output, and `--observar` dumps being 0 —
which is to say, the parked one. Anything else in this repository that reads `obs.f32`
after `--retomar` is under the same suspicion until this is closed.

The cheap guard, until then: **print the median speed of the being you are analysing.**
`0.069 m/s` would have stopped a day's analysis in one line.
