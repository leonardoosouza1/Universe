# `--retomar` loaded a random population and said nothing

Found 2026-08-25, fixed in `5c1ceab`. Filed because the *shape* of the failure is worth
more than the one-line fix: it produced plausible numbers for a brain nobody asked for.

## The bug

```rust
// src/bin/gpu_evolve/main.rs, before the fix
let mut latest: Option<(usize, String)> = None;
for g in 0..n_gens {                                    // <-- bounded by --gens
    let path = format!("{}/evolve_best_gen{:04}.bin", out_dir, g);
    ...
}
```

The search for a checkpoint was bounded by the number of generations **about to run**
rather than by what was on disk. With `--gens 1` it only ever looked at
`evolve_best_gen0000.bin`. Resuming a directory holding `evolve_best_gen0043.bin` found
nothing, and the `else` branch quietly substituted a random population.

The header had already printed `Inicio: retomando de checkpoint`. The contradiction —
`Nenhum arquivo de checkpoint encontrado — população aleatória` — was printed too, forty
lines later in a log full of track diagnostics, and read by nobody.

## What it cost

A day of analysis in this session ran against default genomes. `--observar` dumps being 0,
which under a working resume is the exact elite; under the broken one it was a random
brain that did not drive: median speed 0.069 m/s, never above 1.9 m/s, 12 m covered in
62 s, median throttle 0. `docs/TAU_BANK_DIMENSIONALITY.md` was written on that data and
had to be re-measured in full.

Two wrong diagnoses came before the right one, and both were reasonable:

1. *A regression from the hostile-world commit.* Ruled out properly — binaries built at
   `8afc088` and `75ca9eb` produce a **bit-identical** `obs.f32` on the same command line,
   sha256 `819aeabb6ca1180d…`. That also served as the no-op check that commit owed.
2. *Spawn-dependent fitness*, this project's known "fitness averages the PLACE, not the
   brain". Ruled out by `--same-start` not helping, and by the spread persisting between
   genomes that were supposed to be identical.

Both investigations were spent on a population that was never the champion.

## The tell that existed and was ignored

The run prints its own τ bank every generation:

```
[tau] campeão τ p10 2.000s · mediana 2.000s · p90 2.000s (espalha 1.0×)   <- broken
[tau] campeão τ p10 0.374s · mediana 1.572s · p90 4.000s (espalha 10.7×)  <- working
```

`2.000 s` on all three percentiles is `DECAY_DEFAULT`, the value a genome is born with.
An evolved bank cannot look like that. The evidence was on screen for every broken run.

## The fix

* The search reads the directory and takes the highest-numbered file, in
  `latest_checkpoint()` — extracted so it can be tested. Regression test
  `checkpoint_tests::finds_a_checkpoint_numbered_above_gens` puts `gen0043` in a temp
  directory and asserts it is found, that `gen0007` loses to it, and that malformed names
  are ignored.
* A missing checkpoint under `--retomar` is now a **hard error, exit 1**. Falling back to
  a random population is never what someone who typed `--retomar` wanted.

## Verified

`ereg-m4-m4-s77` gen 43, 48 beings, 20,000 ticks, tracks [8005, 8200, 9001]:

| | before | after |
|---|---|---|
| resume line | absent | `Retomando a partir de gen 43` |
| τ spread | 1.0× (default) | 10.7× (evolved) |
| median | 5 m | **2939 m (1.79 laps)** |
| max | 1684 m | **4289 m (2.62 laps)** |
| being 0 | 118 m @ 0.34 m/s | **2618 m, 41% of ticks above 5 m/s** |

The original record for that generation is `med=8551m max=9870m` over 48 co-evolved
beings. The gap that remains is a different question — one generation from a resumed
checkpoint starts on track 8005 (lap 1639 m) at ladder stage 0, not where generation 43
actually was — and it is now a question that can be asked, because the champion is
finally the thing under test.

## The rule this earns

**A silent fallback is worse than a crash.** A crash costs minutes; a fallback that
reports plausible numbers costs however long it takes someone to doubt the numbers.

And the cheap guard for anything reading `obs.f32`: **print the median speed of the being
being analysed.** `0.069 m/s` would have ended this in one line.
