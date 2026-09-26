## What this changes

<!-- One or two sentences. What behaviour is different after this merges? -->

## Why

<!-- The reason, not the diff. If this fixes something, say what was wrong and how you
     know — file:line, or the measurement that exposed it. -->

## Measurements

<!-- Every number gets the command that produced it. If you did not measure something,
     write "not measured" — never a description standing in for a measurement.

     Behaviour changes need a PAIRED A/B: same seeds, same --gens, same --ticks, same
     tracks, >= 3 seeds, differing ONLY by the new flag.

     Example:
       Median distance 173 m -> 365 m, mortality 0%.
         ./target/release/gpu_evolve --gens 40 --seres 60 --seed 7 --same-start \
             --out /media/llm-data/universe-runs/same-start-s7
         (repeated with --seed 8 / --seed 9; medians 361 m / 370 m)
-->

- [ ] This PR changes no runtime behaviour (docs, translation, formatting, tooling) —
      measurements not applicable.

## Checklist

- [ ] `cargo build --release` is green
- [ ] `cargo test --release` is green
- [ ] `cargo test --release --bin gpu_evolve` is green — **required** if `Genome`,
      `save_genome` or `load_genome` changed
- [ ] `bash scripts/check_language.sh` is green
- [ ] Everything I wrote is in **English**, and the Portuguese comments inside the hunks
      I touched are translated **with their numbers and file:line references intact**
- [ ] `cargo test --lib wgsl_static` passes, if I edited a `.wgsl` file
- [ ] Any physics / perception / fitness change is **behind a flag that defaults to off**,
      the flag is documented where it is parsed, and a paired A/B is above
- [ ] No file grew past the line limit in `docs/STYLE.md`
- [ ] I added no new clippy errors in the files I touched

<!-- See CONTRIBUTING.md for what each of these means and why it exists. -->
