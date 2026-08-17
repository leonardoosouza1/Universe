# `src/bin/gpu_evolve/` — the production evolution binary

The headless loop that actually produces results: it builds a world, runs N generations
of M beings for T ticks each on the GPU, selects, mutates, and writes the best genome per
generation. Everything else in `src/bin/` is a tool that inspects what this binary made.

## Files

| File | Role |
|---|---|
| `main.rs` | CLI parsing, the generation loop, fitness, curriculum, reports (2 582 lines — the biggest file here and the standing refactor target) |
| `context.rs` | `GpuCtx`: owns the device and every buffer; `tick()` lives here |
| `statics.rs` | Process-global flags read inside the hot loop, with a writer/reader ownership table |
| `types.rs` | `BeingState` — per-being CPU state. Deliberately **not** the same type as `gpu_loop`'s (20 fields vs 5); do not unify them |
| `pilots.rs`, `piloto_linear.rs`, `piloto_forja.rs` | Diagnostic drivers that replace the brain behind a flag, to separate "the body cannot" from "the brain cannot" |
| `escada.rs`, `fantasma.rs`, `pistas.rs` | Difficulty ladder (pure state machine), ghost of the previous champion (pressure paid in life, never in fitness), deterministic track rotation |
| `audit.rs`, `painel.rs` | Vision audit (`--auditar-visao`); optional HTTP panel (`--painel`), zero cost when off |
| `tests.rs` | Genome round-trip regression tests — **do not delete** |

## The generation loop

`main.rs:1176`, one iteration per generation, in this order:

1. Track rotation (`:1182`) and curriculum advance when the circuit is dominated (`:1191`)
2. Leader replay recording (`:1245`) and genomes uploaded into the engines (`:1268`)
3. **Recurrent state reset** (`:1355`) — see the traps below
4. Initial being state: one start waypoint for the whole generation (`:1373`)
5. T ticks of simulation (`:1510`), each one a `GpuCtx::tick()`
6. Wall cost (`:1751`), generation statistics (`:1766`), champion reproduction (`:1802`)
7. Best genome saved as `evolve_best_gen<NNNN>.bin` (`:1891`), report printed (`:1903`)
8. Held-out evaluation (`:2415`), then selection + mutation (`:2488`)

## Flags vs global state

CLI arguments are parsed in `main()` (`main.rs:368` onward) into locals. The subset that
`GpuCtx::tick()` needs is published through `statics.rs` instead of being threaded through
the tick signature — the header there explains why (this binary grew flag by flag; every
experiment would otherwise widen `tick()`), and carries the **ownership table: which static
is written by whom and read by whom**. Keep that table current when you add a static, and
do not promote any of them into the library without wrapping them in a config struct first.

Defaults reproduce the old behaviour, always. A flag that changes the distribution of a
brain input — `--visao-periferica-real` (`main.rs:715-718`) is the worked example — is off
by default and needs a paired A/B, because an existing champion was selected under the old
distribution.

## Traps

**Tick ordering: the pilot override must precede the assembly of `phys_states`**
(`context.rs:989-1043`). The block used to sit *after* the `.collect()`, and in that order
the formula pilot's command never reached the motor — `phys_states` had already been built
from the brain's intent, and the only thing the override still touched was `mem0`
(perception channel 120). The 2026-08-16 batteries therefore measured the champion with one
contaminated memory channel while believing they were measuring the clone driving: applied
commands were bit-for-bit identical to the champion's for the first 576 ticks, while the
recorded intent already differed by 0.60 from tick 0. The guard is the test
`piloto_entra_antes_da_fisica` (`context.rs:1440`), which reads its own source file with
`include_str!` and compares the byte offsets of two marker comments — `MARCA-ORDEM-PILOTO`
(`context.rs:1015`) and `MARCA-ORDEM-FISICA` (`context.rs:1043`) — and also asserts there is
exactly one override block, because during the fix there were two. Do not delete or rename
those markers.

**Recurrent state crosses the generation boundary unless it is reset** (`main.rs:1355-1373`).
`buf_cfc` and `buf_fatigue` live on the GPU and were never zeroed, so a child inherited its
parent's CfC state in silence and the champion's re-evaluation was not deterministic. Same
family, found later: `last_intents` and `brain_tick` also survived the boundary, so the first
ticks of a new generation reused the previous generation's last decision. All of them are
reset together at that site.

**Changing the report is not changing the selection** (`main.rs:2488-2496`). The selection
call had `&dist_acum` hardcoded while only the printed metric had been switched, so
`--fitness dist` and `--fitness prog` produced identical numbers to the digit — one edit away
from being recorded as "fitness does not matter". Identical numbers where they should differ
are an alarm, always.

**Fitness measures where a being was born unless the start is fixed.** Beings used to be
spread across waypoints, but track sections are not equally hard: the same genome
re-evaluated the next generation scored ~72% of the previous run. `--same-start` exists for
this; median distance went 173 m → 365 m and mortality to 0% once the whole generation
started from one waypoint.

**Dead beings are pinned to `throttle=0, brake=1`.** Any average over all ticks includes
corpses. Say who the statistic is about.

## Commands

```bash
# fast smoke generation — seconds, enough to prove two conditions diverge
env -u DISPLAY ./target/release/gpu_evolve --gens 2 --seres 12 --ticks 600 \
    --seed 7 --same-start --out /your/own/scratch/smoke

# mandatory after touching Genome, save_genome, load_genome or the tick order
cargo test --release --bin gpu_evolve
```

`--out` defaults to `.` (`main.rs:395`) — always pass your own directory, never `/tmp` and
never inside the repository. Replays are the expensive artefact: every generation × 16
races produced 31 GB once, so on a long run record only the first, middle and last.
