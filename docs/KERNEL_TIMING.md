# Kernel timing — the first per-kernel GPU measurement this project has had

**Measured 2026-08-27** (re-verified during the final whole-branch review; original
instrument built and measured 2026-08-26), RX 6750 XT (RADV, `env -u DISPLAY`), branch
`race-world-v3`, `gpu_evolve --perfil-kernel`.

This document is Phase 0's tracked home. Phase 0's headline result previously survived
only in commit message `098ff4f` and an untracked, gitignored task report
(`.superpowers/sdd/`) — this is the fix: a positive result with nowhere to live, while
Phase 2's negative result got a 594-line document (`SCREEN_CALIBRATION.md`).

---

## 1. Why no external profiler works here

`rocprofv3` produces an **empty trace** against `gpu_evolve`. The reason is architectural,
not a missing flag: `rocprofv3` instruments HIP/HSA kernel launches, and this project's
GPU path runs on `wgpu` over **Vulkan/RADV** — a different driver stack that HSA-level
tooling cannot see at all. No amount of `rocprofv3` configuration fixes this; it is
watching the wrong API.

`wgpu` has its own timestamp mechanism (`wgpu::Features::TIMESTAMP_QUERY`), and the
plumbing for it — `timestamp_writes` fields on every compute pass descriptor — was
already present in this codebase and unused, set to `None`:

- `src/gpu/core.rs:503`
- `src/gpu/cortex_sheet.rs:521` (also `:575`, `:624`)
- `src/gpu/retina_digest.rs:249`

These three sites are still `None` today and that is correct, not an oversight: they
belong to `RetinaGpu`/`CortexSheetGpu`/`RetinaDigestGpu` call paths used by *other*
binaries (the Bevy visualizer, `gpu_validate`, `cortex_evolve`), which `gpu_evolve` never
constructs. `gpu_evolve` opens its **own** device at `src/bin/gpu_evolve/context.rs:254`
and, since this branch, times it directly (§3 below) by bracketing dispatches with
`CommandEncoder::write_timestamp` from `context.rs` itself, not via those three
`timestamp_writes` fields. Wiring those other three sites would only matter if one of the
other binaries needed the same instrument — no test in this branch justifies that, so
they were left alone.

Before this branch, **no per-kernel timing existed for this engine's actual GPU path at
all.** Every optimisation proposal that named a kernel as "the" bottleneck was a guess.

## 2. How to run it

```bash
./target/release/gpu_evolve --retomar --out <dir> --gens 1 --seres <N> --ticks <T> \
    --width 28 --out-beta 0.0 --pistas <seed> --seed 77 --perfil-kernel <N_ticks>
```

`--perfil-kernel <N>` (default `0` = off) prints one block of per-kernel milliseconds
every `N` ticks: `retina`, `perception`, `retina_digest`, `brain` (absent, not `0.000`,
on ticks where `--brain-skip` keeps it from firing), `slow.gather`, `slow`, then `TOTAL`
(sum of the GPU-timestamped kernels), `physics.step` (CPU wall-clock around a separate,
synchronous submit+readback — `PhysicsGpu::step` opens its own encoder, so no timestamp
from the main encoder reaches inside it), and `gaps` (`tick wall − TOTAL − physics.step`,
from a CPU `Instant` wrapping the whole per-tick function body).

**Tick 0 is excluded from every sample.** It is the JIT warm-up tick
(`main.rs:2857`, see that comment) and is not representative — an earlier, retired
4-sample figure (19.68%) turned out to include exactly this kind of outlier undiluted.
`context.rs`'s gating condition is `n_tick > 0 && tick_idx > 0 && tick_idx % n_tick == 0`;
the `tick_idx > 0` term is what keeps tick 0 out.

The flag is off by default and verified inert: with it off, `GpuTimer`'s `begin_frame`,
every `mark_begin`, and `resolve` never run, so a baseline taken with the flag off pays
nothing for having the capability.

## 3. The measured table

Command that produced this table:

```bash
D=/media/llm-data/scratch/kt_final
./target/release/gpu_evolve --retomar --out $D --gens 1 --seres 768 --ticks 310 \
    --width 28 --out-beta 0.0 --pistas 8005 --seed 77 --perfil-kernel 10
```

(genome `ereg-m4-m4-s77/evolve_best_gen0043.bin`, copied into `$D` first via `--retomar`.)
31 steady-state samples (ticks 10, 20, … 310; tick 0 excluded as above).

| kernel | median (ms) | min | max | measured by |
|---|---:|---:|---:|---|
| `retina` | 0.8332 | 0.3448 | 1.0747 | GPU timestamp |
| `perception` | 0.4155 | 0.2289 | 0.5329 | GPU timestamp |
| `retina_digest` | 2.3132 | 0.3777 | 2.7614 | GPU timestamp |
| `brain` | 0.6070 | 0.4427 | 3.7944 | GPU timestamp |
| `slow.gather` | 0.0054 | 0.0038 | 0.0089 | GPU timestamp |
| `slow` | 0.0113 | 0.0074 | 0.0184 | GPU timestamp |
| **TOTAL (GPU kernels)** | **4.4521** | 1.4053 | 7.4714 | sum of the above |
| `physics.step` | 0.3135 | 0.2440 | 0.6254 | CPU wall (blocking submit+readback) |
| **combined (TOTAL + physics.step)** | **4.8226** | — | — | — |
| tick wall clock (from the `gaps` line) | 29.5676 | 22.6712 | 37.6992 | CPU `Instant`, whole `tick()` body |

`brain` and `retina_digest` show the widest spread (8.6× and 7.3× respectively between
min and max) — this is not measurement noise from the instrument; it tracks GPU clock
state. This card idles roughly 85% of every tick (§5) and downclocks (`sclk` drops to a
low P-state) during that idle time, so a kernel sampled right after an idle stretch runs
slower than one sampled while the clock is still up. **Never quote a single sample from
this table as "the" cost of a kernel — quote the median over ≥25 samples, with the
range**, exactly as done above.

This branch's independent Task-3 measurement (2026-08-26, same config, a different
310-tick window) landed in the same band: TOTAL median 3.8371 ms, physics.step median
0.2762 ms, combined 4.1133 ms, clean wall-clock median 26.867 ms (range 26.267–27.600
over 3 runs) — see commit `098ff4f`'s message and `.superpowers/sdd/` (gitignored,
not durable) for that run's own table. The two measurements, a day apart with the same
hardware, agree in kind and disagree by run-to-run clock-state variance, which is the
point of §5's warning to quote ranges.

## 4. Reconstruction

Acceptance criterion (`docs/superpowers/specs/2026-08-26-gpu-resident-scenarios-design.md`
§2): the measured parts must reconstruct the wall-clock tick within 10%. Since `gaps` is
defined as the remainder (`wall − TOTAL − physics.step`), kernels + physics + gaps sum to
wall clock by construction; the informative number is what **fraction of the tick the GPU
kernels plus physics's blocking readback account for**, because that is what decides
whether Phase 1 (GPU-resident state, K ticks per submit) is aimed at the right target.

**Reconstruction (kernels + physics.step, as a percentage of the tick): median 15.8%,
per-sample range 7.4%–23.5%** (this run, 31 samples). Task 3's own post-review sweep
(different day, different window) gave median 15.3%, range 14.9–15.7%. Combined, the
honest statement is **a median around 13–16%, quoted as a band because the underlying
distribution is genuinely wide and tracks GPU clock state — never as a point estimate.**
An earlier, retired 4-sample figure (19.68%/16.44%) is exactly the failure this band
warns against: four samples from a distribution this wide can land almost anywhere.

**The remaining ~84–87% is not idle GPU time waiting on nothing — it is
per-being CPU work**, not per-submission overhead (§5).

## 5. The cost model, and where the money actually goes

Fitted from a scaling sweep (12/48/192/768 beings, 400 ticks each), reproduced by two
independent measurements during this branch's review (Task 3's reviewer: `0.86 +
0.0324 × n_beings`; the original design spec's own sweep: `0.88 + 0.0333 × n_beings`):

```
ms per tick ≈ 0.86 + 0.032 × n_beings
              ----   -----
              fixed  marginal (per being)
```

A quick single-sample-per-point re-check run today (`n = 12, 96, 384, 768`, 400 ticks
each, includes each process's own ~1 s adapter/pipeline-compile startup baked into the
intercept, so not a clean re-derivation) gave `2.26 + 0.039 × n_beings` — a higher
intercept for the reason just stated, but the **same qualitative shape**: a small fixed
term, a marginal term that dominates by roughly an order of magnitude at 768 beings.

**What this means for Phase 1's rationale.** At 768 beings the fixed/per-submission term
is `0.86 ms`, about **3%** of a ~27–30 ms tick. "K ticks per submit" (the mechanism Phase 1
originally sold itself on, per
`docs/superpowers/specs/2026-08-26-gpu-resident-scenarios-design.md` §3) collapses that
fixed term across K ticks — it recovers **3% at best, not a multiple of it.**
**~85% of the tick scales with `n_beings` instead**: the per-being CPU readback inside
`PhysicsGpu::step` and the per-being Rust loops in `context.rs::tick()` that run before
and after it (steer/stagnation bookkeeping, yaw-rotation, state write-back, buffer/
bind-group rebuilds). None of that shrinks by batching GPU submissions; it shrinks only
by moving that state and those loops onto the GPU, which is Phase 1's *other* stated
goal ("State moves to the GPU"). The direction of Phase 1 — get the tick off the CPU —
is unaffected; the mechanism its own spec named is not where the payoff is.

**Brain cost, directly, not by subtraction.** The ablation-table approach (turning the
brain off with `--sem-lento`/`--brain-skip` and subtracting) is retired — see
`docs/superpowers/specs/2026-08-26-gpu-resident-scenarios-design.md` §1, the `share`
column is struck out there. This instrument answers the same question without dividing
two noisy wall-clock numbers: `brain + slow.gather + slow` medians to **0.6237 ms**
(0.6070 + 0.0054 + 0.0113) of a ~28–30 ms tick — **about 2.2%**, not the 21% the retired
ablation implied. Optimising the brain has almost no ceiling left; the retina/perception/
physics/CPU-loop chain is where the tick actually goes.

## 6. Conclusion, for whoever plans Phase 1

- Per-kernel GPU time reconstructs **~13–16%** of the tick (quote the band, not a point;
  it moves with GPU clock state).
- The brain (fast + slow + gather) is **~2.2%** of the tick — not 21%. Optimising it is
  optimising a fraction most of which is already gone.
- The fixed, per-submission cost is **0.86 ms ≈ 3%** of the tick. "K ticks per submit"
  recovers at most that.
- **~85% of the tick scales with `n_beings`**: per-being CPU readbacks and per-being CPU
  loops, not GPU kernel execution and not per-submission overhead. This is the real
  target for Phase 1 — moving that state and those loops onto the GPU — not batching
  submits and not the brain.
