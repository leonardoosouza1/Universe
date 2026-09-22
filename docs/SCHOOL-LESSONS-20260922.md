# Minimal school for organism-taxi-v1 — lesson contracts (Stage 2)

Pre-registered on 2026-09-22 before any code was written. Inputs:
`META/BRAINSTORM-ESCOLA-EVOLUCAO-UNIVERSE-2026-09-22.md` (incl. R2),
`META/RETORNO-ASTRA-TAXI-PAREDE-2026-09-22.md`, `docs/SCHOOL-CONTRACTS-20260922.md`
(Stage 1, commit `9f81d0d`). Results are appended at the end, never edited in place.

## 1. What the school is

A **bench** where every being is judged, per tick, on the GPU, against one declared
motor lesson. Same city physics, same body genome, same action mapper, same brain
runtime as the taxi world. The only things that change in bench mode:

| Element | Taxi world | School bench |
|---|---|---|
| Task channels 176..191 (v3) | passenger / destination | lesson sub-goal (direction + distance) |
| Passenger issuance (`taxi.wgsl post`) | on | **off** (`Params.school = 1`) |
| Taxi–taxi contact (`traffic_contacts.wgsl`) | on | **off** (proven isolation: no physical coupling; retina still renders other cars) |
| Births | lane spawn slots by depot distance | lesson-specific starts, one junction approach per being |
| Fitness | `1000·deliveries + task_score + money + …` | lesson score (section 4) |
| Online plasticity | `--plasticity` flag | unchanged, default **off** (first arm) |

Traffic lights, station stocks, refuel service, body metabolism, wear, damage and
death stay exactly as in the taxi world.

## 2. Lesson record (declared once per run, exported in `organism_genNNNN.json → school`)

`task_id`, `version`, `mode` (`bench` only in Stage 2; `life` is reserved), `env_seed`,
`observable_objective` (what the being sees: body-frame direction cos/sin + d/(d+100) of the
current sub-goal on channels 178/179/183, `has_task` on 180), `initial_condition`
(start pose per being), `start_tick` (0), `horizon_ticks`, `success`, `failure`,
`tolerances`, `assistance` (none: no host pilot, no override, no hidden steering),
`body_version` (`organism-taxi-v1` body, `gpu_evolve` binary SHA in `binary.sha256`).

## 3. Geometry (all in lane frame of the being's start)

`p0` = start position, `f0` = start heading (lane flow, unit), `l0` = `(-f0.z, f0.x)`.
`along = dot(pos - p0, f0)`, `lat = dot(pos - p0, l0)`.
Starts are lane centres 3 m from the street centre line, placed **before** a junction
along the flow, ≥ 18 m from any station, one junction approach per being, ordered by
depot distance, rotated per generation like the taxi spawns (`--fixed-spawn-slots` freezes).

| Lesson | id | start | target | success | explicit failures (code) |
|---|---|---|---|---|---|
| **move** | 1 | 60 m before junction centre | `T = p0 + 30·f0`, radius 3 m | `dist(pos,T) ≤ 3` **and** `propelled_ticks ≥ 16` (applied throttle ≥ 0.1 while speed ≥ 0.5 m/s) | contact (1), death (2), corridor `|lat|>6` or `along<-5` (3), born inside target (4), overshoot `along > 40` (5) |
| **stop** | 2 | 60 m before junction centre | area `A = p0 + 30·f0`, radius 4 m | phase 1 reached (`speed ≥ 3 m/s` before entering the area), then `speed ≤ 0.3 m/s` inside the area for **128 consecutive ticks** (2 s) while alive | 1, 2, 3, 4, 5 (`along > 40`), entered area without ever reaching 3 m/s (8) |
| **turn_near** | 3 | 37.5 m before junction centre | sub-goal 1: junction centre `J`; sub-goal 2: exit gate `E` on the near-side outbound lane, 12 m past `J` | entered the junction box (`|pos-J|∞ ≤ 7.5`), then `dist(pos,E) ≤ 3` with `dot(heading, e) ≥ 0.7` | 1, 2, 4, drove through the junction `along > along(J)+12` (6), reached the opposite exit gate (7) |
| **turn_far** | 4 | as 3 | as 3, exit gate on the far-side outbound lane | as 3 | as 3 |

"Near-side" = the outbound lane reached without crossing the junction centre line
(tight arc, radius ≈ 5–10 m); "far-side" = the one that crosses it (wide arc).
Both are legal flow lanes by the city rule (`organism_city.rs::build_city`).
Time-out at `horizon_ticks` = truncation (status 3), counted in the denominator.

Shortcuts and how they are rejected:

| Shortcut | Rejection |
|---|---|
| born inside the target | code 4 at tick 0 (target is ≥ 30 m away by construction; checked anyway) |
| pushed by another being | contacts off in bench + `propelled_ticks` requirement (move) |
| crash to stop | any contact bit (walls 1/2/4/8, taxi 16) = failure 1 before the hold can complete |
| through obstacles / off road | contact = failure 1; corridor = failure 3 |
| dying = stopped | dwell counts only while alive; death = failure 2 |
| repeated credit | status is written once; judge freezes at the outcome tick |
| oscillation farming | progress = `1 − best_remaining / initial_remaining`, monotone (best ever), bounded [0,1] |
| driving straight through (turn) | failure 6; wrong exit failure 7 |

## 4. Score (per being, bounded)

```
score = 1000·success + 500·progress − 200·failed − 100·(1 − integrity) − 100·(1 − alive)
```

`failed` = any explicit failure code. Truncation keeps `500·progress`. Consumption
(`energy_spent_j = energy_at_start − energy_now`, `distance_m`) and safety
(`contact_ticks`, `collision_energy_j`, `impacts`) are **measured and exported, not
scored** in Stage 2, so that Stage 3 can test incentives against this baseline.
Fitness for selection in school mode = `score`.

Competence: per being and generation, one attempt (`success ∈ {0,1}`); per population
and generation, `p̂ = (S + 1)/(T + 2)`; per lineage, the elite (`genomes[0]`, carried
unmutated) is the only identifiable individual across generations and is reported as
`elite_same_genome` with its own attempt record.

## 5. Judge (GPU, per being, after `taxi.post`, before orientation feedback)

Reads: physics **output** (pose 36/38, velocity 0/2, heading 40/42, contact mask 39,
collision energy 55), physics **input** (applied throttle 51), body (alive 29,
integrity 8, energy 0, distance 98). Writes: `school` buffer (40 f32 per being, see
`src/gpu/school.rs`; slots 32/33 freeze energy and odometer at the outcome so the
judged span is reported apart from the whole episode) and body 70/71 (sub-goal absolute), 80/81 (sub-goal delta),
90 (`has_task` = 1), 65 (onboard = 0) so the v3 observation and the reference
witness both see the lesson without any new channel.

## 6. Pre-registered validation (in order; stop at the first contract failure)

1. `wgsl_validation` parses `school.wgsl`.
2. GPU fixtures (`src/gpu/school_tests.rs`, real Radeon via Vulkan, `--ignored`):
   synthetic pose streams through the judge only — positive path per lesson +
   every failure code + truncation + frozen status after success + not-propelled move
   + hold interrupted (stop) + misaligned gate (turn). Expected: each case yields
   exactly its declared status/code, no other.
3. Reference witness (`--taxi-reference-witness`, one being) on each lesson, one
   run each: the lesson must be achievable by the explicit controller (success,
   zero contact). A witness failure is reported as lesson infeasibility, not hidden.
4. Smoke evolution, 3 generations × 12 beings, lesson `move`, horizon 1280 ticks
   (20 s; ceiling at 10 m/s = 200 m ≫ 30 m): `school_result_genNNNN.json` present,
   statuses and codes populated, fitness dispersion > 0, panel opens the run.
5. Only then: Stage 4 battery (uniform vs adaptive curriculum, 3 seeds).

Binary/config recorded per run in `binary.sha256` and `command.txt`.

## 7. Results (2026-09-22, Stage 2 delivered)

Binary: `target/debug/gpu_evolve` — `ba2e3ef5…` (v1) and `f1078f49…` (v2, record
stride 40 with consumption frozen at the outcome; simulation unchanged). Frozen copies in
`/media/llm-data/scratch/gpu_evolve-school-{ba2e3ef5,f1078f49}`. GPU: RX 6750 XT via
Vulkan (`VK_ICD_FILENAMES=radeon_icd.json`, `unset DISPLAY`, `RAYON_NUM_THREADS=1`).
All commands in `/media/llm-data/universe-runs/school-lessons-20260922/run_witness_and_smoke{,_v2}.sh`.

### 7.1 Files changed

New: `src/gpu/school.wgsl`, `src/gpu/school.rs`, `src/gpu/school_tests.rs`,
`src/organism_school.rs`, this document. Modified: `src/gpu/taxi.{rs,wgsl}` (`school`
param stops passenger issuance), `src/organism_city.rs` (`street_centres_m`),
`src/gpu/mod.rs`, `src/lib.rs`, `src/bin/gpu_evolve/{context,organism_tick,main}.rs`
(`--school`, `--school-mode`, `--school-horizon`; judge after `taxi.post`; contacts off
in bench; `school_result_genNNNN.json`; fitness = score), `tests/wgsl_validation.rs`.
Defaults unchanged: without `--school` the taxi behaves as before.

### 7.2 Pre-registered steps, expected vs observed

| Step | Expected | Observed |
|---|---|---|
| 6.1 `wgsl_validation` | parses | 14/14 pass (`cargo test --test wgsl_validation`) |
| 6.2 GPU fixtures | each case yields exactly its status/code | 10/10 pass, 2.8 s (`cargo test --lib school_tests -- --ignored --test-threads=1`); covers success per lesson, codes 1–8, truncation, frozen record after outcome, sub-goal switch, consumption frozen at outcome |
| 6.3 witness per lesson | success, zero contact | 4/4 success, 0 contact (table below), ~10 s wall each |
| 6.4 smoke evolution | results present, codes populated, dispersion > 0, panel opens | all met (table below), 34 s wall for 3 gens × 12 × 1280 ticks; v1 and v2 trajectories identical to the digit (same seed, same kernels) |

Witness (`--taxi-reference-witness --beings 1 --ticks 1920 --seed 7`, v2 records):

| Lesson | start | outcome tick | progress | score | v_max m/s | judged m / episode m | judged J / episode J |
|---|---|---|---|---|---|---|---|
| move | (99,156) −Z | 399 | 0.901 | 1450.3 | 5.76 | 27.0 / 32.2 | 389 654 / 392 853 |
| stop | (99,156) −Z | 659 | 1.000 | 1499.9 | 5.76 | 30.5 / 32.2 | 390 304 / 392 853 |
| turn_near | (133.5,291) −X | 1294 | 0.941 | 1470.4 | 6.19 | 42.5 / 46.2 | 519 204 / 520 456 |
| turn_far | (133.5,291) −X | 1309 | 0.941 | 1470.7 | 6.11 | 43.6 / 47.2 | 548 543 / 549 765 |

Progress < 1 on success is by construction: success fires at the goal radius, so the
best remaining distance equals the radius (3 m of 30 m = 0.9; 3 m of 51 m ≈ 0.94).

Smoke (`--school move --beings 12 --gens 3 --ticks 1280 --seed 7`, run
`school-lessons-smoke-move-v2`, ceiling 200 m at 10 m/s ≫ 30 m):

| Gen | success | failed (codes) | truncated | deaths | p̂ | progress median | score median / max | fitness σ | distinct |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0/12 | 4 (contact 1, corridor 3) | 8 | 0 | 0.071 | 0.015 | 2.3 / 152.1 | 86.3 | 9/12 |
| 1 | 3/12 | 1 (corridor 1) | 8 | 0 | 0.286 | 0.218 | 14.4 / 1452.1 | 615.8 | 9/12 |
| 2 | 5/12 | 2 (contact 2) | 5 | 0 | 0.429 | 0.855 | 227.0 / 1451.5 | 679.9 | 10/12 |

Successful evolved beings (gen 2): outcome ticks 408–1196, judged distance 27.1–28.4 m,
judged energy 231–389 kJ (witness 390 kJ), judged contacts 0. Starts of the successes
cover −Z, −X and +Z approaches. Panel: run registered in `lab.db`
(`package_run … --db`, exporters from `target/debug` via `UNIVERSE_S2_BIN_DIR`), the
observatory lists it and opens generation 2 — screenshot
`panel-smoke-move-v2-3440x1240.png` in the run directory.

### 7.3 Observations that change the next steps

1. **Same genome, different start, different verdict.** The gen-0 elite (slot 7,
   start (99,348) −Z, progress 0.704, corridor failure at tick 1156, throttle sum 145
   over the episode) was re-evaluated as slot 0 in gen 1 at (285,228) +Z: throttle sum
   0.0 over all 1280 ticks, stationary 1280 ticks, score 0. The gen-1 elite scored
   1452 → 1451 across two different −Z starts. Selection over rotated starts is the
   intended pressure, but any per-individual claim needs the same start; Stage 4 must
   evaluate each genome on the full start set (or `--fixed-spawn-slots`) before ranking.
2. **Consumption and safety are episode-wide unless frozen.** The v1 record reported
   odometer 121.9 m for a 30 m lesson because the being kept driving after success and
   the host read the live body. v2 freezes energy and odometer at the outcome and reports
   both spans; post-outcome contacts (up to 10 ticks in gen 2) are visible only in the
   lifetime counters. The bench does not park a being after its outcome.
3. **Genome exporter must match the producer of the file.** The lab_api packager fails on
   child genomes (99 332 B) with the release `genome_export` of 2026-09-21; the current
   tree parses them. Packaging used `UNIVERSE_S2_BIN_DIR=target/debug`.

### 7.4 Limitations (declared, not hidden)

- Debug binary; host side is slower than release. Simulation results do not depend on it.
- Bench isolation is partial: taxi–taxi contacts and passengers are off, but the retina
  still sees the other cars and traffic lights are neither judged nor removed.
- Life mode (`--school-mode life`) is reserved; the flag exits with code 2.
- Competence per lineage is limited to the elite (`genomes[0]`); other lineages are not
  identifiable across generations without a lineage id in the population file.
- `school_result_genNNNN.json` is written and packaged but not yet surfaced by lab_api or
  the observatory view (Stage 5 work); the panel shows the organism metrics of the run.
- Three generations of one seed show a trend, not learning: the pre-registered claim is
  only that the loop is closed and the judge discriminates.

### 7.5 Next test

Stage 3 incentives on top of this judge (phase progress with best-distance memory, energy
feedback asymmetry, plasticity apply-only mode), then the Stage 4 battery: uniform vs
adaptive curriculum, 3 seeds, equal being-ticks, each genome evaluated on the full start
set, train/validation/test starts, release build.
