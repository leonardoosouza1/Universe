# School Stage 4 — fair comparison of curricula (2026-09-22)

Pre-registration for item 4 of the 2026-09-22 program. Written before the battery
was read; results in §4. Branch `organism/school-20260922`, binary
`gpu_evolve-stage3-v3` (debug, sha256 `bc40df0b…`, frozen in
`/media/llm-data/universe-runs/school-incentives-20260922/`), GPU RX 6750 XT via Vulkan.

## 1. Design

| Element | Value |
|---|---|
| Arms | `--school-curriculum adaptive` (advance when p̂ ≥ 0.5 for 2 gens, default) vs `uniform` (move, stop, turn_near, turn_far cycling) |
| Seeds | 7, 11, 13 — `--seed` also seeds the city, so each seed is a different city and its own fold split; arms are paired within a seed |
| Budget per arm-seed | 16 beings × 16 generations × 1280 ticks = 327 680 being-ticks, identical in both arms |
| Evolution starts | `--school-start-fold train` (candidate index % 3 == 0) |
| Held-out evaluation | champion of gen 0 (control: one generation of selection, no evolution) and of gen 15, each copied 16× (`--resume --resume-mix 1.0 --gens 1`) on the `validation` and `test` folds of all four lessons |
| Bench mode | passengers and taxi–taxi contacts off; fitness = lesson score; online plasticity off |
| Logged per generation | successes / failures / truncated (timeouts) / deaths, p̂ = (S+1)/(T+2), failure codes, `[curriculum]` state |
| Replays | gens 0 and 15 only (`--replay-every 15`) |
| Stop criteria | any non-zero exit stops that arm-seed; a seed whose fold capacity is below 16 for any lesson is skipped in phase 0 (turn lessons have exactly 16 candidates per fold in the seed-7 city) |

Script and log: `/media/llm-data/universe-runs/school-comparison-20260922/battery.{sh,log}`.
Estimated cost: training 6 × ~2.2 min, evaluation 96 runs × ~7 s ≈ 11 min, < 1 GB on
`/media/llm-data` (8.7 GB free at launch).

## 2. Metrics

- Train trajectory: p̂ per generation and arm (adaptive reports the lesson it is on).
- Held-out competence: successes / 16 per lesson × fold for the gen-0 and gen-15
  champions; the paired difference (gen 15 − gen 0) per arm and seed is the learning
  effect; the arm difference on the same seed is the curriculum effect.
- Safety and consumption: deaths, failure codes (`contact`, `corridor`, `overshoot`,
  `drove_through_junction`), truncations, per-run wall time.

## 3. Pre-registered expectations

| Claim | Expected | Refuted if |
|---|---|---|
| Learning | gen-15 champion beats gen-0 champion on the validation fold of `move` in ≥ 2 of 3 seeds, both arms | ≤ 1 seed improves, or the improvement is within ±1 success |
| Generalisation | validation and test folds agree within 3 successes of 16 | test − validation differs by > 3 in most cells |
| Curriculum | uniform ≥ adaptive on the mean of the four lessons (adaptive is expected to stay on `move` under the default 0.5/2 rule); adaptive ≥ uniform on `move` alone | the reverse in ≥ 2 of 3 seeds |
| Safety | zero deaths; `contact` is the dominant failure code | any death (would mean the bench lets a being die) |

## 4. Results

Battery ran 20:41–21:04 on 2026-09-22, binary `gpu_evolve-stage3-v3` (debug, sha256
`bc40df0b…`), adapter `AMD Radeon RX 6750 XT (RADV NAVI22)`, all 102 runs exit 0, no
seed skipped. Analysis: `analyze.py` → `analysis.json`, `report.html`
(`report-3440.png`) in `/media/llm-data/universe-runs/school-comparison-20260922/`.
Wall: 6 training runs 75–77 s each (16 gens × 16 beings × 1280 ticks, ~4.8 s/gen),
96 evaluations 5.6–5.9 s each; ~19 min of GPU in total.

### 4.1 Train trajectories (successes of 16 on the train fold, per generation)

| arm-seed | gen 0 … 15 (lesson: successes) | lessons visited |
|---|---|---|
| adaptive-s7 | move 0 1 8 9 → advanced at gen 4 → stop 0 ×12 | move, stop |
| uniform-s7 | move 0/6/8/11 (gens 0/4/8/12); stop 0/0/0/0; turn_near 1/0/0/4; turn_far 1/1/2/0 | all four |
| adaptive-s11 | move 2 1 8 7 8 3 5 7 7 6 6 9 7 5 6 7 (never reached 2 consecutive gens ≥ 0.5) | move |
| uniform-s11 | move 2/4/4/3; stop 0; turn_near 1/1/0/1; turn_far 0/0/0/0 | all four |
| adaptive-s13 | move 0 1 1 1 2 2 3 1 1 1 2 0 0 3 2 2 | move |
| uniform-s13 | move 0/0/2/1; stop 0; turn_near 0; turn_far 0 | all four |

Deaths: 0 in every generation of every run. Adaptive advanced only in seed 7 (8/16
then 9/16 = p̂ 0.50, 0.56) and then spent 12 generations on `stop` at 0/16 with
`overshoot` as the failure; the stop score (progress-based) kept selecting, so the
population did not lose `move` (see 4.2).

### 4.2 Held-out competence of the champions (successes of 16 starts; gen 0 → gen 15)

| arm-seed | move val | move test | turn_near val | turn_near test | stop val/test | turn_far val/test |
|---|---|---|---|---|---|---|
| adaptive-s7 | 7 → 8 | 6 → 5 | 1 → 0 | 2 → 0 | 0 → 0 / 0 → 0 | 0 → 0 / 2 → 0 |
| uniform-s7 | 7 → 5 | 6 → 4 | 1 → 1 | 2 → 1 | 0 → 0 / 0 → 0 | 0 → 1 / 2 → 1 |
| adaptive-s11 | 5 → **13** | 8 → **12** | 0 → **6** | 0 → **5** | 0 → 0 / 0 → 0 | 0 → 0 / 0 → 0 |
| uniform-s11 | 5 → **11** | 8 → 9 | 0 → **4** | 0 → 2 | 0 → 0 / 0 → 0 | 0 → 0 / 0 → 0 |
| adaptive-s13 | 1 → 1 | 1 → 1 | 0 → 0 | 0 → 0 | 0 → 0 / 0 → 0 | 0 → 0 / 0 → 0 |
| uniform-s13 | 1 → 1 | 1 → 1 | 0 → 0 | 0 → 0 | 0 → 0 / 0 → 0 | 0 → 0 / 0 → 0 |

The gen-0 "champion" is the best of 16 random genomes after one generation of
selection on the train fold, so it is already a selected individual; in seed 7 that
draw already moved on 7 of 16 validation starts, which leaves little room for the
paired difference to show.

Failure codes of the gen-15 champions on validation: `move` fails almost only by
truncation (never covering 30 m in 20 s; s13: 14–15 of 16 truncated in both arms);
`stop` fails by `overshoot` (12/16 in s11 both arms, 5–7/16 in s7) or truncation
(never reaching 3 m/s); the turn lessons fail by `contact` (10/16 in uniform-s7),
`wrong_exit` (6/16 adaptive-s11 turn_far) and `drove_through_junction`.

### 4.3 Claims

| Claim | Verdict | Evidence |
|---|---|---|
| Learning (gen 15 > gen 0 on move validation by > 1, ≥ 2 of 3 seeds, both arms) | **refuted as pre-registered** | 1 of 3 seeds per arm: s11 +8 (adaptive) and +6 (uniform), test +4 / +1; s7 +1 / −2; s13 0 / 0 |
| Generalisation (validation vs test within 3) | holds | 24 of 24 cells |
| Curriculum: uniform ≥ adaptive on the mean of four lessons | **refuted** (reverse in 2 of 3 seeds, tie in the third) | validation gen 15 means: adaptive 2.00 / 4.75 / 0.25 vs uniform 1.75 / 3.75 / 0.25 (s7 / s11 / s13) |
| Curriculum: adaptive ≥ uniform on move | holds | 8 ≥ 5, 13 ≥ 11, 1 = 1 |
| Safety: zero deaths; contact dominant | deaths hold (0 of 102 runs); "contact dominant" holds only for the turn lessons — stop fails by overshoot, move by truncation | codes above |

What the battery shows, stated at the size it has: **in one city of three (seed 11)
both arms produced a champion that moves on 11–13 of 16 held-out starts where the
gen-0 champion managed 5, and that competence transfers to `turn_near` (0 → 4–6) which
the adaptive arm never trained.** In seed 7 the random draw was already competent
and 16 generations did not add to it; in seed 13 neither arm produced a mover
(population maximum 3/16 in 16 generations). `stop` was not learned by anyone (0/16
in all 24 stop evaluations and all 60 stop generations), which repeats Stage 3.

The curriculum result reads mechanistically: the uniform arm spends 3 of every 4
generations on lessons nobody can pass yet (stop, turn_far), and the selection those
generations apply (progress-based score) is a weaker pressure on moving than the move
lesson itself; the adaptive arm stays on `move` (the 0.5/2 rule advanced only once,
in seed 7, and then stalled on stop). Under this rule "adaptive" is effectively
"move-only with an exit that leads into a lesson that cannot yet be passed".

### 4.4 Limitations

- n = 3 cities and one population draw per city; `--seed` seeds both, so the seed-13
  failure cannot be attributed to the city or to the draw without a crossed design.
- 16 generations at 1280 ticks is a short budget; the seed-11 trajectory was still
  noisy at the end (5–9 of 16 over the last five generations).
- The gen-0 control is one generation of selection, not an unselected genome.
- Turn lessons have exactly 16 fold candidates, so each held-out cell is one pass over
  every start, and `stop` never produced a success, so its cells carry no information
  about generalisation.
- Debug binary (~4.8 s/gen); the release build was not used to keep the binary identical
  to Stage 3.

### 4.5 Next test

1. Make `stop` learnable before spending more budget on curricula: the failure is
   `overshoot` (no braking) or never reaching 3 m/s; a shaped score that pays for
   speed reduction inside the area, and a witness check that the scripted pilot passes
   the fold-13 starts, come first.
2. Crossed seeds: a separate population seed (`--pop-seed` was removed; reintroduce as
   `--population-seed`) so city × draw can be separated with 3 × 3 runs.
3. Longer budget on seed 11 (48 gens) with the same held-out evaluation, to see whether
   the 13/16 champion keeps improving or the population saturates.
