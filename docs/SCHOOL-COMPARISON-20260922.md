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

(filled after the battery)
