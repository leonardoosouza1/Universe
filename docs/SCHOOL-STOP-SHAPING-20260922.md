# School — shaped score for the `stop` lesson (2026-09-22)

Follow-up of `SCHOOL-COMPARISON-20260922.md` §4.5 item 1. Flag `--school-stop-shaping`,
off by default (paired A/B required, `CLAUDE.md` rule 3).

## 1. Diagnosis (exploratory, existing GPU records)

192 stop records (adaptive-s7, generations 4–15, 16 beings each, binary
`gpu_evolve-stage3-v3`), read from `school_result_gen00NN.json`:

| group | n | score min / median / max | progress median | max speed median | dwell |
|---|---|---|---|---|---|
| failed `overshoot` | 120 | 230 / 285 / 299 | 0.97 | 8.9 m/s | 0 |
| truncated (no failure) | 39 | 84 / 366 / 499 | 0.73 | 3.3 m/s | 0 |
| other failures (corridor 28, contact 4, no_approach_speed 1) | 33 | 71 / 160 / 281 | 0.72 | 6.2 m/s | 0 |

- `dwell_ticks` = 0 for all 192: no being ever reached ≤ 0.3 m/s inside the area.
- Spearman(score, progress) = 0.69; Spearman(score, max speed) = 0.15 over all 192 and
  0.00 among the overshooters: **the score carries no information about slowing down**.
- Top-3 by score in gens 4, 9 and 15 are truncated creepers with max speed 2.4–3.9 m/s
  and entry speed 3.0–3.3 m/s (just above `v_min` = 3.0): the population converged to
  "drive slowly to the area and never brake", which is the global optimum of the
  current score short of success (500·progress ≈ 499 vs 285 for a fast overshoot).
- H2 (beings cannot brake) is not supported: the same populations record 94–160
  brake ticks per being on average in the turn generations (`brake_ticks`, gen 13–15
  of train-uniform-s7).

Verdict: the score has a plateau between "inside at 3 m/s" (≈ 500) and "held stop"
(≈ 1500); evolution has no gradient across it.

## 2. Change

Two record slots and one flagged score term; the judge's outcomes, codes and
observation slots are untouched.

```
MIN_SPEED_INSIDE (slot 34) = 1e9 at the first judged tick; min(·, speed) on every tick
                             inside the area once phase ≥ 2 (entered after reaching v_min)
MAX_DWELL        (slot 35) = max over ticks of the consecutive-stopped counter DWELL

score = 1000·success + 500·progress − 200·failed − 100·(1 − integrity) − 100·(1 − alive)
      + [flags bit 1 and lesson = stop] · ( 300·clamp(1 − MIN_SPEED_INSIDE / v_min, 0, 1)
                                          + 200·clamp(MAX_DWELL / hold_ticks, 0, 1) )
```

Ordering the term creates (v_min 3 m/s, hold 128 ticks): overshoot at 8 m/s ≈ 285;
creeper at 3.2 m/s inside ≈ 500; slows to 1 m/s inside ≈ 700; holds 0.2 m/s for 64
ticks ≈ 880; success ≈ 2000. Both slots are recorded regardless of the flag and
exported as `min_speed_inside_mps` (null until entered) and `max_dwell_ticks`.

## 3. Flow map (producer → buffer → consumer)

| order | producer (`file:symbol`) | device | input (shape, unit) | output / buffer | consumer | frequency | state |
|---|---|---|---|---|---|---|---|
| 1 | `physics_c.wgsl` (pose, velocity, heading, contact mask) | GPU | physics INPUT f32[92]/being, OUTPUT f32[56]/being (m, m/s) | `physics_out` | school judge | every tick | runtime |
| 2 | `school.wgsl::judge` | GPU | `physics_out[o+0,2]` speed, `[o+36,38]` pos; `body[s+8]` integrity, `[s+29]` alive; Params (v_min 3.0, v_stop 0.3, hold 128, area 4 m, flags) | school record f32[40]/being (`SchoolGpu.buf_state`): slots 15 DWELL, 34 MIN_SPEED_INSIDE, 35 MAX_DWELL, 21 SCORE (written once, in `close`) | host `download` | every tick, `close` once per attempt | runtime, reset per generation (`SchoolGpu::reset`) |
| 3 | `school.rs::SchoolGpu::encode` | CPU→GPU | `Params { flags = bench \| stop_shaping << 1 }` | uniform buffer | kernel | every dispatch | configuration |
| 4 | `main.rs` ≈5769 `records[R_SCORE]` | CPU (readback `download`: copy → map_async → poll → recv) | 40 f32 per being | `fitness: Vec<f64>` when bench | `evolution.rs` selection | once per generation | derived |
| 5 | `school.rs::record_json` | CPU | record | `school_result_genNNNN.json` `beings[]` (+ `min_speed_inside_mps`, `max_dwell_ticks`) | `analyze`, lab_api `school_observatory.py`, panel | once per generation | persisted |

Nothing in the change touches DNA, the brain, the taxi kernel or the observation
channels (65/70/71/80/81/90 keep the same producer and values). The skipped-step
value: with the flag off `shaping = 0` and the score is bit-identical to Stage 2
(parity test below). Schema: `school_result` gains two being keys; consumers that
ignore unknown keys are unaffected.

## 4. Validation ladder (cheap → expensive, stop at the first failure)

1. Static: `cargo test --test wgsl_validation` (parse + type-check).
2. Synthetic on the real GPU (`src/gpu/school_tests.rs`, `--ignored --test-threads=1`):
   the same pose stream (enter at 6 m/s, 50 ticks at 1 m/s inside, 40 ticks stopped,
   leave, truncate) scores `500·progress` with the flag off and
   `500·progress + 300·(1 − 1/3) + 200·(40/128)` with it on; the success stream scores
   > 1900 with the flag; the recorded slots equal 1.0 m/s and 40 ticks in both.
3. Behavioural, two conditions that must diverge: the scripted reference witness on
   `stop` with and without the flag (same seed, one being) — success in both, score
   ≈ 1500 vs ≈ 2000.
4. Battery (pre-registered in §5).

## 5. Pre-registered A/B

Seeds 7 and 11 (cities with movers), `--school stop --school-start-fold train`, 16
beings × 16 generations × 1280 ticks, arms `shaping` (flag on) vs `plain` (flag off),
same binary; gen-15 champions ×16 on the stop validation and test folds.

| Claim | Expected | Refuted if |
|---|---|---|
| Gradient | with shaping, the population median of `min_speed_inside_mps` falls below 2 m/s by gen 15 in both seeds; plain stays ≥ 3 m/s | shaping median ≥ 2 m/s in both seeds |
| Competence | with shaping, ≥ 1 success on the train fold by gen 15 in ≥ 1 seed, and the held-out champion ≥ 1/16 on validation; plain 0 everywhere (replicating Stage 3–4) | shaping 0 successes in both seeds |
| Safety | zero deaths; contact + corridor failures do not rise above the plain arm by more than 3/16 | more deaths or a contact surge |

## 6. Results

(filled after the battery)
