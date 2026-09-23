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

Battery `/media/llm-data/universe-runs/school-stop-shaping-20260922/battery.sh`, binary
`gpu_evolve-stop-shaping` (sha256 `d69ad90cd871b007…`, debug build of the source committed
as `fd84432`), adapter `AMD Radeon RX 6750 XT (RADV NAVI22)`, 21:19–21:25, all 14 steps
exit 0, wall 6 m 23 s (train 76–80 s each, evals 6–7 s, witness 6–7 s). Analysis:
`analyze.py` → `analysis.json`, `report.html`. Verdict: **shaping refuted on gradient and
competence; safety holds.** The mechanism is in §6.5 and it is not the one §1 targeted.

### 6.1 Gate 3 — witness pair (scripted reference pilot, seed 7)

| Arm | Success | Entry speed | Outcome tick | Slowest inside | Longest hold | Score |
|---|---|---|---|---|---|---|
| plain | 1/1 | 5.45 m/s | 659 | 0.00 m/s | 128 ticks | 1499.9 |
| shaping | 1/1 | 5.45 m/s | 659 | 0.00 m/s | 128 ticks | 1999.9 |

Same trajectory, score differs by exactly the full bonus (300 + 200). The two new slots are
recorded in both arms. PASS.

### 6.2 Train trajectories (16 beings, stop, train fold)

"Entered" = beings with a recorded slowest speed inside the 4 m area; "bonus>0" = beings
whose shaped bonus would be non-zero (slowest inside < 3 m/s or any hold).

| Arm | Gen | Success | Failed / truncated | Codes | Score median / max | Entered | Slowest inside, median | bonus>0 |
|---|---|---|---|---|---|---|---|---|
| plain s7 | 0 | 0/16 | 3 / 13 | contact 1, corridor 2 | 0.8 / 95.4 | 0 | – | 0 |
| plain s7 | 7 | 0/16 | 16 / 0 | contact 2, corridor 2, overshoot 12 | 271.6 / 298.7 | 15 | 8.84 m/s | 0 |
| plain s7 | 15 | 0/16 | 12 / 4 | contact 1, corridor 5, overshoot 6 | 268.4 / 345.1 | 6 | 7.83 m/s | 0 |
| shaping s7 | 0 | 0/16 | 3 / 13 | contact 1, corridor 2 | 0.8 / 95.4 | 0 | – | 0 |
| shaping s7 | 7 | 0/16 | 16 / 0 | contact 2, corridor 2, overshoot 12 | 271.6 / 298.7 | 15 | 8.84 m/s | 0 |
| shaping s7 | 15 | 0/16 | 14 / 2 | contact 1, corridor 3, overshoot 10 | 283.7 / 406.1 | 10 | 8.02 m/s | 0 |
| plain s11 | 0 | 0/16 | 3 / 13 | contact 1, overshoot 2 | 2.4 / 281.0 | 2 | 4.99 m/s | 0 |
| plain s11 | 7 | 0/16 | 8 / 8 | corridor 5, overshoot 3 | 275.4 / 487.8 | 4 | 3.69 m/s | 2 |
| plain s11 | 15 | 0/16 | 6 / 10 | corridor 1, no_approach_speed 2, overshoot 3 | 277.6 / 498.4 | 4 | 3.46 m/s | 2 |
| shaping s11 | 0 | 0/16 | 3 / 13 | contact 1, overshoot 2 | 2.4 / 281.0 | 2 | 4.99 m/s | 0 |
| shaping s11 | 7 | 0/16 | 13 / 3 | corridor 6, overshoot 7 | 280.8 / 414.6 | 7 | 4.14 m/s | 2 |
| shaping s11 | 15 | 0/16 | 8 / 8 | corridor 1, no_approach_speed 1, overshoot 6 | 279.7 / 583.3 | 7 | 3.40 m/s | 1 |

Seed 7: the two arms are bit-identical (every being's status and progress) through gen 10
and have the same ranking through gen 9 — the first non-zero bonus (gen 4, slot 15, +245)
went to a being that was already ranked first. Seed 11: rankings differ from gen 2,
populations from gen 3. Deaths: 0 in all 1 024 train episodes.

### 6.3 Held-out champions (gen-15 champion ×16, one generation, no selection)

| Champion | Fold | Success | Codes | Truncated | Score max | Entered | Slowest inside, min | Longest hold |
|---|---|---|---|---|---|---|---|---|
| plain s7 | validation | 0/16 | overshoot 5 | 11 | 476 | 7 | 1.40 m/s | 0 |
| plain s7 | test | 0/16 | overshoot 5 | 11 | 480 | 9 | 0.65 m/s | 0 |
| shaping s7 | validation | 0/16 | contact 2, corridor 3, overshoot 11 | 0 | 298 | 11 | 4.29 m/s | 0 |
| shaping s7 | test | 0/16 | corridor 4, overshoot 11 | 1 | 411 | 11 | 3.67 m/s | 0 |
| plain s11 | validation | 0/16 | corridor 1, no_approach_speed 2, overshoot 3 | 10 | 487 | 7 | 1.78 m/s | 0 |
| plain s11 | test | 0/16 | corridor 1, no_approach_speed 1, overshoot 12 | 2 | 390 | 12 | 2.41 m/s | 0 |
| shaping s11 | validation | 0/16 | overshoot 3 | 13 | 606 | 5 | 1.83 m/s | 0 |
| shaping s11 | test | 0/16 | corridor 3, no_approach_speed 1, overshoot 6 | 6 | 472 | 7 | 1.93 m/s | 0 |

### 6.4 Claims

| Claim | Outcome |
|---|---|
| Gradient | **refuted**: shaping median slowest-inside at gen 15 is 8.02 m/s (s7) and 3.40 m/s (s11), both ≥ 2 m/s; plain 7.83 and 3.46. No arm differs from its pair. |
| Competence | **refuted**: 0 successes in all four train arms and 0/128 held-out; the longest hold is 0 ticks in every one of the 1 154 records of the battery. |
| Safety | **holds**: 0 deaths; contact + corridor at gen 15 is plain 6 vs shaping 4 (s7) and 1 vs 1 (s11). |

### 6.5 Mechanism (exploratory, from the same records)

1. **The bonus almost never reaches anyone.** It is non-zero on 6 of 256 (s7) and 20 of
   256 (s11) train episodes, and when it does the recipient was already a truncated creeper
   scoring ≈ 480–500 against overshooters at ≈ 285 — already ranked first under the plain
   score. Selection decisions therefore differ from the plain arm only rarely (never in
   s7 gens 0–9).
2. **The plateau moved; it did not disappear.** No being in 1 154 records went below
   `v_stop` = 0.3 m/s inside the area for a single tick, so the 200-point hold term paid
   0 everywhere. The slowest inside was 0.55 m/s (s7 gen 4, slot 15, score 719): that being
   then drifted on (max along 36.4 m in 1 280 ticks). Coming to rest needs
   `throttle ≤ 0.01 && brake_cmd > 0.01` sustained (`physics_c.wgsl:292-293`); a brain that
   creeps at 0.5 m/s is 245 points up and gets nothing more until it holds the brake.
3. **One episode per genome per generation, on a rotating junction.** The being → spawn slot
   map is rehashed every generation (`main.rs:4634`, `slot_rotation(gen, seed, n)`), so the
   elite at slot 0 — same genome — is judged on a different approach each generation and its
   progress swings 0.00–0.99 (median 0.83 in s11, 0.97 in s7), flipping between failed and
   truncated. The s7 gen-4 champion (start (285, 228), score 719, slowest 0.55 m/s) was
   re-run as elite at gen 5 on start (99, 348) and failed with progress 0.875. The
   between-start noise on one genome (≈ ±150 points of the progress term) is larger than the
   shaping increment between creeping at 2 m/s (+100) and at 0.5 m/s (+245). For `move` any
   start rewards the same behaviour; for `stop` the braking point is geometry-specific, so
   the noise is the lesson's own signal.

Points 1–2 say the shaped score is still a step function where it matters (the first tick
under 0.3 m/s); point 3 says that even a smooth score would be sampled at n = 1 on a
rotating geometry. Point 3 is testable independently of any score and comes first.

### 6.6 Limitations

Two seeds, 16 generations, debug binary; `--school-mode bench` only. The training runs
had no `--driving-diagnostics`, so "never held the brake" is inferred from speed, not read
from the command trace. Identical early generations between arms are the expected
consequence of a zero bonus (both arms ran the same sha256 binary; the witness pair differs
by exactly 500 points), not a stale-binary artefact. The s11 shaping champion's 606 on the
validation fold is one being in one episode, not a trend.

### 6.7 Next test

1. **Determinism and geometry witness** (`--fixed-spawn-slots`, 4 generations, shaping,
   seed 7): the elite at slot 0 keeps the same start, so its progress must be identical at
   gens 1–3. Identical → the swing in §6.5.3 is geometry; different → the simulation is not
   reproducible and every school number so far carries that noise.
   **Done (`witness-fixed-slots-s7/`, same binary): identical** — progress 0.916544, score
   458.272, energy 249 731.0 J at gens 1, 2 and 3 for the elite on start (99, 156); the same
   genome had scored 95.4 on its gen-0 start. The swing is geometry. Continued in
   [SCHOOL-SHARED-START-20260922.md](SCHOOL-SHARED-START-20260922.md).
2. **Shared start within a generation**: in bench mode taxi–taxi contacts are already off
   (`--school-mode bench`, `cli_flags.rs:149`), so all 16 beings can be judged on the same
   candidate (rotating the candidate across generations, `gen % candidates`). Pre-registered
   A/B against the rotating-slot default, stop + shaping, seeds 7 and 11, 16 generations:
   with a shared start the elite's score must replicate exactly across generations that draw
   the same candidate, and the fraction of beings with a non-zero bonus must grow across
   generations (refuted if it stays ≤ 2/16 at gen 15). Success is still not promised: this
   removes the noise, it does not add the brake.
3. Only after 2: partial credit for the hold term below `v_min` (e.g. ticks under 1 m/s),
   so that the 0.3 m/s cliff of §6.5.2 also has a slope.

