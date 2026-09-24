# School: physical braking envelope for `stop` (`--school-stop-envelope`) — 2026-09-23

Pre-registration. Follows [HANDOFF-CLAUDE-TAXI-SCHOOL-20260923.md](HANDOFF-CLAUDE-TAXI-SCHOOL-20260923.md)
"Próxima correção" and [SCHOOL-STOP-SHAPING-20260922.md](SCHOOL-STOP-SHAPING-20260922.md) §6.
Branch `organism/school-20260922`, base commit `6e6094f`. Status: **implemented behind the flag
(`b6c0ca4`, 21 GPU fixtures pass); baseline finished 0/96 (§1.1); paired A/B running (§5)**.

## 1. Question the baseline answers first

Astra's `stop` runs (K=3 starts per genome, 16 generations) ended 0/48 in every arm. Sixteen
generations is also exactly where the fixed-start run produced its single strict stop
(generation 15, `school-exclusive-train-s7`). Before adding a term to the score we need to know
whether the signal is *absent* or *slow*.

Battery in progress: `/media/llm-data/universe-runs/school-stop-baseline-48g-20260923/`
(`run_battery.sh`, binary `gpu_evolve-6e6094f` = debug build of this commit, seeds 7 and 11,
sequential, same flags as `taxi-school-delivery-task7-evolve-stop-seed7-k3-baseline` but
`--gens 48`).

Decision rule, fixed before results:

| outcome over both seeds (96 genome-generations, 48 starts each) | reading | consequence |
|---|---|---|
| ≥ 1 genome with strict success on ≥ 2 of its 3 starts | signal slow, not absent | envelope still implemented, but promoted only if the paired A/B (§5) beats this baseline on time-to-first-success |
| strict successes only on 1 of 3 starts | fixed-geometry memorisation again | envelope implemented; A/B must show successes on ≥ 2 starts |
| 0 successes in both seeds | signal absent during braking | envelope is the next mandatory step |

### 1.1 Baseline result (2026-09-23 21:03–21:13, Radeon/Vulkan)

| seed | generations | strict successes | slowest speed inside the area (last gen) | longest near-stop hold (last gen) | median progress (last gen) |
|---|---:|---:|---:|---:|---:|
| 7 | 48 | 0 / 2304 replica-episodes | 0.381 m/s | 23 ticks | 0.444 |
| 11 | 48 | 0 / 2304 replica-episodes | 0.791 m/s | 0 ticks | 0.722 |

Third row of the decision table: signal absent during braking. Seed 7 gets within 0.08 m/s of
the strict threshold and never holds; seed 11 never gets under 0.6 m/s. Tripling the generation
count over Astra's 16-generation runs changed nothing. The envelope (§3) is the next step.

## 2. Measured deceleration (real GPU, replays decoded per tick)

Source: `.replay` frames (128 B, per tick, `speed` f32) decoded with
`neuromod-lab/src/lab_api/trace/reference_decoder.py::read_replay`. Deceleration per tick
`a = (v[t-1] − v[t]) · 64`, only while speed decreases and `v[t-1] > 0.3`.

| replay | controller | median a (m/s²) | p90 | median by speed band (m/s → m/s²) |
|---|---|---:|---:|---|
| `taxi-school-delivery-task7-witness-v2/stop/gen0000.replay` | reference witness v2, `brake = 1` at task | 2.20 | 5.56 | 1→2.4, 2→4.7, 3→6.3, 4→5.7 |
| `school-exclusive-train-s7/gen0015.replay` beings 0–3 | evolved stop champion generation | 0.45–0.69 | 1.9–2.3 | 1→0.6–1.1, 2→0.8–1.4, 3→1.0–1.7 |
| `school-static-rest-trace-s7/gen0000.replay` | evolved, static-rest trace | 0.66 | 2.50 | 1→1.1, 2→0.9, 3→2.2 |

Reading: the body brakes at 2–6 m/s² when the pedal is pressed; evolved controllers decelerate
at 0.5–1 m/s², i.e. they coast. The envelope must be satisfiable with the pedal (≤ 2 m/s²) and
violated by coasting from `v_min` (3 m/s) into the 4 m area.

`a_ref = 2.0 m/s²` (below the witness median, above every evolved band). Stopping distance from
`v_min = 3 m/s`: `3² / (2·2) = 2.25 m`. Braking zone radius `r_zone = r_goal + 2.25 = 6.25 m`
with `r_goal = area_radius = 4 m` (Params; no new constant).

## 3. Envelope and new record slots

```
v_allow(d) = sqrt(v_stop² + 2·a_ref·max(d − r_goal, 0))      v_stop = 0.3 m/s (strict criterion)
in_zone    = d ≤ r_zone and phase ≥ 1 (v_min already reached)
excess     = max(speed − v_allow(d), 0)
```

Schema v3: stride 48 → 56. Slots 0..47 frozen. New slots, recorded regardless of the flag:

| slot | name | update |
|---|---|---|
| 48 | `ENV_ZONE_TICKS` | +1 per tick with `in_zone` |
| 49 | `ENV_CONFORM_TICKS` | +1 per tick with `in_zone && excess == 0` |
| 50 | `ENV_EXCESS_INTEGRAL` | `+= excess · dt`, clamped at `v_min · 2 s` (= 6 m) |
| 51 | `ENV_ENTRY_SPEED` | speed at the first `in_zone` tick (1e9 before) |
| 52 | `ENV_MIN_SPEED_APPROACH` | `min(·, speed)` while `in_zone` and `d` decreasing |
| 53 | `ENV_DIST_AT_MIN_SPEED` | `d` at the tick slot 52 last decreased |
| 54–55 | reserved | 0 |

Flag bit 2 (`flags & 4`, CLI `--school-stop-envelope`, lesson = stop only). Score term added to
`close`:

```
envelope = 200·conform_ratio − 100·clamp(ENV_EXCESS_INTEGRAL / 6, 0, 1)
conform_ratio = ENV_CONFORM_TICKS / ENV_ZONE_TICKS   (0 when ENV_ZONE_TICKS == 0)
```

Bounded in [−100, 200]; with the 2026-09-22 shaping (bit 1) also on, the total shaping stays
≤ 700 < 1000 (success). Success remains `DWELL ≥ hold_ticks` at `v_stop` inside the area. The
judge reads only physics (`physics_out` speed/pos) and the body — it never sees the command, so
`brake > 0` and throttle+brake overlap cannot be rewarded by construction.

Exported JSON fields: `envelope_zone_ticks`, `envelope_conform_ratio`,
`envelope_excess_integral_m`, `envelope_entry_speed_mps` (null before the zone),
`approach_min_speed_mps`, `approach_min_speed_distance_m`.

## 4. Fixtures (GPU, `school_tests.rs`, all `#[ignore = "real GPU"]`, run with `-- --ignored`)

Pose streams as in the existing `stop_shaping_*` tests (`Step::at(pos, vel)`), `straight(Lesson::Stop)`.

1. **Decelerating approach** enters the zone at 3 m/s and follows `v_allow` down to 0.3 m/s,
   holds 128 ticks → success; `conform_ratio ≥ 0.95`, `excess_integral < 0.05`, score =
   `1000 + 500·progress + old shaping + ~200`.
2. **Brakes far, parks outside**: 3 m/s then stops at `d = 8 m` (outside the zone) for 300 ticks →
   truncated; `ENV_ZONE_TICKS = 0`, envelope = 0; score < fixture 1 and < a creeping entry
   (0.5 m/s inside for 64 ticks). Variant at `d = 5 m` (in zone, outside area): conform 1 → +200,
   but no dwell, no success, progress `1 − 5/L` → still below the creeping entry.
3. **Fast overshoot** at 8 m/s through the area → failed code 5 (existing); `conform_ratio = 0`,
   `excess_integral` at the clamp, envelope = −100; progress unchanged.
4. **Command-blind**: fixture 1 stream once with `input[51] = 0` and once with `input[51] = 1`
   (throttle held) → identical record bit for bit. Documents that pedal overlap gives no advantage.
5. **Flag off = historical score**: fixtures 1–3 with `flags = 2` (old shaping only) reproduce the
   v2 formula bit for bit; slots 48–55 are still populated.

Host-side (no GPU), `cargo test --lib school::tests` (added 2026-09-23 evening, after the
audit found this paragraph promised tests that did not exist): `record_json` round-trip of the
six new fields, null readings when no zone tick was recorded, `SCHOOL_SCHEMA_VERSION == 3`,
`SCHOOL_STRIDE_F32 == 56`, slot numbers 48–53, `Params` stays 80 bytes. There is no v2 record
decoder in the tree, so "refuses v3 stride" has nothing to guard and was dropped.

## 5. Evolution A/B (after §4 passes, after the §1 battery ends)

Paired, seeds 7 and 11, K=3, `--school-start-fold train`, 16 beings, 48 generations, horizon 1280:

| arm | flags |
|---|---|
| control | `--school-stop-shaping` (= §1 baseline, reused, not re-run) |
| treatment | `--school-stop-shaping --school-stop-envelope` |

Primary: first generation with a strict success on ≥ 2 of 3 starts. Secondary: median progress,
median `ENV_ENTRY_SPEED`, median `conform_ratio`, contact ticks. Promotion (default on in the
curriculum) only if the treatment reaches the primary in both seeds and the control in neither;
one seed → "not replicated", keep behind the flag and report. Frozen champions then go through
validation/test folds without mutation (handoff step 4).

### 5.1 Result of the random-seeded A/B (2026-09-23, binary b6c0ca4, 48 generations, K=3)

| seed | arm | strict successes | first success | best score, mean of last 8 gens | replicas with phase ≥ 1 (last 8) | phase 2 (last 8) | slowest speed inside area | longest near-stop dwell (ticks) | in-zone replicas (last 8): conform median / excess median |
|---|---|---|---|---|---|---|---|---|---|
| 7 | control | 0 / 2304 | — | 776.2 | 20.5 / 48 | 10.2 / 48 | 0.18 m/s | 90 | not recorded (6e6094f) |
| 7 | envelope | 0 / 2304 | — | 752.6 | 15.9 / 48 | 9.8 / 48 | 0.15 m/s | 91 | 106: 0.01 / 5.41 m |
| 11 | control | 0 / 2304 | — | 655.6 | 18.4 / 48 | 13.2 / 48 | 0.78 m/s | 0 | not recorded |
| 11 | envelope | 0 / 2304 | — | 708.1 | 19.5 / 48 | 11.5 / 48 | 0.74 m/s | 0 | 133: 0.00 / 5.98 m |

Generations with a bit-identical phase vector between arms: 22/48 (seed 7), 10/48 (seed 11).
The arms do diverge, so the flag is live, but the primary metric is 0 in all four runs and the
secondary metrics move within seed noise (best score −24 on seed 7, +52 on seed 11; phase-2
count −0.4 and −1.7). **Decision by the §1 rule: not promoted.** The term never rewarded
anyone: the in-zone medians are conformance 0.00–0.01 and excess at or near the 6 m clamp, i.e.
every replica that reaches the zone enters at 2.4–3.8 m/s and never gets under the curve. The
population that does reach the area (phase 2, ~10/48) parks at 0.15–0.8 m/s, above `v_stop` =
0.3 m/s for 128 ticks — seed 7 got to 91 near-stop ticks of the 128 required, seed 11 never
registered a near-stop tick. Wall time: 5 min per run.

### 5a. Interim observation on the random-seeded A/B (generation 16, seed 7)

Both arms are still the same population at generation 16 (identical phase counts 43/3/2 and
fail codes). The population's mode is a creeper: `phase 0`, top speed 1.6–2.4 m/s, parks about
7 m before the area (progress 0.77 → 383 points) because entering under `v_min` is failure
code 8 (−200) and worth less than parking. Only 4 of 48 replicas reach the braking zone; their
envelope readings are conformance 0–0.15 and excess 1–6 m, so the term only *penalises* them.
The envelope cannot flip a selection until a fast approacher exists to reward. Pre-registered
addition, before the results of §5 are in:

### 5b. Paired A/B seeded from the move champions

Same arms and flags as §5, but the population starts as copies of Astra's move champions
(`taxi-school-delivery-task7-evolve-move-seed{7,11}-k3(-v2)/evolve_best_gen0015.bin`,
`--resume --resume-mix 1.0`), which already reach `v_min` and overshoot (Astra's
`stop-from-move` run: progress 0.97, 0/48). That is the population the envelope was designed
for. Run dir `school-stop-envelope-from-move-20260923`, binary `b6c0ca4`, 4 runs
(control/envelope × seeds 7/11), 48 generations each. Same primary/secondary metrics and the
same promotion rule as §5. If §5b also fails, the next hypothesis is the `v_min` gate itself
(fail code 8 makes a slow entry worse than parking), not the braking term.

### 5c. Why the term cannot pay yet: the brake channel is selected out, and the clamp hides it

Per-tick pedal commands decoded from the replays (`intent_brk`, `final_brk` fields; the school
judge itself stays command-blind) of the move-seeded envelope arm, seed 7, generation 47, best
three replicas (scores 850 / 773 / 751):

| quantity | value |
|---|---|
| `final_brk` frames > 0.01 in the whole episode | 0 of 320 (all three) |
| `intent_brk` median, slow tail | −0.35 / −0.12 / −0.12 (the mapper clips negative intent to zero brake) |
| speed inside the area | 0.4–0.9 m/s for ~700 ticks with throttle 0 and brake 0 (coasting) |
| coasting deceleration 1.66 → 0.36 m/s | 560 ticks ≈ 0.15 m/s² |
| replicas that ever brake > 0.01, gen 0 → gen 47 | 0/48 → 6/48 (envelope arm), 15/48 (control) |

The random-init baseline's best stopper (seed 7, gen 40, slot 17, near-stop dwell 90/128,
minimum 0.18 m/s) also never pressed the brake: 0 of 107 slow frames with `final_brk` > 0.01,
`intent_brk` median −0.40. At generation 0 the random population is not biased: seed 11 has
42/48 replicas braking at some point (median intent +0.16), seed 7 27/48 (median −0.14); by
generation 40 seed 7 is at 9/48 and median −0.28. Selection removes the brake because the
progress term (500·progress) pays first and any brake use costs progress, and nothing pays
the brake back later: the strict success needs ≤ 0.3 m/s, coasting from 3 m/s to 0.3 m/s at
0.15 m/s² needs ~18 s = the whole horizon, and the shaping/near-stop terms saturate at the
coasting floor.

The envelope term was designed to pay braking, but its excess integral is clamped at 6 m and
in the last 16 generations 46 % (seed 7), 50 % (seed 11) and 59 % (move-seeded seed 7) of the
in-zone replicas sit exactly at the clamp: a coasting entry at 3.9 m/s accumulates ~20 m of
excess against a 2 m/s² curve, so every non-braking replica looks identical (−100) and a
replica that brakes a little is indistinguishable from one that does not. A hard ceiling hit
by half the population is the metric saturating (CLAUDE.md rule 4), not the population failing.

**Pre-registered next A/B (§5c):** same move-seeded protocol as §5b, envelope on in both arms,
seeds 7 and 11, 48 generations: control = clamp 6 m (`--school-stop-envelope-clamp-m 6`,
the default, identical to §5b treatment), treatment = clamp 30 m. The flag is new (parameter
in `Params`, default unchanged, fixture 3 still asserts the 6 m clamp). Primary metric as §5.
Secondary: replicas with `final_brk` > 0.01 in the last 8 generations (decoded from replays),
minimum speed inside, strict dwell. Promotion of the wider clamp needs both seeds to raise the
braking fraction and the strict dwell; a wider clamp that only lowers scores is not promoted.
Not run yet: the build for the new flag is deferred until CPU use is allowed again.

## 6. Out of scope (declared)

- No change to DNA, brain, action mapper, taxi kernel or observation channels.
- No reward for the brake command value (excluded on purpose, §3).
- The experimental `REFLEX_TASK_BRAKE_NEURON` prior stays out of the tree
  (`/media/llm-data/universe-runs/universe-artifacts-20260923/reflex-task-brake-neuron-20260923.patch`).
- Lessons other than `stop` ignore bit 2.
