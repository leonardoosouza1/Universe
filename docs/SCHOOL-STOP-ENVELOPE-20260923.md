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

Host-side (no GPU): `record_json` round-trip of the six new fields; schema version 3 in the
manifest; `SCHOOL_STRIDE_F32 == 56`; snapshot decode of a v2 record refuses v3 stride (no silent
reinterpretation).

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

## 6. Out of scope (declared)

- No change to DNA, brain, action mapper, taxi kernel or observation channels.
- No reward for the brake command value (excluded on purpose, §3).
- The experimental `REFLEX_TASK_BRAKE_NEURON` prior stays out of the tree
  (`/media/llm-data/universe-runs/universe-artifacts-20260923/reflex-task-brake-neuron-20260923.patch`).
- Lessons other than `stop` ignore bit 2.
