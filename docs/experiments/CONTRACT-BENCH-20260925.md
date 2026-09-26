# Contract bench — frozen genomes on pre-built scenarios (2026-09-25)

Status: EXECUTED 21:05–21:12 (30 GPU runs, 7–20 s each). Owner: Claude (Codex offline since 20:36; his
`DRIVER-MICRO-CONTRACTS-20260925.md` asks the same question and stays "pending
execution" — this record executes it in diagnose-all form and adds input contracts).
No city-delivery or learned-competence claim.

## Question

Can a frozen genome be judged on inputs (what its sensors report) and outputs
(what its body does) in short pre-built scenarios, so that a city battery is only
launched after the contracts pass? Owner's ask (21:0x): "validar sem precisar ir
para a cidade, testar visão e outras entradas … como se fosse contratos, velocidade
mínima, ver parede … larga em algo pré-montado e só analisa o input e output".

## Bench

Script: `scripts/contract_bench.py` (run + report). Root:
`/media/llm-data/universe-runs/contract-bench-20260925/`. Binary: hard link of
`navigation-preview-paired-20260925/gpu_evolve` (sha256 `f294c1dda8522fa9…`, built
19:32, no city-family code). One GPU process, sequential, `nice -n 10`,
`RAYON_NUM_THREADS=1`.

Genomes (sha256 prefix):

| name | file | sha | what it is |
|---|---|---|---|
| witness | — (`--taxi-reference-witness`) | — | scripted controller, positive control, never learning evidence |
| founder | `cross-city-forward-learning-20260925/founder/genome_input.bin` | `6c5e94ba` | = learning gen 0 |
| champion48 | `stable-centered-backbone-thaw-20260925/evolve_best_gen0011.bin` | `547caaa7` | 48 m course 4/4 frozen (`stable-four-start-confirmation`) |
| candidate | `cross-city-forward-learning-20260925/candidate/evolve_best_gen0000.bin` | `20faffb5` | = learning gen 3, city 0/0 |

Scenarios = school bench lessons (`docs/SCHOOL-LESSONS-20260922.md`), 4 starts
each (`--school-start-fold all --school-eval-starts 4 --school-start-pool 4`),
`--beings 1 --gens 1 --resume --resume-mix 1` (frozen, no mutation), seed 11,
`--driving-diagnostics --vision-being 0`:

| lesson | ticks | observation | why |
|---|---|---|---|
| move, stop, turn_near, turn_far | 1280 (20 s) | **v3** | v4/v5 route channels are station-indexed; a free lesson sub-goal encodes as direction 0 / distance 1 (`taxi.wgsl route_senses`) |
| move_v4 (champion48 only) | 1280 | v4 | negative control: input contract I2 must FAIL, showing that degeneracy |
| approach_passenger, pickup | 2048 | v4 | station goals, as trained |
| delivery48 (`--school-service-course 48`) | 4096 | v4 | known 4/4 for champion48; anchors the bench |

Ceiling: 1280 ticks at 10 m/s = 200 m ≫ 30–50 m lesson geometry.

## Contracts (thresholds fixed before the runs)

Inputs, on the observed being (replica 0), records every 16 ticks, i8 channels:

| id | contract | pass |
|---|---|---|
| I1 | has_task ch180 at start | ≥ 120 |
| I2 | route/sub-goal straight ahead at start, ch178 (cos in body frame) | ≥ 100 |
| I3 | front rays open at start, ch19/ch20 (60 m before a junction, street continues) | both ≤ 20 |
| I4 | side rays hit and differ (lane offset 3 m: left/right walls 4.5 m vs 10.5 m), ch16/ch23 | both > 0 and differ ≥ 10 |
| I5 | distance ch183 = 127·d/(d+100) with d from taxi.target_dx/dz | max err ≤ 8 (i8) |
| I6 | speed ch65 = 2·speed | max err ≤ 1 m/s |
| I7 | (turn lessons) lateral cue appears after the box, peak |ch179| | ≥ 60 |

Outputs, from `school_result_gen0000.json` (all 4 starts) and `driving.csv`:

| id | contract | pass |
|---|---|---|
| O1 | judge success | 4/4 |
| O2 | reaches 3 m/s (minimum speed) | min over starts of max_speed ≥ 3 |
| O3 | no contact | contact_ticks = 0 all starts |
| O4 | not standing still | stationary ticks / judged ticks < 0.5 all starts |
| O5 | progress | ≥ 0.5 all starts, or O1 |

## Predictions (registered 21:06, before any result)

- witness: O1 PASS on move/stop/turn/approach/pickup/delivery48 (positive control).
  A witness failure = lesson infeasible on this binary, reported as such.
- champion48: move O1 PASS, O2 FAIL (it drives at 1.4–2.7 m/s on the 48 m course);
  stop O1 FAIL with code 8 (never reaches 3 m/s before the area); turn_near/far O1
  FAIL (no turn training; drive-through code 6 or truncation); approach/pickup
  PASS; delivery48 4/4 PASS (replication of `stable-four-start-confirmation`).
- founder and candidate: same pattern as champion48; candidate no better on turns.
- Inputs: I1–I6 PASS for every v3/v4 run except I2 FAIL on move_v4 (route channel
  degenerate for a free sub-goal). I7 PASS under v3 (body-frame direction of the
  exit gate is lateral once inside the box) — if I7 FAILS while O1 fails, the turn
  failure is perceptual, not motor.
- Reading for the attack order: if every learned genome fails O2 on move and code
  8 on stop, the judge-v3 hypothesis (the judge trains parking) keeps step 1; if
  I7 passes and turns still fail, step 2 (T-turn rung) is a motor-learning gap and
  step 3 (perception v6) can wait.

## Findings while running (21:06–21:10)

- **move_v4 negative control is moot**: the binary refuses it at start (`run.log`:
  "navigation v4/v5 requires station-based service lessons; use
  --navigation-observation v3 for motor lessons"). The degeneracy is guarded, not
  silent. Exit 2, 0 s. Kept in the table as `-`.
- **Wall-ray fan is one-sided by design.** `perception.wgsl:424-456` writes 8 yaw
  rays to ch16-23, but `perception_temporal.wgsl:367` overwrites ch20 with fuel
  ("ch20 replaces the spatial ray at index4") and `perception_motion.wgsl:231`
  overwrites ch21-27 with per-row motion. Witness/move tick 16: ch16-23 =
  [110,110,95,31,**124**,0,0,0] with fuel 0.98·127 = 124. Only rays at yaw
  −fov_half·{1, .71, .43, .14} survive; the other half of the fan never reaches the
  brain through this block (lateral ch76-77, whiskers ch94-97 and the retina remain).
  Contracts I3/I4 were rewritten to the surviving rays and I8 documents the overwrite.
- **Turn horizon.** Witness turn_near/turn_far 2/4 at 1280 ticks: the two failing
  starts are `open_at_deadline` with 20 % stationary ticks — consistent with waiting
  out a red arm (896+128 ticks = 16 s of a 20 s horizon). Turn lessons rerun at 2560
  ticks; the 1280 runs are kept as `*_1280`.

## Results (21:12; full tables in `<root>/CONTRACTS.md`, raw in `contracts.json`)

Wall time: 30 runs, 7–20 s each, ≈6 min GPU. Binary `f294c1dd`. Every run exit 0
(move_v4 exit 2 by the binary's own guard).

### Inputs — all PASS for every genome and lesson

I1 has_task, I2 sub-goal ahead (127), I3 near wall ray (ch16 = 110 → 8 m sample; 95
→ 15 m on service starts), I4 centre ray open, I5 ch183 = 127·d/(d+100) within 0.5
i8 on motor lessons and ≤ 5.1 with a task on service lessons, I6 ch65 = 2·speed
within 0.55 m/s, I8 ch20 = fuel (124–126) confirmed on all 30 runs. I7 lateral cue
after the box: witness 114/97, founder 79, candidate 69; champion48 10 because its
observed replica never reached the box (see below). **The sensors deliver what the
contracts ask; the failures below are not sensory.**

### Outputs — judge success per lesson (4 starts, seed 11)

| lesson | witness | founder (gen 0) | champion48 (thaw gen 11) | candidate (city gen 3) |
|---|---|---|---|---|
| move (30 m) | 4/4, 5.26 m/s | 2/4, max 1.3–2.6 m/s | 2/4, max 1.7–4.5 m/s (start 2 misses the 3 m disc by 0.1 m) | 0/4, max 0.7–1.5 m/s, up to 95 % stationary |
| stop (3 m/s then hold) | 4/4, hold 128 | 0/4 (`no_approach_speed` / deadline) | 0/4 (same) | 0/4 (deadline) |
| turn_near (2560) | 4/4 | 0/4 | 0/4 | 0/4 |
| turn_far (2560) | 4/4 | 0/4 | 0/4 | 0/4 |
| approach_passenger (city stations, 60 m) | 4/4 | 2/4 | **0/4** (max 6.8–13.1 m/s, best remaining 1.8–11 m, blocked 11–20 s, 29–293 pickup requests, 1 contact) | 0/4 (max 1.1–1.4 m/s) |
| pickup | 4/4 | 2/4 | 0/4 | 0/4 |
| delivery48 (artificial course) | 4/4 | 2/4 (`blocked_without_progress`) | **4/4** (replicates `stable-four-start-confirmation`) | 1/4 |

Minimum-speed contract (O2, ≥ 3 m/s on every start): witness PASS everywhere;
founder/champion48/candidate FAIL on move, stop and both turns; champion48 passes it
only on approach/pickup (6.8–13 m/s) and fails it on its own delivery48 course
(2.7–5.4 m/s). No contact on any motor lesson for any genome (O3 PASS).

### Mechanisms seen in the traces (observed replica 0, `observation.jsonl`)

1. **Junction freeze.** turn_near, start 0 (x 133.5 → junction centre 96, red for the
   first 1024 ticks): all three learned genomes crawl at 0.4–2.8 m/s through the red
   (they do not stop for it), then at x ≈ 110–112 (champion48, candidate) or 105.9
   (founder) they stop dead — throttle 0, brake 0.2–0.4, steer 0.1–0.5 — and never
   move again for the remaining 20–30 s, including while the light is green
   (champion48 stops at tick 1296 on green). The box edge is at 103.5 m, the light
   line at 106.5 m, so the stop happens 4–6 m before the line and ≈ 15 m before the
   centre, where the crossing wall is ≈ 22 m ahead. Starts 2/3 of the same genomes
   reach the box at 7–10 m/s and either drive straight through (code 6) or stop inside
   it (`open_at_deadline`, remaining 11–15 m). No learned genome produces a turn.
2. **Course overfit.** champion48 delivers 4/4 on the artificial 48 m straight course
   but 0/4 on approach_passenger against real city stations, where it arrives fast
   (13 m/s), gets within 1.8 m and then spends 18–20 s blocked emitting pickup
   requests that are never accepted. The founder it was bred from passes 2/4 there.
3. **City selection made the being slower and stiller.** candidate (3 generations of
   city selection from founder) is worse than founder on every lesson: move 2/4 → 0/4,
   peak speed 1.3–2.6 → 0.7–1.5 m/s, stationary fraction up to 0.95, delivery48 2/4 →
   1/4. This is the "judge trains parking" diagnosis of the gap analysis measured
   directly, without a city trip.

### Predictions vs measured

- witness all-pass: PASS at 2560 for turns (1280 was too short for a red wait), as
  predicted otherwise.
- champion48 move PASS / O2 FAIL: half right — 2/4, not 4/4; O2 FAIL as predicted.
  stop 0/4 with `no_approach_speed`: as predicted. turns 0/4: as predicted.
  approach/pickup PASS: **wrong** — 0/4 (mechanism 2). delivery48 4/4: as predicted.
- founder/candidate same as champion48: **wrong in the direction that matters** —
  founder beats champion48 on approach/pickup and candidate is worse than both.
- I7 PASS with turns failing → motor gap, not perceptual: holds for founder and
  candidate (cue 69–79 present, no turn produced).

### What this says about the attack order

1. **Judge v3 first — confirmed and sharpened.** The bench shows the reward gradient
   already ran the wrong way inside 3 generations (mechanism 3), and that the stop
   lesson's 3 m/s gate is never met by any learned genome. Step 1 stays first; its
   A/B must use this bench as the gate (all four motor lessons + approach on city
   stations), not the 48 m course.
2. **T-turn rung is real but blocked by the freeze, not by the turn.** No genome even
   arrives at the box under control on 2 of 4 starts. Before a turn rung, a
   "cross a junction at ≥ 3 m/s on green" rung is the missing step.
3. **Perception v6 can wait.** Inputs pass; the lateral cue exists under v3 and the
   being does not use it. The one-sided ray fan (I8) is a design limitation to record,
   not the cause of today's failures.
4. Process: the bench itself is the gate. A genome that fails move/stop/approach here
   must not be sent to a city battery.

### Gate command

```
nice -n 10 .venv-rocm/bin/python Universe/scripts/contract_bench.py run --root <dir>
.venv-rocm/bin/python Universe/scripts/contract_bench.py report --root <dir>
```


## Follow-up 1 (21:15) — speed depends on world heading

Peak speed of the same genome by start heading, motor lessons + approach (n per
heading 7/7/4/2):

| heading | witness | founder | champion48 | candidate |
|---|---|---|---|---|
| −X | 5.1 | 4.4 | 6.8 | 1.8 |
| −Z | 5.2 | 1.9 | 3.2 | 1.2 |
| +Z | 5.3 | 4.6 | 3.7 | 4.5 |
| +X | 5.3 | 10.3 | 10.0 | 7.4 |

The scripted witness is heading-invariant, so body and physics are symmetric. Every
learned genome, including the gen-0 founder, drives 3–5× faster heading +X than −Z.
The asymmetry is inherited and must enter through an input that carries the world
heading. Probe queued: champion48/move with `--vision-being 1..3` so that all four
start headings have a tick-16 wave record; diff the 192 channels by heading.
Prediction: a small set of channels (candidates: ch32-39 wall_curvature "flow ×
forward", ch64 alignment, retina content by lane side under the F1 lane rule) differs
systematically by heading while the lesson channels 178/179/183 are identical.

## Follow-up 2 (21:15) — attack-order step 1 proxy: hard speed gate in the judge

`evolve-stop-champion48/`: school `stop` (success requires ≥ 3 m/s before the area,
then a 128-tick hold), 32 beings × 12 generations × 4 starts × 1280 ticks, seeded from
champion48 with the learning-run mutation rates (`--p-mutate 0.003 --sigma 0.04
--motor-p-mutate 0.15 --motor-sigma 0.08`), seed 11, observation v3.
Prediction: elite stop success 0/4 → ≥ 2/4 within 12 generations, i.e. a judge that
requires speed produces speed on the starts where the body already reaches 4–6 m/s
(−X, +X); the −Z starts may stay below 3 m/s because of the heading asymmetry above.
Rival: stays 0/4 → speed is bounded by the inherited input asymmetry, not by the judge.
The evolved elite is then run through the contract bench (`stop_evolved`).

### Follow-up 2 result (21:17) — arm A, `--school-select worst`, no shaping

| gen | successes/128 | elite success | elite peak speed by start (m/s) | population mean peak | failure codes |
|---|---|---|---|---|---|
| 0 | 0 | 0/4 | 1.7 / 5.9 / 2.2 / 2.2 | 3.0 | no_approach_speed 64, deadline 64 |
| 2 | 14 | 1/4 | 4.5 / 10.3 / 8.0 / 6.7 | 5.3 | overshoot 35, deadline 62 |
| 4 | 12 | 0/4 | 6.1 / 12.5 / 10.1 / 9.3 | 8.1 | overshoot 69 |
| 8 | 3 | 0/4 | 9.8 / 13.1 / 11.8 / 11.9 | 10.2 | overshoot 101 |
| 11 | 1 | 0/4 | 9.5 / 13.6 / 12.0 / 12.3 | 12.1 | overshoot 106, contact 11 |

Prediction half right. A hard speed requirement in the judge produces speed at
once — the 3 m/s gate is cleared by the whole population by generation 6 and the
−Z starts reach 9.5 m/s, so the heading asymmetry is not a speed ceiling. But the
elite never stops: successes peak at 14/128 in generation 2 and decay to 1/128
while `overshoot` climbs to 106/128. Score arithmetic explains it: passing through
the area yields progress ≈ 1 → +500 − 200 (overshoot) = +300, while a slow being
that never reaches the area keeps +500·progress ≈ +250; with worst-of-4 selection
the fast overshooter wins every generation. The current judge fails in both
directions — no speed term trains parking, a speed gate with a −200 overshoot
penalty trains flying through. Judge v3 must make "stop inside" dominate "pass
through" incrementally. Two arms queued on the same seed/population: **B**
`--school-stop-shaping` (+300·(1 − min_speed_inside/v_min) + 200·dwell/hold, Codex
stage 4) and **C** `--school-select mean`. Predictions: B ≥ 2/4 elite successes by
generation 12 and successes/128 rising, not decaying; C alone stays ≤ 1/4 (the
selection rule is not the cause, the score is).

### Follow-up 1 result (21:18) — the heading asymmetry is the F1 lane-rule mirror

`heading-probe/champion48-vb{1,2,3}` + `champion48/move` (vb0): tick-16 wave of
the same genome at the four move starts. Lesson channels 178/179/183 are identical
(127/0/29). What differs is the *side* of the scene: heading −X (horizontal street,
start (156, 99)) sees the mirror image of what the three vertical-street starts see —

| channel | −Z (vb0) | **−X (vb1)** | +Z (vb2) | −Z (vb3) |
|---|---|---|---|---|
| retina FAR pool ch0-3 / ch4-7 | 97 94 88 75 / 39 62 73 81 | **81 76 67 51 / 63 84 92 97** | 97 94 88 75 / … | 96 93 87 74 / … |
| wall rays ch16-19 | 110 110 95 31 | **95 0 52 0** | 110 110 95 31 | 110 110 95 31 |
| lateral distance ch76 / ch77 | 19 / 38 | **38 / 19** | 19 / 38 | 19 / 38 |
| ch81 | −50 | **+50** | −50 | −50 |
| wall EMA ch85 / ch86 | 50 / 60 | **60 / 50** | 50 / 60 | 50 / 60 |
| whiskers ch96 / ch97 | 38 / 0 | **0 / 38** | 38 / 0 | 38 / 0 |
| fovea/periphery HD ch128-159 | mirrored | **mirrored** | as vb0 | as vb0 |

Geometry: with `l0 = (−f.z, f.x)`, the −Z start at x = 99 and the +Z start at
x = 93 both sit at +3 m along l0, but the −X start at z = 99 sits at −3 m: the
grid city's lane rule (`organism_city::build_city`, row ≤ centre → +X, col ≤ centre
→ +Z) puts horizontal-street traffic on the opposite side from vertical-street
traffic — finding F1 of `META/CLAUDE-UNIVERSE-AMBIENTE-20260925.md`, now visible
in the sensors. The learned policy is tuned to one mirror image: 1.2–4.6 m/s on
vertical streets, 4.4–10.3 m/s on horizontal ones (witness: 5.3 everywhere). This
also explains the "every lineage overfits 2–3 of 10 starts" result of the school
stage 5 without invoking memorisation. **Fix belongs to the city (Claude): one
consistent right-hand rule for the grid family, behind a flag so old champions can
be re-measured under both.** Build queued for the idle window.

### Arm A elite through the contract bench (`stop_evolved`, gen 11, 21:17)

| lesson | champion48 (parent) | stop_evolved (12 gens, speed gate) |
|---|---|---|
| move | 2/4, 1.7–4.5 m/s | **4/4, 9.5–12.9 m/s, no contact, 2–5 % stationary** — first learned genome to pass move on all four headings |
| stop | 0/4 no_approach_speed | 0/4 overshoot (+1 contact), 9.5–13.6 m/s |
| turn_near / turn_far | 0/4 deadline | 0/4 drove_through_junction, 7–15 m/s, contacts |
| approach_passenger / pickup | 0/4 blocked | 0/4 blocked, 15–17.6 m/s, 1–3 contacts per start |
| delivery48 | **4/4** | **0/4** blocked_without_progress |

Twelve generations under a speed gate buy motion on every heading (the lane-rule
mirror is not a ceiling once selection demands speed) and destroy every stopping
skill, including the parent's own course. The two failure modes are the two ends
of one dial; judge v3 has to hold both (fast between, stopped inside) at once,
which is what arm B (stop shaping) tests.

### Arms B and C (21:22) — same seed, population and mutation as arm A

| arm | change | successes/128 (gen 2 → 11) | elite success | population peak speed gen 11 |
|---|---|---|---|---|
| A | worst-of-4 selection, no shaping | 14 → 1 | 1/4 (gen 2–3) then 0/4 | 12.1 m/s |
| B | + `--school-stop-shaping` | 14 → 19 → 9 → 2 | 1/4 (gen 2–8) then 0/4 | 11.4 m/s |
| C | `--school-select mean` | 17 → 27 → 22 | **2/4 stable from gen 2 to 11** | 6.7 m/s |

Prediction for B (≥ 2/4) wrong; prediction for C (≤ 1/4) wrong. Under worst-of-4
selection a being with one success and one −200 overshoot scores below a being
that fails all four by deadline, so partial competence is bred out and the shaping
term cannot change the winner (B's generation-10 elite is bit-identical to A's).
Mean selection keeps the partially competent beings alive and holds 2/4 — but no
arm reaches 3/4 in 12 generations. Queued: D = mean + shaping, E = mean × 24
generations. Prediction: D ≥ 3/4 by generation 12 (shaping now has a carrier);
E plateaus at 2/4 without a score change (stopping inside is not yet rewarded
more than passing through).

### Stop successes by start heading, generation 11 (population of 32 × 4 starts)

| arm | −X (horizontal street, mirror image) | −Z (vertical) | +Z (vertical) |
|---|---|---|---|
| A worst | 0/32 | 1/64 | 0/32 |
| B worst + shaping | 0/32 | 1/64 | 1/32 |
| C mean | **0/32** | 16/64 | 6/32 |

Arm C's 2/4 plateau is the F1 mirror: its elite stops on both vertical starts
(min speed inside 0.0) and overshoots the horizontal one at 9.1 m/s. No being in
any arm ever stops on a horizontal street. The lane-rule fix (`--city-lane-rule
consistent`, written, uncompiled) is therefore a prerequisite for judge work, not
a later city item: under the legacy rule half of the starts are a different task.

### Arms D and E — results (21:25)

Same genome, seed and flags as arm C; D adds `--school-stop-shaping`, E runs 24
generations. Column `succ/128` is the whole population (32 genomes × 4 starts);
`elite` is genome 0 of that generation (the elite carried over).

| arm | gen | succ/128 | elite succ/4 | elite v_max | elite min_inside per start | elite codes |
|---|---|---|---|---|---|---|
| C mean | 2 | 17 | 2 | 7.3 | – / 5.3 / 0.0 / 0.0 | 16×2, 0×2 |
| C mean | 11 | 22 | 2 | 9.1 | 0.0 / 7.4 / 0.0 / 0.8 | 0×2, 5, 16 |
| D mean+shaping | 2 | 18 | 2 | 7.3 | – / 5.3 / 0.0 / 0.0 | 16×2, 0×2 |
| D mean+shaping | 7 | 19 | **3** | 8.6 | 0.0 / 6.8 / 0.0 / 0.0 | 0×3, 5 |
| D mean+shaping | 11 | **35** | 3 | 8.6 | 0.0 / 6.8 / 0.0 / 0.0 | 0×3, 5 |
| E mean 24 gens | 13 | 19 | 2 | 9.1 | 0.0 / 7.4 / 0.0 / 0.8 | 0×2, 5, 16 |
| E mean 24 gens | 14 | 17 | **3** | 8.6 | 0.0 / 7.1 / 0.0 / 0.0 | 0×3, 5 |
| E mean 24 gens | 23 | 29 | 3 | 8.6 | 0.0 / 7.1 / 0.0 / 0.0 | 0×3, 5 |

Prediction check. "D ≥ 3/4 by generation 12": PASS (3/4 from generation 7). "E
plateaus at 2/4 without a score change": FAIL — E reached 3/4 at generation 14
with the plain score; shaping bought the same step 7 generations earlier, not a
different ceiling. Both elites end with the identical shape: three starts stopped
(min speed inside 0.0), the fourth overshot (code 5) at 6.8–7.1 m/s.

Population success by start heading, last generation:

| arm | −X (horizontal) | −Z | +Z |
|---|---|---|---|
| D gen 11 | 0/32 | 23/64 | 12/32 |
| E gen 23 | 1/32 | 17/64 | 11/32 |

The fourth start is the horizontal street in every arm (E gen 23: 1 success in
32 across the whole population; D: 0). The ceiling of the stop lesson under the
legacy lane rule is 3/4 for any selection or shaping variant tried, and the
missing start is the mirror image, not a speed problem: the elite drives that
street at 8.6 m/s and never brakes. Judge changes cannot buy the fourth start;
the lane-rule fix can (prediction for the paired legacy/consistent bench, below).

Cost: D 12 gens 1:32 min, E 24 gens 2:18 min (bench.log 21:20:04 → 21:24:54).

### Paired lane-rule bench (pre-registered 21:29, binary rebuilt with the flag)

Script `arms_lane.sh` in the bench directory. One binary for every arm (the
rebuilt tree, which also carries Codex's uncommitted numerical edits — so arm G
repeats arm D on the new binary as the control for that). Stop lesson, 4 starts,
seed 11, same flags as arms A–E.

| arm | genome | rule | gens | prediction |
|---|---|---|---|---|
| eval-eliteD-legacy | arm D elite (3/4) | legacy | frozen | 3/4, −X start overshoots (replicates D gen 11 if Codex's edits are neutral) |
| eval-eliteD-consistent | arm D elite | consistent | frozen | 4/4: the horizontal start is no longer a mirror image; vertical starts unchanged |
| eval-champ-legacy | base champion | legacy | frozen | 0/4 (arm C gen 0) |
| eval-champ-consistent | base champion | consistent | frozen | 0/4; speed on the horizontal start rises toward the vertical ones (4–5 m/s) |
| evo-F consistent, mean+shaping | base champion | consistent | 12 | elite 4/4 by generation 8; −X population successes > 8/32 |
| evo-G legacy, mean+shaping, new binary | base champion | legacy | 12 | replicates arm D (3/4 at gen 7, −X 0/32); a different number flags Codex's edits |

If eval-eliteD-consistent is still 3/4 with the −X start overshooting, the
mirror hypothesis is refuted and the horizontal-street failure lives elsewhere
(start cell geometry or the traffic light at 10.5 m).

#### Lane bench v1 results (21:28–21:31) — pairing invalid, one control passed

| arm | starts (x,z → heading) | result |
|---|---|---|
| eval-eliteD-legacy | (99,156)−Z (156,99)−X (93,228)+Z (99,348)−Z | 3/4; −X overshoot code 5, v_max 8.6, min inside 6.8 — replicates D gen 11 exactly |
| eval-eliteD-consistent | (99,156)−Z (93,228)+Z (99,348)−Z **(291,156)−Z** | 1/4: codes 0, 16, 16, 8 |
| eval-champ-legacy | as legacy above | 0/4, v_max 1.7 / 5.9 / 2.2 / 2.2 |
| eval-champ-consistent | as consistent above | 0/4, v_max 1.6 / 2.2 / 2.3 / 4.7 |
| evo-F consistent, mean+shaping, 12 gens | all four starts vertical | elite 3/4 from gen 4; fourth start (291,156) code 16 with min inside 0.0 |
| evo-G legacy, mean+shaping, new binary | as arm D | **identical to arm D in every generation** (35/128, elite 3/4, v_max 8.6, codes {0×3, 5}) |

Why the pairing broke. `organism_school::lesson_start_candidates` places each
candidate with its own lane offset (`lane_offset_for`, the legacy convention)
and then drops any candidate whose built flow disagrees with the approach
direction. Under the consistent rule every horizontal candidate disagrees, so
the pool lost the −X start and pulled (291,156) in as the fourth. The v1
"consistent" arm also flipped the vertical centre line (`col < centre` instead
of `<=`), which is why the three shared vertical starts did not reproduce the
legacy numbers (start 3: code 0 → 16). Neither difference is the mirror effect
the bench was meant to isolate.

What v1 still says. (1) Arm G is a clean replication: the rebuilt binary,
which carries Codex's uncommitted edits, reproduces arm D to the digit on the
legacy path, so those edits are neutral for the stop lesson under these flags.
(2) Evo-F reached the same 3/4 plateau with four vertical starts, so "one start
of four never passes" is not specific to the horizontal street; the elite
memorises three geometries and misses whichever is fourth. The fold evals
below test that directly.

Fixes (uncommitted until v2 runs): the consistent rule now touches only
horizontal lane cells (vertical streets byte-identical; the unit test counts the
flipped cells and requires each to be an exact sign flip), and the school lane
offset follows `lane_rule()` so horizontal candidates survive.

#### Pre-registered, chain `arms_lane2.sh` (21:32)

| arm | prediction |
|---|---|
| eval-eliteD-legacy-validation (fold 1, 4 new starts) | ≤ 2/4 — memorised starts |
| eval-eliteD-legacy-test (fold 2) | ≤ 2/4 |
| eval-eliteD-legacy-pool8 (8 starts, all folds) | ≤ 4/8; the three trained vertical starts pass, the rest mostly not |
| v2-eval-eliteD-legacy | 3/4, identical numbers to v1 legacy (control) |
| v2-eval-eliteD-consistent | vertical starts identical to legacy (3 successes, same v_max); −X start now at (156,93): if the mirror hypothesis holds it stops (4/4), if not it overshoots as before |
| v2-eval-champ-consistent | vertical starts identical to legacy; −X start speed drops toward the vertical 1.7–2.2 m/s if the mirror made the horizontal street "fast" |
| v2-evo-H consistent, mean+shaping | elite 4/4 by gen 8 if the mirror hypothesis holds; 3/4 plateau otherwise |

#### Fold evals of the arm-D elite (legacy rule, old lane binary, 21:32)

| fold | start (x,z) → heading | result | v_max | min inside |
|---|---|---|---|---|
| validation | (156,99) −X | code 5 overshoot | 8.6 | 6.8 |
| validation | (156,291) −X | code 5 | 9.2 | 7.4 |
| validation | (348,99) −X | code 3 | 10.2 | 8.6 |
| validation | (291,348) −Z | code 5 | 8.6 | 6.3 |
| test | (93,228) +Z (trained) | success | 5.8 | 0.0 |
| test | (228,93) +X | code 5 | 9.7 | 8.1 |
| test | (228,285) +X | code 5 | 11.1 | 9.5 |
| test | (348,291) −X | code 5 | 7.9 | 6.2 |
| pool 8 | the 4 trained + (156,291) −X, (228,93) +X, (291,156) −Z, (348,99) −X | 3/8: only the three trained vertical starts | | |

Prediction "≤ 2/4 on held-out folds": PASS (0/4 and 1/4; the single success is a
trained start, since fold membership is list index mod 3 and the training
pool of 4 spans all three folds — folds are not disjoint from `--school-start-pool 4`).

Two effects, now separated:

1. **Heading, not location.** All six horizontal starts (±X, four different
   junctions) overshoot at 7.9–11.1 m/s with the minimum speed inside the area
   above 6 m/s; the being never brakes on a horizontal street anywhere. Under
   the legacy rule the being drives on the opposite side of the road on those
   streets, so the surviving one-sided wall fan (ch16–19) sees the near wall on
   the other side — the cheapest mechanism for "fast and blind" on one axis.
   v2 of the lane bench tests exactly this.
2. **Memorisation on the vertical axis.** The two untrained vertical starts
   fail as well, in different ways: (291,348) −Z overshoots at 8.6 m/s,
   (291,156) −Z crawls at 3.0 m/s and times out. Three trained geometries
   pass, nothing else does. A 4-start pool with `--school-eval-starts 4` is
   a memorisation bench; the judge-v3 gate must rotate through a pool of ≥12
   starts per generation (`--school-start-pool 12` or larger) before any
   success count means "can stop".

#### Lane bench v2 — paired frozen evals (binary `gpu_evolve_lane2`, 21:33)

| genome | start | legacy | consistent |
|---|---|---|---|
| elite D | (99,156) −Z | success, v_max 5.81, score 1494.8 | **identical** |
| elite D | (93,228) +Z | success, 5.83, 1488.8 | **identical** |
| elite D | (99,348) −Z | success, 4.79, 1488.4 | **identical** |
| elite D | −X start: (156,99) legacy / (156,93) consistent | code 5 overshoot, v_max 8.58, min inside 6.76 | code 8 no approach speed, v_max **2.46**, dist 26.1 |
| champion | (99,156) −Z | code 16, 1.68 | identical |
| champion | (93,228) +Z, (99,348) −Z | code 8, 2.18 / 2.16 | identical |
| champion | −X start | code 16, v_max 5.92, min inside 3.59 | code 16, v_max **1.10**, dist 10.8 |

Control: PASS — the three vertical starts reproduce to the last digit under both
rules and both binaries (v1 legacy, v2 legacy, v2 consistent), as the byte-identical
vertical flow field requires.

Mirror hypothesis, split verdict. **Speed: CONFIRMED.** Put the horizontal
start on the same side of the road as the vertical ones and the "fast street"
disappears: elite D 8.58 → 2.46 m/s, champion 5.92 → 1.10 m/s, both now inside
the range they show on vertical streets (1.7–5.8). The heading asymmetry
measured at 21:22 was entirely the lane mirror. **Stop: REFUTED as the
explanation for the missing fourth start.** With the mirror removed the elite
does not stop on the horizontal start either; it now fails like the untrained
vertical starts of the fold evals (crawl, code 8). The fourth start is missing
because the elite has memorised three geometries, not because of the mirror.

Consequences for the attack order:

- The lane-rule fix is real and cheap, and it removes a systematic perceptual
  asymmetry (one axis "fast and blind"); every future stop/turn/junction rung
  should run with `--city-lane-rule consistent`, and old champions must be
  re-evaluated under it (their horizontal-street speed was an artefact).
- Judge v3 must break the memorisation, and the bench shows how: a rotating
  start pool (≥12 starts, 4 per genome per generation) with held-out folds as
  the gate. Arm I (queued) measures the cost of that change on the same
  champion.

#### Pre-registered arm I (`arms_lane3.sh`, 21:36)

Same champion, consistent rule, mean + shaping, `--school-start-pool 12
--school-eval-starts 4` (each genome sees a rotating batch of 4 of 12 starts
per generation), 24 generations; then the elite of I and the elite of H are
each frozen on a 20-start pool (starts 13–20 never seen by I; 5–20 never seen
by H). Predictions: I's elite ≥ 8/20; H's elite ≤ 4/20 (its 4 trained starts
plus luck). If I ≤ 4/20 too, rotation alone does not buy generalisation and
the judge needs a different signal (e.g. speed-envelope conformity over the
whole approach, not the area).

#### Evo H — consistent rule, mean + shaping, 12 gens (21:34)

| gen | succ/128 | elite | −X successes /32 | −Z /64 | +Z /32 |
|---|---|---|---|---|---|
| 2 | 18 | 2/4 | 2 | 8 | 8 |
| 7 | 25 | 2/4 | 7 | 15 | 3 |
| 10 | 36 | 3/4 | 10 | 18 | 8 |
| 11 | 33 | 3/4 | **13** | 14 | 6 |

Elite at gen 11: (99,156) −Z stop at 4.22 m/s; (156,93) −X **stop** at 4.74;
(93,228) +Z stop at 3.14; (99,348) −Z **fails** (crawl 2.08 m/s, code 16).

Prediction "4/4 by gen 8 if the mirror hypothesis holds": FAIL, and the
failure is informative. The horizontal start is now as learnable as the
vertical ones (13/32 of the population stop there, against 0–1/32 in every
legacy arm), so the lane mirror was what made that street unlearnable. But the
elite still holds three geometries: to gain the −X start it dropped (99,348),
a start the legacy elite D had. Same 3/4 ceiling, different missing start —
capacity, not geometry. Arm I (rotating pool of 12) tests whether the ceiling
moves when no fixed 4-start batch can be memorised.

#### Arm I and the 20-start evaluations (21:37)

Arm I (consistent, mean + shaping, `--school-start-pool 12 --school-eval-starts 4`,
24 gens): population successes 7 → 21 → 18 → 27 of 128 at gens 5/11/17/23; the
elite's own count oscillates 0, 2, 2, 1 — with each genome judged on a
different batch of 4, the elite of one generation is often a lucky draw.

| elite | trained starts | held-out starts | total /20 |
|---|---|---|---|
| I (pool 12) | 5/12 | 0/8 | 5 |
| H (fixed 4) | 4/4 | 3/16 (starts 13, 15, 19) | 7 |

Predictions "I ≥ 8/20, H ≤ 4/20": both FAIL, in the direction opposite to the
bet. Rotating the batch per genome did not buy generalisation in 24
generations; it lowered the selection signal (the score of a genome now
depends on which 4 of 12 starts it drew) and the fixed-batch elite generalises
better (3 of 16 unseen starts) than the rotated one (0 of 8).

Reading. The policy the population finds is scene-keyed: a witness pilot that
reads only ch183 (distance to target) and ch65 (speed) stops 4/4 anywhere,
but evolution from this champion finds start-specific stops first, three at a
time, and a noisier judge just slows that down. The judge-v3 gate therefore
needs two things the bench can measure: (1) the same batch for every genome
in a generation, rotating across generations (`--school-shared-start rotate`
semantics, extended to K starts), so scores are comparable; (2) a held-out
pool in the report every generation, so the plateau is visible as it forms.
Whether the 3-geometry ceiling is brain capacity or a scenery shortcut is the
next intervention: silence the retina during the stop lesson and see whether
the population is forced onto ch183/ch65.

#### Pre-registered arms J and K (`arms_lane4.sh`, 21:39)

No flag silences the retina (only `--ablate cfc/l1/l2` output masks and the
`--research-controller linear` controller swap exist), so the retina control
needs code and waits. Two judge shapes that exist today:

| arm | shape | prediction |
|---|---|---|
| J | fixed batch of 8 starts for every genome, every generation, 24 gens | elite plateaus at 4/8 (capacity ≈ 3–4 geometries); if it reaches ≥ 6/8 the ceiling was the 4-start bench, not capacity |
| K | `--school-shared-start rotate`, K = 1: one start per generation, the whole population on it, 24 gens | elite on pool 20 ≤ 4/20; moving target with a single sample per genome never settles |

Both elites are then frozen on the 20-start pool (J's held-out = starts 9–20;
K saw starts 0–23 mod count once each).
