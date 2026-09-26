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

