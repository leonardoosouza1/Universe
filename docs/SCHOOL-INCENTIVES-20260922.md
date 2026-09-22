# School Stage 3 — incentives and the learning mechanism (2026-09-22)

Pre-registration and results for item 3 of the 2026-09-22 program
(`META/BRAINSTORM-ESCOLA-EVOLUCAO-UNIVERSE-2026-09-22.md` §4–§5, R2). Stage 2
(`docs/SCHOOL-LESSONS-20260922.md`) delivered the lessons and the judge; this stage
fixes what the being is paid for and how a life can change a brain. Branch
`organism/school-20260922`.

## 1. Scope declared

| Item | Delivered | Left out (why) |
|---|---|---|
| S3.1 phase-aware task progress with best-distance memory | `taxi.wgsl` post kernel, body slots 31/61/62 | no reward for parking, odometer or oscillation by construction; no curriculum over passengers (Stage 5 city) |
| S3.2 fitness decomposition per being | `organism_result_gen*.json` `fitness_components` + task memory | UI surface (Stage 5) |
| S3.3 online feedback asymmetry | GPU fixture proving the legacy cycle bonus; `--feedback-mode symmetric` behind a flag, default legacy | no change to the default (needs a paired A/B, rule 3 of `CLAUDE.md`) |
| S3.4 plasticity frozen mode | `LIFE_PLASTICITY_FROZEN` bit 3, `--plasticity-freeze-after <tick>`, fixture checks | plasticity restricted to the 3 motor outputs was already the case (`brain.wgsl` residual on intents 0..2); τ=0.5 s is a gene, not forced |
| S3.5 curriculum + start folds | `--school-curriculum adaptive|uniform`, `--school-lessons`, `--school-advance-p/-gens`, `--school-start-fold train|validation|test`, `--school-mode life` | Stage 4 battery (3 seeds, equal being-ticks) not run here |

## 2. Mechanisms

### 2.1 Task progress (slot 127)

```
key   = onboard ? passenger_id*2+1 : (waiting ? waiting_id*2 : 0)
goal  = onboard ? destination : waiting station
r     = route_distance(pos, goal)             // road BFS metres, -1 unreachable
if key != body[31]: body[31]=key; body[61]=r; body[62]=r        // new phase/target
if r >= 0 and (body[61] < 0 or r < body[61]): body[61]=r        // best so far
approach = (body[62] >= 8 and body[61] >= 0) ? 100*(1 - body[61]/body[62]) : 0
127 = (onboard ? 100 : 0) + approach
```

Consequences: driving toward the passenger pays once per metre closed; driving back,
parking or oscillating adds nothing; a target already within 8 m pays nothing (no
birth bonus); a swapped target resets the memory; an unreachable start pays nothing
for that key. Bench school runs have no passengers, so 127 stays 0 there.

### 2.2 Online reward (`organism_feedback.wgsl`)

| mode | reward |
|---|---|
| legacy (default) | `clamp(delivered − 2·damage + 0.2·max(0,ΔE) + 0.01·min(0,ΔE), −1, 1)` |
| symmetric | `clamp(delivered − 2·damage + 0.01·ΔE + 0.01·min(0, Δmoney/25), −1, 1)` |
| progress | symmetric `+ 0.01·max(0, Δ127)`, paid only while `body[31] != 0` and equal to the key at the previous decision (lifetime slots 401/402) |

ΔE is the energy change since the previous decision divided by capacity; the money
baseline lives in lifetime slot 400. Legacy pays +0.19·q for a loss q followed by a
recovery q, i.e. it rewards refuelling cycles (§4 of the brainstorm). `progress` is the
dense signal for plasticity: 100 points of same-key approach sum to the same reward as
one delivery pulse; it cannot be farmed by shuttling because slot 127 only grows on a new
best distance and resets on a key change (2.1), and it is bounded by 200 per task.

### 2.3 Frozen plasticity

Flag bit 3 with bit 0: eligibility, P update, retention decay and the latent trace are
skipped; the residual `clamp(ΣP·latent/48)·gain` is still added to the 3 motor intents
and recorded at lifetime slot 384..386. `--plasticity-freeze-after T` sets the bit
before the dispatch of tick T; the next generation's `upload_lifetime` clears it.

### 2.4 Curriculum and folds

Adaptive: stay on the current lesson until p̂ = (S+1)/(T+2) ≥ `advance_p` for
`advance_gens` consecutive generations, then advance (stays on the last lesson).
Uniform: lesson = list[gen % len]. Both consume the same generation budget. Folds:
candidate approaches are ordered deterministically; index % 3 → train/validation/test.

## 3. Pre-registered expectations

| Check | Expected |
|---|---|
| `wgsl_validation` | pass |
| `organism_feedback_tests` (GPU) | legacy cycle +0.095 for q = capacity/2; symmetric cycle 0; paid 5 of 25 → −0.002; income → 0 |
| `taxi_progress_tests` (GPU) | first sight 0; halfway ≈50; return keeps 50; born within 8 m 0; grass start −1 and 0 forever; pickup → 100; destination → 200; delivery fitness > carrying > approach > idle; approach − idle = 50 |
| `school_tests` (GPU, regression) | 10/10 unchanged |
| `lifetime_gpu_check` | `frozen_p_constant` true, `frozen_residual_applied` true, all previous checks unchanged |
| witness, short horizon (no school) | `task_key = id·2`, 0 < 127 < 100, best < initial before pickup |
| witness, full horizon | deliveries ≥ 1, `fitness_components.events` = 1000·deliveries |
| freeze A/B (1 being, 1280 ticks, `--plasticity`) | P (life 36..180) changes after tick 640 without the flag, bit-constant with `--plasticity-freeze-after 640`; residual non-zero in both |
| curriculum smoke adaptive (12×4×1280, seed 7, advance_p 0.3, gens 1) | `[curriculum]` lines; lesson advances move→stop once p̂ ≥ 0.3 |
| curriculum smoke uniform | lessons move, stop, turn_near, turn_far in order |

## 4. Results

### 4.1 Files, commands, binaries

Code: `src/gpu/taxi.wgsl` (phase-aware progress), `src/gpu/organism_feedback.{wgsl,rs}`
(modes 0/1/2, lifetime aux slots 400–402), `src/gpu/lifetime.rs` (aux offsets),
`src/gpu/brain.wgsl` + `src/gpu/brain/mod.rs` (`LIFE_PLASTICITY_FROZEN`,
`set_lifetime_flags`), `src/bin/lifetime_gpu_check.rs` (two frozen checks),
`src/organism_school.rs` (`LessonFold`, `lesson_starts_fold`), `src/gpu/school.rs`
(`Lesson::ALL`), `src/bin/gpu_evolve/{main.rs,cli_flags.rs,context.rs,organism_tick.rs}`
(flags, curriculum loop, manifest, result rows, seven one-line arms made scannable),
new GPU fixtures `src/gpu/organism_feedback_tests.rs` and `src/gpu/taxi_progress_tests.rs`,
`src/gpu/mod.rs` (test modules). Docs: this file, `ORGANISM-RESIDENT-RUNTIME.md`.

Run root `/media/llm-data/universe-runs/school-incentives-20260922/` (`build-v1.sh`,
`validate-v1.sh`, `build-and-validate-v2.sh`, `build-and-validate-v3.sh`, each with its
`.log`; `analyze-v1.py <root>` writes `analysis-v1.json`). Frozen debug binaries, one per
validation round, never replaced while a run used them:

| round | binary | sha256 (prefix) | what changed |
|---|---|---|---|
| v1 | `gpu_evolve-stage3` | `b9121224` | S3.1–S3.5 |
| v2 | `gpu_evolve-stage3-v2` | `dfb6fdf9` | CLI registry rows + block-form arms (4 pre-existing `cli_flags_tests` failures fixed) |
| v3 | `gpu_evolve-stage3-v3` | `bc40df0b` | `--feedback-mode progress` (mode 2) |

Common prefix of every run: `--scenario organism-taxi-v1 --seed 7 --driving-diagnostics
--vision-every 16 --no-spectral-init --navigation-observation v3`, GPU env
`WGPU_BACKEND=vulkan VK_ICD_FILENAMES=…/radeon_icd.json MESA_VK_VERSION_OVERRIDE=1.3`,
adapter `AMD Radeon RX 6750 XT (RADV NAVI22)` asserted by every fixture.

### 4.2 Expected vs observed

| Check | Expected (§3) | Observed (v3 unless noted) | Verdict |
|---|---|---|---|
| `wgsl_validation` | pass | 14/14 | ok |
| `organism_feedback_tests` | cycle +0.095 / 0; paid −0.002; income 0 | 5/5 incl. progress: same key 0→50 → 0.5; parked 0; key change 0; 100→130 → 0.3; drop-off 1.0 | ok |
| `taxi_progress_tests` | ordering delivered > carrying > approach > idle | 2/2; initial 36±3 m; halfway 40..60; return keeps credit; born within 8 m → 0; grass start −1 → 0 | ok |
| `school_tests` | 10/10 | 10/10 (v2) | ok |
| `cargo test --bin gpu_evolve` | — | 123/123 (v2; 119/123 at HEAD before the registry fix) | ok |
| `lifetime_gpu_check` | frozen checks true | `frozen_p_constant` true, `frozen_residual_applied` true, 10/10 checks | ok |
| witness short (400 ticks) | 0 < 127 < 100 | 127 = 0, best = initial = 21 m: the witness first drives *away* along the one-way lane, so the undirected route distance does not fall | **miss** (metric limit, see 4.4) |
| witness full (1920 ticks) | deliveries ≥ 1 | 0 deliveries, 127 = 71.4 (best 6 m of 21), components events 0 / progress 71.4 / money 25 / alive 10 / integrity −0.26 | **miss** (the 2026-09-22 witness runs needed 16 384 ticks for one delivery; not a regression) |
| freeze A/B legacy (v2) | P moves before 640 only with flag off | identical before 640; after 640 max ΔE 0.073 (off) vs 0.0 (on); max ΔP 1.1e-11 (off) vs 0.0 (on); residual ≈ 1e-10 both | mechanism ok; **P ≈ 1e-9 in both**, see 4.3 |
| freeze A/B progress (v3) | as above with a dense reward | bit-identical to the legacy pair (P_abs_max 1.402e-9) | explained: the lone random being never closes distance (127 = 0 at the end), so the progress term is 0 and with ΔE < 0 legacy and progress reduce to the same `0.01·ΔE` |
| witness + progress (v3, 1920 ticks) | non-zero reward, P grows | reward non-zero in 45/120 samples, max 5.2e-3; P_abs_max 1.6e-4 (before tick 640: 8.2e-8); residual on motor max 1.0e-5 | mechanism proven end-to-end, magnitude below effect (4.3) |
| curriculum adaptive (12×4×1280, advance_p 0.3, gens 1) | advances move→stop | gen 0 0/12 p̂ 0.071 → gen 1 3/12 0.286 → gen 2 5/12 0.429 → gen 3 lesson stop 0/12 {corridor 4, overshoot 5}; v1 and v2 lines identical | ok |
| curriculum uniform | move, stop, turn_near, turn_far | in order; p̂ 0.071 / 0.071 / 0.143 / 0.143 | ok |
| fold train (move, 80 candidates) | runs | 1/12 in gen 0 | ok |
| life mode (12×3×1280, `--plasticity`) | passengers on, taxi fitness | legacy: 0/12 → 3/12 → 2/12; progress: 0/12 → 3/12 → 2/12; task keys set for 12/12, approach 0–48 points, 0 deliveries, 0 deaths in every run | ok |

Zero deaths, refusals and damage above 0.3 integrity in all 13 runs; wall time 5.6–8.5 s
per single-being run and 18.4 s per 12×3×1280 life run (debug binary).

### 4.3 Finding: the online plasticity path is wired but three orders of magnitude too weak

The P update is `ΔP = rate·elapsed·reward·gain12·alive·energy^k·E` with defaults
rate 0.05 s⁻¹, gain12 1.0, elapsed 1/64 s, |E| ≤ 0.32 (measured). Without a delivery or a
crash the legacy/symmetric reward is `0.01·ΔE ≈ −2e-7` per tick, hence P ≈ 1e-9 in the
freeze pair. Under `progress` the witness earns 71 points in 1900 ticks, mean reward
3.8e-4 per tick, and P reaches 1.6e-4 against a clamp of 0.25; the residual that reaches
the motor is 1e-5 on intents in [−1, 1]. At the gene bounds (rate 1.0, gain12 4.0) the
same life would reach P ≈ 1e-2 and a residual below 1e-3. Conclusion: with the
fitness-consistent scale of 0.01 per point, no gene value in bounds lets a life change the
motor by more than 0.1 % of its range in 1 280–1 920 ticks. Evolution across generations
(the school) remains the only mechanism that changed behaviour in this stage.

### 4.4 Limitations

- `route_distance` is an undirected road BFS; lanes are one-way, so the lawful path can
  be longer and the first stretch of a correct approach may not reduce the metric
  (witness short). A directed distance is a Stage 5 city item.
- The delivery reward is a one-tick pulse (1.0 for 1/64 s) while progress is a rate;
  the plasticity integrator treats a full delivery as 1 point of progress. Any online
  learning claim needs the reward scale re-declared before running (next test).
- Life-mode competence equals bench competence over 2–3 generations; the modes only
  diverge once passengers are reachable, which needs longer horizons.
- One seed (7), debug binary; no claim about generalisation — that is Stage 4.

### 4.5 Next test

Stage 4 (`docs/SCHOOL-COMPARISON-20260922.md`): uniform vs adaptive curriculum, 3 paired
seeds, equal being-ticks, train fold for evolution, champions re-evaluated on the
validation and test folds via `--resume --resume-mix 1.0` (12 identical copies on 12
held-out starts). Online plasticity stays off there. A reward-scale sweep for mode 2
(0.01 → 0.1 → 1.0 per point, genes at the bounds, measured by residual magnitude and
lesson p̂) is the test that decides whether the plastic path deserves a Stage 4 arm.
