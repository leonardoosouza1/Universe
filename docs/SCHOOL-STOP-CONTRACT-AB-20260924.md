# Stop contract A/B — pre-registration, 2026-09-24

Follows `GAP-SWEEP-20260924.md` proposed order **A**. Written before the battery ran;
the results block at the end is empty until it is filled with numbers and commands.

> **Correction 2026-09-24 10:25 — the 50 % deadline fails the scripted reference witness.**
> `--taxi-reference-witness` on the stop lesson (seed 11 and seed 7, identical because the
> witness is scripted and the city start is the same): without a deadline it succeeds at tick
> 784 (max 5.18 m/s, entry 1.72 m/s, hold 128 ticks, score 2085); with `--school-deadline-frac
> 0.5` (640 ticks) it fails with code 16. The ceiling computed below (mean ≥ 3.75 m/s) was
> arithmetic, not a witness run, and `TAXI-FEASIBILITY-AND-SELECTION.md:42` says a failed
> witness blocks a feasibility claim. The `deadline` and `both` arms therefore evolved against
> a contract no known controller satisfies. With `--school-deadline-frac 0.7` (896 ticks) the
> witness succeeds (tick 784). Every later battery uses 0.7. The results below stand as
> measured; the crawler-detector conclusion does not depend on the deadline value, the null
> on strict successes partly does.

## Hypothesis

The stop plateau of 09-22/23 is a selection artefact, not a control limit: with the
09-23 contract a genome that never arrives (status *truncated*, no penalty) outscores a
genome that arrives and overshoots (−200), and the mean over three fixed starts lets one
start pay for two. Changing the contract, with the body, world, seeds and starts held
fixed, should move the population from crawling (median vmax 1.3–1.5 m/s on starts 0–1
at generation 47) to arriving.

**Prediction (direction, not magnitude).** Arms `deadline`, `worst` and `both` each raise
the number of genomes strict on ≥ 2 of 3 starts above the control (control was 0 at
generation 47 for both seeds in 09-23). `both` ≥ either single change. Median vmax on
starts 0 and 1 stays above 3 m/s in every treated arm (the crawler detector). If a
treated arm shows *lower* strict counts than the control, the mechanism to look for is
the −200 wall: under the deadline every genome fails until one arrives, so selection
pressure may be flat for the first generations. That outcome refutes "one change is
enough", not the sweep's diagnosis.

**Refuted if** no treated arm beats the control on the primary metric on either seed at
generation 47 (mean of the last 8 generations), or if the treated arms crawl as well.

## Design

| Arm | Flags added to the common line | What differs |
|---|---|---|
| control | — | 09-23 contract: timeout free, mean selection, 3 fixed starts |
| deadline | `--school-deadline-frac 0.5` | attempt still open at tick 640 closes as failure 16 (`open_at_deadline`, −200) |
| worst | `--school-select worst` | genome score = its lowest start score |
| both | both flags | the two changes together |

Common line (`run_battery.sh` in the run directory):

```
--scenario organism-taxi-v1 --school stop --school-mode bench --school-stop-shaping
--school-eval-starts 3 --school-start-fold train --beings 16 --gens 48 --ticks 1280
--school-horizon 1280 --replay-every 47 --seed {7,11} --resume --resume-mix 1.0
```

Population seeded from the move champions of Astra's task-7 runs (generation 15;
`move-champion-s7.bin` sha256 e5102730…, `move-champion-s11.bin` c59498f2…, 99 332 B
each), the same seeds the 09-23 envelope battery used. Envelope OFF in every arm so the
only difference is the contract. Start pool NOT rotated here (order **B** is a separate
battery), so "≥ 2 of 3" still means two of the same three fixed exams.

Seeds 7 and 11, 8 runs × 48 generations × 16 beings × 3 starts. Expected wall time ≈ 5 min
per run at the 09-23 rate. Disk: replays at generations 0 and 47 only, ≈ 8 × 2 × 30 MB.

## Ceiling of the metric

Goal 30 m from each start. Deadline at 50 % of a 1280-tick horizon = 640 ticks = 10 s at
64 Hz; the success hold is 128 ticks (2 s), so 8 s of travel remain: **mean ≥ 3.75 m/s**
to succeed under the deadline. The move champion enters the goal area at 9–11 m/s
(gen 0 of 09-23), so the deadline is reachable from generation 0. Strict score ceiling per
start is unchanged (1000 + 500 + shaping ≈ 2100); a deadline failure scores
`500·progress − 200`, so a crawler that covers half the distance by tick 640 scores +50,
below the overshooter's 100.

## Primary and secondary metrics

- **Primary:** genomes strict on ≥ 2 of 3 starts, per generation; reported as mean of
  generations 40–47 and first generation reached, per arm and seed.
- **Secondary (crawler detector):** median max speed per start over the population,
  generations 0 / 24 / 47. Plus failures by code (5 overshoot vs 16 deadline), and the
  closing tick of successes (a success at tick > 640 in `control` is a creep).
- Every number cites the arm directory and `school_result_genNNNN.json`.

## Provenance

Binary built from the commit named in `commit.txt` in the run directory; every per-gen
manifest carries `provenance.git_commit` since this commit. Fixtures run before the
battery: `cargo test --bin gpu_evolve`, `--lib school::tests`, `--test wgsl_validation`,
`--lib school_tests -- --ignored` (22 GPU fixtures, including
`deadline_turns_an_open_attempt_into_failure_16`).

## Results (filled 2026-09-24 10:15, battery 09:40–10:10)

Run directory `/media/llm-data/universe-runs/school-stop-contract-ab-20260924/` (`run_battery.sh`,
`commit.txt`: 0151933, binary sha256 45e1182e…, 8 runs × 48 generations × 16 genomes × 3 starts).
Statistics are over all 48 replicas of a generation (every replica alive; no death in any arm).

**Primary metric: 0 in every arm and seed.** No genome was strict on ≥ 2 of 3 starts in any of
the 8 × 48 generations. **Prediction refuted.** The treated arms did not beat the control; on
strict successes the control seed 11 (101, all on start 2, closing at tick ≈ 1140 — the creep
of 09-23 reproduced to the number) beats the treated arms (0).

| arm | seed | strict successes (48 gens) | median vmax per start, gen 0 → 24 → 47 (m/s) | fail codes gen 47 | selection score gen 47 min / med / max |
|---|---|---|---|---|---|
| control | 7 | 0 | 11.4/14.3/13.8 → 6.1/11.9/10.7 → **1.4/2.1/1.3** | 38 truncated, 9 overshoot, 1 code 8 | 65 / 207 / 320 |
| deadline | 7 | 0 | same → 11.1/13.6/12.9 → 10.9/13.9/13.2 | 43 overshoot, 3 deadline, 2 contact | 52 / 94 / 99 |
| worst | 7 | 0 | same → 9.9/13.4/13.0 → 11.3/13.1/13.0 | 45 overshoot | 233 / 290 / 298 |
| both | 7 | 0 | same → 9.9/13.4/13.0 → 11.3/13.2/13.0 | 42 overshoot, 2 deadline, 4 contact | 80 / 290 / 298 |
| control | 11 | 101 (start 2: 100, start 1: 1) | 10.3/13.2/13.2 → 2.2/2.9/3.5 → **1.5/1.7/3.2** | 32 truncated, 7 overshoot, 6 contact | −172 / 191 / 892 |
| deadline | 11 | 0 | same → 12.5/14.1/14.0 → 10.8/13.9/13.8 | 39 overshoot, 3 deadline, 6 contact | −22 / 86 / 99 |
| worst | 11 | 0 | same → 12.4/13.4/13.9 → 12.7/12.9/13.3 | 46 overshoot, 2 contact | 205 / 290 / 298 |
| both | 11 | 0 | same → 10.2/13.3/13.9 → 10.5/12.9/13.6 | 35 overshoot, 4 deadline, 9 contact | 37 / 282 / 298 |

Secondary metric (crawler detector): **confirmed in the direction predicted.** Both controls
crawl by generation 47 (median vmax 1.3–2.1 m/s on starts 0–1, 32–38 of 48 replicas truncated);
every treated arm keeps the move champion's speed (median vmax 12.9–13.9 m/s at generation 47)
for the whole run. The contract change removes the crawling attractor.

**Mechanism of the null (the alternative named above, the −200 wall).** Under the new contract
every replica overshoots at 9–10 m/s entry speed (median `envelope_entry_speed_mps` 9.0–10.4 at
generation 47, unchanged from 8.9–10.1 at generation 0). An overshoot scores `500·progress −
200 + shaping ≈ 290–298` whether it enters at 13 m/s or at 5 m/s, and a slower approach that
is still open at tick 640 scores less (progress < 1, same −200). The score is flat across the
whole population (worst arm: 290 / 290 / 298 at generation 47, 48 replicas), so selection has
nothing to climb. The shaping term alone (`--school-stop-shaping`, near-stop dwell) pays only
inside the goal area and only when slow, which no replica is. Nothing pays *decelerating*.

Note on the historical `mean` rule: the failed replica's `R_SCORE` already carries −200
(overshoot 295.8 = 500·0.99 − 200 + shaping) and `aggregate_school_batch` subtracts the
worst-failure penalty again (92.9 = 295.8 − 200 − …). Under `mean` an overshoot therefore
cost 400 against a truncated crawler's 0, which is the size of the crawling incentive that
09-22/23 evolved against. Recorded, not changed: `worst` and K = 1 do not double-count.

**Consequence.** The contract fix is necessary (it stops selecting crawlers) but not
sufficient: a gradient toward slowing down has to exist before the first strict success.
Follow-up launched 10:12 (`run_battery_envelope.sh`, same seeds, same binary): `both` +
`--school-stop-envelope` with the historical 6 m clamp and with a 30 m clamp, which pays a
decelerating approach through `v_allow(d)` on every replica, successful or not.

### Envelope follow-up (10:12–10:27, `run_battery_envelope.sh`, same binary and seeds)

Arms `both + --school-stop-envelope` with clamp 6 m (historical) and 30 m. **Primary metric 0,
strict successes 0 in all four runs.** Entry speed and speed per start unchanged over 48
generations; the envelope conform ratio is 0.00 in every replica of every generation (nobody
ever drives under `v_allow(d)`), so the +200 conform term never pays and only the −100 excess
term acts:

| arm | seed | median entry speed gen 0 → 47 (m/s) | median excess integral (m) gen 47 | selection score gen 47 min / med / max | fail codes gen 47 |
|---|---|---|---|---|---|
| both-env6 | 7 | 10.1 → 9.2 | 6.0 (clamped) | 131 / 190 / 199 | 42 overshoot, 3 deadline, 3 contact |
| both-env30 | 7 | 10.1 → 8.8 | 11.2 | 161 / 246 / 260 | 39 overshoot, 6 deadline, 3 contact |
| both-env6 | 11 | 8.9 → 10.0 | 6.0 (clamped) | −30 / 159 / 196 | 35 overshoot, 7 deadline, 6 contact |
| both-env30 | 11 | 8.9 → 10.0 | 11.4 | −2 / 250 / 259 | 42 overshoot, 3 deadline, 3 contact |

With clamp 30 the excess term is no longer saturated (11.2–11.4 of 30 m), so a gradient exists
on paper (≈ −38 of a 260 score); it did not move the population in 48 generations of 16.

**Brake use, decoded from `gen0000.replay` / `gen0047.replay` (48 beings, frames every 4 ticks,
`final_brk > 0.05`):**

| arm | gen 0: beings that ever brake / at > 5 m/s | gen 47 | median brake-tick fraction gen 0 → 47 |
|---|---|---|---|
| both-s7, both-env30-s7 | 0 / 0 | 5 / 3, 3 / 0 | 0.00 → 0.00 |
| control-s7 | 0 / 0 | 14 / 5 | 0.00 → 0.00 |
| both-env30-s11 | 48 / 32 | 45 / 29 | 0.28 → 0.07 |
| both-env6-s11 | 48 / 32 | 35 / 22 | 0.28 → 0.02 |
| control-s11 | 48 / 32 | 32 / 4 | 0.28 → 0.09 |

Seed 7's lineage has no brake pathway (brake intent never positive at generation 0) and
selection barely grows one. Seed 11's lineage brakes from generation 0, at speed, on every
being — and still enters the goal area at 9–10 m/s and overshoots on every start. Braking exists
but is not aimed at the goal. The next question is therefore perception, not reward: what the
brain is told about the stop target (distance, bearing) — see the sensor check that follows.

### What the brain sees (2026-09-24 10:50, `--driving-diagnostics` on the env30 seed-11 champion)

`observation.jsonl`, being 0, `wave[176..192]` (i8, `navigation_observation: 2` = dx/dz per
100 m): channel 179 runs −38 → +6 as the target distance runs 30 → 0 m (≈ 1.3 units per
metre, 30 % of full scale); channels 185 and 191 also move monotonically (−76 → −32,
63 → 35). The signal is present and monotonic. The champion applies throttle 0.2–0.6 the whole
approach, brake intent stays negative (applied 0.0), speed 0.4 → 10.1 m/s at the goal, overshoot
at tick 494. Not a blind being: a being that never learned to trade throttle for brake on that
channel. Reference witness on the same lesson: throttle 0.62 → 0.16 → 0 by 22 m, coasts to
rest at 2.7 m, no brake at all — a stop is reachable by throttle alone.

### Next battery (pre-registered 10:32): search scale × body, `run_battery_scale_body.sh`

Contract: deadline 0.7, worst-start selection, envelope clamp 30. Population 32 (96 replicas),
120 generations, seeds 7 and 11, move-seeded. Arms: `fixed` body vs `--evolving-body`
(engine/brake/mass genes mutate ±0.1 per generation, expressed at birth; both arms mutate the
organism metabolism block). Ceiling: witness closes at tick 784 of 896; strict score ≈ 2085.
Prediction: with 4× the search of the 48-gen batteries at least one seed reaches a first
strict success on ≥ 2 of 3 starts; the evolving-body arm reaches it no later than the fixed
arm on the same seed (a bigger brake or a smaller engine makes the throttle-only stop easier).
Refuted if both arms stay at 0 on both seeds (then the search is not the bottleneck and the
next lever is the pedal mapper: negative brake intent clipped, `organism_action.wgsl:61`).
Wall estimate ≈ 10 min per run (measured 4.1 s per generation at 96 replicas), 4 runs; disk ≈ 4 × 3 replays × 90 MB.

Reproduce: `bash /media/llm-data/universe-runs/school-stop-contract-ab-20260924/run_battery.sh`
with the binary in that directory; the table comes from `school_result_gen*.json` of each arm
(`aggregates[].selection_score`, `beings[].max_speed_mps/envelope_entry_speed_mps/fail_code`).

### Results of the scale × body battery (filled 2026-09-24 11:55; runs 10:32–11:48, seed 11 relaunched 11:28 after a reboot killed it at generation 83)

Run directories `scale-{fixed,evolving}-s{7,11}` (120 generations each, 32 genomes × 3 starts,
binary sha256 45e1182e…, commit 0151933). The killed partial run is kept as
`scale-fixed-s11.partial-killed-gen83`; its first 83 generations are bit-identical to the
relaunched run (same seed, same binary), so its generation-60 replay is used below.

**Primary metric: 0 strict successes in all four runs, no genome strict on any start.
Prediction refuted on both clauses** (no seed reaches a first strict success; the evolving
body is not earlier than the fixed one — neither ever gets there).

| arm | seed | strict | median entry speed gen 0 → 119 (m/s) | slowest speed inside the area, gen 119 | longest near-stop dwell (< 0.6 m/s) / hold dwell (< 0.3 m/s), ticks | deadline closures (code 16) gen 119 / 96 | elite selection score final, unchanged since gen | body: median brake / engine torque, mass gen 0 → 119 |
|---|---|---|---|---|---|---|---|---|
| fixed | 7 | 0 | 10.7 → 7.7 | 4.01 | 0 / 0 | 22 | 262, gen 81 | fixed 4000 / 620 Nm, 1350 kg |
| evolving | 7 | 0 | 10.7 → 8.4 | 3.12 | 0 / 0 | 14 | 262, gen 82 | 4000 → 3628 / 620 → 593 Nm, 1350 → 1337 kg |
| fixed | 11 | 0 | 9.7 → 6.3 | **0.25** | **62 / 1** | 47 | 472, gen 101 | fixed |
| evolving | 11 | 0 | 9.7 → 7.5 | 2.94 (1.54 at gens 60–104) | 0 / 0 | 44 | 270, gen 72 | 4000 → 3863 / 620 → 562 Nm, 1350 → 1333 kg |

Seed 11 fixed is the closest any school run has come: from generation 101 one genome touches
0.25 m/s inside the area for one tick and stays under 0.6 m/s for 62 ticks on its best start
(score 720), but its worst start scores 472 and no child improved on it in the remaining
19 generations. Seed 7 never gets under 3 m/s inside the area in either arm.

**Mechanism, read from the pedals (generation-60 replay of seed 11, `intent_brk`/`final_brk`
per tick along the road; `analyze_scale.py` and the decode calls are in the session log).**
Three ways of failing, all with the same root:

1. *Coasting crawler* (genome 0, start 1; genome 1, start 1): the fast brake logit sits at
   −0.4 … −0.7 on every tick of the run; the mapper clips it to 0
   (`organism_action.wgsl`, `fast_brake = max(after2 + hip, 0)`), so the being only ever
   coasts. From 1 m/s at tick 704 it reaches 0.3 m/s at tick ≈ 1200: past the deadline
   (896) and with no room for the 128-tick hold in the 1280-tick horizon. The scripted
   witness needed 24 ticks of full brake to close (driving.csv: brake 1.0 at tick 640).
2. *Pumper* (genome 21, start 0): throttle 0.3 and brake 0.3 alternate every ≈ 16 ticks;
   speed oscillates 0.7–1.0 m/s inside the area, never under 0.6, then the being drives
   off after tick 1000.
3. *Late braker* (slot 2, an overshoot): full stop, brake held at 0.2–0.3 for 700 ticks —
   but 36 m along, 6 m past the goal, after the judge had already closed the attempt.

Census over all 96 replicas of the final replays: seed 7 fixed, mean brake logit per replica
−0.48 / −0.38 / −0.14 (min / median / max), positive on 0.5 % of ticks, 7 replicas never
positive; seed 7 evolving, positive on 2.8 % of ticks; seed 11 gen 60, positive on 45 % of
ticks (the pumpers). The seed-7 lineage lives entirely in the clipped half of the brake head:
every mutation of the brake row is phenotypically silent there, so selection cannot see it.
That is the dead zone the pre-registration named as the next lever.

**Body coevolution reading.** `--evolving-body` is functional (genes mutate every generation
for every non-elite child, are expressed at birth, and drift is measurable: brake capacity
4000 → 3628 / 3863 Nm, engine 620 → 593 / 562 Nm over 120 generations) but it is a random walk,
not selection: the brake capacity drifts *down* in both seeds because no brain presses the
brake, so a stronger brake has no fitness effect. Body evolution cannot be validated as
adaptive until the brain has a usable brake. It is therefore run again as an arm of the next
battery, where that condition is met.

### Next battery (pre-registered 11:58): signed brake head, `run_battery_signed_brake.sh`

Lever: `--organism-signed-brake` (new, default off, organism-taxi-v1 only, incompatible with
`--organism-legacy-pedals`). The negative half of the fast brake logit is pushed into the
exclusive net as drive instead of being clipped: `net = throttle_before + max(−brake_raw, 0)
− brake_before_gate`, throttle `clamp(net, 0, throttle_cap)`, brake `max(−net, 0)`. The
slow-brain inhibition path stays rectified (it is never vetoed by a negative fast logit, the
reason the clip was introduced), and with the flag off the shader output is bit-identical to
before (the new `clamp` on the throttle side is a no-op without the flag because
`net ≤ throttle_before ≤ throttle_cap`). GPU fixture:
`organism_action::gpu_fixed_signed_command_matches_mapper_contract` (intent brake −0.7:
throttle 1.0 with the flag, 0.6 without; flag without the exclusive net is refused).

Same contract, seeds, population, generations and champion seeds as the scale battery
(deadline 0.7, worst, envelope clamp 30, shaping, 32 × 3, 120 generations). Arms `fixed` and
`evolving`, seeds 11 then 7; the four `scale-*` runs above are the controls.

Predictions, in order of confidence:
1. Seed 7 (brake head negative on 99.5 % of ticks): at least one replica reaches a near-stop
   dwell > 0 inside the area within 120 generations in at least one arm (the scale runs: 0
   in 240 generations).
2. At least one of the four runs records a strict success (any start); seed 11 fixed, which
   already touches 0.25 m/s, is the likeliest.
3. Coevolution: in an evolving arm that records strict successes, the median brake capacity
   at generation 119 is not below 4000 Nm (drift stops being downward once the brake is
   pressed). Reported as observed either way.

Refuted if no run records a near-stop dwell on seed 7 and no run records a strict success.
Then the dead zone was not the bottleneck and the next lever is the contract's hold
(a v_stop/hold curriculum stage, e.g. 1.0 m/s / 32 ticks, strict success unchanged) or the
selection rule (worst-of-3 froze the seed-11 elite for 19 generations at 472).
Wall ≈ 4 × 9.5 min; disk ≈ 4 × 3 replays × 90 MB (45 GB free on `/media/llm-data`).

### Results of the signed-brake battery (filled 2026-09-24 12:16; runs 12:04–12:15, `run_battery_signed_brake.sh`, binary `gpu_evolve-signed-brake` sha256 9b1e3430…, commit 12f012c built from its tree pre-commit)

**All three predictions refuted.** No near-stop dwell on any seed or arm (240 generations of
seed 7, 240 of seed 11: 0 replicas under 0.6 m/s inside the area), no strict success, and the
evolving arms' brake capacity moved both ways (seed 11 up to 4548 Nm, seed 7 down to 3604 Nm)
on a flat score: a random walk, not selection.

| arm | seed | strict | median entry speed gen 0 → 119 (m/s) | slowest inside, gen 119 | near-stop dwell | elite selection score (unchanged since gen) | median judged ticks | s per generation |
|---|---|---|---|---|---|---|---|---|
| signed fixed | 11 | 0 | 11.7 → 12.4 | 4.66 (0.68 once, gen 100) | 0 | 259.9 (gen 104; 259.6 since gen 32) | 350 | 1.3 |
| signed evolving | 11 | 0 | 11.7 → 12.6 | 8.32 | 0 | 261 basin | 350 | 1.3 |
| signed fixed | 7 | 0 | 13.4 → 13.1 | 10.28 | 0 | 261 basin | ≈ 350 | 1.3 |
| signed evolving | 7 | 0 | 13.4 → 13.5 | 11.03 | 0 | 261 basin | ≈ 350 | 1.3 |

Controls (`scale-fixed-s11`): entry 9.7 → 6.3 m/s, slowest inside 0.25 m/s, near-stop dwell 62.

**Mechanism.** The move champions carry a brake logit of −0.4 … −0.7; with the sign released
that becomes +0.4 … +0.7 of drive on every tick, so the whole population enters the area at
12–13 m/s and overshoots at tick ≈ 350 (generations take 1.3 s instead of 4.7 s). Above 3 m/s
inside the area the shaping term pays nothing (`300·(1 − min_speed_inside/v_min)` is clamped at
0 for min_speed ≥ v_min = 3), the envelope pays nothing (nobody under `v_allow`, conform 0), and
every overshoot scores 500·progress − 200 + ε ≈ 260. The landscape is flat, so the brake row —
now visible to selection — has nothing to climb: the elite's three starts all score 259.6–259.9
and it is unchanged from generation 32. Removing the dead zone was necessary for the brake to
be selectable at all, but it is not sufficient: nothing pays the first metre per second of
slowing down while the being is still fast.

Two consequences, both pre-registered below as separate levers:
1. the shaping term needs a reference speed above `v_min` so that slowing from 12 to 8 m/s
   inside the area already pays (`--school-stop-shaping-ref-mps`, next code change);
2. the selection rule: worst-of-3 froze the control elite at 472 while its best start scored
   720; under the deadline the crawler exploit of the mean rule is closed, so `mean` can be
   tested (`run_battery_select_mean.sh`, launched 12:15 with the same binary, arms clipped
   vs signed, seeds 11 and 7, fixed body). Prediction: `mean-clipped-s11` records the first
   strict success of the 24/09 contract (its elite already touches 0.25 m/s); the signed arms
   stay in the 261 basin. Refuted if no run records a strict success.

### The best stopper, tick by tick (2026-09-24 12:17; `scale-fixed-s11`, generation 119, genome 0, start 1, score 720)

Decoded from `gen0119.replay` (frames every 8 ticks). Fast brake logit −0.5 … −0.8 on every tick
(clipped, as in every crawler); the applied brake of 0.05–0.10 comes from the slow brain's
inhibition path (`slow_inhibition · 0.8`). Throttle is 0 from tick 512. Speed 3.8 m/s at
tick 512 → 1.75 (616) → 0.77 (704) → 0.44 (896, **deadline: closed as failure 16**) → 0.24
(936) → 0.05 (1024) → 0.14 creep (1104–1240), position frozen at 27.9 m along (goal 30 m,
inside the 4 m area). From tick 936 the speed never exceeds 0.24 m/s, so the 128-tick hold
would have closed as **strict success at tick ≈ 1064**, inside the 1280-tick horizon, had the
deadline been the horizon itself. The same genome's other two starts reach 0.77 and 0.81 m/s
by the deadline (scores 498 and 472).

So for the seed-11 lineage the binding constraint at generation 119 is the 0.7 deadline, not
the brake: the being brakes (weakly, through the slow path) and stops, later than the scripted
witness (784) but before the horizon. The deadline was set to make "never arriving" a failure;
that purpose is served by any deadline ≤ the horizon, including 1.0.

### Next battery (pre-registered 12:17): full-horizon deadline, `run_battery_deadline_full.sh`

`--school-deadline-frac 1.0` (an attempt still open at tick 1280 closes as failure 16, so the
crawler exploit stays closed), arms `--school-select worst` vs `mean`, seeds 11 and 7, clipped
brake (default mapper), fixed body, otherwise the scale contract (shaping, envelope clamp 30,
32 × 3, 120 generations). Same binary (`gpu_evolve-signed-brake`, flag off = bit-identical).
Chained after the selection-rule battery (`chain_after_mean.sh`).

Predictions: (1) seed 11 records strict successes in at least one arm, the first before
generation 120; (2) `mean` reaches it no later than `worst` on seed 11; (3) seed 7 stays at 0
strict in both arms (its lineage never gets under 3 m/s inside the area: no slow-path brake
either). Refuted if seed 11 stays at 0 in both arms; then the remaining lever is the shaping
reference speed (pays slowing while still fast), which is also what seed 7 needs.
Wall ≈ 4 × 10 min (open attempts run to the horizon).

### Results of the selection-rule battery, seed 11 clipped arm (filled 2026-09-24 12:21; run 12:15–12:18, `mean-clipped-s11`, other arms still running)

**First genuine strict stops under the 24/09 contract, and the pre-registered primary metric
(one genome strict on ≥ 2 of 3 starts) is met.** Contract: deadline 0.7, `--school-select
mean`, shaping, envelope clamp 30, default (clipped) pedal mapper, fixed body, move-seeded
seed 11, 32 genomes × 3 starts, 120 generations (2.6 min: successes and overshoots both close
early).

| generation | strict successes (of 96) | first genome with 2 of 3 | elite mean / worst start score | score max |
|---|---|---|---|---|
| 0 | 0 | — | 253 / 238 | 262 |
| 64 | first strict | — | — | 1997 (gen 70) |
| 96 | — | genome 5 | — | — |
| 110 | — | elite (genome 0) 2 of 3 | 1429 / 236 | 2056 |
| 119 | 4 | elite 2 of 3 | 1229 / 236 | 2045 |
| total | 184 genome-starts | | | ceiling ≈ 2085 |

Elite at generation 119, re-evaluated in a fresh process (`/media/llm-data/scratch/eval-mean-clipped-s11-train`, bit-identical scores):

| start | outcome | score | v_max (m/s) | entry speed | slowest inside | hold | closes at tick |
|---|---|---|---|---|---|---|---|
| (99, 156) | **strict success** | 2007 | 5.61 | 1.09 | 0.003 | 128 | 871 |
| (99, 348) | overshoot at 12 m/s | 236 | 15.0 | 12.1 | 12.0 | 0 | 388 |
| (291, 156) | **strict success** | 2045 | 5.44 | 0.94 | 0.008 | 128 | 888 |

Pedal trace of the (99, 156) success (`gen0119.replay`): accelerates to 5.2 m/s by tick 256
(v_min satisfied), throttle 0 from tick 384, brake 0.11 → 0.37 at ticks 384–480 with the
**fast brake row positive** (intent 0.32–0.33 at ticks 464–480: this lineage left the dead
zone on its own), then 0.07–0.19 brake to a full stop at 29.0 m along (goal 30 m, area 4 m)
at tick 752, holding under 0.16 m/s until the judge closes at 871. Both successes close 8–25
ticks before the 896 deadline.

Why `mean` and not `worst`: the control (`scale-fixed-s11`, worst-of-3) reached 0.25 m/s on
its best start but its elite was frozen at its worst start (472) for 19 generations, while
under `mean` a child that improves one start is promoted; the crawler exploit that made `mean`
dangerous on 09-22/23 is closed by the deadline (an open attempt is −200 like an overshoot).
The same brain floors it from (99, 348) (12 m/s entry, overshoot): the stop is not driven by
the target channel alone but by the sensory context of each start.

**Held-out folds: 0 of 3 on validation and 0 of 3 on test** (`--school-start-fold validation`
/ `test`, same champion, same contract): validation (156, 99) and (156, 291) open at the
deadline, (348, 99) overshoots; test (93, 228) overshoots, (228, 93) and (228, 285) open at
the deadline. The policy is start-specific. That is what the fixed three training starts
select for (`start_assignment: per_slot_rotated_by_hash` with pool = K is a fixed assignment,
docs/GAP-SWEEP-20260924.md), and the pre-registered next lever for it is the start pool
(`--school-start-pool N`, real rotation), below.

### Next battery (pre-registered 12:22): start pool, `run_battery_start_pool.sh`

The winning contract (deadline 0.7, `mean`, shaping, envelope clamp 30, clipped mapper, fixed
body) with `--school-start-pool 24`: the train fold has 80 stop approaches; each generation
judges a rotating batch of 3 of the first 24, so every start is seen 30 times in 240
generations. Seeds 11 and 7; controls `mean-clipped-s11` / `-s7` (pool 3 = fixed starts).
Chained after the full-horizon battery (`chain_after_full.sh`). Held-out evaluation of each
final champion on the validation and test folds afterwards, exactly as above.

Predictions: (1) the seed-11 pool champion is strict on ≥ 1 of 3 validation starts (the
fixed-start champion: 0 of 3); (2) evaluated on the 24-start train pool it is strict on ≥ 8;
(3) seed 7 stays at 0 strict. Refuted if (1) fails on both seeds: then rotation does not buy
generalization and the next question is the input (which channels the stop is read from),
not the curriculum. Wall ≈ 2 × 5–20 min (open attempts run to the 896 deadline).

### Results of the selection-rule battery, remaining arms (filled 2026-09-24 12:30; runs 12:18–12:26)

| arm | seed | strict genome-starts (120 gens) | first strict gen | first genome 2 of 3 | slowest inside gen 119 | score max |
|---|---|---|---|---|---|---|
| mean, clipped | 11 | 184 | 64 | gen 96 | 0.00 | 2099 |
| mean, signed | 11 | **311** | **37** | never (max 1 of 3) | 0.00 | 2132 |
| mean, clipped | 7 | 0 | — | — | 0.74 | 573 |
| mean, signed | 7 | 0 | — | — | 10.69 | 262 |

**Prediction partly refuted.** The clipped seed-11 arm behaved as predicted, but the signed
arm on seed 11 did *not* stay in the 261 basin: under `mean` it finds strict stops earlier
(gen 37) and more often (311 genome-starts) than the clipped arm, though never on two starts
of the same genome. So the flat landscape of the signed-brake battery was a property of
worst-of-3 selection on that population, not of the signed mapper itself: with `mean`, a child
that stops on one start is promoted even while its other two starts overshoot at 12 m/s.
Seed 7 confirms the lineage effect: with the clipped mapper it descends (0.74 m/s inside at
gen 119, from 8.98) but has no strict stop in 120 generations; with the signed mapper it
stays fast (10.7 m/s inside). The two seeds are different populations, not different noise.

### Results of the full-horizon deadline battery (filled 2026-09-24 12:41; runs 12:26–12:37, `run_battery_deadline_full.sh`, binary `gpu_evolve-signed-brake`, flag off)

Statistics are over the accepted (strict) genome-starts of each run, all 120 generations;
"last 10" is the mean over generations 110–119.

| arm | seed | strict genome-starts | first strict gen | first genome 2 of 3 | strict / gen (last 10) | genomes ≥ 2 of 3 (last 10) | peak speed of accepted stops, median | speed entering the braking zone, median | outcome tick, median |
|---|---|---|---|---|---|---|---|---|---|
| worst, deadline 1.0 | 11 | 165 | 16 | gen 91 | 4.0 | 1.7 | 4.27 m/s | 1.83 m/s | 1158 |
| mean, deadline 1.0 | 11 | **1572** | 15 | **gen 35** | 20.8 | 6.0 | 3.63 m/s | 1.73 m/s | 1146 |
| worst, deadline 1.0 | 7 | 0 | — | — | 0 | 0 | — | — | — |
| mean, deadline 1.0 | 7 | 0 | — | — | 0 | 0 | — | — | — |
| mean, deadline 0.7 (control, `mean-clipped-s11`) | 11 | 184 | 64 | gen 96 | 5.0 | 1.0 | 5.61 m/s | 3.22 m/s | 880 |

All three predictions held: (1) seed 11 records strict stops in both arms, the first at
generation 15–16; (2) `mean` reaches a 2-of-3 genome at generation 35, `worst` at 91; (3) seed
7 stays at 0 in both arms (its slowest inside speed reaches 0.16 m/s under `mean` by
generation 100, and the slow-brake row still never closes a hold). No genome reached 3 of 3
in any arm. Wall was 2.6 min per run, not the predicted 10: no attempt stays open to the
horizon, every one closes by overshoot, contact or success before tick 1280, so deadline 1.0
costs nothing in wall time.

**The count went up and the driving got worse.** Deadline 1.0 relaxes the time pressure and
the population drifts to slower driving: accepted stops peak at 3.6 m/s (mean arm) and enter
the braking zone (`r_zone = area + v_min²/(2 a_ref)` = 6.25 m with `a_ref` 2.0) at 1.7 m/s,
against 5.6 and 3.2 m/s under deadline 0.7. Their stops also close ~270 ticks later. So the
full horizon buys count, not the skill; the deadline-0.7 lineage is the better stopper.

**What the contract actually accepts.** In every arm the accepted stops enter the 4 m area
at ≈ 1 m/s (median 0.7–1.1 across arms) and hold at the area's edge (best remaining 3.2–3.4 m
of a 3 m goal radius). The `v_min` gate (phase 1 at `school.wgsl:305`) is satisfied on the
drive-up, far from the goal, so the behaviour the judge rewards is "drive, coast down early,
creep in, hold", not "arrive at speed and brake". The mean-clipped-s11 champion does press
the brake (its slot-0 speed trace, § The best stopper), but the contract does not require it.
Measured on the recorded fields: of the mean-clipped-s11 accepted stops, 105 of 184 (57%)
entered the braking zone at ≥ `v_min`; of full-mean-s11's, 99 of 1572 (6%).

**Next contract lever (to be pre-registered when built): a zone-entry gate.** Flag
`--school-stop-zone-entry-gate` (default off, judge flag bit 3): the speed at first entry
into the braking zone must be ≥ `v_min`, otherwise the attempt closes as a new failure 17.
It converts the `v_min` requirement from "once, anywhere" into "still at approach speed
6.25 m out", which is the braking skill itself. Its ceiling on the existing data is the 57%
above; on the full-horizon lineage 6%. Needs a build (queued behind the GPU battery) with the
shaping-reference change already in the tree, then the GPU fixtures.

**Amendment to the start-pool pre-registration (12:40, before any pool result was read).**
The chained launcher started `pool24-mean-s11` at 12:37:18 while I relaunched it believing
it dead (its pid file named the parent, not the live shell); both writing the same directory,
both were killed by pid at 12:38, the directory wiped and the battery relaunched once (log
line "killed both colliding pool24-mean-s11 launches"). The relaunch adds a deadline-1.0 arm
per seed (`pool24-full-mean-s{11,7}`, order s11 0.7, s11 1.0, s7 1.0, s7 0.7). Added
prediction (4): the deadline-1.0 pool champions accept more stops but drive slower — median
peak speed of accepted stops below 4.5 m/s, against above 5 m/s for the deadline-0.7 pool
arm. Held-out evaluation of all four pool champions plus `full-mean-s11` and
`mean-clipped-s11` is chained (`chain_eval_after_pool.sh`, validation/test folds with 3
starts, train fold with the full 24-start pool).

### Results of the start-pool battery, first arm (filled 2026-09-24 12:46; run 12:38–12:43, `pool24-mean-s11`, deadline 0.7)

| arm | seed | gens | strict genome-starts | first strict | starts seen | score max (typical) | slowest inside, gen 239 |
|---|---|---|---|---|---|---|---|
| pool 24, mean, deadline 0.7 | 11 | 240 | **0** | — | 24 of 24 | 260–262 (one spike 421 at gen 220) | 2.70 m/s |

**Predictions (1) and (2) refuted for seed 11.** Rotating the batch of 3 through 24 starts from
the move champion never leaves the 261 basin (`500·progress − 200`) in 240 generations,
while the same contract on 3 fixed starts found its first strict stop at generation 64 and a
2-of-3 genome at 96. Mechanism: on fixed starts the population descended gradually (slowest
speed inside fell 8.6 → 0.7 → 0.0 m/s over 60 generations, the same three geometries each
time); under rotation each generation judges a different geometry, so a child slightly slower
on this batch is not slightly slower on the next, and the elite is re-decided every
generation on incomparable scores. The gradient that carried the fixed-start lineage does
not accumulate. Rotation from a population that cannot stop yet is a curriculum error, not a
generalization test. Seed 7 arms are still queued and are expected at 0 as well.

### Next battery (pre-registered 12:46): from the stopper, `run_battery_from_stopper.sh`

Every arm resumes from the `mean-clipped-s11` generation-119 champion (peak 5.6 m/s, braking
zone entered at 3.2 m/s, strict on 2 of 3 fixed starts) with its contract (mean, deadline 0.7,
shaping, envelope clamp 30, clipped mapper). Seeds 11 and 7 are mutation-RNG replicates of the
one lineage. Chained after the held-out evaluations (`chain_eval_after_pool.sh`).

1. **Body coevolution, fixed 3 starts, 120 generations**: `body-fixed-s{11,7}` (control) vs
   `body-evolving-s{11,7}` (`--evolving-body`, elites keep their body, children mutate each
   body gene by ±0.1 uniform). Question: once the brain presses the brake, does selection move
   brake capacity? Predictions: (B1) the population median brake torque of the evolving arm
   moves in the same direction on both seeds and by ≥ 5% over 120 generations (a random walk
   of ±0.1 per child per generation on the median has no preferred sign, so agreement of sign
   across two seeds and a ≥ 5% drift is the selection signature; the 11:55 scale × body battery,
   whose brains never braked, showed drift with no such structure); (B2) strict genome-starts
   per generation over the last 10 generations: evolving ≥ fixed on both seeds. Refuted if the
   drift signs disagree between seeds or stay under 5%: then brake capacity is not the binding
   constraint of this stopper, and the body question moves to engine/mass.
2. **Rotation from the stopper, 24-start pool, 240 generations**: `rot-stopper-s{11,7}`.
   Predictions: (R1) strict > 0 within 240 generations on both seeds; (R2) seed 11 records
   ≥ 1 strict on at least 6 distinct starts; (R3) the last 30 generations average > 0 strict
   per generation. Refuted if R1 fails on both seeds: the stop is then tied to one start's
   visual cue and rotation must be introduced one start at a time.

Ceilings: 96 genome-starts per generation; a genome is judged on 3 starts per generation, so
"strict on 6 distinct starts" needs at least 2 generations of the elite. Wall ≈ 4 × 3 min +
2 × 6 min. Time stamps were read from `date` before writing.

**Correction to prediction B1 (12:47, before the from-stopper battery started; it is still
queued behind the held-out evaluations).** I re-read the 11:55 scale × body runs with
`analyze_body.py` instead of from memory: their evolving arms did *not* random-walk. Both
seeds shed capacity — population median brake torque 4000 → 3863 N·m (−3.4%, seed 11) and
→ 3628 (−9.3%, seed 7), engine 620 → 562 / 593 (−9.3% / −4.4%), mass −1.2% / −1.0% — with
brains that never braked. That is the cost term of `capacity_mass_tradeoff_v1` selecting
against unused capacity, so "same sign on both seeds and ≥ 5%" would be met by shedding alone.
B1 restated: in the evolving arms resumed from the stopper, the median brake torque drift over
120 generations is **≥ 0% on both seeds** (the shedding stops once the brake is used), and the
selection-for-brake claim needs **≥ +5% on both seeds**. Refuted if either seed sheds brake by
more than 3%: then the stopper's brake use is too light to pay for capacity.

### Results of the start-pool battery, deadline-1.0 arm, seed 11 (filled 2026-09-24 12:51; run 12:43–12:48, `pool24-full-mean-s11`, 240 generations, 5.5 min)

| metric | pool 24, mean, deadline 1.0, seed 11 | pool 24, deadline 0.7 (above) | fixed 3 starts, deadline 1.0 (`full-mean-s11`, 120 gens) |
|---|---|---|---|
| strict genome-starts | **1000** | 0 | 1572 |
| first strict / first 2-of-3 / first **3-of-3** | gen 28 / 34 / **106** (genome 9) | — | 15 / 35 / never |
| strict per generation, last 30 | 6.8 (max 24) | 0 | 20.8 (last 10) |
| distinct starts with ≥ 1 strict | **23 of 24** | 0 | 3 of 3 |
| peak speed of accepted stops, median | **6.12 m/s** | — | 3.63 m/s |
| speed entering the braking zone, median | **2.65 m/s** (q3 3.44) | — | 1.73 m/s |

**Prediction (4) refuted, and the mechanism claimed at 12:46 was incomplete.** Rotation does
not kill the gradient by itself: with deadline 1.0 the same rotating pool finds strict stops
from generation 28 and the first genome strict on all three starts of its batch at 106
(starts (228,477), (348,483), (483,348): peak 6.9 / 6.5 / 4.0 m/s, one of them entering the
area at 4.27 m/s and stopping 1.6 m from the centre). The rotating deadline-1.0 lineage drives
*faster* than the fixed-start deadline-1.0 lineage (6.1 vs 3.6 m/s peak) and enters the zone
faster (2.65 vs 1.73 m/s). What failed at deadline 0.7 was the combination: a generalist's
first slow approaches on unfamiliar geometry run past tick 896 and close as failure 16 (−200,
the same as an overshoot), so slowing down is not rewarded relative to flooring it; at deadline
1.0 the slow approach that reaches the area collects progress + shaping and the descent starts.
Once the population must stop from 24 geometries it cannot memorize a start's timing, which is
the plausible reason its stops are faster than the fixed-start creep-ins; this last step is a
hypothesis, not measured.

**Replay evidence** (`gen0239.replay`, genome 4, slot 13, start (420,861), strict, score
window 904 ticks): throttle 1.0 to tick ~190 (5.3 m/s), then throttle 0 with the brake pedal
applied 0.2–0.4 from tick 336 on, brake intent +0.5…+0.8 throughout (out of the dead zone),
speed 4.8 → 2.5 → 0.8 → 0.05 by tick 816, brake held 0.3–0.4 while stopped. Slot 14 (start
(804,477)) is the same shape from a 3.2 m/s peak. This lineage brakes; it is not a coaster.
Instability remains: the elite of generation 239 is strict on 2 of its 3 starts, generation 120
(the batch that happens to be the three original fixed starts) has 0 strict in the whole
population (best remaining 0.5–0.8 m with codes 5/16), and strict per generation over the last
30 is 6.8 of 96. Held-out evaluation of the gen-239 champion is in `chain_eval_after_pool.sh`
and of the gen-106 champion in `chain_eval2.sh` (after the from-stopper battery). Revised
prediction for both: strict on ≥ 1 of 3 validation starts (the fixed-start champion had 0).
Seed 7 arms still queued; prediction (3) unchanged (0 strict).

### Amendment to the from-stopper pre-registration (12:51, before any arm of it ran): from the generalist, `run_battery_from_generalist.sh`

The queued `run_battery_from_stopper.sh` (still waiting on the evaluations) was killed and
replaced: the natural parent is now the generalist, not the fixed-start creep-in champion.
Every arm resumes from `pool24-full-mean-s11/evolve_best_gen0106.bin` under the contract that
produced it (pool 24, deadline 1.0, shaping, envelope clamp 30, clipped mapper), 240
generations, seeds 11 and 7 as mutation-RNG replicates.

1. `body-fixed-s{11,7}` (mean, fixed body; also the continuation control) vs
   `body-evolving-s{11,7}` (`--evolving-body`). Predictions B1 (brake torque median drift
   ≥ 0% on both seeds; ≥ +5% for the selection-for-brake claim; refuted below −3%) and B2
   (last-10 strict per generation evolving ≥ fixed on both seeds) as corrected at 12:47.
2. `gen-worst-s{11,7}` (`--school-select worst`): can worst-of-3 make the generalist stable
   now that a 3-of-3 genome exists? Predictions: (S1) over the last 30 generations, genomes
   strict on all 3 starts of their batch per generation: worst ≥ 3 × mean (`body-fixed`) on
   both seeds; (S2) last-30 strict per generation: worst ≥ mean on both seeds. Refuted if S1
   fails on both seeds: worst-of-3 then freezes the elite here as it did on fixed starts.

Ceiling: 32 genomes per generation, so S1 saturates at 32. Wall ≈ 6 × 5.5 min. The marker at
the end stays "from-stopper battery done" so `chain_eval2.sh` (gen-106 held-out) fires after it.

### Results of the start-pool battery, seed 7 arms (filled 2026-09-24 13:00; runs 12:48–12:59)

| arm | seed | strict genome-starts (240 gens) | first strict | first 2-of-3 | last-30 strict/gen | distinct starts with ≥ 1 strict |
|---|---|---|---|---|---|---|
| pool 24, mean, deadline 1.0 | 7 | 19 | gen 145 | gen 193 | 0.37 | 8 (peak 4.5 m/s, zone entry 1.8 m/s, medians) |
| pool 24, mean, deadline 0.7 | 7 | 16 | gen 105 | never | 0.20 | 7 (peak 6.3 m/s, zone entry 3.3 m/s, medians) |

**Prediction (3) refuted: seed 7 records its first strict stops ever**, in both deadline arms,
after 0 strict in all six fixed-start arms of the day (scale, signed, mean, full). The
24-start pool contains geometries this lineage can stop on; the three fixed starts did not.
Late (gen 105/145), rare (0.3–0.4 per generation over the last 30) and never a 2-of-3 under
deadline 0.7, so the lineage effect stands, but "seed 7 cannot stop" was a statement about
three starts, not about the lineage. As on seed 11, the deadline-0.7 stops are the faster ones.

### Held-out evaluation of the champions (filled 2026-09-24 13:01; `chain_eval_after_pool.sh`, 13:00:13–13:00:38, 4 copies of each champion, 1 generation, same contract as its run; evals in `/media/llm-data/scratch/eval-pool-20260924/`)

Strict stops of copy 0 (the other three copies are identical: 4 × the same counts everywhere).

| champion (final generation) | validation (3 starts) | test (3 starts) | train pool (24 starts) | failure codes on the train pool |
|---|---|---|---|---|
| `pool24-full-mean-s11` gen 239 (rotation, deadline 1.0) | **1 / 3** | **1 / 3** | **8 / 24** | 13 corridor, 3 open at deadline |
| `pool24-full-mean-s7` gen 239 | 0 / 3 | 0 / 3 | 0 / 24 | 19 overshoot, 4 contact |
| `pool24-mean-s11` gen 239 (rotation, deadline 0.7) | 0 / 3 | 0 / 3 | 0 / 24 | 24 overshoot |
| `pool24-mean-s7` gen 239 | 0 / 3 | 0 / 3 | 0 / 24 | 10 corridor, 10 overshoot, 3 open |
| `full-mean-s11` gen 119 (fixed starts, deadline 1.0) | 0 / 3 | 0 / 3 | 2 / 24 | 15 open at deadline, 7 overshoot |
| `mean-clipped-s11` gen 119 (fixed starts, deadline 0.7) | 0 / 3 | 0 / 3 | 2 / 24 | 10 open, 10 overshoot, 1 corridor, 1 contact |

**First held-out generalization.** The revised prediction for the rotating deadline-1.0
champion (≥ 1 of 3 validation starts) holds on both held-out folds, and the original
start-pool prediction (2) (≥ 8 of 24 on the train pool) holds exactly. Both fixed-start
champions stop only on 2 of the 24 train starts — their own — and on nothing held out, as
measured at 12:21. The generalist's dominant failure on the pool is now the corridor (13 of
24: it leaves the lane or backs past the start), not overshoot; the stop is learned, the lane
keeping under braking is the next skill. Seed 7's rotating champion, whose run recorded 19
strict, holds nothing at its final generation (its stops were not in the elite).

### Results of the from-generalist battery, body arms (filled 2026-09-24 13:23; runs 13:00–13:22, `run_battery_from_generalist.sh`, 240 generations each, all from the gen-106 generalist, pool 24, deadline 1.0, mean)

Population medians at generation 239 (96 bodies); strict counts are genome-starts per
generation averaged over generations 210–239; the body model is
`engine = 1 + 0.25·g0`, `brakes = 1 + 0.25·g1`, mass +15% per engine unit (`src/gpu/body.rs:24-35`),
so engine and brake torque can move at most ±25%.

| arm | seed | brake torque drift | engine torque drift | mass | elite genes (g0, g1) | strict / gen (last 30) | 3-of-3 genomes / gen (last 30) | peak speed of accepted stops, median |
|---|---|---|---|---|---|---|---|---|
| fixed body | 11 | 0 | 0 | 0 | (0, 0) | 6.2 | 0 | 6.4 m/s |
| `--evolving-body` | 11 | **−7.3%** | **+21.4%** | +2.3% | (0.99, −0.32) | **14.7** | 0.07 | 7.06 m/s |
| fixed body | 7 | 0 | 0 | 0 | (0, 0) | 10.4 | 0 | 7.1 m/s |
| `--evolving-body` | 7 | **+6.8%** | −1.6% | 0.0% | (−0.12, +0.24) | 10.2 | 0 | 6.37 m/s |

**B1 refuted, B2 refuted.** Brake capacity is not what selection buys once the brain brakes:
seed 11 sheds brake (−7.3%) and drives the engine gene to its ceiling (g0 = 0.99 of a
maximum 1.0 — a metric at its limit, so +21% is the cap, not the preference), seed 7 raises
the brake (+6.8%) and leaves the engine alone. The body evolves and is selected on — the
seed-11 population is at (0.75…0.97, −0.39…−0.23) by generation 239, not a random walk around
0 — but the *direction* is seed-specific at n = 2, and the stop rate does not separate from
mutation-seed variance: the two fixed-body controls differ from each other by 1.7×
(6.2 vs 10.4), more than seed 7's evolving-vs-fixed gap and less than seed 11's. What can be
claimed: body coevolution is functional under the stop contract (genes express, are inherited
by elites, and move to the bound on one seed); what cannot be claimed yet: that it helps the
stop, or that it selects brake capacity. A third paired seed is chained (`run_body_s3.sh`,
after `chain_eval2.sh`), per the ≥ 3 paired seeds rule; prediction: strict per generation
evolving vs fixed on seed 3 falls between the two observed ratios (0.98 and 2.4), and the
engine gene ends above +0.5 if and only if the strict ratio exceeds 1.5.

### Results of the from-generalist battery, worst-of-3 arms, and the gen-106 held-out evaluation (filled 2026-09-24 13:49; runs 13:22–13:33 `gen-worst-s{11,7}`, evals 13:33 `chain_eval2.sh`)

| arm | seed | strict total (240 gens) | strict / gen (last 30) | 3-of-3 genomes / gen (last 30) | gen 239 fail codes (96 slots) | zone-entry speed, median |
|---|---|---|---|---|---|---|
| `body-fixed` (mean, control) | 11 | 1558 | 6.2 | 0 | — | — |
| `gen-worst` (`--school-select worst`) | 11 | **392** | **0.2** | 0 | overshoot 83, no_approach_speed 3, contact 2 | 9.3 m/s |
| `body-fixed` (mean, control) | 7 | 2168 | 10.4 | 0 | — | — |
| `gen-worst` (`--school-select worst`) | 7 | **98** | **0.0** | 0 | overshoot 90, contact 1, corridor 1 | 10.5 m/s |

**S1 refuted on both seeds, S2 refuted on both seeds.** Worst-of-3 does not stabilise the
generalist; it destroys it. Mechanism, from the fail codes and the score ceiling: under
rotation the worst of a genome's three starts is a failure for almost every genome, so the
worst-of-3 score reduces to the failed-run score (`500·progress + shaping`, maximum 530 and
474 at generation 239 — the strict score of 2000+ never enters the ranking once no genome is
3-of-3). Selection then rewards fast approach and overshoot (entry speed 9–10 m/s at
generation 239 versus 3–4 m/s in the mean arms), and the stop the parent already had is bred
out within the first 100 generations. Worst-of-3 is retired for rotated starts; it was already
the wrong rule for fixed starts (12:26 section).

Gen-106 generalist held out (4 copies of the genome, 1 generation, same contract; folds of 3
starts, train pool of 24; `/media/llm-data/scratch/eval-pool-20260924/pool24-full-mean-s11-gen106-*`):

| fold | strict stops | fail codes |
|---|---|---|
| validation (3 starts) | 0 / 12 | open_at_deadline 12 |
| test (3 starts) | **8 / 12** (2 of 3 starts) | no_approach_speed 4 |
| train (24 starts) | 24 / 96 (6 of 24 starts) | open_at_deadline 60, no_approach_speed 8, overshoot 4 |

Compared with the gen-239 champion of the same run (13:01 section: 1/3 validation, 1/3 test,
8/24 train), the gen-106 genome is worse on the train pool and on validation and better on
test: two snapshots of one lineage disagree by fold, which is what n = 3 starts per fold
looks like. No claim about generalisation beyond the pool is made from either.

### Results of the body coevolution arms, third seed (filled 2026-09-24 13:49; runs 13:33–13:44, `run_body_s3.sh`, seed 3, same contract and parent)

| arm | seed | brake torque drift | engine torque drift | mass | elite genes (g0, g1) | strict total | strict / gen (last 30) |
|---|---|---|---|---|---|---|---|
| fixed body | 3 | 0 | 0 | 0 | (0, 0) | 1563 | 6.1 |
| `--evolving-body` | 3 | +0.1% | **−6.1%** | −1.2% | (−0.24, 0.00) | 1404 | **11.2** |

Prediction of 13:23 checked: the evolving-vs-fixed strict ratio on seed 3 is 1.84, inside the
predicted band (0.98…2.4) — held; the engine gene ended at −0.24 with a ratio above 1.5 —
the "engine above +0.5 iff ratio > 1.5" half is refuted. Over three paired seeds the evolving
arm has the higher last-30 strict rate on two of three (14.7 vs 6.2, 10.2 vs 10.4, 11.2 vs
6.1; mean 12.0 vs 7.6) and the lower 240-generation total on two of three (2893 vs 1558,
1731 vs 2168, 1404 vs 1563). Body direction by seed: engine +21% / brake −7% (s11),
brake +7% (s7), engine −6% (s3). Three seeds, three directions: the body is under selection
but no body trait is consistently selected for by the stop contract at this search scale.
B1 (brake ≥ 0% on all seeds) holds only marginally (+0.1% on s3) and never reaches the +5%
selection-for-brake bar; B2 (evolving ≥ fixed, last 10) holds on 2 of 3. Verdict stands:
coevolution functional, benefit not separable from seed variance. A fourth seed
(`run_body_s13.sh`, seed 13) is running on the idle GPU for the effect-size trend, not as
a decisive test.

### Lab panel restored (13:40–13:47, on the owner's order; code in `lab-ui` and `neuromod-lab`, no Universe change)

The macro/micro observation panel is the home again (`http://192.168.1.54:8130/`, API bound
on the LAN without a login; the observatory prototype moved to `/observatory`). Today's 41
runs of this battery are in `lab.db` as battery `school-stop-20260924` (reconstructed
manifests, `lab_db.py ingest --battery`), with a run picker under the selected battery, the
population grid coloured by judge score, and a "pontuação mediana (escola)" curve metric.
Micro traces exist for three re-evaluated champions recorded with `--driving-diagnostics`
(being 0, 1 generation, `chain_diag.sh`): `diag-pool24-full-mean-gen106-s11` (train pool,
24/96 strict), `diag-pool24-full-mean-gen239-s11` (32/96, corridor 52),
`diag-mean-clipped-gen119-s11` (fixed starts, 8/12, overshoot 4). Path verified in the
browser: battery → run → being 0 → micro (retina, wave, brain 3D, stages, track).

### Lab panel rebuilt around the school (14:00–14:40, owner's order; `lab-ui` 145ac4c, `neuromod-lab` cf347a6 + 17013b1, no Universe change)

The macro is now only the school pipeline: Baterias → Corridas → Evolução / Grade de população /
Campeões / Comparar gerações / Traços / Anotações, every panel fed by `GET /api/school/*` for the
selected run (no mocks, no default battery/run, race-era panels deleted). Campeões lists the
genome selection actually picked in each generation with its per-start outcome (status, failure,
score, entry and peak speed, outcome tick); expanding a generation gives one button per start:
"gravar traço" queues a diagnostic re-evaluation (`POST /api/school/trace-jobs`, file spool under
`/media/llm-data/universe-runs/_school-trace-jobs/`, served by `scripts/school_trace_runner.py`,
which waits while any `gpu_evolve` is alive, runs `--resume --resume-mix 1.0 --beings 4 --gens 1
--driving-diagnostics --vision-being <start>` from the copied checkpoint, packages the trace and
registers it in `lab.db`), "abrir micro" opens it. Traces are indexed by the START the champion
drove, not by population slot, so the micro shows the recorded start's verdict strip (lesson,
start, outcome, score, entry/peak speed, outcome tick) and the sibling starts of the same genome.
Seventeen traces recorded this way (pool24-full-mean-s11 gens 0/30/60/90/106×2/120/150/180/210/239,
mean-clipped-s11 0/60/119, body-evolving-s11 0/120/239). Verified in the browser at 14:38:
select `pool24-full-mean-s11`, expand generation 106 (3/3, starts 6/7/8), open start 6 → micro
plays the recorded stop (entry 0.9 m/s, peak 6.9 m/s, stopped at tick 867). The earlier defect was
that the restored panel showed only a frozen end-of-run champion and slot 0 (the exact checkpoint
on its own start, which crawls), the compare panel crashed because the generation endpoint lacked
the elite summary, and annotations never loaded per run.

### Life battery: the same 30 000-tick production life for every stage of the lineage (16:00–, `school-life-20260924`; `lab-ui` 02bbcf6, `neuromod-lab` 0e26052, no Universe change)

Owner's order at 15:5x: the champion replay "shows the being standing still" because a
lesson is a 30 m episode of 1 280 ticks on a 960 m map; he wants to see a being crossing
the whole map, from the newborn that still has to learn to move up to the adult that
delivers, over many more ticks. Design: one production life (`organism-taxi-v1`,
passengers on, 30 000 ticks = 7.8 min at 64 Hz, 32 beings, `--driving-diagnostics
--vision-being 0 --vision-every 8`) for each stage of one lineage story, all from the
same seed family, plus a service curriculum from the stop generalist and the lives of its
graduates. Stages: `newborn` = random genomes (seed 11); `mover` = move-lesson champion
(`taxi-school-delivery-task7-evolve-move-seed7-k3-v2/evolve_best_gen0015.bin`, seed 7);
`stopper` = stop generalist (`pool24-full-mean-s11/evolve_best_gen0106.bin`, seed 11);
`reference` = scripted witness (`--taxi-reference-witness`, seed 7101, not evolved);
`curriculum` = gen106 through turn_near → turn_far → approach_passenger → pickup →
navigate_with_passenger → delivery (`--school-curriculum adaptive|uniform`, advance at
p̂ ≥ 0.25 for 2 gens, 300 gens, 4 096 ticks, pool 24 × 3 starts); `graduate` = the best
checkpoint of each curriculum living the same 30 000-tick life. Slot 0 of every checkpoint
run is the unmutated clone (`--resume --resume-mix 1.0`); the other 31 are mutants.

Pre-registered (written before the lives ran): no evolved stage delivers a passenger today
(the school only ever produced move and stop; pickup needs a full stop inside the
passenger zone that no lesson has taught); newborn median distance under 100 m; mover
furthest but most deaths (it never learned to stop, so it hits walls); stopper alive but
short; witness 2 deliveries as in the reference run; adaptive curriculum stalls on
turn_near (smoke p̂ 0.038 against a 0.25 gate), uniform cycles without mastering, so
both graduates ≈ stopper.

Measured (organism_result per slot, 32 beings, 30 000 ticks):

| stage | run | max distance | median | deliveries | alive | note |
|---|---|---|---|---|---|---|
| newborn | random-life-s11 | 455 m (slot 11) | 33 m | 0 | 30/32 | random genomes mostly idle |
| mover | move-life-s7 | 3 824 m (slot 7, alive) | 802 m | 0 | 8/32 | slot 0 (clone) 903 m, dead at 196 kJ collision |
| stopper | stopper-life-s11 | 364 m (slot 16) | 142 m | 0 | 32/32 | stops well, barely moves |
| reference | witness-life | 1 211 m | — | 2 | 1/1 | pickup tick 1 536, delivery 19 712 (L-route, shown in the micro) |
| curriculum adaptive | curriculum-adaptive-s11 | — | — | — | — | gens 14–20: turn_near p̂ 0.01–0.11, gate 0.25, no advance (prediction holding) |

UI (verified in Chrome on the LAN address at 16:12–16:16): macro tab "Vida no mundo"
lists the stages with distance / deliveries / survivors and a "▶ ver" per row; the micro
opens `/run/<run>/0/<slot>?view=champion&play=1` with a life strip (stage, distance,
deliveries, money, alive, collision energy), the full observed path on Track2D (faint
dashed, traversed part coloured by speed, "largada" ring, caption "percorrido X m de Y m ·
v · tique"), auto-framed on load. The witness trail is the 1 211 m L-route ending at the
station at tick ~21 000; the mover clone climbs the west edge for 890 m then dies.
Packaging the witness needed the checker to accept `body.motor_trace.source =
resident_reference_action_v1` (`neuromod-lab/src/lab_api/trace/check.py`, uncommitted with
the rest of the 21–23/09 trace-pipeline diff).

Follow-up queued (`run_life_best.sh`, waits for the "school-life battery done" marker): each
stage re-evaluated with the recorder on its BEST slot (most deliveries, then distance) as
`<run>-best`, with a determinism check (same slot, same distance/deliveries as the original
life) logged to battery.log, so the micro can show slot 7 of the mover crossing 3.8 km
instead of the clone. Pending: uniform curriculum, both graduates, their best-slot traces.
Honest status: as of 16:20 no evolved being has delivered a passenger; the only delivery on
the map is the scripted witness.

#### 16:30–17:20 update: curriculum advanced, then panicked; best-slot traces deterministic; stage 2 queued

- Prediction "adaptive stalls on turn_near" was WRONG: turn_near passed at gen 94 and
  turn_far at gen 198 (p̂ ≥ 0.25 for 2 gens, pool 24 × 3 starts). Entering
  approach_passenger at gen 199 `gpu_evolve` panicked (`main.rs:4803`,
  `organism_school::lesson_starts_fold`): passenger lessons have only 10 candidate
  approaches (5 stations × 2) and the pool of 24 is an error instead of a clamp. The
  uniform arm hit the same panic at gen 2. **Universe bug to fix in the next build
  window: clamp the pool to the candidate count (or say so at argument parsing).**
- Graduates' 30 000-tick lives (from the last checkpoint before the panic): adaptive
  gen198 = 318 m max, 141 m median, 32/32 alive, 0 deliveries; uniform gen1 = 531 m max,
  0 deliveries. Both stopper-like, as predicted for a curriculum that never reached pickup.
- Best-slot re-traces (`run_life_best.sh`): all five deterministic (same slot, same distance
  to 0.1 m: newborn 11 = 454.7 m, mover 7 = 3 823.5 m, stopper 16 = 363.9 m, graduate
  adaptive 23 = 318.4 m, graduate uniform 7 = 530.8 m). The micro of the mover's slot 7
  (3.8 km alive) is `/run/move-life-s7-best/0/7?view=champion&play=1`.
- Stage 2 (`run_life_stage2.sh`, 17:19): resumes the turn_far graduate (gen 198) on
  approach_passenger → pickup → navigate_with_passenger → delivery with pool 10 × 3 starts,
  300 gens; then its 30 000-tick life and best-slot trace. First gens: 0/96 on
  approach_passenger, failures contact/corridor, first pickup requests at gen 1.
- UI: Brain3D drew the 14 × 126 retina as a 42 × 42 square (scrambled, "static");
  now rows × cols from block.shape (`e02d465`). Micro did replay-sized work per animation
  frame (32 beings × 3 750 frames scanned ~35× per draw, 3 750 trail strokes, WebGL frame
  every rAF): static offscreen layer + bucketed trail + render-on-demand (`1680671`).

#### 18:20 update: stage 2 advanced to pickup

- approach_passenger passed at gen 238 (27/96 and 28/96 on gens 237–238, p̂ 0.28/0.29,
  two consecutive gens ≥ 0.25). It took 238 gens; the population hovered at 5–20/96 for
  ~100 gens with three single-gen peaks (25, 31, 27) that broke the streak. Dominant
  failures right before passing: corridor (leaving ±6 m near the target) and contact.
- Pre-registered prediction for pickup (before any pickup gen was read): failures migrate
  from corridor to `premature_or_moving_pickup` / `service_goal_not_stopped`; pickup does
  NOT pass within the ~60 remaining gens (p̂ stays < 0.25), because the population has only
  just fixed "arrive" and not "stop inside the radius".

#### 18:40 update: pickup never passed (0/96 × 61 gens); mechanism found; stage-2 graduate life

- pickup: 0/96 on every gen 239–299. The prediction "failures migrate to the stop-contract
  codes" was WRONG: the log kept showing contact/corridor, and the counts grew (contact
  15 → 45, corridor 23 → 50) — the population drifted back toward crashing.
- Hidden bucket: `codes=` only maps codes 1..=15 (`main.rs` by_code loop), so code 16
  `open_at_deadline` never appears. Recomputing failed − Σcodes per gen: 58, 84, 90 beings
  at gens 239–241 reached the passenger, stopped and waited to the deadline; by gen 279 that
  bucket was 0 and by gen 299 it was 1. The population UNLEARNED arriving.
- Why (score, `school.wgsl` service branch): score = progress(0..100) + 1000·delivery_delta
  − 200·failed − integrity − alive. A being that arrives, stops and waits (code 16) scores
  ≈ 95 − 200 = −105; a being that crashes at 95 % scores the same −105. Nothing rewards
  PHASE = 1 (stop contract met) in lesson 6, so selection is blind between "stopped at the
  passenger" and "hit a wall next to it" → drift. `score_max = −100.0` on every pickup gen.
- Second cause: the pickup request channel (`intent[6] > 0.5`) is nearly extinct. In
  approach_passenger every request is an instant fail (code 10 moving, code 15 outside
  4 m), so gens 1–30 had 5–20 requests/gen and gens 200+ had 0–3. The lesson that
  precedes pickup selects AGAINST the channel pickup needs. Success needs a rare fire
  (channel ~0.001 %/tick) coinciding with a rare state (stopped ≥ 64 ticks inside 3 m).
- Stage-2 graduate life (gen 299, 30 000 ticks, seed 11): 0 deliveries, 777 m max,
  548 m median (4× the stopper's 142 m), 31/32 dead at ~300 s with energy_j = 0
  (starvation, not collision: integrity ≥ 0.1 for all), 31 service refusals (it does
  request service in production, moving or off-radius). More driving, more dying,
  no service.
- No evolved being delivers. The witness (scripted) remains the only deliverer (2).

Fixes for the next build window (Universe, `--bin gpu_evolve` only):
1. Lesson 6 partial credit: +100 when PHASE = 1 (stop contract met), before the request.
   Then "stopped at the passenger" (≈ −5) beats "crashed" (≈ −105).
2. Approach lesson: a pickup request while moving/off-radius should be ignored (no-op) or
   cost a bounded penalty, not close the attempt — do not extinguish the channel.
3. Log map 1..=17 so `open_at_deadline` and `zone_entry_slow` are visible.
4. Clamp `--school-start-pool` to the candidate count (10 for service lessons).

#### 18:45 stage 3 queued: replicate stage 2 with seeds 12 and 13 (`run_life_stage3.sh`)

Same curriculum and resume point as stage 2 (stage-1 turn_far graduate, gen 198), seeds 12
and 13, 300 gens each, queued behind the stage-2 best-slot trace. Pre-registered:
- approach_passenger passes (p̂ ≥ 0.25 for 2 gens) in at least 1 of the 2 seeds within
  300 gens; if both pass, the pass gen differs by > 50 (the gen-238 pass was one draw of
  a noisy p̂, not a fixed timescale).
- pickup stays 0/96 in every seed that reaches it, and the hidden deadline bucket
  (failed − Σcodes) collapses toward 0 again — the drift is structural, not seed 11's luck.

#### 19:55 stage 3 (seed 12, gen ~250): p̂ is a per-triple measure; the pool has a duplicate start

Seed 12 shows 0/96 on every gen with gen mod 10 ∈ {1, 4, 7} (34 zero gens in a row on
that lattice, none elsewhere); seed 11 had no such lattice. Cause, from
`school_batch::start_index` and `school_result_gen*.json`:
- Each gen evaluates 3 of the 10 candidate starts: start_index = (slot mod 3 + 3·gen) mod 10.
  Gen mod 10 = 1 → {3,4,5}, 4 → {2,3,4}, 7 → {1,2,3}. Seed 12's population solves only
  starts 0, 6, 7, 9 (success 25–34 % / 25 % / 12 % / 23 % in gens 200–249) and 0 % on
  starts 1–5 and 8. Every triple without 0/6/7/9 is a guaranteed 0/96. Seed 11 was the
  opposite: by gens 200–249 it solved 2, 4, 5, 8 (26 / 27 / 25 / 12 %) and 0–7 % elsewhere,
  which is why it passed on the {1,2,3}+{4,5,6} pair at gens 237–238.
- Consequence: `p̂ ≥ 0.25 for 2 consecutive gens` is not "competence ≥ 25 %". It is
  "solves both of two specific triples" — implicitly ≥ 6 distinct starts — and a
  `streak=1` break is usually the rotation landing on an unsolved triple, not regression.
  The A/B noise I read as "p̂ oscillates" was the rotation.
- Selection is triple-dependent: `selection_score = mean − worst_failure_penalty (200)`.
  In gens whose triple no one solves, every genome scores ≈ −105 and selection is blind;
  in the others the specialist wins. Nothing pushes seed 12 toward starts 1–5.
- Candidates 0 and 6 are the SAME start: (156, 99) heading −x. Two station pairs share
  the origin station and its approach, and only `exit` differs, which approach/pickup
  never use. The "pool of 10" is 9 distinct approaches with one counted twice.

Additional fixes for the build window:
5. Dedupe service candidates by (start, forward) for lessons that ignore `exit`.
6. Advance gate: pool p̂ over a full rotation window (⌈pool / eval_starts⌉ gens) or require
   per-start p̂ ≥ threshold — not 2 consecutive gens.
7. Log per-start successes on the `[school]` line (they exist in `school_result_gen*.json`).
