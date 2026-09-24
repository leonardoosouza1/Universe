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
