# Stop contract A/B — pre-registration, 2026-09-24

Follows `GAP-SWEEP-20260924.md` proposed order **A**. Written before the battery ran;
the results block at the end is empty until it is filled with numbers and commands.

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

## Results

_(empty until the battery finishes — fill with numbers, seeds and the command)_
