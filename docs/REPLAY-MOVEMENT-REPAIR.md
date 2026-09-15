# Replay movement acceptance — 2026-09-14

The delivered `fly-observatory-20260914` recording passed rendering checks but
failed the user's actual requirement: the beings barely moved and completed no
laps. A 512-tick (8-second) untrained reflex recording cannot establish navigation
on a 1,638.84-m track. Rendering correctness is not behavioral acceptance.

## Repair plan and acceptance criteria

- [x] Inspect recorded commands and displacement, not just process exit status.
- [x] Test whether removing retinal body rendering releases the Fly brake.
  It does not: sampled median brake/speed were 0.7174/0.4677 m/s with bodies,
  0.7418/0.4283 m/s without them. This rejects nearby bodies as a sufficient
  explanation. Both runs use the same frozen executable and 512-tick horizon.
- [x] Evaluate a saved native neural champion in the current physics/perception
  implementation. Preserve the checkpoint, all its evolved time constants, and
  use exact clones (`--resume-mix 1`), without a pilot.
- [x] First run a 2,048-tick movement smoke with 16 beings in formation. Then
  evaluate the same checkpoint for 30,000 ticks on tracks 8005 and 8200,
  sequentially. These are two environments for one genotype, not independent
  evolutionary replications. Enable fuel in both; keep other settings fixed.
- [x] Require at least 80% of the 16 beings to complete two full forward circuits
  on track 8005. Report the second track separately, including failure. Count
  circuits from recorded positions and track geometry, independently of fitness
  and the legacy checkpoint counter; reject implausible jumps and shortcuts.
- [x] Make playback state, controller identity, measured motion, and completed
  laps explicit in the viewer. Validate playback by observing a moving being in
  the browser and by checking the full recorded trajectory.
- [x] Record exact commands, checkpoint/executable hashes, results and limitations.

The native champion test restores a previously demonstrated navigation capability.
It does not validate the experimental Fly controller or show new learning. Fixing
Fly's visual-to-motor calibration requires a separate controlled experiment; its
brake must not simply be disabled to produce a moving illustration.

Artifacts: `/media/llm-data/universe-runs/replay-repair-20260914/`.
GPU runs are sequential; generated data remain outside the source tree.

## Results

The checkpoint is `neuromod-lab-taxi/u1v_s077_ctrl/evolve_best_gen0014.bin`,
SHA-256 `40ba411113cb7a3b005a4a234da944d3f40be324db9427b0a7730de6378599fa`.
No time-constant reseeding, pilot or Fly flags were used. Fuel and formation were
enabled. Other beings were not rendered into the retina: this is a navigation
test, not evidence of learned traffic avoidance.

| Recording | Simulated horizon | >=1 complete circuit | >=2 complete circuits | Median odometer | Verdict |
|---|---:|---:|---:|---:|---|
| Native 8005 | 468.75 s | 16/16 | 15/16 | 6,239.55 m | Pass |
| Same checkpoint, 8200 | 468.75 s | 6/16 | 2/16 | 1,741.14 m | Fail |
| Final recorder, 8005 repeat | 468.75 s | 16/16 | 15/16 | 6,239.55 m | Pass |

The final recording took 38.95 seconds wall time. These are repeated environments
for one genotype and a deterministic reproduction, not a new evolutionary study.
The final and first 8005 binary replays are byte-for-byte identical, SHA-256
`de18e6ea167660d144c65bade58cb491b04e10744aeedf499bd66fefd12633da`.
The final executable hash is
`516aa332cfbeb8123b5ab02c4deda6989b0b30df5e49aba49f34028d9329c0ac`.

The behavioral validator ignores recorded fitness, odometer and legacy lap
counts. It projects recorded positions onto the closed track polyline and checks
sequential forward traversal from each being's first recorded position. Its
explicit tolerances are 100 m/s maximum sampled speed plus 5 m numerical margin,
20 m for nearest-segment projection changes at corners, and nominal half-width
plus 3 m for the track corridor. These tolerances are diagnostics, not a proof of
continuous collision correctness; the validator does not inspect unsampled physics
substeps. Missing/corrupt samples, planned-horizon truncation, and duplicate IDs
invalidate the recording. A stationary or off-corridor being fails individually.

`scripts/replay/browser_check.py` verified the delivered artifact in Chrome:
automatic playback, **117.84 m measured displacement** during an 8x playback
interval, pause, the selected being's three completed laps, and desktop/narrow
viewport fit. There were no JavaScript runtime exceptions. Screenshots and the
JSON report are in `native-release/browser/`.

New verification: 12 behavioral-validator tests, 84 `gpu_evolve` tests, 11
replay-filtered library tests, Node model assertions and syntax checks, and one
real GPU/CLI corrupt-checkpoint regression. The latter exits 1 without creating
a replay. Previously, a corrupt checkpoint silently started a random population.
English-language and whitespace checks pass. Passing these software checks alone
is explicitly insufficient for navigation acceptance.

## Delivered entry point and remaining limits

Open `native-release/replay.html` under the artifact directory above. Reproduce
with `scripts/replay/run_champion.py`; see `scripts/replay/README.md`. Its default
lap gate is the preregistered 80% with two complete circuits. Every being remains
visible, including the one that completed only one circuit.

The Fly spiking controller remains uncalibrated for driving. Its failed original
recording is retained as evidence. The native result is not attributed to Fly,
and no unrecorded neural activity is invented for the viewer. Native activations
and a live simulation adapter remain separate work. The poor 8200 result also
means the checkpoint is not a generally reliable driver: held-out-track training
and evaluation are still needed.
