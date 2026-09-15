# Brake and control battery — 2026-09-15

This battery answers whether the repaired perception path, or any one of five
control-contract interventions, makes the frozen native champion brake before
high-speed impacts. It is a paired checkpoint evaluation, not training.

## Protocol

- Vulkan execution through `llvmpipe` because the Codex sandbox exposes the AMD
  device in sysfs but not its `/dev/dri/renderD128` node.
- Checkpoint SHA-256:
  `40ba411113cb7a3b005a4a234da944d3f40be324db9427b0a7730de6378599fa`.
- Track 8005, population seed 77, 16 exact checkpoint clones, 15,000 ticks,
  64 Hz, width 28 m, fuel enabled and hippocampus disabled.
- One selected-being neural observation every 16 ticks; population-wide body
  telemetry every tick.
- All arms use the repaired motion workgroup mapping, packed-channel replacement
  and body-relative whiskers.

Each arm was produced with `scripts/replay/run_champion.py`, changing only the
flag named in the first column. `fixed-canonical` combines all five flags and is
reported as an interaction check, not as evidence for any individual mechanism.
Commands, binary hashes and input hashes are stored in each arm's `manifest.json`.

## Results

| Arm | Beings with >=1 lap | Population distance | Brake ticks / alive | Positive raw brake above 10 m/s | Impact ticks / km | Collision energy | Stationary alive time |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed default | 15/16 | 45.46 km | 0.354% | 0/550 | 29.76 | 54.14 MJ | 14.55% |
| motor head linear | 16/16 | 47.77 km | 0.359% | 0/449 | 25.85 | 65.94 MJ | 14.02% |
| throttle arbitration signed | 16/16 | 45.03 km | 0.262% | 1/515 | 22.81 | 37.46 MJ | 11.67% |
| slow kinematics canonical | 15/16 | 45.19 km | 0.320% | 0/381 | 31.05 | 53.77 MJ | 14.50% |
| wall probes continuous | 15/16 | 43.27 km | 0.518% | 0/462 | 31.15 | 49.57 MJ | 15.83% |
| reverse controller v2 | 9/16 | 33.97 km | 0.228% | 0/468 | 20.14 | 40.28 MJ | 35.50% |
| all five combined | 13/16 | 36.76 km | 0.368% | 0/516 | 19.40 | 35.39 MJ | 14.70% |

Relative to the fixed default, signed throttle kept total distance within -0.95%
while reducing impact ticks/km by 23.37%, collision energy by 30.81% and stationary
time by 2.89 percentage points. It is the only isolated intervention that improved
all three damage/stagnation measures without a material distance loss.

Continuous wall probes fixed measurement resolution: selected-being channel 102
rose from 3 to 193 distinct values and TTC channel 105 from 9 to 117. The frozen
brain did not know the new distribution; distance fell 4.81% and impacts/km rose
4.67%. This is a sensor-contract repair to train under, not an old-checkpoint patch.

The reverse-v2 arm spent 17.14% of alive ticks reversing, versus 3.35% in the fixed
default, and stationary time rose by 20.94 percentage points. Its current trigger
and duration are rejected for production training until separately recalibrated.

## Scientific verdict

The car body and physics do not block braking. The command path accepts positive
brake, records its suppression reason and applies brake torque in the physics
kernel. The frozen checkpoint itself emits nonpositive brake at driving speed:
six of seven arms recorded no positive raw-brake sample above 10 m/s, and the
remaining arm recorded one in 515 samples. Output remapping cannot manufacture a
missing decision.

The perception defects were real and are fixed, but an existing checkpoint cannot
retroactively learn their corrected semantics. The next training battery should
use the repaired kernels, signed throttle arbitration and continuous wall probes.
Canonical slow kinematics should be introduced at birth for that population. The
linear motor head and reverse-v2 should remain separate experimental arms.

The current selection pressure still has a gap: `--wall-cost` penalizes time near a
wall, while exact impact count and collision energy are diagnostic telemetry only.
Training can therefore prefer a fast wall-hugging/collision policy when its progress
outweighs the proximity cost. Before a long evolution, add a selectable exact-impact
or collision-energy objective term, prove cost zero is bit-identical, and run a dose
sweep before the paired multi-seed battery.

## Artifacts

The complete battery is under
`/home/leonardo/projects/LLM/universe-runs/brake-check-20260915/`. Every arm contains
its replay, motor trace, population report, behavioral gate and frozen binaries.
The best immediate old-checkpoint compromise is
`throttle-signed/replay.html`; it still does not demonstrate learned braking.
