# Organism resident runtime — 2026-09-22

This is the executed `organism-taxi-v1` path in `gpu_evolve`, not the legacy race loop.

## Tick contract

One ordered command encoder runs:

1. Resident physics/body/previous intent → sensor probes → road geometry → scalar extras and sensor poses.
2. Previous retina copy → current retina → existing perception → retina digest.
3. Body and taxi interoception overwrite their declared wave channels 160–191.
4. On neural cadence: consequences → lifetime feedback → native recurrent brain. Slow brain runs at world cadence.
5. GPU intent/slow outputs → native action mapper → physics input. Its selected-being trace is copied only at observation ticks.
6. Body pre-step applies inherited capacities, fatigue, wear, maturity and affordable torque.
7. Existing physics A/B/C → yaw integration → rolling telemetry → body resource/thermal update → taxi transaction.
8. Scatter acquired physical output back into the resident input, preserving body parameters and commands.

Host code submits commands and samples state; it does not integrate numerical body/physics/sensors. Observation boundaries are diagnostics, replay frames, each 64-tick window, requested checkpoints and episode end. Sampling still requires GPU→CPU transfers and synchronization. This is not a claim of zero host overhead.

## Interfaces and ownership

- Physics input: 92 floats including typed bit patterns; output: 56 floats. Contact mask in output39 is u32 bits; input39 is the historical numeric mask. Collision energy is output55 in joules.
- Lifetime state: 512 floats per being (`src/gpu/lifetime.rs`). Genes 0..16, feedback 32..36 (reward, energy fraction, alive), P 36..180, E 180..324, previous latent 324..372, previous motor 372..375, runtime 375..387 (residual on the 3 motor intents at 384..386), online-feedback baselines 400 money / 401 task progress / 402 task key (`organism_feedback.wgsl`, modes legacy/symmetric/progress via `--feedback-mode`). Flag bit 3 (`--plasticity-freeze-after`) applies the residual without updating P.
- Body: 128 floats. Slots 31/61/62 are the task-progress memory (key, best remaining, initial remaining; `taxi.wgsl`). Slots98–102 are lifetime metrics; 104–122 are rolling 64-tick metrics; 123 contact-edge memory; 124–126 independent left/right/front probes; 127 bounded active-task score.
- Action: stride96, active channels0–8: throttle, steering, brake, shift up/down, service, pickup, dropoff, near-stop reverse request. These are policy outputs; no curve-braking autopilot was added.
- Action trace: GPU-produced after-head, modulation, steering filter, speed and brake-gate state. Low-speed braking remains available in this scenario.
- Native recurrent state, lifetime adaptation, body, taxi stocks, resident physics, perception history and gear/filter state participate in snapshots. Child reset touches only the newborn's acquired state; hereditary matrices and horizons are refreshed from DNA.
- Tire wear is simulated. The legacy host tire-temperature model is not advanced by this path and is reported absent. Unmeasured host joy/hole/stagnation diagnostics are null, not fabricated zeros.

## Numerical corrections

Full-load fatigue uses a declared 120-second accumulation scale; rest recovers over60 seconds. These are engineering model assumptions, not biological calibration. The earlier 1-second accumulation exhausted capacity almost immediately. Thermal state now follows the two-compartment energy balance without silently clipping temperatures. Positive finite Kelvin is validated.

On the tested RADV stack, dynamic indexing of a private pose array caused a compiler crash before dispatch. Explicit pose writes pass the same fixture. Sensors use three bounded passes; the underlying equations are unchanged. Evidence does not establish a general driver root cause beyond this reproducer.

## Reproduction and verification

Use `scripts/verify_organism_gpu.sh OUTPUT_DIRECTORY` for bounded hardware-GPU fixtures plus uninterrupted/resumed comparison. Output must be new. Never use a software adapter or CPU simulation fallback.

Parent-verified bounded evidence:

- `/tmp/organism-body-final-gpu.log`: four body fixtures pass, including fatigue scales and unclipped thermal balance.
- `/tmp/organism-release-gate.log`: nine resident GPU fixtures and one taxi fixture pass; final complete observation (excluding relocated genome path) and organism result match across checkpoint continuation.
- `/media/llm-data/universe-runs/organism-release-gate-20260922/`: real run artifacts.
- `/tmp/organism-contract-gate.json`: static producer/consumer contract report; this is not GPU evidence.

A mechanical pass does not demonstrate learned deliveries, generalization or a successful evolutionary objective. Those require the separate population runs and held-out evaluation. Native brain3D is a functional schematic, not fly anatomical reconstruction.

## Evolutionary signal and birth viability

The first integrated three-generation run produced zero deliveries and identical fitness in generation1. Mechanical execution was therefore not a success criterion for learning. The objective now includes a transient onboard-task score100–200, based on remaining destination distance, below a completed delivery1000. It is recomputed, never accumulated, so oscillation cannot farm it. The initial depot uses the declared spawn, allowing task exposure before long-distance navigation. No motor command is prescribed.

A newborn receives25% of its own water capacity, debited from the parent while preserving1kg parental reserve. The previous10%-of-parent transfer could leave offspring below their growth threshold. GPU fixtures verify paid growth without free service or money.

## Static map contract

`organism_genNNNN.json.geometry` uses encoding`row-runs-v1`, scalar integer
`size` for a square grid, positive `cell_size_m`, and tuples
`[row,start_column,end_column_exclusive,material_id]`. Coordinates are the same
as GPU sensing: cell(column,row) is centered at `(column*cell_size_m,
row*cell_size_m)` in world(x,z). There is no centered-origin translation or
world-z inversion; only canvas presentation flips z. Each cell extends half
a cell around its center. Grass0 is implicit; road1, wall2 and hole3 are explicit.
A producer/consumer mismatch discovered during browser inspection was fixed
against the real artifact, not merely a synthetic frontend fixture.
