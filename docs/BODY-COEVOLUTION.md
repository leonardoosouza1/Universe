# Experimental body–brain coevolution (E1a historical scope)

The current extension is documented in [COEVOLUTION-RUNTIME.md](COEVOLUTION-RUNTIME.md).
The description below preserves the original two-gene E1a experiment; tank, cooling,
lifetime learning and ecology now exist as separate opt-in extensions.

The production `gpu_evolve` loop can now select a brain and its inherited vehicle
capacities together. This is the first executable slice of the broader
[development roadmap](../../META/EVOLUCAO-AUTOMATICA-CORPO-CEREBRO.md), not an
implementation of developmental growth, new organs, or lifetime learning.

## Executed flow and ownership

`Genome { neural genes, body }` → selection of the entire genome → neural mutation
→ independently seeded body mutation → birth-time body expression → existing
perception/brain/motor mapper → existing GPU physics A/B/C → fitness → selection.

Both training and held-out instantiate bodies through
`src/bin/gpu_evolve/body_evolution.rs::at_birth`. Memory is still reset between
generations. Body genes do not overwrite throttle, steering, or brake decisions.
The native and Fly backends use the same body expression.

Body expression and mutation run on the CPU once per generation. The physical
parameters occupy existing `PhysicsStateGpu` fields. No shader layout, dispatch,
buffer allocation, or CPU/GPU transfer has been added to the tick. The existing
CPU fuel update now scales its load term by engine capacity; it still reads the
previous applied command, exactly as the historical fuel update did.

## Model and assumptions

Two dimensionless genes, `g_engine` and `g_brake`, are bounded to `[-1,1]`:

```text
engine_scale = 1 + 0.25 * g_engine
brake_scale  = 1 + 0.25 * g_brake
mass_scale  = 1 + 0.15 * (engine_scale - 1) + 0.05 * (brake_scale - 1)

max_engine_torque = 620 Nm  * engine_scale
max_brake_torque  = 4000 Nm * brake_scale
mass              = 1350 kg * mass_scale
fuel_delta        = (0.002 + 0.004 * throttle² * engine_scale) / 64
```

Fuel uses the existing simulation resource units, not validated litres. Mass lies
in `[1282.5,1417.5] kg`. A stronger engine costs fuel and mass; stronger brakes
cost mass. Lower investment reduces capability and cost. Suspension stiffness,
damping, and initial normal loads scale with mass to preserve the original static
sag, suspension frequency and damping ratio. Dimensions, tyre compound, thermal
model, tank capacity, and the input/action schemas stay fixed in this experiment.

These coefficients define a testable trade-off, not measured vehicle engineering
or biological metabolism. A new physical model needs its own calibration. In
particular, the existing thermal model is not evidence that larger engines produce
the correct additional heat. No reward for a prescribed action was introduced.

Birth starts at neutral genes. After selection, each non-elite child gets additive
uniform noise in `[-0.1,0.1]` per gene, then clamps to the bounds. Mutation has its
own SplitMix64 stream keyed by population seed, absolute generation, and child
slot. Brain RNG draws are untouched. Later generations can differ in neural genes
because body-dependent performance changes parent selection; this is intentional.

## Running and inspecting

```bash
env -u DISPLAY VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
  MESA_VK_VERSION_OVERRIDE=1.3 ./target/release/gpu_evolve \
  --gens 3 --beings 8 --ticks 512 --seed 7 --same-start --fuel \
  --evolving-body --driving-diagnostics --vision-being 4 --vision-every 64 \
  --out /media/llm-data/universe-runs/body-example
```

Use an unused output directory. Compare against `--fixed-body` with otherwise
identical arguments. Fixed is the default and ignores inherited body traits without
erasing them. `--evolving-body --no-fuel` is rejected; resource costs must be active.
For a checkpoint-only evaluation, use `--resume --resume-mix 1 --gens 1` in a fresh
directory containing a copied checkpoint, and select the intended body mode.

- `body_genNNNN.json`: all beings' genes and expressed mass/engine/brake capacities,
  schema/model version and enabled state. This also exposes whether mutation occurs.
- `observation.jsonl`: the selected being's actual physical capacities and fuel
  load multiplier alongside the existing tick, action and neural stages.
- `genNNNN.meta.json`: enabled state and the body-population file reference.
- Checkpoint/observed genome: the complete inherited body and neural genes.

The UI does not yet plot the new body traits. Recording them is not a claim that
existing dashboards display or aggregate them.

## Persistence

Neutral bodies retain byte-identical UGEN v2 files. Non-neutral bodies use v3:
`UGEN | u32 version=3 | u32 payload_bytes | f32[2] body | unchanged v2 payload`.
All fields are little-endian. The variable-length Fly block remains the payload
tail. The canonical loader reads legacy/v2 as neutral, rejects unknown versions,
and validates v3 lengths and gene bounds. All consumers must be rebuilt before
reading v3: historical release binaries did not reject unsupported versions.

## Scientific gate for the next experiment

Hypothesis: bounded body variation improves held-out survival/progress at the
same training and energy budget. Rival explanation: a larger engine wins a short
training straight without improving control or resource management.

After wiring validation, use at least three paired population seeds, equal starts,
and a frozen set of evaluation worlds. Compare body+brain evolution with body-fixed
brain evolution; separately replay each champion with expressed and neutral bodies.
Report completed laps, ordered progress, survival, impact energy, fuel per progress,
body diversity, and computational cost. Braking frequency alone is not success.

Choose tick/fuel budgets long enough to encounter curves and resource constraints.
The existing held-out path has a different survival envelope from training; label
its result as route generalization, not proof of resource-management generalization.
Stop on nonfinite state, corrupted persistence, broken fixed-body parity, or a
resource loophole. Do not introduce growth/plasticity to explain away an E1 null.

Validation results and exact commands: [BODY-COEVOLUTION-VALIDATION.md](BODY-COEVOLUTION-VALIDATION.md).
