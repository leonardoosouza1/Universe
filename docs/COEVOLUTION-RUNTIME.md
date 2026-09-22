# Experimental body, brain and lifetime coevolution

Implemented on 2026-09-20. All new mechanisms are opt-in. This is a bounded
experimental implementation of the [roadmap](../../META/EVOLUCAO-AUTOMATICA-CORPO-CEREBRO.md),
not evidence of improved intelligence or open-ended biological evolution.
See [validation](COEVOLUTION-VALIDATION.md) before enabling a training campaign.

## Controls and executed flow

| Flag | Effect |
|---|---|
| `--evolving-body` | Inherit/mutate engine, brakes, tank and cooling; requires fuel |
| `--plasticity` | Enable bounded acquired latent-to-motor weights during life |
| `--evolve-learning` | Enable plasticity and mutate its inherited learning rules |
| `--development` | Evolve age/energy-dependent L1/L2 availability |
| `--evolving-topology` | Evolve deterministic L1/L2 connection/neuron masks |
| `--ecology` | Resource-funded reproduction into vacant/dead slots |
| `--ecology-founders N` | Initially occupy N slots in the bounded population |
| `--ecology-maturity-ticks N` | Minimum reproductive age (default 640 ticks) |
| `--ecology-min-progress M` | Required forward progress (default 20 metres) |
| `--adaptive-curriculum --tracks SEEDS` | Advance through an explicit training sequence |

Use `--seed N`, not `--pop-seed`. Unknown arguments now fail before GPU startup;
`--help` prints the accepted flags. Always use a fresh `--out` directory.

The actual native path is:

```text
inherited genome -> birth-time body expression / GPU DNA upload
GPU perception + retina digest -> wave[192]
GPU brain: CfC -> masked BitLinear L1 -> masked L2 -> bounded motor residual
motor mapper -> GPU physics A/B/C
existing body readback -> progress/contact/fuel bookkeeping
previous transition feedback -> next brain decision
selection or resource-funded birth -> hereditary mutation, newborn state reset
sampled diagnostics -> exact observed genome + runtime trace -> replay
```

Physics uses `physics_a.wgsl`, `physics_b.wgsl`, `physics_c.wgsl`. Updating only
`physics.wgsl` does not update this executed path. The A/B/C split is preserved.

The system is not entirely GPU-resident. Existing command mapping, fuel/progress
bookkeeping, lifecycle decisions and evolution remain on the host. This change
adds small per-decision feedback writes, not a new per-tick GPU readback. Each
being reserves 512 f32 lifetime values (2048 bytes); diagnostics copy them only
at the requested sampling cadence through the existing staging operation.

## Inherited body and costs

For engine, brake, tank and cooling genes g in [-1,1], scale = 1 + 0.25 g.
The historical 620 Nm engine and 4000 Nm brake limits are multiplied by their
scales. Chassis mass is 1350[1 + .15(engine_scale-1) + .05(brake_scale-1)] kg;
tank/cooling add 1350[.03(tank_scale-1) + .02(cooling_scale-1)] kg.
Suspension stiffness, damping and initial wheel normal loads follow mass.
Fuel capacity and refill limits use the inherited tank, including body sensing.
The existing load-dependent fuel consumption is multiplied by engine_scale.

When body evolution is enabled, the active physics A kernel applies:

```text
heat    = rpm_ratio * throttle * 15 * dt * engine_scale
cooling = (speed * .3 + 2) * dt * cooling_scale
T_next  = clamp(T + heat - cooling, 20, 150) degrees C
```

These are simulation trade-offs, not calibrated vehicle thermodynamics or
biological organs. Body changes never prescribe steering, braking or throttle.

## Lifetime equations and causal timing

Let x_previous be the previous L2 latent vector (48 entries, clipped to [-1,1]),
a_previous the previous three native motor intents, and dt elapsed world time
since the last decision. The kernel updates 3 x 48 eligibility and plastic weights:

```text
E = exp(-decay * dt) E + (1-exp(-decay * dt)) outer(a_previous, x_previous)
m = previous_transition_reward * reward_gain * alive * fuel_fraction^exponent
P = clamp(exp(-weight_decay * dt) P + learning_rate * dt * m * E, -Pmax, Pmax)
residual[k] = residual_gain * clamp(sum_j(P[k,j] * x_current[j]) / 48, -Pmax, Pmax)
intent[k] = clamp(inherited_intent[k] + residual[k], -1, 1)
```

Feedback averages physical ticks since the last decision. Its source is signed
track progress (delta_metres * 64/30, clipped) minus normalized contact energy
(contact_joules/10000, clipped). It does not reward a hardcoded braking action.
Reward concerns the preceding transition; current sensory data cannot retroactively
cause that reward. This is a bounded Hebbian heuristic, not a proven policy-gradient
estimator; positive progress does not establish correct long-term credit assignment.

The developmental age advances by rate * elapsed_seconds while alive, clipped to
[0,1]. A newborn starts at age zero even when born at a large world tick. L1 retains
a scaffold in each 24-neuron region; L2 retains the motor outputs. Age and energy
scale availability from a nonzero scaffold to the inherited capacity. Topology
masks use a deterministic edge hash independent of population slot. Buffers remain
fixed at 96 L1 / 48 L2 units; masked capacity is not dynamic GPU allocation and is
not a demonstrated compute saving.

| DNA slots | Meaning | Default |
|---|---|---|
| 0..3 | learning rate, eligibility decay, Pmax, residual gain | .05, 2, .25, .10 |
| 4..7 | L1 age start/end, energy minimum, density | 0, 1, 0, 1 |
| 8..11 | L2 age start/end, energy minimum, density | 0, 1, 0, 1 |
| 12..15 | reward gain, weight decay, age rate, energy exponent | 1, .01, .1, 1 |

Finite bounds and ordered age windows are validated by `LifeGenes::is_valid`.
Mutations use a private RNG stream and preserve elites. Only DNA is inherited;
acquired P/E, previous latents, recurrent memory and developmental age reset at birth.

## Persistence and observation contract

The canonical loader is `src/gpu/genome_io.rs`. Do not implement another parser.

- Legacy / UGEN v2: default body, organs and lifetime DNA.
- UGEN v3: 20-byte header with body genes, then unchanged v2 payload.
- UGEN v4: 92-byte header: magic, version, payload byte length, body[2], organs[2],
  lifetime[16], then the unchanged v2 payload. Little endian. Fly tail remains last.
- Neutral new DNA preserves historical v2 output; body-only variation can use v3.
- Checkpoints store hereditary state, not a resumable lifetime/ecology snapshot.

Lifetime trace layout: configuration 0:32; feedback 32:36; P 36:180; E 180:324;
previous latent 324:372; previous motor 372:375; runtime starts at 375. Runtime
contains age, previous tick, reward/energy EMA, elapsed life, dispatch count and
motor residuals at 384:387. Consult the producer metadata for version and stride.

`body_genNNNN.json` records the generation's initial population. Newborn DNA is
stored separately under `births_genNNNN/`. `observation.jsonl` carries selected
being, tick, lineage_id and exact genome_file; the selected slot can change owners.
The HTML exporter attaches each observed genome and switches it with the sample.
The 3D graph shows inherited wiring; the lifetime table reports measured acquired
state. It does not pretend acquired weights are already rendered as 3D synapses.

Historical control runs retain the shared default motor spline. Experimental
body/lifetime/ecology runs use each being's inherited spline, including newborns.
Therefore compare with `--motor-head linear` when isolating these mechanisms from
spline inheritance; the default-spline combined comparison has that extra factor.

## Reproduction and curriculum

Ecology never exceeds allocated population capacity and never replaces a living
being. Eligible mature parents require progress, sufficient energy and cooldown.
The transfer conserves the resource budget:
`parent_before = parent_after + child_energy + reproduction_loss`.
Default child allocation is .20 of the base tank, loss .05, reserve .10.
Child capacity is checked. Dead-slot state is reset across perception, physics,
brain, hippocampus, slow state, contact metrics and lap tracking; the canonical
mutation path creates its genome. Birth/death events have lineage IDs and DNA paths.
Extinction does not silently restart the population. Lineage scope is one generation;
a resumed checkpoint starts fresh founders, not a recovered ecosystem.

Ecology currently rejects Fly, pilot/forced controls, missing fuel and the existing
wiring/sensor subscription-cost modes whose cached per-generation costs cannot
correctly price an in-episode birth. Lifetime mechanisms also require native brain.

Adaptive curriculum advances only after the entire population completes an ordered
lap on the frontier in two consecutive generations. Every fifth episode probes
mastered worlds; success below .8 returns to the failed seed. Completed curricula
continue archive probes. Held-out seeds are rejected from the training sequence,
and held-out scores never enter promotion. `curriculum_genNNNN.json` records decisions.
The frontier/lineage archive is not currently restored from a genome-only checkpoint.

## Explicit boundaries

This release does not invent new body organs, arbitrary sensor modalities, unlimited
network growth or new world geometries. Sensor mutation already available elsewhere
is not a new age-dependent sensor-development program here. Neural masks have no
new calibrated metabolic price; do not label them evolved energy efficiency.

The canonical exporters and lab-ui are now integrated; see
[INTEGRATION-20260921.md](INTEGRATION-20260921.md). Multi-lineage recordings are
packaged as separately identified life traces, preserving the single-genome contract.
The panel selects recorded generation/being/life and displays measured lifetime state.
Long campaigns additionally need
tick storage hardening beyond f32 integer precision (2^24 ticks) and persistence of
curriculum/ecological state. Keep this experimental path in bounded episodes.

These boundaries and failed scientific hypotheses must remain visible to Claude and
future changes; successful GPU execution is not evidence that the combination learns
better. Promote one mechanism at a time only after matched-budget held-out ablations.
