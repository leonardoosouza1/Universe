# Architecture gaps before Taxi World

**Audit date:** 2026-09-17  
**Branch inspected:** `race-world-v3` at `6bfb765`  
**Status vocabulary:** verified = command/artifact; static = code inspection; inference = interpretation.

## Decision

Do not merge the taxi simulation into the race runtime yet. Keep `taxi-world` as the
fast CPU oracle until V1–V9 prove parity with the Python laboratory. Stabilize shared
data contracts in parallel. Extract common runtime traits only after the complete
taxi brain exists, when there are two real implementations to compare.

The current blocker is not GPU performance. Taxi v0 is fast, but simulates a
different world from the laboratory because its effective configuration, retina,
metabolic cost and complete brain are absent.

## Current execution paths

### Race

```text
WorldSpec / VoxelWorld
  -> CPU and GPU perception -> sensory_wave[192]
  -> KAN -> CfC -> BitLinear L1/L2 -> final_intent[96]
  -> intent readback
  -> CPU motor head + slow/hippocampal modulation + overrides
  -> PhysicsGpu -> body-state readback
  -> CPU progress/life/lap/fitness -> race evolution pipeline
```

The authoritative readbacks and removal plan are in
[`KERNELS-TICK-SYNC.md`](KERNELS-TICK-SYNC.md). Its phases 2–6 are not implemented.

### Taxi v0

```text
scenario JSON (partially read) + trace map
  -> Config::default for most laws
  -> 29-column option senses
  -> linear Genome(Vec<f32>, NP=40)
  -> 9 movement + 10 service scores
  -> CPU physics/services/passengers
  -> life fitness -> taxi Simulador + GA + judge
```

The paths share neither observation, action, body, brain-state, genome nor episode
contracts. Moving files between them would preserve names, not semantics.

## Verified `taxi-world` state

Command:

```text
CARGO_TARGET_DIR=/tmp/universe-taxi-target CARGO_BUILD_JOBS=4 \
nice -n 10 cargo test -p taxi-world -- --test-threads=4
```

Result: **94 passed, 0 failed, 6 ignored**. The end-to-end CLI test passed when it
could write its declared artifact under `/media/llm-data`.

The six ignored tests include the real Python trace identity gates. The green suite
proves internal consistency and the synthetic fixture, but does not currently
re-prove P0–P3 against the external oracle.

Static inspection found every v1 deliverable absent:

| Stage | Missing artifact | Consequence |
|---|---|---|
| V1 | `neuromod-lab/src/exporta_mundo.py` and new traces | no effective-world fixture |
| V2 | `src/config_lab.rs` | scenario laws silently remain defaults |
| V3 | `tests/p4.rs` | measured +9.6% fitness bias remains unresolved |
| V4 | `src/retina.rs` | taxi observes NF=29 instead of NF=45 |
| V5 | `src/metabolism.rs` | brain complexity has no physical cost |
| V6 | `src/brain/` | only the linear NP=40 policy exists |
| V7 | `src/replay.rs` | no inspectable taxi life trace |
| V7b | lab-ui taxi adapter | no shared professional visualization |
| V8 | `tests/determinism.rs` | thread-count determinism is unproved |

V9 is consequently blocked.

## Architectural gaps

### G0 — Effective configuration is not a closed contract

The CLI reads a subset of the scenario while the tick uses defaults for other laws.
A run can look valid while being a different experiment.

**Gate:** V1 and V2. Every field used by a tick must be mapped or rejected. Persist
`config_hash`, schema version and the complete effective configuration in every run.

### G1 — The universal substrate is not runtime architecture

`src/substrate.rs` describes `World`, `Percept` and `Agent`, but no production type
implements them. The contract depends on Bevy `Vec3`, embeds decision inside the
agent, lacks batch/device semantics, and has no reset, snapshot, schema or trace
contract. Treat it as a manifesto, not the seam for the taxi port.

**Gate:** no production migration should depend on these traits until two adapters
pass the same contract tests.

### G2 — Observation semantics belong to each world

Race uses a fixed 192-channel signed wave. Taxi uses NF=29 now and NF=45 with retina,
organized per candidate action. Channel number alone cannot define meaning.

**Required contract:** versioned `ObservationSpec` with channel ID, name, shape,
dtype, range, unit, sampling phase, validity mask and producer. Runtime data can stay
flat and GPU-friendly; the descriptor makes it interpretable and testable.

### G3 — Body and motor head are coupled to the world

Race emits continuous throttle/steering/brake and then applies KAN, slow brain,
hippocampus and overrides. Taxi chooses movement and service actions. Sharing the
same final head would make one body imitate the other's action space.

**Required seam:** the reusable brain emits latent intent; a body-specific
`ActionHead` maps it to a versioned `ActionSpec`. Trace raw intent, final applied
action, override cause and physical result separately.

### G4 — “Brain” currently means incompatible boundaries

- `BrainForward` abstracts one layer only.
- `SovereignEngine` owns KAN, CfC, L1/L2, recurrent state and 96 intents.
- taxi V6 specifies a different thalamus/cortex/hippocampus policy over options.

Do not force these into a trait based only on matching method names.

**Extraction gate after V6:** both implementations expose `BrainSpec`, `GenomeSpec`,
reset, recurrent snapshot/restore, observation and latent-intent descriptors,
block activation/cost trace, and a deterministic single-step reference.

### G5 — Genome is untyped at the integration boundary

Taxi stores `Genome(Vec<f32>)` with NP=40; complete taxi expects NP≈765. Race stores
multiple evolved and fixed/runtime blocks. Length checks cannot detect semantic
reorder, fixed defaults or dead genes.

**Required contract:** versioned named regions with dimensions, role, mutability and
runtime source. Export the value actually applied. A gene-to-runtime coverage test
must fail when a region is stored but unused.

### G6 — Evolution harnesses are duplicated

Race evolution is embedded in a Bevy/GPU pipeline; taxi owns another GA and
`Simulador`. The reusable part is episode/evaluation lifecycle, not a universal
fitness formula.

**Required seam after V9:** world adapters own survival laws and measures; a shared
harness owns seeds, population lifecycle, checkpoints, paired evaluation, manifests
and stopping rules. Fitness components remain named and world-specific.

### G7 — Observability is being designed twice

Taxi PRD V7b asks for another static `replay.html`, while `lab-ui` and micro-pack
already exist. A second viewer will diverge in controls, schemas and brain rendering.

**Plan correction:** V7 produces a versioned world-neutral trace envelope plus taxi
payload blocks. V7b becomes a taxi adapter in `lab-ui`: map/district/services, body
curves, events, action scores, block activation, gates and memory. One selected being
may own the detailed trace; the UI must state which beings lack it.

### G8 — The race orchestrator is still a monolith

Static size: `gpu_evolve/main.rs` has 6,172 lines and `evolution/pipeline.rs` has
2,054. Tick phases, observation, control, physics, bookkeeping, recording and
evolution remain difficult to reason about independently.

**Gate before another production world:** extract phase functions and typed phase
outputs without changing behavior. Validate each extraction with the same seed,
genome and tick trace. Do not rewrite mathematics during the refactor.

### G9 — CPU/GPU ownership is unfinished

Race reads final intent and body state each active tick because host code owns control
and lifecycle decisions. Taxi CPU is currently fast enough and should remain CPU
through V9.

**Rule:** finish parity first. Migrate taxi kernels only if V9 profiling shows a
measured bottleneck. For race, remove readbacks only after moving every authoritative
consumer of those values.

### G10 — Generalization is not reliable yet

The B9 artifact (`b9-analise/b9.json`) shows the varied pool did not improve held-out
crossing: pool `0/16`, fixed `2/16`; scaling reached `3/16`, with only `1/4` seeds
strictly better than pool. This concerns the race curriculum, not taxi parity, but it
blocks claims of robust intelligence.

**Gate:** promotion uses a distribution of training worlds and a distinct held-out
suite. Report per-seed success, worst-case world and regression on mastered worlds.
Advance difficulty on demonstrated competence and retain earlier worlds.

### G11 — Artifact paths do not identify the experiment

The race B9 artifacts currently live below
`/media/llm-data/universe-runs/neuromod-lab-taxi/`. Inferring the world from a
directory name can therefore make ingestion, panels and agents classify a race run
as taxi data.

**Required contract:** manifests carry stable `world_id`, `body_id`, `brain_id`,
`runner_id` and schema versions. Databases and UIs use those fields, not path
substrings. Existing artifacts remain in place and are indexed with explicit legacy
metadata rather than silently moved.

## Recommended order

### A — Restore scientific identity

1. V1: export complete effective worlds and real traces.
2. Make the real P0–P3 acceptance command fail when its fixture is absent.
3. V2: complete config mapping with reject-on-unmapped-used-field.
4. V3: isolate the first P4 divergence and close the ≤3% distribution gate.

No architecture extraction should delay these items.

### B — Complete the second organism

5. V4 retina.
6. V5 metabolism and gene cost.
7. V6 complete brain and ablations.
8. V8 deterministic metrics and thread behavior.

Only then are there two real brains/bodies from which to extract common contracts.

### C — Unify contracts and observability

9. Add versioned observation, action, body, brain, genome and trace descriptors.
10. Implement race and taxi adapters with cross-contract tests.
11. Make V7/V7b target the shared trace envelope and `lab-ui`.
12. Extract the episode/evaluation harness, preserving world-specific fitness.

### D — Prove, then optimize

13. Run V9 with its prediction registered before collection.
14. Profile CPU, GPU, transfers and synchronization.
15. Move measured bottlenecks while retaining a CPU oracle per migrated kernel.
16. Add taxi to the visual Universe only after parity and shared replay pass from the
    same run artifact.

## Definition of “architecture round”

- P0–P4 green from a reproducible, non-ignored acceptance command.
- Complete taxi brain/body pass V9 against Rust and Python judges.
- Every run carries effective config and observation/action/genome descriptors.
- Race and taxi share episode, artifact and trace envelopes through adapters.
- `lab-ui` opens either world without a standalone viewer.
- Brain reset/snapshot and body action application are independent contracts.
- CPU/GPU transfers are profiled and every authoritative readback is identified.
- Held-out performance is a promotion gate with regression protection.
- Artifact identity comes from versioned manifest fields, never directory names.

Until then, `taxi-world` is a valuable fast oracle and experimental engine, but not
yet the second production world of Universe.
