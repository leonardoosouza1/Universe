# Organism integration ledger — 2026-09-22

Scope: all eight gates in organismo-taxi-fechamento.md; no claim of completion until each has evidence. Existing race behaviour remains a separate scenario. Simulation runs only on real GPU; builds have one worker.

## Executed path before extension

| Phase | Producer | Device / frequency | Output / consumer |
|---|---|---|---|
| Perception | gpu_evolve/context.rs GpuCtx::tick | GPU each tick | retina + perception wave → brain |
| Neural decision | BrainGpu + slow | GPU scheduled brain tick | resident state, intent readback → host action assembly |
| Action | GpuCtx::tick motor mapping | host each tick | PhysicsStateGpu, 368-byte serialized stride |
| Physics | PhysicsGpu::step | GPU A/B/C each tick | 224-byte output, existing synchronous readback |
| Survival | context.rs + main.rs | host each tick | legacy fuel/life/lap refill and fitness |
| Observation | driving_diagnostics.rs | sampled pre-neural/post-physics | observation.jsonl → pack → lab-ui |

The new organism state will be GPU-owned. Its pre/post physics passes must share the existing encoder and device. Legacy thermal/resource writers must be bypassed explicitly in the new scenario. Numerical telemetry is sampled and must identify units and phase. Existing host physics/action traffic is a known residency gap, not something hidden by the new module.

## Work packages

1. Runtime descriptors and action/body contracts: physics boundary already tested; all other boundaries still require coverage.
2. Resident organism body: implementing energy/water/material, wear, fatigue, paid repair, maturation and senescence.
3. Hereditary persistence: implementing a versioned organism DNA extension; acquired state is separate.
4. Brain integration: pending sensor adapter and explicit interaction channels.
5. Taxi services/passengers: pending deterministic GPU arbitration and connected scenario.
6. Lifecycle/selection/resume: pending integration and conservation gates.
7. Telemetry/UI: correcting physical playback clock and adding numeric body inspection.
8. Final GPU/mechanical/behavioural/visual evidence: pending.

Work-in-progress entries are not release approvals. Scientific improvement remains an experiment separate from mechanical completion.

## Mechanical checks declared before integrated collection

- Device: real Vulkan Radeon, no software adapter; batch 2–4 beings, 64–256 ticks initially, 64 Hz; one compiler worker.
- Body GPU fixtures: negative-capacity DNA never starts over capacity; missing material prevents repair; energy/material differences match paid basal/growth/repair within f32 tolerance; cooling dt refinement converges; chronological age is not reset by repair; dead body cannot power the motor.
- Taxi scripted GPU fixture: moving requests rejected; station stock plus transferred stock conserved; delivery credited exactly once. Fixture positions are injected for transaction testing, not claimed as learned navigation or simulated travel.
- Integrated run: native brain → scenario observations/actions → production physics A/B/C → body → taxi on one device; finite trace with age, resources, temperatures and task state, no legacy lap fuel grants or countdown death.
- Checkpoint: exact continuation requires both inherited population DNA and acquired body/world/brain/host state; compare resumed vs uninterrupted traces before marking PASS.

Model coefficients are engineering assumptions. Energy price is 0.00001 credit/J and service rate 100000 J/s; water 1 credit/kg at1 kg/s, material5 credits/kg at0.2 kg/s. Fare10+0.02*direct-distance metres; initial25 credits is declared currency genesis. These values replace an untested draft priced100× higher that was incompatible with the mechanical energy budget. This is not evidence of economic or evolutionary optimality.

The body uses crankshaft torque×crankshaft angular velocity for mechanical power; wheel-side torque is divided by the same gear/final-drive ratio used in physics. Basal and mechanical costs are paid before growth and repair. Maximum torque is energy-limited before mechanical dispatch. Heat includes conversion losses and paid repair/growth energy; vaporization removes latent thermal energy without charging chemical energy a second time. Temperature has no artificial clamp that discards heat.

## Current evidence (bounded, not release)

- Production binary `cargo check --offline --bin gpu_evolve`: passed before final checkpoint integration.
- New WGSL parse/type validation: passed.
- Basic body GPU fixture: passed.
- Paid repair/growth/cooling/age/death/birth-capacity GPU fixture: passed before the later unclamped-temperature refinement; rerun required.
- Taxi transaction GPU fixture: passed after coefficient correction.
- UI/trace agents report tests passed; integrator must re-run and inspect actual new run/package before release.

## Evidence update — 2026-09-22, bounded integration

The following are observed artifacts and commands from the current worktree. They
do not constitute approval of the eight gates or of the full organism plan.

| Evidence | Command / artifact | Result |
|---|---|---|
| Contract gate | `neuromod-lab/.venv-rocm/bin/python scripts/contract_gate.py verify --output /tmp/ui-contract-gate.json` | PASS: schema equality, 50 focused Python/JS transport tests, fingerprint unchanged, GPU not requested |
| Python trace contract | `.../.venv-rocm/bin/python -m pytest -q tests/test_quantity_contract.py` | PASS: 39 tests; organism/taxi strict codecs and producer shapes included |
| UI contract/telemetry | `lab-ui: npm run typecheck`; Vitest focused body/taxi/decoder tests | PASS: typecheck; 15 focused tests |
| Real package | `/tmp/organism-pack-d.log`; `python -m lab_api.trace.package_run ...organism-integration-20260922-d --db /media/llm-data/lab.db --force` | Trace produced at `trace/gen0000/being00/champion`; API index exposes gen 0 / being 0 / champion |
| API/UI path | API on `127.0.0.1:8131`; Vite `LAB_API_TARGET=http://127.0.0.1:8131` on `127.0.0.1:5176`; proxied health and manifest GETs | HTTP 200; no LAN bind; browser opened the real trace route |
| GPU fixture logs | `/tmp/organism-body-gpu.log`, `balance-gpu.log`, `physiology-final.log`, `taxi-final.log`, `reproduction-gpu.log`, `checkpoint-format.log` | Individual focused fixtures report `ok`; these are fixture evidence, not an integrated campaign |
| Integrated run | `/tmp/organism-run-d.log` | Completed gen 0, 4 beings; 0/4 reached one lap, median 40 m, max 64 m; insufficient generations for trajectory |

The package/UI path is mechanically exercised. The real run is evidence of
execution and finite recording only; it is not evidence of learned taxi success,
economic adequacy, selection quality, or checkpoint continuation.

## Gate status after this evidence update

This is a mechanical status ledger, not a blanket PASS. C01–C12 remain open
unless their boundary has a dedicated artifact and semantic check. Current status:

| Gate | Status | Remaining proof |
|---|---|---|
| C01 execution | PARTIAL | authenticated/queued production boundary and effective command identity |
| C02 DNA → body/brain | PARTIAL | full inherited-vs-acquired state and reset audit |
| C03 world/body → sensors | PARTIAL | scenario geometry/phase/unit audit across the full sensor path |
| C04 sensors → brain | PARTIAL | explicit 192-channel remap and update cadence evidence |
| C05 brain → motor | PARTIAL | action provenance, age and override/readback audit |
| C06 motor → physics | PARTIAL | scenario action ABI and dt/order evidence |
| C07 physics → life/world | PARTIAL | writer ownership and no legacy race grants in the integrated scenario |
| C08 consequence → learning/selection | BLOCKED | no evidence of valid feedback-to-fitness/learning semantics |
| C09 birth/resume | PARTIAL | uninterrupted-vs-resumed trace equivalence with acquired state |
| C10 runtime → recording | PASS for bounded trace shape; PARTIAL overall | post-physics phase proof for every organism/taxi field and live boundary |
| C11 recording → package/API | PASS for current package path; PARTIAL overall | mixed-generation/lineage rejection and production artifact audit |
| C12 API → panel | PASS for current real trace route; PARTIAL overall | complete panel coverage and units/absence audit across all consumers |

Open plan items remain: integrated behavioral taxi delivery, economic/resource
conservation under contention, lifecycle/senescence transitions, selection and
learning success, deterministic resume equivalence, full execution inventory,
and the visual/behavioral evidence required by `ORGANISMO-TAXI-SPEC.md`.

## Consumer audit of the packaged run

The real packaged manifest contains `organism_body` and `taxi_state`; the UI
reads both with the same no-future sample resolver used by replay panels. Body
values retain raw decoded numbers while the presentation applies explicit
Portuguese labels/units, percentage formatting, boolean labels and scientific
notation for sub-`1e-4` values. Taxi refusal codes and passenger state are
presented as named states. Missing fields remain `não registrado` and are never
converted to zero. The playback clock advances from elapsed wall time × replay
tick rate × speed, independently of the recording stride. Brain 3D continues to
label itself as a functional schematic; it does not claim anatomical coordinates.

These are consumer-contract proofs only. They do not close the scientific gates
for causal brain coverage, learned destination choice, resource conservation,
resume equivalence, or selection/fitness validity.


## Resident sensor compiler failure — bounded evidence, 2026-09-22

On AMD Radeon RX 6750 XT (RADV NAVI22), Mesa
`25.2.8-0ubuntu0.24.04.2`, the resident sensor fixture initially terminated with
SIGSEGV. Instrumentation localized the failure inside compute-pipeline creation:
`sensor compiling bridge` appeared, but `sensor compiled bridge` did not.
Naga parsing/type validation succeeded. Splitting the shader into ordered probes,
track geometry and scalar bridge passes allowed the first two pipelines to compile;
the scalar bridge still crashed. Replacing its local `pp[8]` array and dynamically
indexed pose-copy loop with eight explicit assignments allowed all three pipelines
and the unchanged analytic fixture assertions to pass.

Observed evidence: `/tmp/organism-sensor-gpu.log`, command filter
`cargo test --offline --lib gpu_resident_sensor_geometry_fixture -- --ignored --nocapture --test-threads=1`
under the real Radeon adapter environment (`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json`,
`MESA_VK_VERSION_OVERRIDE=1.3`, no DISPLAY). Result: one passed GPU fixture, zero
failed, approximately 0.09 s test execution. The fixture checks known square-track
projection/horizons, boundary-traversal distances, previous digest edges, intent
memory and pose-buffer equality; it contains no CPU physics simulator.

The demonstrated workaround is explicit pose assignments in the split shader.
This does **not** establish register exhaustion, an exact compiler defect, or
portability to other drivers. The dynamic local-array copy was the last isolated
source change between the failing and passing split variants. The fixture must
remain intact. This evidence validates the bounded sensor adapter, not learned
taxi behavior or full uninterrupted-versus-resumed equivalence.

## Integrated resident evidence — later 2026-09-22 update

This section supersedes earlier implementation-status statements, without turning
mechanical evidence into a claim of learned intelligence.

- Resident action, sensors, perception, neural update, body, physics, taxi transaction
  and telemetry are integrated by `src/bin/gpu_evolve/organism_tick.rs`. The host
  samples observations/checkpoints rather than advancing those dynamics.
- `/tmp/organism-release-final.log`: nine focused resident GPU tests plus one taxi
  GPU test pass on RX6750XT. A192-tick execution resumed at96 matches the complete
  final observation (except relocated genome path) and all-being organism result.
  Continued CSV windows match too:65–128 and129–192, each64ticks.
- `/tmp/organism-body-final-gpu.log`: four focused physiology tests pass. Fatigue
  timescales and heat conservation outside former temperature clamps are checked.
- `/tmp/organism-reproduction-final.log`: revised birth water budget conserves
  resources and allows paid newborn growth without free currency/service.
- `/tmp/organism-taxi-shaped-gpu.log`: active-task potential increases on approach,
  decreases on retreat, resets on delivery and never replaces delivery accounting.
- `/media/llm-data/universe-runs/organism-ecology-20260922`: actual ecological
  slot replacements at2240,2304,3136,3968. This run exposed the subsequently fixed
  newborn-water issue; it is not evidence for the corrected birth configuration.
- `/media/llm-data/universe-runs/organism-evolution-20260922-c`:8beings,3generations,
  4096ticks/generation, body/sensor/lifetime mutation and separate held-out seed.
  End-of-generation onboard counts2,1,1; deliveries0,0,0. Maximum odometers
  480.63m,480.38m,450.76m. No monotonic improvement or successful delivery is claimed.
- The real generation2 trace is packaged and loaded through the API/UI:128samples,
  body/taxi tables, motor intermediates, native brain schematic1261nodes/200visible
  top-kconnections. Browser1x advanced approximately946ticks in15wallseconds,
  consistent with64Hz after interaction overhead.

The first unshaped run (`organism-evolution-20260922-b`) had identical fitness for
all8beings in generation1. The bounded task potential and initial depot address
that measured lack of selection signal; they remain a hypothesis about learning.

Remaining scientific work: repeat across seeds, longer horizons, completed
deliveries, ablations of shaping/plasticity/body evolution, resource-economic
calibration and robust held-out task success. Those are not made PASS by compilation,
contracts or additional organs. See `ORGANISM-RESIDENT-RUNTIME.md` for current flow.

### Final ecological and map checks

`organism-ecology-final-20260922` verifies the corrected birth configuration in
production: births at2240,2304,3584; surviving children reach maturity0.454 and0.148
with positive water. Restoring its3072 checkpoint into
`organism-ecology-resumed-20260922` reproduces all four final organism rows exactly;
complete observation differs only in the relocated genome filename.

`organism-visual-final-20260922` records4096ticks (64simulated seconds) for8beings,
with full neural trace for explicitly selected being1. Its static geometry contains
1463 row-runs of real road/wall/hole cells, using the same grid as the GPU. The
selected being carries a passenger at episode end; no delivery is claimed.
The artifact exposes all recorded native stages, not an anatomical fly connectome.

### Final browser check

The actual API-backed route
`http://127.0.0.1:5176/run/organism-visual-final-20260922/0/1`
was visually inspected in the embedded browser after resetting its saved layout.
Road/wall geometry, four station markers, observed trajectory and the native
1261-neuron schematic render together (200 visible top-k connections). Lower
panels retain numerical body telemetry and recorded brain stages. Brain runtime
slot375 is now labeled normalized neural maturation, not physical age in seconds.
The geometry decoder rejects tuple dimensions and unknown material IDs rather
than silently accepting an invented producer contract. `npm run typecheck` and
`npx vitest run src/panels/organism-geometry.test.ts` pass. These are host-side UI
contract checks, not CPU simulation or physics-reference validation.

Full neural telemetry is deliberately recorded for selected being1; physical
population replay does not imply neural traces exist for the other seven beings.
The diagram is a functional native network, not anatomical fly reconstruction.
