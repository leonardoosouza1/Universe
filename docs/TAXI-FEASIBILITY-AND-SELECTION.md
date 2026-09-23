# Taxi feasibility and selection correction — 2026-09-22

## Scope and pre-registered checks

Correct static road-distance shaping and add an explicit GPU reference pilot
feasibility witness. No learned driving claim, no automatic steering for evolved
beings, no CPU dynamics. Existing working changes are retained.

Flow before editing:

| Phase | Producer | Data | Consumer |
|---|---|---|---|
| Geometry | organism_city::build_city (host, once) | road/wall grid, station positions (m) | resident world upload |
| Action | brain.wgsl → organism_action.wgsl (GPU) | intent96 → throttle/steering/brake, gears | resident physics |
| Consequence | physical integration → body post (GPU) | position m, speed m/s, resources | taxi.wgsl::post |
| Task score | taxi.wgsl::post (GPU, each tick) | body127 = bounded current-task potential | organism_scenario::fitness (generation sample) |
| Feedback | organism_feedback.wgsl (GPU, decision cadence) | delivery/damage/energy deltas → lifetime32..35 | brain.wgsl plasticity (motor3×48) |

Existing observation inspected: taxi-observatory-full-20260922 gen2 being0 tick32:
raw brake0.901 → head0.669 → modulation/applied0.674. No delivery in any of
three generations. This demonstrates actuator availability, not good timing.

Static defect: task potential uses straight-line destination distance. Across a
solid block it can prefer the wrong street direction. The correction evaluates
shortest distance on the actual traversable grid. Static distance fields are
prepared once and uploaded; GPU samples them each tick. They are evaluative map
knowledge, not an extra hidden steering command. Unreachable/off-road positions
must never receive a near-destination bonus. The approximation is a four-neighbor
road-cell metric; not a traffic-aware travel-time estimate.

GPU gate prediction: on a U-shaped road, moving along the legal detour increases
potential even while Euclidean distance grows; entering an obstacle gives no
progress bonus. Legacy fixture without a road field retains its old behavior.
Tolerance: 1e-4 score units, exact discrete event counts. Stop on validation error,
non-finite state or failed assertion. Host tests may validate static geometry and
transport only. Radeon/Vulkan, one test thread, build jobs1.

Physical witness: one reference-controlled being, actual city, actual action/body/
physics/taxi transaction, initial placement only, bounded horizon and paid resources.
Success requires a recorded pickup, movement, braking below0.5m/s within4m of the
assigned destination and exactly one paid delivery. Reference results are excluded
from learned-policy campaigns. A failed witness blocks a claim of task feasibility.
Plasticity dose studies remain subsequent to these mechanical gates; a failed or
underpowered learning trial is not a reason to add organs or silently replace a policy.

## First measured witness (exploratory controller check)

`taxi-reference-witness-20260922-a`: RX6750XT RADV, one being, one generation,
16384 ticks (256 s), GPU controller explicitly selected by
`--taxi-reference-witness`, first task from (96,96) to (480,96). This is a straight
route, not a demonstration of turning or learned navigation. One paid delivery
is present by recorded tick6528 at position(479.897,96.199), speed0.03793m/s;
balance42.68 = 25 + 10 + 0.02*384. Zero collision energy and zero red crossings;
two green crossings. Alive with positive energy/water at horizon. Controller
commands traverse the production action mapper with a linear head and zero slow/
hip bias; neural activations are shadow computations and must not be interpreted
as controlling these actions. Map/pose access is privileged reference-controller
knowledge, not the evolved policy's observation contract.

The run found a low-speed dynamics defect: after delivery, throttle0/brake1 still
allowed ~0.0376m/s residual creep and ~9.6m displacement over154s. This is not
hidden behind a successful delivery count. A scenario-only bounded contact-rest
constraint will be checked with real GPU controls before replacing this artifact.
The evidence binary hash is in `/tmp/taxi-witness-binary.sha256`; log is
`/tmp/taxi-reference-witness-a.log`. No binary was replaced during the run.


## Measured correction and navigation evidence — 2026-09-22

The `taxi-reference-witness-20260922-b` run used the same 1-being/16384-tick
command with the corrected static-rest contact constraint. Binary SHA-256:
`d0b0c71f5c881be59de8a8ea0636deb55bc14fdb3147ee2f9530453c9e9f9e72`.
The GPU run completed in 76.8 s wall time. The first observed delivery remains
at tick 6528 (102 s), 0.22456 m from its destination. Recorded speed is zero;
maximum displacement after delivery is now zero over the remaining 154 s
(previous run: 9.57 m). One delivery, zero collision energy, zero red crossings,
alive with positive energy/water. This is privileged reference control on one
straight trip, not evidence of learned navigation, cornering, or generalization.

Evidence and reproduction:

- `python3 scripts/validate_taxi_witness.py /media/llm-data/universe-runs/taxi-reference-witness-20260922-b`
  passed; saved report: `evidence/taxi-navigation-20260922/reference-witness.json`.
- Real-GPU `gpu_static_rest_matrix_real_adapter` passed flag on/off, free rolling,
  insufficient brake torque, zero friction, fast motion and airborne controls
  on AMD Radeon RX 6750 XT (RADV NAVI22).
- `scripts/verify_organism_gpu.sh /media/llm-data/universe-runs/taxi-navigation-gpu-check-20260922`
  passed 9 organism tests and the taxi fixture, plus exact 192-tick observation
  and result equality after resuming the ordinary neural policy at tick 96.
- The road-field fixture passes: the legal U-shaped detour increases progress
  potential while the Euclidean distance increases; entering the obstacle does
  not earn that progress. This affects evolutionary selection. It does not add
  online task reward or demonstrate effective plasticity.
- Reference manifests and UI identify the controller. Neural activity in such a
  run is a shadow computation; the final-intent-to-action connection is marked
  bypassed and must not be presented as evidence of learned causality.

### Actual 3D rendering defect and fixed topology mode

The old line update replaced instanced geometry attributes in-place. Three.js
caches the initial maximum instance count; an initial zero or smaller set could
therefore truncate later geometry regardless of the slider. The update now
replaces and disposes the geometry, invalidating that cache and releasing GPU
buffers. Neuron materials no longer request a vertex-color attribute absent
from SphereGeometry; they use the existing per-instance colors.

The strongest-contribution mode is explicitly time-dependent and capped by a
*total* edge budget. The new **todas disponíveis** mode draws all available
nonzero exported weights and elementwise links. It ignores activation/top-k
filtering and retains its geometry across ticks. Architectural dependencies
without exported matrices remain dashed; the UI does not invent synapses.
Focusing a stage intentionally restricts the view to its adjacent stages;
**enquadrar tudo** clears that restriction.

The real replay `taxi-observatory-full-20260922/2/0` renders **14387/14387**
available connections in all mode. Browser verification exercised the real
canvas, changing the edge budget from 10 to 1000, advancing playback without
changing the all-mode count, and inspecting screenshots with colored neurons
and visible lines. WASD/arrows pan both canvases; Q/E move vertically in 3D;
wheel/+/- zoom; the map supports cursor-anchored zoom, drag and following the
observed being. Re-centering twice is idempotent and user camera positioning
survives tick/resize updates.

Validation commands from `lab-ui`:

```
npm run build
npm exec vitest run src/panels/navigation.test.ts src/panels/brain3d-edges.test.ts src/panels/brain3d-layout.test.ts src/flow/observatory.test.ts
PLAYWRIGHT_BROWSERS_PATH=/media/llm-data/scratch/lab-ui-s5-playwright node scripts/verify-taxi-navigation.mjs
```

The 17 focused UI tests passed, browser checks passed without console/runtime
errors, and screenshots were inspected. The trace contract gate passed and the
API projection tests passed (2 tests). Build retains the existing bundle-size
warning. These are observability and mechanical gates, not a claim that the
current evolved policy delivers passengers. Its near-stationary behavior remains
a measured policy/learning problem; this work does not replace it with the
reference controller.

## Native learned stop after body/action correction — 2026-09-23

The school stop lesson found a separate evolved-policy defect: the resident action
mapper allowed positive throttle and brake simultaneously, while static rest was only
enabled for the privileged witness. The body now resolves antagonistic pedals into one
signed longitudinal command and enables the bounded contact-rest law for every resident
taxi. This changes neither the neural outputs nor the success judge; it makes the body
execute a physically coherent command.

A fresh GPU-only seed-7 run produced one strict stop at generation 15 (128 ticks at
`<=0.3 m/s`, minimum 0.0126 m/s). The saved champion produced 0/16 on validation and
1/16 on test. A 32-generation rotating-start run produced 0/512. The capability is now
real, but generalization is not solved; city/taxi competence must not be inferred from
the single fixed-start success. Evidence lives under `artifacts/school-exclusive-*`.

Two full-city smoke measurements make that boundary explicit. A fresh random population
(8 beings, 2 generations, 640 ticks) completed on the Radeon with zero impacts and zero
pedal overlap, but all fitness values rounded to 35.010 and maximum odometer was 9 m:
selection was blind. Eight clones of the stop champion over 4096 ticks moved 26 m median,
made zero deliveries, spent 80% of alive ticks stationary and accumulated 161 impact
ticks. Applied-pedal overlap remained exactly zero in both runs, so the body correction
works; a stop-only genome does not transfer into a navigation policy. The city must be
fed graduates of a multi-skill, multi-start curriculum and evaluated over multiple starts
per genotype before it can be called functional. Evidence:
`artifacts/final-taxi-smoke-v3` and `artifacts/final-taxi-stop-transfer-v3`.
