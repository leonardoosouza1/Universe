# Taxi observatory and city integration

Scope: replace the default race-oriented dashboard with an organism/taxi catalog,
retain historical race artifacts in an explicit archive, expose measured input to
output flow and body feedback, and replace the race-with-cross-streets environment
with a connected city plus GPU traffic signals. No learned driver or anatomical
fly-connectome claim follows from this infrastructure.

## Executed flow and ownership (before changes)

| Phase | Producer | Device / shape | Consumer / cadence |
|---|---|---|---|
| Static geometry | organism_scenario::connect_city | host grid; stationary map | GPU world upload once per generation |
| Sensors | OrganismSensorsGpu::encode | GPU poses/probes/temporal | retina + perception every tick |
| Perception | retina + percept + retina_digest | GPU retina / wave192 | taxi senses then native brain |
| Interoception/task | TaxiGpu::encode_senses | GPU wave160..191 | brain on configured cadence |
| Mind | brain + slow brain | GPU CfC/L1/L2/intents96 | resident action mapper |
| Modulation/action | OrganismActionGpu | GPU commands and motor checkpoints | body pre + physical A/B/C |
| Consequence | body post + taxi post + telemetry | GPU body128 + station stocks | feedback next brain tick; sampled trace |
| Observation | driving_diagnostics + package_run | host sampled records | API + lab-ui, no model stepping |

Trace inspected: organism-visual-final-20260922 gen0 being1, tick32. Native stage
blocks exist; visual completeness is not guaranteed by that fact. Slot375 is
normalized neural maturation. Physical age is body age_seconds. Keep distinct.

## Acceptance

- Taxi-only default catalog; historical data reachable without deleting evidence.
- Macro: generation/population denominators, deliveries, survival, resources,
  wear/collisions and recorded horizon; missing remains missing.
- Micro: actual recorded stages, dimensions and phases; weights only where the
  exporter provides them; raw/modulated/applied commands separately.
- City: connected road network, legal station locations, defined lane directions,
  crossings and reproducible signal schedule. Signal sensing/events execute GPU.
- Contracts and bounded real GPU fixtures; all model numerical tests GPU only.
- Real generated trace opened visually; no synthetic fallback on API failure.

Scientific boundary: a larger city increases difficulty. Preserve reproducible
small fixtures, distinguish integration smoke from learning, and report task
failure honestly. No rule-based braking override is introduced to fake competence.

## Pre-registered integration run (not a learning claim)

Run: taxi-city-observatory-20260922. Eight beings, three generations, 8192 ticks
(128 simulated seconds) per generation, seed0, native brain, inherited body/sensor/
learning/development/topology enabled. Stop on any non-finite state, transport
failure or GPU validation error. Prediction: the new pipeline records all native
stages and traffic wave channels with finite body budgets; completed deliveries
are measured, not a PASS criterion for this integration check. This jointly
changes geometry/traffic and is not a causal A/B learning experiment.

City-v2 contact approximation: discs of radius1.5m; mutually nearest live pairs,
restitution0.05, equal/opposite impulse with mass weighting. Dissipated energy is
shared between pair members and fed into the existing damage kernel. Pair search
is quadratic; this version makes no 4000-body throughput claim. Road blocks are
solid obstacle cells. Traffic signals use14s green,2s yellow,16s red with opposing
axes offset16s. A red crossing debits up to3credits once; no forced braking.

## Final evidence — 2026-09-22

The intermediate `taxi-city-observatory-20260922` run is superseded: it predates
perpendicular signal phase correction and the complete slow-state observation.
Use `taxi-observatory-full-20260922` (8 beings, 3 generations, 8192 ticks = 128 s
per generation, real RX 6750 XT / RADV). No CPU simulation was run.

The final pack has 27 recorded blocks, 256 observations at ticks 32..8192 per
generation. New blocks: slow_input[48], slow_state[64], slow_fatigue[64],
slow_output[3], lifetime_feedback[4], traffic_state[16]. Slow quantization and
hidden affine/ReLU values are UI reconstructions from the recorded state and
exported weights, explicitly distinct from independently recorded observations.
Full neural observation is of selected being 0, not an asserted champion;
physical replay and episode outcome metrics cover all eight beings.

Observed results: zero deliveries in all three generations. Survivors: 8/8,
8/8, 7/8. In generation 2, distance ranges 15.6546..455.54 m; brake-applied ticks
range 12..6489; two red-light crossings cost six credits total. These are an
integration smoke result, not demonstrated taxi competence or learned thirst.

Verification commands and evidence:

- `cargo test --lib gpu_traffic_crossings_and_pair_conservation -- --ignored --test-threads=1`
  on Radeon: PASS; `/tmp/taxi-traffic-final-gpu.log`.
- `cargo test --lib gpu_temporal_fuel_overwrites_spatial_byte -- --ignored --test-threads=1`
  on Radeon: PASS; `/tmp/taxi-fuel-channel-gpu-final.log`.
- `scripts/verify_organism_gpu.sh <fresh-output-directory>`: PASS, full versus
  resumed observations and results match; `/tmp/taxi-observer-contract-final.log`.
- `cargo test --lib grid_contract_has_connected_axes_and_versioned_signals -- --test-threads=1`:
  PASS, host static geometry/phase contract only; `/tmp/taxi-city-geometry-final.log`.
- `python neuromod-lab/scripts/contract_gate.py verify`: PASS;
  `/tmp/taxi-contract-last.log`. Validates transport, not model learning.
- Focused UI Vitest: 19 tests PASS; `/tmp/taxi-ui-final-tests.log`.
- `npm run build`: PASS; `/tmp/taxi-ui-build-final.log`. Existing large-bundle
  warning remains; this is not a frontend performance certification.
- Actual browser inspection on port 8131: perception, full 3D, layer focus,
  needs/actions and playback tested against the final recorded run.

A real rendering defect was found during inspection: retina positions were
included in the instance position array but omitted from the instance reference
array, displacing subsequent neural cells and edges. Both now use the same
layout; a regression test verifies every stage offset including retina pixels.
Focused 3D isolates the chosen stage and direct dependencies; global weighted
edge selection is budgeted per transition so one large matrix cannot hide all
motor connections. The view is a functional schematic, not fly anatomy.

## Interpretation and remaining science

Water is a finite body resource. Channel 161 carries quantized hydration, not a
hardcoded instruction to drink. Intent 5 requests station service; speed,
location, stock, capacity and money determine the actual transfer. The needs
panel shows each of those signal/action/result stages and distinguishes
pre-physics sensation from post-physics body state. This does not establish
causal use of hydration by the policy. Run paired GPU deprivation/restoration
experiments before asserting learned homeostasis.

The deterministic city is currently shared across world seeds; seed variation
alone is not held-out city geometry. Collision handling is the documented
nearest-pair approximation. Long-run learning, city generalization and traffic
scaling remain unproven. Historical race data is preserved in the archive rather
than deleted. The final app is served at `http://127.0.0.1:8131/`.
