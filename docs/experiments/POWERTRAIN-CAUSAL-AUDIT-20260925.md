# Taxi powertrain investigation — 2026-09-25

Status: OPEN. This document does not certify learned city deliveries.

## Observable acceptance

Correct gear labels; resident powertrain v2 births in first gear. High-gear,
low-coupled-RPM load can stall the engine. Engine-off drive torque is zero.
Restart requires an explicit neural start request, a stationary eligible gear,
one uninterrupted second and prepaid energy. Reverse requires near-stop selection
and actually moves backwards. Shifts interrupt torque. Resource depletion cannot
produce powered motion. Old race/checkpoint semantics must not change silently.
A learned cross-city delivery is a separate behavioral gate, still failing.

## Evidence before edits

Active path: organism_tick -> native brain -> organism_action -> organism pre ->
physics_a/b/c -> organism post -> orientation feedback -> driving_diagnostics.
Run `school-48m-four-starts-confirmed-20260925`, generation 0, being 2:
t16 raw gear3, RPM2516, speed0.423m/s, applied throttle0.2458/brake0;
t48 raw gear3, RPM800, speed0.372m/s, throttle0/brake0.2423.
Raw code3 means second, but UI displayed it as third. Shared constructor starts
in second; resident initialization copied it. Low-speed downshift required
speed>3.5m/s. GearState.cooldown never wrote physics.shift_timer.
The old stall predicate requires full clutch coupling and raw RPM<550, but its
coupling formula implies wheel RPM>=1425: load stall is unreachable.
No paid, explicit starter action exists. Old restart zeroes wheel angular speeds.

## Bounded implementation contract

Explicit `--organism-powertrain-v2`; old profile preserved. Physics flag13.
Action10th active row (index9) = start request; action contract version changes.
Physics input83: previous padding becomes starter authorization under flag13.
Action overwrites it each tick with request; authoritative body pre pays 1500W*dt
only if off, speed<0.5m/s, gear R/N/first, alive and basal reserve affordable, no shift in progress,
then emits 0/1 authorization. Physics advances cranking only on authorization.
No new body slots: 31/61/62 already belong to taxi tasks. Body post retains all
other resource ownership. Starter energy is dissipated outside engine thermal
compartments in this bounded model (no battery/alternator claim).
First/reverse retain automatic launch clutch; higher gears are coupled, with
stall below550RPM outside shift interruption. Automatic transmission downshifts
at low speed; neural shifts respect safety and cooldown. Off engine gear is held
while cranking. Insufficient torque for a hill alone is not a stall condition. Zero usable
engine capacity (including energy starvation) explicitly switches the engine off;
the diagnostic engine_off flag covers both starvation and mechanical load stall.
Channels78..80 are explicitly repurposed only in the new profile from unused
city racing-line horizons to engine-off, crank-progress and shift-progress.
This changes observation semantics and requires matched evaluation, not silent
reuse of old checkpoint claims. Startup energy at birth is not a restart event.
Checkpoint version and configuration fingerprint must reject incompatible state.

## Gates

- Mechanism GPU: PASS, gate-v3 below (22 fixtures + 7 negative controls).
- Original replay gear display: NOT MEASURED after patch.
- Frozen school retention: NOT MEASURED with new profile.
- Learned cross-city delivery: FAIL in previous evaluations, not solved by design.
- Body/brain evolution benefit: NOT MEASURED.

## Explicit school selection penalty and urban approach contract

Under powertrain-v2, service-v2 selection subtracts 5 per running->off transition
and one point per 1500J of paid starter energy. These are declared experimental
weights, not physical constants. Body energy payment remains authoritative; the
judge observes it and cannot trigger or pay a start. Slots38/39 of school record
(previously unassigned, separate from body state) store counts and paid joules;
record schema5. Counters freeze with the judged outcome. Global ecology fitness
is unchanged; that regime still needs a separate experiment before any claim.

Cross-city service flag6 permits turns during pre-pickup navigation: the legacy
straight-approach corridor remains for old school contexts. Taxi physics, wall
contact penalties, timeout, passenger identity, stop and service gates remain.
Static route length alone proves task geometry, not learned completion.

Initial GPU results: four physical fixtures passed in gate-v1, plus all17 previous
fixtures and seven negative controls. Gate-v1 was still FAIL because the runtime
recorder JSON macro exceeded recursion depth. The body diagnostic additions were
split into explicit assignments; focused runtime compilation then passed. A new
full gate is required after the city and scoring additions; do not reuse v1.

## Final mechanism receipt and behavioral separation

`scripts/check_native_flow_gpu.py --out
/media/llm-data/universe-runs/powertrain-cross-city-gate-20260925-v3`
passed all22 physical Radeon fixtures and7 intended negative controls. Receipt:
`/media/llm-data/universe-runs/powertrain-cross-city-gate-20260925-v3/receipt.json`.
Source SHA256: `69c282131ac5bc61f2237b4651710ae984595eca32e835064a22efd462b409af`.
The five new fixtures cover load stall/no powered motion with engine off; paid
start/refusal/reset/reverse; birth/gear/shift safety; engine sensor through native
L1/L2 to start intent; and cross-city judge/event-cost integration. The host-only
schema assertion `gpu::school::tests::schema_v5_stride_and_slots_are_frozen` also
passed; it does not run a numerical simulation.

The matched city-v1 comparison at
`/media/llm-data/universe-runs/cross-city-powertrain-paired-20260925/report.json`
is **FAIL for learned behavior**: legacy0/4 pickups and0/4 deliveries; v2 also0/4
and0/4. The observed being's engine stayed on in both arms; all four judge records
had zero engine-off events and zero paid starter energy. Load stall was a genuine
mechanical bug, but it does not explain this DNA stopping at the corner.
Both runs preserve their own immutable binary, source snapshot and command.

Automatic gearbox was used in the behavioral runs. Therefore learned neural
gear choice, learned restart, school retention with the new sensory profile,
body evolution benefit and ecology-mode ranking remain NOT MEASURED. Starter
payment is tested, not a battery/alternator subsystem. New functionality is
explicitly opt-in, not silently applied to old recordings or default profiles.

Presentation checks: `npx vitest run src/panels/taxi-telemetry.test.ts
src/panels/body-telemetry.test.ts --maxWorkers=1` in Lab UI passed9 tests.
Gear codes are decoded instead of displaying the raw index. Taxi assignment and
target/service coordinates stay visible before boarding; unrecorded occupancy
is no longer displayed as an empty car. New starter telemetry exists in the raw
Universe trace; no claim that the entire Lab UI pack already exposes it.
Visual reinspection remains NOT MEASURED: the user explicitly requested no more
browser opening. No failed city replay was promoted or presented as success.

## Follow-up: observed throttle cancellation at the corner

Same source-bound run: `cross-city-forward-learning-20260925/founder`, being0,
generation0, tick2112. Route codes178/179=[113,57]; L2 effective motor=[.621,.250,.784];
raw throttle/steer/brake=[.815715,0,.874418]; spline=[.617807,0,.653184].
The exclusive signed-net resolver subtracts throttle from brake: applied brake
is .035377, not .653184. Speed before physics is16.818m/s. At tick2160 the car is
near(89.75,662.12), and by2208 it is stationary. The raw trace called this
`brake_suppression=none`, masking the attenuation. This is observed interference,
not proof that preserving brake alone teaches steering.

Bounded intervention: opt-in `--organism-brake-priority`, resident GPU action flag32.
If positive post-modulation brake exceeds1e-6 (numerical-zero tolerance), applied
throttle is0 and the full brake request survives. Otherwise retain requested
throttle and set brake0. No route, speed, steering or reward heuristic. Requires
exclusive low-speed pedals; signed-brake and legacy-pedal combinations rejected.
Old defaults and arithmetic remain unchanged. Manifest `pedal_arbitration` names
the new contract; frozen checkpoint arguments and implementation hash include it.
Trace now reports `throttle_cancellation` under the old subtractive resolver.

Before the next behavioral comparison: run23 physical-GPU fixtures and8 negative
controls; an injected missing-priority flag must fail the new fixture. Then hold
founder DNA, four cross-city-v2 starts, seed11, neutral body and512s horizon fixed,
change only this action flag. Require authoritative pickup then delivery and
contact-free survival for success. Compare movement, contact, progress and
applied pedals regardless of score. Positive brake authority is the predicted
mechanism; improved navigation is an unproven downstream hypothesis. Do not
promote a controller that merely brakes forever. Budget: two sequential frozen
evaluations,4 slots each,32768 ticks, no training or browser.

### Brake-priority comparison completed: behavioral hypothesis rejected

Mechanism gate `brake-priority-gate-20260925-v1/receipt.json` passed23 fixtures
and detected all8 injected faults; source hash
`96b139a6919aa538d02e14f513ce70c278b980514704b11ac4a2d3963fd55880`.
Both executed arms and strict audits are retained at
`/media/llm-data/universe-runs/cross-city-brake-priority-paired-20260925/`.
Commands, frozen input DNA, source snapshot, binary, `report.json` and
`critical-analysis.json` are authoritative. Each ran512 simulated seconds on
four starts. Both had0 pickups and0 deliveries, and start0 was empty at its first
record. Control exactly reproduced the prior judged distances/progress.

| Metric | Signed-net control | Brake priority |
|---|---:|---:|
| Mean judged approach progress | 42.83 | 7.32 |
| Per-start distance (m) | 261/278/485/313 | 52/50/58/54 |
| Total judged contact ticks | 4 | 0 |
| Failure reasons | 4 blocked | 3 blocked,1 death |
| Observed0 samples with positive throttle AND brake heads | 82/2048 | 1873/2048 |
| Observed0 samples reporting throttle cancellation | 82 | 0 |

Inference: preserving braking authority is mechanically correct for that
explicit contract but **not sufficient** to transfer the frozen policy. The
closed-loop trajectory changes the sensory distribution; do not compare the
two sample counts as if they were identical neural inputs. The existing policy
frequently produces conflicting requests in the priority arm and makes little
progress. Reject it as a replacement/default; it remains an explicit experimental
flag only. Do not silently rescale pedals or train more generations solely to
obtain a prettier replay. The next independent question is anticipation and
learned pedal/steering coordination under a declared, fixed actuator contract,
using the original long-task acceptance and a narrowly instrumented turn fixture.

Final frontend checks:9 focused telemetry/marker tests and `npm run typecheck`
passed. Terminal HTTP checks returned200 for `/api/db/batteries` and the two
modified Vue modules on port8131. These are service/build checks, **not visual
inspection**. No browser was opened after the user's prohibition.
