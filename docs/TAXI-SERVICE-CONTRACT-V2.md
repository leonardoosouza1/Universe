# Service curriculum contract v2 — 2026-09-24

Scope: repair service judging and population promotion before claiming learned delivery.
Legacy motor lessons keep their equations. Service v1 remains explicitly selectable.
Map generation and production traffic are subsequent gates, not assumed solved by this change.

## Executed flow

| Phase | Producer | Device | Output / consumer | Frequency |
|---|---|---|---|---|
| Observation | taxi.encode_senses | GPU | packed wave / native brain | tick |
| Action | organism_action.wgsl | GPU | physics input 92 f32; pedals dimensionless | tick |
| Physics | physics GPU | GPU | pose metres, velocity m/s, contacts / body | tick |
| Service | taxi.wgsl post | GPU | body[65,68,72,75], authoritative accepted transitions | tick |
| Judge | school.wgsl | GPU | versioned record / selection, JSON | tick |
| Selection | school_batch.rs | host | genome scores | generation |

Existing evidence: graduate-adaptive-life-s11 being 0 tick 11008: raw brake .322675,
head .275588, applied .275588, no suppression, speed .036736 m/s. This is a
working brake applied at the wrong time, not evidence of a disconnected actuator.

## New contract

- Pickup/dropoff accepted by taxi with matching identity is authoritative. No extra
  hidden dwell requirement may reject a legally accepted service transition.
- Approach/navigate still require the declared dwell. Requests refused by taxi do
  not terminate practice. Refusal state remains observable.
- Small contacts accumulate a bounded physical penalty; death and large route
  deviation remain terminal. Motor lessons retain their strict contact contract.
- Service score separates route progress, best approach readiness, successful task,
  delivery, physical damage and time. No reward is paid for raw pedal activation.
- Time cost is in seconds. Failed/truncated attempts pay the full configured budget
  so early crashing cannot save time cost. Successful attempts pay elapsed time.
- Blocked means no new best task progress or accepted service for
  `max(20s,min(180s,half the episode budget))`, including inside a station.
  The 20-second grace exceeds the city's 16-second red phase. This is a judge
  outcome, not fabricated biological death. An unfinished v2 horizon fails.
- Every judge record, including service requests and accepted-event ticks, freezes
  at the outcome. Later physical/taxi events belong only to lifetime telemetry.
- Promotion requires observed success on every start in the configured pool across
  a complete coverage window. This measures population competence, not frozen-genome
  generalization, which remains a separate held-out evaluation gate.

Acceptance: real GPU fixtures for legal early pickup, harmless rejected exploration,
partial stop credit, collision recovery, blocked timeout and legacy parity. Then
small native runs and a scripted service witness. No learning claim from fixtures.

Sources: CARLA 2.1 https://leaderboard.carla.org/evaluation_v2_1/ and MetaDrive
https://github.com/metadriverse/metadrive/blob/main/metadrive/envs/metadrive_env.py .
Their metrics motivate tests; they do not prove our controller is equally capable.

## Compatibility and exact semantics

Service v2 is the default; `--school-service-legacy` restores service-v1 equations,
termination and legacy generation-mean promotion. Record schema is 4, stride remains
56 f32, with readiness in slot 54 and blocked seconds in slot 55. Old offsets and
motor-lesson equations are unchanged. Score scales across contract versions are not
directly comparable.

Readiness is the best `proximity * slowing` reached after self-propelled movement:
`proximity = clamp(1-max(distance-goal_radius,0)/15m,0,1)` and
`slowing = clamp(1-speed/v_min,0,1)`. Its coefficient is 200, terminal success 500;
valid delivery adds 1000. Damage, death, failure and collision energy have separate
costs. These coefficients are engineering hypotheses, not calibrated guarantees.
Time costs one point per simulated second. The bounded blocked timeout only
exists in this service-school judge; production life termination is not changed.

`--school-advance-gens` retains its CLI name but counts complete coverage windows
under v2. The rate is the minimum success fraction across starts, without a prior
that would turn zero observed successes into positive competence. This can include
different genomes across generations: it is explicitly population competence.
Shared-start runs are diagnostic; adaptive-v2 rejects them rather than pretending
they cover a pool. Oversized pools clamp, but K exceeding the distinct pool fails
explicitly instead of counting a duplicate exam twice. Approach/pickup folds use
physical start/direction identity; old pair-based folds require the legacy flag.

## Historical measured validation (before the 2026-09-25 evidence freeze)

- Build: `nice -n 10 env CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=1 cargo build --bin gpu_evolve`.
- GPU judge: `cargo test --lib gpu::school_tests -- --ignored --test-threads=1`: **30 passed**.
- GPU traffic: `cargo test --lib gpu::traffic::gpu_tests -- --ignored --test-threads=1`: **2 passed**.
  Two earlier filter attempts selected zero tests and are not counted as evidence.
- All numerical fixtures used Vulkan/Radeon, DISPLAY unset, MESA_VK_VERSION_OVERRIDE=1.3,
  RAYON_NUM_THREADS=1, CARGO_BUILD_JOBS=1. Fixtures reject CPU adapters.
- Pickup witness: `service-contract-v2-20260924/witness-pickup`: 1/1 success,
  zero impacts, one accepted pickup. Scripted feasibility only.
- Native paired experiment: `native-check/{legacy,v2}`, same gen198 ancestor,
  seed 11, eight genomes, eight generations, 4096 ticks, same fixed start:
  **0/64 pickups in each arm; zero pickup requests in both**. No learned improvement
  demonstrated. V2 retained attempts until deadline rather than ending on minor
  contact/corridor, but that alone did not recover exploration of the pickup output.
- `native-check/coverage`: four generations, 24 slots; requested pool 24 became nine
  distinct starts. Batches were [0,1,2], [3,4,5], [6,7,8], [0,1,2]; promotion rates
  were null, null, 0, null. Failure bucket counts matched all failed records.
- Final smoke seed 12: four generations, 12 slots, 256 ticks. Same coverage sequence,
  valid schema/contract metadata and per-start counters. Too short for behavior claims.
- Final integrated delivery witness: `final-witness-delivery/school_result_gen0003.json`:
  **1/1 success**, pickup tick 1198, dropoff tick 9872, delivery delta 1,
  zero contact ticks and zero collision joules. Alive at 16000 ticks. Command:
  `target/debug/gpu_evolve --scenario organism-taxi-v1 --taxi-reference-witness --school delivery --school-mode bench --school-horizon 16000 --school-shared-start rotate --gen-offset 3 --beings 1 --gens 1 --ticks 16000 --seed 7101 --driving-diagnostics --out /media/llm-data/universe-runs/service-contract-v2-20260924/final-witness-delivery`
  with the Radeon/Vulkan environment above. This is scripted feasibility, not learning.

Artifacts root: `/media/llm-data/universe-runs/service-contract-v2-20260924/`.
`scripts/validate_service_contract_v2.sh` saves the snapshot binary hash, ancestor
hash, source patch, base commit and per-run commands. The native paired snapshot
predates the final invalid-delivery-credit guard and contact telemetry fix; those
are covered by the final GPU fixtures, not retroactively by the paired experiment.
Final-smoke binary SHA256: `879737ea6bc09e36f9643a2fab2f70bfd158cd22f26f123ddf55c0d42cee7cfe`.
Later CLI help edits only explain coverage-window semantics.

## Remaining project gates

No learned delivery, held-out generalization, city generator, production blocked-life
termination or full legacy race battery is claimed here. The next discriminating
experiment is native pickup-output reachability/exploration from a valid near-target
state, then paired training on diverse target positions. Do not grow the city or
declare a new brain necessary before this gate. No automatic reverse/pilot was added
to evolved policies. Population evaluation still needs a separate frozen-genome gate.
