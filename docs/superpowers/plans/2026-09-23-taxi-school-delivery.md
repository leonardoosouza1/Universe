# Taxi School Delivery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve native Universe organisms through strict city-driving lessons until frozen genomes complete real taxi deliveries on unseen station pairs, without breaking race, legacy school, replay, trace, checkpoint, or observation contracts.

**Architecture:** Extend the existing GPU-resident school with service lessons and deterministic per-being station-pair tasks. Keep pickup/dropoff acceptance in `taxi.wgsl`, using an explicit taxi mode (`production`, `motor_bench`, `service_bench`) so school and production execute the same predicates. Add batched start evaluation before selection, phase-aware telemetry, and scenario-specific console summaries.

**Tech Stack:** Rust 2024, wgpu/Vulkan, WGSL compute kernels, serde JSON/JSONL, existing `gpu_evolve` CLI and GPU fixture harness.

**Spec:** `docs/superpowers/specs/2026-09-23-taxi-school-delivery-design.md`

## Global Constraints

- Universe only; simulation validation runs on Radeon Vulkan with `RAYON_NUM_THREADS=1` and `CARGO_BUILD_JOBS=1`.
- Existing lesson IDs 1–4, genome layout, checkpoint layout, observation v2/v3 dimensions, race defaults and replay fields remain compatible.
- No host pilot, position injection, hidden steering, per-tick host route search, or success inferred from fitness.
- Strict delivery requires the production delivery counter to increment exactly once.
- New trace fields are optional additions; existing readers must continue to parse old artifacts.
- A failed earlier gate blocks claims about later gates.

## Review Focus

- A normal taxi run with no school must issue passengers; a motor bench must not issue them.
- Repeated pickup/dropoff intent must transition each passenger exactly once and never duplicate fare or delivery count.
- Reversed station pairs must remain in the same fold so train and held-out sets cannot leak.
- A genome evaluated on multiple starts must receive the aggregate of its own replicas only, including failure/death penalties.
- Old replay/checkpoint/observation artifacts must load when optional service-event fields are absent.

---

### Task 1: Lock taxi mode and passenger issuance contracts

**Files:**
- Modify: `src/gpu/taxi.rs`
- Modify: `src/gpu/taxi.wgsl`
- Modify: `src/bin/gpu_evolve/context.rs`
- Modify: `src/bin/gpu_evolve/main.rs`
- Test: `src/gpu/taxi_progress_tests.rs`

**Interfaces:**
- Produces: `TaxiMode::{Production, MotorBench, ServiceBench}` encoded as `Params.mode: u32`.
- Produces: `TaxiGpu::set_mode(&mut self, mode: TaxiMode)`.
- Preserves: production passenger issuance and existing `set_school_mode` callers through a temporary compatibility wrapper removed in this task.

- [ ] **Step 1: Add failing mode regression fixtures**

Add GPU fixtures that reset identical station/body buffers and run one taxi tick:

```rust
assert!(production.world.waiting_passenger_id > 0.0);
assert_eq!(motor_bench.world.waiting_passenger_id, 0.0);
assert_eq!(service_bench.world.waiting_passenger_id, 0.0);
```

- [ ] **Step 2: Run the focused GPU fixtures and confirm failure**

Run with Vulkan: `cargo test --lib taxi_progress_tests::taxi_mode_controls_passenger_issuance -- --ignored --test-threads=1`

Expected before implementation: compile failure because `TaxiMode` does not exist.

- [ ] **Step 3: Replace the boolean school flag with `TaxiMode`**

Define the Rust enum with stable discriminants 0/1/2, upload it in `Params`, and change WGSL checks from `p.school` to `p.mode`. Production issues passengers; both bench modes do not. Birth resource isolation applies to both bench modes.

- [ ] **Step 4: Make every call site explicit**

Normal taxi and held-out evaluation use `Production`; lessons 1–4 use `MotorBench`; lessons 5–8 use `ServiceBench`; school life mode uses `Production`.

- [ ] **Step 5: Run fixture, build and normal-world witness**

Run the focused fixture, `cargo build --bin gpu_evolve`, then a 12,288-tick reference witness. Require passenger issuance and one delivery.

- [ ] **Step 6: Commit**

```bash
git add src/gpu/taxi.rs src/gpu/taxi.wgsl src/gpu/taxi_progress_tests.rs src/bin/gpu_evolve/context.rs src/bin/gpu_evolve/main.rs
git commit -m "fix(taxi): make passenger mode explicit"
```

### Task 2: Define service lessons and station-pair folds

**Files:**
- Modify: `src/gpu/school.rs`
- Modify: `src/organism_school.rs`
- Test: `src/organism_school.rs`
- Modify: `src/bin/gpu_evolve/cli_flags.rs`
- Modify: `src/bin/gpu_evolve/main.rs`

**Interfaces:**
- Produces: `Lesson::{ApproachPassenger=5, Pickup=6, NavigateWithPassenger=7, Delivery=8}`.
- Produces: `Lesson::is_service() -> bool` and `Lesson::starts_onboard() -> bool`.
- Produces: deterministic `service_lesson_candidates(...) -> Vec<LessonGeometry>` with passenger in `target` and destination in `exit`.

- [ ] **Step 1: Add parsing and compatibility tests**

Assert IDs/names for all eight lessons, `Lesson::ALL` ordering, and unchanged IDs 1–4.

- [ ] **Step 2: Add fold-leakage geometry tests**

Build the city and assert every service candidate is reachable, at least 60 road metres long for navigation/delivery, includes a turn, and that `(A,B)` and `(B,A)` map to the same fold key.

- [ ] **Step 3: Run tests and confirm new variants are missing**

Run: `cargo test --lib organism_school::tests -- --test-threads=1`.

- [ ] **Step 4: Implement variants and candidate generation**

Keep junction generation unchanged for IDs 1–4. For service lessons, derive stable unordered station-pair keys, orient the selected pair deterministically, locate a valid lane spawn before the passenger station, and populate `target`, `exit`, `exit_dir`, and `depot_distance_m`.

- [ ] **Step 5: Extend CLI messages and default curriculum**

Accept the four new names and list all eight in help/errors. Preserve explicit old lesson lists.

- [ ] **Step 6: Run geometry/parser tests and commit**

```bash
git add src/gpu/school.rs src/organism_school.rs src/bin/gpu_evolve/cli_flags.rs src/bin/gpu_evolve/main.rs
git commit -m "feat(school): define deterministic taxi service lessons"
```

### Task 3: Add canonical service-bench state transitions

**Files:**
- Modify: `src/gpu/taxi.rs`
- Modify: `src/gpu/taxi.wgsl`
- Modify: `src/gpu/school.rs`
- Modify: `src/gpu/school.wgsl`
- Test: `src/gpu/taxi_progress_tests.rs`
- Test: `src/gpu/school_tests.rs`

**Interfaces:**
- Consumes: `TaxiMode::ServiceBench`, lesson IDs 5–8, `LessonGeometry.target/exit`.
- Produces: body slots 65/66/67/68/70/71/72/75/80/81/90 with the same semantic transitions as production.
- Produces: school record fields for accepted pickup tick, accepted dropoff tick, request counts, refusal code, passenger ID and delivery delta.

- [ ] **Step 1: Add failing taxi predicate fixtures**

Cover pickup accepted, pickup while moving, pickup outside 4 m, repeated pickup, dropoff accepted, dropoff wrong destination, repeated dropoff, and exact one-time fare/delivery increment.

- [ ] **Step 2: Add failing school state fixtures**

Assert service lesson initialization, start-onboard behavior, subgoal switch after pickup, and production-equivalent refusal codes.

- [ ] **Step 3: Run both ignored GPU fixture suites and confirm failures**

Run focused tests with Vulkan and one test thread.

- [ ] **Step 4: Generalize `taxi.wgsl::post` for service bench**

Use the same speed `<0.5`, radius `<=4`, request `>0.5`, passenger ID and one-time delivery guards for production and service bench. Service bench reads deterministic task metadata published by school instead of shared station passengers.

- [ ] **Step 5: Extend school record stride without changing old offsets**

Append new fields after existing slot 39, bump the school schema version, and leave every original offset unchanged.

- [ ] **Step 6: Run fixtures and commit**

```bash
git add src/gpu/taxi.rs src/gpu/taxi.wgsl src/gpu/school.rs src/gpu/school.wgsl src/gpu/taxi_progress_tests.rs src/gpu/school_tests.rs
git commit -m "feat(taxi): share service transitions with school"
```

### Task 4: Implement strict service lesson judge phases

**Files:**
- Modify: `src/gpu/school.wgsl`
- Modify: `src/gpu/school.rs`
- Test: `src/gpu/school_tests.rs`

**Interfaces:**
- Consumes: post-taxi body state and native intent buffer.
- Produces: strict success/failure status, monotonic phase progress and service event evidence for lessons 5–8.

- [ ] **Step 1: Add positive-path fixtures for all four lessons**

Drive synthetic pose/action streams through the real GPU judge and require the declared phase sequence. `Delivery` succeeds only after body delivery count changes from its initial value.

- [ ] **Step 2: Add negative-path fixtures**

Cover contact, death, wrong corridor, no propelled movement, never stopped, pickup outside radius, pickup while moving, destination reached without onboard, dropoff outside radius, wrong task ID, and horizon truncation.

- [ ] **Step 3: Bind the native intent buffer to the school pipeline**

Extend `SchoolGpu::encode` with `intent: &wgpu::Buffer`; update the bind layout and call site after taxi post.

- [ ] **Step 4: Implement phase machines and bounded scoring**

Use road-distance best progress 0–100, pickup base 100, destination best progress 100–200, and delivery bonus 1,000. Keep strict status independent from shaping.

- [ ] **Step 5: Run all school fixtures, including IDs 1–4, and commit**

```bash
git add src/gpu/school.rs src/gpu/school.wgsl src/gpu/school_tests.rs src/bin/gpu_evolve/organism_tick.rs
git commit -m "feat(school): judge pickup and delivery phases on GPU"
```

### Task 5: Evaluate each genome over a deterministic start batch

**Files:**
- Create: `src/bin/gpu_evolve/school_batch.rs`
- Modify: `src/bin/gpu_evolve/main.rs`
- Modify: `src/bin/gpu_evolve/cli_flags.rs`
- Test: `src/bin/gpu_evolve/school_batch.rs`

**Interfaces:**
- Produces: `aggregate_school_batch(replica_records: &[Vec<f32>], genome_count: usize, starts_per_genome: usize) -> Vec<SchoolAggregate>`.
- Adds: `--school-eval-starts K`, default 1 for compatibility; scientific curriculum command uses K=3 or greater.

- [ ] **Step 1: Add pure aggregation tests**

Assert replicas never cross genome boundaries, score is mean minus worst-case failure penalty, success rate/deaths aggregate correctly, and `K=1` reproduces historical score exactly.

- [ ] **Step 2: Add failing integration contract**

For two genomes and three starts, assert six physical slots are initialized, all three replicas share the correct genome hash, and selection receives two aggregate scores.

- [ ] **Step 3: Implement replica expansion at generation start**

Upload each genome K times, assign K deterministic starts from the active fold, simulate expanded slots, aggregate records, and evolve only the original genome count.

- [ ] **Step 4: Export batch evidence**

Add per-start rows plus aggregate mean, worst, successes and deaths to `school_result_genNNNN.json`.

- [ ] **Step 5: Run K=1 parity and K=3 integration tests; commit**

```bash
git add src/bin/gpu_evolve/school_batch.rs src/bin/gpu_evolve/main.rs src/bin/gpu_evolve/cli_flags.rs
git commit -m "feat(evolution): select school genomes across start batches"
```

### Task 6: Replace taxi race summaries with service telemetry

**Files:**
- Modify: `src/bin/gpu_evolve/main.rs`
- Modify: `src/bin/gpu_evolve/driving_diagnostics.rs`
- Modify: `src/bin/gpu_evolve/context.rs`
- Test: `src/bin/gpu_evolve/tests.rs`

**Interfaces:**
- Produces optional observation/replay service events: `passenger_issued`, `pickup_accepted`, `dropoff_accepted`, `delivery_incremented`.
- Produces taxi console summary with phase counts, route progress, requests/refusals, deliveries, impacts, resources and survival.

- [ ] **Step 1: Add formatter/schema tests**

Assert taxi summaries contain no `laps`, `voltas`, `setores` or `[pista]`; race summaries remain unchanged. Assert old observations parse without events and new observations include optional event arrays.

- [ ] **Step 2: Record service transitions at sample boundaries**

Compare authoritative previous/current body fields and emit events once with tick, being, passenger ID, station pair and refusal code.

- [ ] **Step 3: Gate console output by scenario**

Keep race progress output only for `race-v1`; print taxi phase/resource summaries for `organism-taxi-v1`.

- [ ] **Step 4: Run tests and inspect one real replay; commit**

```bash
git add src/bin/gpu_evolve/main.rs src/bin/gpu_evolve/driving_diagnostics.rs src/bin/gpu_evolve/context.rs src/bin/gpu_evolve/tests.rs
git commit -m "feat(taxi): report service phases instead of race laps"
```

### Task 7: Validate lesson witnesses and adaptive progression

**Files:**
- Modify: `src/gpu/taxi_witness.wgsl`
- Modify: `src/bin/gpu_evolve/main.rs`
- Create: `scripts/validate_taxi_school_gpu.sh`
- Create: `docs/TAXI-SCHOOL-DELIVERY-RESULTS-20260923.md`

**Interfaces:**
- Produces reproducible GPU commands and machine-readable result locations.
- Consumes all lesson and batch contracts from Tasks 1–6.

- [ ] **Step 1: Extend the reference witness only for declared service lesson subgoals**

Keep it diagnostic-only and route output through the native action mapper. It may choose intent but cannot alter pose, body state or taxi state directly.

- [ ] **Step 2: Run witnesses for all eight lessons**

Require success, zero shortcut flags, zero pedal overlap and finite traces. Require two production station-pair deliveries.

- [ ] **Step 3: Run two-seed evolution per lesson with K=3 starts**

Stop at the first lesson that fails the training threshold; diagnose and correct that lesson before proceeding. Freeze champions and evaluate all validation/test candidates without mutation.

- [ ] **Step 4: Run the complete adaptive curriculum**

Require logged advancement through all eight lessons and no promotion from shaping alone.

- [ ] **Step 5: Record exact commands, seeds, ticks, success counts and failures; commit**

```bash
git add src/gpu/taxi_witness.wgsl src/bin/gpu_evolve/main.rs scripts/validate_taxi_school_gpu.sh docs/TAXI-SCHOOL-DELIVERY-RESULTS-20260923.md
git commit -m "test(taxi): validate delivery curriculum on GPU"
```

### Task 8: Prove frozen-graduate production delivery

**Files:**
- Create: `scripts/validate_taxi_graduates_gpu.sh`
- Modify: `docs/TAXI-SCHOOL-DELIVERY-RESULTS-20260923.md`

**Interfaces:**
- Consumes: two independently evolved frozen genomes from Task 7.
- Produces: final production artifacts with strict delivery event chains.

- [ ] **Step 1: Clone each graduate without mutation into unseen production tasks**

Run at least 16 beings per graduate for a horizon proven sufficient by the reference witness. Use production passengers, traffic contacts, resources and taxi fitness.

- [ ] **Step 2: Verify event chains from artifacts**

For each successful being require ordered `passenger_issued → pickup_accepted → dropoff_accepted → delivery_incremented`, alive final state, finite state and no duplicate delivery ID.

- [ ] **Step 3: Enforce the final acceptance gate**

Require at least one delivery for each independent graduate on unseen station pairs. A zero-delivery graduate fails the project gate.

- [ ] **Step 4: Commit validation artifacts/scripts**

```bash
git add scripts/validate_taxi_graduates_gpu.sh docs/TAXI-SCHOOL-DELIVERY-RESULTS-20260923.md
git commit -m "test(taxi): prove evolved production deliveries"
```

### Task 9: Run the full legacy regression matrix

**Files:**
- Modify: `docs/TAXI-SCHOOL-DELIVERY-RESULTS-20260923.md`

**Interfaces:**
- Consumes the completed implementation.
- Produces the final compatibility verdict.

- [ ] **Step 1: Run shader and focused Rust contracts**

Run WGSL validation, organism school geometry, school GPU fixtures, taxi progress/service fixtures, action arbitration and replay parsing.

- [ ] **Step 2: Run legacy GPU smokes**

Run `race-v1`, each original school lesson, explicit four-lesson curriculum, observation v2/v3, old checkpoint resume and old replay export/import.

- [ ] **Step 3: Run new GPU smokes**

Run each service lesson, full eight-lesson curriculum, normal taxi production and frozen graduate transfer.

- [ ] **Step 4: Inspect generated metadata and replay**

Assert scenario-specific summaries, unchanged old field meanings, optional new events, all brain stages and no non-finite values.

- [ ] **Step 5: Record command evidence and commit**

```bash
git add docs/TAXI-SCHOOL-DELIVERY-RESULTS-20260923.md
git commit -m "docs(taxi): record delivery and legacy validation"
```

