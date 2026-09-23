# Taxi school and end-to-end delivery design

Date: 2026-09-23  
Project: Universe  
Status: proposed for implementation

## Objective

Make `organism-taxi-v1` evolve organisms that can complete real taxi deliveries while
preserving every existing Universe mode and artifact contract. A run counts as working
only when a frozen evolved genome completes strict deliveries on unseen station pairs in
the production taxi world. Fitness improvement, proximity, movement, or a scripted
witness alone are insufficient.

## Existing evidence

- The production taxi world previously inherited `school_bench=true` without a lesson,
  disabling passengers. The corrected call activates isolation only for an actual bench
  lesson.
- The reference GPU controller completes one production delivery in 12,288 ticks.
- Evolved `move` champions generalize to 48/64 unseen city-lane starts.
- `stop` is the first unstable learned skill: 0/64 held-out successes.
- `pickup` and `dropoff` are native actions but do not exist as school lessons.
- Taxi runs still print race-oriented lap summaries that do not describe taxi progress.

## Compatibility rule

The feature extends existing contracts; it does not repurpose them.

- `race-v1` physics, fitness, laps, portals, replay and CLI defaults remain unchanged.
- Existing school lessons retain IDs 1–4, geometry, success criteria, record fields and
  scores.
- New lessons receive IDs after 4. Old result files remain readable.
- `organism-taxi-v1` without school keeps passengers, resources, traffic contacts,
  production fitness and native action arbitration enabled.
- Bench mode keeps tasks per being and disables shared-world interference.
- Life mode keeps production passengers and contacts and uses taxi fitness.
- Existing checkpoints and genomes remain loadable. No genome layout change is needed.
- Existing observation v2/v3 channel meanings remain unchanged.
- Replay and observation schemas gain optional event fields; existing fields are not
  renamed or removed.

Every compatibility item is an automated regression gate.

## Lesson sequence

The default adaptive curriculum becomes:

```text
move
→ stop
→ turn_near
→ turn_far
→ approach_passenger
→ pickup
→ navigate_with_passenger
→ delivery
```

The old four-lesson curriculum remains selectable explicitly with
`--school-lessons move,stop,turn_near,turn_far`.

### 1. `approach_passenger` — ID 5

Initial state: no passenger onboard; one deterministic passenger target per being.

Success requires all of:

- target approached through the road network;
- distance at most 4 m;
- speed at most 0.3 m/s for 64 consecutive ticks;
- alive, no contact, no wrong-way corridor escape;
- at least 16 propelled ticks before the stop.

This teaches perception, navigation and braking without requiring a discrete service
action.

### 2. `pickup` — ID 6

Same initial state as `approach_passenger`.

Success requires the strict approach conditions followed by native intent channel 6
greater than 0.5 while speed is below 0.5 m/s and distance is at most 4 m. The judge
performs a per-being task-phase transition equivalent to production pickup and exposes
the resulting onboard/destination state through the existing taxi observation channels.

Requests while moving, outside the radius, without a passenger, or for the wrong task
produce explicit failure/refusal codes. Proximity alone never succeeds.

### 3. `navigate_with_passenger` — ID 7

Initial state: the being starts with a deterministic passenger onboard and a destination
station chosen from a disjoint station-pair fold.

Success requires reaching the destination through the city, stopping within 4 m at no
more than 0.3 m/s for 64 consecutive ticks, remaining alive, and avoiding contact. No
dropoff intent is required yet.

### 4. `delivery` — ID 8

Initial state: no passenger onboard; deterministic passenger and destination station.

The episode has strict phases:

```text
approach passenger
→ stop
→ pickup action accepted
→ navigate while onboard
→ stop at destination
→ dropoff action accepted
→ delivery counter increments exactly once
```

Success is written only after the delivery counter transition. Reaching the destination,
requesting dropoff from the wrong place, or clearing onboard without incrementing the
delivery counter cannot succeed.

## Geometry and folds

Motor lessons continue using junction approaches. Service lessons use deterministic
station pairs and route-distance fields from the production city.

- Candidate pairs must be mutually reachable and at least 60 road metres apart.
- At least one turn is required for `navigate_with_passenger` and `delivery`.
- Candidate index modulo three defines train, validation and test folds.
- The same station pair must never appear in more than one fold in reversed order.
- Shared fixed/rotating starts remain available for diagnosis.
- Scientific evaluation clones one frozen champion across every candidate of a fold;
  population slots are not compared on different tasks during selection evidence.

## GPU state and kernel flow

The school remains GPU resident. No per-tick host controller or host route search is
introduced.

1. Host builds immutable lesson geometry and station-pair metadata once per generation.
2. `school.wgsl` publishes the current subgoal through existing body/taxi slots.
3. The native brain produces the existing 96-intent ABI.
4. `organism_action.wgsl` applies throttle, steering and braking unchanged.
5. Production physics and body kernels advance unchanged.
6. Taxi service rules or a shared service-rule kernel validate pickup/dropoff.
7. The school judge reads the post-transition state and records progress/outcome.

Pickup/dropoff predicates must have one canonical implementation shared by production
taxi and school. Duplicated constants or subtly different distance/speed checks are not
acceptable. If WGSL composition cannot share source directly, the predicates and
constants move into a generated shared WGSL fragment validated against both pipelines.

## Fitness and anti-shortcut rules

All partial progress is monotonic and bounded.

- Phase approach score: 0–100 from best road distance closed.
- Accepted pickup: phase base becomes 100.
- Destination approach: 100–200 from best road distance closed while the same passenger
  remains onboard.
- Accepted delivery: event bonus 1,000, paid once.
- Contacts, death, wrong target, invalid service requests and corridor escape fail the
  lesson.
- Standing still outside a valid goal earns nothing.
- Oscillation cannot increase best-distance progress.
- A passenger/task key change resets phase memory.
- A delivery event is authoritative; score cannot synthesize it.

The strict success gate is separate from shaping. Shaping may rank incomplete attempts
but cannot mark them successful.

## Stop remediation

Service lessons depend on a robust stop. Before the curriculum advances beyond `stop`:

- training rotates across multiple train starts rather than one memorized pose;
- each genome is evaluated over a small deterministic batch of starts before selection;
- aggregate fitness uses mean score with a worst-case penalty;
- competence requires the strict stop criterion on at least 50% of the batch for two
  consecutive generations;
- frozen champions must pass validation and test starts.

This batching removes the current confound where genomes are compared on different
geometries or memorize one orientation.

## Observability

Taxi telemetry replaces race summaries when `scenario=organism-taxi-v1`.

Required population and selected-being metrics:

- current lesson and phase;
- passenger/task ID and station pair;
- road distance to current subgoal and best distance;
- raw and accepted pickup/dropoff requests;
- refusal code and reason;
- onboard transition tick;
- delivery transition tick;
- throttle, brake, steering, speed and pedal overlap;
- contact energy, impacts, integrity and alive state;
- energy, water, material, engine temperature and wear;
- per-phase duration and final outcome.

Legacy `laps`, `sectors` and race-track progress remain for `race-v1` but are suppressed
from taxi console summaries. Replay files retain their existing fields and receive
optional service events.

## Regression and acceptance matrix

### Contract tests

- All WGSL modules parse and validate.
- Existing school fixtures remain bit-for-bit valid for lessons 1–4.
- New positive and negative fixtures cover every phase and refusal/failure path.
- Production taxi issues passengers without school and does not issue them in bench.
- Pickup/dropoff predicates match between production and school.
- Delivery increments once under repeated requests.
- No throttle/brake overlap after action arbitration.
- No NaN or infinite state in traces or result files.

### GPU witness gates

- Reference witness passes every old and new lesson.
- Reference witness completes a production delivery on at least two station pairs.
- Bench tasks are independent across beings.

### Evolution gates

For two seeds per lesson:

- training competence reaches the declared adaptive threshold;
- frozen champion succeeds on at least 50% of validation candidates;
- frozen champion succeeds on at least 50% of test candidates;
- no shortcut flags, pilot, position injection, or scripted motor override are active.

### Final production gate

At least two independently evolved frozen genomes are placed in the normal taxi city.
Across unseen station pairs, each must complete at least one strict delivery while
remaining alive. The run must record passenger issuance, pickup, onboard navigation,
dropoff and delivery transitions. Zero-delivery runs fail regardless of fitness.

### Legacy gates

- `race-v1` smoke and replay remain valid.
- Four original school lessons and old explicit curriculum remain valid.
- Old genome/checkpoint resume remains valid.
- Observation v2 and v3 dimensions and units remain valid.
- Existing replay readers tolerate new optional event fields.
- CLI help and manifests accurately describe active scenario semantics.

## Delivery order

1. Passenger-mode regression and initialization fix.
2. Canonical shared service predicates.
3. Service lesson geometry and fold contracts.
4. GPU judge phases and fixtures.
5. CLI, curriculum, manifests and telemetry.
6. Stop batching/generalization.
7. Per-lesson witness and evolution batteries.
8. Frozen-graduate production transfer.
9. Race, school, checkpoint, trace and replay regression suite.

No later stage may hide or waive a failed earlier gate.
