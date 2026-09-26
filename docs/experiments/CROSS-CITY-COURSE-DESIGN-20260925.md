# Explicit cross-city school course — design record (2026-09-25)

## Acceptance

`--school-cross-city` must provide four deterministic `Lesson::Delivery` starts.
Each start begins empty on a legal road approach, reaches a pickup station, and
then uses the existing service-v2 delivery judge to reach a distant dropoff.
Every selected station pair must have at least 300 m of host-computed four-neighbor
road distance, at least 200 m of Euclidean displacement, and at least 60 m of
absolute displacement on both X and Z. These constraints require a city turn and
are checked before GPU allocation.

## Scope and compatibility

The legacy `--school-service-course` short straight course remains unchanged.
The new flag is opt-in and rejects short-course mode, prepared starts, non-v4
navigation, and any lesson other than delivery. It uses the existing city station
map and the existing public `ServiceCourse` fields, so station packing, spawn
mapping, lesson metadata, and the GPU judge retain their current contracts.

## Implementation shape

`src/bin/gpu_evolve/cross_city_course.rs` performs deterministic host-only
candidate filtering and selection. It delegates station approach construction to
`organism_school::service_lesson_candidates`, then validates route and displacement
thresholds and emits four `LessonGeometry` values plus the same station/metadata
container used by the short course. `main.rs` adds only flag parsing, compatibility
validation, and course selection. No shader, brain, physics, fitness, or neural
calculation changes are part of this intervention.

## Validation gates

Static source checks and targeted metadata/unit tests are in scope. GPU execution,
full builds, and behavioral claims remain the parent agent's responsibility.

## Root integration and measured v1 rejection

Root fixed the generator's first draft: the approach displacement must use the
assigned distant origin, not the local station of the borrowed spawn. Reused
already-precomputed route fields instead of rebuilding all BFS fields per candidate.
Root also added the explicitly versioned cross-city judge flag: no straight
pre-pickup corridor, with task initialization before the first sensory frame.
The earlier "no shader changes" paragraph describes the delegated scope only.

The v1 frozen GPU evaluation is preserved at
`/media/llm-data/universe-runs/cross-city-powertrain-paired-20260925`:
4 starts x 512 simulated seconds per arm, same native DNA. Both legacy and
powertrain-v2 completed 0 pickups and 0 deliveries. The observed being approached
the first turn at ~19.5m/s and then stopped. This rejects city competence.
Approach lengths were831/705/513/519m; dropoff legs768/768/960/576m.

Course v2 additionally rejects starts whose forward adjacent road cell does not
strictly reduce route distance to the newly assigned pickup. Borrowing a spawn
from another station otherwise permits an accidental initial U-turn. This is an
explicit new task version, not a retroactive change to v1 acceptance. Metadata
records both cells/distances, heading, ordered task geometry and its SHA256.
Only unique starts and station pairs are required; no claim of four compass
headings or coverage of every station.

## Measured cross-city-v2 outcome: rejected

Evidence: `/media/llm-data/universe-runs/cross-city-forward-learning-20260925/`.
`plan.json`, per-arm `command.json`, immutable binary/source, `report.json`,
`critical-analysis.json` and each `strict-audit.json` retain the complete evidence.
Approach legs831/825/633/633m; dropoff legs768/768/960/576m. The task starts empty;
the first observed record has onboard0 and target vector[-252,-579]m. Passenger
ID1 is an assignment, not an accepted boarding event. Actual pickup events remain
absent. Both frozen evaluations ran4 starts x32768ticks (512s), same seed11,
native DNA controller and neutral fixed body. Training ran4 generations of
16DNA x4starts x16384ticks (256s), sequentially on the Radeon6750XT.

| Matched frozen evaluation | Founder | Candidate |
|---|---:|---:|
| Accepted pickups / deliveries | 0 / 0 | 0 / 0 |
| Mean judged approach progress (0..100 points) | 42.83 | 28.44 |
| Worst-start selection score | -834.96 | -704.42 |
| Total judged contact ticks across4 starts | 4 | 1 |
| Observed being0 fraction of samples below0.5m/s | 93.60% | 72.95% |
| Observed being0 median speed | 0m/s | 0.034m/s |
| Observed being0 engine-off fraction | 0% | 0% |

All256 training attempts failed. Best selection scores increased
-578.96,-482.41,-449.82,-449.45 while population mean progress decreased
42.83,40.90,35.28,23.56. Scores across training and evaluation horizons must not
be directly compared: the failure time penalty depends on the declared horizon.
The matched founder/candidate comparison shows reduced contact and improved
fitness without task acquisition. Reject candidate promotion; no city delivery
or body-evolution claim. A body that can move and a changed genome are insufficient.

Founder being0 at t2112 exposes a specific action bottleneck: requested spline
throttle.617807 and brake.653184 resolve to applied brake.035377. The motor
trace incorrectly labeled that cancellation as no suppression. The follow-up
contract and controlled comparison are in `POWERTRAIN-CAUSAL-AUDIT-20260925.md`.
Separately, v4 navigation still has only one-cell road-descent lookahead; a
19m/s car gets no explicit future turn-distance signal. This is a representation
limitation, not proof that anticipation alone fixes the learned policy. Preserve
this failed course and horizon rather than shortening it into a successful replay.

Lab UI task/map presentation now separates assignment from occupancy and shows
the pickup marker before boarding. Absolute delivery and relative pickup/service
positions are distinct; relative markers require a physical frame at the taxi
sample tick. Missing occupancy stays unknown. Frontend tests/typecheck passed;
browser inspection is NOT MEASURED because the user explicitly prohibited it.
