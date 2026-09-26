# Navigation preview investigation — 2026-09-25

Status: frozen comparison COMPLETE / behavioral FAIL; no city competence claim.

Result: `navigation-preview-paired-20260925/report.json`, four frozen starts per
arm, v4/v5 mean progress 42.829784 / 35.009880 points, zero pickups and deliveries
in both, all four attempts blocked. The v5 cue did not improve this transferred
DNA. No exploratory training was launched. Mechanisms passed in
`navigation-preview-gate-20260925-v1/receipt.json` for that source snapshot;
later source changes require a new receipt.

Runner error: the first actual v5 run was stored under `v4-control` because of a
miswritten tuple. The command and manifest exposed the mistake. Original data
were preserved with `label-correction.json`; the missing true v4 arm was then
run with matched settings. Use `actual_path` and `navigation_observation` in the
paired report, not the original directory name, as arm identity.

Acceptance remains empty departure,633–831m to pickup and576–960m to delivery
on the four fixed cross-city-v2 starts. Actual accepted events, clean survival,
all-start coverage and matching frozen horizons are required. No browser per
user instruction. Numerical tests and trajectories run on the physical Radeon.

Evidence before change: `cross-city-forward-learning-20260925/founder`, being0,
generation0, tick2064: position(113.075,670.560), speed19.145m/s,
route codes[125,20], applied steer.045/brake0. At2112 brake is cancelled by the
simultaneous throttle request; by2208 speed0 following contact. The alternative
brake-priority contract failed the frozen task (mean progress42.83→7.32), so
retain signed-net pedals. Do not claim the remaining failure is wholly sensory.

Executed route: immutable road-distance field -> `taxi.wgsl::route_senses`
-> packed wave178/179 -> KAN -> L1 -> L2/native adaptation -> action spline
-> existing pedal arbitration and steering smoothing -> body -> physicsA/B/C
-> task judge -> sampled trace. v4 picks only the next cardinal cell and has no
explicit future turn cue. At19m/s,3m is roughly0.16s. The field does not guarantee
vehicle footprint clearance or a feasible curvature.

Intervention: opt-in navigation v5 changes only direction channels178/179 to the
body-frame bearing of a preview waypoint reached by walking the same strictly
descending road-distance field on GPU. Preview distance is
`clamp(6 + 1.5*speed_mps,18,48)`m, at most16 road steps. Ties keep the preceding
path direction, initially the vehicle forward. Exact station direction within4m,
route distance channel183, reward, map, pedals and body stay unchanged. A bearing
to a future waypoint is information, not permission to cut across a building;
v5 is not a collision-free trajectory planner. No per-tick host routing.

Mechanism prediction: at a fixed approach pose, v4 has no turn cue while v5 has
the correct signed lateral cue before the junction; near-station, missing-route,
identity, other channel bytes and immutable body/physics contracts remain valid.
The GPU negative control must reject v4 masquerading as v5.

Budget: physical-GPU mechanism gate; matched v4/v5 frozen evaluation4starts x512s;
only if the cue reaches the brain without corruption, one bounded exploratory
training run (up to8generations,16DNA x4starts x256s) and a matched512s frozen
candidate evaluation. One GPU process, single-job builds, min5GiB free before
each run, shared immutable executable per binary hash, no duplicate binary copies
per arm. Mutation/selection and failures are recorded; changed weights or higher
fitness alone do not pass. Training horizon scores are not comparable to the
longer frozen horizon. New observation semantics are versioned; old DNA transfer
is an experiment, not presumed compatibility.
