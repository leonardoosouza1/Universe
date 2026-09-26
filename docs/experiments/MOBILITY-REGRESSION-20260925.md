# Mobility regression audit

The owner's acceptance is a being navigating the map and performing useful
service. Four replicas finishing an isolated12m trip do not meet that acceptance.
The most recent training used fresh genomes and did not preserve a known moving
controller. No further success claim follows from that short-course score.

## Predeclared diagnostic

On the exact current GPU binary, compare the recorded historical city mover,
historical race champion, and current short-service champion. Freeze DNA, one
genome with three common move-lesson starts,4096 ticks, native brain, v4 route
observations, neutral body, slow modulation disabled. No mutation, controller
override or CPU numerical simulation. This is a controller substitution test,
not an attempt to reproduce historical racing conditions or prove city service.

Question: does the current taxi body/action path permit the old moving
controllers to accelerate, while the newly selected controller crawls?
Engineering discriminator: an old controller reaching >5m/s before its judged
move outcome falsifies a universal low-speed ceiling in this runtime. Otherwise
inspect actuator/gear/body data before blaming learning. The criterion is not a
statistical MDE: these are deterministic genome interventions, not independent
training seeds. Each run stops at4096 ticks or180 wall seconds.

Executed flow: resident sensors/retina → percept192 → native brain → motor
spline → exclusive throttle/brake arbitration → body capability expression →
GPU physics → taxi → school judge. The production taxi path allows braking below
2m/s; the legacy race path suppressed it there. Changes in sensor semantics,
pedal arbitration and body expression mean an old genome's failure alone cannot
identify a physical regression. Use the trace to discriminate those boundaries.

Current trace before edits: short-service slot0 tick432 has raw intent
[-0.000,-.289,.551] and applied [0,-.203,.458], speed.581m/s. The low speed is
accompanied by active braking; it is not evidence of a hard speed cap. With
`--no-slow`, the runtime clears the action mapper's slow buffer, although the
observer still records the independently computed slow-network output. Do not
attribute the latter's nonzero values to applied modulation.

## First diagnostic results

All three frozen interventions completed on the real Radeon GPU with byte-identical
input/exported DNA. Artifact root: `mobility-regression-20260925`.

| Controller | Peak speed per start (m/s) | Move outcomes |
|---|---|---|
| Historical city mover | 5.58,13.21,11.65 | success, corridor failure, success |
| Historical race champion | 2.86,1.45,.19 | success, success, horizon unfinished |
| Short-service champion | .76,3.51,4.60 | horizon unfinished, success, success |

The current body is capable of13.21m/s (47.56km/h), falsifying a universal
low-speed ceiling. This does not mean its steering is correct: the fastest start
failed its corridor. The historical racer does not automatically transfer through
changed taxi observations/body/pedal semantics. The newly selected service genome
is not a replacement for a competent navigation baseline.

## Next discriminating pair, before execution

Run the recorded city mover and short-service champion in the actual production
taxi loop, without `--school` or artificial stations. One frozen being per run,
same seed0, v4 observations, neutral body and no slow modulation;8192 ticks,
180 wall-second timeout. Record actual path, speed, collisions, passenger events
and final state. This is an exploratory pair, not training or proof of generality.
The move fixture above tests only short directional acceleration. This pair tests
whether that capacity turns into movement across the map. Neither distance alone
nor being alive counts as learned delivery.

## Production pair results

Both8192-tick runs completed with unchanged DNA, one being and no school.
Artifact root: `/media/llm-data/universe-runs/mobility-production-20260925`.

| Controller | Distance | Sampled peak | Deliveries | Collision energy | Final integrity |
|---|---:|---:|---:|---:|---:|
| Historical city mover | 781.01m | 30.86m/s (111.1km/h) | 0 | 575236.6J | .9425 |
| Short-service champion | 317.08m | 14.25m/s (51.3km/h) | 0 | 0J | 1 |

This updates the diagnosis: the new genome can also accelerate and travel on
the actual map. Its12m replay was an inadequate demonstration, not evidence of
a permanent inability to move. It remains a failed taxi policy: zero deliveries
and task progress only3.33 points. The old mover travels farther but crashes;
neither is an acceptable service controller.

The short-service trace at16 has heading(0,1), position(93,114), passenger delta
(3,-18): the nearest pickup begins behind it. At1024 it has moved to
(93.13,128.63), farther from that pickup. At4096 it is at(93.17,174.51), still
farther away. The v4 route hints are present in wave178/179 (approximately0,+127),
so a disconnected direction writer is not established. A rearward pickup and
turnaround were never trained by the four forward-only short-course starts.
This is a concrete out-of-training condition requiring a navigation test, not
more repetitions of the easy forward course.

Static integration gap: `organism_scenario::fitness` still computes
`1000*deliveries + task_progress + money + .001*age + 10*alive
-100*(1-integrity)-100*(1-alive)`. `school.wgsl::close` instead subtracts time
and its service judge terminates stalled attempts. These are separate paths;
school-only fixes did not repair the production objective. The small positive
age term is not, by itself, proven to cause the observed behavior. The production
trace has no pickup/delivery, so no claim of useful navigation is warranted.

Corrective priorities: retain the old moving policies as frozen regression
controls; validate continuous production tasks rather than promote school
scores; train forward/side/rear goals and actual junction routes; explicitly
version and align the production task objective/timeout before another broad
training battery. Do not silently replace the learned controller with a pilot.

## Confirmed objective-routing defect and patch

`main.rs` computed production taxi fitness in life mode, but then unconditionally
preferred `school_aggregates` for `selection_fitness`. That latter vector controls
champion choice and reproduction. Thus adding an allegedly observational school
judge could change which DNA reproduced. This bug does not explain the previous
12m bench result or the no-school production pair: neither used life observation.

The patch makes a school aggregate authoritative only in explicit bench mode.
Life/no-school selection keeps the production vector. School aggregate metadata
now separates `school_score` from actual `selection_score` and declares its
source. A new per-generation `selection_genNNNN.json` records the exact scores
sent to selection, making the contract inspectable rather than merely asserted.
An opposite-ranking regression test covers life, bench and absent-observer cases.

Before execution: validate on the GPU with two generations of four native beings
for each of life and bench, same seed11 and1280-tick horizon. The acceptance here
is objective routing, not learned behavior: life selection must equal recorded
production fitness and bench selection must equal recorded school aggregate
scores. School and production scores must actually differ, or the fixture is
uninformative. File comparisons are host metadata checks; model/physics runs
remain exclusively GPU. Do not label this fix a completed navigation solution.

GPU integration result: PASS on both generations of both modes. In life gen0,
selection received `[35.020004,35.020004,34.823078,35.020004]` (production), while
the recorded school scores were approximately `[0,0,-199.974,0]`. In bench gen0
the latter school vector was selected. Generation1 also matched the declared
objective in both modes; the emitted aggregate selection metadata agreed. The
body/brain simulations used the Radeon Vulkan adapter. Artifact:
`/media/llm-data/universe-runs/life-selection-contract-20260925/report.json`.
Build completed with one job; `git diff --check` passed. The added host-only
ranking unit fixture was not run; the real GPU integration exercised its runtime
consumer instead. No CPU brain/physics simulation was used.

## Rejected replay: temporal audit, not peak-speed acceptance

The owner rejected the production replay too. Exploratory analysis of its
recorded samples confirms why: the short-service policy's mean speed in successive
32-second windows was approximately 0.97, 0.94, 1.55 and 6.50 m/s. Its corresponding
sampled path lengths were 30.63, 29.66, 49.45 and 206.24 m. Thus a 51 km/h peak and
317 m total concealed a first minute spent crawling. The historical mover had the
opposite failure: mean speed 21.39 m/s in the first window, then 2.48, 0.14 and
0.10 m/s, ending against a wall. Neither replay demonstrates sustained navigation.

### Frozen observation-compatibility experiment, before execution

Question: did reinterpreting the navigation inputs contribute to this behavior?
The historical city genome's recorded command used default v2; the short-course
genome was trained with v4. Compare the historical genome under v2/v4, and the
short-course genome under v3/v4. Only observation version changes within each
pair. Use the current frozen binary, one being, seed0, 8192 ticks (128 s), neutral
body, no slow modulation, no school, no evolution or motor overrides. Run four
jobs sequentially on Radeon, with a 180-second wall timeout per job. Record exact
commands, source/binary/DNA hashes, temporal speed statistics, contacts and actual
deliveries. All beings and deaths remain included.

Exploratory engineering discriminator: a >=2x change in time above 2 m/s, or an
actual delivery transition, identifies a materially sensitive input contract in
this one frozen trajectory. This is not a statistical MDE or generalization
claim: there is one genome/environment per pair. A positive result cannot prove
that routing alone fixes taxi service; a negative result rules out treating a
simple version switch as the demonstrated fix. Do not publish another replay as
functional navigation based on path length or peak speed.

**Not executed:** this planned version comparison was suspended before launch
after the owner's request for a complete same-being decision trace. Continue from
[the causal audit](REPLAY-CAUSAL-AUDIT-20260925.md), which records the observed
output attenuation, body state, unresolved mechanical boundary and failed
acceptance. Do not interpret the experiment proposal above as new results.
