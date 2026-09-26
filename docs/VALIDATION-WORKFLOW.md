# Universe validation workflow

**Status:** mandatory working procedure for every agent. The research runner
enforces the GPU mechanism gate below; behavioral/visual gates still require
the responsible agent. This is not a repository-wide CI enforcement claim.
**Scope:** Universe simulation, training, exports, and consumers in Lab UI/API.
**Owner's acceptance:** a learned being navigates the taxi map, reaches passengers,
stops, picks them up and delivers them. Movement alone is a prerequisite, not completion.

Use this entry point rather than reading every historical report. It supplements
the [scientific handbook](../../neuromod-lab/docs/HANDBOOK-ENGENHARIA-CIENTIFICA.md)
and [flow handbook](../../neuromod-lab/docs/HANDBOOK-ANALISE-DE-FLUXO.md).
Record each investigation using [this template](templates/BEHAVIOR-INVESTIGATION.md).
The [September 25 example](experiments/REPLAY-CAUSAL-AUDIT-20260925.md) deliberately
contains failed and unmeasured gates. A truthful failed record is useful evidence.

## 1. Start with the failure, not a new implementation

1. Read the user's acceptance and last failed experiment. Inspect `git status`
   before edits; identify concurrent work and preserve it.
2. Identify the **actual displayed run**, generation, being/lineage, controller,
   URL, pack source and recording interval. A champion filename does not prove
   that its DNA belongs to the selected being.
3. Record binary path and SHA, `--build-provenance`, source SHA/dirty state, genome
   SHA, exact command, seed, environment, body, observation version, action mode,
   neural flags, time step, horizon and sample cadence. Unknown is not default.
4. Write observable pass/fail criteria before the next run. Separate a short
   mechanical fixture, a school task, full production service and generalization.
   A 12 m straight task does not validate city navigation.
5. Choose one failing interval and a useful comparison interval/control. Include
   the beginning, onset of failure and final state; never choose only the best moment.

Do not claim the old system regressed without matching its original configuration.
Reusing old weights with differently encoded sensors or pedals changes the experiment.

## 2. Where to look

Paths below are relative to `Universe/`. They are navigation aids verified on
2026-09-25, not permission to skip checking the current dispatch and schemas.

| Question | Authoritative starting point |
|---|---|
| Which flags/defaults and scenario actually ran? | `src/bin/gpu_evolve/cli_flags.rs`, `main.rs`; run command and per-generation manifest |
| What runs during an organism tick? | `src/bin/gpu_evolve/organism_tick.rs`, `GpuCtx::tick_organism` |
| Which city, spawn, lane and station? | `src/organism_city.rs`, `src/organism_scenario.rs`; exported geometry |
| Which retina and sensory channel semantics? | `src/gpu/retina*.wgsl`, `perception*.wgsl`, `src/organism_channels.rs`; taxi/traffic sensor writers |
| Which destination and route hints? | `src/gpu/taxi.rs`, `taxi.wgsl::senses/route_senses`, `taxi_routes.rs`; observation v2/v3/v4 |
| What reaches L1/L2 and the motor head? | `src/gpu/brain/mod.rs`, `brain.wgsl`; `driving.meta.json` and `observation.jsonl.stages` |
| What modifies the command? | `organism_tick.rs`, `src/gpu/organism_action.wgsl`; slow/hip buffers, pedal flags, smoothing and gearbox |
| What limits body capacity/resources? | `src/gpu/organism.rs`, `organism.wgsl::encode_pre/encode_post`; birth genes and expressed state |
| What physically moves/brakes/collides? | `src/gpu/physics.rs::encode_step` → **`physics_a.wgsl`, `physics_b.wgsl`, `physics_c.wgsl`**; orientation/contact passes |
| Which service transitions are real? | `src/gpu/taxi.wgsl::post`, body task fields, `service_events`, authoritative delivery counter |
| Which non-champion DNA can be investigated? | `diagnostic_candidates/genNNNN.json` with at most 19 genotype checkpoints (three aggregate roles plus up to 16 clean per-start deliveries), explicit omitted-start coverage and hashes when driving diagnostics are enabled; observed school outcomes are not competence certificates |
| Which score selects parents? | `src/organism_scenario.rs::fitness`, `school.wgsl`, `school_batch.rs::selection_scores`, `main.rs` reproduction call, `selection_genNNNN.json` |
| What is actually recorded? | `src/bin/gpu_evolve/driving_diagnostics.rs`, `organism_tick.rs` sampled readback; recorder metadata |
| What is packaged/displayed? | `../neuromod-lab/src/lab_api/trace/producer.py`, `produce.py`, schema; `../lab-ui/src/stores/run.ts`, `flow/adapters.ts`, `panels/` |
| Where are evidence and binaries? | `/media/llm-data/universe-runs/<unique-run>/`; concise conclusions in `docs/experiments/` |

**Historical trap:** `physics.wgsl` is not the current three-pass physics program.
Follow `include_str!`, pipeline entry points, bindings and dispatches. Similarly,
the historical racing tick in `context.rs` does not establish the organism path.
Old README commands and comments may describe obsolete paths; inspect the registry.

## 3. Map the executed loop before attributing a cause

Current organism scheduling outline (verify flags and order for the measured binary):

```text
resident pose/world → retina → perception/digest → body/taxi/traffic senses
→ optional feedback → fast brain when scheduled → slow gather/brain
→ copy intent and applied modulation → action mapper/gearbox
→ body capability expression → physics A/B/C → contacts/orientation/telemetry
→ traffic post → body post → taxi services → optional school judge
→ feedback to next tick's resident input → sampled recorder/readback
→ generation results → actual selection vector → reproduction
```

For each affected edge record: producer symbol, consumer symbol, actual buffer,
shape, dtype, stride/offset, unit, range, coordinate frame/sign, phase/tick,
frequency, initial/reset value, clamp/gate, owner and version. Record submitted
passes and readback cadence. Do not add per-tick CPU round trips for diagnosis.

Distinguish inherited DNA, expressed runtime parameters, executed values and
displayed values. A unit wrapper catches incompatible declared units; it does not
prove a producer labeled its number correctly. Check the equation at that producer.

## 4. Inspect a decision end to end

Use the same `(run, generation, being, tick)` through the following table. Record
actual values, not “connected” or “looks fine”. If a field is missing, mark it and
add bounded sampled instrumentation at its authoritative owner before blaming it.

| Boundary | Required observations and questions |
|---|---|
| World → sensors | Pose/heading, wall sides/distances, target identity and position, route validity, retina. Does the reported geometry agree with the map? Are axes, metres and near/far conventions correct? |
| Sensors → brain | Wave channel codes, decoded values and version; active input dimensions; CfC state; fresh versus reused neural tick. Does the intended input actually reach the consumed channel? |
| Inside brain | `l1_matmul`, `l1_effective`, `l1_to_l2`, `l2_matmul`, `l2_effective`, `l2_std`, `l2_adapted`, `final_intent`. Measure where magnitude/sign changes, zeros appear or values saturate; inspect normalization, fatigue, gates and activation. |
| Brain → action | Raw intent, spline output, **applied** slow/hip values, pedal arbitration, low-speed gate, steering filter, reverse/gear request, final command. A computed but disabled slow output is not applied inhibition. |
| Action → body | Throttle/brake, requested versus available torque, maturity, fatigue, wear, integrity, energy, water, body/oil temperature, repair and resource gates. Is a command suppressed, unaffordable, dead-body overridden or physically weak? |
| Body → physics | Actual torque, RPM, gear/ratio, clutch, TCS/ABS, wheel slip, grip, contact/impact, velocity and displacement. Do not assume throttle 0.2 implies acceleration. Compare delivered drive/brake and resisting forces. |
| Motion → task | Progress toward a stable task identity, accepted pickup/dropoff, passenger identity, stops, refusals and next task. Distance away from the target is not useful progress. |
| Result → evolution | Per-being components, actual parent-selection vector, replica aggregation, preserved controls, mutation provenance. Verify the best displayed score is the score reproduction consumes. |
| Trace → UI | Same identity, tick and phase across source, pack and panel. Missing is distinct from zero. Inherited synapses are not acquired weights; a blinking layer is not causal validation. |

Retina/wave can be pre-physics while body is post-physics in the **same record**.
Use `sensor_pose` and metadata; do not diagnose a sensor bug by comparing it to the
next pose. Sampled traces cannot establish every-tick event frequency. Read
`driving.csv` for its accumulated tick counters and use the recorded sample cadence.

## 5. Critical loop: observe → discriminate → change → verify

1. **Observe:** state the first boundary where actual and expected diverge.
   Separate observed fact, static finding, inference and hypothesis.
2. **Discriminate:** select the cheapest test that could refute the suspected
   mechanism. Keep DNA, body, map, seed and horizon fixed; vary one factor. If more
   change, label a combined intervention, not an isolated causal result.
3. **Predict:** declare the expected intermediate signal change and downstream
   behavior, numerical tolerances and bounded run/time/disk budget. A new target
   metric chosen after seeing results is exploratory.
4. **Change:** patch the owner of the demonstrated failure. Preserve unrelated
   behavior and the legacy path where required. Changes to sensor semantics,
   layouts or action interpretation require an explicit compatibility/version
   decision. Do not compensate for a broken contract in the UI or fitness.
5. **Verify:** run the specific GPU fixture and original failing scenario, then
   relevant adjacent/regression cases. Inspect actual outcomes, not exit status.
6. **Close the loop:** compare prediction to result. A failed prediction means
   revisit the cause; do not stack another speculative patch on it. After repeated
   inconclusive attempts, stop scaling the battery and instrument the missing edge.
   Preserve unsuccessful results and update the next discriminating question.

Output suppression is measurable without proving it is the sole cause of failure.
Disabling suppression and obtaining faster wall collisions would establish neither
navigation nor a good fix. A scripted pilot proves feasibility only; never silently
substitute it for a learned controller in the deliverable.

## 6. Validation has separate gates

| Gate | Necessary evidence | What it does NOT prove |
|---|---|---|
| Build/schema | Correct target built; shader compiled through actual GPU path; layouts/units/flags checked | Body moves or policy learns |
| Mechanism | Small real-GPU fixture isolates changed equation/buffer; invalid/reset/batch cases relevant to change | Learned skill or generalization |
| Behavior | Original failure reproduced and corrected with exact DNA/config; task events and temporal metrics agree | Improvement across training seeds |
| Evolution | Founder vs final under matched frozen evaluation; population distributions, first/middle/last and multiple seeds appropriate to claim | Morphology/memory causally helped merely because genes changed |
| Transfer | Declared held-out starts/maps/tasks and failures; retained driving/stopping competence | Every city or arbitrary unseen complexity |
| UI | Actual published URL loaded, identity verified, played and scrubbed; values compared to source | Simulation correctness by appearance alone |

Use `PASS`, `FAIL`, `NOT MEASURED` or `NOT APPLICABLE` with a reason per gate.
Pass only the claim tested. A docs-only change needs link/content checks, not a
training run; it cannot be described as fixing behavior. No universal “100% ready”.

### Executable mechanism gate (mandatory before research execution)

Run `nice -n 10 python3 scripts/check_native_flow_gpu.py --out
/media/llm-data/universe-runs/<new-gate-directory>` using the real Radeon device.
It builds with one job, executes twenty-three exact GPU tests sequentially, then injects
eight deliberate failures: cancelled brake priority, unreachable engine stall, disconnected L2, the legacy tire-force transform,
explicit wheel integration, instantaneous TCS recovery, corner-indexed retina and lost newborn physics profile.
Each must fail at its intended assertion. A zero-test invocation is an error.
The suite also checks taxi stock/payment/refusals, route-channel semantics,
contact masks and sustained hot/cold physiology at the production time step.
Retina fixtures explicitly test `--retina-road-view` and `--retina-centered-grid`
geometry; they do not certify uncorrected legacy wall distances. The stable-wheel
profile also passes native brake/coast/reset, four-heading rotation and a bounded
timestep-refinement fixture. TCS must preserve a non-slipping pedal request, persist
through a runtime checkpoint and reset only the newborn slot at birth.
Newborns must inherit the active numerical profile while retaining their own body genes. This is not
universal convergence or learned competence.

Only a successful gate builds/certifies the runtime and writes a passed
`receipt.json` bound to the source, binary and gate hashes. Pass that file to
`scripts/run_research_battery.py --execute --gate-receipt <receipt> ...`.
Missing, failed, incomplete or stale receipts stop execution before creating a
run. Any numerical source change requires a new receipt. Direct binary commands
are not intercepted; agents must check the same receipt before diagnostic runs.
The receipt certifies controlled mechanisms, **not learned driving/delivery**.

### Minimum behavioral measurements

- Duration in seconds and ticks; alive/dead time separated, with both reported.
- Median speed and time in declared speed bands; fixed-window summaries across
  the whole episode. Peaks are secondary. Explain excluded periods (e.g. service).
- Idle/stuck intervals, driven distance, net displacement, spatial coverage and
  progress toward the current task. Loops or wall sliding must not masquerade as routes.
- Raw/applied throttle, brake overlap/suppression, steering, RPM/gear, available
  and delivered torque, contacts and collision energy.
- Energy/water/temperature/wear/fatigue trajectories and any threshold crossing;
  no invented zero for unrecorded parameters.
- Accepted service events per passenger and continuous life, refusals and final
  task outcome. Four replicas finishing once are not four consecutive deliveries.
- Distribution across evaluated beings/starts and failures. One selected neural
  trace does not describe the population. Report observation coverage explicitly.

Acceptance thresholds belong to the experiment, with reasons. Do not maximize
speed indiscriminately: stopping for passengers, lights, safety and resources is
part of the task. Require useful progress and safe completion, not perpetual throttle.

### Whole-generation and learned-attempt audit

Run `python3 scripts/audit_taxi_learning.py <completed-run> --require-all-starts`
when acceptance requires a genotype to pass every evaluated start. It verifies
requested generation count, contiguous identities, all body horizons, replica
mapping, the selection vector, frozen event ordering and authoritative delivery
counters. Scripted witnesses and unknown controllers are rejected. Exit 2 means
invalid/incomplete evidence; exit 1 means behavioral acceptance failed; exit 0
means the declared recorded-attempt criterion passed, not city generalization.
Without `--require-all-starts`, one clean recorded full trip is enough, so that
weaker check must never be reported as a four-direction or whole-city pass.
Body control is not inferred from a flag name: `--fixed-body` freezes vehicle
capacity expression only; organism metabolism still evolves. Use
`--research-body fixed` for an explicitly neutral whole-body research control,
and verify `body_genNNNN.json` including all 16 `organism_genes`. The evidence
auditor rejects nonzero genes under a claimed neutral-metabolism control.

Schema regression tests: `python3 scripts/test_audit_taxi_learning.py` (host JSON
validation only, no numerical simulation).

Navigation v4 is station-indexed and is rejected for motor lessons or curricula
containing arbitrary nonstation subgoals. Use explicit v3 for those lessons;
never silently reinterpret old checkpoints under another observation version.
Reference-witness metadata must declare `intent_producer=taxi_witness`,
`learning_evidence=false`, linear motor head and privileged inputs. Recorded
native neural stages in that arm do not control the vehicle.

### Mandatory visual review for a replay claim

Open the exact supplied URL using the browser tool. Verify the life/genome/tick,
play at 1x against the recorded time base, then scrub beginning, failure onset,
turn/stop/service and final state. Compare map/retina/brain/action/body values at
selected ticks with raw data. Check the same car is selected and explain missing
traces for other cars. Record URL, ticks, screenshots when useful, discrepancies
and console/load errors. If the viewer is unavailable, mark visual review unmeasured;
continue other checks without asserting that the replay works.

## 7. Resource limits and useful tools

Numerical brain/body/physics validation is **GPU only** by owner instruction.
Reject software adapters. Run one bounded GPU job at a time; no concurrent heavy
battery. Read/aggregate existing recordings on the host rather than resimulating.
Build only the target needed, with `CARGO_BUILD_JOBS=1` and `nice -n 10`.
Do not run historical CPU or mixed validators, including `gpu_validate` or the
CPU reconstruction in `scripts/validate_brain_trace.py`, under this restriction.

Existing starting points (inspect their scope before running):

| Tool | Scope/limit |
|---|---|
| `scripts/run_research_battery.py --help` | Plans by default; `--execute` launches GPU jobs. Frozen evaluation, body/controller/observation contracts; school success is not production acceptance. |
| `scripts/verify_organism_gpu.sh` and `scripts/README-ORGANISM-GPU.md` | GPU continuation/checkpoint consistency; not learning evidence. |
| `scripts/validate_taxi_school_gpu.sh` | School GPU fixtures; select only relevant cases after inspecting script. |
| `scripts/validate_taxi_witness.py` | Reads recorded scripted-witness transitions; explicitly not learning evidence. |
| `scripts/publish_research_run.py` | Copies completed-run metadata; existing scope is short-course research, not a generic city-success certifier. |

The real AMD runtime uses Vulkan, the Radeon ICD and
`MESA_VK_VERSION_OVERRIDE=1.3`, with `DISPLAY` unset. Confirm the adapter in the
actual log. Sandbox GPU failure must not trigger CPU fallback. Store large
artifacts under `/media/llm-data/universe-runs/`, outside the source tree; use a
unique directory and immutable binary per experiment. Do not overwrite evidence.

## 8. Handoff and completion

The responsible agent checks subagent diffs, commands and actual artifacts.
Every handoff/completion contains:

1. User acceptance and status of each applicable gate.
2. Demonstrated cause versus unresolved hypotheses, with specific tick values.
3. Changed files, affected contracts and compatibility decision.
4. Exact commands, exit results, binary/DNA/source provenance and artifact paths.
5. Before/after behavioral numbers, failures, visual review and limitations.
6. The next bounded test if unresolved; no “continue training” without a reason.

Keep one investigation record current. Supersede claims explicitly instead of
creating contradictory “final” reports. Do not tell the owner “ready” when only
the code, a short benchmark or the panel has passed.


### Curriculum retention is a promotion gate

A new lesson score cannot replace the prior controller by itself. Freeze each
candidate and re-evaluate the same genotype on the previous accepted contexts
and the new context. Record actual start identities, both service events for
full trips, contact, survival, horizons, source/binary/DNA hashes and controller
type. Preserve the accepted ancestor on any regression. A competence archive
stores DNA; it does not prove that one brain retains all archived skills.

The September 25 native-pickup candidate improved to 3/4 native pickups but
regressed from 4/4 to 2/4 original full trips. It was rejected as a replacement.
The same candidate failed all native full trips at a separately declared 512 s
budget, while the scripted feasibility witness passed all four without contact.
Neither pickup counts nor the scripted result can certify a learned delivery.
See `experiments/REPLAY-CAUSAL-AUDIT-20260925.md` for exact artifacts.
