# Rejected city replay: worked investigation record

**Status: OPEN — behavior failed; no behavioral fix claimed.**
Responsible agent: Codex. Date: 2026-09-25.
This record demonstrates the [required workflow](../VALIDATION-WORKFLOW.md) using
the actual failure. It does not certify the entire brain or assign sole causality.

## Latest follow-up: long empty-pickup task still fails

The user's acceptance now explicitly requires a distant pickup AND distant
delivery, starting empty. The new opt-in `--school-cross-city` course has
approach legs633–831m and delivery legs576–960m; the short48m replay below does
not meet that acceptance. Four generations (256 attempts) produced no pickups
or deliveries. Matched512s founder/candidate evaluations also failed0/4 each;
fitness improved while approach progress decreased. Candidate promotion rejected.

See [cross-city evidence](CROSS-CITY-COURSE-DESIGN-20260925.md) and
[powertrain/action investigation](POWERTRAIN-CAUSAL-AUDIT-20260925.md).
The current mechanism gate is `brake-priority-gate-20260925-v1/receipt.json`
under the runs root:23 physical-GPU fixtures and8 detected faults, PASS.
It does not certify learning. A real corner trace showed throttle cancellation
reducing a.653 brake request to.035. An explicit brake-priority profile passed
the mechanism fixture but failed frozen navigation: mean progress42.83→7.32,
0 pickups/deliveries in both arms. It was not made the default. No browser
inspection: the user requested not to open it.

## Previous verified school state (before long-task acceptance)

**The city-learning objective remains open.** A learned native genotype now passes
all four original 48 m full trips, frozen and contact-free. It is preserved at
`stable-centered-backbone-thaw-20260925/evolve_best_gen0011.bin` (runs root:
`/media/llm-data/universe-runs`). Its exact frozen repeat is
`stable-four-start-confirmation-20260925/learned-frozen`. This is a school result.

| Gate | Result | Artifact below runs root |
|---|---|---|
| Numerical mechanisms | 17 GPU fixtures + 6 detected faults; runtime built | `native-flow-gate-20260925-v12/receipt.json` |
| Frozen learned 48 m full trips | 4/4 clean | `stable-four-start-confirmation-20260925/report.json` |
| Frozen 96 m transfer | 2/4 clean; strict gate failed | `stable-route-transfer-20260925/longer-96m/strict-all-start-audit.json` |
| New native-pickup candidate | 3/4 pickups; not deliveries | `native-pickup-transfer-confirmation-20260925/report.json` |
| That candidate's old-skill retention | 2/4; replacement rejected | same report, `retention-48m` |
| Native full trips, 512 s, learned candidate | 0/4; failed | same report, `native-full-512s` |
| Native full trips, 512 s, scripted witness | 4/4, zero contact; feasibility only | same report, `scripted-native-512s` |

The scripted witness travels 815.01/814.96/1008.42/623.00 m before delivery at
ticks 13396/13395/14891/9810. It uses privileged task/route observations and is
**not neural learning evidence**. These original city tasks are executable under
the corrected physics; the current learned controller has not acquired them.
The native trial's 512 s budget is declared separately, not a retroactive change
to the earlier failed 64 s tests.

API and UI proxy both returned HTTP 200 for `/api/db/batteries`. The verified
school replay is available at
`http://127.0.0.1:8131/run/school-48m-four-starts-confirmed-20260925/0/2?view=champion`.
Browser checks covered actual movement, source-matched speed and service state,
1x time progression and visible schematic L2 connections. The combined layout
is still cramped at a small viewport; no UI source change was made.

Next acceptance boundary: a *single* frozen genotype must retain the four
original full trips and succeed on declared native approach/route starts before
promotion. Testing only the newly trained lesson loses old skills, as the
measured 4/4 -> 2/4 regression shows. Extend training contexts and judge both;
do not replace the saved competent ancestor, enable body evolution as a rescue,
raise deadlines without declaring a new condition, or silently install the
scripted witness as the learned controller. Body/metabolic evolution remains
unproven in these fixed, neutral-body diagnostics.

## Acceptance and identity

The owner requires useful navigation across the taxi map and deliveries. The
published replay spends most of its duration crawling and completes no delivery.
Peak speed and total distance were inadequate evidence and should not have been
presented as a satisfactory response to that requirement.

- Source: `/media/llm-data/universe-runs/mobility-production-20260925/short-service-champion/`.
- Published copy: `/media/llm-data/universe-runs/taxi-city-audit-zero-deliveries-20260925/`.
- URL: `http://127.0.0.1:8131/run/taxi-city-audit-zero-deliveries-20260925/0/0?view=champion`.
- Generation0, being0, 8192 ticks at64 Hz,128 s; sampled neural records every16 ticks.
- Exact command: source `command.json`; build and hashes: parent experiment's
  provenance and prior [mobility audit](MOBILITY-REGRESSION-20260925.md).
- Frozen short-course DNA; native controller, v4 navigation, fixed body,
  `--no-slow`, seed0, no school. This is evaluation, not new training.
- Source recording separates pre-physics retina/perception, post-brain stages and
  post-physics body. The comparison below uses a fresh neural dispatch at tick1024.

## Executed/observed boundary map

| Boundary | Owner and consumed data | Evidence/phase |
|---|---|---|
| City → sensory wave | `tick_organism`, perception, taxi senses/route pass;192 signed sensor codes | `sensor_pose`, `wave`, versionv4; pre-physics |
| Wave → brain | `brain.wgsl`, CfC96, L1 192→96, quantized L1, L2 96→48 | recorded `stages`; post-brain, same being |
| L2 → intent | std normalization, gain2, output fatigue beta0.85, activation | `l2_effective`, `l2_std`, `l2_adapted`, `final_intent`; dimensions from metadata |
| Intent → action | `organism_action.wgsl`; spline, applied modulation, exclusive pedals and steer filter | `motor_trace`, `body.applied_motor`; pre-physics command |
| Action → body/physics | `organism.wgsl::encode_pre`, `physics.rs::encode_step` → A/B/C | effective torque/brake capacity, gear, RPM, actual drive torque; body post-physics |
| Physics → task | body post, taxi post; generation selection separate | `taxi`, `service_events`, `organism_result_gen0000.json` |
| Trace → pack → UI | recorder → existing API packer → Lab UI | Original URL was loaded previously; detailed visual verification of these exact ticks is **not yet recorded** |

Static shader inspection must use `physics_a/b/c.wgsl`. The monolithic
`physics.wgsl` was consulted during this investigation, but its similar formulas
cannot establish the active mechanical behavior. No physical cause is concluded
from that legacy file.

### Active-physics discrepancy found by the new GPU flow fixture

On the real Radeon, two neutral bodies with fixed native-brain weights received
opposite visual/CfC steering stimuli and **zero throttle**. Starting at5 m/s,
after128 ticks they reached10.0234 and10.0241 m/s. The full pipeline fixture
initially checked steering direction but omitted passivity. It now explicitly
requires <=5.1 m/s for those coasting cases (0.1 m/s numerical allowance), before
any fix. This is a failure, not a better driving result.

The executed `physics_b.wgsl` computes wheel velocity components
`u = vf*cos(a) + vl*sin(a)`, `v = -vf*sin(a) + vl*cos(a)` and lateral magnitude
`Fy` with the same sign as `v`. Dissipative wheel force is therefore `[Fx,-Fy]`.
Inverse rotation must give body force `[Fx*cos(a)+Fy*sin(a),
Fx*sin(a)-Fy*cos(a)]`. The current front-wheel force instead changes both sine
terms' signs, allowing a steering drag term to add forward work. Rear wheels
have zero steering and do not have these terms.

Prediction before correction: consistent inverse rotation removes the coasting
acceleration and preserves opposite yaw responses. Test the existing legacy
branch separately; do not silently rewrite historical race physics. A correction
of this mechanical contract does not certify the learned taxi controller.

## Observed example: tick1024 (16 s)

| Boundary | Recorded values | Supported conclusion |
|---|---|---|
| Pose/target | Sensor position(93.1347,128.6140), forward(.00398,.99999); task delta(2.8671,-32.6324) in the post-physics task record | Task lies behind the current forward direction; phases differ slightly and are not silently treated as identical |
| v4 wave | Channels178/179 = `[1,127]` | A local route direction reaches the wave; this is not the direct target vector and does not prove the hint is a drivable trajectory |
| Throttle L2 | `l2_effective[0]=.283653`, std`.603607`, gain2 → pre-fatigue magnitude≈`.93986`; recorded adapted`.136372` | A large reduction occurs at the recorded output adaptation boundary |
| Raw → head → pedal | Intent`.135533`, head`.120902`, applied throttle`.120902`, brake0 | Small acceleration command reaches the body; the brake is not held at this sampled tick |
| Steering | Raw`-.072854`, head`-.064989`, filtered/applied`-.160887` | Filtered output includes previous state; it is not an unexplained current-tick mismatch |
| Mechanical output | Gear2 (first forward gear), RPM1657.03, available engine torque490.73 Nm, delivered wheel-side torque54.63 Nm, speed1.18449 m/s | Available torque differs from delivered wheel torque. TCS/clutch/slip and ratio contributions still need an active-path measurement |
| Body | Water10 kg, energy4.889 MJ, body21.10°C, oil26.16°C, integrity1, wear`.000034`, fatigue`.04110` | Depleted water, overheating or damage are not observed at this failing interval |

The recorded `slow_output` is the computed slow network output. In this command
`--no-slow` clears the action buffer; nonzero displayed slow values do not prove
that the motor was inhibited by the slow brain. This observation/applied-value
distinction must be made explicit in any panel interpretation.

## Whole-interval analysis

The first64 s contain256 sampled records. Using the recorded runtime gain/std,
mean absolute pre-adaptation versus adapted values are:

| Motor row | Before adaptation | After adaptation | Nonzero input reduced to exactly zero |
|---|---:|---:|---:|
| Throttle | .545303 | .138785 |36.33% of samples |
| Steering | .247934 | .067518 |45.31% |
| Brake | .367309 | .091529 |42.97% |

These are summaries of GPU-recorded intermediates, not a CPU brain simulation.
They are sampled fractions, not every-tick fractions. The governing code tracks
absolute pre-activation magnitude and subtracts beta times that running magnitude.
With constant input and beta`.85`, its steady-state surviving magnitude is15%.
This establishes attenuation; it does **not** prove that removing it yields a
competent policy. The short-course genome was selected with that adaptation on.

Successive32 s windows have mean speeds approximately`.97, .94, 1.55, 6.50 m/s`.
The run ends with0 deliveries,0 collision energy and317.08 m accumulated distance.
Most distance occurred near the end; the initial pickup was not completed. The
historical mover control is also inadequate: fast initially, then wall-pinned.

Source inspection/aggregation command pattern (read-only; no simulation):

```bash
python3 - <<'PY'
import json, statistics
from pathlib import Path
p = Path('/media/llm-data/universe-runs/mobility-production-20260925/short-service-champion')
meta = json.loads((p / 'driving.meta.json').read_text())
gain = meta['brain_runtime']['output_gain']
with (p / 'observation.jsonl').open() as stream:
    rows = [json.loads(line) for line in stream]
rows = [r for r in rows if r['generation'] == 0 and r['being'] == 0
        and r['tick'] <= 4096 and r['neural_dispatch']]
for j, name in enumerate(('throttle', 'steering', 'brake')):
    before = [r['stages']['l2_effective'][j] / r['stages']['l2_std'] * gain for r in rows]
    after = [r['stages']['l2_adapted'][j] for r in rows]
    print(name, len(rows), statistics.mean(map(abs, before)),
          statistics.mean(map(abs, after)),
          sum(a != 0 and b == 0 for a, b in zip(before, after)) / len(rows))
PY
```

## Hypothesis, unmeasured edges and next test

Output adaptation may impair sustained motor control. Missing disambiguation:
frozen matched GPU evaluation with only adaptation changed, before/after layer
attenuation and action metrics, sustained useful progress and collisions. A faster
crash fails behavioral acceptance. Also inspect actual drivetrain limiting at low
speed before treating weak neural command as the whole explanation.

Do not immediately train another forward-only tiny task or rewrite the entire
brain. No new behavioral intervention has been executed in this record. A planned
observation-version comparison was suspended before launch to follow the owner's
request for end-to-end diagnosis and an explicit workflow.

| Gate | Status |
|---|---|
| Reproduction from recorded artifacts | PASS for the reported crawling/zero delivery |
| Recorded layer attenuation identified | PASS for the sampled stages, not full numerical brain validation |
| Isolated cause and GPU correction | NOT MEASURED |
| Learned production service | FAIL |
| Evolution/generalization improvement | NOT MEASURED |
| Detailed same-tick visual verification | NOT MEASURED in this audit |

The documentation change closes a missing operating procedure. It does not fix
or certify the taxi agent, and this failure remains open for the next correction.

## Next iteration: output fatigue intervention (registered before execution)

Freeze the same short-service genome, current binary, seed0, fixed body, v4
navigation, no slow modulation, one being, no school,8192 ticks. Compare only
`--out-beta 0.85` against `--out-beta 0`. Two sequential Radeon runs,180 s wall
timeout each, full episode metrics and sampled stages. No new training.

Prediction: zero beta removes the discrepancy between scaled/gained L2 and
`l2_adapted` (absolute error <=1e-5). Exploratory behavioral discriminator:
doubling sampled time above2 m/s would implicate output attenuation in crawling;
faster collisions or zero delivery still fail service acceptance. This is a
deterministic intervention, not a statistical learning claim. Record all failures
and body state. Observe before changing any default.


## Executed: output adaptation intervention

Artifact: `/media/llm-data/universe-runs/output-fatigue-ablation-20260925/report.json`.
Same frozen DNA, 128 s, native controller, one body. Beta .85: 317.08 m,
0 deliveries, no collision; sampled time above 2 m/s 28.52%. Beta 0: 588.49 m,
0 deliveries, 494,328 J collision; sampled time above 2 m/s **20.12%**.
After the first 32 seconds the beta-0 body remains stopped for all three remaining
windows, with brake approximately .60. Removing adaptation alone FAILED the
behavioral discriminator. No default was changed on this evidence.

## Executed: native chain contract and passive steering correction

The new GPU fixture starts four identical bodies at 5 m/s. Controlled, fixed
weights test the real KAN/CfC/L1/quantization/L2/motor-head/action/body/physics path.
It does not substitute a pilot for the learned policy. The old steering force
transform accelerated coasting, steering cars from 5 to approximately 10 m/s
in two seconds without propulsion. RED log: `/tmp/universe-passivity-red.log`.

For wheel velocity coordinates `cf = vf*cos + vl*sin`, `cl = -vf*sin + vl*cos`,
the tire force opposing lateral slip must transform back as:
`F_forward = fll*cos + flal*sin`, `F_lateral = fll*sin - flal*cos`.
The old front-wheel formula reversed the cross-term signs. Both front-wheel
blocks now use the power-conjugate inverse transform when physics flag bit10
is enabled. Taxi enables it; historical race keeps the old flag clear.
`--organism-legacy-tire-frame` explicitly selects the control path. Manifest and
sampled body state declare the actual frame. Legacy equations remain available;
bit-identical old trajectories across shader recompilation are NOT guaranteed.

GREEN fixture: propulsion ends at 12.624 m/s; braking .429 m/s; passive opposite
steering ends at 1.729 and 1.635 m/s, with opposite heading changes. Reset repeats
the first neural snapshot. The physical-retina fixture measures a wall at exactly
4.500 / 58.500 m, producing wave97 /32 and applied steering .03905 /.00787.
It explicitly selects physical retina; the first attempt mistakenly selected
legacy geometry and was correctly rejected (6 m instead of 4.5 m).

Gate command: `nice -n 10 python3 scripts/check_native_flow_gpu.py --out
/media/llm-data/universe-runs/native-flow-gate-20260925-v2`.
Receipt: same directory `receipt.json`, status passed. Six exact GPU tests passed;
deliberately disconnected L2 and restored legacy tire transform were both caught.
Four tests cover body thermal budgets, water/energy, paid repair/growth, aging and
extreme morphology reserves. Missing and stale research-runner receipts were
also rejected with exit2 before creating output or starting simulation.
This gate certifies these controlled mechanisms, not learned city service.

## Registered: frozen production tire/retina comparison

Use the same DNA SHA `3b7ead0c24dfa2da12237cda363eda6b94b6c0244901ae86a26ad13fabda5a2d`,
128 s, seed0, fixed body, beta.85, v4 navigation. Compare legacy tire frame to
corrected frame with the SAME new binary; then change only retina to physical
`--retina-road-view 100`. Prediction: passive steering energy disappears; learned
delivery remains an independent behavioral gate. Do not treat distance alone as
improvement. Different optics invalidate claims of a matched historical policy;
this third run is explicitly a sensor intervention. Artifact root:
`/media/llm-data/universe-runs/tire-retina-ablation-20260925`.

## Continuation: complete generations and first demonstrated learning failure

### Mechanical gate and corrected-tire evaluation

The latest source-bound receipt is
`/media/llm-data/universe-runs/native-flow-gate-20260925-v7/receipt.json`.
Command: `nice -n 10 python3 scripts/check_native_flow_gpu.py --out
/media/llm-data/universe-runs/native-flow-gate-20260925-v7`.
All **10 exact GPU tests passed**, and both deliberately broken configurations
(disconnected L2 and the old tire-force rotation) failed at the intended boundary.
The gate rejects missing tests and software adapters; this is not a learning gate.

The body stress fixture uses 512 resident ticks at 1/64 s, with declared mechanical
load, and samples at the end. Both bodies spent 210,432 J. The hot body lost
0.056152 kg of water; the cold body lost none. Water currently represents a
closed coolant reserve: ordinary cool operation need not consume it. Temperature,
energy and water tests therefore do not certify autonomous thirst management.

Frozen tire/retina comparison: `tire-retina-ablation-20260925/report.json`.
Over 128 s, the old tire frame produced 758.617 m and 344,354 J collision energy;
the corrected frame produced 164.356 m without collision. Both delivered zero.
Switching the same genome to physical retina produced zero movement in that
production evaluation. The former tire transform's extra motion was not competence.
No default output-fatigue change was made.

### Whole city-learning battery: FAIL

`corrected-city-learning-20260925/native-s11`: 12 complete generations, 16 genomes
with two starts each, 384 judged attempts, 128 s per generation, seed 11.
Corrected tires, physical retina at 100 m, output beta 0, evolving body.
There were **11 accepted pickups, zero complete deliveries**. Best selection
improved from -313.71 to -177.53 by generation 2, then plateaued. DNA diversity,
higher fitness and changed body capacity do not prove useful morphological evolution.

Read-only audit command:
`python3 scripts/audit_taxi_learning.py
/media/llm-data/universe-runs/corrected-city-learning-20260925/native-s11`.
Expected exit: **1 (behavior failed)**, with all 12 generations accounted for.

The first two station pairs require 768 m and 960 m of road after approximately
60 m of approach. The existing start pool is ordered by identity, not increasing
motor difficulty. A 128 s deadline would require substantial sustained speed;
it is shorter than the reference pilot's travel time at its 5 m/s speed cap.
This does not prove physical impossibility for a faster learned driver, but it
invalidates the assumption that these are elementary beginner tasks.

### Same being/tick: the passenger signal arrives, the brake remains

`city-forensic-learning-20260925/native-s11` preserved up to three diagnostic
candidate genomes per generation. The new recorder reproduced all 32 per-being
outcomes exactly for generations 0–2 compared with the original battery.
`city-slot10-trace-20260925/native-s11` repeated generations 0–1 with
`--vision-being 10`; both full outcome arrays again matched exactly.
This is observational instrumentation, not a different policy or selection rule.

Generation 1, being 10:

| Tick | Onboard / route codes 178,179,183 | L2 effective throttle,steer,brake | Raw throttle,steer,brake | Applied throttle,steer,brake | Speed |
|---|---|---|---|---|---|
| 4832 | 0 / 88,-92,4 | -.124,-.402,.530 | -.425,-.901,.960 | 0,-.667,.694 | .161 m/s |
| 4848 | 1 / -88,91,112 | -.066,-.632,.264 | -.226,-.975,.724 | 0,-.700,.562 | .118 m/s |
| 5120 | 1 / -86,93,112 | .087,-.508,.206 | .300,-.948,.626 | 0,-.670,.246 | .086 m/s |
| 8192 | 1 / -88,92,112 | .039,-.667,.253 | .131,-.979,.698 | 0,-.702,.430 | .170 m/s |

Authoritative pickup tick: **4838**, 75.59 s. At tick 5120 the body retains
4,198,480 J, first forward gear, zero applied propulsion and zero delivered drive
torque. The service transition reaches the wave. L2 itself still requests brake;
the exclusive-pedal mapper removes throttle. This sampled failure is not an
empty tank, missing pickup signal, or disabled motor. The policy fails to restart.
Samples are every 16 ticks; the exact service timestamp comes from the event ledger.

### Feasibility witness: PASS, explicitly NOT learned

`corrected-city-witness-20260925`, command recorded in `command.json`:
one fixed body, physical retina, corrected tires, 16,384 ticks / 256 s,
`--taxi-reference-witness`. Pickup tick 1200; delivery tick 13470 (210.47 s);
815.279 m, zero contact/collision, final energy 2,843,901 J, water 10 kg,
body temperature 27.20 C, alive. The controller reads privileged route/task state.
It establishes feasibility of this route/body/service loop, not neural learning,
traffic generalization, complete map coverage or autonomous resource management.

This exposed an evidence bug: nested research metadata and the driving header
incorrectly labelled the witness as native/spline. `research_contract::configuration`
now records the actual intent producer, learning-evidence flag, privileged inputs
and linear motor head; generation and driving metadata agree. Neural stages may
still be recorded during a witness but are explicitly marked as not driving action.
The old artifact is retained unchanged as evidence of the metadata bug.

### Contracts tightened

- `audit_taxi_learning.py` rejects scripted/unknown controllers, absent requested
  generation count, incomplete or noncontiguous generations, mismatched identities,
  malformed selection vectors, unfinished horizons, nonfinite scores, events after
  frozen outcomes and deliveries unsupported by the body counter. Twelve JSON-only
  regression tests pass: `python3 scripts/test_audit_taxi_learning.py`.
  These tests do not run a CPU brain, body or physics simulation.
- Navigation v4 uses station-indexed route fields. Nonservice motor lessons publish
  arbitrary subgoals; v4 could overwrite these with its missing-route vector.
  CLI now rejects v4 with a motor lesson or a mixed curriculum containing one,
  before GPU allocation. Explicit v3 remains available for those lessons. Direct
  `move` and mixed `pickup,move` negative CLI checks both exited 2 as required.
- A successful short-course audit is only a recorded-attempt pass, never city
  competence. The witness is rejected by the learning auditor with exit 2 even
  though its delivery succeeded.

### Corrected-physics transfer: preserve the trained encoding

`corrected-short-course-20260925`: eight complete generations, 64 bodies,
physical retina, beta 0, evolving body, artificial 12 m trips. **Zero deliveries**.
Its first/main failure is not resolved by making the city task shorter alone.

Frozen factorial test, same DNA/body/geometry/horizon (32 s), corrected tires:
`short-course-transfer-ablation-20260925`, each directory has its command and trace.

| Retina | Output beta | Accepted pickups | Clean complete trips / 4 |
|---|---:|---:|---:|
| Legacy | .85 | 4 | 3 |
| Physical 100 m | .85 | 1 | 0 |
| Legacy | 0 | 4 | 0 |
| Physical 100 m | 0 | 1 | 0 |

Removing output adaptation is **not a demonstrated improvement** for the inherited
policy. Changing the visual encoding also invalidates its transfer. The failed
city-learning arm changed both; it cannot isolate either as its sole cause.

Frozen distance transfer with a fixed 128 s horizon, original retina/beta and
corrected tires: `corrected-distance-transfer-20260925`. Results were **4/4 at
12 m, 1/4 at 48 m, 0/4 at 96 m**; all starts accepted a pickup. These are four
separate lives, not consecutive deliveries by one being. This is a measured
short-range skill with poor distance transfer, not city navigation.

The next bounded experiment is `corrected-48m-learning-20260925`: 16 generations,
16 genomes x four cardinal starts, 64 s per life, fixed vehicle capacities,
original retina/beta, corrected tires. **Correction after gene inspection:**
`--fixed-body` does not freeze organism metabolism; this arm does not isolate
controller adaptation. A separate neutral-metabolism evaluation is required. Acceptance is one genotype completing all four starts without
contact, followed by frozen reevaluation. Its final result must be appended below.

### Observatory service interruption

The frontend was restarted before the API, causing GET `/api/db/batteries` 502.
System Python lacked uvicorn. The existing dependency environment is
`/media/llm-data/venv-rocm/bin/python`; launching it with `-m lab_api --port 8130`
restored the API. Direct port 8130 and frontend proxy 8131 both returned HTTP 200.
Browser reload displayed the battery table. No Lab UI or neuromod-lab source was
changed; these are the existing observation services for Universe recordings.

### Final 48 m generation audit and frozen repeat

The 16-generation battery finished with 1024 judged attempts, **786 accepted
pickups and 135 complete contact-free deliveries** across separate lives.
Generation 0 had zero successes; generation 15 had 9/64. The best genotype rose
from 0/4 to 2/4 successful directions; best selection from -26.82 to 921.09.
This is measured short-course improvement with fixed vehicle capacity genes;
organism metabolism also evolved. It does not isolate either contribution.
**The declared 4/4 acceptance failed.**

`python3 scripts/audit_taxi_learning.py
/media/llm-data/universe-runs/corrected-48m-learning-20260925
--require-all-starts` exits **1**. The auditor now groups clean outcomes by the
actual genome-major replica contract: successes from different genotypes cannot
be combined to invent a universally successful genotype.

`learned-48m-frozen-check-20260925` replays `evolve_best_gen0015.bin` as one
frozen genotype with the same four starts and 64 s budget. Results: **2/4**.
Successful directions 0/1 delivered at ticks 2605/3512, after 54.589/55.902 m,
without collision. Direction 2 approached within 3.147 m but did not complete;
direction 3 came within 0.111 m but did not complete. Reaching a station without
an accepted dropoff remains a failure, even if the distance counter looks good.

`witness-metadata-contract-20260925`: a separate 32-tick real-GPU recording checked
that organism, generation, school, driving-header and per-sample metadata all
identify the actual scripted intent source and linear head. All assertions passed.
This short recording validates metadata only and is not published as a behavior
preview or a successful task.

### Visual check of the failed learning arm

The existing producer packed generation 7, being 0 of
`corrected-short-course-20260925`, using `UNIVERSE_S2_BIN_DIR` pointing to
`Universe/target/debug`. The run's added viewer manifest explicitly marks this
as a failed 12 m training diagnostic. It was not registered as a successful battery.
URL checked: `http://127.0.0.1:8131/run/corrected-short-course-20260925/7/0?view=champion`.

- Tick 1024: UI speed 1.1826 m/s, gear 2, RPM 1668.1406, energy 4,558,824 J
  matched raw values (rounding only). Water 10.1297 kg reflects this body's
  evolved capacity; it is not evidence that water was acquired during the life.
- At 1x, the playhead advanced from tick 16 to 671 in 10.25 wall seconds:
  10.234 simulated seconds at 64 Hz, consistent within interaction timing.
- At tick 2048, the map shows the vehicle curling back and essentially stopped:
  raw speed .04224 m/s, applied throttle 0, brake .65491, zero deliveries.
- The 3D view displays the functional layer sequence and changing contributions.
  This is a schematic, not fly anatomy or proof of useful cognition.
- Remaining observability limitations: “destinations not recorded” on the map,
  and no school summary in the UI for this unregistered diagnostic, despite the
  authoritative school file on disk. Geometry/action data are visible, but the
  absence of catalog metadata must not be confused with missing simulation events.

### Final control review: metabolism and failed dropoff

A control-label error was caught before closing the investigation:
`--fixed-body` freezes vehicle capacity expression, **not** the 16 organism
metabolism genes. In the 48 m training run these genes changed from neutral to
nonzero values. Any earlier claim of isolated brain-only training is withdrawn;
the reports and audit now distinguish capacity evolution, metabolic neutralization
and the number of distinct metabolic genomes. A claimed neutral control with
nonzero recorded genes fails the evidence audit. There are now **12** host-JSON
contract regression tests, all passing; no CPU numerical model is involved.

The explicit matched counterfactual
`learned-48m-neutral-metabolism-20260925` adds `--research-body fixed` to the frozen
final evaluation. The generation body manifest confirms every metabolism gene is
zero. The same final brain still completes **2/4** directions cleanly (ticks
2281/3396, 54.804/56.027 m). This supports retention of a two-direction policy skill
without evolved metabolism. It does not prove that body evolution helped training,
nor that the other directions or city are solved.

`learned-48m-failed-direction-20260925` repeats all four frozen outcomes exactly
while recording neural stages for being 3. Among 222 sampled onboard records,
13 are within 4 m of the destination; **none** are also below 0.5 m/s. At tick
3616: destination distance .160 m, speed 2.670 m/s, raw dropoff .557, applied brake
.107 and throttle 0; taxi refusal code 1 (moving). The destination and request are
present. The policy fails the stop requirement; the world correctly refuses.
These are sampled counts, not all-tick dwell measurements.

Full strict audit files are preserved beside each tested run as
`whole-generation-audit.json`. All five complete batteries/evaluations audited
with `require_all_starts=true` remain **failed** at that criterion. The 135 valid
short trips are retained as partial evidence rather than promoted to a city claim.

Next bounded experiment should preserve the winning input/dynamics contract and
train stopping at a destination over multiple approach distances/speeds, retaining
pickup/restart. Compare the same frozen genotype on all four starts and held-out
routes. Changing retina, output adaptation, body control, objective and task length
together again would destroy the causal comparison. No further GPU job remains
running after the experiments recorded here.

### Resumed investigation: preserved policy, explicit neutral body

The user correctly rejected stopping while learned city service was unresolved.
`neutral-48m-continuation-20260925` resumes the generation-15 DNA above with
`--research-body fixed`, legacy input encoding and output beta .85. Its declared
budget is 24 generations, 16 genotypes × four starts, 4096 ticks per life, one
real-GPU job, at most 900 wall seconds. Its immutable command/plan/binary precede
execution. The v7 mechanism receipt was verified against source and executable.
Acceptance remains one genotype passing all four full trips cleanly, followed by
a frozen evaluation; this does not establish city competence.

Generation 0 reproduces the neutral frozen control's four school records exactly
(2 clean deliveries and 2 failures). Early generations preserve the elite but
mutated offspring mostly regress. Actual runtime motor defaults are mutation
probability .10 and sigma .10, despite the library EvoParams defaults being .02
and .05. These are not interchangeable. A smaller mutation step is a hypothesis
to test if the bounded continuation plateaus, not yet a demonstrated fix.
API and UI proxy were rechecked: GET /api/db/batteries returns HTTP 200 on both
8130 and 8131. Service availability does not certify learned behavior.

Browser root review during the continuation: the battery table loads, but the
existing “Vida no mundo” panel displays `taxi-short12m-champion-20260925` beneath
text claiming “passageiros ligados, sem lição”. That artifact is an artificial
short school service course. The generic panel description is misleading; it
must not be treated as production-city evidence. No Lab UI or neuromod source
was edited in this Universe-only investigation.

The continuation finished normally in 399.65 s: 24 complete generations,
1,536 judged attempts, 1,218 accepted pickups and 273 complete contact-free
trips. The strict whole-generation auditor returned **FAIL**: zero genotypes
passed all four starts. Best selection rose 891.52 → 914.67 without increasing
its two successful directions. Counts across generations are repeated related
attempts, not independent experimental replicates or consecutive city trips.

A paired run, `neutral-48m-smaller-motor-step-20260925`, changes only motor
mutation sigma .10 → .025, preserving the founder, seed, budget and task. Its
plan records the exact comparison before execution. This tests mutation dose;
it does not silently change the simulator, score or service stop threshold.

The control also produced rare successful west/north specialists: generation 12
DNA index 15 delivered west, generation 15 index 9 north, generation 22 index 10
west. Each passed only 1/4 and lost to the 2/4 aggregate-score elite. Their DNA
was not preserved by the three-role forensic saver: best-score/pickup/delivery
all selected the same other candidate. This is a demonstrated forensic coverage
gap, not evidence that combining those brains would solve all starts. Retaining
per-start evidence would allow direct investigation of those skills.

A competing hypothesis (legacy racing-line curvature polluting city decisions)
was checked and rejected for the sampled failed course: city generation clears
`world.racing_line`; wave channels 72, 78–80, 101, 109–113 were zero at the inspected
ticks 16, 512, 1024, 2048 and 3600. No speculative patch was applied there.

The smaller motor-step arm completed 24 generations in 405.72 s: 316 complete
contact-free trips, but still **0 four-start genotypes**. Its champion score rose
to 941.69; west minimum remaining distance 2.89 m with 510 J collision energy,
north .12 m without collision, neither stopped to deliver. This single-seed result
supports a modest search improvement, not resolution of the failure.

The forensic saver now retains a clean delivery per start (up to 16 starts),
beyond the three existing aggregate roles. At most 19 checkpoints are written;
omitted start coverage is explicit. Array lengths and slot/start identities must
match before any write. Three file/identity tests failed on the old saver and pass
on the correction; they do not simulate neural/body/physics numerics. The 12
host-JSON evidence tests pass. Existing compiler warnings remain outside this change.
The refreshed **v8 GPU receipt** passes all ten numerical fixtures and detects both
intentional faults. No CPU numerical oracle or simulation was used.

`neutral-48m-worst-start-20260925` tests the existing worst-start selector against
the smaller-step mean-selection arm. The founder and all numerical configuration
other than selection are unchanged. The binary also includes the forensic-only
recording fix; generation 0's 64 school records exactly equal the paired control,
while the saved files now include the clean delivery from each successful start.
This separates the recording change from the later selection experiment.

The worst-start arm completed 24 generations with 176 complete trips, compared
with 316 for its mean-selection control. Its best minimum score improved
120.46 → 254.58, but the selected champion regressed from 2/4 to **1/4** full trips.
It is not promoted as an improvement in service competence. Individual records
for generations 0 **and 1** match the old recorder/control exactly; subsequent
selection divergence is expected. Rare clean north/west checkpoints now exist.

The next discriminating test is `motor-inheritance-probes-20260925`: nine frozen
GPU conditions (original founder, whole north/west specialists, and each donor's
three motor rows, steering-only row, or brake-only row inherited into the founder).
The canonical Rust genome loader/saver is used for the host-side byte/array copy;
no neural or physics forward pass is computed there. All conditions retain the
four original starts and 4096-tick deadline. This is an exploratory module
counterfactual using previously evaluated DNA, not a claim of new training or
independent generalization. The founder control again reproduces 2/4 exactly.

The motor-focused arm (`--p-mutate 0`, motor sigma .025, mean selection)
completed another 24 generations: 610 contact-free trips, still **no four-start
winner**. This freezes most backbone parameters, not the independent activation
mutation. A canonical DNA diff of its generation-2 winner confirmed only 36
motor L2 coefficients changed in that particular checkpoint. This is a search
improvement on repeated school attempts, not learned city competence.

All nine motor inheritance probes completed. Whole north DNA passed south+north;
whole west DNA passed west only. Transplanting their motor rows into the founder
never exceeded 2/4. Steering/brake rows alone did not transfer the donor's rare
skill. The result rejects the simple motor-row transplantation hypothesis; it
does not prove the skill cannot be combined by further learning.

### Geometry and numerical dynamics isolation (ongoing)

A new opt-in `--retina-centered-grid` preserves legacy pitch/range encoding while
registering cells at their actual centres. A GPU fixture first failed on the old
mode handling (4.885 m slant range instead of 4.5 m legacy horizontal range), then
passed on both axes and both wall faces (4.500/58.500 m). Full physical road-view
remains a separate profile. This is not yet a demonstrated behavioral improvement.
Evidence: `geometry-isolation-20260925/centering-{red,green}.log`.

A second GPU fixture runs identical native neural stimuli/actions in four
rotated empty-road poses for 128 ticks. Intent and applied actions remain equal,
but the original integrator diverges by about .15 m at two seconds under a
quarter-turn. East/west and north/south remain paired. First tiny orientation
rounding differences at tick 4 grow into wheel/RPM differences by tick 32.
An explicit-FMA-only orientation change did not fix it and was **reverted**.
This does not explain the learned founder's east/south versus west/north pattern
by itself; those course scenes are not exact rotated duplicates.

A separate fixed-control GPU experiment refines the whole physics timestep
(1, 2, 4, 8 subdivisions per 1/64 s, same two-second duration, throttle .5,
steer .3, initial speed 4 m/s). At one subdivision, east velocity ended at
(5.197, -6.332) m/s; at eight it ended at (5.706, -11.823) m/s. The two-axis
rotation discrepancy shrank from centimetres to approximately 1e-5 m at eight.
Thus the production timestep is materially sensitive in this controlled case;
changing only arithmetic order is not an adequate fix. This diagnostic's exit
0 certifies finite results only, **not convergence or valid learned behavior**.
Logs: `geometry-isolation-20260925/refinement.log` and `rotation-detail.log`.

An opt-in bit-11 backward-Euler wheel solver is now under GPU evaluation. It
solves I*(omega_next-omega_old)/dt = drive - radius*F(omega_next) - brake,
with a zero-speed Coulomb brake interval, the existing Pacejka/friction circle,
and a bounded 20-step bisection. Chassis velocity/load are held within the wheel
solve; it is not a fully implicit chassis model. Legacy numerical behavior stays
selected without bit 11. No training or deployment uses this candidate yet.
Acceptance: rotation fixture, controlled step refinement, existing brake/coast/
body contracts, then frozen learned-DNA comparisons. No success is inferred
from shader compilation. The latest successful v8 receipt is stale for this
working source and must be replaced before research execution.

Implicit wheel integration alone improved the fixed-control timestep comparison
but still failed the .002 rotation tolerance (about .024 m on one component).
Turning **only** TCS off with implicit wheels passed that fixture. TCS used an
instantaneous torque gain driven by the preceding step's slip. The candidate now
keeps immediate torque cuts and bounds torque recovery with an explicit 100 ms
time constant; no added state/readback is needed. This is a separate bit 12,
while implicit wheels remain bit 11. Both are opt-in via
`--organism-stable-wheels`; no old checkpoint silently adopts them.

The combined profile passed rotation, native brake/coast/reset/energy, and the
added acceleration-to-brake immediate-zero-drive check. GPU timestep refinements
approach each other; coarse versus eight subdivisions differed by .086 m and
.072 m/s in the fixed manoeuvre. Turning off either implicit wheels or bounded
TCS separately failed the rotation test. Restoring the retina's corner indexing
also failed at the intended 4.5 m wall-distance assertion (reported 6 m).

The v9 gate passed all fourteen GPU fixtures and five negative controls, then
correctly failed at runtime compilation: one additional metadata field exceeded
serde_json macro recursion. New metadata assignments were split out of the large
macro instead of raising its recursion limit. **v10 passes the complete gate and
certifies the actual runtime.** The profile equations and limits are documented
in `docs/PHYSICS-NUMERICAL-CONTRACT.md`.

`geometry-physics-frozen-ab-20260925` now evaluates the same original 2/4 DNA in
four predeclared conditions: old profile, centred legacy retina only, stable
wheels only, and both. One generation, four starts, 4096 ticks each; no mutation.
Its command/plan/immutable binary and hashes precede execution. This separates
numerical/sensory changes from additional learning. A corrected mechanism may
require relearning; neither that expectation nor a changed score is a pass.


## Follow-through: corrected profile and learned four-start result

The initial v1 TCS recovery implementation filtered total drive torque. A no-slip
GPU fixture exposed an unintended pedal reduction (125 Nm versus 867 Nm). That
candidate was rejected. The current v2 filters only TCS gain, cuts immediately,
and recovers with a 100 ms time constant. Two half stages of implicit wheel
integration satisfy the original bounded timestep-refinement criteria; the
one-stage candidate failed those criteria and was not accepted. GPU MidState
stores the gain and initialization marker. Runtime checkpoint format is now v4;
old runtime snapshots are rejected, not silently reinterpreted. DNA is unchanged.

A separate birth-path inspection found that ecological newborns lost the selected
numerical profile. `inherit_resident_profile` now preserves that profile and dt
while retaining the child's morphology and capacity genes. The GPU fixture checks
actual uploaded parent/child state; its fault-injected counterpart must fail.
This is a birth mechanism test, not proof of successful ecological reproduction.

`native-flow-gate-20260925-v12/receipt.json` passes 17 exact physical-GPU tests,
6 intended fault detections and runtime compilation. The historical v10 receipt
certified the earlier candidate only. Current numerical contract:
`PHYSICS-NUMERICAL-CONTRACT.md`. No CPU numerical oracle was used.

Frozen factorial v2 (`geometry-physics-frozen-v2-20260925`) retained the original
2/4 result in the control. Centered retina alone, corrected wheels alone and both
each achieved 1/4 with the unchanged old DNA. Numerical correctness does not imply
that a controller trained against the old dynamics transfers without learning.

`stable-centered-48m-learning-20260925` completed 12 generations, 64 genotypes x
4 original 48 m full-trip starts x 4096 ticks. It improved from 1/4 to 3/4 and
recorded 1467 clean trips across 3072 related attempts. West had zero successes.
Those are repeated training attempts, not independent replications or city trips.
The frozen winner repeated 3/4 in `stable-west-diagnosis-20260925/learned-frozen`.
On west, ticks 576–832 already contain a negative route lateral cue while L2 often
emits positive or near-zero steering; at tick 832 adapted steering is +.756 and
final steering +.638 before wall arrest. This discrepancy precedes action output.
The same four tasks are feasible for the explicitly scripted witness (4/4, zero
collision, pickup 271/dropoff 1210). That witness is inadmissible learning evidence.

`stable-centered-backbone-thaw-20260925` then allowed sparse backbone mutation
(p=.002, retaining motor p=.10/sigma=.025 and independent activation mutation).
The frozen body, course, deadline and acceptance criterion were unchanged.
It completed 12 generations; the first all-four clean genotype appeared at gen6,
and gen11 contained five. Total clean trips: 1471/3072 related training attempts.
This exploratory sequence does not isolate mutation dose as a causal treatment.

Crucially, `stable-four-start-confirmation-20260925/learned-frozen` re-ran the
saved gen11 winner with one genotype, no mutation and no scripted controller:

| Start | Pickup tick | Dropoff tick | Collision energy |
|---|---:|---:|---:|
| East | 470 | 2334 | 0 J |
| South | 427 | 2910 | 0 J |
| West | 601 | 1516 | 0 J |
| North | 318 | 2565 | 0 J |

All four original full trips passed, alive and contact-free. This is a verified
school milestone, **not city coverage, body evolution or generalization**. The
body was fixed and metabolism neutralized; water consumption was not exercised.
Transfer to 96 m trips and native starts is being measured separately under
`stable-route-transfer-20260925`; do not infer its outcome from this milestone.


### Transfer and visual inspection (same frozen winner)

The transfer diagnostic completed: 96 m full trips passed east/south only
(dropoff 3907/3960), failed west with contact (~44,969 J) and north at deadline.
Native routes passed zero of four starts. All-start strict audits are **failed**.
The permissive `audit(..., False)` report for 96 m means at least one recorded
trip; it must not be quoted as four-start acceptance. Separate
`strict-all-start-audit.json` records the required criterion.

Native observed slot 2 travelled 49.61 m without contact, then stopped about
11.25 m before pickup. At t2048 its fuel is 4.628 MJ, engine 24.68 C, speed zero,
raw throttle/steering/brake zero; at t4080 raw throttle is -.175 and final throttle
zero. This is policy/output behavior before service, not evidence of a depleted
body or wall obstruction. It also does not establish the adaptation stage alone
as the cause. The native destination is far away; this bounded diagnostic does
not certify that a 4096-tick horizon suffices for the complete city journey.

`stable-native-pickup-learning-20260925` therefore targets the complete native
pickup approach with existing station locations, no prepared start or artificial
course, 12 generations/64 genotypes/4 starts/4096 ticks. Pickup competence has
its own acceptance and is never counted as a delivery. Preserve the full-trip
founder and test retention before replacing it. This is an exploratory curriculum
step, not a production deployment.

The confirmed 48 m replay was copied byte-for-byte into
`school-48m-four-starts-confirmed-20260925` with source hashes and an explicit
school-only claim, packed/registered through the existing trace producer (no Lab
source changes). Browser inspection at t768 shows 14 m travelled and 3.0 m/s;
at t1520 it shows 56 m and .3 m/s, agreeing with raw data (2.980/.320 m/s).
Raw taxi state changes from onboard=1/deliveries=0 at t1504 to 0/1 at t1520;
the authoritative service event is t1516. The brain panel loads 1587 channels
and the trace declares zero missing blocks. Only being 2 has a neural trace;
other slots are correctly marked unavailable.

Visual limitation: at the current ~948x545 viewport the default layout squeezes
the map behind overlapping text and clips most of the brain. Closing the retina
and pedal panels made the actual map inspectable. The map also says station
destinations are not recorded. These are unresolved Lab UI presentation issues;
loading a trace is not equivalent to a usable observatory. No UI source was edited.


### Native pickup training and retention rejection

Native pickup training finished all 12 generations: 513 clean pickups across
3072 related attempts, per-start counts 2/271/84/156. Best coverage reached 3/4,
not four. Frozen repetition confirms pickups at ticks 1343, 1776 and 2542 on
starts 1, 2 and 3; start 0 travels 46.72 m but ends 13.46 m short at the original
4096-tick deadline. This is a partial curriculum result only.

The same new DNA regresses on the original four full trips: only south/west
complete (2333/1901); east/north fail. Therefore it is **rejected as a replacement**
for the verified four-trip founder. A competence archive preserves old DNA but
cannot by itself establish retention within one evolving controller. Future
promotion must measure both prior full-trip and new native-start coverage.

Playback was also exercised in the browser at 1x: about 1646 ticks advanced in
25.716 wall seconds, consistent with 64 ticks/s. Dedicated brain mode renders
the complete schematic; focusing L2 with “all available” shows 4608 connections,
unchanged across t768/t1661 while activation colors change. The cramped combined
layout remains an issue; dedicated mode makes the brain inspectable.

The JSON evidence auditor gained two regression tests: duplicate distinct starts
and a replica index inconsistent with slot mapping must fail. Both failed before
the fix and pass afterwards (15 total file-contract tests, including preservation of valid K=1
populations with different starts; no CPU simulation).
Re-running the stricter auditor on the real four-trip training and frozen repeat
still passes; 96 m transfer still fails. Genotypes in a replicated batch must
share the same distinct start set. This closes a false-coverage loophole without
changing any physics, brain output, selection or previously recorded evidence.
