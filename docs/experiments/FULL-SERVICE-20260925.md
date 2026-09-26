# Complete service experiment — 2026-09-25

Status: training and frozen audits completed; 24m graduation gate failed. This is an engineering experiment, not a claim of
city-wide competence or causal superiority of evolved morphology.

## Why this experiment exists

The earlier `0,6,18,36,native` delivery curriculum starts onboard in every numeric
stage. It practices the final approach/dropoff but never the pickup-to-navigation
transition. Its native stage suddenly requires the whole, much longer task.
The 30-generation experiment at `research-curriculum-20260924-v1` reached only the
6m prepared stage. That is a failed graduation gate, not autonomous taxi service.

There was also a reward discontinuity: pickup reset destination readiness, losing
up to 200 accumulated approach points. The current judge banks that credit for
one authoritative pickup in a complete task. Prepared onboard starts cannot earn
it. Pickup alone still never passes the delivery lesson.

Finally, school event fields used to update after a judged outcome. The new
revision freezes the entire record. Historical request/accepted-event totals
from those binaries must not be interpreted as events inside the judged span.

## Executable task contract

`--school-service-course 12` creates four deterministic straight tasks on existing
city roads. Each has a real pickup station and a distinct dropoff station 12m
away, with disjoint 4m service discs. The body starts 6m before pickup at the
legal lane offset. All four directions are evaluated for every genome, K=4.
Initial phase is seeking pickup, never already onboard.

The same native brain, motor mapper, GPU physics, body metabolism and taxi
service kernel execute. There is no scripted pilot or injected motor command.
The school bench disables inter-agent contacts/shared passenger competition.
The road-route observation v4 is privileged map information, not learned mapping.
The fixed short course does not test junctions, traffic negotiation, scarce
resource competition or unseen city topology.

## Validation before training

Real Radeon Vulkan GPU, DISPLAY unset, one compilation job and one Rayon thread:

```sh
env -u DISPLAY WGPU_BACKEND=vulkan \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
  MESA_VK_VERSION_OVERRIDE=1.3 CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=1 \
  cargo test --lib school_tests -- --ignored --test-threads=1
```

35 passed, including banked pickup credit and record immutability after both
success and failure. Numerical inference/physics validation used no CPU fallback.
Log: `/media/llm-data/universe-runs/full-service-evolution-20260925/gpu-school-validation.log`.

The 2-generation GPU canary in `full-service-canary-20260925` completed without
contract failures: 3 then 6 accepted pickups out of 32 attempts, zero deliveries.
The embedded build-source SHA matched the captured source. It verifies the
specified `src/**`, Cargo manifests/lock and build.rs scope; it is not a claim of
bitwise reproducibility across toolchains or external system libraries.

## Training command and evidence

```sh
nice -n 10 python3 scripts/run_research_battery.py \
  --out /media/llm-data/universe-runs/full-service-evolution-20260925 \
  --suite full-service --lesson delivery --service-course-m 12 \
  --curriculum-body evolved --navigation-observation v4 \
  --seeds 11 --gens 60 --beings 16 --ticks 4096 \
  --timeout-seconds 2400 --execute
```

Every generation evaluates 16 hereditary genomes in 64 GPU slots. Morphology and
brain parameters mutate and are selected together; metabolism genes are kept
neutral in this research arm. This establishes a mechanism, not that morphology
evolution caused a behavioral benefit. `organism_gen*.json`, body provenance,
genome binaries and per-attempt records preserve the actual conditions.

Early observed outcomes: 1 complete delivery in generation2 and 3 in generation3.
The generation3 successes had pickup ticks 910/1629/891 and dropoff ticks
2131/2657/1920; distances 19.05/15.17/20.88m, zero contact ticks/joules, integrity1,
alive. These are training observations, not independent trials or frozen-policy
generalization.

## Observed results

The runner exited successfully with 60/60 generation files and a completed
`report.json`. Generation0 had zero complete trips; generation30 had 35/64;
generation59 had 38/64, all contact-free, alive and integrity1. The last five
generations averaged 38.2/64 complete trips. The first genome to pass all four
starts appeared in generation26. These are dependent training observations from
one evolutionary lineage/seed, not 64 independent experiments.

Frozen audits used exactly the final generation59 DNA (mix1.0, no mutation):

| Audit | Complete clean/alive trips | Interpretation |
|---|---:|---|
| Generation0 champion, evolved-body expression, 12m | 0/4 | Founder baseline |
| Generation59 champion, evolved-body expression, 12m | 4/4 | Reproduced learned short complete service |
| Same generation59 DNA, neutral body, 12m | 4/4 | Success does not require its evolved morphology in this course |
| Same generation59 DNA, evolved body, 24m /8192 ticks | 3/4 | Longer-course gate failed; not a city graduate |

The evolved-body audit reproduced the selected training genome's per-start
score, pickup/dropoff ticks, distance, contact count and integrity exactly, even
with diagnostics enabled. Input and exported genome hashes match. The four
pickup→dropoff tick pairs were 439→2084, 507→1843, 550→1275 and 575→2189. Maximum
speeds were 1.62, 1.70, 1.94 and 1.28m/s: useful service behavior, still very slow.

For observed slot0, the 130 neural/motor samples through its judged outcome had
52 applied-brake values above .05; maximum .642. Tick432, before pickup, had
speed .581m/s and applied [throttle=0, steer=-.203, brake=.458]. Tick2064, before
dropoff, had speed .357m/s and [0,.300,.147]. This proves active braking reaches
the body; it does not establish efficient anticipation or optimal pedal timing.
Neural samples are every16 ticks, so these are nearby samples, not a fabricated
command at the exact service-event tick.

Body records show one neutral morphology in generation0, and 16 distinct
morphologies in generation59 (mass1336.75–1351.72kg). Each genome's four replicas
have identical body genes. Canonical `genoma_view --json` exports, rounded to four
decimals, show 1372 changed L1 coefficients and 264 changed L2 coefficients
between generation0 and59 champions. Thus both body and brain vary; these counts
are not a causal attribution of learning to either component.

Evidence directories under `/media/llm-data/universe-runs/`:

- `full-service-evolution-20260925`: original 60-generation training.
- `full-service-frozen-20260925`: diagnostic frozen final champion.
- `full-service-neutral-body-20260925`: same DNA with neutral expression.
- `full-service-founder-audit-20260925`: frozen generation0 champion.
- `full-service-24m-audit-20260925`: longer-course stress test.

At24m all four starts accepted pickup; three completed clean/alive delivery.
Start3 accepted pickup at575, never accepted dropoff and failed for no route or
service progress at8097. It travelled55.55m, reached a best remaining distance of
3.760m and emitted1360 dropoff requests. Distance travelled and requests are
therefore not competence. It reached the vicinity but did not complete the
legal service conditions. Its detailed neural trace was not recorded (slot0
was selected); do not attribute that failure to a specific layer from these
aggregates. Next discriminating test: record that failing start and examine
speed, service radius and refusal codes at its closest approach. No additional
training was launched automatically after this failed gate.

Binary SHA256: `026bb41d923dbbc537f1d8fd9cf2c50072d6b9d5b15fbd88a457bce6909fe816`.
Source SHA256: `0425cac0ab6ddefbbd40ea9174553ccc0ce3b5d6255f1877e4baae97552bdac1`.
Each directory keeps the actual command, source snapshot and per-attempt records.

## Lab UI publication

### User-visible acceptance correction

The user's replay review rejected this as useful taxi behavior, correctly. The
selected car travels19.29m in32.56s (mean about2.13km/h), peaks at5.82km/h, and
is stationary near the destination bytick3072 through4096. The pickup and
destination centers are only12m apart; service accepts within4m and below.5m/s.
The four reported successes are four replicas on four starts, not four sequential
jobs in one life. Bench mode provides one isolated task and does not exercise
the production passenger-renewal/resource-service loop. This is a short-course
learning result, not an operational taxi or an adequate final demonstration.

The raw observation records `pickup_accepted` (observer tick440, school tick439)
and `dropoff_accepted`/`delivery_incremented` at2084. Nearby sampled positions are
(110.816,94.130) attick448 with passenger onboard and
(123.681,97.465) attick2096 with passenger absent and deliveries1.
However, the published `events.json` contains no service-event markers, despite
those raw records. The generic UI outcome label is “parada válida” even for this
delivery lesson. Thus the publication also fails to explain the very event it
claims to demonstrate. Rendering layers and advancing ticks did not validate
the adequacy of the user-facing result. Do not repeat that acceptance shortcut.

Keep this artifact as a limited school exercise. The next final acceptance must
measure sequential jobs in the production taxi loop on meaningful routes, with
intersection coverage, progress/stall evidence and visible passenger events;
do not count four replicas or a single near-station stop as that acceptance.

The existing API is running locally on8130 and Vite on8131. No neuromod-lab
source was changed. `scripts/publish_research_run.py` makes new catalog copies,
checks each copied file's SHA, and derives wrapper metadata from the actual
completed research report and command. It never fabricates numerical traces.
The existing `lab_api.trace.produce` then validates/packs/indexes the observation.

Catalog battery: `taxi-full-service-20260925`. Runs:

- `taxi-short12m-evolution-s11-20260925`: all60 training generations.
- `taxi-short12m-champion-20260925`: final frozen replay, evaluation generation0.

Open `http://127.0.0.1:8131/run/taxi-short12m-champion-20260925/0/0`.
Browser verification showed moving replay ticks, the correct eight stations,
retina, motor chain and a rendered 3D functional brain graph. There are256 neural
samples (ticks16..4096), with no missing packed blocks. Only slot0 has a recorded
neural trace; other cardinal starts are explicitly disabled as unrecorded.
The 3D graph is schematic native-brain topology, not fly anatomy. The replay
continues after the judged success at2084; post-outcome frames do not extend the
frozen school evidence span.

## Acceptance and remaining gates

Count a complete delivery only when the same attempt has success, pickup before
dropoff, positive distance and events no later than its recorded outcome. Report
contact-free and alive/integrity outcomes separately. Checkpoints for evaluation
must reload with mix1.0 and have identical input/exported DNA SHA.

The bounded run, frozen four-start repeat, neutral-body intervention, founder
baseline and longer-distance audit are complete. The longer-distance gate
failed. A failure outside the short course remains a failure; do not label the
course champion a city graduate.

### Frozen audit protocol, recorded before evaluation

Use the last-generation champion, not whichever checkpoint happens to perform
best in a new evaluation. Reload its exact DNA with mix1.0, one genome and four
cardinal replicas. Stop each run on process/contract failure or 300 wall seconds.

1. Repeat the 12m course for 4096 ticks with the evolved body and diagnostic
   recording. Expected engineering gate: four complete, contact-free, alive
   deliveries, with ordered pickup/dropoff ticks. Compare the per-start outcome
   to training; instrumentation is not allowed to change the policy.
2. Repeat with the neutral body, keeping the DNA and conditions unchanged.
   This is an intervention on one selected policy; no population-level claim or
   statistical superiority is supported by four deterministic orientations.
3. Audit the generation0 champion under the same 12m/4096-tick conditions.
   This describes change through selection; it is not an independent control
   lineage or proof that every proposed mechanism caused improvement.
4. Audit the final champion at 24m for 8192 ticks. This changes distance, station
   placement and budget together and is an extrapolation stress test. Passing
   requires the same four clean full trips; failing triggers a trajectory and
   request audit, not an automatic additional training run.

The minimum gate is a count of four valid events, not a statistical MDE. There is
one evolution seed, so no confidence interval across independent lineages is
claimed. Brake use is measured from applied commands near service events;
slowing down alone does not establish active braking.
