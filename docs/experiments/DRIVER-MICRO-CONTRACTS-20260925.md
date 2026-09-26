# Driver micro-contract investigation — 2026-09-25

Status: OPEN. Owner: Codex. No city-delivery or learned competence claim.

## Question and registered budget

Can the current learned DNA pass short, deterministic driving tasks before we
spend another city training budget? Separate production mechanism fixtures with
designed weights from frozen learned-policy evaluation. Neither replaces the other.

Run the real-GPU mechanism suite sequentially, including canonical service and
judge counterexamples. Then one frozen DNA, four starts per motor lesson, at most
4096 ticks (64 seconds) per lesson: move, stop, turn_near, turn_far. No evolution,
no pilot, no initial passenger. Default stop on the first failed skill; an explicit
diagnose-all option may collect all four failures, never launch a city run.
Acceptance: all four starts succeed alive, without contact, within the declared
lesson horizon. Report per-start peak speed, distance, outcome and failure code;
an instantaneous speed minimum alone is not learned driving. There is no global
minimum-speed rule during stops. One GPU process, one build job, >=5 GiB free,
120 seconds wall ceiling per lesson. No browser under the owner's instruction.

## Existing evidence and flow

`navigation-preview-paired-20260925/report.json`: frozen v4/v5, four starts each,
zero pickups and deliveries. Mean route-score progress 42.8298 / 35.0099 (points,
not metres). Original v5 directory was mislabelled; correction is preserved.
Founder being0/tick2064 in `cross-city-forward-learning-20260925/founder`:
19.145 m/s, route bytes [125,20], steering .045, brake 0; stationary by tick2208
after contact. This is not evidence of a weak engine or successful driving.

| Order | Executed producer -> consumer | Resident contract / cadence |
|---|---|---|
| 1 | retina.wgsl -> perception passes | pixels -> packed 192 signed-byte wave, per brain tick |
| 2 | taxi route/senses + body sensors -> brain.wgsl | versioned overlay, KAN/CfC -> L1 96 -> L2 48 |
| 3 | native intent -> organism_action.wgsl | 96-float stride; spline, adaptation, pedal arbitration |
| 4 | organism pre -> physics A/B/C -> organism post | paid capacity, 92-float input / 56-float output, 1/64 s |
| 5 | taxi post -> school judge | authoritative task transition -> frozen outcome; no policy override |
| 6 | generation readback -> school_result + sampled driving trace | host validation only, no CPU neural/physical simulation |

The learned micro-tests use existing v3 observation for nonstation lesson goals;
v4/v5 are station-indexed and cannot be silently substituted. This is a declared
transfer test, not a comparison with an unchanged city observation distribution.

## Review of the proposed attack order

- Reproduce mechanism and judge contracts FIRST, then the intermediate turn
  lesson, before the proposed 12-generation x 3-seed reward experiment.
- Current service_v2 has explicit canonical pickup/delivery branches without a
  hidden dwell; test them. The report mixes legacy stopping requirements with v2.
- Readiness is a continuous proximity x slowing maximum after an eligibility
  gate, not an intrinsic binary +200 step. A generation score jump cannot identify
  that term as its cause without decomposition.
- `sum((delta_distance/dt)*dt)` equals signed net progress. Rate alone does not
  reward speed. Positive-only deltas can reward loops. Integrated readiness can
  reward dwelling until the cap. Any v3 proposal must reject retreat/loops/idle,
  prefer equally safe faster completed trips, and preserve honest service IDs.
- Do not remove geometric contact reporting to repair legacy acceptance. Contact
  and dissipated impact energy are different contracts; compatibility belongs
  in a versioned judge if needed.
- Fix missing brain shader fingerprint before any stateful resume. Frozen DNA
  loading is not restoration of a complete runtime checkpoint.

## Results

Pending execution. Mechanism, learned skills and judge incentives must be reported
separately. Failure is an actionable result, not a reason to loosen acceptance.
