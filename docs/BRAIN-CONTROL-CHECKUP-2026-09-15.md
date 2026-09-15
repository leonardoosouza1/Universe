# Native brain and control checkup — 2026-09-15

The current checkpoint does not demonstrate reliable braking or wall anticipation.
Completing laps while repeatedly colliding and using the programmed reverse
controller is insufficient evidence of that competence. This checkup separates
recorded observations, verified code defects, experimental changes and pending
physical validation.

## Recording identity and coverage

The three existing `calibrated120-{8005,8200,8006}` recordings under
`/media/llm-data/universe-runs/navigation-improvement-20260914/` evaluate checkpoint
`40ba411113cb7a3b005a4a234da944d3f40be324db9427b0a7730de6378599fa`.
Each contains 16 exact genome clones at different starting positions, seed 77,
30,000 ticks at 64 Hz, fuel enabled and hippocampus disabled. These are frozen
evaluations, not additional training generations. Different tracks and starting
positions are not independent evolutionary seeds.

Physics windows cover all 16 beings. Retina, wave and sensory CfC state were
sampled every 16 ticks for **Being 0 only**, selected by `--vision-being 0`.
This does not automatically select the best being. Switching beings cannot
recover unrecorded neural data. The replay now explains coverage and provides a
button to return to the recorded being.

The existing `fast_state[96]` is sensory CfC recurrent state, **not L1 neuron
activation**. Retina/wave are pre-physics, CfC is post-brain, and body speed is
post-physics. The added `body.motor_trace` records the actual pre-physics speed
and command stages, without another GPU download. Old recordings lack this trace.

## Measured failures

Reproduce the compact analysis for each track:

```bash
python3 scripts/replay/checkup_report.py \
  --run /media/llm-data/universe-runs/navigation-improvement-20260914/calibrated120-8005 \
  --output target/checkup-20260915/checkup-8005.json
```

| Track | Population brake ticks / alive ticks | Impact ticks / km | Stationary alive time | Being 0 samples above 10 m/s with positive raw brake |
|---|---:|---:|---:|---:|
| 8005 | 0.2233% | 20.17 | 14.29% | 0 / 864 |
| 8200 | 0.2161% | 26.95 | 20.98% | 0 / 1261 |
| 8006 | 0.3654% | 20.18 | 11.83% | 2 / 1022 |

Impact ticks are not independent collision events. High-speed subsets use the
recorded post-physics speed; they do not establish exact pre-impact causality.
At high speed the sampled brake intent is almost always already nonpositive.
Removing a downstream low-speed gate cannot create missing positive intent.

The fast throttle head also frequently emits negative values while positive
throttle reaches physics: 557/864, 882/1261 and 656/1022 high-speed samples.
Mean slow output 0 is 0.9292, 0.9274 and 0.9381, respectively. Its contribution to
throttle is approximately +0.46 because the historical arbitration is:

```text
throttle = clamp(max(head(raw_throttle), 0) + 0.5*slow[0] + hip[0], 0, cap)
cap      = 1 - 0.8*max(slow[1], 0)
```

Negative fast output has no authority to cancel the slow positive contribution.
That is a control-authority limitation, not proof that the checkpoint learned
to request braking: its negative outputs evolved under this very clamp.

## Code findings and independent interventions

| Intervention | Verified mechanism | Implementation and limit |
|---|---|---|
| `--throttle-arbitration signed` | The pre-sum floor discards negative fast throttle. | Sum signed head, slow and hip before clamping. May stall this old checkpoint; requires a paired run and subsequently training. |
| `--wall-probes continuous` | CPU bins produce discontinuous distance differences. | DDA to actual voxel boundaries; lateral wall horizon 40 m, frontal wall/hole horizon 50 m. No additional GPU readback. |
| `--slow-kinematics canonical` | Slow slots 4/7 use obsolete ch33/32 instead of speed/alignment ch65/64. | Corrects only these two slots. Other historical slow-map semantics remain unchanged. Rust/WGSL uniform offsets are checked through Naga and actual bytes. |
| `--reverse-controller-v2` | A 3 m obstacle can trigger and immediately cancel reverse; escalated duration can exceed its own limit; timeout lacks consistent cooldown. | Separate trigger/active/exit states, bounded duration and consistent cooldown. Reverse remains programmed assistance. |
| `--motor-head linear` | Already bounded native outputs are passed through another spline, reducing endpoint authority to about 0.71. | Diagnostic identity transfer. Does not activate inactive genome KAN genes or change the checkpoint. |

All five interventions are independently selectable and **off by default** until
paired physical validation. Default motor math is covered against the historical
formula. None of these changes teaches braking by itself.

For the distance defect, at `dt=1/64 s`, a 3 m bin change creates an apparent
approach rate of 192 m/s. The channel normalizer clips at 30 m/s. The recordings
therefore contain only `{-127,0,127}` in channels 102–104 and 106–107. Their TTC
estimate consequently alternates between no urgency and near-immediate urgency.
The continuous-probe test approaches a planar wall at 10 m/s and checks that
finite differences remain 10 m/s across voxel boundaries.

The old frontal path samples only through 30 m but returns 50 m for a miss.
The continuous mode observes and caps at 50 m consistently; its change includes
this horizon correction as well as removal of coarse bins. It is not a pure
quantization-only experiment. Rotation, changes of obstacle identity and genuine
corners can still produce distance discontinuities; exact tracing is not a full
solution to temporal filtering or safe stopping-distance estimation.

Channel 118 means front clearance, increasing with free space; channel 123 means
minimum **lateral** clearance. Channel 105 is minimum TTC across the three probes,
not exclusively frontal TTC. Constant channels in a recording are not necessarily
unwired. Engine temperature already exists at channel 71 and varies here; future
body work should extend its existing model, rather than duplicate it. Tire
temperature is constant in these recordings and requires checking the selected
physics variant before drawing a general conclusion.

The retina digest has recurrent state, but its extra state does not become part
of the 192-channel native input on this path. Its CPU consumers include life
drain and boundary feedback. Exposing additional digest state is a separate input
contract/evolution experiment, not a completed fix in this checkup.

## Follow-up: confirmed GPU perception defects

Two defects in the retinal-motion pass were confirmed after this report's first
version. `perception_motion.wgsl` dispatches one 11-thread workgroup per being,
but it indexed the being with `global_invocation_id.x`. Local rows 1-10 therefore
read and wrote ten different beings instead of the same being as row 0. The pass
now indexes the population with `workgroup_id.x`; all eleven sensor families in a
workgroup share one being.

Channels 21-27 are historical overwrite slots. The spatial pass writes wall
probes first and the motion pass owns their final retinal-motion values. Both
passes previously used `atomicOr`, which merged unrelated signed-byte bit patterns
instead of replacing the old measurement. The motion pass now atomically clears
and writes only its owned packed byte, preserving adjacent channels. The racy
partial clear in the spatial pass was removed.

These defects corrupt inputs that can support anticipation and braking, but their
repair does not prove that the frozen checkpoint will brake. Existing replays were
produced by a binary containing the defects and remain historical evidence. A new
GPU run and subsequent training are required to measure behavior under the fixed
sensor contract.

That new paired checkpoint run was subsequently executed through Vulkan llvmpipe;
see `docs/BRAKE-CHECK-2026-09-15.md`. The repaired default completed at least one
lap for 15/16 clones, but selected-being raw brake remained nonpositive in all 550
samples above 10 m/s. Signed throttle reduced collision energy by 30.81% without a
material distance loss, while no isolated intervention created a braking policy.

## Native brain in the existing replay

The old error occurred because the Brain 3D pane supported FlyWire anatomy only,
while this recording uses the native controller. The exporter now accepts:

```text
replay_view --replay gen0000.replay --html replay.html \
  --diagnostics report/report.json \
  --native-genome inputs/champion.bin --native-being 0
```

The checkpoint bytes must match `checkpoint_sha256` in the replay metadata.
`run_champion.py` now supplies these arguments automatically. The functional 3D
layout separates wave, sensory KAN, sensory CfC, L1, L2, motor transfer, slow
inputs, slow CfC, slow hidden units and slow outputs. Focused connections use
the actual checkpoint matrices. Positions are schematic, not fly anatomy;
stored weights are not measurements of effective signal flow through gates and
fatigue. Decay rate `d` is displayed in s^-1 and time constant as `1/d` seconds.
Unrecorded activations remain unavailable. No animation invents propagation.

`target/checkup-20260915/native-preview.html` is a small four-second preview using
samples from the existing run, not a recording of the new experimental controls.
Regenerate it with `python3 scripts/replay/build_native_preview.py` after building
`replay_view`. Its source replay/checkpoint hashes are retained in the metadata.
The full `/media/llm-data/universe-runs/universe-replay.html` alias has not been
replaced during this checkup: that filesystem is read-only in this session.

## Validation and remaining work

Verified locally:

- `cargo test --release --lib`: 469 tests passed; 16 hardware-GPU tests were ignored.
- `cargo test --release --test wgsl_validation`: all motion and WGSL validation
  tests passed, including population mapping and packed-channel replacement.
- `cargo test --release --bin gpu_evolve`: 101 tests passed after integration.
- `cargo test --release --lib gpu::slow_brain::tests`: uniform offsets, bytes,
  WGSL parse and type validation passed.
- `cargo test --release --lib replay::player`: exporter/trace contracts passed.
- `python3 -m unittest discover -s scripts/replay -p 'test_*report.py'`: analysis
  coverage, dead-being exclusion, recorded gate attribution and independent
  experiment flags passed.
- Native graph and existing replay/retina Node tests passed, including camera
  projection bounds at desktop and narrow sizes. These are not browser screenshots.

Not verified in this session:

- Actual AMD hardware execution: the sandbox does not expose `/dev/dri`. Full
  kernel validation and the paired brake battery did run through Vulkan llvmpipe;
  hardware performance remains unmeasured.
- Visual browser rendering: the browser tool rejected the local `file://` URL
  under its security policy. No alternative browser path was used to bypass it.
- Training convergence under the repaired sensor contract. The new frozen-checkpoint
  battery measures control behavior but contains no evolutionary generation.

After GPU and artifact storage are available, rebuild the release binaries and
run the frozen baseline plus **one intervention at a time**, paired on the same
checkpoint, population seed and tracks 8005/8200/8006. Use `run_champion.py` with
the flags above and `--driving-diagnostics`; all arguments and hashes go into
each manifest. The first smoke test should use a short horizon; full evaluation
uses 30,000 ticks. Do not combine repairs before measuring their individual effects.

Review actual commands, high-speed raw brake, collision energy, impact ticks/km,
time stopped and the population distribution alongside laps. Then train under
the selected contract, with physical collision consequences and held-out tracks.
Compare first/middle/final generations across independent evolution seeds before
claiming learning. The old frozen champion and a lap-only gate cannot supply that
evidence.

## Follow-up: replay layout and apparent disconnections

The delivered full HTML contains both complete stylesheets inline (no external
CSS dependency). A missing stylesheet was not reproduced. The former layout
placed the header, long topology explanation and all inspection controls over
the canvas. They now occupy separate grid rows; inspection details scroll and
the long explanation is collapsed. Short screens scroll the scene area instead
of compressing the brain beneath overlapping controls. Browser rendering is
still unverified under the restriction described above.

The native renderer previously coloured exactly zero weights as positive edges.
It now omits them and reports incoming/outgoing connections and omitted zeros
for the selected unit. Initial focus is L1 unit 0, where sensory mixing happens.
The graph shows all 758 functional units, but only the selected unit's stored
incident mappings. It does not draw recurrent state loops, normalization or all
runtime consumers. Unrecorded activation remains grey.

CPU counts from the verified checkpoint (SHA `40ba4111...8599fa`):

| Quantity | Recorded checkpoint |
|---|---:|
| L1 zero weights / total | 9,994 / 18,432 (54.22%) |
| Entirely zero L1 rows or columns | 0 / 0 |
| Nonzero L2 weights / total | 4,608 / 4,608 |
| L1 inputs to each of the three motor rows | 96 |
| Nonzero L1 paths from channels 64, 65 and 82, each | 48 |
| Nonzero L1 paths from channels 102–107, 118, 119 and 123, each | 24 |
| Slow gather slots with a source | 30 / 48 |

These are structural paths, not measurements of causal sensitivity after
thresholds, quantization and saturation. One-to-one sensory KAN/CfC stages do
not disconnect the controller: mixing follows in L1, then L2
(`src/gpu/brain.wgsl`, stages 1 and 2).

Only outputs 0–2 directly drive standard actions. Outputs 3–4 can request gear
changes in neural gearbox mode (`context.rs`). Crucially, the recorded
`calibrated120-8005/run.log` confirms L2 standard-deviation normalization is ON:
all 48 pre-activation values contribute to the shared denominator
(`brain.wgsl:944`). Thus rows 3–47 are not wholly inert; their weights and memory
parameters can change motor amplitude. Their final activation-function genes
have no motor effect in this mode because activation follows normalization.
All seven non-default L2 activations in this checkpoint occur in those rows.

Other confirmed gaps remain: the legacy slow gather labels wave 33/32 as
speed/alignment instead of 65/64; the existing canonical experiment corrects
these two slots. Sensory KAN coefficients are fixed identity; output KAN genome
knots are stored but not mutated or executed; gate genes mutate even when
regional neuromodulation is disabled. These are limitations in expressed
capacity and search efficiency, not evidence that the entire network is unable
to learn. Changing their contracts requires a separate paired experiment.

The displayed recording has `--gens 1 --resume --resume-mix 1`, 16 checkpoint
clones and no intervening evolutionary generation. `main.rs` calls `evolve()`
only when another generation remains. Re-exporting this recording cannot
improve its braking. This follow-up changes presentation only; it does not
claim improved control or a new GPU run.
