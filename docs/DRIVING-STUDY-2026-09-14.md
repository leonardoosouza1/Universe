# Integrated driving study and scientific audit — 2026-09-14

The replay is now the study interface: track/body, complete indexed FlyWire brain
in 3D, measured retina, neural channels and driving diagnostics share one clock.
This establishes observability. It does not establish competent autonomous driving.

## Open the recordings

- Fly, integrated anatomy + real retina + motor signals:
  `/media/llm-data/universe-runs/driving-audit-20260914/validated/fly-study/replay.html`
- Native checkpoint, track 8005 + real retina + CfC state + contact metrics:
  `/media/llm-data/universe-runs/driving-audit-20260914/validated/baseline-8005/replay.html`
- The same checkpoint on track 8200:
  `/media/llm-data/universe-runs/driving-audit-20260914/validated/baseline-8200/replay.html`

These are separate experiments displayed by the same frontend. Fly activation is
never overlaid on the native controller's driving result.

## Findings and fixes

1. **Native braking is rare, not physically absent.** A real GPU fixture checks
   that brake=0.5 slows the car more than coasting with the same initial wheel speed
   on a straight road. The recorded controller mostly produces nonpositive raw
   brake values, followed by the existing output spline/positive clamp and a
   low-speed brake gate. The default output spline is global; the stored genome's
   output knots are not the spline used by this executable. Do not diagnose this
   run from those unused genes. Slow-brain modulation directly biases throttle
   and steering, not the brake head.
2. **The native controller does use recurrent state.** Erasing the 96 fast CfC
   states before each decision reduces completed laps to zero in both tested
   tracks. The counter, per-neuron memory, fatigue and slow brain remain intact.
   This is evidence about this checkpoint's dependence on CfC state, not proof of
   learned curve recall, online learning, or a general memory-free comparison.
   Sector hippocampus was disabled in these runs.
3. **`--wall-cost` did not affect the default objective.** It subtracted from the
   distance accumulator, while checkpoint/prog/centro used other accumulators.
   The cost now applies after selecting the objective, including held-out scoring.
   Raw distance telemetry remains physical distance. Cost zero preserves behavior.
   A paired 4,096-tick test changed all 16 selection scores while producing identical
   body replay bytes. This verifies the correction, not a trained improvement.
4. **Exact contacts were missing.** The existing physics readback now carries
   probe contact mask, inward impact speed, normal impulse magnitude and wall
   resolution energy, only when diagnostics are enabled. The observer aggregates
   alive ticks, impacts, contact events, brake/throttle overlap, stopping, reverse,
   distance, slip and resources. CPU wall-query proximity remains a separate proxy.
5. **The first energy instrument was wrong.** Ground clamping leaked into its
   collision-energy calculation. A falling car with no wall reproduced a false
   6,075 J. The measurement now ends immediately after wall resolution, before
   ground clamping, and requires a positive wall impulse. The fixture now checks
   zero wall energy for that case. `contact_energy_version: 2` identifies valid
   data; the earlier energy totals are retired and all five runs were repeated.
6. **The blank 3D view had numerical/rendering bugs.** Missing coordinates made
   the camera bounds NaN; Z was not normalized consistently; edge endpoints were
   indexed from encoded strings; and a uniform color hid per-neuron activity.
   The new renderer excludes missing positions from bounds, uses calibrated XYZ,
   typed endpoint arrays and per-neuron activity attributes. Its adjacency index is
   built once; selecting a neuron does not rescan the whole graph each frame.

## What vision actually means here

The retina is **14 rows × 126 columns**, 1,764 ray samples over a nominal 120°
horizontal field. Its pitches in this run range from -0.012 to -0.06 radians;
row zero also uses the existing foveal warp/gaze behavior. Each pixel stores a
material code and distance in millimetres. It is not an RGB camera or 360° vision.
The false-color replay exposes those actual values; hovering gives the raw
material and distance. Material IDs are grass=0, road=1, wall=2, hole=3, being=4.
Other bodies are visible only when `--retina-beings` is enabled.

The three lines on the track are **wall probes**, not the complete retina.
The native brain receives a 192-channel processed wave. Some channels come from
body state and world-derived lookahead/curvature, so this is not purely visual
navigation. In particular, curvature information at several lookahead distances
must be assessed separately from retinal cues when making learning claims.
The Fly eye pathway maps this retina into the connectome's photoreceptor inputs;
that wiring is not by itself a trained driving policy.

## What the 3D brain represents

- 138,639 indexed neurons; 138,625 have a complete annotated soma or position.
  Fourteen remain explicitly unlocated.
- 15,091,983 signed, aggregated connections from the loaded map; these are not
  15 million individually reconstructed synapses.
- All endpoints are indexed; at most 6,000 incident edges are drawn for the
  selected neuron. Straight lines denote connectivity, not anatomical neurites.
- Annotation voxels use 4×4×40 nm units, converted to micrometres before rendering,
  following the [primary annotation API documentation](https://fafbseg-py.readthedocs.io/en/latest/source/generated/fafbseg.flywire.search_annotations.html).
- The map/cache CSR endpoints and weights are compared, root IDs remain exact
  strings, and map/root-index hashes are exported. The old map format itself does
  not embed root IDs; their identity still relies on the canonical companion cache.
- Colors use recorded membrane potentials/spikes. Only being 0 was traced in this
  recording. Samples are held from the latest earlier tick, with age shown;
  another being or a time before the first sample has anatomy without activation.
- Every neural sample's raw motor commands and clock are checked against the body
  replay. Sensory observations are checked against body positions at shared ticks.

## Controls and modularity

`Study` / `Track` / `Brain 3D` select the workspace. Drag rotates, Shift+drag pans,
and the wheel zooms. Click a neuron or search by root ID/cell type; Next advances
through matches. Vision / Behavior / Signals share the lower dock and timeline.

`assets/brain3d/brain3d.js` owns WebGL, camera, selection and adjacency. It knows
nothing about body physics. `assets/replay_player/study.js` aligns samples and
owns the diagnostic dock; the existing model/view still own body playback.
`gpu_evolve/driving_diagnostics.rs` observes without changing neural inputs.
The copy of retina/wave/CfC state is limited to one selected being at 4 Hz by
this battery's configuration: 7,632 additional bytes per sample, about 30.5 kB/s
of simulated time. Exact contacts reuse the existing body readback padding.
The hot loop still contains its pre-existing CPU/GPU synchronizations; this work
has not removed them. Profiling and asynchronous telemetry remain relevant.

## Scientific acceptance and limits

All native runs use the same checkpoint, 16 exact clones in different formation
slots, 30,000 ticks, width 28 m, fuel enabled, and no pilot. The two tracks are
8005 and 8200; only the partial memory ablation changes between paired arms.
These 16 clones are not 16 independent trained seeds. The results describe this
checkpoint and these two environments, not a population confidence interval.

Laps are checked from sampled trajectories independently of the fitness score.
A lap pass does not prove safe driving. Contact fraction, impact severity,
stationary time, fuel and untrained track performance must accompany it.

The Fly controller in this recording still brakes almost continuously and does
not complete a lap. Rendering its full network does not resolve its motor-policy
problem. No new trained champion is claimed by this battery.

Next experiments should prioritize anticipatory braking and pedal conflict,
then matched-state memory tests and held-out curves. Train with measured impacts
and progress as separate reported outcomes, check against a simple controller,
and preserve off-default flags for alternative action mappings/physics. Do not
reward brake usage itself: reward progress with fewer/severer impacts and verify
that the resulting policy brakes before a hazardous turn.

## Verification

- `cargo test --bin gpu_evolve --no-default-features`: 86 tests at the time of this audit.
- `cargo test --lib --no-default-features replay::player`: replay alignment tests.
- `cargo test --release --lib --no-default-features -- --ignored physics_contact_tests`:
  actual GPU braking/contact/ground-energy fixture, run exclusively.
- `target/release/gpu_validate`: before and after logs in the artifact root;
  its explicitly unmeasured subsystems are not counted as passes.
- `python3 -m unittest scripts/replay/test_driving_report.py scripts/replay/test_export_brain3d.py`.
- `python3 scripts/replay/check_study_browser.py ... --brain --vision`: real Chrome,
  visible pixels, changing activity, no future/foreign-being samples, orbit, pan,
  zoom, cell-type search, retinal dimensions, responsive viewport and screenshots.

Exact commands and executable/checkpoint hashes accompany each native run.
See `scripts/replay/README.md` for reproduction and `driving-summary.json` under
`validated/` for the final numerical measurements.

## Repeated native measurements (energy version 2)

All four body replays are byte-identical to their initial instrumented counterparts.
The percentages below use **alive ticks**, not all scheduled ticks. Braking means
applied brake >0.05; stopped means speed <0.5 m/s. An impact tick is not necessarily
an independent crash: repeated resolution of one contact can span several ticks.

| Condition | ≥1 lap | ≥2 laps | Braking | Physical contact | Stopped | Median distance |
|---|---:|---:|---:|---:|---:|---:|
| Native, 8005 | 16/16 | 15/16 | 0.542% | 2.121% | 16.082% | 6,239.90 m |
| Native, 8200 | 6/16 | 2/16 | 0.301% | 1.200% | 41.383% | 1,741.15 m |
| CfC erased, 8005 | 0/16 | 0/16 | 0.518% | 2.786% | 28.023% | 1,855.12 m |
| CfC erased, 8200 | 0/16 | 0/16 | 0.531% | 1.700% | 42.251% | 1,083.12 m |

The native baseline accumulated 2,516 and 3,711 impact ticks respectively.
After CfC erasure the counts were 5,634 and 5,695. These are observations of one
checkpoint; avoid treating the formation clones as independent evolutionary seeds.
The first track's lap count is insufficient evidence of readiness: collisions,
stopping and the second track's failures remain material problems.

Final browser evidence: 99,171 non-background brain pixels and 4,752 colors,
93,340 neurons with changed activity between sampled ticks, zero WebGL/runtime
errors; orbit/pan/zoom, cell-type search, foreign-being/future-sample exclusion,
and the 126×14 real retina passed. The native motion check measured 116.50 m
displacement during playback and verified pause plus three completed laps for
the selected being. Screenshot/JSON evidence lives in each recording's browser
directory. A two-generation held-out smoke also confirmed that reserved-track
observations do not leak into either training-generation report.

The repeated Fly run has zero wall-contact energy with zero physical wall impacts,
while brake >0.05 is active in 100% of alive ticks and speed <0.5 m/s in 68.76%.
Its body replay is byte-identical to the previous recording. This remains a
failed motor-policy demonstration, now inspectable with synchronized anatomy
and vision.
