# Road vision, navigation transfer and visible terrain — 2026-09-14

## Calibration correction and final verification — 2026-09-14

**This section supersedes the height, range labels and verification counts in the
original record below.** The original calculation omitted the resting body height:
the eye is 0.4 m above a body whose position is 0.5 m above ground. The reference eye
height is therefore **0.9 m**, and the legacy ground horizon is approximately **75 m**,
not 33 m. The first implementation of `--retina-road-view 120` actually aimed at
270 m at rest; its `80` setting aimed at 180 m. Their recorded behavior remains valid,
but those runs do not test the advertised ground ranges.

Calibration version 2 uses:

```text
reference_eye_height = 0.5 + 0.4 = 0.9 m
d_r = N * (4 / N)^(r / 13)
pitch_r = -atan(reference_eye_height / d_r)
unoccluded_ground_horizon = (recorded_body_y + eye_offset) / -tan(pitch_0)
```

Thus `N=120` samples nominal ground distances from 4 to 120 m at rest. The actual
body height varies during physics, so the panel calculates the horizon from the
recorded sensor pose, and reports measured ray returns separately. This does not
promise a 120 m return through an occluding wall.

The apparent discontinuity at the top of the retina had a separate presentation
cause: row 0 uses a tangent-based foveal distribution, while the other rows use
uniform azimuths. Stacking their raw column indices makes different directions
look vertically aligned. **Aligned angles** now maps each display column to the
nearest recorded ray at that body-relative azimuth. The original 1,764 values sent
to the brain remain unchanged. Raw columns are still inspectable. Recordings
without measured gaze retain the raw view; missing directions are never invented.

Yellow means the recorded Hole material. The legacy tracer could treat a Hole as
a vertical obstacle, producing a large yellow column. The opt-in physical tracer
tests the ground intersection instead. Hovering a pixel now projects its measured
hit onto the map and checks its cell against the recorded terrain. Chrome verified
a Hole reading at **92.379 m**, cell **63,183**, against the four recorded Hole cells
in the new 8005 scene. Re-exporting an old recording cannot repair its neural inputs.
An independent analytic projection checked all **3,518 Hole readings** across the
three calibrated recordings (898, 1,547 and 1,073). Every ground intersection matched
a recorded Hole cell, with no boundary ambiguities; the largest encoded-distance
error was below 0.001 m. Results are in `hole-ground-validation.json`.

Three further frozen-champion evaluations repeated the original protocol with
calibration version 2; the preregistration is `calibration-v2-plan.json`. These are
transfer evaluations, not new training or independent evolution replicates.

| Track | Legacy ≥2 laps | Calibrated 120 m ≥2 laps | Impact ticks/km, legacy → new | Contact duty, legacy → new | New braking duty |
|---|---:|---:|---:|---:|---:|
| 8005 | 15/16 | 16/16 | 27.14 → 20.17 | 2.12% → 1.94% | 0.22% |
| 8200 | 2/16 | 10/16 | 75.46 → 26.95 | 1.20% → 1.58% | 0.22% |
| 8006 | 16/16 | 16/16 | 22.20 → 20.18 | 2.37% → 3.12% | 0.37% |

Stationary duty is 14.29%, 20.98% and 11.83%, respectively. Contact duty rises on
two tracks despite fewer inward-impact ticks per kilometre. Braking is still rare;
this battery does not establish anticipation, safe driving or curve memory. Track
8200 still fails the preregistered 80% two-lap gate. **The profile stays opt-in.**
`transfer-summary.json` now includes all 12 evaluations and corrects the v1 labels.

The current example is
`/media/llm-data/universe-runs/universe-replay.html`, linked to
`calibrated120-8005/replay.html`. All three calibrated runs and their failures remain
available. `angular-export.json` records the viewer used for the final presentation;
the original manifest retains the binaries used during simulation.

The final default-path recording is byte-identical to the original 8005 baseline
(`final-default-identity/`). The real GPU fixture now checks **8,820 rays** across
five poses, with exact material agreement and distance error below 4 mm. It includes
a 58.272 m wall hit, negative half-cell boundaries and an open-field 120 m ground
intersection with the body at its actual 0.5 m resting height. See
`calibrated-gpu-geometry.log` and `calibrated-optics-tests.log`.

Additional viewer verification:

```bash
node scripts/replay/test_vision_angles.js
node scripts/replay/test_vision_projection.js
python3 scripts/replay/check_study_browser.py /media/llm-data/universe-runs/universe-replay.html --output /media/llm-data/universe-runs/navigation-improvement-20260914/angular-browser --vision
python3 scripts/replay/browser_check.py /media/llm-data/universe-runs/universe-replay.html --output /media/llm-data/universe-runs/navigation-improvement-20260914/angular-motion
```

Chrome confirms reprojection of the foveal row, unchanged other rows and raw sensor
values, the linked Hole hit, actual playback displacement, pause and independently
validated lap display. Pixel inspection is cleared when the sensory sample or
projection changes, so its distance and map marker cannot survive into a different
frame. Layout checks at 1800, 1280, 768 and 390 pixels verify the image, legend and
pixel readout fit their panels; a closed mobile inspector no longer widens the page.
The integrated fly anatomy remains a separate, stationary
driving result; it is not the controller in the successful native replay.

## Original record before calibration correction

The replay now draws actual Hole cells on the track and can focus them with **Inspect
hole**. Its Vision tab exposes measured materials, distances, row pitch, angular
sampling and the nominal ground horizon. A sampled retinal footprint is drawn on the
map when the recording contains the pre-physics sensor pose. It uses three of the 14
rows and every third column for readability; all 1,764 readings remain in the Vision
panel. Lane markings are still visual decoration, not a sensory channel.

The native controller completes laps in the new recordings. The anatomical fly
controller remains an unsuccessful driving experiment: the earlier 512-tick record
has braking above 0.05 on 100% of alive ticks. Anatomical activity is never attached to
a native-controller trajectory.

## Physical correction and compatibility

Legacy vision puts the eye 0.4 m above the body and looks down. At nominal ground
height its farthest pitch, -0.012 rad, intersects the ground at
`0.4 / tan(0.012) ≈ 33.33 m`. At 90 km/h this is 1.33 seconds of travel. The nominal
1,200 m packing limit does not mean the eye can observe the ground that far ahead.
Walls can occlude the ground even sooner.

The opt-in `--retina-road-view N` accepts a finite ground range of 20–300 m. It leaves
all independent wall, lateral and curvature probes at their existing ranges. For row
`r`, with 14 rows:

```
d_r = N * (4 / N)^(r / 13)       horizontal ground distance in metres
pitch_r = -atan(0.4 / d_r)       radians
```

The sensor still returns material plus distance, entirely on the GPU. A modular
surface tracer fixes three issues in the legacy geometry:

- DDA horizontal distances are converted to unit-ray distances before comparison with
  the ground intercept: `t_wall = d_horizontal / length(direction.xz)`.
- Holes are ground materials at the ground intercept, rather than opaque vertical
  columns. The world remains a height-field approximation; this does not render pit
  sidewalls or a pit bottom.
- DDA cells and ground sampling use the same centered-cell convention, and an occupied
  initial wall cell is tested before advancing.

All old optics remain behind the disabled branch. The 96-byte retina uniform reuses a
padding word as the profile gate, with no layout growth. There is no additional GPU
readback for this feature. Observations add the existing host pose to sampled JSON;
terrain is snapshotted once per recorded generation, never once per simulation tick.
The nominal ground horizon assumes body height zero; actual returns are measured and
may differ when the body moves vertically.

## Controlled transfer, including regressions

Nine one-generation evaluations used the same frozen native champion
`u1v_s077_ctrl/evolve_best_gen0014.bin`, SHA-256
`40ba411113cb7a3b005a4a234da944d3f40be324db9427b0a7730de6378599fa`.
Each contains 16 exact genome clones at distinct formation positions, 30,000 ticks
(468.75 s), width 28 m, fuel and holes enabled, and no hippocampus or pilot. These are
three environment seeds, **not** three independently evolved populations. No weights
were trained in this battery. The 80 m range was a subsequent engineering tuning
attempt after the 120 m regression; it is not independent confirmation.

| Track | Legacy: ≥2 laps | 120 m: ≥2 laps | 80 m: ≥2 laps | Legacy / 120 m impact ticks per km |
|---|---:|---:|---:|---:|
| 8005 | 15/16 | 16/16 | 16/16 | 27.14 / 24.01 |
| 8200 | 2/16 | 14/16 | 13/16 | 75.46 / 18.87 |
| 8006 | 16/16 | 11/16 | 11/16 | 22.20 / 36.97 |

Laps are independently validated from trajectory and ordered track sectors, not
inferred from selection fitness. Impact ticks per kilometre use physical contact
instrumentation and actual accumulated path length. They count ticks with inward
contact velocity, not distinct crashes. On track 8200 the 120 m controller is stationary
13.74% of alive ticks versus 41.38% before; its contact duty rises from 1.20% to 1.61%
while impact ticks per kilometre fall. This distinction prevents a misleading claim
that every collision metric improved. Braking remains rare (1.03% of alive ticks on
that 120 m track).

**Decision: neither new range becomes the global default.** There is useful transfer
on 8200 but a substantial regression on 8006. The next learning experiment must adapt
the controller to the corrected sensor geometry, with multiple evolution seeds and
an untouched evaluation set. A single hard-coded reflex or a track-specific choice
of winning replay would not establish general driving competence.

## Terrain provenance and replay inspection

New recordings embed integer Hole cell indices and cell size directly from the
simulation's `VoxelWorld::grid`, together with `no_holes` and provenance
`recorded_world_grid`. Four adjacent 3 m cells are drawn as their actual 6 m by 6 m
footprint. The track centerline/nominal-width ribbon is still a visualization, not an
exact wall collider mesh.

Older recordings can use `replay_view --reconstruct-terrain`, which requires an
explicit `no_holes` metadata boolean and refuses to overwrite an existing snapshot.
For the updated older scenes, that setting was recovered from the preserved run
command (no `--no-holes`, with holes enabled by default). Original raw recordings and
metadata are retained; the separate scene export records this provenance and viewer
hash. Missing terrain is labeled unavailable rather than silently rendered as empty.

Use the native 8200 replay for the improved driving result, the native 8005 replay for
another working circuit, and the fly study for anatomy and the failed fly motor output.
The native scene has no fly connectome attached.

## Verification and artifacts

Artifacts: `/media/llm-data/universe-runs/navigation-improvement-20260914/`.
Each arm includes its frozen binaries/checkpoint, exact command, hashes, raw replay,
contact CSV, sensory JSONL, independent lap validation and rendered HTML. Plans precede
execution (`plan.json`, `road80-plan.json`); `transfer-summary.json` includes all arms.
The initial HTML storage estimate was too low (verbose per-frame JSON dominates).
Redundant pre-validation HTML was removed, freeing 1,355 MiB, while retaining raw
recordings, metrics and all final HTML. `retired-intermediates.json` lists removals.
`run_champion.py` now retires that duplicate automatically after annotation succeeds.

Verified commands:

```bash
cargo test --release --no-default-features --lib vision_geometry
cargo test --release --no-default-features --lib gpu::retina_road_tests -- --ignored --nocapture
cargo test --release --no-default-features --test wgsl_validation
cargo test --release --no-default-features --lib visao
cargo test --release --no-default-features --lib replay::terrain
cargo test --release --no-default-features --bin gpu_evolve
node scripts/replay/test_vision_angles.js
```

The analytic fixtures pass. A real GPU compared 5,292 full rays with exact material
agreement and distance error under 4 mm, including a transverse wall seen at 58.270 m.
`gpu_validate` ran before and after the kernel change; only its measured sections are
claimed. The disabled-profile baseline 8005 raw replay is byte-identical to the prior
validated baseline, SHA-256
`de18e6ea167660d144c65bade58cb491b04e10744aeedf499bd66fefd12633da`.
Two-generation terrain smoke runs on tracks 8005/8200 record 4/8 Hole cells with holes
enabled and 0/0 with `--no-holes`. Actual Chrome checks inspect painted hole pixels,
playback displacement, pause, clock alignment, resize and the integrated anatomical
viewer. Browser evidence lives alongside the corresponding scene export.
