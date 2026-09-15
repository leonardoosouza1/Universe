# Reproducible navigation replay

Record a saved native champion, keeping the evolved time constants. This runs one
evaluation with exact checkpoint clones in different formation positions, fuel
enabled, no pilot, and no Fly-controller substitution. It does not train a new brain.

```sh
cargo build --release --bin gpu_evolve --bin replay_view
python3 scripts/replay/run_champion.py \
  --checkpoint /path/to/evolve_best_gen0014.bin \
  --out /media/llm-data/universe-runs/NEW-REPLAY \
  --track 8005 --ticks 30000 --beings 16
```

The output directory must be new. The runner freezes the checkpoint/executables,
records commands and hashes, then exports `replay.html`. It exits unsuccessfully
unless at least 80% of beings complete two full circuits, measured from recorded
positions. A failed recording remains available for inspection and is marked as
failed. Run GPU jobs sequentially.

`--track`, `--seed`, `--width`, `--ticks`, `--beings`, `--min-laps` and
`--min-fraction` are explicit controls. A short recording is useful for checking
motion but will intentionally fail the lap gate. Passing on one track does not
establish generalization to other tracks.

The shared behavioral validator lives in `scripts/fly/validate_replay.py`. It
uses the exported track polyline, nominal width, tick rate and sampling interval;
it ignores fitness and the legacy lap counter. Its JSON records tolerances and
the SHA-256 of the unannotated input HTML. Keep replay, metadata, manifest and
validation report together. This validates sampled trajectories, not precise
voxel contact or every physics substep.

For browser verification, using local Chrome and Python `websocket-client`:

```sh
python3 scripts/replay/browser_check.py /path/to/replay.html --output /path/to/browser-check
```

This checks actual being displacement during playback, pause, displayed verified
laps and viewport fit. It writes screenshots and `browser-validation.json`.
The replay starts playing automatically unless reduced motion is requested in
the browser; Space pauses/resumes, and Follow tracks the selected being.

CPU tests: `python3 -m unittest scripts/fly/test_validate_replay.py -v`.
The separately invoked `scripts/replay/test_checkpoint_cli.py` requires a built
binary and GPU access; it checks that a corrupt checkpoint cannot silently
substitute a random population.

## Integrated brain, vision and driving study

Add `--driving-diagnostics` to `run_champion.py` to attach the real retinal pixels,
processed sensory channels, selected native CfC state, and exact physical contact
metrics to **the same `replay.html`**. `--vision-being` selects the recorded being;
`--vision-every 16` records at 4 Hz. Other beings remain available in the body replay,
but do not borrow this being's neural/retinal samples. `--erase-fast-memory` is a
partial CfC ablation, not a memory-free or untrained controller.

Native recordings can attach their actual controller to the same Brain 3D pane:

```sh
target/release/replay_view --replay /run/gen0000.replay \
  --diagnostics /run/report/report.json \
  --native-genome /run/inputs/champion.bin --native-being 0 \
  --html /run/replay.html
```

The exporter requires matching checkpoint SHA-256 metadata and an unmodified
generation-0 checkpoint clone (`checkpoint_mix=1`, no tau reseeding).
`run_champion.py` supplies the native attachment automatically. This is a functional
3D layout with exact stored weights, not anatomy. Sensory CfC and L1 are separate;
unrecorded neuron activations remain unavailable. The layer picker inspects
coefficients and the sampled values. Detailed recording is index-selected, not
automatic champion selection; the pane identifies its being and provides a return
button after another being is selected.

`checkup_report.py --run /run --output /run/control-checkup.json` summarizes
all-being physics and explicitly scoped neural samples. New recordings include
`body.motor_trace` (pre-physics speed, transfer, modulation, gating and final
commands), displayed as a stage table under Actions. Existing files cannot recover
those missing stages. Independent opt-in runner options are documented in
[`BRAIN-CONTROL-CHECKUP-2026-09-15.md`](../../docs/BRAIN-CONTROL-CHECKUP-2026-09-15.md).

For a small code-review artifact, `python3 scripts/replay/build_native_preview.py`
extracts the first four seconds of the existing calibrated recording into
`target/checkup-20260915/native-preview.html`. It verifies the checkpoint identity,
preserves actual frames/samples and labels the excerpt. It does not run the new
controls or validate driving improvement.

For a Fly recording with a measured spiking trace:

```sh
# First export the matching trace, including its clock and motor commands.
target/release/fly_map_view --map /path/to/map.bin --trace /run/brain.fspike --json /run/brain.json
python3 scripts/replay/export_brain3d.py --map /path/to/map.bin \
  --annotations /datasets/flywire/neuron_annotations.tsv \
  --trace /run/brain.json --out /run/anatomy
python3 scripts/replay/driving_report.py --run /run --output /run/report
# The complete 3D connectome is embedded into the existing scene player.
target/release/replay_view --replay /run/gen0000.replay \
  --brain3d /run/anatomy/brain3d.json --diagnostics /run/report/report.json \
  --html /run/replay.html
python3 scripts/replay/check_study_browser.py /run/replay.html \
  --output /run/browser --brain --vision
```

`Study` splits track and brain; `Brain 3D` maximizes the anatomical view. Drag to
rotate, Shift+drag to pan, wheel to zoom, click a neuron to inspect it. Search by
root ID or cell type, and use Next to traverse matches. The lower Vision / Behavior /
Signals panels share the replay's timeline. Hover a retinal pixel for its material
and distance. Brain colors show measured voltage/spikes, never an invented pulse.
Straight connection lines represent graph edges, not reconstructed neurite paths.

The exporter checks map/cache connectivity, sorted unique root IDs, and converts
4×4×40 nm annotation coordinates to micrometres. Missing positions remain missing.
All edges are indexed; at most 6,000 incident edges are drawn for a selected neuron.
The full export is about 196 MB, with an additional in-browser adjacency index;
allow several seconds for parsing/preparation. It does not require external JS/CDNs.

For a multi-generation diagnostic recording, pass `--generation N` to
`driving_report.py`. Reports must not combine different genotypes that reused the
same being index. Re-export the matching `genNNNN.replay` with that report.
The HTML exporter rejects brain/body command or observation-position mismatches.

Contact-energy metadata must contain `contact_energy_version: 2`. The first
instrumented batch included vertical ground clamping in that measurement; its
energy totals were retired and the controlled runs repeated. Other contact,
brake and motion counters were unaffected.


Road perception and terrain inspection are documented in
[`docs/ROAD-VISION-AND-TERRAIN-2026-09-14.md`](../../docs/ROAD-VISION-AND-TERRAIN-2026-09-14.md).
`--retina-road-view 120` is opt-in; calibration version 2 uses the 0.9 m eye height
above ground at rest, including body height. Earlier version-1 records with that
flag actually aimed at 270 m. The Vision panel computes the current geometric
horizon from the recorded sensor pose and displays measured returns separately.
**Aligned angles** corrects the foveal row's horizontal presentation using nearest
recorded rays, preserving neural input values; **Raw sensor columns** shows the
original array. Hover a yellow pixel to locate its measured hit on the map, and
use **Inspect hole** to focus the recorded terrain cells.
New recordings embed actual hole cells. In the replay, **Inspect hole** focuses them;
**Sensors** overlays wall probes and a subset of measured retinal rays when sensor
poses were recorded. `run_champion.py --retina-road-view 120` opts into corrected
surface geometry and longer ground sampling; it is experimental and regresses on
some tracks. The flag is not needed merely to display holes on the map.
