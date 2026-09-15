# Universe scene replay

The primary visual interface is the existing Universe replay, upgraded to show the
track and bodies as the main scene. The separate `fly_map_view` HTML remains a
diagnostic export. The scene renderer has no external assets or network dependency.

## Current integrated 3D study

The current interface is documented in [the driving study](DRIVING-STUDY-2026-09-14.md).
It embeds the complete indexed FlyWire 3D graph and real retinal samples inside the
existing replay. Open the recordings under `driving-audit-20260914/validated/`.
The older artifacts below retain their original evidence and UI.

## Earlier validated navigation replay

`/media/llm-data/universe-runs/replay-repair-20260914/native-release/replay.html`

This is the native neural champion: 16/16 beings completed one full circuit and
15/16 completed two or more in 468.75 simulated seconds. The scene starts playing,
identifies the controller and shows independently validated completed laps.
Space pauses/resumes; Follow tracks the selected being. Reproduction and the
failed second-track result are in [the repair report](REPLAY-MOVEMENT-REPAIR.md)
and [the recording commands](../scripts/replay/README.md).

## Earlier Fly calibration recording — not a navigation demonstration

`/media/llm-data/universe-runs/fly-observatory-20260914/universe-replay.html`

This recording contains eight beings, 512 world ticks, measured life/fuel and
motor commands, and eight sampled spiking frames for being 0. It is an integration
demonstration, not evidence of successful driving or evolution. The 40-being
primary population can also be inspected in
`/media/llm-data/universe-runs/fly-battery-20260914/population-replay.html`.
Older recordings lack availability metadata, so unsupported measurements are
shown as unavailable rather than interpreting historical zero/default fields.

The final recording completed in 72.91 s with exit status zero. Independent Chrome
checks loaded 128 body frames per being, verified exact neural alignment at tick
64, suppression for untraced beings, Follow, expanded brain view, pause/seek,
accelerated playback and 1600/768-pixel layouts without runtime exceptions.
The first displayed life/fuel values were 98%/99%, reflecting the first recorded
tick and the byte quantization. Evidence lives beside the HTML in
`run.json`, `browser-validation.json` and `universe-replay.png`.

`cargo test --release --lib --bin gpu_evolve --bin gpu_validate --bin replay_view
--bin replay_analyze --bin fly_map_view --test wgsl_validation` passed 452 library,
84 evolution, 3 validator, 8 diagnostic-viewer and 9 WGSL tests. All 14 explicitly
invoked GPU tests also passed; `cargo check --release --all-targets`, both JavaScript
syntax checks, the replay model tests, the language check and `git diff --check`
passed. Logs are under `/media/llm-data/scratch/scene-final-*` and
`/media/llm-data/scratch/observatory-*`.

## Generate a replay

```sh
cargo build --release --bin replay_view --bin fly_map_view
target/release/fly_map_view --map /path/to/map.bin --trace /path/to/brain.fspike \
  --json /path/to/brain.json --html /path/to/diagnostic.html
target/release/replay_view --replay /path/to/gen0000.replay \
  --brain /path/to/brain.json --width 28 --html /path/to/universe-replay.html
```

Omit `--brain` when no neural trace exists. The replay generator uses the binary
header's world seed and optional `gen0000.meta.json`; recorded width takes
precedence over the fallback CLI width. Track centerline/width are reconstructed
from the current world generator, not from the beings' trajectories. This drawing
does not depict exact voxel contact boundaries. Generator revisions can change
reconstruction; frozen source snapshots remain part of experiment provenance.

## Data ownership

| Module | Responsibility |
|---|---|
| `replay.rs` | Compatible 128-byte frame reading/writing and numerical analysis |
| `replay/player.rs` | World-coordinate export, availability, safe embedded JSON |
| `replay/player_brain.rs` | Recorded trace identity, clock and command alignment |
| `replay_view.rs` | Scene-export CLI |
| `assets/replay_player/model.js` | Frame lookup, camera transforms and sampled-state selection |
| `assets/replay_player/view.js` | Scene, selection, inspector, playback and interactions |
| `assets/replay_player/style.css` | Responsive visual layout |

The renderer keeps metres on both axes. Wall-probe lines use the recorded forward
vector and the documented left/right signs. These are sampled wall queries, not
the full retinal image. Body frames are post-physics; probe values and neural
commands belong to the input/action phase of that tick. The inspector identifies
this phase difference.

The native GPU simulation uses a remaining tick budget in the legacy HP byte.
Metadata identifies that resource explicitly. Fuel is measured when enabled;
disabled or unrecorded fuel is unavailable. Resource values have the legacy
1/127 quantization, so zero in the byte alone does not prove exact physical death.
New recordings capture raw motor commands instead of silently filling zeros.
Fixed-horizon life percentages use that run's initial tick budget; life-earning
mode uses its configured capacity. The previous writer always divided by the
life-earning cap, producing a misleading initial 26% for a 512-tick demonstration.

The legacy `--brain` export sampled nodes/edges by cell type. The current
`--brain3d` interface uses calibrated anatomical positions and the full graph index. Colors use recorded membrane voltages and counts. Only the traced being
has measured activation. Between neural frames, the last earlier sample is shown
with its age. No future sample or invented interpolation is used. Attaching a
brain requires the recorded trace path plus matching per-tick raw commands and
neural clock; a mismatched attachment fails before export. This is not a
cryptographic guarantee against a user replacing files at the same paths.

Binary compatibility was preserved while replacing implicit C-layout padding
with explicit bytes. A derived `Pod` check now prevents that unsafe layout from
returning. Invalid indices, truncated frames and malformed tags fail on open.

## Path to a live visual simulation

This delivery is replay playback. For live operation, keep the same scene/model
and replace the static frame source with bounded, versioned snapshots. The
simulation owns time and state; rendering must not step a second copy of physics.
Transfer compact body/resource/action snapshots at a selected display cadence and
sample brain diagnostics separately. A slow viewer must drop old snapshots rather
than block training or accumulate an unbounded queue. The existing native Bevy
loop and the headless GPU evolution loop have different controller composition;
sharing a visual layout alone does not make their results equivalent.

Acceptance for a live adapter: identical recorded/replayed snapshot values,
reported sample age and dropped frames, bounded host/VRAM use, and profiling with
the viewer disconnected, attached and deliberately slow. No per-neuron full-state
readback should become mandatory just to draw the world.
