# Body, brain and visual diagnostics

The current body is a vehicle in the voxel world. The FlyWire-derived rate and
spiking models are alternative neural controllers for that body, not a simulated
fly with wings, legs or anatomically complete sensory processing.

## Runtime ownership

| Responsibility | Owner | Lifetime |
|---|---|---|
| World geometry and physical state | `VoxelWorld`, `PhysicsGpu`, `GpuCtx` | World / episode |
| Packed retinal pixels and sensory wave | `GpuCtx`, `PerceptionGpu` | GPU resident |
| Neural weights, connectome, recurrent state | Selected GPU brain implementation | Genome / episode |
| Resident sensor view and normalized vehicle actions | `brain_interface` | Borrowed per tick |
| Brain dispatch and command adaptation | `gpu_evolve/brain_backend.rs` | Shared encoder |
| Reduced visual coverage and proximity | `gpu/retina_metrics` | Optional sampled diagnostics |
| CSV recording | `gpu_evolve/visual_telemetry.rs` | Optional output stream |
| Playback and activation display | `fly_map_view_support` | Offline standalone HTML |

The retinal buffer contains 1,764 packed `u32` values per being. Each value stores
distance in millimetres in bits 8–31 and material in bits 0–7. Material 4 means
another visible body; it is not a voxel material. The sensory wave's channel
semantics are registered in the existing `Channel` definition; dimensions come
from configuration constants. The model borrows these buffers instead of copying
pixels through RAM. Dispatch ordering remains retina → perception → brain.

Vehicle actions are throttle `[0,1]`, steering `[-1,1]` and brake `[0,1]`. These are
normalized controls, not force, torque or neural firing rates. The body applies
physics. Missing or nonfinite brain outputs invalidate the run and produce a
nonzero exit status. The final motor conversion also has a defensive braking
fallback. A dead being is stopped regardless of the selected brain.

## Choosing the controller

Existing commands retain their legacy hybrid behavior. Fly outputs in that mode
are combined with the native controller's downstream transformations. Scores from
that mode must not be attributed solely to the connectome.

`--brain-control direct` with `--fly-brain` or `--fly-spiking` selects direct Fly
control. It skips native and slow neural dispatch/readback and bypasses their
downstream KAN/hippocampus motor shaping. Model readout configuration remains part
of the Fly controller. Incompatible controller/activation flags are rejected.
Native buffers are still allocated by the legacy context constructor; eliminating
that allocation is separate from eliminating execution.

To add a brain for this body, implement resident dispatch, three-channel readout
and episode reset through the adapter and context. Test a small CPU reference against
the GPU implementation and check reset isolation. A different body needs a new
action schema and sensory mapping; reusing three driving controls for wings or
limbs would hide an incompatible interface. This is a versioned vehicle contract,
not a claim that all bodies already share the same motor space.

## Viewing a run

Set `UNIVERSE_RETINA_TELEMETRY` to a new CSV path. The simulation reduces retinal
coverage and body proximity on the GPU every 64 ticks, then reads 32 bytes per
being per sample. This diagnostic does not feed the controller. Positions and
applied commands reuse the host state already present in the loop.

For spiking activation, also pass `--fly-spiking-trace <new-file.fspike>` and
`--fly-spiking-trace-every 64`. The trace records the selected being's actual
membrane potentials and accumulated spike counts. Generate the viewer with:

```sh
target/release/fly_map_view --map /path/to/map.bin \
  --trace /path/to/brain.fspike --telemetry /path/to/telemetry.csv \
  --html /path/to/brain-body.html
```

The viewer samples graph nodes/edges and arranges them by cell type. Its graph
coordinates are not anatomical positions. Only the traced being has activation
data. Body trajectories are measured world coordinates. The default alignment
uses 156 LIF substeps per Universe tick; supply `--trace-steps-per-tick` if that
integration configuration changes. Trace recording is optional and transfers
substantially more data than compact retinal diagnostics.

## Remaining transfers and scientific limits

The full loop still contains CPU work and synchronous GPU readbacks, including
physical state and motor commands. The new interface does not remove those
existing round trips. `--kernel-profile 64` reports recorded kernel timings and
host phases; `scripts/fly/profile.py` samples process CPU/RSS and global
DRM utilization counters without GPU readback. Global utilization is not a
per-kernel or per-process GPU measurement.

The current transfer budget includes these payloads (N = population size):

| Payload | Bytes | Frequency |
|---|---:|---|
| Physics input, including WGSL padding | `368 × N` | Every tick |
| Physics output | `224 × N` | Every tick |
| Fly motor commands | `12 × N` | Every tick |
| Native compatibility intent readback | `384 × N` | Native firing ticks; skipped in direct mode |
| Slow output readback | `12 × N` | Legacy mode; skipped in direct mode |
| Retinal diagnostic reduction | `32 × N` | Every 64 ticks when enabled |
| Spiking voltage/count trace | `8 × map_neurons` | Per recorded frame, one selected being |

These are payload sizes from the host layouts, not measured PCIe throughput or a
complete accounting of every upload. Physics still has a separate submission and
blocking readback. The next residency refactor must keep physical state in its
GPU buffer and bind that state directly as the next tick's sensor source; it must
also move the remaining host-side orientation/body-signal calculations before a
claim of removing per-tick transfers is valid.

CSV coverage measures visible retinal pixels. Proximity uses center distance in
the horizontal plane with the retinal body diameter, not collision impulses.
Low progress, life-budget exhaustion and premature exhaustion are separate
endpoints. See `FLY-BATTERY-2026-09-14.md` for the paired experiment and its limits.
