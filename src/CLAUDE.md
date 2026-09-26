# `src/` — library crate

Everything the simulator is made of. `src/lib.rs` is the module list; `src/main.rs` is
the Bevy visualizer, and `src/bin/` holds the headless tools. The core is compute —
Bevy is an optional window over it, so nothing under `gpu/`, `math/` or `analysis/`
may need a window to run.

## Map

| Module | What lives there |
|---|---|
| `config` | Constant hub. Brain, vision and genetics dimensions — the single source of truth |
| `gpu/` | GPU simulation core: one `wgpu` device, resident buffers, every kernel. See `gpu/CLAUDE.md` |
| `math/` | Pure numeric core (stats, linalg, quantization, activations, thresholds). **No GPU, no Bevy** |
| `analysis/` | Activation metrics, runtime alarms, region × activation crossing — built on `math` |
| `perception/` | `channels.rs` is the canonical registry of the 192 sensory channels; `retina.rs` the CPU retina |
| `world` | Voxel world, waypoints, racing line, lap distance |
| `pista_gen` | Track generation by *measured* difficulty (rejection sampling) |
| `engine` | `SovereignEngine` — the CPU-side container of one being's weights |
| `layers`, `float_layers`, `dense_layer`, `cfc`, `kan`, `neuro` | The brain's building blocks |
| `brain_capacity` | Participation ratio + the "is this gain real" test |
| `evolution/` | Bevy-path evolution: `.dna` format (`dna_genesis.rs`), generation pipeline, spawn, telemetry |
| `simulation/` | Bevy per-tick systems — the visualizer's own pipeline, separate from `gpu/` |
| `physics`, `environment`, `metabolism`, `safety/` | CPU-side vehicle, world and life mechanics |
| `ui`, `ui_menu`, `visualization`, `scenery`, `prefabs` | Bevy front-end only |
| `replay`, `rng`, `brain/format.rs` | Replay frames, the single deterministic RNG, the `.brain` container |

## Where **not** to put new code

1. **Not a new literal for an existing dimension.** Channel counts, layer widths and
   retina size come from `config.rs`; channel *indices* come from
   `perception/channels.rs`. A second copy is how `brain.wgsl` once read the sensory
   wave with the wrong stride.
2. **Not a second genome parser.** `gpu/genome_io.rs` is canonical; import it. A
   duplicate parser produced a plausible-looking, silently wrong report once already
   (`gpu/genome_io.rs:1-4`).
3. **Not into `simulation/motor.rs` (3 382 lines) or `world.rs` (1 519).** They are
   already over the 600-line cap and are on the split queue in `docs/STYLE.md` §2.
   New code goes into a new module under the cap.
4. **Not GPU code outside `gpu/`.** Kernels, buffers and `wgpu` types live there so the
   device stays shared; a second device forbids sharing buffers between subsystems.
5. **Not Bevy or `wgpu` imports inside `math/`.** That module exists precisely so the
   numeric core can be tested without a GPU, and `analysis/` depends on that.
6. **Not in `perception_old_backup.rs`.** It is not in `lib.rs` and nothing references
   it — a leftover, not a module. Do not extend it; deleting it is a fine PR.

## Traps

- `SENSORY_WAVE_SIZE` is **computed** in `config.rs:331`, not written down. Comments
  used to quote the old value 223 (`gpu/brain.rs:37` said `223 × 96 = 21.408`, corrected
  at `gpu/brain/mod.rs:94` when the file became a directory;
  the real product is `192 × 96 = 18 432`). Read the constant, not the comment.
- `rng.rs` is the one deterministic generator. Its header records the sign bug that
  motivated it: an initialiser that came out 100% negative, copied across 7 fields,
  collapsed three outputs into one (correlation 0.9999).
- Anything that changes the *distribution* of a brain input invalidates existing
  champions. Gate it behind a default-off flag and measure a paired A/B.

## Commands

```bash
cargo test --release --lib                # library tests; GPU ones are #[ignore]d
cargo test --release --lib -- --ignored   # the GPU parity tests, if you have a GPU
cargo test --release --lib wgsl_static   # every runtime WGSL parses + type-checks (naga)
bash scripts/check_language.sh
```
