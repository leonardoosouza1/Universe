# `src/gpu/` — kernels and resident state

The compute core: one `wgpu` device, every buffer resident on it, every WGSL kernel and
its host. No surface, no swapchain, no window — it runs under `env -u DISPLAY`, in cron,
on a machine with no graphical session. Bevy is a viewer on top, never a requirement.

**The non-negotiable rule** (`mod.rs:29-34`): a kernel must give the *same* answer as the
CPU path. A GPU path that silently answers differently would invalidate every experiment
already measured. `src/bin/gpu_validate.rs` is the check — run it before and after
touching anything here.

## Files

Each `*.rs` is the host for the `.wgsl` of the same name: `mod.rs` (retina + the context
that survives between ticks), `core.rs` (shared device/buffer manager — per-tick byte
budget in its header), `perception.rs` (packs the 192 sensory channels),
`retina_digest.rs`, `brain.rs` (L1 192→96, L2 96→48, KAN), `slow_brain.rs`,
`physics.rs`, `cortex_sheet.rs`. Two have no kernel: `evolution.rs` (`Genome`, tournament
selection, mutation, elitism — selection runs on the CPU, once per generation) and
`genome_io.rs` (**the** genome serializer). Full responsibilities: `ARCHITECTURE.md` §3.

## Buffers and dispatch order

Buffers are owned per subsystem, on the one shared device: `GpuCtx` holds `buf_grid`,
`buf_poses_retina`, `buf_retina_out`; `PerceptionGpu` holds `buf_output` (the 192 packed
channels) and `buf_temporal` (host-fed slots); `BrainGpu` holds `buf_l1w`/`buf_l2w`/
`buf_kan`/`buf_tau`/`buf_neuron_meta`/`buf_gates` (uploaded once per generation) plus the
resident `buf_cfc` and `buf_fatigue`.

Order inside one tick — see `ARCHITECTURE.md` §2 for the full diagram:

```
copy_retina_prev → retina → perception(clear + spatial/temporal/motion)
                → retina_digest → brain → slow_gather → slow_brain
                → submit + poll(Wait) → host decision → physics_a → b → c
```

Three orderings are load-bearing, each of them a bug that was paid for:

- **`clear(buf_output)` must precede the perception kernels** — both write with
  `atomicOr` (`perception.rs:682`, `perception.rs:1050`), so leftover bits from the
  previous tick survive without the clear.
- **`retina_digest` sits between that clear and the brain dispatch.** Before the clear its
  output is erased; after the brain, the brain reads stale channels.
- **`copy_retina_prev` runs before the retina dispatch**, or optical flow compares a frame
  against itself.

## Traps

**One canonical genome parser.** `genome_io.rs` is it. `genoma_view` once kept a duplicate
that broke in silence when the P1.6/P1.7 blocks entered the format — it read
`neuron_activation` believing it was τ and reported "spread 0.0×" for a champion whose real
spread was 3.2× (`genome_io.rs:1-4`). If you read a `.bin`, import
`universe::gpu::genome_io`. Never reorder `Genome`'s fields or the write order: the file
has no field names, only positions. Every new field updates `save_genome` **and**
`load_genome` together — forgetting that dropped 1 091 slow-brain weights on every
`--carregar`, silently. Then run `cargo test --release --bin gpu_evolve`.

**Host and shader must agree on struct layout, and nothing tells you when they don't.**
WGSL aligns a struct to 16 bytes; Rust `#[repr(C)]` does not. `PhysicsStateGpu` is 360
bytes on the host and 368 in the shader's array, so `serialize_input`
(`physics.rs:157-166`) writes each element into a 368-byte slot with `INPUT_WGSL_STRIDE`
(`physics.rs:153-155`). Ship packed 360-byte elements instead and being 0 still reads
correctly while every being after it reads shifted garbage — no error, no panic, just
wrong physics. The same class of failure, already paid for: `retina_digest` used to write
16 extra words into `buf_output`, which is exactly 48 words per being with no slack, so
those writes landed on the *next* being's channels 0-15. The fix is a locked stride,
`EXTRA_OUT_STRIDE = 19` (`retina_digest.rs:88`), guarded by the test
`extra_out_stride_e_dezenove` (`retina_digest.rs:309`). Change it on one side only and the
reader walks into the next being's data.

Growing a uniform struct is the safe direction, and only that one: the host `WorldParams`
carries four climate fields the live kernels do not declare, and it works because
`min_binding_size: None` (`uniform_entry` in `core.rs`) lets the buffer be larger than the
shader's declaration — the kernels keep reading the original 16 bytes, bit-identically
(`physics.rs:284-291`). Shrinking, reordering or inserting a field in the middle is the
silent-corruption case.

**Physics is three kernels on purpose** (`physics.rs:1-30`). The original 756-line
`physics.wgsl` segfaulted the RADV/ACO shader compiler on NAVI22; bisection
(`src/bin/fisica_bisect.rs`) isolated the Pacejka tyre loop. Kernel A (suspension +
engine + aero) → B (Pacejka) → C (collision + integration), each keeping ≤ 50 live f32
locals. Merging them back reintroduces the driver crash. The slow brain is split for the
same reason (`slow_brain.rs:5-16`).

**Readback order is `map_async` → `poll` → `recv`.** Polling before `map_async` is a
silent deadlock (`brain.rs:706-708`, `slow_brain.rs:402-405`).

## Commands

```bash
cargo test --release --test wgsl_validation        # WGSL parse + type check, no GPU
cargo test --release --lib -- --ignored            # GPU parity tests
./target/release/gpu_validate                      # CPU vs GPU equivalence, before and after
```
