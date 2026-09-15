# Universe — Architecture

> **Language policy (2026-08-17):** this project is **English-only** — identifiers,
> comments, doc-comments, log messages, test names and commit messages. The previous
> rule in `NAMING.md` ("identifiers in English, comments and output strings in
> Portuguese, do not translate any comment") is **revoked**. Existing Portuguese
> comments are legacy and are being migrated; new code must not add more.
>
> This file is the living architecture document. The old `ARQUITETURA.md` is a stub
> that points here.

**Scope of this document:** what a new contributor needs in order to change code here
without silently breaking an experiment. Every claim below was checked against the
source at the cited `file:line`. Where the code did not answer a question, this
document says so and marks a `TODO` instead of guessing.

The current body/brain contract, direct Fly control, sampled diagnostics and visual
playback are described in [Brain/body interface](docs/BRAIN-BODY-INTERFACE.md).
The primary scene replay and the boundaries for a future live viewer are documented
in [Replay observatory](docs/REPLAY-OBSERVATORY.md). Its binary recording, export,
brain-trace validation, browser model and rendering are separate modules.
The retinal host, encoding, Bevy resource and tests now live in separate
`src/gpu/retina*.rs` modules; `gpu/mod.rs` preserves their public re-exports.

---

## 1. What this project is

Universe is a neuroevolution simulator. Procedurally generated tracks are populated
with *beings* — vehicles driven by an evolved brain — and a genetic algorithm selects
the genomes that travel the furthest before running out of life. Retina raycasting,
perception encoding, neural dynamics and vehicle physics have `wgpu` compute paths.
The headless loop still prepares body signals and transforms motor commands on the
CPU, with synchronous readbacks between stages. GPU kernels do not yet imply a
fully GPU-resident simulation loop.

Rough size: 121 Rust files (~54.6k lines) under `src/`, plus 22 WGSL compute shaders,
plus 28 command-line binaries under `src/bin/`.

**There are two independent runtimes in this repository. Do not confuse them:**

| Runtime | Entry point | Purpose |
|---|---|---|
| **Headless GPU evolution** (the production path) | `src/bin/gpu_evolve/main.rs` | Runs generations without a window. This is where experiments are measured. |
| **Bevy visualizer / ECS simulation** | `src/main.rs` → `src/simulation/motor.rs` | Renders a race and runs its own CPU/GPU hybrid tick. |

They share constants (`src/config.rs`), layer types (`src/layers.rs`,
`src/float_layers.rs`, `src/cfc.rs`, `src/kan.rs`), the world generator
(`src/world.rs`) and the GPU kernels in `src/gpu/`, but **they do not share tick
code**: `src/bin/gpu_evolve/` contains no reference to `simulation::motor`. A fix
applied to one path does *not* reach the other. `src/gpu/mod.rs:1-12` states the
intent explicitly — compute is the core, Bevy is an optional window over it.

### Building

```bash
cargo build --release
cargo test --release --bin gpu_evolve      # genome serialization + ordering guards
```

> **TODO / build blocker for outside contributors.** `Cargo.toml` declares two path
> dependencies that live *outside* this repository:
> `iara-perception = { path = "../iara-runtime/iara-perception" }` and
> `iara-cfc = { path = "../iara-runtime/iara-cfc" }`. Nothing under `iara-runtime/`
> is tracked by this repo (`git ls-files | grep iara-runtime` → 0 files), so a fresh
> clone will not build without that sibling directory. This must be resolved
> (vendor, submodule, or publish) before external PRs are practical.
>
> The workspace also includes `docs/arquivo/dirs/iara_genesis` as a member — see the
> comment at the top of `Cargo.toml` for why that crate lives under `docs/`.

---

## 2. The simulation tick

One tick is `GpuCtx::tick()` in `src/bin/gpu_evolve/context.rs:299`. Default cadence
is 64 Hz (`dt = 1/64`), 8 000 ticks per generation, 60 beings
(`src/bin/gpu_evolve/main.rs:370-372`).

A tick is **two GPU submissions**, not one: everything up to and including the brains
goes into a single command encoder, then the host makes its motor decision, then
physics is submitted separately. Both submissions end in `poll(Wait)`.

```
                          ── HOST PROLOGUE (CPU) ──
  states[]  ──┬─ pose per being ─────────────────────► buf_poses_retina   (write_buffer)
              ├─ wall raycasts on grid_cpu (L/R/F) ──► buf_temporal[16-18]   ch118/119/123 feed
              ├─ racing-line offsets + corridor ─────► buf_temporal[10-15]   ch108,109,110,118,119,123
              ├─ proprioception (12 slots) ──────────► buf_temporal[20-31]   ch64..75
              ├─ reverse timer, diagnostic pilots ───► buf_temporal[32-48]
              └─ slip/joy/spin/stress/edge/steer-mem ► buf_temporal[49-57]   ch87,99,100,121,122,124..127
                          (context.rs:322-748 — all queue.write_buffer, no dispatch yet)

                    ── GPU ENCODER #1 · single submit (context.rs:752-838) ──
  (a) copy_retina_prev      buf_retina_out ─► buf_retina_prev        optical flow needs t-1  :784
  (b) retina.wgsl           grid + poses    ─► buf_retina_out         n × 1764 u32           :785
  (c) perception            clear(buf_output) then spatial → temporal → motion passes        :796
                            reads buf_retina_out, buf_retina_prev, buf_temporal
                            writes buf_output  (48 u32/being = 192 packed i8 channels)
  (d) retina_digest.wgsl    reads buf_retina_out ─► extra_out (19 f32/being)                 :807
                            = min_wall_ahead, edge_row6, edge_row7, + 16 perception-CfC slots
  (e) brain.wgsl            reads buf_output, buf_l1w/l2w/kan/tau/neuron_meta/gates          :827
                            writes buf_intent (96 f32/being); resident buf_cfc state
                            ── only if `brain_fires` (see brain_skip, context.rs:824)
  (f) slow_gather.wgsl      gathers 48 channels out of buf_output                            :834
      slow_brain.wgsl       CfC-64 → Dense(64→16) → Dense(16→3) ─► 3 f32/being               :835
  (g) queue.submit + device.poll(Wait)                                                       :838

                          ── HOST READBACK + DECISION ──
      brain.read_intent_after_submit(n)      96 f32/being (or reuse last_intents)            :858
      slow.read_outputs(n)                   3 f32/being                                     :867
      M4 gate computed on CPU ─► brain.upload_gates()  (takes effect on the NEXT dispatch)   :871
      retina_digest.read_after_submit(n)     19 f32/being → life / wall / stagnation / fuel  :921
      guard: intents.len() != n ─► abort tick, return false                                  :983
  ►►  PILOT-BY-FORMULA OVERRIDE — must be here (see the ordering invariant below)            :999
      phys_states assembly: KAN eval → slow-brain modulation → steer smoothing →
      reverse / lockout branches → hidden damage → gearbox → raw overrides                   :1044

                    ── GPU ENCODER #2 · physics, own submit (physics.rs:646) ──
      physics_a.wgsl  suspension + engine + aero ─► MidState
      physics_b.wgsl  Pacejka tyre loop (4 wheels) ─► MidState2
      physics_c.wgsl  collision + integration ─► output ─► readback ─► submit ─► poll(Wait)

                          ── HOST EPILOGUE ──
      copy physics back into states; integrate yaw from av_y and renormalize fwd/lat  :1341
        (the physics kernel writes `out.fwd = fwd` unrotated — rotation used to live
         in the Bevy ECS, so headless has to do it by hand, context.rs:1343-1361)
      memory0 = intents[i][0] ─► update_memory0 ─► buf_temporal (channel 120)         :1365
```

### Ordering invariants inside the tick

These are not stylistic. Each one was a real, measured bug.

* **`retina_digest` runs after the perception clear and before the brain**
  (`context.rs:799-809`). `perception` starts with `clear_buffer(buf_output)` because
  both perception kernels use `atomicOr` (`src/gpu/perception.rs:682`,
  `src/gpu/perception.rs:1050`) — zero + OR is the only way the packed bytes come out
  right. `retina_digest` writes into the *same* buffer region, so running it before
  the clear would erase its output; running it after the brain would leave the brain
  reading stale channels.

* **The pilot-by-formula override must precede the assembly of `phys_states`**
  (`context.rs:989-998`). Quoting the code: the block used to sit *after* the
  `.collect()`, and in that order the override never reached the motor — `phys_states`
  had already been built from the brain's intent, and the only thing the formula still
  touched was `mem0` (perception channel 120). The 2026-08-16 batteries therefore
  measured the champion with one contaminated memory channel instead of the clone
  driving: applied commands were bit-for-bit identical to the champion's for the first
  576 ticks while the recorded intent already differed by 0.60 from tick 0. The test
  `piloto_entra_antes_da_fisica` (`context.rs:1440`) locks the order by reading the
  source file itself and comparing the byte offsets of two marker comments
  (`MARCA-ORDEM-PILOTO` at `context.rs:1015`, `MARCA-ORDEM-FISICA` at
  `context.rs:1043`).

* **`copy_retina_prev` happens before the retina dispatch** (`context.rs:784`) —
  otherwise the optical-flow channels (21-27) compare a frame against itself.

---

## 3. Module map

| Module | Responsibility | Main file |
|---|---|---|
| `config` | Constant hub — single source of truth for brain/vision/genetics dimensions | `src/config.rs` |
| `gpu` | GPU simulation core: one `wgpu` device, all resident buffers, all kernels | `src/gpu/mod.rs` |
| `gpu::core` | Shared device/buffer manager; per-tick byte budget documented in the header | `src/gpu/core.rs` |
| `gpu::perception` | Host for `perception{,_temporal,_motion}.wgsl`; packs 192 i8 channels | `src/gpu/perception.rs` |
| `gpu::retina_digest` | Host for `retina_digest.wgsl`: perception CfC + `min_wall_ahead` + edge rows, replacing a 340 KB/tick readback | `src/gpu/retina_digest.rs` |
| `gpu::brain` | Fast brain host: L1 (192→96) + L2 (96→48) + KAN, resident CfC state | `src/gpu/brain.rs` |
| `gpu::slow_brain` | Slow brain host (System 2): CfC-64 → Dense(64→16) → Dense(16→3), 2 chained kernels | `src/gpu/slow_brain.rs` |
| `gpu::physics` | Vehicle physics host — three chained kernels (see §6) | `src/gpu/physics.rs` |
| `gpu::evolution` | `Genome`, tournament selection, gaussian mutation, elitism, spectral L1 init | `src/gpu/evolution.rs` |
| `gpu::genome_io` | **Canonical** genome (de)serializer — `save_genome` / `load_genome` | `src/gpu/genome_io.rs` |
| `gpu::cortex_sheet` | Equipotential cortical sheet + Hebbian plasticity (used by `cortex_evolve`) | `src/gpu/cortex_sheet.rs` |
| `perception::channels` | Canonical registry of the 192 sensory channels — kills magic numbers | `src/perception/channels.rs` |
| `world` | Voxel world, track generation, waypoints, racing line, lap distance | `src/world.rs` |
| `pista_gen` | Track generation by *measured* difficulty (rejection sampling) | `src/pista_gen.rs` |
| `engine` | `SovereignEngine` — the CPU-side container of a being's weights | `src/engine.rs` |
| `layers` / `float_layers` / `dense_layer` | `BitLinear` (fast brain, f32 runtime), `FloatLinear` (no bias, i8 input), `DenseLayer` (with bias) | `src/layers.rs` |
| `cfc` / `kan` | Liquid (closed-form continuous-time) cell; KAN output head with spline knots | `src/cfc.rs`, `src/kan.rs` |
| `math` | Pure numeric core with no GPU/Bevy dependency (stats, linalg, quantization, thresholds) | `src/math/mod.rs` |
| `analysis` | Activation metrics, alarms, region×activation crossing | `src/analysis/activation.rs` |
| `brain_capacity` | Participation ratio and the "is the gain real" test (Garimpo law) | `src/brain_capacity.rs` |
| `evolution` | Bevy-path evolution: DNA format, generation pipeline, spawn, telemetry | `src/evolution/pipeline.rs` |
| `evolution::dna_genesis` | Source of truth for the `.dna` file format (V8 TLV; reads legacy V1/V2/V3) | `src/evolution/dna_genesis.rs` |
| `simulation` | Bevy per-tick system (the visualizer's own pipeline) | `src/simulation/motor.rs` |
| `safety` | Sentinel FCA — emergency braking / soft flags | `src/safety/sentinel.rs` |
| `brain::format` | `.brain` container format (TLV, per-block table, layers can be disabled) | `src/brain/format.rs` |
| `replay` | Replay frame recording and playback | `src/replay.rs` |

Inside the production binary `src/bin/gpu_evolve/`:

| File | Responsibility |
|---|---|
| `main.rs` | CLI parsing, generation loop, fitness, curriculum, reports (2 582 lines — the largest file here and a good refactor target) |
| `context.rs` | `GpuCtx`: owns the device and every buffer; `tick()` lives here |
| `types.rs` | `BeingState` — per-being CPU state (distinct from `gpu_loop`'s own `BeingState`; do not unify) |
| `statics.rs` | Process-global CLI flags read inside the hot loop, with an explicit writer/reader ownership table |
| `pilots.rs`, `piloto_linear.rs`, `piloto_forja.rs` | Diagnostic drivers that replace the brain behind a flag, to separate "body cannot" from "brain cannot" |
| `escada.rs` | Difficulty-ladder controller — a pure state machine, no I/O, no globals |
| `fantasma.rs` | Ghost of the previous champion: time pressure per sector, paid in life, never in fitness |
| `pistas.rs` | Deterministic track rotation (empty set = fixed seed 9001, bit-identical to the old hardcode) |
| `audit.rs` | 3-axis vision audit (`--auditar-visao`) |
| `painel.rs` | Optional real-time HTTP panel (`--painel`); zero cost when off |
| `tests.rs` | Genome round-trip regression tests — **do not delete** |

`.brain` (`src/brain/format.rs`) is defined and tested but **not yet wired into the
evolution loop**: nothing outside that module constructs a `BrainFile`. The format
that is actually written to disk today is `UGEN v2` (next section).

---

## 4. The genome

### What is inherited

`Genome` (`src/gpu/evolution.rs:93`) is everything selection can act on. A being with
a field missing from this struct cannot pass that trait to its children.

| Field | Size | Meaning |
|---|---|---|
| `l1` | 192 × 96 = 18 432 f32 | Fast-brain layer 1 (instinct), `SENSORY_WAVE_SIZE × L1_NEURONS` |
| `l2` | 96 × 48 = 4 608 f32 | Fast-brain layer 2 (cortex) |
| `kan_knots` | 3 × 12 = 36 f32 | KAN output head — throttle / steer / brake curves |
| `slow` | 1 091 f32 | Slow brain, in `SlowBrain::all_weights()` order: dense1 w(1024) + b(16), dense2 w(48) + b(3) |
| `tau` | 96 f32 | Per-channel CfC time constants ("bank of τ") |
| `sensors` | 6 f32 | Sensor genes (off by default → `SENSOR_DEFAULTS`) |
| `neuron_alpha` | 96 + 48 f32 | Per-neuron memory (P1.7 mechanism 1) |
| `neuron_period_l1` | 96 f32 | Per-neuron firing period (P1.7 mechanism 2) |
| `neuron_threshold_l1` | 96 f32 | Per-neuron threshold (P1.7 mechanism 2) |
| `neuron_activation` | 96 + 48 f32 | Heterogeneous activation function selector (P1.7 mechanism 3) |
| `gates_w` | 16 f32 | M4 neuromodulation "conductor" gates |
| `hip` | 29 f32 | C1 sector hippocampus: 27 weights of `W` + write rate + read gain |
| `fly` | 200 + `n_types` f32 | FLY MAP selection: 192 bindings + 6 efferent slots + gain + scale + pruning mask. EMPTY unless `--fly-map` loaded a map (`docs/FLY-MAP-BRAIN.md`) |

Why `slow` is in the genome and not just in `SlowBrain::new()`: with fixed seeds every
being would be born with identical slow weights, so modulation would be a constant for
the whole population — evolution could neither tune it nor discard it
(`src/gpu/evolution.rs:105-111`).

Selection itself runs on the **CPU**, once per generation: tournament (k=3), gaussian
mutation (σ default 0.05, p 0.02), elitism. The cost is in the ticks, not in the
selection (`src/gpu/evolution.rs:1-25`).

### Serialization — format `UGEN v2`

`src/gpu/genome_io.rs:53` (`save_genome`) / `:89` (`load_genome`).

```text
magic    4 bytes  = b"UGEN"
version  4 bytes  = u32 LE = 2
l1                N_L1_WEIGHTS      × f32 LE
l2                N_L2_WEIGHTS      × f32 LE
kan               3 × KAN_OUT_KNOTS × f32 LE
slow              1091              × f32 LE   (optional — read only if the file is long enough)
tau               96                × f32 LE   (optional)
sensors           6                 × f32 LE   (optional)
neuron_alpha      (96+48)           × f32 LE   (optional)
neuron_period_l1  96                × f32 LE   (optional)
neuron_threshold  96                × f32 LE   (optional)
neuron_activation (96+48)           × f32 LE   (optional)
gates_w           16                × f32 LE   (optional)
hip               29                × f32 LE   (optional)
fly               200 + n_types     × f32 LE   (optional, ALWAYS LAST — its length is
                                                not a constant, so it is whatever
                                                remains in the file)
```

Legacy files (written before the header existed) start directly at the `l1` bytes.
`load_genome` sniffs the first 4 bytes: exactly `UGEN` → v2, otherwise legacy from
offset 0. The collision probability of four random f32 weight bytes spelling `UGEN`
is ~2⁻³² per file (`genome_io.rs:24-35`).

Every optional block that is absent falls back to a **documented neutral default**,
never an invented value — a genome saved before the slow brain existed genuinely did
not carry one (`genome_io.rs:129-131`).

### ⚠ There is exactly one canonical parser — do not write a second one

From the module header, `src/gpu/genome_io.rs:1-4`:

> *MOVED out of the `gpu_evolve` binary into the library (2026-08-14, P1.9): `genoma_view`
> kept a DUPLICATE parser that broke silently when the P1.6/P1.7 blocks entered the
> format (it read `neuron_activation` believing it was τ — reporting "spread 0.0×" for a
> champion whose real spread was 3.2×). The canonical parser lives HERE; binaries
> import it.*

`src/bin/gpu_evolve/genome_io.rs` is now a 4-line shim re-exporting the library
module, and `src/bin/genoma_view.rs:41` carries the matching note. If you write a tool
that reads a `.bin` genome, **import `universe::gpu::genome_io`**. A second reader
produces a plausible-looking, silently wrong report — which is worse than a crash.

### Rules when you touch `Genome`

1. **Never reorder the fields of `Genome` or the write order in `save_genome`** — the
   file has no field names, only positions. Reordering corrupts existing `.bin` files
   in silence (`genome_io.rs:37-41`).
2. **Every new field requires updating `save_genome` AND `load_genome` together.** This
   has already been forgotten once: 1 091 slow-brain weights were dropped on every
   `--carregar`, so generations of selection over modulation were discarded with no
   symptom at all (`genome_io.rs:55-60`).
3. Run `cargo test --release --bin gpu_evolve` afterwards. The test
   `genoma_sobrevive_ida_e_volta_no_disco` (`src/bin/gpu_evolve/tests.rs:24`) compares
   both sides of the disk field by field, and marks each block with a value that
   *cannot* come from the default — otherwise "passed" could just mean "both sides fell
   into the same default", which is precisely the original bug.
4. `population_seed_for` (`genome_io.rs:203`) exists because the initial-genome seeding
   used to live in three separate branches and only one was seeded — producing a whole
   battery of "three seeds" that were the same population, and a paired 25σ effect that
   was one sample counted three times.

---

## 5. Invariants that break silently

Ranked by how expensive the failure was, and how invisible it is. Each one has a
guard; if you remove or bypass the guard, nothing will crash — the numbers will just
be wrong.

**1. Pilot override before physics assembly** — `src/bin/gpu_evolve/context.rs:989`
(guard: `context.rs:1440`, markers at `:1015` and `:1043`).
If the override block moves after `phys_states` is collected, the formula pilot's
commands never reach the motor; only perception channel 120 is affected. Symptom: a
full battery reports the clone driving while the applied commands are bit-identical to
the champion's.

**2. Genome field order and the single canonical parser** — `src/gpu/genome_io.rs:37`
(guard: `src/bin/gpu_evolve/tests.rs:24`).
Reordering fields, or adding a field to `Genome` without touching both `save_genome`
and `load_genome`, silently corrupts or drops weights on reload. Already cost 1 091
weights once. A duplicate parser elsewhere in the tree costs a wrong report
(`genome_io.rs:1-4`).

**3. `buf_output` must be zeroed before the perception kernels, and `retina_digest`
must sit between that clear and the brain dispatch** —
`src/gpu/perception.rs:682` and `src/gpu/perception.rs:1050` (`ZERAGEM OBRIGATÓRIA`),
ordering enforced by call site `src/bin/gpu_evolve/context.rs:799-809`.
Both perception kernels write with `atomicOr`; leftover bits from the previous tick
survive without the clear. Related and already paid: until P1.8, `retina_digest` wrote
channels 192-207 into `buf_output`, which is exactly 48 words per being with no slack,
so those writes landed on the *next* being's channels 0-15 and corrupted its FAR/NEAR
pool. The stride is now a locked contract — `EXTRA_OUT_STRIDE = 19`
(`src/gpu/retina_digest.rs:88`), guarded by `extra_out_stride_e_dezenove`
(`src/gpu/retina_digest.rs:309`): change it on one side only and the reader walks into
the next being's data.

**4. Recurrent state must be reset at the generation boundary** —
`src/bin/gpu_evolve/main.rs:1355-1373`.
`buf_cfc` (recurrent state) and `buf_fatigue` live on the GPU and were never zeroed
between generations, so a child inherited its parent's CfC state in silence and the
champion's re-evaluation was not deterministic. Same family, found later in the P1.9
validation battery: `last_intents` and `brain_tick` also crossed the boundary, so the
first ticks of a new generation reused the previous generation's last decision. All
four are reset together at that site — plus `ctx.reset_cfc(n)` and
`ctx.brain.reset_cfc(i)` per being.

**5. Sensory channel indices are a contract between CPU arrays and WGSL kernels** —
compile-time assertions at `src/bin/gpu_evolve/context.rs:402-409` (racing line),
`:446-459` (proprioception) and `:666-676` (extra2), against the single registry
`src/perception/channels.rs`.
The host pushes flat arrays (`rv`, `prop`, `extra`, `extra2`) whose *position* decides
which channel a value lands in — and the mapping is not linear (`rv[0..3]` feed
ch108/109/110 but `rv[3..6]` jump to ch118/119/123). The first version of the registry
had ch124-127 in the wrong order. Note the honest limit stated in the code
(`context.rs:660-665`): those `assert!`s pin the *literals* against the enum, they do
**not** detect someone reordering the `push` itself — the runtime defence for that is
`--auditar-visao`, which measures per-index variance.

**Runner-up, worth knowing:** `reload_world` must also refresh the CPU-side copies
(`racing_line`, `grid_cpu`, `track_curvatura`) — `src/bin/gpu_evolve/context.rs:1407-1420`.
Forgetting `track_curvatura` makes `--regioes` measure the curvature of the *previous*
track against activations from the current one: valid-looking numbers, silently wrong.

### A note on reading the comments here

The comments in this repository are unusually valuable — many of them record a bug
that was paid for in GPU hours, with the measurement attached. They are also, in
places, **stale**. Verified example: `src/gpu/brain.rs:37` says
`// 223 × 96 = 21.408`, but `SENSORY_WAVE_SIZE` is `192` (`src/config.rs:309`), so the
real value is `18 432`. The constant is computed, so the code is correct and only the
comment lies. Treat any comment as a hypothesis until you open the `file:line` it
refers to. This rule is written into the project's own working notes.

---

## 6. Design constraints you cannot refactor away

* **Physics is three kernels on purpose** (`src/gpu/physics.rs:1-30`). The original
  756-line `physics.wgsl` made the RADV/ACO shader compiler (Mesa, NAVI22) segfault.
  Bisection (`src/bin/fisica_bisect.rs`) isolated the Pacejka tyre computation as the
  step that exceeds ACO's capacity. Kernel A (suspension + engine + aero) → Kernel B
  (Pacejka loop) → Kernel C (collision + integration), chained in one encoder, each
  keeping ≤ 50 live f32 locals. Merging them back re-introduces a driver crash.
* **The slow brain is two kernels for the same reason** (`src/gpu/slow_brain.rs:5-16`).
* **One `wgpu` device for everything** (`src/gpu/core.rs:1-14`). Separate devices per
  subsystem would forbid sharing buffers and force readback+reupload between retina and
  brain.
* **`map_async` → `poll` → `recv`, in that order** (`src/gpu/slow_brain.rs:40` and
  `:402-405`; the pattern itself at `src/gpu/brain.rs:706-708`). Polling before
  `map_async` is a silent deadlock.
* **GPU results must match the CPU path** (`src/gpu/mod.rs:29-34`): *a GPU path that
  answers differently in silence would invalidate every experiment already measured*.
  `src/bin/gpu_validate.rs` is the check. Run it before and after touching `src/gpu/`.
* **`src/bin/gpu_evolve/types.rs`'s `BeingState` is deliberately not the same type as
  `gpu_loop`'s** (`types.rs:1-8`) — 20 fields vs 5. Do not unify them.

---

## 7. Where the heavy data goes

The machine this project was developed on keeps simulation output on a **dedicated
90 GB partition, `/media/llm-data`**, and treats the system disk as source-code only.
This is not a preference: batteries in progress were killed when the system disk fell
to 340 MB free. The rule recorded in `CLAUDE.md` is that run outputs, replays,
snapshots, models and heavy scratch all go to `/media/llm-data/…`, never to `/tmp` and
never next to the source tree; the system disk keeps only source, docs and small text
logs.

**As an outside contributor you do not have that partition, and you do not need it.**
The output location is a flag, not a constant:

```bash
# --out defaults to "." (main.rs:395), so ALWAYS pass your own directory
./target/release/gpu_evolve --gens 5 --seres 60 --out /your/own/scratch/run-01
```

What lands in `--out`:

* `evolve_best_gen<NNNN>.bin` — the best genome of each generation, `UGEN v2`
  (`main.rs:1038`, `main.rs:1898`).
* `gen<NNNN>.replay` — replay frames (`main.rs:1249`); the directory is created if
  missing (`main.rs:1251`).

Replays are the expensive artefact: recording every generation × 16 races produced
31 GB once. For long runs record only the first, middle and last generation.

Two hardcoded absolute paths to `/media/llm-data` remain in the tree, both inside a
`#[cfg(test)]` block for the v2 formula pilot
(`src/bin/gpu_evolve/piloto_forja.rs:366-367`). That test declares honestly that
without the partition mounted it does not run and **that is not a PASS**
(`piloto_forja.rs:518`). No production code path depends on the partition.

---

## 8. Where to look next

| Question | File |
|---|---|
| What every CLI flag does | `src/bin/gpu_evolve/main.rs` (argument parsing at `:368`), `src/bin/gpu_evolve/statics.rs` (ownership table) |
| What each sensory channel means | `src/perception/channels.rs` |
| Which constants are authoritative | `src/config.rs` |
| Whether a GPU change is still equivalent to the CPU | `src/bin/gpu_validate.rs` |
| How difficulty is measured rather than guessed | `src/pista_gen.rs`, `docs/ESCADA_DIFICULDADE.md` |
| Planned refactors and their risk register (`R1`…`R8`) | `docs/REFACTOR_PLAN.md` |
| Evolution plan phases (`P0`…`P1.9`) referenced throughout the code | `docs/PLANO_EVOLUCAO.md` |
| Measured results from past sessions | `docs/`, `LEIA_PRIMEIRO.md` (Portuguese, legacy) |
