# GPU tick synchronization and resident-state migration (16/09/2026)

**Current status:** the shared-readback boundary and repeated track-projection removal are
implemented and measured. Resident physics, GPU command/survival/progress bookkeeping, the
summary ring and the final single submission are phases 2–6 below; they are not implemented
yet and the current loop still performs mandatory CPU/GPU synchronization each tick.

## Why the per-being slope was the first speed lever — measured, not argued

The owner sees `GPU% 99`, 41–49 W of a 200 W cap, MCLK at 456 MHz (memory idle) and asks
for more batteries in parallel. The numbers say parallelism is not the lever:

| config (32 beings × 12 000 ticks, frozen binary `gpu_evolve-f5d203b`) | s/generation per run | throughput vs one process |
|---|---|---|
| one process (from the fitted law below: 1.89 ms/tick × 12 000) | ~22.6 | 1.0× |
| 8 processes (battery B5 wave 1, 15/09) | **106** | **1.7×** |
| 12 processes (B5 wave 2, 16/09 09:00–09:07) | **150** | **1.8×** |

`docs/KERNEL_TIMING.md` (27/08) fitted **ms per tick ≈ 0.86 + 0.032 × n_beings** over a
12/48/192/768-being sweep, found the kernels are ~13–16 % of the tick and the brain ~2 %,
and that **~85 % of the tick scales with n_beings**: the per-being CPU readback inside
`PhysicsGpu::step` and the per-being Rust loops in `context.rs::tick()`. The card idles
~85 % of every tick and downclocks, which is exactly what the owner's `rocm-smi` shows.

Two consequences:
- More processes: each one blocks on its own GPU context; the driver serialises them. 8
  processes buy 1.7×, 12 buy 1.8×. CPU is 82 % idle; every process sits at ~19 % of one core,
  i.e. waiting.
- Bigger populations in one process: 256 beings = 0.86 + 0.032 × 256 ≈ 9.1 ms/tick ≈ 109 s
  per generation — the same as 8 × 32 today. The per-being term is linear, so batching beings
  into one process buys nothing either.

The baseline's dominant scaling lever was **0.032 ms (32 µs) per being per tick**. On the
GPU all beings are one dispatch; on the CPU they are a loop and a readback. The later sweep
also exposed a substantial fixed term, so both slope and intercept must be reported rather
than treating either one as the whole runtime.

## Verified inventory (16/09/2026)

Static audit of `src/bin/gpu_evolve/context.rs`, `main.rs` and every read method. Line
numbers below refer to commit `1ba72b5`; guards and consumers are authoritative even when
subsequent edits move the lines.

| line | transfer / wait | exact guard | CPU consumer | required in the same tick? |
|---|---|---|---|---|
| 1888 | first `device.poll(Wait)` | always | originally none: it happened before all `map_async` requests | no; replaced in phase 1 by one poll after the three normal requests |
| 1895 | diagnostic sample staging | `--driving-diagnostics` and `tick % vision_every == 0` | writes the selected being's retina/wave/brain trace | no; observation only |
| 1953 | whole perception wave | `--audit-vision` | temporal/channel audit | no; observation only |
| 1970 | `final_intent`, 96 f32/being | every `brain_fires` tick unless direct control | motor head, pilots, fly fallback, replay, activation and physical command assembly | **yes in the current architecture** |
| 1992 | slow output, 16 f32/being | every tick unless direct control | throttle/steer/brake modulation, regional gates and metrics | **yes in the current architecture** |
| 2002 | fly motor command, 3 f32/being | fly rate or spiking backend | validates and replaces native motor commands | **yes when enabled** |
| 2111 | retina digest, **19** f32/being | every native tick | wall/life costs, retinal edge memory; slots 3..18 are currently copied but not consumed | **yes in the current architecture** |
| 2221/2240 | `brain.download_debug()` | formula/forge diagnostic pilots | CPU diagnostic controller | yes when that diagnostic controller is selected |
| 2633 | `brain.download_debug()` | `--observe` and `brain_fires` | observation file | no; observation only |
| 2680 | `hip_mem[0].read_sector()` | hippocampus observation | reads a **CPU resident** structure | not a GPU readback |
| 2713 | whole perception wave | `--record-wave` and `brain_fires` | open-loop observation file | no; observation only |
| 2761 | `brain.download_cfc_state()` | `--activation`, sampled stride | L1 activation metrics | no; observation only |
| 2837 | `PhysicsGpu::step` output, one struct/being | always | canonical body state, yaw rotation, contact/damage and next-tick sensors | **yes in the current architecture** |
| timer | timestamp staging | sampled `--kernel-profile` tick | profiler rows | no; observation only |

There is another mandatory dependency outside `GpuCtx::tick`: `main.rs` consumes every
being's post-physics CPU state every tick for odometer, track projection, checkpoints,
lap detection, life earning, holes, joy and fitness accumulators. Therefore removing the
physics readback from `tick` without migrating these consumers only moves the same barrier
one stack frame upward.

## Actual dependency boundary

The original requested encoder order cannot be recorded today. `head` and `physics` inputs
do not exist until Rust has read `final_intent`, the slow output and the digest, then applied
KAN/head policy, hippocampal bias, steering smoothing, reverse/gear overrides, damage and
survival rules. The next retina dispatch also needs the CPU copy of the new pose and temporal
body channels. One submit therefore requires a GPU-resident canonical body/survival state,
plus GPU kernels for command arbitration and track progress. Merely making
`PhysicsGpu::step` accept an external encoder would encode stale commands.

The migration is split at boundaries that can be compared against the frozen reference:

1. **Shared readback wait (implemented):** request native intent, slow output and retina
   digest together; service all three with one `poll(Wait)`; retain the old convenience APIs
   for tests and other binaries. Arithmetic and byte layout are unchanged.
2. **Resident physical state:** persistent input/output buffers, yaw and state writeback in
   WGSL, with an old/new first-divergence harness. Keep one readback per tick until parity is
   proven; this isolates math migration from cadence migration.
3. **GPU command assembly and survival:** consume `buf_intent`, `buf_output` and digest
   directly; port the default production path first. Optional diagnostic pilots remain an
   explicitly slow path.
4. **GPU progress/lap/fitness bookkeeping:** port the per-tick consumers in `main.rs`; expose
   a compact summary. Only after this phase may physical state stay resident across ticks.
5. **Readback ring:** double/triple-buffer summaries every 64 ticks and drain replay and
   diagnostics asynchronously. A death bit that affects whether the next tick runs must
   remain GPU resident; the CPU learns it at the next summary boundary.
6. **Single submission:** record retina → perception → brain → slow → command → physics →
   bookkeeping in one encoder. Bind groups are then persistent for the generation.

Phase 1 reduces redundant synchronization but is not expected to meet the target slope by
itself. Claiming `< 0.005 ms/being` before phases 2–6 would confuse a synchronization cleanup
with the requested architecture.

Batching summaries must not change episode semantics. A dead being freezes on the GPU at the
same tick as the reference path; progress, fitness and damage stop accumulating immediately;
the CPU may discover the death up to 63 ticks later, but delayed discovery cannot grant extra
movement, life or fitness. Generation termination occurs at the first summary boundary after
all beings are dead, and the summary reports each being's exact death tick.

## Implementation and measurement log (16/09/2026)

### Shared readback wait

`src/gpu/readback.rs` now provides a two-phase map ticket. Native intent, slow output and
the retina digest request their mappings before one shared `device.poll(Wait)` and decode
after it. Existing one-shot read methods remain compatible with tests and other binaries.

This change is bit-identical but **performance-neutral**, not a completed optimisation.
Same commit, same RX 6750 XT, 40 samples per population:

| version | fitted tick time |
|---|---|
| before shared map requests | `14.3604 + 0.056394 × n_beings` ms |
| after shared map requests | `14.2507 + 0.060005 × n_beings` ms |

The difference is noise. The original poll already waited for the submission; later polls
mostly serviced callbacks for completed copies. The ticket API is useful groundwork for a
future ring, but it does not remove the mandatory CPU dependency.

### Repeated track projection found during the sweep

The current code had grown seven curve horizons plus curvature risk. Each helper projected
the same position again, and nested helpers projected twice. At 768 beings,
`prep.racing_line` had risen to **23.4115 ms median** (40 samples), much larger than the
older 3.742 ms measurement cited by the previous plan.

`RacingLine::project_at` now returns a reusable `Projection`; every racing-line quantity
for a being consumes that one projection. No formula, search order or float expression in
the individual quantities changed. The six-generation frozen reference remains byte for
byte identical for all champion genomes.

| beings | tick median before (ms) | tick median after (ms) | racing-line after (ms) |
|---:|---:|---:|---:|
| 12 | 14.6986 | 11.6141 | 0.0571 |
| 48 | 17.0528 | 12.6035 | 0.2033 |
| 192 | 26.2264 | 17.8448 | 0.8697 |
| 768 | 60.2296 | 33.8454 | 3.1452 |

New fit: **`ms/tick = 11.5092 + 0.029285 × n_beings`**. This removes 51% of the measured
marginal slope in this checkout, but it does not satisfy the `< 0.005` acceptance target.
The remaining target still requires resident command, physics, survival and progress state.

Commands and raw logs:

```text
env -u DISPLAY VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
  ./target/release/gpu_evolve --out <dir> --gens 1 --beings <12|48|192|768> \
  --ticks 400 --width 28 --out-beta 0 --tracks 8005 --seed 77 --kernel-profile 10

/media/llm-data/scratch/codex-tick-sync-baseline-sweep-20260916/
/media/llm-data/scratch/codex-tick-sync-sweep-20260916/
/media/llm-data/scratch/codex-tick-sync-projection-sweep-20260916/
```

Frozen-reference command and result:

```text
./target/release/gpu_evolve --gens 6 --beings 24 --ticks 3000 --seed 7 \
  --same-start --wall-cost 0 --impact-cost 0 --out <dir>

gen0000..gen0005: IDENTICAL by cmp against
/media/llm-data/scratch/costgate-smoke2/c_no_cost/
```

Validation status:

- `cargo test --release --lib`: 481 passed, 16 ignored.
- `cargo test --release --test wgsl_validation`: 12 passed.
- real-GPU `retina_digest_parity`: passed on RADV/NAVI22.
- `cargo test --release`: one unrelated pre-existing failure,
  `sensitivity_gate::the_committed_genome_fixture_is_the_seeded_genome`; reproduced on
  clean commit `1ba72b5` in a detached worktree.
- `cargo fmt -p universe --check`: pre-existing diff in
  `src/bin/gpu_evolve/objective_cost.rs`; all files touched here were formatted directly.
- `BASE_REF=HEAD bash scripts/check_language.sh`: passed.

## Remaining migration, in order

1. **Inventory** — the table above, completed: for every GPU→CPU read in the tick, what the
   CPU actually does with the bytes, and whether it is needed THIS tick or only at the end
   of the generation / every N ticks (fitness, logging, replay, stagnation).
2. **One submit per tick.** Encode retina → perception → brain → slow → head → physics in
   one command encoder; no `poll(Wait)` inside the tick. `PhysicsGpu::step` stops opening its
   own encoder.
3. **Move the per-being loops to WGSL**: steer/stagnation bookkeeping, yaw rotation, state
   write-back, and whatever the inventory shows the CPU computes per being between stages.
   Bind groups built once per generation, not per tick.
4. **Read back only what the CPU needs, batched**: fitness inputs and death flags every N
   ticks (N = 64 is a natural window; the driving diagnostics already use 64), replay and
   diagnostics through a ring buffer drained asynchronously (the shape already agreed in
   `KERNELS-OBSERVATION-L1L2.md`).

## Acceptance

- **Bit-identical.** The reference smoke — `--gens 6 --beings 24 --ticks 3000 --seed 7
  --same-start --wall-cost 0 --impact-cost 0` — reproduces the six champion genomes of
  `/media/llm-data/scratch/costgate-smoke2/c_no_cost/evolve_best_gen000{0..5}.bin` byte for
  byte. If a change of order makes that impossible (e.g. stagnation now updated on GPU in a
  different float order), say so explicitly, show the first diverging tick, and give a new
  reference produced by the old and new binaries on the same seed with the difference
  quantified.
- **Measured with the same sweep.** 12/48/192/768 beings, 400 ticks, `--kernel-profile`,
  median of ≥ 25 samples as `KERNEL_TIMING.md` requires. Report the new fit
  `ms/tick = a + b × n_beings`. Target: `b` from 0.032 to **< 0.005**. At that slope one
  process with 256 beings runs at ~2 ms/tick and a 25-generation battery of 8 × 32 goes from
  ~45 min to under 10.
- `cargo test --release` and the GPU parity tests stay green.

## Not asked

Changing any kernel's arithmetic; the brain; the physics model. This is about where the
work runs and when the bytes cross the bus.
