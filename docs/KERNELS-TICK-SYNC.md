# Request to Astra: take the per-being work and the per-tick readbacks off the CPU (16/09/2026)

## Why this is the only speed lever left — measured, not argued

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

The only thing that moves throughput is the **0.032 ms (32 µs) per being per tick**. On the
GPU all beings are one dispatch; on the CPU they are a loop and a readback.

## What runs synchronously every tick today

`src/bin/gpu_evolve/context.rs` (absolute lines at `f5d203b`):

| line | call | per tick? |
|---|---|---|
| 1888 | `device.poll(wgpu::Maintain::Wait)` | yes |
| 1953 | `percept.read_after_submit` (whole wave buffer) | confirm |
| 1970 | `brain.read_intent_after_submit` | confirm |
| 1992 | `slow.read_outputs` | confirm |
| 2002 | `fly_backend.read_commands_after_submit` | only with `--fly-brain` |
| 2111 | `read_after_submit(.., extra_bytes)` | confirm |
| 2680 | `hip_mem[0].read_sector()` | confirm (off with `--no-hippocampus`?) |
| 2713 | `percept.read_after_submit` (audit path) | only with `--audit-vision` |
| 2837 | `phys_gpu.step` — own encoder, own submit, `poll(Wait)`, map, per-being copy (`src/gpu/physics.rs:155-174`) | yes |
| 1895 | `driving_diagnostics.read_sample` | only on sampled ticks |

"confirm" means I could not establish the guard by reading; the first deliverable is the
authoritative version of this table.

## The ask, in order

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
