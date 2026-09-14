# Fly-map brain — the Drosophila connectome as a fixed brain in `gpu_evolve`

Status: design + implementation spec (14/09/2026). Owner: Leonardo's request of 14/09:
*"não quero gerar as coisas nela e sim já usar o mapa dela e ver como fica o cérebro dela
sendo usado aqui … primeiro a ideia de selecionar, depois o resto"*.

Principle: **the map is never generated or trained; it is loaded once and shared. Evolution
only SELECTS on top of it** — which perception channel drives which afferent cell type, which
efferent cell types are read as throttle / steer / brake, which cell types stay switched on
(pruning), and two scalar gains. This is the pruning finding of `neuromod-lab/docs/94`
(selection beats free mutation) applied to a real wiring diagram instead of 24 weights.

Python side (already built, `neuromod-lab/src/flybrain/`): loader, LIF reference (Shiu et al.
2024), rate model, group bindings, pruning operator, shuffled-map controls, replay reader,
`fly_export.py` (writes the binary below). This document specifies the Rust/WGSL side.

---

## 1. Map file `flymap` v1 (little-endian, written by `neuromod-lab/src/fly_export.py`)

```
magic      8 bytes  "FLYMAP01"
header     8 × u32  n (neurons), m (edges), n_types, k_in (afferent groups), k_out (efferent
                    groups), out_members_len, reserved, reserved
indptr     u32[n+1]     CSR rows = POSTSYNAPTIC neuron j: its input edges are indptr[j]..indptr[j+1]
indices    u32[m]       presynaptic neuron of each edge
weights    f32[m]       signed synapse count (+ excitatory, − inhibitory), as in FlyWire/Shiu
type_of    u32[n]       cell-type id 0..n_types-1 (8 866 types incl. class fallbacks)
in_group_of u32[n]      afferent group id 0..k_in-1, or NONE = 0xFFFFFFFF (267 groups = afferent
                        cell types with ≥ 4 neurons)
out_indptr u32[k_out+1] efferent group membership (516 groups = efferent cell types with ≥ 2 neurons)
out_members u32[out_members_len]
```
Sidecar `flymap_783.json`: type names, in-group names, out-group names, source SHA-256, filters
(`min_weight`, `exclude_optic`). Full map: n = 138 639, m = 15 091 983 (indices 60 MB, weights
60 MB). Variants for speed: `--min-weight 3` (4.9 M edges, 76 % of synapses), `--exclude-optic`
(61 109 neurons, 6.6 M edges). Default file: `/media/llm-data/datasets/flywire/flymap_783.bin`.

## 2. Dynamics (rate model, one step per tick; identical to `flybrain.dynamics.RateBrain`)

Per being b and neuron j (state `x`, output `r`, both `f32`; layout `x[j * B + b]`, being-minor, so
a thread that owns row j reads the CSR row once and serves every being):

```
acc   = Σ_{e ∈ row j} weights[e] · r_old[indices[e], b]
u     = gain[b] · wave_i8(b, k) / 127            if in_group_of[j] = g and group_to_input[b][g] = k ≠ NONE, else 0
x_new = x + α · (−x + scale[b] · acc + u)         α = 1 / tau_steps (uniform, default 4)
r_new = tanh(x_new) · type_mask[b][type_of[j]]    (mask ∈ {0, 1}: a pruned type emits nothing)
```
`r` is double-buffered (read `r_old`, write `r_new`, ping-pong bind groups per tick). Born
state x = 0. The three commands, per being:

```
cmd_c = mean_{i ∈ out_group plus_c} r_new[i] − mean_{i ∈ out_group minus_c} r_new[i]    c ∈ {throttle, steer, brake}
```
read back to the host each tick as `[B][3] f32` and, when the flag is on, they REPLACE the fast
brain's raw intents (`intents[i]`) before `kan_eval` — same downstream governor / sentinel /
clamps as today (mirrors the full-override pattern of `PILOTO_GANHO`, `context.rs` around the
`MARCA-ORDEM-PILOTO` marker; do not move those markers).

## 3. Genome block `fly` (new LAST optional block of the positional `UGEN` format, after `hip`)

`Vec<f32>` of length `192 + 6 + 2 + n_types` (n_types from the map header; the block is absent
when no map is loaded, length 0):

| slice | meaning | birth | mutation (only when `FLY_MUTATE_ON`) |
|---|---|---|---|
| `in_bind[192]` | afferent group id per perception channel, or −1 = unbound | random group with prob 0.25, else −1 | each slot rebinds with prob `fly_p_bind` (default 0.05): new random group, or −1 with prob 0.2 |
| `out_pair[6]` | (plus, minus) efferent group id for thr, str, brk | random | each slot redrawn with prob `fly_p_bind` |
| `gain`, `scale` | input gain and synaptic scale multipliers | 1.0, 1.0 | log-normal σ `fly_sigma` (default 0.1) |
| `type_mask[n_types]` | 1 = type on, 0 = pruned | all 1 | pruning op: off with prob `fly_p_off` (0.05), back on with prob `fly_p_on` (0.015) |

Rules copied from the hippocampus block (`evolution.rs`): a `pub static FLY_MUTATE_ON: AtomicBool`
(default false); the fly block is the **very last rng consumer** per child so a run without
`--fly-brain` draws the exact same rng stream as today; `EvoParams` gets `fly_p_bind`,
`fly_p_off`, `fly_p_on`, `fly_sigma`. `save_genome`/`load_genome` updated together; a genome
saved without the block loads with an empty block; round-trip test in `src/bin/gpu_evolve/tests.rs`.

Per generation the host packs, per being: `group_to_input[k_in]` (u32, NONE default; if two
channels bind the same group the higher channel wins), `out_pair[6]` u32, `gain`, `scale`,
`type_mask[n_types]` f32 — uploaded with `queue.write_buffer` like `upload_gates`.

## 4. GPU module `src/gpu/fly_map/` (host `mod.rs`, kernels `fly_map.wgsl`)

Buffers (resident, created once from the map file like `buf_grid`): `indptr`, `indices`,
`weights`, `type_of`, `in_group_of`, `out_indptr`, `out_members` (read-only storage; each
< 128 MiB, the `downlevel_defaults` binding cap — do not merge indices and weights into one
buffer). Per-being: `x`, `r_a`, `r_b` (`n × B` f32), `group_to_input` (`B × k_in` u32), `out_pair`
(`B × 6`), `gain_scale` (`B × 2`), `type_mask` (`B × n_types`), `cmd` (`B × 3`) + readback.
Params uniform: `n, B, k_in, n_types, alpha, wave_stride_u32 (= 48), NONE`.

Kernels:
1. `fly_step` — `@workgroup_size(128)`, one thread per neuron j (dispatch `ceil(n / 128)`),
   loop over beings inside (B ≤ 64, local `acc[64]`). Reads the perception wave buffer
   (`PerceptionGpu::buf_output`, 48 u32 per being, i8 two's-complement bytes, as in
   `perception.rs:878-886`).
2. `fly_read` — one thread per (being, command), mean over the two efferent groups → `cmd`.

Order inside the tick: after `perception` (and `retina_digest`) wrote the wave, before/alongside
`brain`; same encoder/submit as the fast brain; readback with the same `map_async → poll → recv`
discipline as `read_intent_after_submit`. Cost estimate at B = 40, full map: ~2.4 GB of gathers
per tick → 5–10 ms/tick (today's tick ≈ 1 ms); acceptable for the first closed-loop test; the
`--min-weight 3` map is 3× cheaper.

## 5. Flags (register in `src/bin/gpu_evolve/cli_flags.rs`, consume in `main.rs`)

`--fly-map <path>` (load map; required by the others), `--fly-brain` (use the map's commands
instead of the fast brain's intents; sets `FLY_MUTATE_ON`), `--fly-tau <steps>` (default 4),
`--fly-p-bind`, `--fly-p-off`, `--fly-p-on`, `--fly-sigma`, `--fly-log-every <gens>` (print, for
the elite: bound channels → group names, out pairs, mask density). English names only; no
deprecated aliases needed (new flags).

## 6. Tests and gates (all must pass before any GPU battery)

- `tests/wgsl_validation.rs`: `fly_map_wgsl_is_valid` (naga parse + type check, no GPU).
- `src/fly_map.rs` (lib): map loader with a synthetic 6-neuron map written by the test; CPU
  reference `cpu_step` (same formulas as §2); property test: pruned type emits 0; unbound
  channel drives 0; NONE handling; group means.
- GPU parity (`#[ignore]`, run with `--ignored` like the other GPU parity tests): synthetic map,
  B = 3, 10 ticks, GPU vs `cpu_step` within 1e-5.
- Genome: round-trip with and without the block; a genome from before this change loads with an
  empty block; mutation with `FLY_MUTATE_ON = false` leaves the rng stream and every other block
  byte-identical (same guard style as `evolve_hip_bloco_tem_taxa_propria_e_portao`).
- `cargo test --release --bin gpu_evolve`, `cargo test --release --lib`, `cargo fmt -p universe
  --check` on the touched files only, `scripts/check_language.sh` (English gate).
- With the flags off, `gpu_validate` numbers unchanged (nothing in the existing bindings moves).

## 7. First experiments (predictions go to `docs/PREDICOES-CEREBRO.md` before running)

F1 smoke: `--fly-map … --fly-brain`, 2 gens, 5 000 ticks, 8 beings — runs, tick cost measured.
F2 closed loop, 3 seeds × 15 gens × 40 beings, track 8005, from a random population (not the
champion — the champion's fast brain is bypassed): arms `real map` vs `shuffled-targets map`
(export variant) vs `no fly` (today's brain from random init). Ruler: laps (≥ 1V, ≥ 2V), best
distance, and whether pruning converges on a stable set of cell types across seeds (the
"convergent core" of docs/94).

---

## 8. Implementation notes (14/09/2026)

What was built, where it lives, what departs from the specification above, and what was
measured. Written after the code, so everything here is a statement about the tree, not a
plan.

### Files

| File | What it is |
|---|---|
| `src/fly_map.rs` (+ `src/fly_map_selection.rs`, `src/fly_map_tests.rs`) | GPU-free core: the `flymap` v1 reader/writer, the synthetic six-neuron map, `cpu_step`/`cpu_read` (§2) in the first file; the genome-block operators (§3) and `pack_selection` in the second, re-exported so callers see one module (600-line cap). `NONE` is defined once, in `fly_map.rs` |
| `src/gpu/fly_map/mod.rs` | `FlyMapGpu`: seventeen buffers, two pipelines, `upload_selection` (per generation), `dispatch` (per tick), `read_commands_after_submit`, `reset_state` |
| `src/gpu/fly_map/fly_map.wgsl` | `fly_step` + `fly_read`, 17 bindings. The only mirror of `NONE` |
| `src/gpu/fly_map/parity_test.rs` | `#[ignore]`d GPU parity: synthetic map, B = 3, 10 ticks, GPU vs the CPU reference |
| `src/gpu/evolution.rs` | `Genome::fly` (new last field), `FLY_MUTATE_ON`, the four `EvoParams` rates, the mutation block (new last rng consumer), `evolve_fly_block_has_its_own_rates_and_gate` |
| `src/gpu/genome_io.rs` | `fly` written and read as the new last block |
| `src/bin/gpu_evolve/{cli_flags,main,statics,context,tests}.rs` | the eight flags, the map load, the per-generation upload, the per-tick dispatch/readback/override, two genome round-trip tests |
| `tests/wgsl_validation.rs` | `fly_map_wgsl_is_valid` |
| `src/lib.rs`, `src/gpu/mod.rs`, `src/gpu/dof.rs`, `ARCHITECTURE.md` | module registration, the `Genome` fixture, and the genome tables that were already missing `hip` |

### Deviations from the specification, and why

1. **`in_bind` is `SENSORY_WAVE_SIZE` long, not a literal 192.** The constant is computed
   in `config.rs` and the wave already shrank once from 223 with two hand-written copies
   missed (`docs/STYLE.md` §4). Today the value IS 192, so the block is `200 + n_types` as
   specified.
2. **The `fly` block's length on disk is "whatever is left after `hip`".** §3 does not say
   how a variable-length block is bounded, and a reader of a `.bin` does not necessarily
   have the map that sized it. Being last makes the length recoverable without a header
   bump; a block written against a different map therefore loads at its own length and
   `pack_selection` treats it as INERT rather than reading it at the wrong stride.
3. **`fly_step` serves beings in chunks of 64** instead of assuming B ≤ 64. The private
   `acc` array still has to be a compile-time 64, but a larger population repeats the loop
   (re-reading the CSR row per chunk) instead of silently computing a subset.
4. **Two bind-group layouts, one per entry point.** One shared layout would declare all 16
   storage buffers, which is exactly `max_storage_buffers_per_shader_stage` as `gpu_evolve`
   configures it, leaving no headroom for the next binding.
5. **No device limit had to be raised.** With the full map and B = 64: `indices` and
   `weights` are 15 091 983 × 4 B = 57.6 MiB each, and `x`/`r_a`/`r_b` are
   138 639 × 64 × 4 B = 33.9 MiB each — all under the 128 MiB `downlevel_defaults` binding
   cap, and the largest single allocation is well under the 256 MiB buffer cap.
   `FlyMapGpu::new` checks both and returns `None` with the numbers printed rather than
   letting wgpu fail mid-run. `context.rs:295-302` was NOT touched.
6. **The params uniform carries `k_out`** where §4 lists a spare, and ships `NONE` from the
   host. The shader's `NONE_ID` constant is documentation of the contract, not a second
   source of truth.
7. **`--fly-log-every` prints in-group IDs, not names.** The names live in the sidecar
   `flymap_783.json` and nothing in the Rust tree parses that file; adding a JSON reader
   for a log line was not worth the dependency. The line carries the bound-channel count,
   the first eight `channel->group` pairs, the six efferent slots, mask density, gain and
   scale.
8. **A resumed population gets its fly block reseeded.** A champion saved before this
   change carries no block, and an empty block is a being with every channel unbound — it
   would command exactly zero for a whole battery. `main()` reseeds those at birth values
   and says so on stdout; the rest of the genome is untouched.
9. **`--fly-map` that fails to load is a hard error**, like `--resume` without a
   checkpoint: silently falling back would report plausible numbers for a brain nobody
   asked for.

### What is NOT proven

The seam `main() → GpuCtx::fly → tick()` has never been executed end to end: the task that
produced this code forbade launching any run, so no `gpu_evolve` process has yet loaded a
map. What IS proven: both kernels against the CPU reference on the real GPU, the CPU
reference against hand-computed values, the genome block across the disk, the rng-stream
guarantee, and that the flag table and the parser agree. The first smoke run (§7, F1) is
still the first execution of the wiring.

### Measured

| Check | Result | Wall time |
|---|---|---|
| `cargo build --release` | ok, no new warnings | 2 m 13 s cold, 16 s warm |
| `cargo test --release --lib` | 413 passed, 0 failed, 3 ignored | 0.21 s (after the build) |
| `cargo test --release --bin gpu_evolve` | 77 passed, 0 failed | 0.06 s |
| `cargo test --release --test wgsl_validation` | 6 passed, 0 failed | 0.00 s |
| `cargo test --release --lib -- --ignored fly_map_parity` | 1 passed (GPU vs CPU within 1e-5, 10 ticks, B = 3) | 0.15 s |

The tick cost of the real map is NOT measured — no map file was available to this pass and
no run was launched. The estimate in §4 (5-10 ms/tick at B = 40) stands as an estimate.

### Measured tick cost (serial, `--kernel-profile 250`, 1 gen × 1 000 ticks, track 8005; 14/09/2026)

| map | beings | kernel v1 `fly_map` | kernel v2 `fly_map` | tick wall v2 |
|---|---|---|---|---|
| full (15.1 M edges) | 8 | 20.1 ms | — | — |
| full | 16 | — | 6.8 ms | 8.5 ms |
| full | 40 | 98.1 ms | **7.9 ms** | **10.0 ms** |
| `_w3` (4.9 M) | 8 / 40 | 11.6 / 57.1 ms | — / 3.7 ms | — / 5.5 ms |
| `_w3_nooptic` (2.15 M) | 8 | 5.1 ms | — | — |

v1 = one thread per neuron looping over beings (cost linear in beings); v2 = lanes are beings
(merge c0f86aa): flat in beings, the gather is served by the Infinity Cache.

