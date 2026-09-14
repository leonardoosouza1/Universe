# Kernel request — spiking fly brain on the GPU (for Astra)

**Status:** implemented in the Rust/WGSL tree on 14/09/2026. K1 (delayed event propagation), K2
(optional retina Poisson drive) and K3 (leaky plus/minus readout) are exposed by
`gpu::fly_map::FlySpikingGpu` and selected with `gpu_evolve --fly-spiking`. The CPU reference is
`src/fly_map_spiking.rs`; the GPU source is `src/gpu/fly_map/spiking.wgsl`. The legacy rate path stays
the default and remains available with `--fly-map`/`--fly-brain`.

Validation already run: CPU delay/Poisson/refractory tests, naga WGSL validation, the ignored GPU
two-neuron delay/readout parity test (`fly_spiking_parity`), both existing rate-model GPU parity
tests, and short real-map runs at B=1 and B=40. The full-map sugar→MN9 gate and a stable ≤12 ms
benchmark still require a dedicated long experiment; no biological claim is inferred from the
short smoke.
**Who does what:** the host owns allocation, selection upload and one command readback per tick;
all substeps, propagation, integration, Poisson sampling and readout remain on the GPU.
The final run report separates setup/reset uploads from per-tick uniform uploads and reports
per-tick GPU→CPU maps, so a diagnostic trace cannot be mistaken for the production transfer path.

The first implementation uses the documented wave-uniform row-skip baseline: K1 launches one
invocation per `(presynaptic neuron, packed-being word)` and exits before walking the CSR when the
delayed word is zero. It is event-driven in edge work, while retaining a predictable launch shape.
Compaction of `(neuron, being)` events is a later optimization and must be enabled only after the
full-map benchmark below establishes that row-skip misses the budget. That benchmark now exists:
on the RX 6750 XT the real map measured 11.2–17.9 ms at B=1 and 85.6–114.4 ms at B=40 for the
`fly_map` region over two diagnostic ticks. The baseline is therefore a correctness path, not yet
a production-performance path.

Read first: [`FLY-MAP-BRAIN.md`](FLY-MAP-BRAIN.md) (the rate model already merged, map file format,
measured costs), [`../src/gpu/CLAUDE.md`](../src/gpu/CLAUDE.md) (buffer/binding rules, the traps),
[`../src/gpu/fly_map/`](../src/gpu/fly_map/) (the existing `fly_step`/`fly_read` to copy the style from).

---

## 0. Why this is needed (one paragraph, with numbers)

The Drosophila connectome (FlyWire v783: 138 639 neurons, 15 091 983 signed edges, 54.5 M synapses) is
already resident on the GPU and driving beings through a **rate** model — one step per 15.6 ms tick,
`x ← x + α(−x + scale·W·r + u)`, kernel `fly_step`, 7.9 ms/tick at 40 beings on the full map. That model
is a convenience, not the fly: every published result on this connectome (Shiu et al. 2024, Nature) uses
**leaky integrate-and-fire with spikes**, and the one computation we verified end to end — 20 sugar
receptor neurons → proboscis motor neuron MN9 at 72 Hz, 420 active neurons — only exists in the spiking
model. A rate model with one step per tick cannot represent it: the pathway is 5 synapses deep with a
1.8 ms delay each, so 9 ms of the answer happens *inside* one tick.

**The cost argument that makes it feasible.** Dense evaluation at the LIF timestep is hopeless:
15.1 M edges × 156 substeps (dt 0.1 ms in a 15.6 ms tick) = 2.35 G edge-ops per tick per being. But the
brain is **sparse in time**: at the firing rates we measure, ~0.3 % of neurons spike per 0.1 ms step, so
an event-driven kernel that only walks the rows of neurons that actually spiked touches roughly
15.1 M × 0.003 ≈ 45 k edges per substep. This is a rough upper-bound estimate, not an acceptance result:
“420 active neurons over a one-second run” is not the same as a per-substep spike fraction. The kernel
must report mean and p95 edges touched per substep on the exact stimulus before claiming a speedup.

**Is the wiring worth the effort? Measured 14/09, and the answer is yes.** The same 20 sugar neurons at
100 Hz, on the real map and on three controls that preserve the statistics:

| map | MN9 | trials firing | active neurons | spikes |
|---|---|---|---|---|
| real | **72.4 ± 5.0 Hz** | 30/30 | 420 | 340 k |
| shuffled targets (same degrees, same signs) | **0.0 Hz** | 0/30 | 175 | 76 k |
| shuffled weights (same graph, same signs) | **0.0 Hz** | 0/30 | 96 | 98 k |
| shuffled signs (same graph, same weights) | 29.4 ± 24.2 Hz | 28/30 | **26 022** | **14.0 M** |

Permuting the targets erases the pathway; permuting only the synapse *counts* also erases it (7 % of the
edges carry 43 % of the synapses — the pathway is a narrow ridge of heavy edges); permuting the signs
does not erase it but replaces it with a seizure, 62× more neurons active. So the connectome is not a
generic reservoir: the specific targets AND the specific weights are the computation. The corollary for
this document is that an approximation which blurs either one (random pruning, weight quantisation that
flattens the tail, dropping edges below a threshold) can break the model in a way that microbenchmarks
will not show — which is why acceptance criterion 2 below is the sugar→MN9 gate, not a timing number.

---

## 1. What the three kernels must do

### K1 — `fly_spike_step` (the one that matters)

One substep of LIF for all beings, event-driven. Per neuron `j` and being `b`:

```
v       ← v0 + (v − v0)·decay_v + i_syn·(1 − decay_v) (only if not refractory)
i_syn   ← i_syn·decay_syn                            (only if not refractory)
spike   = v > v_th
if spike: v ← v_rst ; i_syn ← 0 ; refractory ← t_rfc (except for neurons flagged no_rfc)
```
`i_syn` is a current-like voltage drive, matching `LifBrain`; it is not a conductance. A true
conductance model needs a reversal potential and a voltage-dependent term such as `g·(E_rev − v)` and
would be a different model. The drive delivered by spikes emitted `D` substeps ago is:

```
for each neuron i that spiked at substep t−D, for each edge (i → k) of row i:
    s_i          = spike[i, b] · mask[b, type_of[i]]
    i_syn[k, b] += w_syn · weight[e] · s_i · max(scale[b], 0)
```

The mask is pre-synaptic: it controls whether a selected neuron emits a spike into all of its outgoing
edges. Masking `type_of[k]` (the destination) changes the graph and does not match the Python reference.

Constants (Shiu et al. 2024, `default_params`, already used by the Python reference
`neuromod-lab/src/flybrain/dynamics.py`): `dt = 0.1 ms`, `v_0 = v_rst = −52 mV`, `v_th = −45 mV`,
`t_mbr = 20 ms` → `decay_v = exp(−dt/t_mbr)`, `tau_syn = 5 ms` → `decay_syn = exp(−dt/tau_syn)`,
`t_rfc = 2.2 ms` → 22 substeps, `t_dly = 1.8 ms` → `D = 18` substeps, `w_syn = 0.275 mV` per synapse.
The refractory state is a **remaining-step counter**: a neuron with `rfc_remaining > 0` skips the
integration, then decrements once per substep; it is free exactly when the counter reaches zero. The
test must assert the first allowed spike time, because assigning 22 and decrementing only on later ticks
can accidentally produce 2.3 ms rather than 2.2 ms.

**Data layout (fixed, do not change):** the map is CSR **by presynaptic neuron** for this kernel (the
rate model uses CSR by postsynaptic; the exporter will ship both — see §3). Per-being state is
**being-minor**: `v[j*B + b]`, `i_syn[j*B + b]`, so lanes = beings coalesce, exactly like the `fly_step`
rewrite that took 98 ms → 7.9 ms at B = 40 (`@workgroup_size(32, 4)`, 32 lanes = 32 beings, 4 rows of
neurons per workgroup, one wave walks one CSR row and the `indices`/`weights` loads are wave-uniform).

**The event-driven part — the design decision we need from you.** Two options, pick with a measurement:

- **(a) Compaction.** A pass over `spiked[j*B + b]` builds a compact list of (neuron, being) pairs that
  spiked at substep `t − D`, then one thread per (pair, edge-chunk) walks only those rows and
  `atomicAdd`s into `i_syn`. Cost ∝ actual spikes. Needs an atomic counter and a scatter buffer sized for
  the worst case (say 5 % of n×B).
- **(b) Wave-uniform row skip.** Keep one wave per neuron row (lanes = beings, like `fly_step`) but read
  a per-neuron "any being spiked" bit first and `continue` when it is zero. This is a useful baseline,
  but it does **not** remove write conflicts: several presynaptic rows can target the same `i_syn[k,b]`.
  It is correct only with an atomic representation or a separate gather/reduction pass. Its work is
  proportional to (neurons with at least one being spiking) × row length, which degrades as B grows
  because different beings spike at different times.

Our estimate: (a) wins at B ≥ 8, (b) may win at B = 1–4 and is far simpler. **Implement (b) first as the
correctness baseline, then (a), and report both timings** — that is the same discipline that produced the
6× on `fly_step`.

Atomics: `i_syn` is `f32` and WGSL has no `atomicAdd` on floats. The correctness baseline must either
accumulate into an `atomic<i32>`
fixed-point buffer (`i_syn` in µV as i32: `w_syn·weight` is a multiple of 0.275 mV, so 1 µV resolution loses
nothing and the range fits — state the bound you use), or give each of the 32 lanes its own being and
use a destination-gather/reduction layout in which each `(k,b)` has one writer. Being-minor lanes alone
do not make writes non-conflicting: different source rows still converge on the same destination.
Benchmark compaction only after this baseline is bit-for-bit correct.

**Delay ring.** `D = 18` substeps of spike bits: `ring[(t) % D][j*B + b]` as `u32` bitfields packed over
beings (B ≤ 64 → two u32 per neuron) keeps it at 138 639 × 2 × 18 × 4 B = 20 MB. Fine.

### K2 — `fly_eye_drive`

**Read this box before implementing K2 — a measurement on 14/09 changed what this kernel must target.**
Driving the 7 932 R1-6 photoreceptors in the spiking model does NOT reach the visual outputs, at any
rate. The lamina works (L1/L2 reach 26 Hz when photoreceptors are driven at 250 Hz) but the next stage
is net inhibitory: Mi1 receives −3 214 synapse·Hz, Tm3 −1 348, T4 −1 059, T5 −1 997, HS −219. Nothing
past hop 1 ever fires, and raising the drive only deepens the inhibition. The reason is biological: in
the fly, photoreceptors and lamina/medulla neurons are **graded — they do not spike**; inhibition there
modulates a tonic release. A leaky integrate-and-fire model without tonic excitation cannot represent
that stage, and the reference model (Shiu et al. 2024) was validated on excitatory chains such as
sugar → MN9, not on the optic lobe.

The path downstream of the motion detectors IS spiking and works. Measured, same model, 100 Hz drive:

| driven | LPLC2 | HS | VS | DNp01 (giant fiber) |
|---|---|---|---|---|
| T4 + T5 (12 246 cells) | 24.0 Hz | 250 Hz | 241 Hz | **95.6 Hz, 100 % of trials** |
| LC4 + LPLC2 | — | — | — | **230 Hz** |

**So K2 drives whatever neurons the map file says, not photoreceptors by name.** The map carries
`pixel_of_neuron[u32]` (`0xFFFFFFFF` = not eye-driven) and the exporter decides the target set:
`--eye-target photoreceptor` (for the rate model, which handles graded stages correctly) or
`--eye-target motion` (T4/T5, for this spiking kernel). The kernel is identical either way — it reads
the table. Full diagnosis: `neuromod-lab/PREDICOES.md` entry 189v and
`neuromod-lab/src/fly_diag_vision.py`.

Turn the rendered retina into eye-driven input, **on the GPU, no host roundtrip**. Input: the
existing retina/perception buffer (`PerceptionGpu::buf_output`, 48 u32 per being, i8 channels; the raw
retina is `GpuCtx::buf_retina_out`, 1 764 px per being = 14 rows × 126 cols, FOV 120°, pitch 0 to
−0.06 rad). Plus a **binding table** shipped in the map file: for each of the 7 932 R1-6 photoreceptors,
the pixel it looks at (`pixel_of_neuron[u32]`, computed from the target cells' retinotopy — Claude's
`neuromod-lab/src/flybrain/eye.py` produces it). Output: per (photoreceptor, being) a Poisson spike
probability for this substep:

```
rate_hz = max(0, bg_hz · max(gain[b], 0) · (1 − brightness))
p_spike = 1 − exp(−rate_hz · dt_ms · 1e−3)  (exact Bernoulli probability for a Poisson process)
spike   = rand < clamp(p_spike, 0, 1)        → v += w_syn · f_poi (f_poi = 250), no refractory
```
One thread per (bound neuron, being); the RNG must be a counter-based hash of (neuron, being, substep,
run seed) so a run is reproducible and no state buffer is needed. The map's `pixel_of_neuron` table
only supplies luminance. It cannot produce motion selectivity by itself: a `motion` target must consume
a temporal/spatial difference (for example, current minus previous retina or a precomputed T4/T5 drive)
with an explicit time buffer and documented polarity. Feeding static brightness to T4/T5 is not a motion
kernel and must not be presented as one.

### K3 — `fly_spike_readout`

Per being, per command `c ∈ {throttle, steer, brake}`: a leaky spike-count readout over the two chosen
efferent groups (the `out_pair` selection the genome already carries):

```
plus_mean  = (spikes in plus group)  / max(1, size(plus group))
minus_mean = (spikes in minus group) / max(1, size(minus group))
drive_c    = plus_mean − minus_mean
count_c    ← count_c · decay_read + (1 − decay_read) · drive_c
cmd_c      = tanh(count_c / norm_c)          norm_c = 1 by default (dimensionless)
```
Normalising the difference by the mean of the two group sizes is biased when the groups have different
sizes. Use per-group means as above; define an empty-group mean as 0. The `(1 − decay_read)` factor makes
`count_c` an exponential moving average rather than an unbounded accumulated count, so `norm_c = 1` has
a stable interpretation across tick rates and map topologies. Expose `norm_c` only if a deliberate gain
change is measured.
`decay_read = exp(−dt / 50 ms)` (a 50 ms integration window, the timescale of a fly's steering
response). One thread per (being, command); groups come from `out_indptr`/`out_members`, already in the
map file. The host reads `cmd[B][3]` once per tick, like `fly_read` does today.

---

## 2. Acceptance criteria (what "done" means)

1. **Parity with the Python reference, on a synthetic map.** `neuromod-lab/src/flybrain/dynamics.py`
   `LifBrain` is the reference; `src/fly_map.rs` already has the CPU-side pattern for a synthetic map and
   a parity test (`src/gpu/fly_map/parity_test.rs`). A new `#[ignore]` GPU test must match spike *times*
   exactly (they are discrete) for 500 substeps on a 6-neuron map, B = 3, including: refractory blocks a
   second spike and frees it at the documented step, the delay is exactly 18 substeps, an inhibitory edge
   pushes `v` below rest, a pruned pre-synaptic cell emits nothing, and eye Poisson input uses the exact
   probability above.
2. **The published gate reproduces.** (This is the SUGAR gate, not a visual one — see the box in K2:
   looming from photoreceptors is not achievable in a spiking model and is NOT required of you.) With the real map, 20 sugar GRNs driven at 100 Hz for 1 s:
   **MN9 (root id 720575940660219265) fires at 72 ± 10 Hz and 400 ± 100 neurons are active** — that is
   what our torch implementation measures (`neuromod-lab/src/fly_gate.py`, verified 14/09) and what the
   paper reports. A kernel that does not reproduce this is wrong, whatever the microbenchmarks say.
3. **Budget.** ≤ 12 ms per Universe tick at B = 40 on the full map, including the 156 substeps. If that
   is impossible, report the measured number and where it goes (mean/p95 edges touched per substep, occupancy,
   atomic contention) — a negative with the mechanism is a result, and we will cut substeps (dt 0.2 ms)
   or the map (`--min-weight 3`, 4.9 M edges) instead of guessing.
4. **Everything on the GPU.** No per-substep host readback, no CPU loop over neurons or beings. The
   selection is uploaded once per generation; each tick performs one small batched uniform upload
   (the substep schedule) and reads `cmd[B][3]` once. No neuron state crosses the host boundary.
5. The existing rate path (`fly_step`/`fly_read`) keeps working unchanged and stays the default; the
   spiking path is a new flag (`--fly-spiking`), off by default, and with it off **every number in
   `gpu_validate` and the 414 lib / 77 bin tests is unchanged**.

## 3. What Claude ships so you can start

- `flymap` **v2** file: magic `FLYMAP02`, header `n, m, n_types, k_in, k_out, out_members_len, flags,
  reserved` (flags bit 0 = presynaptic CSR, bit 1 = pixel binding, bit 2 = no-refractory bitmap), the v1
  body unchanged, then the flagged blocks in order: presynaptic CSR (`pre_indptr u32[n+1]`,
  `pre_indices u32[m]`, `pre_weights f32[m]`), `pixel_of_neuron u32[n]`, `no_refractory u32[ceil(n/32)]`.
  Sidecar JSON with names, retina geometry, eye target and SHA-256.

  **READY, verified 14/09 12:55** (magic, header and computed size checked byte for byte against the
  file; 12/12 exporter tests pass):

  | file | bytes | eye target | notes |
  |---|---|---|---|
  | `/media/llm-data/datasets/flywire/flymap_783_v2.bin` | 244 269 836 | photoreceptor (R1-6, 7 932 bound) | full map, flags = 7 |
  | `/media/llm-data/datasets/flywire/flymap_783_motion_v2.bin` | 244 269 836 | motion (T4+T5, 12 246 bound) | **use this one for K2** |
  | `/media/llm-data/datasets/flywire/flymap_783_w3_v2.bin` | 81 457 804 | photoreceptor | `--min-weight 3`, for quick iterations |

  Header of the full file, read back: `n=138639 m=15091983 n_types=8866 k_in=268 k_out=516
  out_members=1470 flags=7 reserved=0`. The binding uses `--eye-fit stretch` (the whole eye mapped onto
  the retina's rows); the anatomically strict `strip` fit binds only 77 of 7 932 photoreceptors and
  188 of 12 246 motion cells, because the car's rendered retina spans just 0.69–3.44 degrees of
  elevation — the counts for both fits are in the sidecar JSON and in
  `neuromod-lab/docs/96-flymap-v2.md`.
- The three map variants for controls, already exported:
  `/media/llm-data/datasets/flywire/flymap_783{,_w3,_w3_nooptic,_shuf0}.bin`.
- `neuromod-lab/src/flybrain/dynamics.py` (`LifBrain`) as the executable reference, and
  `neuromod-lab/src/fly_gate.py` as the acceptance test in Python (runs in 23 s on this GPU).

## 4. Rules of this repo that apply (they cost GPU hours to learn)

- English everywhere: identifiers, comments, log lines, test names, commit messages
  (`scripts/check_language.sh` gates it).
- A kernel must give the same answer as the CPU reference — `src/gpu/CLAUDE.md`, first rule.
  `src/bin/gpu_validate.rs` is where equivalence is proven, and `tests/wgsl_validation.rs` needs one
  `#[test]` per new kernel (naga parse + type check, no GPU).
- Host and shader must agree on struct layout and nothing tells you when they do not (the 368 vs 360
  byte `PhysicsStateGpu` story in `src/gpu/CLAUDE.md`). Lock any stride in a `const` and add a test that
  asserts it.
- Device limits are `downlevel_defaults`: **128 MiB per storage binding**, 256 MiB per buffer. The full
  map's `indices` and `weights` are 57.6 MiB each — keep them in separate bindings; the delay ring at
  20 MB is fine; do not merge buffers to save a binding without checking the cap.
- Readback order is `map_async → poll → recv`; polling before `map_async` is a silent deadlock.
- Outputs go to `/media/llm-data/...`, never the repo, never `/tmp`. Heavy scratch:
  `/media/llm-data/scratch/`.
- Never run `gpu_evolve --help` — the flag does not exist, it is ignored, and a real run starts in the
  current directory writing replays to the system disk.
- Build with `CARGO_BUILD_JOBS=10 cargo build --release`; the GPU is shared with running batteries, so
  keep benchmarks under ~30 s and expect a ~10 ms floor on submission latency when it is busy.

## 5. Open questions we would like your opinion on

1. **Substep count.** 156 substeps/tick is the faithful number. Is there a defensible reduction —
   dt 0.2 ms (78 substeps), or running the fly brain at a fixed 4 kHz decoupled from the 64 Hz tick with
   the retina held constant between ticks? Which one keeps the sugar→MN9 gate within tolerance?
2. **Weight precision.** Edge traffic is only 3 % of the total (the `r`/`v` gather dominates and is
   served by the 96 MB Infinity Cache), so f16 weights would save ~1.5 % — we skipped it. Do you see a
   layout where the *gather* gets cheaper instead (block the connectome by neuropil so a wave's reads
   stay inside a cache-resident slice)?
3. **Per-being divergence.** With B = 40 independent beings the spike patterns diverge, so a row is
   "active" for some beings and not others. Is there a better mapping than lanes = beings for the
   event-driven case — e.g. one workgroup per being with a per-being compacted spike list?
