# Kernel observation contract: exact L1/L2 trace for one being (15/09/2026)

The laboratory UI needs to answer a precise question: *which recorded signals contributed
to this motor command at this tick?* The trace must expose the values the production WGSL
kernel actually used. It must not reconstruct plausible activations from a checkpoint and
call them measurements.

This is an observation change only. It must not change the controller, evolution, fitness,
physics, or the output of a run when diagnostics are disabled.

## 1. What exists today

`src/bin/gpu_evolve/driving_diagnostics.rs` samples one selected being at
`--vision-every N` and writes `observation.jsonl`.

| field | shape and type | exact meaning |
|---|---|---|
| `retina` | `u32[14,126]` | packed material in the low byte and distance in millimetres in the upper 24 bits |
| `wave` | `i8[192]` | sensory wave entering `brain.wgsl` |
| `fast_state` | `f32[96]` | resident CfC state after the current brain dispatch, before `tanh(state) -> i8` feeds L1 |
| `body.raw_motor` | `f32[3]` | `final_intent[0..3]` before the output KAN/head |
| `body.motor_trace` | structured | raw intent, output head, slow/hippocampal modulation, suppression and final command |
| `body.applied_motor` | `f32[3]` | command consumed by physics |

The current sampled GPU copy is 7,632 bytes:

```text
retina  1,764 × u32 = 7,056 B
wave      192 × i8  =   192 B
CfC        96 × f32 =   384 B
                         -----
                         7,632 B
```

The complete `final_intent[96]` is already read from GPU for the normal control path. The
diagnostic writer currently keeps only its first three dimensions. The implementation must
reuse that CPU value; it must not copy `buf_intent` a second time.

## 2. The kernel is a graph, not a list of layers

The executed fast-brain graph in `src/gpu/brain.wgsl` is:

```text
wave[0..96] -> sensory CfC -> quantised CfC input --+
                                                     +-> L1 matmul -> L1 effective --+-> L1 activation/gate/i8 -> L2 matmul -> L2 memory -> normalisation/gain/fatigue/activation -> final_intent[0..48]
wave[96..192] -> sensory KAN ------------------------+                              |
                                                                                    +-> gain/fatigue/tanh -> final_intent[48..96]

final_intent[0..3] -> output KAN/head -> slow + hippocampus modulation -> brake gates/overrides -> physics command
```

The split after L1 matters:

- L2 reads `w_l1a`, the post-activation, post-region-gate integer signal.
- The bypass reads `w_l1r[48..96]`, the post-memory and post-threshold floating-point
  signal **before** heterogeneous activation, region gates and quantisation.
- `final_intent[0..48]` is already the final L2 activation. A second field called `l2`
  with those same values would duplicate data and hide the stages before it.

Consequently, a flat `"brain_flow": ["cfc", "l1", "l2", "intent"]` is false. Metadata
must describe nodes and edges, including slices.

## 3. Required sampled stages

Export the following values for the selected being on a diagnostic sample tick. Names are
the stable external contract; the WGSL variables are listed to prevent semantic drift.

| external name | shape/type in JSON | WGSL value | why it is needed |
|---|---|---|---|
| `cfc_state` | `f32[96]` | `buf_cfc[state]` | recurrent state already recorded as legacy `fast_state` |
| `l1_matmul` | `f32[96]` | `w_l1r` immediately after `acc / 127` | validates the L1 weighted sum |
| `l1_effective` | `f32[96]` | `w_l1r` after period/memory/threshold, before activation/gate | actual source of the bypass |
| `l1_to_l2` | `i8[96]` | exact integer `w_l1a` after activation and region gate | exact input consumed by the L2 matrix |
| `l2_matmul` | `f32[48]` | `raw_l2 = acc / 127`, before L2 memory | validates the L2 weighted sum |
| `l2_effective` | `f32[48]` | `w_l2r` after L2 memory, before std normalisation | separates memory from output shaping |
| `l2_adapted` | `f32[48]` | value after std/gain/output fatigue, before heterogeneous activation | explains the transition to L2 output |
| `bypass_adapted` | `f32[48]` | value after gain/output fatigue, before bypass `tanh` | explains `final_intent[48..96]` |
| `final_intent` | `f32[96]` | existing CPU `intents[selected]` | complete head input; reuse existing readback |

Also record the scalar `l2_std` used on that tick. It is `1.0` when L2 normalisation is
disabled. This avoids asking the UI to reproduce a GPU floating-point reduction.

Keep `fast_state` for schema-v1 readers and write `stages.cfc_state` with the same values.
Do not silently rename or reinterpret an existing field.

### Added GPU snapshot size

The selected-being trace region needs 1,924 bytes if the GPU snapshot stores
`l1_to_l2` as `i32[96]`, which is its native WGSL representation:

```text
l1_matmul       96 × f32 = 384 B
l1_effective    96 × f32 = 384 B
l1_to_l2       96 × i32 = 384 B  (serialise as i8 after checking [-127,127])
l2_matmul       48 × f32 = 192 B
l2_effective    48 × f32 = 192 B
l2_adapted      48 × f32 = 192 B
bypass_adapted  48 × f32 = 192 B
l2_std           1 × f32 =   4 B
                              -----
                              1,924 B
```

`final_intent` adds no GPU-copy bytes. Total sampled copy becomes 9,556 bytes, still one
copy and one mapped staging buffer per sampled tick.

## 4. GPU implementation constraints

Use a dedicated region in the existing `buf_debug` allocation or a dedicated observation
storage buffer. Either is acceptable if all of these properties hold:

1. Only the selected being writes the new trace region.
2. Observation is enabled only on ticks that `DrivingDiagnostics::sampling(tick)` selects.
3. The selection reaches WGSL in bits 1..31 of `control_flags` as
   `observed_being_plus_one`: `0` means disabled and `being + 1` identifies the writer.
   Bit 0 independently enables counterfactual ablation. The packed word keeps the uniform
   size and alignment unchanged.
4. The branch is uniform for the workgroup. Every existing workgroup barrier remains in
   uniform control flow.
5. Writes copy values that already exist in workgroup memory. Do not recompute a second
   network for observation.
6. The trace copy is recorded in the same command encoder, after the brain dispatch and
   before its existing submit. No second submit, `device.poll`, map, or GPU-to-CPU path.
7. `final_intent` comes from the existing `read_intent_after_submit` result.
8. The legacy `download_debug()` layout used by `gpu_validate` and `--observe` remains
   compatible, or all its consumers are migrated in the same change with parity tests.

The current `buf_debug` writes only being zero. `--driving-diagnostics-being` can select a
different being, so hard-coding the new trace to being zero is not acceptable.

## 5. Metadata contract

Bump `driving.meta.json.schema_version` because the observation schema changes. Add a
versioned graph, not a flat ordered list:

```json
{
  "observation_schema": 2,
  "brain_graph": {
    "version": 1,
    "nodes": [
      {"name":"wave", "shape":[192], "dtype":"i8"},
      {"name":"cfc_state", "shape":[96], "dtype":"f32"},
      {"name":"l1_matmul", "shape":[96], "dtype":"f32"},
      {"name":"l1_effective", "shape":[96], "dtype":"f32"},
      {"name":"l1_to_l2", "shape":[96], "dtype":"i8"},
      {"name":"l2_matmul", "shape":[48], "dtype":"f32"},
      {"name":"l2_effective", "shape":[48], "dtype":"f32"},
      {"name":"l2_adapted", "shape":[48], "dtype":"f32"},
      {"name":"bypass_adapted", "shape":[48], "dtype":"f32"},
      {"name":"final_intent", "shape":[96], "dtype":"f32"}
    ],
    "edges": [
      {"from":"wave[0:96]", "to":"cfc_state"},
      {"from":"cfc_state", "to":"l1_matmul"},
      {"from":"wave[96:192]", "to":"l1_matmul"},
      {"from":"l1_matmul", "to":"l1_effective"},
      {"from":"l1_effective", "to":"l1_to_l2"},
      {"from":"l1_to_l2", "to":"l2_matmul"},
      {"from":"l2_matmul", "to":"l2_effective"},
      {"from":"l2_effective", "to":"l2_adapted"},
      {"from":"l2_adapted", "to":"final_intent[0:48]"},
      {"from":"l1_effective[48:96]", "to":"bypass_adapted"},
      {"from":"bypass_adapted", "to":"final_intent[48:96]"}
    ]
  },
  "trace_gpu_bytes_per_sample": 1924,
  "sample_gpu_bytes_total": 9556
}
```

The production metadata must also retain the runtime flags that determine the graph's
effective transformations: output gain, output fatigue strength, L2 normalisation,
mechanism-off mask, regional gating, sensor noise and dropout.

Every node declaration is a machine-readable contract, not just a label for the UI. In
addition to `name`, `shape` and `dtype`, it must declare:

- `unit` (`dimensionless`, `sensor_code`, `millimetres`, or the applicable physical unit);
- `scale` and `zero_point` for any quantised value;
- `encoding` when bytes contain packed fields rather than one scalar;
- `phase` (`pre_brain`, `post_brain`, `pre_physics`, or `post_physics`);
- `source` naming the WGSL buffer/workgroup value or the existing CPU readback;
- `optional`, so an older run can omit a stage without being interpreted as zeros.

The producer validates every sampled array against this declaration before writing it. The
pack builder validates it again and refuses inconsistent input. Metadata is authoritative;
the UI must never infer dtype or shape from a filename.

## 6. Recording profiles and batched transport

Neural detail and recording transport are separate decisions. The launcher exposes these
profiles:

| profile | physical replay | neural trace | intended use |
|---|---|---|---|
| `summary` | yes | none | population/evolution overview |
| `champion` | yes | one selected being every 2–4 ticks | normal interactive brain replay |
| `event` | yes | full cadence in bounded windows around brake, impact, death and lap events | detailed incident analysis |
| `full` | yes | every brain dispatch for one selected being | deliberate short scientific capture |

The profile records its actual cadence and event-window boundaries in metadata. A stage
sample exists only when the brain dispatched; skipped-brain ticks reference the previous
decision and must not manufacture a new neural row.

At future `champion`, `event` and `full` profile cadence, do not map one staging buffer for every sample.
Write selected-being snapshots into a bounded GPU ring and copy/map batches asynchronously.
The ring must:

1. have monotonically increasing tick tags beside every slot;
2. never overwrite an undrained slot silently;
3. report dropped samples explicitly in metadata and logs;
4. use at least double buffering so GPU production and CPU serialization can overlap;
5. preserve sample order across ring wrap;
6. drain at generation end and on orderly shutdown.

The sparse historical mode can remain readable, but the new high-cadence profiles are not
accepted if they add `map_async -> device.poll(Wait) -> recv` once per sample. The intended
shape is one kernel dispatch, resident trace writes, batched copies and amortised mapping.

## 7. Scientific meaning

The UI may compute a local edge contribution for a matrix stage:

```text
contribution(i -> j, t) = activation_i(t) * weight(j, i)
```

Call this a **contribution trail**, not a causal trail. It is exact for the recorded
matrix input and weight, but it does not prove causation after recurrent memory,
non-linearities, normalisation, fatigue or motor overrides.

A causal claim requires an intervention, such as replaying the same saved state with a
neuron or edge ablated and measuring `delta(final_command)`. That is a separate offline
feature and is not part of this kernel change.

## 8. Acceptance tests

### 8.1 Off-path preservation

- With `--driving-diagnostics` absent, the observed-being selector is zero.
- Existing `fly_map_parity_*`, `gpu_validate`, shader validation and deterministic smoke
  tests remain green.
- A reproducible committed fixture or checksum must back the deterministic comparison.
  An unversioned scratch directory alone is not an acceptance oracle.

### 8.2 Exact stage values

Validate at least 100 samples from a real GPU run, including a non-zero selected-being
index:

1. Reconstruct `l1_matmul` from the recorded wave/CfC state and checkpoint L1 weights.
   Match within `1e-5`, using the same ordered non-FMA accumulation as the kernel.
2. Assert every `l1_to_l2` value is in `[-127,127]` and compare it with a CPU execution of
   period, memory, threshold, activation and regional gate using the same prior state.
3. Reconstruct `l2_matmul` from recorded `l1_to_l2` and checkpoint L2 weights. Match
   within `1e-5`.
4. Reconstruct `final_intent` from the recorded effective/adapted stages and activation
   identifiers. Match within `1e-5`.
5. Assert `final_intent[0..3] == body.motor_trace.raw_intent` within `1e-6` on the same
   tick.
6. Assert the sampled tick, retina/wave input, neural stages and motor trace all belong to
   the same brain dispatch. A skipped-brain tick must not be labelled as a fresh neural
   evaluation.

### 8.3 Cost

Measure 48 beings with `--kernel-profile 250` under identical seed, adapter and flags:

- diagnostics off;
- diagnostics on at the current default cadence;
- diagnostics on with a stress cadence of every 16 ticks.

Report median and p95 tick wall time, GPU kernel time, bytes copied per sample and number
of submits/readbacks. The required invariant is no additional brain dispatch, no per-sample
synchronous readback and one batched trace-copy path. Report ring capacity, drain batch
size, peak occupancy and dropped-sample count. Do not claim zero cost merely because the
output remains bit-identical.

## 9. Consumer behaviour

The pack builder does not exist yet; the current PRD describes it. When implemented, it
must read the graph from `driving.meta.json` and emit only recorded stages. Missing stages
remain absent and the UI renders them as unavailable.

The retina pack dtype is `uint32`, not `int8`, unless a separate, explicitly lossy decoded
view is produced with its own name and scale. Converting the packed retina directly to
`int8` destroys both distance and material information.

## 10. Not part of this change

- recording all beings;
- changing the default diagnostic cadence;
- changing `.replay`;
- changing weights, formulas or controller behaviour;
- recording the slow-brain hidden/CfC state;
- counterfactual ablation;
- presenting the functional 3D layout as measured fly anatomy.

## 11. Implemented and measured (16/09/2026)

The exact selected-being kernel trace is implemented in `brain.wgsl` binding 15 and decoded
through the stable 481-word layout in `src/gpu/brain/observation.rs`. It exports all eight
intermediate arrays plus `l2_std`; the existing intent readback supplies `final_intent`.
The selector is enabled only on a sampled neural dispatch. A `brain_skip` tick records
`neural_dispatch:false` and `stages:null`, so a stale trace cannot masquerade as a new one.
At each generation boundary the recorder also saves the selected slot as
`observed_genome_genNNNN.bin`; every observation row names that file. Contributions can
therefore be joined to the exact weights that produced them even after evolution changes
the population.

The diagnostic mapping request is now issued before the tick's existing shared
`device.poll(Wait)`. This removes the former second synchronous poll on every sampled tick.
The present `--driving-diagnostics --vision-every N` path is still the sparse historical
profile. The bounded asynchronous ring described in section 6 remains required before a
new full-cadence recording profile is exposed; no such profile is claimed here.

Real-GPU validation on RADV/NAVI22 used 128 consecutive samples of selected population slot
1. A separate CPU reconstruction from the recorded stages and the champion weights measured:

| quantity | maximum absolute error |
|---|---:|
| L1 matrix result | `3.5762787e-7` |
| L1 activation/gate quantisation | exactly `0` integer units |
| L2 matrix result | `4.7683716e-7` |
| L2 standard deviation | `5.9604645e-8` |
| final intent from adapted stages | `1.7881393e-7` |

All are below the `1e-5` contract. `final_intent[0..3]` equalled
`body.motor_trace.raw_intent` exactly in every sample. The run is stored at
`/media/llm-data/scratch/brain-observation-selected1-correct-20260916/`.

A final 128-sample run exercised the generation-to-genome association itself. All rows for
selected slot 1 named `observed_genome_gen0000.bin`; the file was present (99,184 bytes),
and reconstruction directly from that file produced maximum errors of `1.7881393e-7` for
L1, `4.7683716e-7` for L2 and `1.1920929e-7` for final intent. The evidence is under
`/media/llm-data/scratch/final-observation-genome-link-20260916/`.

The matched 48-being, 400-tick cost sweep used 40 profiler samples per arm:

| recording | tick median | tick p95 | summed GPU median | summed GPU p95 |
|---|---:|---:|---:|---:|
| off | 2.2269 ms | 2.5971 ms | 0.4261 ms | 0.4335 ms |
| every 16 ticks | 2.2638 ms | 2.6029 ms | 0.4275 ms | 0.4467 ms |
| every tick | 2.4551 ms | 2.8788 ms | 0.4356 ms | 0.6466 ms |

Normal cadence adds 0.0369 ms to the median tick (1.7%). Mapping every tick adds
0.2282 ms (10.2%), which is direct evidence for retaining the ring requirement before
exposing a production full-cadence profile. Raw logs are in
`/media/llm-data/scratch/brain-observation-cost-20260916/`.
