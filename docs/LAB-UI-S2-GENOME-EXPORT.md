> **CORRECTION (2026-09-16, later the same day, coordinator-routed follow-up).** The
> "Diagnosis" section below ranked "being/genome identity" as the MOST LIKELY cause of
> `brain_step`'s mismatch and said so explicitly as a hypothesis, not a measurement — it
> was wrong to weight it that heavily without testing it. It has now been tested
> directly: 4 real `--resume --resume-mix 1` runs where the recorded being (population
> slot 0) is PROVEN, by direct sha256 comparison, to own the exact genome in `genome.bin`
> (see "GENOME OWNERSHIP: VERIFIED" below for the hashes). **The error did NOT drop.**
> It stayed at the same order of magnitude (0.77-0.996 max, vs. 1.659 on the
> unverified fixture). This falsifies hypothesis #1 below as the dominant cause and
> confirms case (3) from the coordinator's framing: **the reference forward itself is
> incomplete**, not just fed the wrong genome. See "CASE (3) CONFIRMED" below for the
> per-run/per-channel error table and the identified first divergent stage. The original
> text below is left as written — it was a stated hypothesis, correctly labelled as
> unverified ("most likely dominant cause"), and the record of being wrong is worth more
> than deleting it (`docs/CLAUDE.md`'s own rule for this directory).

# lab-ui PRD S2 — genome export and the CPU reference forward (2026-09-16)

Two new CPU-only binaries for the trace-v1 pipeline (`lab-ui/PRD.md` §8, row S2):
`src/bin/genome_export.rs` (`.bin` genome -> `weights.json`) and
`src/bin/brain_step.rs` (genome + a recorded `(wave, cfc_state)` sample -> the reference
forward through `raw_intent`, with optional per-stage ablation). `src/bin/track_export.rs`
(track -> `track.json`) is the third, unrelated to the genome layout, not covered here.

This note is the short version for anyone touching the genome layout; the full reasoning
lives in each binary's own module doc comment and in `neuromod-lab/src/lab_api/DEVIATIONS.md`
(items #4-#7), which this note summarizes.

## `genome_export`: what `brain_flow` means here

There is no literal `brain_flow` array in the genome code. `genome_export`'s `brain_flow`
is `gpu/genome_io.rs`'s own on-disk block order (l1, l2, kan_knots, slow_*, tau, sensors,
neuron_*, gates_w, hip, fly) -- the one place that layout is documented as canonical
("DO NOT reorder"). This is explicitly NOT the per-tick kernel dispatch order (tau feeds
the CfC before L1 runs; kan_knots is applied on the HOST after L1+L2, to `raw_intent`,
not before; `slow_*` is a parallel GPU kernel pair alongside `brain.wgsl`, not
sequential with it) -- `weights.json` itself carries a `brain_flow_note` field saying so,
and ARCHITECTURE.md §2 has the real per-tick order.

Verified: `genome_sha256` in `weights.json` matches the fixture's own recorded
`hashes.checkpoint` (`29c9a51b...c117045`) byte for byte -- see `genome_export`'s own
`cargo test --release --bin genome_export`.

## `CfCCell::quantize_state_i8` (new, extracted from `CfCCell::update`)

`src/cfc.rs`'s `update()` had `output[i] = (filtered * 127.0).clamp(-127.0, 127.0) as i8`
inline as its last line. Extracted into `CfCCell::quantize_state_i8(state: f32) -> i8` so
`brain_step` (which is GIVEN a resident CfC state from a trace sample, not deriving one
from a wave history) reproduces the exact same quantization `update()` uses, instead of a
second copy of the formula. `update()`'s behaviour is unchanged (same line, now a named
call) -- `cargo test --release --lib` (480 passed) and `cargo test --release --bin
gpu_evolve` (119 passed) both stayed green after this change.

## `brain_step` stops at `raw_intent`

`raw_intent` -> `after_head` -> `after_modulation` -> `final_command` is
`map_native_action` (`src/bin/gpu_evolve/native_action.rs`), a pure, already-tested
function -- but it lives in the `gpu_evolve` BINARY crate, not the `universe` LIBRARY
crate, so a separate CPU-only binary cannot call it without either moving it into the
library (a real API change, not done without sign-off) or copying its logic (forbidden
by the task: "do NOT reimplement the brain in a second place"). `raw_intent` is also the
PRD's own literal acceptance target, so this is not a scope cut against the spec.

## Acceptance number: reference forward vs. recorded `raw_intent`, 50 samples

Command (from `.worktrees/universe-s2-pack`):

```
./target/release/brain_step --batch \
  lab-ui/fixtures/trace-v1/universe-b2-gen0/genome.bin \
  <50-sample JSON built from the fixture's wave/cfc/raw_intent blocks, ticks 16..800> \
  /media/llm-data/scratch/s2-pack/brain_step_report_b2gen0.json
```

Result: **max abs error 1.659, mean 0.835, median 0.811, min 0.395** (over the 3
`raw_intent` channels x 50 samples; `raw_intent` values are `tanh`-bounded to [-1,1], so
1.66 is close to the largest possible single-channel disagreement, 2.0). The mismatch is
large and present from the very first sample (tick 16) -- not tuned, per the task's own
instruction to report rather than chase a match.

**Diagnosis, in order of estimated contribution:**

1. **Being/genome identity (most likely dominant cause).** `genome.bin` in this fixture
   is `evolve_best_gen0000.bin`, the generation's FITNESS CHAMPION. The `wave`/`cfc`/
   `raw_intent` samples are population slot 0 (`being=0`). The source run
   (`b2_cost_s113`) did not pass `--resume`, so slot 0 is NOT verified to be the same
   individual as the champion (`lab-ui/contract/DEVIATIONS.md` #2, `main.rs`'s own doc
   comment: "gpu_evolve uses stable population slots; slot zero must not be presented as
   a measured champion"). Feeding one individual's activity through another,
   unrelated individual's weights would produce exactly this kind of large,
   non-decreasing-with-tick disagreement -- two independently random ~18k-weight L1
   matrices have no reason to agree.
2. **No persistent output fatigue.** `brain_step` is stateless (`out_fatigue=0` every
   call); the real `execute_cycle`/production path accumulates fatigue across ticks.
   This shrinks the real signal somewhat; it cannot by itself explain a ~0.83 mean error
   on a [-1,1]-bounded quantity, but it is not zero either.
3. **P1.7 neuron mechanisms not applied.** `brain_step` composes plain `BitLinear`/
   `FloatLinear` calls; it does not gate L1's activation by
   `neuron_alpha`/`neuron_period_l1`/`neuron_threshold_l1`/`neuron_activation` the way
   production does when a run's `disabled_flags` is empty (mechanisms ACTIVE). This
   fixture's `disabled_flags` is empty, so the mechanisms were active in the real run.

Not tuned to close the gap, per instruction. A follow-up that wants a real parity number
should first re-run this same batch against a `--resume` run (where slot 0 is documented
to be the prior generation's elite) before attributing the remaining gap to (2) or (3).

## Tests

`cargo test --release --bin genome_export --bin brain_step --bin track_export`: 12 unit
tests, all passing (6 brain_step, 3 genome_export, 3 track_export -- see each binary's
own `#[cfg(test)] mod tests`). `cargo test --release --bin gpu_evolve` (119 passed) and
`cargo test --release --lib` (480 passed, 16 GPU tests `#[ignore]`d) both stayed green.

## GENOME OWNERSHIP: VERIFIED (2026-09-16 follow-up)

For `--resume --resume-mix 1`, population slot 0 is `best.clone()` with **no scaling
applied at all**, unconditionally (`Universe/src/bin/gpu_evolve/main.rs`):

```rust
pop.push(best.clone()); // elite exato   <- slot 0, BEFORE the mix loop below
while pop.len() < n_seres {
    let mut filho = best.clone();
    for w in filho.l1.iter_mut() { *w *= mix; }   // only slots 1..n get scaled
    for w in filho.l2.iter_mut() { *w *= mix; }
    pop.push(filho);
}
```

Confirmed by direct `sha256sum` of `<run>/evolve_best_gen0000.bin` against its source
checkpoint (`/media/llm-data/snapshots/b1-champions/<arm>_s<seed>.bin`, per
`neuromod-lab/scripts/enqueue_b3_b2.py`'s B3 recipe: the champion is copied to
`<out>/evolve_best_gen0000.bin` before `--resume` runs) for 4 real runs:

```
sha256sum /media/llm-data/snapshots/b1-champions/b1_fixed_s31.bin \
          /media/llm-data/universe-runs/neuromod-lab-taxi/b3_fixed_s31_t8005/evolve_best_gen0000.bin
# 3417a54925eb8da678cefde69735639492156232c12494a5b3ed22ec43f9dc01  (both)

sha256sum /media/llm-data/snapshots/b1-champions/b1_legacy_s31.bin \
          /media/llm-data/universe-runs/neuromod-lab-taxi/b3_legacy_s31_t8005/evolve_best_gen0000.bin
# 7ba04537b799e9a8fb503c6306232d55be85b3145308f5b79502311aa36dc775  (both)

sha256sum /media/llm-data/snapshots/b1-champions/b1_fixed_s113.bin \
          /media/llm-data/universe-runs/neuromod-lab-taxi/b3_fixed_s113_t8200/evolve_best_gen0000.bin
# 77e3a0c5d77220c9c45a70254c8f6ad10e50dbf0076b73fdb55877ab2e7fda45  (both)

sha256sum /media/llm-data/snapshots/b1-champions/b1_legacy_nocost_s77.bin \
          /media/llm-data/universe-runs/neuromod-lab-taxi/b3_legacy_nocost_s77_t8005/evolve_best_gen0000.bin
# ed72e4d31cbd73ccbec34fd062f09c5fc68ce347bc35b85fdb4b8f985b1f3770  (both)
```

All four: exit code `0`, both hashes identical, WHOLE FILE (not just L1/L2). This is now
`producer.py::_counterfactual_valid`'s empirical basis (neuromod-lab worktree,
`src/lab_api/DEVIATIONS.md` #12).

## CASE (3) CONFIRMED: the reference forward does not match even with a verified genome

Samples built from each run's own produced trace (`neural.trace`'s `wave`/`cfc`/
`raw_intent` blocks, all 10 recorded samples per run -- `--vision-every 2000` on
`--ticks 20000`), genome = the run's own `evolve_best_gen0000.bin` (verified above to be
the exact loaded champion):

```
./target/release/brain_step --batch <run>/evolve_best_gen0000.bin <10-sample json> <report.json>
```

| run | control path | max abs err | throttle mean | steer mean | brake mean |
|---|---|---|---|---|---|
| `b3_fixed_s31_t8005` | fixed (continuous probe) | 0.766 | 0.465 | 0.260 | 0.288 |
| `b3_legacy_s31_t8005` | legacy (4 legacy flags) | 0.961 | 0.562 | 0.184 | 0.200 |
| `b3_fixed_s77_t8005` | fixed | 0.984 | 0.498 | 0.308 | 0.623 |
| `b3_legacy_nocost_s77_t8005` | legacy | 0.996 | 0.486 | 0.290 | 0.241 |

Exit code `0` for all four `brain_step --batch` runs. **The error did not drop within
the stage's quantisation** (i8 step ~1/127 = 0.008; these are 100-300x that). Fixed vs.
legacy control path shows no consistent difference (0.766/0.984 fixed vs. 0.961/0.996
legacy) -- confirms the 4 legacy flags (`--throttle-arbitration legacy --wall-probes
legacy --slow-kinematics legacy --legacy-reverse-controller`) do NOT change the forward
`brain_step` computes: all four are Stage-B (native_action mapping) or sensor-generation
concerns (verified by reading their consumers: `throttle_arbitration`/
`reverse_controller` are read only inside `map_native_action`, which `brain_step`
explicitly does not reach; `wall_probes` selects which CPU probe function fills
`dist_wall_*` in the WAVE before it reaches the brain at all -- brain_step takes the
recorded wave as given, so it cannot see this choice; `slow_kinematics` only changes
`slow_gather.wgsl`'s channel map, i.e. the SLOW brain's input, a separate pathway
`brain_step` does not compute). None of the four touch L1/L2/CfC.

**Throttle has the largest mean error in all 4 runs (0.46-0.56).** Steer is next
(0.18-0.31), brake is the most variable (0.20-0.62).

### First divergent stage (best identification available from this trace format)

The trace format does not record `l1`/`l2`/`intent` (all three are in `manifest.absent`,
"not recorded by this binary" -- `driving_diagnostics.rs` only samples `retina`, `wave`,
`fast_state` (= `cfc`) and the motor-chain fields). This means there is no recorded
ground-truth L1 output to diff `brain_step`'s `l1_raw` against directly -- the following
is the best inference available, not a direct numeric L1-level comparison, and that
limitation is itself worth stating plainly.

Inspecting `weights.json` for `b3_fixed_s31_t8005`'s genome (`b1_fixed_s31`, the
champion under test) shows the P1.7 per-neuron mechanisms are NOT neutral:

```
neuron_alpha_l1:      min 0.000  max 0.339  mean 0.045  (41/96 neurons nonzero)
neuron_period_l1:     min 1.0    max 4.0    mean 1.48   (96/96 neurons != the 1.0 "always fires" default)
neuron_threshold_l1:  min 0.000  max 0.165  mean 0.020  (39/96 neurons nonzero)
neuron_activation_l1: min 0.0    max 3.0    mean 0.083  (3/96 neurons on a non-default activation)
```

and this run's `manifest.disabled_flags` is `[]` -- none of `--no-neuron-memory
--no-neuron-period --no-neuron-threshold --no-heterogeneous-activation` were passed, so
all four P1.7 mechanisms were ACTIVE in the production forward that generated the
recorded `raw_intent`. `brain_step` composes plain `BitLinear::forward` /
`FloatLinear::forward` -- it does not apply any of the four. Concretely, with
`neuron_period_l1` averaging 1.48 (some neurons only recompute every 2-4 ticks) and
`neuron_alpha_l1` nonzero on 41/96 neurons (EMA memory blending this tick's raw value
with a REMEMBERED prior value), L1's real output at any given tick is a function of
recent TICK HISTORY that a single stateless sample cannot reconstruct -- this is a
structural gap, not a missing scale factor: closing it requires `brain_step` to carry
persistent per-neuron state across several consecutive ticks (i.e. take a short window
of consecutive samples, not one), which is a larger scope change than "a recorded input
vector" (the task's own phrasing for `brain_step`'s input).

**Conclusion: L1's output (`l1_raw`/`l1_act`) is the first stage the reference forward
diverges at**, and the identified mechanism is the P1.7 per-neuron memory/period/
threshold/activation gates, which `brain_step` does not implement and structurally
cannot implement correctly without persistent cross-tick state. Output fatigue
(`out_fatigue=0` approximation) is a secondary, smaller contributor for the same
structural reason (also needs cross-tick state). **`brain_step` is NOT YET VALID for
raw_intent parity checking against production traces with any P1.7 mechanism active**
(i.e. `disabled_flags` not covering all 4 flags) -- it remains correct as a pure,
ablation-capable forward over the STATIC weights (per its own unit tests), just not a
parity oracle against recorded production behaviour yet.

**For a future `/api/counterfactual` route (PRD S5, not built in this task):** until
`brain_step` carries the P1.7 per-neuron state across a tick window, that route should
return HTTP 501 with a message along these lines: `"counterfactual recomputation is not
yet valid: brain_step does not implement the P1.7 per-neuron memory/period/threshold/
activation mechanisms, which requires cross-tick state; see
Universe/docs/LAB-UI-S2-GENOME-EXPORT.md 'CASE (3) CONFIRMED'. Measured error on 4
genome-verified runs: 0.77-0.996 max abs error on raw_intent, i8-quantisation is
~0.008."` -- gated on `disabled_flags` covering all four P1.7 flags AND
`counterfactual_valid=True`, not on `counterfactual_valid` alone.

### What this does NOT change

`genome_export`'s stage export, `brain_flow` ordering, and `CfCCell::quantize_state_i8`
extraction are unaffected -- none of them claimed raw_intent parity; only `brain_step`'s
forward does, and only that claim is now marked unverified/known-incomplete pending the
cross-tick state work above.
