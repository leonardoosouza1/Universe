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
