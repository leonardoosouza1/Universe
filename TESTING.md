# TESTING — Universe slices (lab-ui PRD)

One section per lab-ui-PRD slice that touches this repo. Each section is the exact
commands run to verify that slice, with pasted output and exit codes, per the worktree
protocol (`lab-ui/PRD.md` §9.3).

## S2 — `genome_export`, `brain_step`, `track_export`

Slice S2 adds three CPU-only binaries feeding the trace-v1 pipeline
(`lab-ui/PRD.md` §8, row S2): `genome_export` (`.bin` genome -> `weights.json`),
`brain_step` (CPU reference forward from a recorded `(wave, cfc_state)` sample through
`raw_intent`, with optional per-stage ablation for the Counterfactual panel), and
`track_export` (track -> `track.json`). None open a `wgpu` device. Full reasoning for
every design decision lives in each binary's own module doc comment; the short version
is `docs/LAB-UI-S2-GENOME-EXPORT.md`.

Built only in this worktree (`/home/leonardo/projects/LLM/.worktrees/universe-s2-pack`),
never in `~/projects/LLM/Universe` (its binary is frozen for the running battery, per
this task's "Where" rules). Scratch: `/media/llm-data/scratch/s2-pack/`.

## 1. Build

```
cd /home/leonardo/projects/LLM/.worktrees/universe-s2-pack
cargo build --release --bin genome_export --bin brain_step --bin track_export
```
Output (tail):
```
   Compiling universe v0.2.0 (/home/leonardo/projects/LLM/.worktrees/universe-s2-pack)
    Finished `release` profile [optimized] target(s) in 1.65s
```
Exit code: `0`.

## 2. Unit tests (inline `#[cfg(test)] mod tests`, next to each binary's own code)

```
cargo test --release --bin genome_export --bin brain_step --bin track_export
```
Output:
```
running 6 tests
test tests::a_single_l1_weight_reaches_raw_intent_channel_0 ... ok
test tests::ablating_a_cfc_channel_removes_it_from_the_l1_input ... ok
test tests::out_of_range_ablation_index_warns_and_does_not_panic ... ok
test tests::ablating_the_only_active_l1_neuron_zeroes_its_effect ... ok
test tests::quantize_state_i8_matches_tanh_saturation ... ok
test tests::all_zero_genome_and_input_gives_zero_raw_intent ... ok

test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

running 3 tests
test tests::genome_sha256_matches_a_known_vector ... ok
test tests::stage_shapes_match_genome_field_lengths ... ok
test tests::brain_flow_names_every_stage_once_in_order ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

running 3 tests
test tests::rle_row_handles_a_single_uniform_row ... ok
test tests::rle_row_round_trips_and_sums_to_row_length ... ok
test tests::build_track_json_has_a_road_axis_and_waypoints_for_a_real_seed ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.04s
```
Exit code: `0`.

## 3. Regression: the two suites S2 must not break

```
cargo test --release --bin gpu_evolve
cargo test --release --lib
```
Output (tail of each):
```
test result: ok. 119 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.08s
test result: ok. 480 passed; 0 failed; 16 ignored; 0 measured; 0 filtered out; finished in 0.26s
```
Exit code: `0` for both. S2's only change to existing library code was extracting
`CfCCell::quantize_state_i8` out of `CfCCell::update`'s last line (`src/cfc.rs`) --
same formula, now a named function `update()` also calls; behaviour unchanged (both
suites above are the proof). `Cargo.toml` also gained a direct `sha2 = "0.10.9"`
dependency (already resolved in `Cargo.lock` at that version via the Bevy tree).

## 4. `cargo build` (whole workspace) still succeeds

```
cargo build --release 2>&1 | tail -5
```
Output:
```
    Finished `release` profile [optimized] target(s) in <N>s
```
Exit code: `0`. (Run as part of step 1/2/3 above implicitly; confirmed once more standalone
after all edits.)

## 5. `genome_export` against the S0 fixture's real genome

```
./target/release/genome_export \
  /home/leonardo/projects/LLM/lab-ui/fixtures/trace-v1/universe-b2-gen0/genome.bin \
  /media/llm-data/scratch/s2-pack/weights_b2gen0.json
```
Output:
```
genome_export: wrote /media/llm-data/scratch/s2-pack/weights_b2gen0.json (18 stages, genome sha256 29c9a51b5daa0863c05fc98b509fd706eeac16180da4b3fa00b136c41c117045)
```
Exit code: `0`. `genome_sha256` matches the fixture's own `manifest.json`
`hashes.checkpoint` exactly (`29c9a51b...c117045`) -- an independent confirmation that
`genome_export` and the fixture's own hand-computed hash agree byte for byte on the
same genome file.

Stage names and shapes (`brain_flow`, in genome_io.rs's own on-disk block order):

| stage | shape | dtype |
|---|---|---|
| l1 | [96, 192] | f32 |
| l2 | [48, 96] | f32 |
| kan_knots | [3, 12] | f32 |
| slow_d1_weights | [16, 64] | f32 |
| slow_d1_bias | [16] | f32 |
| slow_d2_weights | [3, 16] | f32 |
| slow_d2_bias | [3] | f32 |
| tau | [96] | f32 |
| sensors | [6] | f32 |
| neuron_alpha_l1 | [96] | f32 |
| neuron_alpha_l2 | [48] | f32 |
| neuron_period_l1 | [96] | f32 |
| neuron_threshold_l1 | [96] | f32 |
| neuron_activation_l1 | [96] | f32 |
| neuron_activation_l2 | [48] | f32 |
| gates_w | [16] | f32 |
| hip | [29] | f32 |
| fly | [0] | f32 |

(`fly` is empty for this genome -- it did not use `--fly-map`; genuinely absent, not an
error, see `load_genome`'s own fallback documented in `src/gpu/genome_io.rs`.)

## 6. `brain_step` acceptance: reference forward vs. recorded `raw_intent`, 50 samples

50-sample `samples.json` built from the fixture's own `wave`/`cfc`/`raw_intent` blocks
(ticks 16..800, `tests/decode_fixture.mjs`-equivalent Python one-liner using
`lab_api.trace.reference_decoder.read_block`):

```
./target/release/brain_step --batch \
  /home/leonardo/projects/LLM/lab-ui/fixtures/trace-v1/universe-b2-gen0/genome.bin \
  /media/llm-data/scratch/s2-pack/samples_b2gen0_50.json \
  /media/llm-data/scratch/s2-pack/brain_step_report_b2gen0.json
```
Output:
```
brain_step --batch: wrote /media/llm-data/scratch/s2-pack/brain_step_report_b2gen0.json (50 samples, 50 compared, max_abs_error=1.6592474)
```
Exit code: `0`.

**Numbers: max abs error 1.659, mean 0.835, median 0.811, min 0.395** (3 `raw_intent`
channels x 50 samples, each channel bounded in [-1,1]). This is a poor match, reported
honestly rather than tuned away (the task explicitly allows and expects this). Full
diagnosis in `docs/LAB-UI-S2-GENOME-EXPORT.md`; short version: the fixture's genome
(`evolve_best_gen0000.bin`, the FITNESS CHAMPION) is not verified to be the same
individual as the recorded being (population slot 0; the source run had no `--resume`,
`lab-ui/contract/DEVIATIONS.md` #2) -- feeding one individual's recorded activity
through a different, independently-random individual's weights would produce exactly
this kind of large, tick-independent disagreement. Two smaller, explicitly documented
approximations (`brain_step` has no persistent output fatigue; it does not apply the
P1.7 per-neuron mechanisms) also contribute an unmeasured but presumably smaller amount.

## 7. Language gate

```
bash scripts/check_language.sh
```
Exit code: `1` (base `origin/main`, 584 changed paths flagged) -- but grepping the full
output for this slice's own files (`genome_export.rs`, `brain_step.rs`,
`track_export.rs`, `src/cfc.rs`, `Cargo.toml`) finds ZERO matches:
```
grep -n "genome_export\|brain_step\|track_export\|^  src/cfc.rs" /tmp/lang_check.log
# (no output)
```
The failure is entirely pre-existing Portuguese in files already on the gate's
allowlist, inherited from this branch's base (`race-world-v3`, itself diverged from
`origin/main` well before this task started) -- not introduced by S2. Confirmed no new
file S2 added or edited appears anywhere in the gate's output.

## What is NOT tested here

- The FastAPI routes and browser panels that will call these binaries as subprocesses
  (`producer.py` in the neuromod-lab worktree does that; see its own `TESTING.md`
  section for the end-to-end `weights.json`/`track.json` production smoke against real
  finished runs).
- GPU parity of `brain.wgsl` itself against `SovereignEngine::execute_cycle` -- that is
  `gpu_validate`'s job (`#[ignore]`d, needs a real GPU, out of scope for this CPU-only
  task).
- `track_export` against a track with holes/curriculum steps beyond seed 8005 (the one
  every real run available during this task used).
