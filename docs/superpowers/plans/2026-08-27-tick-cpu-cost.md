# Phase 1 — the 21 ms of CPU in a 24 ms tick

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the two CPU costs that own the tick — 7.58 ms preparing per-being inputs in Rust and 13.28 ms blocked waiting on ~2 ms of GPU work — without changing a single simulated number.

**Architecture:** Every task here is a REFACTOR, not a behaviour change. The gate is the same for all of them: `obs.f32` must stay byte-identical (sha256 `5f1cf08e4764c78bec1b9341f625ce62afd7adc760becbf88884b4c81ee5fa35`, median `2939m`). Task 1 sub-profiles the 7.58 ms so Tasks 2-3 attack a measured target instead of a guess — the same order that made Phase 0 work. Task 4 is gated on what Tasks 2-3 leave behind, because the card is downclocked *because* it idles, so removing CPU work may shrink the wait for free.

**Tech Stack:** Rust 2024 (MSRV 1.85), wgpu on Vulkan/RADV, WGSL. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-26-gpu-resident-scenarios-design.md` (§3, rewritten 2026-08-26 after the measurement retired "K ticks per submit" as the mechanism)

## Global Constraints

- **English only** for every new or touched line — identifiers, comments, doc comments, log and panic messages, test names, commit messages. Gate: `BASE_REF=<base> bash scripts/check_language.sh`.
- **`cargo fmt -p universe --check` must pass.** Never `cargo fmt --all` — it rewrites `vendor/`, and `scripts/check_vendor.sh` must keep reporting 5/5 identical.
- **`cargo test --release` must stay green.** Baseline 420 tests as of `c834f78`.
- **The bit-identical gate is the acceptance criterion for Tasks 2 and 3**, not a formality: these are refactors, so any change in `obs.f32` means the refactor changed the simulation and must be reverted, not explained.
- **Every performance claim carries its command and a median over ≥25 samples with its range.** Kernel and CPU-phase numbers move ±10% with GPU clock state (the card sits at DPM level 1, 1205-1995 MHz, 21-24 W against a ~2600 MHz / ~200 W ceiling). A point estimate from 4 samples is how a retired figure got published once already.
- **No GPU training.** All measurement here is `--gens 1` evaluation.
- `src/bin/gpu_evolve/main.rs` is ~2600 lines and `context.rs` ~2300, both against a ~600-line house cap (`docs/STYLE.md`). Do not grow them; new code goes in new modules.
- Heavy outputs go to `/media/llm-data/`, never the repository. `df -h /media/llm-data` before writing — it sits near 90% full.

## Measured starting point

768 beings, 400 ticks, `--perfil-kernel 50`, tick wall 24.10 ms:

| phase | ms | share |
|---|---|---|
| `rec.cpu_prep` — per-being Rust before any GPU call | 7.58 | 31% |
| `rec.gpu_record` — bind groups + recording passes | 0.17 | 1% |
| `cpu.poll` — blocked; the ~2 ms of kernels run in here | 13.28 | 55% |
| `physics.step` | 1.80 | 7% |
| readbacks + per-being loops after the poll | 0.19 | 1% |

`rec.cpu_prep` scales **linearly**: 0.922 / 2.017 / 3.730 / 7.570 ms at 96 / 192 / 384 / 768 beings — 8.2× for 8× beings, against 64× if it were quadratic. The `comp_stress` O(n²) block is NOT the cause; it sits behind `--interocepcao`, which these runs did not pass. That hypothesis was tested and refuted before this plan was written.

What is known: ~9.9 µs per being, spread over ~12 `for i in 0..n` passes, 55 `states[i]` accesses and 12 `Vec` allocations per tick. What is NOT known: which of the twelve dominates. Task 1 answers that.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/bin/gpu_evolve/context.rs` (modify) | The tick. Tasks 1-3 touch its prep region (~lines 500-1285) and its submit/poll boundary. |
| `src/bin/gpu_evolve/tick_buffers.rs` (create, Task 3) | Owns the reusable per-being scratch buffers so `context.rs` stops allocating twelve `Vec`s per tick. One responsibility, one file. |

---

## Task 1: Sub-profile the 7.58 ms

**Files:**
- Modify: `src/bin/gpu_evolve/context.rs` (the prep region, ~500-1285)

**Interfaces:**
- Consumes: the `cpu_phases: Vec<(&'static str, f32)>` accumulator and the `prints_this_tick` gate, both already in `tick()` from commit `c834f78`.
- Produces: `--perfil-kernel` prints one `prep.*` row per per-being pass, summing to `rec.cpu_prep` within 5%.

- [ ] **Step 1: Find every per-being pass in the prep region**

Run: `grep -n "for i in 0..n" src/bin/gpu_evolve/context.rs | head -20`
Expected: roughly twelve sites between the tick's start and the first `create_bind_group`.

List them in the report with a one-line description of what each computes, taken from the code, not guessed.

- [ ] **Step 2: Bracket each pass with a timer**

Follow the pattern already in the file — do not invent a second one:

```rust
let t_prep_dists = prints_this_tick.then(std::time::Instant::now);
// ... the existing pass, unchanged ...
if let Some(t) = t_prep_dists {
    cpu_phases.push(("prep.dists", t.elapsed().as_secs_f32() * 1000.0));
}
```

Name each row for what the pass computes (`prep.dists`, `prep.retina_view`, `prep.prop`, `prep.extra`, `prep.extra2`, `prep.body`, …), not for its line number.

- [ ] **Step 3: Build and measure**

```bash
cargo build --release --bin gpu_evolve
D=/media/llm-data/scratch/prep; rm -rf $D; mkdir -p $D
cp /media/llm-data/universe-runs/ereg-m4-m4-s77/evolve_best_gen0043.bin $D/
./target/release/gpu_evolve --retomar --out $D --gens 1 --seres 768 --ticks 400 \
    --width 28 --out-beta 0.0 --pistas 8005 --seed 77 --perfil-kernel 50
```

Expected: the `prep.*` rows sum to within 5% of `rec.cpu_prep`. **If they do not, the passes were not all bracketed** — find the missing one before continuing, because an unmeasured remainder is exactly what this task exists to eliminate.

- [ ] **Step 4: Verify the instrument is inert when off**

```bash
D=/media/llm-data/scratch/prepnoop; rm -rf $D; mkdir -p $D
cp /media/llm-data/universe-runs/ereg-m4-m4-s77/evolve_best_gen0043.bin $D/
./target/release/gpu_evolve --retomar --out $D --gens 1 --seres 48 --ticks 20000 \
    --width 28 --out-beta 0.0 --mascara-regioes --neuromod-regioes \
    --pistas 8005,8200,9001 --seed 77 --observar $D/obs.f32
sha256sum $D/obs.f32
```

Expected: `5f1cf08e4764c78bec1b9341f625ce62afd7adc760becbf88884b4c81ee5fa35`, median `2939m`.

- [ ] **Step 5: Commit, with the table in the message**

```bash
cargo fmt -p universe
git add -A
git commit -m "perf(gpu_evolve): sub-profile rec.cpu_prep, one row per per-being pass

[paste the measured prep.* table, the sum against rec.cpu_prep, and the command]

Off by default, gated on --perfil-kernel. obs.f32 byte-identical.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 1 result — what it redirected

Measured (`--gens 1 --seres 768 --ticks 400 --perfil-kernel 10`, n=40, median with range):

| pass | median ms | share of `rec.cpu_prep` |
|---|---|---|
| **`prep.racing_line`** | **3.742** (3.140–5.595) | **52.5%** |
| `prep.extra2` (holds the O(n²) block) | 1.362 | 19.1% |
| `prep.dists` | 0.787 | 11.0% |
| `prep.extra` | 0.640 | 9.0% |
| `prep.prop` | 0.594 | 8.3% |
| `prep.retina_view`, `prep.body`, `prep.encoder_setup` | 0.010 | 0.1% |
| sum | 7.136 | 100.13% of `rec.cpu_prep` |

**The plan's original Tasks 2 and 3 were aimed at noise.** Killing the six per-tick `Vec`
allocations and fusing passes together address, at most, the 47.5% that is not
`racing_line` — and allocation is a small part of even that. One function is 52% of the
prep and 15.5% of the whole tick. Tasks 2-4 below are rewritten to attack it.

The mechanism, read from the code rather than assumed:

- `RacingLine::project(pos)` (`src/simulation/racing_line.rs:143`) is a **full linear scan
  over every track segment**, with a `normalize_or_zero()` (a square root) inside the
  best-so-far branch.
- `lateral_offset_from_line` (`:176`) calls `project` once.
- `lookahead_offset` (`:194`) calls `project` **again, with the same `pos`**, then runs a
  **second linear scan** over `self.arc` to find the target segment.

Per being, per tick: two full projections of the same point plus one more linear scan.
At 768 beings and ~77-129 segments per track that is roughly 200 000 segment evaluations
per tick, which is the 3.74 ms.

---

## Task 2: Project once per being instead of twice

**Files:**
- Modify: `src/simulation/racing_line.rs:176-230`
- Test: `src/simulation/racing_line.rs` (`#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `pub struct Projection { pub seg: usize, pub t: f32, pub point: Vec2, pub tangent: Vec2 }`
  - `pub fn project_at(&self, pos: Vec2) -> Option<Projection>` — the existing private `project`, made public and returning a named struct instead of a 4-tuple.
  - `pub fn lateral_offset_from(&self, p: &Projection, pos: Vec2) -> f32`
  - `pub fn lookahead_offset_from(&self, p: &Projection, s: f32) -> f32`
  - The existing `lateral_offset_from_line` and `lookahead_offset` stay, implemented in terms of the new pair, so no other caller breaks.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod projection_reuse_tests {
    use super::*;

    /// A shared projection must produce bit-identical results to projecting twice.
    /// This is the whole point of the task: it is a caching change, not a numerical
    /// one, and the simulation's output is gated on nothing moving.
    #[test]
    fn sharing_one_projection_matches_projecting_twice() {
        let line = RacingLine::from_points(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(100.0, 0.0),
            Vec2::new(200.0, 50.0),
            Vec2::new(300.0, 50.0),
        ]);
        for &(x, y) in &[(10.0, 3.0), (150.0, -8.0), (250.0, 60.0), (299.0, 49.0)] {
            let pos = Vec2::new(x, y);
            let p = line.project_at(pos).expect("projection");

            let lat_old = line.lateral_offset_from_line(pos, Vec2::ZERO);
            let lat_new = line.lateral_offset_from(&p, pos);
            assert_eq!(
                lat_old.to_bits(),
                lat_new.to_bits(),
                "lateral differs at {pos:?}: {lat_old} vs {lat_new}"
            );

            for &s in &[5.0f32, 20.0, 80.0] {
                let look_old = line.lookahead_offset(pos, s);
                let look_new = line.lookahead_offset_from(&p, s);
                assert_eq!(
                    look_old.to_bits(),
                    look_new.to_bits(),
                    "lookahead differs at {pos:?}, s={s}: {look_old} vs {look_new}"
                );
            }
        }
    }
}
```

If `RacingLine` has no `from_points` constructor, read the file and use whichever
constructor exists; do not add one for the test's convenience.

- [ ] **Step 2: Run it and watch it fail**

Run: `cargo test --release --lib projection_reuse 2>&1 | tail -20`
Expected: FAIL — `project_at`, `lateral_offset_from`, `lookahead_offset_from` not found.

- [ ] **Step 3: Implement**

Rename the private `project` to `project_at`, make it public, and return `Projection`
instead of the tuple. Split the two public functions so the projection is a parameter.
**Move the arithmetic, do not rewrite it** — every expression stays byte-for-byte the same,
or the `to_bits()` assertions above will catch it.

- [ ] **Step 4: Run it and watch it pass**

Run: `cargo test --release --lib projection_reuse 2>&1 | grep "test result"`
Expected: `test result: ok. 1 passed`

- [ ] **Step 5: Use the shared projection in the tick**

In `src/bin/gpu_evolve/context.rs`, inside the pass timed as `prep.racing_line`, call
`project_at` once per being and pass the result to both `_from` functions.

- [ ] **Step 6: Prove no simulated number moved**

```bash
cargo build --release
D=/media/llm-data/scratch/proj; rm -rf $D; mkdir -p $D
cp /media/llm-data/universe-runs/ereg-m4-m4-s77/evolve_best_gen0043.bin $D/
./target/release/gpu_evolve --retomar --out $D --gens 1 --seres 48 --ticks 20000 \
    --width 28 --out-beta 0.0 --mascara-regioes --neuromod-regioes \
    --pistas 8005,8200,9001 --seed 77 --observar $D/obs.f32
sha256sum $D/obs.f32
```

Expected: `5f1cf08e4764c78bec1b9341f625ce62afd7adc760becbf88884b4c81ee5fa35`, median `2939m`.
**If it differs, revert.** A projection that is computed once instead of twice must give
the same answer; a difference means the two call sites were not passing the same `pos`,
which is a finding worth reporting, not patching over.

- [ ] **Step 7: Measure and commit**

Re-run the Task 1 profiling command and report `prep.racing_line` median and range before
and after. Expected direction: roughly halved. Put both numbers and the command in the
commit message.

---

## Task 3: Binary-search the arc scan

**Files:**
- Modify: `src/simulation/racing_line.rs` (`lookahead_offset_from`)
- Test: same module

**Interfaces:**
- Consumes: `Projection` and `lookahead_offset_from` from Task 2.
- Produces: no new API. The linear `for k in 0..self.arc.len() - 1` becomes a binary search.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod arc_search_tests {
    use super::*;

    /// `self.arc` is cumulative arc length and therefore sorted ascending, so the
    /// segment containing a target can be found by binary search. The test pins the
    /// EQUIVALENCE, not the algorithm: it compares against the linear scan the code
    /// used before, at every segment boundary and at points either side of one.
    #[test]
    fn binary_search_finds_the_same_segment_as_a_linear_scan() {
        let line = RacingLine::from_points(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(100.0, 0.0),
            Vec2::new(200.0, 50.0),
            Vec2::new(300.0, 50.0),
        ]);
        let arc = &line.arc;
        assert!(arc.len() >= 3, "need a few segments to search");

        let linear = |target: f32| -> usize {
            let mut j = 0usize;
            for k in 0..arc.len() - 1 {
                if arc[k] <= target && target <= arc[k + 1] {
                    j = k;
                    break;
                }
            }
            j
        };

        let mut targets: Vec<f32> = Vec::new();
        for k in 0..arc.len() {
            targets.push(arc[k]);
            if k + 1 < arc.len() {
                targets.push((arc[k] + arc[k + 1]) * 0.5);
            }
        }
        for t in targets {
            assert_eq!(
                super::segment_for_arc(arc, t),
                linear(t),
                "segment mismatch at target {t}"
            );
        }
    }
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cargo test --release --lib arc_search 2>&1 | tail -20`
Expected: FAIL — `segment_for_arc` not found.

- [ ] **Step 3: Implement**

```rust
/// Index of the segment containing `target` in the sorted cumulative-arc table.
///
/// `arc` is cumulative arc length, so it is ascending and `partition_point` applies.
/// Replaces a linear scan that ran once per being per tick: at ~129 segments and 768
/// beings that was ~99 000 comparisons a tick for a value a binary search finds in
/// seven. Clamped to the last valid segment so a target at or past the end behaves
/// exactly like the linear scan's fallthrough, which left `j` at its last match.
pub(crate) fn segment_for_arc(arc: &[f32], target: f32) -> usize {
    if arc.len() < 2 {
        return 0;
    }
    let k = arc.partition_point(|&a| a <= target);
    k.saturating_sub(1).min(arc.len() - 2)
}
```

Then call it from `lookahead_offset_from` in place of the loop.

- [ ] **Step 4: Run it and watch it pass**

Run: `cargo test --release --lib arc_search 2>&1 | grep "test result"`
Expected: `test result: ok. 1 passed`

- [ ] **Step 5: Prove no simulated number moved**

Run the Step 6 command from Task 2. Expected: same sha256, median `2939m`.

**One difference matters and you must check it.** The old linear scan started at `j = 0`
and, if no segment matched, left `j = 0`. `segment_for_arc` clamps instead. Those differ
for a target below `arc[0]` or above the last entry. Read `lookahead_offset_from` and
decide whether such a target can occur — `rem_euclid(total_len)` should prevent it. State
your reasoning in the report either way; if it can occur, match the old behaviour exactly
rather than the "better" one, because this task is a speedup and not a bug fix.

- [ ] **Step 6: Measure and commit**

---

## Task 4: Warm-start the projection from the previous tick

**Files:**
- Modify: `src/simulation/racing_line.rs`, `src/bin/gpu_evolve/types.rs` (one field on `BeingState`), `src/bin/gpu_evolve/context.rs`
- Test: `src/simulation/racing_line.rs`

**Interfaces:**
- Consumes: `project_at` and `Projection` from Task 2.
- Produces: `pub fn project_near(&self, pos: Vec2, hint: usize, window: usize) -> Option<Projection>` — searches `hint ± window` segments and **falls back to the full scan** when the best result lands on the window's edge.

**Why the fallback is not optional.** A being moves roughly 0.3 m per tick, so its nearest
segment almost never changes — that is what makes the hint work. But a track that loops
back near itself has two segments at similar distance, and a purely local search can lock
onto the wrong one and stay there. The edge check is what makes this a speedup rather than
a silent behaviour change: if the local winner sits at the window boundary, a better
segment may lie outside, so redo the full scan.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod warm_start_tests {
    use super::*;

    /// A hinted search must return exactly what the full scan returns, for every
    /// segment index used as a hint — including deliberately wrong hints, which is
    /// where a local search silently diverges.
    #[test]
    fn a_hinted_projection_matches_the_full_scan_from_any_hint() {
        let line = RacingLine::from_points(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(100.0, 0.0),
            Vec2::new(200.0, 50.0),
            Vec2::new(300.0, 50.0),
            Vec2::new(300.0, 150.0),
            Vec2::new(200.0, 200.0),
        ]);
        let n_seg = line.points.len() - 1;
        for &(x, y) in &[(10.0, 3.0), (150.0, -8.0), (250.0, 60.0), (299.0, 120.0)] {
            let pos = Vec2::new(x, y);
            let full = line.project_at(pos).expect("full");
            for hint in 0..n_seg {
                let near = line.project_near(pos, hint, 2).expect("near");
                assert_eq!(
                    near.seg, full.seg,
                    "hint {hint} at {pos:?} gave segment {} instead of {}",
                    near.seg, full.seg
                );
                assert_eq!(near.t.to_bits(), full.t.to_bits(), "t differs for hint {hint}");
            }
        }
    }
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cargo test --release --lib warm_start 2>&1 | tail -20`
Expected: FAIL — `project_near` not found.

- [ ] **Step 3: Implement `project_near` with the edge fallback**

The window scan is the same arithmetic as `project_at`, over `hint.saturating_sub(window) ..= (hint + window).min(n_seg - 1)`, wrapping is NOT required for correctness because the fallback covers it. If the best index equals either end of the searched range and that end is not also the track's end, return `self.project_at(pos)` instead.

- [ ] **Step 4: Run it and watch it pass**

Run: `cargo test --release --lib warm_start 2>&1 | grep "test result"`
Expected: `test result: ok. 1 passed`

- [ ] **Step 5: Store the hint per being**

Add `pub last_seg: usize` to `BeingState` (`src/bin/gpu_evolve/types.rs`), default 0. In the
`prep.racing_line` pass, call `project_near(pos, states[i].last_seg, 3)` and write the
result's `seg` back. **Reset it to 0 whenever the world changes** — `context.rs` rebuilds
the world on track rotation, and a stale hint from another track is exactly the case the
fallback exists to survive, but resetting is cheaper than relying on it.

- [ ] **Step 6: Prove no simulated number moved**

Run the Step 6 command from Task 2. Expected: same sha256, median `2939m`. **This is the
task where a silent divergence is most likely** — if the hash differs, the fallback is not
firing when it should. Report the divergence rather than widening the window until it
passes.

- [ ] **Step 7: Measure and commit**

Report `prep.racing_line` and the whole-tick wall before and after, median with range.

---

## Task 5: Re-measure, then decide about the 13 ms wait

**Files:**
- Modify: `docs/KERNEL_TIMING.md`
- Modify: `src/bin/gpu_evolve/context.rs`, only if Step 2 says it is still worth it.

- [ ] **Step 1: Re-measure the whole tick after Tasks 2-4**

Run the Task 1 profiling command. Report every phase, median over ≥25 samples with range,
plus `rocm-smi --showgpuclocks --showpower` sampled ten times during the run.

- [ ] **Step 2: Decide from the number and write it down**

The card sits at DPM level 1 (1205-1995 MHz, 21-24 W of a ~2600 MHz / ~200 W ceiling)
**because it idles most of every tick**. Removing CPU work raises the duty cycle, which may
raise the clock, which shortens both the kernels and the wait — for free.

- If `cpu.poll` fell substantially and the clock rose, **stop here.** Record it in
  `docs/KERNEL_TIMING.md` and do not restructure the synchronisation. A task that ends in a
  recorded decision and no diff is complete, not skipped.
- If `cpu.poll` is still dominant, continue.

- [ ] **Step 3: Only then, pipeline the submit**

Stop blocking every tick: submit tick N+1's commands before waiting on N's results. This is
the only task here that is not a pure refactor — it changes when the CPU observes GPU state.
Anything reading back per tick (`--observar`, `--gravar-onda`, `--perfil-kernel`,
`--auditar-visao`) forces the old path. Gate it behind a flag, off by default, and make
those flags force the synchronous path with a printed line saying so.

- [ ] **Step 4: The gate is unchanged and stricter here**

Expected: same sha256 and median with the flag both off and on. A pipelined tick that
changes the hash is reading stale state — revert.

- [ ] **Step 5: Commit with the before/after table**

---

## Not in this plan

- **Porting the prep to WGSL.** Tasks 2-4 are mechanical and reversible; a kernel port changes where the sensory inputs are computed and needs its own spec and a paired A/B under `docs/CLAUDE.md` rule 3. Revisit only if they leave `rec.cpu_prep` dominant.
- **The six per-tick `Vec` allocations, and fusing the passes.** These were the original Tasks 2-3. Task 1's measurement demoted them: together the passes they target are 47.5% of `rec.cpu_prep`, and allocation is a small part of even that, against one function at 52%. Revisit after Tasks 2-4 if the remainder is worth a commit.
- **The `comp_stress` O(n²) block.** Measured and refuted as the cause: `rec.cpu_prep` scales 8.2× for 8× beings against 64× if quadratic, and the block sits behind `--interocepcao`. It becomes relevant only when that flag is on at large populations, and then it is its own task.
