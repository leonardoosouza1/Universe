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

## Task 2: Kill the twelve per-tick allocations

**Files:**
- Create: `src/bin/gpu_evolve/tick_buffers.rs`
- Modify: `src/bin/gpu_evolve/context.rs` (prep region), `src/bin/gpu_evolve/mod.rs` or wherever the binary's modules are declared

**Interfaces:**
- Consumes: Task 1's `prep.*` measurement, to say which allocations mattered.
- Produces:
  - `pub struct TickScratch { pub dists: Vec<[f32; 3]>, pub rv: Vec<[f32; 6]>, pub prop: Vec<[f32; 12]>, pub extra: Vec<[f32; 17]>, pub extra2: Vec<[f32; 9]>, pub body: Vec<[f32; universe::perception::channels::BODY_K]> }`
  - `pub fn new() -> TickScratch`
  - `pub fn reset_for(&mut self, n: usize)` — clears each vector's length to 0 while keeping its capacity, growing capacity to `n` once.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::TickScratch;

    /// The whole point is that capacity survives a reset: a `Vec` that reallocates
    /// every tick is 20 000 allocations per generation per buffer, which is what this
    /// type exists to remove. A reset that shrinks capacity would silently keep the
    /// allocation cost while looking like a fix.
    #[test]
    fn reset_keeps_capacity_and_clears_length() {
        let mut s = TickScratch::new();
        s.reset_for(768);
        s.dists.push([1.0, 2.0, 3.0]);
        s.extra2.push([0.0; 9]);
        let cap_dists = s.dists.capacity();
        let cap_extra2 = s.extra2.capacity();
        assert!(cap_dists >= 768, "reset_for must reserve up front, got {cap_dists}");

        s.reset_for(768);
        assert_eq!(s.dists.len(), 0, "length must be cleared");
        assert_eq!(s.extra2.len(), 0, "length must be cleared");
        assert_eq!(s.dists.capacity(), cap_dists, "capacity must survive the reset");
        assert_eq!(s.extra2.capacity(), cap_extra2, "capacity must survive the reset");
    }

    /// Growing the population must not silently keep a too-small buffer.
    #[test]
    fn reset_grows_for_a_larger_population() {
        let mut s = TickScratch::new();
        s.reset_for(48);
        s.reset_for(768);
        assert!(s.prop.capacity() >= 768, "capacity must grow with n");
    }
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cargo test --release --bin gpu_evolve tick_buffers 2>&1 | tail -20`
Expected: FAIL — unresolved module `tick_buffers`.

- [ ] **Step 3: Write the module**

```rust
//! Per-tick scratch buffers, allocated once instead of twelve times per tick.
//!
//! Measured 2026-08-27 (`--perfil-kernel`, 768 beings): `rec.cpu_prep` is 7.58 ms of a
//! 24.10 ms tick, and the region allocated twelve `Vec`s every tick — 20 000 ticks per
//! generation, so 240 000 allocations per buffer per generation. Capacity is reserved
//! once and reused; `reset_for` clears length without touching it.

/// Scratch space reused across ticks. One field per per-being pass in `context.rs`.
#[derive(Default)]
pub struct TickScratch {
    pub dists: Vec<[f32; 3]>,
    pub rv: Vec<[f32; 6]>,
    pub prop: Vec<[f32; 12]>,
    pub extra: Vec<[f32; 17]>,
    pub extra2: Vec<[f32; 9]>,
    pub body: Vec<[f32; universe::perception::channels::BODY_K]>,
}

impl TickScratch {
    pub fn new() -> TickScratch {
        TickScratch::default()
    }

    /// Clears every buffer's length and guarantees capacity for `n` beings.
    ///
    /// `clear()` keeps capacity by contract; `reserve` on an empty vector then only
    /// allocates when `n` grew. Do not replace this with `Vec::with_capacity` — that is
    /// the allocation this type was written to remove.
    pub fn reset_for(&mut self, n: usize) {
        macro_rules! reset {
            ($($f:ident),+ $(,)?) => { $( self.$f.clear(); self.$f.reserve(n); )+ };
        }
        reset!(dists, rv, prop, extra, extra2, body);
    }
}
```

- [ ] **Step 4: Run it and watch it pass**

Run: `cargo test --release --bin gpu_evolve tick_buffers 2>&1 | grep "test result"`
Expected: `test result: ok. 2 passed`

- [ ] **Step 5: Use it in the tick**

Add `scratch: TickScratch` to the context struct, call `self.scratch.reset_for(n)` once at the top of `tick()`, and replace each `let mut x: Vec<..> = Vec::with_capacity(n);` in the prep region with a borrow of the matching field. **Change nothing else in those passes** — same order, same arithmetic, same pushes.

- [ ] **Step 6: Prove it changed no number**

```bash
cargo build --release --bin gpu_evolve
D=/media/llm-data/scratch/scr; rm -rf $D; mkdir -p $D
cp /media/llm-data/universe-runs/ereg-m4-m4-s77/evolve_best_gen0043.bin $D/
./target/release/gpu_evolve --retomar --out $D --gens 1 --seres 48 --ticks 20000 \
    --width 28 --out-beta 0.0 --mascara-regioes --neuromod-regioes \
    --pistas 8005,8200,9001 --seed 77 --observar $D/obs.f32
sha256sum $D/obs.f32
```

Expected: `5f1cf08e4764c78bec1b9341f625ce62afd7adc760becbf88884b4c81ee5fa35`, median `2939m`. **If it differs, revert — a buffer refactor that changes the simulation is a bug, not a trade-off.**

- [ ] **Step 7: Measure the effect, median over ≥25 samples**

Re-run the Task 1 Step 3 command and compare `rec.cpu_prep` before and after. Report median and range for both. If the improvement is under 5%, say so plainly — allocation was then not the cost, and Task 3 carries the weight.

- [ ] **Step 8: Commit**

```bash
cargo fmt -p universe
git add -A
git commit -m "perf(gpu_evolve): reuse per-tick scratch instead of twelve allocations

[before/after rec.cpu_prep medians with ranges and the command]

obs.f32 byte-identical: 5f1cf08e...

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 3: Fuse the per-being passes

**Files:**
- Modify: `src/bin/gpu_evolve/context.rs` (prep region)

**Interfaces:**
- Consumes: `TickScratch` from Task 2; Task 1's `prep.*` table, which says which passes are worth fusing.
- Produces: no new public API. One pass over `states[..n]` filling several scratch buffers, where there were several passes.

- [ ] **Step 1: Pick the fusion set from Task 1's numbers**

Fuse only passes that (a) read the same `states[i]` fields and (b) together account for a meaningful share of `rec.cpu_prep`. **Do not fuse everything on principle** — a fused pass that reads fields no other pass needs gains nothing and costs readability. Record in the report which passes you fused and which you left alone, with the measured share of each.

- [ ] **Step 2: Fuse them, one at a time**

Each fusion is one commit. The arithmetic inside must be copied verbatim, not rewritten — this is a loop-structure change, not a numerical one. The shape, using two real passes from the file as the worked example:

Before — two passes, each walking `states[..n]` and touching the same `phys`:

```rust
for i in 0..n {
    let p = &states[i].phys;
    let dist_l = /* ... existing expression, unchanged ... */;
    let dist_r = /* ... */;
    let dist_f = /* ... */;
    scratch.dists.push([dist_l, dist_r, dist_f]);
}
// ... unrelated code between them stays where it is ...
for i in 0..n {
    let p = &states[i].phys;
    let speed = (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt();
    scratch.rv.push([/* ... six existing expressions, unchanged ... */]);
}
```

After — one pass, both buffers filled, every expression byte-for-byte the same:

```rust
for i in 0..n {
    let p = &states[i].phys;

    let dist_l = /* ... same expression, moved not rewritten ... */;
    let dist_r = /* ... */;
    let dist_f = /* ... */;
    scratch.dists.push([dist_l, dist_r, dist_f]);

    let speed = (p.vel[0].powi(2) + p.vel[2].powi(2)).sqrt();
    scratch.rv.push([/* ... same six expressions ... */]);
}
```

Two rules that make the difference between a fusion and a bug:

* **Only fuse passes with nothing order-dependent between them.** If code between the two passes writes to `states`, or reads a buffer the first pass fills, they are not independent and fusing reorders the program. Check before moving.
* **Never merge two accumulations into one expression.** `a + b + c` computed in a different order is a different `f32`. Keep each push exactly as it was; the loop header is the only thing that changes.

- [ ] **Step 3: Prove each fusion changed no number**

After EACH fusion, run the Step 6 command from Task 2. Expected: sha256 `5f1cf08e4764c78bec1b9341f625ce62afd7adc760becbf88884b4c81ee5fa35`, median `2939m`. A fusion that changes the hash reordered a floating-point accumulation; revert it and say which.

- [ ] **Step 4: Measure and commit each fusion**

Report `rec.cpu_prep` median and range before and after each one, with the command.

---

## Task 4: The 13.28 ms wait — measure before touching

**Files:**
- Modify: `src/bin/gpu_evolve/context.rs` (the submit/poll boundary), only if Step 2 says it is still worth it.

**Interfaces:**
- Consumes: the tick after Tasks 2-3.
- Produces: a decision, recorded in `docs/KERNEL_TIMING.md`, and the change only if the measurement justifies it.

- [ ] **Step 1: Re-measure the whole tick after Tasks 2-3**

Run the Task 1 Step 3 command. Report every phase, median over ≥25 samples with range, plus `rocm-smi --showgpuclocks --showpower` sampled ten times during the run.

- [ ] **Step 2: Decide from the number, and write the decision down**

The card sits at DPM level 1 (1205-1995 MHz, 21-24 W of a ~2600 MHz / ~200 W ceiling) **because it idles most of every tick**. Removing CPU work raises the duty cycle, which may raise the clock, which shortens both the kernels and the wait — for free.

- If `cpu.poll` fell substantially and the clock rose, **stop here.** Record it in `docs/KERNEL_TIMING.md` and do not restructure the synchronisation. That is the cheapest possible outcome and it must be allowed to happen.
- If `cpu.poll` is still the dominant term, continue to Step 3.

- [ ] **Step 3: Only then, pipeline the submit**

The change is to stop blocking on every tick: submit tick N+1's commands before waiting on N's results. This is the one task here that is NOT a pure refactor — it changes when the CPU observes GPU state, and anything reading back per tick (`--observar`, `--gravar-onda`, `--perfil-kernel`, `--auditar-visao`) forces the old path. Gate it behind a flag, off by default, and make those flags force the synchronous path with a printed line saying so.

- [ ] **Step 4: The gate is unchanged, and it is stricter here**

Expected: sha256 `5f1cf08e4764c78bec1b9341f625ce62afd7adc760becbf88884b4c81ee5fa35`, median `2939m`, with the flag both off and on. A pipelined tick that changes the hash is reading stale state — revert.

- [ ] **Step 5: Commit with the before/after table**

---

## Not in this plan

- **Porting the prep to WGSL.** Tasks 2-3 are mechanical and reversible; a kernel port changes where the sensory inputs are computed and needs its own spec and a paired A/B under `docs/CLAUDE.md` rule 3. Revisit only if Tasks 2-3 leave `rec.cpu_prep` dominant.
- **The `comp_stress` O(n²) block.** Measured and refuted as the cause: `rec.cpu_prep` scales 8.2× for 8× beings against 64× if quadratic, and the block sits behind `--interocepcao`. It becomes relevant only when that flag is on at large populations, and then it is its own task.
