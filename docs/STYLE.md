# Universe — Code Style

The rules a change has to satisfy to be merged. They are deliberately few, and every
one of them is either machine-checkable or has a worked example from this repository.

Two audiences read this project: humans who have never seen a neuroevolution
simulator, and coding agents that will be asked to change one file without reading
the other 157. Both need the same thing — a file that explains itself, is small
enough to hold in one read, and does not silently contradict another file.

| Rule | Enforced by |
|---|---|
| English everywhere | `scripts/check_language.sh` (CI) |
| File size caps | review; see the queue below |
| `//!` module header, `///` on public items | review |
| One source of truth per constant | review |
| Comments record measurement and cause | review |
| Bug fix ships with a regression test | `cargo test --release` |
| No `unwrap()` on expected failure | review |

---

## 1. English, everywhere

**Effective 2026-08-17, this project is English-only.** The previous rule — identifiers
in English, comments and output strings in Portuguese — is revoked. See `NAMING.md`.

English applies to:

- identifiers: modules, types, traits, functions, fields, locals, constants, WGSL
  bindings and shader entry points
- file and directory names
- comments and doc comments (`//`, `///`, `//!`)
- log, `println!`, `eprintln!`, `panic!`, `assert!` and error messages
- test function names
- commit messages and PR descriptions
- CLI flags and their help text

Two things stay as they are:

- **Recorded experiment output** under `docs/` (battery reports, verdicts, session
  notes). These are dated records of what was measured; rewriting them would edit
  history. They are out of the checker's scope.
- **`CLAUDE.md`**, the owner's operating instructions for agents.

### Translating without losing the content

The comments in this repository are not decoration. They carry bug post-mortems that
cost real GPU hours: measured strides, seeds that reproduce a failure, the size of an
effect. **Translate the prose; never summarise, shorten, or drop a number.**

A translation is wrong if it loses any of: a numeric measurement, a `file:line`
reference, a command that reproduces something, a date, a seed, or the name of the
mechanism that caused the bug.

### How it is checked

```bash
./scripts/check_language.sh          # what your branch changed — this is the CI gate
./scripts/check_language.sh --all    # whole-repo progress report, never fails
```

The checker looks for Portuguese diacritics and for a short list of function words
that cannot be English (`nao`, `para`, `que`, `com`, `quando`, `porque`, …). Words <!-- check-language: allow -->
shared by both languages are deliberately excluded so it does not fire on ordinary
English.

When English text has to *quote* Portuguese — this paragraph, the legacy symbol names
below, the tables in `docs/MIGRATION.md` — mark the line and the checker skips it:

```
check-language: allow    one line
check-language: off      from here
check-language: on       resume
```
<!-- check-language: on -->
<!-- ^ the fence above contains literal markers, so it opens and closes a region as a
     side effect of documenting them. This line guarantees the region is closed no
     matter what is edited inside the fence. -->

In Markdown, write it as `<!-- check-language: allow -->`; it renders as nothing. Use
it to quote, never to postpone — a source file that needs it belongs on the allowlist.

Because 54k lines cannot be translated in one commit, `scripts/language_allowlist.txt`
holds the files that are still Portuguese. The rule is asymmetric on purpose:

- a file **on** the allowlist may keep its Portuguese, but every line your branch
  **adds** to it must be English;
- a file **not** on the allowlist must be fully English, every line.

Finish a file, delete its line from the allowlist and tick it off in
`docs/MIGRATION.md`, in the same commit. The list only ever shrinks.

---

## 2. File size: 600 lines for `.rs`, 400 for `.wgsl`

These caps are not imported from a style blog. They are where this project already
is, measured on 2026-08-17 over the tracked tree:

| | files | total lines | median | over cap |
|---|---|---|---|---|
| `.rs` | 158 | 60,254 | 236 | **27 (17%)** over 600 |
| `.wgsl` | 22 | 7,761 | ~270 | **4 (18%)** over 400 |

So 83% of the Rust files and 82% of the shaders already comply. The cap ratifies the
project's own normal practice and names the outliers, rather than imposing a foreign
number. The median Rust file is 236 lines; the largest, `src/simulation/motor.rs`, is
**3,382 — fourteen times the median.**

Why 600 and not "whatever fits":

- A reviewer or an agent can read 600 lines in one pass and hold the whole thing.
  At 3,382 nobody does, so changes get made blind — which is how `brain.wgsl` ended
  up reading the sensory wave with the wrong stride (see §4).
- Merge conflicts scale with file size, and this project has 11 branches.

**Code you write now is born under the cap.** A new module that would exceed it gets
split before the first commit, not after.

### The split queue

These files exceed the cap today. They are the standing backlog — picking one up is a
welcome contribution, and each should be its own PR.

| File | Lines | | File | Lines |
|---|---:|---|---|---:|
| `src/simulation/motor.rs` | 3382 | | `src/evolution/dna_genesis.rs` | 1063 |
| `src/bin/gpu_evolve/main.rs` | 2582 | | `src/evolution/spawn.rs` | 991 |
| `src/bin/gpu_report.rs` | 2139 | | `src/gpu/physics.rs` | 908 |
| `src/bin/cortex_evolve.rs` | 1682 | | `src/gpu/cortex_sheet.rs` | 828 |
| `src/gpu/evolution.rs` | 1561 | | `src/bin/fisica_bisect.rs` | 811 |
| `src/evolution/pipeline.rs` | 1535 | | `src/physics.rs` | 769 |
| `src/bin/gpu_validate.rs` | 1525 | | `src/evolution/dna.rs` | 705 |
| `src/world.rs` | 1519 | | `src/neuro.rs` | 680 |
| `src/bin/gpu_loop.rs` | 1494 | | `src/gpu/core.rs` | 671 |
| `src/bin/gpu_evolve/context.rs` | 1457 | | `src/bin/cortex_map.rs` | 669 |
| `src/gpu/perception.rs` | 1421 | | `src/bin/replay_mapa.rs` | 650 |
| `src/perception/channels.rs` | 1151 | | `src/engine.rs` | 609 |
| `src/gpu/brain.rs` | 1139 | | `src/replay.rs` | 604 |

Shaders over 400: `src/gpu/brain.wgsl` (1028), `src/gpu/perception.wgsl` (996),
`src/gpu/physics.wgsl` (756), `src/gpu/perception_temporal.wgsl` (470).

`docs/arquivo/dirs/iara_genesis/src/universal.rs` (637) is also over, but it belongs
to the archived `iara_genesis` crate that `Cargo.toml` still lists as a workspace
member. Where that crate should live is a separate question — TODO, not decided here.

### How to split

The project already contains the pattern to copy: `src/bin/gpu_evolve/` is a binary
split into `main.rs`, `context.rs`, `types.rs`, `statics.rs`, `tests.rs`, and one
module per concern (`escada.rs`, `fantasma.rs`, `pistas.rs`, `piloto_linear.rs`, …).

Split along the seam a reader already perceives — one concern per file, with its own
`//!` header and its own `#[cfg(test)] mod tests`. Do not split by line count into
`motor_part1.rs` / `motor_part2.rs`; that satisfies the number and defeats the reason
for it. A split PR should move code and change nothing else, so the diff is reviewable.

---

## 3. Module and item documentation

**Every file opens with a `//!` header** that answers two questions: what this module
does, and why it exists. "Why it exists" is the one a newcomer cannot reconstruct.

```rust
//! Canonical registry of the 192 sensory-wave channels.
//!
//! Every channel has exactly one writer. This file is the only place that decides
//! which index means what; the perception shaders index into the wave and would
//! otherwise each carry their own private idea of the layout.
```

**Every public item gets a `///` that states its contract**, not its implementation:
what the caller must guarantee, what comes back, what makes it fail or panic, and the
unit of any number involved. The body already shows how; the doc comment exists to
save the reader from having to read the body.

```rust
/// Fuel tank sized for one refuelling interval (one sector), in tank units.
///
/// `lap_dist` is the full lap length in metres. `k = 1.0` means "exactly enough to
/// reach the next sector at 18.75 m/s at full throttle" (consumption 0.006/s);
/// anything slower than that runs dry.
fn tank_per_sector(lap_dist: f64, k: f32) -> f32 { … }
```

Units belong in the doc or in the name (`dist_m`, `lap_ms`). A bare `f32` with no unit
anywhere is how a lap-length ends up divided by a sector count.

---

## 4. One source of truth per constant

**If the same constant or formula is written in two files, it moves into one shared
place and the other file reads it from there.**

This is not a hygiene preference here; the project has already paid for it twice, in
the same week, from the same duplication.

`SENSORY_WAVE_SIZE` is defined once, in `src/config.rs:309`. The packed word count
derived from it, `WAVE_U32S = ceil(192/4) = 48`, is hand-declared **six times**:

```
src/gpu/brain.wgsl               src/gpu/perception.wgsl
src/gpu/perception_motion.wgsl   src/gpu/perception_temporal.wgsl
src/gpu/retina_digest.wgsl       src/gpu/slow_gather.wgsl
```

When the wave shrank from 223 channels to 192, two of those six copies were missed.
Both bugs are recorded in the code. `src/gpu/brain.wgsl:110`:

> was hardcoded 56u here (from when the wave had 223 channels); when it was reduced
> to 192 I missed this site, and the brain started reading perception with stride 56
> while it is written with stride 48 — **every being was reading its neighbour's
> data.** Bug measured and fixed this session.

And `src/gpu/slow_gather.wgsl:27`, independently:

> was 56u (223-channel wave) — the slow brain read perception with the wrong stride,
> scrambled input. Fixed.

The same duplication also rots the comments around it: `src/gpu/brain.wgsl:106` still
says `SENSORY_WAVE_SIZE = 223` two lines above `const WAVE: u32 = 192u`, and
`src/gpu/core.rs:66` documents a `223→96` BitLinear that is really `192→96`. A comment
maintained next to a copied constant will drift; only the original stays true.

The practical rule for this codebase, where WGSL cannot `use` a Rust constant:

- The Rust constant in `src/config.rs` is the **only** definition.
- A shader that needs the value gets it injected at pipeline-build time, or — if it
  must be a literal — is accompanied by an assertion on the Rust side that the two
  agree. `src/gpu/perception.rs:1410` already does this:
  `assert_eq!(WAVE_U32S, 48, "48 × 4 = 192 bytes = 192 channels, no slack")`.
- Never a second copy with a comment claiming it matches.

The same applies to `GRID_SIZE`, currently declared in `src/gpu/core.rs:63` with the
comment "read from world.rs: `let size = 320`" — a copied literal with a note where an
import belongs.

---

## 5. Comments that earn their place

This project values one specific kind of comment: the one that records **a measurement
and a cause** — why the code is shaped this way, with the number and the command that
produced it. Those comments are the reason a five-hour bug does not get reintroduced.

Comments that restate the code are noise and should be deleted rather than translated.

### Good — real, from `src/bin/gpu_evolve/main.rs:145` (shown translated)

```rust
/// Tank sized by the CORRECT denominator: the refuelling interval (one sector),
/// not the whole lap.
///
/// Two bugs were paid for here on 2026-08-16, and neither showed up in the log:
/// 1. the formula used the LAP (`L/18.75 × 0.003 × k`) while refuelling happens
///    every sector (24 per lap) — a tank ~12× larger than the interval it is
///    refilled over, so it NEVER constrained anything;
/// 2. the formula only ran on a TRACK CHANGE, so in a single-track race (and in
///    the first generation of any battery) `--fuel-auto k` did nothing: k=2.0,
///    1.0 and 0.7 gave IDENTICAL results (1923 m, 0% mortality) — the
///    identical-numbers canary caught it.
/// Together they explain why the difficulty-ladder battery had 34 of 35
/// generations at 0% mortality with fuel, noise, damage and weather all enabled.
```

It names the wrong denominator, quantifies the error (~12×), gives the identical
numbers that exposed it, and states the downstream consequence. A future contributor
who changes the refuelling interval knows immediately what else must move.

### Bad — invented, for contrast

```rust
// Calculate the tank.
// TODO: check this later
// NOTE: important!
fn tank_per_sector(lap_dist: f64, k: f32) -> f32 {
    let sector_m = lap_dist as f32 / N_CHECKPOINTS as f32; // divide by checkpoints
    (sector_m / 18.75) * 0.006 * k                         // multiply by k
}
```

Every line restates what the code says. `18.75` and `0.006` — the two numbers a reader
actually needs explained — are the only things left unexplained. `TODO: check this
later` names no condition under which it is checked, so it will never be checked.

**Test:** if the comment would still be true after the function is rewritten to do
something else, it is describing syntax, not the reason. Delete it.

---

## 6. Every bug fix ships with a regression test

A fix without a test is a fix that gets undone. The test must fail against the old
code — write it, watch it fail, then fix. Name it after the behaviour it protects, in
English, as a sentence: `genome_survives_disk_round_trip`, not `test_genome_3`.

Two real examples in this repository, both guarding bugs that shipped silently:

**`genoma_sobrevive_ida_e_volta_no_disco`** — `src/bin/gpu_evolve/tests.rs:24`.
A `slow` field (1,091 weights) was added to `Genome` without updating `save_genome`;
every champion silently lost its slow brain on reload. The test compares the two sides
of the disk field by field, and — the part that makes it work — marks each block with
a value that cannot come from the default initialiser, so "passed" cannot mean "both
sides fell back to the same default". It fails automatically the next time someone adds
a field and forgets the serialiser, which is the error that actually recurs.

**`nao_sobe_com_apenas_2_boas`** — `src/bin/gpu_evolve/escada.rs:406`. <!-- check-language: allow -->
The difficulty ladder was promoting on evidence from the best individual and wiping the
population. The fix added a promotion threshold and hysteresis; this test pins the
threshold by asserting that two consecutive good generations are *not* enough
(`assert_eq!(e.ativo, 0, "2 approvals are not enough, the threshold is 3")`), with
siblings covering demotion after five bad ones and the post-demotion hysteresis.

Both names are on the migration list and will be translated with their files.

Run before pushing:

```bash
cargo test --release --bin gpu_evolve     # 39 tests, no GPU required
cargo test --release                      # full suite
```

GPU-dependent tests are marked `#[ignore]` and run explicitly. **Do not add mocked GPU
tests** — a GPU test in this project talks to a GPU, or it is not a GPU test.

---

## 7. Errors

**No `unwrap()` on a path where failure is expected** — missing file, absent GPU
adapter, malformed genome, CLI argument. Those return `Result` and report. `unwrap()`
and `expect()` are for invariants that a bug, not an environment, could break.

There are 258 `.unwrap()` and 77 `.expect(` calls under `src/` today. They are not all
wrong, and this rule is not a mandate to sweep them. It applies to code you touch.

**An error message says what to do, not only what failed.** The reader is at a terminal
with a broken run and no context.

```rust
// Bad — real, src/bin/cortex_map.rs:64 (shown translated)
.expect("No Vulkan adapter found");

// Good
.ok_or_else(|| anyhow!(
    "no Vulkan adapter found. On the RX 6750 XT this usually means the gfx \
     override is missing: export HSA_OVERRIDE_GFX_VERSION=10.3.0 and retry. \
     Check `vulkaninfo --summary` for a visible device."
))?;
```

The first message tells a user with a working GPU that they do not have one. The second
names the most likely cause and the command that confirms it.

Panic messages follow the same rule: `assert_eq!(a, b, "…")` with the invariant spelled
out beats a bare comparison, because the value alone rarely says which side is wrong.

---

## 8. Before opening a PR

```bash
cargo fmt --all                                  # rustfmt defaults
cargo build --release 2>&1 | grep -E "^error"    # must print nothing
cargo clippy --release --all-targets             # no new warnings
cargo test --release --bin gpu_evolve            # 39 tests
./scripts/check_language.sh                      # English gate
```

Commit messages in English, imperative, subject under 72 characters. If the commit
fixes a bug, the body carries the measurement — that is where this project's history
actually lives.

There is no `rustfmt.toml` or `clippy.toml` in the repository; both tools run on
defaults. TODO: decide whether to pin them.

---

## Open items

- No `rustfmt.toml` / `clippy.toml` — defaults are in force by omission, not decision.
- The `iara_genesis` crate is a workspace member living under `docs/arquivo/`
  (`Cargo.toml:7`); its location has not been decided.
- The cap has no automated check. Adding one to CI is a good first contribution.

*Written 2026-08-17. Measurements in this document were taken against the tracked tree
on that date; re-measure with `./scripts/check_language.sh --all` and
`git ls-files '*.rs' | xargs wc -l | sort -rn` before quoting them.*
