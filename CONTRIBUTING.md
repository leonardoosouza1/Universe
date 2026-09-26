# Contributing to Universe

Current behavioral work must follow
[VALIDATION-WORKFLOW.md](docs/VALIDATION-WORKFLOW.md): inspect the executed
sensor→brain→action→body→task path, reproduce the failure on the real GPU, and
validate the actual replay before claiming success. Use its evidence template.
The owner's current GPU-only restriction overrides historical test commands.

Universe is a neuroevolution simulator written in Rust. Beings with a layered brain
(perception → KAN → CfC → BitLinear L1 → FloatLinear L2 → motor) drive procedural tracks and
are selected by a genetic algorithm. The simulation tick runs on the GPU through `wgpu` and
WGSL compute shaders.

This document is the contract for outside contributions: what language to write in, how to
build and test, and the two project-specific rules that a newcomer cannot guess — how
behaviour changes must be gated, and how experimental numbers must be reported.

Start with [`README.md`](README.md), then [`ARCHITECTURE.md`](ARCHITECTURE.md) for the module
map and [`docs/STYLE.md`](docs/STYLE.md) for the code standards. `LEIA_PRIMEIRO.md` and
`ARQUITETURA.md` are the retired Portuguese predecessors, kept for their measurements while
the migration runs (see [Language policy](#language-policy)).

---

## Before you start: the repository does not build from a fresh clone yet

**Known blocker. Fix this before opening a PR, and do not be surprised by it.**

`Cargo.toml` declares two dependencies by relative path to a sibling directory that is *not*
part of this repository:

```toml
iara-perception = { path = "../iara-runtime/iara-perception" }
iara-cfc        = { path = "../iara-runtime/iara-cfc" }
```

`../iara-runtime` is not tracked here and is not a git submodule, so a clone of this
repository cannot even resolve its own manifest:

```console
$ git archive HEAD | tar -x -C /tmp/clonetest && cd /tmp/clonetest
$ cargo metadata --format-version 1 --no-deps
error: failed to load manifest for workspace member `/tmp/clonetest/.`

Caused by:
  failed to load manifest for dependency `iara-cfc`

Caused by:
  failed to read `/tmp/iara-runtime/iara-cfc/Cargo.toml`

Caused by:
  No such file or directory (os error 2)
```

Both crates are genuinely used — `src/math/cfc_perception.rs` re-exports `iara_cfc::CfcCell`
and `src/perception/retina.rs` builds on `iara_perception::PerceptionPipeline` — so they
cannot simply be dropped.

Until the maintainer publishes or vendors `iara-runtime`, you need it checked out as a
**sibling of this repository**:

```
some-parent/
├── Universe/       ← this repository
└── iara-runtime/   ← the crates above
```

CI has the same problem and does not hide it: the `deps` job fails with an explicit message
rather than reporting a green build it never performed. See
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

> **TODO (maintainer):** publish `iara-runtime` and set the `IARA_RUNTIME_REPO` repository
> variable, add it as a git submodule, or vendor the two crates into this tree. Any of the
> three unblocks CI and outside contributors in one commit.

---

## Language policy

**Everything in this repository is written in English.** No exceptions:

| Where | Rule |
|---|---|
| Identifiers (types, functions, fields, constants, modules, files) | English |
| Comments and doc-comments (`//`, `///`, `//!`) | English |
| Log, `println!`, `eprintln!` and panic/assert messages | English |
| Test names | English |
| New CLI flags | English |
| Commit messages and PR descriptions | English |
| Markdown under `docs/` | English |

This **replaces** the earlier rule in [`NAMING.md`](NAMING.md), which asked for English
identifiers but told agents to keep comments and output strings in Portuguese. That rule was
revoked on **2026-08-17**.

### The migration is a ratchet, not a big bang

The tree is mid-migration. Measured on 2026-08-17:

```console
$ grep -rlE '(//|///|//!).*\b(não|também|porque|quando|pra|para|com|que|isso|cérebro|pista|geração|mundo|é)\b' \
    --include=*.rs src/ tests/ benches/ | wc -l
109
$ find src tests benches -name '*.rs' | wc -l
127
```

109 of 127 Rust files still carry Portuguese comments. You are not expected to fix all of
them. You *are* expected to leave every file you touch better than you found it:

- **Anything you add or rewrite must be English.**
- **Translate the Portuguese comments in the hunks you touch.** Not the whole file — the
  lines your diff already moves.

### Translating without destroying the value

The comments in this codebase are not decoration. They record real bugs paid for with GPU
hours and real measurements: *"the tank was sized per lap but refilled per sector"*,
*"76 of 96 neurons have period > 1"*. When you translate one:

- **Keep every number, unit and file:line reference exactly as it is.** A translated comment
  that lost its measurement is worse than the Portuguese original.
- **Translate, do not summarise.** If the original explains why a fix exists across eight
  lines, the English version is eight lines.
- **Do not delete a comment because it looks verbose.** If you believe a comment is *wrong*,
  that is a separate finding: say so in the PR with the `file:line` evidence.

### The language checker

`scripts/check_language.sh` is the automated gate. Run it before pushing:

```bash
bash scripts/check_language.sh          # gate: only what your branch changed
bash scripts/check_language.sh --all    # progress meter for the whole repo, never fails
```

It needs no Rust toolchain and no GPU, so it is the one gate that works on a plain clone
even while the dependency blocker above is open — which is why it is the first CI job.

The default mode is the gate: it diffs your branch against a base ref (`BASE_REF`, falling
back to `origin/main`) and fails with exit code 1 if Portuguese appears where English is
required. The rule it enforces is monotonic: **a file that has been migrated may never
regress, and a file still in the queue may not grow new Portuguese.** Known-legacy hits live
in `scripts/language_allowlist.txt`, which is what keeps the queue from failing every build.

If it reports a false positive — a proper noun, a track name, a quoted historical log line —
say so in the PR and fix the checker or the allowlist entry. Do not silence it file by file.

### What is *not* being renamed right now

Existing CLI flags are still Portuguese (`--seres`, `--gens`, `--clima`, `--custo-rpm`).
32 files under `scripts/` pass `--gens` alone (`grep -rl -- "--gens" scripts/ | wc -l`, run
2026-08-17), so renaming them is its own change with its own compatibility story, not
something to slip into an unrelated PR. **New** flags are English from birth.

---

## Building and testing

Toolchain: Rust stable. `Cargo.toml` declares `rust-version = "1.85.0"` and `edition = "2024"`.

```bash
# Build everything (binaries included).
cargo build --release

# Full suite.
cargo test --release
```

### When you touch the genome, run the genome suite

`Genome` is serialised by hand in `src/bin/gpu_evolve/genome_io.rs` (`save_genome` /
`load_genome`). Adding a field to the struct without adding it to *both* functions makes a
trained champion silently lose that field on every `--carregar` — it comes back at the
default value, identical across the whole population, and **nothing fails and nothing warns**.
This has already happened, twice (the slow brain's 1091 weights and the 96-entry τ bank).

The regression tests that catch it live in `src/bin/gpu_evolve/tests.rs`:

- `genoma_sobrevive_ida_e_volta_no_disco` — writes a `Genome` with sentinel values in every
  block, reads it back, compares field by field. The sentinels exist so that "passed" cannot
  mean "both sides fell into the same default".
- `genoma_v2_com_cabecalho_sobrevive_ida_e_volta` — same, for the v2 header format.

Run them with:

```bash
cargo test --release --bin gpu_evolve
```

That binary carries 44 `#[test]` attributes across its modules
(`grep -rh "#\[test\]" src/bin/gpu_evolve/ | wc -l`, measured 2026-08-17), covering the
genome round-trip, track parsing, the difficulty ladder, the ghost and the linear pilot.
**Adding a field to `Genome` and not running this command is a rejected PR.**

### GPU tests

Building and the default test run do **not** need a GPU. Exactly one test opens a real
device, and it is `#[ignore]`d for that reason:

```rust
// src/gpu/retina_digest_parity_test.rs:143
#[ignore = "abre GPU real — rodar manualmente: cargo test --release --lib -- --ignored retina_digest_parity"]
fn retina_digest_bate_com_cpu_ao_longo_de_varios_ticks() {
```

`cargo test` skips `#[ignore]`d tests by default, so CI never touches a GPU. If you have one,
run the parity check yourself:

```bash
cargo test --release --lib -- --ignored retina_digest_parity
```

WGSL shaders are the exception worth knowing about: they are embedded with `include_str!`, so
`cargo build` only checks that they are valid *strings*. `tests/wgsl_validation.rs` parses and
type-checks them on the CPU with `naga`, no device required. **If you edit a `.wgsl` file, that
test is your only compile-time safety net — run it.**

### Formatting and lints

The tree is not `rustfmt`-clean and not `clippy`-clean today. Measured 2026-08-17:

```console
$ cargo fmt -p universe --check | grep -oP '^Diff in \K[^ :]+' | sort -u | wc -l
121
$ cargo clippy --release --all-targets
warning: `universe` (lib) generated 186 warnings
error: could not compile `universe` (lib) due to 9 previous errors; 186 warnings emitted
```

121 of 127 files disagree with `rustfmt`, and clippy currently reports 9 deny-by-default
errors in the library (`erasing_op`, `absurd_extreme_comparisons`, `approx_constant`,
`never_loop`). CI therefore runs both **non-blocking**, with the numbers recorded in the
workflow so nobody mistakes yellow for green.

What this means for you:

- **Format the code you write.** `cargo fmt -p universe` reformats the whole package, which
  would bury your change under thousands of unrelated lines — don't. Format your own hunks
  (most editors can format a selection with `rustfmt`).
- **Do not add new clippy errors.** Check your own files:
  `cargo clippy --release -p universe 2>&1 | grep -A5 'your_file.rs'`.
- A PR that is *only* "run `cargo fmt` on everything" is welcome, but it must be exactly that
  — no behaviour change mixed in — so the repo-wide diff can be reviewed as a no-op and the
  CI gates can be flipped to blocking afterwards.

---

## The golden rule: behaviour changes ship behind a flag, off by default

**Any change to physics, perception, fitness, or the shape of the brain's input goes in
behind a CLI flag that defaults to off, and arrives with a paired A/B measurement.**

This is not bureaucracy. Champions in this project are *trained artifacts*: a `.dna` file
evolved over hours of GPU time against one specific input distribution. Changing what the
sensors report, or what the physics charges, invalidates every champion in the repository at
once — and it does so silently, because the simulation keeps running and the numbers just get
worse.

Two flags already in `src/bin/gpu_evolve/main.rs` are the template:

**`--visao-periferica-real`** (main.rs:718) — channels 144-159 stop being a bit-for-bit copy
of the fovea and start looking at the edges of the retina. The static defaults to `false`
(`src/gpu/perception.rs:108`), and the comment above the flag states the reason in one line:

> `Off por padrão — muda a DISTRIBUIÇÃO de entrada, então campeão antigo precisa de A/B pareado, não de fé.`
> ("Off by default — it changes the input DISTRIBUTION, so an old champion needs a paired
> A/B, not faith.")

**`--custo-rpm`** (main.rs:636) — charges life per tick for holding RPM above 90% of maximum.
It reads a float into `CUSTO_RPM` (`src/bin/gpu_evolve/statics.rs:115`), which defaults to
`0.0`, and the consumer at `src/bin/gpu_evolve/context.rs:942` is guarded by
`if custo_rpm > 0.0` — with the flag absent, the code path is provably inert.

Copy that shape:

1. A default value that makes the new code a no-op (`false`, `0.0`, or an empty option).
2. A guard at the consumer so the old path is bit-identical when the flag is off.
3. A comment above the flag stating **what distribution or cost it changes** and why it is
   off.
4. An A/B in the PR: the same seeds, the same generation count, flag off vs flag on.

"Paired" means the two arms differ **only** by the flag: same `--seed`, same `--gens`, same
`--ticks`, same track list, at least 3 seeds. One seed is an anecdote; the project has been
burned by it.

---

## Reporting an experimental result

**A number without the command that produced it is not a result.** If you did not measure it,
write "not measured" — never a description standing in for a measurement.

This is the house rule, and it exists because the alternative kept producing confident,
wrong claims (see the "recurring gaps" section of [`CLAUDE.md`](CLAUDE.md), where each entry
is a real error that was caught after the fact). Every one of them was invisible in prose and
obvious next to a command.

A reportable result looks like this:

```
Median distance rose 173 m → 365 m and mortality went to 0%.

  ./target/release/gpu_evolve --gens 40 --seres 60 --seed 7 --same-start \
      --out /media/llm-data/universe-runs/same-start-s7
  (repeated with --seed 8 and --seed 9; medians 361 m / 370 m)
```

Rules that go with it:

- **State who the statistic is about.** Averages over all ticks include dead beings, whose
  controls are pinned to `throttle=0, brake=1`. "They crawl at 1.7 m/s" was that mistake;
  the living ones were doing 79 km/h. Say *living beings, last N generations* or whatever it
  actually is.
- **Compute the metric's ceiling before you evolve against it.** A "wall at 256 m" that turns
  out to be `--ticks 1200 × 15.5 m/s = 290 m` is a report about the tick budget, not the
  model. A hard 0% or a hard 100% is a metric hitting its limit, not a finding.
- **Claims about *evolution* need a trajectory** (first, middle and last generation), not one
  snapshot. Claims about the *population* need the distribution, not the champion.
- **Identical numbers where they should differ are an alarm, never a confirmation.** Stale
  binary, an edit that did not apply, a non-binding ceiling — find the cause before reporting.
- Write run outputs to `/media/llm-data/universe-runs/<name>/`, never to `/tmp` and never
  into the repository.

---

## Commit style

Imperative mood, English, one line under ~72 characters, no trailing period:

```
Gate peripheral vision behind --visao-periferica-real
Fix refuel writing 1.0 into a tank scaled to 1000
Translate the fuel-metabolism comments to English
```

The body explains **why**, and carries the measurement when there is one:

```
Fix refuel writing 1.0 into a tank scaled to 1000

Every 3rd lap set racer.fuel = 1.0 on a tank whose birth scale is 1000
— a 1000x dump disguised as a full tank, leaving ~166 s of life. This is
why "3+ laps" was 0% in every generation measured: reaching the 3rd lap
was what sentenced the being.

3+ laps: 0% -> 31% over 40 generations, seeds 7/8/9.
  ./target/release/gpu_evolve --gens 40 --seres 60 --seed 7 --fuel
```

- One logical change per commit. Formatting-only commits stay separate from behaviour.
- Reference the file and line when you are fixing something specific.
- No "wip", "fix", "changes" as the whole message.

---

## Pull request checklist

Copy this into your PR (the template in
[`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md) already has it):

- [ ] `cargo build --release` is green.
- [ ] `cargo test --release` is green.
- [ ] `cargo test --release --bin gpu_evolve` is green — **mandatory** if `Genome`,
      `save_genome` or `load_genome` changed.
- [ ] `bash scripts/check_language.sh` is green.
- [ ] Everything I wrote is in English, and the Portuguese comments inside the hunks I
      touched are translated with their numbers intact.
- [ ] `tests/wgsl_validation.rs` passes if I edited a `.wgsl` file.
- [ ] Any behaviour change is behind a flag that defaults to off, the flag is documented
      where it is parsed, and the PR carries a paired A/B (≥3 seeds).
- [ ] Every number in the description is followed by the command that produced it; anything
      unmeasured says "not measured".
- [ ] No file exceeds the line limit in [`docs/STYLE.md`](docs/STYLE.md) — if I grew a file
      past it, I split it.
- [ ] I did not add new clippy errors in the files I touched.

---

## Reporting bugs

Use [the bug template](.github/ISSUE_TEMPLATE/bug_report.md). It asks for three things,
because without them a report about this project is unactionable:

1. **The exact command**, with every flag and the seed.
2. **GPU and driver** — the simulation runs on `wgpu`/WGSL compute, and behaviour differs by
   backend and driver version.
3. **The log**, as text, including the lines before the failure.

---

## License

Contributions are accepted under the project's license (`license = "MIT"` in `Cargo.toml`).
By opening a pull request you agree that your contribution is licensed under those terms.
