# Delivery review — 2026-09-25 22:15

Status: host gates re-measured after fixes; GPU gates NOT MEASURED (owner stopped
GPU work at 21:54: "pare a execução na gpu e só avalie o projeto"). No learned
competence claim; this record is about whether the tree can prove itself.

## Gates, measured on this machine (branch `organism/school-20260922`, HEAD after `5c64120`)

| Gate | Command | Before | After | Verdict |
|---|---|---|---|---|
| Format | `cargo fmt -p universe --check` | 85 files, 411 hunks | 0 (school_tests.rs excluded: unstaged edit by another agent, 28 hunks) | PASS on committed tree |
| Language vs origin/main | `bash scripts/check_language.sh` | 8 non-allowlisted files failing (taxi-world ×7, CONTRIBUTING, README) | branch is 597 commits past origin/main; every allowlisted file counts as "added" — structural | FAIL, not actionable |
| Language vs 2026-09-24 | `BASE_REF=9347ec8 bash scripts/check_language.sh` | 3 files (lines the format pass re-indented) | 0 | PASS |
| Library tests | `cargo test --lib` | 579 pass, 97 GPU-ignored | unchanged (pipeline subset re-run: 25 pass) | PASS |
| Binary tests | `cargo test --bin gpu_evolve` | 144 pass | re-run after renames, see log | PASS |
| taxi-world tests | `cargo test -p taxi-world` | never run by CI | 94 pass, 7 ignored | PASS, now in CI |
| Build | `nice -n 10 env CARGO_BUILD_JOBS=1 cargo build --bin gpu_evolve` | — | Finished, exit 0 | PASS |
| CI workflow | `.github/workflows/ci.yml` | rust job exits 1 on a dead fetch step since 2026-08-17 | step removed; branch has no upstream, so no run exists | NOT MEASURED |
| Mechanism (GPU) | `scripts/check_native_flow_gpu.py --out …` | last receipt source `69c28213…` | tree `e034d826…` (silence ranges, lane rule) | NOT MEASURED |
| Contract bench (GPU) | `scripts/contract_bench.py run --root …` | ran on the 19:32 binary, 4 starts | script fixed, 16 starts, binary bound | NOT MEASURED on current binary |

## What was wrong and what changed

1. **CI could never go green.** The rust job's first step exited 1 unless a
   repository variable named an out-of-tree `iara-runtime`; the crates have been
   vendored since `53ee0f4` (2026-08-17). CONTRIBUTING.md still told newcomers the
   tree does not build from a clone. Both fixed (`3e618d1`).
2. **The fmt gate is blocking and the tree was not formatted.** 85 files drifted
   since the 2026-08-17 format commits. Formatted as a formatting-only commit
   (`9d4a02b`), listed in `.git-blame-ignore-revs`.
3. **The language gate was red on files nobody allowlisted** (`crates/taxi-world`,
   added at `41fd3c2`) and on CONTRIBUTING.md quoting its own Portuguese examples.
   Translated with § anchors intact; quoted examples wrapped in the checker's own
   markers (`a49aa02`).
4. **crates/taxi-world was invisible to CI** (workspace member, no default-members).
   101 tests; 94 pass, 7 ignored. Step added (`5c64120`). Six `.pyc` were tracked.
5. **Contract bench defects** (`f064af5`): a negative control the binary refuses,
   an I5 formula wrong under v4, a union `--only`, two contracts that could not
   fail, a vacuous I5/I6 pass, a tick-0 close read as open. Binary provenance is
   now written to `binary.json` and printed at the top of `CONTRACTS.md`.
6. **Provenance rule violated by today's own runs:** 33 of 40 arms under
   `contract-bench-20260925/` had no `commit.txt`/`run_battery.sh`. Written from
   each arm's manifest provenance (git commit `67c9158`, dirty, source
   `e034d826…`).
7. **CLAUDE.md anchors rotted.** `main.rs` grew from 2 582 to 8 887 lines; every
   `main.rs:`/`context.rs:` anchor from 2026-08-17 pointed at unrelated code, and
   the handbook's canonical "comment lies" example (`brain.rs:37`) referred to a
   file that no longer exists and a comment already fixed. Re-verified line by
   line; the generation loop is now described by grep-able names.
8. **DRIVER-MICRO-CONTRACTS** stayed "Pending execution" after being executed the
   same day; marked superseded with a pointer.

## Not done, and why

- GPU receipts and the bench re-run: owner's stop. Queue, in order:
  `check_native_flow_gpu.py` (new receipt for `e034d826…`) → `contract_bench.py
  run --root /media/llm-data/universe-runs/contract-bench-20260925-v2 --genome
  eliteM=…/lane-evo-M-consistent-fixed16/evolve_best_gen0023.bin` → `arms_lane7.sh`
  (arms N/O).
- Language gate against `origin/main` stays structural until the branch is
  merged or rebased; CI uses `github.event.before`, which is the right base.
- `src/gpu/school_tests.rs` keeps another agent's unstaged edit; not formatted,
  not run (GPU fixture).
- Disk: `/` 53 G free (86 % used), `/media/llm-data` 12 G free (88 % used) —
  both past the 80 % line. `target/` is 17 G; bench dirs hold ~7 binary copies
  of ~0.7 GB. Not cleaned: replays of arms whose verdicts are recorded are the
  candidates (rule: lineage `.bin` and `run.log` stay).

## Second wave (22:40–01:30, host only, GPU still stopped)

Three read-only audits (code health, research-gate script chain, trace consumers in
`lab-ui`/`neuromod-lab`) plus a clippy count. Every claim below was re-opened at the
cited line before anything was changed.

### Fixed in this wave

| # | Finding | Fix | Evidence |
|---|---|---|---|
| 1 | `tests/wgsl_validation.rs` covered 10 of 36 shaders and, as an integration test, linked all 41 binaries (~0.7 GB each); nobody ran it after `brain.wgsl` changed | `src/gpu/wgsl_static_tests.rs` (`cargo test --lib wgsl_static`): all 29 runtime shader modules through naga, a test that fails when a `.wgsl` is neither validated nor declared a fragment, the six source-text canaries ported, and a new test pinning every `WAVE`/`L1_OUT`/`L2_OUT`/`NEURON_*` WGSL literal to `config.rs` | 9 passed, 0.31 s; old file archived at `docs/arquivo/wgsl_validation-superseded-20260925.rs.txt` |
| 2 | `publish_research_run.py` never checked the GPU gate: any hand-built `report.json` + `manifest.json` published into the Lab UI catalog | refuses a manifest without `gpu_contract_gate`; re-hashes the receipt file; copies recursively (skills/, diagnostic_candidates/ were silently dropped from a "byte-identical" copy) | `scripts/publish_research_run.py:35-45,58-70` |
| 3 | `run_research_battery.py` hashed the binary before `shutil.copy2` and never after | re-hash of the copy against the receipt; shared `RUNS_ROOT` | `scripts/run_research_battery.py` |
| 4 | `audit_taxi_learning.py` could not certify shell-launched batteries (no `command.json`): `contract-bench-20260925` arms all rejected | falls back to the binary's own `provenance.argv` in `organism_gen0000.json`; still fails closed without either; `--out` must be below `/media/llm-data` | 17 unit tests pass; lane-evo-M now audited (verdict: no delivery, correct for a stop lesson) |
| 5 | `contract_bench.py`: own copies of `ENV`/`sha`, no subprocess timeout, trusted `--build-provenance` blindly, no `--help` text | imports the runner's helpers; `--timeout-seconds` (1800) records `"timeout"` and continues; `binary.json` carries `binary_matches_tree`; `--help` shows the docstring | `contract_bench.py report` re-run on the existing root |
| 6 | 20 of 73 `pub static`s in `statics.rs` missing from its ownership table (the file's own rule) | 16 rows added, writer flag and reader site for each | `src/bin/gpu_evolve/statics.rs` header |
| 7 | `docs/STYLE.md` split queue measured 2026-08-17: 24 of the 52 files now over the cap were absent, `main.rs` listed at 2 582 (actual 8 887) | regenerated from `scripts/file_size_meter.sh --table` (new, informational); summary table re-measured | 52 `.rs` (19 %) and 7 `.wgsl` (18 %) over cap |
| 8 | `--build-provenance` is the one flag outside `cli_flags.rs`, invisible to the registry round-trip | explicit exemption test `the_build_provenance_query_is_handled_before_the_parser` | `src/bin/gpu_evolve/cli_flags_tests.rs` |
| 9 | Language gate against the policy's own start commit (`29b35fa`, 2026-08-17) still red: ~320 Portuguese lines added *after* the rule, mostly `main.rs` (109), `context.rs` (77), `channels.rs` (32), `world.rs` (21) | translated (two agents, diff reviewed, numbers/`file:line` kept); `MAX_LINES_PER_FILE` now overridable for a full listing | see the commit that follows this section |
| 10 | `src/CLAUDE.md` still told contributors to delete `perception_old_backup.rs`, gone since the "removing trash" commit | paragraph removed | — |

### Measured, recorded, not fixed

- **clippy**: 322 warnings, 0 errors on `--lib --bin gpu_evolve` (24 unused doc
  comment, 19 empty line after doc comment, 27 too-many-arguments, 14 no-effect op,
  13 doc-list indentation). CI keeps clippy non-blocking; the count is in the CI header.
- **Trace consumers drift** (`lab-ui`, `neuromod-lab`; both trees carry someone else's
  uncommitted WIP on exactly these files, so nothing was edited there):
  - written by Universe, dropped at the first hop: `sensor_pose`, `slow_brain`,
    `service_events` (`events-loader.ts` `VALID_KINDS` hardcodes 8 racing kinds and throws
    on the four taxi kinds), 18 per-being `record_json` fields and 11 summary fields
    (`school_observatory.py` `BEING_KEYS`/`SUMMARY_KEYS` allow-lists).
  - consumed, never written: `organism.destinations` (`Track2DPanel.vue:56,299,500`
    always sees `[]`), `universe_commit` (producer reads a key only the publisher writes;
    normal runs get `"unknown"`).
  - three version families (`SCHOOL_SCHEMA_VERSION=5`, `observation_schema=2`, trace-v1
    `=1`) cross-checked nowhere; `contract_gate.py` guards format, not field coverage.
  - genome SHA-256 binding exists for schema-2 staged runs but reaches no panel: the
    "which genome am I looking at" guarantee is invisible in the UI.
  - Portuguese UI strings are pervasive (every panel), not the three that were flagged.
- **Bench wiring**: `contract_bench.py` is still not a precondition of
  `run_research_battery.py --execute`. Proposed shape (not built): `--contract-bench-root`
  that requires `binary.json.sha256 == sha(binary)` and zero graded `false` cells in
  `contracts.json`, recorded next to `gpu_contract_gate`. Needs the bench re-run on the
  current binary first (GPU queue).
- **`source_hash()`** includes editor swap files under `src/`; intended (fail closed),
  noted in the gate docstring only.
