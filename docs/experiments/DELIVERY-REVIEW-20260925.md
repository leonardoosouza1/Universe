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
