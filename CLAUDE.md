# Universe — working notes for agents

## Required workflow: evidence before implementation and completion

Read [AGENTS.md](AGENTS.md) and
[VALIDATION-WORKFLOW.md](docs/VALIDATION-WORKFLOW.md) before diagnosing, editing,
running or publishing results. Use its investigation template and source map.
Trace the same being/tick from sensors through all brain/action/body stages to
motion and task outcome. Inspect the actual supplied replay. Failed acceptance
requires another bounded diagnosis/correction cycle, not a success claim based
on fitness, peak speed, a short lesson, or an agent's report. This current workflow
and the owner's GPU-only restriction take precedence over historical commands
below; it does not authorize CPU numerical validators.

## Current execution limit — owner's instruction

Leonardo reported a desktop freeze and requires **GPU-only numerical validation**
for the current coevolution work. Do not execute CPU test suites, standalone CPU
oracles, or mixed CPU/GPU validators despite the historical commands below. Use
small real-GPU batches and reject software GPU adapters. Host file inspection and
necessary compilation are allowed; build only the required target, at low priority
with one job and no concurrent builds/tests. Do not restart heavy batteries.

## Experimental coevolution handoff (2026-09-20)

Read [runtime contract](docs/COEVOLUTION-RUNTIME.md) and
[measured results](docs/COEVOLUTION-VALIDATION.md) before enabling the new flags.
The combined mechanism has not earned promotion to default. Unknown flags now fail;
use `--seed`, not `--pop-seed`. Active thermal physics is `physics_a.wgsl`.
A 3D inherited graph is not a trace of acquired plastic weights. Multi-lineage
HTML export is supported. Lab-ui now packages separate per-life trace-v1 directories;
see [INTEGRATION-20260921.md](docs/INTEGRATION-20260921.md) for canonical exporters
and the real-run panel path.

## Scientific engineering handbook

For code, metrics, and experiments, read the canonical sibling handbook:
[`../neuromod-lab/docs/HANDBOOK-ENGENHARIA-CIENTIFICA.md`](../neuromod-lab/docs/HANDBOOK-ENGENHARIA-CIENTIFICA.md).
This dependency is explicit; if the file is missing, ask to locate it rather than
silently skipping it.

Neuroevolution simulator in Rust + WGSL. Beings with a brain
(perception → KAN → CfC → BitLinear L1 → FloatLinear L2 → motor) drive procedural
tracks; a genetic algorithm selects them. Perception, native neural computation and physics run on the GPU through
`wgpu`, headless; command mapping, resource bookkeeping and evolution retain host work — Bevy is an optional window over it, never a dependency of the core.

**Read [`ARCHITECTURE.md`](ARCHITECTURE.md) before changing anything.** It has the tick
diagram, the module map, the genome format and the invariants that break silently.
Style rules are in [`docs/STYLE.md`](docs/STYLE.md); the contribution flow is in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Commands

```bash
# Build ONLY the binary you need. A bare `cargo build` links 37 debug binaries of ~0.7 GB
# each (26 GB, filled the system disk on 2026-09-23). School work runs the DEBUG binary.
nice -n 10 env CARGO_BUILD_JOBS=1 cargo build --bin gpu_evolve
cargo test --bin gpu_evolve                  # genome round-trip, CLI registry, K-start aggregation
cargo test --lib school::tests               # school host tests (schema, record JSON)
cargo test --lib school_tests -- --ignored   # 22 school judge fixtures on the real GPU
cargo test --test wgsl_validation            # parses and type-checks the WGSL, no GPU needed
bash scripts/check_language.sh               # language gate: only what this branch changed
bash scripts/check_language.sh --all         # progress meter for the whole repo, never fails
```

`cargo build` only checks that the WGSL is a valid *string* (`include_str!`), which is
why `wgsl_validation` exists. Run it after touching any `.wgsl`.

A short smoke run, before any long battery:

```bash
env -u DISPLAY ./target/debug/gpu_evolve --gens 2 --beings 12 --ticks 600 \
    --seed 7 --same-start --out /your/own/scratch/smoke
```

Every per-generation manifest carries `provenance` (git commit, dirty flag, argv, seed)
since 2026-09-24; a battery directory still needs its `run_battery.sh` and `commit.txt`.

`--out` defaults to `.` (`src/bin/gpu_evolve/main.rs:395`), so always pass your own
directory — see rule 5.

---

## Rules for any agent working here

Every rule below is a real mistake that was made in this repository and paid for in GPU
hours. The one-line lesson attached to each is the incident, not an illustration.

### 1. A claim about behaviour ships with the number and the command that produced it

If you did not measure it, write "not measured" — never a description standing in for a
measurement. State *who* the statistic is about: the simulator pins `throttle=0,
brake=1` on a dead being, and averaging that in is how "they crawl at 1.7 m/s" survived
twenty messages while the living ones were doing 79 km/h.

### 2. Identical numbers where they should differ are an alarm, not a confirmation

This canary appeared 8+ times in one session and never once lied. Causes found: a stale
binary, a different code path, an edit that never applied, a non-binding ceiling, a
display-only change that never touched selection, a unit artefact. Stop and find the
cause before concluding anything.

### 3. New physics or perception enters behind a flag that is off by default, with a paired A/B

The code already says it, at `src/bin/gpu_evolve/main.rs:715-718` for
`--visao-periferica-real`: *off by default — it changes the input distribution, so an
old champion needs a paired A/B, not faith.* Paired means same seeds, `--same-start`,
one arm per flag state. Ablate by silencing the signal, not by deleting the organ
(`src/bin/gpu_evolve/statics.rs:41-48`): removing the organ changes two things at once.

### 4. Compute the ceiling of the metric before you evolve against it

`--ticks 1200` produced a reported "wall at 256 m". At 15.5 m/s, 1200 ticks at 64 Hz is
18.75 s = 290 m of physical ceiling — the wall was the end of the tick budget. A hard 0%
or a hard 100% is a metric hitting its limit, not a model failing. Write the ceiling into
the log before the run.

### 5. Heavy outputs go outside the repository, through `--out`

Never `/tmp`, never next to the source. Replays are the expensive artefact: recording
every generation × 16 races produced 31 GB once. On a long run, record only the first,
middle and last generation.

### 6. A code comment is not evidence — open the `file:line`

Verified example: `src/gpu/brain.rs:37` says `// 223 × 96 = 21.408`, but
`SENSORY_WAVE_SIZE` is 192 (`src/config.rs:309`), so the real figure is 18 432. The
constant is computed, so the code is right and only the comment lies. The same rule
applies to another agent's report: it is a hypothesis until you open the line and read it.

### Also, before announcing a result

Claims about **evolution** need a trajectory (first, middle, last generation), not one
snapshot; claims about the **population** need the distribution, not the champion. Use
≥3 paired seeds and the mean of the last N generations — never the best of all of them.
Follow the effect size as n grows, not σ.

---

## Where to look

| You are touching | Read first |
|---|---|
| Anything at all | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Naming, comments, file size, tests | [`docs/STYLE.md`](docs/STYLE.md) |
| Branch, commit and PR flow | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| A name that is still Portuguese | [`NAMING.md`](NAMING.md), [`docs/MIGRATION.md`](docs/MIGRATION.md) |
| Library code | [`src/CLAUDE.md`](src/CLAUDE.md) |
| Kernels, buffers, genome I/O | [`src/gpu/CLAUDE.md`](src/gpu/CLAUDE.md) |
| The evolution binary | [`src/bin/gpu_evolve/CLAUDE.md`](src/bin/gpu_evolve/CLAUDE.md) |
| Experiment records | [`docs/CLAUDE.md`](docs/CLAUDE.md) |

**Language: English everywhere** — identifiers, comments, doc comments, log and panic
messages, test names, commit messages. The Portuguese-comment rule that used to live in
`NAMING.md` was revoked on 2026-08-17. Translating must never cost content: every
measurement, seed, `file:line` and command inside a comment survives intact.

---

## Maintainer notes

Everything below is specific to the owner's machine and working style. It is not
required to contribute, and an outside contributor can ignore it entirely.

**Storage.** The development machine keeps simulation output on a dedicated 90 GB
partition, `/media/llm-data`, and treats the system disk as source-code only — batteries
in progress were killed when the system disk fell to 340 MB free, and `/media/llm-data`
itself once hit 100% with 803 MB left.

- Run output and replays: `--out /media/llm-data/universe-runs/<name>/`
- Genome and code snapshots: `/media/llm-data/snapshots/`
- Models and weights: `/media/llm-data/models/`
- Heavy scratch: `/media/llm-data/scratch/`
- Permanent target: 20-30 GB free. Every watch cycle starts with
  `df -h /media/llm-data`; below 20 GB, run `~/projects/LLM/tools/limpeza-dados.sh --real`.
- A finished battery whose verdict is already recorded in `docs/` is disposable data:
  replays go, the lineage `.bin` and `run.log` stay.
- Before launching a battery, estimate what it will produce (generations × ~30 MB × beings
  recorded) and check it fits without breaking the target.

No production code path depends on that partition. Two hardcoded `/media/llm-data` paths
remain, both inside a `#[cfg(test)]` block (`src/bin/gpu_evolve/piloto_forja.rs:366-367`),
and that test states honestly that without the partition mounted it does not run — which
is not a PASS (`piloto_forja.rs:518`).

**Continuity (owner's standing instruction).** When the owner says he is stepping away,
writing a report is not stopping — the shift ends when he is back. Never report with the
machine idle: leave a process running first, then schedule the next wake-up, then write.
Keep an explicit queue of targets so the next one starts without asking. Finding a bug and
reporting it is not delivery either: fix it, run the test that proves the fix, measure the
effect on the real target, and only then move to the next bottleneck.

**Agent workflows (the pattern that worked, 2026-08-11).** Substantial work here — a
battery, an audit, a bug hunt, a subsystem validation — runs as a workflow of agents:

1. **Parallel diagnosis, read-only.** One agent per independent front. Each returns
   `cause`, `evidence` (`file:line` + pasted snippet) and `confidence`. Rules in the
   prompt: do not trust comments (several lie); distinguish "no source" from "source not
   fed" from "only varies in time"; "NOT CONFIRMED" is a valid answer.
2. **Sequential repair, one agent.** Fixes touch the same files; parallelism here produces
   conflicts, not speed. Low-confidence proposals get re-confirmed before being applied.
   Compile after each item; run the genome tests.
3. **Validation on the real GPU, two seeds.** Identical numbers between seeds are an alarm
   to investigate, never an approval. Explicit instruction: do not dress it up; a bad
   number gets reported.
4. **A 2-second smoke test before a 30-minute battery** — `--auditar-visao`, `pista_check`,
   a single-generation ablation.
5. While the workflow runs, the main turn does not idle: memory, indexes, diagnostic tools,
   reading the next suspect.

This pattern found and fixed in one pass: dead channels 111-113/118-120/123, cascading
corruption in `set_all_weights`, and a bug in the audit itself.
