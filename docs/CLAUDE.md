# `docs/` — plans, records and measurements

Two kinds of file live here and they must not be confused. **Standards** say what a
change has to satisfy; **records** say what an experiment predicted and what it then
measured. A record is never edited to look better — it is corrected in place, with the
correction dated and the original claim left readable.

Most records are still in Portuguese: they predate the 2026-08-17 English-only decision
and are on the queue in `MIGRATION.md`. Anything written from now on is English.

## Standards

| File | What it is |
|---|---|
| `STYLE.md` | The code style a change must satisfy — English everywhere, 600-line cap for `.rs` and 400 for `.wgsl`, one source of truth per constant, a regression test per bug fix. Every rule is machine-checkable or has a worked example from this repo |
| `MIGRATION.md` | The English migration tracker: 54 000 lines turned into a queue of small, independent, reviewable PRs |
| `REFACTOR_PLAN.md` | Planned refactors with a risk register (`R1`…`R8`), built 2026-08-13 from a static map of five fronts. No file was altered to produce it |
| `MATH_INVENTORY.md` | Every formula in the tree, extracted **from the code, not from the comments** — the header says so explicitly, and entries that could not be confirmed in code are marked `NÃO CONFIRMADO` |

## Plans and records

| File | What it is |
|---|---|
| `PLANO_EVOLUCAO.md` | The phase plan (`P0`…`P1.9`) referenced all over the code. Written 2026-08-13, right after the first complete lap: 3 189 m median, 25 of 48 beings with ≥1 lap, seed 77 |
| `ESCADA_DIFICULDADE.md` | The world-wide difficulty ladder — one rung per battery, on merit, modelled on the gearbox curriculum that worked 3/3 |
| `REGIOES_CEREBRAIS.md` | The brain-regions thesis and its experiment ladder. Note line 61: *prediction registered BEFORE each battery; a negative result documented with its mechanism* |
| `CEREBRO_VIRTUAL.md` | Long-term master plan: functional regions, support cells, and an energy economy where thinking costs |
| `DESTILABILIDADE.md` | Distillability — what "a clone beat the champion" meant. Opens with a ⚠ correction from the 2026-08-16 adversarial audit putting the whole document under review. This is the model to copy: the wrong conclusion stays visible, with the reason it fell |
| `SCREEN_CALIBRATION.md` | The open-loop screen (`src/bin/screen.rs`) calibrated against closed-loop ground truth, 85 candidates, 2026-08-27. A negative result: the screen is bit-exact and 46× cheaper than the closed loop, but its 0.26 threshold rejects brains that drive better than the recording champion, and its score does not rank by quality (ρ = −0.05). Corrects the speed and threshold claims in `superpowers/specs/2026-08-26-gpu-resident-scenarios-design.md` §4 |
| `UNIVERSO_ALEM_DO_CARRO.md` | Expansion brainstorm — the Universe beyond the car |
| `arquivo/` | Records from earlier sessions, kept for provenance. Also holds `arquivo/dirs/iara_genesis`, a workspace member (see the comment at the top of `Cargo.toml`) |
| `lab_specs/`, `dna_v4_genesis/`, `superpowers/` | Lab run specs (JSON), the DNA-V4 note, and older plans/specs |

## The rule for adding a record

**Prediction before, number after.** A battery is registered in two moves:

1. *Before running:* write down what you expect, in what direction, and the **ceiling of
   the metric** under that configuration. A prediction added afterwards is a description.
   A ceiling computed afterwards is how `--ticks 1200` produced a reported "wall at 256 m"
   that was really the 290 m the tick budget allowed.
2. *After running:* the measured number, the seeds, who the statistic is about (living
   beings? last N generations?), and the command that reproduces it. A number without its
   command is not a result.

A negative result is registered with its **mechanism** — why it failed usually points at
the next experiment, which is why it is worth more than a silent deletion. Identical
numbers across conditions that should differ are an alarm to investigate, never a finding.

When a later measurement overturns an earlier document, add a dated correction block at
the top, like `DESTILABILIDADE.md` does, and leave the original text in place. Deleting it
destroys the only evidence of how the error was possible.

## Housekeeping

- Docs are text, so they live in the repository. **Run outputs do not** — replays,
  genomes and logs go to the directory you pass to `--out`, outside the tree.
- When you cite a `file:line`, cite what you actually opened. Comments in this repository
  are unusually valuable and, in places, stale (`src/gpu/brain.rs:37` is the verified
  example); a doc that quotes a stale comment propagates it.
