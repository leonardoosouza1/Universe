# English migration tracker

Universe was written in Portuguese. On **2026-08-17** the project moved to
English-only (`docs/STYLE.md`, `NAMING.md`). This file turns "translate 54,000 lines"
into a queue of small, independent, reviewable pull requests.

**Snapshot taken 2026-08-17** with `./scripts/check_language.sh --all`:

| | files | offending lines |
|---|---:|---:|
| Live code (`src/`, `tests/`, root) | 140 | 11,819 |
| Archived crates under `docs/arquivo/` | 30 | 423 |
| **Total** | **170** | **12,242** |

"Offending lines" counts lines containing Portuguese diacritics or one of the
checker's unambiguous Portuguese function words. It is a workload proxy, not an exact
count of translatable sentences — a line with three Portuguese words counts once.

Regenerate the numbers at any time:

```bash
./scripts/check_language.sh --all
```

**Product documentation is not on this queue.** `README.md`, `CONTRIBUTING.md`,
`NAMING.md`, `docs/STYLE.md`, `docs/MIGRATION.md` and anything under `.github/` are
written under the current policy, so Portuguese in them is a defect to fix now, not
debt to schedule. `--generate-allowlist` refuses to excuse them for the same reason.

> As of this snapshot, `README.md` and `CONTRIBUTING.md` each fail the checker on
> lines that quote Portuguese CLI flags and legacy test names. Both are legitimate
> quotations: mark them with `<!-- check-language: allow -->` rather than deleting the
> quote. `--pista-held-out` and `--held-out-cada` are real flags parsed at <!-- check-language: allow -->
> `src/bin/gpu_evolve/main.rs:739-740`; renaming them is a user-facing change and
> needs the owner's decision (tracked in `NAMING.md`).

---

## How to migrate a file

1. **Claim it.** Open an issue or a draft PR naming the file, and put your handle in
   the Owner column here so two people do not translate the same file twice.
2. **One file per PR.** Translate comments, doc comments, identifiers, log and panic
   strings, and test names. Apply the rename table in `NAMING.md`; if you need a name
   that is not there, add the row in the same PR.
3. **Do not lose content.** These comments record bugs paid for in GPU hours. Every
   number, seed, date, `file:line` and command must survive verbatim. Translate the
   prose; never summarise, shorten, or drop a measurement. If a comment is wrong or
   stale, say so in the PR — do not silently delete it.
4. **Prove nothing broke.** `cargo build --release` must be clean and
   `cargo test --release --bin gpu_evolve` must stay at 39 passing. Renaming a public
   item means updating every call site, including inside `.wgsl` shaders, which the
   compiler will not catch for you.
5. **Close it out.** Delete the file's line from `scripts/language_allowlist.txt`,
   flip its Status here to `done`, and confirm `./scripts/check_language.sh` passes.
   The allowlist only ever shrinks.

### Good first files

The bottom of the table — files with fewer than ~20 offending lines — are complete,
self-contained PRs that take minutes and teach the workflow. Start there, not at
`src/bin/gpu_evolve/main.rs` (850 lines of Portuguese, and the file is 2,582 lines
long and already queued for a split in `docs/STYLE.md`).

### Files worth splitting first

Eight files carry more than 200 offending lines each and are also over the 600-line
size cap. For those, **split before translating**: a 3,382-line file translated in one
PR is unreviewable, whereas the same work as one module per PR is not. See the split
queue in `docs/STYLE.md`.

---

## Queue — live code

Sorted by offending lines, descending. Status is `pending` or `done`.

| File | Occurrences | Status | Owner |
|---|---:|---|---|
| `src/bin/gpu_evolve/main.rs` | 850 | pending | — |
| `src/gpu/evolution.rs` | 636 | pending | — |
| `src/simulation/motor.rs` | 590 | pending | — |
| `src/gpu/brain.wgsl` | 436 | pending | — |
| `src/bin/gpu_evolve/context.rs` | 387 | pending | — |
| `src/bin/gpu_loop.rs` | 385 | pending | — |
| `src/gpu/perception.rs` | 358 | pending | — |
| `src/perception/channels.rs` | 340 | pending | — |
| `src/evolution/pipeline.rs` | 295 | pending | — |
| `src/gpu/brain.rs` | 295 | pending | — |
| `src/bin/cortex_evolve.rs` | 270 | pending | — |
| `src/gpu/perception.wgsl` | 255 | pending | — |
| `src/world.rs` | 228 | pending | — |
| `src/neuro.rs` | 202 | pending | — |
| `src/bin/gpu_report.rs` | 198 | pending | — |
| `src/bin/gpu_evolve/escada.rs` | 161 | pending | — |
| `src/bin/gpu_validate.rs` | 152 | pending | — |
| `src/bin/cortex_map.rs` | 149 | pending | — |
| `src/gpu/cortex_sheet.rs` | 144 | pending | — |
| `src/gpu/core.rs` | 143 | pending | — |
| `src/evolution/spawn.rs` | 133 | pending | — |
| `src/bin/gpu_evolve/piloto_forja.rs` | 131 | pending | — |
| `src/gpu/physics.rs` | 130 | pending | — |
| `src/gpu/perception_temporal.wgsl` | 126 | pending | — |
| `src/bin/gpu_evolve/statics.rs` | 125 | pending | — |
| `src/engine.rs` | 119 | pending | — |
| `src/simulation/racer.rs` | 117 | pending | — |
| `src/bin/gpu_evolve/fantasma.rs` | 109 | pending | — |
| `src/analysis/activation.rs` | 107 | pending | — |
| `src/pista_gen.rs` | 107 | pending | — |
| `src/bin/replay_mapa.rs` | 106 | pending | — |
| `src/bin/genoma_view.rs` | 93 | pending | — |
| `src/gpu/cortex_sheet.wgsl` | 91 | pending | — |
| `src/math/activations.rs` | 91 | pending | — |
| `src/gpu/physics.wgsl` | 89 | pending | — |
| `src/analysis/contextual.rs` | 88 | pending | — |
| `src/gpu/retina_digest.wgsl` | 86 | pending | — |
| `src/analysis/regioes.rs` | 85 | pending | — |
| `src/bin/gpu_evolve/tests.rs` | 85 | pending | — |
| `src/replay.rs` | 84 | pending | — |
| `src/gpu/mod.rs` | 81 | pending | — |
| `src/gpu/slow_brain.rs` | 79 | pending | — |
| `src/config.rs` | 78 | pending | — |
| `src/bin/replay_diag.rs` | 76 | pending | — |
| `src/bin/gpu_evolve/piloto_linear.rs` | 75 | pending | — |
| `src/evolution/dna_genesis.rs` | 74 | pending | — |
| `src/environment.rs` | 73 | pending | — |
| `src/gpu/retina_digest.rs` | 69 | pending | — |
| `src/gpu/slow_brain.wgsl` | 69 | pending | — |
| `src/math/linalg.rs` | 69 | pending | — |
| `src/brain_capacity.rs` | 68 | pending | — |
| `src/gpu/perception_motion.wgsl` | 67 | pending | — |
| `src/bin/gpu_evolve/audit.rs` | 66 | pending | — |
| `src/bin/gpu_evolve/painel.rs` | 66 | pending | — |
| `src/bin/replay_obs.rs` | 66 | pending | — |
| `src/cfc.rs` | 65 | pending | — |
| `src/evolution/dna.rs` | 65 | pending | — |
| `src/brain/format.rs` | 64 | pending | — |
| `src/substrate.rs` | 63 | pending | — |
| `src/bin/sensor_audit.rs` | 62 | pending | — |
| `src/bin/sinapses.rs` | 62 | pending | — |
| `src/gpu/genome_io.rs` | 59 | pending | — |
| `src/math/quant.rs` | 56 | pending | — |
| `src/math/thresholds.rs` | 55 | pending | — |
| `src/bin/fisica_bisect.rs` | 54 | pending | — |
| `src/bin/gpu_evolve/types.rs` | 54 | pending | — |
| `src/bin/visao_check.rs` | 54 | pending | — |
| `src/bin/cortex_test.rs` | 53 | pending | — |
| `src/physics.rs` | 52 | pending | — |
| `src/metabolism.rs` | 51 | pending | — |
| `src/bin/pistas.rs` | 48 | pending | — |
| `src/gpu/retina_digest_parity_test.rs` | 47 | pending | — |
| `src/main.rs` | 44 | pending | — |
| `tests/wgsl_validation.rs` | 44 | pending | — |
| `src/lab.rs` | 43 | pending | — |
| `src/bin/replay_analyze.rs` | 42 | pending | — |
| `src/bin/gpu_evolve/pilots.rs` | 40 | pending | — |
| `src/evolution/settings.rs` | 36 | pending | — |
| `src/bin/pista_check.rs` | 35 | pending | — |
| `src/bin/gpu_evolve/pistas.rs` | 34 | pending | — |
| `src/visao.rs` | 34 | pending | — |
| `src/math/cfc_perception.rs` | 33 | pending | — |
| `src/math/laws.rs` | 32 | pending | — |
| `src/math/stats.rs` | 32 | pending | — |
| `Cargo.toml` | 30 | pending | — |
| `src/bin/dna_console.rs` | 30 | pending | — |
| `src/gpu/slow_gather.wgsl` | 30 | pending | — |
| `src/gpu/retina.wgsl` | 29 | pending | — |
| `src/bin/replay_pop.rs` | 28 | pending | — |
| `src/bin/slow_gpu_equiv.rs` | 28 | pending | — |
| `src/gpu/physics_b.wgsl` | 28 | pending | — |
| `src/kan.rs` | 28 | pending | — |
| `src/specialist.rs` | 28 | pending | — |
| `src/bin/slow_probe.rs` | 27 | pending | — |
| `src/bin/impact.rs` | 26 | pending | — |
| `src/rng.rs` | 24 | pending | — |
| `src/gpu/physics_a.wgsl` | 23 | pending | — |
| `src/layers.rs` | 22 | pending | — |
| `src/memory.rs` | 22 | pending | — |
| `src/simulation/mod.rs` | 22 | pending | — |
| `tests/ablacao_cortex.rs` | 19 | pending | — |
| `src/bin/dna_view.rs` | 18 | pending | — |
| `src/bin/genoma_dump.rs` | 18 | pending | — |
| `src/gpu/physics_c.wgsl` | 18 | pending | — |
| `src/gpu/cortex_sheet_hebb.wgsl` | 17 | pending | — |
| `src/bin/dump_dna.rs` | 16 | pending | — |
| `src/gpu/cortex_sheet_motor.wgsl` | 16 | pending | — |
| `src/gpu/physics_power_b.wgsl` | 16 | pending | — |
| `src/bin/dna_compress.rs` | 15 | pending | — |
| `src/gpu/physics_power_a.wgsl` | 12 | pending | — |
| `src/gpu/physics_power_c.wgsl` | 12 | pending | — |
| `tests/ablacao_controle.rs` | 12 | pending | — |
| `src/gpu/retina_types.rs` | 11 | pending | — |
| `src/math/mod.rs` | 11 | pending | — |
| `src/ui.rs` | 11 | pending | — |
| `src/gpu/physics_phase_integ.wgsl` | 9 | pending | — |
| `src/safety/sentinel.rs` | 8 | pending | — |
| `tests/integration_test.rs` | 8 | pending | — |
| `src/analysis/mod.rs` | 7 | pending | — |
| `src/dense_layer.rs` | 6 | pending | — |
| `src/perception/mod.rs` | 6 | pending | — |
| `src/safety/mod.rs` | 6 | pending | — |
| `src/ui_menu.rs` | 6 | pending | — |
| `src/gpu/physics_minimal.wgsl` | 5 | pending | — |
| `src/gpu/physics_phase_susp.wgsl` | 5 | pending | — |
| `src/prefabs.rs` | 5 | pending | — |
| `src/scenery.rs` | 5 | pending | — |
| `src/brain/mod.rs` | 4 | pending | — |
| `src/evolution/logs.rs` | 4 | pending | — |
| `src/gpu/physics_phase_power.wgsl` | 4 | pending | — |
| `src/lib.rs` | 4 | pending | — |
| `src/bin/gpu_evolve/genome_io.rs` | 3 | pending | — |
| `src/bin/generic_dna.rs` | 2 | pending | — |
| `src/evolution/mod.rs` | 2 | pending | — |
| `src/perception_old_backup.rs` | 2 | pending | — |
| `src/perception/retina.rs` | 2 | pending | — |
| `src/mesh.rs` | 1 | pending | — |
| `src/reality.rs` | 1 | pending | — |
| `src/repominer.rs` | 1 | pending | — |
| `src/types.rs` | 1 | pending | — |

---

## Queue — archived crates

`docs/arquivo/dirs/iara_genesis/` is still a workspace member (`Cargo.toml:7`), so it
compiles and its language counts. `docs/arquivo/dirs/iara_agi/` is archived code kept
for reference. Where these should live has not been decided — see the open items in
`docs/STYLE.md`. **Lowest priority:** do not spend a PR here while live code is
pending.

| File | Occurrences | Status | Owner |
|---|---:|---|---|
| `docs/arquivo/dirs/iara_genesis/src/codec/semantic.rs` | 95 | pending | — |
| `docs/arquivo/dirs/iara_genesis/src/universal.rs` | 53 | pending | — |
| `docs/arquivo/dirs/iara_genesis/src/model/tree.rs` | 48 | pending | — |
| `docs/arquivo/dirs/iara_genesis/src/model/soil.rs` | 29 | pending | — |
| `docs/arquivo/dirs/iara_genesis/src/math/fourier.rs` | 24 | pending | — |
| `docs/arquivo/dirs/iara_genesis/src/neural/autoencoder.rs` | 22 | pending | — |
| `docs/arquivo/dirs/iara_genesis/src/neural/svd.rs` | 19 | pending | — |
| `docs/arquivo/dirs/iara_genesis/src/math/chebyshev.rs` | 18 | pending | — |
| `docs/arquivo/dirs/iara_genesis/src/math/sparse.rs` | 18 | pending | — |
| `docs/arquivo/dirs/iara_genesis/src/codec/mod.rs` | 17 | pending | — |
| `docs/arquivo/dirs/iara_genesis/src/tlv.rs` | 14 | pending | — |
| `docs/arquivo/dirs/iara_genesis/src/lib.rs` | 13 | pending | — |
| `docs/arquivo/dirs/iara_genesis/src/neural/codebook.rs` | 9 | pending | — |
| `docs/arquivo/dirs/iara_agi/src/main.rs` | 7 | pending | — |
| `docs/arquivo/dirs/iara_genesis/src/codec/neural_codec.rs` | 6 | pending | — |
| `docs/arquivo/dirs/iara_genesis/src/codec/math_codec.rs` | 5 | pending | — |
| `docs/arquivo/dirs/iara_agi/src/metabolism.rs` | 4 | pending | — |
| `docs/arquivo/dirs/iara_genesis/src/neural/mod.rs` | 3 | pending | — |
| `docs/arquivo/dirs/iara_agi/src/cfc.rs` | 2 | pending | — |
| `docs/arquivo/dirs/iara_agi/src/engine.rs` | 2 | pending | — |
| `docs/arquivo/dirs/iara_agi/src/kan.rs` | 2 | pending | — |
| `docs/arquivo/dirs/iara_agi/src/layers.rs` | 2 | pending | — |
| `docs/arquivo/dirs/iara_agi/src/mesh.rs` | 2 | pending | — |
| `docs/arquivo/dirs/iara_genesis/Cargo.toml` | 2 | pending | — |
| `docs/arquivo/dirs/iara_genesis/src/model/mod.rs` | 2 | pending | — |
| `docs/arquivo/dirs/iara_agi/src/memory.rs` | 1 | pending | — |
| `docs/arquivo/dirs/iara_agi/src/reality.rs` | 1 | pending | — |
| `docs/arquivo/dirs/iara_agi/src/repominer.rs` | 1 | pending | — |
| `docs/arquivo/dirs/iara_agi/src/resonance.rs` | 1 | pending | — |
| `docs/arquivo/dirs/iara_genesis/src/math/mod.rs` | 1 | pending | — |

---

## Out of scope

Not migration targets, and not scanned by `scripts/check_language.sh`:

- **`docs/` experiment records** — `PLANO_EVOLUCAO.md`, `MATH_INVENTORY.md`,
  `DESTILABILIDADE.md`, `REGIOES_CEREBRAIS.md`, battery verdicts, session notes.
  These are dated records of what was measured. Rewriting them would edit history.
- **`CLAUDE.md`, `LEIA_PRIMEIRO.md`, `ARQUITETURA.md`** — the owner's operating
  notes. `ARQUITETURA.md` is a module map and a natural future translation, but it is
  the owner's call, not a contributor's. TODO: decide.
- **`scripts/*.sh` and `scripts/*.py`** — experiment drivers, named after the
  batteries they ran. Renaming them breaks the commands recorded in the docs above.
  TODO: decide whether new scripts must be English (recommended) while old ones stay.

## Progress

| Date | Files pending | Lines pending |
|---|---:|---:|
| 2026-08-17 | 170 | 12,242 |

Add a row when you finish a batch, so the trend is visible without re-reading the git
history.
