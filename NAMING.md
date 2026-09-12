# Naming — the contract between contributors

## Language policy

**English everywhere.** Identifiers, comments, doc comments, log and panic messages,
test names, file names, CLI flags, and commit messages.

> ### Revoked: the Portuguese-comment rule
>
> Until **2026-08-17** this file carried the opposite instruction:
>
> > *"Every IDENTIFIER in ENGLISH. Comments and output strings stay in PORTUGUESE —
> > do not translate any comment."*
>
> **That rule is revoked**, by the owner's decision, ahead of opening the project to
> outside contributions. Portuguese comments are no longer protected: they are now a
> migration target. The single exception is that translating one must never cost
> content — every measurement, seed, `file:line` and command in a comment survives the
> translation intact. See [`docs/STYLE.md`](docs/STYLE.md) for the full standard and
> [`docs/MIGRATION.md`](docs/MIGRATION.md) for the file-by-file queue.

The rule is enforced on every change by `scripts/check_language.sh`.

## Why this table exists

Contributors and agents work on one file at a time without seeing each other's work.
This table is what makes names agree across files: it records the renames already
performed, so nobody reintroduces an old Portuguese name or invents a second English
one for something that already has one.

**If a name is not in this table and it is in Portuguese, translate it to the obvious
English and move on.** A reasonable English name beats a correct Portuguese one. Add
the row here in the same commit so the next person inherits the decision.

<!-- The tables below are a registry of legacy Portuguese names and exist to be quoted.
     New rows will keep adding Portuguese in the left-hand column, so the region is
     exempted as a whole rather than marking each row. This is the one file where that
     is correct; everywhere else, mark the single line. -->
<!-- check-language: off -->

## Files

| from | to |
|---|---|
| `src/gpu/fisica.rs` / `.wgsl` | `src/gpu/physics.rs` / `.wgsl` |
| `src/gpu/cerebro.rs` / `.wgsl` | `src/gpu/brain.rs` / `.wgsl` |
| `src/gpu/percepcao.rs` / `.wgsl` | `src/gpu/perception.rs` / `.wgsl` |
| `src/gpu/nucleo.rs` | `src/gpu/core.rs` |
| `src/bin/gpu_valida.rs` | `src/bin/gpu_validate.rs` |

Modules in `src/gpu/mod.rs`: `pub mod physics; pub mod brain; pub mod perception; pub mod core;`

## CLI flags

**Decided 2026-09-12: every flag of `gpu_evolve` has an English canonical name, and
every Portuguese spelling keeps working as a deprecated alias.** The table that owns
them is `src/bin/gpu_evolve/cli_flags.rs`, not this file: 112 rows of canonical name,
aliases, whether the flag takes a value, and one help line. Nothing is ever deleted
from `aliases`, so a battery request written months ago is still a valid input, and it
prints one deprecation warning per alias per run.

A rename is therefore: change `name`, move the old spelling into `aliases`. The tests
in `cli_flags_tests.rs` read `main.rs` with `include_str!` and fail if the table and
the parser disagree in either direction — which is what makes this safe to do at all.
The same file carries `VALUE_ALIASES` for a flag whose VALUES are an enum (today only
`--tau-bank`: `livre`/`log-fixo`/`espectral` -> `free`/`log-fixed`/`spectral`).

`cortex_evolve`, `gpu_loop` and `gpu_report` still parse `--seres` with their own
loops and no registry; each needs its own pass before that flag can be renamed.

## Types

| from | to |
|---|---|
| `FisicaGpu` | `PhysicsGpu` |
| `EstadoFisicoGpu` | `PhysicsStateGpu` |
| `CerebroGpu` / `CerebroGpuRes` | `BrainGpu` / `BrainGpuRes` |
| `CerebroParams` / `CerebroInput` | `BrainParams` / `BrainInput` |
| `PercepcaoGpu` | `PerceptionGpu` |
| `NucleoGpu` | `CoreGpu` |
| `IntencaoGpu` | `IntentGpu` |
| `Postura` | `Pose` |
| `PoseGpu` (private in `mod.rs`) | `PoseRaw` — renamed so it does not collide with `Pose` |
| `RetinaGpu`, `RetinaGpuRes`, `Pixel`, `Params` | already English, unchanged |

## Methods and functions

| from | to |
|---|---|
| `novo` | `new` |
| `obter` | `get_or_create` |
| `passo` / `passo_raw` | `step` / `step_raw` |
| `forward` | `forward` (unchanged) |
| `render_legado` | `render_legacy` |
| `codificacao_legada` | `legacy_encoding` |
| `assinatura_do_mundo` | `world_signature` |
| `entrada_uniforme` / `entrada_storage` | `uniform_entry` / `storage_entry` |
| `serializar_entrada` | `serialize_input` |
| `carregar_pesos` | `upload_weights` |
| `baixar_estado_cfc` | `download_cfc_state` |
| `reduzir` | `reduce` |
| `bind_e_dispatch` | `bind_and_dispatch` |
| `buf_saida_raw` | `output_buffer` |
| `validar_cerebro` / `validar_fisica` / `validar_percepcao` | `validate_brain` / `validate_physics` / `validate_perception` |
| `fisica_passo_cpu` / `percepcao_cpu` | `physics_step_cpu` / `perception_cpu` |
| `imprimir_cabecalho_tabela` / `imprimir_linha_tabela` | `print_table_header` / `print_table_row` |

## Fields and variables

| from | to |
|---|---|
| `fila` | `queue` |
| `adaptador` | `adapter` |
| `capacidade` | `capacity` |
| `dispositivo` | `device` |
| `assinatura` | `signature` |
| `pesos` | `weights` |
| `seres` / `n_seres` | `beings` / `n_beings` |
| `saida` / `entrada` | `output` / `input` |
| `leitura` | `readback` |
| `grade` | `grid` |
| `mundo` | `world` |
| `intencao` | `intent` |
| `corpo` | `body` |
| `posturas` | `poses` |
| `ser` (index in WGSL) | `being` |
| `pitch_da_linha` | `row_pitch` |
| `voxel_em` | `voxel_at` |
| `ALTURA_OLHO` (WGSL) | `EYE_HEIGHT` |

Unchanged: `material`, `dist_m`, `params`, `grid_size`, `cell_size`, `fov_half`,
`pitches`, `hit_dist`, `hit_voxel`, `ray_dir`, `side_dist_*`.

## Still to be named

These Portuguese names are still in the tree (verified 2026-08-17). The right-hand
column is a **proposal, not a performed rename** — it exists so that two contributors
migrating two files do not invent two different English words for the same concept.
When you carry one out, move the row up into the tables above.

| from | to |
|---|---|
| `escada` (difficulty ladder) | `ladder` |
| `fantasma` (reference lap) | `ghost` |
| `pista` / `pistas` | `track` / `tracks` |
| `piloto_linear` / `piloto_forja` | `linear_pilot` / `forge_pilot` |
| `genoma` | `genome` |
| `regioes` | `regions` |
| `tanque_por_setor` | `tank_per_sector` |
| `frente_do_waypoint` | `waypoint_forward` |
| `ablacao` | `ablation` |
| `veredito` | `verdict` |
| `degrau` | `step` / `rung` |
| `coesao` | `cohesion` |
Performed on 2026-09-12, so do not propose them again: every `gpu_evolve` flag (see
the CLI flags section above, including `--pista-held-out` -> `--track-held-out` and
`--held-out-cada` -> `--held-out-every`), `tau_bank::banco_tau` -> `tau_bank`,  <!-- check-language: allow -->
`src/bin/sensibilidade.rs` -> `src/bin/sensitivity.rs` (binary `sensitivity`),
`tools/voltas.py` -> `tools/laps.py`, and the `LapRecord` wire key `parcial` ->
`partial`.

<!-- check-language: on -->

*Language policy updated 2026-08-17. The rename tables above predate it and are
unchanged; only the header rule was replaced.*
