# Tabela de nomes — contrato entre agentes

**Regra do usuário:** todo IDENTIFICADOR em INGLÊS. **Comentários e strings de saída seguem em
PORTUGUÊS** — não traduza comentário nenhum.

Vários agentes trabalham em paralelo, cada um num arquivo. Esta tabela é o que faz os nomes
baterem entre arquivos sem que um agente veja o outro.

## Arquivos

| de | para |
|---|---|
| `src/gpu/fisica.rs` / `.wgsl` | `src/gpu/physics.rs` / `.wgsl` |
| `src/gpu/cerebro.rs` / `.wgsl` | `src/gpu/brain.rs` / `.wgsl` |
| `src/gpu/percepcao.rs` / `.wgsl` | `src/gpu/perception.rs` / `.wgsl` |
| `src/gpu/nucleo.rs` | `src/gpu/core.rs` |
| `src/bin/gpu_valida.rs` | `src/bin/gpu_validate.rs` |

Módulos em `src/gpu/mod.rs`: `pub mod physics; pub mod brain; pub mod perception; pub mod core;`

## Tipos

| de | para |
|---|---|
| `FisicaGpu` | `PhysicsGpu` |
| `EstadoFisicoGpu` | `PhysicsStateGpu` |
| `CerebroGpu` / `CerebroGpuRes` | `BrainGpu` / `BrainGpuRes` |
| `CerebroParams` / `CerebroInput` | `BrainParams` / `BrainInput` |
| `PercepcaoGpu` | `PerceptionGpu` |
| `NucleoGpu` | `CoreGpu` |
| `IntencaoGpu` | `IntentGpu` |
| `Postura` | `Pose` |
| `PoseGpu` (privado em mod.rs) | `PoseRaw` — renomeado para não colidir com `Pose` |
| `RetinaGpu`, `RetinaGpuRes`, `Pixel`, `Params` | já em inglês, mantém |

## Métodos e funções

| de | para |
|---|---|
| `novo` | `new` |
| `obter` | `get_or_create` |
| `passo` / `passo_raw` | `step` / `step_raw` |
| `forward` | `forward` (mantém) |
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

## Campos e variáveis

| de | para |
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
| `ser` (índice em WGSL) | `being` |
| `pitch_da_linha` | `row_pitch` |
| `voxel_em` | `voxel_at` |
| `ALTURA_OLHO` (WGSL) | `EYE_HEIGHT` |

Mantêm o nome: `material`, `dist_m`, `params`, `grid_size`, `cell_size`, `fov_half`,
`pitches`, `hit_dist`, `hit_voxel`, `ray_dir`, `side_dist_*`.

## Regra de ouro

Se um nome não está nesta tabela e está em português, traduza para o inglês óbvio e siga.
Melhor um nome inglês razoável que um português correto.
