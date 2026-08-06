# ARQUITETURA.md — Universe (Rust + Bevy)

> Última atualização: 2026-08-04
> Autor: auditoria de duplicação + rede de proteção

---

## 1. Mapa real de módulos

```
src/
├── main.rs              — entry point Bevy; inicializa plugins e AppState
├── lib.rs               — raiz da lib crate (exposta por todos os bins)
├── config.rs            — HUB DE CONSTANTES (fonte de verdade de parâmetros)
│                          Submódulo iara::genetics contém DNA_FAST_SIZE,
│                          DNA_SLOW_SIZE, SCAR_*, MORPHO_*, HYPERNET_* vivos.
├── types.rs             — tipos compartilhados (Vec2, etc.)
│
├── evolution/           — lógica evolutiva
│   ├── mod.rs           — re-exporta de dna_genesis (fonte de verdade)
│   ├── dna.rs           — LEGADO — apenas para leitura de arquivos V1/V2/V3
│   ├── dna_genesis.rs   — FONTE DE VERDADE DO DNA (salva V8 TLV, lê V3/V2/V1)
│   ├── pipeline.rs      — loop de evolução por geração
│   ├── spawn.rs         — instancia racers a partir do pool de elite
│   ├── settings.rs      — RaceSettings, DevSettings
│   ├── telemetry.rs     — log de geração
│   └── logs.rs          — abertura de relatório
│
├── simulation/          — loop de simulação por tick
│   ├── motor.rs         — tick principal (sensory_wave, forward, physics)
│   └── racer.rs         — componente Racer com EliteDna
│
├── gpu/                 — compute shaders wgpu (5 kernels validados)
│   ├── mod.rs           — GPU context (outros agentes — NÃO EDITAR)
│   ├── brain.rs         — forward cortex na GPU (NÃO EDITAR)
│   ├── core.rs          — helpers de bgl, buffers
│   ├── cortex_sheet.rs  — aprendizado Hebb na GPU
│   ├── evolution.rs     — crossover/mutação na GPU
│   ├── perception.rs    — raycast retina na GPU
│   └── physics.rs       — física na GPU
│
├── layers.rs            — BitLinear (fast brain, f32 runtime)
├── float_layers.rs      — FloatLinear (specialist/slow brain, f32 runtime)
├── dense_layer.rs       — DenseLayer (slow brain, f32 com bias)
├── specialist.rs        — SpecialistBrain (3 FloatLinear coordenados)
├── cfc.rs               — CfCCell (slow brain, liquid neural)
├── kan.rs               — KAN output head (spline knots)
├── neuro.rs             — ablação de priming, utilidades neurais
├── perception.rs        — lógica de percepção CPU
├── visao.rs             — retina CPU / pooling
├── world.rs             — geração de pista e voxel world
├── physics.rs           — física CPU (veículo)
├── safety/              — Sentinel FCA (freio de emergência)
└── bin/
    ├── impact.rs        — NOVO: analisa impacto de um símbolo no grafo
    ├── gpu_validate.rs  — valida 5 kernels GPU vs CPU (NÃO EDITAR)
    ├── gpu_loop.rs      — loop GPU headless (NÃO EDITAR)
    ├── gpu_report.rs    — relatório de GPU (NÃO EDITAR)
    ├── dna_console.rs   — inspeção/edição interativa de DNA
    ├── dump_dna.rs      — dump de pesos de um arquivo .dna
    └── ...
```

---

## 2. Fonte de verdade por conceito

| Conceito | Fonte de Verdade | Notas |
|---|---|---|
| **Formato de arquivo DNA** | `src/evolution/dna_genesis.rs` | V8 TLV (iara_genesis). Lê V3/V2/V1 legacy. |
| **Constantes de arquitetura** | `src/config.rs` → `iara::genetics` | `DNA_FAST_SIZE`, `DNA_SLOW_SIZE`, `MORPHO_*`, `HYPERNET_*`, `SCAR_*` |
| **Camada fast brain** | `src/layers.rs` → `BitLinear` | F32 runtime; quantize_for_save() → i8 ao salvar |
| **Camada specialist** | `src/float_layers.rs` → `FloatLinear` | F32 sem bias, input i8 |
| **Camada slow brain (dense)** | `src/dense_layer.rs` → `DenseLayer` | F32 com bias |
| **Retina (visão)** | `src/visao.rs` + `src/config.rs::iara::vision` | Constantes de pitch/fov em config.rs |
| **Loop de evolução** | `src/evolution/pipeline.rs` | Chama `dna_genesis::save_champion_dna_v7` |
| **GPU kernels (5 validados)** | `src/gpu/{brain,physics,perception,cortex_sheet,mod}.rs` | Não editar sem rodar `gpu_validate` antes e depois |

---

## 3. Duplicações auditadas

### 3.1 `dna.rs` vs `dna_genesis.rs` — DIVERGENTES (bug latente resolvido)

**Veredito: DIVERGENTES — duas implementações INCOMPATÍVEIS do serializer.**

| Aspecto | `dna.rs` (LEGADO) | `dna_genesis.rs` (ATIVO) |
|---|---|---|
| `save_champion_dna_full()` | Escreve formato V3 flat (magic+bytes raw) | Delega ao V5→V7 TLV via `compress_organism()` |
| Formato de saída | V3 flat (8478-8755 bytes, magic `IARA`, version=3) | V8 TLV (tamanho variável por entropia) |
| `EliteDna.fast_weights` | `Vec<i8>` | `Vec<f32>` (f32 runtime; quantize ao salvar) |
| `EliteDna` campos extras | — | `cortex_l2`, `cortex_l3`, `car_model_idx`, `kan_knots`, `perception_weights` |
| `DnaPayload` campos extras | — | `kan_knots`, `trinity`, `cortex_dct`, `cortex_raw`, `cortex_l2_raw`, `cortex_l3_raw`, `slow_f32` |
| `DNA_FAST_SIZE` | 8192 (128×64, hardcoded) | importa `DNA_FAST_SIZE_LEGACY` de config.rs |
| `DNA_SLOW_SIZE` | 256 (32×8, hardcoded) | importa `DNA_SLOW_SIZE_LEGACY` de config.rs |
| `DNA_VERSION` | 3 | 8 (via config.rs) |
| Caller em produção | NINGUÉM (exceto `texto_dna.rs` → apenas `DNA_MAGIC`) | `pipeline.rs`, `spawn.rs`, `gpu_loop.rs`, etc. |

**Diff crítico — o que muda entre `save_champion_dna_full` nos dois módulos:**
```
dna.rs::save_champion_dna_full():
  → escreve magic + version=3 + header flat + fast_weights_raw + slow_weights_raw
    + morpho_tail + hypernet_tail
  → arquivo V3, 8755 bytes

dna_genesis.rs::save_champion_dna_full():
  → delega para save_champion_dna_v5() → save_champion_dna_v7()
  → chama compress_organism() (iara_genesis TLV)
  → adiciona chunks META + CRAW + CL2W + SLWF + CEQU
  → arquivo V8 TLV, tamanho variável
```

**Consequência se alguém chamar o save do módulo errado:**
- Arquivo V3 (dna.rs) seria carregado por `load_champion_dna` do `dna_genesis.rs`
  pelo path "legacy V3" — **funciona**, dados básicos são preservados.
- Arquivo V8 TLV (dna_genesis.rs) carregado por `dna.rs::load_champion_dna`
  — **falha silenciosamente** ou lê garbage (dna.rs não conhece TLV).

**Ação tomada:** 4 testes de equivalência cross-módulo adicionados em
`dna.rs::tests` (ver seção 5). NÃO remova `dna.rs` sem verificar que o
legacy path de V1/V2/V3 ainda existe em `dna_genesis.rs::load_champion_dna`
(já está — blocos "v3 with magic, version != 4" e "legacy v1/v2").

**Por que NÃO unificar agora:**
`dna.rs` ainda é necessário para manter tests do formato V3 flat rodando
e como referência de como o formato antigo funcionava. Sua `EliteDna` com
`Vec<i8>` é estruturalmente diferente da `Vec<f32>` em dna_genesis — remover
exigiria adaptar todos os callers de EliteDna (pipeline, spawn) de uma vez.

---

### 3.2 `quantize` e `myelinate` em `layers.rs` vs `specialist.rs`

**Veredito: COMPLEMENTARES — não são duplicação real.**

- `layers.rs::BitLinear::quantize()` copia latent f32 nos weights; também
  aplica `ablar_priming()`. É o comportamento real do fast brain.
- `specialist.rs::SpecialistBrain::quantize()` é **no-op** com comentário
  explícito: "quantize happens at DNA save boundary". Os FloatLinear dos
  especialistas não passam por `quantize()` no runtime.
- `layers.rs::BitLinear::myelinate()` aplica boost de scale (legacy log).
- `specialist.rs::SpecialistBrain::myelinate()` aplica scale diretamente nos
  weights f32 (semanticamente diferente: multiplica os pesos, não só log).

**Ação:** nenhuma. São comportamentos distintos sob a mesma interface `BrainForward`.

---

### 3.3 `forward`, `set_weights`, `mutate` em `dense_layer.rs` vs `float_layers.rs`

**Veredito: COMPLEMENTARES — arquiteturas diferentes, não duplicação.**

| Aspecto | `DenseLayer` | `FloatLinear` |
|---|---|---|
| Input | `&[f32]` (normalizado) | `&[i8]` (sensory_wave) |
| Bias | Sim (learnable) | Não |
| `mutate()` | rate + sigma + clamp configuráveis | rate fixo, sigma=0.02, clamp=±2.0 |
| Uso | SlowBrain dense layers | SpecialistBrain (3 experts 32→1) |

**Ação:** nenhuma. Remover um quebraria o outro caso de uso.

---

### 3.4 `uniform_entry` e `storage_entry` em 5 arquivos de `src/gpu/`

**Veredito: IDÊNTICAS — boilerplate wgpu puro.**

Os 5 arquivos (`core.rs`, `brain.rs`, `cortex_sheet.rs`, `mod.rs`, `physics.rs`,
`perception.rs`) têm cópias byte-a-byte da mesma função helper de 10-15 linhas.

**Por que NÃO foram unificadas:**
1. `brain.rs`, `mod.rs`, `gpu_loop.rs`, `gpu_report.rs` são gerenciados por
   **outro agente** — não devem ser editados nesta sessão.
2. `core.rs` já documenta: "Copiados de mod.rs para não criar dependência circular."
3. O risco de criar um `gpu_helpers.rs` compartilhado e quebrar o build dos
   outros agentes supera o benefício de remover 50 linhas repetidas.

**Ação documentada:** estas cópias são intencionais enquanto dois agentes
trabalham em paralelo nos arquivos GPU. Quando o trabalho paralelo terminar,
criar `src/gpu/helpers.rs` com as duas funções e importar em todos os arquivos.

---

## 4. O que não tem teste e é crítico

| O que | Risco | Status |
|---|---|---|
| Salvo por `dna.rs`, carregado por `dna_genesis.rs` | Corrupção silenciosa | ✅ Coberto (teste `dna_legacy_writer_readable_by_genesis_loader`) |
| Constantes de tamanho DNA iguais nos dois módulos | Parser V3 lê bytes errados | ✅ Coberto (teste `dna_size_constants_match_genesis_legacy`) |
| `MorphoGenes` bytes iguais nos dois módulos | Genes corporais corrompidos | ✅ Coberto (teste `morpho_bytes_identical_across_modules`) |
| `HypernetSeed` bytes iguais nos dois módulos | Hypernetwork corrompido | ✅ Coberto (teste `hypernet_bytes_identical_across_modules`) |
| 5 kernels GPU vs CPU | Divergência silenciosa | ✅ `gpu_validate` (pré-existente) |
| `cfc.rs::state_decays_to_zero` | Falha pré-existente (decay muito lento) | ⚠️ Falha pré-existente |
| `layers.rs::save_load_roundtrip` | Falha pré-existente (precisão i8 1/63.5) | ⚠️ Falha pré-existente |
| `dna_genesis::legacy_v2/v3 loads` | Falha pré-existente (V8 changes sizes) | ⚠️ Falha pré-existente |
| `sentinel::tcs_cuts_throttle_on_slip` | Falha pré-existente | ⚠️ Falha pré-existente |

---

## 5. Ferramenta de análise de impacto

```bash
# Se alterar save_champion_dna, o que muda?
./target/release/impact save_champion_dna

# Se alterar MorphoGenes, quem é afetado? (profundidade 4)
./target/release/impact MorphoGenes --depth 4

# Se alterar DNA_FAST_SIZE, quem quebra?
./target/release/impact DNA_FAST_SIZE --depth 3

# Banco diferente do padrão:
./target/release/impact BitLinear --graph /outro/caminho/graph.db
```

**Aviso:** o grafo é gerado estaticamente. Nomes antigos como `cerebro.rs` podem
aparecer (hoje é `brain.rs`). Use grep para confirmar:
```bash
grep -rn 'MorphoGenes' src/ --include='*.rs'
```

---

## 6. Regras de ouro para não quebrar nada

1. **Sempre rodar `gpu_validate` antes e depois** de qualquer alteração em
   `src/gpu/`, `src/layers.rs` ou `src/evolution/`.

2. **Nunca chamar `dna.rs::save_champion_dna_full`** em código novo — use sempre
   `dna_genesis::save_champion_dna_v7` (ou os wrappers acima dele).

3. **Constantes de tamanho DNA mudam juntas:** se mudar `DNA_FAST_SIZE` em
   `config.rs::iara::genetics`, os testes `dna_size_constants_match_genesis_legacy`
   em `dna.rs` vão travar imediatamente.

4. **`MorphoGenes` e `HypernetSeed` têm cópias em `dna.rs` e `dna_genesis.rs`.**
   Os testes `morpho_bytes_identical` e `hypernet_bytes_identical` verificam
   que os bytes são idênticos. Se as structs divergirem, os testes falham.

5. **`gauss()` helper** existe duplicado dentro de `MorphoGenes::mutate` e
   `HypernetSeed::mutate` em ambos os arquivos. É identicamente implementado
   (Box-Muller). Não é fonte de bug porque cada módulo tem a sua cópia
   inline — mas se a lógica precisar mudar, mudar nos dois lugares.

6. **Testes pré-existentes que falham** (não regredir):
   - `evolution::dna_genesis::tests::legacy_v3_without_morpho_loads_with_default`
   - `evolution::dna_genesis::tests::legacy_v2_loads_with_zero_metadata`
   - `layers::tests::save_load_roundtrip`
   - `safety::sentinel::tests::multiple_soft_flags_compose_and_clamp`
   - `safety::sentinel::tests::tcs_cuts_throttle_on_slip`
   - `cfc::tests::state_decays_to_zero_without_input`
