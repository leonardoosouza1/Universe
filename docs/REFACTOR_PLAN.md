# REFACTOR_PLAN — Universe / Rust + WGSL

> Gerado em 2026-08-13. Baseado em mapa estático de 5 frentes (call-graph, WGSL-host,
> anatomia de gpu_evolve, triagem de bins, wiring do crate). Nenhum arquivo foi alterado
> para produzir este documento.
>
> **Regra de ouro deste plano:** mover arquivo = atualizar TODOS os callers identificados
> no call-graph antes de fechar o estágio. Um estágio só fecha compilando (`cargo build
> --release` sem erro).

---

## 1. Árvore alvo dos módulos

```
src/
├── math/               ← NOVO — funções puras sem deps de Bevy/wgpu
│   ├── mod.rs
│   ├── activations.rs  ← sigmoid, tanh, relu, leaky_relu
│   ├── stats.rs        ← mean, variance, percentis, histograma
│   ├── quant.rs        ← i8 quantização (SNR, round-to-ternary)
│   ├── linalg.rs       ← participation_ratio, SVD simples, raio espectral, rank efetivo
│   └── laws.rs         ← garimpo (search_noise_threshold), metabolismo (AIC/RIC)
│
├── brain/              ← NOVO — tudo sobre o motor cognitivo
│   ├── mod.rs          ← reexporta SovereignEngine, SlowBrain, CfCCell, etc.
│   ├── engine.rs       ← move src/engine.rs inteiro (SovereignEngine, SlowBrain)
│   ├── cfc.rs          ← move src/cfc.rs (CfCCell, TAU_MIN_S, TAU_MAX_S, DECAY_DEFAULT)
│   ├── layers.rs       ← move src/layers.rs (BitLinear, BrainForward)
│   ├── float_layers.rs ← move src/float_layers.rs (FloatLinear)
│   ├── dense_layer.rs  ← move src/dense_layer.rs (DenseLayer)
│   ├── kan.rs          ← move src/kan.rs (SensoryPreFilter, KanOutputHead, KAN_OUT_KNOTS)
│   ├── gate.rs         ← move src/gate.rs (GateBrain)
│   ├── specialist.rs   ← move src/specialist.rs (SpecialistBrain)
│   ├── resonance.rs    ← move src/resonance.rs (ResonantNode, coherence)
│   ├── chaos.rs        ← move src/chaos.rs (ChaosInput, ChaosScore)
│   └── capacity.rs     ← move src/brain_capacity.rs (participation_ratio, growth_verdict)
│
├── perception/         ← NOVO — sensor e percepcao temporal
│   ├── mod.rs          ← reexporta RetinaPerception, CHANNELS (enum)
│   ├── retina.rs       ← move src/perception.rs (RetinaPerception, RetinaSensor)
│   ├── metabolism.rs   ← move src/metabolism.rs (Metabolism, EpisodicTrace)
│   └── channels.rs     ← NOVO — enum Channel com 192 entradas (ver §1.1)
│
├── physics/            ← EXISTE (src/physics.rs) — não move
├── world.rs            ← EXISTE — não move (30+ callers)
├── config.rs           ← EXISTE — não move (12+ callers)
│
├── evolution/          ← EXISTE — não move estrutura
├── simulation/         ← EXISTE — não move estrutura
│
├── analysis/           ← NOVO — métricas de ativação (E4)
│   ├── mod.rs
│   └── activation.rs   ← ActivationStats, ActivationAlarm, collect_activation_stats
│
├── world/              ← NÃO criar agora (VoxelWorld tem 30+ callers, risco máximo)
│
├── lib.rs              ← atualiza declarações de mod após cada estágio
└── bin/
    ├── gpu_evolve.rs   ← extrai seções em E3 (ver §2.3)
    └── ...             ← demais bins não se movem
```

### 1.1 Registro dos 192 canais — `src/perception/channels.rs`

Substitui todos os números mágicos de canal. Cada variante carrega nome, kernel WGSL
de origem, fórmula/semântica e faixa esperada em `[i8]`.

```rust
/// Registro canônico dos 192 canais da onda sensorial.
/// Fonte de verdade única: elimina literais 64, 75, 85, 102, 113, 123, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    // ── Retina espacial (perception.wgsl ch 0-97, 128-191) ──────────────
    // ch 0..97:  colunas retina binocular (RETINA_COLS=126 → grupos de 14 linhas)
    // ch 128..191: reserva espacial estendida
    RetinaRow { row: u8, col_group: u8 },   // formula: packed i8 de distância mm

    // ── Canal temporal (perception_temporal.wgsl) ────────────────────────
    VelFwd           = 64,  // fwd velocity [−127, 127] = [−37 m/s, +37 m/s]
    VelLat           = 65,  // lat velocity
    Yaw              = 66,  // yaw rate [rad/s × escala]
    DistL            = 67,  // dist parede esquerda [0, 127]
    DistR            = 68,  // dist parede direita  [0, 127]
    DistF            = 69,  // dist parede frontal  [0, 127]
    ProgSector       = 70,  // progresso no setor [0, 127]
    SteerMem         = 71,  // memória de volante tick anterior
    ThrMem           = 72,  // memória de acelerador
    BrkMem           = 73,  // memória de freio
    SteerRate        = 74,  // derivada do volante
    EdgeDensL        = 75,  // densidade de borda linha 6
    // ch 76..84: reserva temporal A
    EdgeDensR        = 85,  // densidade de borda linha 7
    Dopamina         = 86,  // sinal dopaminérgico (joy_streak normalizado)
    // ch 87: reserva temporal B
    VidaFrac         = 88,  // life_left / LIFE_CAP_TICKS [0, 127]
    TempMotor        = 89,  // temperatura do motor simulada [0, 127]
    // ch 90..101: reserva temporal C
    SlipSignal       = 102, // sinal de deslizamento (cross product vel/fwd)
    Spinning         = 103, // flag de spin (|slip| > limiar)
    Stress           = 104, // stress cinético
    SteerSensor      = 105, // piloto sensorial: steer sugerido
    ThrSensor        = 106, // piloto sensorial: throttle sugerido
    OnGrass          = 107, // flag grama
    // ch 108..112: reserva temporal D
    RaioEsq          = 113, // fan-ray dist esquerda
    RaioDir          = 114, // fan-ray dist direita
    RaioFwd          = 115, // fan-ray dist frontal
    // ch 116..117: reserva temporal E
    CfcPerceptOut0   = 118, // saída CfC percepcão temporal [0]
    CfcPerceptOut1   = 119, // saída CfC percepcão temporal [1]
    CfcPerceptOut2   = 120, // saída CfC percepcão temporal [2]
    // ch 121..122: reserva temporal F
    MemCfcOut        = 123, // saída memory0 do CfC de percepcão

    // ── Canais de motion (perception_motion.wgsl ch 21-27, 83-84, 98) ───
    MotionDelta0     = 21,
    MotionDelta1     = 22,
    MotionDelta2     = 23,
    MotionDelta3     = 24,
    MotionDelta4     = 25,
    MotionDelta5     = 26,
    MotionDelta6     = 27,
    MotionMag0       = 83,  // magnitude motion linha 6
    MotionMag1       = 84,  // magnitude motion linha 7
    MotionAcc        = 98,  // aceleração de flow ótico
}

impl Channel {
    /// Índice numérico do canal na onda [0, 191].
    pub const fn index(self) -> usize { self as usize }

    /// Kernel WGSL que escreve este canal.
    pub const fn source(self) -> &'static str { ... }

    /// Faixa esperada em i8: (min, max).
    pub const fn range_i8(self) -> (i8, i8) { ... }
}

/// Total de canais — deve bater com config::iara::brain::SENSORY_WAVE_SIZE.
pub const N_CHANNELS: usize = 192;
```

**O que move de onde para onde:**

| Item | Origem | Destino | Callers a atualizar |
|---|---|---|---|
| `canal_tem_fonte()` | `gpu_evolve.rs:2504` | `perception/channels.rs` | `gpu_evolve.rs:2650` (auditoria_relatorio) |
| números mágicos de canal | `gpu_evolve.rs:tick()` (>30 ocorrências) | substituídos por `Channel::VelFwd.index()` etc. | apenas gpu_evolve.rs internamente |

**O que move de onde para onde — módulo `brain/`:**

| Arquivo de origem | Arquivo de destino | Callers a atualizar (do call-graph) |
|---|---|---|
| `src/engine.rs` | `src/brain/engine.rs` | `src/gpu/brain.rs:31/95/341`, `src/gpu/evolution.rs:29/122/138`, `src/gpu/core.rs:368`, `src/evolution/spawn.rs:362`, `src/bin/gpu_loop.rs:61/313/429/520/1421`, `src/bin/gpu_validate.rs:358/424/444/686`, `src/bin/gpu_evolve.rs:53` |
| `src/cfc.rs` | `src/brain/cfc.rs` | `src/gpu/evolution.rs:58/105/111/129/287/288`, `src/gpu/brain.rs:128/274/277/390`, `src/bin/gpu_evolve.rs:1377/2831`, `src/engine.rs:6/40/75/182/267`, `src/bin/gpu_validate.rs:490/503/742/744` |
| `src/layers.rs` | `src/brain/layers.rs` | `src/engine.rs:2/177/247`, `src/specialist.rs:16` |
| `src/float_layers.rs` | `src/brain/float_layers.rs` | `src/engine.rs:178/179/252/259`, `src/specialist.rs:17/82`, `src/memory.rs:9/76/146` |
| `src/dense_layer.rs` | `src/brain/dense_layer.rs` | `src/engine.rs:3/41/42/56/65` |
| `src/kan.rs` | `src/brain/kan.rs` | `src/engine.rs:7/186/268`, `src/gpu/brain.rs:345`, `src/evolution/spawn.rs:917/945`, `src/gpu/evolution.rs:28/34/46`, `src/bin/gpu_loop.rs:292/315/1169`, `src/bin/gpu_evolve.rs:74` |
| `src/gate.rs` | `src/brain/gate.rs` | `src/engine.rs:214/277/331` |
| `src/specialist.rs` | `src/brain/specialist.rs` | `src/engine.rs:215/278`, `src/evolution/spawn.rs:510`, `src/simulation/motor.rs:60` |
| `src/resonance.rs` | `src/brain/resonance.rs` | `src/engine.rs:4/181/221`, `src/chaos.rs:4/30/52/60/67/79` |
| `src/chaos.rs` | `src/brain/chaos.rs` | `src/engine.rs:301/305`, `src/simulation/motor.rs:1903` |
| `src/brain_capacity.rs` | `src/brain/capacity.rs` | `src/bin/gpu_evolve.rs:2340` |

**Reexportar em `src/brain/mod.rs`** todos os itens que os callers externos esperam:
```rust
pub use engine::{SovereignEngine, SlowBrain, set_global_champion_snapshot};
pub use cfc::{CfCCell, TAU_MIN_S, TAU_MAX_S, DECAY_DEFAULT, DC_GAIN_DEFAULT, log_uniform_decays};
pub use layers::{BitLinear, BrainForward};
pub use float_layers::FloatLinear;
pub use dense_layer::DenseLayer;
pub use kan::{SensoryPreFilter, KanOutputHead, KAN_OUT_KNOTS, KAN_OUT_DOMAIN};
pub use gate::{GateBrain, GATE_INPUTS, GATE_OUTPUTS};
pub use specialist::SpecialistBrain;
pub use resonance::{ResonantNode, coherence};
pub use chaos::{ChaosInput, ChaosScore, score};
pub use capacity::{participation_ratio, saturation, search_noise_threshold, growth_verdict, Verdict};
```

Adicionar em `src/lib.rs`:
```rust
pub mod brain;
pub mod math;
pub mod perception;  // renomeia o mod atual + adiciona channels.rs
pub mod analysis;
```

Manter aliases de compatibilidade nos módulos originais enquanto os callers migram (E1 e E2):
```rust
// src/engine.rs — temporariamente:
pub use crate::brain::engine::*;
```

**O que NÃO move neste plano:**
- `src/world.rs` — 30+ callers, risco máximo; permanece onde está
- `src/config.rs` — 12+ callers de caminho `crate::config::iara::brain::*` hardcoded nos shaders
- `src/simulation/` — pipeline ECS inteiro, dependência de Bevy; fora do escopo
- `src/physics.rs` — componentes ECS; fora do escopo
- `.wgsl` — nenhum shader movido nos estágios E1-E3
- `motor.rs` (3382 linhas) — não tocar nesta rodada

**Código candidato a morto** (confirmar antes de remover; não remove nesta rodada):
- `src/reality.rs` (RealityEngine — zero callers externos confirmados)
- `src/repominer.rs` (RepoMiner — zero callers externos confirmados)
- `src/mesh.rs` (TernaryMesh — zero callers externos confirmados)
- `src/substrate.rs` (traits sem implementações concretas)
- `src/memory.rs::MultiverseSpiral`, `consolidate_episodic`, `dream_rollout` — sem callers no loop de produção
- `src/visualization.rs::VisionVisualizationPlugin` — não registrado no ECS

---

## 2. Estágios ordenados da refatoração

Cada estágio fecha compilando com `cargo build --release 2>&1 | grep -c error` == 0.
Antes de começar qualquer estágio: `cargo test --release --bin gpu_evolve` deve passar.

### E1 — Math core + testes unitários (pré-condição dos demais)

**Objetivo:** extrair funções puramente matemáticas sem deps de Bevy/wgpu/serde para
`src/math/`. Esses módulos não dependem de nenhum outro módulo do crate.

**Arquivos criados:**

`src/math/mod.rs`:
```rust
pub mod activations;
pub mod stats;
pub mod quant;
pub mod linalg;
pub mod laws;
```

`src/math/activations.rs` — extrai de onde residem hoje:
- `sigmoid(x: f32)` — usada em `src/cfc.rs:70`, `src/gpu/brain.wgsl` (no host, não no shader)
- `tanh_approx(x: f32)` — usada em `src/cfc.rs`, `src/engine.rs`
- Fonte hoje: inline em `cfc.rs` e `engine.rs` — extrair como `pub fn`

`src/math/stats.rs` — funções novas (usadas por E4):
```rust
pub fn mean(xs: &[f32]) -> f32;
pub fn variance(xs: &[f32]) -> f32;
pub fn percentile(xs: &mut [f32], p: f32) -> f32;  // sort in-place
pub fn histogram(xs: &[f32], n_bins: usize) -> Vec<u32>;
```

`src/math/quant.rs` — extrai de `src/layers.rs`:
```rust
/// SNR de quantização i8: ratio entre sinal f32 e erro de arredondamento
pub fn snr_i8(weights: &[f32], scale: f32) -> f32;
/// Round-to-ternary com threshold
pub fn to_ternary(x: f32, threshold: f32) -> i8;
```
- Origem: `src/layers.rs:40-80` (lógica de quantização de BitLinear)
- Callers após extração: `src/brain/layers.rs` (E2)

`src/math/linalg.rs` — move de `src/brain_capacity.rs`:
```rust
pub fn participation_ratio(activations: &[Vec<f32>]) -> Option<f32>;
// jacobi_eigenvalues fica privada dentro de linalg.rs (é fn privada hoje)
pub fn spectral_radius(w_rec: &[f32], n: usize) -> f32;  // NOVO — raio espectral de W_rec
pub fn effective_rank(singular_vals: &[f32], threshold: f32) -> usize;  // NOVO
```
- Origem de `participation_ratio`: `src/brain_capacity.rs:114`
- Caller atual: `src/bin/gpu_evolve.rs:2340` → atualizar para `universe::math::linalg::participation_ratio`

`src/math/laws.rs` — move de `src/brain_capacity.rs`:
```rust
pub fn search_noise_threshold(sigma: f32, m_alternatives: usize) -> f32;
// Lei do Garimpo: σ√(2 ln M)
pub fn aic_cost(residual_variance: f32, n_params: usize) -> f32;
pub fn ric_penalty(sigma: f32, m_alternatives: usize) -> f32;
```
- Origem de `search_noise_threshold`: `src/brain_capacity.rs:183`
- `saturation`, `growth_verdict`, `Verdict` ficam em `src/brain/capacity.rs` (E2) que usa `math::linalg` e `math::laws`

**Testes unitários** — adicionar `#[cfg(test)]` em cada arquivo de `math/`:
```rust
#[test]
fn mean_de_vetor_uniforme() { assert_eq!(mean(&[2.0; 5]), 2.0); }
#[test]
fn histogram_soma_total() { assert_eq!(histogram(&[0.0, 0.5, 1.0], 5).iter().sum::<u32>(), 3); }
#[test]
fn search_noise_threshold_monotono() {
    let t10 = search_noise_threshold(1.0, 10);
    let t100 = search_noise_threshold(1.0, 100);
    assert!(t100 > t10);
}
```

**Callers a atualizar no E1:**
- `src/bin/gpu_evolve.rs:2340` — `universe::brain_capacity::participation_ratio` → `universe::math::linalg::participation_ratio`
- `src/brain_capacity.rs` — mantém reexports temporários dos itens que sobram lá (saturation, growth_verdict, Verdict) até E2

**Ordem de commits no E1:**
1. Criar `src/math/` com os 5 arquivos, sem remover nada dos originais ainda
2. Atualizar `src/lib.rs` adicionando `pub mod math;`
3. Atualizar caller em `gpu_evolve.rs:2340`
4. `cargo build --release` — deve compilar
5. `cargo test --release --bin gpu_evolve` — dois testes de genoma devem passar

---

### E2 — Registro de canais (elimina números mágicos)

**Objetivo:** criar `src/perception/channels.rs` com o enum `Channel` (192 entradas),
substituir os literais numéricos de canal em `gpu_evolve.rs::tick()` e em
`gpu_evolve.rs::canal_tem_fonte()`.

**Dependência:** E1 completo (math::laws usado por capacity.rs).

**Arquivos criados/modificados:**

Criar `src/perception/channels.rs` — conteúdo na §1.1 acima.

Modificar `src/perception/mod.rs` — adicionar:
```rust
pub mod channels;
pub use channels::{Channel, N_CHANNELS};
```

Modificar `src/lib.rs` — se `perception` ainda é `pub mod perception;` e agora tem
sub-módulo: verificar que o `mod.rs` existente não conflita com `channels.rs` novo.
O arquivo atual é `src/perception.rs` (arquivo único). Para adicionar sub-módulo:
1. Renomear `src/perception.rs` → `src/perception/retina.rs`
2. Criar `src/perception/mod.rs` com `pub mod retina; pub mod channels; pub use retina::*;`

**Callers de `src/perception.rs` (RetinaPerception) a atualizar:**
- `src/engine.rs:8/224/288` → `use crate::perception::RetinaPerception;` (path não muda se reexportado)
- `src/bin/cortex_evolve.rs:57/381/510/606` → idem
- `src/bin/gpu_loop.rs:72/295/298/460/1015` → idem
- `src/bin/gpu_report.rs:55/240/351` → idem

**Literais a substituir em `gpu_evolve.rs`:**
Usando `Channel::VelFwd.index()` no lugar de `64`, etc. Os literais estão nas regiões:
- `tick()` linhas ~820-1250 (propriocepção, extra, extra2)
- `canal_tem_fonte()` linhas 2504-2550

Após substituição, `canal_tem_fonte()` move para `channels.rs` como método `Channel::source()`.

**Validação do E2:**
```bash
cargo build --release
cargo test --release --bin gpu_evolve
# Verificar que canal_tem_fonte continua funcionando:
cargo run --release --bin gpu_evolve -- --auditar-visao --gens 1 --swarm 4 2>&1 | grep "AUDITORIA"
```

---

### E3 — Quebrar gpu_evolve.rs em módulos por seção

**Objetivo:** reduzir `gpu_evolve.rs` de 2834 linhas para ~800 linhas (main + GpuCtx shell).
Mover cada seção para um sub-módulo interno do binário (módulos `mod` dentro do bin, não
na lib — por causa dos statics que não saem do binário).

**Dependência:** E2 completo.

**Não tocar em nenhum .wgsl neste estágio.**

**Seções a extrair por arquivo novo:**

`src/bin/gpu_evolve/` (pasta — Cargo encontra automaticamente como `src/bin/gpu_evolve/main.rs`):

| Módulo novo | Linhas de origem | Conteúdo |
|---|---|---|
| `gpu_evolve/types.rs` | 78-191 | `RetinaParamsGpu`, `PoseRaw`, `BeingState`, `uniform_entry`, `storage_entry`, `pos_of`, `make_physics_state`, `AtomicF32` |
| `gpu_evolve/context.rs` | 266-1305 | `GpuCtx` struct + `GpuCtx::new`, `upload_engines`, `tick`, `reset_cfc` |
| `gpu_evolve/genome_io.rs` | 1320-1397 | `save_genome`, `load_genome`, `population_seed_for` |
| `gpu_evolve/statics.rs` | 1400-1492 | todos os `static` e `const` globais (`FUEL_ON`, `WALL_LIFE_DRAIN`, etc.) |
| `gpu_evolve/audit.rs` | 2500-2756 | `canal_tem_fonte` (ou usa `Channel::source()`), `AuditAcum`, `AUDIT_ACUM`, `auditoria_acumular`, `auditoria_relatorio`, `faixas_contiguas` |
| `gpu_evolve/tests.rs` | 2780-2833 | `genoma_sobrevive_ida_e_volta_no_disco`, `genoma_antigo_sem_lento_nem_tau_ainda_carrega` |
| `gpu_evolve/main.rs` | 1495-2485 | `main()` — importa os módulos acima |

**Estrutura alternativa se Cargo não suportar pasta para bin** (usar inline `mod`):
```rust
// src/bin/gpu_evolve.rs (mantém nome)
mod types;      // arquivo src/bin/gpu_evolve_types.rs
mod context;    // arquivo src/bin/gpu_evolve_context.rs
mod genome_io;  // ...
mod statics;
mod audit;
```

**Ordem de extração segura** (statics são a dependência mais frágil):
1. `statics.rs` primeiro — os demais módulos apenas lêem, não definem
2. `types.rs` — sem deps de statics
3. `genome_io.rs` — depende de `types.rs` (Genome via lib)
4. `audit.rs` — depende de `statics.rs` e `channels.rs` (E2)
5. `context.rs` — depende de tudo acima
6. `tests.rs` — depende de `genome_io.rs`
7. `main.rs` — importa tudo

**Deduplicação com gpu_loop.rs e cortex_evolve.rs:**
- `uniform_entry` / `storage_entry`: mover para `src/gpu/core.rs` como `pub fn` e referenciar nos 3 bins. Callers: `gpu_evolve.rs:95-118`, `gpu_loop.rs:96-119` (verificar) — ambos precisam de update.
- `make_physics_state`: mover para `src/gpu/physics.rs` como `pub fn make_initial_physics_state(start, fwd)`. Callers: `gpu_evolve.rs:198`, `gpu_loop.rs:148` (verificar exact signature), `cortex_evolve.rs` (verificar). Atenção: `gpu_loop.rs` comentário sugere v0 injetada externamente — confirmar antes de unificar.
- `pos_of`: mover para `src/gpu/physics.rs` como `pub fn pos_of(p: &PhysicsStateGpu) -> [f32; 3]`. Callers: `gpu_evolve.rs:193`, `gpu_loop.rs:145`.

**Closures `voxel_at` triplicadas em `tick()`** (gpu_evolve.rs ~494, ~577, ~653):
Extrair como método `GpuCtx::voxel_at(&self, pos: [f32; 3]) -> u8` dentro de `context.rs`.
Não é possível extrair como `fn livre` porque precisa de `&self.grid_cpu`, `grid_size_cpu`, `cell_size_cpu`.

**Validação do E3:**
```bash
cargo build --release
cargo test --release --bin gpu_evolve  # dois testes de genoma
./target/release/pista_check           # smoke test de mundo
./target/release/gpu_evolve --gens 1 --swarm 4 2>&1 | tail -5  # 1 gen roda
```

---

### E4 — analysis/activation.rs com guardas de runtime

**Objetivo:** criar `src/analysis/activation.rs` com métricas por neurônio coletadas durante
o treino. Adicionar alarmes que disparam em stderr quando limiares são violados.

**Dependência:** E1 (usa `math::stats`, `math::linalg`, `math::laws`).

**Arquivo `src/analysis/activation.rs`:**

```rust
use crate::math::{stats, linalg, laws};

/// Limiares — nomeados aqui, não mágicos em nenhum caller.
pub mod thresholds {
    /// Porcentagem de neurônios mortos acima da qual dispara alarme.
    pub const DEAD_NEURONS_PCT: f32 = 20.0;
    /// Porcentagem de saturação (|h| > 0.99) acima da qual dispara alarme.
    pub const SATURATION_PCT: f32   = 30.0;
    /// |lambda_max| de W_rec acima do qual dispara alarme (instabilidade).
    pub const SPECTRAL_RADIUS: f32  = 1.0;
    /// SNR em i8 abaixo do qual dispara alarme.
    pub const SNR_I8_MIN: f32       = 6.0;
    /// Entropia da política (distribuição steer/throttle/brake) abaixo da qual dispara.
    pub const POLICY_ENTROPY_MIN: f32 = 0.5;
}

/// Estatísticas por neurônio de uma camada.
#[derive(Debug, Clone)]
pub struct NeuronStats {
    pub neuron_id: usize,
    pub mean: f32,
    pub sigma: f32,
    pub p5: f32,
    pub p95: f32,
    pub dead_pct: f32,       // fração de ticks com |h| < 0.01
    pub saturated_pct: f32,  // fração de ticks com |h| > 0.99
    pub snr_i8: f32,         // SNR de quantização
}

/// Estatísticas agregadas de uma camada inteira.
#[derive(Debug, Clone)]
pub struct LayerStats {
    pub layer_name: String,   // "L1", "L2", "CfC", "SlowBrain"
    pub neurons: Vec<NeuronStats>,
    pub spectral_radius: f32,  // raio espectral de W_rec (0.0 se não houver W_rec)
    pub effective_rank: usize,
    pub policy_entropy: f32,   // entropia da distribuição de ações (só para camadas de saída)
    pub histogram: Vec<u32>,   // histograma de ativações (32 bins, range [-1, 1])
}

/// Alarme: violação de limiar detectada.
#[derive(Debug)]
pub struct ActivationAlarm {
    pub layer: String,
    pub kind: AlarmKind,
    pub value: f32,
    pub threshold: f32,
}

#[derive(Debug)]
pub enum AlarmKind {
    DeadNeurons,
    Saturated,
    SpectralRadiusHigh,
    SnrLow,
    NanDetected,
    PolicyEntropyLow,
}

/// Coleta stats de ativação de um slice de amostras (n_ticks × n_neurons).
/// Retorna stats + lista de alarmes que violam os limiares em `thresholds`.
pub fn collect_activation_stats(
    layer_name: &str,
    samples: &[Vec<f32>],        // samples[tick][neuron]
    w_rec: Option<&[f32]>,       // pesos recorrentes (para raio espectral)
    n_neurons: usize,
    policy_actions: Option<&[[f32; 3]]>,  // (throttle, steer, brake) por tick
) -> (LayerStats, Vec<ActivationAlarm>);

/// Imprime relatório de stats para stderr (não bloqueia, não tranca Mutex).
pub fn print_stats_report(stats: &LayerStats, alarms: &[ActivationAlarm]);
```

**Integração em gpu_evolve.rs (E4 adiciona ao código de fim de geração):**
```rust
// No fim de cada geração, após coletar slow_samples:
let (stats, alarms) = analysis::activation::collect_activation_stats(
    "SlowBrain",
    &slow_samples,
    None,   // SlowBrain não tem W_rec acessível aqui
    3,      // 3 saídas
    Some(&actions_buffer),
);
analysis::activation::print_stats_report(&stats, &alarms);
// Alarmes vão para stderr — não bloqueiam o loop de evolução
```

**NaN-check**: adicionar no início de `GpuCtx::tick()`, antes do submit, se `cfg!(debug_assertions)`:
```rust
debug_assert!(
    phys_states.iter().flat_map(|s| s.pos.iter()).all(|v| v.is_finite()),
    "NaN detectado em posições antes do submit GPU"
);
```

**Validação do E4:**
```bash
cargo test --release    # testes unitários de math + analysis
cargo build --release
./target/release/gpu_evolve --gens 2 --swarm 8 2>&1 | grep -E "ALARME|dead|saturad"
```

---

### E5 — Painel HTTP tempo-real (`--painel`)

**Objetivo:** servidor HTTP leve dentro do processo `gpu_evolve`, ativado pela flag
`--painel` (porta 3200). Sem dependência nova pesada.

**Dependência:** E4 (usa `ActivationAlarm`, `LayerStats`).

**Detalhes na §3 abaixo.**

---

### E6 — Golden-parity (auditoria antes/depois)

**Objetivo:** verificar que `pista_check` passa antes e depois de toda a refatoração,
e que o genoma de um campeão existente carrega e produz as mesmas saídas em 100 ticks.

**Procedimento:**
```bash
# Antes de qualquer estágio (baseline):
./target/release/pista_check > /tmp/pista_before.txt
./target/release/gpu_evolve --carregar evolve_best_genXXXX.bin --gens 1 --swarm 1 \
    --auditar-visao 2>&1 > /tmp/audit_before.txt

# Após cada estágio:
./target/release/pista_check > /tmp/pista_after.txt
diff /tmp/pista_before.txt /tmp/pista_after.txt   # deve ser vazio

# Após E6 final:
./target/release/gpu_evolve --carregar evolve_best_genXXXX.bin --gens 1 --swarm 1 \
    --auditar-visao 2>&1 > /tmp/audit_after.txt
diff /tmp/audit_before.txt /tmp/audit_after.txt    # deve ser vazio
cargo test --release --bin gpu_evolve              # dois testes de genoma
```

---

## 3. Painel tempo-real

### 3.1 Arquitetura

O painel roda numa thread separada dentro do processo `gpu_evolve`. A thread abre um
`TcpListener` na porta 3200 e atende conexões sequencialmente (um cliente por vez é
suficiente para diagnóstico). Os dados são compartilhados via `Arc<Mutex<PainelState>>`.

`serde_json` já está em `Cargo.toml:38` — zero dependência nova.
`std::net::TcpListener` — stdlib, sem dep nova.

### 3.2 Estrutura de dados compartilhados

```rust
// src/bin/gpu_evolve/painel.rs  (criado no E5)

use std::sync::{Arc, Mutex};
use std::net::TcpListener;

#[derive(Default, serde::Serialize, Clone)]
pub struct PainelState {
    // --- status da geração (atualizado no fim de cada geração) ---
    pub gen: u32,
    pub mediana_m: f64,
    pub elite_m: f64,
    pub mortos_pct: f32,         // % que não completaram 1 volta
    pub obediencia_pct: f32,     // progresso médio / progresso do piloto sensorial
    pub setores: Vec<f32>,       // fitness média por setor (N_CHECKPOINTS valores)
    pub slope: f64,              // OLS slope (m/gen)

    // --- snapshot de tick (atualizado a cada N ticks, default N=64) ---
    pub tick: u64,
    pub vivos: u32,
    pub vel_mediana: f32,        // m/s, só vivos
    pub life_mediana: f32,       // life_left normalizado [0, 1]
    pub fuel_mediana: f32,

    // --- ativação (atualizado no fim de cada geração) ---
    pub activation_alarms: Vec<String>,     // strings prontas para JSON
    pub dead_pct_l1: f32,
    pub saturated_pct_l1: f32,
    pub spectral_radius: f32,
    pub snr_i8: f32,
}

#[derive(Default, serde::Serialize, Clone)]
pub struct BeingSnapshot {
    pub id: usize,
    pub x: f32,
    pub z: f32,
    pub vel: f32,
    pub life: f32,
    pub vivo: bool,
}

pub struct PainelHandle {
    pub state: Arc<Mutex<PainelState>>,
    pub beings: Arc<Mutex<Vec<BeingSnapshot>>>,
}

impl PainelHandle {
    pub fn launch(port: u16) -> Self { ... }  // spawna thread TcpListener
    pub fn update_gen(&self, ...) { ... }
    pub fn update_tick(&self, ...) { ... }
}
```

### 3.3 Endpoints

| Endpoint | Método | Descrição |
|---|---|---|
| `GET /` | GET | HTML único embutido (include_str!) com canvas JS, polling de `/api/status` a 1 Hz |
| `GET /api/status` | GET | `PainelState` serializado como JSON |
| `GET /api/activation` | GET | stats por camada + histogramas (de E4) + alarmes |
| `GET /api/beings` | GET | array de `BeingSnapshot` (posições dos vivos) |

### 3.4 HTML embutido

Arquivo `src/bin/gpu_evolve/painel.html` — carregado via `include_str!("painel.html")`.

Requisitos do HTML:
- Zero CDN externo (canvas + JS puro)
- Polling `fetch('/api/status')` a cada 1000ms
- Gráfico de linha: mediana de distância por geração (últimas 50 gens)
- Mapa 2D dos seres vivos: canvas 400×400, pista como polígono cinza, pontos coloridos por vida
- Tabela de alarmes de ativação (vermelho se presente, verde se vazio)
- Indicadores numéricos: gen atual, mortos %, obediência %, slope

### 3.5 Servidor HTTP mínimo (sem framework)

```rust
fn serve_one(mut stream: TcpStream, state: &PainelState, beings: &[BeingSnapshot]) {
    let mut buf = [0u8; 512];
    let _ = stream.read(&mut buf);
    let path = extract_path(&buf);  // parse mínimo da primeira linha HTTP

    let (status, ct, body) = match path {
        "/" => ("200 OK", "text/html", HTML.to_string()),
        "/api/status"  => ("200 OK", "application/json", serde_json::to_string(state).unwrap()),
        "/api/activation" => { ... },
        "/api/beings"  => ("200 OK", "application/json", serde_json::to_string(beings).unwrap()),
        _ => ("404 Not Found", "text/plain", "not found".to_string()),
    };

    let response = format!(
        "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
        status, ct, body.len(), body
    );
    let _ = stream.write_all(response.as_bytes());
}
```

### 3.6 Coleta de dados

A cada geração, `main()` chama `painel.update_gen(gen, mediana, elite, mortos_pct, ...)`.
A cada 64 ticks, `GpuCtx::tick()` chama `painel.update_tick(tick, vivos, vel_median, ...)`.
Os dados de ativação chegam do E4 (`collect_activation_stats`) no fim da geração.

### 3.7 Flag de ativação

```
./target/release/gpu_evolve --painel [--painel-porta 3200] --gens 100 --swarm 60
```

Sem `--painel`: thread não é lançada, zero overhead.

---

## 4. Riscos críticos e como cada estágio os evita

### R1 — Contrato WGSL (structs Pod, strides, constantes espelhadas)

**Risco:** qualquer mudança em `PhysicsStateGpu`, `PosePerception`, `BrainParams` etc.
que não seja espelhada no WGSL correspondente corrompe dados GPU silenciosamente.

**Mitigação por estágio:**
- E1-E3: não tocar nenhum `.wgsl`. As structs `#[repr(C)] + Pod` só movem de arquivo;
  o layout em memória é invariante.
- E4-E5: não adicionar campos aos structs Pod — só criar structs novas de análise (não Pod).
- E6: `gpu_validate` antes e depois de qualquer mudança de stride.

**Constantes espelhadas críticas** (manter lista sempre atualizada):
| Constante | Host | WGSL | Status |
|---|---|---|---|
| `WAVE` / `SENSORY_WAVE_SIZE` | `config.rs:309 = 192` | `brain.wgsl:86 = 192u` | BATENDO |
| `WAVE_U32S` | `perception.rs:84 = 56` | todos shaders `= 48u` | DIVERGE — host aloca 16% a mais (safe, mas inconsistente) |
| `TEMPORAL_STRIDE` | `perception.rs:188 = 58` | `perception_temporal.wgsl:110 = 58u` | BATENDO |
| `RETINA_COLS` | `config.rs:256 = 126` | todos shaders `= 126u` | BATENDO |
| `INPUT_WGSL_STRIDE` | `physics.rs:80 = 368` | stride manual no serialize_input() | SEM validação runtime |
| `MidState` | `physics.rs:173` | `physics_a/b/c.wgsl` | SEM validação runtime |
| `WEIGHTS_PER_SER` slow | `slow_brain.wgsl:77 = 1091u` | `config.rs DNA_SLOW_SIZE = 1099` | DIVERGE — confirmar upload_weights |
| `N_INPUTS` cortex | `cortex_sheet.rs:55 = 223` | `cortex_sheet.wgsl:34 = 223u` | legado (onda antiga) |

**Ação recomendada antes do E1** (não é estágio de refatoração, é pré-condição):
Adicionar `assert_eq!` em `PerceptionGpu::new()` comparando `TEMPORAL_STRIDE` com a
constante esperada, para detectar divergência futura de forma determinística.

### R2 — Statics globais em gpu_evolve.rs não saem do binário

**Risco:** `FUEL_ON`, `WALL_LIFE_DRAIN`, `STEER_ALPHA`, etc. são lidos dentro de
`GpuCtx::tick()`. Se `tick()` for movido para a lib sem levar os statics, o código não
compila (referência a statics não declarados).

**Mitigação E3:** os statics ficam em `gpu_evolve/statics.rs` — permanecem no binário,
não na lib. `context.rs` os importa com `use super::statics::*;`. Não promover
nenhum deles para a lib enquanto não houver API limpa (parâmetros ou struct de config).

### R3 — Ordem de dispatch no CommandEncoder (gpu_loop.rs:935-999)

**Risco:** a invariante `copy_retina_prev ANTES do dispatch da retina` é frágil.
Qualquer reordenação de chamadas no encoder zera os canais de motion (ch 21-27).

**Mitigação:** não tocar em `gpu_loop.rs` nos estágios E1-E5. E6 (golden-parity)
verifica os canais de motion antes e depois via `--auditar-visao`.

### R4 — BeingState de gpu_evolve vs gpu_loop são structs distintas

**Risco:** `gpu_evolve::BeingState` tem 20 campos (vida, combustível, dopamina, etc.).
`gpu_loop::BeingState` tem 5 campos. Qualquer tentativa de unificação sem cuidado quebra
`gpu_loop`.

**Mitigação E3:** mover `gpu_evolve::BeingState` para `gpu_evolve/types.rs` mas **não**
para a lib. As duas structs ficam separadas por design — não unificar nesta rodada.

### R5 — save_genome/load_genome: formato binário sem cabeçalho

**Risco:** reordenar campos de `Genome` sem atualizar os serializadores corrompe todos os
`.bin` existentes sem erro detectável.

**Mitigação:** o teste `genoma_sobrevive_ida_e_volta_no_disco` detecta campo novo sem
serialização, mas não detecta reordenação. Antes de qualquer mudança em `Genome`:
1. Rodar `cargo test --release --bin gpu_evolve` para garantir baseline
2. Qualquer campo novo em `Genome` exige atualização simultânea de `save_genome` e `load_genome`
3. Não reordenar campos existentes de `Genome` — apenas acrescentar ao final

### R6 — AUDIT_ACUM cresce sem limite entre gerações

**Risco:** com `--auditar-visao` e 100 gerações de 8000 ticks, `hist_ser0` acumula
800.000 entradas × 208 bytes ≈ 166 MB. Sem limpeza entre gerações.

**Mitigação E4:** adicionar limpeza de `hist_ser0` no início de cada geração em
`auditoria_acumular()`:
```rust
// No início de nova geração, zerar o acumulador:
acum.hist_ser0.clear();
acum.n_ticks = 0;
```
Implementar no E3 quando audit.rs é extraído.

### R7 — Código morto confirmado vs não confirmado

Antes de remover qualquer símbolo com "zero callers confirmados":
1. `grep -rn "NomeDoSimbolo" src/ scripts/ docs/` — confirmar ausência
2. Verificar se aparece em algum script externo não listado
3. Só remover após E6 (golden-parity com genoma real confirmado)

Símbolos com zero callers MAS que têm valor histórico / forense:
- `fisica_bisect.rs` — documenta workaround do SIGSEGV do driver RADV; não remover
- `dna_compress.rs` e `generic_dna.rs` — commits "a" de abril; remover após confirmar com Leonardo

### R8 — motor.rs (3382 linhas) não é tocado

**Razão:** `motor.rs` é a maior unidade de compilação e escreve canais temporais por cima
da saída GPU. Qualquer mudança tem custo de recompilação alto e risco de quebrar o
caminho Bevy. Fora do escopo desta rodada.

---

## 5. O que NÃO fazer

1. **Não tocar nenhum `.wgsl` nos estágios E1-E3.** Shaders mudam apenas em E4+ e
   somente para adicionar NaN-check ou validação de stride — nunca para reorganizar lógica.

2. **Não renomear structs `#[repr(C)]` espelhadas no WGSL.** `PhysicsStateGpu`,
   `PosePerception`, `BrainParams`, `ParamsGpu`, `MidState`, `MidState2`,
   `RetinaParamsGpu`, `PoseRaw` — qualquer renomeação é segura em Rust mas enganosa:
   o contrato real é o layout em bytes, não o nome. Se renomear, o WGSL permanece com
   o struct de mesmo significado e deve ser atualizado em comentário.

3. **Não mexer em `motor.rs` nesta rodada.** 3382 linhas, dependência de todo o pipeline
   ECS, escreve canais temporais por cima da GPU — risco desproporcional ao ganho.

4. **Não mover `VoxelWorld` ou `VoxelWorldResource`.** 30+ callers, incluindo todos os
   bins de replay, todos os crates de GPU e o motor ECS. Custo de migração maior que
   qualquer benefício de organização.

5. **Não mover `src/config.rs` nem reorganizar `iara::brain::*`.** Os caminhos
   `crate::config::iara::brain::SENSORY_WAVE_SIZE` etc. são usados por 10+ arquivos
   com caminhos absolutos. Uma reorganização exigiria atualizar todos os shaders que
   usam as constantes derivadas.

6. **Não usar `sed` ou `.replace()` sem verificar unicidade.** Para cada substituição
   de literal de canal: `grep -c "numero_magico" arquivo.rs` deve retornar 1 antes da
   substituição. Usar `assert_eq!(s.matches(old).count(), 1)` em scripts de migração.

7. **Não promover statics de gpu_evolve para a lib sem API limpa.** `FUEL_ON`,
   `STEER_ALPHA`, `WALL_LIFE_DRAIN`, etc. controlam comportamento via estado global
   invisível à assinatura de `tick()`. Promovê-los para a lib sem encapsulá-los em
   uma struct de config cria acoplamento global na biblioteca.

8. **Não unificar `replay::ReplayFrame` com `telemetry::ReplayFrame`.** São structs com
   propósitos distintos: uma é `bytemuck::Pod` de exatamente 128 bytes (arquivo de
   disco), a outra é estrutura de trace de cérebro. Unificar introduz UB se qualquer
   caller assumir o tamanho errado.

9. **Não remover `genoma_sobrevive_ida_e_volta_no_disco` nem `genoma_antigo_sem_lento_nem_tau_ainda_carrega`.**
   Esses testes em `gpu_evolve.rs` são a barreira de regressão da serialização. Ao
   mover para `tests.rs` no E3, confirmar que `cargo test --release --bin gpu_evolve`
   ainda os encontra.

10. **Não criar arquivos de relatório/análise em Markdown.** Achados de análise ficam
    neste documento ou em comentários de código — não gerar `FINDINGS.md` ou similar.

---

## Apêndice A — Sequência de comandos de validação por estágio

```bash
# Antes de qualquer estágio — salvar baseline
cargo test --release --bin gpu_evolve 2>&1 | tee /tmp/test_baseline.txt
./target/release/pista_check 2>&1 | tee /tmp/pista_baseline.txt

# Após cada estágio
cargo build --release 2>&1 | grep -E "^error" | wc -l  # deve ser 0
cargo test --release --bin gpu_evolve 2>&1 | grep -E "FAILED|passed|test result"
diff /tmp/pista_baseline.txt <(./target/release/pista_check 2>&1)

# E4 específico
./target/release/gpu_evolve --gens 2 --swarm 8 2>&1 | grep -E "ALARME|dead_pct|saturad"

# E5 específico — painel
./target/release/gpu_evolve --painel --gens 5 --swarm 8 &
sleep 3 && curl -s http://localhost:3200/api/status | python3 -m json.tool
curl -s http://localhost:3200/api/beings | python3 -m json.tool
kill %1
```

---

## Apêndice B — Dual-source de dna.rs / dna_genesis.rs

`src/evolution/dna.rs` e `src/evolution/dna_genesis.rs` exportam símbolos com nomes
idênticos (`MorphoGenes`, `EliteDna`, `SCAR_TTL_INITIAL`, etc.) ambos via
`evolution/mod.rs`. Esta duplicata é um ciclo lógico de duas fontes de verdade.

**Não resolver nesta rodada** (fora do escopo de refatoração estrutural), mas registrar:
- `dna_genesis.rs` é a versão mais recente (DNA V8/TLV)
- `dna.rs` é o legado (DNA V3)
- Ao remover o legado, verificar callers de `dna_compress.rs` e `generic_dna.rs` que
  importam `DNA_MAGIC` de `dna_genesis.rs` apenas

---

*Fim do REFACTOR_PLAN.md — versão 1.0, 2026-08-13*
