# Plano de encaixe — o mundo do táxi no Universe

Escrito em 2026-09-10 pelo arquiteto (Opus). **Documento de plano; nada foi compilado nem
executado.** A spec do mundo é `/home/leonardo/projects/LLM/neuromod-lab/docs/71-spec-mundo-taxi.md`
(daqui em diante: **SPEC**). Este arquivo diz onde o código mora, quais são as interfaces e
como 6 implementadores trabalham em paralelo sem colidir.

> **Idioma deste documento.** `docs/CLAUDE.md` manda escrever registro novo em inglês.
> Este plano está em português por ordem explícita do dono (10/09). **O CÓDIGO nasce em
> inglês** — identificadores, comentários, nomes de teste e commits (`NAMING.md:5`,
> `scripts/check_language.sh`). Quando o v0 fechar, este arquivo pode ser traduzido.

Motivo: `neuromod-lab/docs/70-reprojeto.md` §2.1 — o tique em torch custa **4 486
lançamentos de kernel ≈ 11.0 ms**; o Universe faz o tique inteiro em WGSL em **0.63 ms**.
O ganho buscado é de lançamento, não de aritmética.

---

## 1. Onde entra

### 1.1 O que o Universe tem hoje (levantado, com linha)

| fato | consequência para o táxi |
|---|---|
| `Cargo.toml:8` — `[workspace] members = [".", "docs/arquivo/dirs/iara_genesis"]`; o resto de `vendor/*` são path-deps | acrescentar um membro é mudança de 1 linha |
| `bevy` é dependência **direta e obrigatória** do crate `universe` (`Cargo.toml:26-40`), sem feature-gate; até `src/gpu/mod.rs:59` importa `bevy::math` | um mundo headless dentro de `universe` arrasta o Bevy inteiro para cada `cargo test` |
| **dois runtimes que não compartilham código de tique**: `src/bin/gpu_evolve/context.rs:473` (`GpuCtx::tick`, GPU) e `src/simulation/motor.rs:242` (`update_racers`, CPU+Bevy, 3839 linhas) | escolher UM. Escolha: nenhum dos dois — ver §1.2 |
| `VoxelWorld` (`src/world.rs:63-86`) é grade 320×320 × 3 m com `waypoints_world`, `racing_line`, `lap_distance`; `VoxelType` só tem `Grass/Road/Wall/Hole` (`src/world.rs:29-35`) | não há como representar posto/distrito/passageiro sem reescrever o enum e a struct |
| `PhysicsStateGpu` (`src/gpu/physics.rs:186-238`) é carro de 4 rodas com Pacejka, RPM, marcha, temperatura de motor | zero reuso: o táxi é um ponto 2D com 8 direções |
| GA em `src/gpu/evolution.rs:704-773`: torneio k=3, σ=0.05, `p_mutate`=0.02, `n_elite`=3, pop 60 | **incompatível** com o GA do lab (elite 15 %, σ_mut 0.1, TODOS os genes mutam, P=128×5 sementes). Não reusar |
| `docs/STYLE.md:96` — teto de **600 linhas** por `.rs` e **400** por `.wgsl`; `motor.rs` (3839) e `world.rs` (1827) já estouram e estão na fila de split | não empilhar nada nesses dois |
| `NAMING.md:5-6`, `CLAUDE.md:110-113` — **inglês em tudo** (identificadores, comentários, testes, commits), com `scripts/check_language.sh` | o código do táxi nasce em inglês; só os docs ficam em português |
| `docs/STYLE.md:295-299` — teste nomeado como frase em inglês descrevendo o comportamento, sem prefixo `test_` | ex.: `first_tick_senses_match_the_python_trace` |
| `CLAUDE.md:60-66` — física/percepção nova entra atrás de flag off-by-default com A/B pareado | o táxi não toca nada existente ⇒ regra satisfeita por construção |
| `CLAUDE.md:75-79` — saídas pesadas nunca em `/tmp` nem dentro do repo | tudo em `/media/llm-data/universe-runs/taxi-world/` |
| `src/predator.rs:9-13`, `src/resources.rs` — padrão do repo: módulo puro, sem I/O e sem GPU, documentado como "NOT wired" até alguém integrar explicitamente | é o padrão a copiar, um degrau acima (crate próprio) |
| `docs/ROADMAP.md` M3 "The second universe" e `docs/UNIVERSO_ALEM_DO_CARRO.md` | o táxi É esse marco; nada disso está implementado |
| **`WDNA` não existe no repositório** (0 ocorrências); o DNA em uso é `.dna` **V8** (`src/config.rs:343`) no caminho Bevy e `UGEN v2` (`src/gpu/genome_io.rs:53`) no caminho GPU | não prometer compatibilidade de genoma no v0 |

### 1.2 Decisão: crate novo `crates/taxi-world`, membro do workspace

```
/home/leonardo/projects/LLM/Universe/crates/taxi-world/
```

Registrado em `Cargo.toml:8` como terceiro membro. **Sem dependência de `universe` e sem
Bevy.** Dependências: `serde` + `serde_json` (ler o traço/mapa), `rand`+`rand_chacha`
(PRNG próprio, só na bateria), `bytemuck` (v1, buffers), `wgpu` (v1, atrás da feature
`gpu`, default-off). `rayon` opcional para o v0 CPU.
`edition = "2024"`, `rust-version = "1.85.0"` (herdados do workspace, `Cargo.toml:13-14`;
`rustc` local = 1.93.1) ⇒ **`f32::round_ties_even` está disponível** (estável desde 1.77),
que é o que o risco 2 do §5 exige.

Por quê crate e não módulo em `src/`:

1. O crate `universe` linka Bevy sempre. Um portão de identidade que precisa rodar em
   segundos não pode pagar o build do Bevy.
2. `cargo test -p taxi-world` fica isolado: não recompila `motor.rs` (3839 linhas) nem os
   28 binários de `src/bin/`.
3. O teto de 600 linhas por arquivo é confortável num crate novo e impossível dentro de
   `world.rs`/`motor.rs`.
4. Quando o táxi for para a tela, um `TaxiWorldPlugin` dentro de `universe` depende de
   `taxi-world`; a seta aponta para o lado certo (UI depende do núcleo, nunca o contrário)
   — que é justamente o compromisso declarado e hoje violado em `src/gpu/mod.rs:1-13`.

**O que NÃO se reusa, e por quê** (declarado, não silenciado):

| peça do Universe | veredito |
|---|---|
| `VoxelWorld`, `VoxelType`, `RacingLine`, waypoints | não — semântica de pista embutida |
| `PhysicsStateGpu`, `physics_{a,b,c}.wgsl` | não — carro de 4 rodas |
| `retina.wgsl`, `perception*.wgsl`, `retina_digest.wgsl` | não no v0 (SPEC §12: sem retina); **candidatos reais no v2**, quando a retina do táxi entrar |
| `brain.wgsl`, `SovereignEngine` | não — o cérebro do v0 é um produto interno de 29 termos |
| `src/gpu/evolution.rs` (GA) | não — parâmetros incompatíveis (§1.1) |
| `genome_io.rs` / `UGEN v2` / `.dna` V8 | não no v0. O genoma do táxi é 40 f32; formato próprio, versionado desde o início |
| `src/gpu/core.rs` (padrão de abrir device, bind groups, `#[repr(C)] + bytemuck::Pod`) | **sim, como PADRÃO copiado**, não como código chamado |
| `tests/wgsl_validation.rs` (naga estático, sem GPU) | **sim** — replicar no crate novo no v1 |
| `docs/STYLE.md`, `NAMING.md`, convenção de nome de teste | sim, integralmente |

### 1.3 Regra do device único

`ARCHITECTURE.md:381-383` e `src/gpu/core.rs:1-14`: um só `wgpu::Device` no processo.
O `taxi-world` **nunca cria device próprio quando embutido**: a API é

```rust
pub fn new_gpu(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, world: &World, n: usize) -> GpuTaxi
```

O binário headless `taxi_evolve` cria o device (é o único processo) e injeta. Quando o táxi
for chamado de dentro do `universe`, recebe o device de lá. Sem `OnceCell` global.

---

## 2. Arquitetura em 2 estágios

```
        traco_ref.json (Python, GPU)
                 │
      ┌──────────┴───────────┐
      ▼                      ▼
   v0: tique em Rust CPU   ─── mesmos portões P0..P3 ───▶  v1: tique em WGSL
   determinístico                                          (mesma SPEC, mesmo teste)
   1 ser = 1 iteração                                      1 ser = 1 invocação
```

**v0 — CPU, `taxi-world/src/`.** Um tique = uma função pura sobre um `WorldState` em SoA.
Sem GPU, sem paralelismo obrigatório (rayon opcional por ser, mas a ordem das operações
dentro do ser é a da SPEC §3.1). Objetivo único: **fechar o portão de identidade**.
Custo esperado: 16 seres × 2400 tiques é instantâneo; 128×7200 tiques (juiz) na casa de
segundos.

**v1 — WGSL, `taxi-world/src/gpu/`.** O mesmo tique, um `@workgroup_size(64)`, uma
invocação por ser, **um único dispatch por tique** (o táxi não tem retina no v0, então não
há a cadeia de 6 dispatches do carro). O mapa vira buffers read-only; o estado vira SoA
read_write. **O v1 só existe depois de o v0 fechar P0..P3**, e roda os MESMOS quatro
portões contra o MESMO `traco_ref.json`.

**Por que dois estágios e não direto o WGSL:** um portão que falha no WGSL não distingue
"a spec foi mal lida" de "o shader diverge do f32 do host". Com o v0 fechado, a segunda
falha é a única possível, e o v1 vira um diff de duas implementações Rust/WGSL da mesma
função — que é depurável.

---

## 3. Layout de dados

### 3.1 Estado — SoA por ser

Um `Vec<f32>` por variável, comprimento `B`. Nada de array-de-struct: o `PhysicsStateGpu`
do carro (`src/gpu/physics.rs:186-238`) precisa de `INPUT_WGSL_STRIDE = 368` e de
serialização manual (`physics.rs:174-183`) porque o WGSL alinha arrays de struct a 16
bytes e o `#[repr(C)]` do Rust não. **SoA elimina a classe inteira desse bug.**

```rust
pub struct Beings {
    pub n: usize,
    // posição e corpo
    pub pos_x: Vec<f32>, pub pos_y: Vec<f32>,
    pub fuel: Vec<f32>, pub water: Vec<f32>, pub hunger: Vec<f32>,
    pub fatigue: Vec<f32>, pub heat: Vec<f32>, pub life: Vec<f32>,
    pub money: Vec<f32>, pub reserve: Vec<f32>,
    pub stopped_ticks: Vec<f32>, pub metabolism: Vec<f32>, pub consumption: Vec<f32>,
    pub home_district: Vec<u32>, pub phase: Vec<f32>,
    // passageiro
    pub carrying: Vec<f32>,            // -1.0 ou +1.0  (f32, não bool: é lido como sentido)
    pub dest_x: Vec<f32>, pub dest_y: Vec<f32>,
    pub board_x: Vec<f32>, pub board_y: Vec<f32>,
    pub origin_district: Vec<u32>,
    pub since_pickup: Vec<f32>,
    pub px: Vec<f32>,      // n * NPX * 2, layout [being][slot][xy]
    pub pxd: Vec<f32>,     // idem
    pub px_ok: Vec<u32>,   // n * NPX  (u32, não bool: WGSL não tem bool em storage)
    // contadores
    pub alive_ticks: Vec<f32>, pub deliveries: Vec<f32>, pub revenue: Vec<f32>,
    pub breakdowns: Vec<f32>, pub damage_total: Vec<f32>,
    pub refuels: Vec<f32>, pub liters: Vec<f32>, pub meals: Vec<f32>,
    pub repairs: Vec<f32>, pub drinks: Vec<f32>, pub faults: Vec<f32>,
    // índice do mapa (lote com NC mapas × R réplicas × P genomas)
    pub map_idx: Vec<u32>, pub genome_idx: Vec<u32>,
}
```

**Índice do lote (SPEC §10):** `b = p·(NC·R) + c·R + r`. `map_idx[b] = c`,
`genome_idx[b] = p`. Errar isso faz o genoma ser avaliado no mapa errado e o resultado
fica *plausível*.

### 3.2 Mapa — buffers imutáveis, um por mapa

```rust
pub struct World {
    pub side: f32,                    // L = 200.0
    pub cost_tex: Vec<f32>,           // [tex*tex] = 64*64,  índice iy*tex + ix
    pub district_field: Vec<u32>,     // [GR*GR] = 128*128,  índice iy*GR + ix
    pub centers: Vec<[f32; 2]>,       // [K]
    pub tax: Vec<f32>,                // [K]  aliq
    pub price_mul_base: Vec<f32>,     // [K]  pmul0
    pub rent: Vec<f32>,               // [K]  alug
    pub demand: Vec<f32>,             // [K]
    pub price_phase: Vec<f32>,        // [K]
    pub services: [Vec<[f32; 2]>; 5], // ordem TIPOS: posto, comida, agua, descanso, oficina
}
```

Achatado para GPU: `services` vira um buffer `[max_n * 5 * 2]` + `counts: [u32; 5]`.
**Sem máscara `1e9`** no Rust — a máscara existe no Python só porque `_empilha`
(`campo.py:123-127`) padeia os tipos ao mesmo comprimento; no Rust cada tipo tem seu
`count` real e o laço para lá. Isso é numericamente idêntico (o `1e9` nunca vence um
`argmin`) e evita 32 distâncias falsas por tipo por tique.

**Ordem dos eixos: `[iy][ix]`.** Está na SPEC §2.3 e é o erro que produz um mundo
transposto que "quase funciona".

### 3.3 Genoma

`[f32; 40]` (`NP = NF + S + 1 = 29 + 10 + 1`, SPEC §4.3). Serialização própria:

```
magic  b"TAXG"   4 B
version u32 LE   = 1
nf      u32 LE   = 29        ← validado contra a build
s       u32 LE   = 10
n       u32 LE                (quantos genomas)
weights f32 LE   [n * (nf + s + 1)]
```

`nf` no cabeçalho porque `NM_RISCO`/`NM_PONTO` mudam `NF` do lado Python (SPEC §4.2) e um
genoma de `NF = 27` carregado numa build de `NF = 29` produz um ser silenciosamente
errado. **Recusar, não alinhar por posição.** O alinhamento por NOME
(`e43_campo.py:alinha_semente`) fica numa ferramenta separada, do lado Python.

---

## 4. Grafo de tarefas

Todas as 6 tarefas são **independentes** porque as interfaces abaixo são contrato deste
documento — cada implementador escreve contra elas e o T1 só as materializa. Quem chegar
primeiro num arquivo compartilhado (`lib.rs`, `Cargo.toml`) mexe só na sua linha de `mod`.

Regra comum a todas:
- **Inglês** em identificadores, comentários e nomes de teste (`NAMING.md:5-6`).
- Teste nomeado como frase de comportamento, sem `test_` (`docs/STYLE.md:295-299`).
- Teto de 600 linhas por `.rs` (`docs/STYLE.md:96`).
- Comando: `CARGO_BUILD_JOBS=4 nice -n 10 cargo test -p taxi-world <nome>` — **nunca**
  `cargo build`/`run` largado, e **nunca** sem `nice`: a fila `nmd` na GPU é limitada por
  CPU de lançamento de kernel e roubar núcleo atrasa a bateria em curso.
- Nada de escrita em `/tmp` nem dentro do repo (`CLAUDE.md:75-79`): saídas em
  `/media/llm-data/universe-runs/taxi-world/`.

---

### T0 — Emendas ao gerador do traço (Python, `neuromod-lab`)

**Não é Rust. É pré-requisito de T5 e não bloqueia T1..T4, T6.**

Arquivo a editar: `/home/leonardo/projects/LLM/neuromod-lab/src/traco_referencia.py`
(23 linhas hoje; a emenda são ~15).

Fazer, exatamente o que a SPEC §11.1 lista:

| # | emenda |
|---|---|
| a | gravar `alvo (T,B,2)` por tique — **obrigatório**: `alvo()` (`campo.py:256-259`) lê `px` sem a máscara `px_ok`, então `d_alvo` passa a depender do RNG do torch a partir do 1º embarque e o Rust não pode reproduzi-lo |
| b | gravar `F (B,9,NF)`, `sc`, `svn`, `sv`, `tudo` no **tique 0** |
| c | gravar `_ptos (B,5,2)` e `_dist (B,5)` no tique 0 |
| d | gravar `margem (T,B)` = maior menos segundo maior de `tudo` |
| e | gravar `esc_livre (T,B)` = nº de opções não mascaradas |
| f | `os.environ['NM_RISCO'] = '1'` e `NM_PONTO = '1'` **antes** do `from campo import *` |

Depois, quando a GPU liberar (fila `nmd`), rodar:

```bash
cd /home/leonardo/projects/LLM/neuromod-lab/src
NM_RISCO=1 NM_PONTO=1 NM_COMPILE=0 \
  /home/leonardo/projects/LLM/.venv-rocm/bin/python traco_referencia.py \
  --B 16 --T 2400 --seed-mapa 1000 \
  --out /media/llm-data/universe-runs/neuromod-lab-taxi/e81
```

**Aceite:** o JSON existe, tem as chaves `alvo`, `f_t0`, `sc_t0`, `svn_t0`, `sv_t0`,
`tudo_t0`, `ptos_t0`, `dist_t0`, `margem`, `esc_livre`; `len(nomes_f) == 40`;
`len(nomes_corpo) == 19`; `len(genoma) == 40`. Um script de validação de esquema
(`scripts/valida_traco.py`, ~30 linhas, roda em CPU, sem torch) é parte do aceite.
**Não tocar** em `campo.py`, `lab.py`, `e43_campo.py` — a bateria em curso os usa.

Estimativa: 1 sessão curta para as emendas + validador; a geração é 1 comando enfileirado.

---

### T1 — Esqueleto do crate, carregador de mapa e geometria

**Cria:**
```
crates/taxi-world/Cargo.toml
crates/taxi-world/src/lib.rs
crates/taxi-world/src/constants.rs      // toda a tabela da SPEC §3.2
crates/taxi-world/src/world.rs          // struct World + geometria
crates/taxi-world/src/trace.rs          // serde do traco_ref.json (só leitura)
crates/taxi-world/src/genome.rs         // Genome + formato TAXG
crates/taxi-world/tests/fixtures/tiny_trace.json   // fixture escrito à mão
```
Edita `Cargo.toml:8` do workspace (uma linha: acrescenta `"crates/taxi-world"`).

**Interface:**
```rust
pub const G: usize = 9;
pub const GRID: usize = 128;
pub const TEX: usize = 64;
pub const N_TYPES: usize = 5;
pub const N_SERVICES: usize = 10;
pub const DIRS: [[f32; 2]; G];               // SPEC §0, diagonal = FRAC_1_SQRT_2

pub struct Config { /* todos os campos da SPEC §3.2, f32 */ }
impl Default for Config { /* valores da SPEC §3.2 */ }

pub struct World { /* §3.2 deste plano */ }
impl World {
    pub fn terrain_cost_at(&self, x: f32, y: f32) -> f32;      // tex-1=63, round-ties-even
    pub fn district_at(&self, x: f32, y: f32) -> u32;          // GRID=128, truncamento
    pub fn nearest_services(&self, x: f32, y: f32) -> ([[f32;2]; 5], [f32; 5]); // ordem TIPOS
}

pub struct Trace { /* todas as chaves do JSON, incl. as emendas de T0 */ }
impl Trace { pub fn load(path: &Path) -> Result<Trace, TraceError>; }
impl Trace { pub fn to_world(&self) -> World; pub fn initial_beings(&self) -> Beings; }

pub struct Genome(pub Vec<f32>);   // len == NF + S + 1
impl Genome { pub fn save(&self, w: impl Write) -> io::Result<()>;
              pub fn load(r: impl Read) -> Result<Vec<Genome>, GenomeError>; }
```

**Testes de aceite** (`cargo test -p taxi-world`):
- `terrain_cost_uses_sixty_three_and_ties_to_even` — três pontos escolhidos à mão,
  incluindo um exatamente no meio de duas células (`x/L*63 == k + 0.5`), provando que
  `f32::round_ties_even` (Rust ≥ 1.77) é usado e não `f32::round`.
- `district_lookup_truncates_on_a_grid_of_one_twenty_eight` — inclui `x = L` (deve dar
  índice 127, pelo clamp) e `x = 0`.
- `texture_rows_are_indexed_by_y_not_x` — mapa fixture assimétrico; falha se transposto.
- `nearest_service_scans_every_point_of_its_type` — 3 pontos, o mais próximo é o do meio.
- `genome_survives_disk_round_trip` — grava e lê 40 f32, bit a bit.
- `genome_with_the_wrong_sense_count_is_refused` — cabeçalho com `nf = 27` numa build de
  29 devolve `Err`, não um genoma torto.
- `trace_fixture_loads_every_documented_key`.

**Não tocar:** nada fora de `crates/taxi-world/` e da linha `members` do `Cargo.toml`
raiz. **Não** editar `src/world.rs`, `src/config.rs`, `src/gpu/*`.

Estimativa: 1–1.5 sessão. ~450 linhas + fixture.

---

### T2 — Corpo e física do passo

**Cria:**
```
crates/taxi-world/src/beings.rs     // struct Beings (SoA) + construtores
crates/taxi-world/src/physics.rs    // step_physics()
```

**Interface:**
```rust
/// Executa os itens 1..28 e 32..35 da SPEC §3.1 (tudo menos passageiros).
/// `choice[b]` é o índice já decidido (0..=18). `t` é o contador APÓS o incremento.
pub fn step_physics(
    b: &mut Beings, w: &[World], cfg: &Config, t: i64,
    choice: &[u32],
    nearest: &NearestCache,       // _ptos/_dist do obs() DESTE tique (T3 produz)
    service_avail: &[f32],        // sv (n*10) do obs() deste tique (T4 produz)
    service_cost: &[f32],         // cs (n*10) deste tique (T4 produz)
);
pub fn update_daily_prices(w_price_mul: &mut [f32], w: &World, cfg: &Config, t: i64);
pub fn is_night(t: i64, phase: f32, cfg: &Config) -> bool;
pub fn hour(t: i64, phase: f32, cfg: &Config) -> f32;
```

Enquanto T3/T4 não existem, o implementador usa dublês triviais (`NearestCache` com
distâncias fixas, `service_avail` tudo zero) — os testes desta tarefa não dependem deles
para nada além de assinatura.

**Testes de aceite:**
- `staying_put_still_burns_idle_fuel` — `esc = 8` gasta `fuel_passo·0.33`, não zero
  (SPEC §13.11).
- `a_dry_tank_triggers_a_tow_to_the_nearest_station` — `fuel` abaixo do necessário ⇒
  `pos` vai para `nearest.ptos[0]`, `fuel = 15`, `stopped_ticks = 18`, dano
  `D_PANE + (20 − kcal)`.
- `overheating_resets_heat_and_skips_the_driving_penalty_that_tick` — SPEC §13.7:
  o `D_RODAR` lê o `calor` **já** resetado a 60.
- `exhaustion_parks_the_taxi_for_twelve_ticks_and_halves_fatigue`.
- `hunger_thirst_and_debt_damage_match_the_three_formulas` — três casos por fórmula,
  incluindo os pontos de saturação (`fome = 100`, `agua = 0`, `dinh = −60`).
- `rent_is_charged_once_per_simulated_day` — varre 480 tiques e conta exatamente 2 cobranças.
- `the_stopped_counter_is_decremented_before_being_overwritten` — SPEC §13.8.
- `reserve_is_only_spent_by_the_beings_that_actually_broke_down` — SPEC §13.9.

**Não tocar:** `senses.rs`, `decision.rs`, `services.rs`, `passengers.rs`.

Estimativa: 1.5–2 sessões. ~500 linhas (física) + ~250 (testes).

---

### T3 — Sentidos (F), nota linear, centragem G1 e argmax

**Cria:**
```
crates/taxi-world/src/senses.rs     // observe() -> F
crates/taxi-world/src/decision.rs   // score + G1 + mask + argmax
```

**Interface:**
```rust
pub const N_OP: usize = 10;
pub const N_BODY: usize = 19;
pub const NF: usize = N_OP + N_BODY;   // 29

pub struct NearestCache { pub points: Vec<[f32;2]>, /* n*5 */ pub dist: Vec<f32> /* n*5 */ }

/// F em layout [being][option][feature], comprimento n*9*NF. SPEC §4.
/// Preenche `nearest` (usado depois por physics e services NO MESMO TIQUE).
pub fn observe(
    b: &Beings, w: &[World], price_mul: &[f32], cfg: &Config, t: i64,
    target: Option<&[f32]>,          // Some => usa o `alvo` do traço (portão); None => calcula
    out_f: &mut [f32], nearest: &mut NearestCache,
);

/// sc = F·W[0..NF]; sc -= média das 9; ss = svn·W[NF..NF+S] + W[NF+S];
/// ss[a] = -1e9 onde sv[a] <= 0; argmax de [sc(9); ss(10)] com desempate no PRIMEIRO índice.
pub fn decide(
    f: &[f32], need: &[f32], avail: &[f32], genome: &Genome, n: usize,
    out_choice: &mut [u32], out_margin: &mut [f32], out_scores: &mut [f32],
) ;
```

`target: Option<&[f32]>` é o gancho que torna o portão de identidade possível apesar do
RNG (SPEC §7.2): no modo portão o Rust **lê** `alvo` do traço; no modo bateria calcula.

**Testes de aceite:**
- `option_features_follow_the_zero_one_two_four_three_order` — um mapa fixture com um
  ponto de cada tipo em posições distintas; falha se oficina e descanso trocarem
  (SPEC §4.1).
- `body_features_are_identical_across_the_nine_options`.
- `each_of_the_twenty_nine_columns_matches_its_formula` — um caso por coluna, valores
  calculados à mão da SPEC §4.
- `the_common_mode_of_the_body_cancels_after_centering` — soma o mesmo valor nas 9 opções
  e confere que `sc` não muda (é o achado E58 da SPEC §4.3).
- `a_tie_in_the_scores_picks_the_lowest_index`.
- `masked_services_never_win` — todas as 10 notas de serviço em `-1e9` ⇒ escolha < 9.
- `the_margin_is_the_gap_between_the_best_and_the_runner_up`.

**Não tocar:** `physics.rs`, `passengers.rs`.

Estimativa: 1.5–2 sessões. ~400 linhas + ~300 de teste.

---

### T4 — Serviços e passageiros

**Cria:**
```
crates/taxi-world/src/services.rs     // availability + cost + need
crates/taxi-world/src/passengers.rs   // pickup, delivery, fare, respawn
```

**Interface:**
```rust
pub const ACTIONS: [(ServiceKind, f32); 10];   // SPEC §6

/// SPEC §6.1 e §6.2. `nearest` vem do observe() do MESMO tique (distância da posição ANTIGA).
pub fn service_state(
    b: &Beings, w: &[World], price_mul: &[f32], nearest: &NearestCache,
    cfg: &Config, t: i64,
    out_avail: &mut [f32], out_cost: &mut [f32], out_need: &mut [f32],   // n*10 cada
);

/// SPEC §7.2 e §7.4. Itens 29 e 31 da SPEC §3.1.
pub fn step_passengers(
    b: &mut Beings, w: &[World], price_mul: &[f32], cfg: &Config,
    free: &[bool], acting_service: &[bool], rng: Option<&mut ChaCha8Rng>,
);
```

`rng: Option<&mut _>` — `None` no modo portão (respawn desligado, SPEC §7.3);
`Some` na bateria. **No modo portão a vaga usada é apenas marcada `px_ok = 0`; `px`/`pxd`
não são reescritos** (SPEC §7.2: o valor sorteado pelo Python nunca é lido de volta,
exceto por `alvo()`, que T3 resolve lendo do traço).

**Testes de aceite:**
- `the_shop_at_the_station_sells_food_and_water_in_that_order` — tabela-verdade dos 4
  casos, cobrindo o encadeamento `agua |= posto | comida_já_ampliada` (SPEC §6.1).
- `the_fuel_station_is_closed_at_night`.
- `sleeping_at_home_is_free` — `distrito(pos) == casa` ⇒ custo do descanso = 0.
- `the_refuel_need_splits_between_tank_and_reserve` — os 5 valores de `nec` para
  `q ∈ {25,50,75,100,150}` com `fuel = 60`, `res = 100`.
- `a_service_you_cannot_afford_is_not_offered` — `paga = 0` ⇒ `avail = 0`.
- `overflow_beyond_a_hundred_liters_goes_to_the_reserve_minus_fifteen_percent`.
- `a_passenger_can_board_and_be_delivered_in_the_same_tick` (SPEC §7.4).
- `the_fare_uses_the_district_of_the_pickup_not_of_the_drop_off`.
- `a_used_slot_never_comes_back_when_respawn_is_off`.

**Não tocar:** `senses.rs`, `physics.rs`.

Estimativa: 1.5 sessões. ~350 linhas + ~300 de teste.

---

### T5 — Harness de identidade contra `traco_ref.json`

**Cria:**
```
crates/taxi-world/src/tick.rs             // o tique completo (une T2+T3+T4)
crates/taxi-world/src/gate.rs             // comparador + classificação por margem
crates/taxi-world/src/bin/taxi_identity.rs
crates/taxi-world/tests/identity.rs
```

**Interface:**
```rust
/// Um tique completo, na ordem da SPEC §3: observe → decide → step_physics → passageiros.
pub fn step(st: &mut Sim, target: Option<&[f32]>) -> StepRecord;

pub struct GateReport {
    pub actions_equal: usize, pub actions_total: usize,
    pub first_divergence: Vec<Option<usize>>,   // por ser
    pub bugs: Vec<Divergence>,                  // margem > 1e-2
    pub grey: Vec<Divergence>,                  // 1e-3 .. 1e-2
    pub max_abs: BTreeMap<&'static str, f32>,   // por variável de estado
}
pub fn run_gate(trace: &Trace, level: GateLevel) -> GateReport;
pub enum GateLevel { Senses, Score, FirstDay, FullLife }
```

Portões P0..P3 exatamente como a SPEC §11.2. Classificação de divergência pela `margem`
do traço (SPEC §11.2): `< 1e−3` empate aceitável; `1e−3..1e−2` cinzento (reporta);
`> 1e−2` **BUG**.

Na falha, o relatório imprime `t`, `b`, `esc_py`, `esc_rust`, os 19 valores de `tudo` dos
dois lados e o `argmax` da diferença por coluna de `F`. **Um portão que diz "falhou" sem
dizer onde não conta como entregue.**

**Testes de aceite:**
- `the_tiny_fixture_reproduces_its_own_trace` — roda contra
  `tests/fixtures/tiny_trace.json` (2 seres, 3 tiques, valores calculados à mão da SPEC).
  **Sempre roda**, mesmo sem o `traco_ref.json`.
- `a_divergence_below_the_margin_is_reported_as_a_tie_not_a_bug` — injeta uma divergência
  sintética com margem 1e−4 e confere a classificação.
- `senses_match_the_python_reference_on_the_first_tick` — `#[ignore]`, P0.
- `scores_and_the_first_choice_match_the_python_reference` — `#[ignore]`, P1.
- `every_action_of_the_first_simulated_day_matches_python` — `#[ignore]`, P2.
- `a_full_life_diverges_only_on_numerical_ties` — `#[ignore]`, P3.

Os quatro `#[ignore]` seguem a convenção do repo para testes que dependem de recurso
externo (`CLAUDE.md:17-24` usa o mesmo mecanismo para os testes de GPU). Comando:

```bash
CARGO_BUILD_JOBS=4 nice -n 10 cargo test -p taxi-world --release -- --ignored
```

**Depende de T0** (o arquivo) para os quatro `#[ignore]`; **não depende** para o resto.

**Não tocar:** os módulos de T1..T4 além do que a interface expõe.

Estimativa: 2 sessões. ~350 linhas + o fixture à mão (o custo real está no fixture).

---

### T6 — Laço GA e juiz

**Cria:**
```
crates/taxi-world/src/evolution.rs
crates/taxi-world/src/judge.rs
crates/taxi-world/src/bin/taxi_evolve.rs
```

**Interface:**
```rust
pub struct GaParams {           // SPEC §10 — defaults do cenarios/mundoA_seres.json
    pub pop: usize,             // 128
    pub generations: usize,     // 30
    pub patience: Option<usize>,// 15
    pub replicas: usize,        // 4
    pub seeds: Vec<u64>,        // [0,1,2,3,4]
    pub sigma_mut: f32,         // 0.1
    pub elite_frac: f32,        // 0.15
    pub clamp: f32,             // 5.0
    pub ticks: usize,           // 7200
}
pub fn evolve(worlds: &[World], cfg: &Config, p: &GaParams) -> BTreeMap<u64, Champion>;

pub struct Measures { /* fit, deaths, deliveries, breakdowns, ..., life: LifeStats */ }
pub struct LifeStats { pub mean: f32, pub median: f32, pub p10: f32, pub p90: f32, pub ceiling_frac: f32 }
/// SPEC §9: 128 cópias, semente 7777, R=1, sem jardim comum.
pub fn judge(g: &Genome, worlds: &[World], cfg: &Config, ticks: usize) -> Measures;
```

Régua: `fit = alive_ticks/240 + w_trab · revenue/300` (SPEC §8). Só a régua `vida`.

**Testes de aceite:**
- `the_elite_passes_untouched_to_the_next_generation` — os `max(2, ⌊P·0.15⌋)` primeiros
  são idênticos bit a bit.
- `every_gene_of_a_child_is_mutated_and_clamped_to_plus_minus_five`.
- `the_patience_window_compares_the_best_of_the_window_against_the_best_before_it` —
  série sintética de fitness; para na geração certa (SPEC §10, item 6).
- `fitness_averages_over_maps_and_replicas_for_each_genome` — com `P=2, NC=2, R=3` e
  fitness sintético, confere `b = p·(NC·R) + c·R + r`.
- `a_life_that_hits_the_horizon_is_flagged_by_the_ceiling_fraction` — `ceiling_frac > 0.05`
  precisa aparecer no relatório (SPEC §9).
- `the_judge_never_shares_the_evolution_seed` — o juiz usa 7777, a evolução `777000+13·it`.

**Não tocar:** `gate.rs`, `senses.rs`, `physics.rs`.

Estimativa: 1.5 sessões. ~400 linhas + ~250 de teste.

---

### 4.1 Grafo

```
T0 (Python) ─────────────────────────────────┐
                                             ▼
T1 (crate + mapa) ─┐                    T5 (harness) ── portão
T2 (física)        ├─ interfaces do §4 ─┤
T3 (sentidos)      ├────────────────────┘
T4 (serviços/pax) ─┘
T6 (GA + juiz) ────── independente das 5 (usa só `step` e `Genome`)
```

Ordem de mérito para 3 implementadores em paralelo: **(T1, T3, T4)** primeiro (são os que
outros leem), depois **(T2, T5, T6)**. T0 pode sair a qualquer momento e não colide com
ninguém.

---

## 5. Riscos e como o portão os pega

| # | risco | como se manifesta | quem pega |
|---|---|---|---|
| 1 | **Ordem de redução do produto interno de 29 termos.** `torch.einsum` no ROCm usa uma ordem própria; o Rust soma sequencialmente | `sc` difere ~3e−4; em ~1 tique de cada milhares isso vira `argmax` diferente | P1 (mede `Δsc` no tique 0) e P3 (classifica pela margem: `< 1e−3` = empate) |
| 2 | **`round` do torch é ties-to-even; `f32::round` do Rust é ties-away-from-zero.** Afeta `custo_em` (SPEC §2.3) | um ser numa fronteira de célula lê custo 1.0 em vez de 3.0 ⇒ combustível, calor e nota mudam | P0 (`ΔF` na coluna `custo_frente`/`custo_aqui`) e o teste `terrain_cost_uses_sixty_three_and_ties_to_even`. **WGSL `round` já é ties-to-even por spec** — o cuidado é só no Rust (`f32::round_ties_even`) |
| 3 | **RNG.** O Philox do torch não é reproduzível em Rust nem em WGSL | trajetória diverge no 1º embarque | resolvido por desenho: v0 lê mapa e estado inicial do JSON; a vaga usada é só desativada (SPEC §7.2); e `alvo` vem do traço no modo portão (T0-a, T3 `target`) |
| 4 | **`alvo()` ignora `px_ok`** (SPEC §7.2) | sem a emenda T0-a o portão morre em < 50 tiques e o diagnóstico aponta o lugar errado | T0-a é bloqueante para P2/P3; se T0 não sair, P0/P1 ainda fecham |
| 5 | **Contração de ponto flutuante no WGSL** (`fma`, `dot`) | o v1 diverge do v0 sem nenhum bug lógico | P0..P3 rodam idênticos no v1; mitigação: nas notas, somar explicitamente em cadeia e **não** usar `dot`/`fma` dentro do laço de 29 termos até o portão fechar |
| 6 | **Alinhamento de 16 bytes do WGSL em array de struct** — foi o bug do `INPUT_WGSL_STRIDE = 368` (`src/gpu/physics.rs:172-183`) | corrupção silenciosa do ser seguinte | eliminado por desenho: SoA (§3.1), `Vec<f32>`/`Vec<u32>` puros, sem struct em storage |
| 7 | **`obs()` vê `t`, `passo()` vê `t+1`** (SPEC §3) | o ser abastece à noite (posto fechado) ou não abastece de dia, em 2 tiques por dia | P2 pega no tique 240 ou antes; a série de `esc` diverge exatamente na virada de hora |
| 8 | **Índice do lote `b = p·(NC·R) + c·R + r`** | genoma avaliado no mapa errado ⇒ resultado plausível e falso | `fitness_averages_over_maps_and_replicas_for_each_genome` (T6) |
| 9 | **`NF` depende de variável de ambiente do lado Python** (`NM_RISCO`/`NM_PONTO`, SPEC §4.2) | traço com 26 colunas contra build de 29 | T0-f fixa no arquivo; T1 recusa o genoma pelo cabeçalho `nf` |
| 10 | **Respawn espúrio no traço** (`respawn = 1e9` ⇒ ~0.1 % de chance em 460 800 sorteios, SPEC §7.3) | uma divergência isolada com margem grande, num tique só | o relatório de P3 aponta o tique; regerar o traço resolve. Não afrouxar a tolerância por causa disso |
| 11 | **Bevy entrando pela porta dos fundos** — `universe` linka Bevy sempre | `cargo test -p taxi-world` demora minutos e o portão deixa de ser barato | `taxi-world` **não** depende de `universe`; um teste de CI (`cargo tree -p taxi-world | grep -q bevy` deve falhar) trava isso |
| 12 | **Regressão no caminho da corrida.** O táxi não toca `world.rs`, `motor.rs`, `gpu_evolve` nem os `.wgsl` existentes | — | `cargo test --release --bin gpu_evolve` e `cargo test --release --test wgsl_validation` continuam verdes sem nenhuma mudança; se algum falhar, alguém saiu do escopo |

---

## 6. Estimativa

| tarefa | sessões Sonnet | linhas de código (aprox.) |
|---|---|---|
| T0 emendas do traço (Python) | 1 curta | 15 + 30 (validador) |
| T1 crate, mapa, geometria, genoma | 1.0–1.5 | 450 + fixture |
| T2 corpo e física | 1.5–2.0 | 500 + 250 teste |
| T3 sentidos, nota, argmax | 1.5–2.0 | 400 + 300 teste |
| T4 serviços e passageiros | 1.5 | 350 + 300 teste |
| T5 harness de identidade | 2.0 | 350 + fixture à mão |
| T6 GA e juiz | 1.5 | 400 + 250 teste |
| **v0 total** | **10–11** | **~2 450 + ~1 100 de teste** |
| v1 (WGSL, depois de P0..P3 fecharem) | 3–4 | ~400 WGSL + ~350 host |

O v1 é barato porque o tique do táxi é **um dispatch**, não os seis encadeados do carro
(`context.rs:752-838`): sem retina, sem percepção espacial/temporal/motion, sem digest,
sem cérebro lento, sem os três kernels de física. Um `@workgroup_size(64)`, uma invocação
por ser, o mapa em buffers read-only e o estado em SoA read_write.

---

## 7. O que fica fora deste plano — e por quê

| fora | por quê | quando |
|---|---|---|
| retina, regime, reprodução, prefeito, ponto de táxi, jardim comum, temperatura de decisão, custo de gene | SPEC §12 — cada um muda `NF` ou consome RNG por tique, e mata o portão | v2, um por vez, cada um com seu portão "desligado ⇒ idêntico" |
| cérebros não-lineares (`Reflexo`, `Produto`, `Modos`, `Cerebro`, `SerCompleto`) | o portão precisa de UM cérebro; o `Linear` é a espinha | v1, um por vez, com portão "genes 0 ⇒ idêntico ao Linear" |
| geração própria do mapa em Rust | `gerar_mapa` usa `torch.rand` + `conv2d` de suavização; irreproduzível bit a bit (SPEC §2.2) | v1, com portão **estatístico** (distribuição de custo, contagem de serviços por perfil), nunca de identidade |
| pilha `(C, GR, GR)` de canais e `varredura = 'grade'` | é a otimização de `docs/70` §2.2 (13–21×, erro 0.37 m), não a spec | v1, depois do portão, com tolerância própria declarada |
| UI Bevy do mundo do táxi | o Universe não tem câmera 2D nem sprites; tudo é `Mesh3d` (`src/world.rs:707`). O caminho barato é `bevy_gizmos` (`Cargo.toml:32`) sobre uma câmera ortográfica | v1, como `TaxiWorldPlugin` no crate `universe`, dependendo de `taxi-world` |
| integração com `UGEN v2` / `.dna` V8 / o GA de `src/gpu/evolution.rs` | parâmetros e formato incompatíveis (§1.1) | não planejado. Se for pedido, é uma ponte de conversão, não uma unificação |
| migração do táxi para dentro de `gpu_evolve` ou de `motor.rs` | os dois runtimes já não compartilham tique (`ARCHITECTURE.md:41`); um terceiro isolado é a escolha honesta | não planejado |

## Decisão do dono (10/09 11:10)
Sem Bevy, sem visual: "a parte visual com Bevy não é necessária; era usada para os sensores do carro (CPU–GPU); hoje tudo roda na
GPU no modo headless; com replay não precisa mais". Logo: v0 = tique na CPU SÓ como portão de identidade contra o traço do Python;
v1 = tique inteiro em WGSL headless (wgpu compute, padrão do `gpu_evolve`), replay em arquivo para olhar. `TaxiWorldPlugin` (Bevy)
sai do plano.
