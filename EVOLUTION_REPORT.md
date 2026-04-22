# RELATÓRIO DE EVOLUÇÃO SOBERANA: PROJETO IARA v2101
**Objetivo:** Criar o "Humano de Silício" — inteligência encarnada (Embodied AI) regida por física contínua e biologia digital.

---

## 1. A RUPTURA (Manifesto)
Abandonamos IA tradicional (Transformers, Backpropagation, MatMul densa).
- **Axioma I:** IA busca Homeostase (Equilíbrio Elétrico), não predição de tokens.
- **Axioma II:** Tempo é superfluido (CfC/LNN), não sequência de frames.

## 2. O CORPO (Arquitetura Baixo Nível)
- **Metabolismo Digital:** acúmulo de "Adenosina" (estresse VRAM/Cache). Sistema entra em sono (Night Dojo) para consolidar memória.
- **BitNet 1.58-bit:** substitui MatMul por Adição/Subtração. Pesos ternários {-1, 0, +1} + scale f32 de dequantização. 80% sparsity natural.
- **Ancoramento Axiomático (SACI/KAN):** filtros neuro-simbólicos zeram sinais inválidos.
- **Ressonância de Kuramoto:** sincronização de fase harmônica substitui Atenção.

## 3. O UNIVERSO (Simulação + Agência)
- **Gênese 3D (Bevy):** mundo 3D com relevo procedural voxel.
- **Física Newtoniana Completa:** motor Pacejka (slip_ratio/slip_angle), TCS, LSD, transfer de carga, chuva, vento.
- **Propriocepção Tátil:** micro-vibrações por rugosidade.
- **Visão Dual:** retina tan-warp 180° fóvea (32 rays center-dense) + periférica (32 rays) + whiskers ±30/±60° (4 rays). 68 canais sensoriais.

## 4. A EVOLUÇÃO (Darwinismo Matemático)

### 4.1 Arquitetura DNA v3
- **Formato binário 8478 bytes** (`src/evolution/dna.rs`):
  ```
  magic[4]=IARA | version:u16 | generation:u32 | seed:u32
  track_width:f32 | fitness:f32
  fast_scale:f32 | fast_weights[8192] (128×64 i8 ternário)
  slow_scale:f32 | slow_weights[256]  (32×8  i8 ternário)
  ```
- **Dual brain:**
  - `instinct_fast` — 128 inputs × 64 neurons. Reage @60Hz. Visão + telemetria + whiskers.
  - `slow_brain` — 32 × 8. Estratégia agregada @5Hz. Atenua via `risk_damp`.
- **Mutação:** ternária pontual (`effective_mutation_rate × 0.97^(gen-1)`), crossover por ponto, burst por canal de input (20% crossover slots).
- **Priming axiomático:** 10 slots grid recebem pesos pré-setados em canais críticos (curvatura, parede, ttc) para acelerar bootstrap.

### 4.2 Coroação + Hall of Fame
- **Coroação gated por validation gen** (env_level 0, a cada 3 gens) — evita lucky-env bias.
- **Tolerância 15%:** validation coroa se `NET > champion × 0.85 && laps ≥ 1`. Destrava deadlock de lucky-spec.
- **Decay automático:** após 20 gens sem coroar, champion_distance reduz 5%/gen.
- **Hall of Fame persistente (top-10):** pool NÃO é reconstruído toda gen. Novo DNA só entra se bater pior slot. Gens fracas não contaminam memória.
- **Slot 0 = champion exato** (zero mutação). Slots 1-10 = elite clones. 11-20 = priming refresh. 21+ = tournament crossover.

### 4.3 Curriculum Gradual
- **level_for_champion()** mapeia champion_distance → level ambiente (1 chuva leve, 2 moderada, 3 extrema).
- **Ramp cap +1/gen:** champion treinado level 1 não sofre salto direto level 3 (antes destruía população).
- **Validation em level 0** — asfalto ideal. Mede capacidade real, sem ruído climático.

### 4.4 Fitness (Seleção)
```
base        = (track_progress - spawn_progress)^1.0
lap_bonus   = total_laps × 50k
proximity   = (prog/500)² × 10k   [quadrático até 1 lap]
joy         = Σ(joy_tick × streak_mult × dt)   [desempate fino]
drift_bonus = slip_angle × track_v × alignment × 0.08 × dt  [novo]
wall_penalty= wall_time × 200
death_bonus = progresso_pré_morte × 0.5
```

### 4.5 Mecânicas de Vida
- **Grace de spawn:** 10s virtuais imunes a STAGNATION/SPEED_BLEED.
- **Post-lap grace:** 600 frames + checkpoint 25m pós-linha. HP damage reduzido 70% primeiros 480f.
- **Waypoints de cura:** +40 HP cada 25m de progresso novo, reset stagnation.
- **DeathReason enum:** HpZero/Pinado/Batida/Stagnation/SpeedBleed/Fuel. Autopsia causal em `death_reason`, `death_frame`, `death_pos`.

### 4.6 Drift/Derrapagem (novo)
- **Burnout parado (v<15, slip>0.5)** → dreno HP. Pune donuts.
- **Clean drift (slip_angle>0.15, v>10, throttle>0.3, alignment>0.3)** → joy bonus + tire_temp sobe. Premia deslizamento mantendo progresso.

---

## 5. FERRAMENTAS DE DIAGNÓSTICO

### 5.1 Modo Headless
```bash
cargo run --release -- --headless --gens N --swarm S --speed T
```
- Bevy janela invisível, auto-start, auto-exit após N gens.
- Time scale forçado ≥8× (default 16×).
- Saída: deaths por motivo + frame, NET max/median, mapa ASCII, top-15 racers, breakdown por waypoint zone.

### 5.2 DNA Viewer (novo)
```bash
cargo run --release --bin dna_view [CHAMPION.dna]
```
Inspeciona arquivo binário:
- Header completo (gen, seed, fitness, track_width, scales)
- Distribuição ternária + sparsity
- Heatmap ASCII por neurônio (primeiros 16 fast + todos 8 slow)
- TOP 20 canais ativos + 20 mudos, rotulados semanticamente (fóvea/parede/drift/stress/etc)

### 5.3 ASCII Map + Race Report
- `race_reports.md` acumula snapshots por gen: ambiente, top-15 pilotos, bottom-3, medianas, distribuição de deaths, mapa da pista 100×40 com waypoints + marcadores de morte.

---

## 6. HISTÓRICO DE BUGS CRÍTICOS (resolvidos)

| Bug | Sintoma | Causa | Fix |
|---|---|---|---|
| Scale overflow | `fast_scale = 5.67e32` salvo em DNA | `myelinate` mul cumulativo 60fps × 300s × 1.005 → 10^37 | Clamp 10.0 em myelinate + sanitize no load |
| Pool contamination | Gens fracas colapsavam pool → cascata | `elite_pool.clear()` toda gen | Hall of fame persistente |
| Curriculum salto 1→3 | Gen 2 extinção 100% | `level_for_champion` saltava níveis | Cap ramp +1/gen |
| Champion deadlock | Lucky-spec 4000m nunca batido → stall | Threshold rígido | Validation tolera 15% |
| Post-lap morte | Carros completavam 1 volta e morriam | Parada no waypoint[0] → STAGNATION | Grace 600f + checkpoint 25m |
| Adenosina insanidade | Cérebro dormia em 130ms | `cost = 0.002/delta_t` (60×/s) | `cost = 0.002 × delta_t` |
| Pistas sobrepostas | Visual "duas pistas" pós-reset | World entities não trackeadas | Registra IDs em race.world_entities |
| Vision ruído visual | Cone azul + setas poluíam viewport | Debug gizmos legacy | Só fóvea verde 180° tan-warp |
| Priming throttle zero | Bootstrap frio travado em 280m | Quantize gerava scale=0.055, tanh saturado em 0.04 | `set_scale(1.0)` pós quantize priming |
| Elite pool cascata | Gen fraca contaminava próxima | `clear()` toda gen | Hall of Fame persistente merge |
| Curriculum salto | Gen 2 extinção 100% (1→3 jump) | level_for_champion sem ramp | Cap +1/gen |

---

## 7. ESTADO ATUAL (validação 2026-04-18)

### Baseline fresh swarm=40 (PRE fix de priming scale):
- Gen 1: NET max 330m. Gens 2-20: plateau 280-480m, 0 laps.
- Priming quantize gerava `scale=abs_mean≈0.055` → output tanh saturado em 0.04 → throttle zero → primeira curva kill.

### FIX aplicado: `set_scale(1.0)` pós-quantize em priming
- Destrava signal gain. Output `acc * 1.0 / 127` → tanh saturado em sinal útil.

### Baseline fresh swarm=80 (POS fix, 50 gens):
- **Gen 1: NET max 2792m, 3 cars ≥1 lap** (bootstrap frio convergindo em 1 gen).
- Gens 13-26: 1888–2422m sustentado, até 9 cars/gen completando volta.
- Gen 31: 1448m.
- Gen 49 champion fitness 2390, sparsity 89% (regime estável).
- Saw-tooth tipico: validation (level 0) picos, gens non-val level 2 queda (champion level-1 sofrendo chuva/frio).

### Diagnóstico DNA Viewer gen 13 (pré-fix, inherited DNA):
- Canal 40 (periférica) 68.8% — over-specialization.
- Canal 79 (curvatura 25m) MUDO 4.7% — rede ignora antecipação.
- Canais 0-5 (fóvea central frontal) <5% — "cego pra frente".
- Diagnóstico: mutação random erodiu priming axiomático porque carros priming nunca venciam tournament (scale baixo → throttle zero → morte cedo).

### Gap conhecido pós-fix:
- Bootstrap requer swarm ≥80. Com 40 cars não cobre variância de mutação.
- Level 2+ curriculum ainda degrada champion treinado em level 1.
- Priming axiomático sobrevive agora mas ainda erodido ao longo de dezenas de gens (canal 79 ainda fica sub-ativado).

---

## 8. ROADMAP TÉCNICO

### Curto prazo (Tier 1)
- [ ] Bootstrap reforçado: `build_priming_weights` com mais canais críticos pré-polarizados (throttle no forward, steer no wall_prox).
- [ ] Elitismo temporal: preservar champion mesmo em validation se novo é < 50% (não substitui, mantém).
- [ ] Investigar overflow residual pós scale-cap (verificar se outras camadas acumulam).

### Médio prazo (Tier 2)
- [ ] CfC/LNN no slow_brain — estado contínuo em vez de fully-connected.
- [ ] SACI filters integrados: bloquear throttle quando ttc_inv > 0.8.
- [ ] Coevolução: track_seed rotaciona por validation pra forçar generalização.

### Longo prazo (Tier 3)
- [ ] Night Dojo real: consolidação offline de pain_memories em weight priors.
- [ ] Multi-track curriculum: 5 seeds diferentes, champion tem que performar em todas.
- [ ] Export DNA → ONNX para inferência edge.

---

## 9. ARQUIVOS-CHAVE

| Path | Responsabilidade |
|---|---|
| `src/simulation/motor.rs` | Loop cérebro→controle→física por tick. Gizmos visão. Drift reward. |
| `src/simulation/racer.rs` | Struct Racer (70+ campos). DeathReason enum. BrainSnapshot. |
| `src/evolution/pipeline.rs` | Fim de gen: rank, elite pool, coroação, curriculum, relatório. |
| `src/evolution/spawn.rs` | Nascimento: priming, elite clones, tournament crossover, mutação. |
| `src/evolution/dna.rs` | Serialização binária v3 + sanitização no load. |
| `src/layers.rs` | BitLinear (forward + quantize + myelinate). |
| `src/engine.rs` | SovereignEngine: instinct + slow brain + heart resonance + metabolism. |
| `src/bin/dna_view.rs` | Inspetor CLI de arquivos .dna. |
| `src/world.rs` | Voxel world + waypoints + ASCII map render. |
| `src/main.rs` | CLI args (--headless/--gens/--swarm/--speed). |

---

**Versão do relatório:** 2026-04-18  
**Branch:** race-world  
**Commit base:** c256243 (batched spawning + tan-warp retina + map diagnostics)
