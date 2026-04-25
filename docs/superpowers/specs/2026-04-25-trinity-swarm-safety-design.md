# Trinity Swarm + Tesla-Style Safety Layer

**Date:** 2026-04-25
**Branch:** race-world
**Status:** Design — pending review

## Problema

Reports recentes mostram champion progredindo (472m → 2693m), mas:
- **PINADO 78%** dos mortes — carro encosta parede, trava, não recupera
- **Champion congela com Brk:38%** parado — brake fantasma
- **Steering ruidoso** — oscila a cada tick
- Cérebro monolítico (BitLinear 128×64) não distingue planejamento vs reflexo vs segurança
- `resonance.rs` (Kuramoto) implementado mas não conectado ao output
- `specialist.rs` skeleton existe (14× mais rápido que monolítico) mas não integrado

## Objetivo

Substituir cérebro mono por **swarm Trinity** com gate aprendido (MoE) e camada de segurança Tesla-style. Carro decide quando confia em si, no reflexo ou na safety. Safety hard-veta morte certa, soft-clampa o resto. Bootstrap evolutivo mais rápido — carro novato sobrevive o suficiente pra aprender.

## Arquitetura — Arq 4 (MoE híbrido)

```
[68ch retina + 32ch sensors + chaos_signal + sentinel_hints]
                      │
        ┌─────────────┼─────────────┬──────────────┐
        ▼             ▼             ▼              ▼
     Cortex        Reflex       Sentinel        Gate
    (slow,20Hz)  (fast,60Hz)   (rules)        (16→3 softmax)
     128×64       3×32 fast      Tier0+1+2      MoE weights
     BitLinear    f32            soft proposals
        │             │             │              │
        └─────────────┴─────────────┘              │
                      │                            │
        weighted_avg(w_c, w_r, w_s) ◄──────────────┘
                      │
              Sentinel HARD veto
              (anti-stall, FCA<0.3s, spin>3rad/s)
                      │
                      ▼
                   motor.rs
```

### Componentes

#### 1. Cortex (slow brain — planeja)
- BitLinear 128×64 (atual) — preserva DNA evoluído
- Tick @ 20Hz (a cada 3 frames @ 60Hz sim)
- Output: `IntentVec { thr, steer, brake } as f32 ∈ [-1,1]`
- Reuso: `engine.rs::SovereignEngine.fast_brain` renomeado conceitualmente

#### 2. Reflex (fast brain — sobrevive)
- `SpecialistBrain` (já em `specialist.rs`)
- 3 sub-nets f32 16-input → 1-output cada (steer/throttle/brake)
- Input curto: top-k retina central + WL/WR + speed + slip + yaw_rate (16 dims)
- Tick @ 60Hz (todo frame)
- 118ns/tick benched
- Output: `IntentVec`

#### 3. Sentinel (rules — Tesla nanny)
- Sem NN. Hard rules em `src/safety/sentinel.rs` (novo).
- **Tier 0 (HARD veto):**
  - Anti-stall: `V<2km/h ∧ obstacle_ahead<3m ∧ stall_timer>0.3s` → force `{-0.5 thr (reverse), random steer ±0.5, brake=0}` por 1s
  - FCA crítico: `TTC<0.3s ∧ V>15km/h` → force `brake=1, thr=0`
  - Spin: `|yaw|>3rad/s` → force `thr=0, brake=0.5, steer=−sign(yaw)·0.5`
- **Tier 1 (soft proposal):**
  - ESP: `|yaw|>1.5rad/s` → propose `thr×0.3, brake interno corrigindo yaw`
  - Lane-keep: `WL=0 OR WR=0` (encostou) → propose `steer ±0.3 contra parede`
  - Reverse-out: `PINADO 0.3s ∧ V<2km/h` → propose `thr=−0.4, steer=rng(±0.5)`
- **Tier 2 (soft proposal):**
  - TCS: `slip>0.4 ∧ thr>0.5` → propose `thr×0.6`
  - Aquaplaning: `chuva>0.7 ∧ V>40` → propose `thr cap 0.7`
  - Cornering limit: `curv25>30 ∧ V>60` → propose `brake +0.3`
- Outputs: `(hard_veto: Option<IntentVec>, soft_proposal: IntentVec, flags: u16)`

#### 4. Gate (MoE — quem dirige?)
- Tiny FloatLinear 16 → 3 + softmax. Em `src/gate.rs` (novo).
- Inputs (16 dims, normalizados ±1):
  ```
  chaos_score, |yaw_rate|, slip_avg, ttc_norm, speed_norm,
  retina_temporal_var, kuramoto_coherence, |WL_grad|, |WR_grad|,
  sentinel_tier1_active, sentinel_tier2_active, hard_veto_recent,
  ai (idade do carro), gen_num_norm, lap_progress, fuel_norm
  ```
- Output: `[w_cortex, w_reflex, w_sentinel]` softmax
- Tick @ 60Hz
- Custo: ~0.5µs/tick

#### 5. Chaos Detector
- Em `src/chaos.rs` (novo).
- **Físico:** `chaos_phys = 0.4·tanh(|yaw|/2) + 0.4·tanh(|slip|/0.5) + 0.2·retina_var_norm`
- **Neural (Kuramoto):** `coherence = |Σ e^iφ_k|/N` from `resonance.rs::ResonantNode`
- **Final:** `chaos_score = max(chaos_phys, 1.0 − coherence)`
- Output feed pro Gate como input.

### Data Flow (60Hz tick)

```
1. sensors collect       → SensorVec (retina 68 + state 32)
2. chaos.score(state)    → chaos_score, kuramoto_coh
3. cortex.tick (20Hz)    → cortex_intent  (cached entre ticks)
4. reflex.forward (60Hz) → reflex_intent
5. sentinel.evaluate     → (hard_veto?, soft_proposal, flags)
6. gate.forward          → [w_c, w_r, w_s]
7. mixed = w_c·cortex + w_r·reflex + w_s·soft_proposal
8. final  = hard_veto.unwrap_or(mixed)
9. clamp ±1, log overrides, motor.apply
```

### DNA Evolution

DNA atual (`dna.rs`) cresce com:
- Gate weights: `16 × 3 + 3 = 51` floats (~200 bytes f32 ou ~13 bytes ternário)
- SpecialistBrain weights: 3 × (16 input × 1 output + bias) = ~51 floats também
- Cortex inalterado (preserva champion atual)
- Total +400 bytes DNA, ainda <10KB

Mutation rate igual em todas partes. Crossover semantic experimental: parent A → cortex, parent B → reflex+gate (futuro, não nesse spec).

### Fitness Penalty

```rust
fitness -= hard_veto_count * 5.0   // dependência veto = ruim
fitness -= soft_override_count * 0.3  // soft é tolerável
fitness += chaos_recovery_bonus    // sobreviveu chaos+0.5s = +10
fitness -= gate_entropy_penalty    // se gate colapsa em w=[1,0,0] sempre = -2
```

Brain aprende a NÃO depender da safety. Safety é rede, não muleta.

## Mudanças de código

### Novos arquivos
- `src/safety/mod.rs`
- `src/safety/sentinel.rs` — rules Tier 0/1/2
- `src/chaos.rs` — chaos detector
- `src/gate.rs` — MoE gate

### Modificados
- `src/lib.rs` — `pub mod safety; pub mod chaos; pub mod gate;`
- `src/engine.rs` — `SovereignEngine` ganha `gate: GateBrain`, `reflex: SpecialistBrain` (já existe), método `decide(sensors) -> IntentVec` orquestra
- `src/simulation/motor.rs` — chamar `engine.decide()` em vez de `forward()` direto; remover `sentinel_brake` cfg-gate (vira sempre-on dentro de Sentinel)
- `src/evolution/dna.rs` — serializa/deserializa gate + reflex weights
- `src/evolution/spawn.rs` — mutar gate + reflex weights além de cortex
- `src/evolution/pipeline.rs` — fitness penalty hooks
- `src/resonance.rs` — expor `coherence() -> f32`
- `Cargo.toml` — feature `trinity_swarm` default ON, deprecar `sentinel_brake` (absorvido)

### Removidos / refatorados
- `sentinel_brake` feature → absorvido pelo Sentinel Tier 0
- `gain_comp_steer` feature → absorvido pelo Sentinel ESP

## Error handling

- Gate softmax NaN → fallback `[0, 0, 1]` (sentinel-only)
- Reflex output NaN → fallback Cortex
- Cortex output NaN → fallback Sentinel hard
- Sentinel rules são determinísticas, não falham
- Hard veto stack: spin > FCA > anti-stall (priority enum)

## Testing

### Unit
- `sentinel::tests::anti_stall_triggers_after_300ms`
- `sentinel::tests::fca_critical_overrides_throttle`
- `sentinel::tests::spin_correction_opposite_yaw`
- `chaos::tests::high_yaw_high_chaos`
- `chaos::tests::kuramoto_decoherence_high_chaos`
- `gate::tests::softmax_sums_to_one`
- `gate::tests::nan_fallback_to_sentinel`

### Integration (headless 1-car)
- Spawn champion atual → run 30s plain track → no hard vetos triggered
- Spawn dumb-zero brain → run 30s → sentinel keeps alive >10s
- Spawn near wall → anti-stall reverses out within 1.5s

### A/B Evolution (validação real)
- Baseline: `simple_elitism` atual (champion 472m em 8 gens)
- Trinity: 8 gens × 80 cars × speed 16x
- Métricas: champion peak NET, PINADO%, BATIDA%, gen_to_first_lap
- Sucesso = PINADO < 40% E champion progride monotônico

## Riscos & mitigação

| Risco | Mitigação |
|---|---|
| Brain atrofia (sentinel faz tudo) | Fitness penalty `hard_veto_count×5` + entropy bonus no gate |
| Gate colapsa em w=[0,0,1] sempre | Penalty entropia + curriculum: aos 30 gens, sentinel Tier 2 desliga |
| DNA backward-incompat com champion 2693m | Migration: champion existente carrega cortex, gate=uniform 1/3, reflex=zeros |
| Reflex 60Hz × 500 carros = 500 × 118ns × 60 = 3.5ms/s | Trivial, sim corre 16× já |
| Sentinel rules conflitam (anti-stall vs FCA) | Priority enum, hard veto único por tick |

## Não-mexido

- Cortex BitLinear 128×64 (preserva champion DNA)
- KAN splines (post-process steer/thr/brake)
- CfC temporal gating
- Memory FloatLinear 32×8
- `simple_elitism`, `survival_pressure` flags
- Track curriculum, environment params

## Roll-out

1. Implementar Sentinel sozinho (Tier 0+1+2). A/B vs sentinel_brake atual.
2. Implementar Chaos detector + conectar resonance. Log apenas, não usar ainda.
3. Implementar Reflex live (specialist runtime). A/B mux fixo (chaos>0.5 → reflex).
4. Implementar Gate (MoE). A/B vs mux fixo.
5. Fitness penalties + curriculum desligamento Tier 2.

Cada step = commit + race report + decisão go/no-go.
