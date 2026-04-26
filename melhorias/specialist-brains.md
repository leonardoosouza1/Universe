# Cérebros Especialistas — Arquitetura Modular

> Abstração do **SwarmBrain + SwarmCoordinator** do SACI-Trader.
> Fonte: `~/projects/LLM/saci-trader/backend/src/brain/models/` — 5 workers CfC M5→H4, cross-attention coordinator, long/short expert heads, Sentinel gating.
> **Não** copiar trading. Copiar o **padrão**.

## Por que trocar o monolítico

BitLinear 128→64 é um único tensor ternário. Faz TUDO: ver, decidir steer, decidir gas, decidir brake, mediar medo. Problemas:

1. **Gradiente genético conflituoso** — melhorar steer pode degradar throttle (pesos compartilhados).
2. **Priming axiomático frágil** — mutação random em 1 peso do "canal curvatura" muda gas, steer e brake juntos.
3. **Sem inspetabilidade** — DNA viewer mostra heatmap mas não separa competências.
4. **Sem capacidade de especialização assimétrica** — retina frontal precisa de muito mais resolução que laterais.

## Proposta: Swarm de especialistas

```
                     ┌─── REFLEX (Sentinel, hard-coded) ───┐
                     │  wall<2.5m → emerg_steer            │
                     │  front<2.5m + ttc<0.6s → hard_brake │
                     │  surface=grass+slip → no-op         │
                     └──────────┬───────────────────────────┘
                                │ override
  sensory_wave[128] ──┐         │
    (retina 0-63,     │         ▼
     wall_probes,     │    ┌──────────────┐
     ttc, slip,       ├───▶│  VisionCore  │ 64 → 32 CfC (liquid time)
     curvature,       │    └──────┬───────┘ memória 2D da retina
     speed, surface,  │           │
     fuel, stress)    │           ▼
                      │    ┌──────────────┐
                      ├───▶│  SteerBrain  │ 32→1, BitLinear ternário
                      │    └──────┬───────┘ (perif+curvatura+line_hint)
                      │           │
                      │    ┌──────────────┐
                      ├───▶│ ThrottleBrain│ 32→1, BitLinear ternário
                      │    └──────┬───────┘ (ttc+speed+safety_cap+grip)
                      │           │
                      │    ┌──────────────┐
                      └───▶│  BrakeBrain  │ 32→1, BitLinear ternário
                           └──────┬───────┘ (ttc+wall_front+cornering)
                                  │
                                  ▼
                      ┌─────────────────────┐
                      │  Coordinator (slow) │ 16→4 FloatLinear
                      │  aggression, risk,  │ fuse: throttle_cap,
                      │  damp, brake_anticip│ steer_bias, damp
                      └─────────────────────┘
```

### Camada por camada

1. **Sentinel (reflex)** — C code hard-coded. Sempre liga. Não evolui. Single responsibility: não deixar bater. (ver [[gain-asymmetry]])
2. **VisionCore** — CfCCell 64→32. **Já existe** em `src/cfc.rs`. Extrair da retina, dar saída compacta compartilhada.
3. **SteerBrain** — BitLinear 32→1. Inputs: visão comprimida + perif_L/R + curvatura + line_hint. Output: steer_intent.
4. **ThrottleBrain** — BitLinear 32→1. Inputs: visão + speed + safety_cap + grip + ttc. Output: pedal_intent.
5. **BrakeBrain** — BitLinear 32→1. Inputs: visão + ttc + wall_front + slip. Output: brake_intent.
6. **Coordinator** — FloatLinear 16→4 (já existe como `slow_brain`). Recebe resumos dos 3 especialistas + metabolismo. Emite bias modulador: `throttle_cap`, `risk_damp`, `steer_bias`, `brake_anticip`. **Não substitui** — modula.

### DNA v4 — layout por cérebro

Em vez de blob monolítico 8192 + 256 pesos:

```
magic[4]=IARA | version:u16=4 | generation:u32 | seed:u32
track_width:f32 | fitness:f32

[ VISION_CORE ]  (CfC state NOT saved, runtime only)
[ STEER    ]  scale f32 + 32 pesos ternários
[ THROTTLE ]  scale f32 + 32 pesos ternários
[ BRAKE    ]  scale f32 + 32 pesos ternários
[ COORD    ]  256 pesos f32 (slow brain)
```

**Crossover por cérebro:** pai A doa SteerBrain, pai B doa BrakeBrain — recombinação semântica, não posicional. Evolução pode encontrar "melhor freador" × "melhor curvador".

### Mutação especializada

- Rate base por cérebro. SteerBrain mais alto (domínio caótico). BrakeBrain baixo (não quer perder reflexo ganho).
- Burst mutation por input-canal continua valendo, mas por cérebro.
- Priming axiomático: cada cérebro nasce com pesos pré-setados nos canais que IMPORTAM PRA ELE. Não mais priming global dispersivo.

## Analogia SACI

| SACI | Universe (proposto) | Papel |
|---|---|---|
| Worker M5 | SteerBrain | specialist micro-scala (curva imediata) |
| Worker H4 | Coordinator / Slow | specialist macro-scala (estratégia de volta) |
| LOBEncoder | VisionCore | encoder sensorial compartilhado |
| Sentinel | Reflex hardcoded | supressor físico em DEFCON |
| LFM Strategist | ~~n/a~~ | Universe não tem LLM (tempo real) |
| Cross-attention coordinator | slow_brain atual | fusão ponderada |

## Incremental rollout

**Não** reescrever do zero. Ordem:

1. Extrair **VisionCore** (mover CfC pra lib compartilhada, output 32-dim disponível).
2. Introduzir `BrainForward` trait já em [[../src/layers.rs]]. ✅ feito.
3. Criar `src/engine/specialist.rs` com `SpecialistBrain { steer, throttle, brake }` wrapando 3× BitLinear.
4. Adaptar `SovereignEngine` pra aceitar **Box<dyn BrainForward>** OU `SpecialistBrain` via enum.
5. DNA codec v4: `iara_genesis` **já** suporta TLV arbitrário (ver `docs/dna_v4_genesis/`). Salvar por cérebro é trivial.
6. Manter codec v3 loadável (back-compat) — carro legado vira SpecialistBrain com cabeças zeradas e steer monolítico migrado.
7. Feature flag `specialist_brains` default OFF. Run A/B.

## Expected win

- **Inspetabilidade**: DNA viewer mostra heatmap por cérebro — vê QUEM é bom em quê.
- **Evolução paralela**: crossover semântico acelera convergência.
- **Reflexo hard-coded**: impede morte burra (bater em reta).
- **Caminho pra GPU futuro**: especialistas pequenos (32×1) são batch-friendly — `rocblas::sgemm_batched` natural.

## Links

- [[gain-asymmetry]]
- [[gpu-roadmap]]
- Obsidian: [[SACI-Trader/components/BRAIN]], [[SACI-Trader/ARCHITECTURE]], [[SACI-Trader/concepts/CFC]]
