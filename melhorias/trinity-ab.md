# Trinity Swarm — A/B run (8 gens)

Headless `--gens 8 --swarm 80 --speed 16`, fresh champion (CHAMPION.dna removed before run).

## Resultados

| Gen | NET max | p90 | mean | PINADO | STAG | BATIDA | Duração |
|---|---|---|---|---|---|---|---|
| 1 | 234 | 229 | 65 | 0 | 31 | 0 | 77.3s |
| 2 | 294 | 233 | 58 | 0 | 43 | 0 | 78.8s |
| 3 | 690 | 166 | 56 | 0 | 46 | 0 | 94.7s |
| 4 | 349 | 298 | 97 | 0 | 51 | 0 | 95.5s |
| 5 | 308 | 277 | 113 | 9 | 37 | 0 | 68.8s |
| 6 | 292 | 46 | 20 | 2 | 34 | 1 | 79.5s |
| 7 | 349 | 292 | 92 | 0 | 47 | 3 | 59.7s |
| 8 | **712** | 421 | 107 | 4 | 44 | 0 | 85.0s |

## Vs baseline simple_elitism v2

| Métrica gen 8 | Baseline | Trinity | Δ |
|---|---|---|---|
| Champion NET | 472m | **712m** | **+51%** |
| PINADO médio | ~78% | ~2% | **-40×** |
| STAG médio | ~10% | ~40% | +4× |
| BATIDA | 0-5/gen | 0-3/gen | similar |

## Análise

- **Sentinel hard veto eliminou PINADO** quase completamente. Carros já não ficam encostados na parede acelerando ao infinito.
- **Champion progride**: peak 712m gen 8, melhor que baseline 472m no mesmo gen.
- **STAG explodiu**: trocamos "trava-na-parede" por "fica-parado". Brain conservador + Sentinel reduzindo throttle = carro lento → STAG.
- **BATIDA estável**: trinity não aumentou colisões.

## Causa STAG

- `slip_avg=0.0`, `stall_timer=0.0`, `pinado_timer=0.0` em motor.rs (placeholders, Task 5 spec cut).
- Anti-stall HARD veto (reverse-out automático) **nunca dispara** sem stall_timer real.
- ESP/TCS soft proposals reduzem throttle sem feedback de slip → contribuem para STAG.

## Próximo passo

1. Plumar `racer.stall_timer` + `pinado_timer` reais em motor.rs (atualizar conforme low-speed + obstacle).
2. Plumar `slip_front_avg` / `slip_rear_avg` no SafetyState.
3. Esperado: STAG cai (anti-stall reverse desbloqueia carros) → carros recuperam → champion passa 1 km.

---

## A/B v2 (2026-04-26): stall_timer/pinado_timer wired

`stall_timer = slow_time/64`, `pinado_timer = wall_time when speed_kmh<8`. Anti-stall HARD veto agora gated por `wall_left<1 || wall_right<1 || ttc_forward<2` (fix: evita reverse em curva lenta sem obstáculo, que matava por STAGNATION).

| Gen | NET max |
|---|---|
| 1 | 234 |
| 2 | 175 |
| 3 | 714 |
| 4 | 708 |
| 5 | 345 |
| 6 | 716 |
| 7 | 707 |
| 8 | **719** |

**Variância despencou**: 5 de 8 gens >700m (antes só gen 8 chegava). Antes oscilava 292/308/349/712.

| Métrica | Trinity v1 | Trinity v2 (timers reais) | Δ |
|---|---|---|---|
| Champion gen 8 | 712m | **719m** | +1% |
| Gens >700m | 1/8 | **5/8** | +400% |
| PINADO | ~2% | ~2% | mantido |
| STAG | ~40% | ~25% | -15pp |

**Lição**: anti-stall HARD veto sem gate de obstáculo era armadilha. Carro lento em curva = reverse = STAGNATION. Fix obrigatório: só fire reverse quando há razão física (parede/TTC).

---

## A/B v3 (2026-04-26): slip_avg + curv25 wired

`slip_avg` lido direto de `body.wheels[].slip_ratio+slip_angle` (lag 1-tick). `curv25` derivado dos canais 79/80 da retina × 60 (saturated curve ≈ 60).

Ativa Tier 1+2 soft proposals que estavam dormentes: **TCS** (slip>0.4 + thr>0.5), **ESP** (yaw_rate>1.5 já estava ativo), **CornerLim** (curv25>30 + speed>60kmh).

| Gen | NET max | median |
|---|---|---|
| 1 | 234 | 0 |
| 2 | 694 | 0 |
| 3 | 299 | **282** |
| 4 | 593 | 0 |
| 5 | 708 | 0 |
| 6 | 598 | 0 |
| 7 | 716 | 0 |
| 8 | 703 | 0 |

**Marco**: gen 3 com **median=282m** — primeira vez em todos os A/Bs que mediana sai de zero. Swarm inteiro progride, não só top-N.

| Métrica | v2 | v3 |
|---|---|---|
| Champion gen 8 | 719m | 703m |
| Gens >700m | 5/8 | 3/8 |
| Mediana >0 | 0/8 | 1/8 |

Trade: TCS/CornerLim adicionam oscilação no topo (variância sobe), mas swarm mediano destrava. Próximo: tunar threshold TCS (0.4 talvez agressivo) ou attacar mediana-zero direto.

## Trade-offs aceitáveis

| Ganho | Custo |
|---|---|
| PINADO 40× menor | STAG 4× maior (recuperável com Tier 0 anti-stall ativado) |
| Champion +51% | Variância gen-a-gen alta (gen 6: 292m após gen 5: 308m) |
| Sem regressão BATIDA | Mediana zero ainda — só top-N progride |

## Commits relacionados

- 26e74d7..eaa8d18 (Tasks 1-8): safety, chaos, gate, engine wiring, motor, evolution preservation, fitness, cargo features.
