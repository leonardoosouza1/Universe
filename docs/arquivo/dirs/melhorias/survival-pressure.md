# Pressão de Sobrevivência

Feature flag `survival_pressure` (default ON).

## Mecanismos existentes (antes)

- HP drain off-road: 5/s
- HP drain redline: 5/s
- HP drain slip: 2/s
- HP drain spin: 30/s se yaw>1.5 rad/s parado
- HP drain grass: 10/s
- HP drain batida: -50 + prox_wall
- Fuel: `0.002 + pedal² × 0.004` /s
- Lap refill: cada 3 laps → fuel=1.0
- Waypoint heal: +40 HP / 25m novo
- Stagnation: death se sem progresso

## Adicionado (survival_pressure)

### 1. `corner_fuel_burn`

```
fuel_rate = 0.002 + pedal² × 0.004 + |steer| × pedal × 0.003
```

Curva sustentada com pedal queima 37% mais rápido em full-lock + full-gas. Força aprender linha eficiente, não arrastar volante.

### 2. `idle_brake_penalty`

```
if brake > 0.5 AND speed < 3 AND !reversing:
    hp -= 2.0 × dt
```

Brake parado dreno HP. Pune estratégia "freio pra não morrer por stagnation" sem andar.

### 3. (reservado) `spin_extra_drain` — **não aplicado**, existente já forte.

## Refinamentos ao Sentinel brake

Antes: dispara sempre em TTC<0.6 + speed>8.
Agora: **só dispara se pedal>0.4** (carro ATIVAMENTE acelerando contra parede).

Evita amplificar stagnation em cruising lento próximo à parede.

## A/B baseline (2026-04-24)

Run: 3 gens × swarm 40 × speed 16×.

| Gen | Metric | Baseline | Sentinel v1 (agressivo) |
|---|---|---|---|
| 1 | NET max / median | 330 / 38.6 | 330 / 41.5 |
| 1 | BATIDA / STAG | 1 / 15 | 1 / 15 |
| 2 | NET max / median | 519 / 154.5 | 330 / 74.0 |
| 2 | STAG / SBLD | 12 / 25 | 14 / 21 |
| 3 | NET max / median | 330 / 157.8 | 330 / 20.4 |
| 3 | HP / STAG | 4 / 12 | 10 / 17 |

**Veredito v1:** Sentinel agressivo degrada — amplificou stagnation via cruising próximo parede.
**Veredito v2 (refinado):** precisa rerun (pedal-gated).

## A/B v2 (pedal-gated Sentinel + survival_pressure) — **VALIDADO**

Run: 4 gens × swarm 80 × speed 16× (2026-04-24).

| Métrica | Baseline (no-default) | Sentinel+Survival | Δ |
|---|---|---|---|
| NET max Gen 1 | 330 | 330 | 0 |
| NET max Gen 2 | 330 | **679.7** | **+106%** |
| NET max Gen 3 | 330 | 422.0 | +28% |
| NET max Gen 4 | 330 | 330 | 0 |
| NET median Gen 2 | 128.2 | 152.4 | +19% |
| NET median Gen 3 | 128.3 | 158.9 | +24% |
| NET mean Gen 2 | 133.9 | 177.1 | +32% |
| BATIDA total | 8 | 12 | +50% (carros ousam mais) |
| STAG total | 60 | 54 | -10% |
| HP total | 65 | 62 | -5% |

**Veredito:** baseline plateau em 330m (ninguém atravessa curva 1). Sentinel+Survival
achou **breakthrough em Gen 2: 679.7m** — primeiro carro ultrapassou waypoint 2.
Stagnation caiu, batida subiu (efeito esperado: carros tentam velocidade em
curvas mal treinadas — pressão pra aprender frenagem).

## Próximo

- Rodar 10 gens continuação pra ver se breakthrough consolida (Hall of Fame ativa).
- Testar flags isoladas: só sentinel, só survival, p/ medir efeito individual.
- Habilitar `brake_symmetric` (remove threshold 0.2) — esperado: mais modulação neural.
- Habilitar `gain_comp_steer` em swarm fresh — full lock real pra bootstrap.

## Links

- [[gain-asymmetry.md]]
- [[specialist-brains.md]]
- `EVOLUTION_REPORT.md` §7 (bootstrap swarm ≥80)
