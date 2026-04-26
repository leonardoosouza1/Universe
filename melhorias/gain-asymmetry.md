# Assimetria de Ganho Percepção ↔ Ação

## Sintoma

Carro vê parede (retina + TTC), deriva temporal confirma aproximação, **mas bate**.

## Diagnóstico da cadeia de steer

`src/simulation/motor.rs`:

```
brain_out[1]                     ∈ [-1, 1]           # saturado
  ↓ KAN spline + steer_bias + line_hint
  ↓ cortisol_steer_mult
  ↓ clamp(-1, 1)
  ↓ 0.65 · prev + 0.35 · new     # smoothing temporal
  ↓ × (1 - risk_damp)             # slow brain atenua
  ↓ + emerg_steer (ECAS ≤ 2.5m)
  ↓ + exploration_steer (curiosity > 0.7)
  ↓ clamp(-1, 1)  → final_steer_intent
  ↓ lerp racer.memory[1] @ rate dt*20
  ↓ = physical_steer              # SEM clamp
  ↓ × dynamic_steer_limit = 1/(1 + speed * 0.02)
  ↓ = engine.steer
```

## A raiz: `dynamic_steer_limit`

| Speed (m/s) | Limit |
|---|---|
| 0 | 1.00 |
| 25 | 0.67 |
| **50** | **0.50** |
| 100 | 0.33 |

Brain satura em `steer_raw = 1.0` mas a 50m/s só 0.5 chega na roda.
Pra virar "100%" a 50m/s, brain precisaria emitir **2.0** — mas clamp em ±1 trava.
Canal 74 (sensory_wave) expõe `steer_limit × 127` mas a rede só aprende a usar se variância na evolução encontrar. Não tem priming axiomático pra isso.

Resultado: em curva rápida, brain faz o que sabe (output saturado), mundo divide por 2, carro "entende tarde" e bate.

## Assimetria de throttle → brake

Brain emite intent[2] = brake, passa por spline + threshold 0.2. Se brake_spline ≤ 0.2 → zero. Brain precisa descobrir "apertar firme". Enquanto isso, throttle vem de `(intent[0]+1)/2` → sempre em [0,1] sem threshold. **Throttle é gratuito, brake é caro.** Pressão evolutiva empurra gas.

Front wall safety cap (`forward_wall_dist / 3.0`) só corta throttle, **não aciona brake**. Brain vê parede, throttle cai, mas **continua rolando por inércia até bater**. Não há reflexo de freio como há ECAS lateral.

## Propostas

### P1 — Sentinel brake reflex (inspirado SACI Sentinel)

Hard-brake non-learnable quando:
- `forward_wall_dist < 2.5` **E** `speed > 8` **E** `forward_wall_dist / speed < 0.6` (TTC < 0.6s)
- Força `physical_brake = max(physical_brake, 1.0 - ttc/0.6)`
- Ativo SEMPRE, não sujeito a evolução. Mesma filosofia do ECAS lateral.
- Feature flag `sentinel_brake` (default ON em experimentos, OFF em replay de champion legado).

**Expectativa:** reduz HpZero por colisão frontal em reta longa / chicane. Libera pressão seletiva pra aprender outras coisas (linha, drift, throttle modulation).

### P2 — Gain compensation no steer

Multiplicar neural steer por `1 / dynamic_steer_limit`, capped em 2×, ANTES do smoothing:

```rust
let gain_comp = (1.0 / dynamic_steer_limit).min(2.0); // max 2× boost
let steer_raw_unclamped = (steer_spline + steer_bias + line_hint)
    * cortisol_steer_mult
    * gain_comp;
```

Brain fala em "unidades físicas reais". Output 1.0 = full lock verdadeiro em qualquer velocidade. Priming axiomático passa a fazer sentido (canal 74 deixa de ser "bicho de sete cabeças").

**Risco:** invalida scale calibrado do champion atual. Precisa bootstrap novo.
**Mitigação:** feature flag `gain_comp_steer`, default OFF. Experimento em branch.

### P3 — Brake spline threshold simétrico

Reduzir threshold 0.2 → 0.05 OU remover. Brain aprende modulação em vez de bang-bang. Throttle continua sem threshold — simetria restaurada.

## Ordem de ataque

1. **P1 Sentinel brake** — baixo risco, ganho imediato.
2. Medir: 3 gens headless × 3 seeds → wall_death_rate vs baseline.
3. Se P1 reduz HpZero > 20%, aplica **P3**.
4. **P2 gain comp** só depois de ter baseline estável com P1+P3 (invalida scales).
