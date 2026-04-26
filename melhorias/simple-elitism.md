# Simple Elitism

Feature flag `simple_elitism` (default ON).

## Tese

Hall of Fame + tournament + priming-refresh + crossover + niching +
validation-gate = complexidade que impedia campeão de progredir.

Esquema simples:
- **Slot 0** = champion puro, zero mutação.
- **Slot 1..N** = champion + mutação ternária pontual (rate 2× padrão).
- **Sem distributed spawn** — todos partem do grid. Comparação justa.
- **Crown imediato**: qualquer gen em que best > champion → coroa. Sem
  `env_level == 0` gate.

Se o champion é o melhor, ele continua. Se um mutante bate, assume.
Sem lucky-spec: todos partem do zero.

## Mudanças de código

- `Cargo.toml`: feature `simple_elitism` default ON.
- `src/evolution/spawn.rs`:
  - Slots != 0 → champion weights + mutação ternária pontual (rate_boosted = 2× × 0.25 cap).
  - Slow brain também recebe mutação leve (f32 delta ±0.15).
  - `GRID_SPAWNS = usize::MAX` → zero distributed spawn.
  - Grid 4-col em vez de 2-col (compacta 80 carros em 120m).
- `src/evolution/pipeline.rs`: `simple_ok = this_gen_best_dist > champion_distance`
  adicionado ao `should_crown` (OR com validation-gate original).

## A/B (8 gens × swarm 80 × speed 16×)

### Antes (priming v2 + sentinel + survival sem simple_elitism)
4 gens rodados: Gen 2 NET max 682m, p90 Gen 2 = 330m, mean 161.
Coroação ~1 vez. Champion não consolida — Gen 3 volta a 330m.

### Com simple_elitism (v1 — com distributed spawn)
- 3 coroações (Gen 1/2/3: 235→237→344).
- Depois disso champion `344m`. Gen 5/8 batem 619m/673m mas distributed —
  não podiam coroar (filtro `is_grid_spawn`).

### Com simple_elitism v2 (grid full, 80 cars no grid, 4 cols)

| Gen | NET max | Champion novo |
|---|---|---|
| 1 | 236 | **237** |
| 2 | 273 | **273** |
| 3 | 338 | **338** |
| 4 | 368 | **368** |
| 5 | 376 | **376** |
| 6 | 368 | — |
| 7 | 372 | — |
| 8 | 472 | **472** |

**6 coroações, champion progressivo 237 → 472m em 8 gens.**

## Observações

- NET max absoluto menor que v1 (472 vs 682) porque ninguém nasce a 15% da pista.
  Mas champion é HONESTO — comparação justa.
- Mediana 0 em gens iniciais = muitos carros morrendo rápido (STAG 43-44).
  Grid 20 rows × 6m = 120m atrás da start. Parte pode ficar off-track.
  Tradeoff aceitável pra simplicidade.
- BATIDA cresce (0-5 por gen) — carros arriscam mais com pressão.
- Mutação 2× rate mantém exploração sem precisar de priming-refresh.

## Não-mexido

- `effective_mutation_rate` decay exponencial preservado.
- Priming axiomático ainda funciona em slot 0 quando champion NÃO existe
  (bootstrap frio via build_priming_weights).
- `is_grid_spawn` filter em pipeline — agora redundante mas inofensivo
  (todos grid).

## Trade-offs

| Ganho | Custo |
|---|---|
| Champion evolui limpo | Menos diversidade exploratória |
| Bootstrap previsível | NET max menor em gens intermediárias |
| Coroação frequente | Sem pool de genótipos alternativos |
| Código 30% menor | Perde niching ecológico |

Se champion estagnar por 20+ gens, pode precisar reintroduzir crossover
ou priming-refresh ocasional. Por enquanto, simple > complexo.
