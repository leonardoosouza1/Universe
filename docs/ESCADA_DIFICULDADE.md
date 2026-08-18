# Escada de Dificuldade — o currículo do mundo inteiro

> Leonardo (2026-08-16): "ainda está muito fácil" → decisão: TUDO em escada,
> um degrau por bateria/mérito, como o currículo de câmbio (que funcionou 3/3).

## Diagnóstico do fácil (medido)
Promoção 33/33 gens (nada segura o campeão) · same-start+pista fixa = decorável ·
erro barato (17% na parede E voltas duplas; mortalidade ~0) · percepção sem ruído ·
sem adversário · nada degrada (por isso o maestro M4 empatou: sem estado lento).

## A escada (D1→D8), critérios e doses

Cada degrau: flag própria (off por padrão), braço A/B na entrada, e SOBE por
mérito populacional (mesmo gatilho do currículo). **Janela de dose certa**:
promoção cai pra 30-60%/gen · mortalidade 5-20% · mediana cai e recupera ≤15
gens. Fora disso = dose errada (ajustar ANTES de concluir) — lição teto6/energia2.

| # | mecanismo | flag | status |
|---|---|---|---|
| D1 | fuel-auto k apertando 1,4→1,0→0,9 | `--fuel-auto k` (+currículo do k) | PRONTO (falta currículo do k) |
| D2 | largada aleatória por geração (mesma pra todos na gen — comparável; derivada da seed da gen) | `--largada-aleatoria` | IMPLEMENTADO 2026-08-16 |
| D3 | ruído sensorial σ + dropout de canal por tick | `--ruido-sensorial σ` | spec (mexe em BrainParams+shader) |
| D4 | DANO oculto acumulado (RPM alto + parede degradam torque/direção na vida; sem sensor direto — o estado lento do maestro/EXP10) | `--dano` | IMPLEMENTADO 2026-08-16 |
| D5 | clima: μ por trecho + vento lateral | `--clima` | spec |
| D6 | fantasma do campeão anterior na pista | `--fantasma` | spec |
| D7 | competição por recurso (M<N postos de energia) | `--recurso-escasso M` | IMPLEMENTADO 2026-08-17 (BODY PHASE) — doses 16→12→8→6→4 no plano; falta CANÁRIO |
| D8 | percepção custa energia (canal = gene com preço) | `--custo-sensor DOSE` | IMPLEMENTADO 2026-08-17 (BODY PHASE) — doses 0,15→0,30→0,45→0,60 no plano; falta CANÁRIO |

## D4 — DANO (detalhe, porque é o que destrava o maestro)
dano_i ∈ [0,1], começa 0 a cada vida; acumula por tick:
  + 0,00015 × excesso_rpm (mesma rampa do custo-rpm)
  + 0,0004 quando colado na parede (<3 m)
Efeito: throttle_efetivo ×= (1 − 0,5·dano) e steer_autoridade ×= (1 − 0,3·dano)
— o corpo ENVELHECE dentro da vida. SEM canal sensorial direto (oculto): o ser
só sente pelos EFEITOS — exatamente o regime onde neuromod/maestro paga (EXP10:
dano τ longo oculto, G=25,6%). Predição M4 sob D4: gates correlacionam com dano
e M4 > ctrl (registrar na bateria).

## Currículo da escada (quando rodar)
Bateria única longa (200+ gens): começa D1(k=1,4)+D2; cada promoção-de-mundo
(população completa na dose atual) APERTA o degrau ativo ou LIGA o próximo
(D1 aperta k → D3 liga → D4 liga → ...), um de cada vez, nunca dois juntos —
degrau a degrau, corpo antes de pista, como o câmbio. Telemetria [escada] por
gen com o degrau ativo. Reprovar = recuar um degrau (histerese, lição do
currículo por população).

## Integração concluída (2026-08-16) + correções de registro

- 4 agentes paralelos (posse disjunta) entregaram D3/D5/D6/controlador; fiação
  em main.rs feita pelo orquestrador. Flags: --ruido-sensorial σ, --ruido-dropout,
  --clima mu, --clima-freq, --vento x z, --fantasma, --escada (controlador
  automático completo: FuelK→Ruído→Dano→Clima→Fantasma com histerese).
- CORREÇÃO DE REGISTRO (honestidade): physics_power_*.wgsl é a ferramenta de
  BISSECÇÃO do crash ACO, não o pipeline vivo (vivo = physics_a/b/c.wgsl).
  Isso derruba minha citação anterior de "térmica do pneu implementada" — o
  vivo faz PASSTHROUGH da temperatura; o ch70 "variando" na bateria de 139 era
  o sinal PÓS-CfC (denoise + dinâmica do filtro sobre entrada constante), não
  o sensor. O clima D5 foi portado por MIM para o pipeline vivo (μ no grip do
  kernel B, vento no somatório do A, espelhos de layout no C). Térmica de pneu
  REAL (CPU-side, sem risco ACO) fica na fila junto do degrau Clima — sem ela
  o μ variável não é sentível pelo ch70.
- Comando da bateria-escada (quando a GPU for liberada):
  gpu_evolve --seres 48 --gens 200 --ticks 20000 --fitness centro --width 28
    --same-start --life-earn 12 --life-cap 8000 --out-beta 0.0 --sem-buracos
    --ativacao --retomar --retomar-mix 0.85 --pistas 9001,8005,8200,9003
    --trocar-ao-completar 4 --seed {77,31} --cambio auto --cambio-teto 6
    --custo-rpm 2.0 --mascara-regioes --tau-semeado --largada-aleatoria
    --fuel-auto 1.4 --escada --out /media/llm-data/universe-runs/escada-s{N}

## CANÁRIO PERMANENTE: pressão nova PROVA que diferencia dose antes da bateria

> Nasceu de um erro caro (2026-08-16) e a sessão-forja adotou o mesmo canário no
> lado do Qwen no mesmo dia, depois de ele explicar uma anomalia deles em 10 min.

**Regra:** antes de qualquer bateria com um mecanismo de pressão novo, rodar a
varredura de dose e mostrar que doses diferentes produzem números DIFERENTES —
incluindo o braço com o mecanismo DESLIGADO. Dose que não diferencia não é pressão
fraca: é pressão desligada.

**E a varredura tem de rodar na CONFIG DA BATERIA.** Calibrei a primeira vez com
6000 ticks, largada fixa e pista única; na bateria real (20k ticks, largada
aleatória, rotação) a mesma dose colapsou a população na primeira geração
(mediana 84 m). Config de smoke serve pra provar que o mecanismo LIGA, não pra
escolher a dose.

### O que o canário pegou aqui (dois bugs, ambos mudos no log)

1. **Denominador errado**: tanque dimensionado por VOLTA e recarregado por SETOR
   (24 por volta) — ~12× folgado, nunca prendia.
2. **Nunca aplicado**: a conta só rodava na TROCA de pista; em pista única e na
   primeira geração `--fuel-auto` não fazia efeito nenhum.

Sintoma que denunciou: `k = 2,0`, `1,0` e `0,7` deram resultado IDÊNTICO (1923 m),
igual ao controle sem combustível. E a consequência silenciosa: **34 de 35
gerações da bateria-escada com mortalidade 0% mesmo com combustível, ruído, dano
e clima LIGADOS ao mesmo tempo.**

### Curva de dose do combustível (config real da bateria, 2026-08-16)

| k | tanque | mediana | promoção |
|---|---|---|---|
| desligado | — | 7665 m | 100% |
| 8,0 | 0,312 | 7640 m | 98% |
| **5,0** | 0,195 | 1442 m | **42% — dentro da janela 30-60%** |
| 3,0 | 0,117 | 813 m | 0% |
| 2,0 | 0,078 | 450 m | 0% |

Penhasco entre 5 e 3: o plano da escada agora anda fino nessa faixa
(8,0 → 6,5 → 5,5 → 5,0 → 4,5 → 4,0) em vez de saltar por cima dela.

### Correção do critério: mortalidade é TETO, não piso

O doc pedia "mortalidade 5-20%" como parte da janela de dose certa. Medido: esta
pressão age TRAVANDO, não matando — mortalidade 0% em toda a varredura, enquanto a
promoção varia de 98% a 0%. Exigir mortalidade ≥5% tornava a janela inalcançável e
a escada só enxergava "boa" ou "ruim", passando reto pela dose calibrada.
`avaliar()` agora usa mortalidade só como teto.

## BODY PHASE (2026-08-17) — o que entrou e o que AINDA NÃO foi medido

Integração dos quatro mecanismos do mundo hostil no `gpu_evolve`. **Nenhum número
de corrida saiu daqui**: a regra 1 desta fase proíbe treino de GPU, então tudo
abaixo é código + teste de CPU + aritmética de teto. A curva de dose de cada um
é dívida aberta, e até ela fechar nenhuma dose está calibrada.

| flag | mecanismo | moeda | canais que o ser sente |
|---|---|---|---|
| `--interocepcao` | liga o BODY BLOCK (8 valores CPU→onda sensorial) | — | ch70, ch71, ch98, ch114-117, ch124 |
| `--recurso-escasso M` | D7: M postos < N seres; SUBSTITUI o refil por setor | fuel → vida | ch114 (rumo), ch115 (proximidade), ch116 (carga) |
| `--predador f` | perseguidor CPU a `f × 18,75 m/s`, dreno `D·(1−d/R)²` | VIDA | ch98 (rumo), ch124 (proximidade) |
| `--pneu-termico g` | térmica CPU-side → `env_grip` por ser | aderência | ch70 (temperatura real) |
| `--custo-sensor D` | D8: assinatura por família de canais | VIDA | — (é imposto, não sinal) |

**Correção de registro (medida, não inferida).** A térmica do pipeline VIVO estava
travada de um jeito que nenhuma auditoria tinha nomeado: `wheels[1].x` nasce em
**60.0** (`gpu/physics.rs:104`, `WheelGpu::from_cpu(..., tire_temp = 60.0, ...)`) e
o kernel B compara esse valor contra `optimal_temp = 0.5` em
`clamp(1 − |T − 0,5|·0,5, 0.7, 1.0)` (`physics_b.wgsl:190`). Celsius contra
normalizado: o fator satura no piso e vale a CONSTANTE **0,70** a corrida inteira.
E `prop[6] = média(wheels[j*2+1][0])/120` (`context.rs`) dá **0,5 constante**, ou
seja **ch70 era canal morto no gpu_evolve** — não "passthrough vivo".

Consequência para o A/B: `--pneu-termico` NÃO toca `wheels[1].x` nem os
`config_flags`. Aquele 0,70 é idêntico nos dois braços e cancela; consertar a
unidade mexeria no envelope de pneu dos DOIS braços ao mesmo tempo. A térmica real
chega à física por `env_grip` (por ser, multiplica `mu_base` em
`physics_b.wgsl:141`) e ao cérebro pelo slot 6 do BODY BLOCK.

**Ganho DC normalizado em TODA dose.** Medido (teste
`every_thermal_dose_keeps_unit_dc_gain`): ganho DC = 1,00000 em g ∈ {0,5; 1; 1,5;
2; 3}. A tabela de equilíbrio a 18,75 m/s, calculada ANTES de rodar:

| dose | piso | trim | parado | cruzeiro | curva certa | forçada | derrapagem |
|---|---|---|---|---|---|---|---|
| g=1 | 0,70 | 1,1372 | 0,922 | 1,001 | 1,130 | 0,847 | 0,796 |
| g=2 | 0,40 | 1,3179 | 0,818 | 1,001 | 1,302 | 0,645 | 0,528 |
| g=3 | 0,10 | 1,5671 | 0,676 | 1,003 | 1,538 | 0,366 | 0,157 |

Dose maior é BALANÇO MAIOR, nunca menos aderência média — senão o A/B mediria o
offset em vez da dinâmica (a lição do ganho DC, de novo). O teto `max_factor` é
DERIVADO da banda, não fixo: com o 1,20 herdado o trim de g=2 (1,3179) seria
cortado e o braço LIGADO viraria um nerf global. A mutação que fixa 1,20 derruba o
teste — conferido.

**CANÁRIO DE DOSE: dívida aberta para os quatro.** A regra permanente desta seção
exige, na CONFIG DA BATERIA, que doses diferentes deem números DIFERENTES incluindo
o braço desligado. Nada disso foi rodado. Até rodar:

- D7: braços `off / 16 / 8 / 4`, exigir que os quatro difiram; se M=16 ≈ off,
  procurar bug mudo (a oferta/demanda impressa em `[recurso] CEILING` já diz se a
  dose é escassa por construção — se ela cobre 100%, a pressão está desligada).
- Predador: braços `off / 0,40 / 0,55 / 0,70 / 0,85`; mortalidade e ser-ticks sob
  dreno têm de ser MONOTÔNICOS na dose.
- Pneu: braços `off / 0,5 / 1,0 / 2,0`; a linha `[pneu]` tem de mostrar temperatura
  média DIFERENTE por dose (igual = a térmica não está integrando).
- D8: braços `off / 0,15 / 0,30 / 0,45 / 0,60`; a linha `[sensor]` tem de mostrar
  LEQUE de imposto entre indivíduos — leque ~0 é o modo de falha do E-REG1
  (imposto constante não seleciona), e é o mesmo alarme dos números iguais.
