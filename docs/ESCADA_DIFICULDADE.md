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
| D7 | competição por recurso (M<N postos de energia) | `--recurso-escasso` | spec (1ª frente competitiva) |
| D8 | percepção custa energia (canal = gene com preço) | `--custo-sensor` | spec (liga em FASE SENSORES) |

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
