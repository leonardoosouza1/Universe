# A bateria de serviço completo: a primeira capacidade de ponta a ponta do programa (com uma retratação minha no §5)

Figura: `full-service.png`. Fontes: `universe-runs/full-service-{evolution,frozen,neutral-body,founder-audit,canary}-20260925/report.json`,
`course-12m-audit/driving.csv` e a geometria do curso registrada em `research.service_course`.
Protocolo: `docs/experiments/FULL-SERVICE-20260925.md`.

---

## 1. O que aconteceu, e é um resultado

O treino fechou as 60 gerações. As três auditorias congeladas — protocolo escrito
**antes** da avaliação, usando o campeão da última geração e não o melhor checkpoint:

| auditoria | entregas completas, sem contato, vivo | por partida |
|---|:--:|---|
| campeão congelado, corpo evoluído | **4 / 4** | 1/1 em cada uma das quatro |
| mesmo DNA, corpo neutro | **4 / 4** | 1/1 em cada uma das quatro |
| fundador (geração 0) | **0 / 4** | 3 bloqueados, 1 vencido no prazo |

O fundador aceita 3 embarques e faz **zero** entregas. Mesmo curso, mesmas condições,
mesmo executável. **A seleção produziu a capacidade**, e existe o controle antes-depois que
sustenta essa frase — coisa que nenhuma bateria anterior deste programa tinha.

E o portão de identidade de DNA fecha: `3b7ead0c24dfa2da` é o mesmo SHA no braço congelado,
no braço de corpo neutro e no campeão `gen0059` do treino. O fundador é `baf6463221160c20`.

## 2. O corpo evoluído não é necessário neste curso

Corpo neutro, DNA idêntico por SHA: **também 4/4**, escore mediano 2058.7 contra 2055.0 —
0.2 % de diferença. É um negativo limpo sobre a coevolução de morfologia *neste curso*, e o
protocolo já tinha se antecipado a ele ("no population-level claim ... is supported by four
deterministic orientations"). Registrar como tal: o cérebro faz o trabalho todo aqui.

## 3. A população satura em 59 %, o campeão em 100 %

Entregas completas por geração, sobre 64 tentativas (16 genomas × 4 partidas):

| gerações | taxa |
|---|---|
| 0–9 | 12.7 % |
| 25–34 | 50.3 % |
| 50–59 | **59.2 %** |

As últimas dez gerações vão de 32 a 42 de 64 **sem tendência**. O campeão resolve o curso;
a população não. A frase defensável é "um campeão selecionado completa o curso nas quatro
orientações", não "o táxi entrega".

Zero mortes em 60 gerações, e 59 de 60 gerações com ao menos uma entrega completa.

## 4. Um alerta sobre a métrica que a seleção usa

O escore mediano salta **198 → 2003 entre as gerações 24 e 30** (10×), enquanto as entregas
completas vão de 23 para 35 (1.5×). O `binary_sha256` é o mesmo nas 60 gerações, então o juiz
não mudou no meio: o salto é o crédito de embarque "banked" entrando em regime.

Consequência prática: **o escore não é métrica de progresso neste curso** — ele é dominado
por uma parcela que liga de uma vez. Quem lê a curva de escore vê um degrau onde o
comportamento teve uma subida suave. As entregas completas são a métrica; o escore é o que a
seleção otimiza, e as duas divergem por um fator 7 no mesmo intervalo. Vale conferir se essa
descontinuidade é a intenção do desenho.

## 5. Retratação: o curso EXIGE controle lateral, e o contrato não é posicional

A primeira versão deste documento afirmava que o curso não exige controle lateral, porque
"os discos de serviço são declarados com 4 m" e o desvio lateral da reta de partida é 3.0 m.
**Está errado, e o erro é meu: eu li o raio de um texto em prosa em vez do kernel.**

Lido do fonte congelado da própria auditoria:

```
src/gpu/school.wgsl:196   inside_passenger   = distance(pos, goal) <= p.goal_radius
src/gpu/school.wgsl:197   inside_destination = distance(pos, exit) <= p.goal_radius
src/gpu/school.rs:219     goal_radius_m: 3.0     <- o raio que ACEITA
src/gpu/school.rs:220     area_radius_m: 4.0     <- zona de FRENAGEM, nao de aceitacao
```

O raio de aceitação é **3.0 m**, e o desvio lateral é **3.0 m**. A reta de partida é
**tangente** ao disco: margem zero, no ponto único de maior aproximação. Não há folga — o
corpo tem de corrigir lateralmente. O 4.0 m é a zona de frenagem (`school.wgsl:363,381`).

E o `STATION_MIN_SEPARATION_M = 8.0` que aparece no JSON **não é raio nem gap de disco**: é
a restrição mínima de colocação de estação verificada em `service_course.rs:274`.

**A segunda afirmação errada era "a aceitação é posicional, não temporal".** O contrato do
kernel cobra um envelope longitudinal inteiro, pelos códigos de falha declarados no
cabeçalho do `school.wgsl`:

| falha | o que cobra |
|---|---|
| 8 | entrou na área de parada sem nunca atingir `v_min` (3.0 m/s) |
| 10 | embarque prematuro ou **em movimento** |
| 14 | chegou ao alvo e **não completou a parada mantida** (`hold_ticks` = 128) |
| 15 | embarque fora do raio do passageiro (os 3.0 m) |
| 17 | entrou na zona de frenagem abaixo da velocidade mínima de entrada |

Então aceitar exige: chegar acima de 3 m/s, frear, parar abaixo de 0.3 m/s dentro de 3 m da
estação e **sustentar 128 tiques**. A razão de 0.3 % entre aceitos e pedidos não é sinal de
que pedir é de graça — é o custo desse envelope. E isso explica com precisão o traço do
campeão: parado em 64–76 % dos tiques, freio em 23–52 %, 20–30 m percorridos num curso de
18 m.

## 6. O piso continua valendo a pena — mas sem predição minha

O fundador não é o piso: é um cérebro aleatório que fica bloqueado, a referência "anda sem
rumo". O piso continua sendo **a coisa simples**: uma política de comando fixo que percorre
o trajeto e cumpre o envelope — acelerar, frear, parar dentro dos 3 m, manter 128 tiques,
pedir. A máquina existe (portão 2 de `RESEARCH-GATES-20260924`, política linear de pesos
analíticos rodada na GPU).

**Mas eu retiro a predição de que ela completa 4/4.** Com o envelope acima — velocidade
mínima na chegada, parada mantida, tangência lateral de margem zero — não tenho base para
prever que um cronograma fixo passe. O valor da medição não mudou: ela é o único jeito de
saber se o 4/4 mede competência de serviço ou o acerto de um cronograma. O que mudou é que
eu não sei o resultado, e dizer que sabia era o mesmo erro de novo.

## 7. Duas pendências

1. **A auditoria 4 não rodou.** Os 24 m estão pré-registrados e não há diretório. Ela muda
   distância, colocação de estação e orçamento juntos, como o próprio protocolo diz — e
   deveria levar o seu próprio piso.
2. **"Sem contato" é quase automático aqui.** Zero tique de contato e zero joule em todas as
   auditorias, num curso reto, sem outros agentes e com discos disjuntos. O protocolo já
   separa esse relato do de sucesso, o que está certo; vale só não ler o zero como prova de
   condução cuidadosa. Ele volta a informar quando houver cruzamento ou trânsito.
