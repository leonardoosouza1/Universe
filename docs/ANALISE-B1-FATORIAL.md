# A b1 completa: o custo de colisão não compra segurança, e possivelmente custa a descoberta

Figura: `b1-fatorial.png`. Fonte: `universe-runs/neuromod-lab-taxi/b1_*/population_gen*.json`.
Fatorial **{contrato consertado, legado} × {com custo, sem custo de colisão} × 6 sementes**
(31, 77, 113, 137, 191, 233), 30 gerações, 48 seres, 12 000 tiques, pista 8005 — **24
células, todas completas**. Sucede a análise das 8 células parciais (`ANALISE-B1-DESCOBERTA.md`).

Desfecho primário: **geração em que o avanço mediano da população cruza 1000 m**, analisado
por sobrevivência. O limiar e o método foram declarados no documento anterior, antes destas
18 células terminarem.

**Nota de inventário:** `b1_fixed_nocost_s{113,137,191,233}` é **bit a bit idêntico** a
`b1_nocost_s{113,137,191,233}` — mesmas flags, mesmos números. São quatro corridas
redundantes (~4 h de GPU). Excluídas da análise para não duplicar sementes.

---

## 1. O desfecho, por célula

| semente | consertado, com custo | legado, com custo | consertado, sem custo | legado, sem custo |
|---|:--:|:--:|:--:|:--:|
| 31 | 21 | 22 | 22 | **15** |
| 77 | 18 | **14** | 17 | 18 |
| 113 | — | — | 26 | 21 |
| 137 | — | — | — | 21 |
| 191 | — | — | 27 | 22 |
| 233 | — | — | — | — |
| **descobrem** | **2/6** | **2/6** | **4/6** | **5/6** |

(— = nunca cruzou em 30 gerações.)

O desfecho é fortemente estruturado pela semente: 31 e 77 descobrem nos quatro braços, 233
não descobre em nenhum. É exatamente por isso que o pareamento por semente é obrigatório
aqui.

## 2. O custo de colisão: quatro testes, mesma direção, nenhum abaixo de 0.05

| teste | efeito | p |
|---|---|---|
| descoberta, Fisher exato | 9/12 sem custo contra 4/12 com custo, razão de chances 6.0 | 0.100 |
| descoberta, log-rank | χ² = 3.16 | 0.075 |
| avanço final, pareado, contrato consertado | +1079 m | 0.100 |
| avanço final, pareado, contrato legado | +1686 m | 0.060 |

Quatro canais independentes — desfecho binário, tempo até o evento, e a medida contínua nos
dois contratos — apontam todos para o mesmo lado, e **nenhum cruza 0.05**. A leitura honesta
é essa: direção consistente, magnitude grande, evidência insuficiente.

**E o poder explica por quê.** Simulando o efeito observado, o teste desta bateria tem
**42 % de poder**. Para 80 % seriam **12 sementes por braço** (24 por grupo), o dobro do que
rodou. Isto é uma medida do instrumento, não uma desculpa: mesmo este fatorial — o melhor
desenhado do programa — está subdimensionado para o efeito que ele encontrou.

## 3. O achado mais forte é um NULO: o custo não compra segurança

O custo de colisão existe para reduzir impacto. Pareado por semente, n = 6:

| contrato | com custo | sem custo | menor com custo em | p |
|---|---:|---:|:--:|---:|
| consertado | 1.95 /km | 2.56 /km | 3 de 6 sementes | 0.64 |
| legado | 3.80 /km | 3.18 /km | 1 de 6 sementes | 0.67 |

**Nenhum efeito detectável na grandeza que o custo existe para controlar** — e no contrato
legado o sinal é até invertido, com o braço sem custo batendo *menos*. Somando com o §2: a
pressão não entrega o que promete e aponta para custar descoberta. Isso é mais forte que o
resultado do §2, porque um nulo sobre o efeito *pretendido* não precisa de poder para ser
incômodo: se a pressão funcionasse, ela apareceria justamente no canal que ela manipula.

O mecanismo continua o do documento anterior: no começo da evolução todo mundo bate, porque
ninguém dirige. Penalizar a batida seleciona contra as linhagens que se mexem, e a população
converge para o piso "não se mexe, não bate" — **o piso vence a tentativa**, o mesmo padrão
do piso do prêmio no mundo do táxi.

## 4. O contrato consertado: nulo na descoberta, menos impacto

Descoberta: 6/12 contra 7/12, log-rank **p = 0.53**. Nulo, e desta vez o nulo não é por
falta de poder — os dois braços são quase idênticos.

Na taxa de impacto, pareado por semente:

- com custo: consertado **1.95** contra legado **3.80** /km, diferença −1.85, IC95
  [−4.18, +0.48], p = 0.097, **menor em 5 de 6 sementes**
- sem custo: consertado **2.56** contra legado **3.18** /km, p = 0.42, menor em 4 de 6

Direção consistente nos dois níveis de custo, mais forte com custo. É a assinatura do
conserto, e ela está no canal de impacto, não no de avanço. Deveria ser o desfecho
secundário registrado.

*(Nota de estatístico: são médias das taxas por semente, o que é o que o teste pareado exige.
A razão entre as médias agrupadas dá números diferentes — 1.12 e 2.93 — porque as células que
não dirigem têm distância pequena no denominador. Os dois são legítimos; só não se misturam.)*

## 5. O que eu rodaria

1. **Currículo, com a predição já registrada.** Sem custo até cruzar 1000 m, custo ligado
   depois. **Predição:** avanço na geração 29 dentro de 10 % do braço sem custo *e* taxa de
   impacto dentro de 30 % do braço com custo. Se bater, o custo é pressão de acabamento e
   não de descoberta — e o resultado é geral sobre *quando* aplicar uma pressão, não sobre
   esta pista. Com 6 sementes já mede, porque o contraste é contra dois braços conhecidos.
2. **Dobrar o n do eixo do custo, não dos quatro braços.** O contrato deu nulo com poder;
   gastar GPU nele é desperdício. Doze sementes em `nocost` contra `fixed` custa metade de
   repetir o fatorial e leva o poder a 80 %.
3. **Semente 233 como caso de estudo.** Não descobre em nenhum dos quatro braços. Ou é uma
   pista/formação intratável, ou é um bug de inicialização. Um retrato dela na geração 15
   responde qual, e é barato.
4. **Não repetir `fixed_nocost`.** É `nocost` com outro nome.
