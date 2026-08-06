# Como acrescentar corpo e ambiente sem quebrar o experimento

Guia para a próxima fase — tração diferente, pistas diferentes, gasto de energia, e o resto
do corpo. Sai de quatro reestruturações pagas na bancada de sobrevivência em Python
(`~/projects/LLM/new/CORPO_DIARIO.md`), onde um mundo aparentemente rico se revelou trivial.

---

## A lei

> **Toda ação protetora precisa custar numa MOEDA DIFERENTE daquela contra a qual protege.**

Abrigo protege da predação → seu preço tem de ser energia (forrageio perdido).
Esforço rende energia → seu preço tem de ser térmico.

Esse cruzamento é o que torna o ótimo **interior** (nem sempre, nem nunca) e o que o faz
**depender da situação**. Sem ele não há nada para o cérebro decidir.

**Traduzindo para o carro:** frear protege da batida, então o preço do freio tem de ser tempo
(posição perdida). Acelerar rende tempo, então o preço da aceleração tem de ser aderência ou
temperatura de pneu. Se frear fosse grátis, o carro ótimo andaria devagar sempre e todo
sensor de curva viraria decoração.

---

## O sintoma diagnóstico — rodar SEMPRE antes de evoluir

**Uma política degenerada — *sempre X*, *nunca X* — vence a política que usa os sensores.**

Medido na bancada: com abrigo grátis, "sempre abrigo" dava **100%** de sobrevivência contra
92,5% da política que decidia pelo sensor. O sensor não decidia nada, e o canal que carregava
65% do ganho teórico era inalcançável. O experimento teria dado zero, e eu teria lido esse
zero como refutação da tese.

Ao acrescentar tração/energia/pistas, rodar antes: *acelerador sempre no fundo*, *nunca*,
*freio sempre*, *nunca*. Se alguma delas empatar com o campeão, a variável nova não está
acoplada e é enfeite.

---

## O corolário da colinearidade

Não basta separar as ações; os **custos não podem ser a mesma função da mesma variável**.

Medido: `C_MOV·v²` (energia) e `κ·v²` (aquecimento) são colineares — o térmico virava só um
custo de movimento maior, nunca mordia antes do energético, e saber a fase do dia não mudava
nada. **Dois custos colineares = um canal decorativo.**

No carro: se "gasta mais energia" for outra função de `v²`, ela não acrescenta decisão
nenhuma ao que o consumo de aderência já impõe. Para acrescentar decisão, o custo novo tem de
depender de algo que os outros não vejam — o estado do pneu, a molhada da pista, o
combustível restante contra as voltas que faltam.

---

## Os dois modos de falhar

| | sintoma | por que arruína |
|---|---|---|
| **armadilha mortal** | toda política morre | não há o que descobrir |
| **berçário** | toda política vive | a morte não pressiona; o corpo vira espectador |

As quatro reestruturações alternaram entre os dois: superaquecimento universal (armadilha) →
fadiga saturada, um dos quatro estados virou constante → predador com dano de equilíbrio 0,59
contra letal 1,00, matematicamente incapaz de matar (berçário) → abrigo grátis (berçário) →
abrigo que bloqueia comida mas ainda isola do frio (armadilha, 24/24 políticas mortas).

**Calibrar pelo ponto de equilíbrio, nunca por chute.** Todas as quatro foram achadas com uma
conta de uma linha: `T_amb + τ·κ·E[v²] = 42,9` contra letal 43,0; `Dm_eq = taxa·dano·τ = 0,59`
contra letal 1,00. Fazer a conta antes custa um minuto e evita a rodada inteira.

---

## Antes de acreditar num mecanismo, medir o teto

A forma fechada prometia ganho de **24,20%** para adaptar a constante de tempo ao regime. O
teto **realizável**, medido no objetivo real com o regime dado de graça, deu **+0,11%**.
Duzentas vezes de perda na tradução, por três motivos medidos: o sinal do mundo não era o que
a fórmula supunha, a decisão é um limiar (e limiar destrói melhora de erro quadrático), e
mudança de ruído é absorvível por um limiar fixo.

**Regra:** antes de gastar evolução num mecanismo, rodar o "oráculo" — a mesma política com o
mecanismo ajustado pela verdade, de graça. Se der ~0, um resultado nulo depois não distingue
"a ideia está errada" de "não havia o que ganhar neste mundo".

E cuidado com a armadilha irmã: **comparar condições de informação com classes de política
diferentes não mede informação, mede graus de liberdade.** Medido — um "oráculo" com regra
binária rígida PERDEU para a política de sensor ruidoso com limiar ajustável, porque o ruído
randomizava a ação e implementava de graça uma política mista.
