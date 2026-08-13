# Diário: as voltas completas — o relógio, não o cérebro

Leonardo pediu voltas completas e mais tempo de vida. O caminho até elas foi principalmente
**desfazer uma medição errada minha**, e isso é o que este diário registra.

---

## 1. O zero que não era fracasso

`≥2 voltas = 0,0%` em toda geração, em duas condições independentes de treino. Eu estava
prestes a creditar isso ao cérebro e evoluir contra.

Zero cravado em condições independentes não é dificuldade — é parede. A conta que resolveu
(`scripts/teto_de_voltas.py`), feita ANTES de olhar mais dados:

```
pista 1658 m
volta mais rápida FISICAMENTE possível: 50,5 s
   (perfil de velocidade: teto de curva √(μg(r+W/2)) + passada de frenagem
    + passada de aceleração, iteradas 4× porque o circuito é FECHADO —
    sem isso o começo da pista não sabe da curva que existe logo antes dele)
orçamento: 120 s  ->  2,38 voltas no teto OTIMISTA
```

O carro real não anda no teto. Duas voltas nunca couberam.

## 2. O experimento controlado

Mesmo campeão de partida, mesma semente, mesma fração exploradora. **Só o relógio muda.**
A geração 1 é livre de confundimento: nenhuma evolução aconteceu ainda.

| gen | ≥2v a 120 s | ≥2v a 180 s | melhor a 120 s | melhor a 180 s | MEDIANA a 120 s |
|---|---|---|---|---|---|
| 1 | 0% | 28% | 1,93 v | 2,89 v | 0,42 v |
| 2 | 0% | 20% | 1,74 v | 2,73 v | 0,40 v |
| 3 | 0% | 10% | 1,77 v | 2,70 v | 0,40 v |
| 4 | 0% | **42%** | 1,94 v | 2,96 v | **1,78 v** |

**0,0% contra 25,0%, com o mesmo cérebro.**

A coluna mais dura é a última: na geração 4, a **mediana** da população cobria 1,78 voltas e
a contagem inteira ainda escrevia 0%. O degrau engolia 89% de uma volta na mediana.

E o estrago passava da leitura: `lap_bonus = 50000` domina a fitness. Com a segunda volta
inalcançável, o gradiente que puxaria para ela estava cepado — **a seleção também sofria.**

## 3. A física se validou sozinha

O teto de 50,5 s/volta foi calculado antes de ver estes dados:

| orçamento | melhor observado | teto | eficiência |
|---|---|---|---|
| 120 s | 1,93 v | 2,38 v | **81%** |
| 180 s | 2,89 v | 3,56 v | **81%** |

Dois orçamentos independentes, mesmo fator. Um modelo que errasse a física daria fatores
diferentes. É o que me deixou confiar no número em vez de só no experimento.

## 4. O orçamento novo, escolhido por conta

220 s. Mantendo os 81%, três voltas exigem ~187 s: a terceira fica confortável e a quarta
vira alvo de esticada. **De propósito com folga** — encostar a métrica no teto foi
exatamente o erro que gerou o zero.

Resultado imediato (gerações 1–5, `telemetry/treino_voltas/`):

```
GEN 1: 3+v=18  net_max 5782m = 3,49 voltas
GEN 2: 3+v=32  net_max 5791m  mediana 5116m = 3,09 voltas
GEN 4: 3+v=41  net_max 6000m  mediana 5621m = 3,39 voltas
GEN 5: 3+v=23  net_max 5841m
```

**A MEDIANA da população passou a fazer mais de três voltas.** Antes a contagem dizia zero.

## 5. O que apareceu quando o relógio esticou: o degrau do grace

A geração 3 morreu inteira nas DUAS repetições:

```
r1  GERAÇÃO 3 | Fim: Extinção | Duração: 62,0s | SBLD:54 PINADO:1 STAG:4
r2  GERAÇÃO 3 | Fim: Extinção | Duração: 58,0s | SBLD:45 PINADO:14 STAG:1
```

A causa está em `motor.rs:2558`:

```rust
// v3: 3200 frames (50s). ... Agora 50s de bootstrap + 70s de pressão seletiva real
let out_of_grace = racer.grace_ticks >= 3200;
```

O comentário diz a premissa em voz alta: **"50s + 70s" é uma geração de 120 s.** O grace é
uma constante absoluta calibrada para um orçamento que eu acabei de mudar.

Aos 50 s a pressão entra como DEGRAU: SpeedBleed passa a valer para todo mundo ao mesmo
tempo, e quem estiver devagar naquele instante morre junto. Nas gerações 1, 2, 4 e 5 a
população estava rápida na hora e sobreviveu; na 3, não. É uma moeda ao ar, e ela seleciona
"esteja rápido exatamente em t=50 s" em vez de "seja consistentemente bom".

O `wall_time` piora: ele acumula DURANTE o grace, então no frame 3200 quem já raspou 9–21 s
de parede morre instantaneamente (o log mostra `PINADO frame 3200` com `wall_time=13,28s`).
Não é uma morte por raspar — é a conta inteira do bootstrap sendo cobrada de uma vez.

**Não mexi nisso agora, de propósito:** alterar a paisagem de seleção no meio de um treino
invalidaria a corrida. Fica registrado para depois, com o conserto óbvio sendo rampa em vez
de degrau — e com a obrigação de MEDIR, porque "obviamente melhor" já me traiu várias vezes.

## 6. Por que a evolução seguia plana: mais dois degraus

Consertado o relógio, a população passou a dar 3 voltas — mas a tendência ficou em 0,0–0,4σ
em 26 gerações. Procurando o motivo, apareceram dois mecanismos, ambos da mesma doença.

### 6a. O fingerprint saturado — a diversidade nunca esteve ligada

```
[NICHE] gen=1  pool=10 clusters=1 largest=10
...  em TODA geração de TODO treino
```

`BehaviorFingerprint::from_weights` acendia o bit de uma região se
`energia > (nº de pesos)/10`. Duas falhas empilhadas:

1. **Unidades trocadas** — compara SOMA DE MAGNITUDES contra CONTAGEM/10. Os pesos entram
   como `w*127`, então **um único peso não-zero já estoura o limiar**. Fator ~1270×.
2. **Limiar absoluto não serve para pesos densos** — `CHAMPION.dna` tem 52,6% de não-zeros;
   mesmo com as unidades certas, 52,6% passa de 10% em toda região. Há teste provando que
   consertar só a unidade NÃO resolveria.

Conta: 21408 pesos / 64 = 334 por região; limiar 33; energia ≈ 22.317. Saturado por três
ordens de grandeza. O fitness-sharing dividia a fitness de todos pelo mesmo 10 — fator
uniforme, que não separa ninguém.

Conserto por POSTO (as 32 regiões de maior energia): não há limiar para calibrar, metade
acende por construção em qualquer densidade. No campo virou `clusters=2 largest=6`.

> Os testes que existiam usavam vetores de 256 elementos uniformes e passavam **por causa da
> saturação** — um deles afirmava `hash == u64::MAX`. Codificavam o bug.

E derruba uma calibração histórica: o comentário `v14: tentou 10, piorou (tudo 1 cluster)`
foi escrito quando toda distância era 0. **Aquela conclusão é nula.**

### 6b. O degrau DENTRO da fitness

```rust
lap_bonus = total_laps as f32 * 50000   // INTEIRO, domina prog (~5800 m) por 9×
```

| carro | bônus | contra o de 3,01 |
|---|---|---|
| 3,01 voltas | 150.000 | — |
| 3,99 voltas | 150.000 | **idêntico** |
| 4,00 voltas | 200.000 | +33% |

A população vive exatamente aí (melhor carro 3,72 de um teto de 4,36). **Ir de 3,0 para 3,99
voltas — a melhoria que se quer selecionar — não valia nada.** O mesmo degrau que escondeu as
voltas no relatório estava dentro do SINAL DE SELEÇÃO.

Contínuo (`prog / comprimento_da_volta * bonus`) atrás de flag, para o A/B ser possível.

## 7. Dois erros meus nesta etapa, e as ferramentas que sobraram

**Confiei numa tendência carregada por uma geração.** O `+2,17pp / 2,2σ` do contínuo virava
0,4σ ao tirar duas gerações. `scripts/tendencia.py` agora recalcula a INCLINAÇÃO em janelas
encolhidas — inclinação e não σ, porque σ depende de n e oscila.

**Ia cometer a falácia da diferença de significâncias.** "Contínuo 2,0σ, degrau 0,2σ, logo
diferem" é inválido; o teste certo regride a DIFERENÇA pareada. Feito isso, o resultado caiu
de "óbvio" para 1,4σ. A ferramenta faz a interação sozinha quando recebe duas condições.

## 8. O que continua em aberto

**A monocultura é real.** Com o fingerprint honesto, o elite colapsa para um cluster nas duas
condições do A/B. Não era artefato. Instrumentei `dist_med`/`dist_max` par a par porque
`clusters=1` sozinho não distingue monocultura genuína de limiar frouxo — e o limiar precisa
ser recalibrado do zero, já que a calibração antiga é nula.

## 9. A regra que sobrou

Terceira roupa do mesmo erro numa sessão só:

| onde | o que escondia | mecanismo |
|---|---|---|
| `net_max` | metade da população parada | métrica de EXTREMO |
| fração exploradora | um 4,3σ que não existia | janelas DIFERENTES |
| `≥2 voltas` | carros a 1,86 de 2 | contagem INTEIRA na fronteira |

> **Zero cravado (ou 100% cravado) é quase sempre a métrica batendo num limite, não o modelo
> falhando.** Calcular o teto por conta fechada antes de gastar GPU. Se o observado está a
> mais de 85% do teto, o problema é a métrica — troque por contínua.

`scripts/relatorio_voltas.py` já traz voltas fracionárias ao lado da contagem inteira, e
avisa sozinho quando o melhor carro passa de 85% do teto do orçamento.
