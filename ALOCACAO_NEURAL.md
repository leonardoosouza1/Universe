# Alocação neural no mundo de corrida

Instrumentação para responder três perguntas sobre o cérebro do carro: **quanto cada parte
carrega**, **o que cada sinapse diz**, e **o que mudou entre duas gerações**.

Vem da bancada de sobrevivência em Python (`~/projects/LLM/new`, `~/projects/LLM/novo`), e
carrega para cá as guardas de método que lá custaram caro.

---

## 1. Quanto cada via carrega — `src/neuro.rs`

O cérebro já era feito de partes com pesos explícitos de mistura:

```
engine.rs decide():   w[0]·córtex + w[1]·reflexo + w[2]·sentinela
motor.rs:            slow_brain MODULA agressão, teto de acelerador e viés de esterço
```

O que faltava era medir o quanto cada uma contribui — e a medida certa não é o peso do portão.

> O peso do portão diz o quanto uma via é **USADA**. Só a ablação diz o quanto ela **SERVE**.

Isso não é preciosismo. Medido no exp14 da bancada Python: genes moduladores derivaram até
`b=−1,67` e `c=8,02` **enquanto a ablação daqueles mesmos genes custava exatamente +0,0%**.
Genes vagueiam livres justamente quando são invisíveis à seleção, então qualquer estatística
sobre o genoma mede deriva junto com função.

```bash
cargo build --release
scripts/alocacao.sh 4 3 150      # reps · gerações · carros
```

A ablação zera a intenção **depois** do portão, não os pesos do portão: assim o misturador
continua vendo o mesmo mundo e distribuindo os mesmos pesos, e o que se mede é a falta da
via, não uma mudança de comportamento do misturador. O veto duro da sentinela só cai em
`sentinela_total` — sem ele o carro bate em segundos e o número mediria "morreu cedo".

**Resultado (5 repetições por condição, campeão treinado, 2026-07-31):**

| via ablada | net_max | custo | uso do portão (cx/rf/sn) | vetos |
|---|---|---|---|---|
| base | 3198 ± 25 | — | 0,370 0,342 0,288 | 2583 |
| **córtex** | 170 ± 82 | **94,7 ± 2,6%** | 0,363 0,348 0,290 | **15171** |
| reflexo | 3061 ± 23 | 4,3 ± 1,0% | 0,378 0,351 0,271 | 4685 |
| lenta (modulador) | 3112 ± 21 | 2,7 ± 1,0% | 0,362 0,355 0,283 | 4037 |
| **marcha** | 3177 ± 18 | **0,7 ± 0,9%** | 0,376 0,351 0,273 | 3321 |
| sentinela (suave) | 3140 ± 17 | 1,8 ± 0,9% | 0,372 0,347 0,281 | 2010 |

> O portão dá **37%** ao córtex e **34%** ao reflexo — quase igual. O córtex carrega
> **94,7%**, o reflexo **4,3%**. Descolamento de **22×** entre o que o portão diz e o que
> importa. É a tese deste módulo, medida.

E o peso do portão mal se move entre condições (0,362–0,378 no córtex), inclusive quando a
via está desligada: ele continua distribuindo peso para quem não está dizendo nada.

Sinal secundário: sem córtex os vetos da sentinela vão de 2583 a **15171** — a rede de
segurança passa a pilotar sozinha.

**A predição da via `marcha` foi CONFIRMADA:** teto calculado 0,36%, medido **0,7 ± 0,9%** —
indistinguível de zero. Dar o câmbio ao cérebro é neutro.

**A predição da via `lenta` ficou PARCIAL:** eu previa custo alto em corrida (tarefa de
instante) contra os +0,11% do mundo de sobrevivência. Medido 2,7% — 25× mais que lá, o que
apoia a leitura taxa-vs-instante, mas longe de "caro" perto dos 94,7% do córtex. Modulação
segue sendo contribuinte menor em todo domínio testado.

### Predição registrada ANTES do resultado — a via `marcha`

O Leonardo perguntou se dar o câmbio ao cérebro ajuda *"ou pior, não sei kk"*. Em vez de
rodar e ver, dá para calcular o teto: `scripts/ponto_de_troca.py` resolve onde o torque na
roda da marcha seguinte passa o da atual.

| marcha | RPM ótimo de troca | perda da regra automática (85%) |
|---|---|---|
| 1ª | 100% | 0,00% |
| 2ª | 99% | 0,41% |
| 3ª | 97% | 0,66% |
| 4ª | 95% | 0,35% |

**Teto da via `marcha` para escolher ponto de troca: 0,36%.** O ótimo é segurar quase até o
limitador, a regra troca um pouco cedo, e isso quase não custa.

Logo, o resultado da bateria já vem interpretável:

- **custo ≈ 0,4%** → o cérebro só está afinando o ponto de troca, e não vale a pena
- **custo ≫ 0,4%** → ele usa o câmbio para algo que a regra fixa não faz, e a hipótese mais
  provável é **segurar marcha na curva** (a regra decide por RPM, que não sabe o que vem
  pela frente)

A sobreposição neural é o último `else if` de `motor.rs:2107`, alcançada com embreagem colada
e RPM entre 30% e 85% — **55% da faixa útil**. O cérebro tem o piso; a pergunta é se usa
melhor que a regra.

---

## 2. O que cada sinapse diz — `src/bin/sinapses.rs`

```bash
./target/release/sinapses CHAMPION.dna --top 12
./target/release/sinapses novo.dna --contra antigo.dna     # o que nasceu, morreu, engrossou
./target/release/sinapses CHAMPION.dna --neuronio 2        # só o motor de freio
```

Uma sinapse é uma aresta `canal → neurônio`. Com o nome do canal, vira uma frase sobre
pilotagem:

```
-2.00  SATURADA    inibe   curvatura a 70 m (com sinal)   → n0  MOTOR acelerador
+1.95  muito forte excita  curvatura a 70 m (com sinal)   → n2  MOTOR freio
+2.00  SATURADA    excita  curvatura a 120 m (com sinal)  → n1  MOTOR esterço
+1.92  muito forte excita  alinhamento com o fluxo        → n0  MOTOR acelerador
```

Ou seja: *vê curva a 70 m → tira o pé e freia; vê curva a 120 m → já começa a virar; está
alinhado com a pista → acelera.*

**`SATURADA` não é ênfase, é informação:** o `i8` do DNA trava o peso em ±2,0 (127/63,5). O
priming nasce em ±3,0 a ±10,0 e tudo isso vira ±2,0 no disco. Chamar de "forte" esconderia um
limite do formato.

**Canal sem origem localizada no código aparece como `canal N (sem nome mapeado)`.** Um nome
inventado num relatório de interpretabilidade é pior que índice cru, porque parece
conhecimento. Os nomes saem de onde os canais são escritos (`motor.rs:455-524, 480-513`) e do
priming esparso (`spawn.rs:64-90`).

---

## 3. Duas armadilhas que este código já evita

**O DNA v8 tem duas cópias dos pesos.** `fast_weights` é a versão TERNÁRIA {−1,0,+1} do
compressor; os pesos de verdade estão no chunk `CRAW` (±127 → ±2,0). Ler a fonte errada não
dá erro — dá um relatório silenciosamente falso, com todo cérebro parecendo vazio. A
ferramenta prefere `CRAW` e avisa em voz alta quando cai no outro.

**O simulador sobrescreve `CHAMPION.dna` no diretório de trabalho.** Um teste de 2 gerações
coroa um campeão fraco e apaga o treinado — aconteceu aqui, e só deu para desfazer porque o
arquivo estava versionado. Por isso `scripts/alocacao.sh` roda **cada corrida num diretório
isolado em /tmp**, com a sua própria cópia do DNA. Há também `dna_backup/`.

**As corridas não são reprodutíveis.** `spawn.rs` usa `rand::rng()`, thread-local e não
semeado. Comparar uma variante contra outra numa rodada só compara ruído — por isso o script
faz repetições e reporta média com erro padrão.

---

## 4. Por que este caminho, e não o modulador

Em 2026-07-30 a bancada Python mediu duas coisas que decidem o que vale construir aqui:

**Ter um banco de τ variados é selecionado.** A faixa de constantes de tempo dos campeões
evoluídos ficou em **40×** (15,7 · 75,9 · 28,5 nas três sementes) contra **5,3×** de um
genoma aleatório. Confirma a observação do Leonardo de que os melhores carros tinham
neurônios rápidos *e* lentos.

**Trocar τ conforme o regime não paga.** Teto realizável medido com o regime dado de graça:
**+0,11%**, contra os 24,20% que a forma fechada prometia. E não é falta de esforço da
evolução — o teto foi medido por busca direta, com oráculo.

> Ter escalas de tempo variadas: sim. Modular a escala de tempo: não.

O teste de fumaça acima é a primeira réplica disso em outro domínio: a via lenta do carro
(que é exatamente um modulador — mexe em ganho e teto, não age direto) custou ≈0% quando
ablada. **Isto contraria a minha própria predição registrada** em `neuro.rs`, que dizia que
corrida, sendo crítica no instante, faria a modulação pagar. Não pagou.

Falta confirmar com repetições (`scripts/alocacao.sh`) — 2 gerações com 60 carros é ruído.
