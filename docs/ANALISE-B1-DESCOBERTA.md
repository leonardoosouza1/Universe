# A bateria b1: a evolução passou a funcionar, o desfecho virou binário, e o custo de colisão pode estar bloqueando a descoberta

Figura: `b1-descoberta.png`. Fonte: `universe-runs/neuromod-lab-taxi/b1_*/population_gen*.json`
e `manifest.json`. Desenho: fatorial {contrato consertado, contrato legado} × {com custo,
sem custo de colisão} × 6 sementes, 30 gerações, 48 seres, 12 000 tiques, pista 8005.

**Escopo desta análise:** só as **8 células completas em 30 gerações** —
`fixed_s{31,77,113}`, `legacy_s{31,77,113}`, `nocost_s{31,113}`. As outras 6 pararam em
26–27 gerações e 14 pedidos continuam na fila. Uma célula de 27 gerações não pode ser
comparada com uma de 30: é o confundidor de `ger_efetiva` que já custou a pred 180 no
laboratório do táxi.

---

## 1. A evolução funciona — primeira vez na pista

O avanço mediano da população vai de **~0 m na geração 0 para 455–4380 m na geração 29**.
Para comparação, a era `u1v`/`u10b` mostrava ~1600 m de distância percorrida em três
voltas, e a bateria da mosca 39–201 m de avanço. Isto é outra ordem de grandeza, e é a
primeira vez que a seleção produz melhora grande neste mundo.

## 2. O desfecho é quase binário — e censurado

Das 8 células completas, **2 nunca cruzam 1000 m de avanço mediano em 30 gerações** — e as
duas são a **mesma semente (113)**, nos dois braços com custo de colisão. Entre as que
cruzam, a geração do cruzamento é: **14, 18, 21, 22, 22, 26**.

| endpoint | coeficiente de variação entre células |
|---|---|
| avanço final (contínuo) | 0.55 |
| geração da descoberta | **0.20** |

A geração da descoberta tem **2.8× menos ruído relativo**. E o endpoint contínuo é
inutilizável no n disponível: as diferenças consertado-menos-legado por semente são
**−1109, +1778, −134 m** (troca de sinal), média +178 m, e o **MDE com n = 3 é 4547 m** —
maior que o avanço inteiro da célula mais lenta (455 m).

**Recomendação de método:** o desfecho primário deveria ser a **geração da descoberta**,
analisada por **sobrevivência** (Kaplan–Meier + log-rank), não teste t sobre o avanço
final. Dois de oito censurados é exatamente dado censurado à direita, e a análise de
sobrevivência **usa** as células censuradas em vez de descartá-las. Com 6 sementes × 4
braços isso fica viável.

## 3. Contrato consertado contra legado: nulo no avanço, 3–4× menos impactos

No avanço, nulo e não mensurável (acima). Mas na taxa de impacto — impactos médios por ser
divididos pela distância média, que é o estatístico certo porque o ser mediano tem zero
impactos:

| semente | consertado | legado | razão |
|---|---:|---:|---:|
| 31 | 0.69 /km | 2.07 /km | **3.0×** |
| 77 | 0.42 /km | 1.58 /km | **3.7×** |
| 113 *(nunca dirige)* | 2.96 /km | 1.18 /km | 0.4× |

Nas duas sementes que dirigem o conserto reduz a taxa de impacto em 3–4×, consistente em
direção. Na semente que não dirige o padrão se inverte — o que faz sentido, porque ali a
taxa é dominada por bater parado. **O benefício do conserto aparece em impactos, não em
avanço** — e impactos é o canal de menor variância. Vale registrar como desfecho
secundário. n = 2 entre as que dirigem: direção, não resultado.

## 4. O custo de colisão pode estar bloqueando a descoberta

É o achado mais interessante da bateria.

| semente | com custo | sem custo | | taxa de impacto |
|---|---:|---:|---|---|
| 31 | 2780 m (cruza na ger. 21) | 4078 m (ger. 22) | +1.5× | 0.69 → 0.82 /km |
| 113 | **455 m (nunca cruza)** | **4062 m (cruza na ger. 26)** | **8.9×** | **2.96 → 0.47 /km** |

A semente 113 não descobre dirigir com o custo, e descobre sem ele. E o ponto que fecha o
argumento: **a taxa de impacto CAI** quando o custo sai (2.96 → 0.47 por km). O ser sem
custo não está batendo mais — ele está dirigindo.

**Mecanismo proposto.** No começo da evolução todo mundo bate, porque ninguém sabe dirigir.
Penalizar a batida seleciona contra as linhagens que se mexem, e a população converge para
o piso "não se mexe, não bate". É a mesma lógica do piso do prêmio do mundo do táxi numa
roupa nova: **o piso vence a tentativa**. O custo é uma pressão de acabamento aplicada
antes de existir o comportamento que ela deveria refinar.

n = 2 sementes. É direção, não resultado — mas é uma direção grande, com mecanismo, e
testável por um experimento barato.

## 5. Uma correção à minha recomendação do ciclo passado

Eu recomendei trocar o desfecho primário para **eficiência = avanço / distância**. Nesta
bateria ela **satura**: 99–102 % na geração 29 em todas as células. Ela discrimina no
regime quebrado (bateria da mosca: 12–42 %) e não discrimina quando o carro já dirige.
Então: **eficiência como verificação de saúde** (o carro está dirigindo ou circulando?) e
**avanço como desfecho**, com a geração da descoberta como primário.

*Nota de definição:* a eficiência passa de 100 % (até 104.7 %). Isso é esperado se cortar o
lado de dentro da curva deixa o caminho percorrido mais curto que o arco de linha central
que conta como avanço — a "volta ideal" do automobilismo. Vale confirmar que é essa a
semântica pretendida de `progress_m` contra `distance_m`, e não um artefato de amostragem.

---

## 6. O que eu rodaria, em ordem

1. **Fechar o fatorial.** 14 pedidos na fila, 6 células truncadas em 26–27 gerações.
   Terminar ou descartar — não comparar com as de 30.
2. **Trocar o desfecho primário para geração da descoberta + log-rank**, com o limiar
   registrado antes (1000 m é o que eu usei aqui; o valor deve ser declarado, não escolhido
   depois de ver as curvas).
3. **Teste de currículo, com predição registrada.** Evoluir sem custo de colisão até o
   avanço mediano cruzar o limiar, e então ligar o custo pelas gerações restantes.
   **Predição:** o braço com currículo chega à geração 29 com avanço dentro de 10 % do
   braço sem custo **e** taxa de impacto dentro de 30 % do braço com custo. Se chegar, o
   custo é pressão de acabamento e não de descoberta — e isso é um resultado geral sobre
   *quando* aplicar uma pressão de seleção, não só sobre esta pista.
4. **Semente 113 como caso de estudo.** Ela é a única que falha, falha nos dois contratos e
   é destravada pelo custo. Vale um retrato: o que a população dela está fazendo na geração
   15 que as outras não estão.
