# Cérebro emergente na GPU — contrato

Leonardo: *"quero ver eles evoluírem, não quero que faça coisas hardcode para eles serem mais
inteligentes, preciso ver a inteligência surgir, ver a distribuição e afins. Poderia fazer o
modelo ter um cérebro virtual mesmo, onde tem uma parte que cuida da visão, outra memória e
afins, como realmente o cérebro humano é"*.

Fonte do mecanismo: `~/projects/LLM/neuromod/docs/08-desenvolvimento-alocacao-frequencia.md`.

---

## O que muda de rumo

O laço todo-GPU funciona (5 kernels validados, seres dirigindo por decisão do cérebro), mas
**0/60 completam volta**. O diagnóstico é que o `CHAMPION.dna` foi evoluído DENTRO do
`motor.rs`, contando com dois módulos que o laço GPU não tem: o *curvature speed governor* e o
*slow brain*.

**A saída NÃO é portar esses dois.** Eles são exatamente as muletas hardcoded que o Leonardo
não quer. A saída é **evoluir um campeão dentro do laço GPU, do zero**, com um cérebro cuja
organização EMERJA.

---

## Princípio central (do doc 08)

> Você não aloca neurônios por região. Você projeta quatro forças locais — **cronograma,
> aferência, competição e custo** — e a anatomia emerge. Nenhum mecanismo precisa de
> conhecimento global da arquitetura.

E o critério de medida:

> **A fronteira entre "visual" e "auditivo" não é desenhada. Ela é onde a correlação cruzada
> cai.**

---

## As quatro forças, e como implementar cada uma

### 1. Aferência — como uma região vira "visual"

Evidência (ferrets, Rakic): redirecionar a retina para o córtex AUDITIVO faz ele desenvolver
mapas visuais. **A identidade da região vem da ENTRADA que ela recebe, não de genes dela.**

PROIBIDO:
```
regioes = { "visual": 200, "memoria": 150, "motor": 120 }   // ❌ alocação explícita
```

FAZER — folha equipotente + gradientes fracos (análogo a FGF8/Emx2):
```
pos  = grade_2d(N)                 // a folha cortical, (N,2)
P(aferência_k → unidade_i) ∝ exp(−‖pos_i − alvo_k‖² / σ²)
```
`σ` grande → modalidades se sobrepõem no início e segregam depois por atividade.
Os `alvo_k` são pontos na folha, NÃO rótulos de região.

### 2. Competição — a fronteira emerge da correlação

Hebb com **normalização subtrativa** (recurso finito):
```
Δw_ik = η · ( a_i·a_k − α·w_ik·Σ_j a_j·a_k )
```
Entradas que disparam juntas ficam juntas; descorrelacionadas se separam. Isso reproduz
sozinho colunas de dominância, mapas retinotópicos e fronteiras entre áreas.

### 3. Frequência — o canal que faltava

Correlação não é "ativo ao mesmo tempo", é **em fase, na mesma banda**. Cada unidade tem seu
τ, e o banco deve ser **log-espaçado** (razão ~2,3× entre bandas vizinhas):

| banda | f (Hz) | τ = 1/(2πf) | papel |
|---|---|---|---|
| gamma | 30–100 | 1,6–5,3 ms | processamento local |
| beta | 13–30 | 5,3–12,2 ms | estado motor |
| alpha | 8–13 | 12,2–19,9 ms | gating de atenção |
| theta | 4–8 | 19,9–39,8 ms | memória, sequências |
| delta | 0,5–4 | 39,8–318 ms | consolidação |

O sinal da plasticidade depende da frequência: ~1 Hz deprime, 50–100 Hz potencia. Versão
STDP com janela ±20 ms: pré antes de pós fortalece; pós antes de pré enfraquece.

**Já validado no nosso próprio ecossistema:** a evolução escolheu sozinha τ = 1,5 e 23,4
(spread 16×) sem instrução, e no Universe os campeões têm spread de τ de 40× contra 5,3× do
controle aleatório.

### 4. Ondas espontâneas — antes de haver mundo

A retina gera ondas correlacionadas ANTES de o animal abrir os olhos, e elas já esculpem os
mapas. **É preciso um gerador rítmico endógeno na fase inicial**, senão a segregação não tem
o que correlacionar.

### 5. Cronograma / custo

Neurogênese sequencial e exponencial (Finlay & Darlington): as partes começam e terminam de
proliferar em ordem fixa. Isso é o que produz alometria — dobrar o total multiplica o córtex
por 2,55 e o resto por 1,62, **sem nenhum gene específico para córtex**.

Teste de sanidade do doc: córtex/cerebelo mantêm razão ~1:4 em número de neurônios em quase
toda a evolução dos mamíferos. Se o modelo produzir razão muito fora disso, algo está errado.

---

## O que medir (é o que o Leonardo quer VER)

1. **Mapa de alocação:** para cada unidade da folha, de qual modalidade ela mais recebe.
   Plotar/imprimir a folha 2D com a modalidade dominante. As regiões devem aparecer sozinhas.
2. **Onde a correlação cai:** a fronteira. Medir correlação cruzada entre unidades vizinhas e
   mostrar que ela desaba nos limites das regiões.
3. **Expoente alométrico:** crescer o total de unidades e medir `b` em `N_parte = a·N_total^b`
   por região emergida. Comparar com córtex ~1,35 / resto ~0,70.
4. **Distribuição de τ:** histograma. Deve ficar log-espaçado se a lei valer.
5. **Evolução de verdade:** voltas por geração ao longo de N gerações, com a MEDIANA da
   população ao lado do máximo.

---

## Proibições

- **Nada de região pré-alocada.** Nenhum `if canal_é_visão { manda_para_area_visual }`.
- **Nada de muleta motora.** Sem *curvature governor*, sem *slow brain*, sem `throttle` forçado.
  Já foram pegas 4 provas fabricadas hoje (`throttle=0.3`, `v0=1.0 m/s`, teste comparando zero
  com zero, e um "escape de pânico"). O escape mediu: com ele 1744 m, sem ele 650 m — **2,7%
  do resultado era muleta.**
- **Prova real é HETEROGÊNEA.** Indivíduos em situações diferentes têm de dar números
  diferentes. Valor idêntico entre eles denuncia que o número não veio do que se quer medir.
- **Todo veredito sai ao lado da MAGNITUDE do sinal.** Um PASS com sinal zero tem de ser
  impossível de confundir com um PASS real.

## Restrição operacional

A máquina do Leonardo trava com build pesado. SEMPRE `nice -n 19 cargo build --release -j 2
--bin <alvo>`. Nunca sem `nice`/`-j 2`, nunca `cargo test`, nunca build do workspace inteiro,
nunca dois builds ao mesmo tempo, nunca o binário `universe`.
