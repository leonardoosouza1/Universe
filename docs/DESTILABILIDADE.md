# Destilabilidade — o que o clone que venceu o campeão significa

> ## ⚠ CORREÇÃO GRAVE (2026-08-16, auditoria adversarial) — TUDO ABAIXO ESTÁ SOB REVISÃO
>
> O override do piloto estava aplicado em `intents[0]` **depois** de `phys_states` já
> ter sido montado (`context.rs`: montagem na linha 989, `.collect()` em 1178, override
> em 1190, física em 1302). Nessa ordem o comando da fórmula **nunca chegava ao motor**:
> o único caminho que restava era `mem0` (linha 1328), ou seja UM canal de percepção
> (ch120, família memória). Prova que matou: os comandos aplicados ao carro foram
> bit-a-bit iguais aos do campeão nos 576 primeiros ticks, enquanto o intent gravado já
> diferia em 0,60 desde o tick 0 — e a divergência começou exatamente no tick seguinte
> ao primeiro em que `clip(intent0, 0, 1)` mudou.
>
> **O que as corridas mediram de fato: o cérebro campeão com um canal de memória
> contaminado pela fórmula.** Não "o clone dirigindo". Isso atinge o 108,6%, o colapso
> de 10% em pista nova e os 100,2% com portão — todos os três.
>
> Achado por agente cético dedicado a derrubar o resultado (o padrão de workflow
> adversarial do projeto pagou o próprio custo aqui). Ordem consertada, teste de
> regressão `piloto_entra_antes_da_fisica` trava a posição lendo o próprio fonte, e a
> bateria inteira foi relançada — agora com controle de DETERMINISMO (duas corridas
> idênticas) porque o mesmo cético mostrou que os 47 "controles de graça" mudaram entre
> corridas (um deles −5915 m). Os números corrigidos entram abaixo quando fecharem.
>
> Lição de método, sem atenuante: eu tinha declarado "divergência verificada, o override
> chega ao motor exatamente como resolvido" apoiado em ver a velocidade divergir no tick
> 577. Divergir prova que ALGO mudou, não que mudou pelo caminho que eu supus. O teste
> certo era comparar o comando APLICADO (throttle/steer/brake na física) com o intent
> gravado — dois campos que eu já tinha no obs.f32 e não cruzei.

> Leonardo (2026-08-16): *"então ele saiu melhor que o campeão, o que aprendemos com isso
> além do ruído? Como podemos aplicar isso já no projeto, o que isso fez em si no projeto"*

## O fato

Campeão m23-s77: ~24 mil parâmetros evoluídos em 80 gerações de GPU (L1 18432 + L2 4608 +
lento 1091 + τ + KAN + ativação). Clone: **1728 números resolvidos em 10 segundos de CPU**,
sem gradiente. Na mesma pista, mesma largada, mesmo corpo: **7156 m contra 6592 m**.

## O que isso ensina, além do "ruído se cancela"

### 1. O mundo atual NÃO EXIGE um cérebro (a leitura desconfortável)

Se uma matriz 576×3 dirige tão bem quanto 80 gerações de evolução, então a política ótima na
região que o ser visita é **quase linear** nas features pós-CfC. Isso não é elogio ao ridge —
é diagnóstico da PISTA. Bate exatamente com o "ainda está muito fácil" do Leonardo, e explica
achados que pareciam soltos:

- o maestro M4 empatou (não havia estado lento pra reger);
- o custo de fiação não selecionou (imposto constante, mas também: nada a reorganizar);
- promoção 33/33 gerações (nada segura o campeão).

**Um mundo que uma fórmula resolve não seleciona cérebro.** A escada de dificuldade
(`ESCADA_DIFICULDADE.md`) deixa de ser "deixar mais difícil por gosto" e passa a ter um alvo
mensurável: *tornar o mundo indestilável*.

### 2. Onde mora o valor do cérebro (medido, não suposto)

O clone manteve percepção + KAN + **CfC** vivos e substituiu só a decisão L1→L2. E o peso do
volante ficou dominado pelo **ch121 (família memória)**, com FÓVEA/PERIFÉRICA sendo os blocos
mais fracos do linear. Somado a "os EMAs extras não ajudaram porque X já é pós-CfC":

> a parte **contextual** (percepção + CfC) é o órgão que fica; a parte **pontual** (a decisão)
> é forjável.

A sessão-forja mediu o mesmo teorema no Qwen por outro caminho (caderno RBF substitui MLP,
nada substitui atenção). Dois ecossistemas, mesma lei.

### 3. Onde o cérebro ainda ganha: o freio

Erro do ridge por ação: acelerador 0,177 · volante 0,156 · **freio 0,453**. O resíduo aponta
exatamente onde a política deixa de ser linear. Não é detalhe: é o mapa do que falta.

## O instrumento que isso cria: índice de destilabilidade D

`tools/destilabilidade.py` (novo): dado um `obs.f32` de campeão, mede que fração da política é
extraível por forma fechada (R² por ação, held-out dos últimos 15% TEMPORAIS — split aleatório
vaza em série de 64 Hz).

| D | leitura | ação |
|---|---|---|
| alto (~1) | a evolução está refinando uma função linear | apertar a escada — o mundo é fácil demais |
| caindo | o cérebro passou a fazer algo que fórmula não pega | a dose do degrau está certa |
| baixo | política genuinamente não-linear/contextual | aqui evoluir vale o custo de GPU |

**D vira telemetria da escada**: medido a cada N gerações da bateria, ele responde
objetivamente à pergunta "o degrau que liguei tornou o mundo mais exigente ou só mais lento?".
Nenhuma métrica que temos hoje responde isso — completagem, mediana e voltas medem
desempenho, não EXIGÊNCIA.

## Aplicações imediatas no projeto (o que isso já muda)

1. **Medidor de dificuldade** (acima) — o uso mais valioso, e é de graça a cada campeão.
2. **Gen-0 destilada**: campeão → W em 10 s → semear a próxima evolução. Cuidado registrado
   (aviso da forja): **maldição do vencedor ~30%** — julgar mutantes-semente em corrida/semente
   FRESCA antes de promover.
3. **Baseline de regressão barato**: o clone é determinístico e roda em 1 matvec. Toda mudança
   de física/percepção pode ser testada contra ele sem gastar geração de GPU.
4. **Leitura de importância por sensor de graça**: |W| por canal, comparável com a ablação
   causal. Já rendeu um contraste: |W| linear diz memória/derivados; a ablação diz visão 77%.
   Não é contradição — é a diferença entre "o que o linear usa" e "sem o que o sistema morre".
5. **Piloto v2 não-linear** (aviso (c) da forja): forja_auto nos mesmos X/Y. Se o não-linear
   também não ganhar muito, é mais uma prova de que o gargalo é o MUNDO, não o controlador.

## VEREDITO CORRIGIDO (fiação consertada, 2026-08-16) — a destilação v1 FALHA

Números com o piloto REALMENTE no motor, corridas em diretório limpo, simulador
verificado determinístico (3 corridas idênticas, 48/48 seres bit-a-bit):

| corrida | ser0 | % do campeão | vel média | parado | cérebro reassumiu |
|---|---|---|---|---|---|
| campeão em 9001 (treino) | 6592 m | 100% | 22,2 | 8,2% | — |
| **clone puro em 9001** | **1027 m** | **15,6%** | 3,7 | 77,9% | — |
| clone + portão em 9001 | 6771 m | **102,7%** | 22,8 | 7,6% | 5,5% |
| campeão em 9007 (nova) | 5932 m | 100% | 20,4 | 6,6% | — |
| clone puro em 9007 | 2559 m | 43,1% | 9,0 | 35,1% | — |
| clone + portão em 9007 | 4203 m | 70,9% | 14,7 | 13,8% | 23,4% |

**A tese "evolução vira refino" FALHA na v1.** Um controlador linear resolvido sobre
20 mil pares (sentiu→fez) não dirige nem na pista em que foi ajustado: 15,6% do
campeão, 78% do tempo parado. O erro por tick de 0,26 (held-out honesto) COMPÕE em
malha fechada — cada desvio leva a um estado um pouco menos visitado, e o eixo de
menor resolução (freio) escapa primeiro. É exatamente a armadilha "rajada/deriva"
que a própria spec listou; ela não apareceu antes porque a fiação com bug fazia o
campeão dirigir.

**Achado que sobrevive e vale mais que o original: o portão.** Com o mesmo clone
ruim, devolver ao cérebro os ticks em que a fórmula não reconhece o estado leva de
15,6% para 102,7% (treino) e de 43,1% para 70,9% (pista nova). A taxa de devolução é
a medida direta de "onde a fórmula não sabe": **5,5% dos ticks na pista conhecida,
23,4% na nova**.

**Controle obrigatório em andamento** (`--portao-aleatorio p`, implementado agora):
abstenção na MESMA dose mas ignorando o estado. Sem ele não dá para distinguir
"o portão escolhe os ticks certos" de "qualquer respiro do cérebro na mesma taxa
salvaria". Resultado entra aqui — e se o aleatório empatar, o portão é decoração.

---

## HISTÓRICO — a versão anterior deste veredito (fiação com bug)

O que está abaixo foi medido com o override do piloto chegando só ao canal de
memória (ver aviso no topo). Fica registrado porque o FATO observado continua
estranho e testável: contaminar UM canal de memória (ch120) com um sinal linear
derivado do próprio replay levou o campeão de 6592 a 7156 m in-distribution e o
destruiu (600 m) em pista nova. Se replicar com fiação limpa, é neuromodulação
acidental — linha EXP10/neuromod, não destilação.

## E-PIL1 — VEREDITO: o clone NÃO generaliza, e o conserto mede o cérebro

**Predição registrada antes falhou nos dois braços — o que saiu é melhor.**

Primeiro, a limpeza: das 4 pistas novas, **3 eram teste inválido**. Em 8010/8150/8250
(perfis) campeão E clone rodaram 75-95% do tempo FORA da pista, com desvio de eixo de
35-52 m (meia-largura = 14 m); a projeção de arco satura e os dois deram o MESMO número
(9827,9 em 8250 — o canário de números idênticos pegou o artefato). O campeão m23 treinou
só em 9001 e 9003, ambas procedurais: **9007 é o único teste legítimo**, e nela os dois
ficaram 100% na pista.

| pista 9007 (nova, válida) | metros | vel média | parado | freio médio | volante \|·\| |
|---|---|---|---|---|---|
| campeão | 5932 | 20,42 | 6,6% | 0,00 | 0,59 |
| clone linear | **600 (10,1%)** | 2,28 | 78,8% | **0,56** | 0,10 |
| **clone + portão de abstenção** | **5944 (100,2%)** | 20,53 | 6,4% | — | — |

**Mecanismo da queda (medido, não suposto):** fora do que viu, o linear extrapola sem
limite justo no eixo que ajusta pior — o freio (erro 0,45 no held-out). Freio alto → carro
para → estado fica ainda mais estranho → trava para sempre: velocidade por quarto de
corrida 8,8 → 0,3 → 0,1 → 0,1 m/s. O cérebro, com tanh e dinâmica do CfC, não tem esse modo.

**O conserto e o número que ele revela:** portão de abstenção (512 referências do treino,
limiar = quantil 0,99 da distância² ao vizinho mais próximo). Fora do domínio o piloto
se abstém e **quem decide é o cérebro** — fallback melhor que o "frear/reduzir marcha" da
spec, já que frear é exatamente o que trava. Resultado: 100,2% do campeão na pista nova
com o cérebro reassumindo **9,4% dos ticks**; na pista de treino, 7512 m (114% do campeão,
acima até do clone puro) com o cérebro reassumindo **1,6%**.

> **É a medida que faltava: o cérebro é necessário em ~9% dos ticks numa pista nova e
> ~2% na conhecida. Os outros 90-98% são forjáveis.** "Onde o cérebro importa" deixa de
> ser opinião e vira porcentagem — e a arquitetura que sai disso é córtex (cérebro) +
> reflexo forjado (fórmula) + portão de confiança, a mesma forma do verificador que
> abstém do resto do ecossistema IARA.

Correções de método pagas neste experimento (as duas já tinham mordido hoje):
1. **Binário velho**: rodei `cargo test` achando que rebuildava o release — o portão saiu
   com `limiar inf` e 0,0% de abstenção. Sintoma clássico do canário.
2. **Diretório contaminado**: a corrida SALVA `evolve_best_gen0000.bin` no `--out`, então
   a segunda corrida no mesmo diretório retoma de outro genoma (9001 deu 7674 em vez de
   7156). Toda corrida de avaliação agora usa diretório limpo + cópia do campeão.
3. **Erro silencioso**: `thr_abster` é escalar e o parser só lia arrays, caindo em
   INFINITO sem reclamar. Consertado com parser de escalar + `[ERRO]` ruidoso + teste de
   regressão `portao_le_limiar_escalar_e_abstem`.

## Registro do experimento (predição feita ANTES)

**Predição registrada ANTES (2026-08-16, corridas lançadas):**
- Se o "ruído" do campeão NÃO era funcional (ridge só suavizou): o clone mantém ≥100% em
  pistas NUNCA VISTAS (9007, 8010, 8150, 8250) em ≥3 das 4.
- Se o ruído ERA generalização (exploração/robustez): o clone cai abaixo de 70% em ≥2 das 4 —
  e aí "aluno supera professor" vale só dentro do treino, que é a maldição do vencedor.

Pareado: mesma pista/semente/genoma, com e sem `--piloto-formula`, ser 0 dos dois lados.
Script: `/media/llm-data/scratch/epil1_generalizacao.sh`.

## Registro de honestidade

O 108,6% é UMA corrida de cada lado. Dois agentes céticos foram postos para derrubá-lo
(assimetria de comparação, métrica `voltas`=0 vs `pct_volta`=244%, variância entre corridas,
e se o override é híbrido com câmbio/ré). Veredito deles entra aqui antes de qualquer uso do
número como base de decisão.

## Bug colhido de graça pela bateria (2026-08-16)

A escada rodou 4 gerações e a de índice 2 saiu `med=0m max=0m morto=100%` com
`fitness 2377..2377 · valores distintos 1/48` (o próprio log gritou SELEÇÃO CEGA).
Não era a pista: a gen 3 na MESMA pista rendeu 7474 m. Era a largada — a lista de
waypoints fecha o laço repetindo o primeiro ponto (`world.rs:1214`), e
`--largada-aleatoria` sorteou justo esse índice: `wp == wp_next` → versor `[0,0,0]`
→ orientação degenerada, velocidade EXATAMENTE 0,00 com acelerador 0,4-0,6 aplicado
e RPM girando. Probabilidade 1/n_wp por geração ≈ 1% — numa bateria de 200 gerações,
~2 gerações inteiras viravam sorteio puro.

Consertado em `frente_do_waypoint()` (procura adiante o primeiro waypoint distinto),
provado com controle pareado: wi=105 saiu de `0 m / 1 nota distinta / 100% mortos`
para `1327 m / 46 notas / 0% mortos`, e wi=50 (controle) ficou **idêntico ao de
antes** — mudança cirúrgica. O log agora imprime `[largada] gen N nasce no wp k/n`;
sem isso o diagnóstico exigiu uma sonda dedicada.

## E-PIL2 — "caderno misto" (transferência da sessão-forja) — PREDIÇÃO REGISTRADA

A forja mediu 4× que **cobertura vence aglomeração**: roteador de especialistas perdeu
para um caderno ÚNICO com dados misturados (código CE 2,128 roteado vs 1,563 misto,
apesar de 99% de acerto no roteamento; centros especializados aglomeram e interpolam
mal na borda). O análogo aqui: em vez de um piloto por tipo de pista, um piloto só com
os ticks de VÁRIAS pistas.

Desenho: piloto-A = só 9001 (o atual) · piloto-M = 9001 + 9003 (as duas pistas de
treino do campeão), mesmas features, mesmo portão (quantil 0,99 das refs do conjunto
correspondente). Avaliação em 9007 (nova, o único mapa em que os dois seres ficam na
pista) e em 9001 (controle in-distribution).

**Predições, antes de rodar:**
1. A abstenção do piloto-M em 9007 cai abaixo dos 9,4% do piloto-A — é a medida direta
   de cobertura, e é o que "misturar" deveria comprar.
2. Metros em 9007 ficam ≥ os do piloto-A (5944 m); se caírem, a mistura DILUIU a
   política em vez de cobrir, e a lei da forja não transfere para controle contínuo.
3. Em 9001 a abstenção do piloto-M ≥ a do piloto-A (1,6%): mais referências de outra
   pista não podem tornar a pista de casa mais familiar — se cair, o limiar está
   frouxo e a métrica de cobertura não significa nada.

O ponto 3 é o controle que separa "cobertura real" de "limiar mais permissivo": o
quantil 0,99 é calculado no próprio conjunto de treino, então um conjunto mais
heterogêneo tem limiar MAIOR por construção. Sem esse controle, qualquer queda de
abstenção seria artefato do limiar, não cobertura.

## Controle do portão — PREDIÇÃO REGISTRADA (antes de olhar os dados)

Escrito antes de abrir qualquer CSV do diretório `controle/`. A forja registrou a
predição espelhada do lado deles no mesmo momento (malha aberta vs fechada).

1. **O portão bate o aleatório na mesma dose**, em ambas as pistas — margem grande
   em 9001 (≥2× os metros do aleatório) porque lá a dose é baixa (5,5%) e o resgate
   precisa cair exatamente nos ticks de fuga.
2. **Mecanismo previsto, e é o que separa as duas hipóteses**: a abstenção do portão
   vem em BLOCOS CONTÍGUOS (o estado sai do domínio e fica fora por um trecho),
   enquanto a aleatória vem em ticks ISOLADOS. Um tick solto de cérebro não tira um
   carro parado do lugar; um bloco tira. Previsão numérica: comprimento MEDIANO de
   sequência de abstenção ≥ 4 ticks no portão contra ≈ 1 no aleatório.
3. **Se o aleatório empatar**: a lei degrada para "o órgão precisa respirar X% do
   tempo, tanto faz quando" — mais barata de implementar e menos bonita. Qualquer
   desfecho paga, mas os dois dizem coisas MUITO diferentes sobre o que o cérebro faz.
4. Dose 0,200 (bem acima da medida) serve de terceiro ponto: se o aleatório só
   funciona com dose alta, a curva dose-resposta separa "escolher bem" de "intervir
   muito".

## Controle do portão — VEREDITO: o portão vale, e o mecanismo é a CONTIGUIDADE

Dose casada, sorteios independentes (`--portao-semente`), campeão = 100%.

| pista 9001 (dose 5,5%) | metros | % |
|---|---|---|
| portão | 6771 | 102,7% |
| aleatório (4 sorteios) | 909 · 3838 · 1070 · 7132 → **3237 ± 2532** | 49,1% |

O aleatório não empata: ele é LOTERIA. Um sorteio deu 7132 (foi esse que apareceu na
primeira rodada e me fez escrever "empatou"), outro deu 909. A média é metade do portão,
com desvio do tamanho da própria média.

**O mecanismo, medido pela telemetria nova de contiguidade:**

| corrida | abstenções | blocos | média/bloco | maior bloco |
|---|---|---|---|---|
| portão 9001 | 1109 (5,5%) | 247 | 4,5 | **345 ticks (5,4 s)** |
| aleatório 9001 | 1100 (5,5%) | 1099 | 1,0 | 2 |
| portão 9007 | 4682 (23,4%) | 514 | 9,1 | **1227 ticks** |
| aleatório 9007 | 4680 (23,4%) | 4600 | 1,0 | 2 |

**O portão não ganha escolhendo ticks avulsos melhores. Ganha entregando o volante por
TRECHOS INTEIROS quando o estado está fundo fora do domínio.** Piscadas de 1-2 ticks não
tiram um carro de uma excursão; um bloco de 5 segundos tira.

Minha predição registrada dizia "mediana de bloco ≥4 no portão contra ≈1 no aleatório".
A mediana é 1 nos DOIS — a maioria das abstenções do portão também é avulsa. **Errei na
mediana; o que separa é a CAUDA.** Registro o erro porque ele muda o desenho: quem for
implementar isso deve olhar o maior bloco, não o típico.

**Reformulação da lei** (mandada para a sessão-forja): não é "o órgão precisa respirar X%
do tempo" nem "o portão escolhe o tick certo" — é **o órgão precisa poder ASSUMIR e
SEGURAR**. A unidade de resgate é o trecho, não o tick.

### n=12: o efeito ENCOLHEU, como manda a lição de replicação

Sorteios (pista 9001, dose 5,5%): 909 · 932 · 1070 · 1070 · 3838 · 3847 · 5536 · 6318 ·
6387 · 6783 · 7132 · 7189 → **4251 ± 2526**. Portão 6771 = **1,59× a média** (+1,0σ).

O tamanho do efeito caiu de 2,1× (n=4) para 1,59× (n=12) — exatamente o padrão que a
lição `feedback_replicacao_vs_sigma` manda seguir: acompanhar o EFEITO conforme n cresce,
não o σ. Registro para não repetir o erro de cravar o número da amostra pequena.

E a distribuição é BIMODAL, não ruidosa: 4 de 12 sorteios falham feio (~1000 m), 6 de 12
chegam ao nível do portão (~6500 m), 2 ficam no meio. **5 de 12 alcançam ≥90% do portão.**

**Enunciado honesto:** o resgate aleatório funciona em cerca de metade das corridas. O
portão não é dramaticamente melhor que um sorteio de sorte — ele converte uma moeda em
certeza. Num campeão que precisa ser reproduzível, é isso que importa; num paper de
"olha como o portão é inteligente", não seria suficiente.

## PILOTO v2 (não-linear) — PREDIÇÃO REGISTRADA antes de rodar

Pergunta única: o erro por tick 0,1794 (pirâmide n=1024 + arctanh) fica ABAIXO do
limiar em que o erro deixa de acumular em malha fechada? O v1 (0,2620) não ficava —
1027 m, 15,6% do campeão, na própria pista de treino.

**Predição:** o v2 melhora mas NÃO dirige — espero entre 20% e 60% do campeão em 9001.
Razão: a queda de erro se concentra no acelerador e no volante; o freio, que é o eixo
por onde a fuga acontece, ainda erra 0,292 (contra 0,137 do volante). Se der ≥70%, a
tese de destilação ressuscita e eu estava errado sobre o mecanismo da fuga.

Barra de sucesso real, a que importa: o híbrido tem de bater o CÉREBRO SOZINHO
(6592 m em 9001, 5932 m em 9007) — não o clone v1.

## VEREDITO FINAL: a destilação FUNCIONA — com a base certa e com caderno MISTO

A predição registrada (v2 entre 20% e 60%) foi **refutada por cima**. Existe um limiar
de erro por tick entre 0,26 e 0,18 onde o erro DEIXA de acumular em malha fechada:

| piloto | erro held-out | 9001 (treino) | 9003 (treino) | 9007 (NUNCA VISTA) |
|---|---|---|---|---|
| cérebro campeão (referência) | — | 6592 m | 5242 m | 5932 m |
| v1 linear | 0,2620 | 1027 m (15,6%) | — | 2559 m (43,1%) |
| v2 pirâmide+arctanh, 1 pista | 0,1794 | **7485 m (113,5%)** | — | 4985 m (84,0%) |
| **v2 CADERNO MISTO (9001+9003)** | **0,1388** | 7139 m (108,3%) | **5575 m (106,4%)** | **6072 m (102,4%)** |

**O piloto misto bate o cérebro nas três pistas, inclusive numa que nunca viu.**

Verificação de que o piloto REALMENTE dirige (o teste que faltou na primeira rodada e
me fez publicar número falso): o comando APLICADO à física diverge do braço-cérebro no
tick **0** (no bug era só no 576) e a correlação entre o comando aplicado e o intent
pedido pelo piloto é 0,989 (acelerador) e 0,999 (freio) — a mesma do cérebro com o
próprio intent (0,988).

### Os dois ingredientes, em ordem de importância

1. **`arctanh` na saída** — ajustar no espaço PRÉ-compressão do `tanh` e devolver com
   `tanh`. Vale −23% de erro com ZERO parâmetro, mais que toda a não-linearidade da
   entrada (−12%). Mecanismo: 61-74% dos alvos vivem no platô onde mover 1,0 na
   pré-ativação move 0,004 na saída; o mínimo quadrado gastava capacidade raspando
   platô em vez de decidir.
2. **Caderno MISTO** (transferência da sessão-forja, lei "cobertura > aglomeração",
   4ª aparição lá e 1ª aqui): ajustar com ticks de VÁRIAS pistas em vez de uma.
   Fecha o buraco de generalização (84% → 102,4%) e custa 5 pontos na pista de casa
   (113,5% → 108,3%) — o trade que a lei prevê.

### O que isso destrava no projeto

Um campeão que custou **80 gerações de GPU** vira, em **10 segundos de CPU**, um piloto
que dirige melhor que ele — inclusive em pista nova. "Evolução vira refino" deixa de ser
tese e passa a ter número. E a leitura dos pesos vem de graça junto.

### O que ainda NÃO está resolvido

- O piloto não é um GENOMA: não dá para semear a população com ele diretamente. Ligar
  as duas coisas (destilar → devolver para dentro do cérebro evoluível) é a próxima
  ponta solta, e não é óbvia — o L1 é ternário e o v2 é uma pirâmide de 202k features.
- Custo por tick do v2: 202 752 multiplicações, mais caro que o próprio cérebro. Serve
  como professor/baseline/instrumento, não como economia de computação.
- Só duas pistas no caderno. A lei da cobertura prevê que mais pistas melhoram;
  não medi.

## Refinamento da lei do pré-squash (correção vinda da sessão-forja, verificada aqui)

Eu tinha apostado que o `arctanh` consertaria também uma derrota deles em
CLASSIFICAÇÃO. A forja mostrou a matemática contra: para alvo **dicotômico puro** o
ridge é afim-invariante à codificação, então 0/1, ±1 e ±arctanh(0,999) dão as MESMAS
predições ordenadas. Verifiquei nos nossos dados em vez de aceitar:

```
alvo dicotômico (frear ou não):  corr(pred 0/1, pred ±1)      = 1.0000000000
                                 corr(pred 0/1, pred arctanh) = 1.0000000000
alvo graduado (freio real):      identidade 0,2613 → arctanh 0,2071  (+20,7%)
```

(A correlação sai exatamente 1 porque Z é centrado no treino: `Zᵀ1 = 0`, então o termo
constante da transformação afim resolve para peso ZERO e sobra só a escala.)

**Enunciado corrigido: o pré-squash paga quando o alvo saturado tem MAGNITUDES VARIADAS
comprimidas no platô — não quando é dicotomia.** No nosso caso 61,5% do freio vive em
|y|>0,9 com valores distintos ali dentro: é essa variação esmagada que o `arctanh`
devolve. Num alvo 0/1 não há variação a recuperar.

Nota de método: minha primeira tentativa de verificar deu ganho NEGATIVO (−7%) porque eu
tinha esquecido de padronizar o alvo — sem intercepto o modelo gasta pesos reproduzindo
a média (o freio tem média −0,852). Refeito com o protocolo correto, reproduzi o número
do agente de forma independente (+20,7% contra os +23% dele).

## Limite da lei da cobertura: só vale com demonstração BOA

Estendi o caderno de 2 para 4 pistas esperando mais cobertura. Deu o contrário:

| caderno | erro held-out | 9001 (casa) | 9007 (nunca vista) |
|---|---|---|---|
| 1 pista, n=1024 | 0,1794 | 114% | 84% |
| **2 pistas, n=1024** | **0,1388** | 108% | **102%** |
| 4 pistas, n=1024 | 0,1509 | 109% | 93% |
| 4 pistas, n=2048 | 0,1470 | 110% | 100% |

Dobrar a capacidade recupera de 93% para 100%, mas nunca alcança o caderno de 2 pistas.
A causa não é capacidade — é a QUALIDADE do que foi acrescentado:

| pista do dump | professor fora da pista |
|---|---|
| 9001 | 0,0% |
| 9003 | 0,0% |
| **8005** | **77,0%** |
| **8200** | **87,5%** |

As duas pistas novas são de PERFIL, e nelas o próprio campeão dirige fora da pista 77-87%
do tempo (ele treinou só em 9001/9003, ambas procedurais). Metade do caderno virou
demonstração de fracasso.

**Enunciado do limite: cobertura > aglomeração vale enquanto o que se acrescenta é
demonstração BOA. Acrescentar região onde o PROFESSOR falha não amplia cobertura —
envenena o caderno.** Regra prática que sai disso: filtrar o caderno por qualidade do
professor (`fora_pct ≈ 0`) antes de misturar; é um número que o `replay_obs` já dá.

Corolário para o projeto, e é útil: as pistas onde o campeão falha (perfil) são as que
valem EVOLUIR, não destilar. Destilação amplia o que o professor sabe; ela não inventa
competência que ele não tem.

## Destilação ITERADA — predição registrada antes de rodar

O piloto misto-2 dirige melhor que o campeão nas três pistas. Se destilação amplia o que
o professor sabe, então destilar do PILOTO (professor melhor) deveria render um piloto
ainda melhor — refino sem gastar uma única geração de GPU.

Desenho: piloto misto-2 dirige em 9001 e 9003 com `--observar` → novo caderno → mesmo
solve → piloto-g2. Avaliação nas mesmas três pistas, incluindo 9007 (nunca vista).

**Predição: NÃO melhora — espero empate ou queda leve (dentro de ±5%).** Razão: o
caderno novo é gravado na distribuição do PILOTO, não na do cérebro, e o que limita não
é a qualidade do professor e sim o que a base consegue representar (o erro held-out do
misto-2 já é 0,1388 com 202k features). Se melhorar acima de +5%, existe um laço de
auto-melhoria sem evolução e isso muda a prioridade do projeto inteiro.

## Destilação ITERADA — VEREDITO: não há almoço grátis, e a métrica de ajuste MENTE

Predição registrada (empate ou queda leve) confirmada, pelo lado da queda:

| pista | cérebro | piloto g1 (do cérebro) | piloto g2 (do piloto) | g2 vs g1 |
|---|---|---|---|---|
| 9001 | 6592 | 7139 | 6961 | −2,5% |
| 9003 | 5242 | 5575 | 5177 | −7,1% |
| 9007 (nunca vista) | 5932 | 6072 | 5536 | **−8,8%** |

**O achado não é a queda — é a DISSOCIAÇÃO.** O erro held-out do ajuste CAIU 60%
(0,1388 → 0,0549, com volante em 0,024 e freio em 0,067) enquanto dirigir piorou até
8,8%. Copiar o piloto é fácil justamente porque ele é uma função suave e determinística
de X; copiar bem não acrescenta nada.

Duas consequências práticas:

1. **Não existe laço de auto-melhoria por re-destilação.** Refino de verdade exige
   informação NOVA do mundo — que é o que a evolução fornece e a cópia não. A queda é
   pior na pista nunca vista (−8,8%), ou seja, o que se perde primeiro é generalização:
   a cópia herda os pontos cegos do pai e não vê nada além deles.
2. **Erro de ajuste deixa de ser proxy de desempenho assim que se destila um
   destilado.** Se eu tivesse aceitado o 0,0549 como progresso, teria publicado uma
   melhora de 60% que na pista é uma piora. Vale como canário: quando a métrica de
   ajuste melhora MUITO sem que nada no mundo tenha mudado, medir no mundo antes de
   comemorar.

## Veneno por malha — fecha a dicotomia (medição da sessão-forja)

Eles rodaram o espelho do nosso teste: 25% do caderno vindo de texto onde o modelo denso
vai mal. Resultado **inerte** em malha aberta (prosa 3,138→3,089; código +0,06,
explicável pelo orçamento menor de centros bons; diálogo empate).

**Enunciado conjunto:** a qualidade do professor é critério de seleção de dados
PROPORCIONAL ao fechamento da malha. Em malha aberta, filtrar é economia (centro ruim
desperdiça orçamento); em malha fechada, filtrar é SEGURANÇA (centro ruim é armadilha
que a trajetória pisa, porque a própria dinâmica carrega o controlador para lá).
Corolário: onde o professor falha, em malha fechada é território de EVOLUIR; em malha
aberta é só espaço desperdiçado.

## A PONTE: a política destilada volta para dentro do genoma (cirurgia de L2)

O piloto dirige melhor que o campeão mas NÃO é um genoma — não dá para semear evolução
com ele. A ponte é uma cirurgia por forma fechada: congelar L1 e recalcular só as três
linhas de L2 que produzem acelerador/volante/freio, tendo o PILOTO como alvo.

**Para isso foi preciso reconstruir a cadeia de decisão fora da GPU**, e a reconstrução
foi ganhando precisão à medida que cada mecanismo entrou:

| modelo | erro vs intent gravado | corr (acel/volante/freio) |
|---|---|---|
| L1 → L2 → tanh | 0,1841 | 0,975 / 0,871 / 0,860 |
| + memória e limiar de L1 | 0,0833 | 0,984 / 0,944 / 0,909 |
| + PERÍODO por neurônio | **0,0531** | **0,993 / 0,981 / 0,958** |

O período era a peça grande: **76 dos 96 neurônios de L1 têm período > 1** (só 20
recalculam todo tick). Sem ele, qualquer análise offline do cérebro erra ~2× mais.
Ferramenta nova: `genoma_dump` (dump pelo parser CANÔNICO — nunca reimplementar o
formato, ver o bug do parser duplicado) com modo `--patch-l2`.

**Resultado, e a âncora é o que decide:**

| condição | 9001 (treino) | 9007 (nunca vista) |
|---|---|---|
| cérebro original | 6592 | 5932 |
| piloto por fórmula (não herdável) | 7139 | 6072 |
| genoma operado, encolhimento para ZERO | 7134 (108%) | **600 (10%)** |
| **genoma operado, encolhimento para W₀ (λ=1)** | **7083 (107,4%)** | **5846 (98,5%)** |
| genoma operado, âncora forte (λ=100) | 7217 (109%) | 3738 (63%) |

Encolher para ZERO reproduz a política em casa e DESTRÓI a generalização — a mesma
assinatura do clone linear (600 m, o mesmo ponto de trava). Encolher para os pesos
ORIGINAIS mantém a estrutura que o cérebro tinha e absorve só o ganho: **+7,4% em casa
com 98,5% preservado na pista nova.**

> **A destilação fecha o ciclo: campeão → fórmula melhor → de volta ao genoma, herdável
> pela evolução.** O que a evolução herda não é a fórmula (que é uma pirâmide de 202k
> features), é o efeito dela nas três linhas de L2 que viram ação.

Ressalvas honestas: uma corrida por condição; λ=100 saiu pior que λ=1 em 9007 (3738 vs
5846), não-monotônico — pode ser ruído de corrida única em malha caótica, não medi
réplicas. E a reconstrução ainda erra 0,053: a cirurgia é feita sobre um modelo BOM, não
exato, o que provavelmente explica parte da perda fora da distribuição.
