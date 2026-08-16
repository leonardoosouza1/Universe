# Destilabilidade — o que o clone que venceu o campeão significa

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
