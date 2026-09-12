# Equações, DNA e desenvolvimento do ser — análise e proposta

2026-09-11. Revisão estática pelo Codex com dois agentes de leitura; sem implementação, treino ou benchmark. O Claude estava alterando o projeto: HEAD observado passou de `f515192` para `5c3a8bf` durante a leitura. Conferir o checkout antes de transformar achados em correções.

Escopo: cálculos do cérebro, memória, plasticidade, seleção e representação hereditária, no caminho `gpu_evolve` e nas variantes relevantes. Não é auditoria de cada fórmula da física, pneus, colisões ou economia. Complementa [o plano de kernels](../plano-kernels.md); propostas aqui não modificam o protocolo experimental já aprovado.

## 1. O que a ideia do ser que se desenvolve significa

Separar quatro processos:

| Processo | O que muda | Escala |
|---|---|---|
| Dinâmica recorrente | Ativação, fadiga, memória de trabalho | Ticks/segundos |
| Aprendizagem individual | Memória episódica, conexões plásticas, estratégias adquiridas | Durante uma vida |
| Desenvolvimento | Expressão de módulos, maturação e eventualmente estrutura/corpo | Durante uma vida |
| Evolução | Distribuição dos genes herdáveis na população | Gerações |

O ambiente não determina uma mutação desejada. Quando há variação herdável e diferenças de reprodução, características vantajosas podem tornar-se mais frequentes. Ser mais alto não é garantido: depende de vantagens, custos, restrições e variação disponível. Nutrição também pode alterar crescimento individual sem alterar o DNA. Essa distinção é explicada pelo [NIH/MedlinePlus](https://www.medlineplus.gov/genetics/understanding/mutationsanddisorders/evolution/).

Proposta central: evoluir um programa de nascimento e adaptação. Um indivíduo nasce com competências mínimas e regras plásticas; a experiência modifica sua vida. A seleção favorece genes que produzem bons aprendizes. Isso não exige que a memória adquirida dos pais seja herdada.

## 2. O que existe de verdade

`src/gpu/evolution.rs::Genome` contém L1, L2, KAN, pesos do lento, taxas CfC, sensores, metadados neuronais, gates e o hipocampo recém-adicionado. Portanto, o Universe já faz neuroevolução de pesos e parâmetros funcionais, com topologia de dimensões fixas.

`evolve()` seleciona um pai por torneio, clona e muta; elites passam sem mutação. Não há crossover no caminho inspecionado. Reprodução assexuada é uma escolha válida; crossover não é requisito para evolução.

`src/gpu/genome_io.rs` define UGEN v2, posicional, com blocos opcionais. Estados CfC, fadiga e memória setorial são estados de vida, não genes. `src/evolution/dna_genesis.rs` é outra representação usada por outros caminhos/ferramentas; seus genes morfológicos são descritos como armazenados, ainda sem ligação física naquele mecanismo. Não unificar os formatos por semelhança de nomes.

Evidências verificadas por `sed`/`rg`: `evolution.rs` struct `Genome` e função `evolve`; `genome_io.rs` cabeçalho/formato; `gpu_evolve/main.rs` chamada `evolve` e `kan_knots_global`; chamadas `dispatch_hebb` em todo `src`; `hippocampus.rs` leitura/escrita. Afirmações abaixo são inferidas do código, não resultados de execução.

## 3. Inventário de equações e insights

### 3.1 CfC rápido: memória temporal com ganho controlado

`src/gpu/brain.wgsl`, estágio CfC:

```text
x = sign(x_raw) max(0, |x_raw| - beta * fatigue)
d = exp(-lambda * dt)
s' = d*s + G*(1-d)*x + noise
s' = clamp(s', -2, 2)
```

Ponto positivo: com entrada constante, sem fadiga/ruído/clamp, o equilíbrio é `s*=G*x`, independente de lambda. O código já corrige o acoplamento entre retenção e amplitude. O campo chamado `tau` contém aqui uma taxa em s^-1: a constante de tempo física é `1/lambda`. Registrar essa unidade evita inverter o sentido de uma mutação.

Insight: ruído, shrinkage e clamp alteram a dinâmica real. Testar entrada constante, pulso e degrau por canal; medir retenção e saturação, não concluir “memória melhor” apenas porque lambda mudou. Isso é estado recorrente, não alteração dos pesos sinápticos.

### 3.2 Fadiga

```text
f' = f + k * (|s'| - f) * dt
```

O passo é uma média convexa quando `0 <= k*dt <= 1`. Se taxas/dt evoluírem, essa condição deve fazer parte do contrato. Sinais persistentes importantes, como combustível baixo, podem ser atenuados junto com estímulos redundantes. Comparar canais transitórios e homeostáticos separadamente. Uma mudança para integração exponencial seria mudança matemática, não simples otimização.

### 3.3 Memória por neurônio: retenção e ganho misturados

`brain.wgsl::neuron_mem_gain` e estágios L1/L2:

```text
h' = alpha*h + (1-alpha)*z
output = h' * sqrt((1+alpha)/(1-alpha))
```

A compensação preserva variância para entrada temporalmente não correlacionada sob as hipóteses da derivação. Sensores de movimento são frequentemente correlacionados. Para entrada constante, `h*=z`, mas o multiplicador continua: em alpha=0.95, ganho ≈6.24. Logo o gene seleciona retenção E amplificação; isso pode aumentar saturação e confundir a interpretação.

Experimento recomendado: manter controle atual e comparar retenção sem compensação ou ganho independente, com entradas branca, constante e correlacionada. Não declarar a compensação um bug universal; sua justificativa depende do sinal real.

### 3.4 Período, limiar, ativação e quantização

```text
fires = tick % period == 0
v = 0 se |v| < threshold
q = trunc(clamp(activation(v)*gate*127, -127, 127))
```

Períodos alteram o instante de atualização. Para EMA atualizada a cada p ticks, a escala efetiva é `tau_eff = -p*dt/log(alpha)` para `0<alpha<1`. Mutar alpha e período juntos pode alterar a mesma propriedade por dois caminhos.

Limiar e quantização criam regiões em que pequenas mutações não mudam comportamento; ReLU/linear seguidas de clamp podem acabar iguais em muitos estados. Medir fração de zeros, saturação, diversidade dos comandos e sensibilidade por bloco. O código calcula a L1 antes de aplicar `fires`: período maior não demonstra economia computacional.

### 3.5 L1/L2 e normalização opcional

```text
z_j = sum_i W[j,i]*x_i
z_normalized = z / max(std(z), epsilon)        quando a flag está ligada
```

Não centralizar preserva o viés, mas se todos os z forem quase iguais e não nulos, std tende a zero e a amplitude é amplificada até o limite imposto por epsilon/clamps. O cálculo `E[z²]-E[z]²` também sofre cancelamento. Criar canários de entradas quase constantes antes de escolher variância mais estável ou RMS como alternativa; RMS mudaria a semântica.

Auditar influência até o atuador: tamanho de camada não prova capacidade útil. `context.rs` consome as primeiras intenções para comandos e alguns canais para marcha, e o lento tem outros consumidores. Identificar conexões realmente ativas para cada flag antes de adicionar neurônios.

### 3.6 Cérebro lento

`slow_brain.wgsl` executa CfC64, Dense64→16→3 a cada 8 ticks:

```text
s' = exp(-0.5*dt)*s + dt*x + noise
out = tanh(W2*ReLU(W1*q+b1)+b2)
```

Não assumir que usa a mesma discretização normalizada do rápido. Validar dt do host, cadência e persistência das saídas. O risco de corrida nos contadores já consta da especificação de kernels. Novamente: pesos densos herdados + estado variável não equivalem a aprendizagem sináptica durante a vida.

### 3.7 Gates regionais e tálamo

```text
gate_region = 1 + 0.5*tanh(W*slow_output+b)
gate_family = 2*sigmoid(a+b*need)              proposta C2
```

São mecanismos diferentes: um modula regiões neuronais, outro famílias de sensores. Ambos precisam de sinais que ajudem a escolher ações. Gate neutro é excelente controle. Medir saturação e contribuição causal, não só magnitude do gate.

O `need=max(deficits)` do Neuromod perde qual necessidade causou o máximo: dois corpos com déficits diferentes podem ter o mesmo escalar. Isso só é suficiente se os canais originais preservarem a identidade necessária. Primeiro testar dois estados de mesmo need com ações ótimas diferentes; depois considerar vetor de necessidades por família. Não ampliar genes sem esse diagnóstico.

### 3.8 Hipocampo: boa memória, mas atenção ao nascimento neutro

`hippocampus.rs` e PRD C1:

```text
M[sector]' = M[sector] + sigmoid(g_tau)*(event-M[sector])
bias = tanh(g_gain)*tanh(W*[M[current],M[next],1])
```

Escrita por evento é coerente com memória de uma passagem e tem custo limitado. Taxa de escrita mede eventos, não segundos. Preservar setor concluído, reset e significado de cada coluna.

Com W e g_gain ambos zero, mutar apenas um deles não liga o órgão; perto de zero, o efeito é de segunda ordem, aproximadamente `g_gain*(W*x)`. Isso não bloqueia evolução por mutações conjuntas, mas cria uma região inicialmente sem sinal seletivo para mutações isoladas. Um canário deve medir probabilidade de mutação produzir efeito, além do teste com todos os genes artificialmente ativados.

Alternativa experimental: pesos internos pequenos e determinísticos, apenas ganho de saída zero. O Neuromod já usa entradas aleatórias com saídas neutras em sua camada livre. Mudaria inicialização/dof do C1, portanto comparar em braço separado e preservar o controle aprovado.

Apagar M mantém o viés da coluna constante; desligar o bloco remove tudo. Incluir também memória embaralhada entre setores para separar conteúdo contextual de offset aprendido. Resultados nas voltas completas têm viés de sobrevivência: reportar completagem e mortes junto com desempenho da segunda volta.

### 3.9 Núcleo Neuromod

`campo.py::Cerebro` e `completo.py` combinam:

```text
h = ReLU(U*features+b)
z' = z + (ReLU(U2*body+b2)-z)/tau
W_effective = W0 + z*M + memory*H + context_delta + gate_delta + reflex_delta
```

Ponto positivo: memória modifica o significado dos sentidos por produtos bilineares; não é apenas mais um escalar na entrada. Porém isso continua sendo modulação de pesos efetivos, não escrita de W0. Escalas de features, pesos e ganhos precisam ser comparadas na ação final; contagem de neurônios e nomes anatômicos não demonstram inteligência.

`completo.py` usa EMA por evento para renda/gasto e por tick para espera: uma mesma parametrização tau não representa a mesma duração física nessas três colunas. Testar se essa assimetria é desejada, antes de transportar o mecanismo.

### 3.10 Folha cortical: plasticidade existente em outra frente

`cortex_sheet.wgsl` soma aferências gaussianas, pesos plásticos e onda espontânea; filtra com `exp(-dt/tau)`, aplica tanh, inibição local e fadiga. `cortex_sheet_hebb.wgsl` atualiza:

```text
delta_w = eta * (a_i*a_k - alpha*w*abs(a_k)*sum_j abs(a_j))
```

Isso é mudança sináptica durante a vida. Entretanto, `rg 'dispatch_hebb\(' src` encontra chamadas em `cortex_evolve`, `cortex_test` e `cortex_map`, não no `gpu_evolve` principal. A folha usa 223 entradas; o cérebro principal usa 192. Não conectar os buffers sem mapeamento.

Achado: `cortex_evolve.rs` usa média entre seres de `act_sum_prev`, e a estimativa de cada soma é `abs(mean)*1024*2`. Em geral `abs(mean(a))*N` não é `sum(abs(a))`: ativações positivas e negativas cancelam. Além disso, um ser pode mudar a normalização de outro. Corrigir/explicitar essa semântica antes de reaproveitar a regra. Teste: mesmo ser isolado e em lote com indivíduos de atividades diferentes.

### 3.11 Fitness, custo e seleção

O modo padrão de `gpu_evolve/main.rs` combina checkpoints ordenados com progresso; outros modos usam distância/progresso/centralização. Isso seleciona desempenho de corrida. Sobrevivência influencia indiretamente quando afeta esse desempenho; não equivale automaticamente a um ecossistema com reprodução autônoma.

Para selecionar aprendizagem, o mundo deve oferecer informação reaproveitável e exigir mudança de estratégia. Para selecionar desenvolvimento, o benefício futuro precisa compensar custo/risco de crescer. Memória não ganha prêmio se a tarefa termina antes de poder usá-la.

`dof` por limiar de peso e magnitude de ativação são métricas úteis, mas não são medição direta de energia. Reparametrizações podem produzir o mesmo comportamento com magnitudes diferentes. Separar custo biológico simulado de tempo/memória reais da GPU; não cobrar redução de FLOPs que o kernel não executa.

### 3.12 Gene armazenado precisa ter efeito

`evolution.rs` documenta que KAN knots deixaram de mutar porque o caminho principal usa `default_kan_knots()` global em `main.rs`, não os knots por genoma. Isso já é uma restrição conhecida, não um novo bug de mutação ativa. Antes de chamar KAN de característica evoluída, conectar por indivíduo ou marcar como parâmetro fixo.

Para todo gene novo, exigir a cadeia: arquivo → carregamento → fenótipo → buffer → cálculo → comando/mundo → seleção → filho → novo carregamento. Perturbação que só muda bytes não demonstra expressão genética.

## 4. Organização proposta de DNA, pesos e neurônios

| Objeto | Conteúdo | Onde fica | Herança/reset |
|---|---|---|---|
| Genome | Pesos iniciais ou gerador, parâmetros dos neurônios, plasticidade, desenvolvimento, sensores/corpo | Arquivo versionado + host; parâmetros necessários na GPU | Herdado/mutado |
| Phenotype | Pesos base expandidos, regiões, conexões, máscaras iniciais | Buffers GPU produzidos no nascimento | Reconstruído do genoma |
| LifeState | Ativações, fadiga, memória, traces, deltas plásticos, idade e expressão de módulos | Buffers GPU por indivíduo | Zerado em nova vida, preservado ao retomar a mesma vida |
| Lineage | IDs, pais, geração, seed, ambiente e métricas de seleção | Host/artefatos | Registro histórico |

Nomes acima são uma proposta conceitual, não novos tipos implementados. Preservar o leitor UGEN atual; qualquer formato novo deve ter versão, blocos identificados, shapes e migração testada. Diferenciar checkpoint reprodutivo (genes) de checkpoint de continuação (genes + todo estado + RNG/tick/mundo).

Um neurônio é um índice dentro de buffers, não precisa ser um objeto Rust individual. Seus parâmetros herdados (tau/taxa, ativação, limiar, região) ficam separados de estado (atividade, fadiga, elegibilidade). Cada réplica do mesmo genoma compartilha apenas dados imutáveis; memória e plasticidade são privadas.

Tamanhos de referência, só pesos f32: L1 `192*96*4 = 73,728 bytes` (72 KiB), L2 `96*48*4 = 18,432 bytes` (18 KiB); não incluem restante do cérebro, réplicas ou mundo. Duplicar deltas/traces para todas as conexões aumenta memória por vida. Começar plasticidade num bloco 16×16 precisaria 1 KiB por matriz f32 por vida; W0, delta e trace são matrizes distintas. Perfil e orçamento decidem layout final.

## 5. Como fazer o ser nascer capaz de aprender

Proposta experimental mínima: manter o cérebro atual e adicionar um pequeno bloco plástico residente. Genes especificam W0, taxas e regras; a vida modifica um delta local. Exemplo de família de regras a validar, não fórmula já presente:

```text
e_ij(t+1) = rho*e_ij(t) + (1-rho)*pre_j(t)*post_i(t)
D_ij(t+1) = clip((1-decay)*D_ij(t) + eta*m(t)*e_ij(t+1), -Dmax, Dmax)
W_eff(t) = W0 + strength * D(t)
```

`rho`, `decay`, `eta`, `strength` e limites são parametrizados por genes, inicialmente por bloco para reduzir dimensão da busca. O modulador m deve ter fonte explícita e observável; não pode ler sucesso futuro nem fitness oculta. Pode começar por erro/resultado corporal local documentado. Hebb com neuromodulação não garante aprender o comportamento desejado; é hipótese a testar. Taxas são por atualização e precisam de contrato de dt/cadência.

O genoma permanece fixo durante a vida; D e e começam zerados no nascimento. A seleção compara o desempenho dos indivíduos depois da experiência. Assim, filhos herdam predisposição para aprender, não lembranças particulares. Herança de pesos adquiridos (lamarckiana em termos algorítmicos) pode ser outra experiência, nunca efeito acidental do checkpoint.

Base científica: [Differentiable Plasticity, Miconi et al., 2018](https://proceedings.mlr.press/v80/miconi18a.html) demonstra otimização de redes com componentes plásticos. O trabalho usa backpropagation; a proposta aqui é selecionar parâmetros por evolução, portanto não é uma reprodução do método nem garantia dos mesmos resultados.

## 6. Desenvolvimento sem nascer com toda capacidade expressa

Primeira versão viável: reservar capacidade máxima compatível com os kernels e começar com poucos módulos expressos. Genes definem parâmetros de maturação e resposta à experiência; LifeState guarda idade, atividade acumulada e quais módulos estão ativos. Mudanças acontecem em fronteiras explícitas de poucos em poucos ticks, com custo, limites e histerese para não alternar incessantemente.

Ativar neurônios reservados é desenvolvimento de capacidade dentro de um substrato limitado, não criação irrestrita de topologia. Se kernels ainda calculam módulos mascarados, não há economia real. Crescimento computacional precisa também de despacho/armazenamento que evite trabalho desativado, demonstrado por medição.

Segunda versão: genes codificam um gerador `W0_ij = f_G(position_i, position_j, region_i, region_j)` e expressão de conexões. Isso permite regularidade e repetição sem um gene independente por sinapse, mas impõe um viés e pode dificultar detalhes arbitrários. [ES-HyperNEAT, Risi e Stanley](https://direct.mit.edu/artl/article/18/4/331/2720/An-Enhanced-Hypercube-Based-Encoding-for-Evolving) é referência direta para gerar conectividade e colocação de neurônios.

Outra linha é crescimento estrutural por mutações entre gerações, como [NEAT](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf). NEAT não significa desenvolvimento durante a vida. Topologias variáveis também exigem representação, compilação/agrupamento e política de diversidade; não adicionar apenas uma mutação “criar neurônio” ao vetor atual.

[Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/) mostra desenvolvimento de padrões e regeneração por regras locais aprendidas. Serve de inspiração para morfogênese; os experimentos são padrões-alvo, não evidência de inteligência geral nem motivo para reescrever o Universe agora.

No mundo de corrida, um primeiro traço corporal poderia ser alcance visual, capacidade energética ou resistência térmica, desde que realmente ligado ao sensor/física e com custos coerentes. Se aumentar um atributo só trouxer benefício, a seleção pode simplesmente empurrá-lo ao máximo. A analogia de altura exige um mundo onde tamanho afete ações e tenha consequências; renomear um gene não cria isso.

## 7. Experimento que separa promessa de resultado

Executar em etapas, sem liberar treino longo por este documento:

1. Validar baseline/hipocampo atual e corrigir dependências matemáticas identificadas.
2. Comparar cérebro atual, bloco adicional fixo e mesmo bloco plástico, com orçamento de avaliação controlado. Incluir plasticidade congelada e memória apagada na avaliação.
3. Usar ambientes com mudança dentro da vida e pistas/contextos não vistos na seleção. Mesma genética em experiências diferentes deve produzir adaptação mensurável e útil. Medir desempenho absoluto, velocidade de adaptação, sobrevivência e esquecimento; ganho relativo sozinho pode premiar nascer deliberadamente ruim.
4. Só depois comparar capacidade fixa com módulos que amadurecem sob as mesmas tarefas, limite máximo e custos. Medir ativações/neurônios ativos e gasto real GPU separadamente.
5. Manter sementes de evolução independentes, linhagens registradas, resets auditáveis e avaliação held-out fora da seleção. Reservar nichos/diversidade é possibilidade futura; não substituir o controle atual sem experimento.

Critério final: o genoma produz indivíduos que se adaptam melhor a experiências novas, e essa capacidade passa aos descendentes mesmo quando o estado aprendido é zerado. Crescer neurônios, alterar pesos ou aumentar tamanho por si só não demonstra esse objetivo.
