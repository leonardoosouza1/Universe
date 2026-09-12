# Especificação de execução e validação dos kernels

Data: 2026-09-11. Executor: Claude. Revisor posterior: Codex. Entrada: [plano-kernels](../plano-kernels.md).

Este documento define uma proposta de implementação, não resultados de benchmark. As equações de hipocampo/tálamo abaixo vêm do [PRD C0–C5](PRD-CEREBRO-NEUROMOD.md); detalhes ambíguos devem ser fixados em contrato antes do port. Não copiar fórmulas do Neuromod sobre sensores de corrida sem a adaptação definida nesse PRD.

## 1. Base existente e verificação inicial

Universe usa Rust edition 2024, `wgpu = 24.0`, Naga 24 e WGSL. Consultar a versão resolvida no Cargo.lock antes de escolher APIs. Backend esperado nesta máquina: Vulkan/RADV; confirmar adapter, backend, driver, limites e features no manifesto. Hardware descrito no workspace: RX 6750 XT, 12 GB. Não presumir suporte a recursos opcionais com base apenas no nome da GPU.

| Responsabilidade | Ponto inicial de inspeção |
|---|---|
| Execução real da geração e decisão no host | `src/bin/gpu_evolve/main.rs`, `context.rs` |
| Cérebro rápido e buffers | `src/gpu/brain/mod.rs`, `src/gpu/brain.wgsl` |
| Cérebro lento e gather | `src/gpu/slow_brain.rs`, `slow_brain.wgsl`, `slow_gather.wgsl` |
| Percepção e retina | `src/gpu/perception.rs`, `retina_digest.rs`, shaders correspondentes |
| Física e variantes | `src/gpu/physics.rs`, shaders `physics_*` |
| Cortex sheet opcional | `src/gpu/cortex_sheet.rs`, shaders correspondentes |
| Medição GPU | `src/gpu/timing.rs` (`GpuTimer`) |
| Pesos, evolução e persistência | `src/gpu/evolution.rs`, `genome_io.rs` |
| Referência Neuromod e motor HIP | `../../neuromod-lab/src/campo.py`, `vida_hip.py`, `hip/vida.cu` |

K0 deve identificar quais variantes realmente executam no braço alvo. Arquivo existente não comprova uso. Documentos antigos citam `brain.rs`, mas o host está em `brain/mod.rs`; também não assumir que a alegação de “tick inteiro GPU” elimina decisões em `context.rs`.

## 2. Arquitetura proposta

Sequência lógica inicial, a confirmar contra a ordem efetiva do controle:

```text
estado(t) → sensores(t) → gates(t) → cérebro rápido/lento
          → leitura da memória(t) → decisão/alisamento/clamps
          → física → eventos e métricas → escrita da memória(t+1)
```

Um evento produzido pela física não pode modificar retroativamente a decisão do mesmo tick. Caso o controle use outra ordem, explicitar e preservar essa ordem no primeiro port.

CPU envia configurações, mapas, pesos e estados iniciais; GPU mantém estados e executa etapas; CPU recebe agregados/checkpoints ou amostras para visualização. Visualizador não pode ser necessário para avançar o treino. Telemetria deve ter frequência configurável e modo desligado para benchmark.

As unidades abaixo são responsabilidades, não obrigação de criar um arquivo/dispatch para cada linha:

| Unidade | Entradas → saídas | Estratégia inicial |
|---|---|---|
| Percepção existente | Mundo e histórico visual → canais | Reusar; preservar clear/atomicOr e cópia do frame anterior |
| Interocepção/gates, futuro C2 | Corpo e genes → 6 gates por ser | Comparar pré-cálculo pequeno com fusão na preparação de entrada |
| Inferência existente | Canais, pesos, estado → intenção e estado novo | Preservar divisão rápido/lento e cadência |
| Leitura hipocampal C1 | Setor, memória e 29 genes → 3 vieses | Começar simples, distribuição por ser; medir antes de paralelizar matriz pequena |
| Decisão motora | Intenção, lento, vieses e estado do controlador → comandos | Portar também reflexos/flags ativos, alisamento e clamps |
| Física existente | Comandos e mundo → próximo mundo | Preservar passes separados; não fundir por princípio |
| Eventos/memória | Transições de setor e acumuladores → memória atualizada | Um escritor por ser/setor; registrar setor encerrado explicitamente |
| Métricas | Estado/eventos → acumuladores e resumo | Reduções hierárquicas quando necessário; evitar contadores globais disputados por tick |

Não introduzir compartilhamento HIP↔Vulkan neste plano. HIP serve à frente Neuromod; Universe continua com buffers do mesmo device wgpu. Seleção/mutação permanecem inicialmente por geração no host; migrar somente se medição demonstrar custo relevante.

## 3. Contratos matemáticos

Para cada bloco, entregar tabela com símbolos, unidades, shapes, domínio, clamps, estado inicial, frequência, ordem de leitura/escrita, comportamento quando morto e precisão. Marcar separadamente equação científica e aproximação de implementação.

### Hipocampo por setor — C1

Estado proposto no PRD: `M[being,24,4]`; genes `W[3,9]`, `g_tau`, `g_gain`.

```text
alpha = sigmoid(g_tau)
M[s] ← M[s] + alpha * (event_features - M[s])     apenas ao concluir o setor s
x = concat(M[current_sector], M[next_sector], 1)
bias = tanh(g_gain) * tanh(W @ x)
commands ← commands + 0.5 * bias                 antes do alisamento e clamps
```

Fixar antes de implementar: normalização de dano; população temporal da média de velocidade e da fração de grama; setor terminado versus setor recém-entrado; tratamento de teleporte, vários cruzamentos, morte e reset; `dt_improvement=0` na primeira passagem ou duração anterior nula. Não ler dados da passagem futura. A regra `alpha=sigmoid(g_tau)` do PRD não é a mesma parametrização de tau do Neuromod; manter a escolha explícita.

Genes neutros devem preservar os comandos do controle bit a bit. Preferir caminho de bypass neutro quando necessário para preservar arredondamentos. Testar `memory erased` (memória zero, constante da entrada preservada) separado de `block disabled` (viés zero). Não tratar as duas ablações como equivalentes.

### Tálamo — C2 condicionado ao protocolo científico

```text
need = max(1 - fuel_norm, 1 - life_norm, tire_temp_norm)
gate[f] = 2 * sigmoid(a[f] + b[f] * need)
input[channel] *= gate[family[channel]]
```

Definir normalizações/unidades e domínio dos sensores; não inserir clamps novos silenciosamente. Cobrir os 192 canais exatamente uma vez. Genes zero dão gate 1. Testar saturação, extremos finitos e entrada inválida. Não adicionar passageiro como variável no mundo de corrida.

### Dinâmica recorrente existente

Em `slow_brain.wgsl`, a atualização inclui:

```text
x = sign(x_raw) * max(0, abs(x_raw) - beta * fatigue)
s_new = clamp(s * exp(-lambda * dt) + gain * dt * x + noise, -2, 2)
fatigue_new = fatigue + alpha * (abs(s_new) - fatigue) * dt
q = trunc(clamp(tanh(s_new) * 127, -127, 127)) / 127
```

Preservar parâmetros, truncamento e cadência de 1 execução a cada 8 ticks. Conferir o `dt` efetivamente passado pelo host e pela referência: não multiplicar por 8 como “otimização”. Alterar integração, retirar quantização, trocar exp/tanh ou adotar precisão menor exige outro experimento; não é simples port equivalente.

## 4. Correções bloqueantes e regressões

1. **Contadores do lento:** `run_cfc` lê contadores em todas as invocações e escreve na invocação zero sem sincronização. Proposta: invocação zero lê/incrementa, publica valores em memória de workgroup; barreira uniforme antes de todos decidirem executar. Não colocar barreira depois de retorno divergente. Testar fronteiras 7/8/9 e 15/16/17, estado e ruído por canal, várias entidades e repetições. Consultar o modelo de memória WGSL [R1].
2. **Amplitude efetiva HIP:** `Campo` zera oscilação quando nenhum mapa é distinto, mas wrapper envia `cfg.amp_preco`. Passar valor efetivo, preservando inclusive a semântica atual de lotes mistos. Testar mapas comuns/distintos/mistos, amplitude zero e mudança de dia.
3. **Stream HIP:** usar dispositivo do tensor e stream atual do PyTorch, validar device/dtype/contiguidade/shapes antes dos ponteiros e verificar erro de lançamento. Testar preparação e consumo em stream não padrão. Não remover a sincronização global atual até a ordem e a vida útil dos buffers estarem corretas [R5].

Achados por inspeção; regressões executáveis ainda devem confirmar os efeitos. Se Claude já corrigiu algum, anexar evidência em vez de refazer.

## 5. Memória, sincronização e aleatoriedade

- Documentar cada binding: produtor, consumidor, acesso, tamanho, stride, offsets, alinhamento, frequência de upload e reset. Verificar limites reais do adapter; “restam 2 bindings” no PRD não substitui contagem por pipeline.
- Não assumir que todo struct WGSL alinha a 16 bytes. Derivar layout dos tipos/membros e regras de storage/uniform; conferir com Rust e fixtures de pelo menos dois seres [R1].
- Campos por ser podem usar SoA/AoSoA; pesos compartilhados por genoma devem ser comparados com duplicação por réplica. Escolher por padrão de acesso medido, não por preferência abstrata.
- Se atualização lê vizinhos ou estado global anterior, usar buffers anterior/próximo ou passes separados. Barreiras de workgroup não sincronizam todos os workgroups.
- Em vários ticks enfileirados, não sobrescrever um único uniform de tempo supondo que cada dispatch capturou o valor antigo. Usar slots/offsets ou estado temporal no device com dependências explícitas.
- RNG: documentar chave `(seed, individual_id, episode, absolute_tick, channel/event)`. Compactação ou mudança de batch não pode trocar a identidade aleatória. Portar chave de 64 bits do HIP para WGSL exige representação/algoritmo deliberado; não truncar silenciosamente.
- Debug pode injetar os mesmos sorteios nos dois motores. Igualar apenas sementes não iguala algoritmos RNG diferentes.
- Fazer orçamento: mapas + pesos + réplicas + estados + temporários + staging + telemetria. Não exceder limite local de 80%; particionar população preservando identidade e resultados.

## 6. Medição e otimizações candidatas

Baseline por braço/configuração com `B` pequeno, típico e grande viável; um tick, episódio curto e horizonte real. Manter mesmo número de seres, mapas, ticks, flags, pesos e protocolo RNG. Separar compilação/aquecimento da execução estável; reportar custo frio separadamente.

Registrar tempo GPU por estágio, encoding/submit no host, espera/readback, bytes H2D/D2H, pico de memória e segundos por geração. Reportar ticks programados e ticks vivos úteis; morte precoce não pode parecer aceleração. Usar ao menos cinco repetições curtas, mediana e dispersão; aumentar somente se ruído impedir conclusão.

Universe: reaproveitar `GpuTimer`/timestamp queries com features verificadas; medir geração completa também. Neuromod: eventos GPU e profiler HIP quando suportado. Não usar rocprofv3 como medidor presumido do caminho Vulkan; a infraestrutura local já registra essa limitação em `timing.rs`.

Ordem candidata após perfil:

1. Remover transferências e esperas necessárias a cada tick, fechando dependências no device.
2. Reusar buffers/pipelines; agrupar submissões em blocos curtos sem mudar `dt` nem ordem. Validar interrupção e reset entre blocos.
3. Comparar layouts, blocos/workgroups e redução cooperativa para as dimensões reais. Não fixar tamanho de wave por suposição.
4. HIP: comparar vida inteira por thread com blocos de ticks. Medir pressão de registradores/scratch quando ferramentas suportarem; arrays locais indexados são hipótese de spill, não diagnóstico medido [R3].
5. Avaliar distâncias quadradas, pré-cálculo de invariantes e remoção de transcendentes repetidas. Distância quadrada pode mudar desempate por arredondamento; preservar a regra e testar fronteiras.
6. Só então experimentar precisão reduzida/fast math, em braço separado, com ganho científico validado.

Preservar divisão da física/lento enquanto não houver evidência que justifique mudança: `src/gpu/CLAUDE.md` documenta crashes anteriores do compilador RADV/ACO. A meta é ganho end-to-end superior ao ruído e sem regressão nos cenários alvo; não há fator de aceleração prometido.

## 7. Validação e entrega para revisão

Portões, nesta ordem:

1. **Estático:** WGSL parse/type-check; layout host/shader; capacidades; contratos de configuração. Naga não substitui testes de concorrência/paridade [R2].
2. **Um passo:** fixtures determinísticas, mesmos estados e sorteios, comparação por campo com `abs_error <= atol + rtol*abs(reference)`. Fixar tolerância por variável antes de comparar; flags, índices e contadores exatos.
3. **Trajetória curta:** primeira divergência, margem da decisão, eventos, morte, reset e memória; testar batch 1 e tamanhos não múltiplos do workgroup. Sem acesso entre seres ou estado herdado indevidamente.
4. **Neutralidade/ablações:** flags desligadas e genes neutros reproduzem controle; genes não neutros realmente influenciam saídas. Persistência com valores que não sejam defaults.
5. **Equivalência estatística quando RNG difere:** margens pré-definidas por métrica e intervalos de confiança; comparar campeões fixos nos dois motores. `p > 0.05` isolado ou médias próximas não provam equivalência. Respeitar sementes de evolução como unidades independentes, sem contar réplicas do mesmo campeão como treinos independentes.
6. **Desempenho:** repetir protocolo da seção 6 após os portões anteriores. Mudança de arquitetura científica requer ablações do PRD, além de benchmark.

Comandos existentes para o Claude conferir e executar quando aplicáveis:

```bash
# Em Universe, CPU; confirmar disponibilidade no checkout de implementação.
CARGO_BUILD_JOBS=4 nice -n 10 cargo test --release --test wgsl_validation
CARGO_BUILD_JOBS=4 nice -n 10 cargo test --release --bin gpu_evolve
CARGO_BUILD_JOBS=4 nice -n 10 cargo build --release --bin gpu_validate
```

Execução de `gpu_validate` e testes que criam device: somente pela fila `nmd` do PRD C4; não executar diretamente nem assumir que todo teste de biblioteca é CPU. Wrapper/comandos de fila devem ser verificados no checkout, não inventados. Artefatos pesados em `/media/llm-data/universe-runs/universe-cerebro/`. Não houve execução desses testes nesta entrega documental.

Pacote final: commits/diff, hashes de fontes/shaders/genomas, config efetiva, sementes, hardware/driver/toolchain, comandos exatos e exit codes, logs e medições brutas, tabela antes/depois, discrepâncias conhecidas, caminho de rollback. Separar correções de port equivalente e experimentos aproximados. O revisor deve reproduzir os checks pertinentes, não aceitar apenas o relato do executor.

## 8. Linguagem e ferramentas — referências oficiais

Consultadas em 2026-09-11; documentação latest pode descrever recursos ausentes da instalação. Usar Cargo.lock e versões locais como limite operacional.

- **[R1] [WGSL — especificação W3C](https://www.w3.org/TR/WGSL/):** tipos, layout host-shareable, uniformidade, barreiras, modelo de memória e precisão das funções. Ler essas seções antes de desenhar bindings e recorrência compartilhada.
- **[R2] [wgpu 24.0.5](https://docs.rs/wgpu/24.0.5/wgpu/) e [Naga 24](https://docs.rs/naga/24.0.0/naga/):** `Device`, `Limits`, `Features`, buffers, `CommandEncoder`, compute passes, timestamp queries e validação WGSL. Não atualizar dependências apenas para seguir exemplos de outra versão.
- **[R3] [AMD HIP — performance guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html):** acesso à memória, ocupação, recursos por thread e desenho dos lançamentos. Aplicar ao motor HIP, com medições na GPU disponível.
- **[R4] [AMD ROCprofiler-SDK — rocprofv3](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html):** tracing/profiling HIP/HSA; verificar versão, permissões e suporte do hardware antes de selecionar contadores. Falta de contador não impede medir tempo end-to-end.
- **[R5] [PyTorch — C++ CUDA Streams](https://docs.pytorch.org/cppdocs/api/cuda/streams.html):** stream atual e guards. Para a extensão ROCm, verificar APIs equivalentes disponíveis via hipificação na versão instalada; não copiar um launcher que ignora a stream do tensor.

Essas referências orientam implementação e segurança de execução. Nenhuma delas demonstra que um kernel proposto é mais rápido neste projeto: essa evidência deve vir do K0/K6.
