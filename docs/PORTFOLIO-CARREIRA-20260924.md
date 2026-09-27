# Universe como evidência profissional

Revisão de 24/09/2026, HEAD observado: bd04bae. Objetivo: transformar trabalho de pesquisa e engenharia em material para entrevistas. Sem publicação ou candidatura automática.

## Posicionamento

“Desenvolvedor de software construindo uma plataforma experimental para evolução e observação de agentes em simulação GPU, com Rust, wgpu/WGSL e ferramentas de análise.”

Não se apresentar como criador de AGI, especialista em neurociência ou autor manual de todo o código. Diferenciar decisões pessoais, implementação assistida por IA e verificações realizadas. O repositório prova a existência do sistema; a entrevista deve demonstrar domínio pessoal sobre ele.

## O que foi conferido

Leitura de `git log`, diff desde c43521e, `school_batch.rs`, pontos de promoção em `main.rs` e JSONs existentes. Nenhuma nova simulação, build ou inspeção visual nesta revisão.

| Evidência | Resultado | Limite |
|---|---|---|
| school-life-20260924/curriculum-adaptive-s11-stage2 | 239 gerações em approach; depois 61 em pickup, todas sem sucesso | Promoção do currículo não prova domínio geral |
| curriculum-adaptive-s12-stage2 | 300 gerações em approach, sem promoção | Sucessos pontuais não bastam para progresso consistente |
| gen106 congelado, treino | 24/96 sucessos: 6/24 largadas, quatro cópias cada | Cópias não são amostras independentes |
| gen106 congelado, validação | 0/12: 0/3 largadas | Falha de generalização nesse conjunto |
| gen106 congelado, teste | 8/12: 2/3 largadas | Apenas três geometrias de teste |
| move-life / stopper-life / graduate-stage2-life | zero entregas em cada população | Entrega aprendida não demonstrada |
| witness-life | duas entregas, piloto programado | Controle de viabilidade, limitado a duas por código |

Fontes: `/media/llm-data/universe-runs/school-life-20260924/` e `/media/llm-data/scratch/eval-pool-20260924/pool24-full-mean-s11-gen106-{train,validation,test}/`, arquivos `school_result_gen*.json` e `organism_result_gen0000.json`.

## Três estudos de caso

1. **Um contrato de avaliação pode selecionar o comportamento errado.** Mostrar como truncamento, overshoot e agregação por largada alteram o ranking. Incluir a hipótese de worst-of-three e seu resultado negativo, não vender a primeira correção como solução.
2. **Rotação que não rotacionava.** Quando pool=K, `(geração*K) mod K=0`. A correção anterior do Codex evitou o panic de capacidade, mas deixou exames fixos. Mostrar diff, exemplo pequeno e teste de rotação. Explicar também que duas gerações consecutivas em triplas diferentes não estimam competência estável do pool.
3. **Replay como ferramenta de investigação.** Mostrar entrada sensorial, intenção, comando aplicado, efeito físico e resultado do juiz para um único ser/tick. A demonstração visual atual ainda precisa ser conferida e gravada. Não chamar pesos×ativação de prova causal sem intervenção.

## Pacote mínimo de apresentação

- README curto em inglês: problema, arquitetura, execução, resultado comprovado e limitação principal.
- Vídeo de 90–120 segundos: cenário, percepção, ação aplicada, resultado e um bug explicado. Rotular piloto e agente evoluído separadamente.
- Um estudo de caso completo com gráfico de todas as sementes, não só o melhor campeão.
- Pequeno conjunto de artefatos e instruções para reproduzir um resultado; não exigir baixar todas as baterias.
- Diagrama do fluxo real, incluindo limites CPU/GPU e pontos de readback. Não afirmar ganho de performance sem benchmark comparável.
- Registro “minha contribuição”: decisões, revisões e experimentos que Leonardo consegue explicar; agentes e ferramentas utilizados.

## Texto-base para currículo

Projeto pessoal — Universe, laboratório de agentes simulados. Condução de um projeto com implementação assistida por IA, integrando simulação GPU em Rust/wgpu/WGSL, currículo evolutivo e observação de decisões. Investigação de falhas de avaliação entre múltiplas condições iniciais e análise de generalização em cenários separados de treino, validação e teste.

Adaptar “condução” às atividades pessoais reais. Só incluir implementação de kernels, profiling ou desenho experimental como competência própria quando for possível explicar e realizar essas tarefas em entrevista. Não incluir números de speedup, redução de custo ou entregas autônomas não demonstrados.

## Texto-base para post técnico

“Estou desenvolvendo, com apoio de agentes de código, um laboratório de evolução de agentes em GPU. Um dos aprendizados foi que corrigir um bug de capacidade podia deixar outro problema: a rotação de cenários avaliava sempre as mesmas três largadas. A plataforma agora permite investigar isso com resultados por largada e avaliação separada. Há avanço em locomoção e parada, mas a transferência para entregas ainda falha. Vou mostrar o erro, a correção e os limites dos resultados.”

Revisar e incluir imagens verificadas antes de publicar.

## Direção profissional

Priorizar vagas próximas da experiência profissional já existente: engenharia backend/full-stack em produtos de IA e ferramentas para desenvolvedores. Universe funciona como diferencial de profundidade. Engenharia de sistemas/ML infrastructure é uma segunda direção, dependendo de domínio de Rust, Linux, memória, concorrência e medição. Research Engineer é uma trajetória possível, mas requer fortalecer estatística, otimização, baselines e desenho experimental. O projeto sozinho não determina senioridade.

Referências consultadas em 24/09: [Ubuntu Software Engineer](https://canonical.com/careers/6707824), [C++/Rust Graphics Engineer](https://canonical.com/careers/4439935), [Anthropic careers](https://www.anthropic.com/careers/jobs). São exemplos de famílias de função, não promessa de elegibilidade, contratação ou trabalho remoto do Brasil.

## Próximos passos orientados a emprego

1. Escolher uma família de vagas com base no currículo atual, inglês e experiência comprovada.
2. Fechar um estudo de caso e uma demo estável; não esperar terminar todo o Universe.
3. Ensaiar explicar sem assistência: equação de agregação, falha da rotação, contrato de buffer, diferença entre piloto e política aprendida, limites estatísticos.
4. Selecionar vagas compatíveis e adaptar o currículo aos requisitos, sem inventar experiência.
5. Medir candidaturas → conversas → entrevistas → propostas; ajustar a apresentação ao feedback.

Retorno financeiro esperado não estimado: faltam salário atual, função, senioridade demonstrada, inglês, disponibilidade e mercado-alvo. O mecanismo pretendido é aumentar acesso a entrevistas e demonstrar capacidade de resolver problemas; o projeto não garante aumento salarial.
