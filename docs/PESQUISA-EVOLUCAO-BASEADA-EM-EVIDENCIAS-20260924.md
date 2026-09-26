# Universe: pesquisa aplicada à evolução de agentes

Data: 2026-09-24. Escopo: pesquisa e propostas, sem implementar novos mecanismos ou executar treino nesta revisão.

## Diagnóstico e limite da evidência

O objetivo é um organismo que percebe, dirige, presta serviço e regula recursos, com corpo e controlador separáveis. A prioridade experimental é conseguir pickup e entrega aprendidos, preservando navegação e parada. Acrescentar mecanismos biológicos antes disso pode multiplicar as causas de falha.

Evidência local da rodada anterior, descrita em `TAXI-SERVICE-CONTRACT-V2.md`: o witness programado completou uma entrega; a comparação curta de políticas nativas legacy/v2 teve zero pickups em ambos os braços, sem pedidos de pickup. Isso verifica uma trajetória executável, não demonstra aprendizado nem identifica sozinho a causa da estagnação. Os coeficientes novos de fitness são hipóteses de engenharia, não parâmetros cientificamente validados.

As propostas abaixo são inferências para o Universe. Resultados de artigos não são resultados nossos. A consulta incluiu artigos, páginas dos autores e documentação oficial; não é uma revisão sistemática exaustiva, nem reprodução independente dos papers.

## 1. Ensinar o último passo antes de exigir a viagem inteira

**Base:** [Reverse Curriculum Generation, 2017](https://arxiv.org/abs/1707.05300) e [Jump-Start Reinforcement Learning, ICML 2023](https://proceedings.mlr.press/v202/uchendu23a/uchendu23a.pdf). O primeiro explora começos próximos do objetivo; o segundo usa uma política guia para levar o aprendiz a estados úteis e reduz a assistência conforme aprende.

**Adaptação proposta:** começar pickup dentro da área válida, velocidade zero e recursos suficientes. Depois variar posição e velocidade, aumentar distância e incluir curva. Treinar delivery inicialmente com passageiro embarcado e destino próximo, preservando episódios que exigem embarque. A escola deve amostrar uma mistura de competências, não substituir completamente a anterior pela seguinte.

Separar três avaliações: estados preparados; assistência seguida de autonomia; autonomia desde o nascimento. Só a terceira demonstra o ciclo completo. Posições fáceis devem variar para impedir memorização de um ponto.

**Experimento decisivo:** se não aprende a solicitar pickup já parado na zona válida, investigar representação da ação, limiar, mutação e crédito antes de trabalhar mapas maiores. Registrar probabilidade/frequência de pedido, aceite, recusa e motivo. Se aprende parado e falha em aproximação, o gargalo está na composição de habilidades.

## 2. Não perder estados úteis durante a exploração

**Base:** [Go-Explore, 2021](https://arxiv.org/abs/2004.12919): arquivar estados promissores, voltar a eles e então explorar ajuda problemas difíceis de exploração. A abordagem original possui requisitos de restauração e posterior robustificação; não basta copiar a ideia do arquivo.

**Adaptação proposta:** arquivo pequeno de estados perto de passageiros, após embarque e antes de cruzamentos. Célula definida por região, orientação, faixa de velocidade e fase do serviço; incluir estado interno quando necessário. Explorar a partir desses pontos pode revelar sequências que um episódio completo raramente alcança.

Um snapshot válido precisa restaurar corpo, cérebro recorrente, plasticidade, passageiro, timers, semáforos e RNG. Restaurar só a pose cria um estado artificial. Alternativa inicialmente mais simples: reconstruir uma condição inicial válida e identificá-la como currículo, sem fingir restauração fiel.

**Critério:** mais eventos úteis por milhão de ticks e posterior sucesso autônomo. Benefício limitado a resets privilegiados não prova que o agente sabe voltar ao estado.

## 3. Preservar especialistas em vez de selecionar um único comportamento

**Base:** [MAP-Elites, 2015](https://arxiv.org/abs/1504.04909), [populações de agentes RL com MAP-Elites, 2023](https://arxiv.org/abs/2303.12803), [sinergias QD/RL, 2023](https://arxiv.org/abs/2303.06164).

**Adaptação proposta:** arquivo de genomas com diferentes competências: navegar, parar perto do alvo, embarcar, entregar. Descritores devem medir comportamento útil, não apenas posição final ou velocidade. Um bom motorista pode ser preservado mesmo quando outro indivíduo maximiza o score da lição atual.

Manter duas coisas distintas: arquivo de genomas/competências e arquivo de estados de exploração. Um não substitui o outro. Cruzar pesos arbitrariamente também não garante composição de habilidades; redes podem representar funções com neurônios permutados.

**Teste:** arquivo versus seleção atual, mesmo total de ticks, medindo retenção das habilidades e entregas fora do treino. Muitas células ocupadas sem serviço útil significam diversidade sem avanço. Combinações de QD e RL adicionam critic e otimização; não são um plugin gratuito para a evolução atual.

## 4. Medir quanto a mutação destrói antes de aumentar a rede

**Base:** [Safe Mutations, 2018](https://arxiv.org/abs/1712.06563). O método adapta perturbações à sensibilidade das saídas aos pesos; a variante principal exige gradientes de saída.

**Proposta local:** medir pai e filhos em sequências sensoriais fixas, restaurando o mesmo estado recorrente inicial. Registrar alteração de motor, frequência dos serviços, saturação, deriva do estado e retenção de competências. Uma perturbação pequena nos pesos pode atravessar um limiar e alterar completamente o comportamento.

Uma primeira ablação pode comparar mutação global e por módulo, ou calibrar magnitude pelo efeito comportamental. Isso é uma aproximação nossa, não uma implementação equivalente a SM-G. BitLinear, clipping e limiares exigem tratamento explícito se forem usados gradientes.

**Critério:** maior fração de filhos viáveis, sem eliminar exploração ou reduzir diversidade a cópias. Confirmar em rollouts: estabilidade em sequências gravadas não garante estabilidade em laço fechado.

## 5. Separar reward de treino e avaliação final

**Base:** [Ng, Harada e Russell, 1999](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf): shaping potencial usa `F(s,s') = gamma * Phi(s') - Phi(s)` sob hipóteses específicas de MDP e retorno. [CARLA Leaderboard 2.1](https://leaderboard.carla.org/evaluation_v2_1/) fornece critérios de avaliação, incluindo conclusão, infrações e término por bloqueio.

Proposta: distância geodésica na rede viária, fase da tarefa e tempo restante explícitos. Distância euclidiana pode premiar avançar contra uma parede quando a rota exige se afastar primeiro. Se usar potencial por fase, tratar transições e terminal para impedir ganho repetido ao alternar pickup/dropoff.

A garantia de invariância não se transfere automaticamente ao score episódico atual: clipping, bônus de melhor aproximação e terminais mudam o problema. Além disso, um potencial que telescopa com terminais iguais pode não diferenciar os candidatos na seleção por retorno completo, embora ajude métodos de crédito temporal. Não trocar o score simplesmente por uma fórmula famosa.

Avaliação final independente: entrega correta, colisões por km, tempo, energia por entrega, abandono, violações e tarefas sem solução. Custo por tempo deve ser testado contra fuga por morte precoce e contra punição de espera legítima em semáforo. Timeout por bloqueio precisa distinguir fila, serviço e incapacidade de progredir.

## 6. Uma referência de aprendizado simples para localizar a falha

**Base:** [PPO, 2017](https://arxiv.org/abs/1707.06347). É uma referência estabelecida de otimização por gradiente, não uma promessa de sucesso neste simulador.

Comparar o cérebro atual com uma política pequena: primeiro a mesma evolução sobre arquitetura simples; depois, se necessário, PPO com ações contínuas para direção/tração e discretas para serviços. Ambos recebem exatamente a mesma informação e atuadores. Uma observação privilegiada deve ser uma terceira condição, explicitamente identificada.

Se a rede simples evolui e a atual não, investigar arquitetura/modulação/mutação. Se PPO aprende e evolução não, investigar atribuição de crédito e busca. Se nenhum aprende, revisar sensores, ambiente e orçamento; não concluir impossibilidade. Se só a política privilegiada aprende, investigar percepção e memória.

Isso exige trabalho de integração real. wgpu/WGSL não oferece automaticamente autodiff nem interoperabilidade sem cópias com PyTorch/JAX. Na AMD, verificar suporte antes de escolher biblioteca. Um treinador com readback a cada tick violaria a restrição operacional do projeto. Não portar tudo para CUDA apenas por causa de um paper.

## 7. Evoluir o mundo na fronteira do que o agente consegue

**Base:** [MetaDrive, 2022](https://arxiv.org/abs/2109.12674), [ACCEL, 2022](https://accelagent.github.io/) e [POET, 2019](https://arxiv.org/abs/1901.01753).

MetaDrive estuda diversidade composicional de cenários; ACCEL edita níveis e prioriza desafio com potencial de aprendizado; POET mantém pares ambiente/agente e transfere soluções. Nenhum prova que dificuldade crescente sem controle sempre melhora um agente.

Proposta: blocos com contratos geométricos — reta, curva, T, cruzamento, parada lateral — com largura, raio, distância entre serviços e tráfego parametrizados. Validar conectividade, folga física, spawn, curvas transitáveis e disponibilidade de recurso. Uma rota no grafo não prova que o corpo consegue percorrê-la.

Inicialmente usar dificuldade observável: cobertura, sucesso por cenário, regressão de habilidade. Chamar isso de currículo adaptativo; só chamar de ACCEL quando houver implementação fiel da estimativa e seleção correspondentes. Agente que fracassa sempre não distingue nível impossível de nível útil. Um witness ajuda a verificar existência de uma solução, sem medir aprendibilidade.

Reter mapas por topologia/família, além de seed. Aumentar um parâmetro visual não constitui novo desafio. Novos bairros podem variar tarefas, densidade, custos e recursos; centro/campo apenas estéticos não induzem adaptação.

## 8. Tráfego convincente sem evoluir todos os carros do zero

**Base:** documentação dos autores de [IDM](https://traffic-simulation.de/info/info_IDM.html) e [MOBIL](https://traffic-simulation.de/info/info_MOBIL.html).

IDM modela aceleração longitudinal com velocidade própria, distância e velocidade relativa ao líder. MOBIL adiciona critérios de segurança e incentivo para mudança de faixa. Servem para tráfego de fundo e controles de referência; não seriam evidência de inteligência evoluída do nosso agente.

Proposta: primeiro faixas, prioridade e parada em sinal; depois perfis de motorista e mudança de faixa. NPCs usam a mesma física/colisão, com fonte da ação marcada. Semáforos precisam coordenar movimentos conflitantes e lidar com filas. IDM sozinho não resolve planejamento de cruzamentos.

Evitar sobreposição no spawn e testar passo temporal, frenagem e congestionamento. Modelos contínuos precisam de integração cuidadosa; não afirmar ausência de acidentes por usar uma equação conhecida.

## 9. Corpo, homeostase e evolução morfológica

**Base:** [Homeostatic RL, eLife 2014](https://elifesciences.org/articles/04811), [DERL, Nature Communications 2021](https://www.nature.com/articles/s41467-021-25874-z), [Evolution Gym, NeurIPS 2021](https://arxiv.org/abs/2201.09863). Há modelos computacionais de regulação interna e experimentos de cootimização corpo/controlador. Não demonstram que acrescentar sede produz inteligência geral.

Proposta nossa: estado interno `h = [fuel, coolant, temperature, integrity]`, unidades e observabilidade explícitas; dinâmica física gera necessidade, ação executada altera o recurso, consequência futura afeta retorno. Registrar tanto necessidade quanto pedido, aceite e efeito. Estar perto de água não equivale a beber.

Um drive possível é `D(h)=sum_i w_i * deviation_i(h_i)^2`, com desvios adimensionais por faixa segura. Redução de drive pode orientar aprendizado, mas exige custos físicos e testes contra ciclos de danificar/reparar. Não apresentamos essa escolha quadrática como fórmula exata do artigo.

Carro não recupera pneu por descanso biologicamente. Descanso pode resfriar; reparo requer material/estação ou uma hipótese explícita de corpo regenerativo e seu custo. Distinguir analogia de modelo mecânico.

Depois de serviço básico, evoluir poucos parâmetros com compensações: massa, bateria/tanque, torque, dissipação e sensores. Mais potência/tanque não pode ser grátis. Comparar corpo fixo, corpo aleatório e corpo evoluído com igual orçamento; controlar vantagem de tamanho e energia. Genes herdados, estado do indivíduo e aprendizado durante a vida são objetos separados. DERL usa aprendizado dentro da vida e evolução entre gerações; copiar apenas mutação de corpo não reproduz esse mecanismo.

## 10. O que adiar

- [HER](https://arxiv.org/abs/1707.01495): útil com RL off-policy condicionado a metas; não é relabelar o CSV de fitness. Chegar a outro ponto pode virar meta de navegação, mas não entrega um passageiro no destino errado.
- [DreamerV3](https://arxiv.org/abs/2301.04104): modelo aprendido do mundo e treinamento em imaginação. Exige outra infraestrutura de treinamento e validação de erro acumulado. Não precisa de física diferenciável, mas precisa de componentes aprendidos treináveis; prioridade posterior.
- Novas camadas biológicas, caos controlado e campos hormonais: exigem predição e ablação próprias. Não há evidência local de que resolvam a ausência de pedidos de pickup.
- Cidade gigante e corpo inteiro evoluindo simultaneamente: ampliar só após atribuir a causa das falhas básicas.

## 11. GPU e observatório como instrumentos científicos

[GPUDrive, versão consultada 2024](https://arxiv.org/html/2408.01584v2) é referência próxima de engenharia: múltiplos mundos, sensores, física e aprendizado em lote. Usa CUDA/Madrona; não é dependência diretamente compatível com nosso Vulkan. Distingue passos totais de passos de agentes controláveis. A versão consultada também declara metas inalcançáveis em parte dos dados, ilustrando por que validar cenários importa.

Copiar o rigor de medir throughput útil e custo completo: agentes vivos/controláveis, reset, seleção, gravação, transferências e wall time até competência. FPS de kernel isolado não mede velocidade de aprendizado. Reusar geometria entre mundos é aceitável; corpos/recorrência/eventos de réplicas precisam ser isolados.

No painel mostrar funil de serviço: percebe alvo → aproxima → desacelera → entra na zona → pede → recebe aceite → transporta → entrega. Cada transição com contagem e motivo da perda. Mostrar intenção bruta, cada transformação e ação aplicada no mesmo tick; campos ausentes não viram zero.

Animação de sinapses e `peso × ativação` não demonstram causalidade. Para testar influência, repetir estado/RNG com intervenção controlada em sensor ou módulo e medir mudança na ação e no resultado. Uma intervenção pode produzir estado fora da distribuição; registrar essa limitação.

## 12. Sequência de experimentos proposta

Os números abaixo são orçamento piloto sugerido, não parâmetros publicados nem execução realizada. Rodar simulação exclusivamente na GPU, em baterias limitadas.

| Etapa | Comparação isolada | Evidência principal | Próxima decisão |
|---|---|---|---|
| E0 | Pickup parado em locais variados | Pedido e aceite por episódio; thresholds e suppressions | Corrigir exploração/contrato se nem isso surgir |
| E1 | Início atual vs currículo próximo→longe | Sucesso autônomo desde o início normal | Expandir distância e curva se houver transferência |
| E2 | Seleção atual vs arquivo de especialistas | Retenção, pickups e entregas | Manter arquivo só com ganho útil |
| E3 | Mutação atual vs magnitude calibrada por módulo | Filhos viáveis e progresso por tick | Ajustar busca antes de ampliar rede |
| E4 | Cérebro atual vs política pequena | Mesmos sensores/corpo, orçamento e cenários | Localizar custo da arquitetura |
| E5 | Família única vs blocos variados | Mapas retidos, colisões/km, entregas | Escalar mundo se generalização aumentar |
| E6 | Corpo fixo vs corpo evoluído | Entregas/energia e adaptação em novos cenários | Ampliar genes apenas com benefício |

Começar com três seeds independentes para triagem; ampliar para cinco ou mais finalistas conforme incerteza/custo. Repetir clones da mesma política não cria seeds de treino. Usar cenários pareados entre braços, orçamento em ticks e wall time, versões/hashes de kernels, genoma, física, mapa, observação e reward. Congelar candidatos antes da avaliação retida. Treino/currículo não pode consultar o teste final.

[Statistical Precipice, NeurIPS 2021](https://arxiv.org/abs/2108.13264) fundamenta reportar incerteza e perfis de desempenho. Para nosso caso, reportar por seed e por família, com intervalos que respeitem os agrupamentos. Não transformar milhares de ticks correlacionados em milhares de amostras independentes.

Não iniciar E2–E6 todos juntos. Critério operacional inicial: conseguir pedidos aceitos em E0; depois autonomia de aproximação; depois ciclo completo. Nenhum limiar único de score substitui essas observações.

## Conclusão de engenharia

O projeto tem uma hipótese pesquisável: evolução de controle incorporado em um simulador GPU observável. Para sustentá-la, precisamos demonstrar aprendizado útil e transferência com controles comparáveis. A aposta imediata mais justificável é currículo de estados úteis + preservação de competências + auditoria da mutação. Mundo adaptativo e evolução do corpo vêm depois, mantendo contratos físicos e avaliação independente.

Correção da discussão anterior: não há base para afirmar que todos os projetos citados têm cérebro inferior ao nosso, que penalidade por tick basta, ou que colisão terminal sempre bloqueia aprendizado. A versão de GPUDrive consultada reporta aprendizado com objetivo esparso e término por colisão. O conjunto algoritmo, observação, exploração, reset e avaliação determina o resultado; uma regra isolada não explica tudo.
