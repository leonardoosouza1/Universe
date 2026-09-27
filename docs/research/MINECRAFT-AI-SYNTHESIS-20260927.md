# Vila Viva: pesquisa de IA no Minecraft e caminho para controle verificável

**Data:** 27/09/2026. **Status:** pesquisa e proposta experimental; nenhuma execução ou alteração dos servidores. Fontes primárias: artigos dos autores, documentação e código. O levantamento cobre as principais famílias, sem prometer inventário exaustivo de todos os projetos existentes.

**Conclusão:** o relatório recebido mostra um preditor funcionando e um controlador ainda sem eficácia demonstrada. A prioridade é provar a cadeia intenção → autoridade de controle → movimento → tarefa. Mais treino sobre um atuador ineficaz não resolve essa cadeia.

## 1. Evidência local e limites desta análise

A fonte do estado atual é o [relatório enviado pelo dono](VILA-VIVA-REPORT-20260927.md), corte 17:46 UTC. Ele cita servidor `ba0e492` e Universe `06eec1e` no notebook, ainda sem push. **Não inspecionei esses commits nem seus episódios neste desktop.** Os números abaixo são reportados, não medições refeitas nesta pesquisa.

| Elo | Evidência reportada | Interpretação permitida |
|---|---|---|
| Percepção → predição | GRU de 27.016 parâmetros, 23 entradas, previsão de atividade/deslocamento | Existe aprendizado preditivo; saída não comanda o aldeão |
| Política → aplicação | Uma ordem de cama correlacionada, uma atualização; replay rejeitado | Transporte/identidade funcionaram naquele caso; benefício não demonstrado |
| Aplicação → deslocamento | `walk_to`, raio 0,25, cinco refreshes: terminou 0,326 bloco mais distante | O ensaio de atuação falhou; causa interna ainda não isolada |
| Tarefa → benefício | Vanilla dormiu em 12/20 frames; piloto em 8/20 | Esse piloto não mostrou ganho; frames correlacionados não são 20 ensaios independentes |
| Pressão de sobrevivência | Fome desligada, pouca comida, saciedade volátil | Longa execução não equivale a aprender a buscar comida |

Hipóteses concorrentes para o movimento: caminho substituído pelo Brain vanilla; pré-condição/estado da entidade impedindo locomoção; alvo/geometria mal representados; término ou refresh incorreto; comando obsoleto por relógio/cadência. **Não escolhemos a primeira como causa provada.**

O relógio já produziu um erro concreto no relatório: timestamp textual de `CANT_REACH` e confusão entre `gameTime` e tempo do episódio. Tipar apenas como número não detectaria os dois domínios temporais. O contrato precisa incluir origem, unidade, época e conversão explícita.

## 2. As formas reais de integrar IA

Há dois eixos independentes: **como acessar o corpo** e **como escolher/aprender ações**. Trocar um pelo outro não fecha a integração.

| Integração do corpo | Referências | O que entrega | Limite para nós |
|---|---|---|---|
| Cliente bot por protocolo | [Mineflayer](https://github.com/PrismarineJS/mineflayer), [Pathfinder](https://github.com/PrismarineJS/mineflayer-pathfinder) | Jogador sem renderização obrigatória; mundo estruturado, inventário, interação, navegação | Controla jogador, não aldeão. Mesmo nome de ação não implica mesmo corpo |
| Automação em cliente modado | [Baritone](https://github.com/cabaletta/baritone) | Busca de caminho e processos orientados a metas no cliente | Não é biblioteca de navegação Paper para mobs |
| Plugin no servidor | [Paper Pathfinder 1.21.4](https://jd.papermc.io/paper/1.21.4/com/destroystokyo/paper/entity/Pathfinder.html), [Citizens](https://github.com/CitizensDev/Citizens2) | Acesso a entidades/NPCs, navegação e ciclo de vida | Precisa resolver propriedade do controle e semântica do corpo |
| Mod do servidor/jogo | [SmartBrainLib](https://github.com/Tslat/SmartBrainLib), [MineColonies](https://github.com/ldtteam/minecolonies) | Controle mais profundo de sensores, memórias, comportamentos e entidades próprias | Fabric/Forge/NeoForge não são plugins Paper intercambiáveis; migração precisa de prova isolada |
| Ambiente de pesquisa instrumentado | [Malmo](https://github.com/microsoft/malmo), [MineRL](https://github.com/minerllabs/minerl), [MineDojo](https://github.com/MineDojo/MineDojo), [MineStudio](https://github.com/CraftJarvis/MineStudio) | Episódios, observações, ações, datasets e avaliação | Versões e ações próprias; resultados não certificam Paper 1.21.4 nem NPCs |
| Ferramentas/RPC/comandos | Plugin, HTTP, arquivos, RCON ou MCP como transporte | Liga processo externo ao jogo | Transporte não fornece política nem prova efeito. Teleporte/comandos administrativos alteram a tarefa |

As versões são relevantes: o runtime publicado de Voyager declara Fabric 1.19; MineRL v1.0 documenta Minecraft 1.16.5. Não são instaladores diretos para o nosso servidor. Detalhes nos dois levantamentos complementares, ao final.

## 3. Métodos de decisão e aprendizado: o que aproveitar

| Família | Trabalhos/fontes | Como funciona | Uso no Vila Viva |
|---|---|---|---|
| Regras, estados e busca | [MineColonies](https://github.com/ldtteam/minecolonies), [SmartBrainLib](https://github.com/Tslat/SmartBrainLib/wiki/How-do-Brains-Work), Pathfinder | Comportamentos programados, pré-condições, memórias, metas e caminhos | Referência obrigatória de competência. Profissões/trabalho funcionando não significam pesos aprendidos |
| LLM + ferramentas | [Mindcraft](https://github.com/mindcraft-bots/mindcraft) | Modelo escolhe ações sobre Mineflayer, com perfis e tarefas | Planejamento eventual; ações limitadas e verificadores locais. Não colocar chamada remota em cada tick |
| Currículo + biblioteca de habilidades | [Voyager](https://arxiv.org/abs/2305.16291) | Propõe tarefas, gera código, usa feedback e reutiliza habilidades | Copiar habilidades com pré/pós-condições e histórico. Aprendizado de código/memória não é treinamento dos pesos do LLM |
| Planejamento com falhas e memória | [DEPS](https://arxiv.org/abs/2302.01560), [JARVIS-1](https://arxiv.org/abs/2311.05997), [Optimus-1](https://arxiv.org/abs/2408.03615) | Decompõe objetivos, consulta experiência, seleciona subobjetivos e revisa planos | Distinguir falha de caminho, recurso ausente e tarefa impossível; impedir repetição infinita da mesma ação |
| Hierarquia com controle aprendido | [Optimus-2](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Optimus-2_Multimodal_Minecraft_Agent_with_Goal-Observation-Action_Conditioned_Policy_CVPR_2025_paper.pdf) | Planejador multimodal acima de política condicionada a objetivo, observação e histórico de ação | Separar escolha da tarefa de sua execução. O acrônimo GOAP deste artigo não é o planejador clássico Goal-Oriented Action Planning |
| Imitação de ações/vídeos | [VPT](https://github.com/openai/Video-Pre-Training), [paper](https://cdn.openai.com/vpt/Paper.pdf) | Dinâmica inversa rotula vídeos; clonagem comportamental aprende controles do jogador; há ajuste posterior | Copiar demonstrações locais e rótulos de ação. Pré-treino original usa 70 mil horas; não reproduzir isso no notebook |
| Política condicionada por linguagem/vídeo | [STEVE-1](https://arxiv.org/abs/2306.00937), [GROOT](https://arxiv.org/abs/2310.08235) | Objetivo representado por texto ou demonstração visual condiciona a ação | Ensinar “chegar/trabalhar/descansar”, não apenas prever qual atividade acontecerá |
| Aprender com objetivos retrospectivos | [Pan / Pantograph](https://pantograph.com/journal/pan-1), [HER](https://arxiv.org/abs/1707.01495) | Usa estados futuros como objetivos de treino; ações continuam necessárias na etapa de controle | Ideia para dados vetoriais locais após validar motor. Pan reporta 500 mil horas de vídeo e 2 mil com ações: escala muito diferente; HER original é robótica, não Paper |
| Recompensa aprendida/humana | [MineCLIP](https://github.com/MineDojo/MineCLIP), [BASALT](https://github.com/minerllabs/basalt-benchmark) | Correspondência texto-vídeo ou avaliação humana ajuda a avaliar comportamentos difíceis de especificar | Útil para naturalidade; entregas, dano e inventário devem ter verificação objetiva |
| RL com modelo do mundo | [DreamerV3](https://danijar.com/project/dreamerv3/), [Dreamer 4](https://arxiv.org/abs/2509.24527) | Aprende dinâmica e otimiza ações em trajetórias imaginadas; Dreamer 4 usa dados offline | Referência de aprendizado, não correção de atuador. Exige dados de ações e validação das previsões fora da distribuição |
| Modelo visual de mundo | [MineWorld](https://github.com/microsoft/mineworld) | Prediz próximo frame condicionado à ação | Simulador aprendido não é por si controlador. Repositório recomenda GPU de alto desempenho, sem encaixe pronto na iGPU |
| Coordenação multiagente | [VillagerAgent](https://arxiv.org/abs/2406.05720), [Mindcraft/MineCollab](https://arxiv.org/abs/2504.17950), [Project Sid](https://arxiv.org/abs/2411.00114) | Dependências de tarefas, cooperação, memória e papéis | Só depois de ações individuais confiáveis. Apesar do nome, VillagerAgent usa bots Mineflayer, conforme seu repositório |
| Prazos e colaboração dinâmica | [TickingCollabBench, junho/2026](https://arxiv.org/abs/2606.15684) | Tarefas com capacidades distintas, dependências, eventos e limites de tempo | Expõe falhas por latência/observação parcial. Bom desenho para comunidade, sem assumir que conversa equivale a cooperação |

**Evolução/NEAT/estratégias evolutivas** continuam sendo alternativas para otimizar uma política pequena ou suas preferências, mas não encontrei neste levantamento uma demonstração comparável de aldeões Paper aprendendo sobrevivência via esses métodos. Isso não prova impossibilidade. A seleção precisa de ações eficazes, episódios comparáveis e objetivo estável, como os outros métodos. Alterar genes de corpo só tem efeito se o servidor realmente expressar atributos/custos correspondentes.

**Craftax** é uma alternativa rápida para estudar algoritmos em um ambiente 2D JAX, mas não é Minecraft. Seus resultados não validam o atuador Paper. [Projeto original](https://github.com/MichaelTMatthews/Craftax).

Um protótipo mais próximo do corpo desejado é [LLM Villager/NeoForge 1.21.1](https://github.com/davideconte/llm-villager-neoforge): declara sensores, memória persistente, tool calling Ollama e executor. É exemplo de organização, **não evidência de melhoria contra vanilla**. Não foi instalado/auditado integralmente aqui.

## 4. O achado mais diretamente aplicável: Brain e goals são controles diferentes

Confirmei código histórico do Citizens com dependência **1.21.4** no [pom, revisão 41953403](https://github.com/CitizensDev/Citizens2/blob/41953403a345213a474a32d99b21df0d900cbca8/v1_21_R3/pom.xml#L12). O [VillagerController](https://github.com/CitizensDev/Citizens2/blob/41953403a345213a474a32d99b21df0d900cbca8/v1_21_R3/src/main/java/net/citizensnpcs/nms/v1_21_R3/entity/VillagerController.java#L104) atualiza o estado da IA antes do processamento vanilla. Em [updateMinecraftAIState](https://github.com/CitizensDev/Citizens2/blob/41953403a345213a474a32d99b21df0d900cbca8/v1_21_R3/src/main/java/net/citizensnpcs/nms/v1_21_R3/util/NMSImpl.java#L2696), desativar IA Minecraft salva/limpa **goals e o mapa de comportamentos do Brain**; reativar restaura esses componentes.

É evidência estática de que uma biblioteca madura resolve essa distinção explicitamente. Não prova que copiar esse método corrigirá o nosso ensaio. Também não garante restauração perfeita de toda memória/rotina nem compatibilidade de um jar atual com 1.21.4.

O Paper documenta `findPath` como cálculo de caminho, podendo retornar o ponto alcançável mais próximo; `moveTo` retorna início da navegação. Nenhum dos dois certifica chegada, persistência do alvo ou conclusão da tarefa. Portanto `path_accepted` não é recompensa de sucesso. [API da versão usada](https://jd.papermc.io/paper/1.21.4/com/destroystokyo/paper/entity/Pathfinder.html).

Não usar `setAware(false)` como solução automática: a API informa que isso impede ações próprias e pode desabilitar outros comportamentos, inclusive afogamento. [Mob 1.21.4](https://jd.papermc.io/paper/1.21.4/org/bukkit/entity/Mob.html#setAware(boolean)).

Duas decisões possíveis, a resolver por prova curta:

- **Preservar aldeão vanilla, nossa preferência inicial:** inserir comportamento de assistência com prioridade e duração limitadas no mecanismo que realmente controla movimento; devolver controle ao terminar e testar sono/trabalho depois. Adaptador NMS, se necessário, deve ficar isolado e preso à versão exata.
- **NPC de controle próprio:** Citizens ou entidade/mod próprio podem tornar o controle mais explícito. Isso muda o objeto experimental. A [API Citizens](https://github.com/CitizensDev/CitizensAPI/blob/master/src/main/java/net/citizensnpcs/api/npc/NPC.java) documenta proteção por padrão; dano, colisão, comércio, POIs e demais capacidades precisam ser igualados/testados. Não trocar o corpo e atribuir todo ganho à política.

Uma mensagem JSON com `owner`/`lease` não suspende o Brain vanilla. A exclusividade só existe se os escritores efetivos forem coordenados no ponto de aplicação. Se essa intervenção for inviável com a API atual, registrar a limitação e decidir o adaptador; não aumentar a frequência de `moveTo` indefinidamente.

## 5. Arquitetura mínima recomendada

```text
Minecraft / aldeão real
  → observação local com identidade, versão, gameTime e validade
  → escolha da intenção: regra de referência OU política aprendida
  → árbitro integrado ao controlador real do corpo
  → habilidade limitada: iniciar / avançar / concluir / falhar / cancelar
  → movimento ou interação observados no servidor
  → resultado da tarefa + saúde + retorno à rotina
  → dataset de ação/resultado → treinamento → avaliação congelada
```

O preditor shadow pode continuar como auxiliar. Um planejador LLM é opcional, assíncrono, acionado por tarefa nova ou falha relevante. Nem um LLM maior nem acrescentar camadas cerebrais são pré-requisitos para essa arquitetura.

Contrato de decisão mínimo: episódio, UUID, sequência, versão da observação, tick de origem, validade em ticks, objetivo, ação solicitada e parâmetros tipados. Execução registra ação efetivamente aplicada, controlador responsável, alvo/caminho atual, motivo de recusa/preempção e evento terminal. Resultado referencia a mesma sequência e versão da política. Campo ausente é distinto de zero; comando expirado não é nova amostra bem-sucedida.

A [implementação de física do Mineflayer](https://github.com/PrismarineJS/mineflayer/blob/master/lib/plugins/physics.js) consultada usa passo de 50 ms. Nossa operação reportada usa 80 TPS. Isso **não prova incompatibilidade**, mas impede assumir aceleração uniforme de servidor, cliente, polling e modelo. Medir idade da observação na aplicação, latência p50/p95, taxa efetiva de decisões e unidades de cada timeout. Se testar um bot auxiliar, começar em instância separada a 20 TPS e avaliar 80 depois; não mudar o principal sem necessidade.

Também precisamos declarar o que cada corpo pode fazer: aldeão vanilla não recebe automaticamente todo o inventário de ações de um jogador. “Craftar”, “depositar no baú” ou “beber água” exigem mecanismos reais e limites. Para economia, registrar estoque antes/depois, origem, destino e conservação; não criar item por comando e chamar isso de produção aprendida.

## 6. Como aproveitar nossos dados e treinar algo útil

Definir explicitamente uma política condicionada ao objetivo:

`h[t] = GRU(h[t-1], observação[t], objetivo[t], ação_aplicada[t-1])`

`ação[t] ~ πθ(ação | h[t], objetivo[t], máscara_de_ações_válidas[t])`

Isto é **proposta**, não descrição da GRU atual. Uma regressão ou classificador pequeno sobre candidatos pode bastar inicialmente; a recorrência deve ganhar de uma referência sem memória. O objetivo distingue “prever que vai dormir” de “selecionar uma ação para alcançar uma cama válida”.

Sequência de aprendizado proposta:

1. Habilidade determinística validada gera pares observação/ação/resultado. Demonstração do mesmo corpo; ação registrada antes de seu efeito. Posição futura não revela unicamente a ação que a causou.
2. Clonagem comportamental aprende escolhas cobertas pelas demonstrações. Avaliar também recuperação de erros; acurácia offline não basta.
3. Para escolhas locais com resultado curto bem definido, testar um seletor contextual simples. Para dependências longas, avaliar RL hierárquico; não tratar todas as decisões como bandits independentes.
4. Testar objetivos retrospectivos somente em transições válidas, recomputando recompensa/terminação do novo objetivo. Um teleporte de resgate não é demonstração de caminhar. HER pode reaproveitar experiências de metas alcançadas, mas não cria ações que nunca foram executadas. [HER](https://arxiv.org/abs/1707.01495).
5. Evolução ou aprendizado online vêm depois do gate motor; comparar com regra simples usando o mesmo corpo, informações e orçamento.

O modelo atual de próximo estado, sem condicionamento adequado à ação, não estima o efeito de todas as ações alternativas. Um modelo de dinâmica para planejamento teria que aprender `p(estado_seguinte | histórico, ação)` e ser testado nas intervenções relevantes. Dados apenas observacionais do vanilla não garantem cobertura das nossas ações.

Se precisar de recompensa densa, uma base estudada é `r' = r_tarefa + γΦ(s') − Φ(s)`, com potencial limitado e convenção terminal consistente. A garantia de preservar a política ótima depende das hipóteses do MDP e do desconto; não transferir automaticamente para observação parcial e macros de duração variável. Evitar pagar repetidamente pelo mesmo progresso bancado. [Ng, Harada e Russell](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf).

## 7. Bateria proposta — ainda não executada

Prerregistrar configuração/critério antes dos resultados. Instâncias isoladas, sequenciais, um aldeão investigado; manter orçamento de CPU/RAM do notebook já usado, com limite de tempo/disco. Não executar treinamento pesado nem reinstalar o servidor nesta etapa.

| Etapa | Ensaio e medida | Critério proposto / decisão |
|---|---|---|
| M0: validade | Campo textual/numérico, idade, gameTime, UUID, replay, comando expirado, alvo em outra dimensão, entidade morta, ausência de sensor | Rejeições corretas e explícitas; nenhuma mutação para entrada inválida |
| M1: autoridade motora | Oito snapshots simples: quatro direções, duas posições, alvo livre a 2 blocos. Até 120 ticks por tentativa. Comparar integração atual e adaptador com autoridade explícita; vanilla sem ordem serve de trajetória de referência | Distância final ≤0,25 bloco em pelo menos 7/8 tentativas, sem dano/teleporte; nenhum comando marcado sucesso fora do raio. Limiar é proposta de engenharia, não capacidade já medida |
| M1 negativo | Alvo fisicamente bloqueado, cancelamento, novo comando que substitui anterior, término e devolução ao vanilla | Falha/timeout/preempção identificados; comando antigo não retoma; sono/trabalho podem voltar após devolução |
| M2: tarefa útil | Doze episódios pareados elegíveis para travamento real: vanilla, regra simples e candidata congelada, mesmo orçamento de 600 ticks de acompanhamento | Piloto discriminante. Medir retomada de tarefa, tempo, dano e reincidência; não declarar superioridade populacional a partir desse n |
| M3: aprendizado | Demonstrações próprias, treino separado, avaliação congelada em UUIDs/geometrias/horários retidos | Comparar tarefa completa contra regra e vanilla, incluir falhas. Se empatar, manter regra e registrar ausência de benefício |
| M4: comunidade | Dois agentes, recurso/POI disputado, tarefa com dependência e uma interrupção | Handoff real de recurso, sem duplicação; cancelamento/replanejamento observados. Só executar após competência individual |

Em M1 registrar **a cada tick da janela curta**, antes/depois das fases relevantes: comando, alvo, caminho e próximo nó, WALK_TARGET, atividade/behaviors ativos quando disponíveis, velocidade, posição, estado de solo, colisão e motivo de término. Fora da janela, amostragem leve. Se o alvo mudar após a fase vanilla, investigar o escritor; se caminho persistir e velocidade não surgir, investigar controle/estado; se houver velocidade sem progresso, investigar colisão e geometria. Ausência de instrumentação não autoriza escolher a causa por aparência.

Uma cópia do save não garante duas trajetórias idênticas: registrar RNG disponível, ordem de entidades, configurações e controles; randomizar a ordem dos braços e repetir estados iniciais. A unidade de comparação é episódio/aldeão, não frame. Para afirmação confirmatória de benefício, definir efeito mínimo e dimensionar a amostra usando a variância do piloto; incluir intervalos e censura.

“Menos resgates” sozinho é manipulável: pode significar que o aldeão deixou de tentar trabalhar ou que o detector parou de disparar. Métrica primária proposta: **retomar atividade útil e mantê-la sem novo travamento no horizonte**, com dano/morte como restrições. Resgates por aldeão-dia são secundários. Parado durante sono ou interação válida não conta como travado; `fell` precisa distinguir queda perigosa de descer para terreno legítimo.

Inspeção visual obrigatória para afirmar comportamento: início, pedido, resposta, falha/sucesso e retorno à rotina do mesmo UUID, sincronizados com os números. Nenhum replay novo foi aberto/produzido nesta pesquisa.

## 8. Decisões que destravam, e o que adiar

**Agora:** obter no notebook os commits e o trace que o relatório cita; executar M0/M1 com atenção à integração Brain/navegação. Se uma regra não consegue realizar a ação, não iniciar treino. Se a regra consegue e a política não, o próximo problema já está isolado.

**Depois:** seletor pequeno sobre habilidades, memória de falhas e objetivo explícito. Usar biblioteca crescente de habilidades e verificação de efeito, inspiradas em Voyager/Project Sid. Registrar taxa de conclusão, custos, cobertura e regressões por habilidade.

**Adiar:** fome agressiva, economia completa, muitos agentes e evolução morfológica simultâneas. No relatório há pouca comida, HP baixo e saciedade que não persiste; aumentar pressão agora mistura escassez/configuração com capacidade de decisão. Primeiro implementar/validar persistência, abastecimento e aquisição/consumo no mesmo contrato.

Operação 24/7 também depende de retenção: o relatório estima cerca de 29,5 h até a reserva de disco. Não é nova leitura do disco. O orçamento de logging deve preservar eventos/manifestos, limitar traces densos a janelas e ter política explícita de rotação/arquivamento; rodar por mais tempo sem ações e feedback úteis só acumula observações.

O resultado mínimo que merece chamar de avanço é um aldeão com controle demonstrado, uma tarefa completa e um ensaio mostrando onde a política agrega valor à regra. O objetivo de longo prazo continua sendo o aldeão adaptativo; um jogador-bot auxiliar ou NPC diferente não deve ser apresentado como conclusão desse objetivo.

## 9. Material complementar e rastreabilidade

- [Métodos de aprendizado, plataformas, versões e custo](MINECRAFT-AI-LEARNING-20260927.md).
- [Bots, LLMs, habilidades e coordenação](MINECRAFT-AI-SYMBOLIC-20260927.md).

Pesquisa dividida entre dois agentes de menor custo; a síntese revisou suas recomendações e verificou diretamente as fontes decisivas. Foram corrigidas extrapolações sobre trocar aldeões por bots, suporte de hardware, disponibilidade de Project Sid e eficácia de contratos sem integração ao escritor real.

Checagens realizadas: leitura do relatório fornecido; consulta web às fontes ligadas; leitura via `urllib.request` do código público Citizens (revisões atual e histórica), API NPC, wiki SmartBrainLib e física Mineflayer; validação documental de links locais. Não houve benchmark dos frameworks, deploy, atualização de pesos nem teste comportamental do projeto nesta bateria bibliográfica.
