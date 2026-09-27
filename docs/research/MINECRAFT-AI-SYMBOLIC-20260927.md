# IA útil para Vila Viva no Paper 1.21.4 — pesquisa dirigida

Data: 2026-09-27. Escopo: arquitetura e evidência pública primária; não houve instalação, execução nem alteração do servidor. “Compatível” abaixo significa apenas o que a fonte declara: a prova real continua sendo um episódio controlado no seu Paper 1.21.4.

## Decisão arquitetural

Há dois corpos que não se devem confundir.

| Corpo | Onde roda e o que controla | Papel recomendado |
|---|---|---|
| **Jogador-bot** | Cliente do protocolo Minecraft; entra no Paper como uma conta/jogador. Mineflayer cria esse bot e expõe mundo, entidades, inventário, crafting, mineração e ações. [Mineflayer](https://github.com/PrismarineJS/mineflayer) | Operário/testemunha: coleta, constrói, testa receitas, executa uma tarefa verificável. |
| **Aldeão/NPC** | Entidade que já existe no processo do servidor. Paper é a superfície de plugin; não é um cliente Mineflayer. [Paper docs](https://docs.papermc.io/paper/dev/) | Personagem persistente e social. O plugin deve observar o estado e arbitrar metas; mudar navegação/goal selector de aldeão é acoplamento de servidor, não “ligar o Pathfinder”. |

Portanto, não permitir que o piloto Rustbed e o `Brain` escrevam ambos no mesmo controlador sem um dono explícito. Como o objetivo é aldeões úteis, o primeiro produto pode ser o **aldeão** com intenção observável e execução vanilla/controlada. Mineflayer é um controle auxiliar: ótimo para testar mundo, executar tarefas externas e comparar políticas, mas não substitui o aldeão sem essa decisão explícita.

O `mineflayer-pathfinder` é A* para um bot Mineflayer: metas estáticas/dinâmicas, custos configuráveis, quebrar/colocar bloco, atualização de caminho e eventos `success/partial/timeout/noPath`. A própria documentação avisa que o gerador pode bloquear o event loop até o limite por tick e expõe limites como `thinkTimeout`, `tickTimeout` e `searchRadius`. [Pathfinder README/API](https://github.com/PrismarineJS/mineflayer-pathfinder) Isso é excelente para transformar uma meta simbólica em locomoção, mas não é IA de aldeão vanilla nem planner de alto nível.

Baritone é a outra referência de navegação, porém é um mod de cliente (Forge/Fabric) e sua API é voltada a processos/metas do cliente, não a uma API de plugin Paper. [Baritone, repositório oficial](https://github.com/cabaletta/baritone) Ele é útil para estudar políticas de custo, metas e cancelamento; não o adotaria como dependência central de um notebook headless/Paper sem antes aceitar a pilha de mod de cliente e validar a versão exata.

## O que “aprende” de fato

**Planejar/código não é treinar pesos.** O padrão prático para o i7/20 GB é memória explícita: episódio → verificação → habilidade versionada (JSON/Rust/JS) → recuperação por pré-condições. O LLM pode decompor e escrever uma proposta, mas só uma máquina de estados/validador executa primitivas permitidas. Isso aprende novas **habilidades e planos**, sem treinar a GRU nem fazer fine-tuning local. A GRU preditora Rust de 27k pode continuar como sinal barato (risco, próximo estado, anomalia), não como autoridade de movimento.

Voyager é a fonte mais útil para esse desenho: currículo automático, biblioteca crescente de código executável e ciclo de prompt com feedback/erros/autoverificação; relata 3,3× mais itens únicos, 2,3× maior distância e até 15,3× mais rapidez em marcos tecnológicos frente aos baselines que avaliou. [Artigo](https://arxiv.org/abs/2305.16291), [repositório](https://github.com/MineDojo/Voyager). A limitação decisiva é operacional: o repositório declara teste em **Fabric 1.19**, requer mods Fabric e GPT-4. Logo, copiar a **biblioteca de skills + verificadores**, não tentar encaixar seu runtime inteiro no Paper 1.21.4.

Mindcraft é uma base mais próxima do uso atual: combina Mineflayer e LLM, tem perfis para modelos de conversa/código/visão/embedding, exemplos de tarefas com meta, inventário inicial, alvo, timeout e contagem de agentes. [Mindcraft README](https://github.com/mindcraft-bots/mindcraft) Seu artigo descreve colaboração ação a ação/multiagente. [Mindcraft paper](https://arxiv.org/abs/2504.17950) Mas o README alerta que, se habilitado, o LLM pode escrever/executar código e que há risco de prompt injection mesmo com sandbox; essa capacidade vem desabilitada. Para Vila Viva, copie o esquema de perfil/tarefa e a comunicação entre agentes, **não** a execução livre de código. O projeto declara suporte Java até 1.21.11, mas isso não substitui teste de login, inventário, blocos e tarefas no seu 1.21.4.

Project Sid/Altera é inspiração para populações de agentes, agora com fonte primária: o artigo apresenta PIANO, *action-awareness* que compara efeito esperado e observado, e um controlador cognitivo que coordena módulos; seus resultados são preliminares. [Project Sid, artigo/HTML](https://arxiv.org/html/2411.00114v1) Isso é diretamente útil como desenho de laço de erro e coordenação, não como kit: não localizei ali um controlador NPC/Paper, pesos, protocolo de aldeão ou benchmark reproduzível para Paper 1.21.4. Não transfira seus números para Vila Viva nem o ponha na cadeia de dependências.

## Contratos de motor propostos

1. **Observação (Paper → Brain):** `tick, entity_id/uuid, espécie, posição/velocidade, health, inventário permitido, alvo/atividade vanilla, blocos e entidades locais, eventos`. O servidor produz um snapshot imutável e registra a mesma sequência.
2. **Intenção (Brain/LLM/GRU → árbitro):** `actor, goal, prioridade, ttl_ticks, orçamento, pré-condições, ações permitidas, razão`. Não enviar teclas, NMS ou código. LLM só pode emitir esse objeto validável.
3. **Arbitragem (único escritor real):** prioridade de segurança > regra de mundo > tarefa confirmada > navegação > idle. Lease por ator, cancelamento idempotente e `generation` só funcionam se o escritor vanilla do Brain/Rustbed for integrado ao árbitro e todos os caminhos de escrita passarem por ele; JSON isolado não impede sobrescrita. Comando velho é descartado no ponto que realmente aplica o caminho/meta.
4. **Execução:** para jogador-bot, adaptador Mineflayer converte intenção em `goto`, coleta, craft, interação; o pathfinder retorna êxito/falha. Para aldeão, adaptador Paper inicialmente realiza efeitos seguros de alto nível/observação; um futuro controlador de entidade é outro adaptador, não outro Brain.
5. **Resultado e memória:** `started/progress/succeeded/failed/cancelled`, diferença de inventário/mundo e motivo. Só promover uma skill depois de N execuções reproduzíveis; guardar versão, seed, versão do jogo, pré-condições e pós-condições. Múltiplos agentes compartilham apenas skills aprovadas e blackboard de fatos com TTL, não conversa crua.

Isso mantém o tick crítico local e determinístico: LLM/planejador assíncrono nunca bloqueia Paper, e o executor aceita apenas ações curtas, com timeout e rollback/compensação definidos. Para 20 GB RAM/iGPU, priorize modelo remoto ou modelo pequeno quantizado apenas como planejador eventual; CPU local deve ficar para Paper, bot, logs, busca de memória e a GRU. Visão por pixels é opcional e cara: Mineflayer já fornece estado estruturado de bloco/entidade; use imagem só quando a decisão depender de interface/estética que a API não expõe.

## Roteiro de adoção e gates

1. Fixar Paper **1.21.4** e dependências; a API oficial lista versões/builds. [Paper downloads API: 1.21.4](https://api.papermc.io/v2/projects/paper/versions/1.21.4) Em mundo descartável, provar no mesmo aldeão/tick: observação, intenção, decisão do árbitro, escrita integrada no Brain vanilla, caminho observado e resultado visual.
2. Validar primeiro `ir_a` no aldeão. Depois considerar `coletar_item_permitido` e `depositar_em_bau`, implementando explicitamente as capacidades que esse corpo não oferece; elas não vêm prontas por existir uma API equivalente para jogador. Cada skill tem orçamento de blocos/ticks, alcance, validação de inventário e falhas explícitas. Mineflayer seria controle auxiliar opcional, em ensaio separado e limitado; seu Pathfinder permite cavar e construir se configurado. [Pathfinder movements](https://github.com/PrismarineJS/mineflayer-pathfinder#movement-class)
3. Só então adicionar planner: ele decompõe “abasteça a vila” em chamadas fechadas às skills, recebe resultado esperado/observado e propõe revisão. Multiagente começa com papéis (observador, planejador, executor), um executor por ator e custo/timeout por turno.

## O que copiar / evitar

**Copiar:** a biblioteca de habilidades executáveis e verificadas do Voyager; o efeito esperado/observado e coordenação modular de Project Sid; metas/custos/eventos do Pathfinder; task schema, perfis e colaboração por ação do Mindcraft; observabilidade estruturada do Mineflayer. O Mineflayer declara suporte à família 1.21, mas a configuração deve fixar e testar 1.21.4, em vez de supor compatibilidade por faixa. [Mineflayer README](https://github.com/PrismarineJS/mineflayer)

**Evitar:** tratar Mineflayer como substituição autorizada do aldeão; chamar um aldeão Paper de “bot Mineflayer”; usar Baritone como se fosse API de servidor; colocar um LLM no loop de tick; supor que JSON/lease sem integração ao escritor impede sobrescrita; deixar LLM gerar/executar código ou comandos sem allowlist; interpretar métricas de Voyager/Sid como garantia para Paper 1.21.4; chamar armazenamento de código/episódios de “treino de pesos”.
