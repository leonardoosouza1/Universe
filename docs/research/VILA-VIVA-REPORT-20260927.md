> Fonte: relatório fornecido pelo dono em 27/09/2026. Cópia para consulta após clone; os caminhos e números abaixo são históricos do notebook, não medições refeitas nesta publicação.

# Vila Viva — relatório independente do projeto

**Corte:** 27/09/2026, ~17:46 UTC (14:46 BRT). **Objetivo do dono:** manter
um servidor Minecraft headless 24/7 no notebook, acessível pela LAN, com
aldeões que aprendam a tomar ações úteis e superem o Brain vanilla em tarefas
medidas. O projeto ainda **não demonstrou IA controladora melhor que vanilla**.
Este relatório separa operação estável, instrumentação, pilotos de controle
e o que falta provar. Os dois repositórios locais são
`/home/leonardo/projetos/IA/m/vila-viva-server` (deploy/Paper/painel/piloto)
e `/home/leonardo/projetos/IA/m/Universe` (política Rust canônica e pesquisa).
Os episódios e segredos ficam fora do Git.
Revisões locais com código e pesquisa deste ciclo: servidor `ba0e492`
(`main`) e Universe `06eec1e` (`organism/school-20260922`); nenhum push
externo foi feito.
O adendo de 17:46 corrige a interpretação do campo `CANT_REACH`; o
relatório original até17:30 não detectava o erro de tipo/relógio.

## Estado real agora

| Componente | Estado comprovado | Limite |
|---|---|---|
| Minecraft Paper | Plugin principal `0.1.1-notebook`, ~20 aldeões, Brain vanilla, LAN em `0.0.0.0:25565` | Piloto IA **não** implantado no mundo principal |
| Painel | `0.0.0.0:8131`, pronto, 20 saídas do modelo aprovado | Pode exibir alvo visual antigo; confirmar `tick query` e taxa observada |
| Modelo `shadow-20260926-220823` | GRU Rust/CPU, 27.016 parâmetros, 23 entradas por aldeão; prevê atividade e deslocamento futuro | Saída de movimento **não é política**; não escreve ação |
| Política Rust de cama | `vila-goal-policy/v2` canônica em Universe e cópia byte-idêntica no servidor; decide `claim_bed`/`return_home` em opt-in | Só uma atualização experimental; nenhum ganho contra vanilla |
| Perfil operacional | `80 TPS` alvo e observado, `power-saver`, Knyon1 OP nível4, serviços automáticos ativos | 400 TPS foi o padrão solicitado antes; 80 TPS é preferência noturna posterior para menor CPU e não deve subir sozinho |
| Recursos recentes | Na janela17:10–17:22: CPU host média42,8% de8 threads durante boot isolado, RAM disponível ≥10,045 GiB, CPU pacote51–54 °C, disco livre ≥44,55 GiB; sem novos throttles, gaps ou resets | A margem até reserva10 GiB é ~29,5 h à taxa conservadora1,17 GiB/h; cópias de teste aceleram consumo de disco |

No mundo há **34 camas completas para20 aldeões**, mas só **3 postos**
(2 fazendeiros, 1 ferreiro; 17 sem profissão). Quatro casas foram adicionadas
em duas fases; um corredor inicialmente bloqueado por tábuas foi corrigido,
e as memórias de HOME subiram de7/20 para20/20. Portanto mais camas, por si,
não resolvem o gargalo. Um teste com14 postos extras não gerou profissões e
teve custo de TPS; esses postos foram removidos. O diagnóstico correto para
profissões é reivindicação/POI/acesso por um ou dois postos em cópia pareada.

A fome do plugin está **desligada no mundo principal**. Na leitura das16:44,
os20 aldeões carregavam só18 itens comestíveis (4 pães e14 batatas,
30 unidades na escala do plugin), distribuídos por8 aldeões; três tinham
1 HP e sete HP≤6. Ativar fome mais rígida agora poderia matar por falta de
suprimento, sem testar uma política de busca/comida. `Hunger.java` mantém
`satiety` apenas em memória, outro bloqueio para um ensaio persistente.

## O que realmente aconteceu nas ~12 horas analisadas

1. O notebook recebeu deploy headless Docker/systemd, monitor de CPU/RAM/
   temperatura/ACPI/disco/TPS, guarda térmica e de disco, backup consistente,
   LAN e OP. A linha de base a20 TPS teve CPU host média18,97%, pacote
   médio70,2 °C; no ensaio inicial a400 TPS CPU host média35,37%, pacote
   médio73,8 °C e Minecraft ~2,44 GiB. O acervo cresceu1,54 GiB/h a400 TPS.
   Depois de mudanças operacionais, o dono pediu menos CPU para dormir;
   80 TPS/power-saver mantêm o principal estável. Ver
   `NOTEBOOK-HARDWARE-20260926.md` e `NOTEBOOK.md`.
2. Foram construídas/copíadas casas, corrigido corredor para camas e
   retirados obstáculos locais depois de duas mortes por arbusto/fogueira,
   sempre após backup. Houve benefício ambiental/vanilla, não aprendizado.
3. O piloto Java opt-in passou a aceitar comandos de um UUID por arquivo
   atômico, com episódio/UUID/seq/ticks e revalidação no Paper. Rust propôs
   `return_home` a partir de observação Paper, despachou uma ordem, Paper
   registrou resultado e o ingestor fechou `learn` uma vez: `updates=1`,
   recompensa5,1877, replay rejeitado. No par do mesmo save, o vanilla
   dormiu **12/20 frames** de forma contínua, contra **8/20** com piloto.
   Isso é falha do gate de benefício, não IA melhor.
4. Uma rota HOME que o sensor `Pathfinder.canReachFinalPoint()` chamou
   inalcançável ainda permitiu sono pelo Brain vanilla após abrir dois blocos
   **só numa cópia**: controle0/20 frames dormindo, variante18/20. Esse
   resultado identifica um efeito de geometria e uma divergência entre o
   sensor do plugin e a navegação do Brain; não prova vantagem da política.
5. Um preditor offline de quedas foi refeito após descobrir vazamento de
   acionabilidade. Na validação dias16–20, marcou6/24 resgates com47 alertas,
   praticamente igual à regra física (6/24 com52). No holdout congelado
   dias21–25, o escore marcou5/16 com49 alertas, regra física7/16 com67.
   Hipótese de controlador de queda **rejeitada**, pesos não promovidos.
6. Dados fechados dos dias21–27 do episódio corrente: **38 resgates**
   (24 `fell`, 14 `stuck`) em7 dias, média5,43/dia ou0,271 por
   aldeão-dia. Uma primeira soma de agente estava errada e foi recontada
   diretamente nos arquivos. Nos dias31–33 houve22 resgates totais
   (7,6,9): a antiga fileira de camas teve só1 `stuck`; uma célula de
   queda teve3 eventos, um/dia, de três UUIDs. Sem aresta causal do terreno
   identificada, não foi feita obra no vivo.
7. Para atacar `stuck`, o novo produtor Java opt-in
   `vila-survival-observation/v2` expõe um snapshot Paper fresco de um UUID:
   episódio/v/tick, posição/saúde/chão/sono/profissão, comida/saciedade
   nullable, HOME/JOB_SITE/WALK_TARGET/CANT_REACH e idade, estagnação e
   quatro saídas cardeais com materiais e rejeição de segurança. O mesmo
   `NudgeSafety` é usado pelo sensor e atuador. Paper isolado publicou
   quatro snapshots consecutivos13780–13840, 3/4 direções seguras,
   zero drops. Um `nudge` norte seq1 teve ação/resultado correlacionados,
   mas deslocou0,230 bloco e terminou **-0,106 bloco** na direção pedida.
   Segurança/identidade passaram; eficácia falhou.
   **Correção posterior 17:46 UTC:** o v1 não expunha de fato a idade de
   `CANT_REACH`, pois `BrainReader` entrega `"4939"` (texto) e o parser
   exigia número. O relógio da memória é `World.getGameTime()`, não o
   `tick`/`getFullTime()` do episódio. Em nova cópia Paper, o v2 publicou
   tick13780–13840, gameTime4944–5004, since4939 e idades5/25/45/65;
   exit0/zero drops, sem comando. Logo o v10 tinha esse sinal, embora
   não tenha havido resgate `stuck` na janela. A decisão Rust e o ganho
   seguem ausentes. O v9 real perdeu profissão no tick11308838 após
   memória persistente de rota incompleta até sua bancada; causa física/
   estado atual do POI ainda precisam de ensaio isolado.
8. Um par seguinte testou o `walk_to` já existente para passo local de
   um bloco. Controle sem ordem avançou **-0,047** bloco ao norte;
   variante com caminho de2 pontos aceito terminou com só **0,053** bloco
   ao norte. Causa concreta: tolerância fixa de chegada1 bloco marcou
   chegada quando ainda faltavam0,98 bloco e suprimiu refresh. Código opt-in
   agora aceita `arrival_radius` validado0,20–1,00 (padrão1,00 para
   compatibilidade); build passou. Com raio0,25 e cinco refreshes,
   o Paper aceitou a rota, porém o aldeão terminou **0,326 bloco mais
   longe** do alvo e avançou **-0,347 bloco** ao norte, sem dano, exit0/
   zero drops. Isso refuta o Pathfinder local como atuador eficaz neste
   fixture; nenhuma variante foi implantada no principal.

## Contrato de IA, elo por elo

| Elo | Entradas/saídas | Estado |
|---|---|---|
| Mundo → Brain vanilla | Paper vê terreno, POIs, entidades, memórias e ações nativas; o `Recorder` expõe parte disso por UUID/tick | PASS para telemetria; Brain continua controlador principal |
| Mundo → shadow | 23 features em ordem fixa: posição, velocidade, direção, corpo/comida, sono, luz, relógio, profissão; GRU prevê atividade e deslocamento | PASS preditivo; **não** dá objetivo, terreno detalhado ou recompensa de ação |
| Mundo → Rust cama | `vila-bed-candidates/v3`: noite, HOME, sono, estagnação, camas/rotas; cinco features para escolha | PASS isolado; `findPath` não equivale à capacidade de dormir |
| Mundo → Rust sobrevivência | Novo snapshot Java tem WALK_TARGET/CANT_REACH/saídas/comida; Rust ainda não o consome | **FAIL**: política `unstick`/comida/profissão ausente |
| Rust → Paper | `--dispatch-once` de cama escreve comando atômico com episódio/UUID/seq/frescor; Paper revalida e aplica | PASS isolado para uma ação de cama; sem invocação contínua no principal |
| Ação → resultado | `pilot_action/outcome` de `nudge` têm identidade/ticks; `walk_to` informa progresso, saúde, sono, HOME | PASS mecânico isolado; **FAIL/NOT MEASURED** para benefício de `unstick` |
| Resultado → aprender | Ingestor de cama exige episódio fechado, zero drops, identidade/tick/alvo, persiste estado e rejeita replay | PASS uma amostra experimental; sem aprendizado útil demonstrado para outras tarefas |
| Avaliação → promoção | Parear mesmo save/UUID/tempo, incluir vanilla e regra simples, medir tarefa final/saúde/TPS/CPU | FAIL para ganho; nenhum peso promovido, mundo principal segue vanilla |

O maior erro de arquitetura até aqui foi confundir **ver um sinal**, **emitir
uma decisão** e **melhorar um resultado**. O shadow observa comportamento
vanilla e pode prever movimento, mas não aprendeu uma política de controle.
O Rust de cama usa um campo de estagnação, mas a tarefa não tinha falha
vanilla persistente e o piloto interferiu no sono. Para `unstick`, faltava
terreno; agora existe observação, porém o atuador ainda não deslocou o ser
de modo significativo. Treinar sobre `safe=true` ou `path_accepted=true`
seria aprender um proxy errado.

## Bloqueios, em ordem causal

1. **Motor e tarefa:** validar um ato curto que gere progresso dirigido
   sem dano em um aldeão realmente elegível (`cant_reach_age` e posição
   estagnada), antes de aprender. Não premiar comando aceito como sucesso.
2. **Política de sobrevivência:** definir no Universe contrato versionado
   de objetivo/ação/resultado para `unstick`, consumindo o snapshot real,
   com `continue_vanilla` como opção, identidade/frescor, ação limitada,
   recompensa por resgate evitado e dano; sincronizar runtime servidor.
3. **Comparação:** no mesmo save consistente, várias sementes/UUIDs e
   horizonte igual, comparar vanilla, regra cardinal fixa e política
   congelada. Métrica primária `stuck` por aldeão-dia; secundárias avanço
   dirigido, saúde/morte, sono, quedas, CPU/TPS e rejeições.
4. **Alimento:** fome desligada, comida insuficiente e `satiety` volátil.
   Persistir estado, garantir suprimento e uma ação de adquirir/consumir
   comida; só então ensaiar intervalo mais curto em cópia, sem misturar
   com mudança de população/postos.
5. **Profissões e densidade:** 17 sem trabalho apesar de casas/camas;
   ensaiar1–2 POIs acessíveis com verificação de reivindicação/rota,
   e redução de20→12 somente em outro braço. Nenhum dos dois pode ser
   chamado de melhoria de IA sem política, baseline e resultado.
6. **Autonomia operacional:** principal está saudável, mas episódios
   crescem continuamente. Manter guarda de disco/temperatura, backup
   consistente e previsão de espaço; nunca apagar o acervo para “ganhar”
   uma métrica. Não alterar BIOS/ACPI/sensores nem publicar portas RCON.

## Onde verificar e como reproduzir sem tocar no vivo

- Estado curto e próxima ação: `LOOP-STATE.md`; histórico operacional:
  `HANDOFF-NOTEBOOK.md`, `AUTONOMOUS-24H.md` e diário
  `~/.local/state/vila-viva/AUTONOMOUS-JOURNAL.md`.
- Mapa detalhado de contratos: `docs/AI-FLOW-CONTRACT-AUDIT.md`;
  arquitetura e critérios: `docs/SURVIVAL-BRAIN-ROADMAP-20260927.md`;
  piloto: `docs/AI-CONTROL-PILOT.md`.
- Preregistros e resultados canônicos de política:
  `Universe/docs/experiments/VILA-POLICY-PILOT-20260927.md`.
  Runtime: `Universe/crates/vila-policy-cpu/` e cópia idêntica
  `vila-viva-server/cpu-policy/`.
- Dados privados: `~/.local/share/vila-viva/episodes/`; medições:
  `~/.local/state/vila-viva/measurements/24h.jsonl`;
  ensaios isolados: `~/.local/share/vila-viva-ai-eval-20260927/`.
  Não versionar dados, senhas, `deploy/.env` nem comandos RCON com senha.
- Testes de código: `cd plugin && ./gradlew --offline --max-workers=1 shadowJar`,
  `cd cpu-policy && CARGO_BUILD_JOBS=1 cargo test --offline --locked`,
  `python3 -m unittest discover -s tests`, `git diff --check`.
  No host, os testes Python exigiram uma `.venv` local ignorada pelo Git
  com `deploy/requirements-panel.txt` e `duckdb`; nessa venv **15/15**
  testes do servidor passaram. A política Rust passou **10/10** testes
  nas duas árvores, o ingestor Python **3/3** em ambas, `cargo fmt --check`
  e `shadowJar` passaram. O teste sem venv falhou apenas por falta das
  dependências `duckdb`/`uvicorn`/`fastapi` no Python global.
  Ensaios Paper levam minutos de boot mesmo com apenas centenas de ticks;
  executá-los **sequencialmente**, com container0,5 CPU/3 GiB/sem porta,
  stop limpo e `episode_end.lines_dropped=0`.

**Perguntas para a outra IA:** qual tarefa tem ganho final plausível nos
dados atuais? Qual variável Paper mede o obstáculo antes de agir? Um comando
aceito realmente altera o corpo contra o Brain vanilla? Como parear
ação/resultado/recompensa sem vazamento temporal e comparar com uma regra
simples? Que mudança resolve o primeiro elo FAIL sem arriscar o mundo 24/7?
