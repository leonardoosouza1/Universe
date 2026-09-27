# Investigation: ação de abrigo para um aldeão

Status: OPEN
Date and responsible agent: 2026-09-27, Codex no notebook de Leonardo
Related acceptance: aldeão usa uma política de IA para decidir ações reais e
supera vanilla em uma tarefa relevante. O mundo 24/7 deve permanecer seguro.

## Diagnóstico preregistrado: por que HOME não tem rota — 15:18 UTC

O v10 opt-in `policy-stall-adult-1508` estava em `(-499.52,95,-369.85)`
com HOME lembrada `(-500,96,-368)`; observação final mostra duas
aproximações seguras, duas consultas ao Pathfinder, zero rotas completas.
`no_complete_path` agrega causas diferentes: `findPath=null`, destino não
alcançado ou caminho >64 pontos. Antes de alterar objetivo, waypoint ou
mundo, instrumentar **somente** `BedCandidates` opt-in para anexar uma lista
limitada de diagnósticos dessas consultas já realizadas: coordenada/cota
do alvo, blocos nos pés/cabeça/suporte, distância, se o caminho é nulo,
`canReachFinalPoint` e contagem de pontos. Publicar também bloco sob corpo,
bloco da HOME e parte/orientação da cama. Manter `vila-bed-candidates/v3`:
os novos campos são diagnósticos opcionais e Rust continua ignorando-os;
não ampliar raio nem número de consultas. Nenhum comando/ação nesta etapa.

Teste planejado: build Java offline, rodar cópia nova parada da fonte
`policy-feedback-1425`, UUID v10, 0,5 CPU/3GiB/sem portas até pelo menos
uma observação noturna com HOME e `home_path_queries>0` (ou prazo finito),
guardar payload/episódio, parar limpo/zero drops. Diferenciar obstrução
geométrica, consulta nula, caminho incompleto e limite de64. Se o resultado
for inconclusivo, registrar o que falta; não modificar mundo principal.

Resultado 15:26 UTC: cópia `policy-route-diagnostics-1520` fechou exit0 no
tick13782, 1.107 linhas/zero drops, sem comando. HOME é cama azul válida;
corpo em `(-499.82,95,-369.96)` sobre grama. Das oito aproximações, duas
em `(-501,96,-368)` e `(-501,96,-367)` tinham pés/cabeça no ar e piso de
pedra, mas `findPath` retornou **incomplete/2 pontos** para ambas. As outras
eram cama/pedra/vidro/baú e falharam no gate de segurança; nenhuma rota >64
ou consulta nula. Consultas RCON **somente leitura no principal** confirmaram
parede cobblestone em `(-501,96,-369)` e glass_pane em
`(-501,97,-369)` imediatamente antes da entrada livre; também pedra em
`(-500,96/97,-369)`. Isto sugere barreira geométrica, não prova única causa.

Perturbação causal preregistrada 15:27 UTC, **somente nova cópia** parada da
fonte `policy-feedback-1425`: mesmo v10, jar diagnóstico, 0,5 CPU/3GiB,
sem portas/sem ordem de IA. Primeiro captar observação pré-intervenção com
HOME `no_complete_path`, preservar JSON. Depois abrir apenas os dois blocos
`(-501,96,-369)` cobblestone e `(-501,97,-369)` glass_pane com RCON nessa
cópia (`fill ... air`), preservando a fonte/world original como backup
consistente. Captar observação pós-intervenção, rota/contagem de pontos e
sono/saúde por janela até +200 ticks; parar limpo. Hipótese: a parede bloqueia
as entradas seguras; se rota continuar incompleta, não atribuir causalidade
nem repetir aberturas arbitrárias. O principal não recebe a abertura.

Execução 15:33 UTC: `policy-route-opening-1528` captou o baseline real no
tick13780 e o RCON removeu exatamente os dois blocos somente na cópia.
O monitor rejeitou por engano a resposta textual `Successfully filled 2
block(s)` (esperava `2 blocks`) e acionou seu `finally`, parando o servidor
limpo no13788, episódio com 1.187 linhas/zero drops. Não há observação
pós-intervenção; este ensaio é inconclusivo. Repetir **a mesma perturbação
preregistrada**, em cópia nova da fonte original parada, corrigindo apenas
o reconhecimento da resposta RCON e aguardando amostras até +200 ticks.
Não usar o mundo já modificado como baseline, nem alterar o principal.

Preparação da primeira repetição `policy-route-opening-retry-1535` falhou
antes da intervenção: a cópia herdou da fonte o UUID do v0 no config, em vez
do v10 preregistrado. O monitor exigia v10, não achou observação elegível;
cópia interrompida e parada limpa sem abrir blocos/emitir comando. Nova
cópia deve conferir o UUID v10 **antes do boot**. Não misturar observações
do v0 com a série v10.

Resultado exploratório 15:50 UTC: terceira cópia `policy-route-opening-v10-1542`
confirmou UUID v10 antes do boot, captou observação Paper pré no tick13780:
HOME BLUE_BED, duas consultas `incomplete`/2 pontos, `no_complete_path`.
O RCON confirmou materiais e removeu só os dois blocos na cópia. No tick
13800, o plugin ainda marcou `no_complete_path`, agora `path_null` para as
duas aproximações; o v10 estava em movimento/subindo. No keyframe13813,
o Brain vanilla já o pôs a dormir (observação13820 `sleeping=true`, saúde20,
zero resgates). Cópia exit0 no13822, 2.094 linhas/zero drops, nenhum
`pilot.json`/evento de comando; fonte e principal intactos. O monitor
encerrou antes de +200 por bug ao indexar `home_route_diagnostics=[]` quando
o aldeão dorme. Comparador sem abertura `policy-stall-adult-1508` da mesma
fonte/UUID/tick13773, sem comando, permaneceu acordado em 21 frames até
14153, saúde20/zero resgates. É um par exploratório sujeito à variação
entre boots, **não** evidência de IA nem prova causal geral. O achado mais
firme de contrato é que `findPath(approach).canReachFinalPoint()` do plugin
não equivale à possibilidade de sono pelo Brain vanilla.

Próximo teste preregistrado: corrigir o monitor para aceitar diagnósticos
vazios quando `sleeping=true` e não encerrar antes do horizonte; executar
pares repetidos da **mesma fonte original**, um controle sem edição e uma
abertura idêntica, com UUID v10 verificado antes do boot, sem ordem IA e
horizonte fixo até tick14153. Métricas: primeiro tick de sono, fração de
frames dormindo, saúde, resgates e status de rota. Exigir fechamento limpo
e zero drops. Só depois considerar correção da geometria no mundo principal
com backup consistente; isso será melhoria ambiental, não da política.

Resultado do primeiro par repetido 16:10 UTC: monitor robusto em
`vila-viva-server/analysis/route_access_pair.py`, com testes dos dois erros
anteriores, usou o mesmo jar diagnóstico, fonte parada/tick13773/UUID v10,
0,5 CPU/3GiB e horizonte14153, sem ordem/portas. Controle intacto
`policy-route-pair-control-1558`: primeira observação13780 com duas rotas
`incomplete`, 20/20 observações `no_complete_path`, **0/20 keyframes
dormindo**, saúde mínima20, resgate0, episódio encerrado14165 com9.515
linhas/zero drops. Variante `policy-route-pair-opening-1602`: mesma
observação pré13780, precondições dos materiais confirmadas, `fill` removeu
dois blocos só na cópia; no13800 o sensor ainda marcava `no_complete_path`
e `path_null`, mas o primeiro keyframe dormindo foi13813 e houve
**18/20 keyframes dormindo** até14153, saúde mínima20, resgate0; episódio
encerrado14168 com10.030 linhas/zero drops. Ambos sem `pilot.json`.
Isso reproduz a melhora **ambiental local do Brain vanilla** observada no
par exploratório, não mostra IA superior. A consulta do plugin continua
um falso negativo operacional para o sono nesse cenário.

Consulta read-only ao mundo principal às16:10 UTC: os dois blocos ainda
existem, mas **nenhum dos 20 aldeões atuais** tem HOME `(-500,96,-368)`
ou está perto da casa. Portanto não há benefício atual demonstrado para
editar essa estrutura no principal; não fazê-lo por extrapolação do save
antigo. Próximo bloqueio do produto é selecionar uma tarefa observada nos
aldeões atuais e medir política vs Brain vanilla/regra simples, com entrada
de rota que não confunda `canReachFinalPoint` com sono possível. Exigir
fixture atual consistente/backup antes de qualquer mutação do mundo.

## Hipótese preregistrada: antecipar resgate de queda — 16:19 UTC

Análise somente leitura dos dias fechados17–20 do principal mostrou que
sono não é falha persistente: v4 passou uma noite0/546 frames dormindo
tentando HOME, mas a noite seguinte teve239/239; v2/v8 perderam HOME e
voltaram a dormir após realocação. Em contraste, houve34 resgates nos
dias16–20, 24 por `fell`, dez `stuck`. Exemplo de evento real v1 tick
10853100: frames anteriores tinham Brain `MoveToTargetSink` para alvo
`(-556,78,-378)` enquanto o corpo descia de y84 a y78, saúde1, antes do
resgate. A política Rust atual **não vê** alvo do Brain, declive, suporte
ou evento de resgate em tempo de decisão. Seu `return_home` não é
instrumento válido para evitar essa queda.

Antes de adicionar política/ação, medir capacidade preditiva no acervo
fechado sem tocar no principal: por frame/índice/tick, parsear WALK_TARGET
e `dy=target_y-current_y` (blocos), `on_ground`, velocidade vertical e
distância horizontal; rotular resgate `fell` do mesmo índice nos próximos
80 ticks. Dividir por dia, sem vazamento: desenho nos dias0–15,
validação retida16–20. Comparar taxa-base e regra simples
`dy<=-3 && MoveToTargetSink && on_ground` em precisão, recall e alertas
por dia/ser. Não afirmar que prever resgate evita resgate. Se o sinal não
for discriminativo ou o aviso chegar tarde demais, rejeitar a hipótese.
Se houver sinal, próximo contrato opt-in: observação de alvo/terreno Paper
fresca e identidade/tempo, ação segura `nudge`/`walk_to`, resultado com
resgate e saúde correlacionados, e baseline vanilla/regra simples em cópia
consistente do mundo atual. `nudge` atual ainda não produz os campos de
identidade/tempo para o ingestor Rust, portanto esse elo é FAIL e requer
correção antes de qualquer `learn`. Nenhum peso aprovado será alterado.

Resultado 16:24 UTC, análise frame a frame dos dias fechados0–20: 93
eventos `fell`, 504.020 frames vivos. Regra preregistrada alertou 3.262
vezes em384.020 frames de desenho (precisão0,77%, recall9,06% dos frames
positivos; 15/69 eventos tiveram alerta nos80 ticks anteriores). Na janela
retida16–20, 725 alertas em120.000 frames: 11 verdadeiros, 714 falsos,
precisão1,52%, recall11,46% dos frames positivos, 7/24 eventos com
alerta (em geral61 ticks antes). **Rejeitar essa regra como controlador:**
alerta frequente e cobertura insuficiente. `target_y-current_y` isolado
não resolve risco de queda; nenhuma intervenção foi aplicada.

Próximo ensaio offline, sem alterar produção: usar dias0–15 para ajustar
um pequeno escore experimental com alvo do Brain, posição/velocidade,
distância ao alvo e altura relativa ao limite de resgate; dias16–20 são
validação/model selection, dias21–25 ficam **intocados** para teste futuro
após fechamento. Reportar AUPRC, precisão/recall por evento e alertas por
ser-dia contra taxa-base e regras simples no mesmo orçamento de alertas.
Se não superar controles em dados retidos, não construir ação de segurança.
Não usar o modelo shadow aprovado como controlador nem promover este escore.

Primeiro escore offline, concluído nesta rodada de 16:13–16:27 UTC (arquivo experimental
`~/.local/state/vila-viva/analysis/fall-warning-v1-20260927.json`):
logística pequena treinada nos dias0–15 teve AP0,339 em16–20 contra
prevalência0,0008; limiar de desenho gerou921 alertas e marcou24/24
eventos. **Falha de acionabilidade descoberta:** ao exigir que o aviso
ocorra `on_ground` e corpo ao menos um bloco acima de y80, só8/24
eventos foram marcados, e a precisão dos alertas elegíveis caiu a2,4%.
O resultado24/24 explorava frames já dentro da queda/baixo do nível de
resgate; não é prevenção. Regra física mais específica
`target_y<80 && MoveToTargetSink && on_ground` marcou15/24 eventos com
89 alertas/precisão33,7% (antes de filtrar altura do corpo). Ambos são
apenas preditores offline; não despachar ação.

Correção preregistrada: restringir **treino, calibração e avaliação** a
frames em que o aldeão está no chão e y>=82 (margem de2 blocos acima do
resgate). Manter rótulo futuro80 ticks e divisão temporal0–15/16–20;
usar o volume de alertas da regra física no desenho para fixar limiar
do escore. Comparar evento, precisão e alertas por ser-dia com essa regra
no mesmo conjunto elegível. Os dias21–25 seguem intocados como holdout
futuro. Se desempenho elegível não superar a regra física com aviso útil,
rejeitar a política de queda antes de criar controle.

Resultado da correção na mesma rodada, arquivo experimental
`~/.local/state/vila-viva/analysis/fall-warning-v2-actionable-20260927.json`:
nos frames elegíveis `on_ground && y>=82`, a prevalência da validação
16–20 era0,000128 e AP do escore0,153. A regra física
`target_y<80 && MoveToTargetSink && on_ground` produziu52 alertas,
precisão19,23% e alcançou6/24 eventos. O escore treinado com orçamento
de alertas fixado no desenho produziu47 alertas, precisão19,15% e
também6/24 eventos. Na janela de desenho, regra e modelo alcançaram
14/69 e15/69, respectivamente; a diferença **não se repetiu** na
validação. A regra frouxa `dy<=-3` gerou721 alertas/7 de24 eventos na
validação elegível. Portanto o escore não supera a regra física para uma
intervenção plausível. **Rejeitar esta hipótese/controlador**, sem acionar
`nudge`, mexer em pesos aprovados ou espiar dias21–25. Sinal de terreno
local/frescor do Brain ausente é o primeiro bloqueio se a tarefa de
segurança for retomada; não assumir que um escore maior resolverá.

Checagem retida preregistrada às 16:35 UTC: os dias21–25 agora estão
fechados. Executar **uma vez** o escore v2 já congelado em
`~/.local/state/vila-viva/analysis/fall-warning-v2-actionable-20260927.json`
nessa janela, sem ajustar pesos, limiar, elegibilidade ou orçamento. Excluir
os últimos80 ticks do dia25 porque o dia26 segue aberto e o rótulo futuro
seria censurado. Reportar AP, prevalência, alertas, precisão e eventos
antecipados versus a regra física existente. Esta é auditoria de
generalização de uma hipótese já rejeitada; mesmo ganho eventual não
autoriza comando Paper. Não selecionar novos parâmetros a partir dessa
janela e preservar os artefatos.

Resultado retido 16:37 UTC, sem reajuste: 16 resgates `fell` em21–25;
escore congelado alertou49 vezes, precisão16,33%, antecipou5/16;
regra física alertou67 vezes, precisão16,42%, antecipou7/16.
AP do escore0,248 contra prevalência0,000111, mas o objetivo operacional
por evento foi inferior à regra. Resultado em
`~/.local/state/vila-viva/analysis/fall-warning-v2-holdout-20260927.json`.
Confirma a rejeição; não usar essa cabeça como controlador.

## Preregistro: contrato de resultado para destravamento — 16:37 UTC

No episódio atual há52 resgates `stuck` em26 dias fechados, nove nos cinco
últimos. O `Rescue` vanilla já identifica `cant_reach` prolongado e
movimento <1,5 bloco, mas só teleporta; a ação opt-in `nudge` admite pulso
horizontal em corredor seguro e registra deslocamento, porém **não**
inclui episódio/UUID/índice do ser/tick de aplicação no par ação/resultado.
Sem isso, a ingestão de feedback não pode atribuir efeito ao ser/tarefa.

Primeiro ajuste limitado ao plugin opt-in: acrescentar identidade/tempo
às mensagens `pilot_action`, `pilot_deferred`, `pilot_reject` e
`pilot_outcome` de `nudge`, preservando os gates de segurança e física.
Compilar; validar que um comando sintético em cópia Paper, com UUID/episódio
e prazo, gera par correlacionável ou rejeição explícita. **Não** afirmar
aprendizado: Rust ainda só tem `claim_bed`/`return_home`, sem observação
de terreno nem recompensa de `unstick`. Próximo experimento específico:
um aldeão atual com `cant_reach` e posição estável, observar candidatos
cardeais seguros no Paper, comparar nenhum comando, regra de direção fixa
e política com estado congelado em cópias pareadas; medir resgate `stuck`,
deslocamento dirigido, saúde e custo. Fazer backup consistente do mundo
atual antes da cópia; nenhuma mudança no principal até gates e baseline.

## Preregistro: percepção multiobjetivo de um UUID — 16:46 UTC

Após o pedido do dono por revisão macro/micro, a falha seguinte é a cegueira
da política opt-in: ela só recebe noite/HOME/camas, enquanto o gravador
já lê WALK_TARGET/cant_reach, corpo, saúde e inventário; fome nem está
ativa no episódio A. Construir **em código Java opt-in**, sem deploy
principal, `vila-survival-observation/v1` em arquivo atômico separado:
identidade (episódio, UUID, v, tick Paper), relógio, posição/saúde/sono/
chão/profissão, comida/saciedade nullable, memórias HOME/JOB_SITE/
WALK_TARGET/CANT_REACH com idade, estagnação observada em ticks e quatro
ações cardeais de pulso, com segurança do **mesmo** validador do atuador,
materiais de pés/cabeça/apoio e razão de rejeição. Limitar a um UUID,
cadência20 ticks, no máximo quatro direções/um bloco, sem pathfinding
extra. Teste: build offline e payload Paper real com UUID/tick e valores
válidos; nenhum comando/treino até demonstrar saída útil. O Rust canônico
deve consumir contrato versionado depois, sem reutilizar a cabeça shadow.

Fome mais apertada e casas profissionais são **cenários de avaliação**,
não variáveis ocultas em produção: hoje 20 aldeões têm só três postos e
comida inventariada escassa; habilitar fome já no principal poderia matar
indivíduos de saúde1. Primeira variante de fome será cópia isolada com
configuração versionada e métricas de comida/dano/morte, seguida de
política que de fato possa obter/comer alimento. Novos postos/população
também requerem ensaio sequencial porque 14 postos prévios custaram TPS
sem criar profissões. Não alterar tudo no mesmo braço; separar efeitos.

Resultado 17:05 UTC: Java opt-in `vila-survival-observation/v1` compilou;
Paper real em cópia `survival-obs-1650` publicou quatro amostras consecutivas
13780–13840, 3/4 saídas seguras e estagnação0→60, episódio encerrado
exit0/zero drops. Segundo ensaio sequencial na mesma cópia parada enviou
um único `nudge` norte seq1 após observação13860: ação13880 e resultado13920
com episódio/UUID/v/tick correlacionados, exit0/zero drops e saúde20→20.
Entretanto o deslocamento total foi0,230 bloco e o progresso dirigido
**-0,106 bloco**. Isto valida somente percepção, aceitação e correlação;
refuta eficácia daquele pulso. V10 não estava preso, portanto resgate
evitado não foi medido. Nenhuma recompensa/atualização Rust ou mudança no
principal. Não treinar a política com `safe=true` como sucesso.

**Preregistro do próximo teste de atuador:** um único passo local pelo
Pathfinder do plugin Java já existente (`walk_to`), para destino a um
bloco numa direção classificada segura em observação Paper fresca.
Usar cópias sequenciais da mesma fonte parada/UUID/jar: controle sem ordem
e variante com ordem atômica; horizonte fixo, comparar caminho aceito,
progresso dirigido, estagnação, saúde/sono/resgates e integridade dos
episódios. Segurança deve ser revalidada no tick de aplicação, não confiar
apenas no snapshot. Se o caminho for rejeitado, declarar motor indisponível
nesse local; se aceito mas não melhorar controle, não usar como política.
Somente depois escolher caso atual com `cant_reach` e testar objetivo final.

Resultado do par 17:20 UTC: controle `survival-step-control-1708`, mesma
fonte/v10/tick13800/jar, terminou14040 exit0/zero drops, progresso norte
-0,047 bloco, saúde20/sem sono/resgate. Variante
`survival-step-action-1714`, mesma observação inicial, enviou `walk_to`
para exatamente 1 bloco ao norte; Paper aceitou rota de2 pontos no13820,
mas o piloto marcou `arrived_tick=13840` ao ficar0,98 bloco do destino,
sem refresh, e progresso final até14020 só0,024 bloco rumo ao alvo;
deslocamento norte do keyframe0,053 bloco. Sem dano/sono/resgate, exit0/
zero drops. Isso **não** valida um motor útil: tolerância fixa de1 bloco
invalida passos de1 bloco e a diferença versus controle é mínima.

**Novo preregistro de correção do atuador, antes do código:** permitir
`arrival_radius` opcional no comando opt-in `walk_to`, faixa fechada
[0,20;1,00] bloco, padrão1,00 para compatibilidade; guardar no estado
pendente, usar em detecção de chegada, refresh e resultado. O despacho Rust
de cama continua sem campo e preserva semântica. Repetir somente a variante
do passo local a partir da mesma fonte parada/UUID/tick, agora raio0,25;
controle anterior é referência exploratória. Critério: caminho aceito,
progresso dirigido substancial além de0,1 bloco sem dano ou resgate, e
integridade exit0/zero drops. Se continuar ineficaz, não tratar caminho
aceito como controle; investigar interferência do Brain e mudar atuador.

Resultado 17:30 UTC: variante `survival-step-radius-1722`, mesma fonte/v10/
tick13800, jar opt-in com `arrival_radius=0,25`, comando aplicado13820 e
rota de2 pontos aceita. Foram cinco refreshes, zero falhas de pathfinder;
no tick14020 a distância ao alvo cresceu1,000→1,326 blocos, progresso
dirigido norte no keyframe final **-0,347 bloco**. Saúde20, sem sono ou
resgate, exit0/zero drops. O raio conserta o falso positivo de chegada,
**mas não** dá controle físico útil; a hipótese de motor local por refresh
é rejeitada nesse fixture. Rust `unstick` não deve usar `path_accepted` ou
`refresh_success` como recompensa. Próximo: observar conflito com
`WALK_TARGET`/navegação do Brain e testar mecanismo delimitado que mantenha
ação sem dano, em caso realmente preso, contra vanilla e regra fixa.

## Preregistro: relógio e tipo de `CANT_REACH` — 17:39 UTC

Auditoria read-only do episódio Paper `survival-step-radius-1722` encontrou
`brain_mem_cant_reach_walk_target_since="4939"` nos keyframes de v10,
enquanto `vila-survival-observation/v1` publicava `cant_reach_since_tick:null`
e `cant_reach_age_ticks:null`. Causa no produtor: `BrainReader.describe()`
serializa memórias como texto e `SurvivalObservation` só aceita `Number`.
No mundo principal, RCON `time query gametime`≈11.329.457 e `tick` do
snapshot≈11.329.459; os valores de `CANT_REACH` são números nesse relógio,
mas não devemos pressupor que `world.getFullTime()` é sempre igual a
`world.getGameTime()` quando comandos de tempo são usados.

Corrigir somente o plugin opt-in em contrato **v2**: expor `brain_game_time`
via `World.getGameTime()`, converter memória decimal textual ou `Number` para
inteiro não negativo finito, rejeitar formato inválido/futuro; publicar
`cant_reach_since_game_tick` e `cant_reach_age_ticks` em unidades de game
ticks. `tick` permanece `getFullTime()` para frescor do comando. Atualizar
monitores de novas cópias para exigirem v2; v1 histórico permanece artefato
da falha. Compilar e, em cópia Paper parada/UUID v10, exigir sinal numérico
e idade coerente, sem ordem, exit0/zero drops. **Não** concluir que o ser
está preso só pela presença da memória: testar persistência, movimento,
alvo e benefício de ação em outro passo. Principal não muda.

## Ensaio preregistrado: evidência de estagnação antes de `return_home` — 14:57 UTC

O par `policy-feedback-home-1434`/`policy-feedback-vanilla-1443` refutou ganho
de sono sustentado para v2: o vanilla dormiu sozinho, e a ordem antecipada
interrompeu o sono. Hipótese limitada: exigir pelo menos 160 ticks **contíguos**
de posição quase estacionária (deslocamento <=0,25 bloco a cada amostra),
acordado, à noite, com HOME, impede intervenção nesse caso. Isso é um indício
de estagnação, **não** prova de falha vanilla nem política melhor.

Mudança pretendida só no produtor Java opt-in e runtime Rust espelhado no
Universe: `vila-bed-candidates/v3` publica `stationary_home_ticks` em unidade
de tick Paper; zera ao mudar UUID/posição >0,25 bloco, dormir, perder HOME,
sair da noite, retroceder tick ou perder continuidade de amostras. Rust
requer o campo e só propõe `return_home` após >=160; `claim_bed` mantém os
gates existentes. v2 antigo deve falhar por schema, v3 com 0/159 deve dar
`decision:null`, v3 com160 e HOME alcançável pode propor ação. Não promover
estado experimental nem implantar no principal.

Teste Paper planejado: cópia nova da fonte parada `policy-feedback-1425`
(tick13773), mesmo UUID v2, 0,5 CPU/3GiB, sem portas, jar novo, nenhum
despacho. Amostrar observações v3 e decisão Rust ao longo de até ~400 ticks;
comparar quando ele começa a se mover/dormir, verificar que a política não
propõe `return_home` antes da janela. Encerrar graciosamente com zero drops.
Se o ser continuar parado por160 ticks e a política propuser, não despachar
nesta rodada: o gate do produtor estará validado, mas eficácia/superioridade
seguirão NOT MEASURED. Um experimento por vez.

Resultado parcial 15:07 UTC: `policy-stall-gate-1458` da mesma fonte parou
limpo no tick14157, 9.402 linhas/zero drops, 19 observações v3, nenhum
comando/erro/decisão. O v2 dormiu por conta própria já no tick13800 e o
contador ficou0, validando esquema/sono mas **não** o ramo >=160. A variação
do instante de sono entre cópias reforça que um par não prova causalidade.

Segundo fixture preregistrado, sem sobrepor ensaios: keyframes vanilla da
mesma fonte mostram o v10 adulto UUID
`1a8bdbe6-c062-4ff6-b85c-2dccf9c4ce80` acordado, no chão, com HOME
`(-500,96,-368)` e quase parado de13773 a14133 (360 ticks); v0/v4 também
pararam, mas são bebês. Copiar novamente a fonte `policy-feedback-1425`
parada, selecionar v10, novo jar v3/0,5 CPU/3GiB/sem portas, nenhum
despacho, amostrar `stationary_home_ticks` e Rust `decide_paper` até14144.
Esperado: contador cresce de0 a>=160 se condições persistirem; antes de160
decisão nula; a partir de160, decisão só se HOME tiver rota completa.
Se dormir/mover/resetar antes, registrar refutação e encerrar. Não chamar
parado = falha vanilla sem rota/segurança verificadas. Parar cópia no limite.

Resultado 15:15 UTC: `policy-stall-adult-1508` terminou limpo no tick14159,
9.486 linhas/zero drops, sem ordem nem evento de piloto. De 19 amostras v3
do v10, contador subiu0→100, zerou no13900 após movimento, tornou a subir
e chegou240 no14140; cinco amostras >=160. Todas as 19 tinham
`home_route_status=no_complete_path`, portanto Rust retornou `decision:null`
nas 19, sem erro. Isso valida **produtor Paper do contador**, reset por
movimento e composição com gate de rota; não prova o ramo de decisão
positiva com >=160 + rota alcançável em Paper. A primeira obstrução causal
para esse adulto é navegabilidade da HOME, não falta de espera. Não repetir
`walk_to` para rota que o próprio Paper considera incompleta. Próximo passo:
diagnosticar geometria/POI/altura e caminho até HOME em cópia, sem alterar
produção, depois definir uma tarefa que permita baseline pareado e regra
simples. Estado aprendido da ação v2 anterior permanece experimental.

## Contrato v2 — 27/09/2026 13:12 UTC

O primeiro bloqueio causal da auditoria foi tratado **offline**, ainda sem
controle do Paper: `vila-goal-policy/v2` adiciona `decide_paper` que aceita o
envelope `vila-bed-candidates/v1` inteiro. Para aldeão com HOME e rota
`reachable`, converte `home_approach` validado em proposta distinta
`goal=return_home`, `action=walk_to`, `candidate_id=home_route`; para sem HOME
mantém seleção de cama. Rejeita esquema/contagem inconsistentes. `learn` agora
separa `decision_tick` de `action_tick` real, com atraso permitido até40 ticks,
pois o comando é lido pelo Paper depois da observação. O esquema de estado
mudou de v1 para v2; estado anterior não é promovido automaticamente.

`cargo test --offline --locked` passou sete testes nas duas árvores; build e
fmt passaram. Fontes servidor/Universe byte-idênticas, SHA256
`6e842d8baabfe2240e62fa976fb72e3e83136b597c6c60664586757805f80be7`.
Com estado recém-inicializado, a CLI v2 retornou `return_home`/`walk_to`
para `(-532.5,86,-412.5)` em dois snapshots Paper reais (ticks16040 e16860),
ambos com `policy_updates=0`. Nenhum comando foi enviado ao Paper por essa
política, nenhum feedback/atualização real ocorreu. O `walk_to` manual anterior
para o mesmo destino foi aceito mas não avançou; não declarar ganho. O próximo
gate é ponte com identidade/expiração e resultado Paper completo, seguida de
eficácia mecânica e baseline pareado.

## Ensaio preregistrado: estado de sono no produtor — 27/09/2026 13:28 UTC

Mudança proposta: `BedCandidates` passa a `vila-bed-candidates/v2` e acrescenta
`observation.sleeping` diretamente de `selected.isSleeping()`; a política
`decide_paper` requer v2 e não propõe ação se `sleeping=true`. Sem campo ou
esquema antigo, a deserialização/validação deve falhar fechada. Não mexer no
mundo principal. Teste: copiar `night-route-base` parada para destino novo,
plugin jar novo, observação a cada20 ticks, Paper isolado 0,5 CPU/3GiB sem
portas; obter JSON v2 e conferir tipo booleano/episódio/UUID/tick, depois
rodar `init`+`decide_paper` nessa observação. Para `sleeping=true`, exigir
`decision:null`; para acordado com HOME `reachable`, exigir `return_home`
consistente. Não emitir `pilot.json`, não medir sucesso de navegação neste
ensaio. Parar a cópia ao atingir uma observação v2 válida ou limite de tempo;
preservar episódios e registrar se o estado necessário não ocorrer.

Resultado 13:35 UTC: implementação Java `vila-bed-candidates/v2` acrescentou
`observation.sleeping` booleano de `selected.isSleeping()`. O Rust v2 exige o
campo e retorna `null` quando verdadeiro; observação histórica v1 rejeitada
com `missing field sleeping`. `shadowJar` offline passou, jar
`e6594f31ab919fe86cfe0bed5ea359254a8b610d707936e6efbfc2b1baf248ee`.
Oito testes Rust passaram nas duas árvores; fontes byte-idênticas SHA256
`77d946ba65bb9544a7fff5f230c4ac76f8a562cabfad7b6f3b73be6f7ba0f959`.

A cópia nova `sleep-gate-1330` veio de `night-route-base` parada, observação
20 ticks, container `vila-ai-sleep-gate-1330` limitado a 0,5 CPU/3 GiB e sem
porta. Episódio `A-6784215452709343147-20260927-133420`: no tick14960,
`sleeping=false`, `has_home=true`, `is_night=true`, `home_route_status=no_complete_path`
e política devolveu `null`; no tick15000 a rota era `reachable` e o Rust
propôs `return_home`/`walk_to` para `(-532.5,86,-412.5)`, atualizações0.
Nenhum `pilot.json` foi escrito, logo nenhuma ação/resultado/aprendizado real.
Container terminou com exit0, `episode_end.json` presente; fonte e dados
preservados. Não ocorreu amostra Paper com `sleeping=true`, portanto esse ramo
tem apenas teste Rust sintético. O principal não recebeu plugin/política novos.

## Ensaio preregistrado: despacho Rust de uma ação — 27/09/2026 13:50 UTC

Fonte: `sleep-gate-1330` parada com exit0; copiar para novo destino, preservar
fonte e episódios, mesma config opt-in (`candidate_every_ticks=20`) e jar
`e6594f31...`. Novo `vila-policy-cpu` SHA256 de fonte `860fb6b7...` tem modo
`--dispatch-once OBS LIVE EPISODE STATE DECISION PILOT SEQ`. Ele exige arquivos
Paper com mtime<=10s, episódio opt-in igual, tick live no intervalo
`[observation.tick, observation.tick+20]`, UUID vivo/acordado em ambas as
leituras, estado de política válido e nenhum despacho prévio no destino. Grava
primeiro registro de decisão e depois `pilot.json` por hard link de arquivo
temporário sincronizado, sem sobrescrever. Teste unitário de escrita/identidade
e recusa de repetição passou; execução real ainda pendente.

Rodar Paper em cópia isolada 0,5 CPU/3 GiB/sem portas, sem teleporte e sem
alterar tempo. Esperar no máximo oito minutos após boot por observação v2
`sleeping=false`, HOME `reachable`, live do mesmo episódio/UUID e tick fresco.
Executar um único despacho seq1; comprovar `pilot_action` ou `pilot_reject`
com o mesmo seq, episódio, UUID, tick aplicado, e `pilot_outcome` até200 ticks
se aceito. Medir progresso dirigido/chegada, saúde, sono/resgate na janela
disponível. Parar cópia graciosamente após resultado ou limite; não invocar
`learn` nem declarar vantagem. Se não houver estado fresco, nenhum comando é
emitido e o gate fica NOT MEASURED. Principal não recebe binário/jar novos.

Resultado 14:00 UTC: cópia `policy-dispatch-1351` iniciada da fonte parada
`sleep-gate-1330`, 0,5 CPU/3 GiB/sem portas. Episódio
`A-6784215452709343147-20260927-135557`: `--dispatch-once` consumiu
observação v2 tick15060 e live15069, registrou decisão Rust `return_home`
para v9, `policy_updates=0`, e publicou seq1 com expiração15109. Paper
registrou `pilot_action` no15080 com mesmo seq, alvo e UUID, caminho aceito
de12 pontos, distância inicial7,6037 blocos. `pilot_outcome` no15280:
`reason=deadline`, distância restante7,6052, progresso **-0,0015 bloco**,
sem chegada, saúde5→5, quatro refreshes e uma falha. Frames seguintes até
15329 não mostraram sono nem resgate; HOME ainda era a memória vanilla.
Episódio encerrado no15353 com `lines_dropped=0`, container exit0 e cópia
desligada. A ponte decisão→ordem→resultado passou; deslocamento/objetivo
falharam. Nenhum `learn` foi executado e não há evidência de vantagem.

O próximo gate é derivar `Outcome` apenas dos eventos de um episódio fechado,
com junção por seq/UUID/ticks e ausência de sono/resgate verificável, salvar
estado após `learn` e rejeitar replay. Antes de outro ensaio, investigar duas
mortes independentes no principal em 27/09 às13:49 e13:59 UTC, por arbusto
de bagas e fogo; preservar dados e proteger os perigos. Principal continua
vanilla e shadow apenas preditivo.

## Contrato de feedback implementado, ainda sem execução Paper — 14:17 UTC

O episódio `policy-dispatch-1351` não prova ausência de sono a cada tick:
keyframes são amostrados. A nova fonte Java opt-in encerra `walk_to` quando
observa o ser dormindo em um tick, rejeita comando se ele dormiu antes da
aplicação e acrescenta `episode`, UUID, índice `v`, `action_tick` e
`slept_during` ao resultado. `shadowJar` offline passou; esse jar ainda não
rodou em Paper. O antigo resultado Paper falha fechado no ingestor por falta
dessas entradas, preservando `updates=0`.

`ingest_feedback.py` no servidor e Universe são byte-idênticos. Ele exige
episódio fechado/sem drops, eventos gzip, ação aceita única e correlação exata
de seq, episódio, UUID, `v`, alvo e ticks. Só então deriva progresso/dano,
sono observado pelo Java e resgate do evento do mesmo `v`, chama `learn` Rust
e cria `state.json`/`feedback.json` sem sobrescrever. Três testes em cada
árvore passaram, incluindo atualização sintética do Rust para `updates=1`,
replay recusado, ausência de sono e drops rejeitados. Isto demonstra o
contrato offline, **não** aprendizagem com Paper real nem vantagem sobre
vanilla. Próximo ensaio requer novo episódio Paper com o jar enriquecido e
ação plausível, com pré-registro de alvo, resultado e baseline.

## Ensaio preregistrado: feedback Paper fechado em fixture sem HOME — 14:25 UTC

Fonte parada `~/.local/share/vila-viva-ai-eval-20260927/pilot`, aldeão UUID
`00150f0b-f8fe-42c3-b656-d74910557447`, vivo/saúde20/sem HOME no último
episódio, com deslocamento anterior registrado nesse cenário. Criar cópia
nova `policy-feedback-1425`, preservar fonte/episódios, remover apenas comando
e observação **stale da cópia**, instalar jar opt-in `89e2d5f6...`, intervalo
de candidatos20 ticks, container 0,5 CPU/3GiB sem portas, sem mexer no
principal. Não alterar relógio/posição do ser. Esperar até oito minutos após
boot por envelope v2 fresco com `sleeping=false`, `has_home=false`, pelo menos
um candidato de cama válido e live do mesmo UUID/episódio/tick. Só então
despachar seq1 via Rust v2; se não houver candidato, nenhum comando e gate
NOT MEASURED. Parar cópia ao obter resultado ou limite, exigir exit0,
`episode_end.lines_dropped=0`, eventos gzip íntegros.

Gate de contrato: `pilot_action` aceito com episódio/UUID/v/tick/alvo/seq
iguais ao despacho e `pilot_outcome` com `slept_during` booleano; ingestor
produz `feedback.json` e `state.json` com `updates=1` somente do episódio
Paper fechado, replay rejeitado. Registrar progresso, chegada, casa, sono,
saúde, resgate e custo. Se a ordem for rejeitada ou o aldeão ficar imóvel,
isso não é ganho. Não mover esse estado para o principal nem concluir
superioridade; vanilla e regra simples continuam baselines pendentes.

Resultado 14:33 UTC: cópia `policy-feedback-1425` iniciou com jar
`89e2d5f6...`, observações v2 do UUID correto em noite e sem HOME.
Nos ticks13460 e13760, `candidate_count=0`; no final, havia quatro blocos de
cama próximos, dois HEADs e **ambos reivindicados** por outros aldeões,
`beds_checked=0`, `path_queries=0`. Nenhum `pilot.json`/decisão/feedback
foi criado. A cópia parou voluntariamente no13773, exit0, 9.315 linhas,
zero drops. Gate decisão/feedback Paper NOT MEASURED neste cenário; não
modificar pesos nem chamar ausência de candidato de falha de navegação.

## Ensaio preregistrado: retorno HOME perto da cama — 14:34 UTC

Fonte é a cópia acima, agora parada/fechada. Criar `policy-feedback-home-1434`
sem apagar origem/episódios, selecionar UUID
`4e388df0-5990-4926-a79f-d2f42c4c925b` (v2 no último live, saúde20,
acordado, HOME `(-527,88,-371)`, distância2,5 blocos), limpar apenas os
arquivos de controle stale na cópia, usar mesmo jar `89e2d5f6...`, observar
a cada20 ticks, 0,5 CPU/3GiB/sem portas. Sem alterar mundo, relógio ou
posição. Até oito minutos após boot, exigir observação Paper v2 fresca,
`sleeping=false`, `has_home=true`, HOME `reachable`, `home_approach` válido,
live/episódio/UUID/ticks correspondentes. Somente então Rust despacha seq1.
Se dormir antes da aplicação, Java deve rejeitar; se aplicar, exigir ação
e resultado enriquecidos. Parar graciosamente após resultado ou prazo,
checar zero drops e rodar ingestor **apenas** em episódio fechado. Gate de
contrato: `updates=1` em estado experimental e replay rejeitado, com
recompensa derivada de sono/saúde/resgate reais. Medir chegada/sono/dano e
distância, sem concluir vantagem sobre vanilla por ser uma posição próxima.

Resultado 14:42 UTC: cópia `policy-feedback-home-1434` publicou HOME
`reachable` para v2 no tick13800, aproximação `(-527.5,88,-371.5)` a
3,0948 blocos. Rust despachou seq1 com episódio/UUID/tick frescos e estado
`updates=0`; Paper aceitou `walk_to` no13820 com caminho12 pontos.
`pilot_outcome` no13887 trouxe `reason=sleeping`, `slept_during=true`,
progresso1,5767 blocos (razão0,5095), saúde20→20, zero resgates e HOME
preservada. `arrived=false` no critério <=1 bloco; não transformar sono em
chegada ao waypoint. O episódio fechou no14144, 8.675 linhas, zero drops,
exit0. Ingestor correlacionou evento e decisão e chamou Rust `learn`:
estado experimental `updates=1`, recompensa5,1877, pesos mudaram;
`feedback.json`/`state.json` persistidos. Segunda invocação recusou replay
com arquivos inalterados. Isto é **primeiro ciclo real de feedback Paper**,
mas ainda não prova que o controle causou sono, que generaliza, ou que supera
vanilla. Modelo aprovado e principal não mudaram.

## Baseline preregistrado para o mesmo save — 14:43 UTC

Criar uma nova cópia da fonte parada `policy-feedback-1425` no tick13773,
igual à fonte do braço de política acima. Mesmo jar `89e2d5f6...`, 0,5 CPU/
3GiB/sem portas e mesmo UUID v2, mas `ai_pilot.enabled=false` e nenhuma
ordem; não mudar posição/relógio. Encerrar após tick>=14144, verificar
episódio fechado/zero drops. Comparar do tick13773 ao14144 a primeira
observação de sono em keyframes (resolução20 ticks), HOME, saúde, resgates e
posição do mesmo UUID. A política dormiu por evento no13887; baseline não
possui evento por tick, logo comparar apenas dentro da resolução dos frames.
Uma cópia por braço é exploratória: diferenças estocásticas/boot impedem
conclusão causal. Se vanilla também dormir no mesmo horizonte, não alegar
vantagem; se não dormir, repetir pares controlados antes de alegar ganho.

Resultado 14:52 UTC: a primeira tentativa do container sem rede falhou antes
do Paper iniciar (o entrypoint quis resolver o download); a mesma cópia foi
iniciada com rede bridge padrão, **nenhuma porta publicada** e os limites
previstos. Não houve tick de jogo na tentativa falha. Episódio vanilla
`A-6784215452709343147-20260927-145041` começou no mesmo tick13773,
UUID v2 e posição `(-526.9,88,-368.5)`, terminou limpo no14195, 9.116 linhas,
zero drops, sem `pilot_action`/`pilot_outcome` nem resgate do v2.
Até o horizonte pareado14144, primeiro frame dormindo: política13893,
vanilla13913 (diferença20 ticks, resolução dos frames20). Mas o braço da
política acordou nos frames13913–13993 e voltou a dormir no14013; foram
**8/20 frames dormindo**, contra **12/20** vanilla, que permaneceu dormindo
de13913 a14133. Ambos conservaram HOME/saúde20 e zero resgates. A política
não venceu nesta tarefa; o pequeno adiantamento do primeiro sono é
insuficiente e a continuidade foi pior. O `learn` experimental recompensou
sono observado, porém a comparação mostra que isso também ocorreria sob
vanilla; não promover esse estado. Próximo bloqueio causal: só agir quando
houver evidência de que a rotina vanilla de retorno falhou, e medir sono
sustentado como resultado, antes de novo ensaio de controle.

## Auditoria de contratos — 27/09/2026 13:00 UTC

O mapa de ponta a ponta e as perguntas obrigatórias de cada rodada estão em
`vila-viva-server/docs/AI-FLOW-CONTRACT-AUDIT.md`. O diagnóstico atual corrige
o handoff antigo: não falta apenas seleção de objetivo. Não existe invocador
que entregue observações Paper ao `vila-policy-cpu`, conversor que publique a
decisão em `pilot.json`, nem consumidor que correlacione `pilot_outcome` a uma
decisão e salve o estado de `learn`. O evento Java não traz todos os campos
exigidos pelo `Outcome` Rust (`candidate_id`, sono, resgate, razão de progresso
e identidade completa). Cinco testes unitários provam apenas lógica sintética.

O produtor `BedCandidates` v1 é opt-in. Nos snapshots Paper reais
`home-approach-1219` tick16040 e `walk-refresh20-1243` tick16860,
`has_home=true`, zero candidatos e `home_approach` válido. O binário Rust
compilado offline respondeu `decision:null` em ambos; `decide()` retorna isso
explicitamente para aldeão com casa e nem recebe `home_approach`. Portanto o
fixture v6 de retorno à HOME não exercita a política de seleção de cama. A
política não vê geometria local, suporte, ameaça ou memória de HOME. O shadow
aprovado recebe 23 features de telemetria apenas para previsão e não está
conectado ao controle. O principal continua vanilla.

Gate imediato: fixar uma tarefa e um contrato coerentes com ela. Para retorno
à HOME, definir entradas/ações/resultados correspondentes, verificar uma
decisão não nula em snapshot real, correlacionar comando e feedback e obter
`updates=1` por episódio Paper isolado antes de comparar com vanilla e regra
simples. Sem ação mecanicamente eficaz, treinar pesos não tem interpretação.
Qualquer edição deste runtime deve continuar byte-idêntica à cópia do servidor;
na auditoria, SHA256 de `src/main.rs` em ambas as árvores foi
`9693ee0f2a280913f8775c4a58d4040a34d0f29e204115ade099dfb8f188e318`.

## Acceptance and budget (before the next run)

- Tarefa: aldeão sem casa escolhe uma cama livre e alcança uma entrada segura;
  sucesso final requer memória de casa e sono numa janela noturna, saúde
  preservada e resgates explicitados.
- Unidade: um aldeão opt-in, cópias isoladas do mundo, mesma versão de plugin,
  tempo/posição de partida pareados ou repetições independentes. Antes de
  declarar vantagem, comparar taxa de casa/sono e tempo com vanilla e um
  controlador trivial de cama mais próxima; distância até waypoint é gate
  mecânico, não sucesso da tarefa.
- Orçamento inicial: 0,5 CPU lógica no Paper isolado, sem porta exposta;
  executar ensaios em série e parar ao alcançar o horizonte. Sem GPU para o
  novo runtime CPU do Minecraft; os gates GPU legados de taxi permanecem
  independentes e não são tocados.
- Limite da próxima etapa: contrato e teste unitário de decisão/atualização
  de recompensa; isso não demonstra habilidade aprendida.

## Reproduction identity

- Servidor contínuo: Paper 1.21.4-232, plugin `0.1.1-notebook`, episódio
  `A-6784215452709343147-20260927-063347`, modelo preditivo aprovado
  `shadow-20260926-220823`. Vanilla controla; o modelo não deve ser promovido.
- Ponte isolada: `ControlPilot` Java `0.2.0-pilot`, comando `walk_to` restrito a
  UUID/episódio/tempo/caminho completo. No episódio `...091254`, UUID
  `00150f0b-f8fe-42c3-b656-d74910557447`, seq2 tick 3820, caminho de dez
  pontos, chegada a <=1 bloco no tick 3920; sem casa/sono, vanilla retoma
  o movimento depois. Evidência detalhada em
  `vila-viva-server/docs/AI-PILOT-BASELINE-20260927.md` e gzip isolado em
  `~/.local/share/vila-viva-ai-eval-20260927/pilot/episodes/`.
- Branch Universe `organism/school-20260922`, HEAD `74d12e1`; há um documento
  shadow não rastreado que deve ser preservado. Novo runtime de política terá
  formato próprio, sem alterar `vila-shadow-cpu` ou seus pesos.

## Executed path and first discrepancy

Paper/Brain vanilla → `Recorder`/`live.json` → (ainda ausente) seleção de
objetivo por política → arquivo atômico `pilot.json` → `ControlPilot` →
`Pathfinder` → corpo/Brain vanilla → evento `pilot_outcome`.
O primeiro elo ausente é a seleção de objetivo por uma política com
recompensa/estado explícitos; atualmente um operador escolhe coordenadas.
`pilot_outcome` mede percurso, casa, profissão e saúde; sono e ocupação da cama
ainda precisam de contrato e janela adequados.

Hipótese: uma política contextual limitada a candidatos de cama seguros,
atualizada por resultado de chegada/casa/saúde, pode aprender a preferir
alvos úteis. Refutação: ações aceitas que não obtêm casa/sono ou desempenho
não superior ao vanilla em partidas retidas. Waypoint aceito não valida a
hipótese. A saída de movimento do shadow não será usada como comando.

## Gate verdicts

| Gate | Estado | Evidência |
|---|---|---|
| Ponte de ação | PASS limitado | Um waypoint alcançado em Paper isolado; sem casa |
| Política e atualização | PASS sintético | `vila-policy-cpu/v1` seleciona e atualiza estado em cinco testes; sem feedback Paper ainda |
| Tarefa de abrigo | FAIL neste ensaio | `mem_home` permaneceu ausente |
| Vantagem vs vanilla | NOT MEASURED | Primeiro par rejeitou alvo e trajetórias divergiram antes da ordem |
| Generalização/produção | NOT MEASURED | Mundo 24/7 permanece vanilla |

## Handoff

Foi criado `crates/vila-policy-cpu` como processo CPU isolado, com contrato
`vila-bed-policy/v1`: estado versionado (cinco pesos mais média por cama),
observação de noite/casa, candidatos com caminho completo/disponibilidade e
resultado com chegada, casa, sono, progresso, dano e resgate. Escolha inicial
entre empates usa menor distância; após feedback negativo sintético, o teste
mostra mudança de cama escolhida. O estado não altera `vila-shadow-cpu`.
`CARGO_BUILD_JOBS=1 cargo test --offline --locked` passou cinco testes;
`cargo build` e CLI `init` passaram; `cargo fmt --check` passou. SHA256 do
binário debug `5092b05926818cc91080f844d5f68dd57aef49b3423c3cb313d5e901a2fe9a77`.
A cópia byte-idêntica em `vila-viva-server/cpu-policy` passou os mesmos cinco
testes; SHA256 do fonte nas duas árvores
`9693ee0f2a280913f8775c4a58d4040a34d0f29e204115ade099dfb8f188e318`.

Próximo: conectar produção de candidatos e ingestão de resultado reais em
cópia isolada, medir quantas decisões/atualizações ocorreram e exigir casa/sono
na tarefa antes de comparar. Cinco testes unitários não demonstram política
aprendida útil; o mundo 24/7 continua vanilla.

Atualização 27/09 10:08 UTC: o produtor Java em Paper isolado examinou oito
camas livres próximas, mas encontrou zero entradas seguras e não chegou a
consultar caminhos. A causa foi ambiental: as quatro casas novas tinham
tábuas sólidas nos corredores à altura dos pés por causa do piso gerado por
`fill ... outline`. Abrir um corredor na cópia fez o aldeão adquirir casa
sem comando da política. A correção delimitada das quatro casas no servidor
principal, após backup consistente, elevou as memórias de casa de 7 para 20;
17 dormiam na noite observada. Isso demonstra que a tarefa de abrigo vanilla
estava bloqueada pela geometria e muda o baseline. O núcleo Rust de política
continua desconectado, sem atualização por feedback real; não inferir
superioridade de IA deste resultado. Próximo teste: candidato positivo e
piloto opt-in em cópia já acessível, com comparação contra o novo vanilla.

## Resultado do preregistro de relógio CANT_REACH — 17:46 UTC

`shadowJar` offline passou. O teste isolado `survival-clock-v2-1742` partiu
da mesma fonte parada e UUID v10, sem ordem, limite0,5 CPU/3GiB e sem porta
publicada. Quatro snapshots em ticks fullTime13780/13800/13820/13840
trouxeram gameTime4944/4964/4984/5004, since4939 e idades5/25/45/65:
o desvio entre relógios foi **8.836 ticks** neste fixture, apesar de
estarem próximos no principal. Episódio
`A-6784215452709343147-20260927-174550` terminou exit0/zero drops,
3/4 candidatos locais seguros, comida0/saciedade null. **PASS** para
tipo/relógio/identidade em Paper; **FAIL** do v1 histórico; **NOT MEASURED**
para `unstick` útil, Rust, recompensa ou vantagem. `CANT_REACH` presente
não é sinônimo de resgate `stuck`.

Auditoria read-only posterior do episódio principal fechado dia38: v9
UUID08368130-915e-417a-aac3-5ca660220b56 teve `LOSING_JOB` no tick
11308838. Antes, memórias `CANT_REACH` persistiram entre keyframes e
o caminho para o job site `SMITHING_TABLE` (-501,85,-406) estava
incompleto (`#4!`). Não há edição registrada desse bloco, porém seu
estado físico/claim atual não foi verificado. Próximo par em mundo
consistente deve testar acesso físico vs controle intacto, medindo rota,
profissão e trabalho; não adicionar postos em lote nem tratar correlação
como causa.

## Preregistro de diagnóstico do posto atual — 18:04 UTC

Backup completo consistente `vila-data-20260927T175655Z.tar.gz` foi criado
com Minecraft parado, índice tar verificado pelo script e principal religado.
O RCON do principal confirmou `SMITHING_TABLE` física em (-501,85,-406)
após o reinício; isso não prova POI livre/reivindicável. Uma fonte isolada
`profession-source-1756` contém os três mundos extraídos do backup e os
outros arquivos de servidor de um fixture anterior, sem episódios/segredos
no Git. UUID selecionado v9 `08368130-915e-417a-aac3-5ca660220b56`.

Primeiro teste **diagnóstico**, sem variante ambiental: em cópia com0,5CPU/
3GiB/sem portas, verificar UUID e bancada por RCON, teleportar só ali v9 à
posição histórica (-495.7,88,-403.7), ajustar manhã apenas na cópia e
observar por1200 game ticks. O mesmo comando inicial seria dado aos dois
braços de um par futuro. Medir profissão/JOB_SITE/CANT_REACH e Brain path
em keyframes; saúde, resgates, drops e tick. Uma profissão adquirida no
controle ou rota completa refuta bloqueio durável no novo snapshot; ausência
de trabalho não prova geometria sem diagnóstico de clearance/POI. Só propor
remoção de um bloco após identificá-lo e registrar mesmo início/horizonte
para controle/intervenção. O teste não é política IA nem altera o principal.

## Preregistro: disputa de navegação — 18:15 UTC

Crítica externa revisada: `Pathfinder.moveTo` confirma início da navegação,
não chegada nem persistência; `setAware(false)` desativa comportamento
próprio e contamina a comparação. Hipótese principal: Brain vanilla
substitui ou cancela rota piloto entre ticks. Alternativas: rota permanece
mas corpo/colisão impedem avanço; cenário anterior não tem chão livre.
O par de passo local anterior refuta apenas aquela integração no fixture.

Implementar traço Java **opt-in** de no máximo40 ticks por ordem `walk_to`,
com `getCurrentPath`/`hasPath`, ponto final e próximo índice, `WALK_TARGET`
e behaviors do Brain lidos pelo Recorder, posição/velocidade/chão/sono,
distância ao alvo, identidade seq/episódio/UUID/tick e motivo final.
Capturar imediatamente após `moveTo` e em cada tick seguinte antes do
refresh; não desativar Brain. Baseline controlado primeiro em chão livre,
mesmo save/UUID/tick e alvo1 bloco, comparando nenhuma ordem vs uma
ordem sem refresh; só depois local problemático e variante com refresh.
Sucesso técnico = deslocamento dirigido além do controle sem dano, não
`moveTo=true`; tarefa final `unstick` ainda exige sair da elegibilidade,
retomar atividade útil e não recair/resgatar num horizonte definido.
Traço não será implantado no principal sem gate.

Auditar também o rótulo `fell`: `Rescue.tick` atual marca `fell` somente se
Y < centroY-belowY; isso é altitude relativa, não prova queda ou dano.
`lethal_fall` separado usa evento de dano fatal e é mais específico.
Não treinar recompensa de prevenção sobre `fell` antes de redefinir
resultado com trajetória vertical/dano/chão. Monitorar HP baixo e reserva
de disco; fome principal continua desligada.

O diagnóstico de profissão `profession-v9-diagnostic-1805` não será usado
como par: `/time set 1000` na cópia reiniciou fullTime para1000, enquanto
gameTime seguiu em11.418.652, e v9 voltou de imediato ao alvo HOME. Nenhum
JOB_SITE apareceu em1200 game ticks; isso não testa acesso ao posto.
Container exit0/zero drops e principal intacto. Corrigir harness com
`time add` monotônico e início acordado se a tarefa de profissão voltar à
prioridade, sem misturar este resultado ao baseline.

### Resultado do primeiro traço e próximo gate — 18:24 UTC

No fixture v10 isolado, `walk_to` para um bloco ao norte, sem refresh nos
primeiros 40 ticks, retornou `moveTo=true` no tick 13820. O caminho ativo
imediato terminava em (-500,95,-371); no tick 13822 já terminava em
(-500,95,-370), enquanto `WALK_TARGET` vanilla apontava para
(-500,96,-368). O caminho alternou presente/ausente repetidamente e o
corpo ficou sem velocidade do tick 13837 ao 13860. Houve só 0,053 bloco de
progresso norte no horizonte; 42 eventos de traço, episódio exit0/zero
drops. Isso demonstra substituição/cancelamento da rota **nesta
integração e neste fixture**; o logger amostra no callback do plugin, por
isso não identifica sozinho o autor de cada escrita NMS.

Próxima alteração limitada: encerrar uma ordem assim que o ponto final do
caminho ativo divergir do final aceito em mais de meio bloco, com motivo
`path_replaced`, sem refresh nem recompensa positiva. Reexecutar a mesma
cópia para validar resultado correlacionado, e então um alvo em chão livre
para separar disputa geral de obstáculo local. A alternativa de controle
exclusivo via player bot será prototipada separadamente: ele não possui
Brain de Villager, mas tampouco é Villager para profissões/reprodução.

## Preregistro: piloto exclusivo solicitado pelo dono — 18:31 UTC

O dono pediu isolamento: um aldeão selecionado deve ficar parado, sem
decisões do Brain vanilla, até receber saída da política. Fazer **somente em
cópia** com configuração opt-in padrão desligada. Usar `Mob.setAware(false)`
como primeiro ensaio de exclusividade por uma janela limitada, registrar
`aware` em cada traço e restaurar `true` ao encerrar; marcador persistente
permite recuperar a entidade após reinício inesperado. O teste deve mostrar
se `Pathfinder.moveTo` ainda produz deslocamento sob esse modo; a API pode
desativar também parte da navegação. Se falhar, não forçar sucesso com
teleportes ou chamar `moveTo=true` de movimento: experimentar um atuador de
velocidade discreta com checagem de colisão, ou um player bot em processo
separado. Nenhuma alteração no principal, no modelo aprovado ou na
população. Medir caminho/posição/velocidade/saúde por40 ticks, resultado
correlacionado e recuperação do `aware` no save após stop. Antes de
qualquer comparativo de tarefa, separar controle motor de qualidade da
política: esta ainda não escolheu a ação neste fixture.

### Primeiro resultado exclusivo — 18:40 UTC

No mesmo fixture v10, `aware=false` começou no tick 13773. `moveTo=true`
no 13820 criou rota até (-500,95,-372); o ponto final ficou idêntico nos
40 ticks seguintes, `WALK_TARGET` permaneceu null, mas posição e velocidade
ficaram constantes. `progress=0`, saúde20, exit0/zero drops. No stop,
`aware=true` foi emitido no tick13884. Portanto `setAware(false)` isola a
decisão vanilla, mas **também paralisa o atuador Pathfinder neste ensaio**.
Não declarar controle motor. Próximo teste discriminante: aplicar uma única
velocidade horizontal norte de magnitude0,2, com `NudgeSafety` existente,
sob o mesmo modo exclusivo, comparar deslocamento e saúde; se físico,
projetar atuador contínuo com validação de cada passo. Não usar teleporte
para mascarar imobilidade.

### Pulso físico exclusivo — 18:46 UTC

Com `aware=false` no v10, um comando `nudge` atômico seq1 (vx0,
vz-0,2), autorizado pelo `NudgeSafety` no tick13820, deslocou 0,5223
bloco exatamente ao norte até tick13860; lateral0, saúde20→20, zero drops,
container exit0. O outcome traz episódio/UUID/v/seq e foi fechado por
deadline; awareness foi devolvida no stop normal. Não houve chegada a uma
meta útil, política Rust, `learn` ou superioridade sobre vanilla. O
atuador de velocidade **funciona em uma aplicação** com Brain suprimido.
Próximo gate: verificar save/restart com piloto desligado, depois definir
contrato de pulsos curtos com observação/resultado a cada passo e comparar
uma tarefa real contra vanilla e regra fixa. Se a física ou colisão mudar
em outro terreno, abortar e restaurar; não extrapolar 0,52 bloco.

No restart da **mesma cópia** com `ai_pilot.enabled=false`, a consulta
RCON do UUID não encontrou `NoAI` nem `BukkitValues`; o segundo episódio
fechou no tick14207, exit0/zero drops. Isso valida restauração no stop
normal e persistência do estado aware nesse ensaio. O monitor de restart
esperava um campo `NoAI:0b` explícito e foi interrompido após a consulta;
o container foi parado limpo. Resultado manual está em
`~/.local/state/vila-viva/exclusive-restart-check-20260927.json`.
# Preregistro 27/09 19:08 UTC — contrato de motor exclusivo

Primeiro bloqueio causal auditado independentemente: `vila-survival-observation/v2`
não é aceito pelo despachante Rust `vila-bed-candidates/v3`; adaptar só o nome
misturaria ações e estado de cama com motor. Criar contrato `vila-motor-policy/v1`
separado para um Villager explicitamente `aware=false`, usando snapshot Paper
com episódio/UUID/v/tick, solo/sono, candidatos cardeais seguros e sinal de
estagnação. O comando deve escolher um candidato publicado sem inventar
velocidade, carregar seq/TTL e ser recusado para snapshot antigo, identidade
divergente ou ownership ausente. A hipótese testável é que um comando Rust
gerado dessa observação moverá o v10 isolado em chão livre como o pulso manual
anterior (progresso dirigido >0, sem dano), com evento de aplicação e outcome
na mesma chave. Controle sem ordem deve permanecer parado. Esse primeiro teste
**não** mede melhora de tarefa nem atualiza pesos. O aprendizado motor terá
estado e ingestão separados, após resultado Paper correlacionado e comparação
com vanilla/regra fixa. Nenhuma ativação no principal ou promoção automática.

Resultado 19:17 UTC: produtor v3 e novo crate independente
`vila-motor-policy-cpu` passaram build Java e 2 testes Rust em ambas as
árvores byte-idênticas. Ensaio Paper `motor-policy-v10-1909` aplicou
decisão Rust de regra fixa norte da amostra v10/tick13800 (live13813)
no tick13820; outcome13860 reportou progresso dirigido0,52228198,
lateral0, saúde20→20, exit0/zero drops. Testes negativos derivados do
snapshot real recusaram lease ausente, awareness vanilla, amostra antiga,
UUID diferente, velocidade adulterada e replay. A hipótese de ponte
comando/resultado passou; aprendizado e ganho de tarefa seguem abertos.
Não alimentar o estado de cama com esse resultado.

## Preregistro 27/09 19:29 UTC — cenário de stuck real, percepção do motor

No episódio principal `...180123`, v0 UUID08368130-915e-417a-aac3-5ca660220b56
foi resgatado por `stuck` no fullTime11799800, posição(-524.5,91,-370.7).
Keyframes duráveis11797001–11799781 mostraram a mesma posição por2780
ticks, acordado, saúde9, CANT_REACH presente e WALK_TARGET/HOME próximos
de(-527,88,-371). A regra Rescue exige2400 ticks, imobilidade<1,5 bloco
e CANT_REACH contínuo; Y91 não é `fell` (limiar Y80). O dado ainda não
prova que um pulso local resolve a dificuldade vertical.

Usar **cópia isolada** do backup consistente 17:56 (`profession-source-1756`),
um UUID, sem portas, 0,5CPU/3GiB e limite explícito. Posicionar só nessa
cópia o v0 na coordenada exata do evento (intervenção artificial marcada),
com piloto desativado para ação e observação v3 ativa. Capturar blocos
feet/head/support e `safe/reject_reason` dos quatro candidatos, HOME,
WALK_TARGET, posição e estado corporal por pelo menos dois snapshots;
encerrar e conferir zero drops. Hipótese: quatro pulsos cardeais planos
não representam a descida de Y91 para HOME Y89; se o sensor disser seguro
mas não oferecer redução vertical, o primeiro bloqueio é contrato de meta/
terreno 3D, antes de treino ou controle contínuo. Não inferir baseline de
resgate de uma entidade teleportada nem alterar o principal.

Resultado 19:37 UTC: na cópia de mundo do backup, o v0 foi colocado
somente ali em(-524.5,91,-370.7). Snapshots2240/2260 mantiveram posição,
saúde9, HOME(-527,88,-371), pés/cabeça em AIR. Norte/sul eram seguros
sob o gate plano; oeste (na direção da HOME) rejeitado por corredor
inseguro, com SPRUCE_LOG/MOSSY_COBBLESTONE; leste rejeitado por folhas.
No tick2280 o vanilla havia se deslocado para z-369.667 e CANT_REACH
apareceu, reforçando que o teleporte reproduz geometria, não a trajetória
pré-resgate. Container exit0/zero drops. Hipótese de limitação 3D passou:
as quatro direções seguras não representam descida até HOME dois blocos
abaixo; escolher norte por ordem fixa seria tarefa cega.

Preregistro de contrato v4: estruturar HOME como deslocamento dx/dy/dz em
blocos relativos ao corpo, preservar string antiga, e exigir no Rust motor
que um pulso seguro reduza a distância horizontal com alinhamento mínimo
e que |dy|<=1 bloco. Sem isso, decisão explícita de abstenção e nenhum
comando. O caso do telhado deve abster-se; o primeiro caso plano validado
antes pode continuar como baseline de pulso, mas não conta como tarefa
concluída. Testar parser/ausência/frescor e saída offline agora; validar
publicação Paper v4 e par de regra depois, sem deploy no principal.

Resultado 19:49 UTC: Java v4 compilou e Rust motor v2 passou2/2 testes
em Universe e espelho servidor. Em cópia isolada
`stuck-v0-v4-exclusive-1941`, índice local v9/UUID original sob lease
verdadeiro, Paper publicou HOME dx=-2/dy=-2/dz≈0,2 da posição imposta;
Rust em observação3200/live3209 decidiu candidato null, motivo
`vertical_route_unavailable`, sem pilot.json. Saúde9, sem deslocamento
durante lease, zero drops/exit0; `pilot_exclusive_end` restaurou
aware=true no tick3212. Esta é uma recusa correta da ação plana, não
sucesso da tarefa. O próximo experimento deve ampliar sensores/ações
para rota 3D e medir chegada segura vs vanilla/regra simples com fontes
pareadas; não usar a estagnação causada por aware=false como rótulo.

Auditoria do input antes de fechar o contrato: o snapshot exclusivo
registrou `stationary_ticks=960`, gerados pelo próprio `aware=false`.
Esse número é movimento observado do corpo, **não** duração de falha do
Brain vanilla. Acrescentar `stationary_context` ao v4 (`controller_paused`
ou `vanilla_motion`) e transportar ao registro Rust, sem usá-lo como
gatilho ou recompensa até haver pré-lease confiável. Testar compilação e
paridade; validação Paper do campo fica para o próximo ensaio isolado.

Resultado do contrato aditivo ~19:55 UTC: Java v4 agora publica
`stationary_context` segundo o lease; Rust motor v2 transporta esse
contexto à decisão e rejeita `vanilla_motion` quando o snapshot afirma
lease exclusivo. A duração não entra na seleção da ação. Build Java,
Rust 2/2 testes nas duas árvores, paridade byte a byte e 15 testes Python
passaram. Ainda **não** houve ensaio Paper do campo novo ou ação positiva
v4: esses elos permanecem NOT MEASURED. Nenhum deploy/peso alterado.

Preregistro ~20:32 UTC, motivado por
`vila-viva-server/docs/AI-ACTION-FAILURE-20260927.md` e pesquisa nova:
repetir a ordem `walk_to` de v10 em uma cópia parada da mesma fonte,
sem mudar alvo/velocidade/raio/UUID, com trace opt-in de40 ticks.
Acrescentar ao log o identificador do objeto NMS `Path` por trás do
wrapper Paper (ou null se inacessível) e capturar
`EntityPathfindEvent` desse UUID dentro da janela, inclusive destino
e call stack NMS limitado. Hipótese A: outro `Path` é instalado entre
13821–13822, possivelmente por `MoveToTargetSink` vanilla; hipótese B:
o mesmo `Path` tem pontos/final modificados por progressão ou
replanejamento. Se a API não emitir evento, classificar escritor como
não medido; não inferi-lo só do endpoint. O trace permanece diagnóstico,
não ação nova, sem mudança do principal. Sucesso do ensaio é classificar
A/B com zero drops/exit0/UUID+seq/ticks, **não** chegar ao alvo nem
provar benefício. Não alterar política/pesos ou chamar concorrência
Brain de causa até identificar a fase que escreveu a rota.

Resultado ~20:32 UTC: `writer-identity-2036` repetiu UUID v10,
seq1, origem13800, aplicação13820, mesmo alvo/raio/velocidade em cópia
limitada. Após `moveTo=true`, o `Path` NMS id379902552 persistiu até
13821. O `EntityPathfindEvent` no13822 para(-500,97,-368) trouxe
stack `MoveToTargetSink.tryComputePath → Behavior.tryStart → Brain.tick
→ Villager.customServerAiStep`; o caminho ativo passou a
id835786466/final(-500,95,-370), distinto do alvo aceito
final(-500,95,-371). Outcome `path_replaced`, distância0,628>0,25,
exit0/zero drops/saúde20. Eventos `BedCandidates.checkHomeRoute`
ocorreram no13820, mas não mudaram o id ativo no13821. Bytecode Paper
1.21.4-232 da própria cópia confirma que `MoveToTargetSink.start`
chama `PathNavigation.moveTo(this.path,speedModifier)`: neste fixture
o Brain instala outra rota. Ainda não se mediu quem escreveu originalmente
`WALK_TARGET`, nem se intervenção seletiva preserva locomoção.
Nenhum deploy/peso alterado; M1 e ganho continuam não medidos.
Preregistro 27/09 ~20:43 UTC, próximo experimento de propriedade da rota,
antes de alterar Java. O traço `writer-identity-2036` identificou
`MoveToTargetSink.tryComputePath` como fonte da rota substituta no13822.
Hipótese: vetar **somente** `EntityPathfindEvent` desse comportamento para
o UUID selecionado durante até40 ticks de uma ordem `walk_to` já aceita
impedirá a substituição sem desligar `PathNavigation`/física. O bytecode
Paper 1.21.4-232 mostra que evento cancelado remove o destino antes do
pathfinder; porém `MoveToTargetSink` pode parar a navegação por outra via,
portanto preservar a rota e mover são desfechos a medir, não assumir.
Variável única: flag opt-in `veto_brain_pathfind`, default false. Comparar
com a cópia `writer-identity-2036` de mesma fonte, UUID, alvo, velocidade,
raio, comando e horizonte; registrar evento cancelado, identidade/final
do Path a cada tick, WALK_TARGET, velocidade, distância, saúde, sono,
motivo terminal, restauração pós-lease, drops/exit. Critério de mecanismo:
sem rota substituta de `MoveToTargetSink`; critério motor separado:
chegada <=0,25 bloco antes do prazo e saúde intacta. Se rota persistir
mas corpo parar, classificar motor FAIL; se outro escritor substituir,
propriedade FAIL. Não classificar isso como IA melhor ou executar M1 antes
de um caminho livre chegar. Sem deploy/pesos, e a cópia deve encerrar.

Resultado ~20:50 UTC: cópia `path-veto-2046`, mesma fonte/UUID/seq/
alvo/raio, encerrou exit0/zero drops/saúde20, sem chegada. A ordem
foi aplicada13820 e o Path do piloto (id358901963) persistiu em13821–13822;
47 `EntityPathfindEvent` de `MoveToTargetSink` foram cancelados até13860.
Porém no13823 o caminho ativo já era null; o corpo desacelerou e ficou
~0,528 bloco distante do alvo até13860. No13861, fora da janela, o
Brain instalou outra rota e o outcome fechou `path_replaced`, `arrived=false`.
Portanto o veto bloqueou a rota substituta durante a janela, mas NÃO
preservou navegação nem completou a ação. O bytecode de
`MoveToTargetSink.stop` inclui `PathNavigation.stop`; não medimos essa
chamada específica no tick13823, então atribuir o null a ela é hipótese.
Rejeitar veto de evento como solução de controle. Próximo ensaio, se
justificado: adquirir lease antes da ordem e suspender o comportamento
`MoveToTargetSink` inteiro de modo reversível, sem desativar o mob,
testando caminho/movimento/saúde/retorno em cópia; manter flag veto off.
Preregistro 27/09 ~21:05 UTC, próximo elo após veto falhar. No Paper
1.21.4-232, `Brain.availableBehaviorsByPriority` contém sets de
`BehaviorControl`; tanto start quanto tick/getRunningBehaviors percorrem
esse mapa. `MoveToTargetSink.stop` para `PathNavigation`, logo cancelar
seu evento não basta. Hipótese: remover **a única instância** de
`MoveToTargetSink` do set do Brain do UUID selecionado ANTES de emitir
`walk_to`, sem chamar `doStop` e sem `setAware(false)`, permitirá que o
`Path` aceito progrida até<=0,25 bloco. Restaurar a mesma instância no
mesmo set ao chegar ou até40 ticks, registrando status e identidade;
falha de reflexão/topologia deve rejeitar comando sem mutar o Brain.
Variável única: flag experimental `suspend_brain_move`, default false,
na mesma fonte/UUID/alvo/velocidade/raio/seq de `writer-identity-2036`.
Medir por tick rota/objeto/endpoint, velocidade, distância, saúde,
sono, eventos Brain e retorno vanilla pós-lease; exit0/zero drops.
Critérios separados: registry ausente durante lease e restaurado (mecanismo),
chegada <=0,25 antes de40 ticks (motor), sem dano e vanilla volta
(recuperação). Caso falhe, registrar primeiro ponto sem aumentar prazo,
raio ou velocidade. Não implantar no principal nem chamar isso IA aprendida.
Resultado ~21:11 UTC de `brain-move-lease-2105` (flag experimental
`suspend_brain_move`, default false): o Brain v10/seq1 tinha uma única
instância `MoveToTargetSink`, status STOPPED, removida do registro antes
do `walk_to` e reinserida com a **mesma identidade** no tick13860
(`restored=true`). Aware permaneceu true; nenhum pedido de caminho do
`MoveToTargetSink` apareceu na janela. Path do piloto id1545862012
persistiu até13822, com próximo nó já final, mas a API Paper passou a
reportar caminho ausente no13823; o corpo parou a0,676 bloco do alvo,
sem chegada<=0,25. Resultado `brain_move_lease_end` no13860,
saúde20/exit0/zero drops. O ganho mecânico de evitar preempção PASS;
chegada FAIL; retorno funcional vanilla após restaurar NOT MEASURED.
Fonte/UUID/tempo/regra norte1 iguais, mas a posição inicial diferiu
~0,153 bloco em X da cópia anterior; portanto não são trajetórias
rigorosamente pareadas nem o alvo absoluto foi idêntico.

Bytecode Paper esclarece uma hipótese rival ao `stop` externo:
`PaperPathfinder.getCurrentPath()` retorna null se o NMS Path está null
**ou isDone**. `PathNavigation.followThePath` avança o último nó quando
o corpo fica dentro da tolerância por eixo em torno do centro do bloco.
No13822 corpo(-499,529;-370,299) estava a(dx~0,029,dz~0,201) do
centro do nó final(-499,5;-370,5), enquanto a0,678 do alvo contínuo.
Assim, término natural do Path por tolerância é plausível; o trace
atual não lê `Path.isDone` diretamente, então não descarta
`PathNavigation.stop` por outro comportamento. Próximo teste mínimo:
instrumentar **raw NMS Path/null/isDone** e causa de término sem mudar
alvo, velocidade, raio ou recompensa; só então escolher correção motora
precisa. O teste atual não valida M1 nem IA aprendida.
Preregistro 27/09 ~21:25 UTC — desambiguação do fim do Path no lease seletivo. Hipótese A: no tick13823 o Path NMS ainda existe, mas `isDone=true` porque `followThePath` aceitou o centro do nó final enquanto o alvo contínuo segue >0,25 bloco. Hipótese B: o Path NMS passa a null por um `stop`/limpeza ainda sem autor identificado. Em uma única cópia limitada da fonte parada `policy-feedback-1425`, mesmo UUID v10, comando `walk_to` norte1, velocidade1, raio0,25, seq1, `suspend_brain_move=true` e horizonte40 ticks de `brain-move-lease-2105`, adicionar somente telemetria por tick: identidade do Path NMS bruto obtido uma vez de `PathNavigation.getPath`, presença, `isDone`, índice e número de nós, junto à visão Paper, posição, velocidade, saúde e restauração do lease. Sem alterar motor, alvo, raio, tempo ou recompensa. Se raw Path presente+done no13823, classificar término normal do Path e chegada ao alvo contínuo FAIL; se raw null, classificar limpeza da referência e autoria NOT MEASURED; se raw presente+not done com Paper null, investigar fase de amostragem. Exit0/zero drops/saúde preservada e cópia encerrada são guardrails. Este ensaio não valida M1, política aprendida ou vantagem.

Resultado ~21:30 UTC — `raw-nms-path-2127` confirmou hipótese A nesse fixture. No13822 o Path NMS id1855315346 estava presente, `isDone=false`, índice1/2, Paper presente, distância0,6277. No13823 o **mesmo objeto** continuou presente, `isDone=true`, índice2/2, Paper null, distância0,5460. Deriva posterior estabilizou perto de0,528 bloco, ainda fora do raio0,25; outcome13860 `brain_move_lease_end`, arrived=false, progresso0,4721, saúde20. Mesmo sink restaurado, exit0, zero drops, cópia encerrada. A referência não foi removida nesse tick; não atribuir o null a `nav.stop` neste ensaio. A chamada intratick que avançou o último nó ainda não foi identificada, mas o contrato motor já difere: término do Path por nós != chegada ao alvo contínuo. Próximo ensaio isolado, preregistrado separadamente antes de codificar: corretor final curto e seguro acionado só para Path raw concluído com distância>raio, mesmo alvo/velocidade/raio, mantendo lease; medir chegada, dano, oscilação e restauração. M1/vanilla/regra/learn continuam não medidos.

Preregistro 27/09 ~21:44 UTC — correção motora final. Auditor read-only sugeriu um único `findPath/moveTo` repetido e alertou que impulso poderia exceder o alvo. Conferi o mecanismo: o Path original já terminou no mesmo nó de bloco antes do alvo contínuo; um retry para o mesmo bloco não discrimina a semântica restante. Hipótese: até três pulsos horizontais de velocidade0,2 rumo ao ponto exato, separados por pelo menos dois ticks e cada um precedido pelo `NudgeSafety` atual, podem reduzir a distância de~0,53 para<=0,25 sem dano. Única variável ativa é flag opt-in `final_correction=false` em cópia com `suspend_brain_move=true`, `aware=true`; mesmo UUID v10, save parado, seq1, norte1, alvo/velocidade/raio0,25 e horizonte40. Entrada só se o Path NMS bruto estiver presente+`isDone`, distância>raio e<=0,75, diferença vertical<=0,25, aldeão em chão/acordado/válido, saúde não abaixo da inicial e lease válido. Falha de reflexão/corredor/solo/saúde ou três pulsos sem chegada deve terminar sem repetir indefinidamente; ao chegar, remover velocidade horizontal e restaurar o Brain. Registrar tick/seq/distância/velocidade/segurança por pulso e outcome com distância, chegada, saúde, número de pulsos, restauração; observar após retorno ao vanilla até14040. Critério positivo estrito: <=0,25 em até40 ticks, saúde20, zero resgate/drop, sem oscilação para fora do raio logo após o fim; caso contrário rejeitar a correção. Nada disso prova política aprendida ou superioridade. Um ensaio limitado, sem porta, com término garantido.

Resultado ~21:51 UTC — cópia `final-correction-2144` atingiu o alvo pontualmente, mas **não passou o critério completo preregistrado**. A mesma fonte/UUID/tick/alvo iniciou a rota no13820; Path raw ficou `isDone=true` no13823 a0,5460 bloco. Um pulso seguro vx=-0,1529/vz=-0,1289 aplicado nesse tick reduziu a distância para0,3460 no13824 e0,23684 no13825, quando outcome `arrived=true` e o sink Brain original foi restaurado. Saúde20, nenhum resgate, exit0/zero drops; só um pulso. Porém no frame13833, oito ticks após devolver o Brain vanilla, distância já era0,8902; no13853,1,0498 e no14033 o aldeão estava perto da origem. É chegada física pontual em um fixture, **não permanência nem realização de tarefa**. A política Rust não escolheu a ação, nem recebeu recompensa/learn. Primeiro elo novo: ownership temporal acaba no instante de chegada, vanilla retoma outra meta e desfaz o resultado espacial. Não aumentar raio, nem chamar isso IA melhor. Próximo: validar motor em M1 oito metas livres com critérios de chegada e segurança, e definir objetivo durável/lease ou sequência de política com feedback final e baselines separados.

Preregistro 27/09 ~22:05 UTC — triagem M1 em arena limpa. A síntese define 4 direções × 2 posições, alvo a2 blocos, horizonte120 ticks e chegada<=0,25 em>=7/8, sem dano/teleporte. Primeiro lote desta rodada: posição A no mesmo save parado/v10, norte2, dois braços sequenciais em cópias separadas: vanilla sem ordem e piloto `BrainMoveLease+final_correction` opt-in; não contar como M1 concluído. Em cada cópia, construir somente nela plataforma stone x[-502,-484],z[-383,-367],y109, limpar ar y110–113 e teleportar o mesmo UUID para (-496,5;110;-375,5). Aguardar observação Paper fresca/mesmo UUID, chão, saúde; alvo = posição observada + (0,0,-2), velocidade1, raio0,25, TTL <=120 ticks, sem outro comando. Controle usa mesmo alvo hipotético e nenhuma ordem. Medir por tentativa: estado inicial/target, trajetória por frame, eventos piloto/tick de chegada/distância no horizonte120, dano/resgate, path/raw/lease, retorno após restaurar, exit/drops; registrar qualquer diferença real de estado inicial/RNG como limitação de pareamento. Todos containers0,5CPU/3GiB/sem porta, apenas um por vez e parada no horizonte; erro de teleporte/precondição é falha de setup, não sucesso motor. Critério local piloto: chegada<=0,25 em até120 com saúde intacta e sem resgate; permanência pós-lease avaliada separadamente. Não atribuir aprendizagem nem vantagem a duas cópias. Se setup inviável ou disco abaixo da reserva, parar sem alterar principal.

Emenda pré-comparação ~22:11 UTC: a primeira cópia vanilla `m1-a-north-vanilla-2206` iniciou corretamente no tick13780 na posição exata do teleporte e terminou13900 exit0, mas observação/comando estavam configurados a cada20 ticks. O primeiro frame a13793 já estava3,5 blocos do alvo, e uma ordem piloto só poderia ser aplicada muito mais tarde; logo essa trajetória é exploratória, **fora do denominador pareado M1**. Repetir controle e piloto em cópias novas com as mesmas cadências de observação e checagem a1 tick, exigir snapshot tick posterior ao teleporte e posição<=0,8 da base, emitir ordem nesse snapshot e monitorar horizonte120. Não aplicar lentidão/NoAI ou congelar o mob na preparação, pois isso alteraria o motor. Registrar tick entre teleporte/observação/aplicação. Só então comparar; não reinterpretar o braço inicial como falha de setup, pois ele teve início observado válido.

Emenda de contrato ~22:19 UTC, antes de mudar Java: a tentativa `m1-a-north-vanilla-fast-2212` abortou setup sem comando; `SurvivalObservation` clampa `everyTicks` a **mínimo20** apesar de config1, e o script podia teleportar antes de existir a primeira observação, tratando o snapshot antigo como posterior. Container exit0/sem mundo principal alterado; resultado não entra em M1. Corrigir somente o produtor opt-in para honrar `survival_every_ticks=1` (default20 permanece) e fazer o script aguardar um snapshot inicial antes do teleporte. Recompilar e executar cópia controle nova, verificando diferença de tick entre pré-teleporte/pós-teleporte e coordenadas. Não modificar motor/recompensa/raio nem aproveitar a tentativa inválida como evidência de falha física. Se a frequência1 causar carga/erros, abortar e escolher sincronização por tick do RCON numa rodada posterior.

Resultado exploratório ~22:32 UTC: após corrigir cadência e snapshot, `m1-a-north-vanilla-sync-2219` iniciou em(-496,5;110;-375,5) tick13780 e ficou10,249 blocos do alvo norte2 no fim120, saúde20/zero resgate/drop/exit0. `m1-a-north-pilot-sync-2225` iniciou nas mesmas coordenadas tick13778 e emitiu seq1 nesse tick; no13779 Java rejeitou `unreachable_target` **antes de aplicar a ordem**. Saúde20/zero drop/exit0; trajetória final também se afastou. A diferença de dois ticks e o gate de rota impedem interpretar como comparação do motor/corretor; M1 célula ainda FAIL de elegibilidade da integração, não sucesso. A arena era plataforma artificial em cópia, não o mundo principal. Próximo teste causal preregistrado antes da edição: manter a mesma fonte/plataforma/UUID/posição/direção/alvo/raio e só adicionar evento diagnóstico ao gate `findPath`: path null vs parcial, quantidade de pontos, último ponto, `canReachFinalPoint`, posição/solo do corpo, materiais nos pés/cabeça/suporte do alvo e tick. Rodar uma cópia piloto curta; não afrouxar gate, mover arena, alterar alvo/velocidade ou concluir que Pathfinder é incapaz. Se `findPath` vier parcial, investigar por que o bloco é inacessível apesar de plataforma; se null, investigar inicialização pós-teleporte. Guard exit0/zero drops/sem dano e cópia encerrada.

Resultado ~22:40 UTC — `m1-a-north-route-probe-2233` repetiu o mesmo setup com diagnóstico e confirmou `findPath` **null** no13778; destino AIR/AIR com suporte STONE, origem(-496,5;110;-375,5), mas `selected.isOnGround=false` naquele tick. O snapshot logo após teleporte que iniciou a tentativa tinha on_ground=true; há transição corporal durante a preparação. Sem ordem aplicada, saúde20/zero drop/exit0. Isso torna prematura a conclusão de que o Pathfinder falha em chão livre; pode ser estado pós-teleporte, mas causalidade de `on_ground` ainda não isolada. Próxima emenda preregistrada: manter arena/UUID/controle/ação e aguardar pelo menos três ticks observados após teleporte, `on_ground=true`, y estável, sem sono/dano e deslocamento <=2 blocos da base antes de fixar posição inicial e alvo relativo2. Exigir o mesmo gate nos braços vanilla e piloto, observação cada tick; abortar setup se não estabilizar até20 ticks. Sem lentidão/NoAI, sem mexer no gate Java ou no atuador. Comparar tick da emissão/aplicação e rota completa; se ainda null com chão estável, investigar estado do navegador/forma da plataforma.

Resultado ~22:55 UTC — primeira célula M1 **parcial** norte2/posição A validada após gate de solo. `m1-a-north-grounded-pilot-2241`: v10 iniciou13780 em(-496,5;110;-375,5), ordem seq1 emitida13780/aplicada13781, Path3 nós aceito, distância de aplicação2,061→0,1803 no13787, `arrived=true`, zero pulsos corretivos, saúde20/zero resgate/drop/exit0, sink restaurado. Trace por tick teve passo máximo0,502, sem teleporte durante ação. Controle `m1-a-north-grounded-vanilla-2248` com mesmo gate iniciou13782 em(-496,5;110;-375,43875), sem ordem; nenhum frame observado perto do alvo (primeiro3,439, fim10,190), saúde20/zero resgate/drop/exit0. Estados iniciais diferiram2 ticks e0,061 bloco, então é comparação exploratória, não réplica idêntica. Após lease, piloto afastou-se do alvo: tarefa/permanência FAIL. Contar **1/8 células M1 com chegada**, não afirmar 7/8 nem IA superior; Rust motor ainda é regra fixa e não decidiu este `walk_to`. Próximo: outras direções/posição B em cópias sequenciais com ordem dos braços alternada, todos os resultados inclusive rejeições no denominador; depois M1 negativos e tarefa útil. Uma célula não estima ganho geral.

Preregistro 27/09 ~23:00 UTC — célula M1 A/sul2. Manter o mesmo save parado, UUID v10, jar, arena de cópia, teleporte A, gate de três ticks e duas observações consecutivas em solo, observação a cada tick, destino relativo `(0,+2)` e horizonte120 ticks. Executar **controle vanilla primeiro**, depois piloto, cada qual em cópia isolada sem porta e limite0,5 CPU/3GiB; nenhuma execução paralela. Critério pontual do piloto: ordem aplicada a partir de snapshot fresco, chegada medida por `pilot_outcome`/trace a distância<=0,25 até120 ticks, saúde sem queda, nenhum resgate/teleporte durante ação, zero drops, exit0, sink restaurado. Registrar rejeição do gate como falha de integração, não esconder a célula. No controle medir proximidade em frames disponíveis, explicitando que `live` a cada~20 ticks pode perder uma chegada transitória; não declarar ausência absoluta de chegada sem telemetria equivalente. Permanência após devolver Brain é desfecho separado. Registrar diferenças de tick/posição inicial e alvo relativo; não inferir IA aprendida, vantagem estatística ou tarefa útil a partir deste par. Auditor independente alertou que `frames` do resumo são esparsos apesar da observação Paper a1 tick; `pilot_outcome` e trace por tick são fonte primária para chegada no piloto.

Resultado ~23:14 UTC — A/sul2, braço vanilla `m1-a-south-grounded-vanilla-2301`: tick inicial13781 em(-496,5;110;-375,5), alvo(-496,5;110;-373,5), seis frames no horizonte120 com mínimo0,5 e fim6,5299 blocos; saúde20, resgate0, drops0, exit0. Esses frames a ~20 ticks não excluem passagem transitória por<=0,25. Braço piloto `m1-a-south-grounded-pilot-2308`: tick inicial13783 na mesma posição/alvo relativo, seq1 aplicada13784, Path3 nós, outcome `arrived` no13789 a0,032935 bloco; trace de ação por tick com passo máximo0,4971, saúde20, zero resgate/drop, sem pulso corretivo, sink original restaurado, exit0. Quadro final pós-lease6,441 blocos: permanência/tarefa FAIL. Diferença de início2 ticks; não é réplica perfeita. M1 positivo para chegada motora pontual em **2/8** células norte2+sul2 A, não para resultado durável. Controle contínuo, outras seis células, política Rust escolhendo `walk_to`, feedback/learn e vantagem vs vanilla+regra simples permanecem NOT MEASURED. Antes de próxima célula, preregistrar captura mais densa do controle, preservando motor e critérios.

Preregistro 27/09 ~23:22 UTC — fechar lacuna de amostragem do controle M1. Auditor read-only confirmou `SurvivalObservation` por UUID/tick a1, mas arquivo atômico sobrescrito e polling do harness a0,1s perdem ticks; `m1_motor_trial.py` resume `live-*.jsonl` com `live_every=20`. A menor intervenção sem novo código Java é uma **configuração somente de cópia** com `live_every=1` e `position_step=0.001` para fornecer todos os ticks com erro de quantização <0,001 bloco por eixo; plugin/jar/motor/arena/UUID/gate/raio/horizonte permanecem idênticos. Modificar apenas harness para aceitar caminho explícito dessa configuração e exigir ticks distintos consecutivos de `start` a `start+120` no arquivo live, com episódio e UUID selecionados; falha de densidade é falha de medição, não sucesso/fracasso motor. Teste discriminante único: repetir controle vanilla A/sul2, sem comando, mesmo alvo relativo, 120 ticks; medir mínimo, primeiro tick dentro de0,25 se houver, distância final, saúde/resgates, duração/CPU/disco, exit0/zero drops. Comparar somente com o controle esparso anterior como auditoria de medição, sem reivindicar igualdade contrafactual/RNG; se a coleta densa ameaçar TPS/disco ou tiver gaps, abortar essa via e usar coletor Java único opt-in em rodada seguinte. Após confirmar, usar a MESMA config densa nos dois braços das células M1 restantes; não misturar sucesso motor pontual com tarefa/learn.

Resultado ~23:30 UTC — a lacuna de amostragem alterou a conclusão do controle. Sem mudar jar/motor/arena/gate, cópia vanilla `m1-a-south-vanilla-dense-2322` usou `live_every=1` e `position_step=0.001`. O harness exigiu 121 ticks únicos consecutivos de13780 a13900 para UUID v10, mesma origem(-496,5;110;-375,5) e alvo sul2; validou 121/121, zero lacunas/drops/dano/resgates, exit0, sem comando. **Vanilla entrou no raio0,25** em13795,13796,13797, mínimo0,004 no13796 e final6,508. Os seis frames da cópia anterior haviam omitido os três ticks de chegada. Baseline sul2 tem chegada pontual vanilla observada; não existe ganho exclusivo do piloto nesse alvo. Em cópia0,5CPU, live denso custou~2,52MB/120 ticks e TPS médio~15,1, comparável aos braços esparsos~14,5–14,8; TPS da cópia não representa principal80. Repetição independente não pareia RNG/estado; piloto chegou em6 ticks de seu start e vanilla em15 do seu, mas isso não demonstra vantagem sem réplicas/controlos robustos. M1 2/8 chegadas do piloto ainda vale como prova motora pontual, não como superioridade. Próximo preregistro separado: A/oeste2 com ambos braços densos, ordem piloto→vanilla, mesma arena/gate e critério; medir também tarefa/permanência separada.

Preregistro 27/09 ~23:40 UTC — M1 célula A/oeste2, agora com baseline denso. Após o achado de chegada vanilla transitória em sul2, executar piloto **primeiro** e vanilla depois, em cópias sequenciais da mesma fonte parada, mesmo UUID v10, jar, arena plana só de cópia, posição A, gate de>=3 ticks pós-teleporte e dois snapshots consecutivos em solo, alvo relativo(-2,0), velocidade1/raio0,25/horizonte120. Ambos usam a configuração privada `m1-dense-config-2322.yml` (`live_every=1`, posição0,001) e o harness exige121 ticks consecutivos por UUID/episódio; cap0,5CPU/3GiB, sem porta, parada/exit0 e zero drops. Critério pontual do piloto: ordem válida aplicada, distância<=0,25 em até120 ticks, sem dano/resgate/teleporte, sink restaurado; qualquer rejeição conta como falha da célula. Controle: registrar primeira e mínima distância<=0,25 se ocorrer, duração dentro do raio e posição final. Em ambos medir TPS/tick_ms da cópia, saúde, resgates, deslocamento máximo por tick e permanência no fim separada da chegada. Registrar diferenças de tick/posição inicial/RNG; mesmo seed não garante trajetórias idênticas. Se medição densa falhar, não inferir resultado de baseline. Nenhuma alteração de motor, arena ou limiar após ver resultado. Isso testa alcance físico local, não tarefa útil, decisão Rust ou aprendizagem.

Esclarecimento de critério ~23:41 UTC, **antes de ler A/oeste2**: auditor independente encontrou divergência entre o relatório M1 original e o contador operacional. A síntese especifica **distância FINAL <=0,25 em >=7/8 células**, não mera passagem transitória. Por isso manter duas colunas separadas para cada braço: (1) primeira chegada pontual/tempo dentro do raio, diagnósticos do motor; (2) distância no tick `start+120`, o verdadeiro critério M1. As células norte2 e sul2 do piloto registradas como 2/8 de chegada pontual NÃO são 2/8 aprovações M1, pois se afastaram ao final. Nenhuma meta/raio/horizonte/arena mudou. A/oeste2 seguirá o mesmo critério final preregistrado, incluindo falhas/rejeições no denominador conforme aplicável. Se o piloto atingir e depois sair, marcar chegada pontual PASS e M1 final FAIL.

Resultado ~23:54 UTC — A/oeste2 denso, conforme preregistro. Piloto `m1-a-west-pilot-dense-2340` e controle `m1-a-west-vanilla-dense-2347` iniciaram ambos tick13784 exatamente em(-496,5;110;-375,5), alvo(-498,5;110;-375,5), 121/121 ticks sem gaps, saúde20/zero resgate/drop, exit0. Piloto seq1/Path3 aplicada13785, `pilot_outcome arrived`13790 a0,1707; seis frames no raio, mínimo0,026, mas **distância FINAL7,8519**, M1 FAIL. Sink restaurado no mesmo tick da chegada. Vanilla sem ordem não entrou no raio nos121 ticks (mínimo2,000), final7,8564, M1 FAIL. Passo máximo observado piloto0,4918/vanilla0,1358 bloco por tick, sem teleporte detectado; TPS médio das cópias cap0,5CPU 15,26/15,09. Mesmo início/tick não torna cópias sequenciais contrafactuais idênticas. Piloto agora 0/3 aprovações FINAIS (norte final10,2489; sul6,4413; oeste7,8519), apesar de3/3 chegadas pontuais. Com três falhas, >=7/8 está matematicamente impossível neste lote; suspender cinco células restantes. Primeiro FAIL causal observado é autoria temporal: `ControlPilot.finish("arrived")` solta o lease e reintegra MoveToTargetSink ao chegar, enquanto o resultado final exige permanecer. Não usar chegada transitória como aprovação M1 nem treinar como tarefa concluída.

Preregistro ~23:55 UTC — experimento de propriedade temporal após chegada, antes de codificar. Em uma única cópia A/oeste2 da mesma fonte/UUID/arena/config densa, manter aware=true, o mesmo Pathfinder, alvo relativo2, velocidade1, raio0,25 e limite120; adicionar flag opt-in default false para **reter apenas o BrainMoveLease seletivo após a primeira chegada até o tick start+120**, em vez de chamar `finish("arrived")` imediatamente. Registrar tick da primeira chegada, todos os ticks dentro/fora do raio, posição/velocidade e Brain WALK_TARGET/behaviors quando disponíveis, saúde/resgate/teleporte, motivo de liberação e confirmação de restauração no120/shutdown. Não aplicar força extra, teleporte, novo caminho ou mudar geometria nesta primeira discriminação; se sair mesmo com sink ausente, bloquear atribuição ao retorno vanilla e examinar inércia/outro comportamento. Critério de validação motor M1 local: distância no tick120<=0,25, saúde intacta, zero resgate/teleporte/drop, exit0, restauração verificada. Controle histórico piloto oeste final7,8519 sem hold; cópias podem divergir por estado/RNG. Isso ainda não cria política aprendida nem tarefa útil; se passar, repetir/avaliar baselines e segurança antes de qualquer principal.
