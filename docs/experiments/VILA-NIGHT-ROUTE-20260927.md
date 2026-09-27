# Investigação: aldeão com casa fica imóvel à noite

Status: OPEN

Responsável: Codex, 27/09/2026 10:20 UTC

Aceite do dono: um aldeão escolhe ações com política de IA e supera vanilla
numa tarefa relevante, sem comprometer o mundo 24/7.

## Falha observada e identidade

No Paper principal 1.21.4-232, plugin `0.1.1-notebook`, episódio
`A-6784215452709343147-20260927-100317`, UUID
`9fb7f1ca-c8da-4a5e-b08c-c1530786a68f` (v6, saúde 4), o Brain vanilla
manteve `walk_target=-534,86,-413@0.5/1` na segunda noite. Entre ticks
9229606 e 9236006, ficou em cerca de `(-530.3,84,-405.5)` e não dormiu,
apesar de ter `mem_home=-534,86,-413` naquele intervalo. Noite anterior,
esse mesmo UUID dormiu em outra cama. No snapshot posterior consultado por
RCON, a memória de casa já era `(-549,84,-401)` e o ser estava dormindo;
logo, a falha não deve ser tratada como permanente.

Fluxo executado: mundo/POI → Brain vanilla e `walk_target` → navegação do
Paper → corpo → `live-0.jsonl`; `ControlPilot` e `vila-bed-policy/v1` não
foram executados no principal. A primeira discrepância demonstrada é entre
alvo de casa persistente e falta de deslocamento/sono; ainda não se sabe se
o Paper oferece rota completa à cama, se o Brain substitui rotas ou se há
impedimento ambiental. Um mapa somente de blocos de ar mostrou degraus e
barreiras entre `(-530,-405)` e a entrada, mas não prova falta de caminho.

## Teste discriminante preregistrado

- Usar somente cópia isolada do backup consistente, plugin opt-in
  `0.2.0-pilot`, 0,5 CPU, sem porta publicada, RCON interno. Não tocar o
  mundo principal e parar após uma observação noturna relevante.
- Na cópia, abrir os mesmos corredores internos já corrigidos no principal,
  colocar apenas o UUID acima em `(-530.3,84,-405.5)` e definir sua memória
  de casa para a cama `(-534,86,-413)` via comando RCON, se suportado.
  Registrar explicitamente essas intervenções; se o comando de memória não
  funcionar, o teste fica **não medido**, não se infere rota.
- O produtor Java opt-in consulta até oito abordagens seguras da cama,
  com caminho completo no Pathfinder, e emite `home_route_status`, contagens
  e duração a cada 400 ticks. Resultado `reachable` refuta a hipótese de
  rota impossível para esta posição; `no_complete_path` a apoia somente
  neste fixture. Não executar ação da política neste teste.
- Orçamento: uma inicialização, uma observação e parada limpa; verificar
  limite de memória, temperatura, TPS do principal e nenhuma porta extra.

## Gates

| Gate | Estado |
|---|---|
| Diagnóstico Java compilado | PASS (`shadowJar` offline; Paper pendente) |
| Rota no fixture | FAIL: `no_complete_path`, 2 entradas seguras/2 consultas em dois snapshots |
| Decisão aprendida/aplicada | NOT MEASURED |
| Ganho contra vanilla | NOT MEASURED |

O modelo aprovado `shadow-20260926-220823` continua apenas preditivo. O
núcleo Rust de cama ainda não recebe observação real. Se a rota for possível,
próximo teste deve distinguir interferência do Brain vanilla com intervenção
opt-in temporária e baseline pareado; se não, avaliar outro alvo/entrada
segura antes de agir. Não alegar IA melhor com esse diagnóstico.

## Execução e decisão (10:27 UTC)

O plugin Java com `home_route_status` compilou (`nice -n 10 ./gradlew
--offline --max-workers=1 shadowJar`); jar SHA256
`08fdd07a3a0f3d7dcbaa4862f82060e13a1394ad9fb053ddd4647143f387937b`.
Rodou no Paper isolado `vila-ai-houses-0927`, 0,5 CPU e 3 GiB, sem portas
publicadas, episódio `A-6784215452709343147-20260927-102303`. As quatro
passagens foram corrigidas nessa cópia, como no principal. A memória HOME
do UUID foi ajustada somente na cópia para `(-534,86,-413)` e confirmada
por `data get`; o ser foi teleportado para `(-530.3,84,-405.5)` e o horário
foi posto à noite. Uma primeira tentativa de snapshot enquanto ele ainda
dormia reposicionou-o na cama e não reproduziu o início desejado; **não**
conta como evidência da rota.

Depois de acordá-lo de dia, congelou-se o mundo de teste, repetiu-se o
teleporte, ajustou-se o tempo para 13999 e executou-se um tick. No tick
14000, a observação registrou x=-530.3,y=84,z=-405.5, casa presente,
`home_route_status=no_complete_path`, duas abordagens seguras e duas
consultas de caminho. Após descongelar, no tick 14800 (~800 ticks depois),
o aldeão estava a menos de 0,01 bloco da mesma posição, continuava sem
dormir e com o mesmo alvo HOME; o diagnóstico repetiu `no_complete_path`
com duas consultas. A cópia foi parada graciosamente e preservada.

Isso reproduz neste fixture a ausência de caminho para a cama atual e
explica por que manter o mesmo alvo vanilla não basta. Não demonstra ainda
que uma cama alternativa seja alcançável, que a política a escolha ou que
melhore o vanilla. Próximo teste: preservar este estado como base de dois
braços, gerar candidatos alternativos acessíveis e comparar política
opt-in de realocação contra vanilla em janela noturna delimitada.

## Segundo teste preregistrado (10:37 UTC)

Com o mesmo UUID acordado e mundo isolado congelado no tick 16000 em
`(-530.3,84,-405.5)`, o produtor encontrou 20 cabeceiras próximas, 14
ocupadas, seis livres, dez entradas no alcance e **zero caminhos completos**
para camas alternativas. Uma observação natural de tick 15200 em posição
~0,21 bloco diferente havia retornado `home_route_status=reachable`; a
sensibilidade a posição/estado impede concluir que a cama seja
permanentemente inacessível. Para distinguir travamento local de barreira
global, testar somente o comando mecânico `walk_to(-530.5,84,-403.5)` por
200 ticks, ponto com pés/cabeça de ar e apoio não vazio. Gate: caminho
completo aceito, deslocamento >=1,5 bloco e ausência de dano; depois medir
novamente caminho à HOME e sono. Ordem rejeitada ou sem progresso refuta
esse waypoint. O alvo foi escolhido pelo operador, **não pela IA**. Mesmo
sucesso não será comparado causalmente com vanilla até rodar dois braços
pareados da base parada.

Resultado do segundo teste (10:38 UTC): compilação `shadowJar` offline passou,
jar SHA256 `d542f8515a9fc706c1a80a41453d37aab7690420e8d06f3e908c450f56623202`.
No episódio isolado `A-6784215452709343147-20260927-103447`, uma
observação natural do tick 15200 já retornou HOME `reachable` após mudança
de ~0,21 bloco na posição, sem ação. Repetido o fixture exato com mundo
congelado em `(-530.3,84,-405.5)`, tick 16000: HOME
`no_complete_path` (duas entradas seguras/duas consultas) e seis camas
alternativas livres examinadas, dez consultas, **zero caminhos completos**.
Portanto a simples realocação de cama não resolve a partida nesse ponto.
Comando manual `walk_to(-530.5,84,-403.5)` seq1 foi rejeitado no mesmo
tick com `reason=unreachable_target`; não houve ação aplicada. A observação
seguinte de HOME `reachable` surgiu com pequena oscilação de posição, mas
o ser não percorreu o waypoint. A cópia foi parada limpidamente.

Conclusão limitada: o Pathfinder pode ficar sem rota completa a partir
daquele estado preciso, inclusive para uma posição aberta a ~2 blocos.
A mudança sub-bloco e outros estados do mundo tornam a classificação
instável, então não afirmar barreira geométrica permanente. Próxima
hipótese a testar é um controle físico local e seguro (sem teleporte)
para sair do ponto de partida, seguido de nova consulta de rota e sono.
Esse mecanismo deve ter contrato de ação/resultado e baseline vanilla
pareado; um empurrão escolhido manualmente continua sendo testemunha
mecânica, não IA aprendida.

## Terceiro teste preregistrado — saída física local (10:47 UTC)

No mesmo mundo isolado, testar uma única ação `nudge` para o UUID v6 acordado
em `(-530.3,84,-405.5)` com HOME `(-534,86,-413)` e noite. Direção manual
`vx=0`, `vz=0.20` (rumo +Z), sem componente vertical. O plugin só aceitará
o pulso se o aldeão estiver no chão e acordado e todo o corredor varrido de
um bloco, incluindo largura de 0,6 bloco, tiver pés/cabeça livres, chão
sólido, chunk carregado e posição dentro da borda. O pulso é único; não há
teleporte ou repetição de velocidade. Registrar rejeição ou deslocamento
horizontal em 40 ticks, saúde antes/depois e diagnóstico HOME no próximo
snapshot, seguido por até ~800 ticks de sono/resgate. Gate mecânico: deslocar
ao menos 0,2 bloco sem dano; gate de tarefa: HOME alcançável e dormir nessa
janela. Se rejeitar, registrar o motivo; não relaxar o limite sem inspecionar
os blocos. Mesmo sucesso não provará IA, pois a direção foi manual.

Resultado (10:56 UTC): `shadowJar` offline passou; jar SHA256
`98e2bf3334163d2abe895ea4fda974c930627334a3f9d6e91ff1c47e47b0be6e`.
O Paper isolado carregou `0.2.0-pilot` no episódio `...105217`, sem porta
host e com limite de 0,5 CPU. A primeira ordem seq2 foi rejeitada
`stale_or_invalid_time`: o operador usou `gametime` global (7060168), enquanto
o contrato recebe `world.getFullTime()` (16020 após `time set`). Ela não
aplicou ação e não é evidência mecânica. A ordem corrigida seq3, no fixture
exato e tick16400, foi rejeitada `nudge_not_grounded_awake`; o produtor no
mesmo tick marcou HOME `no_complete_path` com duas consultas. O frame do
aldeão registrava `sleeping=0`, `pose=STANDING`, porém `on_ground=0`.
Portanto nenhum pulso foi aplicado, nenhum deslocamento pode ser atribuído à
ação e o gate mecânico falhou sem expor o aldeão a dano. Bloco adjacente
`(-530,84,-406)` não é ar, ao lado da borda da entidade; isso ainda não
explica sozinho a falta de chão. O teste foi parado limpidamente e preservado.

Próximo gate: inspecionar suporte e colisão ao redor do centro real, inclusive
o valor fracionário de Y e o estado `on_ground`, e escolher uma posição
estável no chão antes de qualquer novo pulso. Não retirar a checagem de chão
nem concluir que a política de IA funcionou. Se o fixture só existir em
estado sem suporte, tratar o travamento como problema de geometria/mundo.

Correção de interpretação após ler os keyframes do principal: no intervalo
9229606–9236006, o v6 original estava `on_ground=1` em **238/321** amostras,
inclusive no ponto quantizado `(-530.3,84,-405.5)`; em 83 amostras estava
`on_ground=0`. Portanto a rejeição desta cópia pode refletir o instante
logo após teleporte ou diferença no estado exato, e não prova ausência
permanente de apoio no mundo. O próximo fixture deve esperar/medir
`on_ground=1` antes da ordem, sem editar blocos nem relaxar o gate.

## Quarto teste preregistrado — estado natural após restart (11:06 UTC)

Leitura RCON somente de blocos no principal mostrou chão não-ar em y83,
ar em x=-532/-531,z=-406,y84 e bloco não-ar em x=-530,z=-406,y84.
Isso sustenta direção manual oeste para fugir da borda; não atribuir a uma
política aprendida. Reiniciar a mesma cópia isolada parada **sem teleporte**
e deixar no máximo 400 ticks naturais. Só emitir `nudge` seq4
(`vx=-0.20,vz=0`) se o UUID estiver acordado, `on_ground=1`, a no máximo
0,5 bloco de `(-530.3,84,-405.5)`, com HOME `(-534,86,-413)` e
`home_route_status=no_complete_path`. A ordem deve usar `tick` da observação,
com expiração em 200 ticks. Se qualquer condição falhar, não enviar
pulso e parar a cópia. Se aceitar: comparar posição/saúde após 40 ticks,
diagnóstico HOME no próximo snapshot até 400 ticks, e sono/resgate até
800 ticks; gate mecânico >=0,2 bloco sem dano. Container sem portas,
0,5 CPU, não modificar o principal. Ao final, parar limpidamente.

Resultado (11:15 UTC): a cópia subiu sem teleporte, episódio `...111109`.
O v6 estava acordado, `on_ground=1` no frame16480, na posição quantizada
`(-530.3,84,-405.5)`, com HOME e `no_complete_path`. Congelamos no
tick16481 e enviamos seq4 para oeste. No tick16500, quando a ordem foi
lida, `on_ground=0`; evento final `pilot_reject` com
`nudge_not_grounded_awake`. Não houve `pilot_action`, deslocamento oeste,
dano ou resultado de política. O teste foi parado limpidamente; mundo e
episódios preservados. Essa oscilação mostra que o gate no instante da
ordem importa, mesmo sem teleporte. Não se mede ainda superioridade.

Mudança proposta e compilada após o resultado, ainda sem ensaio Paper:
`wait_for_ground_ticks` opcional, máximo 40, permite que uma ordem já
validada aguarde um tick no chão antes de aplicar o mesmo corredor seguro.
Caso o prazo expire, rejeita sem agir; sono/indisponibilidade também rejeitam.
`pilot_deferred` e `ground_wait_ticks` tornam a espera auditável. Jar
SHA256 `5e42fabc2551ab64c9a10e9c55f3548ddbe674686a6b4c094d880003a9eaa1ac`.
Próximo teste preregistrado deve usar a mesma posição natural, direção
oeste, espera explícita de 40 ticks e medir aceitação, deslocamento e
saúde. Um pulso manual continua sendo teste mecânico, não IA aprendida.

## Quinto teste preregistrado — espera por chão (11:17 UTC)

Reiniciar a cópia isolada parada no tick16540, com o jar SHA256 `5e42fabc...`,
sem teleporte ou edição de blocos. Quando o snapshot do novo episódio
mostrar v6 acordado a <=0,5 bloco do ponto, com HOME
`(-534,86,-413)` e `home_route_status=no_complete_path`, congelar.
Emitir uma única ordem `nudge` oeste (`vx=-0.20,vz=0`,
`wait_for_ground_ticks=40`, prazo do comando <=200 ticks). Não exigir
`on_ground=1` na amostra; o próprio gate deve esperar. Validar
`pilot_deferred` se inicialmente sem chão, depois `pilot_action` com
`ground_wait_ticks<=40` ou `pilot_reject` ao expirar/colidir. Se aplicada,
medir deslocamento horizontal >=0,2 em 40 ticks, saúde sem queda e
diagnóstico HOME/sono no próximo snapshot. Se as precondições de posição,
HOME ou noite falharem, não emitir ordem. Parar a cópia ao final.

Resultado (11:25 UTC): a cópia abriu o episódio `...112310` com o jar novo.
Entre ticks16580–16900, v6 permaneceu acordado a 0–0,3 bloco do ponto,
com `on_ground` alternando 0/1. O produtor só publicou diagnóstico do
episódio novo no tick16800: HOME `reachable` (posição 0,3 bloco distinta).
No tick16900, ainda era `reachable`; a precondição `no_complete_path`
falhou. Nenhuma ordem foi enviada, portanto **não há validação Paper**
de `wait_for_ground_ticks`. A cópia foi parada limpa, sem portas e sem
efeito no principal. O produtor a cada 400 ticks perdeu a janela com
falha de rota que existiu em frames anteriores. Próximo ensaio pode usar
cadência de observação 20 ticks **apenas na cópia** (configuração isolada,
sem alterar produção), com limite de CPU e prazo, para capturar o estado
no instante da decisão. Registrar custo da sondagem e parar se a carga
impedir o teste; não escolher ação usando observação vencida.

## Sexto teste preregistrado — diagnóstico a cada 20 ticks (11:29 UTC)

Preservar a configuração anterior da cópia desligada e mudar somente
`ai_pilot.candidate_every_ticks` de 400 para 20 na cópia isolada. Mesmo jar
`5e42fabc...`, mundo parado do episódio anterior, sem teleporte, portas ou
alteração do principal, limite 0,5 CPU/3 GiB. Observar no máximo 500 ticks
naturais após a inicialização; comparar custo de `BedCandidates.duration_ms`
e TPS efetivo do teste com a configuração anterior. Emitir uma única ordem
`nudge` manual oeste (`vx=-0.20,vz=0`, `wait_for_ground_ticks=40`) somente
quando observação do novo episódio tiver idade <=20 ticks, HOME
`no_complete_path`, v6 acordado, a <=0,5 bloco do ponto e saúde >=5.
Se o aldeão estiver no ar quando lida, exigir `pilot_deferred` e então
`pilot_action` em até 40 ticks ou rejeição de prazo; se já estiver no chão,
aceitação direta é possível. O gate de corredor permanece. Após uma ação,
medir deslocamento >=0,2 em 40 ticks, saúde, rota HOME e sono até 800 ticks;
sem precondição ou com custo excessivo, parar sem ordem. Encerrar a cópia
sempre. Isso continua testemunha mecânica manual, não política IA.

Resultado parcial (11:38 UTC): a configuração isolada foi salva em
`~/.local/state/vila-viva/validation/ai-houses-config-before-cadence-20260927-1129.yml`
(SHA256 `f339bc3b...`) e ajustada para 20. O jar anterior, porém, força
`Math.max(100,everyTicks)`: observações saíram nos ticks17000/17100/17200,
não a cada 20. No tick17200 houve HOME `no_complete_path`, mas a tentativa
de congelar terminou no tick17223, além da idade máxima preregistrada.
O diagnóstico seguinte 17300 manteve `no_complete_path`, mas a captura
repetida não congelou dentro dos 20 ticks; o limite de 500 ticks passou.
Nenhuma ordem foi enviada. O episódio `...113446` foi parado limpo.
Custos observados ~4,88 ms para o diagnóstico do tick17200; ainda não há
medida suficiente para certificar cadência20. O plugin opt-in foi corrigido
para mínimo20, default400 preservado; `shadowJar` offline passou, jar
SHA256 `4dc54c938cb5f7ba341dee02d8f3d4d25ec8382391b7f11cbf7d26afa43feb76`.

## Sétimo teste preregistrado — passos controlados (11:39 UTC)

No mesmo mundo isolado parado, iniciar jar `4dc54c...` com config20 e
congelar ao fim do startup. Avançar em blocos de 20 ticks e ler a observação
do novo episódio após cada passo, por no máximo 500 ticks. Só escrever uma
ordem manual para oeste (`vx=-0.20,vz=0`, espera40, validade<=200) se o
snapshot recém-publicado indicar HOME `no_complete_path`, aldeão acordado,
saúde>=5 e distância<=0,5 do ponto; conferir que `candidate_every_ticks`
efetivo é20 pelo incremento dos ticks e registrar `duration_ms`. A ordem
é lida no próximo múltiplo de20, ainda dentro da idade<=20; se a rota
oscilar, o resultado da ordem continua limitado ao estado observado.
Depois avançar até o evento `pilot_action` ou `pilot_reject`; se ação,
mais40 ticks para resultado, saúde e deslocamento. Deixar no máximo800
ticks após o pulso para observar HOME/sono e parar limpo. Se não houver
snapshot válido ou custo alto sob meia CPU, parar sem ordem. Mundo principal
sem mudança. Mesmo aceitação é prova mecânica, não política aprendida.

Resultado (11:48 UTC): o plugin novo carregou no episódio isolado
`...114454`, meia CPU, sem portas. Observações passaram a sair nos ticks
17700, 17720, 17740 e 17760, validando a cadência efetiva de 20.
O primeiro diagnóstico pós-start custou 197,10 ms; os seguintes
1,71/0,19/2,43 ms, evidenciando aquecimento e custo baixo nas amostras
posteriores, sem provar desempenho sustentado. No tick17760, v6 estava
acordado, saúde5, ~0,05 bloco do ponto, HOME `no_complete_path`.
Ordem seq5 manual oeste, espera40: `pilot_deferred` no tick17760,
`pilot_action` no 17761 após **1 tick** de espera por chão, e
`pilot_outcome` no 17801. Saúde 5→5, deslocamento horizontal total
0,26861 bloco, mas `x` só foi de -530,30000 a -530,33299:
**progresso oeste 0,03299**. `z` mudou +0,26657, em direção lateral,
compatível com a oscilação observada antes da ordem. HOME estava
`reachable` no tick17820, mas já alternava sem ação; não atribuir
causalidade. O aldeão não dormiu nos frames até17813; janela de sono
longa não foi ensaiada. Cópia parou limpa. Esse teste valida que a ordem
adiada pode ser aplicada sem dano, **não** que saiu do travamento ou
superou vanilla. O gate de progresso dirigido >=0,2 falhou.

A métrica original só registrava deslocamento total e podia confundir
desvio lateral com progresso da ação. `ControlPilot` opt-in passou a
registrar `directed_progress`, `lateral_displacement`, `start_tick` e
`window_ticks`; `shadowJar` offline passou, novo jar SHA256
`31200a843913b3bb596a032c21bc18635fb5cfc4dcaf665b1ca37685a12af4a4`.
Essa instrumentação ainda não rodou em Paper. Próximo teste: base parada
pareada antes da ordem, um braço vanilla e um piloto, e comparar
**componente oeste**, acesso à HOME, sono e dano em janelas iguais.
Não conectar política aprendida ou promover produção por deslocamento
total. O custo de boot sob 0,5 CPU foi ~260 s, fator operacional para
reduzir reinícios isolados desnecessários.

## Par exploratório preregistrado — deslocamento dirigido (11:53 UTC)

Fonte única: `~/.local/share/vila-viva-ai-eval-20260927/night-route-base`, uma
cópia parada consistentemente antes do primeiro pulso. Copiar, sem apagar a
fonte, para dois diretórios novos. Usar o mesmo jar `31200a84...`, configuração
de observação a cada 20 ticks, limite de 0,5 CPU/3 GiB e nenhuma porta
publicada. Executar **sequencialmente** um braço sem ordem e um braço com
`nudge` oeste `vx=-0.20,vz=0,wait_for_ground_ticks=40`, sempre para o mesmo
UUID. Ajustes de horário/posição, se necessários para reproduzir o fixture,
devem ser idênticos e registrados nos dois braços. Medir posição inicial,
saúde, HOME/rota, estado no chão, relógio e diferença já existente antes da
ordem. Janela primária: 40 ticks após a oportunidade de ação, com progresso
oeste `x_inicial-x_final`, desvio lateral `abs(z_final-z_inicial)` e saúde.
Janela secundária: até 800 ticks para HOME acessível, sono, resgates e dano.
Se os estados iniciais divergem materialmente, se a rota não tem o estado
requerido ou se a ordem é rejeitada, classificar o par como inválido para
inferência de benefício. Uma única comparação exploratória não estabelece
superioridade; repetição e política aprendida são gates separados. Parar cada
cópia de forma limpa e preservar seus dados/episódios.

Braço vanilla concluído às 12:03 UTC. Mesmo jar `31200a84...`, 0,5 CPU,
3 GiB, sem portas. `time set 16000` na cópia congelada; sem teleporte.
No tick16060, v6 saúde5 e HOME `no_complete_path` em
`(-530.3000,84,-405.5020)`. No instante de oportunidade16080, HOME
oscilou para `reachable`, posição
`(-530.3000000119,84,-405.5012123)`; no tick16120, x igual e
z `-405.5007366`, progresso oeste **0**, lateral 0,00048. Até o tick16880,
saúde5, nenhum resgate, zero sono em 40 frames do UUID (ticks16089–16869),
posição final `(-530.3000000119,84,-405.5051390)` e HOME `reachable` no
snapshot final. Episódio `...120013`, parada limpa. Artefato estruturado
`~/.local/state/vila-viva/validation/night-nudge-pair-20260927-1153.json`.
O braço piloto ainda está iniciando; nenhum efeito causal concluído.

Braço piloto concluído às 12:12 UTC, mesmo jar/limites/save e mesmo ajuste de
horário, sem teleporte. V6 no tick16040 estava em
`(-530.3000,84,-405.5037)`, saúde5, HOME `no_complete_path`;
no tick16060, posição `(-530.3000000119,84,-405.5021953)` e HOME
`reachable`. Ordem manual oeste seq1 foi lida e **aplicada no tick16060**
(sem espera por chão); o resultado no16100 registrou 40 ticks,
`directed_progress=0`, deslocamento horizontal0,00154 apenas em z, saúde5→5.
A instrumentação nova funcionou em Paper. No tick16880, v6 ainda acordado,
saúde5, HOME `reachable`, nenhum resgate; zero sono em 40 frames. A posição
final teve z `-405.1407`, mas x continuou `-530.3000`.

Os braços partiram de posição/saúde quase idênticas, porém o estado
`HOME` oscilou em fases diferentes (vanilla `no_complete_path` no16060 e
`reachable` no16080; piloto o inverso). Além disso, as janelas primárias
capturadas não coincidem: piloto16060–16100, vanilla16080–16120. Portanto
o par **não sustenta inferência causal de melhora**; a conclusão mecânica
é menor e direta: o pulso aplicado falhou no gate de avanço oeste. Ambos
foram parados limpos e preservados. Não repetir esse pulso nem conectar o
bandit de camas a essa ação sem mudar tarefa/mecanismo e produzir baseline
comparável. O principal segue sem controlador IA.

## Contrato de aproximação HOME — preregistro (12:19 UTC)

O produtor Java opt-in passou a incluir `home_approach` somente quando o
diagnóstico de HOME encontra caminho completo até uma entrada segura:
coordenadas centrais `x/y/z`, distância e quantidade de pontos do mesmo
caminho já consultado. Caso contrário é `null`, sem segunda busca, sem
comando. `shadowJar` offline passou, jar SHA256
`b130a0b2a495ace335f547ec1bf8b686fc4bf85a9ff51a2a167e4f23a35a90b1`.
O runtime Rust do Universe não muda: ainda decide apenas entre camas para
aldeão sem casa, e não deve consumir a aproximação HOME como candidato
de cama. Contrato `vila-bed-candidates/v1` ganha campo opcional; ver
`vila-viva-server/docs/AI-CONTROL-PILOT.md`.

Teste planejado: copiar novamente `night-route-base` parado (SHA256 de
`world/level.dat` `85f5f621...`) para diretório novo, config isolada de
20 ticks, iniciar Paper com 0,5 CPU/3 GiB e nenhuma porta host. Congelar,
ajustar a cópia para noite (`time set 16000`) e avançar até uma observação
`home_route_status=reachable` para v6 acordado, saúde>=5, a <=0,5 bloco do
ponto. Verificar `home_approach` não nulo, finito, distância<=16,
`path_points<=64`, ponto distinto da posição atual e `home_path_queries`
sem acréscimo por serialização. Quando `no_complete_path`, exigir
`home_approach=null`. Medir custo da geração; preservar mundo/episódio e
parar cópia. **Não enviar ação neste gate.** Um ensaio de `walk_to` com
baseline separado depende do resultado deste contrato.

Resultado do contrato (12:26 UTC): cópia `home-approach-1219`, episódio
`...122422`, jar `b130a0...`, 0,5 CPU/3 GiB, sem portas. No tick16000,
HOME `reachable` publicou aproximação `(-532.5,86,-412.5)`, distância7,60,
12 pontos e uma consulta HOME. No tick16020, HOME `no_complete_path`
publicou `home_approach=null`, duas consultas HOME; no16040 voltou a
`reachable` com o mesmo ponto, distância7,60, 12 pontos, uma consulta e
0,148 ms de geração. JSON finito e limites passaram; nenhum comando foi
enviado. O primeiro diagnóstico visto no16000 custou0,564 ms; no16020
0,291–0,981 ms conforme captura assíncrona. Cópia parada limpa.
O campo foi validado em Paper; sua utilidade como ação ainda não.

## `walk_to` para a aproximação HOME — preregistro (12:28 UTC)

Criar uma **nova** cópia do save de teste parado no16040, preservando o
original, e iniciar o mesmo jar com 0,5 CPU/3 GiB, sem porta publicada.
Congelar após boot; manter noite na cópia, sem teleporte. Avançar no máximo
500 ticks até uma observação fresca `home_route_status=reachable`,
`home_approach` não nulo com distância entre2 e16, v6 acordado/saúde>=5,
posição a <=0,5 bloco de `(-530.3,84,-405.5)`. Emitir **uma** ordem manual
`walk_to` para o ponto observado, velocidade0,6, validade<=200 ticks,
episódio/UUID/tick exatos. Medir `pilot_action`/rejeição, caminho completo,
distância inicial, chegada <=1 bloco/progresso em200 ticks, refreshes,
saúde e sono até800 ticks, resgates e custo de observação. Se a precondição
falhar, parar sem ação. Sucesso mecânico não é política aprendida nem
vantagem sobre vanilla; requer baseline posterior da mesma base. Parar
limpo e preservar mundo/episódio.

Resultado da ação (12:37 UTC): nova cópia `walk-home-1228`, episódio
`...123249`, sem teleporte, mesmo jar/limites e parada limpa. No tick16040,
v6 saúde5 estava a ~0,004 bloco do ponto; HOME `reachable` publicou
`(-532.5,86,-412.5)`, distância7,60 e 11 pontos. Ordem seq1 com velocidade
0,6 foi aplicada no16060: `path_accepted=true`, 11 pontos, distância real
7,7298. O resultado no16260 teve cinco refreshes (uma falha),
`remaining_distance=7,8701`, progresso **-0,1402**, `arrived=false`,
saúde5→5 e HOME preservada. Até16860, zero sono/resgate em43 frames e x
permaneceu `-530.3000`. O caminho completo do Paper não se converteu em
locomoção neste fixture; vanilla mantinha `walk_target` da HOME enquanto o
piloto tentava conduzir. Logo falhou o gate mecânico, sem baseline de
superioridade e sem motivo para conectar a política Rust a essa ação.
Leitura somente de blocos ao redor no principal indicou pés em corredor de
ar estreito com paredes a norte/leste; isso sugere colisão/Brain em disputa,
mas não identifica a causa sem trajetória de waypoints. Próxima hipótese:
testar saída local segura para sul/sudoeste ou outro aldeão/fixture onde
o comando já mostrou eficácia, com caminho/primeiros waypoints registrados;
não repetir `walk_to` para a mesma HOME esperando ganho espontâneo.

## Diagnóstico da rota e cadência de refresh — preregistro (12:43 UTC)

`ControlPilot` opt-in passou a registrar somente os seis primeiros
waypoints e a posição inicial no evento `pilot_action` de `walk_to`, sem
mudar o movimento. Build offline passou, jar SHA256
`d8cea6b77ba36e81ea383e34e6c7b4a69c395c7ad07f2169fb277aa8199f0e22`.
Em uma **nova** cópia parada de `home-approach-1219`, preservar a fonte e
ajustar só `refresh_every_ticks:40→20`; `candidate_every_ticks:20` já é
isolado. Paper com 0,5 CPU/3 GiB, sem portas, uma cópia por vez. Depois
do boot, congelar, exigir v6 acordado/saúde>=5 a <=0,5 bloco de
`(-530.3,84,-405.5)` e observação fresca de HOME `reachable` com
`home_approach` não nulo/distância2–16. Uma ordem manual `walk_to`
velocidade0,6, até200 ticks, depois até800 ticks para sono/resgates.
Critério mecânico: chegar <=1 bloco sem dano e sem degradação do TPS/custo
da cópia; se não, registrar progresso, refreshes/falhas, primeiros pontos
e posição/ground do v6. Se precondição falhar em500 ticks, nenhuma ordem.
Este é contraste exploratório de cadência com o ensaio anterior40, não
comparação causal perfeita porque o boot pode divergir. Não conectar a
política, não tocar o principal, parar limpo. Se o caminho começa em
direção bloqueada, priorizar correção de ação/fixture em vez de aumentar
frequência indefinidamente.

Resultado refresh20 (12:51 UTC): cópia `walk-refresh20-1243`, episódio
`...124817`, mesma fonte do ensaio refresh40, jar `d8cea6...`, parada
limpa. Ordem aceita no16060 a 7,7298 blocos; preview dos seis primeiros
nós: `(-531,84,-406)`, `(-531,84,-405)`, `(-530,85,-406)`,
`(-530,85,-407)`, `(-530,85,-408)`, `(-530,86,-409)`.
No16260 faltavam7,6052: progresso0,1246, nove refreshes/três falhas,
`arrived=false`, saúde5→5. Zero sono/resgate em43 frames até16860,
x=-530,3. Refresh20 produziu leve melhora numérica frente a -0,1402 do
refresh40, mas ambos falharam em sair do mesmo local; não se pode atribuir
a diferença à cadência com uma cópia por braço. O trajeto sobe de y84 para
y85 junto à parede; escolher alvo local em piso nivelado para separar
colisão/geométrica de disputa com Brain.

## Passo local nivelado — preregistro (12:52 UTC)

Nova cópia de `home-approach-1219` parada, jar `d8cea6...`, refresh20,
0,5 CPU/3 GiB, sem portas. Após boot, congelar e exigir v6 acordado,
saúde>=5, a <=0,5 bloco de `(-530.3,84,-405.5)`; sem teleporte.
Testar **uma** ordem manual `walk_to(-532.5,84,-404.5)` velocidade0,6,
200 ticks, somente se Paper aceitar caminho completo e alvo tiver
pés/cabeça livres/apoio sólido. Esse ponto de piso plano fica a oeste/sul
do local e a ~2,4 blocos, fora da subida mostrada no preview. Medir
waypoints, chegada<=1 bloco, x oeste>=0,5 e z sul>=0,5, saúde, rota HOME,
sono/resgates até800 ticks, custo e refreshes. Se rejeitar/nenhum progresso,
parar, não aumentar frequência de novo; investigar autoridade do Brain ou
fixture. É teste mecânico manual, sem política ou alegação de superioridade.
Preservar original e encerrar cópia limpa.
