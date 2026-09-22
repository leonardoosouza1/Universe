# Handoff para Claude — replay, pista e cobertura do cérebro

Data: 21/09/2026. Pedido: revisar e documentar; **não implementar nesta etapa**.

**Atualização — segunda revisão, mesmo dia:** ler também a seção final “Fechamento para hoje”. Foram encontrados erros adicionais na projeção dos sensores e dados já gravados que podem completar a inspeção motora sem nova corrida.

**Terceira revisão — foco no painel:** [REVISAO-PAINEL-CLAUDE-20260921.md](REVISAO-PAINEL-CLAUDE-20260921.md). Inclui divergência de tick entre painéis, seleção 3D desconectada, comparadores sem integração, métricas inválidas de saturação e proposta de organização por tarefa.

## Conclusão

O painel ainda não é uma ferramenta confiável para julgar o comportamento completo do ser. Há defeitos concretos de tempo e representação do mundo, além de cobertura incompleta do cérebro. Corrigir esses contratos antes de interpretar as imagens como falha de evolução ou alterar recompensa, física e arquitetura neural.

Minha entrega anterior verificou transporte, identidade dos artefatos e disponibilidade de blocos, mas não fechou a validação funcional do player. HTTP 200, hashes e L1/L2 presentes não demonstram reprodução na velocidade correta, mapa fiel ou fluxo neural completo. A afirmação de integração concluída foi ampla demais.

Nesta revisão: somente leitura de fontes e artefatos existentes, mais este documento. Nenhum código, servidor ou corrida alterado; nenhuma bateria ou simulação CPU/GPU executada. A revisão não incluiu interação visual instrumentada no navegador. Tempos de reprodução abaixo são calculados, não cronometrados.

## Escopo e evidência

- UI auditada: `lab-ui`, rota `/run/integration-champion-20260921/0/0`, servida anteriormente em `http://127.0.0.1:8131`.
- Corrida: `/media/llm-data/universe-runs/neuromod-lab-taxi/integration-champion-20260921`.
- Pacote: `trace/gen0000/being00/champion/` nessa corrida.
- HEADs consultados: Universe `ad944b6`, lab-ui `310bc60`, neuromod-lab `28310ab`. Há alterações locais: esses commits **não identificam sozinhos** o código auditado.
- Bundle existente: `lab-ui/dist/assets/index-DX7PzS4r.js`; contém os rótulos de pista e o temporizador `1e3/16` encontrados nas fontes. Isso não prova qual versão está em cache no navegador do usuário.
- As conclusões sobre esta corrida não valem automaticamente para outros replays ou para o antigo player HTML independente.

Classificação: **estático** = fonte; **verificado** = leitura do artefato; **inferido** = consequência calculada; **hipótese** = falta discriminar.

## 1. P0 — “1x” depende do stride e neste pacote equivale a 16x

**Estático:** `lab-ui/src/components/Scrubber.vue:20–38` inicia com multiplicador 4 e, a cada `1000/16` ms, soma `manifest.stride * speedMultiplier`. O relógio ignora `replay.header.tickRate`.

**Verificado:** manifesto tem stride 64, 187 amostras, ticks 64…11968; replay declara 64 Hz. Assim:

```text
velocidade efetiva = 16 callbacks/s × stride × multiplicador / tick_rate
                 = 16 × multiplicador
```

**Inferido:** “1x” reproduz 186 segundos observados em aproximadamente 11,625 segundos de parede; o default “4x” em aproximadamente 2,906 segundos, sem contar atrasos do navegador. Isso explica diretamente a reclamação de avanço rápido.

Correção a implementar: relógio por tempo monotônico decorrido, taxa física registrada e velocidade selecionada. O stride serve à consulta de amostras, não à passagem do tempo. Carregar fração de tick; explicitar comportamento ao ocultar a aba; não acelerar para compensar renderização lenta.

Aceitação: após 1 segundo de reprodução em primeiro plano, 1x avança aproximadamente 64 ticks e 4x aproximadamente 256, tanto com stride 4 quanto com 64. Separar teste determinístico do relógio e medição real do navegador. O replay físico vai de tick 4 a 12000: intervalo de 187,4375 s; o horizonte simulado de 12000 ticks corresponde a 187,5 s.

## 2. P0 — painel não desenha a pista, embora diga que desenha

**Estático:** `lab-ui/src/panels/Track2DPanel.vue`:

- `bounds` usa `replayWorldBounds`, não o mapa.
- `drawSource()` liga todos os frames de cada ser com `lineTo`, inclusive o futuro em relação ao tick exibido.
- `draw()` chama esse desenho e escolhe o rótulo `pista: track.json` pela existência de `source.track`.
- Não há renderização de `voxel_grid_rle`, paredes, buracos ou eixo da pista nesse componente.

Portanto, múltiplas trajetórias, desvios e retornos podem parecer ramificações da estrada. **Inferência plausível para a aparência relatada**, não prova de qual segmento o usuário viu. Uma trilha aberta também não prova circuito aberto. Em ecologia, conectar frames apenas pelo slot pode criar linhas entre vidas distintas; confirmar isso na corrida ecológica antes de atribuir um salto específico a nascimento.

**Verificado no pacote:** seed 8005, largura 28 m, volta nominal 1638,8385 m; primeiro e último waypoint são iguais. Grid 320×320, células de 3 m, RLE com linhas válidas. Contagem: 95.686 células de grama, 5.077 de estrada, 1.633 de parede e 4 de buraco. O grid de origem é 2D (`Universe/src/world.rs:64–115`), portanto não há evidência de erro de achatamento 3D neste exportador.

**Verificado no replay:** ser 0 tem 3000 frames; contador de voltas passa para 1, 2 e 3 nos frames 3716, 7348 e 11060. `laps_gen0000.jsonl` registra fins em 3712, 7346 e 11058. Não exigir igualdade literal sem reconciliar fase/indexação do writer e amostragem a cada quatro ticks. Os registros sustentam três voltas contadas; não certificam isoladamente a qualidade do detector de voltas.

Correção: renderizar materiais do mapa em camada própria, com transformada comum para carro, sensores, largada e overlays. Trajetórias devem ser opcionais, identificadas e limitadas no tempo, interrompidas entre vidas/reset. Usar limites do mapa, não somente posições visitadas. Verificar fechamento e conectividade reais, não simplesmente fechar qualquer polilinha visualmente.

Aceitação: com trilhas desligadas ainda aparecem circuito, largura, paredes e as quatro células de buraco nos índices `[63,183]`, `[64,183]`, `[63,184]`, `[64,184]` da metadata. Respeitar `alignment: cell_center` e convenções de índices. Trilha desligada não pode deixar apenas seres e raios sobre um fundo vazio.

## 3. P1 — cone sensorial perde a pose que realmente gerou os raios

**Verificado:** `observation.jsonl` guarda `sensor_pose`. `Universe/src/bin/gpu_evolve/driving_diagnostics.rs:298–303` a exporta. Porém `neuromod-lab/src/lab_api/trace/producer.py:_block_specs` só empacota `pose` a partir de `body`. O painel usa posição/direção de `physical.replay` para a origem e orientação do cone.

Traço do ser 0:

| Tick | posição sensorial `(x,z)` | posição do corpo no pacote e replay `(x,z)` |
|---|---|---|
| 64 | `(611.382080, 712.148804)` | `(611.345947, 712.119202)` |
| 128 | `(608.034119, 710.150635)` | `(607.960754, 710.112366)` |

As direções também diferem. Isso demonstra poses de fases diferentes; não demonstra percepção errada no kernel. O cone da observação deve usar a pose sensorial daquela observação, inclusive orientação/gaze, sem aplicar a rotação duas vezes. O corpo pode continuar na pose física posterior, identificando as fases.

Além disso, `data.ts:sampleIndexForTick` escolhe a amostra **mais próxima**, eventualmente futura. Entre 64 e 128, pode mostrar retina/ativação do tick 128 com corpo anterior. Estabelecer política explícita: modo causal usa amostra passada; modo análise pode usar a mais próxima, mostrando tick de origem e defasagem. Não interpolar silenciosamente ativações ou materiais categóricos.

Aceitação: reproduzir os dois casos acima e um tick intermediário; exibir tick físico, tick sensorial e tick neural. Só então julgar se raios atravessam paredes. Esta revisão não mediu taxa de erro de interseção.

## 4. P1 — camadas presentes não significam cérebro completo

**Verificado:** manifesto contém CfC 96, L1 96 (`matmul`, `effective`, `to_l2`), L2 48 (`matmul`, `effective`, `adapted`), `l2_std`, bypass 48, intenção 96 e quatro estágios motores de 3 canais. L1/L2 não estão ausentes neste pacote.

**Estático:** `lab-ui/src/flow/adapters.ts:buildFlowWeights` cria apenas quatro transições: KAN sensorial 1:1, matriz L1, matriz L2 e cabeça motora 1:1. A própria lista `unsupported` registra junção L2+bypass, slice motor, modulação lenta e controle final. Pesos `slow_d1`, `slow_d2`, gates e parâmetros `hip` existem em `weights.json`, mas pesos não substituem estados internos gravados.

`brain3d-layout.ts:buildStageSequence` monta uma sequência de estágios; isso não resolve ramos, memória recorrente, gates ou junções. `Brain3DPanel.vue:computeTopKEdges` seleciona os maiores valores globais de contribuição, default 200. Uma transição pode ficar visualmente sem arestas por causa desse filtro mesmo estando mapeada. Ausência visual de aresta não prova desconexão do runtime.

Correção a planejar antes de novos desenhos:

1. Inventariar nós e conexões do caminho **GPU realmente executado**, com condição de ativação, frequência e buffer. Não copiar automaticamente descrições CPU/host presentes nos textos `unsupported`.
2. Representar um grafo com concatenação, slice, bypass, recorrência, transformação elemento a elemento e portas de controle. Não fingir que cada operação é uma matriz aprendida.
3. Separar topologia estrutural de overlay de contribuição. Mostrar contagem total, filtrada e carregada, por transição; permitir inspecionar uma transição sem competir pelo top-k global.
4. Declarar cobertura: gravado, derivado exatamente, parâmetro somente, não gravado, desativado, não aplicável. `manifest.absent=[]` atualmente não certifica cobertura integral: módulos fora do inventário também ficam fora dessa lista.
5. Acrescentar observação dos estados faltantes somente após definir utilidade e orçamento de gravação. Manter buffers residentes; coletar a amostra selecionada em bloco, evitando readback por camada/tick.

Não converter `lifetime_state[512]` em 512 neurônios: é um layout misto de estado. Nesta corrida `lifetime_flags=0`, `ecology=false`. Tampouco apresentar grafo nativo como anatomia real de mosca. Novas regiões exigem existência no runtime e contrato observacional correspondente.

Aceitação: ser possível seguir uma entrada até o comando final através de operações e ramos reais; qualquer trecho sem estado medido precisa estar explicitamente marcado. Testar top-k reduzido para assegurar que o painel não anuncie desaparecimento de uma conexão estrutural.

## 5. P1 — domínio temporal e troca de fonte ainda são incompletos

- `lab-ui/src/stores/run.ts:120` limita `tickRange` ao trace neural (64…11968), apesar do replay físico cobrir 4…12000. Excluir bordas silenciosamente não é reproduzir a corrida inteira. Fora da janela neural, mostrar “sem observação”, preservando reprodução física.
- `Scrubber.vue:54` pausa ao mudar `store.traceDir`; cargas API mantêm esse campo `null`. É insuficiente para garantir pausa/reset entre duas fontes API. Usar identidade efetiva de corrida/geração/ser/perfil/linhagem, não o modo de transporte.
- `framesForBeing` e `nearestFrame` percorrem arrays repetidamente e o painel redesenha trajetórias inteiras por atualização. É um custo estático evitável, ainda **não perfilado**: indexar frames por ser/vida uma vez e reutilizar mapa rasterizado. Não chamar isso de gargalo medido.

Aceitação: trocar duas corridas API durante play, trocar linhagem, selecionar ser sem trace e chegar às bordas físicas. Nenhum detalhe neural antigo deve permanecer como se fosse do novo ser; não ligar trajetórias de recém-nascido ao ocupante anterior.

## 6. P1 — exportação do mundo tem limite de fidelidade

`Universe/src/bin/track_export.rs:38–42` reconstrói `VoxelWorld::new(seed,width)`. A metadata opcional restaura `no_holes` e confere seed. Porém o simulador também aceita `VoxelWorld::new_from_spec` (`gpu_evolve/main.rs:748`), com parâmetros adicionais.

**Estático:** seed/largura não são contrato suficiente para mundos parametrizados, nem garantem reprodução histórica depois de mudar o gerador. **Não foi demonstrado mapa incorreto para seed 8005 nesta revisão.**

Próxima correção: persistir descrição resolvida/versionada do mundo ou snapshot dos dados necessários ao desenho e colisão, com hash. Consumir o mundo da geração selecionada. Não reconstruir silenciosamente mapa diferente com binário novo. Validar padrão, cenário parametrizado e mudança de mundo entre gerações.

## Fluxo efetivamente seguido na auditoria

| Etapa | Produtor → consumidor | Dado / frequência | Situação |
|---|---|---|---|
| Observação | GPU readback → `driving_diagnostics.rs` | retina, CfC/L1/L2, pose sensorial; amostra a cada 64 ticks nesta corrida | campos presentes no JSONL |
| Corpo/comando | writer físico → `.replay` | posição em metros, comando aplicado; 64 Hz, gravação a cada 4 ticks | decodificado diretamente |
| Pacote | `producer.py` → `neural.trace`/manifest | 187 linhas; retina u32, wave i8, estados f32 | pose sensorial descartada |
| Mundo | `track_export` → `track.json` | mapa estático RLE por geração | pacote presente, painel não o desenha |
| API/store | artefatos → `PanelSource` | carga do pacote, seleção e caches | fonte real; disponibilidade não prova semântica visual |
| Relógio | `Scrubber` → `currentTick` | stride × multiplicador × 16/s | velocidade incorreta |
| Desenho | `Track2DPanel`/`Brain3DPanel` | frame físico e amostra neural mais próximos | mapa omitido; fluxo parcial |

Checkpoint adicional: no tick 128, `neural.trace.final_command` e os campos finais do `.replay` são ambos `(0.65307170, 0.47826490, 0.0)`. No tick 64, ambos `(0.0, -0.50413436, 0.0)`. A concordância desses checkpoints limita a hipótese de corrupção genérica do parser; não valida a cadeia neural inteira nem o freio em curvas.

## Ordem de trabalho recomendada ao Claude

1. Corrigir relógio e seleção temporal; provar 1x e bordas físicas.
2. Renderizar o mapa real, separar trilhas, largada e vidas; provar geometria e buracos.
3. Preservar pose sensorial e fases do tick; alinhar cone com a observação original.
4. Fechar inventário de cobertura e contrato de grafo; depois conectar os estados faltantes. Validar mecanismo ativo versus desativado.
5. Rever exportação de mundo parametrizado e identidade entre gerações.
6. Abrir o consumidor real, limpar dúvidas sobre bundle/cache, gravar evidência visual e temporal. Só depois voltar a diagnosticar aprendizagem e ampliar ambiente/táxi.

Usar primeiro os artefatos existentes. Se precisar de nova simulação, executar **somente na GPU real**, em bateria curta e limitada, respeitando a restrição do usuário contra validação numérica CPU. Não executar bateria ampla para diagnosticar um bug do player. Não compensar defeito visual com piloto, reward ou freio automático.

## Como reproduzir a inspeção, sem simular

Na raiz LLM:

```sh
nl -ba lab-ui/src/components/Scrubber.vue
nl -ba lab-ui/src/panels/Track2DPanel.vue
nl -ba lab-ui/src/flow/adapters.ts
nl -ba lab-ui/src/panels/brain3d-layout.ts
nl -ba neuromod-lab/src/lab_api/trace/producer.py
nl -ba Universe/src/bin/track_export.rs
```

Os números foram extraídos com Python 3/stdlib (`json`, `struct`, `pathlib`), lendo esses artefatos existentes. Para repetir o decode físico: cabeçalho 64 bytes; entradas de 130 bytes; tag e índice do ser nos dois primeiros bytes; frame de 128 bytes; posição no offset 0, volta u16 no 32, tick u32 no 40, comando final f32×3 no 88. Todos little-endian. `tick_rate` f32 está no offset 12 do cabeçalho. Para `neural.trace`, usar offsets/dtypes/shapes do manifesto, nunca um layout presumido.

Antes de declarar concluído, guardar: URL e identidade do pacote; versão/hash do bundle; medição de 1x; imagens com trilhas ligadas/desligadas; comparação de cone no tick rastreado; cobertura de módulos do cérebro; troca de fonte/ser/vida; limitações restantes. Esta revisão é um handoff com defeitos demonstráveis, **não uma certificação do treino ou da UI**.

## Segunda revisão — fechamento para hoje

Reabri fontes e artefatos após o pedido de nova revisão. O avanço por stride de `Scrubber.vue` e o desenho somente de trajetórias em `Track2DPanel.vue` continuam presentes. Não há evidência nesta inspeção de que esses dois defeitos tenham sido corrigidos. Novamente, não executei simulações, builds ou testes; apenas leitura e atualização deste documento.

### 7. P1 — corrigir a pose não basta: a projeção da fóvea também está errada

**Estático:** `Universe/src/gpu/retina.wgsl:126–141` usa, na linha 0, com `TAN_K=1.25`:

```text
u = 2 * col / (cols - 1) - 1
yaw = tan(u * 1.25) / tan(1.25) * fov_half + gaze
```

As demais linhas usam distribuição linear. `Track2DPanel.vue:computeRayFan` seleciona precisamente a linha 0, mas desenha todas as colunas com distribuição linear e FOV constante de 120°. Não aplica a deformação tangente nem gaze. A metadata já tem `fov_degrees`, mas esse valor não é consumido ali.

**Inferido numericamente:** na coluna 94 de 126, FOV 120° e gaze zero, o kernel aponta a aproximadamente **14,536°**, enquanto a UI desenha **30,24°** em relação ao eixo frontal. É uma divergência de aproximadamente 15,704°, independente do deslocamento de pose da seção 3. Portanto, um cone pode parecer atravessar uma parede mesmo se distância e material estiverem corretos.

Correção: publicar/consumir o modelo de projeção versionado, parâmetros e pose sensorial. Usar o vetor forward/right real, convenção de yaw, gaze e pitch. Não adotar fórmula única para todas as linhas. Auditar também o significado de distância: no caminho legado há cálculos horizontais de DDA e parâmetros do raio de interceptação do chão; não adicionar `cos(pitch)` indiscriminadamente antes de explicitar esse contrato.

Aceitação: conferir colunas 0, 31, 62, 63, 94 e 125 na linha 0, uma linha periférica e gaze não zero. Só extremos simétricos não detectam a deformação errada. Comparar endpoints com o mapa usando o mesmo modelo de coordenadas.

### 8. P1 — legenda de alcance confunde deslocamento do olho com altura absoluta

**Estático:** `retina.wgsl:122` posiciona o olho em `pose.y + EYE_HEIGHT`; no ramo legado o intercepto do chão usa `-eye.y / sin(pitch)`. `data.ts:retinaGroundReachM` usa somente `eye_height_m / tan(abs(pitch))`, com fallback 0,4 m. O pacote atual não preserva `eye_height_m` em `recorder_config`, apesar de existir na metadata de origem.

**Verificado/inferido para o tick 64:** `sensor_pose.y=0,5 m`, deslocamento de olho 0,4 m e pitch −0,012 rad. O kernel legado usa altura absoluta 0,9 m: parâmetro até chão aproximadamente **75,002 m**, alcance horizontal aproximadamente **74,996 m**. A legenda informa aproximadamente **33,332 m**. Isso é alcance teórico antes de oclusão; não promete ver através de paredes. No modo de terreno físico, considerar também altura local e o contrato específico desse modo.

Correção: diferenciar `eye_offset_m`, pose vertical e altura relativa à superfície. Se não há dado suficiente, omitir a estimativa ou marcá-la como indisponível. Não usar a legenda atual como evidência de que o cérebro só vê 33 m.

Outra limitação de apresentação: `RetinaPanel.vue` chama `distanceShade` com teto padrão de 30 m (`colors.ts`). Assim, 35 m e 75 m recebem a mesma intensidade no modo distância. O valor numérico ainda existe; não é saturação do sensor. Mostrar escala/legenda explícita e permitir alcance apropriado à gravação.

### 9. Caminho curto: expor informações motoras que já existem

**Verificado nas 187 amostras:** `body.motor_trace` já contém `slow_output[3]`, `hip_bias[3]`, `pre_physics_speed_mps`, `steering_filtered`, `brake_suppression` e `override_cause`. `_block_specs` não leva esses campos ao pacote atual. Não é necessário executar nova corrida para recuperá-los desta gravação.

Exemplo tick 64:

```text
raw_intent.brake       = -0.4581913650
after_head.brake       = -0.3846862316
slow_output.brake      = -0.0888861567
after_modulation.brake = 0
final_command.brake    = 0
brake_suppression     = nonpositive_after_modulation
override_cause        = none
hip_bias              = [0, 0, 0]
```

Esse é um diagnóstico de supressão registrado, não prova de que o ser deveria frear naquele ponto. Em 185/187 amostras o `override_cause` é `damage`; **não interpretar como 185 colisões nem como freio bloqueado**. `context.rs:2575–2603` aplica dano acumulado a acelerador/direção e marca esse motivo quando `d>0`. O trecho não reduz o freio. Há 58 amostras com `nonpositive_after_modulation`, mas isso também não estabelece uma falha por si só.

Entrega prática: painel “decisão motora neste tick” com intenção → cabeça → modulação → comando aplicado → velocidade antes/depois, acrescido dos motivos, fase e idade da amostra. Acrescentar saídas lentas e bias do hipocampo ao grafo como **saídas medidas de módulos**, sem afirmar que representam todos os seus neurônios internos. Estados intermediários ausentes continuam exigindo instrumentação para completar o cérebro.

Métricas que ajudam a distinguir “não quer frear” de “não consegue frear”:

- Quando a intenção de freio é positiva, em qual estágio ela zera ou diminui?
- Com freio aplicado, qual desaceleração longitudinal ocorre e quais contatos/overrides coexistem?
- Quanto tempo antes da curva começa a frenagem, qual velocidade de entrada e qual resultado físico?
- A sequência muda entre passagens da mesma curva e com memória ativa/inativa? Esta última pergunta exige experimento pareado, não somente replay.

Não concluir competência de frenagem apenas pela presença de comando: 57/187 amostras têm freio >0,05; `events.json` tem 244 transições `brake_on` e 244 `brake_off`. São fontes com cadências diferentes e não contraditórias. A quantidade de transições merece inspeção de duração e curva temporal; sozinha não prova oscilação defeituosa nem frenagem útil.

### 10. P1 — eventos agregados não têm precisão de um tick

`producer.py:_impact_events_from_csv` gera um evento no `last_tick` de uma janela com `impact_ticks>0`. Portanto, esse tick é o fim da janela, não necessariamente o instante da colisão. A corrida de referência não tem evento desse tipo, mas o caminho existe e afeta outras gravações.

Correção de contrato/apresentação: preservar `[first_tick,last_tick]`, fonte e resolução. Para brake/lap extraídos do replay, indicar resolução de quatro ticks. Exibir janela de colisão agregada em vez de um marcador que promete causalidade exata. Não usar ordem entre eventos incertos para explicar a ação cerebral.

### Plano de execução com dependências e provas de término

| Ordem | Trabalho delimitado | Prova para aceitar |
|---|---|---|
| 1 | Relógio, intervalo físico, pausa/reset por identidade | 10 s em 1x → aproximadamente 640 ticks; independente do stride; troca de corrida durante play não reaproveita relógio anterior |
| 2 | Mapa real e trilhas opcionais | circuito permanece com trilhas desligadas; paredes/buracos correspondem ao RLE; nenhum segmento une duas vidas |
| 3 | Contrato de sensores: pose, projeção, FOV, altura e fase | exemplo coluna 94 correto; ticks 64/128 alinhados; legenda de alcance e distância coerentes |
| 4 | Empacotar motor_trace já existente e mostrar diagnóstico | valores do tick 64 reproduzidos sem nova corrida; categorias ausentes diferenciadas de zero |
| 5 | Grafo de operações e cobertura neural completa | bypass/junção/slice/recorrência explícitos; filtro visual não apaga topologia; estados faltantes medidos antes de declarar cérebro completo |
| 6 | Consumidor real e regressões | abrir URL real, acompanhar trecho com curva/freio e três voltas; verificar seleção, resize, erros e bundle efetivamente servido |

Os passos 1 e 2 não precisam esperar instrumentação neural. O contrato do passo 3 precede alterações independentes de API e UI. O passo 4 pode reutilizar esta gravação. O passo 5 distingue conectar dados existentes de coletar novos estados: não bloquear correções do replay enquanto se projeta instrumentação, nem dar como terminado o requisito completo só porque o módulo aparece com um rótulo.

**Insight de organização:** o problema recorrente é fechar componentes isoladamente sem fechar seus contratos. Para cada item, registrar `produtor → pacote → consumidor → prova visível`. Um responsável deve validar a rota final inteira. Se Claude delegar, particionar arquivos e contrato primeiro; não deixar agentes mudarem simultaneamente schema, store e nomes de estágios sem coordenação.

Os testes existentes `lab-ui/tests/values-equal-source.test.ts` verificam valores de `debugValue()` contra uma fixture. Isso é útil, mas não prova que o mapa foi desenhado ou que 1x dura o tempo certo. Comentários de `Scrubber.vue` citam `tests/scrubber.spec.ts`, que não está no caminho atual consultado. Não aceitar comentário como evidência de teste disponível. Adicionar verificações semânticas dos defeitos reproduzidos, sem disparar suíte pesada nem simulação CPU.

**Critério de fechamento hoje:** publicar um único endereço de replay, identidade do artefato e checklist com evidências dos seis passos. Registrar claramente qualquer impedimento real à cobertura integral. Acrescentar táxi, novos órgãos, recompensas e novos métodos de evolução não corrige esses contratos e deve ficar para depois deste fechamento. Prazo urgente exige reduzir trabalho paralelo sem dependência resolvida, não reduzir a prova de funcionamento.
