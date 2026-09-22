# Revisão do painel — contratos, investigação e experiência de uso

21/09/2026 — terceira revisão, foco no painel. Handoff para Claude.

**Plano mais recente:** [corpo, métricas, números e painel de objetivos](PLANO-TELEMETRIA-E-PAINEL-CLAUDE-20260921.md), com inventário de sensores reais e checkpoint de aceitação. Usar como entrada para a implementação; manter os achados abaixo como requisitos de correção.

Complementa [a auditoria de replay](REVISAO-REPLAY-CLAUDE-20260921.md). Esta revisão não modifica código. Inspecionei fontes e fiz leituras pequenas dos artefatos existentes; não rodei treinamento, testes, navegador automatizado ou benchmark. Achados de interação abaixo são **estáticos**, não relatos de cliques executados. Não houve inspeção visual nova nem medição de contraste/FPS.

## Direção de produto

O painel deve permitir responder: **quem estou observando, o que percebeu, que ação produziu, quais transformações ocorreram e qual foi o resultado no corpo/mundo?** Hoje há componentes úteis, mas vários contratos entre eles estão incompletos. Organizar isso é mais importante que aumentar a quantidade de janelas.

Entrega recomendada: uma interface de investigação com seleção e relógio compartilhados, evidência acessível e estados de indisponibilidade precisos. Preservar Vue/Dockview/Three.js; não reescrever o frontend para resolver problemas de integração. O requisito de cérebro completo continua: saídas de módulos ajudam já, mas não substituem observação dos seus estados internos.

## Achados novos, com consequência e aceitação

### P0 — painéis podem mostrar ticks neurais diferentes

- `lab-ui/src/flow/ticks.ts:sampleIndexForTick` escolhe amostra anterior; é usado pelo cérebro 3D.
- `lab-ui/src/panels/data.ts:sampleIndexForTick` escolhe a mais próxima; retina, pedais e estágios usam essa função.
- Exemplo estático com o manifesto real `[64,128,…]`: no tick **100**, cérebro mostra **64**, enquanto retina/pedais mostram **128**. Deslocamento de um segundo de simulação entre painéis sob um relógio comum.

**Aceitação:** resolver o contexto temporal uma vez no store e compartilhá-lo: tick solicitado, tick físico, tick sensorial/neural utilizado, fase e defasagem. Testar tick 100 explicitamente, limites, lacunas e perfil com amostragem não uniforme. Distinguir fora do intervalo de amostra válida; não prolongar silenciosamente atividade após morte/fim da gravação. Isso é pré-requisito para qualquer comparação visual de causa e efeito.

### P1 — seleção não fecha o caminho cérebro → inspetor

- `Brain3DPanel.vue:onClick` emite `select-neuron`, mas `PanelHost.vue:microExtraProps` não registra `onSelectNeuron`.
- O mesmo host conecta seleção dos pedais ao fluxo de contribuição: existe mecanismo compartilhado, porém falta conectar o cérebro.
- `Track2DPanel.vue` não tem hit-test/listener para selecionar o ser clicado; trocar o dropdown de traces é outro mecanismo, restrito a gravações disponíveis.

**Aceitação:** clicar num neurônio atualiza inspector, destaque 3D e contribuição com o mesmo `(run,gen,being,lineage,tick,stage,index)`. Clicar num carro seleciona seu corpo; se não possui trace neural, manter dados físicos e dizer “atividade neural não gravada para este ser”. Não continuar mostrando o cérebro do campeão como se fosse do carro clicado. Disponibilizar seleção textual/teclado alternativa ao canvas.

### P1 — comparadores estão expostos, mas não integrados

- `PanelHost.vue` passa `runStore.panelSource` também como `ghost`: comparação consigo mesmo e no mesmo tick. `GhostPanel.vue` ainda anuncia por padrão “mesma fixture, deslocada no tempo”, embora o host não aplique deslocamento.
- `GenomeDiffPanel.vue` exige `weightsA` e `weightsB`, mas o host genérico só envia `source/tick`. Assim o painel declara “weights.json ausente” mesmo quando o pacote tem pesos; falta a segunda fonte e sua ligação, não necessariamente o arquivo.

**Aceitação:** seletores A/B com identidade e hash de genoma, alinhamento explicitado e incompatibilidades informadas. Para condução, oferecer alinhamento por tick ou progresso na pista; para pistas diferentes, não sobrepor coordenadas como se fossem equivalentes. Enquanto não conectado, mostrar “selecione duas gravações”, sem auto-comparação disfarçada. Comparar A=A deve produzir diferença zero e identidade claramente igual.

### P1 — “neurônios mortos/saturados” faz inferência inválida

`DeadNeuronsPanel.vue:computeStats` usa `|v|<1e-3` e `|v|>0.95` para todos os blocos com `stage`, exceto motor. Isso inclui inteiro `l1_to_l2`, somas pré-ativação, estado CfC e até o escalar `l2_std`, sem contratos de faixa comuns.

**Verificado**, pacote `integration-champion-20260921/trace/gen0000/being00/champion`:

| Bloco | Faixa lida | Percentual que a regra atual chama saturado |
|---|---|---|
| `l1_to_l2` (i8, 17.952 valores) | −107…111 | **87,58%** |
| `cfc` (f32) | −1,5518…1,6241 | 14,59% |
| `l2_std` (f32, estatística) | 0,6382…1,0025 | 2,67% |

Nenhum valor de `l1_to_l2` chega a `|126|`. Essa comparação não define sozinha o limiar correto; demonstra que 0,95 aplicado a inteiros não mede saturação de quantização. CfC não tem necessariamente domínio normalizado; `l2_std` não é neurônio. Percentual de amostras próximas de zero também não prova neurônio morto ou inútil.

**Aceitação:** renomear como atividade, faixa e ocupação; definir métricas por estágio/unidade. Separar silêncio observado, valor constante, clipping, saturação da não linearidade, quantização e ausência de dado. Exibir janela e número de amostras. “Morte funcional” exige evidência adicional, não só threshold universal. Não tomar decisões de poda/evolução com a métrica atual.

### P1 — barras de freio escondem o sinal antes da aplicação física

`PedalsPanel.vue:DOMAIN` fixa freio `[0,1]` também para `raw_intent` e `after_head`, que podem ser negativos. `draw()` usa `Math.max(1, t*usableH)`, tornando valores negativos barras mínimas semelhantes a zero. O tick 64 tem intenção de freio −0,458 e saída da cabeça −0,385, conforme auditoria anterior.

`suppressed()` compara `raw-final > 0.02` para todos os canais. Para direção negativa reduzida em magnitude (`−0,8 → −0,4`), não acusa redução; para transformações não lineares, diferença de valores também não estabelece motivo.

**Aceitação:** domínios por estágio, eixo zero explícito, valor assinado e unidade; separação visual de intenção versus atuador. Consumir `brake_suppression` e `override_cause` já registrados para explicar motivos. Redução de magnitude, inversão de sinal, clamp e filtragem são transformações distintas.

### P1 — o contrafactual real não tem contrato de ponta a ponta

- `CounterfactualPanel.vue` inicia em modo toy e manda `trace` como objeto e `remove` como objetos `{stage,index}`.
- `neuromod-lab/src/lab_api/routes/counterfactual.py:CounterfactualRequest` exige `trace: str` e `remove: list[str]`; handler é stub 501.
- A UI faz fetch próprio sem `x-csrf-token`; a API exige esse header no modo LAN, conforme `auth.py:require_csrf`. No loopback essa proteção é no-op.
- O pedido usa `manifest.profile` sem resolver o sufixo de linhagem. Não há invalidação de `apiResult` por mudança de tick/identidade no componente.

**Consequência estática:** nem o pedido atual corresponde ao schema; não confundir 422/403 com inexistência matemática de contrafactual. Não executei o POST nesta revisão.

**Aceitação:** uma especificação única de operação, identidade completa e disponibilidade declarada pelo backend. No laboratório real, indisponibilidade deve ser o estado inicial se não há executor; toy deve ser exemplo explicitamente separado. Um resultado deve ficar preso ao experimento original, identificado, ou ser invalidado ao trocar contexto. Um futuro contrafactual GPU precisa restaurar estado recorrente, fase e RNG pertinentes; remover peso em rede toy não estima efeito causal sobre a corrida real. Não usar oracle/simulação CPU neste fechamento.

### P1 — cores e contadores podem sugerir mudanças que não aconteceram

- `Brain3DPanel.vue:updateColorsForCurrentTick` normaliza por máximo absoluto **de cada plano no tick atual**. Sinal de 0,001 pode ficar tão luminoso quanto 1,0 em outro tick. Isso destaca posição relativa, mas impede comparação temporal sem legenda.
- `computeTopKEdges` refaz top-k global por tick. Aresta desaparecer pode ser apenas mudança de ranking.
- `neuronCount` conta instâncias dos vários estágios desenhados. Representações sucessivas de L1 e estatísticas auxiliares não são novos neurônios biológicos. O rótulo atual anuncia “neurônios”.
- `flow/contribution.ts` calcula `a*w`; para uma spline mapeada apenas como topologia 1:1 com peso unitário, esse produto não é a transformação não linear nem atribuição causal exata. O 3D trata essas arestas com brilho da saída, diferentemente desse cálculo.

**Aceitação:** modo de escala fixa por estágio para comparar ticks; modo relativo com legenda persistente. Fixar arestas selecionadas e separar topologia de contribuição. Contar “valores/nós exibidos” e, separadamente, unidades únicas quando houver mapeamento. Na spline, exibir entrada, saída e função; reservar soma de contribuições para operação linear compatível. Não chamar `peso×ativação` de prova causal.

### P2 — cobertura de sensores não é probabilidade de colisão

`SensorCoveragePanel.vue:computeCoverage` usa `distWallF/speed` até mesmo em frames cujo probe retorna seu teto (“nenhuma parede detectada”). A função compartilha o nome TTC com o mapa. Sem parede detectada, 50 m/30 m/s = 1,67 s não é previsão de impacto em 1,67 s. Velocidade escalar também não é necessariamente velocidade de aproximação normal à parede.

**Aceitação:** separar nenhuma detecção, desconhecido, aproximação e tempo geométrico aproximado. Rotular `distância/velocidade` como aproximação sob suas hipóteses. Não colocar inválido/NaN automaticamente em “seguro”. Cobertura de um probe frontal não equivale à cobertura de todos os sensores ou à compreensão de curva.

## Proposta concreta de organização do painel

Manter Dockview e disponibilizar três layouts por tarefa, com restauração do arranjo do usuário:

1. **Condução:** pista dominante, carro selecionado, retina, pedais/velocidade/vida/combustível e eventos. Pergunta: “o que aconteceu nesta curva?”.
2. **Cérebro:** 3D dominante, árvore pesquisável de módulos, inspetor de neurônio/operação e séries temporais. Miniatura da pista mantém o contexto corporal.
3. **Comparação:** duas fontes identificadas, alinhamento, métricas e diferenças. Somente quando houver duas fontes válidas; sem placeholders como resultados reais.

Esboço do layout de condução (proposta, não tela implementada):

```text
Corrida / geração / ser / vida | tipo de cérebro | cobertura dos dados
Play · pausa · 1x · tempo em s · tick físico · tick observado · copiar link
┌────────────────────────────┬───────────────────────────┐
│ Pista, carro e sensores     │ Decisão no tick           │
│ seleção / seguir / enquadrar│ intenção → ação → corpo   │
│ mapa e trilhas independentes│ freio, velocidade, motivo │
├────────────────────────────┼───────────────────────────┤
│ Retina + legenda e cursor   │ Cérebro 3D / inspetor      │
└────────────────────────────┴───────────────────────────┘
Linha temporal: voltas | contato | freio | nascimento/morte | seleção A–B
```

O cabeçalho deve mostrar identidade **carregada**, não só parâmetros pedidos. Informação crítica — trace parcial, outro tick, sem atividade neural do ser — não deve depender de abrir `<details>`. Metadados extensos, hashes e termos internos ficam no inspetor de evidência.

No 3D: enquadrar tudo, enquadrar seleção, ocultar/exibir módulo, transparência, escala de cores e legenda, seleção persistente, entradas/saídas do nó, busca por índice/nome, valores exatos e evolução temporal. É um grafo funcional esquemático, como a UI já informa; coordenadas anatômicas só quando houver modelo e dataset anatômicos correspondentes. Não renomear camadas artificiais como regiões de mosca.

## Fluxo de investigação que precisa funcionar na interface real

**Caso “por que não freou?”**: selecionar carro → escolher curva/evento → pausar alguns segundos antes → observar amostra sensorial com fase correta → selecionar canal de freio → seguir intenção, cabeça, modulação e comando → verificar velocidade/contato depois. Mostrar dados ausentes precisamente no ponto onde a explicação termina.

**Caso “ele aprendeu?”**: escolher gerações/linhagens comparáveis → manter mundo e horizonte explícitos → comparar voltas, sobrevivência, colisões, velocidade de curva, uso de freio e custo corporal. Campeão isolado e clones não viram evidência de evolução populacional. A UI deve conseguir apresentar distribuição e número de seres, não somente o máximo.

**Caso “compartilhar um bug”**: copiar link com corrida, geração, ser, linhagem, tick e seleção. Hoje `MicroView.vue` observa identidade e perfil, mas não implementa restauração de tick/seleção pela URL. O link precisa abrir exatamente a evidência, ou explicar por que ela não está disponível. Não salvar tokens/sessão na URL.

## Desempenho e estados: corrigir com medição, sem travar a máquina

Achados estáticos, sem benchmark:

- `stores/run.ts:openApi` carrega todos os blocos inteiros com `Promise.all`, apesar de existir loader por intervalo. Escala de pacotes longos precisa de janela/cache limitado, carregamento por necessidade e cancelamento de requests obsoletos. O controle de sequência evita sobrescrita tardia, mas não cancela transferência.
- `useCanvas.ts:render` reatribui width/height do canvas em todo redraw, mesmo sem resize; separar redimensionamento de desenho.
- `Brain3DPanel.vue:animate` mantém RAF até desmontar. Não há guarda local para pausa, aba oculta ou painel não visível. Confirmar ciclo de montagem do Dockview antes de atribuir consumo de abas ocultas. Renderizar quando dados/câmera mudam, com janela curta para damping.
- O hook `measureFrames` mede chamadas de render, não todo o custo de carregar, escolher amostra, atualizar cores, ordenar arestas e manter a UI responsiva. Não usar sua mediana como FPS completo do produto.
- `PanelHost.vue` mostra “carregando traço” sempre que não há `panelSource`, mesmo após erro no store. Exibir erro, vazio, sem observação, parcial e loading como estados distintos, com ação de recuperação.

Aceitação proposta: pausar realmente reduz trabalho; scrub rápido permanece responsivo; memória não cresce sem limite ao alternar fontes; janela neural carregada e volume em memória aparecem no diagnóstico. Registrar navegador/GPU/resolução/DPR, bytes, tempo de abertura, latência de scrub e p95 de atualização. Não fixar alegações de desempenho sem medir. Nenhum benchmark pesado nem fallback de simulação CPU para esta tarefa.

Acessibilidade e uso diário: `PedalsPanel.vue:195` tem canvas clicável sem equivalente de teclado; cérebro faz raycast de mouse sem seletor textual integrado; erros assíncronos do cabeçalho não anunciam estado via região acessível. Adicionar ações sem gesto obrigatório, foco visível, nomes de controles e texto alternativo de valores. Não depender apenas de cor. Referência auxiliar: [Web Interface Guidelines](https://raw.githubusercontent.com/vercel-labs/web-interface-guidelines/main/command.md); achados científicos e de dados acima vêm das fontes locais, não desse guia.

## Sequência de fechamento para Claude

1. **Contexto único:** identidade e amostragem compartilhadas; corrigir 1x/mapa/projeção da auditoria anterior. Prova: tick 100 consistente entre todos os painéis.
2. **Investigação utilizável:** seleção do carro/neuronio, inspetor motor com dados existentes e escala correta. Prova: seguir um caso real de freio sem perder identidade/fase.
3. **Métricas honestas:** retirar inferências inválidas de saturação/morte, TTC e contagem de neurônios; tipar operações e unidades.
4. **Comparadores reais:** conectar A/B, corrigir estados vazios; alinhar contrato de contrafactual e declarar executor ausente até implementado/validado.
5. **Layout por tarefa e navegação reproduzível:** foco, fullscreen/enquadramento, URL de evidência, teclado, erros recuperáveis.
6. **Validar a rota completa:** não somente componentes montados isoladamente. Exercitar cache antigo/layout salvo, 1440×900 e ultrawide, troca de ser sem trace, perfil de uma única amostra, nascimento/morte, falha de API e recarga do link.

Resultado esperado: relatório final com cada item como corrigido e demonstrado, pendente ou não aplicável; evidência do navegador e pacote usados. Não declarar “painel completo” só porque todas as abas abrem. Para fechar o requisito integral do cérebro, ainda é necessário inventariar e instrumentar os estados internos ausentes — explicitá-los na UI é necessário, mas não substitui essa implementação.

## Rastreabilidade desta revisão

Leituras: `sed`/`rg` nos arquivos citados; contrato backend versus corpo do fetch; `git -C lab-ui status --short`. Estatística de faixa extraída por Python 3/stdlib, com `json` para offsets/dtypes/shapes do manifesto e `struct.unpack_from` para os blocos do `neural.trace` existente. Os percentuais são descritivos deste artefato, não testes do cérebro em CPU.

Nenhum código de aplicação alterado nesta revisão. Próxima validação discriminante: consumidor real com fonte identificada, clock corrigido, todos os painéis no tick 100 e seleção 3D chegando ao inspetor.
