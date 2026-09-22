# Plano para Claude — corpo, métricas e painel de investigação

21/09/2026. Pedido do usuário: revisar dados e apresentação, fazer brainstorming e preparar implementação posterior. **Não implementar neste documento.**

**Escopo ampliado pelo usuário após esta revisão:** água, desgaste, envelhecimento e reparação agora integram a release. Seguir [o plano único de fechamento](../organismo-taxi-fechamento.md) e [a especificação corporal/táxi](ORGANISMO-TAXI-SPEC.md). As menções abaixo a “futuro” permanecem como histórico da auditoria; não excluem esses mecanismos do novo plano.

Este é o detalhamento de telemetria; o ponto de entrada da implementação é o plano único de fechamento acima. Complementos técnicos: [defeitos do replay](REVISAO-REPLAY-CLAUDE-20260921.md) e [integração dos painéis](REVISAO-PAINEL-CLAUDE-20260921.md).

## Resultado da revisão e limite de validação

Li produtores, cálculos e consumidores e confrontei JSONL com o pacote binário existente. Não executei simulação, treinamento, build ou alteração de código.

Tentei abrir `http://127.0.0.1:8131/run/integration-champion-20260921/0/0` pelo navegador integrado: `net::ERR_CONNECTION_REFUSED`. O usuário confirmou que o endereço é esse e está desligado. **Não houve revisão visual da tela atual.** A proposta visual abaixo se baseia na inspeção dos componentes e nas necessidades expressas; precisa ser conferida no navegador ao implementar. Não reiniciei o serviço.

Conclusão principal: falta levar dados que já existem até uma apresentação interpretável. Alguns indicadores pedidos ainda não correspondem a estados físicos implementados. Adicionar um mostrador sem fonte produziria falsa confiança.

## 1. O que mostrar: inventário de métricas

Fonte principal auditada: `Universe/src/bin/gpu_evolve/driving_diagnostics.rs:record`, que grava `observation.jsonl.body`. O pacote `neuromod-lab/src/lab_api/trace/producer.py:_block_specs` conserva somente 12 canais de corpo e quatro estágios motores. “Existe no JSONL” não significa “chega à UI”.

| Informação | Fonte/situação atual | Apresentação proposta | Cuidado necessário |
|---|---|---|---|
| Velocidade | `body.speed`, norma de velocidade no plano XZ; presente no pacote | km/h, número principal; m/s no detalhe | `km/h = m/s × 3,6`. É módulo, não distingue ré |
| Acelerador | `applied_motor[0]` e `motor_trace.final_command[0]`; comando final no pacote | % aplicado e curva temporal | intenção assinada em trilha separada; não chamar pedal de força ou torque |
| Freio | `applied_motor[2]` e cadeia motora; final no pacote | % aplicado, duração, episódios e efeito na velocidade | freio positivo não garante frenagem suficiente; separar pedido e aplicação |
| Direção | comando aplicado, cadeia e `steering_filtered` | esquerda/direita, valor assinado | não rotular graus sem conversão definida; filtrado e aplicado podem divergir |
| Marcha | `body.gear`, presente no pacote/replay | R, N, 1ª, 2ª… conforme tabela de transmissão | código 0=R, 1=N; códigos 2 e 3 são 1ª e 2ª no caminho auditado |
| Motivo da troca | lógica em `context.rs`, registro detalhado ainda incompleto | automática por RPM / pedido neural / ré / bloqueada | não atribuir troca ao cérebro só porque houve mudança de marcha |
| Rotação | `rpm`, `rpm_limit`, `rpm_fraction`; limite não segue no bloco body | RPM e limite, ex.: 6872 / 8500 RPM | registrar excedente real; não limitar a leitura a 100% |
| Combustível | `fuel` e `fuel_capacity` no JSONL; só `fuel` no bloco body | `% = 100×fuel/capacidade`, absoluto no detalhe | unidades internas, **não litros**; capacidade pode variar entre seres |
| Consumo | derivável do dreno/eventos e tempo, não indicador pronto | unidades/s e por volta, após separar reabastecimentos | diferença líquida do tanque inclui entrada; não é consumo bruto |
| Temperatura do motor | `engine_temperature_c`, gravada, omitida do body empacotado | °C, tendência e efeito térmico se ativo | modelo simplificado, flag e limiares runtime; não usar faixas reais presumidas |
| Temperatura do óleo | `oil_temperature_c`, gravada, omitida do pacote | °C no detalhe do motor | temperatura **não é nível de óleo** |
| Água/líquido de arrefecimento | nenhum estado de volume/nível encontrado no caminho de carro auditado | “não simulado” em capacidades do corpo | não fabricar tanque de água a partir da temperatura |
| Capacidade de refrigeração | gene em `gpu/organs.rs:cooling_scale`, usado condicionalmente | propriedade do corpo e status ativo/desativado | capacidade não é quantidade de água restante |
| Vida restante | `body.life = life_left`, pacote preserva valor bruto | orçamento de sobrevivência; segundos equivalentes na taxa base | ticks de vida, **não saúde de 0 a 1**; penalidades e ganhos mudam previsão |
| Dano | `body.damage`, presente no pacote | % de dano acumulado com explicação | não somar a “vida”; dano também pode resultar de RPM/proximidade |
| Torque de tração | `drive_torque_nm`, gravado, omitido do pacote | N·m no detalhe técnico | não confundir com torque máximo do motor antes da transmissão |
| Massa e capacidades | `mass_kg`, `engine_max_torque_nm`, `brake_max_torque_nm` gravados | ficha do corpo, não indicadores piscando a cada tick | registrar procedência herdado/default/expressado, versão do modelo |
| Pneus | dois campos: `tyre_temperature_native` e `tyre_temperature_model` | diagnóstico avançado com unidade e modelo explícitos | não misturar modelos nem tratar valor normalizado como °C |
| Aderência/derrapagem | `terrain_grip`, `environment_grip`, slip e ângulo; pacote parcial | alertas contextualizados e tabela por evento | “peak” de roda não é direção do carro; unidade em rad ou graus identificada |
| Colisão/buraco | contact mask, impacto, `hole_hits`, CSV e eventos | lista de eventos e contagens por intervalo | contato persistente não é uma nova colisão a cada tick |
| Voltas/progresso | replay, arquivo de voltas e objetivo | voltas completas + progresso atual e distância | distância percorrida, progresso ordenado e fitness são métricas distintas |

As métricas devem declarar quatro estados separados: **medida**, **derivada**, **não gravada** e **não simulada**. Acrescentar **mecanismo desativado** quando existe implementação mas não estava ativa na corrida. Zero é um quinto caso, um valor legítimo, nunca substituto desses estados.

## 2. Checkpoint numérico para aceitar a implementação

Corrida existente: `/media/llm-data/universe-runs/neuromod-lab-taxi/integration-champion-20260921`, geração 0, ser 0, tick **2432**, tempo nominal 38 s a 64 Hz.

| Indicador | Valor de origem / apresentação correta |
|---|---|
| Velocidade | 28,2571945 m/s → **101,73 km/h** |
| RPM | 6871,6274 → **6872 RPM**, limite 8500 |
| Marcha | código **3** → **2ª** |
| Acelerador aplicado | 0,4867311 → **48,67%** |
| Freio aplicado | 0,6814047 → **68,14%** |
| Combustível | 0,68256750 de capacidade 0,78664243 → **86,77%** |
| Motor / óleo | **20,00 °C / 70,93 °C** |
| Vida | 3839 ticks → **59,98 s na taxa base**, não promessa de duração |
| Dano | 0,2331337 → **23,31%** |
| Contato / impacto | máscara 0, impacto 0 m/s naquele registro |

Comparação verificada por leitura com `json`/`struct` de Python 3, sem executar modelo: velocidade, vida, marcha, dano e RPM coincidem entre JSONL e `neural.trace.body`. Combustível passa de f64 0,6825675004623405 para f32 0,6825674772262573, diferença esperada de representação. Isso valida o transporte desses campos **nesse checkpoint**, não o gráfico, a validade física nem toda a corrida.

Texto factual que o painel pode produzir nesse instante:

> Ser 0, 38,00 s: 101,73 km/h em 2ª marcha. Acelerador aplicado 48,67% e freio 68,14%. Tanque com 86,77%. Sem contato registrado neste instante. Há aceleração e frenagem simultâneas; examine a janela anterior e posterior para avaliar o efeito.

Não escrever “freou porque viu a curva” sem evidência causal. Não exigir um LLM para esses textos: templates determinísticos com valores, unidades, condições e fonte são suficientes.

## 3. O corpo precisa de auditoria física, não apenas de novos mostradores

### Temperatura

Nesta gravação, 168/187 amostras do motor estão exatamente em 20 °C. O óleo varia entre aproximadamente 31,48 e 85,18 °C; a temperatura nativa do pneu fica em 0,5, e a do modelo opcional em 0,15. Valores constantes não provam defeito, mas precisam de status de mecanismo e fonte para interpretação.

Em `gpu/physics.wgsl`, o ramo térmico usa aquecimento associado a RPM/acelerador, resfriamento associado à velocidade e capacidade de refrigeração, piso 20 °C e redução de potência acima de 105 °C, condicionados às flags. Outros kernels têm versões semelhantes. O valor no piso pode ser efeito das equações, flag, caminho executado ou clamp: **causa não determinada aqui**. Registrar pipeline/flags efetivos, calor gerado, resfriamento e fator de potência quando necessários para discriminar. Não ajustar fórmula para o termômetro parecer mais realista antes dessa verificação GPU.

### Combustível e vida

`context.rs` aplica, quando combustível está ativo, dreno por tick `(0.002 + throttle²×0.004×engine_cost)/64`, em unidades internas. Há reabastecimento; metadata da corrida declara `refuel_laps=1` e capacidade 0,78664243. Logo o painel deve explicar reposição por volta, não apresentar autonomia infinita ou consumo negativo.

Vida perde ao menos um tick por tick vivo nesse caminho, com ganhos e custos adicionais. Mostrar “reserva na taxa base” e histórico de crédito/débito. Somente mostrar saúde física percentual se houver uma definição distinta e contratada. Uma barra de vida 0–100 que receba diretamente 3839 seria incorreta.

### Marcha e sobreposição de pedais

`context.rs` permite regras automáticas e, em um modo, pedidos neurais pelos canais 3/4 da intenção. Gravar `shift_requested`, `shift_applied`, origem, bloqueio/cooldown e modo efetivo para atribuir corretamente a troca. Uma mudança automática de marcha não demonstra aprendizagem da transmissão.

Medir acelerador/freio simultâneos em tempo e intensidade; isso não é automaticamente inválido em todo modelo, mas pode dissipar recursos. Mostrar resultado: desaceleração, torque e contato, sem aplicar correção de comportamento dentro do painel.

### Água: sugestão de evolução futura, sem mostrador fictício hoje

Faz sentido modelar líquido de arrefecimento se houver uma decisão associada: gerir temperatura, reparar vazamento, buscar manutenção/reabastecimento, escolher capacidade corporal. Antes disso, duplicaria o indicador térmico sem dar informação nova ao agente.

Uma fase futura deve definir quantidade/capacidade, transferência de calor, condições de perda/reposição, efeito na potência/dano, sensores e possibilidade de ação. Conservar unidades e balanço de recurso; um número que diminui arbitrariamente para criar dificuldade não basta. Óleo em volume, pressão de óleo, bateria e desgaste dos freios seguem a mesma regra. **Não incluir essa física nova no fechamento atual da UI.**

## 4. Proposta visual: cena + estado numérico + explicação

Três alternativas consideradas:

- Só gauges/barras: leitura rápida de estado, ruim para precisão, comparação e explicação.
- Só tabela técnica: boa para auditoria, ruim para entender curva, posição e relações no cérebro.
- **Cena e mapa 3D ligados a um inspetor numérico e narrativo:** recomendado. Gráficos temporais aparecem para a variável selecionada, sem dezenas de gráficos competindo.

```text
CORRIDA · GERAÇÃO · SER/VIDA · TEMPO · AMOSTRA · ATIVO/PAUSADO
┌──────────────────────────────┬───────────────────────────────┐
│ Pista + corpo selecionado    │ Estado do corpo               │
│ paredes, buracos, visão      │ 101,73 km/h · 2ª · 6872 RPM    │
│ seguir / enquadrar / trilhas │ freio 68,14% · aceler. 48,67%  │
│                              │ tanque 86,77% · motor 20 °C   │
├──────────────────────────────┼───────────────────────────────┤
│ Cérebro 3D funcional         │ O que aconteceu               │
│ módulos, nós e seleção      │ texto factual + eventos       │
│ entradas/saídas destacadas   │ tabela + evolução da seleção  │
└──────────────────────────────┴───────────────────────────────┘
REPLAY: tempo em segundos · tick · eventos · intervalo A–B · exportar
```

Valores nesse esboço vêm do checkpoint acima; **não é screenshot nem painel implementado**. Em viewport menor, alternar foco “Condução”/“Cérebro” preservando inspetor e contexto, em vez de encolher tudo até ficar ilegível.

### Sempre visível

Velocidade, marcha, RPM, pedais aplicados, combustível, motor, situação de sobrevivência; identidade do ser e idade da observação. Número + unidade + estado em texto. Mini-tendência discreta quando houver histórico válido; cor complementa, nunca substitui informação. Sem termômetro de água artificial.

### Ao selecionar corpo, módulo ou variável

Tabela: nome compreensível, valor atual, unidade, mínimo/máximo/média no intervalo escolhido, fonte, frequência, fase e disponibilidade. Exportar CSV/JSON dos dados selecionados, com contexto e campos ausentes explícitos. Estatísticas ponderadas pelo tempo quando intervalos são desiguais; indicar amostragem e extremos possivelmente não capturados.

Ao selecionar freio: curva de intenção e comando, motivos registrados, velocidade, marcha e contato na mesma janela. Ao selecionar motor: RPM, temperatura, torque, fatores de perda e flag térmica. Ao selecionar neurônio: camada/operação, índice, ativação, entradas/saídas e histórico. Não abrir todos esses gráficos ao mesmo tempo.

### Cérebro 3D

Manter mapa funcional, com legenda “não anatômico” quando for o cérebro nativo. Diferenciar módulos existentes, etapas de uma mesma camada e estados recorrentes; múltiplos snapshots da L1 não são múltiplos cérebros. Topologia persistente, contribuição dinâmica opcional, escala de cor explícita, busca de nó, enquadrar seleção e ligação com tabela/inspetor.

Mostrar perto de cada módulo: dimensão, frequência efetiva, amostra usada e estado medido/não gravado/desativado. Deixar nomes técnicos disponíveis, acompanhados de explicação curta em português. Ex.: “Estado recorrente — conserva informação entre atualizações”; não afirmar que determinada memória já aprendeu a curva sem teste.

## 5. Painel de objetivos: traduzir a tarefa e separar os números

Não identifiquei nesta leitura um componente único chamado “Objetivo”; tratar esta seção como desenho do resumo de tarefa/avaliação, a conferir com a tela real depois. Não remover campos arbitrariamente sem ver o painel que o usuário indicou.

Separar quatro perguntas:

1. **Qual a tarefa configurada?** Ex.: avançar pelos checkpoints na ordem e completar voltas; explicar reposição de vida/combustível realmente ativa.
2. **O que aconteceu?** Voltas completas, progresso, tempo, sobreviventes, colisões, consumo e mortes, com horizonte e população.
3. **Como é selecionado o próximo ser?** Nome e versão do objetivo, componentes e custos efetivos. Fitness é pontuação de seleção, não “QI” ou inteligência comprovada.
4. **Por que a dificuldade mudou ou não?** Critério do currículo, valor atual, limiar e evidência. Não inventar meta ou promoção automática no front.

Na corrida de referência: `fitness_mode=checkpoint`, custos de parede e impacto declarados como **zero**, combustível ativo, quatro clones do mesmo campeão, uma geração. Isso não significa ausência de dano corporal, nem é demonstração nova de evolução. O texto precisa mostrar essas condições, sem dizer que os quatro seres representam quatro aprendizagens independentes.

No caminho de custo opcional, `objective_cost.rs` usa uma divisão não linear: `progress/(1 + pain_metres/scale)²`. Portanto, não representar todo fitness como “bônus menos penalidades” se a fórmula ativa não for aditiva. Pedir ao produtor os componentes finais e a fórmula/versionamento; não reconstituir resultado a partir de rótulos.

Vocabulário de apresentação:

| Campo técnico | Nome curto e explicação |
|---|---|
| fitness | Pontuação de seleção — critério usado para escolher descendentes |
| progress_m | Progresso válido — avanço reconhecido pela regra da pista |
| distance/odometer | Distância percorrida — movimento total, inclusive trechos sem avanço válido |
| life_left | Reserva de sobrevivência — orçamento do organismo, com créditos e custos |
| raw_intent | Intenção do cérebro — antes das transformações do controle |
| final_command | Comando aplicado — entrada entregue ao corpo naquela fase |
| override_cause | Ajuste do controle — motivo registrado da transformação final |
| l2_std | Dispersão usada na normalização — estatística, não neurônio |

Ficam no detalhe técnico: offsets, hashes, flags em bits, índices de buffer, saída bruta e fórmulas completas. Ficam visíveis: tarefa, resultado, disponibilidade, condições que mudam a interpretação e critérios de sobrevivência. Não ocultar falhas de dados sob uma seção técnica fechada.

## 6. Plano de implementação para Claude

1. **Dicionário versionado de métricas:** nome, unidade, domínio, origem, fase, tick, entidade/vida, flag e regra de interpretação. Corrigir antes relógio e amostragem divergentes das auditorias anteriores.
2. **Contrato de telemetria corporal:** adicionar campos já gravados (temperaturas, capacidade, limites, torque, motivos), sem alterar silenciosamente o layout posicional antigo. Preferir bloco adicional versionado/descritor de canais e compatibilidade explícita; não incrementar índices em consumidores dispersos.
3. **Ficha numérica do corpo e inspetor:** implementar primeiro o caso do tick 2432; nomes claros, unidades e estados ausentes. Não depender de novo treino para recuperar o que já está no JSONL.
4. **Resumo de objetivos e eventos:** separar tarefa, pontuação, recursos e resultado. Criar textos factuais com templates, sem inferir causalidade pelo aspecto dos gráficos.
5. **Ligar pista, cérebro e tabelas:** seleção compartilhada e mesmos ticks/fases; modos de layout e gráficos sob demanda; exportação do recorte com identidade.
6. **Instrumentação adicional mínima:** se faltarem motivos de marcha, capacidades/flags ou fatores térmicos, registrá-los com orçamento definido. Reusar coleta agrupada por amostra; não criar leitura GPU→CPU por painel aberto ou por neurônio.
7. **Validação:** pacote → API → tela real, seguida de corrida GPU curta apenas quando dado novo ou modelo físico exigir. Respeitar a proibição do usuário de simulação/validação numérica CPU pesada. Water/coolant é etapa futura, não requisito para produzir valores honestos agora.

Critérios de aceitação:

- Checkpoint 2432 mostra os números desta revisão; marcha não exibe código 3 como “3ª”, tanque não aparece como 68,26%, vida não vira 383900%.
- Motor/óleo ausentes em gravação antiga aparecem como não gravados; água aparece como não simulada; mecanismo desativado não parece atividade nula medida.
- RPM acima do limite continua visível; frio no clamp não é rotulado automaticamente como “motor saudável”.
- Identidade, fase e amostragem são comuns à cena, 3D, pedais, tabelas e textos; o usuário consegue copiar link para o recorte.
- Consumo distingue saída e reposição; tempo na taxa base não é apresentado como previsão certa de morte.
- Contato, colisão, proximidade e dano acumulado têm contagens e significados diferentes.
- Usuário consegue responder “qual a velocidade, por que esse freio foi alterado, qual marcha, quanta reserva resta?” sem decifrar barras ou índices.
- Claude anexa evidência da tela real e limitações restantes. Abrir abas não é validação; gráficos presentes não provam que o corpo ou o cérebro estão fisicamente corretos.

## Prioridade recomendada

**Agora:** dados corporais existentes bem nomeados, números e textos, mapa/cérebro integrados, objetivos compreensíveis e relógio correto. **Depois:** instrumentação fina de trocas/termodinâmica e comparação científica pareada. **Futuro:** água, óleo em volume, manutenção e novos órgãos quando tiverem efeito físico, percepção e ação definidos.

O objetivo é um instrumento para estudar o ser. Cada novo indicador deve responder a uma pergunta verificável; cada gráfico deve levar de volta a valores e fontes.
