# Organismo-táxi: especificação integrada de implementação

**Universe — 22/09/2026:** contratos de ação/corpo integrados à física, validação de retorno e checagem ABI Rust/WGSL; 9 testes estáticos e 1 bateria GPU passaram. Escopo e pendências: [contratos executados](CONTRACTS-RUNTIME.md). Isso não conclui todos os contratos nem comprova aprendizado.

**Implementação iniciada em 22/09/2026:** entregue a base de contratos de telemetria: quantidades validadas Python/JS, registro compartilhado de unidades e semântica dos blocos, validação corporal no produtor, validação de manifesto nos consumidores e gate com fingerprints. Ver [operação do gate](../../neuromod-lab/docs/CONTRACT-GATE.md). As afirmações abaixo de “somente documentação” descrevem as revisões anteriores. Contratos Rust/WGSL completos, fisiologia nova, táxi, relógio do painel, treinamento e proteção remota de integração **não estão concluídos por esta entrega**.

21/09/2026 — responsável pela implementação posterior: Claude. Planejamento e leitura de fontes feitos nesta revisão; nenhum kernel, treino ou painel alterado.

**Revisões posteriores:** a seção 12 identifica lacunas do código e desta própria especificação; a seção 13 corrige conclusões excessivas e registra três passagens de revisão. O plano ainda não equivale a projeto implementado ou calibrado. A seção 13 delimita o alcance dos achados anteriores.

## 1. Decisão de escopo

Construir **um organismo simulado com corpo veicular regenerativo**, cérebro substituível e ambiente de táxi. O corpo tem metabolismo, recursos, fadiga, temperatura, desgaste recuperável e deterioração ligada à idade. É uma escolha explícita de mundo artificial: um pneu convencional não regenera borracha ao estacionar. Aqui a regeneração será um mecanismo corporal com consumo de energia/material, capacidades herdadas e observação verificável.

O cérebro escolhe ações para obter recursos, transportar passageiros, repousar e acessar serviços. O corpo executa recuperação automaticamente quando as condições físicas permitem. Não há uma regra escondida “vá descansar quando vida < X” controlando o campeão.

Esta versão entrega o ciclo completo. Não entrega garantia de inteligência geral, evolução ilimitada, anatomia integral de mosca ou competência perfeita em qualquer mundo. Esses conceitos não devem ser usados como sinônimo de passar os gates abaixo.

### Três mudanças distintas

- **Desenvolvimento:** o indivíduo amadurece; capacidades se expressam conforme idade, energia e genes.
- **Aprendizagem:** estado recorrente, memória e plasticidade mudam durante a vida.
- **Evolução:** mutação/seleção alteram DNA herdado entre gerações ou nascimentos.

O jovem não fica competente apenas porque o relógio avançou: maturação permite capacidade física/neural, mas a tarefa ainda depende de política, memória e seleção. O envelhecimento pode reduzir capacidade, mesmo com manutenção. Nenhum desses processos exige que o cérebro “edite o motor” diretamente.

## 2. Inventário do que aproveitar e do que falta

| Peça | Evidência estática atual | Decisão |
|---|---|---|
| Carro GPU | `physics_a.wgsl`, `physics_b.wgsl`, `physics_c.wgsl`; corpo e órgãos em `gpu/body.rs`, `gpu/organs.rs` | estender o caminho executado, não só `physics.wgsl` monolítico |
| Genes e plasticidade | `gpu/lifetime.rs`: 16 genes, 512 f32/ser; maturação normalizada e máscaras | reusar; separar idade cronológica de maturação existente |
| Nascimentos | `ecology.rs`, `bin/gpu_evolve/ecology_runtime.rs`: recursos, linhagens, resets | reusar conservando recursos; completar snapshot retomável |
| Recursos escassos | `resources.rs`: estações e transferência, ainda com decisões host | portar caminho quente necessário e generalizar estoque com cuidado |
| Táxi histórico | `crates/taxi-world/src`: água, comida, descanso, oficina, passageiros e economia | reusar semântica/configuração compatível; não transplantar pressupostos do corpo discreto |
| Execução do táxi | `lib.rs`/`sim_cpu.rs`: CPU, cérebro/observações próprios | não é runtime GPU pronto; implementar adaptador de cenário no runtime escolhido |
| Água, reparo e desgaste no carro | não há ciclo corporal completo no caminho auditado | requisito novo desta release |
| Observação | JSONL/pacote/API/UI, com falhas nas auditorias | completar e validar como parte da release, não após treinar |

Os documentos históricos que exigiam “sem GPU”, oracle CPU ou adiar integração descrevem outro objetivo. Para esta entrega prevalece a restrição atual do usuário: simulações e validações numéricas na GPU real. Não apagar histórico nem chamar o cenário novo de reprodução exata do táxi antigo.

## 3. Contratos: DNA, estado e ambiente separados

**DNA herdado**, com limites e custo: capacidade do motor/freio/tanques, refrigeração, eficiência metabólica, durabilidade de motor/pneu, taxa e eficiência de reparação, tempo de maturação, resistência à senescência; genes cerebrais já existentes de pesos, memória, aprendizagem e topologia limitada.

Começar com esse conjunto pequeno. Genes de “ser melhor em tudo” sem custo são proibidos pelo contrato: mais capacidade cobra massa/manutenção; mais durabilidade cobra construção/massa; reparo rápido cobra energia/material e gera calor; maturação lenta posterga capacidade/reprodução. Fixar relações, unidades e limites na configuração versionada. Valores numéricos são parâmetros de calibração, não constantes biológicas universais.

**Estado adquirido**, nunca confundido com DNA: idade em segundos, maturidade, combustível/energia, água, reserva de material, temperatura motor/óleo, fadiga, desgaste por componente, integridade estrutural, estado de serviço, memória, plasticidade, RNG, dinheiro, passageiro e destino.

**Ambiente:** temperatura externa, superfície/aderência, rede viária, estoques e taxas de reposição, oficinas/abrigos, pedidos de transporte, preços, oferta de material/água/energia e condições de reprodução. Genoma não pode mutar as leis, preços, juiz ou limiares de segurança global.

Criar descritores versionados `BodySpec`, `ObservationSpec`, `ActionSpec`, `WorldSpec` e `TelemetrySpec`, com IDs semânticos, unidades, dimensões e fases. Manter buffers compactos SoA/alinhados. Não ocupar silenciosamente canais 192 existentes nem reutilizar padding sem versão/layout e validação.

A migração do genoma deve passar pelo loader canônico `gpu/genome_io.rs`. Novo formato identificado; v2/v3/v4 continuam legíveis com defaults explícitos. Legado preserva comportamento no cenário legado; novo organismo requer migração explícita e versão registrada. DNA não é snapshot de uma vida.

## 4. Modelo corporal proposto

Estas são **equações de projeto**, não fórmulas já implementadas ou calibradas. Usar `dt` em segundos e registrar unidades de todos os coeficientes. Uma única implementação autoritativa por mecanismo.

### Recursos e metabolismo

Usar combustível em unidades de modelo, energia em J com conversão `q_fuel` declarada, água/material em unidades físicas definidas ou unidades de modelo explicitamente nomeadas. Não exibir litros sem essa definição.

```text
E_next = E + input_energy - basal_cost - brain_cost - motion_cost
         - repair_cost - growth_cost - reproduction_cost
W_next = W + water_received - leak_loss - phase_change_loss
M_next = M + material_received - repair_material - growth_material
```

Todos os termos de consumo acima são incrementos no passo; taxas devem ser multiplicadas por `dt`. Reservas e transferências limitadas pelo estoque real antes da atualização. Guardar balanço e erros; clamp não pode esconder criação de recurso ou consumo impossível. Não cobrar movimento duas vezes como combustível e energia independente: registrar a conversão e qual reservatório alimenta cada custo.

Metabolismo neural: distinguir proxy de custo biológico de watts reais da GPU. Se usar atividade, conexão e memória como custo, definir fórmula/escala; não afirmar que máscara de neurônio reduz tempo computacional sem perfil medido.

### Temperatura e água

Modelo inicial concentrado de motor/óleo, com capacidade térmica efetiva `C` e condutância `K` dependente da refrigeração expressa, fluxo de ar e fração de água. Água não desaparece só porque transportou calor num circuito fechado; perdas vêm de vazamento e/ou regime de fervura declarado. Escassez pode existir por reposição limitada e danos, sem vazamento obrigatório artificial em todo tick.

```text
C dT/dt = P_heat - K(W, speed, cooling_gene) (T - T_ambient) - Q_phase_change
T_next = [C*T + dt*(P_heat - Q_phase_change + K*T_ambient)] / [C + dt*K]
```

`P_heat` e `Q_phase_change` em W; `C` em J/K; `K` em W/K, não negativos. Derivar calor da energia não convertida em trabalho mais dissipação aplicável; verificar eficiência no intervalo válido. Se água sai por vaporização, contabilizar calor latente correspondente. A discretização acima trata o resfriamento linear implicitamente com coeficientes congelados no passo; não dispensa teste de convergência dos demais acoplamentos.

Movimento **pode resfriar** pelo ar e simultaneamente aquecer pelo esforço; proibir resfriamento durante uso seria uma regra incorreta. O resultado depende do balanço. Não corrigir o piso de 20 °C atual apenas desenhando outro termômetro. Condições de potência reduzida, pane e dano térmico devem vir de `BodySpec`, com histerese onde necessário para evitar alternância rápida.

### Fadiga, desgaste e reparação

Separar: fadiga recuperável, desgaste reparável e deterioração estrutural/senescência não apagada por descanso.

```text
dF/dt = exertion_rate - recovery_rate * rest_gate
dw_engine/dt = engine_load_wear + thermal_wear + lubrication_wear - R_engine
dw_tire/dt   = distance_wear + slip_wear + brake_wear - R_tire
```

Desgaste e fadiga normalizados em [0,1]; coeficientes das parcelas são taxas 1/s, construídas a partir de carga, temperatura, velocidade, deslizamento e trabalho dissipado com referências/unidades explícitas. Evitar cobrar o mesmo contato simultaneamente em vários estados sem explicar seus efeitos distintos.

O corpo inicia reparação quando vivo, em baixa velocidade/esforço, com temperatura compatível e reserva. O agente pode buscar abrigo, parar e pedir serviço; a regeneração não precisa de um botão mágico. `rest_gate` é uma condição física contínua/com histerese, não uma recompensa por estacionar.

Para cada componente, calcular reparo solicitado e repartir recursos antes de aplicar:

```text
repair_request_i = rate_i(DNA, maturity, age, temperature) * rest_gate * dt
repair_i <= current_wear_i
sum(energy_per_repair_i * repair_i) <= available_repair_energy
sum(material_per_repair_i * repair_i) <= available_material
```

Arbitragem determinística entre motor/pneus; registrar prioridade, recurso limitante e reparo efetivo. Recuperar pneu consome material; descansar sem material recupera fadiga/esfria, mas não cria borracha. Motor muito danificado pode exigir oficina; a oficina transfere material/energia e cobra tempo/recursos, não zera gratuitamente todo estado. Perda irreversível de integridade e idade permanecem.

### Maturação e envelhecimento

```text
age_next = age + dt
dm/dt = growth_gate(E,M,T) * (1-m) / tau_mature
capacity_i = inherited_capacity_i * juvenile_to_adult_i(m)
             * wear_factor_i(w_i) * age_factor_i(age)
```

Pagar crescimento antes de permitir seu incremento; se faltar recurso, limitar incremento, não gerar dívida negativa. `juvenile_to_adult` começa acima de zero para permitir sobrevivência e melhora até capacidade adulta. Senescência pode começar após idade herdada/configurada e reduzir eficiência/resistência, sem decréscimo a cada tick desde o nascimento. Limites e custos evitam seleção trivial de juventude eterna ou reparo infinito.

Não usar a idade normalizada existente em `lifetime.rs` como cronômetro: ela satura em 1. Expor idade cronológica, maturidade e estado corporal separadamente. Sono/repouso não zera idade, não copia experiência para DNA e não prova consolidação neural; consolidação só pode ser alegada se houver mecanismo/teste específico.

## 5. Cérebro: perceber, decidir e lembrar

Reusar CfC/L1/L2, estado lento, hipocampo e plasticidade onde efetivamente executados; não aumentar camadas para compensar sensores/atuadores ausentes. Definir adaptador corporal versionado, preservando cérebro alternativo e replay.

Observações mínimas adicionais: água/capacidade, reserva energética/material, motor/óleo e tendência térmica, desgaste motor/pneus, fadiga, maturidade/idade, potência/freio/aderência efetivos, disponibilidade/estado de serviço, carga passageiro, destino e relógio da tarefa. Sinais internos são propriocepção/interocepção do corpo, não conhecimento mágico do mapa.

Percepção externa: ruas/paredes/buracos, serviços e destinos observáveis. A informação de destino pode vir do passageiro; posição global/mapa, se disponibilizados, precisam constar do cenário. Não orientar automaticamente o ser ao serviço mais próximo com dados ocultos e chamar isso de inteligência.

`ActionSpec` cobre pedais/direção, solicitações de marcha e interações de passageiro/serviço. Antes de aumentar saídas, conferir os canais de intenção já usados para câmbio e controlador. Interação requer distância, velocidade, disponibilidade e custo; falha retorna motivo observável. Corpo valida ação, não escolhe rota.

Sinal de aprendizagem usa consequências da transição anterior: consumo, dano, tarefa e alívio de déficit fisiológico, com componentes registrados. Evitar “recompensa porque apertou freio numa curva”: benefício deve resultar de sobreviver, preservar recursos e cumprir tarefa. Evolução pode selecionar política sem plasticidade; se esta piorar desempenho, registrar e corrigir estimador/escala, não concluir que mais aprendizagem sempre ajuda.

## 6. Táxi GPU: mundo com decisões necessárias

Nova versão `organism-taxi-v1`: dinâmica contínua do carro GPU, serviços e passageiros inspirados no táxi existente. **Não é port bit a bit do mundo discreto antigo.** Não liberar essa versão como paridade histórica.

Entrega mínima completa do cenário:

- Rede viária conectada com interseções legítimas, obstáculos e destinos. Validar conectividade e acessibilidade física dos serviços, sem premiar ramificações aleatórias impossíveis.
- Passageiro com origem/destino, embarque/desembarque condicionado e identificador; pagamento uma única vez por entrega válida. Tempo, carga e percurso têm efeitos registrados.
- Postos de energia/combustível e água, oficinas com material e abrigo/descanso. Estoques, transferência e reposição com taxas explícitas; disputa por serviço determinística e sem prioridade fixa injustificada por slot.
- Economia com dinheiro distinto de energia: tarifa paga serviço, não vira combustível sem transferência. Sem dinheiro negativo silencioso, venda/recompra lucrativa em loop ou entrega repetida do mesmo passageiro.
- Descanso reduz esforço e permite recuperação paga, mas tem custo basal/oportunidade. Ficar parado para sempre não domina a tarefa de táxi; se vira estratégia ótima, rever recursos/objetivo, não matar artificialmente quem descansa corretamente.
- Clima/temperatura e oferta de recursos escalonáveis; esses fatores devem exigir escolhas reais. Evitar ativar todas as dificuldades no primeiro nascimento e tornar toda política inviável.

Modo novo usa orçamento de recursos e integridade como sobrevivência; não misturar reposição infinita de combustível por volta, cronômetro de corrida e oficina do táxi sem regra. Manter essas regras antigas somente no cenário de controle. A perda de capacidade com idade aumenta a necessidade de cuidado, não impõe uma solução única codificada.

## 7. Evolução automática, com controle experimental

Escolher **um modo de reprodução por execução**: gerações episódicas ou ecologia de nascimentos. Não combinar silenciosamente seleção global e reposição local. Ambos compartilham contratos e artefatos; começar a integração em gerações, fechar ecologia no mesmo release após os gates de nascimento.

DNA muta com RNG independente e limites válidos. Nascimento paga construção, energia inicial e custos; nenhum recurso vem de reaproveitar memória de slot morto. Preservar todos os seres vivos. Estado adquirido reinicia por contrato: idade, maturidade, temperatura inicial plausível, estoques transferidos, desgaste inicial, cérebro/memórias/plasticidade, passageiros, serviços, métricas e RNG de vida.

Seleção avalia desempenho de tarefa, sobrevivência e custo sob condições comparáveis. Fixar fórmula, coeficientes e horizonte antes da coleta; nunca deixar o gene alterar seu juiz. Progressão/fitness não são somente distância percorrida: não aceitar loops de ida-e-volta, camping e reparo gratuito como sucesso. Não zerar mortalidade ocultando mortos das médias.

Currículo automático tem estágios registrados: condução básica → recursos/temperatura → desgaste/manutenção → maturação/reprodução → entregas/rotas → população/escassez combinadas. Isso é ordem de treinamento **dentro da entrega completa**, não desculpa para omitir o ambiente final. Cada estágio define critérios de promoção, orçamento máximo e regressão; não promover só por atingir geração N.

Usar cenários de avaliação separados do treino, seeds ambientais distintas das evolutivas e um arquivo de campeões/linhagens. Em extinção ou ausência de avanço, salvar diagnóstico e parar/retornar ao estágio conforme política explícita. Nunca reiniciar escondido até aparecer um resultado bom.

## 8. Ordem do tick e residência GPU

Uma ordem candidata que Claude deve confirmar contra o ledger real antes de editar:

```text
estado de corpo/mundo t
→ sensores internos/externos (GPU)
→ atualização neural quando devida, usando feedback anterior (GPU)
→ ação validada/controle corporal
→ física A/B/C e energia mecânica
→ recursos/térmica/desgaste/fadiga/reparação/maturação (GPU)
→ serviços/passagens/transações e eventos de vida (GPU, arbitragem explícita)
→ consequências/feedback da transição e escrita de observação agrupada
→ próxima decisão; seleção/nascimento em fronteira declarada
```

Claude deve estabelecer se calor da reparação entra no mesmo passo ou no seguinte, quando reservas de serviço ficam disponíveis e quem é o único writer de temperatura/combustível. Não aplicar térmica antiga mais a nova, nem transferir combustível duas vezes. Se mudar ordem, versionar cenário/trace e atualizar aceitação.

Estado corporal, neural, mundo mutável, desgaste e recursos ficam residentes. CPU prepara cenário/genoma e salva métricas/checkpoints; scheduler de geração/nascimento pode consumir eventos compactos em fronteiras declaradas, sem copiar população inteira a cada tick. Kernels de serviços precisam resolver estoque compartilhado sem corrida de dados, por arbitragem/redução determinística. Não prometer determinismo bit a bit de atomics float não ordenadas.

GPU real obrigatória. O runtime atual ainda tem caminho host para controle/bookkeeping: medir ledger e eliminar as trocas necessárias à nova integração, sem afirmar que “tudo já é GPU”. Permitir readback amostrado/agrupado e snapshots, registrar bytes/frequência; UI não dispara transferência por gráfico/neuronio.

## 9. Persistência, observação e painel fazem parte do organismo

Checkpoint de execução inclui corpo adquirido, idade, recursos, mundo/estoques, finanças, passageiros, linhagens, RNG, estado neural, contador temporal e currículo. Salvar só genoma continua útil, mas deve se chamar exportação de DNA. Teste de retomar compara continuação de uma vida, não renascimento com o mesmo peso.

Aplicar [o plano de telemetria](PLANO-TELEMETRIA-E-PAINEL-CLAUDE-20260921.md), **com água, desgaste, reparação e idade agora no escopo desta release**. Onde o plano anterior diz “futuro”, ler como dependência dos kernels corporais, não exclusão. Não apresentar dado novo em replay histórico como se já tivesse sido medido.

Painel mostra mapa/pista real, organismo, módulos 3D e ficha numérica: velocidade, marcha, RPM, pedais, combustível/energia, água, material, temperaturas, desgaste, fadiga, idade/maturidade, integridade, dinheiro, passageiro, serviço e objetivo. Texto factual informa “recuperando pneu: X/s; limitado por material”, quando registrado, e não inventa intenção do cérebro.

UI precisa separar genes/capacidades, estado adquirido e leis do cenário. Mesma identidade/tick/fase entre 3D, tabela, retina e mapa. Resolver 1x, pose/projeção sensorial e comparadores das auditorias. Topologia herdada e plasticidade medida são overlays distintos; não desenhar valor não gravado como zero.

## 10. Gates numéricos e comportamentais

Executar casos pequenos em GPU e parar no primeiro gate vermelho. Coeficientes, tolerância float, horizonte e resultado esperado devem ser escritos antes da execução. Para nova fórmula, usar invariantes/valores fechados como referência de teste, não rodar simulador CPU paralelo.

| Gate | Caso e propriedade exigida |
|---|---|
| Recursos | transferência/reparo/nascimento conservam balanço; falta de estoque limita efeito; nenhum valor negativo escondido por clamp |
| Térmica | carga isolada aquece conforme balanço; resfriamento com fonte desligada converge ao ambiente; pouca água altera dissipação prevista; dt versus dt/2 dentro da tolerância registrada |
| Desgaste | carga/slip sustentados deterioram o componente correto; fadiga/reparo não mascaram dano estrutural |
| Recuperação | parar esfria/recupera fadiga; reparar reduz desgaste e debita recursos; estoque zero bloqueia reparação material; oficina não rejuvenesce |
| Idade | maturação progride apenas com recursos; adulto ganha capacidade; senescência afeta a função declarada; descanso não zera idade |
| Genes | dois DNAs diferem no parâmetro expresso e no custo previsto; genes neutros preservam controle legado no modo legado |
| Interface | cada sensor é provocado e observado no cérebro; cada ação altera corpo ou gera motivo explícito de recusa |
| Táxi | embarque/entrega reais; pagamento único; serviço depende de acesso e estoque; política parada não entrega passageiros |
| Reprodução/retomada | recurso inicial debitado, nova linhagem, resets sem vazamento; snapshot retomado conserva estado e sequência |
| Painel | valores do buffer chegam ao texto/tabela/mapa/3D no mesmo contexto; tick 100 e 1x corretos; sem série fictícia |

Bateria comportamental depois dos gates: comparar controle conhecido com corpo novo; incluir política sem reparação, sem interocepção, sem memória e corpo fixo em ablações separadas, com seeds/horizonte pareados. Primeiro uma seed curta; ampliar somente após prova de execução. Sem desligar mecanismos secretamente para obter demonstração boa.

Condição de sucesso operacional: um cenário completo roda, registra e retoma; há entrega de passageiro, compra/uso de recurso, repouso, reparação com custo, maturação e nascimento rastreáveis. Esses eventos podem ser provocados por controle de teste identificado para validar mecanismos, **mas isso não conta como habilidade aprendida**.

Condição de sucesso científico: políticas evoluídas melhoram tarefa/sobrevivência em ambientes não usados no treino, com distribuição por seed, mortes e custos incluídos. Registrar critério quantitativo e orçamento antes da bateria, após verificar que o cenário é fisicamente viável. Se não melhorar, o release mecânico pode estar integrado, mas a meta de evolução competente continua pendente; não anunciar o contrário.

## 11. Entrega final exigida do Claude

Um cenário `organism-taxi-v1` reproduzível; comando GPU real e manifest completo; código e kernels integrados; relatório de gates; checkpoints de DNA e snapshots distintos; replay e painel com telemetria completa; tabela de cobertura do cérebro/corpo; resultado de avaliação e limitações. Um único índice de status, este plano e esta spec, em vez de novos documentos que adiem indefinidamente a integração.

**Origem da especificação:** pedido atualizado do usuário + inspeção de `COEVOLUTION-RUNTIME.md`, `gpu/body.rs`, `gpu/organs.rs`, `gpu/lifetime.rs`, `ecology_runtime.rs`, `resources.rs`, `crates/taxi-world/PRD.md` e módulos `beings/services/physics/sim_cpu`. Os achados são estáticos; o modelo proposto não foi executado nesta revisão.

## 12. Revisão adversarial: gaps que ainda bloqueiam uma implementação correta

Revisão adicional em 21/09. HEAD consultado `ad944b6`, com alterações locais; HEAD não identifica sozinho o runtime em desenvolvimento. Sem novos testes, simulações ou validação visual. **Estático** identifica código lido; **proposta** define contrato futuro; **pendente** exige medição/decisão antes de afirmar correção.

### G1 — interocepção que chega ao vetor, mas não influencia a escolha

**Estático:** `crates/taxi-world/src/senses.rs` coloca 19 sinais corporais iguais nas nove opções de movimento. `decision.rs:decide` calcula soma linear e subtrai a média das opções. Seja `s_g = w_x·x_g + w_b·b`: o termo corporal comum some em `s_g - mean(s)`. Mesmo sem centralização, adicionar a mesma constante não altera `argmax` entre direções. Há outra via para serviços, portanto isso não significa que todo o corpo é ignorado em toda ação.

**Correção de projeto:** não reaproveitar esse scorer linear como prova de interocepção útil. No adaptador novo, contexto corporal precisa interagir com opções/destinos por representação condicionada ou termos não lineares. Não basta acrescentar colunas de água/calor ou retirar a média.

**Gate GPU:** mesma geometria e pesos de fixture preparados para o teste, variar somente uma necessidade corporal e verificar que a preferência entre destinos pode mudar. Depois, ablação comportamental avalia se a política evoluída realmente usa o sinal. Primeiro gate prova capacidade representacional; segundo prova utilidade.

### G2 — novas ações não cabem automaticamente no aprendizado e desenvolvimento atuais

**Estático:** `brain.wgsl` usa plasticidade de três saídas (`k<3`, `tid<3`) e mantém garantidamente os três primeiros neurônios motores durante desenvolvimento (`layer==2 && neuron<3`). Novas saídas de água/oficina/passageiro não ficam automaticamente protegidas ou plásticas. O layout de 512 f32 e matrizes P/E assume essa cardinalidade.

**Correção de projeto:** antes do shader, entregar tabela `ActionSpec` completa, com índices versionados, sinais contínuos versus solicitações discretas, domínio, arbitragem e máscara de disponibilidade. Preservar acesso juvenil a todos os atuadores essenciais, mesmo quando capacidade neural reduz. Definir quais cabeças evoluem e quais recebem plasticidade; uma cabeça herdada não plástica é válida, desde que declarada. Se estender P/E, atualizar strides, persistência, reset, gravador e UI juntos.

**Gate:** novo atuador alcançável pelo cérebro, alteração detectada na ação, recusa explícita quando indisponível; máscara de desenvolvimento não torna o juvenil incapaz de obter o recurso necessário à própria maturação. Não chamar “desacoplado” um corpo que só funciona com índices implícitos da cabeça antiga.

### G3 — descanso pode ser punido pelas regras antigas de estagnação

**Estático:** `context.rs` aplica custos de vida por estagnação, decresce vida por tick e zera vida quando o tanque seca. Esses mecanismos pertencem à sobrevivência antiga da pista; simplesmente adicionar recuperação corporal não os remove.

**Contrato novo:** selecionar um único conjunto de leis efetivas por cenário. Em `organism-taxi-v1`, repouso/serviço válido não recebe penalidade artificial de “não está avançando”; continua pagando metabolismo, tempo e recursos. Progresso perdido entra na tarefa, não como morte instantânea escondida. Remover refill por volta e ratchet de circuito deste modo; preservá-los no legado.

**Gate:** dois seres parados, um reparando com recursos e outro sem reposição, devem ter consequências explicadas pelo metabolismo e estado de serviço. Parar para manutenção não pode acionar silenciosamente o juiz antigo de corrida. Manifest deve listar regras efetivas, inclusive desligadas.

### G4 — pane, falta de combustível e morte precisam ser estados diferentes

**Gap do plano anterior:** pediu energia e combustível separados, mas não fechou transições de falha. Sem isso, um ser pode “morrer” sem poder reparar uma pane recuperável, ou reparar depois da morte.

**Proposta de contrato:** `alive` é estado terminal de linhagem somente quando uma condição de morte declarada ocorre. Dentro de uma vida há modos locomotor normal/limitado/pane e atividade conduzindo/repousando/em serviço. Tanque vazio bloqueia combustão; reserva metabólica, se existente e financiada, ainda pode sustentar percepção/recuperação. Integridade estrutural fatal ou ausência sustentada de energia vital encerra a vida segundo lei explícita. Recursos zerados não autorizam criação de energia para “salvar” o ser.

Registrar eventos e motivos distintos: `engine_stalled`, `fuel_empty`, `metabolic_depleted`, `irreversible_failure`, `death`. Setpoints, tolerâncias e prazos entram no cenário. Corpo morto não consome posto, atende passageiro ou ressuscita por serviço; novo nascimento recebe outra linhagem.

**Gate:** provocar cada transição e demonstrar reversíveis versus terminais, inclusive pane junto à oficina e falha durante reparação. Não iniciar treino sem definir a tabela de transições.

### G5 — competição atual favorece slots baixos

**Estático:** `resources.rs:step` distribui em ordem do vetor; comentário e teste `dead_beings_do_not_drink_and_index_order_breaks_ties` tornam explícito que índices menores bebem primeiro. Determinismo não é imparcialidade. O viés pode contaminar seleção se linhagens permanecem nos mesmos slots.

**Proposta para o cenário novo:** requisições simultâneas válidas recebem rateio proporcional à demanda limitada por capacidade individual, sem gasto acima do estoque. Quando recurso/serviço for indivisível, usar fila ou desempate pseudoaleatório por evento e identidade de linhagem, nunca prioridade fixa por slot. Essa escolha deve integrar o kernel transacional, não ser remendo estatístico na análise.

**Gate:** permutar os slots preservando estados, IDs e seeds não muda o resultado físico por ser, dentro da tolerância declarada. Paridade histórica pode manter a regra antiga só no cenário legado, nomeada no manifesto.

### G6 — reprodução ecológica não seleciona automaticamente trabalho de táxi

**Estático:** `ecology.rs` habilita pais por idade, energia, cooldown e progresso acumulado mínimo. Não verifica serviço de táxi. Depois de atingir o progresso mínimo, um ser pode continuar elegível enquanto obtém recursos; se o mundo permite coleta gratuita suficiente, há uma hipótese plausível de reprodução sem trabalho útil. Isso não foi demonstrado numa corrida nesta revisão.

**Contrato novo:** tarefa econômica e ecologia precisam ser compatíveis. Financiamento recorrente de manutenção/reprodução deve vir de receita legítima ou recursos ambientais explicitamente orçados; não de bônus oculto de nascimento. Comparar política que só usa postos com política que trabalha. Escolher/documentar se sucesso é descendência ecológica ou desempenho de táxi; um não implica o outro. No modo episódico, seleção por tarefa continua separada da reprodução dentro da vida.

**Gate:** auditoria de fluxo de dinheiro/energia/material e orçamento por vida. Ações de espera, compra/venda, passagem repetida e reprodução em cascata não podem gerar saldo sem fonte. Não resolver isso impondo um roteiro de serviço ao cérebro.

### G7 — escassez deve deixar o problema viável para um juvenil

**Gap do plano anterior:** genes, maturação, desgaste e custos foram especificados sem orçamento inicial e trajetos viáveis. Ativar tudo pode matar toda a população antes de qualquer seleção útil.

**Contrato:** cada configuração publica distância acessível a recursos, capacidades iniciais, consumo basal, reserva mínima, janela até pane/morte e capacidade juvenil. Estimar e verificar na GPU que existe pelo menos uma sequência válida de obter recursos e completar uma entrega. Um piloto de teste pode testemunhar viabilidade, identificado como controle; não entra como política aprendida.

O horizonte deve alcançar os fenômenos avaliados: se termina antes de desgaste relevante/senescência, genes correspondentes não foram submetidos a essa pressão. Definir idades iniciais/episódios de estresse separados quando necessário, sem chamar envelhecimento acelerado de tempo real. Parâmetro calibrado só entra depois em treino congelado/versionado.

**Pendente:** valores numéricos de coeficientes, estoques, horizonte e critérios de promoção. A spec anterior não os fechou. Claude deve entregar configuração inicial finita e ensaio de viabilidade antes da bateria evolutiva; “ajustável” não é uma configuração executável.

### G8 — o modelo térmico anterior não fecha sozinho conservação com massa variável

**Gap da proposta:** a equação de temperatura assume `C` fixo durante o passo. Água perdida/reposta, crescimento e troca de material alteram capacidade térmica e transportam energia. Alterar `C` sem balanço de entalpia pode criar aquecimento/resfriamento gratuito.

**Correção de contrato:** estado térmico conservativo com energia interna/entalpia e conversão para temperatura; entradas e saídas de fluido/material transportam energia na temperatura correspondente. Congelar coeficientes dentro de um subpasso continua possível, mas a mistura/transferência deve ser contabilizada explicitamente. Definir se água é reservatório de reposição ou massa ativa do circuito; não usar as duas interpretações alternadamente.

**Gate:** sem fonte/troca externa, energia total conservada dentro da tolerância; adicionar fluido na mesma temperatura não muda T; fluido frio muda T segundo mistura; perda de fluido e vaporização debitam massa e energia corretas. Não basta “tanque cheio resfria mais” e curva visual plausível.

### G9 — reparação e crescimento disputam a mesma reserva

**Gap da proposta:** cada mecanismo foi descrito com seu limite, mas isso não garante que a soma das operações do tick respeite energia/material. Dois kernels podem ler o mesmo saldo e gastar ambos.

**Contrato:** um orçamento por ser/passo com reserva e commit único. Separar solicitação, alocação, aplicação e registro dos efeitos. Ordem de prioridade metabólica/crescimento/reparo/reprodução explicitada; não deixar a ordem de dispatch decidir a biologia acidentalmente. Contrapartidas de calor, massa e integridade pertencem à mesma transação. Morte/pane no tick tem precedência declarada sobre serviços/nascimento.

**Gate:** recursos insuficientes para duas solicitações simultâneas; soma debitada coincide com reparo/crescimento realizado, inclusive sob troca de ordem de solicitação. Falha de escrita/checkpoint não pode gerar uma segunda cobrança ou segundo nascimento ao retomar.

### G10 — sensores e decisões operam em relógios diferentes

**Gap:** temperatura lenta, reação motora rápida e reparação prolongada não podem depender de “brain tick” como unidade física. O plano de ordem do tick ainda precisa virar um ledger de buffers, frequências e writers no runtime concreto.

**Contrato:** física e reservas avançam pelo tempo de mundo; a ação neural permanece válida até nova decisão dentro dos limites do atuador. Cada sensor informa idade/fase; feedback cobre a transição efetivamente ocorrida. Modos de amostragem e thresholds não podem variar secretamente com stride do replay. Mecanismos corporais de proteção, se adotados, precisam ser nomeados e observáveis, sem virar um piloto escondido.

**Gate:** variar apenas frequência neural/gravação não altera a quantidade física de recurso processada por segundo em um teste com ação imposta constante. Retomar entre atualizações neurais preserva recorrência, último comando, tempo acumulado e feedback pendente.

### G11 — anatomia, função e custo do cérebro continuam sendo coisas distintas

“Cérebro completo” no painel exige o caminho efetivamente executado: sensores, recorrência, gates, bypass, modulação, plasticidade, memória e decisões novas. Mostrar todos os planos já gravados não cobre automaticamente esses mecanismos. Contar planos como neurônios ou multiplicar atividade por um coeficiente e chamar de watts mistura níveis diferentes.

**Gate:** inventário de execução com cobertura de trace por módulo; uma decisão de serviço rastreada até efeito corporal. CPU/GPU, estados ativos/desativados, contribuição local e causalidade experimental separados. Coordenadas anatômicas só para modelo com anatomia real correspondente; native continua grafo funcional.

### G12 — não há evidência de que combinar tudo melhore evolução

**Evidência documental, não bateria repetida:** `COEVOLUTION-VALIDATION.md` reporta regressão do braço combinado em smokes curtos e registra o confundidor de cabeça motora herdada. Esse resultado não condena a proposta; impede tratá-la como melhoria já estabelecida. O log HEAD também registra que variedade de pistas sem domínio pode piorar o resultado; inspecionar artefatos antes de usar números históricos em nova conclusão.

**Correção de método:** fechar uma trajetória vertical sem aprendizado para provar mecânica; depois controle evolutivo pequeno com mundo fixo; depois dificuldade/corpo/plasticidade em condições discriminantes. Release pode conter todos os mecanismos, mas a campanha não precisa ligar todas as pressões no primeiro instante. Mais mecanismos não substituem sinais informativos, ações acessíveis, exploração e horizonte suficiente.

### Quadro de bloqueio atualizado

| Prioridade | Bloqueio | Evidência de fechamento exigida |
|---|---|---|
| P0 de integração | Condicionamento das direções no táxi linear legado; contrato das novas ações ainda aberto | fixtures GPU de entrada→preferência e cabeça→atuador; não generalizar a falha ao cérebro native |
| P0 | Repouso versus morte/estagnação | tabela de transições + lei efetiva nova sem penalidade legada oculta |
| P0 | Recursos, calor e transações sem orçamento fechado | balanço material/energético e concorrência de solicitações |
| P0 de projeto | Viabilidade e alinhamento reprodução/tarefa ainda não demonstrados | configuração executável, testemunho de viabilidade e auditoria econômica; hipóteses de risco, não falhas observadas no cenário novo |
| P1 | Viés de slot / relógios / snapshot | permutação de slots, invariância temporal e continuação retomada |
| P1 | Painel incompleto ou enganoso | ledger, cobertura e decisão rastreada na UI com dados reais |
| P1 | “Evoluiu” sem avaliação independente | protocolo congelado, seeds e resultados com mortes/limitações |

**Diagnóstico final desta revisão:** o maior risco não é faltar mais um órgão, mas adicionar órgãos sem fechar os caminhos observação→decisão→ação→consequência e os balanços que dão valor a essas decisões. Estes gaps foram documentados e contratados com mais precisão; continuam pendentes de implementação e validação. Não anunciar como corrigidos no código.

## 13. Autorrevisão e duas verificações adicionais

### Passagem 1 — correções das minhas conclusões

- **Generalizei o achado do táxi.** O cancelamento algébrico está em `taxi-world/src/decision.rs:decide`, para as nove opções de movimento com contexto corporal comum. Serviços usam outra expressão dependente de necessidade. Isso não demonstra cegueira corporal do native nem explica, por si só, as colisões da corrida.
- **Confundi plasticidade incompleta com ação impossível.** Os limites de três saídas em `gpu/brain.wgsl` restringem o mecanismo local auditado. Uma saída herdada pode funcionar sem plasticidade. Faltam mapear e testar as novas ações; não foi provado que sejam inacessíveis.
- **Deixei o plano parecer mais fechado do que está.** Descrever equações e gates não escolhe índices de ações, coeficientes, orçamento inicial, fitness ou transições de morte. São pendências de contrato/calibração, não kernels concluídos. Claude deve fixar esses itens antes da campanha, sem inventar defaults silenciosos.
- **Misturei descoberta com síntese de problemas já conhecidos.** O cancelamento já está documentado e testado no próprio táxi. Reconfirmá-lo é útil para evitar transplantá-lo; não é descoberta nova desta revisão. Testes existentes foram lidos, não executados.
- **Não isolei uma causa única da estagnação.** Mais órgãos, custos ou camadas não são uma correção demonstrada para frear/dirigir melhor. O trace já contém freio aplicado; ainda é preciso separar capacidade de frear, escolha de momento, intensidade suficiente e efeito físico.

As três passagens são revisão estática/documental. Não houve treino, simulação CPU/GPU nem inspeção visual de painel ligado nesta revisão.

### Passagem 2 — implementação existente versus modelo proposto

1. **Unidades na ecologia:** `ecology_runtime.rs:step` preenche `LifeSnapshot.energy` com `state.fuel`. Os custos e transferências atuais estão nessa unidade de combustível, não em J. A nova reserva metabólica exige conversão explícita e schema distinto; renomear no painel não converte a grandeza. Manter ledger separado para combustível, energia, material e dinheiro.
2. **Retomar DNA não continua a vida:** `gpu_evolve/main.rs` registra `ecology_lineage_scope = generation; resumed genomes become fresh founders, acquired state is not resumed`. O snapshot de execução da seção 9 é trabalho novo. Não aceitar carregar `.bin` como gate de retomada de memória, idade, água ou linhagem.
3. **Falha durante nascimento não é transação recuperável demonstrada:** o scheduler avança antes de `record_birth`; o observer escreve genoma/JSONL antes da aplicação completa e da sincronização dos bancos. O chamador encerra o processo em erro. Portanto não há evidência de duplicação durante uma corrida que continua; há uma janela de artefato parcial em falha/interrupção. Para snapshots novos, publicar um checkpoint completo por commit identificável, distinguir evento preparado de aplicado e testar interrupções nas fronteiras. `flush` do JSONL sozinho não certifica durabilidade/consistência de vários arquivos.
4. **Térmica existente não é balanço energético:** `physics_a.wgsl` usa aquecimento `rpm_fraction * throttle * 15 * dt` e resfriamento `(speed * 0.3 + 2) * dt`, com escala opcional de órgãos e piso de 20 °C. Isso explica a possibilidade de longos períodos no piso, mas não mede sozinho eficiência real, água ou desgaste. O modelo conservativo proposto precisa substituir esse writer no cenário novo; não se soma a ele.
5. **Freio pede investigação por transição:** `context.rs` pode substituir comandos por piloto, ré e overrides; `THR_FIXO != 0` força freio zero. `physics_b.wgsl` converte comando em torque nas rodas, com ramo condicionado por flag. Conferir configuração efetiva e trace da run antes de atribuir efeito a qualquer ramo. Pedal positivo num quadro não prova redução suficiente de velocidade, e ausência visual de barra não prova comando zero.

**Gate de frenagem antes de elevar dificuldade:** caso controlado GPU com mesmo corpo, piso, velocidade e estado inicial; comparar acelerador solto com freio 0/parcial/máximo, medindo velocidade longitudinal, distância até reduzir à velocidade-alvo, slip e energia dissipada. Separar testes de propulsão simultânea e curva. Na política aprendida, registrar distância ao perigo, velocidade de entrada/saída, instante de percepção/decisão/aplicação, comando pedido/aplicado e colisão. A estimativa `d ≈ v*latência + (v²-v_alvo²)/(2*a)` só é referência sob desaceleração positiva aproximadamente constante; usar trajetória medida para concluir suficiência. Não diagnosticar “cérebro lento” a partir do replay amostrado.

### Passagem 3 — o que falta para executar e aceitar o plano

Releitura de `lab-ui/src/components/Scrubber.vue`, `flow/ticks.ts`, `panels/data.ts` e do manifesto real de `integration-champion-20260921/trace/gen0000/being00/champion`:

- O player ainda soma `stride * speedMultiplier` a cada `1000/16` ms; o manifesto tem stride 64. O relógio deve usar tempo decorrido e frequência física registrada, não frequência de observação. O manifesto examinado não possui `tick_hz`: propagar a fonte autoritativa de tempo até o consumidor, sem deduzi-la do stride.
- Os dois seletores de amostra usam regras diferentes. O tick exibido deve incluir tick solicitado, amostra usada, fase e idade do dado; snapshot neural anterior não deve ser apresentado como simultâneo a um pedal futuro.
- O pacote contém L1/L2 e `lifetime_state`, mas `lifetime_layout.enabled_flags=0`. Presença do bloco não comprova aprendizagem/desenvolvimento ativos. Corrigir disponibilidade e estado no painel antes de interpretar atividade como evolução.
- Este pacote é do ser 0/perfil `champion`. Selecionar outro carro não cria trace neural daquele indivíduo. Permitir seleção para pose/dados disponíveis; informar claramente quando detalhe neural não foi gravado. Chave de detalhe inclui corrida, geração, linhagem/vida, ser, perfil e fase; nunca manter silenciosamente o cérebro do ser anterior.

**Contratos ainda abertos, que a implementação deve materializar:**

| Contrato | Conteúdo mínimo verificável | Bloqueia |
|---|---|---|
| Ações e sensores | índices, unidades, domínio, escritor/leitor, frequência, disponibilidade e feedback de recusa | conexão cérebro/corpo; não adivinhar novos canais |
| Sobrevivência | estados, limiares/histerese, orçamento inicial e transições reversíveis/terminais | fisiologia e descanso |
| Recursos e térmica | conversões, coeficientes finitos, prioridades de gasto, fonte/destino e writers únicos | kernels e comparação de balanços |
| Cenário viável | mapa acessível ao corpo, estoques, preços, horizonte e testemunho de rota executável | campanha evolutiva |
| Seleção | fórmula, pesos, modo de reprodução, condições de avaliação, seeds e orçamento | alegação de melhoria |
| Persistência | snapshot completo, IDs, compatibilidade, publicação e recuperação de falha | continuidade de vidas |
| Telemetria | unidade, fase, frequência física, disponibilidade e identidade por bloco | leitura científica do painel |

**Ordem corrigida para evitar mais redesign:** congelar baseline e configuração efetiva; demonstrar comando→frenagem/curva e trace→painel; integrar uma entrega de táxi com consumo/reposição; incorporar e verificar recuperação/desenvolvimento/nascimento; então ampliar população, pressões e avaliação independente. Os órgãos continuam no escopo final, mas não são pré-condição para corrigir o relógio nem substituem a prova de frenagem. Reusar o modo episódico como controle; não abrir duas arquiteturas novas em paralelo para satisfazer gerações e ecologia.

**Aceite do painel:** primeiro mapa e identidade/tempo; depois ficha com números, unidades e estados ausentes; depois cadeia comando→efeito e eventos; por último exploração 3D sincronizada. Cobertura funcional completa não significa desenhar todas as sinapses simultaneamente. Top-k/filtros precisam declarar o que ocultam. Mostrar custo e quantidade de dados antes de gravar todos os seres em toda frequência.

**Limites destas três passagens:** `sed`/`rg` confirmaram os caminhos estáticos citados; leitura JSON do manifesto confirmou stride, blocos e flags. Não executaram o binário nem comprovaram que os mesmos overrides estavam ativos em todas as runs anteriores. O painel desligado impediu aceite visual. Não há evidência para declarar o organismo completo, evolução destravada ou todos os gaps encerrados.

## 14. Contratos entre partes — revisão de integração

21/09/2026. Esta seção transforma pendências em fronteiras de responsabilidade. **Proposta de contrato, ainda não aplicada ao código.** Fontes lidas são indicadas abaixo; a inspeção é estática, não uma certificação de todos os ramos/flags. Não criar outro runtime, outro formato de pacote ou schemas paralelos apenas para atender esta seção.

### 14.1 Propósito e limites de responsabilidade

O propósito atual é obter um organismo veicular que perceba, controle o corpo, sobreviva e melhore por seleção/aprendizagem em tarefas progressivamente mais exigentes, chegando ao táxi, com comportamento investigável. Não é obter um painel convincente nem implementar todos os mecanismos biológicos possíveis.

| Parte | Responsabilidade exclusiva | Não pode fazer implicitamente |
|---|---|---|
| Ambiente/cenário | geometria, recursos externos, serviços, tarefa e leis | mudar dificuldade/juiz durante avaliação congelada |
| Corpo | executar ação, dinâmica, reservas, limites e recuperação | escolher destino ou substituir a política por um piloto não declarado |
| Sensores | transformar estado acessível em observações versionadas | entregar verdade oculta do mundo como percepção local |
| Cérebro | produzir intenção e atualizar estado aprendido permitido | escrever posição, estoque, fitness ou leis físicas |
| Adaptador motor | converter intenção em solicitação corporal válida | esconder overrides ou atribuí-los ao cérebro |
| Avaliador/evolução | medir consequências e gerar descendentes conforme protocolo | editar o juiz por gene ou herdar estado adquirido sem contrato |
| Gravador/API/painel | registrar, transportar e apresentar evidência | alterar trajetória simulada, inventar medições ou completar dados ausentes |
| Fila/runner | executar configuração validada e rastreável | substituir binário/configuração de uma run silenciosamente |

Um reflexo corporal é permitido se fizer parte do corpo declarado; controle de teste também é permitido como controle identificado. Ambos precisam de procedência diferente de uma ação aprendida.

### 14.2 Mapa executado e fronteiras a preservar

Este mapa descreve o caminho native auditado, não presume paridade com fly ou táxi histórico. Ordem interna condicional deve constar do ledger da configuração executada.

| Fronteira | Produtor → consumidor existente | Dados/tempo/device auditados | Obrigação |
|---|---|---|---|
| C01 Execução | `lab_api/routes/queue.py` → `scripts/nmd.py` → `gpu_evolve/main.rs` | pedido validado; fronteira de execução, host | configuração efetiva e identidade do binário |
| C02 DNA → corpo/cérebro | `gpu/genome_io.rs`, `body_evolution.rs` → bancos GPU/estado inicial | genes persistidos versus parâmetros expressos, nascimento/geração | versão/layout, custos e reset explícitos |
| C03 Mundo/corpo → sensores | `GpuCtx::tick` → retina/percepção/digest | poses e estado anterior à física; GPU com preparação host | geometria, unidades, visibilidade, fase e validade |
| C04 Sensores → cérebro | `percept.bind_and_dispatch`/digest → `brain.bind_and_dispatch` | canais compactos, atualização neural condicional, GPU | sem reutilizar índice ou deslocamento implicitamente |
| C05 Cérebro → motor | intenções native, lento/hip → montagem em `context.rs` | intenções `[N,96]`, lento `[N,3]`; há readbacks e montagem host | transformação, idade da decisão e procedência de cada override |
| C06 Motor → física | `PhysicsStateGpu` → `PhysicsGpu::step` | ação aplicada; A/B/C GPU, retorno host | domínio, torque/força, `dt`, ordem dos kernels |
| C07 Física → vida/mundo | retorno da física → `context.rs`/laço de `main.rs` | orientação também atualizada no host; dano/vida/progresso/recursos em fases distintas | writer e fase próprios; não chamar tudo de um único pós-tick |
| C08 Consequência → aprendizagem/seleção | feedback corporal → lifetime; métricas → fitness/evolve/ecologia | feedback de transição versus julgamento de geração | sem vazamento de futuro, recompensa e fitness separados |
| C09 Nascimento/retomada | `ecology_runtime.rs`/loader → slot/bancos/gravador | slot pode ser reutilizado; DNA não é snapshot | identidade de vida, reset completo, commit rastreável |
| C10 Runtime → gravação | `driving_diagnostics.rs:after_physics`, replay/live → arquivos/endpoints | sensores pre-physics, neurais post-brain, corpo post-physics; live em fronteira posterior | fase por campo/bloco e cobertura declarada |
| C11 Gravação → pacote/API | `trace/producer.py:produce` → `routes/trace.py` | seleção de amostras/genoma, blocos e hashes; host offline | derivação determinística e rejeição de mistura incompatível |
| C12 API → painel | loaders/store/adapters → painéis | replay detalhado e live resumido são fontes diferentes | contexto único, unidades, ausência e versão compatíveis |

**Correção importante ao desenho simplificado:** `after_physics` recebe `diagnostic_alive`, capturado no início de `GpuCtx::tick`, e grava `life_left` consultado depois. O live é montado no laço externo, após outras atualizações. Logo, mesmo número de tick não torna `alive`, `life` e estado live um snapshot atômico. Nomear `alive_at_step_start` versus estado final, ou escolher uma fronteira comum ao exportar uma versão nova; não corrigir replays históricos reinterpretando o campo sem procedência.

### 14.3 Conteúdo obrigatório de cada contrato

Para C01–C12, registrar na seção/documento técnico dono da interface:

1. ID e versão; produtor, consumidor e responsável pela validação.
2. Schema/layout: shape, dtype, ordem de eixos, alinhamento, stride, signedness e unidade. Matriz informa orientação e operação; pesos KAN não viram pesos lineares por conveniência visual.
3. Identidade: run, geração/episódio, slot, vida/linhagem quando aplicável e hash do genoma/configuração. Slot não é identidade vital.
4. Tempo: tick físico, fase, frequência física, decisão de origem e validade. Stride de gravação não define velocidade do mundo.
5. Domínio e precondições: finitude, limites, enumerações, disponibilidade do módulo e capacidades do corpo.
6. Poscondições/invariantes: transformação permitida, conservação e autoridade de escrita.
7. Falha: rejeitar, interromper execução ou marcar módulo opcional indisponível; nunca completar resultado científico com zeros sem status.
8. Evidência de aceite: fixture pequena compartilhada, teste semântico e artefato de integração. Para matemática/simulação, apenas GPU nesta tarefa.

Não criar doze frameworks. Reusar structs Rust/WGSL e schema JSON existentes; documentar semântica junto ao dono, exportar descritores e checar cópias/consumidores. Geração automática só onde remover duplicação real. Valores de faixa e coeficientes ainda abertos continuam bloqueantes, não devem ser preenchidos arbitrariamente nesta revisão.

### 14.4 Contratos semânticos que faltam ou divergem

| Problema confirmado por leitura | Contrato exigido | Como aceitar |
|---|---|---|
| `life` live é `life_left/life_cap`, trace guarda ticks brutos | IDs semânticos distintos: orçamento em ticks, fração e integridade; capacidade e conversão explícitas | mesmo estado convertido pelos dois adaptadores sem confundir saúde/tempo |
| `energy` da ecologia vem de combustível | unidade de modelo atual separada de nova energia metabólica | migração com fator e ledger; não trocar apenas legenda |
| Montagem motora tem múltiplos overrides | intenção, comando após cada estágio e comando efetivo; origem/razão por canal | fixture de freio e override, efeito físico e registro concordam |
| Intenções ausentes invalidam tick; lento incompleto pode apenas avisar | capabilities efetivas: módulo obrigatório falha explicitamente; módulo desligado é ausência esperada | falha injetada do módulo habilitado invalida resultado; modo sem módulo não gera falso erro |
| Cérebro pode pular dispatch e reutilizar intenção | comando mantém ID/tick de origem, expiração/regra de retenção declarada; feedback consumido uma vez | alterar apenas amostragem não muda trajetória; nascimento limpa comando antigo |
| Store, helpers micro e fluxo têm seletores diferentes | uma resolução temporal compartilhada; anterior/próximo/interpolado identificados | tick 100, antes da primeira amostra, lacunas e morte sem mostrar futuro como presente |
| Pacote comporta um genoma por conjunto de pesos | segmentar por vida/genoma; não misturar descendentes no mesmo pacote antigo | filtro por linhagem já existente no produtor preservado; erro explícito quando consumidor não suporta perfil |
| Arquivos de corpo/genoma/trace têm versões distintas | compatibilidade declarada entre modelo corporal, canais, layout neural e schemas | recusar combination incompatível antes de carregar/dispatch/renderizar |

O produtor já possui filtro `lineage_id` e recusa vários `genome_file` nas amostras. Portanto o requisito é propagar essa identidade até índice/API/UI; não remover a proteção nem afirmar que segmentação está totalmente ausente.

### 14.5 Um único writer por estado e transações explícitas

Claude deve preencher o ledger concreto antes de portar kernels. Um estado pode ter várias contribuições, mas uma fase autoritativa de commit por passo:

| Estado | Autoridade proposta no novo cenário | Outros módulos |
|---|---|---|
| pose, velocidades, rodas | integração física e orientação | leem; não corrigem trajetória para satisfazer objetivo |
| combustível/energia/água/material | commit do orçamento corporal e transferências | pedem consumo/transferência, não debitam isoladamente |
| temperatura/desgaste/maturação | integrador corporal após alocação | geram cargas/taxas; não aplicam também modelo legado |
| passageiro, pagamento, estoques | transação do mundo com IDs | emitem solicitações; entrega/reparo não duplica crédito |
| recorrência/plasticidade | backend cerebral e reset de vida | gravador apenas observa; evolução altera DNA nas fronteiras permitidas |
| alive/linhagem/nascimento | ciclo de vida após consequências | UI e fitness não ressuscitam/reclassificam estado |

`request → allocate → apply → commit` define precedências entre serviços, morte e nascimento. Não exige copiar esses quatro estágios para CPU. Não confundir essa proposta com residência já implementada: o código atual ainda prepara comandos, atualiza orientação e mantém sobrevivência no host.

### 14.6 Compatibilidade, falhas e observabilidade

- **Preservar o que existe:** `sha256sum` confirmou schemas `lab-ui/contract/trace-v1.schema.json` e `lab_api/trace/trace-v1.schema.json` idênticos nesta revisão (`c2ac9efe…9770b`). Transformar igualdade em gate contínuo; igualdade textual não valida unidades/fases.
- **Evoluir com versão:** novos campos precisam de estratégia para leitores antigos, sobretudo onde o schema rejeita propriedades desconhecidas. Campo opcional só é compatível se o consumidor suportar sua ausência. Alteração de unidade, índice, fase ou significado é mudança de contrato mesmo sem mudar dtype.
- **Falha científica versus dado opcional:** NaN/shape incorreto em estado obrigatório invalida a execução. Trace opcional perdido marca cobertura incompleta; não invalida automaticamente toda física, mas bloqueia as conclusões que dependem daquele trace. Não fazer fallback para CPU.
- **Live não é replay:** atraso, reconexão e tick repetido devem aparecer como estado de conexão/frescura. Parear status e seres pela mesma revisão/tick; requisições separadas não asseguram atomicidade. Não inventar atividade neural entre snapshots populacionais.
- **Imutabilidade de pacote:** publicar conjunto completo com manifesto/hashes coerentes e identidade de conteúdo. Cache depende da fonte/configuração/versão do produtor, não só do nome da run. API já valida hashes: manter essa defesa e recusar pacote truncado.
- **Observação não comanda simulação:** ligar painel/gravação não pode mudar RNG, quantidade de passos, política ou seleção. Testar mesma configuração com observação ligada/desligada; medir overhead e bytes/readbacks separadamente. Tolerância numérica e orçamento de overhead precisam ser definidos antes do gate.
- **Fila não é shell livre:** preservar validação tipada, limites, snapshot de binário, autenticação/CSRF nas mutações e publicação atômica do pedido existentes. Aceitação de flag deve corresponder ao binário congelado, não apenas ao código-fonte atual. Cancelar e reenviar não pode duplicar o experimento sem uma nova identidade.

### 14.7 Aceitação ponta a ponta e proteção do propósito

**Fixture de integração inicial:** um genoma, corpo, cenário e seed congelados; um intervalo incluindo aproximação e frenagem. Identificar sensor/tick → representação neural → intenção → transformações/override → comando físico → desaceleração/contato → consequência → trace → pacote → valor no painel. O checkpoint 2432 do plano de telemetria serve para transporte de valores existentes, não prova toda essa causalidade nem competência de condução.

Depois adicionar fixtures de serviço com estoque insuficiente, morte/nascimento no mesmo slot, snapshot retomado e schema incompatível. Não trocar runtime inteiro antes de fechar essa cadeia. Critérios de integração mecânica, comportamento aprendido e apresentação visual ficam separados; todos são necessários para encerrar o objetivo completo.

Uma mudança só entra no cenário principal se declarar: qual comportamento ou defeito atual atende; contrato afetado; migração/default legado; custo de GPU/VRAM/readback esperado; gate; resultado e limitação. Ideia experimental sem ganho demonstrado fica atrás de configuração identificada, fora do baseline congelado. Isso evita que “mais biologia” substitua a meta atual de dirigir, sobreviver e cumprir tarefa.

**Decisão arquitetural proposta:** endurecer as fronteiras existentes em vez de reescrever módulos ou criar um barramento genérico. Benefício: migração incremental e consumidores verificáveis; custo: adapters e checagens temporariamente duplicados. Rever essa decisão somente se um segundo cérebro/corpo concreto não puder cumprir os descritores sem vazamento de detalhes internos.

**Status:** mapa e requisitos revisados; implementação, integração GPU, perfil de transferências e aceite visual continuam pendentes. Esta revisão não executou simulação, testes de aplicação nem servidor.

## 15. Validação obrigatória para trabalho por agentes

**Escopo:** todas as fronteiras do ecossistema Universe + neuromod-lab + lab-ui e seus consumidores, não apenas corpo/cérebro. Esta seção especifica a infraestrutura de validação a implementar; não instala hooks, CI ou proteção de branches. Não impõe o domínio táxi aos demais projetos independentes da pasta LLM.

### 15.1 Estado real das proteções existentes

Leitura de `Universe/.github/workflows/ci.yml`: o step efetivo de `fmt` é bloqueante; `clippy` tem `continue-on-error: true`; há `cargo test`, sem gate GPU explícito nesse workflow. O cabeçalho que diz que ambos não bloqueiam está desatualizado. Não executar esse workflow localmente nesta revisão: ele não foi classificado quanto à restrição atual de simulações somente em GPU.

`lab-ui/package.json` oferece typecheck/build/test; existem testes de contratos e igualdade com fonte. A API tem checker de amostras, validação de pacote e hashes. Isso é base para aproveitar, não prova de aprovação do sistema integrado. Não foram executados esses checks aqui, nem verificada proteção remota de branches.

### 15.2 Cinco níveis de garantia, sem depender de obediência ao Markdown

| Nível | Mecanismo a implementar/aproveitar | O que impede | O que não prova |
|---|---|---|---|
| Compilação | tipos nominais para unidades/IDs relevantes, enums de fase, interfaces de módulos; asserts de offsets/strides Rust–WGSL | misturar índices, formas e unidades conhecidas estaticamente | fórmula correta ou comportamento inteligente |
| Fronteira runtime | validar versão/capabilities, shape, finitude, identidade e recursos antes de aceitar dados externos | carregar genoma incompatível, trace inválido ou módulo obrigatório ausente | semântica completa de um número plausível |
| Teste de contrato | fixtures do produtor consumidas por API/UI, invariantes e casos inválidos | produtor e consumidor divergirem silenciosamente | que cenário seja aprendível |
| Integração científica | caso GPU discriminante e cadeia completa de observação/ação/consequência | declarar sucesso por mock ou teste isolado | generalização sem avaliação independente |
| Aprovação da mudança | runner confiável + revisão do diff dos contratos/testes + evidência da combinação exata | integrar código não verificado ou tratar teste enfraquecido como correção | impedir fraude se o autor controla também o runner e suas permissões |

Contrato do cérebro deve permitir `capabilities`, inicialização/reset, decisão com contexto temporal, leitura de telemetria e snapshot quando suportado. Corpo recebe solicitações tipadas e devolve efeitos/recusas. Não pressupor que todo backend tenha L1/L2, plasticidade ou anatomia de mosca. Adaptadores explícitos traduzem os layouts compactos GPU; não inserir objetos dinâmicos ou validação por neurônio no caminho quente.

Interfaces com tipos específicos e validadores **podem e devem detectar unidade errada e invariantes de domínio**. Um campo declarado apenas como float é que não distingue km/h de m/s. Validadores relacionais também devem rejeitar `fuel=0.5` quando a capacidade é 0.3; isso não precisa esperar uma bateria científica. Já demonstrar que `brake=0.8` produz frenagem exige verificar a execução física. A seção 15.8 concretiza essa distinção e corrige a formulação anterior excessivamente genérica.

### 15.3 Registro executável de contratos e dependências

Criar um índice legível por ferramenta com C01–C12 e demais contratos descobertos. Cada entrada referencia o schema/struct autoritativo existente, em vez de manter uma nova cópia concorrente. Campos mínimos:

```text
contract_id, version, owner_module
producer_paths, consumer_paths, depends_on
schema_or_layout_source, capabilities_required
semantic_invariants, compatibility_policy
checks: command, execution_class, budget, required, evidence_outputs
```

O índice precisa associar arquivos alterados a consumidores transitivos: mudar um sensor exige checar percepção→cérebro→gravador→pacote→painel; mudar fitness exige conferir seleção→métricas→comparação; mudar loader exige conferir exportador/checkpoint/fila. Arquivo relevante sem classificação é pendência de triagem, não autorização para executar zero checks. Dependências, flags, scripts, assets e schemas também entram no cálculo de impacto.

Unificar os dois schemas do trace por geração/cópia verificada a partir de fonte declarada, preservando os consumidores atuais. Contratos HTTP/WS, banco/migrações, pedidos da fila, checkpoint, eventos e layouts de kernel também entram no índice. Interface TypeScript sozinha não valida resposta HTTP; JSON Schema sozinho não valida ABI WGSL.

### 15.4 Comando único de integração — interface proposta

Um entrypoint futuro do ecossistema, com três operações lógicas: `plan` calcula impacto/checks; `verify` executa o conjunto permitido; `report` explica aprovação ou bloqueio. Os nomes são proposta, não comandos disponíveis hoje. Reaproveitar testes existentes e adicionar apenas gaps semânticos relevantes.

| Classe de execução | Conteúdo | Política |
|---|---|---|
| static | schemas, imports, tipos, dependências, validação de fixtures e metadados | host permitido, concorrência e memória limitadas |
| build | compilação dos módulos afetados | host permitido e limitado; não recompilar binário em uso |
| gpu | kernels, dinâmica, aprendizagem, invariantes numéricos | GPU real verificada; fila exclusiva/orçamento; nunca fallback CPU |
| browser | fluxo real de seleção/tempo/unidades e inspeção do painel | servidor/configuração identificados; mock não substitui fonte real |

Um script de teste não pertence à classe static só porque usa Python/Rust: se executar modelo/física, pertence à classe gpu. Sem GPU disponível, resultado obrigatório é **BLOCKED**, não PASS nem execução CPU. Revisão documental não exige iniciar essas baterias.

Saída estruturada por gate: `PASS`, `FAIL`, `BLOCKED` ou `NOT_APPLICABLE` justificado. Gate obrigatório sem execução impede aprovação final. Não transformar ausência de artefato em sucesso; timeout, processo morto e log truncado são estados explícitos. Orçamento excedido encerra o caso e preserva diagnóstico.

### 15.5 Prova vinculada ao código realmente testado

O relatório deve identificar a combinação Universe/neuromod-lab/lab-ui, dependências/lockfiles, contratos, binário, kernels incorporados, configuração, device/driver e fixtures. Com árvore suja, HEAD é insuficiente: incluir fingerprint dos conteúdos relevantes, inclusive arquivos não rastreados. Registre comandos, exit codes, tempos e hashes dos artefatos.

Qualquer edição relevante depois do teste invalida sua evidência ou exige recálculo explícito do impacto. Não reutilizar PASS obtido antes de uma correção. Publicar um manifesto de integração do conjunto de versões que passou, sem presumir que três branches chamadas `main` sejam compatíveis. Mudanças coordenadas entre repositórios precisam desse conjunto testado, não de três aprovações isoladas.

Para compilar e testar enquanto agentes editam, usar checkout/snapshot estável de integração. Não executar a bateria contra arquivos que mudam durante o teste. Agentes trabalham em mudanças isoladas por fronteira; o integrador aplica e valida a combinação. Não exige criar worktrees nesta revisão.

### 15.6 Regras contra enfraquecimento da validação

- Mudança de contrato atualiza produtor, consumidores, migração, fixture e documentação na mesma entrega coordenada. O teste deve demonstrar incompatibilidade quando ela existe, não normalizá-la silenciosamente.
- Autor pode alterar testes quando o comportamento autorizado muda, mas deve justificar alteração de expectativa. Remover assert, aumentar tolerância, regenerar golden, adicionar skip/allowlist ou `continue-on-error` exige revisão explícita do integrador. Nunca atualizar golden apenas copiando a saída que falhou.
- Fixtures semânticas precisam de resultado independente: valor analítico, balanço, propriedade física ou artefato congelado revisado. Gerar entrada e resposta pelo mesmo algoritmo apenas testa autoconsistência.
- Casos negativos obrigatórios nas fronteiras relevantes: unidade/versão errada, dimensões trocadas, NaN/Inf, arquivo truncado, consumidor antigo, tick fora da cobertura, slot renascido, módulo desativado, interrupção de escrita e dispositivo indisponível.
- Exceção tem escopo, motivo, responsável e validade; aparece no relatório e não vira PASS científico. Dívida antiga deve ter baseline identificada, sem permitir regressões novas na parte modificada.
- Hooks locais são atalho de feedback, não barreira confiável: podem faltar ou ser ignorados. O gate obrigatório roda novamente no integrador/CI. Se houver repositório remoto protegido, aprovação deve depender desse status; confirmar/configurar permissões separadamente. Não alegar proteção instalada sem verificá-la.
- Se agentes tiverem permissão de editar runner, política e integração final, nenhuma regra em arquivo oferece isolamento real. A garantia forte exige separação de permissões/runner confiável; revisão independente reduz risco, mas não substitui essa separação.

### 15.7 Ordem de implementação para Claude

1. Inventariar checks existentes, classificar execução e registrar dependências C01–C12; não iniciar centenas de testes sem orçamento.
2. Implementar validação compartilhada de identidade/tempo/unidades e compatibilidade de schemas/layouts, com construtores e conversões da seção 15.8. Começar pelas divergências confirmadas nas seções 14.2/14.4 e pelos casos inválidos reproduzidos abaixo.
3. Integrar entrypoint/relatório com fingerprint do conjunto de projetos e regras de invalidar evidência após alterações.
4. Acrescentar gates GPU pequenos e consumer tests reais para as fronteiras alteradas; conservar restrição de CPU e limites do desktop.
5. Conectar aprovação obrigatória ao fluxo efetivo de integração e verificar bloqueio com uma quebra intencional de fixture/contrato em ambiente isolado, depois restaurá-la. Testar também que correção legítima volta a passar; não basta um gate permanentemente vermelho.

**Critério final:** um agente deve receber uma falha específica antes de integrar uma mudança incompatível: contrato afetado, produtor/consumidor, esperado/observado, evidência e correção necessária. Isso precisa funcionar mesmo se ele esquecer de ler a documentação. Ainda assim, melhoria de inteligência exige o experimento comportamental separado.

### 15.8 Correção: interfaces que validam unidades e preservam invariantes

**Erro desta revisão:** dizer genericamente que uma interface não detecta unidade errada confundiu um campo numérico primitivo com uma interface de domínio validada. Classes/construtores, tipos nominais e schemas acompanhados de validadores podem bloquear essa mistura. Essa é a solução exigida, não uma limitação a contornar apenas por documentação.

#### Evidência adicional e limite da conclusão

- `lab-ui/contract/trace-v1.schema.json` declara `unit` como string livre; sua descrição mistura unidade física com lista de nomes de canais. Não basta validar que o valor é string para conferir compatibilidade física. `TraceBlock.unit: string`, no consumidor TypeScript, também não restringe unidades por canal.
- **Verificado no checker isolado:** `lab_api.trace.check.check_sample` aceitou `retina=[True]`, `wave=[False]`, `fast_state=[True]`, com dimensões 1 e `body={}`. Python considera bool uma subclasse de int, e os predicados usados permitem essa entrada.
- **Verificado no mesmo checker:** outra chamada aceitou `body={"speed":"not-a-speed"}`, com retina/wave/fast_state numéricos válidos. A função confere que body é objeto, mas não valida ali todos os campos físicos.

Comando: `python3 -B`, import path `neuromod-lab/src`, importação de `check_sample`, metadata `retina_rows=retina_cols=wave_channels=fast_state_values=1` e as duas fixtures acima, com tick 1. Ambas retornaram sem exceção. A segunda usou retina/wave `[0]` e fast_state `[0.0]`. Foi checagem de formatos, sem simulação/modelo. Isso demonstra lacunas **nessa função**, não que todo pacote inválido atravesse os demais validadores. A tentativa inicial de importação por arquivo falhou por import relativo; a verificação válida usou o pacote Python.

#### Construção e uso obrigatórios

```text
entrada externa
  → parser estrito: tipo, campos, versão, tamanho
  → validação: unidade, identidade, domínio e relações entre campos
  → conversão explícita para representação canônica
  → valor validado com invariantes preservados
  → método do módulo ou codec GPU autorizado
  → validação do resultado na fronteira correspondente
```

1. **Grandeza no tipo/descritor:** `Speed<MetersPerSecond>`, `Speed<KilometersPerHour>`, `Angle<Radians>`, `LifeBudgetTicks`, `HealthFraction`, `FuelModelUnits`. Nomes propostos, não tipos existentes. Fração de dano e pedal de freio também são tipos semânticos distintos, apesar de ambos variarem entre 0 e 1.
2. **Construção controlada:** campos privados/imutáveis onde aplicável; factories/construtores com erro de domínio. Desserialização também valida. Em TypeScript, `as SomeType` não valida runtime: entrada permanece `unknown` até decodificação. Em Python, excluir bool dos campos numéricos. Em Rust, não expor um campo/construtor bruto que contorne o wrapper validado.
3. **Conversão explícita:** endpoint canônico pode rejeitar km/h; adaptador de importação pode aceitá-lo e converter. A política pertence à fronteira. `36 km/h → 10 m/s` ocorre uma vez no adaptador; cast ou troca de rótulo não é conversão. UI converte para exibir sem alterar estado canônico.
4. **Validade após mutação:** setters/operações preservam invariantes; combustível/capacidade são validados conjuntamente após atualização. Clamp não mascara dado externo corrompido. Saturação física legítima registra pedido/aplicação; solicitações inviáveis retornam recusa explícita.
5. **Descritor por canal:** blocos mistos declaram nome semântico, unidade, índice, domínio e transformação por canal. Não criar classe por elemento de matriz. Buffers compactos usam descritor validado e codec autorizado. Preservar leitura de trace-v1 por adaptador legado explícito; novo contrato exige migração/versionamento coordenado.

#### Gaps adicionais e bloqueios concretos

| Gap | Validação exigida | Caso de aceite |
|---|---|---|
| Bool/string usados como número; unidade desconhecida | parser estrito, enum/registro de unidades e dimensão permitida | `true`, `"36"` e `m` num campo canônico de velocidade rejeitados; coerções só no adaptador autorizado |
| Conversão ausente/duplicada | DTO externo separado do valor canônico; conversão entre tipos distintos | km/h→m/s→exibição preserva quantidade na tolerância definida |
| Graus/radianos e temperatura absoluta/diferença | tipo de grandeza e transformação correta, inclusive offset | 180°→π rad; 0 °C→273,15 K; diferença de 10 °C→10 K |
| Overflow, perda de identidade, NaN/Inf | integralidade, limites, finitude e representabilidade após conversão | tick fracionário, u32 negativo e overflow f32 rejeitados; u64 além do inteiro seguro JS tem representação sem perda |
| Estado impossível com campos válidos isoladamente | validator do agregado e precondições de método | combustível acima da capacidade, débito acima do saldo e serviço após morte recusados |
| Unidade correta, referencial errado | frame espacial e convenção de eixos/ângulos | vetor local não entra como posição global sem transformação; módulo de velocidade não é velocidade longitudinal assinada |
| Duração/instante/fase confundidos | tipos e contexto temporal distintos | tick de outra vida/geração recusado; futuro não apresentado como medição atual |
| Shape igual, canais/eixos trocados | IDs/ordem dos canais e eixos, orientação e layout versionado | throttle/brake trocados e matriz transposta detectados mesmo com dimensões iguais |
| Ausência confundida com zero | estado discriminado com motivo | não gravado/desativado/não aplicável não viram medição zero |
| Wrapper contornado por buffer ou desserialização | codecs validados; gate ABI e teste semântico | fixture com marcador por canal detecta offset/stride/permutação no round-trip |

Enums/strings usam identificadores exatos ou normalização por tabela explícita na importação, não inferência por substring. Um produtor que calcula errado mas declara `unit="m/s"` pode passar pela validação de unidade: esse caso exige teste da fórmula/produtor. Isso não diminui a utilidade dos tipos e validadores.

Valores float64 finitos podem transbordar ao converter para float32: validar após conversão e definir tolerância de quantização. Não exigir identidade bit a bit entre precisões diferentes. IDs e contadores exatos não atravessam floats por conveniência.

#### GPU, diagnóstico e aceite

Validar entradas e descritores ao carregar/preparar dispatch; manter buffers compactos na VRAM. Tipos de domínio no host não precisam aumentar o buffer. Invariantes dos valores produzidos pelos kernels podem ser conferidos por checks GPU com resumo de violações; não adicionar readback por campo/neuronio a cada tick. ABI (alinhamento, offsets, stride, endian) e semântica de canais têm gates separados.

Erros informam contrato/versão, campo/canal, código (`TYPE_MISMATCH`, `UNIT_MISMATCH`, `OUT_OF_RANGE`, `INVALID_STATE`, `STALE_CONTEXT`), esperado/recebido e contexto disponível. Limitar payload/log, sem despejar buffers completos. Exemplo: `body.speed: expected m/s, received kg — UNIT_MISMATCH`.

Prioridade para Claude: fechar os dois casos reproduzidos no checker; definir descritores de velocidade/ângulo/vida/combustível/pedais; implementar construtores/codecs e conversões; migrar consumidores; demonstrar rejeição das entradas inválidas e preservação das válidas até o painel. Testar só a classe não basta se API/gravador a contornam.

**Status:** documentação corrigida e dois gaps de validação reproduzidos; código de aplicação não alterado. Wrappers, validadores e schemas propostos ainda não foram implementados.
