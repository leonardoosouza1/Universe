# Fechamento: organismo evolutivo e táxi na GPU

**Universe — 22/09/2026:** contratos de ação/corpo integrados à física, validação de retorno e checagem ABI Rust/WGSL; 9 testes estáticos e 1 bateria GPU passaram. Escopo e pendências: [contratos executados](docs/CONTRACTS-RUNTIME.md). Isso não conclui todos os contratos nem comprova aprendizado.

**Status em 22/09:** implementada a primeira integração de contratos de telemetria (quantidades, validação produtor/consumidor e gate executável); instruções em [CONTRACT-GATE.md](../neuromod-lab/docs/CONTRACT-GATE.md). Isso não conclui os oito gates do organismo abaixo. Nenhuma física/treinamento foi executada ou modificada nesta etapa.

Pedido atualizado em 21/09/2026: entregar o ciclo completo de corpo, cérebro, recursos, reparação, envelhecimento, reprodução, tarefa de táxi e observação. Este arquivo é o **plano de execução para Claude**; não é declaração de implementação concluída.

**Especificação obrigatória:** [ORGANISMO-TAXI-SPEC.md](docs/ORGANISMO-TAXI-SPEC.md). Ela substitui a decisão anterior de adiar água/desgaste para fora do escopo: agora esses mecanismos fazem parte da entrega. Auditorias anteriores continuam como requisitos de correção, não como planos concorrentes.

**Revisão e correções:** começar pela seção 13 da spec, depois consultar os achados da seção 12. Foram feitas a autorrevisão e duas passagens adicionais. O cancelamento corporal foi confirmado nas opções de movimento do táxi linear legado, não em todo cérebro; plasticidade de três saídas não prova que novas ações sejam impossíveis. Ainda faltam contratos de ações/sensores, pane/morte, unidades/orçamentos e configuração viável. São pendências de integração e avaliação, não causas únicas demonstradas da estagnação.

**Primeiro marco verificável:** configuração efetiva congelada; frenagem/curva medida na GPU; uma decisão rastreada até o painel com identidade, fase e relógio corretos. Em seguida integrar entrega/recursos e completar os mecanismos abaixo. Não esperar todos os órgãos para corrigir o replay, nem usar mais órgãos como prova de melhor inteligência. Nenhuma simulação foi executada nestas revisões.

**Contratos de integração:** a seção 14 da spec mapeia C01–C12, do pedido de execução até o painel. Toda alteração deve identificar fronteira, produtor/consumidor, identidade, unidade, fase, autoridade de escrita, compatibilidade, falha e gate. Reusar os contratos existentes; não abrir schemas ou runtimes concorrentes. Prioridade: corrigir divergências semânticas e rastrear uma decisão inteira antes de ampliar os mecanismos.

**Validação automática para agentes:** seção 15 define índice executável de contratos, análise de impacto, gates por classe de execução, manifesto do conjunto testado e bloqueio de integrações sem evidência. É infraestrutura ainda a implementar. Markdown/hooks locais não equivalem a proteção de integração; sem GPU, gate numérico obrigatório fica bloqueado, sem fallback CPU.

**Unidades e interfaces validadas:** seção 15.8 exige construtores/decoders estritos, tipos de grandeza, conversões explícitas e invariantes preservados após mutação. Corrige a afirmação anterior: interfaces validadas podem bloquear unidade errada. Há dois casos inválidos aceitos pelo checker isolado já reproduzidos; começar por eles e migrar os consumidores sem contornar a validação.

## Uma entrega, com gates internos

| Ordem | Entrega e arquivos de entrada | Gate antes de prosseguir |
|---|---|---|
| 1 | Congelar baseline; contratos de corpo/ação/observação/cenário. Entradas: `gpu/lifetime.rs`, `genome_io.rs`, `gpu/physics.rs`, `crates/taxi-world/src/` | mapa do tick e schema versionado; legado preservado; nenhum sensor/atuador sem significado |
| 2 | Recursos, térmica, fadiga, desgaste e reparação residentes. Entradas: `physics_a/b/c.wgsl`, `gpu/body.rs`, `gpu/organs.rs` | balanço de recursos; calor sob carga; desgaste e recuperação paga; descanso não apaga idade |
| 3 | Genes corporais e cerebrais, maturação e senescência; nascimento/reset/snapshot | evolução altera capacidades com custo; estado adquirido não vaza entre vidas; retomar conserva estado |
| 4 | Interocepção, ações de serviço/repouso e modulação; integração com cérebro existente | cada sinal muda na condição certa e chega ao cérebro; comandos chegam ao corpo, sem autopiloto oculto |
| 5 | Cenário táxi GPU: ruas, destinos, passageiros, estoque, oficinas e áreas de descanso | entrega contabilizada uma vez; serviços conservam estoque; mapa navegável; carro não teleporta |
| 6 | Seleção/ecologia/currículo automático e limites de execução | extinção e estagnação explícitas; transferências de nascimento conservadas; seeds de avaliação isoladas |
| 7 | Pacote/API/painel: mapa, cérebro, ficha numérica, causas e objetivos | tick/ser/vida coerentes; 1x correto; água/temperatura/desgaste visíveis e reais; sem dados fictícios |
| 8 | Bateria integrada GPU + inspeção visual + relatório de release | relatório mecânico e comportamental separado; artefatos reproduzíveis; pendências bloqueantes resolvidas |

Cada gate deve produzir evidência curta. Não esperar a bateria final para descobrir que corpo, sensores e painel usam unidades diferentes. Não iniciar outra rodada de redesign antes de integrar o caminho vertical.

## Regras de execução

- **Não rodar simulação ou referência numérica em CPU.** O pedido atual prevalece sobre o PRD histórico do táxi, que prescrevia CPU/“sem GPU”. Compilação e I/O continuam necessários, com concorrência limitada; não iniciar suíte ampla que trave o desktop.
- Manter corrida legada e mundo táxi histórico intactos; a nova combinação é cenário/versionamento **`organism-taxi-v1`**, não alegação de paridade com o Python antigo.
- Sem reescrever todo o cérebro, frontend ou runtime. Reusar módulos e trocar os contratos que impedem integração.
- Uma integração principal. Se houver delegação, contratos aprovados no próprio plano antes de dividir arquivos; cada agente entrega diff, gate e artefato. Nenhum relatório substitui validação do integrador.
- Resultados grandes em `/media/llm-data/universe-runs/`; toda bateria limitada por tempo, população e armazenamento; falha de device não autoriza fallback CPU.
- Entregar código, cenário completo executável, checkpoints retomáveis, replay e painel. Uma aba ou flag existente não satisfaz o requisito correspondente.

“Completo” significa que todos os mecanismos desta versão estão integrados e demonstrados. Melhoria evolutiva é uma conclusão experimental separada; não pode ser garantida pelo número de órgãos ou por uma data de entrega.
