# Auditoria dos sensores e fluxo CPU/GPU

Data: 2026-09-12. Base: `4b87614`, caminho `src/bin/gpu_evolve/`. Revisão estática por Codex e dois agentes de leitura, com confirmação dos achados prioritários pelo revisor. Sem edição de código, treino ou execução GPU. Não confundir com `gpu_loop`, Bevy ou `cortex_evolve`.

## Veredito

O ciclo NÃO está inteiramente na GPU. Retina, boa parte da percepção, inferência e três passes de física usam kernels; a CPU ainda calcula sensores, mundo auxiliar, decisões, estado corporal e orientação. Também há problemas de contrato sensorial que devem preceder uma migração de desempenho.

## Fluxo real verificado

```text
states CPU
 → mundo auxiliar/corpo, poses, sondagens e sensores CPU
 → uploads para buffers
 → retina / percepção espacial / temporal / movimento / digest / cérebros GPU
 → submit, espera, readback de intenção + lento + digest
 → vida/combustível, hipocampo/gates, comandos/reflexos CPU
 → upload de estado físico
 → física A → B → C GPU
 → readback de estado
 → rotação yaw/normalização CPU
 → memória sensorial + eventos/checkpoints no host
 → próximo tick
```

Pontos principais em `context.rs`: preparo corporal a partir de 609; poses 785–798; sondagem de distâncias 815–884; racing line até 954; propriocepção/body até 1101; extras até 1363; cadeia GPU até 1575; submit/poll 1594–1596; intenção 1666, lento 1680, digest 1749; decisão motora a partir de 1902; física 2370; yaw 2385–2403; memory0 2407–2408. O laço externo de `main.rs` também processa vida, combustível, checkpoints e progresso por tick.

## Achados priorizados

### P1 — canais 21–27 combinam dois sensores com OR binário

`perception.wgsl:446` escreve 16–23 e o bloco seguinte escreve 24–31. `perception_motion.wgsl:219` escreve 21–27. Ambos usam `atomicOr`, com apenas uma limpeza inicial em `perception.rs:1363`, antes de spatial → temporal → motion.

Portanto, quando os dois produtores têm valores não nulos, o resultado é OR dos bytes, não sobrescrita nem soma física. Exemplo: espacial=64, movimento=32 → leitura=96. Mesmo movimento com outra parede passa a significar outro valor.

Correção proposta: definir um único proprietário por canal no braço alvo e impedir a escrita conflitante, ou desenhar fusão explícita com semântica documentada. Preservar um modo legado para avaliar campeões treinados com a mistura. Teste requerido: espacial isolado, movimento isolado e ambos; uma alteração de parede não pode modificar indevidamente um canal definido como movimento.

### P1 — escrita legada 192–207 excede o espaço por ser

`perception_temporal.wgsl:144–149` calcula `being*48 + channel/4`; o laço 437–456 escreve canais 192–207. A onda tem 192 canais/48 palavras por ser (`perception.rs:80–84`). Para um ser não final, os endereços coincidem com os primeiros 16 canais do seguinte; no limite final ultrapassam a faixa alocada. Não é erro de tipo que Naga necessariamente detecte.

**Qualificação importante:** `context.rs:1221–1231` passa os 16 hidden legados zerados em `gpu_evolve`. Nesse caminho, a escrita é OR com zero; não demonstramos corrupção numérica atual. A rotina compartilhada permanece incorreta para hidden não nulo, e outros chamadores precisam ser auditados antes de remoção. O `retina_digest.wgsl:217–220` já guarda os valores em `extra_out` separado.

Teste requerido: dois seres com padrões distintos e hidden não nulo; invariância dos 192 canais do vizinho. Não aumentar o stride para “acomodar” dados que o cérebro não consome.

### P1 de objetivo — CPU ainda participa obrigatoriamente de cada tick

Os readbacks de intenção, lento, digest e física são consumidores necessários, não apenas debug. `brain/mod.rs:1192`, `slow_brain.rs:448` e `physics.rs:855–874` fazem map/poll/recv. A física envia estado com stride 368 bytes por ser em cada chamada. O lento retorna 3 f32/ser e a intenção 96 f32/ser quando recalculada. Não somar isso como estimativa completa sem inventariar todos os uploads/flags.

Migrar somente sensores não remove o retorno: decisão motora, mundo auxiliar e yaw ainda precisam do host. Reaproveitar o device compartilhado existente é positivo, mas compartilhar device não elimina transferências.

### P2 — três canais sem escritor na cadeia principal

Na inspeção de spatial, temporal, motion e digest, não há escrita dos canais 78, 79 e 80. O registro `channels.rs:1037–1054` atribui esses canais a `motor.rs`, que não preenche a onda dessa cadeia. Após clear, ficam zero. O mesmo registro atribui 20, 28 e 32 ao motor, embora os blocos espaciais escrevam esses índices no caminho GPU.

Não usar a tabela de nomes como prova de cobertura. Entregar inventário de 192 linhas gerado/validado contra produtores reais, com proprietário por flag, fonte física, unidade, escala, atraso e consumidor. A análise identificou esses três sem produtor; não mediu variância dos demais em execução.

### P2 — CPU calcula geometria e faz muitas pequenas escritas

`context.rs:815–884` sonda grade na CPU para paredes/frente, apesar de existir percepção geométrica em WGSL. Propriocepção, distâncias, racing line e extras chamam métodos que fazem `write_buffer` por ser; exemplo `perception.rs:1070–1087` e `1188–1204`.

Poucos bytes não significam custo desprezível: há chamadas e preparação. Oportunidade: primeiro extrair sensores do estado GPU; enquanto a migração não chega, medir packing em lote. Não substituir sondagens CPU por resultados GPU sem conferir amostragem, alcance e critérios de parede/buraco.

### P2 — tempo e significado dos sensores não são uniformes

`context.rs` usa dados anteriores de bordas para canais 99/100, memória0 vem da intenção `intents[i][0]`, enquanto steer/brake vêm do estado físico aplicado. Gates regionais são calculados depois da inferência e enviados para o próximo despacho. Essas defasagens podem ser legítimas, mas precisam constar no contrato.

A distinção intenção/ação aplicada precisa ser explícita: piloto diagnóstico, reflexo, KAN ou clamp pode alterar o comando depois da intenção. Ao migrar, não realimentar acidentalmente um sinal diferente.

### P2 — corpo troca o significado de canais conforme flag

`channels.rs::BODY_SLOTS` e `perception_temporal.wgsl` transferem propriedade de 98/114–117, além das fontes de temperatura e significado de 124. Neste caso há guardas no motion para evitar dupla escrita: é um ponto positivo, diferente da colisão 21–27.

Mesmo sem colisão binária, um campeão depende desse contrato. Persistir versão do mapa e flags efetivas com resultados/genomas; testar corpo ligado/desligado. Não supor que canal válido significa mesmo sensor em todos os braços.

### P2 — aproximação documentada como tanh tem erro muito maior

`perception_temporal.wgsl:451` afirma erro menor que 2% para `h/(1+|h|)` versus tanh. Em h=1, os valores são 0.5 e aproximadamente 0.7616: erro absoluto 0.2616, relativo aproximadamente 34%. O digest preserva essa função por compatibilidade (`retina_digest.wgsl:208–220`).

Corrigir a descrição; trocar a função é alteração comportamental que exige comparação. No principal, esses 16 valores do digest não entram como canais 192–207 no cérebro de 192 entradas; verificar consumidores antes de gastar esforço para “melhorar visão” por esse bloco.

## Cobertura funcional e o que falta provar

| Grupo | Fonte/cálculo no principal | Situação |
|---|---|---|
| Retina e sondas espaciais | GPU; poses produzidas no host | GPU parcial; depende de uploads |
| Fluxo/movimento visual | GPU sobre frames atual/anterior | Colisão 21–27 requer correção |
| Velocidade/alinhamento/lateral, marcha/RPM, vida | Host → temporal GPU | Codificação GPU não torna origem GPU |
| Temperaturas, estações e predador | Modelos/preparo CPU conforme flags | Semântica configurável |
| Distâncias, racing line, alguns sinais de risco | CPU e GPU em caminhos distintos | Paridade/duplicação a validar |
| Memórias temporais e filtros | Parte GPU, parte host/retorno | Atrasos precisam de contrato |
| Contato por roda, carga, suspensão, aceleração corporal completa | Não identifiquei um conjunto dedicado completo na onda principal | Validar necessidade e influência antes de adicionar |

Cada canal deve passar três provas: física muda leitura; leitura pode mudar decisão; leitura melhora desempenho numa situação que precisa dela. A última exige experimento causal, não inspeção estática.

## Testes e próximos passos

`wgsl_validation` faz parse/tipos dos shaders listados, não valida limites dinâmicos de índices nem semântica entre produtores. Não inclui toda a família de física na lista inspecionada. Há testes de encoding/stride corporal e paridade GPU ignorada por padrão; `gpu_validate` compara física com uma implementação CPU separada. Isso não constitui cobertura ponta a ponta dos 192 canais.

Ordem sugerida: (1) testes de colisão/isola­mento/cobertura; (2) contrato de canais e versões; (3) medir `--perfil-kernel` com cenário/flags fixos; (4) mover extração corporal, comando final e yaw/eventos para buffers residentes; (5) retirar readbacks somente após remover seus consumidores de controle; (6) ampliar sensores em experiências específicas.

Todos os testes com device e perfis passam pela fila `nmd`, sem concorrência GPU. Esta auditoria não executou testes GPU nem mediu percentuais de uso/tempo CPU. O alvo técnico é remover dependências CPU do ciclo por ser/tick; CPU para despachar, salvar e coordenar continua apropriada.
