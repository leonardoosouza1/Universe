# Plano de kernels — execução pelo Claude, revisão pelo Codex

Data: 2026-09-11. Base inspecionada: Universe `f515192`, neuromod-lab `8417497`, incluindo arquivos locais existentes. Nenhum kernel foi implementado ou medido nesta entrega.

## Objetivo e escopo

Executar na GPU as equações por ser e por tick, com estados residentes e resultados cientificamente equivalentes. Rust coordena recursos, experimentos e persistência, e mantém referências matemáticas para testes.

Este plano complementa [PRD-CEREBRO-NEUROMOD](docs/PRD-CEREBRO-NEUROMOD.md): propõe substituir a localização host de C1/C2 por kernels, preservando inicialmente fórmulas, flags, layout de genoma e protocolo experimental. A referência Rust pode ser feita primeiro para validação, mas não é o destino de produção. Não antecipa C2 nem libera as baterias que o PRD condiciona ao dono. O mundo táxi e seu PRD V1–V9 continuam uma frente separada.

## Ordem de execução

| Etapa | Entrega do Claude | Aceite | Dependência |
|---|---|---|---|
| K0 | Inventário do caminho ativo, contratos de buffers/equações e baseline | Cada etapa ligada a chamada real, bytes transferidos e tempos; manifesto reproduzível | — |
| K1 | Correções de contadores WGSL, amplitude de preço HIP e stream/dispositivo HIP | Regressões específicas e equivalência; relatar separadamente cada repositório | K0 |
| K2 | Referências puras e fixtures para hipocampo, gates, decisão e eventos | Fórmulas, ordem temporal, neutralidade e serialização cobertas | K0; alinhar C0/C1/C3 |
| K3 | Hipocampo GPU: leitura, aplicação e escrita por evento | Paridade com referência, reset, isolamento entre seres, ablações | K1, K2 |
| K4 | Fechar decisão motora/eventos/métricas na GPU; remover dependências CPU por tick | Traço mostra ausência de readback necessário ao próximo tick; paridade do mundo | K3 |
| K5 | Interocepção/tálamo GPU quando C2 estiver habilitado pelo protocolo | Gates neutros, cobertura dos canais e canários do C2 | K4 e portão científico C2 |
| K6 | Otimizações medidas: layout, despacho, blocos de ticks e matemática | Ganho end-to-end repetível com todos os portões de correção verdes | K4; K5 se ativo |
| K7 | Pacote final para revisão independente | Diff, comandos/logs, manifests, resultados brutos, limitações e rollback | Etapas escolhidas concluídas |

K1 HIP é independente do port WGSL após K0; não colocar HIP no caminho de produção Vulkan para cumprir esta etapa. Claude deve revalidar o estado do código antes de aplicar um achado já corrigido.

## Entrega e limites

Seguir a [especificação técnica](docs/KERNELS-ESPECIFICACAO.md). Fazer mudanças pequenas, revisáveis, separando correção, migração equivalente e mudança científica. Cada etapa entrega evidência antes da seguinte; caixas não devem ser marcadas por intenção.

Ficam fora: reescrever o projeto, trocar backend/driver, portar toda a evolução sem perfil, alterar arquitetura científica ou prometer fator de aceleração. Esses trabalhos não são necessários para estabelecer primeiro um ciclo GPU correto e mensurável.

Toda execução GPU passa pela fila `nmd`, conforme o PRD C4; sem competir com o Claude ou outro experimento. Builds com `CARGO_BUILD_JOBS=4` e prioridade reduzida. Respeitar o limite local de 80% dos recursos. Treino longo continua dependendo da liberação existente no PRD.
