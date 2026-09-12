# Resposta à auditoria de sensores/fluxo CPU–GPU (Astra/Codex, 12/09)

Auditoria de `docs/AUDITORIA-SENSORES-CPU-GPU-2026-09-12.md` e `docs/INSIGHTS-NEUROEVOLUCAO-DESENVOLVIMENTO.md`,
conferida contra o código em `70bc9e9` por um agente de leitura (Sonnet) e consolidada por Fable, 12/09 11:10.
Nenhum binário rodou; nenhum código foi alterado por causa destes docs. Os dois arquivos entram no repo como recebidos.

## Auditoria de sensores — o que confere

| # | achado do Astra | veredito | evidência |
|---|---|---|---|
| 1 | canais 21–27 são OR binário espacial×movimento (um `clear` só antes da cadeia) | **confirmado** | `perception.rs:1363` (comentário do próprio autor: "ambos os kernels usam atomicOr") |
| 2 | escrita legada ch 192–207 estoura o buffer por ser (`WAVE_U32S = 48`, `ch/4 = 51 > 47`) e cai no ser seguinte | **confirmado, inofensivo hoje** | `perception.rs:80-84`; `context.rs:1215-1231` zera `hidden[1..17]` no `gpu_evolve`; `retina_digest.wgsl` já registra o mesmo aliasing (P1.8) e desvia para `extra_out` |
| 3 | a CPU consome readbacks obrigatórios todo tick (intenção, lento, física) | **confirmado** | `gpu/brain/mod.rs:1192`, `gpu/slow_brain.rs:449`, `gpu/physics.rs:855-874` — `map_async → poll(Wait) → recv` |
| 4 | canais 78/79/80 sem escritor | **refutado** | `perception.wgsl:702-722` (`tid == 59u`) escreve 78/79/80 para todo ser; `perception.rs:831` cobre "0–97, 128–191". O que está errado é o REGISTRO `channels.rs:1037-1054`, não o kernel |
| 5 | comentário "erro < 2 %" do tanh aproximado `h/(1+|h|)` | **confirmado falso** | em h = 1: 0,762 vs 0,5 (34 %); em h = 3: 24,6 %. É comentário, não bug de cálculo: o cérebro evoluiu COM essa função |

Não conferidos neste orçamento: sonda de geometria na CPU, semântica temporal não uniforme, troca de canal por `BODY_SLOTS`.

## O que fazer, e quando

- **Correção pura (1, 2, 5) e o registro (4):** entram como flag com A/B pareado (regra 3 do `CLAUDE.md`), DEPOIS do `u1`
  — o `u1` compara hip × controle com o MESMO binário; trocar o OR por overwrite no meio muda os dois braços e some com a
  comparação. Ordem: corrigir `channels.rs` (só texto) → apagar o comentário "< 2 %" → flag `--canais-overwrite` com
  teste de identidade desligada → smoke pareado.
- **Migrar extração corporal/comando/yaw para buffers residentes e remover readbacks:** é o K3/K4 do
  `plano-kernels.md`, e a resposta continua a de `docs/RESPOSTA-PLANO-KERNELS.md`: só com o perfil do `u1` na mão e um
  gargalo medido. O `KERNEL_TIMING.md` já diz onde o tempo vai (85 % em trabalho de CPU por ser) — o que falta não é
  saber, é decidir se a bateria da tese precisa dessa velocidade.

## Insights de neuroevolução — o que confere

Confirmado: composição do `Genome` (`evolution.rs:93-129`), torneio k = 3 + elitismo sem crossover, `dispatch_hebb` só
em `cortex_evolve/test/map` (folha com 223 entradas contra 192 do cérebro principal — `cortex_evolve.rs:63` se declara
"OUTDATED"), fórmulas do hipocampo idênticas a `gpu/hippocampus.rs`. Impreciso: os `kan_knots` CONTINUAM mutando no
genoma (`evolution.rs:1019-1025`), só não são lidos de volta — "sem efeito", não "sem mutação".

Pedidos (bloco plástico à la Miconi, ambientes com mudança intra-vida, desenvolvimento/NEAT/NCA): agenda paralela, sem
conflito com `u1`/`u2` — o próprio doc diz que não altera o protocolo aprovado nem libera treino. Fica em
`META/PLANO-FECHAMENTO-2026-09-11.md` Parte 3b como "depois do veredito C1".
