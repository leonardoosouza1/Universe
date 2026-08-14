---
name: vigilia
description: Ciclo autônomo 24h do Universe — o script que o Claude segue ao se auto-acionar (ScheduleWakeup) enquanto Leonardo está fora. Use ao iniciar operação autônoma ou ao acordar de um wakeup sem instrução nova.
---

# Vigília — operação autônoma contínua

Leonardo: *"você deve usar loop, uma ferramenta que te aciona a cada X tempo e você
valida o que está acontecendo — rodar 24h por dia sem eu ter que pedir."*

O mecanismo: **ScheduleWakeup** (1200-1800 s entre ciclos; mais curto só se estiver
vigiando corrida específica). TODO turno termina com um wakeup armado E processo rodando.
Máquina ociosa = falha do ciclo.

## O ciclo, a cada acordar

1. **Colher** — processos vivos? (`ps -eo pid,etime,args | grep gpu_evolve`) Baterias
   terminaram? (`grep -c '^gen ' <run.log>`) Workflows notificaram?
2. **Validar** — números contra o esperado, com o protocolo `experimento-universe`:
   teto da métrica, ≥3 sementes, média das últimas 5 gens, vivos vs mortos, canário de
   números idênticos. Cérebro: skill `medir-cerebro` (cosseno gen0→genN é o juiz).
3. **Decidir** — consultar a fila (abaixo). Passou → próxima etapa. Falhou → voltar uma
   casa e fatiar (não-regressão ≥80%). Travou → matar, diagnosticar com smoke de
   segundos, relançar.
4. **Registrar** — resultado relevante vai para a memória persistente e/ou
   `docs/PLANO_EVOLUCAO.md`. Descoberta nova = arquivo de memória novo.
5. **Re-armar** — lançar o próximo trabalho pesado (bateria/workflow) e SÓ ENTÃO
   agendar o wakeup. Nunca o contrário.

## A fila canônica (ordem de docs/PLANO_EVOLUCAO.md + docs/CEREBRO_VIRTUAL.md)

- **P0** equivalência estatística pós-aceleração (3 sementes vs referências:
  s77=3189m/25 voltas · s31=519m · s113=903m)
- **P0.5** init espectral |λ_max|≈1 (hoje 7,0) — matemática fechada, maior ganho barato
- **P1** anti-decoreba: sementes rotativas + generalização (razão nova/treino)
- **P2** corpo: grip→temperatura→pneu→furo, um por vez, ablação cada
- **F2** metabolismo: custo por spike (moeda separada da vida), aluguel por peso,
  eficiência na fitness
- **F3** regiões (gatilho: L2 saturada persistente) · **F4** glia · **F5** sono/competição

## Regras duras (aprendidas pagando)

- Workflow morto por limite de sessão → retomar com `{scriptPath, resumeFromRunId}`
- Saída SEMPRE em `/media/llm-data/universe-runs/` (hook bloqueia /tmp)
- Bateria longa só depois de smoke de segundos
- Paridade bit-a-bit é impossível com raio espectral 7 — usar equivalência estatística
- Relatório honesto: número ruim se reporta; "não medi" em vez de descrição
- Se Leonardo mandou mensagem nova, ela SEMPRE vence a fila
