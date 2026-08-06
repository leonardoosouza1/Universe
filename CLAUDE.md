
---

## Regra de continuidade (Leonardo, 2026-08-03)

> **Quando Leonardo avisa que vai sair, reportar NÃO é parar.** O turno só acaba quando ele
> voltar. Ele deixa a máquina ligada horas contando com isso — parar depois do relatório
> desperdiça o tempo todo.

Como cumprir, sem depender de memória:

1. **Nunca reportar com a máquina ociosa.** Deixar processo rodando (`nohup`) ANTES de escrever.
2. **Agendar o próprio re-acionamento** com `/loop` (auto-pacing) ou `ScheduleWakeup`. Ao
   acordar: checar o que os processos produziram, validar, seguir para o próximo alvo.
3. **Fila de alvos explícita** — quando um fecha, o próximo já está definido, sem perguntar.
4. **Agentes em paralelo** para frentes independentes, com posse de arquivo sem sobreposição.
