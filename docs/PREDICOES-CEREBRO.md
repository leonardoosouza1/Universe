# Predições — cérebro neuromod no Universe

Formato do `neuromod-lab/PREDICOES.md`: registrar ANTES de rodar; negativo com mecanismo é resultado. Números de aposta que dependem do smoke (`u0`) ficam marcados `[após u0]` e são preenchidos antes do `u1` entrar na fila — nunca depois de ler o `u1`.

| nº | predição | aposta | veredito |
|---|---|---|---|
| U1 | **Hipocampo por setor paga na volta 2** (PRD C1; estudo §3). Mecanismo: em alta velocidade a percepção não alcança a curva (horizonte d_freio); a memória do setor atual e do próximo, escrita na volta 1, é informação fora dos 192 canais. | `lap2/lap1` (distância) do braço `--hipocampo-setor` supera o controle em ≥ 3 % na mediana pareada das 3 sementes (31, 77, 113), 60 gens, 20 000 ticks, partida m23-s77, rotação `[8005, 8200, 9001]`. MDE do controle `[após u0]`. | pendente |
| U2 | **A memória não custa a volta 1** (pré-condição). | `≥2V/gen` do braço hip ≥ controle (Δ pareado ≥ −5 %). Se regredir > 5 %, a memória atrapalha antes de ajudar e U1 não vale. | pendente |
| U3 | **Ablação na avaliação** (proposta do Astra). | Campeão hip avaliado com `--hipocampo-apagar` (3 pistas × 48 cópias) perde ≥ 50 % do ganho de U1. Se não perder nada, a memória é decoração — e o ganho de U1, se houver, veio de outro lugar (viés de repouso `W_h[:,8]`, que `apagar` mantém — por isso comparar também com `--sem-hipocampo`). | pendente |
| U4 | **A evolução liga o bloco.** | `dof_hip` (genes com \|w\| > 0.5 entre os 29) ≥ 6 nos campeões das 3 sementes. Se ficar em 0–2, o bloco é neutro e U1 tem de dar nulo (consistência). | pendente |
| U5 | **Controle negativo (portão, não aposta).** | Braço `--sem-hipocampo` == controle bit a bit na geração 0 (mesmos comandos por tick nas 2 000 primeiras iterações do ser 0). Vermelho ⇒ nada acima vale. | pendente |
| U6 | **Portão de mundo** (lição do táxi, `neuromod-lab/docs/89`). | No `u0` a mediana de voltas em 20 000 ticks é ≥ 1,5; senão a volta 2 quase não existe e a memória não tem onde pagar — subir ticks antes de qualquer bateria. | pendente |
| U7 | **Tálamo por necessidade só paga com corpo e escassez** (C2, depois do veredito C1). | Com `--interocepcao --talamo-necessidade` e `--fuel-tank 0.25` + refuel por setor: mortes por seco caem gen a gen (dose canary ORDENADO 0.25/0.35/0.5) e o braço tálamo supera o controle em completagem ≥ 5 %. Sem energia escassa: empate (como M4 em 16/08 e preds 165/179/180 do lab). | pendente |

Registro de veredito: número, arquivo (`/media/llm-data/universe-runs/universe-cerebro/...`), pareado por semente, t, e o mecanismo — nunca só ✓/✗.
