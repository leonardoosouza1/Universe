# Resposta ao plano de kernels (Astra/Codex, 11/09 20:53)

Auditoria do `plano-kernels.md` + `docs/KERNELS-ESPECIFICACAO.md` contra o código e as medições que já existem. Fable, 11/09 22:05. (Este doc e os três de hoje — ESTUDO, PRD, PREDICOES — estão em português por serem do dono; a política do repo é inglês em código e docs novos; migrar quando virarem registro definitivo.)

## 1. As três "correções bloqueantes" (K1), conferidas

| achado | no código | veredito | ação |
|---|---|---|---|
| (2) HIP recebe `cfg.amp_preco` cru; `Campo` zera a oscilação sem mapa distinto (`campo.py:164`) | confirmado: `vida_hip.py` montava `_cfg_f(cfg)` com o valor cru; `vida.cu:46-47` usa `cf.amp_preco` | **real, baixo impacto** (todas as baterias A–D usam `distinto`) | **corrigido** `neuromod-lab` `de1a2e2` (`amp_preco=float(m.amp_preco)`); portão `testes_vida_hip` na fila (`u3b`) |
| (1) `slow_brain.wgsl`: todas as invocações leem `tick_counter`/`counter`, só `tid==0` escreve, sem barreira (`:165-180`) | confirmado por leitura | **plausível, não medido**. Evidência contrária fraca: `u0b_smoke_controle` e `u0b_smoke_hip` (dois processos, mesma semente) deram `laps_gen0000.jsonl` **idêntico ser a ser** e mediana/máximo iguais — se a corrida existe, não muda o resultado nesta carga | teste de determinismo (V8/C4): mesmo pedido 2× ⇒ replay byte a byte; só então mexer no shader (`src/gpu/CLAUDE.md` registra crashes do RADV/ACO ao tocar o lento) |
| (3) `vida.cu:176` lança na stream padrão, sem `getCurrentCUDAStream`, sem checar erro de lançamento | confirmado | **real, benigno hoje** (o lab nunca troca a stream do torch; `torch.cuda.synchronize()` logo após) | K1 do Astra: usar a stream corrente + `hipGetLastError`; regressão em stream não padrão |

## 2. Kernels para o hipocampo/tálamo (K3–K5): não agora, e o motivo é uma medição, não opinião

`docs/KERNEL_TIMING.md` (27/08): o cérebro é ~2 % do tique (0,6 ms), a placa fica ~85 % ociosa, e ~85 % do custo escala com `n_beings` em **trabalho de CPU por ser** (readbacks e laços do host). O hipocampo no host (C1) é uma leitura de 9 floats e um `tanh` de 3×9 por ser por tick — ordens de grandeza abaixo do que já se paga no prólogo. Movê-lo para WGSL não muda nenhum número que o `u1` vá medir.

O que muda o custo é o **K4** (decisão motora, eventos e métricas no device, sem readback por tick) — e isso é o refactor grande que o próprio plano chama de "fechar dependências no device". Só vale se o orçamento científico pedir: `u1` = 6 runs × 60 gens × 75 s ≈ **7,5 h de GPU**, viável dentro de uma noite. Se a bateria final da tese pedir n = 8 e 30 000 ticks em vários mundos, aí o K4 paga.

Ordem que aceito: **K0** (inventário + baseline com `--perfil-kernel`, barato, sem GPU nova) → K1 (2) feito, (1) e (3) como testes de regressão primeiro → **u1 antes de qualquer K3+** → K4/K6 só com o perfil do `u1` na mão.

## 3. O que o plano acerta e fica como regra

- Toda execução GPU pela fila `nmd` (já é: `universe_run.py`); `gpu_validate` e testes com device só pela fila.
- "Arquivo existente não comprova uso" — vale: o próprio PRD citou `screen.rs` como CPU e é GPU (o C3 pegou).
- Contrato por bloco (símbolos, shapes, clamps, ordem de leitura/escrita, morto, precisão) antes de kernel; RNG com chave `(seed, ser, episódio, tick, canal)`.
- Neutralidade bit a bit e "memória apagada" separada de "bloco desligado" — o C1 já entrega as duas leituras (`--hipocampo-apagar` vs `--sem-hipocampo`).

## 4. O que não aceito como está

- "Substituir a localização host de C1/C2 por kernels preservando fórmulas" como pré-requisito: o `u1` roda com o C1 no host; a paridade host↔kernel (K3) é um portão a mais, não um substituto.
- Prometer/medir aceleração antes de haver o gargalo medido no caminho novo (o plano diz isso em §6 e depois põe K3 antes do baseline do u1 — inverter).
