# PRD — cérebro neuromod no Universe: hipocampo por setor, tálamo por necessidade, método

Guia para agentes em paralelo (Opus/Sonnet/Astra). Base: `docs/ESTUDO-CEREBRO-NEUROMOD.md` (o que existe, o que é novo). Repositório `Universe`, branch `race-world-v3`, caminho de produção `gpu_evolve` (`src/bin/gpu_evolve/`, `src/gpu/`). Toda linha citada foi conferida em 11/09.

## 0. Regras (não negociáveis)

1. **Inglês em tudo** no código (`ARCHITECTURE.md:3-8`, `scripts/check_language.sh`); testes nomeados como frase (`docs/STYLE.md:295-299`); teto 600 linhas por `.rs`, 400 por `.wgsl` — **não empilhar em `motor.rs` nem `world.rs`**.
2. **Flag off por padrão + braço de controle bit-idêntico** (`CLAUDE.md:60-66`). Todo gene novo nasce em zero e reproduz o cérebro antigo bit a bit (contrato `src/gpu/evolution.rs:975-982`).
3. **Genoma é contrato de disco**: campo novo entra por ÚLTIMO em `Genome` (`src/gpu/evolution.rs:93`), em `save_genome` **e** `load_genome` (`src/gpu/genome_io.rs:53, 89`), como bloco opcional com default neutro documentado; estender `genoma_sobrevive_ida_e_volta_no_disco` (`src/bin/gpu_evolve/tests.rs:24`) com valor que não pode vir do default. Nunca reordenar.
4. **Sem GPU fora da fila.** Agentes não rodam `gpu_evolve`, `gpu_loop`, `gpu_validate` — só `cargo build`/`cargo test` de testes sem device. A GPU é compartilhada com a fila `nmd` do neuromod-lab (um processo na GPU); qualquer execução na GPU vira pedido na fila (§C4). `nice -n 10`, `CARGO_BUILD_JOBS=4`.
5. **Treino proibido até o dono liberar** (`docs/ROADMAP.md:90-92`). Smokes de 1 geração curta são permitidos, pela fila.
6. **Predição antes de qualquer veredito**, negativo com mecanismo, pareado por semente. Registro em `docs/PREDICOES-CEREBRO.md` (criado no C4).
7. Saídas pesadas só em `/media/llm-data/universe-runs/universe-cerebro/`.
8. Relatório final obrigatório: arquivos, testes antes/depois, números impressos, o que ficou fora, discrepâncias com este PRD. Nunca Fable em subagente.

## 1. Alvo do experimento

**Braço controle**: campeão m23-s77 (M2+M3), `--mascara-regioes --tau-semeado`, sem gates. **Braço novo**: o mesmo + `--hipocampo-setor`. Depois (C2): + `--interocepcao --talamo-necessidade` com energia escassa. Comparação pareada por semente, mesma pista por geração, 20 000 ticks (para haver volta 2), a partir do mesmo campeão.

Hipótese mecanicista (estudo §3): na volta 2 o ser tem informação que a percepção não alcança em alta velocidade — o que aconteceu neste setor e no próximo na volta 1. Teste de ablação: `--hipocampo-apagar` na avaliação.

## 2. Partes

Grafo: **C0 ∥ C1 ∥ C3 ∥ C4** (independentes) → smoke (C4) → **bateria** (dono libera) → **C5** análise → **C2** (só depois da bateria e com energia rodada 3).

### C0 — Telemetria por volta e flags de ablação

**Objetivo.** Hoje não existe tempo/distância por volta (`cruzamentos` guarda só o 1º cruzamento por setor, `src/bin/gpu_evolve/fantasma.rs:94-99`; `ReplayFrame.total_laps` permite derivar offline, `src/evolution/telemetry.rs:24`). A métrica principal do experimento é volta 2 vs volta 1.

**Entrega.**
- `src/bin/gpu_evolve/laps.rs` (novo, ≤ 200 linhas): por ser, `Vec<LapRecord{lap, tick_start, tick_end, dist, sector_ticks: [u32; 24], grass_ticks, damage, dry_events}>`, alimentado nos pontos que já existem: cruzar setor `main.rs:3055-3060`, volta `main.rs:3085-3094`, grama `context.rs:937/984`, dano `context.rs:1949-1964`, seco `context.rs:1727-1733`.
- Gravação por geração em `<out>/laps_gen<N>.jsonl` (uma linha por ser: sementes, braço, laps[]) e resumo no log da geração: `laps: med 1.8 | lap2/lap1 dist +3.1 % | ≥2V 31/60`.
- Flags (`src/bin/gpu_evolve/statics.rs:40-148`, padrão atomic global): `--hipocampo-apagar` (C1 lê MEM como zero), `--sem-hipocampo` (congela mutação dos genes do C1 e força bias 0 — o braço controle com o MESMO layout de genoma).

**Aceite.** `tests/laps.rs`: sequência sintética de cruzamentos/voltas ⇒ `LapRecord` esperado; ser sem volta ⇒ `laps` vazio; `lap2/lap1` só calculado para quem tem 2 voltas. `cargo test --release --bin gpu_evolve` verde.

### C1 — Hipocampo por setor (host-side, sem WGSL)

**Objetivo.** Memória por contexto escrita por evento e lida como viés nos comandos. Implementação **no host**, no mesmo lugar onde o lento já modula os comandos (`src/bin/gpu_evolve/context.rs:1862-1907`) — nada de binding novo (restam 2, `context.rs:261-270`; guardar para o C2).

**Fonte da verdade (neuromod-lab, para a forma, não para copiar).** `completo.py:246-248` (`_mem`: memória do contexto atual × ganho `tanh(g)`), `:274-286` (`atualizar`: EMA com `1/tau_h`, escrita gateada por evento em `oh_o`/`oh_n`), `:224-226` (`tau_h = 1 + 200·sigmoid(g − 4)`, `g_hip = tanh(g)`: genes 0 ⇒ ganho 0 exato).

**Desenho.**
- Estado por ser: `mem: [[f32; NM]; 24]`, `NM = 4`: `[speed_mean/40, grass_frac, damage_norm, dt_improvement]` do setor na última passagem (`dt_improvement = clamp((t_prev − t_now)/t_prev, −1, 1)`, 0 na 1ª passagem). Zerado no nascimento (por geração/vida).
- Escrita: no cruzamento do setor `s` (evento C0/`main.rs:3055`), `mem[s] += α·(x − mem[s])`, `α = sigmoid(hip_tau_gene)` (gene 0 ⇒ α .5).
- Leitura, a cada tick: `x = [mem[s], mem[(s+1) % 24], 1]` (9 valores; `s` = setor atual, `main.rs:3037-3050`); `bias = hip_gain · tanh(W_h · x)`, `W_h ∈ R^{3×9}` (27 genes), `hip_gain = tanh(gene)`.
- Aplicação (`context.rs:1862-1907`, **depois** do lento e **antes** do alisamento α): `throttle += 0.5·bias[0]`, `steer += 0.5·bias[1]`, `brake += 0.5·bias[2]`, depois os clamps já existentes. Escalas 0.5 documentadas no código.
- Genoma: `hip: [f32; 29]` (27 + tau + gain), **último** bloco do `UGEN v2` (depois de `gates_w`), mutação gaussiana com `p_mutate`/σ do slow (`evolution.rs:897-1055`); `--sem-hipocampo` congela.
- `--hipocampo-apagar`: `x = [0…0, 1]` ⇒ bias constante `hip_gain·tanh(W_h[:,8])` — **não** é zero; para ablar de verdade zera também o bias. Documentar as duas leituras: `apagar` (memória zero, viés de repouso fica) e `sem-hipocampo` (tudo zero).
- Observação: `<out>/hip_gen<N>.f32` para o ser 0: por tick `[tick, s, mem[s][0..4], mem[s+1][0..4], bias[0..3]]` (14 f32) — arquivo separado do `obs.f32` (394 colunas fixas, `context.rs:2059-2091`; não mudar).
- Módulo puro `src/gpu/hippocampus.rs` (sem GPU, sem I/O; padrão `src/predator.rs:9-13` "NOT wired" até o `context.rs` chamar).

**Aceite.** `tests/hippocampus.rs`: (a) genes zero ⇒ `bias == [0,0,0]` para qualquer `mem` (bit a bit); (b) `--hipocampo-setor` com genes zero em 2 000 ticks determinísticos (fixture de comandos gravada do controle — usar `screen_brain` open-loop, `src/gpu/brain/screen.rs`) ⇒ comandos idênticos ao controle; (c) sensibilidade: genes `W_h` em ±0.5 e `mem` sintético ≠ 0 ⇒ `|bias| > 1e-3` em cada uma das 3 saídas, e `bias` cresce monotonicamente com `hip_gain`; (d) escrita: 3 passagens sintéticas no setor 5 com α = .5 ⇒ `mem[5]` esperado por conta fechada; (e) `genoma_sobrevive_ida_e_volta_no_disco` com o bloco `hip` marcado; (f) genoma antigo (sem bloco) carrega com `hip` neutro.

### C2 — Interocepção + tálamo por necessidade (só depois da bateria C1)

**Objetivo.** Dar ao cérebro o corpo (M1 do ROADMAP) e, sobre ele, o portão de entrada por família × necessidade — o tálamo do lab.

**Fonte.** `completo.py:236-244` (`_need = max(déficits)`, `gate = 2·sigmoid(a + b·need + c·pax)`, genes 0 ⇒ 1.0 exato; normalizações teto/fixo); famílias de canais `src/neuro.rs:171-182`; M4 como referência de plumbing (`context.rs:1600-1624`, `brain.wgsl:871`).

**Entrega.**
- `--interocepcao`: `fuel`, dano oculto, `engine_temp`, `life_left` nos canais mortos ch114-117/98/124 (ROADMAP M1; hoje só `ch117 OwnFuel` com body-block, `channels.rs:907-911`). Com a flag off, canais como hoje.
- `--talamo-necessidade`: host calcula por ser 6 gates de família `g_f = 2·sigmoid(a_f + b_f·need)`, `need = max(1−fuel, 1−life, tire_temp_norm)`; genoma `tal: [f32; 12]` (a, b por família), bloco opcional após `hip`; upload em `buf_in_gates` (6 f32/ser, binding 14); `brain.wgsl` multiplica cada canal pelo gate da sua família **antes** da L1 (onde a onda é lida para o produto com `buf_l1w`).
- Só faz sentido com energia escassa: `--fuel-tank 0.25` + refuel por setor (rodada 3, `docs/REGIOES_CEREBRAIS.md` fim; flags em `main.rs:1193-1211`).

**Aceite.** (a) gates ≡ 1 ⇒ `gpu_validate` cérebro idêntico (`src/bin/gpu_validate.rs:390-510`, pela fila); (b) mapa de famílias cobre os 192 canais exatamente uma vez (teste); (c) genes zero ⇒ gate 1.0 exato em f32; (d) dose canary ORDENADO (`docs/M0_DOSE_CANARIES.md`): tanque 0.25/0.35/0.5 dão mortes por seco em ordem.

### C3 — Método: dof, contribuição por bloco, sensibilidade

**Objetivo.** As três réguas do lab que o Universe não tem.

**Entrega.**
- `dof`: por geração e por ser, nº de genes novos com `|w| > 0.5` (`hip`, `tal`, `gates_w`), no log e no `laps_gen<N>.jsonl` (`dof_hip`, `dof_tal`).
- Contribuição: média de `|bias_hip|` por saída e de `|g_f − 1|` por família, por geração (no mesmo jsonl).
- `src/bin/sensitivity.rs` (sem GPU): para cada grupo de genes novos, perturba ±δ e mede Δ dos comandos num traço open-loop gravado (`src/gpu/brain/screen.rs`); imprime matriz grupo × saída; falha se algum grupo não move nenhuma saída. É o `testes_sentidos.py` do lab (`neuromod-lab/src/testes_sentidos.py:1-3`), que teria pego em 30 s o achado 84.

**Aceite.** `sensitivity` verde para `hip` (C1) antes de qualquer pedido na fila; `dof` = 0 na geração 0 dos dois braços.

### C4 — Fila, smoke, bateria, predições

**Objetivo.** Rodar na GPU sem violar "um processo na GPU": tudo pela fila `nmd` do neuromod-lab.

**Entrega.**
- `neuromod-lab/src/universe_run.py` (wrapper, sem torch): recebe `--bin <nome>` e `--args-json '[...]'`, executa `~/projects/LLM/Universe/target/release/<bin>` com `cwd = Universe`, `nice -n 5`, stdout/stderr no log do pedido, `--out` do pedido vira `--saida` do binário; exit code propagado. O binário tem de estar **compilado antes** (`cargo build --release`, CPU, fora da fila).
- `docs/PREDICOES-CEREBRO.md` com a tabela `| nº | predição | aposta | veredito |`, no formato do `neuromod-lab/PREDICOES.md`.
- Pedidos (ids em ordem byte a byte): `u0_smoke_controle` e `u0_smoke_hip` (1 geração, 60 seres, 20 000 ticks, `--retomar` do m23-s77, medir s/geração e mediana de voltas — **portão de mundo**: mediana ≥ 1.5 voltas, senão subir ticks); `u1_{ctrl,hip}_s{31,77,113}` (60 gens, `--gens 60 --ticks 20000`, pista rotação `[8005,8200,9001]`, mesma partida); `u2_aval_{ctrl,hip}_{mem,apagar}` (campeões, `--gens 1`, 3 pistas × 48 cópias).
- Predições a registrar ANTES do `u1` (aposta numérica, com o lado do lab: MDE do controle medido no smoke):
  1. `lap2/lap1` dist do braço hip > controle em ≥ 3 % na mediana das 3 sementes (mecanismo §3 do estudo).
  2. ≥2V/gen do hip ≥ controle (não regride) — pré-condição: memória não pode custar volta 1.
  3. `u2`: campeão hip com `--hipocampo-apagar` perde ≥ 50 % do ganho de (1); se não perder nada, a memória é decoração.
  4. `dof_hip` ≥ 6 nos campeões (a evolução liga o bloco); se ficar 0–2, o bloco é neutro e (1) tem de dar nulo.
  5. Controle negativo: `--sem-hipocampo` == controle bit a bit na gen 0 (portão, não predição).

**Aceite.** `u0` verde pela fila com as duas linhas de resumo (s/geração, mediana de voltas) no log; `PREDICOES-CEREBRO.md` com 1–5 preenchidos antes do `u1` entrar na fila.

### C5 — Análise

`tools/laps.py` (Python padrão + matplotlib, CPU, sem torch): lê `laps_gen*.jsonl` dos 6 runs; tabela pareada por semente (controle × hip) de `lap2/lap1`, `≥2V/gen`, `net_max`, `dof_hip`, `|bias|`; t pareado, MDE; curva por geração; PNG em `<out>/voltas.png`. Lê também `u2` (mem × apagar). Saída Markdown ≤ 40 linhas para o doc de veredito.

## 3. Ordem de lançamento

| onda | agentes | depende |
|---|---|---|
| 1 | **C0**, **C1**, **C3**, **C4** (wrapper + PREDICOES esqueleto) — 4 worktrees | — |
| 2 | merge C0+C1+C3 em `race-world-v3`; `cargo build --release`; pedidos `u0` (smoke) | onda 1 verde |
| 3 | dono libera treino → `u1` (6 runs) → `u2` | u0 + predições |
| 4 | **C5** + veredito em `docs/VEREDITO-HIPOCAMPO.md` | u1, u2 |
| 5 | **C2** (interocepção + tálamo) com energia rodada 3, mesmo protocolo | veredito C1 |

Merge só com testes verdes e relatório; conflito em `context.rs` resolve-se mantendo a ordem: KAN → lento → **hipocampo** → alisamento → clamps.

## 4. Fora deste PRD

Portar `SerCompleto` inteiro; córtex por distrito (a L2+KAN já condicionam por curvatura vista); reflexo no `BrainGpu` (sem combustível que acabe não há o que refletir — entra com C2 se a rodada 3 morder); custo de fiação (reprovado 2×); WGSL para o hipocampo (host basta: 60 seres × 24 setores).
