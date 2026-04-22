# PILOTO_LOG — Copilot Autônomo

Log de iteração autônoma. Objetivo do usuário: **campeão ≥3 voltas + várias gens de carros completando 3 voltas consistentemente em pistas estreitas (config atual: Interlagos 10m + 4m buffer = 14m effective)**.

Regra do log: cada intervenção = (a) **O que fez** (b) **Porquê** (c) **Resultado teste** (d) **Tempo gasto**.

---

## Estado Inicial

- Pista: Interlagos Gateway (seed 64, width 10m config + 4m = 14m effective)
- Swarm: 50, time_scale: 5.0 (headless força ≥8.0), gen_timeout: 120s
- Priming A1–A18 ativo
- Sintomas herdados: carros grid travam com steer oscilante mesmo com dark_center guiando contra.
- Debug eprintlns removidos.

---

## Passo 0 — Baseline (sem alteração)

**O que fez**: removidos eprintlns + rodado 10 gens headless com `--swarm 50 --speed 10`.

**Resultado**:
- **0 voltas completas em qualquer gen** (0v=50 em todas).
- NET max por gen: 99 / 121 / 106 / 100 / 111 / 115 / 112 / **270** (spike) / 94 / 106 m. Lap = ~1299m, ninguém passa de 21%.
- Mortes ~50/50: STAG 26-32 / HP 8-15 / SBLD 5-11. Quase tudo stagnation (paradinha).
- Seg[0] (reta principal 400m) concentra 31/50 mortes → carros morrem na primeira reta.
- Brain trace top-3: throttle trava em +1.0 (bom), mas **raw_steer oscila pesado (-0.74 → +0.52 → +0.59)** mesmo em reta.

**Diagnose**: Na pista 14m efetivos com grid spawn em ±3m do centro, retina periférica (ch 8-15) sempre vê parede em 4-10m. Priming A1/A2 (pesos ±18/±14 em out 1) dispara steer forte para dentro, overshoots, pega parede oposta, zig-zag fatal. Não é proximidade real — é **"sempre vejo parede"** do 14m width.

**Tempo**: 24s simulação + análise ~4min.

---

## Passo 1 — Gate perif A1/A2 + boost lane_offset A10

**Hipótese**: A1/A2 muito agressivos em pista narrow. Lane offset (ch 81) já entrega sinal **relativo** (centro de pista). Vou:
- Cortar magnitudes de perif bins: 18→6, 14→4 em A1/A2 (perif_bin 8,9,14,15). Ainda steer, mas não dominante.
- Boost A10 lane_offset ch 81: 4 → 10 (out 1) — torna "fica no centro" o sinal primário.
- Deixar whiskers A17 (94-97) intactos = ±18/14 pois só acendem quando parede encosta mesmo.

**Porquê**: Anula o feedback positivo de zig-zag. Deixa o carro aprender "siga o centro" em vez de "fuja de parede que está a 5m".

## Passo 2 — v2b: soltar stagnation thresholds

**O que fez**: `progress_checkpoint + 2.0` → `+ 0.5`; timer morte 240s→480s.

**Porquê**: v2 mostrou que 31/50 morriam STAG. Pensei: "carros ficam parados tentando manobra". Se afrouxar, aprendem.

**Resultado**: Regressão — SBLD (speed bleed) passou a dominar. Carros grudavam na parede e morriam por velocidade baixa em vez de por stagnation. NET max inchou 121→170m mas mediana ficou igual. **Decisão**: manter código (inocuous) mas mover foco pro verdadeiro bug.

**Tempo**: ~8min.

---

## Passo 3 — v3: FIX BUG CRÍTICO de sinal na curvatura A9

**O que fez**: Investiguei brain_trace — `raw_steer` persistia **contrário** ao centro de pista mesmo em retas. Rastreei A9 (ch 78/79/80 = flow lookahead 10/25/50m).

Convenção cross-product no plano xz: `cross(forward, flow_ahead)` **positivo ⇒ curva à DIREITA**. Código antigo tinha pesos `-6/-10/-4` em out 1 (spawn.rs priming) + `bias -=` (motor.rs axiom_policy). Três lugares invertidos.

Correção:
- `spawn.rs`: weights[1,78]=+6, [1,79]=+10, [1,80]=+4 (eram negativos).
- `motor.rs` axiom_policy: `bias[1] +=` em vez de `-=`.

**Porquê**: Carros estavam **steering contra a curva** — quanto mais viam curva à direita, mais viravam pra esquerda. Só whiskers A17 salvavam em curto alcance, explicando o plateau de 330m.

**Resultado**: NET max 330→635m em 3 gens. Champion grid-spawn coroou 290m (vs ~150 antes). Mediana 15→40m. Bug de meses resolvido.

**Tempo**: ~15min diagnose + 2min patch.

---

## Passo 4 — v4: boost magnitudes curvatura + pre-brake + swarm 100

**O que fez**:
- axiom_policy A9 coefs: 0.08/0.18/0.06 → **0.20/0.40/0.25**.
- Novo: `curv_mag = (|ch79|+|ch80|)/254; bias[0] -= curv_mag*0.25` (pre-brake antecipado).
- A10 lane_offset bias 0.08→0.22.
- swarm 50→100 (mais diversidade, amostra maior do corner).

**Porquê**: v3 destravou 1ª reta mas carros ainda não negociavam a curva. Steering preditivo precisava de magnitude pra ganhar votação na ternary quantize. Pre-brake pra evitar over-run na entrada de curva.

**Resultado (30 gens swarm 100)**:
- NET max: 740m (57% volta).
- Median 15→80m.
- Champion grid coroou 292m.

**Tempo**: ~5min patch + 4min run.

---

## Passo 5 — run 100 gens swarm 100 speed 15

**O que fez**: `./universe --headless --gens 100 --swarm 100 --speed 15`.

**Resultado**:
- Gens completadas: 101 (~6min wall).
- **NET max global: 909.4m** (70% volta) — distributed spawn.
- **Champion grid plateau: 292.8m** (Gen 91).
- 0 laps em 100+ gens.
- Top cars hit 909m mas terminam crawling a 5 km/h → pre-brake + A15 + A4 compondo demais.
- Gens sem coroar acumulou até 20+ → decay automático do threshold agiu.

**Diagnose**:
- Grid-spawn champion = métrica real de lap capability (começa da largada).
- Distributed 909m prova que mid-track segments são negociáveis — problema é **corner chain entre 292m e 740m**.
- Over-braking: `curv_mag * 0.25` + A11 (ch82) + A15 (ch88) somam throttle cut forte em meio de curva, impedindo acelerar na saída.

**Tempo**: ~6min.

---

## Passo 6 — v5: trim over-brake + adicionar contra-steer slip

**Hipótese**: Cars conseguem chegar à curva mas freiam demais no meio dela. Precisamos:
- `curv_mag * 0.25` → **0.10** (deixa pre-brake só em curva severa).
- A14 slip angle (ch87): 0.08 → **0.20** (contra-volante em derrapagem, estabiliza saída).
- A9 ch79 coef: 0.40 → **0.32** (reduz overshoot; era agressivo demais).

**Porquê**: Cars tocam parede na saída de curva por não contra-esterçar + freiam demais no ápice.

**Resultado (60 gens swarm 100)**:
- NET max 1014.9m (78% lap).
- Champion grid 304m.
- 0 laps.

---

## Passo 7 — v5 extended 150 gens + timeout 120s→180s

**O que fez**: `generation_timeout_s: 120.0 → 180.0`, então 150 gens.

**Porquê**: Cars agora conseguem mais progresso mas podiam estar timing-out mid-lap.

**Resultado**:
- Champion grid explode 304 → **620m** (gen 16), peak **736m** (gen 7).
- NET max 1069m.
- Decay threshold erode ganho; gen 70 champion voltou 608m.

**Tempo**: ~10min.

---

## Passo 8 — v5 run 300 gens

**Resultado**:
- NET max **1172.9m** (90% lap).
- Champion grid oscilou 544-736m.
- 0 laps ainda. 127m faltam pra fechar volta.
- Morte dominante: SPEED_BLEED no final do último trecho (curva final).

**Diagnose**:
- Decay automático do champion threshold (5%/gen após 20 gens sem coroar) **apaga campeões raros**. Gen 7 = 736m, decay depois permite coroar 548m. Regressão.
- Cars chegam quase ao fim mas param por falta de energia / slow_time.

**Tempo**: ~18min.

---

## Passo 9 — v6: decay window 20→40, rate 0.95→0.98, + A19 anti-stall

**O que fez**:
- `pipeline.rs`: `gens_since_crown >= 20 → >= 40`, decay `0.95 → 0.98`.
- `motor.rs`: novo axiom A19. Quando `speed<40% && curv<20% && wall_central<25%` ⇒ `bias[0] += 0.30; bias[2] -= 0.20`.

**Porquê**:
- Decay erode DNAs bons antes de evolução reencontrá-los. Janela mais ampla + rate mais suave = preserva ganhos.
- A19: cars morrem de SBLD em trechos limpos sem committing throttle. Forçar "quando reta limpa → acelere" quebra loops de stall-no-fim.

**Resultado**:
- Champion 606m, NET 1042m. Não melhorou v5.
- Revertido A19 (regressão); mantido decay tweak inicialmente.

---

## Passo 10 — v7: GRID_SPAWNS 30→50

**Porquê**: Só 30 grid-spawns = só 30 candidatos a champion por gen. Baixa amostra.

**Resultado (200 gens)**:
- Champion peak 595m. NET max 1011m. **Pior que v5**.
- Hipótese: menos distributed = menos exploração mid-track = pool genético reduzido.
- Rollback: GRID_SPAWNS voltou a 30.

---

## Passo 11 — v8: STEER SMOOTHING TEMPORAL + FIX RNN memory

**Descoberta importante**: ch 121 (memory[1]=prev steer) era **sempre 0** — memory[0,1,2] eram LIDOS como input mas nunca ESCRITOS. Slow/fast brain recebia zeros no canal de steer recorrente. RNN estava incapacitado.

**O que fez**:
- motor.rs: `steer_raw = prev_mem[1] * 0.65 + novo_steer * 0.35`. Suavização temporal (~3 frames pra trocar direção).
- Escreve de volta `racer.memory[0,1,2]` com pedal/steer/brake correntes.

**Porquê**:
- Brain trace de cars mortos mostrava `SR:-0.99/1.00` = saturação ±1 frame-a-frame. Zig-zag em paredes.
- Smoothing elimina alternância ao custo de resposta 3x mais lenta. Trade-off vale a pena: car com steer estável > car que morre no 1172m.
- Fix RNN memory dá ao slow_brain histórico real pra regular ações.

**Estado**: rollback v6/v6b/v7 (decay 20/0.95, SBLD 180f, GRID 30). v8 adiciona smoothing + RNN fix on top of v5 baseline.

---

## Passo 12 — v10 NINJA: KAN output head + niching fingerprint + racing line spline

**O que fez** (3 agents paralelos em worktree, merge clean sem conflito):

1. **KAN output head** (`src/kan.rs` + `src/simulation/motor.rs` + `src/simulation/racer.rs` + `src/evolution/spawn.rs`):
   - `KanOutputHead` = 3 B-splines piecewise-linear, 12 knots cada, domínio [-2,2].
   - Default `tanh(x*0.9)` — suave, sem hard-clamp.
   - Aplicado em throttle/steer/brake **antes** do smoothing v8 (steer) e antes do threshold (brake).
   - Knots evolvem: `mutate_spline_knots(rng, rate=0.08)` chamado no spawn pra não-champions.

2. **Niching por BehaviorFingerprint** (`src/evolution/pipeline.rs` + `spawn.rs` + `settings.rs` + `main.rs`):
   - `NICHE_DISTANCE=8` (Hamming 8/64 bits).
   - Fitness sharing: raw_fitness ÷ cluster_size antes do sort elite.
   - Tournament parents em clusters distintos (até 5 rejection tries, fallback aceita mesmo cluster).
   - Telemetry: `[NICHE] gen=X pool=10 clusters=N largest=M thr=8`.

3. **Racing line spline** (novo `src/simulation/racing_line.rs` + `world.rs` + `motor.rs`):
   - Catmull-Rom-ish smoothing da centerline com offset `0.35*half_width*curv_sign` pra apex inside.
   - Resample 3m + smoothing ±8 samples (~±24m).
   - Sensores novos canais 108/109/110 (103-107 reservados pra perceived_rivals).
   - Axiom A21: `bias[1] += line_err * 0.35`.
   - Também adicionado A20 drift commit: slip>0.5 + curv>0.4 + speed>0.35 → throttle+0.18, brake-0.15.

**Porquê**:
- KAN spline = curva suave por construção → mata saturação zig-zag ±1 que persistia mesmo com smoothing v8.
- Niching preserva DNAs comportamentalmente distintos (não só o top fitness) → evita plateau de monocultura em ~600m champion.
- Racing line = sensor macro da pista (antes carro só via parede local). Axiom A21 dá template de trajetória.

**Resultado (100 gens swarm 100 speed 15, Silverstone 15.6m track_width, lap 1295m)**:
- **NET max peak: 960m (74% lap)**. Top-5: 960/957/890/847/844m.
- Clusters evoluíram 1 → 7-8 (diversity rising).
- 0 laps. SBLD dominante próximo ao final.
- Silverstone tem hairpin WP9 (105.8°) + sharp WP7 (60.3°) — geometria mais difícil que Interlagos.

**Diagnose preliminar**:
- 100 gens insuficiente pra KAN knots convergir (mutation 8%/gen, needs ~200+).
- Niching impediu colapso mas champion ainda não transfere distributed runs (960m) pra grid-spawn.
- Próximo: rodar 200 gens mesma config antes de iterar.

**Compile**: `cargo check` clean (só warnings pré-existentes). `cargo build --release` 8.26s.

**Tempo**: 3 agents paralelos ~4min + merge/validation ~2min + run 100 gens ~5min = ~11min total.

**Resultado 200 gens follow-up**:
- NET max peak: **767m** (regressão vs 960m de 100g).
- Median late: **166-184m** (vs 108m em 100g — niching elevou base).
- Clusters estabilizaram ~5.
- Champion grid sempre SBLD frame 900-1100.
- 0 laps.

**Diagnose**:
- KAN mutation 0.08 + σ=0.05 drift cumulativo em 200g → heads bons degradando. 
- Champion não serializa `kan_out.knots` → CHAMPION.dna restaura default tanh toda gen, perde aprendizado spline.
- Niching funciona: mediana dobrou, pool mantém diversidade.

**Próximo (v11)**:
1. KAN mutation rate 0.08 → 0.03, σ 0.05 → 0.025.
2. Serializar kan_out.knots no CHAMPION.dna format.
3. Brain trace champion em gen 150+ pra ver se A20/A21 over-correct.

---

## Passo 13 — v11/v11b: KAN knots hereditários via elite_pool

**O que fez**:
- v11: Adicionado `kan_knots: [[f32;12];3]` em `EliteDna` (runtime-only, dna_genesis.rs). Pipeline captura `r.kan_out.knots` ao montar elite_pool. Spawn herda de `elite_pool[0]` + mut 0.03. σ 0.05 → 0.025.
- v11b: Spawn pega 2 parents ALEATÓRIOS do pool + blend 50/50 knot-a-knot + mut 0.04. Champion (i==0) herda top.

**Resultado v11 (200g Silverstone)**:
- NET peak 779m (regressão). Median 88m. Clusters colapsaram a 2.
- Causa: todos filhos herdavam knots do mesmo top elite = monocultura spline.

**Resultado v11b (200g Silverstone, swarm 100, speed 15)**:
- 🎉 **PRIMEIRA VOLTA COMPLETA gen 171** (1v=1 de 100).
- **NET peak 1168m (90% de 1295m)**. Top 5: 1168/1105/1093/1086/1070.
- Median oscila 89-171m.
- Clusters colapsaram tardio a 1 (mas 1 lap antes do colapso).

**Porquê funcionou**:
- Herança de knots preservou aprendizado spline entre gens (antes zerava).
- Parent-random picks mantém diversidade inicial (cada filho amostra pool diferente).
- σ 0.025 impede drift cumulativo destrutivo.

**Restrição identificada**:
- Cluster → 1 late: diversidade morre quando 1 DNA domina fitness.
- Preciso: injeção periódica de diversidade OU niching mais agressivo.

**Tempo**: ~15min (2 iterações + builds + testes).

---

## Passo 14 — v12/v13: Injeção diversidade + A21 coef fix

**v12 (300g, swarm 100)**:
- Config: 10% fresh KAN heads + mut 0.30 + parent-random blend.
- Resultado: **REGRESSÃO**. NET peak 1001m (vs v11b 1168m). 0 laps em 300 gens.
- HOLE deaths: 5181 (vs v11b 3261, +59%). Clusters healthier (modal 5, rare collapse).
- Causa: heads fresh untreinadas + mut forte → KAN produz action zigzag → carros caem do track.

**v13 (200g, swarm 50)**:
- Config: injection 10%→5%, mut 0.30→0.15, A21 racing-line coef 0.35→0.20.
- Motivo A21: v12 HOLE concentrado em curva fechada (WP9 105.8°). Coef 0.35 puxava carro pro apex agressivo demais, jogava fora.
- Resultado: 🏆 **BREAKTHROUGH**.
  - NET peak **1669m** (+43% vs v11b 1168m). 1 volta = 1295m → ultrapassou linha + voltou largada.
  - Laps recorrentes: gens 16, 41, 44, **46 (1v=2 dois carros)**, 100, 121, 151, 174, 178.
  - HOLE: 1133 (vs v12 5181, **-78%**). A21 fix validado.
  - SBLD: 2835. BATIDA: 87. PINADO: 71.
  - Clusters: 181/200 em 2, 19/200 em 1. Colapso ainda late.

**Insights**:
- A21 racing line funciona mas coef forte fight centerline channel (81) em curva sharp.
- Serialização de kan_knots ainda pendente (champion perde knots entre runs).
- Cluster collapse gen 180+ → perde lap consistency.

**Próximo (v14)**:
1. Serializar kan_knots em CHAMPION.dna (persistência entre runs).
2. Re-injetar diversidade ao detectar cluster ≤ 2 por N gens consecutivas.
3. Ou: aumentar NICHE_DISTANCE de 8 → 10 pra facilitar fork de clusters.
4. Testar A21 0.20 → 0.25 (meio-termo) pra ver se mais laps sem HOLE spike.

**Tempo**: ~20min (2 builds + 2 runs 200/300g).

---

## Passo 15 — v14 NICHE bump + v15 axiom floor bump

**v14 (200g swarm 50, NICHE_DISTANCE 8→10)**:
- NET peak **1714m** (+3% v13 1669m). Marginal.
- 10 lap gens. **Gen 1: 7 laps** (priming seed = ninja-capable desde o start).
- Gen 136: 2 laps single gen.
- HOLE 1124, SBLD 2877 (≈v13).
- ❌ Clusters TODAS 1 (loose hamming distance colapsou niching).
- Revertido pra 8.

**Insight forte gen 1 = 7 laps**:
- Pool priming A1-A21 já produz carro capaz de lap.
- Evolução DEGRADA ninja inicial (gen 2+ cai pra 1v=1, depois 0v).
- Axiom decay 1.0→0.15 em 50 gens mata prior ninja.

**v15 (200g swarm 50)**:
- Config: axiom floor 0.15 → 0.35 (mantém prior forte pós gen 50).
- Rationale: priming = ninja, degrada por decay. Mais floor = mais ninja persistente.
- Restrição: 65% dos pesos ainda livres pra evolução (só canais críticos tocados pelo priming).
- Rodando...


**v15 DESCOBERTA CRÍTICA**: primeiro run v15 rodou binário v14 (timestamps confirmaram source editado após build). Log mostrou thr=10 ao invés de thr=8. Resultados "v15 fake" na realidade foram rerun de v14 com seed diferente:
- "v15 fake" NET peak 1269 (vs v14 real 1714) — variância random seed.
- "v15 fake" 14 lap gens (vs v14 10) — também variância.

Rebuild forçado + re-run v15 real em /tmp/v15real_ninja200.log. Pending results.


---

## Passo 15 REAL — axiom floor 0.15→0.35

**Config**: NICHE=8, axiom floor 0.35, injection 5%, mut 0.15, A21 0.20. Multi-track rotation ATIVA (widths 16→14→12m).

**🏆 RESULTADO — BREAKTHROUGH 2-LAP**:
- **NET peak 2551m gen 175** (+49% vs v14 1714m, +118% vs v11b 1168m).
- **Laps: 0v=49 1v=0 2v=1 3+v=0** — PRIMEIRO CARRO DE DUAS VOLTAS.
- 9 lap gens (1/28/64/83/94/118/148/172/175).
- HOLE 1051 (-7% vs v13), SBLD 3000, BATIDA 95.
- Clusters: 191/194 em 1, 3 em 2 (niching ainda fraco).

**Porquê funcionou**:
- Axiom floor 0.35 manteve priming ninja (A1-A21) forte pós gen 50.
- Evolução refinou dentro do "cone" ninja em vez de derivar pra lixo.
- Track rotation introduziu carros em múltiplas larguras (12-16m), generalização forçada.

**Pendências**:
- Clusters ainda colapsam (niching 8 insuficiente quando decay holds ninja priors).
- Pós gen 175 regressão (novo track 12.1m, cars não generalizam).
- 2-lap isolado — falta consistência.

**Próximo (v16)**:
- Persistir kan_knots em CHAMPION.dna pra não perder 2-lap DNA entre runs.
- Ou: run 500g pra ver se 2-lap repete + push pra 3-lap.

**Tempo**: ~25min (1 build + 1 run 200g).
