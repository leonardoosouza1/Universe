# MATH_INVENTORY — Universe (Rust + WGSL)

> Gerado em 2026-08-13. Fonte: inventário bruto de 6 frentes de análise de código.
> Regra: **comentários mentemneste repo** — toda fórmula foi extraída do código, não do
> comentário. Quando não foi possível confirmar diretamente no código, a entrada indica
> "NÃO CONFIRMADO".

---

## Índice

1. [A matemática que o projeto TEM](#1-a-matemática-que-o-projeto-tem)
   - 1.1 [Neurônio — forward pass](#11-neurônio--forward-pass)
   - 1.2 [Evolução e seleção](#12-evolução-e-seleção)
   - 1.3 [Percepção e codificação dos 192 canais](#13-percepção-e-codificação-dos-192-canais)
   - 1.4 [Física do corpo](#14-física-do-corpo)
   - 1.5 [Métricas e medição](#15-métricas-e-medição)
2. [Constantes mágicas](#2-constantes-mágicas)
3. [Duplicações Rust ↔ WGSL](#3-duplicações-rust--wgsl)
4. [O que FALTA — lacunas](#4-o-que-falta--lacunas)
5. [Mapa de reorganização proposto](#5-mapa-de-reorganização-proposto)

---

## 1. A matemática que o projeto TEM

### 1.1 Neurônio — forward pass

O caminho GPU segue 5 etapas sequenciais por ser, por tick:
**KAN sensorial → CfC → L1 BitLinear → L2 FloatLinear + Bypass → fadiga de saída**.

---

#### F01. CfC — fator de decaimento por canal

**Arquivo:** `src/cfc.rs:189` | `src/gpu/brain.wgsl:405`

```
decay = exp(-d_i * dt)
```

onde `d_i = 1/τ_i` é a taxa de decaimento do canal `i`, herdada do genoma.

**Matemática:** decaimento exponencial com constante de tempo `τ_i = 1/d_i`. A 64 Hz
(`dt = 1/64 s`), `τ = 2 s → decay ≈ 0.9845`.

**Propósito:** cada canal tem uma memória diferente. Canal com `τ = 25 ms` esquece em 1,6
ticks; canal com `τ = 4 s` ancora em 256 ticks.

---

#### F02. CfC — ganho DC normalizado

**Arquivo:** `src/cfc.rs:190` | `src/gpu/brain.wgsl:408`

```
gain_dt = DC_GAIN * (1 - decay)   com DC_GAIN = 2.0
```

**Matemática:** garante que o estado estacionário sob entrada constante `x` seja `x · DC_GAIN`
para qualquer τ. Sem isso, canais rápidos saíam 80× mais fracos que os lentos — a evolução
descartaria o rápido por ser mudo, não por ser rápido.

**Propósito:** neutralizar o viés de amplitude que tornaria τ heterogêneo um artefato.

---

#### F03. CfC — fadiga de input (limiar suave / spike-frequency adaptation)

**Arquivo:** `src/cfc.rs:185-186` | `src/gpu/brain.wgsl:418-419`

```
threshold = FATIGUE_BETA * fat_i       (FATIGUE_BETA = 0.4)
x = sign(x_raw) * max(0, |x_raw| - threshold)
```

**Matemática:** encolhimento suave (soft-thresholding) da magnitude do input. A fadiga `fat_i`
acumula o nível médio de `|state_i|`; quando o canal satura, o input efetivo é reduzido sem
trocar seu sinal.

Ponto fixo de entrada máxima sustentada: resolvendo `s = x·DC_GAIN = 2(1 - β·s)` chega-se a
`s* = 2 / (1 + 2β) = 2 / 1.8 ≈ 1.111`, ou seja `tanh(1.111) ≈ 0.804 → ±102/127`.

**Propósito:** quebrar pontos fixos de saturação sem inverter o sinal — modela "adaptação de
frequência de disparo" como ocorre em neurônios reais.

---

#### F04. CfC — atualização do estado (equação principal)

**Arquivo:** `src/cfc.rs:191` | `src/gpu/brain.wgsl:422`

```
s_i = s_i * decay + x * gain_dt
```

**Matemática:** integrador de primeira ordem discreto (Euler). Forma canônica do CfC
("Closed-form Continuous-time"):
```
ds/dt = -d·s + x·DC_GAIN·d
```
com solução exata para passo constante `dt`.

**Propósito:** memória temporal entre ticks — o ser detecta "parede se aproxima" vs "parede
estática".

---

#### F05. CfC — atualização da fadiga de input

**Arquivo:** `src/cfc.rs:204` | `src/gpu/brain.wgsl:442`

```
fat_i += FATIGUE_ALPHA * (|s_i| - fat_i) * dt     (FATIGUE_ALPHA = 1.0)
```

**Matemática:** filtro passa-baixa sobre `|state_i|`. Com `α = 1.0`, `τ_fadiga = 1/α = 1 s`
a 1 Hz de ganho unitário; em 64 ticks (~1 s) a fadiga sobe para 63% do alvo.

**Propósito:** rastrear saturação lenta do canal para alimentar o limiar suave (F03).

---

#### F06. CfC — injeção de ruído determinístico (hash de Knuth)

**Arquivo:** `src/cfc.rs:194-199` | `src/gpu/brain.wgsl:428-433`

```
h = counter * 2654435761 + i      (Knuth multiplicative hash)
if (h % 7 == 0):                   # ~14.3% dos canais por tick
    noise = (h >> 8) / 16777216.0 - 0.5) * 2 * DENOISE_MAG
    s += noise                     (DENOISE_MAG = 0.10)
```

**Matemática:** `2654435761` é a constante de Fibonacci (hash de Knuth), primo adequado para
hashing multiplicativo em 32 bits. `16777216 = 2^24`; os bits `[8..31]` do hash geram ruído
quasi-uniforme em `[-DENOISE_MAG, +DENOISE_MAG]`.

**Propósito:** quebrar viés de exposição simétrica — validado no byte-lab (KL = 0.003).

---

#### F07. CfC — clamp de estado e quantização de saída

**Arquivo:** `src/cfc.rs:201, 206-207` | `src/gpu/brain.wgsl:436, 449`

```
s = clamp(s, -2.0, +2.0)
output_i = clamp(tanh(s) * 127, -127, 127)  →  i8
```

**Matemática:** `tanh` mapeia `[-2, 2]` para `[-0.964, 0.964]`, aproveitando ~96% do range i8.
O clamp em ±2.0 evita blow-up numérico.

**Propósito:** converter estado contínuo para representação i8 usada pelo BitLinear.

---

#### F08. KAN sensorial — mapeamento de índice e interpolação

**Arquivo:** `src/gpu/brain.wgsl:280-281, 286` (WGSL) | `src/kan.rs` (NÃO CONFIRMADO)

```
idx_f = (raw_f32 + 127.0) * (7.5 / 127.0)     # mapeia [-127,127] → [0,15]
idx   = clamp(idx_f, 0, 15)                    # índice inteiro
saída = clamp(raw_f32 * coeff[idx], -127, 127) # coeficiente do knot
```

**Matemática:** lookup em tabela de 16 knots por canal. `7.5 / 127.0 = 15.0 / 254.0` — divide
o range i8 em 16 intervalos uniformes.

**Propósito:** filtro adaptável por canal sensorial (16 coeficientes aprendidos por evolução).
Atualmente os coeficientes partem de `1.0` (identidade) — ver Lacuna 5.

---

#### F09. KAN de saída — interpolação B-spline linear (12 knots)

**Arquivo:** `src/bin/gpu_evolve.rs:1104-1114` | `src/bin/gpu_loop.rs:1168-1181` | `src/kan.rs`

```
xc  = clamp(x, -2.0, 2.0)
u   = (xc + 2.0) / 4.0          # normaliza [−2,2] → [0,1]
fpos = u * 11.0                  # escala para [0,11]
i0  = floor(fpos)
i1  = min(i0+1, 11)
frac = fpos - i0
saída = clamp(a[i0] + (a[i1]-a[i0])*frac, -1, 1)
```

**Matemática:** interpolação linear entre 12 nós distribuídos uniformemente em [-2, 2].
`11 = N_KNOTS - 1 = 12 - 1`.

**Propósito:** cabeça de saída não-linear do cérebro; mapeia o intent para os atuadores
(throttle, steer, brake) em [-1, 1].

---

#### F10. L1 BitLinear — produto interno (192 → 96)

**Arquivo:** `src/layers.rs:45-49` | `src/gpu/brain.wgsl:489-500`

```
acc = Σ_i W[j][i] * input_i        (W em f32, input em i8)
l1_raw[j] = acc / 127.0
l1_act[j] = clamp(tanh(l1_raw[j]) * 127, -127, 127)  →  i8
```

**Matemática:** produto escalar f32-sobre-i8, depois normalizado por 127 e não-linearidade
`tanh`. Pesos L1 em f32 runtime, mas comprimidos para i8 no disco com escala 63.5
(range ±2.0 → veja F15).

**Propósito:** primeira camada densa do cérebro; mistura os 192 canais sensoriais em 96
unidades internas.

---

#### F11. L2 FloatLinear — produto interno (96 → 48)

**Arquivo:** `src/float_layers.rs:29-33` | `src/gpu/brain.wgsl:561-567`

```
acc = Σ_i W[j][i] * l1_act_i       (W em f32, l1_act em i8)

CPU:  l2_raw = acc * (1/127.0)  e  depois  raw = val * 2.0   (engine.rs:498-500)
GPU:  l2_raw = acc * (2.0/127.0)
```

**Matemática:** CPU multiplica por `1/127` e depois por `2.0` em duas etapas separadas; GPU
multiplica por `2.0/127` diretamente. São matematicamente equivalentes, mas a ordem diferente
pode gerar até 1 ULP de diferença em FP32 — **divergência potencial menor, não é bug
funcional**.

**Propósito:** segunda camada densa, reduz 96 → 48 unidades de intenção.

---

#### F12. Fadiga de saída — atualização e encolhimento

**Arquivo:** `src/engine.rs:502-504` | `src/gpu/brain.wgsl:574, 584-585`

```
out_fat += OUT_ALPHA * (|l2_raw| - out_fat) * dt     (OUT_ALPHA = 4.0)
adapted = sign(l2_raw) * max(0, |l2_raw| - OUT_BETA * out_fat)   (OUT_BETA = 0.85)
intent  = tanh(adapted)
```

**Matemática:** mesmo mecanismo da fadiga de input (F03/F05), mas aplicado à saída de L2.
`OUT_ALPHA = 4.0` → `τ_fadiga_out ≈ 0.25 s` a 64 Hz (16 ticks). Ponto fixo: sem adaptação
máxima o intent fica em `tanh(|raw| * 0.15)` se `raw` for sustentado — quebra o volante
cravado.

**Propósito:** prevenir que a saída congele num valor fixo (o problema do "volante cravado"
registrado em vários relatórios de sessão).

---

#### F13. Bypass L1 (canais 48..95 do intent)

**Arquivo:** `src/engine.rs:509-514` | `src/gpu/brain.wgsl:592-599`

```
raw_bp  = l1_raw[tid] * 2.0         (tid em [48, 95])
out_fat += OUT_ALPHA * (|raw_bp| - out_fat) * dt
adapted_bp = sign(raw_bp) * max(0, |raw_bp| - OUT_BETA * out_fat)
intent[tid] = tanh(adapted_bp)
```

**Propósito:** alimenta os slots 48..95 do `final_intent[96]` diretamente a partir de L1 (sem
passar por L2), duplicando o range de ação disponível.

---

#### F14. SlowBrain CfC — diverge do cérebro rápido

**Arquivo:** `src/gpu/slow_brain.wgsl:39-40` | `src/engine.rs:48-82`

```
decay   = exp(-0.5 * dt)       # DECAY_DEFAULT = 0.5 s⁻¹, τ único = 2 s (sem banco de τ)
gain_dt = 1.0 * dt             # dc_gain = 1.0  (≠ 2.0 do cérebro rápido)
```

**DIVERGÊNCIA confirmada:** cerebro rápido usa `dc_gain = 2.0` e banco de τ por canal;
cerebro lento usa `dc_gain = 1.0` e τ fixo único. O lento é **2× mais fraco em amplitude
de sinal** em regime permanente.

---

#### F15. Serialização de pesos para disco (DNA)

**Arquivo:** `src/layers.rs:91, 96-97`

```
save:  i8 = clamp(w * 63.5, -127, 127)    # comprime f32 [-2,2] → i8
load:  w  = i8 / 63.5                      # restaura
```

**Matemática:** `63.5 = 127 / 2.0`. O range de pesos em runtime é ±2.0 (pesos mutados com
clamp ±2.0 na evolução); a escala de disco é diferente da escala de ativação (127.0).

---

#### F16. Inicialização de pesos (Box-Muller)

**Arquivo:** `src/gpu/evolution.rs:87-93`

```
u1, u2 ~ Uniform(0, 1)
g = sqrt(-2 * ln(u1)) * cos(2π * u2)    # Box-Muller
w = clamp(g * 0.30, -0.9, +0.9)         # N(0, 0.30) clamped em 3σ
```

**Propósito:** inicialização de pesos L1/L2 no genoma aleatório. O sigma 0.30 coloca pesos na
escala observada no CHAMPION.dna (~0.1-0.5).

---

#### F17. Inicialização L2 no engine (identidade perturbada)

**Arquivo:** `src/engine.rs:253-255`

```
W_L2 ≈ I + ε     onde  ε ~ N(0, 0.05),  diagonal += 1.0
```

**DIVERGE** da inicialização do genoma evolutivo (que usa `N(0, 0.30)` sem diagonal).
O `engine.rs::new()` inicia L2 como "quase identidade", que é diferente de qualquer filho
gerado pelo evolucionário. **`cortex_l3` é inicializado da mesma forma mas nunca entra no
forward** (`engine.rs:517: let _ = &self.cortex_l3;`).

---

### 1.2 Evolução e seleção

---

#### F18. Fitness — modo padrão (checkpoint)

**Arquivo:** `src/bin/gpu_evolve.rs:2196-2199`

```
fitness = checkpoint * lap_dist + max(0, prog_acum)
```

`checkpoint` = número de setores cruzados EM ORDEM (inteiro dominante);
`prog_acum` = avanço contínuo monotônico dentro do setor (desempate fino).

---

#### F19. Progresso monotônico com wrap-around

**Arquivo:** `src/bin/gpu_evolve.rs:2014-2017`

```
d = cur - prog_prev
if d < -lap/2: d += lap
elif d > lap/2: d -= lap
avanco = clamp(d, -0.1*lap, +0.1*lap)
prog_acum += avanco
```

**Matemática:** trata o cruzamento da linha de chegada nos dois sentidos; o clamp de 10% da
volta descarta teleporte numérico.

---

#### F20. Proximity ao eixo da pista

**Arquivo:** `src/bin/gpu_evolve.rs:2078`

```
prox = clamp(1.0 - dist_eixo / half_width, 0, 1)
```

`1.0` no eixo, `0.0` na borda ou além.

---

#### F21. Elitismo — seleção dos melhores

**Arquivo:** `src/gpu/evolution.rs:203-211`

Os `n_elite` melhores genomas (padrão 3, 5% de 60) passam intactos para a próxima geração.

---

#### F22. Seleção por torneio (k = 3)

**Arquivo:** `src/gpu/evolution.rs:245-251`

```
candidatos = sample_sem_reposição(k=3)
pai = argmax(fitness, candidatos)
```

**Propósito:** menos suscetível a deriva genética que seleção proporcional.

---

#### F23. Mutação gaussiana dos pesos (L1 e L2)

**Arquivo:** `src/gpu/evolution.rs:258-270`

```
com prob p_mutate = 0.02:
    w += N(0, sigma)       sigma padrão = 0.05
    w = clamp(w, -2.0, +2.0)
```

---

#### F24. Mutação do cérebro lento (3× taxa, 1.5× sigma)

**Arquivo:** `src/gpu/evolution.rs:273-277`

```
com prob p_mutate * 3.0:
    w += N(0, sigma * 1.5)
    w = clamp(w, -2.0, +2.0)
```

**Justificativa:** o lento tem ~1091 pesos vs ~23040 do L1+L2; taxa maior compensa o menor
número de parâmetros.

---

#### F25. Mutação do banco de τ (multiplicativa em log)

**Arquivo:** `src/gpu/evolution.rs:291`

```
com prob p_mutate * 3.0:
    d *= exp(N(0, 0.20))
    d = clamp(d, 1/TAU_MAX_S, 1/TAU_MIN_S)    # [0.25, 40.0] s⁻¹
```

**Matemática:** mutação multiplicativa preserva a escala logarítmica. `exp(N(0, 0.20))` tem
mediana 1.0 e p95 ≈ 1.49.

---

#### F26. Mutação dos knots KAN (muito esparsa)

**Arquivo:** `src/gpu/evolution.rs:297-299`

```
com prob 0.005:    # hardcoded, não configurável via CLI
    k = clamp(k + N(0, 0.05), -1, 1)
```

---

#### F27. Participation Ratio (PR) das ativações

**Arquivo:** `src/brain_capacity.rs:159-167`

```
PR = (Σ λ_i)² / Σ λ_i²
```

onde `λ_i` são autovalores da covariância empírica. `PR = 1` → uma dimensão efetiva;
`PR = N` → todas as N dimensões igualmente usadas.

**Chamado em:** `gpu_evolve.rs:2340` — uma vez por geração sobre ≤20.000 amostras das
**3 saídas** do SlowBrain. **Não é chamado para L1 ou L2** (ver Lacuna 1).

---

#### F28. Lei do Garimpo — limiar de ruído de busca

**Arquivo:** `src/brain_capacity.rs:187`

```
limiar = sigma * sqrt(2 * ln(M_alternativas))
```

O melhor de `M` amostras de ruído `N(0, σ²)` tem valor esperado igual ao limiar. Um ganho
de fitness só é "real" se exceder isso.

**STATUS: FUNÇÃO ÓRFÃ** — `growth_verdict()` que a chama não é invocada em nenhum ponto do
fluxo de produção (confirmado por grep). A fórmula existe, está testada, mas não guia nenhuma
decisão automaticamente.

---

#### F29. OLS inclinação da mediana (progresso por geração)

**Arquivo:** `src/gpu/evolution.rs:329-345`

```
slope = Σ(x_i - x̄)(y_i - ȳ) / Σ(x_i - x̄)²
slope_stderr = sqrt(SSR / ((n-2) * Σ(x_i - x̄)²))
```

`x_i` = índice de geração, `y_i` = mediana de `prog_acum`. Reportado em m/geração.

---

#### F30. Ganho de vida por avanço

**Arquivo:** `src/bin/gpu_evolve.rs:2143-2144`

```
life_left = min(life_left + passo * LIFE_EARN, LIFE_CAP_TICKS)
```

Cada metro avançado ganha `LIFE_EARN` ticks de vida, com teto `LIFE_CAP_TICKS` (padrão 1920 = 30 s).

---

### 1.3 Percepção e codificação dos 192 canais

O vetor sensorial tem 192 canais i8 por ser, produzidos por 4 shaders WGSL + código CPU em
motor.rs. **Regra de prioridade:** quando GPU e CPU escrevem no mesmo canal, o que vem DEPOIS
prevalece — veja o mapa abaixo.

---

#### F31. Codificação legada (legacy_encoding) — base de todos os pools

**Arquivo:** `src/gpu/mod.rs:123-130` | `src/gpu/perception.wgsl:255-268`

```
Wall próxima  (d < 30m):    v = 127.0 - sqrt(clamp(d/30, 0,1)) * 72.0
Wall média    (30≤d<120m):  v = 55.0  - clamp((d-30)/90, 0,1) * 55.0
Wall longe    (120≤d<1200m):v = 0.0   - clamp((d-120)/480, 0,1) * 25.0
Wall longe+   (d≥1200m):    v = -25.0 - clamp((d-1200)/1200, 0,1) * 25.0
Grass:                       v = 40.0  - clamp(d/1200, 0,1) * 40.0
Road:                        v = -128.0 + clamp(d/1200, 0,1) * 96.0
Hole:                        v = 90.0  - clamp(d/50, 0,1) * 30.0
```

**Idêntica em Rust e WGSL** (confirmado).

---

#### F32. Sensores angulares de parede (ch 16-23) — distância mínima em 8 ângulos

**Arquivo:** `src/gpu/perception.wgsl:385-392` | `src/simulation/motor.rs:768`

```
encoded = clamp(127 * (60 - wall_dist_min) / 60, 1, 127)  se dist < 60m
          0                                                  caso contrário
```

WGSL e CPU **idênticos**.

---

#### F33. Distância lateral esq/dir (ch 76-77) — DIVERGÊNCIA CONFIRMADA

**Arquivo GPU:** `src/gpu/perception.wgsl:631`
**Arquivo CPU:** `src/simulation/motor.rs:1492-1493`

```
GPU:  encoded = clamp(dist / 40.0 * 127, 0, 127)    # normaliza por 40m
CPU:  encoded = clamp(dist / 15.0 * 127, 0, 127)    # normaliza por 15m
```

**BUG LATENTE:** fator 2.67× de diferença. DNA treinado com CPU (motor.rs) verá escala
diferente ao rodar na GPU. Para a mesma distância real de 10m, GPU codifica `≈32`, CPU
codifica `≈85`.

---

#### F34. EMA de distância lateral (ch 85-86) — DIVERGÊNCIA CONFIRMADA

**Arquivo GPU:** `src/gpu/perception_temporal.wgsl:221-227`
**Arquivo CPU:** `src/simulation/motor.rs:1597-1602`

```
alpha = clamp(dt * 3.0, 0, 1)    # meia-vida ≈ 21 ticks ≈ 0.33 s
GPU:  ema = ema*(1-alpha) + dist*alpha;  encoded = clamp((1-ema/40)*127, 0, 127)
CPU:  ema = ema*(1-alpha) + dist*alpha;  encoded = clamp((1-ema/15)*127, 0, 127)
```

**BUG LATENTE:** mesmo normalizador divergente (40m GPU vs 15m CPU) dos ch 76-77.

---

#### F35. Dark-center tracker (ch 90-93) — DIVERGÊNCIA TOTAL

**GPU:** escreve 4× o mesmo valor (centroide de sliding-window W=8 sobre pixels escuros).
**CPU (motor.rs):** escreve 4 métricas diferentes: `dc_40m`, `dc_25m`, `lr_diff_norm`, `dc_15m`.

Os 4 canais têm semântica completamente diferente entre GPU e CPU. **BUG REAL.**

---

#### F36. Urgência frontal (ch 82) — semântica diferente

**GPU:** `encoded = clamp((1 - fwd_probe/30) * 127, 0, 127)` — função de distância.
**CPU:** `encoded = clamp(speed/fwd_probe / 10 * 127, 0, 127)` — função de TTC.

Mesma posição no vetor, grandeza física diferente.

---

#### F37. Curvatura por atan2 (ch 78-80) — DIVERGÊNCIA POR SOBRESCRITA

**Arquivo:** `src/simulation/motor.rs:1524-1530` e `1769`

Motor.rs escreve ch78 **duas vezes**: primeiro com `cross * 127` sem atan2 (bloco "CURVATURA
ANTECIPADA", a 10m), depois com `atan2/(π/2) * 127` (bloco "CURVATURA LOOKAHEAD", a 40m).
**A segunda sobrescreve a primeira.** GPU usa atan2. O canal final é atan2 em ambos os casos,
mas a sequência de escritas em motor.rs pode causar confusão na leitura do código.

---

#### F38. Curvatura por atan2 — fórmula

**Arquivo:** `src/gpu/perception.wgsl:660-665` | `src/gpu/perception_temporal.wgsl:143-156`

```
cross_y = fwd_x * flow_y - fwd_z * flow_x
dot     = clamp(fwd_x*flow_x + fwd_z*flow_y, -1, 1)
angle   = atan2(cross_y, dot)
encoded = clamp((angle / (π/2)) * 127, -127, 127)
```

**Propósito:** curvatura com sinal do campo de fluxo em 3 horizontes (40m, 70m, 120m) e mais
3 ultra-longos (200m, 275m, 350m).

---

#### F39. Motion per-row (ch 21-27) — DIVERGÊNCIA de normalizador

**GPU:** `encoded = clamp((delta_dist_m / 2000.0) * 127, 0, 127)`
**CPU:** `encoded = clamp((delta_bytes / 8000.0) * 127, 0, 127)`

As unidades são diferentes (metros vs bytes de diferença i8). O comentário do kernel GPU
menciona que 2000m corresponde a ~1882m para equivalência com 8000 bytes (4.25 bytes/m),
mas usa 2000 por margem para curvas bruscas. **Divergência intencional, documentada.**

---

#### F40. Raycasting da retina — DDA com packed output

**Arquivo:** `src/gpu/retina.wgsl:141-160`

```
# DDA (Digital Differential Analyzer)
d_side0 = ((map_x - ray_start.x + (1-step_x)/2) / ray_dir.x) * cell_size
# ... até 400 passos
packed = (mm << 8) | (material & 0xFF)    # mm = dist em milímetros
```

**Desempacotamento:** `src/gpu/mod.rs:350-354`
```
dist_m   = (packed >> 8) as f32 / 1000.0
material = (packed & 0xFF) as u8
```

**Propósito:** raycasting GPU por pixel da retina (14 linhas × 126 colunas = 1764 pixels/ser),
retornando distância e material.

---

#### F41. Ângulo de yaw da retina (linha 0 foveal vs demais)

**Arquivo:** `src/gpu/retina.wgsl:90-95`

```
linha 0 (fovea):   yaw = (tan(norm_i * TAN_K) / tan(TAN_K)) * fov_half + gaze
demais linhas:      yaw = norm_i * fov_half
```

`TAN_K = 1.25` — maior densidade angular no centro da linha 0 (fovea com distorção tangente).

---

#### F42. CfC de percepção — aproximação de tanh

**Arquivo:** `src/gpu/perception_temporal.wgsl:373-379`

```
tanh_h ≈ h / (1 + |h|)     # aproximação, erro < 2% para |h| < 3
val = clamp(tanh_h * 127 * 0.3, -127, 127)
```

**DIVERGÊNCIA:** motor.rs usa `tanh` exato; GPU usa aproximação `x/(1+|x|)`. Para `|h| > 3`
o erro pode exceder 2%.

---

### 1.4 Física do corpo

O pipeline GPU usa **3 kernels** em sequência: Kernel A (forças) → Kernel B (pneus + LSD) →
Kernel C (colisão + integração).

---

#### F43. Steer → ângulo da roda (speed-sensitive)

**Arquivo:** `src/physics.rs:519-524` | `src/gpu/physics_b.wgsl:158-161`

```
max_steer_deg = 35.0                        se speed < 15 m/s
              = max(35 - (speed-15)*0.4, 12) se speed ≥ 15 m/s
wheel_steer = steer * max_steer_deg * (π/180)
```

**Matemática:** redução linear de 35° (parado) a 12° (acima de ~72.5 m/s). Quatro cópias
idênticas (physics.rs, physics_b.wgsl, physics.wgsl, physics_phase_integ.wgsl).

---

#### F44. Pacejka longitudinal (tração/frenagem)

**Arquivo:** `src/physics.rs:539-541` | `src/gpu/physics_b.wgsl:188, 227, 265, 302`

```
F_long = load * mu * sin(c_long * atan(b_long * slip_ratio))
         com b_long=10.0, c_long=1.5
```

**Matemática:** fórmula de Pacejka simplificada (Magic Formula). Pico de força em
`slip_ratio ≈ tan(π/(2·c_long)) / b_long ≈ 0.16` (16% de escorregamento).

---

#### F45. Pacejka lateral (cornering)

**Arquivo:** `src/physics.rs:547-550` | `src/gpu/physics_b.wgsl:190, 229, 267, 304`

```
slip_angle = atan2(v_lateral_contato, |v_frontal_contato|)    se |v_frontal| > 0.5
F_lat = load * mu * sin(c_lat * atan(b_lat * slip_angle))
        com b_lat=12.0, c_lat=1.35
```

---

#### F46. Círculo de Kamm — limitador combinado

**Arquivo:** `src/physics.rs:558-564` | `src/gpu/physics_b.wgsl:192-194`

```
F_total = sqrt(F_long² + F_lat²)
se F_total > mu*load:
    scale = mu*load / F_total
    F_long *= scale;  F_lat *= scale
```

**Nota:** CPU usa `sqrt()`, WGSL usa `length(vec2)` — numericamente equivalentes.

---

#### F47. Yaw — integração Euler + amortecimento speed-sensitive

**Arquivo:** `src/gpu/physics_c.wgsl:260-262` | `src/physics.rs:743-751`

```
drag_y = -av_y * (400.0 + speed * 60.0)    # N·m·s/rad (speed-dependent)
ang_acc = (torques_y + drag_y) / yaw_inertia
av_y_new = av_y + ang_acc * dt
```

**BUG CONFIRMADO:** `yaw_inertia = mass * (wheel_base/2)²` = 1964 kg·m² (ponto de massa nos
eixos). O valor correto para corpo retangular seria `m*(L²+W²)/12 ≈ 905 kg·m²` — 2.17× maior,
suprimindo a resposta de yaw. Combinado com o amortecimento 400+60v, o déficit de torque de
yaw para completar a curva de 51m da pista 9001 é ~30× (ver observações da frente de física).

---

#### F48. Yaw inertia — fórmula incorreta (bug estrutural)

**Arquivo:** `src/physics.rs:631` | `src/gpu/physics_a.wgsl:282`

```
yaw_inertia = mass * (wheel_base * 0.5)²    # INCORRETO para corpo rígido
```

**Correto para paralelepípedo:** `I = m*(L²+W²)/12`. Todas as cópias usam a mesma fórmula
incorreta.

---

#### F49. Curva de torque do motor (parábola deslocada)

**Arquivo:** `src/physics.rs:382-388` | `src/gpu/physics_a.wgsl:249-251`

```
rpm_ratio = rpm / max_rpm
torque_mult = 1 - (rpm_ratio - 0.6)²  * 2    # pico em 60% do max_rpm
se rpm_ratio > 1.0: torque_mult = 0           # limitador
se rpm_ratio < 0.2: torque_mult *= rpm_ratio/0.2   # rampa de baixa
```

---

#### F50. Suspensão mola-amortecedor por roda

**Arquivo:** `src/physics.rs:207-217` | `src/gpu/physics_a.wgsl:170-172`

```
spring_force = k * compression
damper_force = c * (compression - prev_compression) / dt
total = clamp(spring_force + damper_force, 0, ∞) + downforce_per_wheel
carga = clamp(total, 0.2*carga_estatica, 6.0*carga_estatica)
```

---

#### F51. Colisão com parede (impulso inelástico)

**Arquivo:** `src/physics.rs:677-690` | `src/gpu/physics_c.wgsl:186-191`

```
impulso = -(1 + 0.05) * vel_penetração * massa    # restitution = 0.05
vel += (impulso/massa) * dir_penetração
vel -= vel_tangencial * 0.5                        # atrito tangencial 50%
av_y += cross(fwd, pen_dir).y * vel_penetração * 0.05   # kick de yaw (sem dividir por I)
pos -= pen_dir * 0.5                               # push-back 0.5m
```

**Nota:** o kick de yaw não divide por `yaw_inertia` — não é fisicamente correto, mas é
reproduzido identicamente em CPU e GPU.

---

#### F52. Drag aerodinâmico — BUG histórico na CPU

**Arquivo:** `src/physics.rs:416` (Kernel A) e `src/physics.rs:717` (integração)

```
Kernel A:       F_drag = 0.5 * 1.225 * v² * Cd * 2.2    # correto
Integração CPU: F_drag = 0.5 * 1.225 * v² * Cd * 2.5    # aplica drag 2× com Cd diferente!
```

**GPU (physics_c.wgsl:243):** reaplicamento **removido** em 2026-08-03. **CPU ainda tem o
bug.** O drag duplo na CPU faz o carro desacelerar mais rápido que na GPU.

---

### 1.5 Métricas e medição

---

#### F53. Odômetro (distância euclidiana acumulada)

**Arquivo:** `src/bin/gpu_evolve.rs:2006-2007`

```
passo = sqrt(dx² + dz²)
dist_acum += passo
```

---

#### F54. Velocidade média por ser vivo

**Arquivo:** `src/bin/gpu_evolve.rs:2310-2312`

```
seg = max(ticks_alive, 1) / 64.0
vel_media = dist_acum / seg
```

---

#### F55. Aproveitamento de pista

**Arquivo:** `src/bin/gpu_evolve.rs:2325-2326`

```
aproveitamento = 100 * prog_acum / dist_acum    se dist_acum > 1.0
```

---

#### F56. Auditoria 3 eixos — σ temporal e σ entre seres

**Arquivo:** `src/bin/gpu_evolve.rs:2615-2634`

```
σ_temporal = std(canal[ch] ao longo de todos os ticks do 1º ser VIVO rastreado)
σ_entre    = std(canal[ch] entre seres no último tick com ≥4 seres vivos)

Canal APROVADO ↔ σ_temporal ≥ 2.0  E  σ_entre ≥ 2.0  E  canal tem fonte definida
```

---

#### F57. Curvatura discreta da pista (kappa)

**Arquivo:** `src/bin/replay_mapa.rs:155-158`

```
cross = v1.x*v2.z - v1.z*v2.x
dot   = v1.x*v2.x + v1.z*v2.z
ang   = atan2(cross, dot)
κ     = ang / (0.5 * (l1 + l2))     # rad/m
```

---

#### F58. Obediência (quociente de somas com sinal)

**Arquivo:** `src/bin/replay_mapa.rs:270-271, 291`

```
soma_proj += (w_real * sign(w_exig))    # projeção no sentido da curva
soma_exig += |w_exig|
obediencia = soma_proj / soma_exig      # ∈ [-1, 1]
```

Filtro: apenas quadros com `|κ*v| > 0.03 rad/s` e `v > 1.0 m/s`.

**Propósito:** detectar se o cérebro vira no sentido certo. `0.5 = acaso; 1.0 = perfeito`.

---

#### F59. Correlação de Pearson volante × curvatura

**Arquivo:** `src/bin/replay_diag.rs:325-332`

```
r = Σ(x-x̄)(y-ȳ) / (std(x) * std(y))
```

---

#### F60. Joy (dopamina) — sinal de recompensa

**Arquivo:** `src/bin/gpu_evolve.rs:2032, 2049`

```
progress_speed = clamp(v * max(align, 0) / 30, 0, 1)
joy_tick = progress_speed * surface_factor
    # surface_factor: Road=1.0, Grass=0.2, Wall=-0.5, Hole=-1.0
```

---

#### F61. Joy streak (acúmulo de "flow")

**Arquivo:** `src/bin/gpu_evolve.rs:2052-2055`

```
se joy_tick > 0.5:  streak = min(streak + dt, 5.0)    # acumula a 1 s/s, teto 5 s
senão:              streak = max(streak - dt*2.0, 0)   # decai 2× mais rápido
```

---

## 2. Constantes mágicas

Tabela das constantes sem justificativa formal. "Vira param?" = se deveria ser um parâmetro
nomeado em `config.rs`.

| Valor | Onde (arquivo:linha) | O que provavelmente é | Vira param? |
|---|---|---|---|
| `2.0` (DC_GAIN) | `src/cfc.rs:30`, `brain.wgsl:115` | Ganho DC do CfC — reproduz par antigo (input_gain 1.0 / decay 0.5). Justificado. | Não — faz parte da especificação |
| `2.0` (CFC_CLAMP) | `src/cfc.rs:104`, `brain.wgsl:116` | Limite de saturação do estado CfC em ±2.0 | Sim — `CFC_CLAMP_RANGE` |
| `0.10` (DENOISE_MAG) | `src/cfc.rs:105`, `brain.wgsl:117` | Magnitude do ruído determinístico (~14% canais/tick) | Sim — validado em byte-lab; deveria ser `CFC_NOISE_MAG` |
| `1.0` (FATIGUE_ALPHA) | `src/cfc.rs:108`, `brain.wgsl:143` | τ_fadiga ≈ 1 s a 64 Hz | Sim — `FATIGUE_ALPHA` |
| `0.4` (FATIGUE_BETA) | `src/cfc.rs:116`, `brain.wgsl:144` | Força do encolhimento suave; ponto fixo s*=1.111 | Sim — `FATIGUE_BETA` |
| `4.0` (OUT_ALPHA) | `src/engine.rs:494`, `brain.wgsl:174` | τ_fadiga_saída ≈ 0.25 s (16 ticks) | Sim — duplicado como const local em engine.rs |
| `0.85` (OUT_BETA) | `src/gpu/brain.rs:47`, `engine.rs:495` | Força do encolhimento de saída; duplicado como const local | Sim — `OUT_BETA_PADRAO` já existe mas não é usada por `engine.rs` |
| `127.0` | `src/layers.rs:42`, `float_layers.rs:25` e ~15 locais | Fator de normalização i8 (range ±127 → ±1.0) | Sim — definir `I8_SCALE: f32 = 127.0` |
| `63.5` | `src/layers.rs:91,97` | Escala de disco L1: 127/2.0 (range ±2.0 → i8) | Sim — `I8_DISK_SCALE: f32 = 63.5` |
| `0.30` (sigma_init) | `src/gpu/evolution.rs:87` | Desvio padrão da inicialização N(0,0.30). Justificado por champion.dna | Sim — `INIT_SIGMA` |
| `0.05` (sigma_mutação) | `src/gpu/evolution.rs:173` | Sigma da mutação gaussiana padrão | Sim — `MUT_SIGMA` |
| `0.02` (p_mutate) | `src/gpu/evolution.rs:171` | Probabilidade de mutação por peso por geração | Sim — `MUT_RATE` |
| `0.20` (sigma_tau_log) | `src/gpu/evolution.rs:291` | Largura da mutação multiplicativa de τ | Sim — `TAU_MUT_LOG_SIGMA` |
| `0.005` (p_kan_knots) | `src/gpu/evolution.rs:297` | Taxa de mutação dos knots KAN de saída. **Hardcoded, não configurável via CLI** | Sim — urgente |
| `2.0` (clamp pesos) | `src/gpu/evolution.rs:261,268,276` | Limite de amplitude dos pesos mutados | Sim — `WEIGHT_CLAMP` |
| `640` (LIFE_START) | `src/bin/gpu_evolve.rs:1441` | Fôlego inicial = 10 s a 64 Hz | Sim — `LIFE_START_TICKS` |
| `1920` (LIFE_CAP) | `src/bin/gpu_evolve.rs:1451` | Teto de vida = 30 s a 64 Hz; foi escolha histórica que virou parâmetro real | Sim — `LIFE_CAP_TICKS` |
| `24` (N_CHECKPOINTS) | `src/bin/gpu_evolve.rs:1425` | Setores da pista (~122m cada em 2928m) | Sim — `N_TRACK_SECTORS` |
| `128` (stagnation ticks) | `src/bin/gpu_evolve.rs:1065` | 2 s de parado antes de cobrar STAG_LIFE_DRAIN | Sim |
| `3.0` (WALL_NEAR_M) | `src/bin/gpu_evolve.rs:1402` | Distância frontal para "colado na parede" | Sim |
| `6000.0` (edge norm) | `src/bin/gpu_evolve.rs:1091`, `motor.rs:919` | Normalizador de edge density; max teórico é 31875 — 6000 = ~19% do max | Sim — calibrar |
| `30.0` (v_norm) | `src/bin/gpu_evolve.rs:630,2032` | Velocidade de normalização (108 km/h) | Sim — `V_NORM_MS` |
| `400.0` (yaw_damp_base) | `src/physics.rs:743`, `physics_c.wgsl:260` | Amortecimento de yaw base (N·m·s/rad). Causa o déficit de curva ~30×. Historicamente ajustado | **Sim — crítico para física** |
| `60.0` (yaw_damp_speed) | `src/physics.rs:744`, `physics_c.wgsl:260` | Coeficiente speed-dependent do amortecimento de yaw | Sim |
| `10.0` (Pacejka b_long) | `src/physics.rs:539` | Rigidez longitudinal do pneu | Sim — `PACEJKA_B_LONG` |
| `1.5` (Pacejka c_long) | `src/physics.rs:540` | Fator de forma longitudinal | Sim — `PACEJKA_C_LONG` |
| `12.0` (Pacejka b_lat) | `src/physics.rs:548` | Rigidez lateral do pneu | Sim — `PACEJKA_B_LAT` |
| `1.35` (Pacejka c_lat) | `src/physics.rs:549` | Fator de forma lateral | Sim — `PACEJKA_C_LAT` |
| `2.5` (drag/downforce ratio) | `src/physics.rs:421` | Razão Cl/Cd do carro GT3; causa drag duplo na CPU | Sim — `AERO_LIFT_DRAG_RATIO` |
| `1.25` (TAN_K retina) | `src/gpu/retina.wgsl:54` | Distorção tangente da linha 0 (fovea) | Sim — `RETINA_FOVEA_TAN_K` |
| `0.4` (EYE_HEIGHT) | `src/gpu/retina.wgsl:55` | Altura do olho em metros | Sim — `RETINA_EYE_HEIGHT_M` |
| `40.0` (LAT_MAX) | `src/gpu/perception.wgsl:617,631` | Alcance máximo dos sensores laterais. Expandido de 15→27→40m | Sim — `LATERAL_SENSOR_RANGE_M` |
| `0.85` (TTC threshold) | `src/simulation/motor.rs:194` | TTC_norm > 0.85 = perigo crítico nos axiomas | Sim |
| `2654435761` | `src/cfc.rs:194`, `brain.wgsl:428` | Constante de Knuth (hash multiplicativo de Fibonacci). Não é mágica | Não — define como `KNUTH_HASH_MULT` |
| `16777216.0` | `src/cfc.rs:197`, `brain.wgsl:431` | `2^24` — divisor para mapear bits do hash para [0,1) | Não — define como `HASH_FLOAT_DIV` |
| `0.5` (DECAY_DEFAULT) | `src/cfc.rs:28` | τ = 2 s herdado; ainda usado pelo SlowBrain | Sim — revisar ou remover |
| `64.0` (tick rate) | múltiplos arquivos (`gpu_evolve.rs:255`, `gpu_loop.rs:204`, etc.) | Taxa da simulação em Hz. Aparece como literal em 8+ arquivos | **Sim — urgente: `SIM_HZ: f32 = 64.0`** |
| `1350.0` (massa) | `gpu_evolve.rs:198`, `gpu_loop.rs:149` | Massa do carro em kg; repetida em 4 cópias de `make_physics_state` | Sim — `VEHICLE_MASS_KG` |
| `620.0` (max_torque) | `gpu_evolve.rs:236`, `gpu_loop.rs:186` | Torque máximo do motor em Nm | Sim — `ENGINE_MAX_TORQUE_NM` |
| `8500.0` (max_rpm) | `gpu_evolve.rs:238`, `gpu_loop.rs:188` | RPM máximo. Comentário registra que 7000 causava bug de relação de marcha | Sim — `ENGINE_MAX_RPM` |
| `0.85` (OUT_BETA_PADRAO) | `src/gpu/brain.rs:47` | Já existe como constante nomeada mas engine.rs usa `0.85` como literal local | Usar a constante já definida |

---

## 3. Duplicações Rust ↔ WGSL

### 3.1 Pares idênticos (sem divergência)

| Fórmula | Rust | WGSL | Status |
|---|---|---|---|
| CfC decay | `cfc.rs:189` | `brain.wgsl:405` | IDÊNTICA |
| CfC gain_dt | `cfc.rs:190` | `brain.wgsl:408` | IDÊNTICA |
| CfC limiar suave (fadiga) | `cfc.rs:185-186` | `brain.wgsl:418-419` | IDÊNTICA |
| CfC state update | `cfc.rs:191` | `brain.wgsl:422` | IDÊNTICA |
| CfC atualiz. fadiga | `cfc.rs:204` | `brain.wgsl:442` | IDÊNTICA |
| CfC ruído Knuth | `cfc.rs:194-199` | `brain.wgsl:428-433` | IDÊNTICA |
| CfC clamp + tanh→i8 | `cfc.rs:201,206-207` | `brain.wgsl:436,449` | IDÊNTICA |
| L1 BitLinear forward | `layers.rs:45-49` | `brain.wgsl:489-500` | IDÊNTICA |
| L1 tanh→i8 | `engine.rs:469-470` | `brain.wgsl:527` | IDÊNTICA |
| Fadiga saída (update) | `engine.rs:502` | `brain.wgsl:574` | IDÊNTICA |
| Fadiga saída (encolhimento) | `engine.rs:503-504` | `brain.wgsl:584-585` | IDÊNTICA |
| Bypass L1→intent | `engine.rs:509-514` | `brain.wgsl:592-599` | IDÊNTICA |
| legacy_encoding (todos os materiais) | `gpu/mod.rs:123-130` | `perception.wgsl:255-268` | IDÊNTICA |
| ch 16-23 wall_distance_fov | `motor.rs:768` | `perception.wgsl:385-392` | IDÊNTICA |
| ch 24-31 wall_width_fov | `motor.rs:786` | `perception.wgsl:422-423` | IDÊNTICA |
| ch 40-47 road_ahead NEAR | `motor.rs:822` | `perception.wgsl:469-478` | IDÊNTICA |
| ch 94-97 whiskers ±30/±60° | `motor.rs:1515-1520` | `perception.wgsl:774-827` | IDÊNTICA |
| ch 102-107 approach rate + TTC | `motor.rs:1270-1279` | `perception_temporal.wgsl:199-213` | IDÊNTICA |
| ch 160-191 horizon scan | `motor.rs:718-721` | `perception.wgsl:886-893` | IDÊNTICA |
| ch 88 proximity_sustain | `motor.rs:1618-1624` | `perception_temporal.wgsl:231-237` | IDÊNTICA |
| ch 81 lane_offset | `motor.rs:1534-1536` | `perception_temporal.wgsl:246-248` | IDÊNTICA (max(0.1) vs condicional >0.1 — equiv.) |
| Steer speed-sensitive | `physics.rs:519-524` | `physics_b.wgsl:158-161` | IDÊNTICA (4 cópias) |
| Pacejka longitudinal | `physics.rs:539-541` | `physics_b.wgsl:188 e 3 mais` | IDÊNTICA |
| Pacejka lateral | `physics.rs:547-550` | `physics_b.wgsl:190 e 3 mais` | IDÊNTICA |
| Círculo de Kamm | `physics.rs:558-564` | `physics_b.wgsl:192-194 e 3 mais` | IDÊNTICA (sqrt vs length — equiv.) |
| Colisão com parede | `physics.rs:677-690` | `physics_c.wgsl:186-191` | IDÊNTICA |
| Scrub lateral | `physics.rs:730-735` | `physics_c.wgsl:249-253` | IDÊNTICA |
| Curva torque motor | `physics.rs:382-388` | `physics_a.wgsl:249-251` | IDÊNTICA |
| Termodinâmica motor | `physics.rs:357-360` | `physics_a.wgsl:233-235` | IDÊNTICA |
| Suspensão spring-damper | `physics.rs:207-217` | `physics_a.wgsl:170-172` | IDÊNTICA (parênteses diff, equiv.) |
| Yaw + amortecimento | `physics.rs:743-751` | `physics_c.wgsl:260-262` | IDÊNTICA |
| RPM / clutch blend | `physics.rs:325-337` | `physics_a.wgsl:217-223` | IDÊNTICA |
| Box-Muller (Genome::random) | `evolution.rs:91` (só cos) | — | Sem par WGSL |
| Box-Muller (Rng::gauss) | `evolution.rs:230-234` (cos+sin, spare) | — | Sem par WGSL |
| Retina packed output | `retina.wgsl:159-160` | `gpu/mod.rs:350-354` (desempacota) | IDÊNTICA (protocolo) |

### 3.2 Pares que DIVERGEM (bug latente ou diferença intencional)

| Canal/Fórmula | Rust (CPU/motor.rs) | WGSL (GPU) | Impacto |
|---|---|---|---|
| **ch 76-77** distância lateral | norm por **15m** | norm por **40m** | **CRÍTICO**: fator 2.67× — DNA treinado numa versão vê escala errada na outra |
| **ch 85-86** wall EMA | norm por **15m** | norm por **40m** | **CRÍTICO**: mesmo fator — os canais de alerta de parede têm amplitude radicalmente diferente |
| **ch 90-93** dark-center | 4 métricas diferentes (`dc_40m`, `dc_25m`, `lr_diff`, `dc_15m`) | 4× o mesmo centroide sliding-window | **BUG REAL**: semântica completamente divergente |
| **ch 82** urgência frontal | TTC = `speed/dist` (dimensão: `s⁻¹`) | `1 - dist/30` (dimensão: adimensional) | **BUG**: física diferente — CPU sobrescreve |
| **ch 32-39** curvatura flow | step fixo 2m por ângulo | `10 + i*1.5m` por ângulo | Leve — distância de amostragem diferente |
| **ch 48-51** apex_signal | normal 3D completo (inclui hz) | só hx, omite hz | Leve — equivalente se terreno flat |
| **L2 acumulação** | `acc*(1/127)` depois `*2.0` em 2 passos | `acc*(2.0/127.0)` direto | **Irrelevante**: ≤1 ULP em FP32 |
| **SlowBrain CfC** | dc_gain=1.0, τ fixo único | mesmo | **DIVERGE do cérebro rápido**: amplitude 2× menor |
| **Drag aerodinâmico** (CPU) | aplicado 2× com Cd diferente (`physics.rs:717`) | reaplicamento removido em `physics_c.wgsl:243` | **BUG na CPU**: carro CPU desacelera mais rápido |
| **tire_temp** | atualizada todo tick | congelada (copiada da entrada no Kernel C) | Leve: mu calculado com temperatura do tick anterior |
| **CfC percepção tanh** | tanh exato | aprox `x/(1+\|x\|)` (erro <2% para \|h\|<3) | Leve: diverge para ativações grandes |
| **Motion ch 21-27** | `delta_bytes / 8000` | `delta_dist_m / 2000` | Documentado como intencional |
| **WAVE_U32S** | host calcula `ceil(223/4)=56` | shader usa `48=ceil(192/4)` | **POTENCIAL BUG**: stride incorreto no decode se host aloca 56 e shader lê em 48 |

---

## 4. O que FALTA — lacunas

### Lacunas do checklist de análise matemática de redes

As seguintes análises **não existem** em nenhum arquivo do projeto:

1. **Estatística de ativação por neurônio (L1 e L2)**
   O único PR calculado é das 3 saídas do SlowBrain. Não existe histograma, nem média/σ,
   nem percentis das 96 unidades de L1 ou 48 unidades de L2.
   _Fórmula proposta:_ para cada neurônio `j`, coletar `act_j[t]` ao longo de N ticks:
   `μ_j = mean(act_j)`, `σ_j = std(act_j)`, `p5_j`, `p50_j`, `p95_j`.
   _Onde integrar:_ `gpu_evolve.rs` no loop de ticks, amostrando `l1_act` e `l2_raw`
   a cada K ticks.

2. **Neurônios mortos — % com ativação sempre zero**
   Um neurônio com `|act_j| < ε` para todos os ticks é "morto" (não contribui).
   _Fórmula:_ `morto_j = max(|act_j[t]|) < 0.01` ao longo de N ticks.
   _Onde integrar:_ na auditoria 3 eixos existente (`gpu_evolve.rs:2615`), adicionar
   um 4º critério `max_abs >= 1.0` (em escala i8).

3. **Neurônios saturados — % com |tanh(s)| > 0.99**
   Saturação é `|s| > 2.65` (pois `tanh(2.65) ≈ 0.99`). O clamp em ±2.0 já limita
   isso, mas `|state| = 2.0` implica `tanh(2.0) ≈ 0.964` — ainda informativo.
   _Fórmula:_ `saturado_j = mean(|act_j[t]|/127 > 0.96)`.
   _Onde integrar:_ mesma auditoria.

4. **Raio espectral de W_rec — estabilidade echo-state**
   Para uma rede recorrente, a condição de estabilidade é `|λ_max(W_rec)| < 1`.
   O CfC não tem W_rec explícito (é um integrador sem recorrência lateral), mas L1
   processa a concatenação `[w_cfc || w_kan]`, o que cria dependência temporal indireta.
   _Verificação:_ calcular `|λ_max|` dos pesos L1 pelo Jacobi já implementado em
   `brain_capacity.rs:36` para a matriz W_L1 (192×96).
   _Onde integrar:_ `brain_capacity.rs`, chamado uma vez por geração sobre o campeão.

5. **Participation Ratio das camadas L1 e L2** (já existe, mas é ÓRFÃO para essas camadas)
   A função `participation_ratio()` em `brain_capacity.rs:159` existe e funciona, mas
   **só é chamada para as 3 saídas do SlowBrain**. Chamar também para as ativações de
   L1 e L2 do cérebro rápido custo ≈ zero (já tem o Jacobi).
   _Onde integrar:_ `gpu_evolve.rs:2340`, coletar `l1_act[96]` e `l2_raw[48]` junto
   com as amostras do lento.

6. **Rank efetivo / SVD dos pesos**
   Verificar se W_L1 (192×96) tem posto efetivo << 96 (muitas linhas redundantes).
   _Fórmula:_ `rank_efetivo = Σσ_i / σ_max`, onde `σ_i` são valores singulares.
   _Onde integrar:_ `brain_capacity.rs`, função nova `effective_rank(weights, rows, cols)`.

7. **Ganho por camada (alpha de propagação)**
   `α_k = ||h_{k+1}|| / ||h_k||` — detecta se as camadas amplificam ou atenuam o sinal.
   _Fórmula:_ razão de normas L2 entre saídas de camadas consecutivas.
   _Onde integrar:_ calcular sobre as amostras coletadas no loop de ticks.

8. **Análise de inicialização — Xavier/He vs o que fazemos**
   Xavier recomenda `σ = sqrt(2/(fan_in + fan_out))`. Para L1 (192→96):
   `σ_xavier = sqrt(2/(192+96)) ≈ 0.083`. Usamos `σ = 0.30` (3.6× maior) com clamp
   em 3σ = 0.9. Isso pode causar saturação na primeira geração — não verificado.
   _Verificação imediata:_ calcular `mean(|l1_act|)` na geração 0 e ver se está > 100
   (saturação de início).

9. **Sensibilidade jacobiana `d(saída)/d(canal_sensorial)`**
   Detecta quais dos 192 canais realmente influenciam o atuador.
   _Fórmula:_ backpropagação da saída `intent[0]` (throttle) até a entrada, ou
   perturbação finita canal a canal.
   _Onde integrar:_ ferramenta de diagnóstico nova em `src/bin/`, não no loop de treino.

10. **Espectro de τ por geração**
    Hoje é reportado como p10/p50/p90 do campeão (`gpu_evolve.rs:2365-2376`). Falta:
    a) a dispersão p90/p10 por canal (não só do campeão, mas da população); b) correlação
    entre τ de um canal e sua contribuição (via ablação ou jacobiano).

11. **Erro de quantização i8 (SNR)**
    _Fórmula:_ `SNR = 10 * log10(signal_power / quant_error_power)`
    onde `quant_error = w_f32 - round(w_f32 * 127) / 127`.
    Para pesos L1 com σ ≈ 0.1, o erro de quantização é ≈ 1/(2·127) ≈ 0.004
    → SNR ≈ 28 dB. Para pesos próximos de zero, SNR cai — não monitorado.

12. **Entropia da política de ação**
    _Fórmula:_ `H = -Σ p(a) * log p(a)` onde `p(a)` é obtida por binning do `intent[0:3]`
    ao longo dos ticks. Baixa entropia = comportamento repetitivo (volante cravado);
    alta entropia = exploração.
    _Onde integrar:_ `replay_diag.rs` ou nova seção da auditoria.

13. **Correlação ativação ↔ fitness por neurônio**
    _Fórmula:_ para cada neurônio `j`, `r_j = pearson(act_j[t], fitness_proxy[t])`
    onde `fitness_proxy[t] = passo[t]` (avanço no tick `t`).
    Neurônios com `|r_j|` alto são "úteis"; com `r_j ≈ 0` são neutros.
    _Onde integrar:_ `gpu_evolve.rs`, coletando `l1_act` e `passo` por tick.

### Lacunas do código (bugs e mortos)

14. **cortex_l3 é código morto com 1152 pesos mutados.**
    `src/engine.rs:517: let _ = &self.cortex_l3;` — existe, é inicializado, é mutado pelo
    evolucionário, mas nunca entra no forward. Os 1152 pesos gastam memória e ciclos de
    mutação sem efeito.
    _Ação:_ remover do `Genome` ou conectar ao forward após L2.

15. **KAN sensorial (SensoryPreFilter) — adapt() nunca é chamado.**
    Os 3072 coeficientes KAN sensoriais (192 canais × 16 knots) são carregados no kernel
    GPU mas inicializados como identidade (`coeff = 1.0`) e nunca evoluem via gradiente ou
    regra Hebbiana. O KAN sensorial só age como passthrough.
    _Ação:_ conectar à mutação do genoma ou documentar como desativado.

16. **growth_verdict() e search_noise_threshold() são funções ÓRFÃS.**
    A "Lei do Garimpo" está implementada e testada em `brain_capacity.rs`, mas
    `growth_verdict()` não é chamada em nenhum ponto do fluxo de produção. A decisão de
    crescer a rede nunca é tomada automaticamente.
    _Ação:_ chamar em `gpu_evolve.rs` após calcular o PR das camadas L1/L2.

17. **saturation() também é órfã** — só chamada por `growth_verdict()`, que é órfã.

18. **Raio mínimo de viragem nunca é calculado.**
    Dado o déficit de torque de yaw (~30×), o carro não consegue completar a curva de 51m
    da pista 9001 acima de ~3-4 m/s em regime estacionário. Não existe nenhuma fórmula
    que compute `R_min(speed)`. Proposta:
    ```
    yaw_torque_max = max(F_lat) * wheel_long_pos    # Pacejka máximo
    yaw_damp = 400 + 60 * speed
    av_y_max = yaw_torque_max / (yaw_inertia * yaw_damp / yaw_inertia)
    R_min = speed / av_y_max
    ```
    _Onde integrar:_ `pista_check.rs`, validar cada curva da pista contra `R_min`.

19. **Divergência WAVE_U32S: host usa 56, shaders usam 48.**
    `perception.rs:84` calcula `ceil(223/4) = 56`; todos os WGSL usam `48 = ceil(192/4)`.
    Se o buffer é alocado com stride 56 (224 bytes) e o shader lê com base `being * 48`,
    qualquer ser > 0 lê offset errado (400 bytes vs 448 bytes do 2º ser).
    _Verificação urgente:_ ler `perception.rs:84` e o código de `create_buffer` para
    confirmar qual stride é usado no dispatch.

20. **STAGNATION_FOR_REVERSE**: `gpu_evolve` e `gpu_loop` usam 60 ticks mas têm
    comportamento diferente em stagnation 32..59 (escape_ticks só em gpu_loop com
    `IARA_ESCAPE=1`). Comportamento do DNA treinado em evolução diverge em produção.

21. **Propriocepção slots 3/7/8/11**: `gpu_loop` usa constantes (`hp=1.0`, `downforce=0`,
    `draft=0`, `wetness=0`) enquanto `gpu_evolve` injeta valores dinâmicos (`vida_frac`,
    `temp_motor`, `risco_cinetico`, `estagnação`). O DNA evoluído espera entradas dinâmicas
    mas recebe constantes em `gpu_loop`.

22. **Transferência de carga longitudinal/lateral ausente**: o modelo de suspensão calcula
    carga por roda com heightmap, mas não há transferência de carga por aceleração/curva.
    Um carro acelerando não aumenta carga nas rodas traseiras.

23. **Tire temp congelada na GPU**: o Kernel C não atualiza `tire_temp`; o `mu` é calculado
    com temperatura do tick anterior. CPU atualiza a cada tick.

24. **Nenhuma metrica de diversidade genética**: apenas a dispersão da fitness é monitorada.
    Não há distância média entre genomas, variância de pesos por posição, nem entropia
    da população.

25. **SlowBrain: sem banco de τ, dc_gain 2× menor**: já documentado como DIVERGÊNCIA do
    cerebro rápido (item F14). A evolução do lento opera com sinal 2× menor — pode ser a
    causa da PR baixa reportada.

---

## 5. Mapa de reorganização proposto

A estrutura atual tem os problemas: motor.rs com 3382 linhas em uma função; 4 cópias de
`make_physics_state` e `uniform_entry`; módulos mortos declarados em `lib.rs`; WGSL partida
em 8 arquivos de physics sem índice.

### Árvore nova proposta

```
src/
│
├── math/                   # NOVO: matemática pura, sem dependência de wgpu/bevy
│   ├── cfc.rs              # MOVE de src/cfc.rs (já isolado)
│   ├── pacejka.rs          # EXTRAI de physics.rs:539-564 (Pacejka + Kamm)
│   ├── brain_capacity.rs   # MOVE de src/brain_capacity.rs (já isolado)
│   ├── stats.rs            # NOVO: PR, OLS slope, pearson, activation_stats
│   └── quantize.rs         # EXTRAI de layers.rs:91-97 (save/load quantized)
│
├── brain/                  # MOVE de src/ (núcleo neural)
│   ├── mod.rs              # re-exports
│   ├── engine.rs           # MOVE src/engine.rs
│   ├── layers.rs           # MOVE src/layers.rs
│   ├── float_layers.rs     # MOVE src/float_layers.rs
│   ├── dense_layer.rs      # MOVE src/dense_layer.rs
│   ├── kan.rs              # MOVE src/kan.rs
│   ├── gate.rs             # MOVE src/gate.rs
│   └── neuro.rs            # MOVE src/neuro.rs
│
├── perception/             # MOVE de src/perception.rs + visao.rs
│   ├── mod.rs
│   ├── cpu.rs              # lógica de percepção CPU (extraída de motor.rs)
│   └── config.rs           # MOVE src/visao.rs
│
├── physics/                # MOVE de src/physics.rs
│   ├── mod.rs
│   ├── vehicle.rs          # EXTRAI de physics.rs: struct + constantes do veículo
│   ├── tires.rs            # EXTRAI de physics.rs: Pacejka + Kamm (usa math/pacejka.rs)
│   ├── dynamics.rs         # EXTRAI de physics.rs: integração Euler, yaw, suspensão
│   └── collision.rs        # EXTRAI de physics.rs: colisão parede
│
├── evolution/              # JÁ EXISTE — apenas mover gpu/evolution.rs para cá
│   ├── mod.rs
│   ├── genome.rs           # RENOMEIA gpu/evolution.rs → evolution/genome.rs
│   ├── pipeline.rs         # MOVE src/evolution/pipeline.rs
│   ├── spawn.rs            # MOVE src/evolution/spawn.rs
│   └── dna.rs              # MOVE src/evolution/dna.rs + dna_genesis.rs
│
├── gpu/                    # PERMANECE — só elimina duplicatas
│   ├── core.rs             # uniform_entry / storage_entry (FONTE ÚNICA)
│   ├── brain.rs            # BrainGpu (remove duplicatas de layout)
│   ├── slow_brain.rs       # SlowBrainGpu
│   ├── perception.rs       # PerceptionGpu
│   ├── physics.rs          # PhysicsGpu — renomeia kernels physics_a/b/c como pipeline único
│   ├── vehicle.rs          # make_physics_state (FONTE ÚNICA — hoje 4 cópias)
│   └── mod.rs
│
├── analysis/               # NOVO: ferramentas de diagnóstico (não entram no treino)
│   ├── activation_stats.rs # stats de ativação L1/L2 (Lacuna 1-3)
│   ├── jacobian.rs         # sensibilidade d(saída)/d(canal) (Lacuna 9)
│   ├── spectral.rs         # raio espectral, rank efetivo, SVD (Lacunas 4, 6)
│   └── diversity.rs        # diversidade genética, entropia da política (Lacunas 12, 24)
│
├── world/                  # MOVE de src/world.rs + environment.rs + pista_gen.rs
│   ├── mod.rs
│   ├── voxel.rs            # EXTRAI de world.rs: VoxelWorld
│   ├── track.rs            # EXTRAI de world.rs: waypoints + pistas
│   ├── environment.rs      # MOVE src/environment.rs
│   └── generator.rs        # MOVE src/pista_gen.rs
│
├── simulation/             # JÁ EXISTE — foco: dividir motor.rs
│   ├── mod.rs
│   ├── motor.rs            # REFATORA: extrair funções auxiliares (hoje 1 função de 3148 linhas)
│   ├── sensors.rs          # EXTRAI de motor.rs: pipeline de percepção CPU
│   ├── brain_interface.rs  # EXTRAI de motor.rs: chamada do cérebro + modulação lenta
│   ├── control.rs          # EXTRAI de motor.rs: reverse logic, steer smooth, axiom_policy
│   ├── racer.rs            # MOVE src/simulation/racer.rs
│   └── racing_line.rs      # MOVE src/simulation/racing_line.rs
│
├── config.rs               # PERMANECE — adicionar SIM_HZ, VEHICLE_*, I8_SCALE, etc.
├── rng.rs                  # PERMANECE
└── lib.rs                  # REMOVE módulos mortos (memory, repominer, reality, mesh, substrate)

src/bin/
├── gpu_evolve.rs           # REFATORA: extrair make_physics_state → gpu/vehicle.rs
├── gpu_loop.rs             # idem
├── gpu_report.rs           # idem
├── cortex_evolve.rs        # idem
└── [diagnósticos]          # mantém como estão
```

### O que move de onde para onde (resumo executivo)

| Hoje | Para onde | Por quê |
|---|---|---|
| `cfc.rs` | `math/cfc.rs` | É matemática pura — sem deps de wgpu |
| `brain_capacity.rs` | `math/brain_capacity.rs` | Idem |
| `gpu/evolution.rs` | `evolution/genome.rs` | Pertence à evolução, não à GPU |
| 4× `make_physics_state` | `gpu/vehicle.rs` (fonte única) | Elimina divergência de `vel=[0;3]` em cortex_evolve |
| 4× `uniform_entry` | `gpu/core.rs` (fonte única) | DRY |
| `physics.rs:539-564` | `math/pacejka.rs` | Reutilizável e testável isoladamente |
| Módulos mortos | Removidos de `lib.rs` | `memory`, `repominer`, `reality`, `mesh`, `substrate` não têm referência externa |
| `motor.rs:234-3382` | Dividir em `sensors.rs + brain_interface.rs + control.rs` | Uma função de 3148 linhas é ilegível |
| Lacunas 1-3, 5, 9, 12 | `analysis/activation_stats.rs` etc. | Análises que Leonardo pediu, ainda inexistentes |
| `stats.rs` (novo) | `math/stats.rs` | PR, OLS, pearson já existem mas dispersos |

---

_Fim do inventário. Total de fórmulas catalogadas: 61. Total de constantes mágicas: ~55.
Duplicações confirmadas: 35+ pares; com divergência confirmada: 12 pares._
