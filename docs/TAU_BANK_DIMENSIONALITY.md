# What the CfC's 96 channels are actually worth

> Measured 2026-08-25 on `ereg-m4-m4-s77` generation 43 reloaded: 48 beings, 20,000
> ticks, track rotation [8005, 8200, 9001], `--width 28 --out-beta 0.0
> --mascara-regioes --neuromod-regioes`. Median 2939 m (1.79 laps), max 4289 m. The
> being behind the numbers (slot 0, the exact elite) covered 2618 m with 41% of its
> ticks above 5 m/s. No training run: one generation, evaluation only.
>
> **An earlier version of this file, committed the same day, was measured on a PARKED
> being** — `--retomar` was silently loading a random population (`docs/CHAMPION_RELOAD.md`,
> fixed in `5c1ceab`). Every number below has been re-measured against a being that
> drives. The comparison is kept in section 5 because the difference is instructive:
> the headline survived and the worst confound disappeared.

## 1. The law, and where it does hold

Two first-order EMA channels with rates `a = 1/τ₁`, `b = 1/τ₂` fed **the same** white
input. One integral gives the correlation:

```
Cov = ∫₀^∞ e^(-au)e^(-bu) du = 1/(a+b),   Var_i = 1/(2a_i)

              1/(a+b)          2√ρ
r(ρ) = ───────────────────── = ─────── ,   ρ = τ₂/τ₁
       √(1/(2a) · 1/(2b))       1 + ρ
```

It depends on the RATIO only, and is symmetric under ρ → 1/ρ.

Against the four cross-correlations the iara-semente session measured on 399,998 bytes of
real Wikipedia-PT surprise signal — not fitted to them:

| pair | ρ | law | measured | error |
|---|---|---|---|---|
| τ4 ↔ τ16 | 4 | 0.800 | 0.793 | +0.007 |
| τ16 ↔ τ64 | 4 | 0.800 | 0.829 | −0.029 |
| τ4 ↔ τ64 | 16 | 0.471 | 0.510 | −0.039 |
| τ1 ↔ τ64 | 64 | 0.246 | 0.206 | +0.040 |

Worst error 0.040. The law is sound where its premise is: **one signal, many filters.**

## 2. The law does not transfer here

Prediction registered before measuring: the 96×96 matrix `r(τᵢ/τⱼ)` built from the
champion's own τ bank should reproduce the measured correlation of the post-CfC state.

Measured over all 4560 channel pairs:

| ρ band | pairs | law | measured |
|---|---|---|---|
| 1–2× | 1786 | 0.984 | 0.375 |
| 2–4× | 1247 | 0.878 | 0.385 |
| 4–8× | 942 | 0.711 | 0.347 |
| 8×+ | 585 | 0.538 | 0.305 |

**Correlation between the law's prediction and the measurement: +0.101.** The measured
value is roughly flat near 0.35 whatever the τ ratio, where the law spans 0.98 down to
0.54. Refuted, with a mechanism: in iara-semente the four bands filter one shared scalar,
whereas each of the 96 CfC channels here filters a **different** sensory channel. τ spread
is not what sets this state's redundancy.

Two naming traps found on the way, both worth their own fix:

* `Genome::tau` holds **decay rates** (s⁻¹), not τ. `evolution.rs:562` fills it from
  `cfc::log_uniform_decays` and `genome_io.rs:177` defaults it to `DECAY_DEFAULT = 0.5`.
  τ = 1/value.
* `obs.f32` columns 10..106, read back as `cfc_in`, are the **post**-CfC output —
  `brain.wgsl:375` declares `w_cfc` as "CfC output (tanh(state) × 127)".

## 3. The state spans 4.68 of 96 dimensions

```
PR = 4.68   out of 96 live channels     (all 96 alive)
```

| mode | share of variance | cumulative |
|---|---|---|
| 0 | 43.1% | 43.1% |
| 0–4 | — | 72.4% |

**Mode 0 tracks speed**: `corr(mode 0, velocity) = −0.797`. Five modes carry 72% of
everything 96 channels of liquid memory represent.

## 4. The dashboard explains the REGIME, not the command

A global least squares from telemetry to the 96 post-CfC channels and to the command
`intent[0..3)` looks damning at first:

| target | predictors | global R² |
|---|---|---|
| post-CfC state | speed + rpm + squares | 0.585 |
| command | speed + rpm + squares | 0.542 |
| command | all 96 post-CfC channels | 0.839 |
| post-CfC state | the clock alone (t, t²) | 0.092 |

Read as "half the driving is dashboard-following", which is what an earlier version of this
file claimed. **That reading is wrong**, and the control that kills it is stratification by
speed.

The being spends 49% of its ticks below 1 m/s and 26% above 20 m/s. That is a bimodal
regime, and a global regression on speed buys the separation between "stopped" and "fast"
for free — which is not the same as predicting what the being does. Repeat the fit inside
speed bins and compare like with like:

| target | global | within speed bins |
|---|---|---|
| state ← dashboard | 0.585 | **0.123** |
| command ← dashboard | 0.542 | **0.113** |
| command ← 96 post-CfC channels | 0.839 | **0.815** |

The brain's number barely moves. The dashboard's collapses by a factor of five. Within a
regime the 96 channels explain 0.815 of the command against the dashboard's 0.113 — the
brain contributes about **seven times** what speed and rpm do, and the surplus is +0.70.

Bins used: [0,1), [1,5), [5,12), [12,20), [20,40) m/s, averaged over the bins holding at
least 400 ticks. Stratification is itself a strong control — conditioning on speed removes
part of the very variance being tested — so 0.113 is a floor and 0.542 a ceiling. What is
not ambiguous is the CONTRAST: one number survives the control and the other does not.

**The being is not following its own dashboard.** That conclusion replaces the opposite
one, drawn here on the same day from the same data without this control.

## 5. What this measurement cannot do

The natural next step was to rank the M0 mechanisms by whether they reduce the command's
dependence on the dashboard. Measured, on the frozen champion, average of the surplus
(brain minus dashboard) over speed bins:

| arm | dashboard | brain | surplus |
|---|---|---|---|
| off | 0.113 | 0.815 | 0.701 |
| `--interocepcao` | 0.127 | 0.824 | 0.697 |
| `--pneu-termico 2.0` | 0.140 | 0.848 | 0.708 |
| `--predador 0.70` | 0.183 | 0.812 | 0.629 |

They are the same to within the spread. **The metric does not separate the mechanisms**,
and the reason is structural rather than a matter of dose: a frozen genome computes one
fixed function, so switching on a world pressure can only change its INPUT distribution,
never what it computes. "This world demands more brain" is a claim about what EVOLUTION
selects, and no single-generation evaluation can hold it.

The honest boundary of every measurement in this file, then: it can show that a mechanism
is wired, that it is inert when its flag is off, and what dose stalls the being. It cannot
rank mechanisms by difficulty. That needs a training run.

## 6. What survives for the roadmap

`docs/ROADMAP.md` opens on distillability: a closed-form ridge pilot beats the champion
(7139 m vs 6592 m). Section 4 was written as a mechanism for that result and does not
support it — within a regime, a regression on two dials reaches 0.113 of the command, so
the pilot's success is not explained by the being being trivially predictable from its own
telemetry. Where the closed form's advantage does come from remains open, and
`docs/DESTILABILIDADE.md` still owns the question.

What section 3 does contribute stands on its own: 96 channels of liquid memory span 4.68
effective dimensions, and one mode holding 43% of the variance tracks speed. That is a
statement about capacity, and it is the input the `src/brain_capacity.rs` criterion needs.

## 7. A separate gap, independent of all the above

`TAU_MAX_S = 4.0` (`cfc.rs:24`) is documented as "double the single τ the brain used to
have". The champion's evolved bank runs τ ∈ [0.200 s, 13.20 s] — a range of 66×, with mutation
having pushed the slowest channel more than three times past the declared ceiling. Median
τ is 1.55 s.

At the ~20 m/s the population reaches, median τ spans **31 m of the 1639 m lap**, and the
slowest channel 264 m. The race is 312 s (20,000 ticks at 64 Hz). No channel can hold "this lap versus the last one".

That is worth fixing on its own evidence, but note what section 3 implies: widening the
range will not by itself raise PR much while one common mode owns 47% of the variance.
Fix the common mode first, then re-measure whether the range binds.

## Reproducing

```bash
cargo build --release
G=/media/llm-data/universe-runs/ereg-m4-m4-s77/evolve_best_gen0043.bin
cp "$G" "$D"/
./target/release/gpu_evolve --retomar --out "$D" --gens 1 --seres 48 --ticks 20000 \
    --width 28 --out-beta 0.0 --mascara-regioes --neuromod-regioes \
    --pistas 8005,8200,9001 --seed 77 --observar "$D"/obs.f32
./target/release/genoma_dump --arquivo "$G" --saida "$D"
```

`obs.f32` rows are 394 f32: `[0]` tick, `[1]` speed, `[2]` rpm, `[3]` gear, `[4]` engine
temp, `[5..8)` throttle/steer/brake, `[8]` life, `[9]` joy, `[10..106)` post-CfC,
`[106..202)` post-KAN bypass channels, `[202..298)` L1 raw, `[298..394)` intent.

Check the median speed of column 1 before trusting anything computed from this file.
On a stationary being every result in sections 2 to 6 changes.
