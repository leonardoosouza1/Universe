# What the CfC's 96 channels are actually worth

> ## CORRECTION 2026-08-25 — sections 2 to 5 are MEASURED ON A PARKED CAR. Do not cite them.
>
> Every measurement below marked (†) was taken from `obs.f32` after `--retomar`, and
> `--observar` always dumps **being 0**, which after a resume is the exact reloaded elite.
> That being does not drive: median speed 0.069 m/s, never above 1.9 m/s, 12 m covered in
> 62 s, median throttle 0. See `docs/CHAMPION_RELOAD.md` — the champion that scored
> 9870 m reproduces at 118-124 m on reload, in five different configurations.
>
> A state measured on a stationary car cannot support any claim about what a *driving*
> being computes. "76% of the state is an affine function of speed" is, on that data,
> mostly a statement about a speed that barely varies.
>
> What survives: **section 1** (the closed form and its validation against the
> iara-semente session — nothing to do with this repository) and **section 6** (the τ
> range, read from the genome file, not from a run). Sections 2-5 have to be re-measured
> against a being that actually drives before any of it can be believed.
>
> The process failure is worth naming: this file checked the semantics of the CfC columns
> carefully and never checked whether the car was moving. The cheapest possible sanity
> check — print the median speed — would have caught it before any of the analysis.


> Derived 2026-08-25 from the spectral analysis of the iara-semente τ bands (session
> "Ondas de Tau"). A closed form was carried over from that session, **predicted a number
> for this repository, and was refuted here**. What survived the refutation is a measured
> mechanism for the open problem in `docs/ROADMAP.md`: why a closed-form pilot beats the
> evolved champion.
>
> Scope: one being (slot 0), one champion (`escada2-s31` gen 137), one track (seed 9001),
> 4000 ticks, no training run. Every number below comes from that single `obs.f32`.

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

## 2. (†) It does not transfer here — and that is the point

Prediction registered before measuring: build the 96×96 matrix `r(τᵢ/τⱼ)` from the
champion's own τ bank and it should reproduce the measured correlation of the post-CfC
state, giving `PR = (Σλ)²/Σλ² ≈ 1.75`.

Measured, over all 4560 channel pairs:

| ρ band | pairs | law | measured |
|---|---|---|---|
| 1–2× | 2068 | 0.983 | 0.408 |
| 2–4× | 1259 | 0.883 | 0.429 |
| 4–8× | 845 | 0.726 | 0.416 |
| 8–16× | 328 | 0.565 | 0.400 |
| 16×+ | 60 | 0.413 | 0.480 |

**Correlation between the law's prediction and the measurement, across pairs: −0.007.**
The measured correlation is FLAT at ≈0.41 regardless of the τ ratio. The law is refuted
for this repository, cleanly and with a mechanism: in iara-semente the four bands filter
one shared scalar, whereas here each of the 96 CfC channels filters a **different**
sensory channel. τ spread is simply not what sets the redundancy of this state.

Two naming traps found while doing this, both worth fixing separately:

* `Genome::tau` holds **decay rates** (s⁻¹), not τ. `evolution.rs:562` fills it from
  `cfc::log_uniform_decays` and `genome_io.rs:177` defaults it to `DECAY_DEFAULT = 0.5`.
  τ = 1/value.
* `obs.f32` columns 10..106, read back as `cfc_in`, are the **post**-CfC output —
  `brain.wgsl:375` declares `w_cfc` as "CfC output (tanh(state) × 127)".

## 3. (†) The state spans 3.77 of 96 dimensions

Participation ratio of the post-CfC correlation matrix, measured:

```
PR = 3.77   out of 96 live channels
```

The spectrum says where it went:

| mode | share of variance | cumulative |
|---|---|---|
| 0 | 47.0% | 47.0% |
| 1 | 13.6% | 60.5% |
| 2 | 11.4% | 71.9% |
| 3 | 8.0% | 79.9% |
| 4 | 5.0% | 84.9% |

**Mode 0 is speed.** `corr(mode 0, velocity) = −0.923`. It is spread almost flat across
the bank — median channel weight 0.115 against a maximum of 0.145, with 62 of 96 channels
above half the maximum — and `corr(|weight|, log τ) = −0.425`, so τ does not decide who
joins it. Removing that one mode lifts PR from 3.77 to 6.29.

Half the representational budget of a 96-channel liquid memory is one scalar the being
already receives directly.

## 4. (†) Three quarters of the state is an affine function of the dashboard

Least squares from raw telemetry columns to all 96 post-CfC channels, aggregate R²:

| predictors | R² |
|---|---|
| speed alone | 0.642 |
| rpm alone | 0.132 |
| **speed + rpm + their squares + speed·rpm** | **0.758** |
| 5 body scalars + squares + speed·rpm | 0.795 |
| the same, plus the clock | 0.795 |

And on the command the physics actually receives, `intent[0..3)`:

| predictors | R² |
|---|---|
| speed + rpm (+ squares) | 0.541 |
| all 96 post-CfC channels | 0.882 |

**The clock control matters and changes the reading.** `life_left` has `corr(t) = −1.000`
and is strictly decreasing — it is the tick counter in disguise, not a body signal.
`engine_temp` is nearly as bad at −0.923. The clock alone explains 53.2% of the state, so
any result that leans on those two columns is suspect. It survives anyway: speed and rpm
with no clock term reach 0.758, and adding the clock on top of the full model moves
nothing (0.795 → 0.795). Speed and rpm subsume the clock, not the other way round.

## 5. (†) Why this would matter for the roadmap

`docs/ROADMAP.md` makes distillability the project's first number: a closed-form ridge
pilot beats the champion (7139 m vs 6592 m), and the target is to drive that down. This is
a mechanism for it, not just a restatement — if 76% of the state a pilot regresses on is a
smooth function of two dashboard numbers, a closed form is not clever, it is sufficient.
A world that demands a brain has to make the state depend on something a regression on
speed and rpm cannot reach.

It also makes the M0 mechanisms testable on a cheaper axis than lap distance: re-run this
same measurement with `--recurso-escasso`, `--predador` or `--interocepcao` on, and see
whether R²(speed, rpm → state) falls. A hostile mechanism that leaves it at 0.76 did not
change what the being computes, whatever it did to the distance.

## 6. A separate gap, independent of all the above

`TAU_MAX_S = 4.0` (`cfc.rs:24`) is documented as "double the single τ the brain used to
have". The champion's evolved bank runs τ ∈ [0.140 s, 8.78 s] — 3 channels pinned at the
`d = 0.25` ceiling and one mutated past it. Median τ is 1.66 s.

At the ~20 m/s the population reaches, median τ spans **33 m of a 2928 m lap**, and the
slowest channel spans 176 m. A lap is ~146 s and the race is 312 s (20,000 ticks at
64 Hz). No channel can hold "this lap versus the last one".

That is worth fixing on its own evidence, but note what section 3 implies: widening the
range will not by itself raise PR much while one common mode owns 47% of the variance.
Fix the common mode first, then re-measure whether the range binds.

## Reproducing

```bash
cargo build --release
cp /media/llm-data/universe-runs/escada2-s31/evolve_best_gen0137.bin "$D"/
./target/release/gpu_evolve --retomar --out "$D" --gens 1 --seres 8 \
    --ticks 4000 --seed 31 --same-start --observar "$D"/obs.f32
./target/release/genoma_dump \
    --arquivo /media/llm-data/universe-runs/escada2-s31/evolve_best_gen0137.bin --saida "$D"
```

`obs.f32` rows are 394 f32: `[0]` tick, `[1]` speed, `[2]` rpm, `[3]` gear, `[4]` engine
temp, `[5..8)` throttle/steer/brake, `[8]` life, `[9]` joy, `[10..106)` post-CfC,
`[106..202)` post-KAN bypass channels, `[202..298)` L1 raw, `[298..394)` intent.
