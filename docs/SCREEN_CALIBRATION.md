# Screen calibration — the open-loop screen against closed-loop ground truth

**Measured 2026-08-27**, RX 6750 XT (RADV, `env -u DISPLAY`), branch `race-world-v3`,
binaries built at commit `1ff71af`.

`src/bin/screen.rs` replays a recorded 192-channel sensory wave through the real GPU
brain kernel and scores a candidate genome's commands against the recording's own,
as mean absolute error per tick over the three commands. This document answers the
three questions that decide whether that score may be used to prune candidates:

1. **How short can the replay window be** before the ranking changes?
2. **Does the score order candidates by quality**, or only separate them into two bins?
3. **How often does the screen reject a candidate that drives?** (the false-negative
   rate — the number the funnel lives or dies on)

**Verdict up front.** The screen is **bit-exact** (the recording genome scores 0.0000 at all 20 000 ticks) and
**cheap** (21.8 ms per candidate at the recommended window, 46× cheaper than the closed
loop). It is also, on 85 measured candidates, **unusable as the pruning filter the design
spec assumes it to be.**

* **Shortest window that preserves the ranking: 2 000 ticks** (Spearman +0.945 against the
  full 20 000-tick ranking). But no threshold at any shorter window reproduces the
  accept/reject partition that 0.26 produces at 20 000 — the score moves with the window.
* **The metric is not monotone in genome distance** (13.5% of strictly-ordered pairs are
  inverted; three perturbation directions of identical magnitude score 3.2× apart), and it
  **does not rank by quality at all**: Spearman against closed-loop driving distance is
  **−0.05** over 85 candidates, negative at essentially every window. It is a filter at
  best, and the filter is wrong.
* **False rejects: 6 of the 7 candidates the screen rejects drive at ≥75% of the recording
  champion; 2 of the 7 drive better than it.** `gen30` — a real champion from the
  recording's own lineage — scores 0.2618 (rejected) and drives **162%** of the champion.
  **The 0.26 threshold is unsafe for pruning and must not be used to discard candidates.**

The screen's one reliable capability is telling a *destroyed* brain from a functioning
one. Recommendation in §6: demote it from a funnel to a tripwire at ≈0.60.

---

## 1. What was measured

### The recording

`/media/llm-data/scratch/wave/wave.f32`, 20 001 rows × (192 channels + 3 commands),
`UWAV` v1 header, made by Task 5's re-recording of champion `ereg-m4-m4-s77` gen 43:

```bash
D=/media/llm-data/scratch/wave
./target/release/gpu_evolve --retomar --out $D --gens 1 --seres 48 --ticks 20000 \
  --width 28 --out-beta 0.0 --mascara-regioes --neuromod-regioes \
  --pistas 8005,8200,9001 --seed 77 --gravar-onda $D/wave.f32 --observar $D/obs.f32
```

Row 0 is the JIT warm-up tick and is replayed but not scored; 20 000 ticks are scored.
The recording genome scores **0.0000 at every one of those 20 000 ticks** against its own
recording (`mix100` in every table below, and `mix100.bin` is byte-identical to
`evolve_best_gen0043.bin` — `cmp` reports no difference). There is no screen-side bias to
subtract.

### The candidate sets

**Set A — 25 degraded copies of the recording champion** (the brief's mixes, extended).
Built by the same mechanism `--retomar-mix` uses (`src/bin/gpu_evolve/main.rs:2100-2112`:
multiply L1 and L2 by a scalar), plus two further perturbation directions so that
*magnitude* and *direction* can be separated:

| family | what it does | members |
|---|---|---|
| `mixNNN` | L1, L2 × `NNN/100` — attenuation, the `--retomar-mix` mechanism. Direction preserved, norm falls | 1.00, 0.98, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.30 |
| `ampNNN` | the same but > 1 — amplification | 1.05, 1.15, 1.30 |
| `noiSSS_sK` | L1, L2 += `N(0, SSS/100 × std(W))` — an actually *different* brain, not a timider one | σ 0.02…2.00, seeds 1-3 at σ 0.10 and 0.50 |

Every candidate carries a measured **relative perturbation magnitude**
`‖G − C‖₂ / ‖C‖₂` over the 23 040 L1+L2 weights. For `mix`/`amp` it is exactly `|1 − m|`;
for `noi` it comes out within 1% of σ. This is the axis question 2 is about: three
directions reach the same magnitude, so "is the score a function of magnitude?" is
directly testable.

Generator (the exact script that produced the files, run against
`evolve_best_gen0043.bin`):

```python
raw = bytearray(open(SRC, "rb").read())          # b"UGEN" | u32 version | L1 | L2 | ...
O1, END = 8, 8 + (18432 + 4608) * 4              # src/gpu/genome_io.rs:56-100
base = np.frombuffer(bytes(raw[O1:END]), dtype="<f4").astype(np.float64)
out = bytearray(raw); out[O1:END] = w.astype("<f4").tobytes()   # w = base*m, or base+noise
```

**Set B — the recording's own lineage, 60 real champions.**
`ereg-m4-m4-s77/evolve_best_gen0000.bin` … `gen0059.bin`. These are not degraded copies:
they are brains that *selection actually produced*, at relative distance 0.05-0.42 from
the recording genome. Set A can only ever contain candidates that are worse than or equal
to the champion, so it cannot contain the case the funnel is most dangerous for — a brain
that is **different and good**. Set B can, and does.

### The closed-loop ground truth

One `--gens 1` evaluation per candidate, the recording's own command with
`--retomar-mix 1.00` so that all 48 beings carry the candidate's weights (τ is drawn from
`--seed 77` and is therefore identical across candidates — the comparison is paired):

```bash
env -u DISPLAY ./target/release/gpu_evolve --retomar --out $O --gens 1 --seres 48 \
  --ticks 20000 --width 28 --out-beta 0.0 --mascara-regioes --neuromod-regioes \
  --retomar-mix 1.00 --pistas 8005,8200,9001 --seed 77
```

with `$O/evolve_best_gen0000.bin` = the candidate. Reported quantity: the run's own
`med=` (median metres over the 48 beings) and `≥1V=` (beings completing a real lap).
Median over the population, not the best individual, per `CLAUDE.md`.

**Why 48 beings and not 1.** `--seres 1` is *not* the same closed loop: channel 125 is
`CompStress`, the distance to the nearest being ahead within 20 m
(`src/bin/gpu_evolve/context.rs:1177-1226`). Recording being 0 alone and comparing against
the 48-being recording, the waves differ at row 0 in exactly channel 125, and the
recorded commands differ from row 0 onward. Measured, not assumed:

```bash
cmp $W1/wave.f32 $D/wave.f32   # differ: byte 563  → row 0, channel 125
```

**Ceiling of the metric** (`CLAUDE.md` rule 4), computed before reading any result:
gen 0 of a `--pistas 8005,8200,9001` run uses track 8005, lap 1 639 m
(`[pista] gen 0 usa seed=8005 (77 wp, volta=1639m)`). <!-- check-language: allow -->
20 000 ticks at 64 Hz is 312.5 s; at the ~22 m/s the champion reaches that is ≈6 900 m,
≈4.2 laps. The best number anywhere below is 4 335 m, so nothing here is a metric
hitting its limit.

---

## 2. Result 1 — the shortest window

The screen's own ranking is stable well below the full recording. Spearman of each
window's ranking against the 20 000-tick ranking, over all 85 candidates that have both
a screen score and closed-loop ground truth:

| window | ρ vs 20000-tick ranking | ρ vs closed-loop median | ρ vs ≥1 lap |
|---|---:|---:|---:|
| 250 | +0.8232 | -0.015 | +0.013 |
| 500 | +0.8444 | -0.046 | -0.019 |
| 1000 | +0.8552 | -0.011 | +0.008 |
| 2000 | +0.9451 | -0.036 | -0.004 |
| 5000 | +0.9855 | -0.058 | -0.017 |
| 10000 | +0.9906 | -0.079 | -0.026 |
| 20000 | +1.0000 | -0.050 | +0.001 |

**Recommended window: 2 000 ticks** (ρ = +0.945 against the full-window ranking over
A+B; +0.988 over Set A alone). 5 000 buys +0.04 for 2.3× the cost; below 2 000 the
ranking degrades fast (+0.855 at 1 000, +0.823 at 250) because the first ~1 000 ticks are
the launch, where every candidate's commands are saturated and nearly identical.

**But the 0.26 threshold does not survive the shortening, and barely survives at all.**
A short window scores *higher* than the full one (the launch and the first corner
dominate the mean), so the number moves with the window. Asking which threshold at
window *W* reproduces the accept/reject partition that 0.26 produces at 20 000:

| window | highest keep | lowest reject | threshold that reproduces the partition |
|---|---:|---:|---|
| 250 | 0.5611 | 0.2432 | **none exists** |
| 500 | 0.5532 | 0.2502 | **none exists** |
| 1000 | 0.5280 | 0.2723 | **none exists** |
| 2000 | 0.4013 | 0.3585 | **none exists** |
| 5000 | 0.3214 | 0.3204 | **none exists** |
| 10000 | 0.3169 | 0.2933 | **none exists** |
| 20000 | 0.2581 | 0.2618 | any in (0.2581, 0.2618] |

At every window shorter than the full recording, some candidate that 0.26 keeps at
20 000 outscores some candidate that 0.26 rejects — **no rescaling of the threshold fixes
it.** (Over Set A alone the partition does survive rescaling at 2 000, 5 000 and 10 000
ticks; it is the 60 real lineage champions of Set B that break it, which is exactly the
population a funnel would be applied to.) That is why `screen.rs` suppresses verdicts on partial windows (Task 5, finding I4);
this measurement says the suppression is correct and should stay.

Even at the full 20 000 ticks the partition survives only inside a **0.0037-wide band**:
the highest keep is `gen14` at 0.2581 and the lowest reject is `gen30` at 0.2618. There is
no gap in the distribution at 0.26. The threshold is not cutting a valley; it is cutting
through the middle of a dense cluster.

---

## 3. Result 2 — the metric is not monotone in genome distance, and does not rank

### It is not a function of how far the genome moved

Over Set A at the full window, `ρ(score, ‖ΔW‖/‖W‖) = +0.8894` — high, and misleading.
Of the 297 pairs whose magnitudes differ, **40 (13.5%) are inverted**: the candidate that
moved further scores lower. Grouping candidates by measured magnitude:

```
d=0.05: amp105 0.0189, mix095 0.0196, noi005_s1 0.0599                     spread 3.2x
d=0.10: mix090 0.0354, noi010_s2 0.1006, noi010_s3 0.1091, noi010_s1 0.1137  spread 3.2x
d=0.15: amp115 0.0440, mix085 0.0503                                       spread 1.1x
d=0.20: mix080 0.0651, noi020_s1 0.1305                                    spread 2.0x
d=0.30: amp130 0.0654, mix070 0.0941                                       spread 1.4x
d=0.50: mix050 0.1622, noi050_s1 0.1874, noi050_s3 0.3086, noi050_s2 0.4287  spread 2.6x
```

Three directions of perturbation reaching the *same* measured magnitude score up to
**3.2× apart** (`mix090` 0.0354 vs `noi010_s1` 0.1137, both at ‖ΔW‖/‖W‖ = 0.10). Three
*seeds* of the same σ = 0.50 noise span 2.6× (0.1622 … 0.4287). And the brief's specific
suspicion is confirmed: a larger perturbation can score better —

| pair | magnitude | score @20000 |
|---|---:|---:|
| `noi035_s1` | 0.35 | 0.2633 |
| `noi050_s1` | 0.50 (**43% larger**) | **0.1874** |
| `noi050_s2` | 0.50 | 0.4287 |
| `noi100_s1` | 1.00 (**2× larger**) | **0.3272** |

Direction dominates magnitude. Attenuation (`mix`) keeps the weight vector's direction
and only shrinks its norm, and the network's saturating non-linearities absorb a large
part of that: `mix030` throws away 70% of the weights and still scores 0.2359, while
`noi035_s1` — half the magnitude, but in a random direction — scores 0.2633.

### It does not rank by quality either

The question that matters is not "does the score track the genome" but "does it track
whether the brain drives". Spearman of the screen score against the closed-loop median,
per replay window and per candidate set:

| window | Set A (n=25) | Set B (n=60) | A+B (n=85) | A+B, vs beings completing ≥1 lap |
|---|---:|---:|---:|---:|
| 250 | −0.279 | −0.240 | −0.015 | +0.013 |
| 500 | −0.335 | −0.254 | −0.046 | −0.019 |
| 1000 | −0.354 | −0.165 | −0.011 | +0.008 |
| 2000 | −0.396 | −0.137 | −0.036 | −0.004 |
| 5000 | −0.367 | −0.144 | −0.058 | −0.017 |
| 10000 | −0.337 | −0.153 | −0.079 | −0.026 |
| 20000 | −0.364 | −0.100 | −0.050 | +0.001 |

**−0.05 over 85 candidates.** Per set: −0.36 (Set A, 25 degraded copies) and −0.10
(Set B, 60 real champions), at the full window; the sign is negative at essentially every
window of both sets. The screen's score carries **no usable information about driving
quality**, and what little signal there is points the wrong way: within this candidate
population, being *more* divergent from the recording champion is weakly associated with
driving *better*.

The mechanism is not mysterious: the recording champion `gen43` is one point in a
population where 56 of the 85 candidates outdrive it on this track (closed-loop medians
span 71 m to 4 949 m). A metric whose zero is pinned at that one point measures distance
from a mediocre reference, and distance from a mediocre reference is not quality.

**The over-fitting explanation does not rescue it.** `gen43` was selected on track 8200
(generation index 43 mod 3 = 1 of `--pistas 8005,8200,9001`) and everything above is
evaluated on 8005, so "perturbing an over-fitted champion helps" is a live alternative.
It was tested — the same 11 candidates re-run on 8200, the champion's own track:

| candidate | score @20000 | 8005 med (m) | 8200 med (m) |
|---|---:|---:|---:|
| `mix100` | 0.0000 | 2418 | 4193 |
| `mix090` | 0.0354 | 2758 | 4742 |
| `mix070` | 0.0941 | 2819 | 6377 |
| `noi010_s1` | 0.1137 | 2558 | 4464 |
| `noi020_s1` | 0.1305 | 2828 | 3995 |
| `mix050` | 0.1622 | 2452 | 6714 |
| `mix030` | 0.2359 | 2029 | 5644 |
| `noi035_s1` | 0.2633 | 2192 | 2730 |
| `noi100_s1` | 0.3272 | 1936 | 3158 |
| `noi050_s2` | 0.4287 | 2986 | 4316 |
| `noi200_s1` | 0.9066 | 71 | 42 |

`ρ(score, 8200 median) = −0.491` (n = 11), against `ρ(score, 8005 median) = −0.382` over
the same 11.

The correlation is *more* negative on the champion's home track, not less. (Caveat on this
control: 8200's lap is ~2 377 m and 20 000 ticks at 64 Hz allow ≈6 900-7 300 m, so the top
rows here are close to the tick-budget ceiling and their ordering among themselves is not
to be trusted. The sign of the correlation does not depend on them.)

**Verdict on question 2: the screen is neither a ranker nor a trustworthy filter at 0.26.**
It is a *similarity* measure that behaves monotonically only along a single fixed
perturbation direction, and similarity to this champion is uncorrelated with fitness.

---

## 4. Result 3 — the false-negative rate, which is the answer that matters

A false reject is a candidate scoring above 0.26 that nevertheless drives. Ground truth is
one `--gens 1` closed-loop evaluation per candidate, 85 of them.

**6 of the 7 rejects drive at ≥75% of the recording champion. 2 of the 7 drive better
than it.**

| bar for "drives well" | false rejects, out of 7 rejects / 85 candidates |
|---|---|
| ≥50% of champion (1 209 m) | **6** |
| ≥75% (1 814 m) | **6** |
| ≥90% (2 176 m) | 4 |
| ≥100% (2 418 m) | 2 |

The two that beat the champion outright:

| candidate | score @20000 | closed-loop median | ≥1 lap | vs champion |
|---|---:|---:|---:|---:|
| `noi050_s2` | 0.4287 (65% above threshold) | 2 986 m | 36/48 | **123%** |
| `gen30` | 0.2618 | 3 906 m | 41/48 | **162%** |

`gen30` is the one that should end the argument. It is not a synthetic perturbation — it
is a champion this lineage's own selection produced, `ereg-m4-m4-s77/evolve_best_gen0030.bin`.
The screen rejects it, and it drives 62% further than the brain the recording was made
from, with 41 of 48 beings completing a lap against the champion's 22.

The only candidate the screen rejects that genuinely cannot drive is `noi200_s1` (σ = 2.0
noise, 71 m, 0/48 laps, 2% mortality) — a brain that has been destroyed rather than
changed.

In the other direction, the screen's *keeps* are almost uninformative: 78 of 85 candidates
pass, and their closed-loop medians run from 1 723 m (`gen59`, 71% of champion) to 4 949 m
(`gen06`, 205%). Two keeps drive below 75% of the champion.

**The 0.26 threshold is UNSAFE for pruning.** Used as the spec proposes, it would have
discarded `gen30` and `noi050_s2` — both better than the brain it is protecting — while
keeping everything else regardless of quality. Over a long search that is not a slow
funnel, it is a mechanism that removes good brains at a rate of roughly 6 per 7 rejections
while providing no ordering among survivors.

### Full data

#### Set A — 25 degraded copies of the recording champion

Columns 250…20000 are the screen's error/tick at that replay window.

| candidate | ‖ΔW‖/‖W‖ | 250 | 500 | 1000 | 2000 | 5000 | 10000 | 20000 | closed loop med (m) | ≥1 lap | % of champion | verdict @0.26 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mix100` | 0.000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2418 | 22/48 | 100% | keep |
| `mix098` | 0.020 | 0.0106 | 0.0176 | 0.0232 | 0.0178 | 0.0128 | 0.0122 | 0.0093 | 2566 | 25/48 | 106% | keep |
| `amp105` | 0.050 | 0.0267 | 0.0378 | 0.0434 | 0.0338 | 0.0266 | 0.0248 | 0.0189 | 2467 | 23/48 | 102% | keep |
| `mix095` | 0.050 | 0.0246 | 0.0355 | 0.0436 | 0.0351 | 0.0263 | 0.0259 | 0.0196 | 2551 | 24/48 | 106% | keep |
| `noi002_s1` | 0.020 | 0.0296 | 0.0413 | 0.0490 | 0.0379 | 0.0286 | 0.0270 | 0.0206 | 2632 | 26/48 | 109% | keep |
| `mix090` | 0.100 | 0.0468 | 0.0607 | 0.0775 | 0.0586 | 0.0459 | 0.0465 | 0.0354 | 2758 | 30/48 | 114% | keep |
| `amp115` | 0.150 | 0.0543 | 0.0749 | 0.0952 | 0.0724 | 0.0596 | 0.0549 | 0.0440 | 2466 | 22/48 | 102% | keep |
| `mix085` | 0.150 | 0.0645 | 0.0796 | 0.0992 | 0.0744 | 0.0623 | 0.0657 | 0.0503 | 2860 | 30/48 | 118% | keep |
| `noi005_s1` | 0.050 | 0.0815 | 0.1083 | 0.1324 | 0.0990 | 0.0796 | 0.0743 | 0.0599 | 2552 | 24/48 | 106% | keep |
| `mix080` | 0.200 | 0.0834 | 0.1005 | 0.1201 | 0.0916 | 0.0792 | 0.0856 | 0.0651 | 2855 | 32/48 | 118% | keep |
| `amp130` | 0.300 | 0.0808 | 0.0974 | 0.1242 | 0.0936 | 0.0832 | 0.0805 | 0.0654 | 2327 | 22/48 | 96% | keep |
| `mix070` | 0.300 | 0.1279 | 0.1496 | 0.1571 | 0.1214 | 0.1123 | 0.1233 | 0.0941 | 2819 | 30/48 | 117% | keep |
| `noi010_s2` | 0.100 | 0.1697 | 0.2086 | 0.2392 | 0.1707 | 0.1378 | 0.1196 | 0.1006 | 2361 | 22/48 | 98% | keep |
| `noi010_s3` | 0.100 | 0.1168 | 0.1533 | 0.2151 | 0.1660 | 0.1246 | 0.1294 | 0.1091 | 2147 | 16/48 | 89% | keep |
| `noi010_s1` | 0.100 | 0.1229 | 0.1482 | 0.1885 | 0.1506 | 0.1344 | 0.1352 | 0.1137 | 2558 | 23/48 | 106% | keep |
| `mix060` | 0.400 | 0.1723 | 0.2026 | 0.2118 | 0.1630 | 0.1522 | 0.1610 | 0.1236 | 2934 | 30/48 | 121% | keep |
| `noi020_s1` | 0.200 | 0.2455 | 0.2822 | 0.3108 | 0.2221 | 0.1775 | 0.1582 | 0.1305 | 2828 | 29/48 | 117% | keep |
| `mix050` | 0.500 | 0.2209 | 0.2721 | 0.2853 | 0.2246 | 0.2033 | 0.2087 | 0.1622 | 2452 | 23/48 | 101% | keep |
| `noi050_s1` | 0.499 | 0.1960 | 0.2312 | 0.2948 | 0.2294 | 0.2208 | 0.2369 | 0.1874 | 1754 | 8/48 | 73% | keep |
| `mix030` | 0.700 | 0.3012 | 0.3569 | 0.3931 | 0.3207 | 0.2872 | 0.2900 | 0.2359 | 2029 | 17/48 | 84% | keep |
| `noi035_s1` | 0.350 | 0.4143 | 0.5003 | 0.5887 | 0.4413 | 0.3607 | 0.3174 | 0.2633 | 2192 | 16/48 | 91% | REJECT |
| `noi050_s3` | 0.494 | 0.2445 | 0.3471 | 0.4668 | 0.3807 | 0.3638 | 0.3750 | 0.3086 | 2197 | 14/48 | 91% | REJECT |
| `noi100_s1` | 0.995 | 0.2432 | 0.2502 | 0.2723 | 0.3585 | 0.3204 | 0.2933 | 0.3272 | 1936 | 16/48 | 80% | REJECT |
| `noi050_s2` | 0.500 | 0.4493 | 0.4961 | 0.6037 | 0.5316 | 0.4820 | 0.4828 | 0.4287 | 2986 | 36/48 | 123% | REJECT |
| `noi200_s1` | 1.987 | 1.0175 | 1.0465 | 1.0604 | 0.9726 | 0.9102 | 0.8858 | 0.9066 | 71 | 0/48 | 3% | REJECT |

#### Set B — the recording's own lineage, 60 champions of `ereg-m4-m4-s77`

| candidate | ‖ΔW‖/‖W‖ | 250 | 500 | 1000 | 2000 | 5000 | 10000 | 20000 | closed loop med (m) | ≥1 lap | % of champion | verdict @0.26 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `gen43` | 0.000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2418 | 22/48 | 100% | keep |
| `gen42` | 0.054 | 0.0623 | 0.0695 | 0.0734 | 0.0574 | 0.0456 | 0.0460 | 0.0364 | 2401 | 17/48 | 99% | keep |
| `gen38` | 0.122 | 0.1431 | 0.1595 | 0.1840 | 0.1436 | 0.1142 | 0.1047 | 0.0877 | 2851 | 32/48 | 118% | keep |
| `gen44` | 0.160 | 0.1925 | 0.1962 | 0.1975 | 0.1537 | 0.1343 | 0.1287 | 0.1051 | 2363 | 21/48 | 98% | keep |
| `gen35` | 0.154 | 0.1767 | 0.2086 | 0.2626 | 0.1940 | 0.1534 | 0.1397 | 0.1159 | 2372 | 22/48 | 98% | keep |
| `gen53` | 0.173 | 0.0866 | 0.1354 | 0.1729 | 0.1544 | 0.1501 | 0.1594 | 0.1300 | 2397 | 18/48 | 99% | keep |
| `gen56` | 0.190 | 0.1111 | 0.1468 | 0.1662 | 0.1577 | 0.1537 | 0.1713 | 0.1472 | 2307 | 21/48 | 95% | keep |
| `gen32` | 0.261 | 0.2704 | 0.2922 | 0.3290 | 0.2472 | 0.2024 | 0.1882 | 0.1535 | 3747 | 35/48 | 155% | keep |
| `gen31` | 0.259 | 0.2152 | 0.2534 | 0.3104 | 0.2385 | 0.1954 | 0.1814 | 0.1546 | 3236 | 36/48 | 134% | keep |
| `gen41` | 0.315 | 0.3044 | 0.3143 | 0.3298 | 0.2494 | 0.2047 | 0.1948 | 0.1562 | 2185 | 20/48 | 90% | keep |
| `gen29` | 0.244 | 0.2883 | 0.3064 | 0.3507 | 0.2580 | 0.2114 | 0.1964 | 0.1604 | 4349 | 37/48 | 180% | keep |
| `gen39` | 0.302 | 0.2754 | 0.3040 | 0.3428 | 0.2619 | 0.2159 | 0.2025 | 0.1628 | 3311 | 33/48 | 137% | keep |
| `gen34` | 0.280 | 0.3134 | 0.3259 | 0.3306 | 0.2472 | 0.2094 | 0.1977 | 0.1633 | 2691 | 27/48 | 111% | keep |
| `gen23` | 0.385 | 0.2420 | 0.2507 | 0.2615 | 0.2074 | 0.1968 | 0.2013 | 0.1661 | 3266 | 38/48 | 135% | keep |
| `gen50` | 0.144 | 0.2373 | 0.3007 | 0.3378 | 0.2602 | 0.2163 | 0.2064 | 0.1709 | 2287 | 21/48 | 95% | keep |
| `gen22` | 0.381 | 0.2677 | 0.2689 | 0.2693 | 0.2125 | 0.2042 | 0.2164 | 0.1734 | 2856 | 29/48 | 118% | keep |
| `gen36` | 0.285 | 0.2963 | 0.3205 | 0.3558 | 0.2685 | 0.2293 | 0.2100 | 0.1739 | 4224 | 41/48 | 175% | keep |
| `gen40` | 0.306 | 0.2556 | 0.2835 | 0.3415 | 0.2736 | 0.2294 | 0.2117 | 0.1784 | 4650 | 43/48 | 192% | keep |
| `gen17` | 0.306 | 0.2363 | 0.2573 | 0.3093 | 0.2419 | 0.2140 | 0.2230 | 0.1814 | 3211 | 38/48 | 133% | keep |
| `gen25` | 0.232 | 0.3024 | 0.3441 | 0.4187 | 0.3083 | 0.2480 | 0.2206 | 0.1837 | 4071 | 43/48 | 168% | keep |
| `gen05` | 0.377 | 0.2217 | 0.2495 | 0.2671 | 0.2347 | 0.2298 | 0.2333 | 0.1853 | 3282 | 36/48 | 136% | keep |
| `gen10` | 0.330 | 0.2821 | 0.2938 | 0.3409 | 0.2672 | 0.2320 | 0.2194 | 0.1863 | 3143 | 36/48 | 130% | keep |
| `gen15` | 0.353 | 0.2544 | 0.2805 | 0.3166 | 0.2473 | 0.2221 | 0.2301 | 0.1884 | 3696 | 40/48 | 153% | keep |
| `gen08` | 0.386 | 0.2877 | 0.2843 | 0.3041 | 0.2618 | 0.2313 | 0.2278 | 0.1912 | 4278 | 39/48 | 177% | keep |
| `gen06` | 0.381 | 0.2039 | 0.2341 | 0.2627 | 0.2386 | 0.2281 | 0.2285 | 0.1954 | 4949 | 47/48 | 205% | keep |
| `gen47` | 0.331 | 0.3298 | 0.3619 | 0.3985 | 0.3076 | 0.2579 | 0.2462 | 0.1964 | 2763 | 28/48 | 114% | keep |
| `gen46` | 0.337 | 0.3584 | 0.3970 | 0.4389 | 0.3373 | 0.2660 | 0.2425 | 0.1991 | 2492 | 23/48 | 103% | keep |
| `gen03` | 0.369 | 0.2017 | 0.2437 | 0.2853 | 0.2585 | 0.2408 | 0.2406 | 0.2010 | 3930 | 42/48 | 163% | keep |
| `gen21` | 0.360 | 0.4438 | 0.4285 | 0.4365 | 0.3258 | 0.2624 | 0.2543 | 0.2014 | 2717 | 27/48 | 112% | keep |
| `gen33` | 0.266 | 0.3329 | 0.3611 | 0.4154 | 0.3099 | 0.2577 | 0.2431 | 0.2030 | 4314 | 42/48 | 178% | keep |
| `gen12` | 0.320 | 0.3015 | 0.3385 | 0.4004 | 0.3093 | 0.2620 | 0.2408 | 0.2034 | 3476 | 40/48 | 144% | keep |
| `gen13` | 0.404 | 0.3523 | 0.3376 | 0.3184 | 0.2662 | 0.2505 | 0.2468 | 0.2072 | 2623 | 30/48 | 108% | keep |
| `gen57` | 0.378 | 0.3931 | 0.4169 | 0.4267 | 0.3193 | 0.2588 | 0.2597 | 0.2090 | 2101 | 16/48 | 87% | keep |
| `gen19` | 0.351 | 0.4232 | 0.4281 | 0.4530 | 0.3456 | 0.2763 | 0.2642 | 0.2126 | 3875 | 41/48 | 160% | keep |
| `gen07` | 0.385 | 0.3615 | 0.3417 | 0.3263 | 0.2791 | 0.2620 | 0.2600 | 0.2135 | 2762 | 29/48 | 114% | keep |
| `gen04` | 0.349 | 0.2506 | 0.2782 | 0.3252 | 0.2804 | 0.2596 | 0.2344 | 0.2147 | 4816 | 47/48 | 199% | keep |
| `gen45` | 0.330 | 0.3498 | 0.4025 | 0.4627 | 0.3459 | 0.2878 | 0.2560 | 0.2149 | 2693 | 25/48 | 111% | keep |
| `gen49` | 0.347 | 0.3435 | 0.3952 | 0.4510 | 0.3412 | 0.2808 | 0.2576 | 0.2149 | 2245 | 19/48 | 93% | keep |
| `gen24` | 0.365 | 0.2228 | 0.2510 | 0.2999 | 0.2559 | 0.2526 | 0.2546 | 0.2152 | 3658 | 39/48 | 151% | keep |
| `gen01` | 0.353 | 0.2141 | 0.2519 | 0.2966 | 0.2700 | 0.2533 | 0.2444 | 0.2167 | 4767 | 43/48 | 197% | keep |
| `gen20` | 0.422 | 0.3435 | 0.3586 | 0.3596 | 0.2945 | 0.2752 | 0.2670 | 0.2193 | 2368 | 25/48 | 98% | keep |
| `gen00` | 0.356 | 0.1938 | 0.2470 | 0.3080 | 0.2790 | 0.2643 | 0.2647 | 0.2240 | 4317 | 40/48 | 179% | keep |
| `gen55` | 0.372 | 0.3413 | 0.4032 | 0.4618 | 0.3434 | 0.2830 | 0.2720 | 0.2248 | 2361 | 18/48 | 98% | keep |
| `gen37` | 0.299 | 0.4018 | 0.4283 | 0.4338 | 0.3442 | 0.2814 | 0.2477 | 0.2258 | 2631 | 26/48 | 109% | keep |
| `gen59` | 0.377 | 0.5479 | 0.5189 | 0.5078 | 0.3593 | 0.2997 | 0.2655 | 0.2262 | 1723 | 12/48 | 71% | keep |
| `gen02` | 0.364 | 0.1977 | 0.2745 | 0.3371 | 0.3005 | 0.2750 | 0.2753 | 0.2279 | 1935 | 23/48 | 80% | keep |
| `gen18` | 0.419 | 0.3963 | 0.4036 | 0.3917 | 0.3136 | 0.2861 | 0.2765 | 0.2279 | 2198 | 18/48 | 91% | keep |
| `gen26` | 0.379 | 0.2708 | 0.2984 | 0.3440 | 0.2808 | 0.2760 | 0.2733 | 0.2304 | 3745 | 41/48 | 155% | keep |
| `gen28` | 0.388 | 0.5611 | 0.5532 | 0.5280 | 0.4013 | 0.3184 | 0.2997 | 0.2343 | 3109 | 35/48 | 129% | keep |
| `gen51` | 0.355 | 0.3597 | 0.4099 | 0.4743 | 0.3614 | 0.3047 | 0.2688 | 0.2346 | 3015 | 33/48 | 125% | keep |
| `gen52` | 0.355 | 0.3597 | 0.4099 | 0.4743 | 0.3614 | 0.3047 | 0.2688 | 0.2346 | 3015 | 33/48 | 125% | keep |
| `gen48` | 0.343 | 0.3522 | 0.4096 | 0.4739 | 0.3656 | 0.3094 | 0.2750 | 0.2359 | 2645 | 27/48 | 109% | keep |
| `gen58` | 0.384 | 0.3655 | 0.4061 | 0.4434 | 0.3433 | 0.2842 | 0.2718 | 0.2397 | 2635 | 26/48 | 109% | keep |
| `gen54` | 0.363 | 0.3612 | 0.4083 | 0.4566 | 0.3592 | 0.3105 | 0.2840 | 0.2418 | 4394 | 39/48 | 182% | keep |
| `gen27` | 0.383 | 0.3319 | 0.3523 | 0.3911 | 0.3096 | 0.2959 | 0.2926 | 0.2441 | 3659 | 40/48 | 151% | keep |
| `gen16` | 0.415 | 0.5276 | 0.5100 | 0.4582 | 0.3563 | 0.3111 | 0.2907 | 0.2457 | 2134 | 15/48 | 88% | keep |
| `gen09` | 0.392 | 0.4727 | 0.4471 | 0.4231 | 0.3494 | 0.3214 | 0.3169 | 0.2541 | 1986 | 15/48 | 82% | keep |
| `gen14` | 0.409 | 0.4973 | 0.4760 | 0.4368 | 0.3574 | 0.3155 | 0.2998 | 0.2581 | 2308 | 23/48 | 95% | keep |
| `gen30` | 0.396 | 0.6159 | 0.6124 | 0.5870 | 0.4434 | 0.3518 | 0.3314 | 0.2618 | 3906 | 41/48 | 162% | REJECT |
| `gen11` | 0.397 | 0.5251 | 0.5179 | 0.5051 | 0.4075 | 0.3576 | 0.3439 | 0.2753 | 2083 | 15/48 | 86% | REJECT |

(`gen51` and `gen52` are byte-identical — `cmp` confirms; the elite was carried through
that generation unchanged — which is why their rows agree exactly. Identical numbers with
a found cause, per `CLAUDE.md` rule 2. Set B is 60 files, 59 distinct genomes.)

---

## 5. Result 4 — cost per candidate, and the correction to the spec

All measured today on the RX 6750 XT, same recording, `env -u DISPLAY`.

**Closed loop**, `/usr/bin/time` around the exact ground-truth command, two repeats
(identical result, 2 418 m, both times — the simulator is deterministic; only wall time
moves):

```
wall 51.07 s      wall 46.37 s      # 48 beings x 20 000 ticks
```

→ **0.97-1.06 s per genome.** The `~18 s` in the spec was never measured and is retired.

**Screen**, from the tool's own per-batch line:

| | 20 000 ticks | 2 000 ticks |
|---|---:|---:|
| `--batch 1` | 8 404 ms/genome | 896 ms/genome |
| `--batch 48` | **208.6 ms/genome** | **21.8 ms/genome** |
| `--batch 60` | 184.2 ms/genome | 21.1 ms/genome |

| configuration | vs 1.0 s/genome closed loop |
|---|---|
| full window, unbatched | **8.4× more expensive** |
| full window, batch 48 | **4.8× cheaper** |
| 2 000-tick window, batch 48 | **46× cheaper** |

Batching is the entire win: at `--batch 1` the screen is *slower* than simulating the
whole world, because both sides are launch-latency bound, not arithmetic bound. The
FLOP count behind the spec's estimate measured the one thing neither side is limited by.

**Correction applied to `docs/superpowers/specs/2026-08-26-gpu-resident-scenarios-design.md`:**
"three orders of magnitude per candidate" is wrong by between two and three orders of
magnitude itself. The measured best case is **46×**, and only with the window shortened
to 2 000 ticks and 48 candidates batched per dispatch.

---

## 6. What the screen may be used for

Rejected: **pruning at 0.26**. 6 of 7 rejects drive; 2 beat the champion.

Rejected: **ranking**. ρ = −0.05 against closed-loop quality over 85 candidates.

What survives is much narrower. The single thing the screen did reliably is separate a
*destroyed* brain from a functioning one: `noi200_s1` scored 0.9066 while every candidate
that could still drive scored ≤0.4287 — a factor of 2.1 with nothing in between. As a
**tripwire for catastrophic divergence**, with a threshold around **0.60** (conservatively
above the highest-scoring brain that still drove, and well below the destroyed one), it
costs 21.8 ms per candidate and, on this evidence, produces no false rejects. That is a
much smaller claim than the funnel the spec describes, and it is the only one the data
supports.

**Limits of this calibration, stated so the next reader does not over-generalise it.**
One recording, one champion (`ereg-m4-m4-s77` gen 43), one track for the ground truth
(8005, with an 11-candidate control on 8200), one brain architecture, `out_beta = 0.0`.
The false-reject rate is measured against *this* recording; a recording made from a
stronger champion — one that is not outdriven by 56 of 85 nearby candidates — would
plausibly give a less hostile picture. What that cannot change is the mechanism: the score
is a distance from one reference point, and nothing in it knows which direction is better.
The two candidate sets disagree on how *strongly* the correlation is negative (−0.36 vs
−0.10) but not on its sign, at any window.

Two things would have to change before the screen could prune:

1. **A reference that is not one champion.** Scoring against a single recording measures
   similarity to one point. A screen that scored against a *set* of recordings, or against
   the recorded outcome rather than the recorded commands, would at least be measuring
   something whose sign is right.
2. **A threshold derived from this kind of measurement**, not transplanted. The 0.26/0.18
   pair in `docs/DESTILABILIDADE.md` was measured for a *distilled pilot* — a closed-form
   controller replacing the brain entirely — not for perturbed genomes of the same
   architecture. The spec adopted it across that gap without a check. This document is
   that check, and it fails.

---

## 7. Reproducing

```bash
D=/media/llm-data/scratch/wave                 # recording (Task 5)
C=/media/llm-data/scratch/screen-calib         # candidates + ground truth

# screen, all windows
for T in 250 500 1000 2000 5000 10000 20000; do
  env -u DISPLAY ./target/release/screen --wave $D/wave.f32 \
      --genomes $C/cand --ticks $T --batch 25
done

# closed-loop ground truth, one --gens 1 evaluation per candidate
env -u DISPLAY ./target/release/gpu_evolve --retomar --out $O --gens 1 --seres 48 \
  --ticks 20000 --width 28 --out-beta 0.0 --mascara-regioes --neuromod-regioes \
  --retomar-mix 1.00 --pistas 8005,8200,9001 --seed 77
```

Raw logs: `$C/screen_windows.txt`, `$C/screen_lineage.txt`, `$C/cl*/`,
`$C/cl.log`, `$C/cl_lin.log`, `$C/cl8200.log`. Wall time for the whole battery:
85 + 11 closed-loop evaluations ≈ 85 min, plus ~4 min of screening.

