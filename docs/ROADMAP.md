# Roadmap — making Universe live up to its name

> Owner (2026-08-17): "I want a much more aggressive environment, a much more sensitive
> model, a hell of a brain. The model has to survive any universe. Everything is too easy
> for it."

This document turns that into targets that can be **measured**, because the project has
already paid for the lesson that unmeasured ambition produces plateaus that look like
progress. Every milestone below has a number, and the number has a command.

## Where we actually are (measured 2026-08-16/17)

| Fact | Evidence |
|---|---|
| The world does not select for a brain | A closed-form ridge pilot **beats** the champion (7139 m vs 6592 m, and 102% on a never-seen track). If a formula can drive, evolution is refining something a formula can express. `docs/DESTILABILIDADE.md` |
| Four difficulty mechanisms at once still kill nobody | Ladder battery: mortality 0% in 34 of 35 generations with fuel+noise+damage+weather all on (fuel was silently broken; fixed, now bites: k=1.0 halves distance) |
| The being cannot feel its own body | `engine_temp`, `oil_temp`, `fuel`, `life_left`, hidden damage are computed and never reach the 192 sensory channels |
| The being is half-blind | ch144-159 ("peripheral vision") was a bit-for-bit copy of the fovea until 2026-08-16; 6 more channels are constant the whole race |
| Neuromodulation has nothing to govern | M4 (per-region maestro) tied against control — the EXP10 line proved modulation only pays when there are slow internal body states, and the being has none it can sense |
| There is exactly one universe | One task (lap racing), one body (car), one fitness. "Survive any universe" is currently unfalsifiable |

## The two numbers that define "extraordinary"

**1. Distillability D must go DOWN.** `tools/destilabilidade.py` measures what fraction
of the champion's policy a closed formula can express. Today the formula *wins*. A world
that demands a brain is a world where D falls and stays low — this is the only honest
"difficulty meter" we have, and it becomes ladder telemetry. Target: the best
closed-form pilot drops below 70% of the champion on the training tracks themselves.

**2. Transfer must beat tabula rasa.** Intelligence is not lap count — it is how much
faster a brain evolved in world A reaches competence in world B than a random
population does. Requires a second world (foraging, spec'd from
`docs/UNIVERSO_ALEM_DO_CARRO.md` F-A). Target: champion-seeded population reaches the
marker in ≤ half the generations of the from-scratch arm, across 3 paired seeds.

Everything else — laps, medians, mortality — is instrumentation for these two.

## Milestones

### M0 — The world can hurt (in flight, workflow `universe-hostil`)
- **D7 scarce resource**: fewer energy stations than beings, finite charge, slow
  recharge. First *competitive* pressure — beings finally interact.
- **D8 perception costs energy**: channel families charge life proportional to the
  brain's use of them. The sensor budget becomes evolvable.
- **Predator**: a scripted hunter on the track, life-drain by proximity (continuous
  ramp, never instant death — sparse rewards taught nothing before). Speed is the dose.
- **Tire thermal (CPU-side) + feelable weather**: temperature accumulates with slip and
  speed, degrades grip, and finally feeds ch70 — closing the loop weather → grip →
  slip → heat → *sensation*.
- Gate for all: the dose canary. Off / weak / strong must produce different numbers in
  the battery config, or the mechanism goes back.

### M1 — The body reaches the brain
- `--interocepcao`: fuel, hidden damage, engine temp, life into the dead channels
  (ch114-117, ch98, ch124). Off by default; paired A/B mandatory.
- The M4 maestro re-test: with real slow body states, per-region neuromodulation gets
  what the EXP10 line says it needs. Prediction registered there: M4 stops tying.

### M2 — The brain earns its size
- `cortex_l3` (1152 mutated weights outside the forward pass): wire it or delete it —
  decided by the objective criterion in `src/brain_capacity.rs` (participation ratio),
  not by taste. L1's PR is 6.7/96: capacity is not the bottleneck *today*, so the
  probable verdict is delete now, re-add when PR crosses the recorded threshold.
- Sensitivity inventory switched on in measured order: real peripheral vision,
  heterogeneous τ, KAN adaptation (3072 frozen coefficients) — one A/B each.

### M3 — The second universe
- Foraging world v1: same substrate (grid, physics, 192 channels, brain), no track, food
  as scattered energy stations, fitness = energy accumulated. The transfer metric goes
  live here. Cut is deliberately minimal — the point is the *measurement*, not the biome.

### M4 — The ladder becomes the trainer
- D7/D8/predator/thermal join `escada.rs` with calibrated doses (sweep in battery
  config — smoke-config calibration already burned us once).
- Distillability measured every K generations, printed as `[exigencia]` telemetry.
- **Then, and only on the owner's explicit go: the long battery.** Training stays
  forbidden until released.

## Standing constraints

- Training runs are **forbidden** until the owner releases them. Light validation smokes
  (single short generation, dose sweeps) are allowed.
- Every new mechanism: flag off by default, cross-currency cost (life/time — never raw
  fitness), dose canary before any battery, prediction registered before any verdict.
- Heavy outputs never in the repository.
