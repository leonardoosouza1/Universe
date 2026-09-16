# Request to Astra: a neuron mask that switches on at tick t (16/09/2026)

## What it is for

The lab-ui micro view has a **Counterfactual** panel: pick a neuron (or a connection) at
tick t and show how the brake, throttle and steering would have differed without it. That
is the only causal question the application can answer; the contribution panel
(`a_i · w_ji`) is explicitly local and not causal.

## Why not a CPU twin — measured

Slice S2 built `brain_step`, a CPU reference forward from the recorded wave and CfC state to
the raw motor intent, reusing the CPU types `gpu_validate` checks. On a dense recording where
the recorded being is proven to own the genome (sha256, `--resume --resume-mix 1`,
`champ_nocost_s77_dense_t8005`, 3 000 samples) it misses the recorded intent by **0.40 /
0.29 / 0.31 mean abs** (throttle / steer / brake), max ~1.0. The cause is structural: the
P1.7 per-neuron mechanisms are active in the champion (`neuron_period_l1` 96/96 non-default,
`neuron_alpha_l1` 41/96, `neuron_threshold_l1` 39/96) and their state (`buf_cfc`,
`buf_fatigue`, `buf_neuron_mem`) lives across ticks on the GPU and is never downloaded. A
CPU twin would chase the kernel forever. Details:
`docs/LAB-UI-S2-GENOME-EXPORT.md` on branch `ui/s2-pack`.

## Why a GPU replay works — measured

Two independent runs of the same frozen champion with identical arguments
(`champ_nocost_s77_dense_t8005` and `_rep`: `--resume --resume-mix 1 --gens 1 --ticks 12000
--beings 8 --seed 77 --same-start --driving-diagnostics --vision-every 4`, frozen binary
`gpu_evolve-f5d203b`) produced **byte-identical** `driving.csv`, `population_gen0000.json`,
`observation.jsonl` and `gen0000.replay` (md5 16/09 10:14). The run costs **84 s**. So an
exact counterfactual needs no state restore: replay from tick 0 with the mask inactive until
tick t, and compare the two recordings from t onward. Everything before t is identical by
construction; everything after t is the counterfactual.

## The ask

1. **A per-being, per-neuron mask** over the recurrent and dense stages (L1 96, L2 48; CfC
   if it is a separate population) — 144 f32 per being as S2 estimated, or bits — applied as
   a multiplicative zero on the neuron's OUTPUT, so its downstream contribution vanishes while
   its own state keeps evolving as recorded (say explicitly if the kernel makes that
   distinction impossible and which one you chose).
2. **`mask_active_from_tick`** in the brain uniforms: before that tick the mask is ignored.
3. **Flags**: `--ablate <stage>:<index>[,<stage>:<index>...] --ablate-from-tick <t>`, off by
   default, registered in `cli_flags.rs`. Optional stretch: connections (`<stage>:<row>:<col>`)
   if the weight buffers make it cheap.
4. `driving.meta.json` records the ablation spec so the trace producer can put it in the
   manifest.

## Acceptance

- **Off is bit-identical**: without `--ablate`, the two dense runs above are reproduced byte
  for byte.
- **Before t is bit-identical**: with `--ablate l1:17 --ablate-from-tick 6000`, every
  `observation.jsonl` line and `driving.csv` row with tick < 6000 equals the unablated run.
- **After t differs where it should**: at least the recorded `l1` (once
  `KERNELS-OBSERVATION-L1L2.md` lands) shows index 17 at zero from tick 6000; before that
  lands, the motor chain must differ at some tick ≥ 6000 for at least one of 5 indices tried,
  and the doc must say which indices changed nothing (a neuron that changes nothing is a
  result, not a bug).
- `cargo test --release` green; cost of one ablated run reported (expected ≈ the 84 s of the
  unablated one).

## Not asked

Restoring state at arbitrary ticks; a CPU forward; changing any stage's arithmetic.
