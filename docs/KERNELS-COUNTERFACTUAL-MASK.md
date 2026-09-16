# Counterfactual neuron mask activated at simulation tick t (16/09/2026)

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
tick t, and compare the two recordings from t onward. Everything before the first treated
brain dispatch is identical by construction. At that dispatch the comparison is a controlled
intervention from the same prior state; later differences are the total causal effect after
the changed action has also changed the body, sensors, environment and recurrent state.

## Implemented contract

1. **A packed per-being mask over three distinct populations:** CfC 96, L1 96 and L2 48.
   The layout is eight `u32` words per being: CfC `[0..3)`, L1 `[3..6)`, L2 `[6..8)`.
   A set bit removes the emitted value and leaves that unit's internal recurrence, EMA and
   fatigue updates intact. Those states evolve under the counterfactual trajectory after the
   intervention; they do not remain equal to the baseline recording.
2. **Stage semantics:** a CfC mask zeros `w_cfc` before L1 without zeroing `buf_cfc`; an L1
   mask zeros its quantised `w_l1a` contribution to L2 and, for indices 48..95, also zeros
   the direct bypass into `final_intent`; an L2 mask zeros `final_intent[index]` after its
   activation. Pre-output diagnostic stages remain visible.
3. **World-clock activation:** `world_tick >= ablation_from_tick`. `world_tick` is the
   generation-relative simulation tick, not the CfC dispatch counter. With `--brain-skip N`,
   the effective intervention is the first neural dispatch at or after the requested tick.
4. **Flags:** `--ablate <stage>:<index>[,...] --ablate-being <population-slot>
   --ablate-from-tick <t>`. `--ablate-being` defaults to zero; the other two flags are
   required together. Stages are `cfc`, `l1` and `l2`; invalid indices fail before GPU work.
5. `driving.meta.json` and replay metadata record the selected being, canonical target list,
   requested world tick and effective-tick semantics.

Per-connection ablation is intentionally separate. Testing an override list inside every
L1/L2 accumulation would add a branch to 18,432 + 4,608 products per being per dispatch. It
needs its own kernel design and cost measurement rather than being hidden as a stretch goal.

## Implementation (16/09/2026)

- `src/gpu/brain/ablation.rs`: validated parser and eight-word mask construction.
- `src/gpu/brain.wgsl`: binding 14 and output masks at CfC, L1/L1-bypass and L2.
- `src/gpu/brain/mod.rs`: resident mask buffer, unchanged 48-byte uniform layout by replacing
  former padding with `world_tick`, `ablation_from_tick` and `control_flags`. Bit 0 enables
  ablation; the remaining bits independently select the observation writer.
- `gpu_evolve`: CLI validation, selected-being upload and diagnostic/replay metadata.
- Off by default. No mask read can affect an output while bit 0 of `control_flags` is zero.

## Acceptance

- **Off is bit-identical**: without `--ablate`, the two dense runs above are reproduced byte
  for byte.
- **Before t is bit-identical**: with `--ablate l1:17 --ablate-being 0
  --ablate-from-tick 6000`, every
  `observation.jsonl` line and `driving.csv` row with tick < 6000 equals the unablated run.
- **After t differs where it should:** `l1_matmul` and `l1_effective` remain observable;
  `l1_to_l2[17]` becomes zero on treated dispatches. For L1 indices 48..95 the bypass output
  is also zero. `cfc_state` remains observable while the future `cfc_to_l1` emission is zero.
- **Positive controls:** L2 indices 0, 1 and 2 test the direct throttle, steer and brake
  outputs. L1 candidates are chosen by their recorded absolute contribution to those rows.
  Include one silent/unused neuron as a negative control; no motor change there is expected.
- Only the selected population slot is treated. Every other slot remains byte-identical until
  shared-world interactions transmit an effect; record whether social sensors or body-body
  occlusion are enabled.
- `cargo test --release` green; cost of one ablated run reported (expected ≈ the 84 s of the
  unablated one).

## Not asked

Restoring state at arbitrary ticks; a CPU forward; changing any stage's unmasked arithmetic;
per-connection masks.

## Measured validation (16/09/2026)

On the real RADV/NAVI22 path, ablating L2 outputs 0, 1 and 2 from world tick 128 produced
byte-identical observation rows through tick 127 and first diverged at tick 128. The three
raw motor intents were exactly zero from that dispatch onward. Scheduling the same mask for
tick 1000 in a 256-tick run produced byte-identical `driving.csv`, population JSON, replay
and champion genome relative to the unablated run.

The other two stages were also exercised on the real GPU from tick 64. `l1:17` preserved
all 63 preceding samples, forced `l1_to_l2[17]` to zero in all 65 treated samples and changed
the downstream intent. `cfc:17` preserved the recurrent `cfc_state` at the intervention tick
while changing the L1 matrix result at that same dispatch, confirming that the mask removes
the emission rather than erasing internal state. In a two-being run, treating slot 0 while
recording slot 1 left all 128 samples of slot 1 identical to baseline.

After adding the L1/L2 observation binding, the unablated 256-tick run still reproduced the
pre-observation `driving.csv`, population JSON, replay and champion genome byte for byte.
The combined evidence is under `/media/llm-data/scratch/brain-observation-current-*`.
The stage and population-isolation controls are under
`/media/llm-data/scratch/final-mask-*-20260916/`.
