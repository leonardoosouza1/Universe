# Request to Astra: L1, L2 and head intents in `observation.jsonl` (15/09/2026)

Context: the laboratory is building one application over `lab.db` (`LLM/lab-ui/PRD.md`)
whose micro view answers *why did the being brake here* by walking the recorded brain
stages backwards from the pedal to the retina, weighting each connection by
`weight × activation` at that tick. The walk is only as complete as what the recorder
writes, and today it stops one stage after the eye.

## What `--driving-diagnostics` records today, per sample, for the observed being

From `src/bin/gpu_evolve/driving_diagnostics.rs` (`before_brain` / `after_physics`):

| block | size | stage |
|---|---|---|
| `retina` | 14 × 126 **u32** (material in the low byte, distance in mm in the high 24 bits — `driving.meta.json:retina_encoding`) | eye, pre-physics |
| `wave` | 192 i8 | sensory wave entering the brain |
| `fast_state` | 96 f32 | `fast_state_stage: "sensory_cfc_pre_l1"` |
| `body.raw_motor` | 3 f32 | head output before command assembly |
| `body.applied_motor` | 3 f32 | throttle, steer, brake after `map_native_action` |
| `body.slow` | 3 f32 | slow brain output |

Missing between `fast_state` and `raw_motor`: everything L1 and L2 compute, and the 96
`final_intent` (48 L2 + 48 L1 bypass) that the KAN head reads. Without them the trail can
only say "the CfC state was X and the pedal was Y" with the weights in between — it cannot
show which L1/L2 cells carried the decision, which is the picture the owner asked for.

## The ask

Extend the sample readback (`SAMPLE_BYTES`, `driving_diagnostics.rs:21-24`) with three
blocks, in kernel execution order, and name them in `driving.meta.json` exactly as the
kernel names them:

1. **`l1`** — the 96 L1 activations after its nonlinearity, as the kernel hands them to
   the next stage (f32, or i8 with the scale written to meta if the stage is i8).
2. **`l2`** — the 48 L2 activations, same rule.
3. **`intent`** — the 96 `final_intent` values the head reads (48 L2 + 48 L1 bypass), so
   the bypass is visible as a separate path in the trail.

And one line in `driving.meta.json`: `"brain_flow": [...]` listing every recorded stage in
the order the kernel runs them, taken from the code, so the front never has to guess the
architecture from a document.

## Contract, so the consumer can refuse bad data instead of drawing it

Every block the recorder writes must be declared in `driving.meta.json` with `dtype`,
`shape`, `unit`, `scale` (for quantised stages), `stage` (the kernel's own name) and
`encoding` (as `retina_encoding` already is). The trace producer validates each sample
against that declaration and refuses the run on mismatch; `lab-ui up` runs the same check
and refuses to start. The retina dtype error in the first draft of this document is the
case in point: had the producer trusted the document, it would have drawn garbage
confidently. Please treat the meta as the contract and the document as commentary.

## Recording profiles and cost (from your review, adopted)

Profiles are a launcher field: `summary` (replay + events, no neural blocks), `champion`
(one being, every 2–4 ticks), `event` (one being, full cadence only in windows around
impact/brake/death/lap), `full` (every tick, asked for explicitly). Collection must not
add a `device.poll(Wait)` or a GPU→CPU map per tick: a GPU ring buffer with asynchronous
batched copies, drained every N ticks, is the shape you proposed and the one the front
assumes. State the measured per-tick cost of each profile at 48 beings.

## What must hold

- **Off is bit-identical.** With `--driving-diagnostics` absent, nothing changes: the
  existing parity tests (`fly_map_parity_*`, `gpu_validate`) stay green and a 2-generation
  smoke on seed 7 reproduces today's champion genomes byte for byte
  (`/media/llm-data/scratch/costgate-smoke2/b_gate_shut/evolve_best_gen*.bin` are the
  reference for `--wall-cost 0 --impact-cost 0`).
- **The readback is what the kernel computed.** A CPU-side recomputation of L1 from the
  recorded `fast_state` and the genome weights must match the recorded `l1` within the
  stage's quantisation (i8 diff ≤ 1 if i8, 1e-5 if f32) on at least 100 samples of a real
  run. That test is the acceptance; a plausible-looking stream is not.
- **Cost.** State the added bytes per sample and the measured change in tick wall time
  with `--kernel-profile 250` on 48 beings; today the observed-being readback is one extra
  `SAMPLE_BYTES` copy per `sample_every` ticks and should stay one copy.
- **Cadence stays a flag.** `--vision-every N` already sets `sample_every`; the new blocks
  follow it.

## What the consumer will do with it

`neuromod-lab/src/lab_api/pack/builder.py` reads `observation.jsonl` and writes one
`stage_<name>.f32` per recorded stage plus `meta.json:stages` in the order of `brain_flow`.
It is already written to tolerate absence: a stage not in the file is not in the pack, and
the front draws a dashed placeholder. So there is no ordering dependency — the front ships
first and lights the two stages up when your change lands.

## Not asked

Recording every being (48× the volume); changing the sample cadence default; touching
`.replay`.
