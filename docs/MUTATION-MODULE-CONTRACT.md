# Mutation module contract

This contract adds an opt-in mutation profile for the organism-taxi service
outputs. It is a genotype operator at the generation boundary. It is not
backpropagation, an output-gradient method, or a claim that a mutation is
"safe" because it changed a particular command.

## Executed mapping

The production brain writes L2 row `j` to `final_intent[j]`. The service
interaction ABI consumes these rows:

| L2 row | `final_intent` slot | request |
| ---: | ---: | --- |
| 5 | 5 | service / refuel |
| 6 | 6 | pickup |
| 7 | 7 | dropoff |

The mapping is visible in `src/gpu/brain.wgsl` (L2 output stage) and
`src/gpu/taxi_witness.wgsl` (the service request conditions). The driving motor
rows remain 0..2; gearbox requests use rows 3..4. The service profile therefore
targets rows 5..7 only.

## API and presets

`MutationProfile::legacy()` selects the historical `evolve` operator. The public
`evolve` function still delegates to that path, so existing callers retain their
exact random draw order and outputs.

`MutationProfile::head_only(p_mutate, sigma)` validates a per-weight probability
in `[0, 1]` and a finite, non-negative Gaussian sigma. It emits a
`MutationScope::ServiceHeadOnly` profile. The parser accepts the stable presets
`legacy` and `head-only`; the latter is a reproducible starting dose of
`p_mutate=0.05`, `sigma=0.05`. That dose is a calibration starting point, not a
measured service optimum.

`evolve_with_profile` preserves tournament selection and elitism, then clones the
selected parent and mutates only the L2 weights in rows 5..7 for the head-only
scope. L1, motor and gearbox rows, slow brain, tau/sensor banks, asynchronous
metadata, hippocampus, fly map, KAN knots, body and organism genes are frozen.
Frozen blocks consume no mutation RNG draws. Selection still consumes its normal
tournament draws.

`MutationProfile::metadata()` returns additive typed provenance for a run
manifest or offspring record: scope, service rows, and the selected dose. The
genome binary layout is unchanged.

## Calibration rule for rare service outputs

Use the service-event prevalence and a target genotype effect size to choose a
study-specific dose. Record the profile, rows, generation seed, population,
elite count, tournament size, and the number of non-elite offspring. Compare a
head-only arm with the frozen-backbone control and a legacy arm under paired
evolution/environment seeds. A mutation changing a service row does not prove
that the GPU service output changed; confirm producer-to-consumer reachability in
a GPU run before making a behavioral claim.

No default path consumes extra draws. The service profile is opt-in and does not
alter the existing genome serialization or per-tick host inference/readback
path.
