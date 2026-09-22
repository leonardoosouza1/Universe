# E1a validation — 2026-09-20

Scope: bounded engine/brake genes co-selected with the existing brain. Baseline
source HEAD: `ad944b6c`. GPU: AMD Radeon RX 6750 XT (RADV NAVI22).
Artifacts: `/media/llm-data/universe-runs/body-evolution-20260920/` (14 MB).

## Checks completed

| Check | Command / evidence | Result |
|---|---|---|
| Library regression | `cargo test --release --lib` | 507 passed, 17 ignored |
| Evolution executable | `cargo test --release --bin gpu_evolve` | 121 passed |
| Real GPU body response | `cargo test --release --lib body_capacities_reach_gpu_physics -- --ignored --nocapture` with driver environment below | passed; explicitly executes one of the ignored tests |
| Builds | `cargo build --release --bin gpu_evolve --bin gpu_validate --bin genoma_dump --bin genoma_view --bin replay_view` | passed |
| Existing CPU/GPU physics gate | `./target/release/gpu_validate --fisica`, before and after | both passed, logs identical |
| Language, this change | `BASE_REF=HEAD bash scripts/check_language.sh` | passed |
| Whitespace | `git diff --check` | passed |

The ordinary language gate against `origin/main` fails on existing branch-wide
Portuguese text (669 changed paths). This change does not translate that history.
The physics gate is tolerance-based: maximum 100-step position error remains
0.141869 m and velocity error 0.133116 m/s. It is not CPU/GPU bit identity.

The four-body GPU test uses equal initial velocity and commands. Engine extremes
produce drive torques 1904.4838 and 3174.1394 Nm; brake extremes leave wheel speeds
28.42803 and 27.17803 rad/s after one step. This isolates capacity transmission to
GPU mechanics from whether a brain chooses to brake.

## Closed-loop runs

Every GPU command uses:

```bash
env -u DISPLAY VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
  MESA_VK_VERSION_OVERRIDE=1.3 ./target/release/gpu_evolve ...
```

Before/after fixed-body comparison:

```text
--gens 2 --beings 8 --ticks 128 --seed 7 --same-start --fuel
--driving-diagnostics --vision-every 64 --out <fresh-directory>
```

All saved `.bin` checkpoint/observation files matched byte for byte. All legacy
fields in `observation.jsonl` matched after excluding the four newly added physical
capacity fields. This is a tested configuration, not a claim about every possible run.

Paired integration battery (four arms plus one repeat):

```text
--gens 3 --beings 8 --ticks 512 --seed <7|19> --same-start --fuel
--driving-diagnostics --vision-every 64 --vision-being 4
--track-held-out 991 --held-out-every 1 --<fixed|evolving>-body
--out <fresh-directory>
```

Directories: `validated/fixed-{7,19}`, `validated/evolving-{7,19}` and
`validated/evolving-7-repeat`. Each has 24 observation samples. Checks confirmed:

- Generation zero has neutral bodies; non-elites mutate at the next generation.
- Generation-one elites retain their body genes.
- Every sampled physical capacity agrees with its being/generation body manifest.
- Non-neutral selected genomes save as v3, not as genes appended to the Fly tail.
- The repeated evolving run has identical observations and observed-genome bytes.
- Both final populations contain eight distinct body genotypes. Seed 7's engine
  range is 606.715–635.461 Nm; mass range is 1345.201–1355.770 kg.

Final training median progress reported by these short runs:

| Seed | Fixed | Evolving |
|---|---:|---:|
| 7 | 7 m | 7 m |
| 19 | 24 m | 22 m |

There is **no evidence of a performance improvement** here. Eight seconds per
generation with untrained brains and two seeds is an integration smoke, not a
learning study. The small distances also make held-out/train ratios unstable;
do not interpret a large ratio when its denominator is near zero as a discovery.

## Persistence and controls

Copied `validated/evolving-7/observed_genome_gen0001.bin` into fresh directories as
`evolve_best_gen0001.bin`, then executed:

```text
--resume --resume-mix 1 --gens 1 --beings 4 --ticks 128 --same-start
--seed 7 --fuel --<fixed|evolving>-body --driving-diagnostics --vision-every 64
--out <fresh-directory>
```

`validated/resume-evolving` and `validated/resume-fixed` preserve the complete
inherited DNA byte for byte in the observed slot. Actual capacities match the
inherited body in evolving mode and the historical body in fixed mode.
`--evolving-body --no-fuel` exits with code 2 before GPU allocation.
`genoma_view` successfully reads the v3 checkpoint using the canonical loader.

The unit envelope test also covers legacy-neutral output, nonempty Fly tails,
complete round trips, truncated headers/payloads, unknown versions, NaN, infinity
and out-of-bound body genes. Mutation tests cover selection inheritance, elite
preservation, separate RNG, bounds over 1000 mutation rounds, and unchanged brain
fields during body mutation.

## Remaining limitations

No new UI panels, variable topology, developmental growth or lifetime plasticity
were implemented. The body manifest and observation fields make those parameters
available to consumers; they do not make existing dashboards display them.

A direct native-brain HTML export of the resumed smoke was rejected by the existing
association guard because its metadata lacks `checkpoint_sha256`. The guard was
not bypassed. Native-brain HTML export is therefore **not validated end to end**
by this battery; canonical v3 loading and neural/physical observation are validated.
Resolving that replay provenance path is separate from demonstrating body evolution.

The full E0 architecture audit remains open. E1a is experimental, default-off;
the experiment proposed in `BODY-COEVOLUTION.md` must precede any claim that body
coevolution improves survival or generalization.
