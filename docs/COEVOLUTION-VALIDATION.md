# Coevolution validation — 2026-09-20

**Integration update, 2026-09-21:** the separate-exporter and per-life lab-ui gaps
were addressed in [INTEGRATION-20260921.md](INTEGRATION-20260921.md). The historical
experimental results and limitations below remain unchanged.

**Decision:** bounded experimental mechanisms are implemented, but the combined
configuration is not approved as a better learner or as the default. GPU numerical
checks pass; short evolutionary results include regressions. No long training,
CPU numerical oracle, `cargo test`, or mixed CPU/GPU validator was run in this pass.

The hardware reported by every accepted simulation was **AMD Radeon RX 6750 XT
(RADV NAVI22)**. Host work was restricted to necessary compilation, orchestration,
artifact inspection/export and existing simulation bookkeeping. Compilation used
one job, one CPU core and low priority; GPU batches ran sequentially.

## Reproducible evidence

All paths below are relative to `Universe/`:

- `artifacts/coevolution-20260920-evidence/summary.json`: inspected metrics,
  inheritance/conservation checks and current gpu_evolve binary SHA-256.
- `artifacts/coevolution-20260920-evidence/lifetime-gpu-check.log`: device fixture results.
- `artifacts/coevolution-20260920-evidence/baseline-reference/`: preserved pre-change checkpoint/observation reference.
- `artifacts/coevolution-20260920-paired/commands.json`: exact accepted commands,
  seeds 7 and 19, fixed versus all mechanisms, each 3 generations x 8 beings x 1024 ticks.
- `artifacts/coevolution-20260920-ablation/commands.json`: six isolated linear-head
  arms, seed 7, same population/generation/tick budget.
- `artifacts/coevolution-20260920-final-checks/commands.json`: profile and v4 resume.
- `artifacts/coevolution-20260920-final-v2/commands.json`: disabled-path identity,
  two births and curriculum non-promotion checks. Its old `fixed-*`/`coevolved-*`
  entries are **invalid as independent-seed comparisons**, as explained below.

Commands use this real-GPU environment:

```bash
env -u DISPLAY VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json \
  MESA_VK_VERSION_OVERRIDE=1.3 RAYON_NUM_THREADS=1 \
  nice -n 15 target/release/lifetime_gpu_check
```

Build used `CARGO_BUILD_JOBS=1 RAYON_NUM_THREADS=1 nice -n 19 taskset -c 0
cargo build --release --offline` with only the required binaries. Builds succeeded
with existing/unused-code warnings; this is not a warning-free claim. Syntax check:
`node --check src/assets/replay_player/study.js`; whitespace check: `git diff --check`.
Neither executes a CPU numerical simulation.

## Device and integration checks

The lifetime GPU fixture passed: disabled-path identity with changed DNA;
finite/bounded acquired weights; nonzero P and eligibility under feedback; zero
reward leaves P zero with energy/alive positive; resetting one slot clears its
acquired state while preserving the other's; motor ablation overrides plasticity;
juvenile masking, maturation, newborn age at a late world tick, and silent pruned
connections. These are semantic GPU checks, not a CPU parity claim.

The real A/B/C physics comparison ended at **89.87114 C with stronger cooling**
versus **90.41015 C with weaker cooling**, starting from the same hot body. This
caught and corrected an earlier change limited to the unused monolithic shader.
It establishes ordering, not thermodynamic calibration.

With all new flags off, every historical observation field and checkpoint byte in
the 2-generation/4-being/128-tick smoke matched the pre-change reference. Effective
seed was 0 in both. New fields do not count against the old-field equality check.

Ecology produced births at ticks **77 and 82**, filling slots 2 and 3 from two
founders. Both newborn files use UGEN v4. For each birth, the recorded identity
`parent_before - parent_after - child_energy - loss = 0` held exactly. The selected
slot's trace changed from lineage 3 to 5 with a new observed genome; age/P began
again in that life. The isolated GPU reset check is the direct zero-state evidence.
The v4 resume smoke retained the exact 80-byte body/organ/lifetime DNA block.

Curriculum integration stayed on seed 8005 at zero completed-lap success across
three short generations. It did not promote on distance alone. Training seeds were
8005/9001; held-out was 991. Promotion/regression/archive transitions are implemented
and reviewed, but were not reached by a trained GPU population in this pass.
Extinction, deaths followed by replacement, and long-lived ancestry were also not
established by this two-birth smoke.

## Measured learning outcomes

Final-generation median **ordered progress**, calculated from population JSON
using the conventional average of the two central values for an even population:

| Population seed | Historical fixed configuration | All mechanisms + inherited motor spline |
|---|---:|---:|
| 7 | 25.10 m | 2.67 m |
| 19 | 108.68 m | 13.29 m |

The historical console uses a different median convention; do not silently equate
its printed integer with this table. These are short random-population smokes,
not trained champions. No useful held-out generalization was established. Logs
report zero held-out median at generation 2 and no held-out survivors.

The combined arm also activates inherited motor splines whereas the historical
control shares its spline. It therefore cannot isolate development/plasticity.
To remove this factor, the following six arms all used `--motor-head linear`:

| Seed 7 arm | Median ordered progress | Best ordered progress |
|---|---:|---:|
| Fixed | 35.927 m | 45.092 m |
| Body only | 35.223 m | 63.847 m |
| Learning rules + plasticity only | 35.993 m | 48.718 m |
| Development only | 30.561 m | 85.825 m |
| Topology only | 34.297 m | 41.690 m |
| Combined | 31.120 m | 77.756 m |

Plasticity is numerically active, but the near-identical median does not demonstrate
learning benefit. Development reduced the median here while raising the best
individual: this suggests heterogeneity/capacity restriction worth testing, not a
proven cause or robust improvement. One-seed ablations cannot establish statistical
significance. Keep the mechanisms separate and disabled by default.

## Profiling and replay

The 4-being/256-tick profile recorded a late sampled native-brain time of **0.0533 ms**
and a **0.4420 ms sum of timed GPU kernels**. The same sample had **0.6327 ms host
poll wait** and **1.1902 ms tick wall time**. Physics.step's 0.1530 ms is explicitly
CPU wall time around a blocking GPU submission, not a GPU timestamp. These figures
are illustrative samples, not a production throughput benchmark. They show existing
synchronization still matters; do not describe the whole pipeline as CPU-free.

Real binary replays with position/clock-matched diagnostics exported successfully:

- [Ecology replay: two recorded births](../artifacts/coevolution-20260920-final-v2/ecology/preview.html).
- [Combined linear-head replay: generation 2](../artifacts/coevolution-20260920-ablation/combined/preview.html).

Both carry exact observed genomes. The native inherited motor-head export also
uses the correct per-being knots for experimental runs. The browser tool refused
the local file URL under its security policy; **visual rendering and interaction
were not verified**. Export success is not a claim that the UI is bug-free.

The separately deployed lab-ui `genome_export` comes from the
`ui/brain-bins-main` worktree. It has not been rebuilt against this new v4 parser;
that consumer must be ported/rebuilt before v4 runs enter the existing micro.pack
pipeline. Multi-lineage runs additionally require trace segmentation. The local
`replay_view` and `genoma_dump` use the updated canonical loader. Do not silently
feed new files to old exporters or weaken the producer's provenance checks.

## Errors found and corrected / retained evidence

1. Thermal expression initially changed only `physics.wgsl`; fixed in executed A.
2. Header/layout alignment and newborn state ownership were checked on device.
3. Per-being motor-head inheritance had been stored without being executed; enabled
   for these opt-in experiments and made explicit in replay provenance.
4. The first paired script passed `--pop-seed`; the old CLI silently ignored it.
   Both labels used seed 0. Those runs remain as failed-protocol evidence, and the
   paired comparison was rerun with `--seed`, checking metadata 7/19. Unknown
   arguments now exit 2; `gpu_evolve --pop-seed 7` verified the rejection.
5. Curriculum seed 9002 failed geometry checks (including a hole on its centreline).
   The guard was preserved; the integration run used validated 8005/9001 instead.

Next scientific gates: matched-budget multi-seed ablations long enough to reach
curves/laps; separate inherited-spline effect; extinction/replacement stress;
trained promotion/regression transitions; developmental sensor programs and priced
neural capacity; lineage-aware lab-ui export; long-episode tick precision and
resumable ecosystem state. These are explicit limits, not completed validations.
