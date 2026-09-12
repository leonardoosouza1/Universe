# Response to the external review of the u1v battery (12 Sep 2026)

Reviewer: Astra/Codex, reviewing Universe up to `8ce51dc` and the lab up to `c973454`.
Author: Fable, 12 Sep 15:20. Every item below was re-checked on this machine; commands are listed.

## Accepted, with the fix already in the tree

| # | reviewer's point | verdict | what changed | evidence |
|---|---|---|---|---|
| 1 | "the memory-dependent part is ~0" was not supported: comparing the max of three components hides throttle behind a constant brake offset | **accepted, retracted** | `hip_check --memory` computes Δb(t) = b(mem_t) − b(empty) per component; U1 text corrected | `4b722a9`; s031 mean\|Δ\| throttle 0.0839 / steer 0.0236 / brake 0.0227, max throttle 0.1667 — the reviewer's numbers, reproduced to 4 dp; s077 0.0718 / 0.0513 / 0.0137; s113 0.0184 / 0.0095 / 0.0244 |
| 2 | a partial second pass counted a sector where the being died after 1 tick as a 200× speed-up | **accepted, bug fixed** | `tools/voltas.py` excludes the interrupted sector (highest index reached in 0..22; sector 23 is the first sector of the following lap and stays eligible); regression tests with the reviewer's two cases; the six u1v runs have no partial records, so their numbers are unchanged (Δ +0.0016, sd 0.0070, MDE 0.0229, byte-identical JSON) | `65d392e`, self-test 27/27 |
| 3 | U10 mixed two changes and used the MDE as a threshold; "first lap neutral" is false at sector 23 | **accepted** | U10 split into U10a (drop the constant column, mutation unchanged) and U10b (mutation 0.3 / 0.1, column kept); ruler = pre-specified effect size (Δ ≤ −0.02 with a CI excluding 0), not the MDE; the sector-23 lookahead is stated | `363184a` |
| 4 | run the U3 ablation before attributing the regression | **done** | `u3`: 9 evaluations, one generation, 40 beings, same seed ⇒ same population, hip champions of u1v gen 14; `mem` / `apagar` (memory read zeroed, resting offset kept) / `sem` (organ off) | `363184a`, run dirs `u3_s{031,077,113}_{mem,apagar,sem}` |

## What U3 says

≥2 laps out of 40, per seed (mem / erase / off): s031 39 / 40 / 38 · s077 30 / 37 / 35 · s113 27 / 29 / 32.
`mem − erase` = −1, −7, −2 (3/3): the memory READ costs laps. `erase − off` = +2, +2, −3: the constant-column offset is
neutral. Per-sector time on the second pass vs the first (r = t2/t1): `mem − off` = −0.012, +0.005, −0.011 — the second pass
is ~1 % faster with memory in 2 of 3 seeds, below the MDE (0.023). So the regression in U2 comes from a maladapted memory
read, not from the offset. The reviewer's alternative explanation was the right one; my "offset + mutational load" story
was half wrong (the offset half) and half untested (the mutational-load half, now U10b).

## Status labels (handbook)

- Verified: the four rows above, the U3 counts, the hip_check reproduction.
- Inferred: "maladapted read" as the cause of the lap deficit (consistent with mem − erase in 3/3; a mechanism at the level
  of which sectors lose time is not isolated yet).
- Hypothesis: mutational load of p = 1.0 / σ = 0.2 prevents W from settling — tested by U10b (running, 3 arms, manifest
  in PREDICOES-CEREBRO).

## Not changed

- The u1v verdict stands: no detectable second-pass gain (Δ +0.0016 ± 0.0070, MDE 0.023) and a completion deficit in 3/3.
  "Not detected" is not "absent": 15 generations and one track, stated in U1‴.
- The champion is shared across the three replicas by design (replica = mutation stream + world seeds); they are not
  independent lineages and are not called that.
