# School: one lesson start per generation (`--school-shared-start`) — 2026-09-22

Follow-up of [SCHOOL-STOP-SHAPING-20260922.md](SCHOOL-STOP-SHAPING-20260922.md) §6.5–6.7.
Branch `organism/school-20260922`. Status: **pre-registered, battery running**.

## 1. Why (verified on the real GPU before this change)

The school gives each of the 16 beings its own junction approach and rehashes the
being → slot map every generation (`src/bin/gpu_evolve/main.rs`, `slot_rotation`). One
genome is therefore judged once per generation, on a different geometry each time.

Fixed-slot witness (`witness-fixed-slots-s7/`, binary `d69ad90cd871b007…`, `--fixed-spawn-slots`,
stop + shaping, seed 7, 4 generations): the elite at slot 0 — same genome, same start
(99, 156) — scored progress 0.916544, score 458.272, energy 249 731.0 J at generations 1, 2
and 3, bit-identical. The same genome had scored 95.4 (corridor failure) on its gen-0 start.
So the simulator is reproducible for (genome, start) and the 0.00–0.99 progress swing of the
elite in every arm of the stop-shaping battery is the geometry, not the being. Selection was
comparing beings across geometries at n = 1 — the same regression-to-the-mean the track world
paid for and fixed with `--largada-aleatoria` (comment at `main.rs` ≈ 4652: the champion
"had been born in the right place").

## 2. Change (host only, no kernel touched)

`--school-shared-start rotate|fixed`, default off:

- `rotate`: all beings of generation *g* start on candidate `g mod count` of the fold.
- `fixed`: all beings start on candidate 0 of the fold in every generation.

`src/organism_school.rs`: `SharedStart` (`parse`, `name`, `pick`) and
`lesson_shared_start_fold(...) -> (starts, index, count)`; host test
`shared_start_gives_every_being_the_same_candidate_and_wraps` (static geometry).
`src/bin/gpu_evolve/main.rs`: flag parse, validation (requires `--school`), start assignment
per generation with a `[school-start]` log line, `start_assignment` and `shared_start`
`{index, count}` in `organism_genNNNN.json` and `school_result_genNNNN.json`.
`cli_flags.rs`: the flag. Judge, score, physics and perception unchanged.

Why a shared start is clean here: `--school-mode bench` (the default under `--school`) turns
taxi–taxi contacts off (`cli_flags.rs:149`, `src/gpu/school.rs:255`) and the retina occludes on
other beings only under `--retina-beings` (default off), so 16 beings on one start neither
collide nor see each other. Held-out evaluation keeps the per-slot recipe, so a champion is
still measured on 16 distinct approaches.

## 3. Flow map

| Producer | Buffer / file | Consumer |
|---|---|---|
| `lesson_shared_start_fold` (host, per generation) | `school_starts: Vec<LessonGeometry>` (n identical) | `city_spawns` → being pose; `SchoolGpu` geometry upload (unchanged path) |
| `slot_rotation` | permutation of identical spawns (no effect) | pose |
| `school_shared_pick` | `organism_gen*.json`, `school_result_gen*.json` (`shared_start`, `start_assignment`) | `analyze.py`, lab observatory (`start` per being already exported) |

## 4. Validation ladder

1. Static: `cargo test --lib organism_school::tests` (host geometry, allowed by
   `Universe/CLAUDE.md`).
2. Gate A (regression, GPU): the untouched per-slot path on the new binary reproduces the
   gen-0 `[school]` line of the previous battery's shaping arm (seed 7) exactly.
3. Gate B (determinism on the new path, GPU): `--school-shared-start fixed`, 3 generations —
   the elite's score must repeat at generations 1 and 2.
4. Battery (§5). Any non-zero exit stops the script.

## 5. Pre-registered battery

`/media/llm-data/universe-runs/school-shared-start-20260922/battery.sh`, binary
`gpu_evolve-shared-start` (debug build of this commit; sha256 in `battery.log`).
Arms `default` (per-slot, rotated by hash), `shared-rotate`, `shared-fixed`; seeds 7 and 11;
`--school stop --school-stop-shaping --school-start-fold train --beings 16 --gens 16
--ticks 1280`; gen-15 champion ×16 on validation and test (per-slot starts, one generation,
`--resume --resume-mix 1.0`). Expected wall ≈ 10 min (6 × 80 s + 12 × 7 s + gates).

| Claim | Expected | Refuted if |
|---|---|---|
| Replication | `shared-fixed`: the elite's score is identical at every generation ≥ 1, both seeds | any generation differs |
| Reach | mean over gens 13–15 of beings with a non-zero shaping bonus ≥ 4/16 in a shared arm in ≥ 1 seed; `default` ≤ 2/16 (replicating §6.5.1) | every shared arm ≤ 2/16 in both seeds |
| Competence | ≥ 1 train success by gen 15 in ≥ 1 shared arm; `default` 0 | 0 successes in all four shared trains |
| Transfer | if a shared arm succeeds on train, its champion ≥ 1/16 on validation | champion 0/16 on both folds |
| Safety | 0 deaths; contact + corridor at gen 15 not above `default` by more than 3/16 | more deaths or a contact surge |

Honest expectation: Reach is the claim this change can win; Competence still needs the
first tick under `v_stop` = 0.3 m/s, which no being has produced yet (§6.5.2 of the
stop-shaping record). If Reach holds and Competence fails, the next change is the slope
under 0.3 m/s (partial credit for ticks under 1 m/s), not another evaluation change.

## 6. Results

(filled after the battery)
