# Gap analysis at the Codex handover — 2026-09-25 20:36

Claude took over the whole Universe taxi front (navigation, battery, cities,
physics) after Codex went offline. This record lists what the being can do,
what it cannot, and the gaps Codex left, each with the artefact or `file:line`
that proves it. Three read-only sonnet audits (diff, runs vs claims, pipeline)
fed it; every number below was recomputed or opened by Claude before being
written. No simulation was run for this document.

## 1. What the being does today (measured)

| Task | Result | Artefact |
|---|---|---|
| Straight 12 m course, champion frozen, 4 orientations | 4/4 deliveries; 3/4 at 24 m | `full-service-{frozen,24m-audit}-20260925/course-*/school_result_gen0000.json` |
| Straight 48 m course, population gen 11 (16 DNA × 4 starts × 4 replicas) | 120/256 deliveries; onboard leg median 35.2 s = **1.36 m/s**, max speed median 3.5 m/s | `stable-centered-backbone-thaw-20260925/school_result_gen0011.json` |
| Same course, `--school-select worst`, gen 23 | **2/64** deliveries (mean speed 1.08 m/s) | `neutral-48m-worst-start-20260925/school_result_gen0023.json` |
| Straight 96 m, frozen 48 m champion | 2/4 | `stable-route-transfer-20260925/longer-96m` |
| City course (≥300 m, turns), founder / 4 gens training / frozen candidate / v4 / v5 | **0 pickups, 0 deliveries in every arm**; 4/4 `blocked_without_progress`; being 0 below 0.5 m/s in 93.6 % (founder) / 72.9 % (candidate) of ticks | `cross-city-forward-learning-20260925/*`, `navigation-preview-paired-20260925/v4-control`, `navigation-preview-evaluation-20260925/v4-control` (this folder is really v5, see `label-correction.json`) |
| Scripted witness on the 48 m starts | 4/4 (not evolved) | `stable-west-diagnosis-20260925/scripted-feasibility` |

Reading: the evolved brain has learned to creep to a point and hold still,
on a straight lane, from a handful of memorised starts. It has never turned a
corner into a delivery, never driven at road speed while on task, and never
completed any service on the city. The 2026-09-25 selection analysis
(`META/CLAUDE-UNIVERSE-AMBIENTE-20260925.md` §6) shows why the city score
"improved": the collision penalty shrank while progress fell.

## 2. Gaps, ranked by how much they block a city delivery

### A. The judge trains stopping, not driving (structural, highest)

1. Acceptance envelope is a parking test. `school.wgsl:196-197`
   `goal_radius = 3.0 m`; the start lane is offset 3.0 m, so the approach is
   tangent to the disc (zero lateral margin). Arrival needs >3 m/s once
   (fail 8/17), then <0.3 m/s inside 3 m, held 128 ticks (fail 14), request
   while `speed<0.5` (`taxi.wgsl:270,286,303`). Codex's own retraction:
   `docs/ANALISE-FULL-SERVICE-20260925.md` §5.
2. No incentive for speed. `school.wgsl:112,124`: time cost is `-seconds`
   (1 point per second) against `+200` readiness, `+200` pickup, `+1000`
   delivery, `+500` success; on any failure the full budget is charged
   regardless of behaviour, so a failing being gains nothing by moving. A
   1.4 m/s delivery and a 10 m/s delivery differ by ~30 points out of ~2000.
3. Progress is a banked maximum. `school.wgsl:226`
   `PROGRESS=max(previous, clamp(body[127],0,200))`: once the best point is
   reached, freezing keeps the credit; collision (`min(J/1000,200)`) and
   corridor (fail 3, `school.wgsl:250-253`) are cliffs. Rational optimum for
   a weak driver: creep, bank, stop. Matches `route_revisit ρ=+0.64`,
   `distance ρ=-0.79` in the selection analysis.
4. Readiness bonus is a step. `school.wgsl:233,122`: `+200*max(proximity*slowing)`
   banked once; the score jumps 198→2003 between generations 24 and 30 while
   deliveries go 23→35 (`ANALISE-FULL-SERVICE` §4). Selection follows the step.
5. Kill timers are invisible. `BLOCKED_SECONDS` (`school.wgsl:236-237,310`)
   and the corridor edge exist only in the judge; no wave channel carries
   them, so there is no gradient before fail 18 or fail 3. Only `TaskAge`
   (ch 181, /300 s) is observed.

### B. The being cannot see the road ahead (perception)

6. Default observation schema is v2 (`main.rs:1014`): straight-line
   `(target-pos)/100` clamped ±1 (`taxi.wgsl:340-345,399`) — saturates
   beyond 100 m and ignores walls. City runs used v4 (one-cell route descent,
   `taxi.wgsl:64-66`: "no turn anticipation") or v5 (preview bearing ≤48 m).
   The v5 arm did run, mislabelled as `v4-control`, and scored **worse**
   (progress 35.0 vs 42.8, 0/0 both): the cue alone did not help a brain that
   never moves.
7. No channel for distance to the next junction, the lane after the turn,
   or the remaining route length at metre resolution (ch 183 is
   `d/(d+100)`: at 600 m one i8 step ≈ 39 m).
8. Ring/curved geometry is unseen by every helper that assumes the grid
   (`organism_school.rs` lane rule; patch delivered, not applied).

### C. Selection cannot separate a driver from a specialist

9. `--school-eval-starts` defaults to **1** (`main.rs:1051`); default
   `Mean` aggregation (`school_batch.rs:58-133`) lets one good start pay for
   three bad ones. With `worst` the population collapses to 2/64 on the
   easiest course: no lineage has a general policy to select.
10. Prior finding (school stage 4/5, `SCHOOL-STOP-CONTRACT-AB-20260924.md`):
    every lineage specialises on 2–3 of 10 starts, max p̂ 0.087, never passes
    the 0.25 gate; corner-only approach is an ancestor property.
11. ~38 of 48 L2 rows are frozen by default (`congela_l2_cegas`, `main.rs:893`,
    `evolution.rs:1732-1736`); taxi rows 0–9 stay mutable, so this limits
    capacity rather than blocking service. Not measured.

### D. The curriculum has a cliff between 96 m and 300 m

12. Successes exist only on straight lanes ≤96 m with the station in the
    start lane. The next rung in use is the cross-city course: ≥300 m, ≥2
    turns, empty start, 512 s. Nothing in between exists (one T turn,
    ~150 m). Every city arm dies of `blocked_without_progress` before the
    first junction.
13. Lesson order gates delivery behind pickup (`main.rs:3077-3079`), and the
    native controller on the gen-198 ancestor emitted zero pickup requests
    (`docs/TAXI-SERVICE-CONTRACT-V2.md`); later 48 m runs do request, so the
    ancestor was the problem, not the head — but the adaptive curriculum
    still has no rung that teaches "drive 150 m, turn once, then park".
14. In-world progression (school area inside the city, promotion by p̂ over
    all starts, regression, retention) is a contract draft only
    (`META/CLAUDE-UNIVERSE-AMBIENTE-20260925.md` §4).

### E. Process gaps in the tree Codex left (46 files, +4216/−456, uncommitted)

15. Checkpoint fingerprint omits `brain.wgsl` (`organism_checkpoint.rs:63-105`
    lists `brain/mod.rs` only) while Codex edited `brain.wgsl` in this diff:
    a resumed checkpoint will not detect a brain kernel change. **Fix before
    any resume.**
16. `traffic_contacts.wgsl:40` sets the taxi contact mask bit outside the
    `closing>0` guard; under `--school-service-legacy` any static touch now
    kills the episode (`school.wgsl:245`), breaking the "legacy = old judge"
    parity claim. Default `service_v2` unaffected.
17. `navigation-preview-evaluation-20260925/v4-control` is the v5 arm
    (`label-correction.json`); `NAVIGATION-PREVIEW-20260925.md` is still
    `Status: OPEN` with no result written. The v4/v5 comparison is
    complete and negative; it must be recorded as such.
18. `school.wgsl:8-22` header omits fail code 18 and flag bits 4–6
    (`school.rs:121`). `evidence-evolution-plan.md` P1–P7 all unchecked though
    P1–P5 shipped. `ANALISE-FULL-SERVICE` §7 says the 24 m audit did not run;
    it did (3/4).
19. All new judge/route/controller logic is proven only by `#[ignore]` real-GPU
    tests; none has a recorded pass since the last edits. `check_native_flow_gpu.py`
    must be rerun before the next battery.
20. City family (`src/city_family`, ring) and the school patch are written but
    uncompiled (build queued for an idle window).

## 3. Attack order (each step: prediction, ceiling, paired A/B, number)

1. **Judge v3** (gap A): continuous credit = road progress **per second**
   (rate, not banked max), `blocked_seconds` and corridor margin exposed as
   channels, readiness bonus made continuous (proximity × slowing integrated,
   capped), time cost proportional to the deadline fraction on failures too.
   Envelope kept (parking is part of the job) but `goal_radius` measured
   against the lane offset so the approach has ≥1 m margin. Paired A/B on the
   48 m course, 12 gens, 3 seeds; prediction: mean onboard speed rises from
   1.4 to >4 m/s and deliveries stay ≥100/256; ceiling: 48 m at 10 m/s = 5 s.
2. **Rung between 96 m and the city** (gap D): 150 m with one T turn on the
   grid (or ring chord), station after the turn, empty start; then 300 m with
   two turns; promotion by p̂ over **all** starts, `eval-starts ≥ 4`, selection
   = number of starts passed then mean score (gap C).
3. **Perception for turns** (gap B): junction distance + post-turn lane bearing
   channel (v6), retested on the T rung, not on the 300 m course.
4. **Process fixes** (gap E): fingerprint `brain.wgsl`, contact mask under the
   closing guard (or legacy exemption), relabel the v5 folder, close the
   navigation doc with the negative result, rerun `check_native_flow_gpu.py`.
5. **City family** goes live only after step 2 passes on the grid: same rung
   on `ring` seeds, train seeds vs eval seed 9.

Nothing above is done. Step 4 is pure host work and can start in the next
idle window; steps 1–3 need the GPU and a paired protocol registered here
before the first run.
