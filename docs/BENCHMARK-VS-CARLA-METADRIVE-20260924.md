# Universe taxi vs CARLA / MetaDrive / Waymax / CarRacing / GT Sophy (2026-09-24)

Question from Leonardo: "based on CARLA and others, what are we doing wrong and what can we copy?"
Sources: two Sonnet research passes (CARLA docs + Leaderboard evaluation pages; MetaDrive/PGDrive
papers; Waymax paper; gym CarRacing source; GT Sophy Nature/Sony AI; Togelius & Lucas; Cardamone
et al.; curriculum papers 2103.07903, 2207.04361, 2608.22549), cross-checked against what we
measured today in `SCHOOL-STOP-CONTRACT-AB-20260924.md`.

## What we do wrong (each line pairs a measurement of ours with how the field does it)

| # | Ours (measured) | Field | Copy |
|---|---|---|---|
| 1 | 32 learners share one city; stations at junction centres; 2 310 of 2 765 in-box stationary samples of the mover are contact piles | CARLA: one ego per world, NPCs by Traffic Manager, hybrid physics beyond 50 m; MetaDrive single-agent envs, MARL only as a deliberate task, traffic recycled when far; Sophy: 20 cars per console spaced apart | one being per world instance (GPU batch = 32 worlds), or no learner–learner contacts, stations off junction centres |
| 2 | service score = progress + 1000·delivery − 200·fail; "stopped at passenger" = "crashed" = −105; no time cost; stopper lineage survives 32/32 at 142 m | CARLA Driving Score = route completion × Π penalty^count (collision 0.6, red 0.7, …), off-road cancels progress, **agent blocked 180 s = shutdown**; MetaDrive R = Δprogress + 0.1·v/v_max ± terminal; CarRacing −0.1 per frame + 1000/N per tile | dense progress reward, per-tick time cost, blocked-timeout failure, multiplicative bounded penalties instead of instant kill on a pickup request |
| 3 | approach lesson closes the attempt on any pickup request (codes 10/15) → the request channel is extinct (5–20 → 0–3 per gen) before the pickup lesson needs it | CARLA infractions multiply the score, they do not end the route; stop-sign 0.8, red 0.7 | infractions cost, never terminate (except collision with death) |
| 4 | one city (seed 8005), 9 distinct approaches, one counted twice; seed 12 solves 4 of 9 and never generalises | MetaDrive composes 7 block types (Straight, Curve, Ramp, Fork, Roundabout, T, X) procedurally, trains on N maps, tests on held-out maps, success rises monotonically with N; CarRacing new random track per episode; Togelius/Cardamone: a controller evolved on one track drives only that track | block-based procedural city + curved-track family; N maps per generation; held-out test maps; report success per map |
| 5 | p̂ = success on the 3 starts of the gen; gate = 2 consecutive gens ≥ 0.25 → measures a triple, not competence | MetaDrive reports success rate over a fixed held-out set; CARLA Leaderboard = fixed route set; curriculum papers advance on fixed budgets, CL4AD samples scenarios by utility | evaluate every start (or a fixed held-out subset) each gen; advance on a full-window estimate or a budget; log per-start success |
| 6 | red light = 4 sensor channels + fine ≤ 3 cash; beings stop at red and never resume (stopper 463 samples, 0 resumes) | CARLA: red does not touch physics either; compliance is the controller's job; `is_at_traffic_light` only when red; running red ×0.7; being blocked 180 s ends the route | same sensor model is fine; add the blocked timeout so "stop forever" cannot win |
| 7 | the witness is capped at 2 deliveries (`body[68] ≥ 2` → brake) and used only as a demo | CARLA autopilot/TM is the expert for imitation (Learning by Cheating, Roach: privileged teacher → student) | uncap the witness; use it as a privileged teacher for distillation into the evolved brain (we already validated "distil the champion" in Piloto por Fórmula) |
| 8 | observation: retina 14×126 material+distance (33 m), wall probes; navigation input to the evolved brain unverified | MetaDrive: 240-ray lidar 50 m + ego state + relative vector to checkpoints 50 m apart; CARLA agents get the GlobalRoutePlanner waypoints | verify the evolved brain receives a next-waypoint vector from the route field; if not, add it |

## What we already do like the field
- Fixed-step deterministic GPU simulation (CARLA sync mode; Waymax batches 44 000 scenarios in JAX). Re-traces reproduce to 0.1 m.
- Judge with typed failure codes per lesson = ScenarioRunner criteria classes (CollisionTest, RouteCompletionTest, ActorBlockedTest, RunningRedLightTest…). Ours lacks the blocked test and hides code 16.
- Route distance fields per station = a GlobalRoutePlanner; lessons = scenario classes triggered at a start.

## Order of adoption (cheapest, highest expected effect first)
1. Blocked timeout + per-tick time cost + dense progress in the life fitness (fixes 2, 6).
2. Infractions cost instead of terminate in lessons; partial credit for the stop contract (fixes 3).
3. One being per world instance in lives and lessons; stations off junction centres (fixes 1).
4. Evaluate all starts per gen; gate on the full window; per-start log (fixes 5).
5. Procedural map family (blocks + curved track) with N maps per gen and held-out maps (fixes 4).
6. Witness uncapped as teacher (fixes 7).
