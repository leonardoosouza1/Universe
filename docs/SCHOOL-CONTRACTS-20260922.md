# Organism-taxi contract corrections — 2026-09-22 (stage 1 of the school program)

Branch `organism/school-20260922`, base checkpoint `b301c90` (Astra's uncommitted
tree committed verbatim). Handbook: `../neuromod-lab/docs/HANDBOOK-ANALISE-DE-FLUXO.md`.
Every claim below is labelled **verified** (command + output), **static** (read in
code), or **inferred**.

## Flow map before editing

| Contract | Producer (GPU/host) | Data | Consumers | Defect (static) |
|---|---|---|---|---|
| Contact mask | `physics_c.wgsl:321` writes numeric `f32(contact_mask)` into output slot 39 | bits 1/2/4/8 = probes 0..3 | `traffic_contacts.wgsl:39` (OR bit 16 via `bitcast`), `organism_telemetry.wgsl:31-36` (`bitcast`, stores body[123] as bit pattern), `organism_orientation.wgsl:29` (`f32(bitcast<u32>(...))` → 1065353216 into physics input 39), host `physics.rs:334` (`as u32`), `driving_diagnostics.rs:244/527`, lab-ui BodyTelemetryPanel | numeric producer, bit-pattern consumers: telemetry never counts contact edges after physics reports one; JSONL shows `contact_mask=1065353216` |
| Cell index | `world.rs::get_voxel_at` Rust `round` (half away from zero); `organism_sensors.wgsl::cell_round` same | cell i ↔ centre i·3 m | `taxi_routes.rs:16` station cell `floor`; `taxi.wgsl::route_distance` `floor`; `taxi_witness.wgsl` `floor` + waypoint `(q+0.5)·cell`; `physics*.wgsl::voxel_at` WGSL `round` (half to even) | route field and physics disagree by up to one cell (1.5 m); witness waypoints offset 1.5 m from cell centres |
| Spawns | `organism_city::spawn_positions` depot + lattice rows from z=0 step 2 cells | positions, heading fixed +x in `main.rs:4371` | physics input, taxi post, replay | slot 0 born on the depot (instant pickup), others at z=90 (1.5 m from the wall), `--same-start`/`--random-start` ignored in the city branch, no per-slot geometry recorded |
| Life metadata | `context.rs:503` `life_left = alive flag` | 0/1 | replay meta `main.rs:3947` announces `remaining_tick_budget`, `player.rs:285`, `view.js:214` | organism replay labelled as tick budget |
| Navigation observation | `taxi.wgsl::senses` word 44/46: dx/100, dz/100 clamped separately | channels 178/179, 184/185 | brain.wgsl, lab-ui WavePanel via manifest | saturates: (1000,100) and (100,1000) both quantise to (127,127); direction and distance lost beyond 100 m |

## Corrections (design)

- **A. Contact mask numeric everywhere.** `traffic_contacts.wgsl`: `f32(u32(max(snapshot,0))|16)`. `organism_telemetry.wgsl`: `u32(max(x,0))`, body[123] stored numeric. `organism_orientation.wgsl`: copy slot 39 numerically. Host unchanged (`as u32` already numeric). Bit 16 = taxi–taxi contact.
- **B. One cell convention: half away from zero, cell centre = i·cell.** `taxi_routes.rs` station cell → `round`; `taxi.wgsl` / `taxi_witness.wgsl` → `sign(v)*floor(abs(v)+0.5)`, waypoint = `q·cell`; `physics.wgsl` / `physics_c.wgsl` `voxel_at` → same formula (differs from WGSL `round` only at exact ties). Retina ground voxel keeps its documented `floor`.
- **C. Spawns on lane centres, nobody on a station.** Candidates along every street at 18 + 12k m from the intersection centres, lane offset ±3 m from the street centre (4.5 m from the nearest wall edge), heading = lane flow, ordered by depot distance; slot map rotated per generation by hash(gen, seed) (`--fixed-spawn-slots` disables). Every slot's position, heading and depot distance recorded in `organism_genNNNN.json`.
- **D. Life provenance.** Organism replays announce `life_semantics = "alive_flag"`, `life_model`, capacity 1; legacy keys untouched for the race scenario.
- **E. Navigation observation v3** (`--navigation-observation v3`, default v2 so old runs reproduce): 178/179 = cos/sin of the target direction in the body frame (forward, lateral axes from the physics pose), 183 = compressed distance d/(d+100), 184/185/186 = same for the nearest service, 187 = min station stock fraction, 188 = refusal, 190 = fare; 182/191 stay traffic-owned (unchanged). Fare/refusal/stock rows move; deliveries (182) and alive (191) were already overwritten by traffic in v2, so no information is lost.

## Pre-registered validation (real GPU, RADV, one test thread)

| Fixture | Prediction | Tolerance |
|---|---|---|
| traffic pair contact with preset slot 39 ∈ {1, 5} | 17.0 and 21.0 exactly; isolated being stays 0 | exact |
| telemetry window with slot 39 = 17.0 | body[123] = 17.0, one contact edge counted, touching ticks = 1 | exact |
| orientation feedback | physics input 39 = output 39 (17.0), never 1065353216 | exact |
| route cell at x = 97.6 m (32.53 cells) with station cell 32 | remaining = 3 m (cell 33), not 0 (floor) | 1e-4 |
| host station cell at 95.0 m | target index = cell 32 (round), field 0 there | exact |
| spawns n = 12 on the real city | all on Road, ≥ 4.5 m from any wall edge, pairwise ≥ 8 m, ≥ 18 m from every station, heading axis-aligned, rotation deterministic and bijective | exact |
| senses v3, targets (1000,100) and (100,1000) | distinct channel bytes; v2 gives (127,127) for both | exact bytes |

Stop rules: non-finite state, any assertion failure, software adapter. No campaign
before all rows pass.

## Results

Binary: `target/debug/gpu_evolve`, SHA256 `26ecabef21a772e1ddf2e39b818efbf625b1d6e3582cb2cd6220d315de54b0d8`
(recorded in `/media/llm-data/universe-runs/school-contracts-20260922/binary.sha256`).
Environment for every GPU command: `unset DISPLAY; WGPU_BACKEND=vulkan
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json MESA_VK_VERSION_OVERRIDE=1.3
RAYON_NUM_THREADS=1 CARGO_BUILD_JOBS=1`; adapter reported by every fixture:
`AMD Radeon RX 6750 XT (RADV NAVI22)`, Mesa 25.2.8. Tests ran with
`cargo test --lib '<filter>' -- --ignored --nocapture --test-threads=1`.

### Pre-registered fixtures — all rows passed (verified)

| Fixture (test name) | Expected | Observed |
|---|---|---|
| `gpu::traffic_tests::gpu_contact_mask_is_numeric_across_traffic_feedback_and_telemetry` | pair 1,5 → 17,21; isolated 8 stays 8; feedback input[39] = 17/21; host decode 17; telemetry body[123] = 17, one edge, touching 1→2 | exactly as predicted (asserts exact) |
| `gpu::organism_telemetry::gpu_window_and_lifetime_fixture` | body[123] = 17.0 numeric incl. taxi bit | 17.0 |
| `gpu::organism_orientation::gpu_orientation_quarter_turn_and_slot_bounds` | slot 39 copied numerically (5.0 in, 5.0 out), never 1065353216 | numeric copy; suite 9/9 |
| `gpu::taxi::tests::taxi_gpu_scripted_stock_payment_once_and_moving_refusal` (cell-tie case) | z = 15.3 → cell 5 (score 100 + 100·(1 − 110/120)), 14.7 → cell 5, 15.0 tie → cell 5 (half away from zero) | matches within 1e-4 |
| `gpu::taxi_routes::tests::station_cell_uses_round_half_away_from_zero` | station (5.0, 6.0) m on a 3 m grid → cell (2,2) field 0; (4.5,4.5) → cell 2 | pass |
| `organism_city::tests::spawn_slots_sit_on_lanes_away_from_walls_and_stations` (n = 12) | all on Road, ≥ 4.5 m from wall edges, pairwise ≥ 8 m, ≥ 18 m from stations, axis-aligned heading, rotation bijective (`slot_rotation_is_deterministic_and_covers_every_slot`) | pass |
| `gpu::taxi::tests::taxi_gpu_navigation_v3_keeps_direction_and_distance_beyond_100m` | v2: (1000,100) and (100,1000) both → (127,127); v3: distinct bytes within ±1 of cos/sin·127 and d/(d+100)·127; idle → (0,0,127) | pass |
| `wgsl_validation` (12) and host static tests (13) | parse + type-check, static geometry | 12/12, 13/13 |

### Smoke runs (verified, real GPU, exit 0 both)

```
./target/debug/gpu_evolve --scenario organism-taxi-v1 --beings 8 --gens 2 --ticks 640 --seed 7 \
  --driving-diagnostics --vision-every 16 --no-spectral-init --navigation-observation {v2|v3} \
  --out /media/llm-data/universe-runs/school-contracts-20260922/smoke-{v2|v3}
```

- `[spawn] gen 0 rotation 0: 0->(93,114) 18m 1->(114,93) 18m 2->(99,120) 24m 3->(120,99) 24m`;
  `gen 1 rotation 5: 0->(126,93) 30m … 3->(93,114) 18m`. Nobody born on the depot (96,96);
  `driving.csv` window 1 of being 0: `wall_left 10.5, wall_right 4.5` (lane centre, 4.5 m from the wall face).
- `organism_gen0000.json`: `spawn_slots` (8 rows with position, heading, depot distance), `spawn_assignment.mode =
  rotated_by_hash_generation_seed`, `observation = organism-city-observation-v2|v3`, `sensory_channels.schema` matches.
- `gen0000.meta.json`: `life_semantics = alive_flag`, `life_capacity = 1`, `life_model = organism-alive-flag: body[29]>0.5 -> 1 else 0`,
  `life_earn_ticks_per_new_progress_metre = null`, `navigation_observation = 2|3`.
- `observation.jsonl` (80 rows each): `body.contact_mask = 0` in all rows (no contact in 640 ticks at ≤ 0.06 m/s — random brains crawl);
  taxi wave bytes 176..191 differ exactly where the manifest says: v3 178 = −126 (task behind the being: pickup at the depot,
  being heading +z), 183 = 19..29 (d/(d+100) for 17–30 m), 187 = 95 (min stock 0.75), 188 = 0..106 (refusal moved from 183).
  v2 keeps 178/179 = dx/100, dz/100 (−38..4, −23..4).
- Not a learning claim: 640 ticks, random genomes, no deliveries.

### Reference witness on the corrected binary (verified, exit 0, validator FAILED)

```
./target/debug/gpu_evolve --scenario organism-taxi-v1 --taxi-reference-witness --beings 1 --gens 1 --ticks 16384 \
  --seed 7 --driving-diagnostics --vision-every 16 --no-spectral-init \
  --out /media/llm-data/universe-runs/school-contracts-20260922/witness-c
python3 scripts/validate_taxi_witness.py <run>   →  AssertionError: collision during witness
```

| Quantity | Value |
|---|---|
| deliveries | 1 (onboard at tick ≈ 4100, delivered by tick 12304, money 25 → 42.68) |
| distance | 632 m; at rest from tick ≈ 11 264 at (484.4, 96.0), speed 0.04 m/s |
| contact ticks / events / impact ticks | 72 / 3 / 2 |
| collision energy | 51 322 J (impacts at tick 832 (102, 162.6) 6.1 m/s and tick 4032 (102.8, 89.7) 5.9 m/s; wall scraping 70 ticks at x ≈ 88.8, z ≈ 100–105) |
| `observation.jsonl` contact_mask values | {0: 1020, 4: 4} — numeric probe bit, never 1065353216 (contract A in a real run) |

Reading: with lane-centre spawns the witness slot 0 is (93,114) heading +z, i.e. away from the depot where its
passenger waits. The reference controller (`taxi_witness.wgsl`, `steer = dot(desired, lat)·1.8`) has no reverse or
U-turn rule, so it drove north to the z = 160 junction, hit the wall, came back scraping the west wall, hit again near
the depot, then delivered one straight trip east along z = 96. The trip is real (contract B routes and A masks agree
with the diagnostics), the safety is not: the witness contract requires zero collision energy. Extending the reference
controller to a legal turn-around is stage-4 work ("extend the reference to curves"); it is not hidden by choosing a
friendlier slot.

### Panel (lab-ui, real artifact)

Packaged with `python -m lab_api.trace.package_run <run> --db /media/llm-data/lab.db` after moving the runs to
`/media/llm-data/universe-runs/school-contracts-smoke-v3` and `…/school-contracts-witness-c` (the trace route
resolves `runs_root/<run>/trace/…`, so nested run directories are not addressable; symlinks left under
`school-contracts-20260922/`). Trace `organism.json` carries `sensory_channels.schema =
organism-city-observation-v3` with rows 178 TaskDirectionForward, 179 TaskDirectionLateral, 183 TaskDistance,
187 StationStockMinFraction, 188 TaskRefusal — what WavePanel's channel table reads.
Two defects sat between the artifact and the panel, both outside this repository (neuromod-lab working tree,
left uncommitted next to Astra's changes there, reported separately):

- `lab_api/routes/trace.py::get_artifact` only accepted the short keys (`organism`, `track`) while lab-ui
  `stores/run.ts` requests `organism.json` / `track.json` → HTTP 404 (verified with curl before the patch), so
  the sensory channel table never loaded. Patched to accept both spellings; after restarting the local API with
  its original arguments: `organism.json http 200 345931B`, `track.json http 200`.
- `lab_db.ingest_run` deleted the `run` row on every re-ingest and the foreign-key cascade erased the trace
  registrations made a moment earlier (only gen 1 survived; `trace-index` listed one row). Patched to keep the
  trace rows across re-ingestion; `trace-index` now lists gen 1 and gen 0.

Real panel (`http://127.0.0.1:8131/run/school-contracts-smoke-v3/0/0`), headless Chrome at 3440×1240:
screenshot `/media/llm-data/universe-runs/school-contracts-20260922/panel-smoke-v3-3440x1240.png`
(retina, 192-channel wave, 3D brain, stages, mind→body observatory, 2D city with 5 stations and the taxi at
(93,114)). Rendered DOM (`--dump-dom`) contains `TaskDirectionForward`, `TaskDirectionLateral`, `TaskDistance`,
`StationStockMinFraction`, `TaskRefusal` once each — the WavePanel channel table under "Inspecionar todos os
inputs" is fed by the v3 manifest. Body block served by `/block/body` (40 × 12 f32) decodes `contact_mask` = 0
for every sample and `alive` = 1 (no contact in this run). Not exercised: the "Telemetria corporal" tab
(needs a click; headless run stays on the default layout) and a packaged run with a non-zero mask — the witness
cannot be packaged because `lab_api/trace/check.py` only allows motor-trace sources
`native_action_mapper_v1` / `resident_native_action_v1`, not the witness's `resident_reference_action_v1`.
Pre-existing, unrelated: `tests/test_lab_api.py::TestTraceRoutes` (10 tests) fail with 404 in the current
neuromod-lab tree because the fixture is a symlink outside the test runs root and the uncommitted
`_trace_dir` hardening rejects it; the same 404 occurs on the manifest route my patch does not touch.

### Limitations

- Contact edge count in a real run only exercised by the witness (4 samples with mask 4); the multi-being taxi–taxi
  bit 16 is covered by the fixture, not yet by a run with traffic contact.
- The cell convention still differs from WGSL `round` (half to even) only at exact ties; `organism_sensors.wgsl`
  already used half-away-from-zero, the retina ground voxel keeps `floor` (documented, unchanged).
- `check_language.sh` (diff vs `origin/main`) flags pre-existing Portuguese headers in `main.rs`, `context.rs`,
  `statics.rs` (lines 1–6); no line added in this stage is flagged.
- v3 is opt-in; no evolved champion exists under v3 yet. Task age (channel 181) read 0 in every smoke sample of both
  versions — pre-existing v2 behaviour, not touched, to be checked when the school judge reads task age.

### Next test

Stage 2: lesson contracts (move-to-target, stop-in-area, complete-a-turn) with a per-tick GPU judge, run with
`--navigation-observation v3`, lane-centre spawns and fixed slots for the bench; first battery = 3 seeds × 2 arms
(uniform vs adaptive curriculum) with equal being-ticks, stop rule on any non-finite state or any shortcut flag.
