# Minecraft pivot — what the new branch keeps, adapts and removes

*2026-09-26. Assessment only. No branch created, nothing deleted. The owner will
send a planning document; the branch is cut after it arrives.*

## 0. Decision the owner made

The procedural taxi/track world stops being the environment. Minecraft (Paper
1.21.4 server, already running in `~/projects/LLM/mc-lab/`) becomes the world:
rules, physics, day/night, hunger, hostiles, villagers and terrain come from the
game. Universe keeps only what is world-independent: the GPU brain, the genome,
the evolutionary loop and the evidence/gate tooling. A new branch removes the
rest and adapts the two contracts that touch the world (sensory input, motor
output).

Sizes below are `wc -l` on `organism/school-20260922` at `db7ac1e`.

| Area | Lines | Files | Verdict |
|---|---:|---:|---|
| `src/gpu/` (all) | 34 493 | 86 | ~9 900 keep, ~1 700 adapt, ~22 900 remove |
| `src/bin/gpu_evolve/` | 24 990 | 38 | ~2 500 keep, ~9 000 adapt (mostly `main.rs`/`context.rs` slices), rest remove |
| `src/bin/*.rs` (37 tools) | 21 165 | 37 | ~1 900 keep, rest remove |
| `src/` library (non-gpu) | ~28 000 | — | ~6 500 keep, rest remove |
| `crates/taxi-world` | 7 913 | 17 | remove |
| WGSL runtime kernels | 10 980 | 38 | 3 199 keep, ~200 adapt, 7 600 remove |

Rough total: the branch keeps about 20 % of the tree.

## 1. KEEP — world-independent, moves as-is

### 1.1 GPU brain kernels (WGSL, 3 199 lines)

| Kernel | Lines | Why it stays |
|---|---:|---|
| `brain.wgsl` | 1 369 | KAN → CfC → BitLinear L1 (192→96) → L2 (96→48) → intent. Reads `buf_wave` (192 i8 per being) and writes `buf_intent`; it does not know what a channel means. |
| `slow_brain.wgsl` | 332 | 48-input slow CfC (`INPUT_SIZE=48`, `CFC_SIZE=64`, 3 outputs). Fed by `slow_gather`. |
| `slow_gather.wgsl` | 104 | packs the slow input from the fast brain state. |
| `cortex_sheet.wgsl` + `_motor` + `_hebb` | 414 | alternative sheet brain with Hebbian plasticity. |
| `fly_map/fly_map.wgsl` + `spiking.wgsl` | 647 | FlyWire fixed connectome brain (2026-09-14). Its eye-binding drive and the parity tests are world-neutral. |
| `organism_feedback.wgsl` | 47 | online adapter; only the money/progress baselines are taxi-flavoured (see §2.3). |
| `organism_telemetry.wgsl` | 81 | per-being telemetry copy; body slots renamed but layout stays. |
| `retina_metrics.wgsl` | 39 | retina statistics — keep only if the MC eye keeps a 14×126 retina; otherwise remove. |

### 1.2 GPU hosts (Rust)

`gpu/core.rs` (device + buffer manager — **one edit**: `new(world: &VoxelWorld, …)`
at `core.rs:199` takes the voxel world to size the grid; becomes `new(capacity)`),
`gpu/brain/` (4 028 lines: `mod.rs`, `ablation.rs`, `observation.rs`, `screen.rs`,
tests), `gpu/slow_brain.rs`, `gpu/cortex_sheet.rs`, `gpu/fly_map/` (3 328),
`gpu/lifetime.rs` (16 heritable lifetime genes, 512-f32 resident state),
`gpu/hippocampus.rs` (sector memory), `gpu/readback.rs`, `gpu/timing.rs`,
`gpu/state_snapshot.rs`, `gpu/wave_record.rs`, `gpu/sensitivity.rs`,
`gpu/contribution.rs`, `gpu/dof.rs`, `gpu/screen_brain.rs`,
`gpu/wgsl_static_tests.rs` (module list shrinks with the kernels).

### 1.3 Genome and evolution

`gpu/evolution.rs` (`Genome`, `EvoParams`, `MutationProfile`, tournament +
elitism, ancestry proof `prove_mutation_ancestry`, `immutable_genome_sha256`),
`gpu/genome_io.rs` (**the** positional serializer; it imports `BodyGenes` and
`OrganGenes` — see §2.4), `evolution/dna_genesis.rs` (default KAN knots, genetics
constants), `rng.rs`, `config.rs` (brain/genetics blocks; the vision block shrinks),
`engine.rs` (`SovereignEngine`, host copy of one being's weights), `kan.rs`,
`cfc.rs`, `layers.rs`, `float_layers.rs`, `dense_layer.rs`, `neuro.rs`,
`tau_bank.rs`, `brain_capacity.rs` (participation ratio), `brain/format.rs`
(`.brain` container), `math/` (1 830, no GPU/Bevy), `analysis/` (1 474),
`perception/channels.rs` (registry — rewritten for the MC wave, §2.1),
`gate/`, `safety/sentinel.rs`, `contracts/`.

### 1.4 gpu_evolve modules that carry over

`brain_backend.rs`, `genome_io.rs` (re-export), `genome_diagnostics.rs`,
`competence_archive.rs` (elite archive), `research_contract.rs`,
`body_evolution.rs` (generic coevolution driver), `statics.rs` (flag table —
trimmed), `cli_flags.rs` + tests (registry pattern; taxi flags deleted),
`types.rs`, `escada.rs` (curriculum ladder — mechanism, not world),
`visual_telemetry.rs`, `audit.rs` (σ_between test), `survival.rs`.

### 1.5 Tools and scripts

Bins: `genoma_view`, `genoma_dump`, `genome_export`, `sinapses`, `cortex_test`,
`cortex_map`, `cortex_evolve`, `slow_probe`, `slow_gpu_equiv`, `lifetime_gpu_check`,
`fly_map_view`, `sensitivity`, `impact`, `screen`. Scripts:
`check_native_flow_gpu.py` (receipt), `run_research_battery.py` (RUNS_ROOT,
provenance), `publish_research_run.py`, `audit_taxi_learning.py` (renamed, the
generation-count/provenance logic is generic), `check_language.sh`,
`file_size_meter.sh`, `contract_bench.py` (judge harness, §2.5), `language_allowlist.txt`.

Docs: `ARCHITECTURE.md` (§1–3 brain/genome, rewritten tick diagram),
`docs/STYLE.md`, `CONTRIBUTING.md`, `NAMING.md`, `docs/VALIDATION-WORKFLOW.md`,
`docs/COEVOLUTION-RUNTIME.md`, the school judge *concept* (`docs/UNIVERSE-SCHOOL*.md`),
`docs/experiments/` as history.

## 2. ADAPT — the two world contracts and their neighbours

### 2.1 Sensory input: farol frame → 192-channel wave

Today `perception.wgsl` (956) + `perception_temporal` (503) + `perception_motion`
(323) + `retina_digest` (349) build the wave from a raycast retina and the voxel
grid on the GPU. In Minecraft the world lives in the Java server; the observation
arrives on the host as a **farol frame** (WebSocket :8765, 5 frames/s at
`tick_every=4`): 21-byte header (`tick u64`, `world_time u32`, `weather u8`,
`n_entities u16`) + 50 bytes per entity (`id`, `type`, `x y z`, `yaw pitch`,
`health`, `max_health`, `food`, `chunk_x/z`, `light`, `on_ground`, `target_id`,
`flags` player/villager/monster/onFire/sneaking/swimming/leashed/baby), plus
JSON events `damage`, `death`, `block_break`, `block_place`, `item_pickup`, `trade`.
`mc-agent` additionally defines a 105-float named signal vector
(`cortex/signals.py`, groups vitals/position/threats/resources/inventory/milestones)
and the `SignalsSchema` versioning rule (append only, `version++`).

So the perception stage becomes a **host encoder** (`perception/mc_wave.rs`,
new): frame + events + inventory → `[i8; 192]` per being, uploaded with the
existing `queue.write_buffer(&self.buf_wave, …)` (`brain/mod.rs:1295`). The
kernel side needs nothing: `buf_wave` is already the only sensory binding of
`brain.wgsl` (`@binding(1)`). The channel registry `perception/channels.rs`
(`N_CHANNELS = 192`, `CHANNEL_FAMILIES`, `range_i8`, `formula_doc`) is rewritten
with MC families; a first cut that respects the `SignalsSchema` groups:

| Family (proposed) | Channels | Source in the frame |
|---|---|---|
| vitals | 0–8 | health/max, food, light, on_ground, on_fire, swimming, weather, day phase |
| self pose | 8–16 | y (depth band), yaw/pitch sin-cos, speed from Δpos, chunk edge distance |
| nearest hostile ×4 | 16–48 | bearing, elevation, distance ramp, type class, is_targeting_me |
| nearest villager/player ×2 | 48–64 | same 8-slot pattern |
| local voxel ring | 64–128 | 8 bearings × 8 bands from `mc-agent` voxel observation (needs a farol extension or the Mineflayer body) |
| inventory / milestones | 128–160 | counts of log, planks, cobblestone, coal, iron, food, tool tier, milestone bits |
| memory / drives | 160–176 | hippocampus sector bias, hunger urge, threat EMA, night flag |
| events last window | 176–192 | damage taken, block broken, item picked, death (decayed) |

**Open question for the planning doc:** the frame today has no blocks. Either
farol gains a `voxel_ring` record (Java, ~150 lines in `FrameEncoder.java`) or the
body runs on Mineflayer (`mc-agent/bot`) which already has voxel/inventory
observation. This decides whether channels 64–160 exist in v1.

### 2.2 Motor output: intent → farol / mc-agent action

`organism_action.wgsl` (162) maps `intent[96]` + slow + heads to steer/throttle/
brake/gear and writes physics. In MC the actuator is a **command**: farol
`walk_to/look_at` (continuous-ish, pathfinder) or `mc-agent` `Action`
(`move dx dz`, `goto`, `mine`, `dig_down`, `tunnel`, `craft`, `place`, `smelt`,
`attack`, `eat`, `equip`, `look`, `skill`). Host side (`native_action.rs`, 423 →
`mc_action.rs`): the 3 continuous heads become `look {yaw,pitch}` + `move {dx,dz}`;
the discrete vocabulary is a softmax head over `ACTION_VOCAB` fed from L2 or the
slow-brain outputs. Keep the arbitration/clamp structure, drop the gearbox and
tyre lines. The kernel is dropped: at 5 Hz per being, host mapping is cheaper
than a dispatch + readback.

### 2.3 Life, metabolism, fitness

`gpu/organism.rs` + `organism.wgsl` (226) model energy/water/material/thermal
for a car body. In MC the game already meters health, food, fire, drowning and
death. Keep the **life gene block** (`lifetime.rs`, plasticity/eligibility/age)
and the `organism_feedback` adapter; replace the body ledger with the game's
values (health, food, xp, inventory delta). `organism_feedback.wgsl` money and
task-progress baselines (`LIFE_AUX_MONEY_OFFSET` 400, progress 401, key 402)
become reward EMA / milestone index. `gpu/body.rs` and `gpu/organs.rs` gene
blocks are removed from the genome (§2.4).

### 2.4 Genome format bump

`genome_io.rs` is positional; it embeds `BodyGenes` (16 f32, engine/brake/
cooling/…) and `OrganGenes` (taxi organs). Removing them or replacing them with
MC body genes (reach, run speed cost, night vision, …) requires a version bump
and a converter for the archive champions (`--carregar` of an old `.bin` must
fail loudly, not read shifted floats — the trap at `gpu/CLAUDE.md` "one canonical
genome parser"). Test to keep: `cargo test --bin gpu_evolve` round-trip.

### 2.5 Judge and evidence

School judge v3 (16 shared starts, mean, held-out starts) is the right *shape*
for MC: N shared seeds/spawn points, elite scored on unseen ones. `school.rs`
(host) and `contract_bench.py` keep the harness; the pass criterion changes from
"station reached" to milestone events from the farol event stream (`item_pickup
oak_log`, `block_break stone`, …). `school.wgsl` (416), `taxi_witness.wgsl` (158)
and the city lane rule go. `check_native_flow_gpu.py` TESTS/FAULTS lists shrink
to the kept kernels. Trace-v1 producer (`--trace` in `context.rs`) keeps the
schema; the sensor names in `lab-ui/contract` change with §2.1.

### 2.6 gpu_evolve binary

`main.rs` (8 887) and `context.rs` (3 655) are mostly world orchestration: track
loading, `reload_world`, laps, waypoints, pilots, wall probes, fuel, tyre, city
scenario. What survives is the generation loop (spawn → tick loop → fitness →
`evolve_with_profile` → manifest with provenance), the archive, the flag
registry and the trace writer — an estimated 1 500–2 500 lines. The tick loop's
inner body changes from `physics_a→b→c` to `recv frame → encode wave → brain
dispatch → read intent → send commands`, gated by the 5 Hz frame clock instead
of the 64 Hz physics clock. Population per server: farol tracks ≤ a few hundred
entities in radius 8 chunks; a first target is 32–64 beings on one server.

## 3. REMOVE — track/vehicle world

| Group | Lines | Items |
|---|---:|---|
| Bevy visualizer | ~9 000 | `src/main.rs` (595), `src/simulation/` (4 959), `ui.rs`, `ui_menu.rs`, `visualization.rs`, `scenery.rs`, `prefabs.rs`, `mesh.rs`, `resources.rs`, `evolution/pipeline.rs`/`spawn.rs`/`telemetry.rs`/`logs.rs` (Bevy path), deps `bevy`, `bevy_atmosphere`, `bevy_hanabi` and all `bevy_*` features |
| Voxel world + tracks | ~4 500 | `world.rs` (2 145), `world_spec.rs`, `pista_gen.rs` (504), `pista_gen_spec.rs`, `pistas.rs`, `track_export`, `pista_check`, `pistas` bins, `vision_geometry.rs`, `visao.rs` |
| Vehicle physics | ~6 500 | `physics.rs` (862), `tire_thermal.rs` (609), `gpu/physics.rs`, `physics.wgsl` (764, unused), `physics_a/b/c.wgsl` (1 107), `physics_phase_*.wgsl` + `physics_power_*.wgsl` + `physics_minimal.wgsl` (2 156, `fisica_bisect` fragments), `fisica_bisect.rs` (1 003), `gpu/organism_action.wgsl`, `native_action.rs` after the port |
| GPU raycast perception | ~3 300 | `perception.wgsl`, `perception_temporal.wgsl`, `perception_motion.wgsl`, `retina.wgsl`, `retina_geometry.wgsl`, `retina_digest.wgsl`, `gpu/perception.rs` (host raycast/pack), `gpu/retina*.rs`, `perception/retina.rs`, `perception/retina_map.rs`, `retina_map_export`, `visao_check`, `sensor_audit` |
| Taxi / city / school world | ~5 000 | `gpu/taxi.rs` + `taxi.wgsl` (421), `taxi_witness`, `taxi_routes`, `traffic.rs` + `traffic.wgsl` + `traffic_contacts.wgsl`, `organism_city.rs` (979), `city_family.rs`, `organism_scenario.rs`, `organism_school.rs` (906), `school.wgsl`, `organism_sensors.wgsl`, `organism_orientation.wgsl`, `organism_reproduction.wgsl`, `crates/taxi-world` (7 913) |
| gpu_evolve taxi modules | ~7 000 | `driving_diagnostics.rs`, `piloto_forja.rs`, `piloto_linear.rs`, `pilots.rs`, `laps.rs` + tests, `starting_grid.rs`, `service_course.rs`, `cross_city_course.rs`, `wall_probes.rs`, `reverse_controller.rs`, `fantasma.rs`, `painel.rs`, `pistas.rs`, `organism_tick.rs`, `organism_checkpoint.rs`, `school_batch.rs`, `ecology_runtime.rs`, `objective_cost.rs`, `diagnostic_candidates.rs` |
| Replay / report tools | ~7 500 | `replay.rs` (597), `replay_*` bins (3 061), `gpu_report.rs` (2 768), `gpu_loop.rs` (1 711), `gpu_validate.rs` (2 136 — CPU vs GPU parity of physics/retina; the brain parity part moves into `--ignored` lib tests), `world_census`, `lab.rs`, `dna_*` + `generic_dna` + `dump_dna` bins (1 550) and the `iara_genesis` workspace member (only `dna_compress`/`generic_dna` use it) |
| Library CPU mechanics | ~3 500 | `environment.rs`, `metabolism.rs`, `ecology.rs`, `predator.rs`, `chaos.rs`, `reality.rs`, `substrate.rs`, `repominer.rs`, `specialist.rs`, `resonance.rs`, `adaptive_curriculum.rs`, `memory.rs`. **Caveat:** `engine.rs` (`SovereignEngine`) imports `metabolism`, `resonance`, `specialist`, `chaos`, `safety`, `gate` and `perception::RetinaPerception` (`engine.rs:1-10, 259-266, 344-349`); the CPU engine is the trinity-swarm host path and must be trimmed to the weight container the GPU brain actually uploads before these files go |
| Integration tests | 200 K | `tests/*.rs` (all link 41 bins; already superseded by lib tests) |

Vendored crates: `vendor/iara-perception` (used by `engine.rs` `RetinaPerception`
→ drop with the retina) and `vendor/iara-cfc` (keep), `iara-gate`, `iara-byte`,
`iara-hdc` (check `Cargo.toml` — only two are declared; the others are dead dirs).

## 4. Order of work on the new branch

1. Cut the branch from `organism/school-20260922` after the planning doc.
2. Delete §3 groups one at a time, building `--bin gpu_evolve` after each
   (`nice -n 10 env CARGO_BUILD_JOBS=1`), fixing imports as they surface.
   Expected to shrink `Cargo.toml` to `wgpu, pollster, bytemuck, rand, serde,
   serde_json, sha2, bitflags, iara-cfc, naga (dev)`.
3. Genome version bump (§2.4) with a loud failure on old files; round-trip test.
4. `perception/mc_wave.rs` encoder + registry rewrite, unit-tested on recorded
   farol frames (a fixture recorded from `farol_client.py`, stored under
   `/media/llm-data/mc-farol/fixtures/`).
5. Farol client in Rust (`tokio-tungstenite` or blocking `tungstenite`), decoding
   header + `[u8;50]` records with `bytemuck`; commands as JSON. Handshake
   validates `frame_entity_size == 50`, `version == 1`.
6. `mc_action.rs` mapping + one smoke generation on the real server (GPU brain
   dispatch, 8 beings, 1 minute), `check_native_flow_gpu.py` receipt regenerated
   with the reduced kernel list.
7. Judge: milestone events, N spawn seeds, held-out spawns; `contract_bench.py`
   repointed.

## 5. Risks and open points

- **mc-farol has no git repository.** It must be initialised (or moved under
  `mc-lab/` git) before the branch depends on it; the jar and `config.yml` are
  part of the experiment provenance.
- **Frame lacks blocks and inventory.** v1 brain sees entities, light, time,
  vitals only; milestone tasks (wood → tools → diamond) need the voxel ring and
  inventory from either a farol extension or the Mineflayer body. Decide in the
  planning doc.
- **Clock mismatch.** 5 Hz frames vs 64 Hz brain ticks: run the brain once per
  frame (CfC `dt` = 0.2 s) or sub-tick with the last frame held. The CfC τ ranges
  (`sensory_tau_range`) were tuned for 64 Hz and need a paired check.
- **Population and RAM.** Server 1.9 GiB, GPU brain for 64 beings is small, but
  the machine has 15 GB and froze twice this week; keep one server, ≤ 64 beings,
  no Bevy.
- **Determinism is gone.** Minecraft ticks are not reproducible across runs;
  the "same seed replicates" gates (`--same-start`, judge shared starts) become
  "same world seed + same spawn" with variance reported, never assumed zero.
- **GPU is still stopped by the owner.** Nothing in this document was measured
  on the GPU; sizes are `wc -l`, contracts are read from source.
