# World generator — what a seed actually controls

Owner's request (2026-09-17): "a system of different worlds, generated randomly from
seeds like Minecraft, so we can test thousands of environments." This is the current
state, not the target: today's tracks are all easy, and this doc says exactly why.

## What `--tracks <seed>` does today

`world::track_waypoints_for_seed` (`src/world.rs:1321`) routes an integer seed to one
of five generators, all producing waypoints in **cell-space** (3 m/cell,
`pista_gen::CELULA`, `src/pista_gen.rs:58`):

| Seed range | Generator | What varies with the seed | What is HARDCODED |
|---|---|---|---|
| 6 literal seeds (123, 64, 256, 512, 768, 1024) | Hand-written waypoint lists | Nothing — the seed is a lookup key | Every coordinate, `track_holes_for_seed` (`world.rs:1763`) |
| 8000-8099 | `pista_gen::gerar_perfil(seed, "curta", 28.0)` | Vertex count (6-9), radius (60-110 cells) | Width **28 m literal** (call site), length band 1200-2000 m (`FAIXA_PERFIL`, `pista_gen.rs:282`) |
| 8100-8199 | same, `"curvas"` | Vertex count (12-16), radius (95-150 cells) | Width 28 m, length band 1800-2600 m |
| 8200-8299 | same, `"retao"` | Vertex count (5-7), radius (170-260 cells) | Width 28 m, length band 2200-3200 m |
| >= 1,000,000 | `pista_gen::do_curriculo` (`pista_gen.rs:361`) | `d_freio` difficulty band via `degrau`/`variante` in the seed digits | 4 fixed bands (`DEGRAUS`, `pista_gen.rs:352`), width **14 m literal** |
| anything else | `world::procedural_waypoints` (`world.rs:1470`ish) | Layout, via `generate_track_candidate` + `validate_track` | No radius/length/curve-count knob at all |

Track **width** is a single run-wide CLI flag, `track_width: f32 = 28.0`
(`src/bin/gpu_evolve/main.rs:873`), passed once into `VoxelWorld::new(seed, track_width)`
— every track in a run shares it; a seed never changes it. **Holes** are 1-2 per track,
chosen by a seed-derived coin flip (`world::procedural_holes`, `n_holes = 1 +
(hash(seed) & 1)`) placed at the sharpest turn apexes — never a density, never zero,
never more than 2.

## `WorldSpec` — what `--world` adds (`src/world_spec.rs`)

`--world seed=8005,radius=9,curves=16,width=18,holes=0.4,length=3500` parses into a
`WorldSpec { seed, radius, curves, width, holes, length }` (all but `seed` optional —
omitted = today's hardcoded value above). `--world-file <path.json>` loads an array of
the same object for a world pool; `--world` may repeat and composes with
`--switch-on-complete`/track rotation exactly like `--tracks` (`gpu_evolve/pistas.rs`).

**A seed-only spec is byte-identical to `--tracks <seed>`** —
`world::track_waypoints_for_spec` (`world.rs:1462`) special-cases
`WorldSpec::is_seed_only()` and defers straight to `track_waypoints_for_seed`. Verified
by `world_spec_seed_only_matches_track_waypoints_for_seed` (`world.rs`, compares the
full `VoxelWorld.grid`) and by an `md5sum driving.csv` smoke run — see the report.

### Honestly, what each field moves (`pista_gen::waypoints_for_spec`, `pista_gen.rs:397`)

| Field | Honoured? | How |
|---|---|---|
| `seed` | Yes, always | Only entropy source into the RNG |
| `curves` | Yes | Sets `tracado`'s vertex count directly |
| `length` | Yes, approximate | Rejection-sampled against `medir().comprimento`, tolerance ±35% |
| `radius` | Yes, approximate | `raio_base` scaled from the target; search keeps the closest `r_min`. `tracado`'s `raio_base` sets overall track scale, not one curve — a correlation exploited by rejection, not a guarantee. Physical floor `r_min >= 6 m` always wins |
| `width` | Yes, but downstream only | NOT used by the geometry search (`medir` only needs it for `d_freio`, which this search ignores); applied to the voxel grid by `VoxelWorld::new_from_spec` (`world.rs:99`) |
| `holes` | Yes, downstream only | `world::track_holes_with_density` (`world.rs:1866`) replaces the old 1-2 coin flip with `round(density * n_sharpest_turns)` |

If no candidate clears the physical floor (`r_min>=6m`, length>=400m, span<=280 cells)
in 4000 attempts, the search logs the miss to stderr and returns its closest candidate
— never a silent success (same discipline as `pista_gen::gerar`'s own rejection loop).

## Census results (`world_census`, `src/bin/world_census.rs`)

See the report for the exact commands and numbers from `--seeds 8100..8299
radius=9,width=20`.
