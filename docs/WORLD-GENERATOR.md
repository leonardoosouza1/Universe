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

### Honestly, what each field moves (`pista_gen_spec::waypoints_for_spec`, `pista_gen_spec.rs`)

> **Correction (2026-09-17, second pass).** The first version of this table was
> measured against a generic default ("10 curves, no length target") instead of the
> seed's own default — that violated `WorldSpec`'s own contract ("key omitted =
> today's hardcoded value") and was silently changing `curves`/`length` on a
> radius-only request: `--world seed=8005,radius=9` used to come back 17 curves / 622 m
> instead of seed 8005's own 6 curves / 1639 m. Fixed by measuring the seed's default
> track (`track_waypoints_for_seed`) up front and threading it through as the search's
> target AND its floor (the search can only replace it with something that measures a
> smaller `r_min`, never something worse). The table below reflects the fix; the census
> numbers further down are the rerun with it in place.

| Field | Honoured? | How |
|---|---|---|
| `seed` | Yes, always | Only entropy source into the RNG |
| `curves` | Yes | Defaults to the seed's OWN measured curve count when omitted, else sets `tracado`'s vertex count directly |
| `length` | Yes, approximate | Defaults to the seed's OWN measured length when omitted; rejection-sampled against `medir().comprimento`, tolerance ±35% |
| `radius` | Yes, approximate, sometimes unreachable | `raio_base` scaled from the target; search keeps the closest `r_min` that still respects `curves`/`length`. Seeded with the DEFAULT geometry as its floor, so a failed search returns the seed's own default, never something wider. For seed 8005's own family (6 vertices, ~1639 m) a 20 000-attempt sweep never beat r_min≈14 m — its default (12.6 m) is already near that family's limit, so `radius=9` alone cannot be met there without also moving `curves`/`length`. Physical floor `r_min >= 6 m` always wins when reachable |
| `width` | Yes, but downstream only | NOT used by the geometry search (`medir` only needs it for `d_freio`, which this search ignores); applied to the voxel grid by `VoxelWorld::new_from_spec` (`world.rs:99`) |
| `holes` | Yes, downstream only | `world::track_holes_with_density` (`world.rs:1866`) replaces the old 1-2 coin flip with `round(density * n_sharpest_turns)` |

If no candidate reaches the requested radius in 4000 attempts, the search logs the miss
to stderr and returns the tightest candidate found — the seed's own default geometry if
nothing in the search beat it — never a silent success and never a regression past the
default (same discipline as `pista_gen::gerar`'s own rejection loop).

## Census results (`world_census`, `src/bin/world_census.rs`)

```
world_census --seeds 8100..8299 radius=9,width=20 \
  --out /media/llm-data/scratch/world-gen-smoke/census_8100_8299_v2.tsv
```

> **Correction (2026-09-17, second pass).** Rerun after the default-derivation fix
> above. The first run (46/200 valid, 23.0%) used a generic 10-curve/no-length-target
> default that does not reflect what `--world radius=9,width=20` alone now produces —
> kept below for provenance, superseded by this one.

200 seeds, **114 valid (57.0%)**, 86 invalid, 0 parse errors, run in 11.9 s CPU-side (no
GPU) — `world_census --seeds 8100..8299 radius=9,width=20`. Among the valid ones,
measured `r_min` ranges 7.5-28.0 m (mean 14.4 m): most of the 8100-8199 range is the
"curvas" profile (12-16 vertices, `FAIXA_PERFIL` 1800-2600 m), whose own default `r_min`
is already fairly tight, so keeping curves/length pinned to that default (instead of a
generic 10/no-target) lets more seeds land inside `pista_check`'s tolerance at
`width=20` than the first (buggy-default) run did. Higher validity here is a
consequence of the fix, not a re-tuned target.

Superseded first run, kept for provenance: 200 seeds, 46 valid (23.0%), 154 invalid,
0 parse errors, 5.4 s CPU-side; `r_min` 6.0-11.0 m (mean 8.7 m) among the valid ones.
Most rejections were `pista_check`-style defects (axis off track / bottleneck) at
`width=20` on curves generated for a *different*, generic-default width target — width
still does not steer the geometry search (see the table above), so a narrow request can
collide with a track shape searched for a looser one.

## Known limitation found while smoke-testing (`holes` density on a narrow/tight track)

`--world seed=8005,radius=8,width=18,holes=0.5` — the literal hard-world combination —
is REJECTED by `pista_check` at every radius from 8 to 60 m tested. Isolating each
field: `radius=8,width=18,holes=0` is ACCEPTED (0 defects); any `holes>=0.05` at
`width=18` fails; `holes=0.5` fails at every width from 18 to 60 m tested. The blocker
is `holes`, not `radius` or `width` — a real limit of `track_holes_with_density`'s
fixed 30%-of-track-width lateral offset (`world.rs`, `hole_position`): at `holes=0.5`
roughly half of a dozen-plus curves get a hazard, and on this seed's geometry that is
enough hazards near enough to apexes to create an unavoidable bottleneck somewhere on
the lap. `holes<=0.02` (rounds to ~0 real curves hazarded) passes; nothing between
0.02 and 0.5 was swept finely enough to find where it starts failing on every seed —
only confirmed for 8005. Not a search bug (a real, prior counting bug in
`track_holes_with_density` was found and fixed in the same session — see its own doc
comment — this is what remains after that fix).
