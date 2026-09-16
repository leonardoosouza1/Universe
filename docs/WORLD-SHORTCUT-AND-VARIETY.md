# World request: optional shortcut at tight V curves, and more world variety (16/09/2026)

## Why

Batteries B3–B7 (`docs/PREDICOES-CEREBRO.md`) show the same failure outside the training
track: the being reaches a tight hairpin at 23–30 m/s, hits the outer wall, then stays
stuck until life runs out (54/60 held-out evaluations). With a fixed start the population
memorises one track; overfitting is the default outcome, not an accident.

Leonardo's direction: when a being masters a track, the world must grow (bigger tracks,
more curves, more difficulty); when it stagnates, help it move on. And at a very tight V
curve there could sometimes be a **shortcut**: a population that has overfitted the curve
must discover and evaluate the alternative — someone eventually tries it. The question is
whether evolution finds and keeps a better route when the world changes under it.

What already exists and will be used first (no kernel change): `--switch-on-complete N`
(track switch on population mastery, 25 % floor), `--tracks` rotation, `--random-start`,
`--ladder` (fuel / damage / weather / noise / ghost rungs with hysteresis), `--fuel-tank`,
intentional holes (on by default), `--stag-life-drain`, `--retina-range`.

## Requested: shortcut at tight V curves

1. **Flag** `--shortcuts P` (off by default): at generation of a track, each curve whose
   minimum radius is ≤ `--shortcut-max-radius R` (default 12 m) gets, with probability P
   and deterministically from the track seed, a straight asphalt connector between the
   entry and exit legs of the V.
2. **Geometry:** connector width `--shortcut-width W` (default 8 m, narrower than the
   28 m track), optional lip of loose surface (lower grip) so the shortcut is faster but
   not free. It must not create an opening that confuses the existing "open side"
   geometry more than the V already does (B6x: stops concentrate where legs merge).
3. **Progress accounting (critical):** progress/lap/sector counters must credit the
   shortcut as the arc length of the curve it bypasses, never more; `get_track_progress`
   currently takes the nearest segment without visit order (see the comment near the
   `--switch-on-complete` promotion in `main.rs`), so a connector could alias progress.
   Real laps (`voltas_reais`, ordered sectors) must stay correct.
4. **Telemetry:** per generation `[shortcut] curve@arc taken k/N beings`; in
   `driving.csv` a per-being count of shortcut uses; the replay marks the connector.
5. **`pista_check`:** prints the connectors and refuses a connector whose axis leaves
   asphalt (same P1.8 rule).
6. **Acceptance:** `--shortcuts 0` is byte-identical to today (driving.csv, population,
   replay); with `--shortcuts 1` a scripted straight-line driver through the connector
   gets exactly the bypassed arc credited; lap counts of a normal driver unchanged.

## Requested: world variety knobs (only if missing)

- Hole density `--holes-density D` (today holes are on/off via `--no-holes`).
- Track generator length/curvature targets for a "harder pool": `--track-min-curves`,
  `--track-length-m` — or a documented seed census that the lab can pick from
  (`neuromod-lab/scripts/track_census.sh` already lists radius and defects per seed).

## Not asked

Changing the brain, fitness, or the existing ladder semantics.
