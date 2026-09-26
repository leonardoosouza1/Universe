# Organism city geometry

`organism_city::build_city` replaces the former two-cross overlay with a
deterministic orthogonal street grid. Road cells, lane axes, intersections,
station coordinates, and traffic-light phases share the same cell coordinates.
The function updates static `VoxelWorld` geometry and returns the existing taxi
station ABI plus versioned metadata. It does not run vehicle dynamics or choose
actions.

## City families (`src/city_family`, 2026-09-25)

`--city-family ring --city-seed N` builds a rounded loop with one to three
cardinal chords instead of the grid: circular corners, T and X junctions and
stations spread over several straights, all seeded (`seed % 10 == 9` is an
evaluation seed). The family writes the same `VoxelWorld` fields, station ABI
and traffic-light ABI as the grid, with metadata `organism-city-v2`. The
reference grid (`build_city`, `grid`/`compact`/`wide`) is unchanged. Lanes in a
family city follow one right-hand rule; the grid's horizontal streets do not,
so school helpers that assume the grid rule need the flow-matching patch
recorded in `META/CLAUDE-UNIVERSE-AMBIENTE-20260925.md`. Contract and gate
status live in that file.
