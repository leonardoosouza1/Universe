# Organism city geometry

`organism_city::build_city` replaces the former two-cross overlay with a
deterministic orthogonal street grid. Road cells, lane axes, intersections,
station coordinates, and traffic-light phases share the same cell coordinates.
The function updates static `VoxelWorld` geometry and returns the existing taxi
station ABI plus versioned metadata. It does not run vehicle dynamics or choose
actions.
