//! Continuous planar distance probes for the existing CPU sensory upload.
//!
//! This optional path replaces coarse sample bins with voxel-boundary traversal.
//! Front probes retain hole avoidance; lateral probes detect walls only. A miss
//! returns the finite sensing horizon, without a separate discontinuous sentinel.

/// Distance in metres to an obstacle, capped at `horizon_m`.
/// Cells are centred at integer multiples of `cell_size_m`, as in `VoxelWorld`.
pub fn distance(
    grid: &[u32],
    size: usize,
    cell_size_m: f32,
    position: [f32; 2],
    direction: [f32; 2],
    horizon_m: f32,
    detect_holes: bool,
) -> f32 {
    assert!(size > 0 && grid.len() == size * size);
    assert!(cell_size_m.is_finite() && cell_size_m > 0.0);
    assert!(horizon_m.is_finite() && horizon_m >= 0.0);
    assert!(
        position
            .iter()
            .chain(direction.iter())
            .all(|v| v.is_finite())
    );
    let magnitude = direction[0].hypot(direction[1]);
    if magnitude <= 1e-6 {
        return horizon_m;
    }
    let ray = [direction[0] / magnitude, direction[1] / magnitude];
    let mut cell = [
        (position[0] / cell_size_m).round() as i32,
        (position[1] / cell_size_m).round() as i32,
    ];
    let obstacle = |c: [i32; 2]| {
        if c.iter().any(|&x| x < 0 || x >= size as i32) {
            return true;
        }
        let value = grid[c[1] as usize * size + c[0] as usize];
        value == 2 || (detect_holes && value == 3)
    };
    if obstacle(cell) {
        return 0.0;
    }
    let mut step = [0; 2];
    let mut next = [f32::INFINITY; 2];
    let mut delta = [f32::INFINITY; 2];
    for axis in 0..2 {
        if ray[axis].abs() > 1e-6 {
            step[axis] = if ray[axis] > 0.0 { 1 } else { -1 };
            let boundary = (cell[axis] as f32 + 0.5 * step[axis] as f32) * cell_size_m;
            next[axis] = ((boundary - position[axis]) / ray[axis]).max(0.0);
            delta[axis] = cell_size_m / ray[axis].abs();
        }
    }
    // Every crossing advances at least one cell. Leaving the bounded grid hits a
    // wall, so 2*size+2 steps suffice even for a corner-to-corner diagonal.
    for _ in 0..2 * size + 2 {
        let crossing = next[0].min(next[1]);
        if crossing > horizon_m {
            return horizon_m;
        }
        // At an exact corner, enter the diagonal cell; zero-length tangencies do
        // not count as a surface segment occupied by the ray.
        for axis in 0..2 {
            if next[axis] == crossing {
                cell[axis] += step[axis];
                next[axis] += delta[axis];
            }
        }
        if obstacle(cell) {
            return crossing;
        }
    }
    horizon_m
}

pub fn sample(
    grid: &[u32],
    size: usize,
    cell_size_m: f32,
    position: [f32; 2],
    forward: [f32; 2],
) -> [f32; 3] {
    let left = [forward[1], -forward[0]];
    let right = [-forward[1], forward[0]];
    [
        distance(grid, size, cell_size_m, position, left, 40.0, false),
        distance(grid, size, cell_size_m, position, right, 40.0, false),
        distance(grid, size, cell_size_m, position, forward, 50.0, true),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_velocity_gives_continuous_approach_without_bin_spikes() {
        let mut grid = vec![0; 100 * 100];
        for z in 0..100 {
            grid[z * 100 + 40] = 2;
        }
        let dt = 1.0 / 64.0;
        for i in 0..128 {
            let x = 90.0 + i as f32 * 10.0 * dt;
            let before = distance(&grid, 100, 3.0, [x, 120.0], [1.0, 0.0], 50.0, true);
            let after = distance(
                &grid,
                100,
                3.0,
                [x + 10.0 * dt, 120.0],
                [1.0, 0.0],
                50.0,
                true,
            );
            assert!(((before - after) / dt - 10.0).abs() < 1e-3);
        }
    }

    #[test]
    fn left_right_symmetry_and_hole_semantics_are_preserved() {
        let mut grid = vec![0; 100 * 100];
        grid[40 * 100 + 35] = 2;
        grid[40 * 100 + 45] = 2;
        grid[45 * 100 + 40] = 3;
        assert_eq!(
            sample(&grid, 100, 3.0, [120.0, 120.0], [0.0, 1.0]),
            [13.5, 13.5, 13.5]
        );
        assert_eq!(
            distance(&grid, 100, 3.0, [120.0, 120.0], [0.0, 1.0], 40.0, false),
            40.0
        );
    }

    #[test]
    fn cell_boundaries_and_horizon_have_no_extra_sentinel_jump() {
        let mut grid = vec![0; 100 * 100];
        grid[40 * 100 + 40] = 2;
        assert_eq!(
            distance(&grid, 100, 3.0, [120.0, 120.0], [1.0, 0.0], 50.0, true),
            0.0
        );
        assert_eq!(
            distance(&grid, 100, 3.0, [-1.5, 120.0], [1.0, 0.0], 50.0, true),
            0.0
        );
        assert_eq!(
            distance(&grid, 100, 3.0, [67.5, 120.0], [1.0, 0.0], 50.0, true),
            50.0
        );
        assert!(
            (distance(&grid, 100, 3.0, [68.6, 120.0], [1.0, 0.0], 50.0, true) - 49.9).abs() < 1e-4
        );
    }
}
