//! Static lesson geometry for the organism school (`docs/SCHOOL-LESSONS-20260922.md`).
//!
//! Host-only map arithmetic on the immutable city grid: where each being starts,
//! what it must reach and which exits count. No vehicle dynamics, no controller.
//! Every start is a lane centre placed before a junction along the lane flow, and
//! the lane rule is cross-checked against `world.flow` so a change in
//! `organism_city::build_city` cannot silently move a lesson into oncoming traffic.
use crate::gpu::school::{Lesson, LessonContract, LessonGeometry};
use crate::organism_city::{street_centres_m, SPAWN_STATION_CLEARANCE_M};
use crate::world::{VoxelType, VoxelWorld};

/// Lane centre offset from the street centre line, one cell.
fn lane_offset_m(world: &VoxelWorld) -> f32 {
    world.cell_size
}

fn is_road_cell(world: &VoxelWorld, x_m: f32, z_m: f32) -> bool {
    let cell = world.cell_size;
    let size = world.size as i32;
    let col = (x_m / cell).round() as i32;
    let row = (z_m / cell).round() as i32;
    if col < 0 || row < 0 || col >= size || row >= size {
        return false;
    }
    world.grid[(row * size + col) as usize] == VoxelType::Road
}

/// The cell and its eight neighbours are road (a car body fits).
fn is_road_block(world: &VoxelWorld, x_m: f32, z_m: f32) -> bool {
    let cell = world.cell_size;
    (-1..=1).all(|dr| (-1..=1).all(|dc| is_road_cell(world, x_m + dc as f32 * cell, z_m + dr as f32 * cell)))
}

fn flow_at(world: &VoxelWorld, x_m: f32, z_m: f32) -> Option<[f32; 2]> {
    let cell = world.cell_size;
    let size = world.size as i32;
    let col = (x_m / cell).round() as i32;
    let row = (z_m / cell).round() as i32;
    if col < 0 || row < 0 || col >= size || row >= size {
        return None;
    }
    let f = world.flow[(row * size + col) as usize];
    if f.length_squared() < 0.5 {
        None
    } else {
        Some([f.x, f.y])
    }
}

/// Lane offset (perpendicular to the travel direction) for the lane whose flow is
/// `dir`, by the city rule: rows/cols at or before the centre carry +X/+Z.
fn lane_offset_for(dir: [f32; 2], lane: f32) -> [f32; 2] {
    match (dir[0] > 0.5, dir[0] < -0.5, dir[1] > 0.5) {
        (true, _, _) => [0.0, -lane],
        (_, true, _) => [0.0, lane],
        (_, _, true) => [-lane, 0.0],
        _ => [lane, 0.0],
    }
}

fn straight_is_road(world: &VoxelWorld, from: [f32; 2], dir: [f32; 2], length_m: f32) -> bool {
    let step = world.cell_size * 0.5;
    let mut d = 0.0;
    while d <= length_m {
        if !is_road_block(world, from[0] + dir[0] * d, from[1] + dir[1] * d) {
            return false;
        }
        d += step;
    }
    true
}

/// Which third of the deterministic candidate list a run may start from
/// (`--school-start-fold`). Candidate `i` belongs to fold `i % 3`, so every fold
/// spreads over the whole city instead of clustering near the depot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LessonFold {
    All,
    Train,
    Validation,
    Test,
}

impl LessonFold {
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "all" => Some(Self::All),
            "train" => Some(Self::Train),
            "validation" => Some(Self::Validation),
            "test" => Some(Self::Test),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Train => "train",
            Self::Validation => "validation",
            Self::Test => "test",
        }
    }

    fn keeps(self, index: usize) -> bool {
        match self {
            Self::All => true,
            Self::Train => index % 3 == 0,
            Self::Validation => index % 3 == 1,
            Self::Test => index % 3 == 2,
        }
    }
}

/// Deterministic lesson starts: one junction approach per being, ordered by the
/// junction's distance to the depot, then x, then z. Returns an error when the city
/// cannot host `n` distinct approaches for this lesson.
pub fn lesson_starts(
    world: &VoxelWorld,
    stations: &[[f32; 8]],
    lesson: Lesson,
    contract: &LessonContract,
    n: usize,
) -> Result<Vec<LessonGeometry>, String> {
    lesson_starts_fold(world, stations, lesson, contract, n, LessonFold::All)
}

/// `lesson_starts` restricted to one fold of the candidate list.
pub fn lesson_starts_fold(
    world: &VoxelWorld,
    stations: &[[f32; 8]],
    lesson: Lesson,
    contract: &LessonContract,
    n: usize,
    fold: LessonFold,
) -> Result<Vec<LessonGeometry>, String> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut candidates: Vec<LessonGeometry> = lesson_start_candidates(world, stations, lesson, contract)
        .into_iter()
        .enumerate()
        .filter(|(i, _)| fold.keeps(*i))
        .map(|(_, g)| g)
        .collect();
    if candidates.len() < n {
        return Err(format!(
            "city has only {} lesson approaches for {} in fold {}; requested {n}",
            candidates.len(),
            lesson.name(),
            fold.name()
        ));
    }
    candidates.truncate(n);
    Ok(candidates)
}

/// Every valid approach for the lesson in the city, in the deterministic order
/// (`lesson_starts`). The full list is the population of starts a genome can be
/// evaluated on; folds are taken from it by index.
pub fn lesson_start_candidates(
    world: &VoxelWorld,
    stations: &[[f32; 8]],
    lesson: Lesson,
    contract: &LessonContract,
) -> Vec<LessonGeometry> {
    let lane = lane_offset_m(world);
    let centres = street_centres_m(world);
    let depot = world.start_pos;
    let before = lesson.start_before_junction_m();
    let mut candidates: Vec<LessonGeometry> = Vec::new();
    for &cx in &centres {
        for &cz in &centres {
            let junction = [cx, cz];
            for dir in [[1.0f32, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]] {
                let offset = lane_offset_for(dir, lane);
                let start = [junction[0] + offset[0] - dir[0] * before, junction[1] + offset[1] - dir[1] * before];
                if !is_road_block(world, start[0], start[1]) {
                    continue;
                }
                // The lane rule must agree with the built flow field at the start.
                match flow_at(world, start[0], start[1]) {
                    Some(f) if (f[0] - dir[0]).abs() < 0.01 && (f[1] - dir[1]).abs() < 0.01 => {}
                    _ => continue,
                }
                if stations
                    .iter()
                    .any(|s| (s[0] - start[0]).hypot(s[1] - start[1]) < SPAWN_STATION_CLEARANCE_M)
                {
                    continue;
                }
                let depot_distance_m = (junction[0] - depot.x).hypot(junction[1] - depot.z);
                let geometry = if lesson.is_turn() {
                    let left = [-dir[1], dir[0]];
                    let right = [dir[1], -dir[0]];
                    let gate = |e: [f32; 2]| {
                        let o = lane_offset_for(e, lane);
                        [junction[0] + o[0] + e[0] * contract.exit_gate_distance_m, junction[1] + o[1] + e[1] * contract.exit_gate_distance_m]
                    };
                    // Near side: the outbound lane lies before the junction centre line
                    // along the approach direction (dot(offset, dir) < 0), a tight arc.
                    let near_is_left = {
                        let o = lane_offset_for(left, lane);
                        o[0] * dir[0] + o[1] * dir[1] < 0.0
                    };
                    let (near_e, far_e) = if near_is_left { (left, right) } else { (right, left) };
                    let (near_gate, far_gate) = (gate(near_e), gate(far_e));
                    if !is_road_block(world, near_gate[0], near_gate[1]) || !is_road_block(world, far_gate[0], far_gate[1]) {
                        continue;
                    }
                    // Both outbound lanes must carry the exit flow the gate assumes.
                    let flow_ok = |g: [f32; 2], e: [f32; 2]| match flow_at(world, g[0], g[1]) {
                        Some(f) => (f[0] - e[0]).abs() < 0.01 && (f[1] - e[1]).abs() < 0.01,
                        None => false,
                    };
                    if !flow_ok(near_gate, near_e) || !flow_ok(far_gate, far_e) {
                        continue;
                    }
                    if !straight_is_road(world, start, dir, before - contract.box_half_m) {
                        continue;
                    }
                    let (exit, exit_dir, wrong) = if lesson == Lesson::TurnNear {
                        (near_gate, near_e, far_gate)
                    } else {
                        (far_gate, far_e, near_gate)
                    };
                    LessonGeometry { lesson, start, forward: dir, target: junction, exit, exit_dir, wrong_exit: wrong, depot_distance_m }
                } else {
                    let target = [start[0] + dir[0] * contract.move_distance_m, start[1] + dir[1] * contract.move_distance_m];
                    if !straight_is_road(world, start, dir, contract.move_distance_m + contract.goal_radius_m) {
                        continue;
                    }
                    LessonGeometry { lesson, start, forward: dir, target, exit: target, exit_dir: dir, wrong_exit: [0.0, 0.0], depot_distance_m }
                };
                candidates.push(geometry);
            }
        }
    }
    candidates.sort_by(|a, b| {
        a.depot_distance_m
            .total_cmp(&b.depot_distance_m)
            .then(a.start[0].total_cmp(&b.start[0]))
            .then(a.start[1].total_cmp(&b.start[1]))
    });
    candidates
}

#[cfg(test)]
mod tests {
    //! Static geometry only (host arithmetic on the immutable grid), permitted by
    //! Universe/CLAUDE.md: no simulation, no CPU oracle of a kernel.
    use super::*;
    use crate::organism_city::build_city;

    fn city() -> (VoxelWorld, Vec<[f32; 8]>) {
        let mut world = VoxelWorld::new(7, 12.0);
        let plan = build_city(&mut world);
        (world, plan.stations)
    }

    #[test]
    fn move_starts_are_lane_centres_with_road_ahead() {
        let (world, stations) = city();
        let starts = lesson_starts(&world, &stations, Lesson::Move, &LessonContract::default(), 12).unwrap();
        assert_eq!(starts.len(), 12);
        for g in &starts {
            assert_eq!(flow_at(&world, g.start[0], g.start[1]).unwrap(), g.forward);
            assert!(is_road_block(&world, g.target[0], g.target[1]));
            assert!(((g.target[0] - g.start[0]).hypot(g.target[1] - g.start[1]) - 30.0).abs() < 1e-3);
        }
        let mut sorted = starts.clone();
        sorted.sort_by(|a, b| a.depot_distance_m.total_cmp(&b.depot_distance_m));
        assert_eq!(sorted.iter().map(|g| g.start).collect::<Vec<_>>(), starts.iter().map(|g| g.start).collect::<Vec<_>>());
    }

    #[test]
    fn turn_gates_are_outbound_lanes_on_opposite_sides() {
        let (world, stations) = city();
        let near = lesson_starts(&world, &stations, Lesson::TurnNear, &LessonContract::default(), 8).unwrap();
        let far = lesson_starts(&world, &stations, Lesson::TurnFar, &LessonContract::default(), 8).unwrap();
        for (a, b) in near.iter().zip(&far) {
            assert_eq!(a.start, b.start);
            assert_eq!(a.exit, b.wrong_exit);
            assert_eq!(b.exit, a.wrong_exit);
            assert_eq!(flow_at(&world, a.exit[0], a.exit[1]).unwrap(), a.exit_dir);
            assert_eq!(flow_at(&world, b.exit[0], b.exit[1]).unwrap(), b.exit_dir);
            // Exit directions are perpendicular to the approach and opposite to each other.
            assert!((a.exit_dir[0] * a.forward[0] + a.exit_dir[1] * a.forward[1]).abs() < 1e-6);
            assert!((a.exit_dir[0] + b.exit_dir[0]).abs() < 1e-6 && (a.exit_dir[1] + b.exit_dir[1]).abs() < 1e-6);
            // The near gate sits before the junction centre line along the approach.
            let near_along = (a.exit[0] - a.target[0]) * a.forward[0] + (a.exit[1] - a.target[1]) * a.forward[1];
            let far_along = (b.exit[0] - b.target[0]) * b.forward[0] + (b.exit[1] - b.target[1]) * b.forward[1];
            assert!(near_along < 0.0 && far_along > 0.0, "near {near_along} far {far_along}");
        }
    }

    #[test]
    fn folds_partition_the_candidates_without_overlap() {
        let (world, stations) = city();
        let contract = LessonContract::default();
        let all = lesson_start_candidates(&world, &stations, Lesson::Move, &contract);
        let take = |fold: LessonFold| {
            let n = all.iter().enumerate().filter(|(i, _)| fold.keeps(*i)).count();
            lesson_starts_fold(&world, &stations, Lesson::Move, &contract, n, fold).unwrap()
        };
        let (train, validation, test) = (take(LessonFold::Train), take(LessonFold::Validation), take(LessonFold::Test));
        assert_eq!(train.len() + validation.len() + test.len(), all.len());
        assert!(train.len() >= 12 && validation.len() >= 12 && test.len() >= 12, "{} {} {}", train.len(), validation.len(), test.len());
        for a in &train {
            assert!(!validation.iter().any(|b| b.start == a.start) && !test.iter().any(|b| b.start == a.start));
        }
        for a in &validation {
            assert!(!test.iter().any(|b| b.start == a.start));
        }
        assert_eq!(lesson_starts(&world, &stations, Lesson::Move, &contract, 12).unwrap(), all[..12].to_vec());
    }

    #[test]
    fn too_many_beings_is_an_error_not_a_reuse() {
        let (world, stations) = city();
        assert!(lesson_starts(&world, &stations, Lesson::Stop, &LessonContract::default(), 10_000).is_err());
    }
}
