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
/// Lane centre for a travel direction, following the city's lane rule
/// (`organism_city::lane_rule`): the legacy grid mirrors horizontal streets, the
/// consistent rule puts every street on the ring side. Candidates are checked
/// against the built flow field afterwards, so a mismatch drops starts silently
/// (the 2026-09-25 lane bench lost every horizontal start that way).
fn lane_offset_for(dir: [f32; 2], lane: f32) -> [f32; 2] {
    let horizontal_sign = match crate::organism_city::lane_rule() {
        crate::organism_city::LaneRule::Legacy => -1.0,
        crate::organism_city::LaneRule::Consistent => 1.0,
    };
    match (dir[0] > 0.5, dir[0] < -0.5, dir[1] > 0.5) {
        (true, _, _) => [0.0, horizontal_sign * lane],
        (_, true, _) => [0.0, -horizontal_sign * lane],
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

/// Stable identity of an unordered station pair. Reversing a passenger trip must
/// never move it between train, validation and test folds.
pub fn service_pair_fold_key(a: usize, b: usize) -> usize {
    let (low, high) = if a < b { (a, b) } else { (b, a) };
    high
        .checked_mul(high.saturating_sub(1))
        .and_then(|value| value.checked_div(2))
        .and_then(|value| value.checked_add(low))
        .expect("station-pair key overflow")
}

fn station_approach(
    world: &VoxelWorld,
    station: [f32; 2],
    before_m: f32,
    selector: usize,
) -> Option<([f32; 2], [f32; 2])> {
    const DIRECTIONS: [[f32; 2]; 4] = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]];
    let lane = lane_offset_m(world);
    for offset_index in 0..DIRECTIONS.len() {
        let dir = DIRECTIONS[(selector + offset_index) % DIRECTIONS.len()];
        let offset = lane_offset_for(dir, lane);
        let start = [
            station[0] + offset[0] - dir[0] * before_m,
            station[1] + offset[1] - dir[1] * before_m,
        ];
        if !is_road_block(world, start[0], start[1]) {
            continue;
        }
        if flow_at(world, start[0], start[1]) != Some(dir) {
            continue;
        }
        if straight_is_road(world, start, dir, before_m) {
            return Some((start, dir));
        }
    }
    None
}

/// Deterministic production-station tasks for lessons 5–8. One canonical
/// orientation is emitted per unordered pair, so reversed trips cannot leak into
/// another fold. Runtime route lookup remains in the GPU taxi kernel; this host
/// pass only validates immutable map geometry at generation setup.
pub fn service_lesson_candidates(
    world: &VoxelWorld,
    stations: &[[f32; 8]],
    lesson: Lesson,
    _contract: &LessonContract,
) -> Vec<LessonGeometry> {
    if !lesson.is_service() || stations.len() < 2 {
        return Vec::new();
    }
    let packed = match crate::gpu::taxi_routes::pack_distance_fields(world, stations) {
        Ok(packed) => packed,
        Err(_) => return Vec::new(),
    };
    let cells = world.size * world.size;
    let cell_index = |station: &[f32; 8]| {
        let col = (station[0] / world.cell_size).round() as usize;
        let row = (station[1] / world.cell_size).round() as usize;
        row * world.size + col
    };
    let route_distance = |from: usize, to: usize| {
        packed[4 + to * cells + cell_index(&stations[from])]
    };
    let requires_turn = matches!(lesson, Lesson::NavigateWithPassenger | Lesson::Delivery);
    let before_m = lesson.start_before_junction_m();
    let depot = [world.start_pos.x, world.start_pos.z];
    let mut candidates = Vec::new();

    for low in 0..stations.len() {
        for high in low + 1..stations.len() {
            let key = service_pair_fold_key(low, high);
            // The orientation is stable but not biased toward one station index.
            let (origin, destination) = if key & 1 == 0 { (low, high) } else { (high, low) };
            let target = [stations[origin][0], stations[origin][1]];
            let exit = [stations[destination][0], stations[destination][1]];
            let forward_distance = route_distance(origin, destination);
            let reverse_distance = route_distance(destination, origin);
            if forward_distance < 60.0 || reverse_distance < 60.0 {
                continue;
            }
            if requires_turn
                && ((target[0] - exit[0]).abs() <= world.cell_size
                    || (target[1] - exit[1]).abs() <= world.cell_size)
            {
                continue;
            }
            let Some((start, forward)) = station_approach(world, target, before_m, key) else {
                continue;
            };
            if stations.iter().enumerate().any(|(index, station)| {
                index != origin
                    && (station[0] - start[0]).hypot(station[1] - start[1])
                        < SPAWN_STATION_CLEARANCE_M
            }) {
                continue;
            }
            let exit_dir = station_approach(world, exit, before_m, key.rotate_left(1))
                .map(|(_, direction)| direction)
                .unwrap_or(forward);
            candidates.push((
                key,
                LessonGeometry {
                    lesson,
                    start,
                    forward,
                    target,
                    exit,
                    exit_dir,
                    wrong_exit: [0.0, 0.0],
                    depot_distance_m: (target[0] - depot[0]).hypot(target[1] - depot[1]),
                },
            ));
        }
    }
    candidates.sort_by_key(|(key, _)| *key);
    candidates.into_iter().map(|(_, geometry)| geometry).collect()
}

/// Which third of the deterministic candidate space a run may start from
/// (`--school-start-fold`). Motor candidate `i` belongs to `i % 3`; service
/// candidates use their stable unordered station-pair key, so filtering a lesson
/// or reversing a trip cannot leak the pair into another fold.
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

fn candidate_fold_index(
    stations: &[[f32; 8]],
    lesson: Lesson,
    list_index: usize,
    geometry: &LessonGeometry,
) -> usize {
    if !lesson.is_service() {
        return list_index;
    }
    let station_index = |point: [f32; 2]| {
        stations
            .iter()
            .position(|station| station[0] == point[0] && station[1] == point[1])
    };
    let origin = station_index(geometry.target)
        .expect("service lesson target must be a production station");
    let destination = station_index(geometry.exit)
        .expect("service lesson exit must be a production station");
    service_pair_fold_key(origin, destination)
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
        .filter(|(i, geometry)| fold.keeps(candidate_fold_index(stations, lesson, *i, geometry)))
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

/// Bounded pool for replicated evaluation. Approach/pickup identity excludes the
/// unused destination; identical physical exams cannot cross folds or count twice.
/// Legacy callers keep `lesson_starts_fold` and its original candidate numbering.
pub fn lesson_start_pool(
    world: &VoxelWorld,
    stations: &[[f32; 8]],
    lesson: Lesson,
    contract: &LessonContract,
    requested: usize,
    fold: LessonFold,
) -> Result<Vec<LessonGeometry>, String> {
    let approach_only = matches!(lesson, Lesson::ApproachPassenger | Lesson::Pickup);
    let mut seen = std::collections::HashSet::new();
    let mut pool = Vec::new();
    for (index, geometry) in lesson_start_candidates(world, stations, lesson, contract).into_iter().enumerate() {
        let identity = [geometry.start[0].to_bits(), geometry.start[1].to_bits(),
            geometry.forward[0].to_bits(), geometry.forward[1].to_bits()];
        let fold_index = if approach_only {
            if !seen.insert(identity) { continue; }
            // Stable integer mixing; independent of list order and destination.
            let hash = identity.iter().fold(2166136261u32, |hash, value| {
                value.to_le_bytes().iter().fold(hash, |h, byte| (h ^ *byte as u32).wrapping_mul(16777619))
            });
            hash as usize
        } else { candidate_fold_index(stations, lesson, index, &geometry) };
        if fold.keeps(fold_index) { pool.push(geometry); }
    }
    if pool.is_empty() { return Err(format!("no distinct starts for {} in {}", lesson.name(), fold.name())); }
    pool.truncate(requested.max(1));
    Ok(pool)
}

/// How the lesson starts of one generation are assigned (`--school-shared-start`).
/// Default (`None` at the call site): one candidate per slot, slots rotated by hash, so
/// one genome is judged on a different approach every generation and its score swings
/// with the geometry (docs/SCHOOL-STOP-SHAPING-20260922.md §6.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SharedStart {
    /// Candidate `generation mod count` of the fold: the population sees every start
    /// over the generations while all beings of one generation share it.
    Rotate,
    /// Candidate 0 of the fold in every generation.
    Fixed,
}

impl SharedStart {
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "rotate" => Some(Self::Rotate),
            "fixed" => Some(Self::Fixed),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Rotate => "shared_rotate",
            Self::Fixed => "shared_fixed",
        }
    }

    pub fn pick(self, generation: usize) -> usize {
        match self {
            Self::Rotate => generation,
            Self::Fixed => 0,
        }
    }
}

/// One lesson start for the whole generation: every being gets candidate `pick mod count`
/// of the fold. Returns the starts, the candidate index used and the fold's count. The
/// elite re-judged on the same candidate must score identically (fixed-slot witness,
/// docs/SCHOOL-STOP-SHAPING-20260922.md §6.7).
pub fn lesson_shared_start_fold(
    world: &VoxelWorld,
    stations: &[[f32; 8]],
    lesson: Lesson,
    contract: &LessonContract,
    n: usize,
    fold: LessonFold,
    pick: usize,
) -> Result<(Vec<LessonGeometry>, usize, usize), String> {
    let candidates: Vec<LessonGeometry> = lesson_start_candidates(world, stations, lesson, contract)
        .into_iter()
        .enumerate()
        .filter(|(i, geometry)| fold.keeps(candidate_fold_index(stations, lesson, *i, geometry)))
        .map(|(_, g)| g)
        .collect();
    if candidates.is_empty() {
        return Err(format!("city has no lesson approach for {} in fold {}", lesson.name(), fold.name()));
    }
    let index = pick % candidates.len();
    Ok((vec![candidates[index]; n], index, candidates.len()))
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
    if lesson.is_service() {
        return service_lesson_candidates(world, stations, lesson, contract);
    }
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
    fn lesson_ids_names_and_service_contract_remain_stable() {
        let cases = [
            (Lesson::Move, 1, "move", false, false),
            (Lesson::Stop, 2, "stop", false, false),
            (Lesson::TurnNear, 3, "turn_near", false, false),
            (Lesson::TurnFar, 4, "turn_far", false, false),
            (Lesson::ApproachPassenger, 5, "approach_passenger", true, false),
            (Lesson::Pickup, 6, "pickup", true, false),
            (Lesson::NavigateWithPassenger, 7, "navigate_with_passenger", true, true),
            (Lesson::Delivery, 8, "delivery", true, false),
        ];
        assert_eq!(Lesson::ALL.len(), cases.len());
        for (index, (lesson, id, name, service, onboard)) in cases.into_iter().enumerate() {
            assert_eq!(Lesson::ALL[index], lesson);
            assert_eq!(lesson.id(), id);
            assert_eq!(lesson.name(), name);
            assert_eq!(Lesson::parse(name), Some(lesson));
            assert_eq!(lesson.is_service(), service);
            assert_eq!(lesson.starts_onboard(), onboard);
        }
    }

    #[test]
    fn service_pairs_are_reachable_long_and_fold_safe() {
        let (world, stations) = city();
        let contract = LessonContract::default();
        let packed = crate::gpu::taxi_routes::pack_distance_fields(&world, &stations).unwrap();
        let cells = world.size * world.size;
        let station_index = |point: [f32; 2]| {
            stations
                .iter()
                .position(|station| station[0] == point[0] && station[1] == point[1])
                .expect("lesson endpoint must be a production station")
        };
        let distance_to = |from: usize, to: usize| {
            let col = (stations[from][0] / world.cell_size).round() as usize;
            let row = (stations[from][1] / world.cell_size).round() as usize;
            packed[4 + to * cells + row * world.size + col]
        };

        let mut pair_folds = std::collections::BTreeMap::new();
        for lesson in [
            Lesson::ApproachPassenger,
            Lesson::Pickup,
            Lesson::NavigateWithPassenger,
            Lesson::Delivery,
        ] {
            let candidates = service_lesson_candidates(&world, &stations, lesson, &contract);
            assert!(!candidates.is_empty(), "{} has no candidates", lesson.name());
            let mut lesson_pairs = std::collections::BTreeSet::new();
            for (candidate_index, geometry) in candidates.iter().enumerate() {
                let origin = station_index(geometry.target);
                let destination = station_index(geometry.exit);
                assert_ne!(origin, destination);
                let forward = distance_to(origin, destination);
                let reverse = distance_to(destination, origin);
                assert!(forward >= 60.0 && reverse >= 60.0, "{origin}->{destination}: {forward}/{reverse}");
                assert!(is_road_block(&world, geometry.start[0], geometry.start[1]));
                assert_eq!(flow_at(&world, geometry.start[0], geometry.start[1]), Some(geometry.forward));
                let key = service_pair_fold_key(origin, destination);
                assert_eq!(key, service_pair_fold_key(destination, origin));
                let fold = key % 3;
                assert!(lesson_pairs.insert(key), "station pair duplicated within {}", lesson.name());
                if let Some(previous) = pair_folds.insert(key, fold) {
                    assert_eq!(previous, fold, "station pair changed fold between lessons");
                }
                let selected = match fold {
                    0 => LessonFold::Train,
                    1 => LessonFold::Validation,
                    _ => LessonFold::Test,
                };
                assert!(selected.keeps(candidate_fold_index(
                    &stations,
                    lesson,
                    candidate_index,
                    geometry
                )));
                if matches!(lesson, Lesson::NavigateWithPassenger | Lesson::Delivery) {
                    assert!(
                        (geometry.target[0] - geometry.exit[0]).abs() > world.cell_size
                            && (geometry.target[1] - geometry.exit[1]).abs() > world.cell_size,
                        "navigation pair must require at least one turn"
                    );
                }
            }
            let mut counts = [0usize; 3];
            for geometry in &candidates {
                let key = service_pair_fold_key(
                    station_index(geometry.target),
                    station_index(geometry.exit),
                );
                counts[key % 3] += 1;
            }
            for (fold, expected) in [
                LessonFold::Train,
                LessonFold::Validation,
                LessonFold::Test,
            ]
            .into_iter()
            .zip(counts)
            {
                assert!(expected > 0, "{} has an empty {} fold", lesson.name(), fold.name());
                let selected = lesson_starts_fold(
                    &world,
                    &stations,
                    lesson,
                    &contract,
                    expected,
                    fold,
                )
                .unwrap();
                assert_eq!(selected.len(), expected);
            }
        }
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
    fn shared_start_gives_every_being_the_same_candidate_and_wraps() {
        let (world, stations) = city();
        let contract = LessonContract::default();
        let train = lesson_starts_fold(&world, &stations, Lesson::Move, &contract, 12, LessonFold::Train).unwrap();
        let (starts, index, count) =
            lesson_shared_start_fold(&world, &stations, Lesson::Move, &contract, 12, LessonFold::Train, 5).unwrap();
        assert_eq!(starts.len(), 12);
        assert_eq!(index, 5);
        assert!(count >= 12);
        assert!(starts.iter().all(|g| *g == train[5]));
        let (wrapped, wrapped_index, _) =
            lesson_shared_start_fold(&world, &stations, Lesson::Move, &contract, 12, LessonFold::Train, count + 2).unwrap();
        assert_eq!(wrapped_index, 2);
        assert_eq!(wrapped[0], train[2]);
        assert!(lesson_shared_start_fold(&world, &stations, Lesson::Move, &contract, 0, LessonFold::Train, 0).unwrap().0.is_empty());
        assert_eq!(SharedStart::Fixed.pick(7), 0);
        assert_eq!(SharedStart::Rotate.pick(7), 7);
    }

    #[test]
    fn too_many_beings_is_an_error_not_a_reuse() {
        let (world, stations) = city();
        assert!(lesson_starts(&world, &stations, Lesson::Stop, &LessonContract::default(), 10_000).is_err());
    }
}
