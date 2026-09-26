//! Deterministic static geometry for the short full-service delivery course.
//!
//! This module only chooses road cells and declares the lesson geometry.  The
//! Delivery lesson remains responsible for the runtime approach, pickup, route,
//! and dropoff sequence on the GPU.

use serde_json::{Value, json};
use universe::gpu::school::{Lesson, LessonGeometry};
use universe::organism_city::street_centres_m;
use universe::organism_scenario::validate_city;
use universe::world::{VoxelType, VoxelWorld};

const APPROACH_DISTANCE_M: f32 = 6.0;
const LANE_OFFSET_M: f32 = 3.0;
const STATION_MIN_SEPARATION_M: f32 = 8.0;
const INTERSECTION_CLEARANCE_M: f32 = 9.0;
const BODY_HALF_LENGTH_M: f32 = 2.25;
const BODY_HALF_WIDTH_M: f32 = 1.55;
const SWEEP_STEP_M: f32 = 1.5;

const CARDINALS: [([f32; 2], &str); 4] = [
    ([1.0, 0.0], "east"),
    ([0.0, 1.0], "south"),
    ([-1.0, 0.0], "west"),
    ([0.0, -1.0], "north"),
];

pub struct ServiceCourse {
    pub stations: Vec<[f32; 8]>,
    pub starts: Vec<LessonGeometry>,
    pub metadata: Value,
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    direction: [f32; 2],
    direction_name: &'static str,
    origin: [f32; 2],
    destination: [f32; 2],
    start: [f32; 2],
}

/// Build the artificial short straight full-trip course on the already-built city.
///
/// `trip_distance_m` is a road distance in metres.  It is deliberately kept in
/// the same 3 m lattice as the city so that the static route check and the GPU
/// route field agree exactly.
pub fn build(world: &VoxelWorld, trip_distance_m: f32) -> Result<ServiceCourse, String> {
    validate_trip_distance(trip_distance_m)?;

    let mut centres = street_centres_m(world);
    centres.sort_by(f32::total_cmp);
    centres.dedup_by(|a, b| (*a - *b).abs() < 0.001);
    if centres.len() < 3 {
        return Err(format!(
            "service course requires at least three city streets; found {}",
            centres.len()
        ));
    }

    let candidates = collect_candidates(world, &centres, trip_distance_m);
    let mut selected = Vec::new();
    let mut directions_found = [false; CARDINALS.len()];

    // One distinct pair per cardinal direction keeps every generation's K=4
    // batch tied to the same four directional tasks.
    for (direction_index, (direction, _direction_name)) in CARDINALS.iter().enumerate() {
        if let Some(candidate) = candidates.iter().copied().find(|candidate| {
            candidate.direction == *direction && distinct_from_selected(candidate, &selected)
        }) {
            directions_found[direction_index] = true;
            selected.push(candidate);
        }
    }
    if selected.len() != CARDINALS.len() {
        return Err(format!(
            "city has only {} distinct valid cardinal short delivery task pairs; need exactly 4",
            selected.len()
        ));
    }

    let mut stations = Vec::with_capacity(selected.len() * 2);
    let mut starts = Vec::with_capacity(selected.len());
    let mut pair_metadata = Vec::with_capacity(selected.len());
    for (pair_index, candidate) in selected.iter().enumerate() {
        let origin_station = station(candidate.origin);
        let destination_station = station(candidate.destination);
        let origin_station_index = stations.len();
        stations.push(origin_station);
        let destination_station_index = stations.len();
        stations.push(destination_station);

        let geometry = LessonGeometry {
            lesson: Lesson::Delivery,
            start: candidate.start,
            forward: candidate.direction,
            target: candidate.origin,
            exit: candidate.destination,
            exit_dir: candidate.direction,
            wrong_exit: [0.0, 0.0],
            depot_distance_m: (candidate.origin[0] - world.start_pos.x)
                .hypot(candidate.origin[1] - world.start_pos.z),
        };
        starts.push(geometry);
        pair_metadata.push(json!({
            "id": format!("delivery-pair-{pair_index}"),
            "origin_station_id": format!("station-{origin_station_index}"),
            "destination_station_id": format!("station-{destination_station_index}"),
            "start_position_m": candidate.start,
            "origin_position_m": candidate.origin,
            "destination_position_m": candidate.destination,
            "forward": candidate.direction,
            "direction": candidate.direction_name,
            "distance_m": trip_distance_m,
        }));
    }

    // Keep the course contract self-contained: all station resources are from
    // this city, all pair endpoints are road cells, and every pair has a finite
    // four-neighbor road route.  No vehicle, controller, or GPU state is touched.
    validate_city(world, &stations)
        .map_err(|error| format!("service course city validation failed: {error}"))?;
    let packed_routes = universe::gpu::taxi_routes::pack_distance_fields(world, &stations)
        .map_err(|error| format!("service course route preprocessing failed: {error}"))?;
    let cells = world.size * world.size;
    for (pair_index, geometry) in starts.iter().enumerate() {
        validate_geometry(world, geometry)
            .map_err(|error| format!("delivery pair {pair_index} invalid: {error}"))?;
        let origin_cell = station_cell(world, geometry.target).ok_or_else(|| {
            format!("delivery pair {pair_index} origin is outside the route grid")
        })?;
        let destination_cell = 2 * pair_index + 1;
        let route = packed_routes[4 + destination_cell * cells + origin_cell];
        if !route.is_finite() || route < 0.0 {
            return Err(format!(
                "delivery pair {pair_index} has no reachable station route"
            ));
        }
        if (route - trip_distance_m).abs() > world.cell_size * 0.5 {
            return Err(format!(
                "delivery pair {pair_index} route is {route:.1} m, expected {trip_distance_m:.1} m"
            ));
        }
    }

    let attempted = CARDINALS.iter().map(|(_, name)| *name).collect::<Vec<_>>();
    let found = CARDINALS
        .iter()
        .enumerate()
        .filter_map(|(index, (_, name))| directions_found[index].then_some(*name))
        .collect::<Vec<_>>();
    let metadata = json!({
        "schema_version": 1,
        "course_id": "artificial-short-straight-full-trip-v1",
        "course_type": "artificial_short_straight_full_trip",
        "lesson": "delivery",
        "trip_distance_m": trip_distance_m,
        "claim_scope": "This artificial short straight full-trip course makes no general-city competence claim.",
        "directions_attempted": attempted,
        "directions_selected": found,
        "station_count": stations.len(),
        "stations": stations
            .iter()
            .enumerate()
            .map(|(index, station)| json!({
                "id": format!("station-{index}"),
                "position_m": [station[0], station[1]],
                "resources": {"energy": station[2], "water": station[3], "material": station[4]},
            }))
            .collect::<Vec<_>>(),
        "pairs": pair_metadata,
        "static_validation": {
            "road_cells": true,
            "swept_body_footprint": true,
            "legal_lane_flow": true,
            "four_neighbor_bfs": true,
            "interaction_disc_separation_m": STATION_MIN_SEPARATION_M,
        },
    });

    Ok(ServiceCourse {
        stations,
        starts,
        metadata,
    })
}

fn validate_trip_distance(distance: f32) -> Result<(), String> {
    if !distance.is_finite()
        || !(12.0..=96.0).contains(&distance)
        || (distance / 3.0).fract().abs() > 0.0001
    {
        return Err("service trip distance must be a finite 3 m multiple in 12..=96 m".into());
    }
    Ok(())
}

fn collect_candidates(world: &VoxelWorld, centres: &[f32], distance: f32) -> Vec<Candidate> {
    let cell = world.cell_size;
    if !cell.is_finite() || cell <= 0.0 {
        return Vec::new();
    }
    let lower = centres[0] + 4.0 * cell;
    let upper = centres[centres.len() - 1] - 4.0 * cell - distance;
    if lower > upper {
        return Vec::new();
    }
    let mut candidates = Vec::new();
    for (direction, direction_name) in CARDINALS {
        let horizontal = direction[0].abs() > 0.5;
        for &line in centres {
            let mut along = lower;
            while along <= upper + 0.001 {
                let origin_along = if direction[0] + direction[1] > 0.0 {
                    along
                } else {
                    along + distance
                };
                let destination_along = if direction[0] + direction[1] > 0.0 {
                    along + distance
                } else {
                    along
                };
                let origin = if horizontal {
                    [origin_along, line]
                } else {
                    [line, origin_along]
                };
                let destination = if horizontal {
                    [destination_along, line]
                } else {
                    [line, destination_along]
                };
                if clear_of_intersections(centres, origin, horizontal)
                    && clear_of_intersections(centres, destination, horizontal)
                {
                    let lateral = lane_offset(direction);
                    let start = [
                        origin[0] - direction[0] * APPROACH_DISTANCE_M + lateral[0],
                        origin[1] - direction[1] * APPROACH_DISTANCE_M + lateral[1],
                    ];
                    let candidate = Candidate {
                        direction,
                        direction_name,
                        origin,
                        destination,
                        start,
                    };
                    if validate_candidate(world, &candidate).is_ok() {
                        candidates.push(candidate);
                    }
                }
                along += cell;
            }
        }
    }
    candidates
}

fn clear_of_intersections(centres: &[f32], point: [f32; 2], horizontal: bool) -> bool {
    let along = if horizontal { point[0] } else { point[1] };
    centres
        .iter()
        .all(|centre| (along - *centre).abs() >= INTERSECTION_CLEARANCE_M)
}

fn distinct_from_selected(candidate: &Candidate, selected: &[Candidate]) -> bool {
    let points = [candidate.origin, candidate.destination];
    selected.iter().all(|other| {
        let other_points = [other.origin, other.destination];
        points.iter().all(|point| {
            other_points.iter().all(|other_point| {
                (point[0] - other_point[0]).hypot(point[1] - other_point[1])
                    > STATION_MIN_SEPARATION_M
            })
        })
    })
}

fn station(point: [f32; 2]) -> [f32; 8] {
    [
        point[0],
        point[1],
        50_000_000.0,
        200.0,
        500.0,
        0.0,
        0.0,
        0.0,
    ]
}

fn validate_candidate(world: &VoxelWorld, candidate: &Candidate) -> Result<(), String> {
    if !is_road(world, candidate.origin) || !is_road(world, candidate.destination) {
        return Err("station endpoint is not a road cell".into());
    }
    let flow = flow_at(world, candidate.start).ok_or("approach start has no lane flow")?;
    if (flow[0] - candidate.direction[0]).abs() > 0.01
        || (flow[1] - candidate.direction[1]).abs() > 0.01
    {
        return Err("approach start is not in the requested legal lane".into());
    }
    if !swept_footprint_is_road(world, candidate) {
        return Err("start-to-destination swept footprint leaves the road".into());
    }
    Ok(())
}

fn validate_geometry(world: &VoxelWorld, geometry: &LessonGeometry) -> Result<(), String> {
    let candidate = Candidate {
        direction: geometry.forward,
        direction_name: "selected",
        origin: geometry.target,
        destination: geometry.exit,
        start: geometry.start,
    };
    if geometry.lesson != Lesson::Delivery || geometry.exit_dir != geometry.forward {
        return Err("geometry is not a straight Delivery task".into());
    }
    validate_candidate(world, &candidate)
}

fn lane_offset(direction: [f32; 2]) -> [f32; 2] {
    match direction {
        [1.0, 0.0] => [0.0, -LANE_OFFSET_M],
        [-1.0, 0.0] => [0.0, LANE_OFFSET_M],
        [0.0, 1.0] => [-LANE_OFFSET_M, 0.0],
        [0.0, -1.0] => [LANE_OFFSET_M, 0.0],
        _ => [0.0, 0.0],
    }
}

fn swept_footprint_is_road(world: &VoxelWorld, candidate: &Candidate) -> bool {
    let first_length =
        (candidate.origin[0] - candidate.start[0]).hypot(candidate.origin[1] - candidate.start[1]);
    let second_length = (candidate.destination[0] - candidate.origin[0])
        .hypot(candidate.destination[1] - candidate.origin[1]);
    let sample_segment = |from: [f32; 2], to: [f32; 2], length: f32| {
        let samples = (length / SWEEP_STEP_M).ceil() as usize;
        (0..=samples).all(|index| {
            let t = if samples == 0 {
                0.0
            } else {
                index as f32 / samples as f32
            };
            let centre = [
                from[0] + (to[0] - from[0]) * t,
                from[1] + (to[1] - from[1]) * t,
            ];
            [-BODY_HALF_LENGTH_M, 0.0, BODY_HALF_LENGTH_M]
                .iter()
                .all(|longitudinal| {
                    [-BODY_HALF_WIDTH_M, 0.0, BODY_HALF_WIDTH_M]
                        .iter()
                        .all(|lateral| {
                            let point = [
                                centre[0] + candidate.direction[0] * *longitudinal
                                    - candidate.direction[1] * *lateral,
                                centre[1]
                                    + candidate.direction[1] * *longitudinal
                                    + candidate.direction[0] * *lateral,
                            ];
                            is_road(world, point)
                        })
                })
        })
    };
    sample_segment(candidate.start, candidate.origin, first_length)
        && sample_segment(candidate.origin, candidate.destination, second_length)
}

fn is_road(world: &VoxelWorld, point: [f32; 2]) -> bool {
    let col = (point[0] / world.cell_size).round() as isize;
    let row = (point[1] / world.cell_size).round() as isize;
    if col < 0 || row < 0 || col >= world.size as isize || row >= world.size as isize {
        return false;
    }
    world.grid[row as usize * world.size + col as usize] == VoxelType::Road
}

fn flow_at(world: &VoxelWorld, point: [f32; 2]) -> Option<[f32; 2]> {
    let col = (point[0] / world.cell_size).round() as isize;
    let row = (point[1] / world.cell_size).round() as isize;
    if col < 0 || row < 0 || col >= world.size as isize || row >= world.size as isize {
        return None;
    }
    let flow = world.flow[row as usize * world.size + col as usize];
    (flow.length_squared() > 0.5).then_some([flow.x, flow.y])
}

fn station_cell(world: &VoxelWorld, point: [f32; 2]) -> Option<usize> {
    let col = (point[0] / world.cell_size).round() as isize;
    let row = (point[1] / world.cell_size).round() as isize;
    if col < 0 || row < 0 || col >= world.size as isize || row >= world.size as isize {
        None
    } else {
        Some(row as usize * world.size + col as usize)
    }
}
