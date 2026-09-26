//! Deterministic long-form service course on the existing city station map.
//!
//! This module only selects immutable road geometry. The GPU Delivery lesson
//! still owns pickup, route progress, and dropoff acceptance.

use serde_json::json;
use sha2::{Digest, Sha256};
use universe::gpu::school::{Lesson, LessonContract, LessonGeometry};
use universe::organism_school::{lesson_start_candidates, service_lesson_candidates};
use universe::world::VoxelWorld;

const REQUIRED_STARTS: usize = 4;
const MIN_ROAD_DISTANCE_M: f32 = 300.0;
const MIN_EUCLIDEAN_DISTANCE_M: f32 = 200.0;
const MIN_AXIS_DISPLACEMENT_M: f32 = 60.0;

pub fn build(
    world: &VoxelWorld,
    stations: &[[f32; 8]],
) -> Result<super::service_course::ServiceCourse, String> {
    if stations.len() < 2 {
        return Err("cross-city course requires at least two city stations".into());
    }
    let contract = LessonContract::default();
    let candidates = service_lesson_candidates(world, stations, Lesson::Delivery, &contract);
    let pickup_starts = lesson_start_candidates(world, stations, Lesson::Pickup, &contract);
    let routes = universe::gpu::taxi_routes::pack_distance_fields(world, stations)
        .map_err(|error| format!("cross-city route preprocessing failed: {error}"))?;
    let cells = world.size * world.size;
    let mut selected = Vec::with_capacity(REQUIRED_STARTS);

    for mut geometry in candidates {
        if selected.len() == REQUIRED_STARTS {
            break;
        }
        let Some(origin) = station_index(stations, geometry.target) else {
            continue;
        };
        let Some(destination) = station_index(stations, geometry.exit) else {
            continue;
        };
        let origin_cell = station_cell(world, geometry.target)?;
        let route_distance = routes[4 + destination * cells + origin_cell];
        let dx = (geometry.exit[0] - geometry.target[0]).abs();
        let dz = (geometry.exit[1] - geometry.target[1]).abs();
        if !route_distance.is_finite()
            || route_distance < MIN_ROAD_DISTANCE_M
            || dx.hypot(dz) < MIN_EUCLIDEAN_DISTANCE_M
            || dx < MIN_AXIS_DISPLACEMENT_M
            || dz < MIN_AXIS_DISPLACEMENT_M
        {
            continue;
        }
        let Some((start, forward)) = pickup_starts.iter().find_map(|pickup| {
            let start_cell = station_cell(world, pickup.start).ok()?;
            let start_route = routes[4 + origin * cells + start_cell];
            let forward_cell = forward_cell(world, pickup.start, pickup.forward).ok()?;
            let forward_route = routes[4 + origin * cells + forward_cell];
            if !start_route.is_finite()
                || !forward_route.is_finite()
                || forward_route < 0.0
                || forward_route >= start_route - 0.001
            {
                return None;
            }
            let start_dx = (geometry.target[0] - pickup.start[0]).abs();
            let start_dz = (geometry.target[1] - pickup.start[1]).abs();
            (start_route >= MIN_ROAD_DISTANCE_M
                && start_dx.hypot(start_dz) >= MIN_EUCLIDEAN_DISTANCE_M
                && start_dx >= MIN_AXIS_DISPLACEMENT_M
                && start_dz >= MIN_AXIS_DISPLACEMENT_M
                && selected
                    .iter()
                    .all(|existing: &LessonGeometry| existing.start != pickup.start))
            .then_some((pickup.start, pickup.forward))
        }) else {
            continue;
        };
        geometry.start = start;
        geometry.forward = forward;
        geometry.depot_distance_m =
            (start[0] - world.start_pos.x).hypot(start[1] - world.start_pos.z);
        if selected.iter().any(|existing: &LessonGeometry| {
            existing.target == geometry.target && existing.exit == geometry.exit
        }) {
            continue;
        }
        selected.push(geometry);
    }

    if selected.len() != REQUIRED_STARTS {
        return Err(format!(
            "city has only {} deterministic cross-city delivery starts; need exactly {} meeting route >= {:.0} m, displacement >= {:.0} m, and both axis differences >= {:.0} m",
            selected.len(),
            REQUIRED_STARTS,
            MIN_ROAD_DISTANCE_M,
            MIN_EUCLIDEAN_DISTANCE_M,
            MIN_AXIS_DISPLACEMENT_M,
        ));
    }

    let pairs = selected
        .iter()
        .enumerate()
        .map(|(index, geometry)| {
            let origin = station_index(stations, geometry.target).expect("validated origin");
            let destination = station_index(stations, geometry.exit).expect("validated destination");
            let origin_cell = station_cell(world, geometry.target).expect("validated origin cell");
            let route_distance = routes[4 + destination * cells + origin_cell];
            json!({
                "id": format!("cross-city-delivery-{index}"),
                "origin_station_id": format!("station-{origin}"),
                "destination_station_id": format!("station-{destination}"),
                "start_position_m": geometry.start,
                "forward": geometry.forward,
                "start_cell": station_cell(world,geometry.start).expect("validated start"),
                "forward_cell": forward_cell(world,geometry.start,geometry.forward).expect("validated forward cell"),
                "forward_route_m": routes[4+origin*cells+forward_cell(world,geometry.start,geometry.forward).expect("validated forward cell")],
                "origin_position_m": geometry.target,
                "destination_position_m": geometry.exit,
                "road_distance_m": route_distance,
                "pickup_road_distance_m": routes[4+origin*cells+station_cell(world,geometry.start).expect("validated start")],
                "pickup_euclidean_displacement_m": (geometry.target[0]-geometry.start[0]).hypot(geometry.target[1]-geometry.start[1]),
                "pickup_axis_displacement_m": [(geometry.target[0]-geometry.start[0]).abs(),(geometry.target[1]-geometry.start[1]).abs()],
                "euclidean_displacement_m": (geometry.exit[0] - geometry.target[0]).hypot(geometry.exit[1] - geometry.target[1]),
                "axis_displacement_m": [
                    (geometry.exit[0] - geometry.target[0]).abs(),
                    (geometry.exit[1] - geometry.target[1]).abs()
                ],
            })
        })
        .collect::<Vec<_>>();
    let mut hash = Sha256::new();
    hash.update(b"explicit-school-cross-city-v2:strict-forward-distance-decrease");
    hash.update((world.size as u32).to_le_bytes());
    hash.update(world.cell_size.to_bits().to_le_bytes());
    for cell in &world.grid {
        hash.update([*cell as u8]);
    }
    for flow in &world.flow {
        hash.update(flow.x.to_bits().to_le_bytes());
        hash.update(flow.y.to_bits().to_le_bytes());
    }
    hash.update(serde_json::to_vec(&pairs).map_err(|error| error.to_string())?);
    let geometry_sha256 = format!("{:x}", hash.finalize());
    let metadata = json!({
        "schema_version": 1,
        "course_id": "explicit-school-cross-city-v2",
        "geometry_sha256": geometry_sha256,
        "coverage_claim": "four distinct starts and station pairs, not four compass headings or all city stations",
        "course_type": "cross_city_empty_pickup_delivery",
        "lesson": Lesson::Delivery.name(),
        "claim_scope": "Four deterministic long-form service tasks; learned city competence requires behavioral GPU validation.",
        "required_starts": REQUIRED_STARTS,
        "constraints": {
            "minimum_road_distance_m": MIN_ROAD_DISTANCE_M,
            "minimum_euclidean_displacement_m": MIN_EUCLIDEAN_DISTANCE_M,
            "minimum_axis_displacement_m": MIN_AXIS_DISPLACEMENT_M,
        },
        "starts_empty": true,
        "initial_heading_contract": "cardinal forward adjacent road cell strictly reduces distance to assigned pickup by >0.001m; no runtime steering override",
        "pairs": pairs,
        "station_count": stations.len(),
    });

    Ok(super::service_course::ServiceCourse {
        stations: stations.to_vec(),
        starts: selected,
        metadata,
    })
}

fn station_index(stations: &[[f32; 8]], point: [f32; 2]) -> Option<usize> {
    stations
        .iter()
        .position(|station| station[0] == point[0] && station[1] == point[1])
}

fn station_cell(world: &VoxelWorld, point: [f32; 2]) -> Result<usize, String> {
    let col = (point[0] / world.cell_size).round();
    let row = (point[1] / world.cell_size).round();
    if !col.is_finite()
        || !row.is_finite()
        || col < 0.0
        || row < 0.0
        || col >= world.size as f32
        || row >= world.size as f32
    {
        return Err("cross-city station is outside route grid".into());
    }
    Ok(row as usize * world.size + col as usize)
}

fn forward_cell(world: &VoxelWorld, start: [f32; 2], forward: [f32; 2]) -> Result<usize, String> {
    let cardinal = (forward[0].abs() == 1.0 && forward[1] == 0.0)
        || (forward[1].abs() == 1.0 && forward[0] == 0.0);
    if !cardinal {
        return Err("cross-city spawn heading must be cardinal".into());
    }
    let cell = station_cell(
        world,
        [
            start[0] + forward[0] * world.cell_size,
            start[1] + forward[1] * world.cell_size,
        ],
    )?;
    if world.grid[cell] != universe::world::VoxelType::Road {
        return Err("cross-city forward cell is not road".into());
    }
    Ok(cell)
}
