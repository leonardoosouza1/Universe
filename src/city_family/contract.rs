//! Metadata document, plan assembly and the static contract every family city
//! must pass before a generation starts on it. Host geometry only.

use serde_json::{Value, json};

use super::{
    ARM_NAMES, CITY_FAMILY_SCHEMA_VERSION, CityLayout, ROAD_WIDTH_CELLS, RoadPiece, RouteStats,
    detect_junctions, rasterize, road_at, route_stats, seed_split, spawn_slots_generic,
    station_records, traffic_lights,
};
use crate::organism_city::CityPlan;
use crate::world::VoxelWorld;

/// Build the shared metadata document for a rasterised layout.
pub fn metadata(
    world: &VoxelWorld,
    layout: &CityLayout,
    light_metadata: Vec<Value>,
    stats: &RouteStats,
) -> Value {
    let cell = world.cell_size;
    let half_cells = ROAD_WIDTH_CELLS / 2;
    let roads = layout
        .pieces
        .iter()
        .map(|(id, piece)| match *piece {
            RoadPiece::Straight { a, b } => json!({
                "id": id, "kind": "straight", "from_m": a, "to_m": b,
                "length_m": piece.length_m(), "width_m": ROAD_WIDTH_CELLS as f32 * cell, "lanes": 2
            }),
            RoadPiece::Arc {
                centre,
                radius,
                start_rad,
                sweep_rad,
            } => json!({
                "id": id, "kind": "arc", "centre_m": centre, "radius_m": radius,
                "start_rad": start_rad, "sweep_rad": sweep_rad, "from_m": piece.start_point(),
                "to_m": piece.end_point(), "length_m": piece.length_m(),
                "width_m": ROAD_WIDTH_CELLS as f32 * cell, "lanes": 2
            }),
        })
        .collect::<Vec<_>>();
    let junctions = layout
        .junctions
        .iter()
        .enumerate()
        .map(|(index, j)| {
            let arms = ARM_NAMES
                .iter()
                .zip(j.arms.iter())
                .filter(|(_, present)| **present)
                .map(|(name, _)| *name)
                .collect::<Vec<_>>();
            json!({"id": format!("j{index}"), "x_m": j.centre[0], "z_m": j.centre[1],
                "kind": j.kind().name(), "arms": arms})
        })
        .collect::<Vec<_>>();
    let stations = layout
        .stations
        .iter()
        .enumerate()
        .map(|(index, s)| {
            json!({"id": format!("station-{index}"), "x_m": s[0], "z_m": s[1],
            "role": if index == 0 { "depot" } else { "station" }})
        })
        .collect::<Vec<_>>();
    json!({
        "schema_version": CITY_FAMILY_SCHEMA_VERSION,
        "size": world.size,
        "cell_size_m": cell,
        "road_width_cells": ROAD_WIDTH_CELLS,
        "extent_cells": [
            [(layout.extent_m[0][0] / cell).round() as i32 - half_cells, (layout.extent_m[0][1] / cell).round() as i32 - half_cells],
            [(layout.extent_m[1][0] / cell).round() as i32 + half_cells + 1, (layout.extent_m[1][1] / cell).round() as i32 + half_cells + 1]
        ],
        "lane_axes": {"x": "east_west", "z": "north_south"},
        "lane_rule": "two-way; the lane on the right-hand side of travel direction t (dot(offset, (-t.z, t.x)) > 0 in the x/z map frame) flows +t; junction boxes carry zero flow",
        "roads": roads,
        "junctions": junctions,
        "stations": stations,
        "traffic_lights": light_metadata,
        "spawn": {"x_m": world.start_pos.x, "z_m": world.start_pos.z, "safe": true},
        "route_stats": {
            "min_station_pair_road_m": stats.min_pair_road_m,
            "max_station_pair_road_m": stats.max_pair_road_m,
            "cross_city_capable_ordered_pairs": stats.cross_city_capable_pairs,
            "unreachable_ordered_pairs": stats.unreachable_pairs
        },
        "variant": {
            "family": layout.family,
            "seed": layout.seed,
            "split": seed_split(layout.seed),
            "signature": layout.signature.clone(),
            "generator": layout.generator.clone(),
            "limitations": "cardinal straights joined by circular arcs; junctions only where cardinal straights cross; no ramps, block grammar or elevation; surface grip does not vary by material"
        }
    })
}

/// Assemble the plan after rasterisation. Shared by every family.
pub fn finish_plan(world: &mut VoxelWorld, layout: &CityLayout) -> Result<CityPlan, String> {
    rasterize(world, layout);
    let stations = station_records(layout);
    crate::organism_scenario::validate_city(world, &stations)?;
    let (lights, light_metadata) = traffic_lights(layout);
    let stats = route_stats(world, &stations)?;
    if stats.unreachable_pairs > 0 {
        return Err(format!(
            "{} station pairs are unreachable by road",
            stats.unreachable_pairs
        ));
    }
    let metadata = metadata(world, layout, light_metadata, &stats);
    Ok(CityPlan::from_parts(stations, lights, metadata))
}

/// Static contract of a family city, checked before the first generation.
/// Host geometry only; it ticks no vehicle and runs no model.
pub fn validate(world: &VoxelWorld, plan: &CityPlan, n: usize) -> Result<(), String> {
    let metadata = plan.metadata();
    if metadata.get("schema_version").and_then(Value::as_str) != Some(CITY_FAMILY_SCHEMA_VERSION) {
        return Err("family city metadata has the wrong schema version".into());
    }
    if crate::organism_city::is_orthogonal_grid_city(world) {
        return Err(
            "family city rasterised into an orthogonal grid; the grid path owns that geometry"
                .into(),
        );
    }
    crate::organism_scenario::validate_city(world, &plan.stations)?;
    let stats = route_stats(world, &plan.stations)?;
    if stats.unreachable_pairs > 0 {
        return Err(format!(
            "{} station pairs unreachable",
            stats.unreachable_pairs
        ));
    }
    if stats.min_pair_road_m < 60.0 {
        return Err(format!(
            "station pair road distance {:.1} m below the 60 m service minimum",
            stats.min_pair_road_m
        ));
    }
    let declared = metadata
        .get("junctions")
        .and_then(Value::as_array)
        .ok_or("family city metadata is missing junctions")?;
    let detected = detect_junctions(world);
    if detected.len() != declared.len() {
        return Err(format!(
            "detected {} junctions from road cells; metadata declares {}",
            detected.len(),
            declared.len()
        ));
    }
    let mut expected_lights = 0usize;
    for (j, d) in detected.iter().zip(declared.iter()) {
        let x = d
            .get("x_m")
            .and_then(Value::as_f64)
            .ok_or("junction without x_m")? as f32;
        let z = d
            .get("z_m")
            .and_then(Value::as_f64)
            .ok_or("junction without z_m")? as f32;
        if (j.centre[0] - x).abs() > world.cell_size || (j.centre[1] - z).abs() > world.cell_size {
            return Err(format!(
                "junction at ({x:.1}, {z:.1}) not recovered from road cells: detected ({:.1}, {:.1})",
                j.centre[0], j.centre[1]
            ));
        }
        let arms = d
            .get("arms")
            .and_then(Value::as_array)
            .map(|a| a.len())
            .unwrap_or(0);
        if arms != j.arm_count() {
            return Err(format!(
                "junction at ({x:.1}, {z:.1}) declares {arms} arms; road cells show {}",
                j.arm_count()
            ));
        }
        expected_lights += arms;
    }
    let lights = plan.traffic_lights();
    if lights.len() != expected_lights {
        return Err(format!(
            "family city has {} traffic lights; expected {expected_lights}",
            lights.len()
        ));
    }
    if lights.iter().any(|light| {
        !light
            .position
            .iter()
            .chain(light.approach.iter())
            .all(|v| v.is_finite())
            || !light.half_width.is_finite()
            || light.half_width <= 0.0
            || (light.approach[0] * light.approach[0] + light.approach[1] * light.approach[1] - 1.0)
                .abs()
                > 0.0001
            || light.green_ticks == 0
            || light.yellow_ticks == 0
    }) {
        return Err("family city contains an invalid traffic light".into());
    }
    for (index, station) in plan.stations.iter().enumerate() {
        let col = (station[0] / world.cell_size).round() as i32;
        let row = (station[1] / world.cell_size).round() as i32;
        let footprint = (-1..=1).all(|dr| (-1..=1).all(|dc| road_at(world, col + dc, row + dr)));
        if !footprint {
            return Err(format!(
                "station {index} does not have a 3×3 road footprint"
            ));
        }
    }
    spawn_slots_generic(world, &plan.stations, n)
        .map(|_| ())
        .map_err(|e| format!("family city spawn contract failed: {e}"))
}
