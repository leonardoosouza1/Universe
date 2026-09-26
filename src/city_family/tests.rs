//! Host geometry tests for the city families. No vehicle is ticked and no GPU
//! is used: they check the static contract every family must satisfy before a
//! generation is allowed to start on it.

use super::ring;
use super::*;
use crate::organism_city::{build_city, is_orthogonal_grid_city, spawn_slots, CityFamily};
use crate::world::{VoxelType, VoxelWorld};

fn city_world() -> VoxelWorld {
    VoxelWorld::new(7, 28.0)
}

fn road(world: &VoxelWorld, x_m: f32, z_m: f32) -> bool {
    world.get_voxel_at(x_m, z_m) == VoxelType::Road
}

#[test]
fn seed_split_holds_out_one_seed_in_ten() {
    assert_eq!(seed_split(9), "eval");
    assert_eq!(seed_split(19), "eval");
    assert_eq!(seed_split(0), "train");
    assert_eq!(seed_split(8), "train");
    assert_eq!((0..100u64).filter(|s| seed_split(*s) == "eval").count(), 10);
}

#[test]
fn seeded_rng_is_deterministic_and_bounded() {
    let mut a = SeededRng::new(42);
    let mut b = SeededRng::new(42);
    for _ in 0..1000 {
        let (x, y) = (a.unit(), b.unit());
        assert_eq!(x, y);
        assert!((0.0..1.0).contains(&x));
    }
    let mut c = SeededRng::new(43);
    assert_ne!(SeededRng::new(42).next_u64(), c.next_u64());
}

#[test]
fn arc_nearest_matches_straight_geometry() {
    let arc = RoadPiece::Arc { centre: [100.0, 100.0], radius: 60.0, start_rad: 0.0, sweep_rad: std::f32::consts::FRAC_PI_2 };
    // Point on the arc at 45 degrees.
    let p = [100.0 + 60.0 * 0.5f32.sqrt(), 100.0 + 60.0 * 0.5f32.sqrt()];
    let n = arc.nearest(p);
    assert!(n.distance_m < 1e-3, "{n:?}");
    assert!((n.tangent[0] + 0.5f32.sqrt()).abs() < 1e-3 && (n.tangent[1] - 0.5f32.sqrt()).abs() < 1e-3);
    // Point outside the sweep snaps to the nearest endpoint.
    let q = [100.0, 40.0];
    let m = arc.nearest(q);
    assert!((m.foot[0] - 160.0).abs() < 1e-3 && (m.foot[1] - 100.0).abs() < 1e-3, "{m:?}");
    assert!((arc.length_m() - 60.0 * std::f32::consts::FRAC_PI_2).abs() < 1e-3);
    let s = RoadPiece::Straight { a: [0.0, 0.0], b: [10.0, 0.0] };
    let n = s.nearest([4.0, 3.0]);
    assert!((n.distance_m - 3.0).abs() < 1e-6 && n.foot == [4.0, 0.0] && n.tangent == [1.0, 0.0]);
}

#[test]
fn junction_detection_recovers_the_grid_city() {
    let _quiet = crate::world::silence_map_diag();
    let mut world = city_world();
    let plan = build_city(&mut world);
    assert!(is_orthogonal_grid_city(&world));
    let junctions = detect_junctions(&world);
    assert_eq!(junctions.len(), 25, "grid city has 5×5 crossings");
    // Streets end at the map border: 9 interior crossings are X, the 12 edge
    // crossings are T (3 arms) and the 4 corners have 2 arms.
    let x_count = junctions.iter().filter(|j| j.kind() == JunctionKind::X).count();
    assert_eq!(x_count, 9, "{junctions:?}");
    assert!(junctions.iter().all(|j| j.arm_count() >= 2), "{junctions:?}");
    let declared: Vec<[f32; 2]> = plan.metadata()["intersections"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| [i["x_m"].as_f64().unwrap() as f32, i["z_m"].as_f64().unwrap() as f32])
        .collect();
    for j in &junctions {
        assert!(declared.iter().any(|d| (d[0] - j.centre[0]).abs() < 0.01 && (d[1] - j.centre[1]).abs() < 0.01), "{j:?}");
    }
    // The generic spawn finder also works on the grid and stays on lanes.
    let slots = spawn_slots_generic(&world, &plan.stations, 32).unwrap();
    assert_eq!(slots.len(), 32);
    for slot in &slots {
        assert!(road(&world, slot.position[0], slot.position[2]));
    }
}

#[test]
fn grid_detector_rejects_the_ring_city() {
    let _quiet = crate::world::silence_map_diag();
    let mut world = city_world();
    ring::build(&mut world, 1).unwrap();
    assert!(!is_orthogonal_grid_city(&world));
}

#[test]
fn ring_layout_is_deterministic_and_seed_sensitive() {
    let world = city_world();
    let a = ring::layout(&world, 5).unwrap();
    let b = ring::layout(&world, 5).unwrap();
    assert_eq!(a.signature, b.signature);
    assert_eq!(a.stations, b.stations);
    let c = ring::layout(&world, 6).unwrap();
    assert_ne!(a.signature, c.signature);
}

#[test]
fn ring_city_passes_the_family_contract_for_train_and_eval_seeds() {
    let _quiet = crate::world::silence_map_diag();
    for seed in 0..12u64 {
        let mut world = city_world();
        let plan = ring::build(&mut world, seed).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        validate(&world, &plan, 32).unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        let metadata = plan.metadata();
        assert_eq!(metadata["schema_version"], CITY_FAMILY_SCHEMA_VERSION);
        assert_eq!(metadata["variant"]["family"], "ring");
        assert_eq!(metadata["variant"]["split"], seed_split(seed));
        assert_eq!(plan.stations.len(), ring::STATION_COUNT);
        assert_eq!(plan.stations[0][0], world.start_pos.x);
        assert_eq!(plan.stations[0][1], world.start_pos.z);
        // Every junction is a T or an X and carries one light per arm.
        let junctions = detect_junctions(&world);
        assert!(!junctions.is_empty());
        let arms: usize = junctions.iter().map(Junction::arm_count).sum();
        assert_eq!(plan.traffic_lights().len(), arms);
        assert!(junctions.iter().all(|j| matches!(j.arm_count(), 3 | 4)), "{junctions:?}");
        // Layout junctions and detected junctions agree one to one.
        let layout = ring::layout(&world, seed).unwrap();
        assert_eq!(layout.junctions.len(), junctions.len());
        for (l, d) in layout.junctions.iter().zip(junctions.iter()) {
            assert!((l.centre[0] - d.centre[0]).abs() <= world.cell_size && (l.centre[1] - d.centre[1]).abs() <= world.cell_size, "{l:?} vs {d:?}");
            assert_eq!(l.arms, d.arms, "seed {seed} junction {l:?} vs {d:?}");
        }
        // The dispatching spawn finder takes the generic path on a ring city.
        let slots = spawn_slots(&world, &plan.stations, 32).unwrap();
        assert_eq!(slots, spawn_slots_generic(&world, &plan.stations, 32).unwrap());
        for slot in &slots {
            let flow = world.get_flow_at(slot.position[0], slot.position[2]);
            assert!((flow.x - slot.forward[0]).abs() < 1e-6 && (flow.y - slot.forward[2]).abs() < 1e-6);
        }
    }
}

#[test]
fn ring_lane_flow_is_right_hand_and_zero_in_junction_boxes() {
    let _quiet = crate::world::silence_map_diag();
    let mut world = city_world();
    ring::build(&mut world, 3).unwrap();
    let layout = ring::layout(&world, 3).unwrap();
    let [[min_x, min_z], [max_x, _]] = layout.extent_m;
    let radius = layout.generator["corner_radius_m"].as_f64().unwrap() as f32;
    // North straight: travel +x on the row south of the centreline (right-hand
    // side of +x is +z), travel -x on the row north of it.
    // 20 m past the tangent point: chords keep 70 m from the arc, so no
    // junction box can cover this column.
    let x = min_x + radius + 20.0;
    let _ = max_x;
    let south = world.get_flow_at(x, min_z + world.cell_size);
    let north = world.get_flow_at(x, min_z - world.cell_size);
    assert!(road(&world, x, min_z));
    assert!((south.x - 1.0).abs() < 1e-6 && south.y.abs() < 1e-6, "{south:?}");
    assert!((north.x + 1.0).abs() < 1e-6 && north.y.abs() < 1e-6, "{north:?}");
    // On the arc the flow follows the tangent and has unit length.
    let corner = [min_x + radius, min_z + radius];
    let angle = 1.25 * std::f32::consts::PI;
    let p = [corner[0] + (radius + world.cell_size) * angle.cos(), corner[1] + (radius + world.cell_size) * angle.sin()];
    let f = world.get_flow_at(p[0], p[1]);
    assert!((f.length() - 1.0).abs() < 1e-3, "{f:?}");
    assert!((f.x * (p[0] - corner[0]) + f.y * (p[1] - corner[1])).abs() < 0.05 * radius, "flow not tangent: {f:?}");
    for j in &layout.junctions {
        assert_eq!(world.get_flow_at(j.centre[0], j.centre[1]).length_squared(), 0.0);
    }
}

#[test]
fn ring_family_dispatches_through_organism_city() {
    let _quiet = crate::world::silence_map_diag();
    assert_eq!(CityFamily::parse("ring"), Some(CityFamily::Ring));
    assert_eq!(CityFamily::Ring.name(), "ring");
    let mut world = city_world();
    let plan = crate::organism_city::build_city_variant(&mut world, 2, CityFamily::Ring).unwrap();
    crate::organism_city::validate_city_postconditions(&world, &plan, 32).unwrap();
    assert_eq!(plan.metadata()["variant"]["family"], "ring");
}

#[test]
fn ring_city_offers_cross_city_station_pairs() {
    let _quiet = crate::world::silence_map_diag();
    let mut world = city_world();
    let plan = ring::build(&mut world, 4).unwrap();
    let stats = route_stats(&world, &plan.stations).unwrap();
    assert_eq!(stats.unreachable_pairs, 0);
    assert!(stats.min_pair_road_m >= 60.0, "{}", stats.min_pair_road_m);
    assert!(stats.cross_city_capable_pairs > 0);
}

#[test]
fn reference_grid_city_is_unchanged_by_the_module() {
    let _quiet = crate::world::silence_map_diag();
    let mut world = city_world();
    let plan = build_city(&mut world);
    assert_eq!(plan.metadata()["schema_version"], crate::organism_city::CITY_GEOMETRY_VERSION);
    assert_eq!(plan.traffic_lights().len(), 100);
    let road_cells = world.grid.iter().filter(|c| **c == VoxelType::Road).count();
    // 5 streets × 5 cells × 261 cells extent, each axis, minus 25 overlaps of 25 cells.
    assert_eq!(road_cells, 2 * 5 * 5 * 261 - 25 * 25);
}
