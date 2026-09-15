//! Deterministic starting grids that follow the track centerline.
//!
//! Formation rows are measured in track arc metres, then offset from the local
//! polyline tangent. This keeps every slot on the generated road at corners and
//! gives paired runs exactly the same positions.

use bevy::math::Vec2;
use universe::resources::point_at_arc;
use universe::world::{VoxelType, VoxelWorld};

pub const FORMATION_COLUMNS: usize = 4;
pub const FORMATION_LATERAL_SPACING_M: f32 = 3.0;
pub const FORMATION_LONGITUDINAL_SPACING_M: f32 = 3.5;
const MIN_SLOT_DISTANCE_M: f32 = 2.0 * 0.775;

/// A validated starting pose and its track progress in metres.
#[derive(Clone, Copy, Debug)]
pub struct StartingSlot {
    pub position: Vec2,
    pub forward: Vec2,
    pub progress_m: f32,
}

/// Returns the canonical start pose selected by the track generator.
///
/// `VoxelWorld::start_pos` intentionally lies on the launch straight. Starting
/// at waypoint zero instead is ambiguous on a closed polyline because that
/// point is shared by the first and last segments.
pub fn canonical_start(world: &VoxelWorld) -> Result<StartingSlot, String> {
    let position = Vec2::new(world.start_pos.x, world.start_pos.z);
    let forward = world
        .get_flow_at(position.x, position.y)
        .normalize_or_zero();
    if !position.is_finite() || !forward.is_finite() || forward.length_squared() < 1e-8 {
        return Err("canonical start has a non-finite or degenerate track flow".into());
    }
    if world.get_voxel_at(position.x, position.y) != VoxelType::Road {
        return Err(format!(
            "canonical start is off-road at ({:.3}, {:.3})",
            position.x, position.y
        ));
    }
    Ok(StartingSlot {
        position,
        forward,
        progress_m: world.start_progress,
    })
}

/// Returns the longitudinal and lateral offset for a population index.
pub fn formation_offset(being: usize) -> (f32, f32) {
    let row = being / FORMATION_COLUMNS;
    let col = being % FORMATION_COLUMNS;
    let lateral =
        (col as f32 - (FORMATION_COLUMNS as f32 - 1.0) * 0.5) * FORMATION_LATERAL_SPACING_M;
    (row as f32 * FORMATION_LONGITUDINAL_SPACING_M, lateral)
}

/// Places and validates a deterministic formation on a closed waypoint polyline.
pub fn formation_slots(world: &VoxelWorld, count: usize) -> Result<Vec<StartingSlot>, String> {
    let waypoints = &world.waypoints_world;
    let arcs = &world.waypoint_arcs;
    let total = arcs.last().copied().unwrap_or(0.0);
    if waypoints.len() < 2 || arcs.len() != waypoints.len() || total <= 0.0 {
        return Err("formation requires a non-degenerate waypoint polyline".into());
    }

    let mut slots = Vec::with_capacity(count);
    for being in 0..count {
        let (longitudinal, lateral) = formation_offset(being);
        let arc = longitudinal.rem_euclid(total);
        let (center, tangent) = point_at_arc(waypoints, arcs, arc);
        let tangent = if tangent.length_squared() > 1e-8 {
            tangent.normalize()
        } else {
            return Err(format!("slot {being} has a degenerate polyline tangent"));
        };
        let position = center + Vec2::new(-tangent.y, tangent.x) * lateral;
        if !position.is_finite() || !tangent.is_finite() {
            return Err(format!("slot {being} is non-finite"));
        }
        if world.get_voxel_at(position.x, position.y) != VoxelType::Road {
            return Err(format!(
                "slot {being} is off-road at ({:.3}, {:.3})",
                position.x, position.y
            ));
        }
        slots.push(StartingSlot {
            position,
            forward: tangent,
            progress_m: world.get_track_progress(position.x, position.y),
        });
    }
    for i in 0..slots.len() {
        for j in 0..i {
            if slots[i].position.distance(slots[j].position) < MIN_SLOT_DISTANCE_M {
                return Err(format!("slots {i} and {j} overlap"));
            }
        }
    }
    Ok(slots)
}

#[cfg(test)]
mod tests {
    use super::*;
    use universe::simulation::racing_line::RacingLine;

    #[test]
    fn formation_offsets_are_separated() {
        let mut slots: Vec<(f32, f32)> = Vec::new();
        for being in 0..40 {
            let (longitudinal, lateral) = formation_offset(being);
            for &(other_longitudinal, other_lateral) in &slots {
                assert!(
                    (longitudinal - other_longitudinal).hypot(lateral - other_lateral)
                        >= MIN_SLOT_DISTANCE_M
                );
            }
            slots.push((longitudinal, lateral));
        }
    }

    #[test]
    fn offsets_keep_four_columns_and_three_point_five_metre_rows() {
        assert_eq!(formation_offset(0), (0.0, -4.5));
        assert_eq!(formation_offset(3), (0.0, 4.5));
        assert_eq!(formation_offset(4), (3.5, -4.5));
    }

    #[test]
    fn registered_track_has_finite_road_poses_with_independent_progress() {
        let world = VoxelWorld::new(8005, 28.0);
        let slots = formation_slots(&world, 40).unwrap();
        assert_eq!(slots.len(), 40);
        for (i, slot) in slots.iter().enumerate() {
            assert_eq!(
                world.get_voxel_at(slot.position.x, slot.position.y),
                VoxelType::Road
            );
            assert!(slot.progress_m.is_finite());
            assert!((slot.forward.length() - 1.0).abs() < 1e-6);
            for other in &slots[..i] {
                assert!(slot.position.distance(other.position) >= MIN_SLOT_DISTANCE_M);
            }
        }
        assert!(
            slots
                .windows(2)
                .any(|pair| pair[0].progress_m != pair[1].progress_m)
        );
    }

    #[test]
    fn invalid_track_is_rejected_before_starting_physics() {
        let mut world = VoxelWorld::new(8005, 28.0);
        world.waypoints_world.clear();
        assert!(formation_slots(&world, 40).is_err());
    }

    #[test]
    fn canonical_start_uses_the_launch_straight_instead_of_the_closed_seam() {
        let world = VoxelWorld::new(8005, 28.0);
        let slot = canonical_start(&world).unwrap();
        let seam = world.waypoints_world[0];
        assert!(slot.position.distance(seam) > 1.0);
        assert_eq!(
            world.get_voxel_at(slot.position.x, slot.position.y),
            VoxelType::Road
        );
        assert!((slot.forward.length() - 1.0).abs() < 1e-6);

        let line = RacingLine::from_centerline(&world.waypoints_world, 28.0);
        for horizon in [40.0, 80.0, 140.0] {
            let curve = line.lookahead_curve(slot.position, horizon);
            assert!(
                curve.abs() < 0.5,
                "launch straight reports an implausible curve {curve} at {horizon}m"
            );
        }
    }
}
