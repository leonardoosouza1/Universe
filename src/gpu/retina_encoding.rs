//! Physical retinal pixels and compatibility encoding.
use bevy::math::{Vec2, Vec3};

/// One retina pixel, in PHYSICAL quantities: what a lidar would hand back.
///
/// The old retina returned an `i8` produced by a hand-written salience curve that resolved
/// 4 useful levels for road and 3 for grass (it normalised by 1200 m while the ray was
/// capped at 33.3 m) and put walls at the top of the scale, so the consumer's `max`-pool
/// erased the road from any bin that contained a wall. The decision about what matters now
/// happens downstream of here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pixel {
    /// Distance to the first opaque surface, in metres.
    pub dist_m: f32,
    /// 0 = Grass, 1 = Road, 2 = Wall, 3 = Hole (same order as `VoxelType`) — or
    /// [`BEING_MATERIAL`] (4), a code this kernel PRODUCES when `--retina-beings`
    /// intersects the ray with another being's body first. `VoxelType` (`src/world.rs`)
    /// still has only the four grid materials; `Being` is never written into `grid[]`.
    pub material: u8,
}

/// Material code the retina emits when the nearest thing the ray hits is another
/// being's body, not a grid voxel (`--retina-beings`, `docs/RETINA-BEINGS.md` §2).
///
/// Deliberately outside `VoxelType`'s 0-3 range: this is a code the RETINA produces,
/// never a voxel the world stores. Mirrored in `src/gpu/retina.wgsl`
/// (`const BEING_MATERIAL: u32 = 4u;`) and consumed by `legacy_encoding` below and by
/// `gpu/fly_map/fly_map.wgsl::eye_pixel_drive`'s CPU counterpart, `fly_map_eye.rs`.
pub const BEING_MATERIAL: u8 = 4;

/// The OLD encoding, reconstructed from the physical quantities.
///
/// It exists for two reasons, both methodological: to prove the GPU kernel is equivalent to
/// the CPU pixel by pixel, and to allow comparing physical × encoded in an A/B instead of
/// assuming the new one is better. Assuming without measuring was exactly the mistake of
/// whoever wrote the 1200 m.
pub fn legacy_encoding(px: Pixel) -> i8 {
    let d = px.dist_m;
    let depth_fade = (d / 1200.0).clamp(0.0, 1.0);
    let v = match px.material {
        2 => {
            if d < 30.0 {
                127.0 - ((d / 30.0).clamp(0.0, 1.0).sqrt() * 72.0)
            } else if d < 120.0 {
                55.0 - (((d - 30.0) / 90.0).clamp(0.0, 1.0) * 55.0)
            } else if d < 1200.0 {
                0.0 - (((d - 120.0) / 480.0).clamp(0.0, 1.0) * 25.0)
            } else {
                -25.0 - (((d - 1200.0) / 1200.0).clamp(0.0, 1.0) * 25.0)
            }
        }
        0 => 40.0 - (depth_fade * 40.0),
        1 => -128.0 + (depth_fade * 96.0),
        3 => 90.0 - ((d / 50.0).clamp(0.0, 1.0) * 30.0),
        // Being (docs/RETINA-BEINGS.md §2): a code the RETINA produces, never a grid
        // voxel — flat -29 rather than a curve over distance, because the raw
        // `Pixel.dist_m` already carries the real distance; this function only
        // reconstructs the OLD salience-curve encoding for the legacy consumers
        // (perception.wgsl's i8 channels, the fly eye binding). -29 sits in the one gap
        // no other material's formula reaches AT ANY distance, not only at the same one:
        // Wall spans [-25, 127] (minimum at d=1200: -25.0 - clamp(0..1)*25.0 = -25.0),
        // Grass [0, 40], Hole [60, 90], Road [-128, -32] (maximum at d=1200:
        // -128.0 + 1.0*96.0 = -32.0) — so the open interval (-32, -25) is untouched by
        // all four, and -29 sits inside it.
        4 => -29.0,
        m => unreachable!(
            "legacy_encoding: material {m} is not Grass/Road/Wall/Hole/Being (0-4) — a \
             new retina material code needs its own branch here, not the old `_` default \
             (which used to silently encode anything unknown as Hole)"
        ),
    };
    v.clamp(-128.0, 127.0) as i8
}

/// A being's pose at the instant of the dispatch: where it is and where it is looking.
#[derive(Clone, Copy, Debug)]
pub struct Pose {
    pub pos: Vec3,
    pub forward: Vec2,
    pub right: Vec2,
    pub gaze_yaw: f32,
}
