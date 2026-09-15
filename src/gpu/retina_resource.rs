//! Persistent Bevy resource for the standalone retinal renderer.
use super::retina::RetinaGpu;
use crate::world::VoxelWorld;

/// Bevy resource that keeps the GPU context alive between ticks.
///
/// Recreating the device and recompiling the shader once per tick would cancel out the
/// whole gain, so this lives in the application. The world signature is what makes a track
/// change detectable, so the grid can be re-uploaded — if that slipped through, the beings
/// would see the PREVIOUS track and nothing would flag it.
#[derive(bevy::prelude::Resource, Default)]
pub struct RetinaGpuRes {
    pub ctx: Option<RetinaGpu>,
    pub signature: u64,
}

impl RetinaGpuRes {
    /// Returns a context valid for this world, creating or recreating it when needed.
    /// `None` = no usable GPU; the caller falls back to the CPU instead of killing the race.
    pub fn get_or_create(&mut self, world: &VoxelWorld, capacity: usize) -> Option<&RetinaGpu> {
        let sig = world_signature(world);
        let needs_rebuild = self.ctx.is_none()
            || self.signature != sig
            || self
                .ctx
                .as_ref()
                .map(|c| c.capacity < capacity)
                .unwrap_or(true);
        if needs_rebuild {
            self.ctx = RetinaGpu::new(world, capacity.max(64));
            self.signature = sig;
            if let Some(c) = &self.ctx {
                println!("🖥  retina on the GPU: {} ({} beings)", c.adapter, capacity);
            } else {
                println!("⚠️  no usable GPU — the retina stays on the CPU");
            }
        }
        self.ctx.as_ref()
    }
}

/// Cheap and sufficient: size, lap length and a sampling of the grid.
/// Exposed as `pub` so that motor.rs can pass the same signature to the GPU perception
/// context without recomputing or duplicating the logic.
pub fn world_signature(world: &VoxelWorld) -> u64 {
    let mut h = world.size as u64 ^ ((world.lap_distance as u64) << 20);
    let step = (world.grid.len() / 512).max(1);
    for (i, v) in world.grid.iter().step_by(step).enumerate() {
        h = h
            .wrapping_mul(1099511628211)
            .wrapping_add((*v as u64) ^ i as u64);
    }
    h
}
