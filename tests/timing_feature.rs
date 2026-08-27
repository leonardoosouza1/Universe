//! The feature must be REQUESTED when available and its absence must not be fatal.
//! `#[ignore]`d because it opens a real adapter: CI runners have none, and a test that
//! silently passes on a missing device is worse than one that is skipped.
//!
//! # Deviation from the task brief
//!
//! The brief's draft referenced `universe::gpu::GpuCore::new()` and `core.device()`.
//! Neither exists: the struct that owns the `request_device` call this task changes
//! (`src/gpu/mod.rs:208-215`) is `RetinaGpu`, and its constructor is
//! `RetinaGpu::new(world: &VoxelWorld, capacity: usize) -> Option<Self>` (there is a
//! second, unrelated device-opening struct, `CoreGpu` in `src/gpu/core.rs`, used by the
//! physics/brain/perception tick — not the one this task's brief points at). `RetinaGpu`
//! already exposes `pub fn device(&self) -> &wgpu::Device` (`src/gpu/mod.rs:404`), which
//! is the real accessor used below in place of the brief's `core.device()`.

#[test]
#[ignore = "opens a real GPU — run manually: cargo test --release --test timing_feature -- --ignored"]
fn timestamp_query_is_requested_when_the_adapter_offers_it() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("no adapter");

    let offers = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);

    // `RetinaGpu::new` opens its own adapter+device internally (same backends,
    // same `HighPerformance` preference), so this is the device the production
    // code path actually builds today, not a stand-in for it.
    let world = universe::world::VoxelWorld::new(9001, 18.0);
    let retina = universe::gpu::RetinaGpu::new(&world, 64).expect("device");
    let granted = retina
        .device()
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY);

    assert_eq!(
        offers, granted,
        "adapter offers TIMESTAMP_QUERY = {offers} but the device was created with {granted}"
    );
}
