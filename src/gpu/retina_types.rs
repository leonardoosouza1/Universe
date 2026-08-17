//! `Pod` structs for the retina uniform/params — single canonical source.
//!
//! # Why this file exists
//!
//! `RetinaParamsGpu` (retina uniform: grid, being count, per-row pitches) and
//! `PoseRaw` (pose packed for the shader) existed as **identical** structs,
//! copied file to file, in `src/gpu/mod.rs` (as `ParamsGpu`), `gpu_evolve.rs`,
//! `gpu_loop.rs` and `cortex_evolve.rs`. Each copy is `#[repr(C)] + Pod` — the real
//! contract is the byte layout the WGSL shader expects (see `retina.wgsl`), not the type
//! name. Having four definitions of the same layout is four chances for one of them to
//! diverge without the compiler warning (distinct `Pod` structs give no type error
//! against each other).
//!
//! Moved here in E3 (REFACTOR_PLAN.md §2.3) without changing any `.wgsl` — the byte
//! layout is identical to the previous copies.

/// Retina uniform: parameters shared by every being in the dispatch.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RetinaParamsGpu {
    pub grid_size: u32,
    pub n_beings: u32,
    pub cell_size: f32,
    pub fov_half: f32,
    /// 4 × vec4 = 4 rows × 4 pitches per row (rows 0-15).
    pub pitches: [[f32; 4]; 4],
}

/// Pose of one being, packed for the retina shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PoseRaw {
    /// [x, y, z, gaze_yaw]
    pub pos_gaze: [f32; 4],
    /// [fwd.x, fwd.z, right.x, right.z]
    pub dirs: [f32; 4],
}
