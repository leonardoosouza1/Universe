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
///
/// Grown by 16 bytes (80 → 96) for `--retina-beings` (`docs/RETINA-BEINGS.md` §2):
/// `beings_on` gates the occlusion test the same way `fly_map::FlyParams::eye_on` gates
/// the eye binding — 0 (default) takes the exact kernel path this struct had before the
/// field existed. `_pad0`/`_pad1`/`_pad2` keep the Rust and WGSL sizes equal (both must
/// be a multiple of the struct's 16-byte alignment, forced by `pitches`'
/// `array<vec4<f32>, 4>` — see the "Host and shader must agree on struct layout" trap in
/// `src/gpu/CLAUDE.md`). EVERY construction site of this struct across the tree
/// (`gpu/mod.rs`, `gpu_evolve/context.rs`, `cortex_evolve.rs`, `gpu_loop.rs`, plus the
/// independent local copies in `gpu_report.rs` and `gpu_validate.rs`) must set these
/// four fields, or the buffer it uploads is 16 bytes short of what `retina.wgsl` now
/// declares.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RetinaParamsGpu {
    pub grid_size: u32,
    pub n_beings: u32,
    pub cell_size: f32,
    pub fov_half: f32,
    /// 4 × vec4 = 4 rows × 4 pitches per row (rows 0-15).
    pub pitches: [[f32; 4]; 4],
    /// `--retina-beings` gate: 0 = off, 1 = on. See the struct doc above.
    pub beings_on: u32,
    /// Optics gate in the reserved word: 0 = legacy, 1 = physical road-view geometry.
    pub _pad0: u32,
    pub _pad1: u32,
    pub _pad2: u32,
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
