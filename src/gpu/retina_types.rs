//! Structs `Pod` do uniform/params da retina — fonte canônica única.
//!
//! # Por que este arquivo existe
//!
//! `RetinaParamsGpu` (uniform da retina: grade, contagem de seres, pitches por linha) e
//! `PoseRaw` (postura empacotada para o shader) existiam como structs **idênticas**,
//! copiadas de arquivo em arquivo, em `src/gpu/mod.rs` (como `ParamsGpu`), `gpu_evolve.rs`,
//! `gpu_loop.rs` e `cortex_evolve.rs`. Cada cópia é `#[repr(C)] + Pod` — o contrato real
//! é o layout em bytes que o shader WGSL espera (ver `retina.wgsl`), não o nome do tipo.
//! Ter quatro definições do mesmo layout é quatro chances de uma delas divergir sem que
//! o compilador avise (structs `Pod` distintas não dão erro de tipo entre si).
//!
//! Movida para cá em E3 (REFACTOR_PLAN.md §2.3) sem alterar nenhum `.wgsl` — o layout em
//! bytes é idêntico ao das cópias anteriores.

/// Uniform da retina: parâmetros compartilhados por todos os seres do dispatch.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RetinaParamsGpu {
    pub grid_size: u32,
    pub n_beings: u32,
    pub cell_size: f32,
    pub fov_half: f32,
    /// 4 × vec4 = 4 linhas × 4 pitches por linha (rows 0-15).
    pub pitches: [[f32; 4]; 4],
}

/// Postura de um ser, empacotada para o shader da retina.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PoseRaw {
    /// [x, y, z, gaze_yaw]
    pub pos_gaze: [f32; 4],
    /// [fwd.x, fwd.z, right.x, right.z]
    pub dirs: [f32; 4],
}
