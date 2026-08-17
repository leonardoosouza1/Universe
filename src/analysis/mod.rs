//! Activation metrics and runtime guards (E4 — REFACTOR_PLAN.md §E4).
//!
//! Consumes only `math::*` (E1) and primitive types — no GPU/Bevy dependency in
//! here. Whoever reads the GPU buffers and assembles the sample `Vec<f32>` is the
//! caller (`src/bin/gpu_evolve/context.rs`); this module only sees numbers.
pub mod activation;
/// Contextual sparsity: *who* fires for *this* situation (E1-E4 of the protocol).
pub mod contextual;
/// Track regions × activation: does the brain use different neurons on a straight and in a curve?
/// Crosses `contextual`/`activation` (activation) with the local track curvature (position).
pub mod regioes;
