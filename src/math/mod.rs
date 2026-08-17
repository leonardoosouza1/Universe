/// Pure math core — no GPU, Bevy or wgpu dependencies.
///
/// Every submodule exports pure functions (no global state, no I/O).
/// They are the base for the analysis modules (E4) and can be tested
/// deterministically offline.
///
/// Organisation:
/// - `activations` — activation functions and clamp helpers
/// - `stats`       — mean, variance, percentiles, histogram
/// - `quant`       — encode/decode i8↔f32, BF16↔F32, SNR
/// - `linalg`      — participation_ratio, spectral radius, effective rank
/// - `laws`        — Prospecting Law, AIC/RIC cost
/// - `thresholds`  — named thresholds for use in the analysis modules
/// - `cfc_perception` — canonical bridge to `iara_cfc::CfcCell` (the perception CfC),
///   mirrored by `gpu/retina_digest.wgsl`. The only place in this crate with the CfC
///   buffer layout — it does not reimplement the equation (that lives in the
///   `iara-cfc` crate).
pub mod activations;
pub mod cfc_perception;
pub mod laws;
pub mod linalg;
pub mod quant;
pub mod stats;
pub mod thresholds;
