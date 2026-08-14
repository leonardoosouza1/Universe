/// Núcleo matemático puro — sem dependências de GPU, Bevy ou wgpu.
///
/// Todos os submódulos exportam funções puras (sem estado global, sem I/O).
/// São a base para os módulos de análise (E4) e podem ser testados de forma
/// determinística offline.
///
/// Organização:
/// - `activations` — funções de ativação e helpers de clamp
/// - `stats`       — média, variância, percentis, histograma
/// - `quant`       — encode/decode i8↔f32, BF16↔F32, SNR
/// - `linalg`      — participation_ratio, raio espectral, rank efetivo
/// - `laws`        — Lei do Garimpo, custo AIC/RIC
/// - `thresholds`  — limiares nomeados para uso nos módulos de análise
/// - `cfc_perception` — ponte canônica para `iara_cfc::CfcCell` (CfC de percepção),
///   espelhada por `gpu/retina_digest.wgsl`. Único lugar deste crate com o layout de
///   buffer do CfC — não reimplementa a equação (mora na crate `iara-cfc`).
pub mod activations;
pub mod stats;
pub mod quant;
pub mod linalg;
pub mod laws;
pub mod thresholds;
pub mod cfc_perception;
