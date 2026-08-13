/// Módulo de percepção sensorial — retina e canais (E2).
///
/// # Estrutura
///
/// - `retina` — adaptador da retina sobre iara-perception (movido de `src/perception.rs`)
/// - `channels` — registro canônico dos 192 canais da onda sensorial
///
/// Os callers que usavam `use crate::perception::RetinaPerception;` continuam
/// funcionando via reexport abaixo.
pub mod retina;
pub mod channels;

// Reexporta tudo que os callers externos esperam do módulo original.
// Callers: src/engine.rs, src/bin/cortex_evolve.rs, src/bin/gpu_loop.rs,
//          src/bin/gpu_report.rs, src/bin/gpu_evolve.rs
pub use retina::*;

// Reexporta os símbolos principais de channels para conveniência.
pub use channels::{Channel, ChannelSource, N_CHANNELS, source_for, name_for};
