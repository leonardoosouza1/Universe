pub mod channels;
/// Sensory perception module — retina and channels (E2).
///
/// # Structure
///
/// - `retina` — retina adapter over iara-perception (moved from `src/perception.rs`)
/// - `channels` — canonical registry of the 192 sensory-wave channels
///
/// Callers that used `use crate::perception::RetinaPerception;` keep
/// working via the reexport below.
pub mod retina;

// Reexports everything external callers expect from the original module.
// Callers: src/engine.rs, src/bin/cortex_evolve.rs, src/bin/gpu_loop.rs,
//          src/bin/gpu_report.rs, src/bin/gpu_evolve.rs
pub use retina::*;

// Reexports the main channels symbols for convenience.
pub use channels::{Channel, ChannelSource, N_CHANNELS, name_for, source_for};

pub mod retina_map;
