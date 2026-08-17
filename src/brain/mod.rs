//! Brain — file format and (in the future) the topology itself.
//!
//! Today it holds only the `.brain` format. The module exists on its own because the
//! intent is for it to also receive the layer definitions: when the brain gains regions
//! (vision, decision, motor), the description of each region lives here, next to the
//! format that writes it.
pub mod format;

pub use format::{Block, BlockKind, BrainFile, Encoding, FLAG_ACTIVE};
