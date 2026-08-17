//! Shim: the real content moved to `universe::gpu::genome_io` (lib) on 2026-08-14
//! so the canonical parser could be used by `genoma_view` and future readers —
//! see the header there for the bug that motivated it (duplicate parser reading τ wrong).
pub use universe::gpu::genome_io::*;
