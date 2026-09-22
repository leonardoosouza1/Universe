//! GPU simulation, with no graphics.
//!
//! # The architecture the owner asked for
//!
//! > *"I need it to run on the GPU with no graphics at all, because that was supposed to
//! > be in the foundation of the project: ways to visualize with Bevy and without, and the
//! > one without is not supposed to render Bevy — Bevy would be just a frontend that shows
//! > in a graphics engine what is happening"* (translated from the original request)
//!
//! In other words: the core is compute, and Bevy is an optional window over it. This module
//! is the start of that core — it talks to the GPU through `wgpu` (already in the tree via
//! Bevy) **with no surface, no swapchain and no window**, so it runs under
//! `env -u DISPLAY`, in cron, on a machine with no graphical session.
//!
//! # Why the retina came first
//!
//! Measured over 2440 PERF samples from that day's races:
//!
//! | component | share of the time |
//! |---|---|
//! | **retina** | **57.0%** |
//! | physics and the rest | 26.9% |
//! | brain | 14.3% |
//! | perception | 1.9% |
//!
//! The retina is 14 rows × 126 columns × N beings — with 60 beings, **105,840 DDA marches
//! per tick**, all independent. It is the most parallel work in the simulator and the most
//! expensive. The brain, which is what the thesis studies, costs a quarter of that.
//!
//! # The non-negotiable rule
//!
//! The kernel has to give the SAME result as the CPU, pixel by pixel. A GPU path that
//! silently answers differently would invalidate every experiment already measured — and
//! after a full day of uncovering constants that carried dead assumptions, that is exactly
//! the class of error that cannot get in. The original note here pointed at
//! `testes::gpu_igual_a_cpu`; that test no longer exists in the tree (checked 2026-08-17,
//! no match anywhere under `src/` or `tests/`). The equivalence check that does exist is
//! the `src/bin/gpu_validate.rs` binary — run it before and after touching this module.

// ── GPU simulation core, written in parallel by agents ────────────────────────
// Each module brings a WGSL kernel + its wgpu host. The rule is the same in all of them:
// equivalence with the CPU demonstrated before any claim of speed-up.
pub mod body;
pub mod brain;
pub mod lifetime;
pub mod organs;
// C3 of docs/PRD-CEREBRO-NEUROMOD.md — the three method rulers the Universe did not
// have: how many genes of a block are on (`dof`), how much a block moves the command
// (`contribution`), and whether a block can move it at all (`sensitivity`). All three
// are PURE and GPU-free on purpose: the sensitivity gate has to be runnable before
// anything is queued on the shared GPU.
pub mod contribution;
pub mod core;
pub mod cortex_sheet;
pub mod dof;
pub mod evolution;
// Host + kernels of the fly map (`docs/FLY-MAP-BRAIN.md`). The map itself, its file
// format and the CPU reference the kernels are compared against live in the GPU-free
// `crate::fly_map`.
pub mod fly_map;
pub mod genome_io;
pub mod hippocampus;
pub mod perception;
pub mod physics;
pub mod readback;
pub mod retina_digest;
pub mod retina_metrics;
pub mod retina_types;
pub mod screen_brain;
pub mod sensitivity;
pub mod slow_brain;
pub mod state_snapshot;
pub mod timing;
pub mod wave_record;

pub mod retina;
pub mod retina_encoding;
pub mod retina_resource;
pub use retina::RetinaGpu;
pub use retina_encoding::{legacy_encoding, Pixel, Pose, BEING_MATERIAL};
pub use retina_resource::{world_signature, RetinaGpuRes};

/// Column count in one binocular retinal row.
pub const COLS: usize = crate::config::iara::vision::RETINA_COLS;

#[cfg(test)]
mod retina_road_tests;
#[cfg(test)]
mod retina_tests;

/// Shared source for all retinal dispatchers, including the opt-in physical ray tracer.
pub const RETINA_SHADER: &str = concat!(
    include_str!("retina_geometry.wgsl"),
    "\n",
    include_str!("retina.wgsl")
);

pub mod organism;
pub mod organism_action;
pub mod organism_telemetry;
pub mod taxi;
pub mod taxi_witness;

pub mod organism_feedback;

#[cfg(test)]
mod organism_tests;

pub mod organism_reproduction;

pub mod organism_orientation;

pub mod organism_sensors;

pub mod traffic;

pub mod taxi_routes;

pub mod school;

#[cfg(test)]
mod school_tests;
