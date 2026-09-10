//! `taxi-world` — a standalone, GPU-free port of the taxi survival world.
//!
//! See `docs/plano-mundo-taxi.md` (the implementation plan, "plan" below) and
//! `/home/leonardo/projects/LLM/neuromod-lab/docs/71-spec-mundo-taxi.md` ("SPEC"
//! below, the executable source of truth for every formula). This crate depends on
//! NEITHER `universe` NOR Bevy, by design (plan §1.2): the whole point of a
//! separate crate is that `cargo test -p taxi-world` never links the GPU
//! visualizer, so the world's identity gate (T5, SPEC §11) stays a
//! seconds-not-minutes command that does not compete with the GPU battery this
//! machine also runs (plan §4, "Comando" rule: always `nice -n 10`, never steal a
//! launch-bound core from a battery in progress).
//!
//! # Scope of this crate right now (T1..T4 and T6; T5 pending)
//!
//! T1..T4 (plan §4) are in: the crate skeleton, fixed dimensions (`constants`), map
//! geometry (`world`), the read-only reference-trace loader (`trace`), the on-disk
//! genome format (`genome`), the SoA being state and per-tick physics
//! (`beings`/`physics`), senses/decision (`senses`/`decision`) and
//! services/passengers (`services`/`passengers`). T6 (`evolution`, `judge`) is also
//! in — the GA loop and the "vida" fitness judge, written against a minimal
//! `Simulador`/`LifeSampler` trait pair rather than T5's tick function (T5 was
//! being written in parallel and does not exist in this crate yet — see
//! `evolution.rs`'s and `judge.rs`'s own doc comments for the full contract and the
//! divergence from the plan's literal `evolve`/`judge` signatures). T5
//! (`tick`, `gate`) is the one task still deliberately NOT declared below:
//!
//! ```text
//! T5  tick, gate            — the full tick, and the identity gate against SPEC §11
//! ```
//!
//! `Trace::initial_beings` is listed in the plan's own T1 interface block but is
//! NOT implemented here — see `trace.rs`'s doc comment on `Trace` for why (it needs
//! T2's `Beings` type, and this task's hard rules forbid creating `beings.rs`).

pub mod constants;
pub mod genome;
pub mod trace;
pub mod world;
pub mod beings;
pub mod physics;
pub mod senses;
pub mod decision;
pub mod services;
pub mod passengers;

// T5 (plan §4) — left commented so the module names are discoverable:
// pub mod tick;
// pub mod gate;
pub mod evolution;
pub mod judge;

pub use constants::{Config, DIRS, G, GRID, N_SERVICES, N_TYPES, NF, NP, NPX, S, TEX};
pub use genome::{Genome, GenomeError};
pub use trace::{Trace, TraceError};
pub use world::World;
