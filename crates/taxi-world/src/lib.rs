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
//! # Scope of this crate right now (T1 only)
//!
//! This is task T1 (plan §"T1 — Esqueleto do crate, carregador de mapa e
//! geometria"): the crate skeleton, fixed dimensions (`constants`), map geometry
//! (`world`), the read-only reference-trace loader (`trace`) and the on-disk genome
//! format (`genome`). Tasks T2..T6 (plan §4) each own one or more modules that are
//! deliberately NOT declared below, so a later `git merge` between implementers
//! collides on nothing but this file's `mod` list:
//!
//! ```text
//! T2  beings, physics       — the SoA being state and the per-tick physics
//! T3  senses, decision      — observe() -> F, the 29-term score, argmax
//! T4  services, passengers  — service availability/cost, boarding/delivery
//! T5  tick, gate            — the full tick, and the identity gate against SPEC §11
//! T6  evolution, judge      — the GA loop and the "vida" fitness judge
//! ```
//!
//! `Trace::initial_beings` is listed in the plan's own T1 interface block but is
//! NOT implemented here — see `trace.rs`'s doc comment on `Trace` for why (it needs
//! T2's `Beings` type, and this task's hard rules forbid creating `beings.rs`).

pub mod constants;
pub mod genome;
pub mod trace;
pub mod world;

// Not part of T1 (plan §4) — left commented so the module names are discoverable
// without pre-committing to their shape:
// pub mod beings;
// pub mod physics;
// pub mod senses;
// pub mod decision;
// pub mod services;
// pub mod passengers;
// pub mod tick;
// pub mod gate;
// pub mod evolution;
// pub mod judge;

pub use constants::{Config, DIRS, G, GRID, N_SERVICES, N_TYPES, NF, NP, NPX, S, TEX};
pub use genome::{Genome, GenomeError};
pub use trace::{Trace, TraceError};
pub use world::World;
