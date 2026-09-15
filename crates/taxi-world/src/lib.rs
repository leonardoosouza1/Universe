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
//! # Scope of this crate right now (T1..T7)
//!
//! T1..T4 (plan §4): the crate skeleton, fixed dimensions (`constants`), map
//! geometry (`world`), the read-only reference-trace loader (`trace`), the on-disk
//! genome format (`genome`), the SoA being state and per-tick physics
//! (`beings`/`physics`), senses/decision (`senses`/`decision`) and
//! services/passengers (`services`/`passengers`). T5 (`tick`, `gate`): the full
//! tick (`tick::step`, SPEC §3 `um_passo`) and the identity gate against SPEC §11
//! (`gate`, `tests/identity.rs`). T6 (`evolution`, `judge`): the GA loop and the
//! "vida" fitness judge, written against a minimal `Simulador`/`LifeSampler` trait
//! pair rather than calling `tick::step` directly, because T5 was being written in
//! parallel and did not exist yet when T6 landed — see `evolution.rs`'s and
//! `judge.rs`'s own doc comments for the full contract and the divergence from the
//! plan's literal `evolve`/`judge` signatures. T7 (`sim_cpu`, `taxi_evolve_cli`,
//! `src/bin/taxi_evolve.rs`): the integrator — `SimCpu` implements BOTH T6
//! traits over T5's real `tick::Sim`; `taxi_evolve_cli::run` is the CLI logic
//! (reads a lab scenario JSON + a champion `.json` + a map trace, runs
//! `evolve` + `judge` per seed, writes an `e43`-schema JSON `scripts/
//! analise_campo.py` reads unchanged) and lives in the library rather than
//! directly in `src/bin/taxi_evolve.rs` (a thin wrapper) so a `tests/*.rs`
//! integration test can drive it in-process — see `taxi_evolve_cli`'s own doc
//! comment.
//!
//! `Trace::initial_beings` is listed in the plan's own T1 interface block but is
//! NOT implemented here — see `trace.rs`'s doc comment on `Trace` for why (it needs
//! T2's `Beings` type, and T1's own hard rules forbade creating `beings.rs`).
//! `beings::Beings::from_trace` covers the identity-gate path (reading a trace's
//! `t=0` state); `sim_cpu`'s `build_batch` covers the GA-battery path (SPEC §1/§7.1
//! birth sampling, no trace involved).

pub mod beings;
pub mod constants;
pub mod decision;
pub mod evolution;
pub mod gate;
pub mod genome;
pub mod judge;
pub mod passengers;
pub mod physics;
pub mod senses;
pub mod services;
pub mod sim_cpu;
pub mod taxi_evolve_cli;
pub mod tick;
pub mod trace;
pub mod world;

pub use constants::{Config, DIRS, G, GRID, N_SERVICES, N_TYPES, NF, NP, NPX, S, TEX};
pub use genome::{Genome, GenomeError};
pub use trace::{Trace, TraceError};
pub use world::World;
