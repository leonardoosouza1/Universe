//! Tesla-style safety layer — pure rules, no neural network (E6: doc added
//! in the final readability review; the module itself was not touched in that round).
//!
//! `Sentinel` sits BETWEEN the brain's output (proposed `IntentVec`) and the actuator: Tier 0 is
//! a HARD veto (replaces the whole intent, e.g. imminent collision), Tier 1/2 are SOFT
//! proposals that the `gate` weighs. Consumed today only by the legacy Bevy path
//! (`src/engine.rs`, `src/simulation/motor.rs`) — the GPU path (`gpu_evolve`) does not go
//! through here yet.
pub mod sentinel;
pub use sentinel::{IntentVec, SafetyFlags, SafetyState, Sentinel, SentinelOutput};
