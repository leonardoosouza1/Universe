//! # Substrate — the generalized metaphysical engine
//!
//! This module formalizes the project's thesis: **the simulation is not a game with AI
//! on top. It is a physical world where perception, instinct and affect are emergent
//! properties of the substrate.**
//!
//! Today the only instantiated experiment is a race (`Racer` + `VoxelWorld` +
//! `pista`). Tomorrow it can be an organism foraging energy fields, an ant
//! navigating pheromones, or any other cognition situated in a field.
//!
//! The graphics engine (Bevy) **renders** the substrate — it does not *simulate* it. The
//! substrate lives in the data (voxels, pain synapses, ternary weights); the GPU
//! only translates that data into photons that hit the human retina. The car also
//! has a retina — but its retina is a DDA raycast over the same voxel field.
//! Human and agent see the SAME world through the SAME physical mechanism: light
//! reflected off the energy field.
//!
//! ## The three abstractions
//!
//! - [`World`] — the physical field. Voxels, light, pheromones, gradients. The single
//!   answer to the question "what exists at (x, y, z)?".
//! - [`Percept`] — the slice of the world an agent can capture. Binocular DDA
//!   retina, scalar smell field, local pressure, temperature.
//! - [`Agent`] — a situated body. It perceives, feels (pain/affect), decides, acts.
//!   What distinguishes agents is the *way of perceiving* and the *ternary ethics* of
//!   deciding — not "which game it plays".
//!
//! ## Why ternary (BitLinear 1.58-bit) belongs here
//!
//! Ternary weights collapse magnitudes into `{-1, 0, +1}`. That is not an optimization —
//! it is a **discrete ethics**: each neuron answers the question "with respect to
//! this stimulus, am I in favor, neutral, or against?". There is no "48.3% in favor".
//! Agents on this substrate decide by categorical voting.
//!
//! ## Priming (a priori axioms)
//!
//! Before evolution, each agent is born with minimal survival structures
//! (A1..A15 in the race: "wall on the left → turn right"). This is applied
//! Kant: there are a priori forms of the understanding that precede experience.
//! Without priming, initial randomness exterminates the population before
//! selection can operate.

use bevy::prelude::Vec3;

/// The physical field. Answers "what exists here?" for any point.
///
/// Concrete implementations: [`crate::world::VoxelWorld`] (track with asphalt,
/// grass, wall). Future ones: `PheromoneField`, `ResourceField`, `SemanticSpace`.
pub trait World {
    /// Type of this world's "matter" (e.g. [`crate::world::VoxelType`]).
    type Substance;

    /// Samples the world at a 3D point.
    fn sample_at(&self, pos: Vec3) -> Self::Substance;

    /// Local gradient — the "natural" flow direction at this point. In the race it is
    /// the track vector; in a smell world it would be the pheromone gradient.
    fn flow_at(&self, pos: Vec3) -> Vec3;
}

/// The sensory slice an agent extracts from the world.
///
/// Concrete implementations: binocular DDA retina (2×32 pixels, dual-scale FOV),
/// lateral sensors, optical flow. What matters: **every `Percept` is derived
/// from the `World` by the same physical mechanism** — photons, fields, collisions.
pub trait Percept {
    /// Projects the percept onto the brain's input vector (channels 0..N).
    /// By convention, signed 8-bit integers; the ternary instinct_layer
    /// quantizes everything on entry.
    fn project(&self) -> &[i8];
}

/// A body situated in the world. It perceives, suffers, decides, acts.
///
/// Concrete implementations: [`crate::simulation::Racer`]. Future ones: Forager,
/// Predator, Writer (an agent that perceives a semantic space and emits tokens).
///
/// ## Cycle contract
///
/// Each tick, the external harness (Bevy ECS) calls:
///
/// 1. `perceive(world)` — agent samples the field → sensory_wave
/// 2. The brain does: `sensory_wave → instinct_layer (ternary) → intent`
/// 3. `act(intent)` — agent applies forces to the world via physics
/// 4. `feel(delta)` — agent updates pain/joy/fitness from what happened
pub trait Agent {
    type Intent;
    type Affect;

    /// Apposition: the agent looks at the world and extracts its percept.
    fn perceive<W: World>(&mut self, world: &W);

    /// Volition: the agent transduces percept into motor intent.
    fn decide(&mut self) -> Self::Intent;

    /// Suffering/Pleasure: the agent integrates affective feedback from the previous step.
    /// In the race: `joy_integral` grows with flow state, `pain_memories`
    /// mark collisions. This signal is the DRIVER of evolutionary selection —
    /// fitness is the affective footprint of a life.
    fn feel(&mut self, affect: Self::Affect);
}

/// Philosophical landmark — not used at runtime, but it is the project's manifesto.
///
/// None of the axioms below is "the programmer's opinion". They are *invariants*
/// of the substrate that define what can be observed in it.
pub mod axioms {
    /// A1 — **Everything is field.** There is no fundamental "object"; only regions
    /// of the voxel field with distinct properties (wall = high density,
    /// grass = high friction, asphalt = low resistance).
    pub const FIELD_PRIMACY: &str = "World is a field, not a scene.";

    /// A2 — **To perceive is to collide with light.** The agent does not receive a
    /// ground-truth Vec3; it receives photons that ricochet off the field and hit a retina.
    pub const PERCEPTION_IS_COLLISION: &str = "To see is to be hit by light.";

    /// A3 — **Decision is ternary voting.** Each weight is a categorical vote.
    /// There are no continuous preference gradients — there is {against, neutral, in favor}.
    pub const ETHICS_IS_TERNARY: &str = "To decide is to vote -1, 0, or +1.";

    /// A4 — **Fitness is affect integrated over time.** Evolution selects not
    /// "who got the farthest" but "who accumulated the most joy and the least pain".
    pub const FITNESS_IS_AFFECT: &str = "Fitness is felt life, integrated.";

    /// A5 — **Instincts precede experience.** The A1..A15 priming is the a priori
    /// structure without which randomness exterminates the population.
    pub const A_PRIORI_PRIMING: &str = "Instincts come before experience.";
}
