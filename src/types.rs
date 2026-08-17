/// IARA v2100 - Proprietary Data Structures in Rust
/// Based on Codex Part 6: Ontologia de Dados

/// Ternary logic for the "Lago de 2-Bits" (-1, 0, 1)
/// 0 is homeostasis (rest), 1 is positive acceleration, -1 is negative acceleration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TernaryState {
    Negative = -1,
    Homeostasis = 0,
    Positive = 1,
}

/// Replaces media arrays. Stores raw frequencies (audio/video) after Wavelet Transform.
/// Part of the "Prisma de Ingestão".
#[derive(Debug, Clone)]
pub struct WaveformTensor {
    pub frequencies: Vec<f32>, // Decomposed frequencies
    pub energy: f32,           // Total signal energy
    pub timestamp: f64,        // Continuous time (ODE-based)
}

#[derive(Debug, Clone)]
pub struct SplineNode {
    pub position: [f32; 3],            // (x, y, z) spatial coordinates
    pub momentum: [f32; 3],            // p (vector) for kinetic calculations
    pub torque: [f32; 3],              // tau (vector) for error redirection (Hapkido)
    pub next: Option<Box<SplineNode>>, // Recursive pointer to next node
    pub mass: f32,                     // Semantic mass
}

#[derive(Debug, Clone)]
pub struct PotentialWell {
    pub semantic_mass: f32, // M (mass) - depth of gravitational well
    pub theta: f32,         // Angular coordinate (theta)
    pub radius: f32,        // Complexity level (R)
    pub concept_id: String, // Unique concept identifier
}

#[derive(Debug, Clone)]
pub struct QuantumPill {
    pub harmonics: Vec<i8>, // Quantized frequencies
    pub amplitude: f32,
    pub phase: f32,
}

#[derive(Debug, Clone)]
pub struct InputBoundary {
    pub sensory_flow: Vec<WaveformTensor>,
    pub active_anomalies: Vec<SplineNode>,
}

/// Motor control interface for the Virtual Body.
pub trait MotorActuator {
    fn apply_rotation(&mut self, matrix: [[f32; 3]; 3]);
    fn shift_viewport(&mut self, delta_x: f32, delta_y: f32);
    fn trigger_spline_action(&mut self, action: SplineNode);
}
