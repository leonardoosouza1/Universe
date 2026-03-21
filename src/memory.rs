/// IARA v2100 - Multiverse Spiral & Semantic Gravity (Fibonacci Matrix)
/// Based on Codex Part 2 & 5

use std::collections::HashMap;
use crate::types::PotentialWell;

pub struct MultiverseSpiral {
    /// The "Fibonacci Cylinder" - Memory grows along the Z-axis, rotating by the Golden Angle.
    pub cylinders: HashMap<String, Vec<PotentialWell>>,
    
    /// The "Einstein-Rosen Bridge" (Z-Axis absolute bus).
    pub z_bus: Vec<f32>,
    
    /// Global counter to track the Fibonacci node index
    pub global_index: usize,
}

impl MultiverseSpiral {
    pub fn new() -> Self {
        Self {
            cylinders: HashMap::new(),
            z_bus: Vec::new(),
            global_index: 0,
        }
    }

    /// Store a concept using the Fibonacci Spiral logic (Phyllotaxis)
    pub fn store_concept(&mut self, domain: &str, mass: f32, id: &str) {
        let phi = std::f32::consts::PI * (3.0 - (5.0f32).sqrt()); // Ângulo de Ouro
        let n = self.global_index as f32;
        
        let theta = n * phi;
        let radius = 15.0 * n.sqrt(); // Crescimento Fibonacci

        let entry = self.cylinders.entry(domain.to_string()).or_insert(Vec::new());
        entry.push(PotentialWell {
            semantic_mass: mass,
            theta,
            radius,
            concept_id: id.to_string(),
        });
        
        self.z_bus.push(radius);
        self.global_index += 1;
    }

    /// Semantic Gravity: Find the closest Potential Well for a given coordinate.
    pub fn resolve_gravity(&self, domain: &str, query_theta: f32) -> Option<&PotentialWell> {
        let wells = self.cylinders.get(domain)?;
        
        wells.iter().min_by(|a, b| {
            let diff_a = (a.theta - query_theta).abs();
            let diff_b = (b.theta - query_theta).abs();
            diff_a.partial_cmp(&diff_b).unwrap()
        })
    }
}

