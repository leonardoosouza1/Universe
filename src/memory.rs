/// IARA v2100 - Multiverse Spiral & Semantic Gravity
/// Based on Codex Part 2 & 5

use std::collections::HashMap;
use crate::types::PotentialWell;

pub struct MultiverseSpiral {
    /// The "1D Spiral Tape" stored as nodes on a cylindrical coordinate system.
    /// Memory grows along the Z-axis (time).
    pub cylinders: HashMap<String, Vec<PotentialWell>>,
    
    /// The "Einstein-Rosen Bridge" (Z-Axis absolute bus).
    /// Provides O(1) direct access from Core to Present.
    pub z_bus: Vec<f32>,
}

impl MultiverseSpiral {
    pub fn new() -> Self {
        Self {
            cylinders: HashMap::new(),
            z_bus: Vec::new(),
        }
    }

    /// Store a concept in the spiral at a specific coordinate.
    pub fn store_concept(&mut self, domain: &str, theta: f32, radius: f32, mass: f32, id: &str) {
        let entry = self.cylinders.entry(domain.to_string()).or_insert(Vec::new());
        entry.push(PotentialWell {
            semantic_mass: mass,
            theta,
            radius,
            concept_id: id.to_string(),
        });
        
        // Update the Z-Bus for O(1) tracking of this domain's growth
        self.z_bus.push(radius);
    }

    /// Semantic Gravity: Find the closest Potential Well for a given coordinate.
    /// This is the "Gravitational Pull" that routes incoming data (Splines).
    pub fn resolve_gravity(&self, domain: &str, query_theta: f32) -> Option<&PotentialWell> {
        let wells = self.cylinders.get(domain)?;
        
        // In a real implementation, we use spatial hashes or Morton Curves
        // to find the minimum energy state in O(1).
        wells.iter().min_by(|a, b| {
            let diff_a = (a.theta - query_theta).abs();
            let diff_b = (b.theta - query_theta).abs();
            diff_a.partial_cmp(&diff_b).unwrap()
        })
    }
}
