use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Resource, Debug, Serialize, Deserialize, Clone)]
pub struct LabParams {
    // Metadados
    pub experiment_name: String,
    pub variant_id: u32,

    // Axiom policy
    pub axiom_wall_avoidance: f32,
    pub axiom_dark_center: f32,
    pub axiom_whisker: f32,
    pub axiom_racing_line: f32,
    pub axiom_decay_divisor: f32,
    pub axiom_floor: f32,

    // Fitness weights
    pub k_smooth: f32,
    pub k_speed: f32,
    pub k_novel: f32,
    pub k_stab: f32,
    pub k_phi: f32,
    pub k_aesth: f32,
    pub lap_bonus: f32,
    pub survival_bonus: f32,
    pub death_penalty: f32,

    // Survival thresholds
    pub speed_bleed_damage: f32,
    pub stagnation_secs: f32,
    pub wall_t_threshold: i8,
}

impl Default for LabParams {
    fn default() -> Self {
        Self {
            experiment_name: "default".to_string(),
            variant_id: 0,
            axiom_wall_avoidance: 0.18,
            axiom_dark_center: 0.45,
            axiom_whisker: 0.15,
            axiom_racing_line: 0.26,
            axiom_decay_divisor: 50.0,
            axiom_floor: 0.35,
            k_smooth: 100.0,
            k_speed: 0.5,
            k_novel: 50.0,
            k_stab: 200.0,
            k_phi: 300.0,
            k_aesth: 200.0,
            lap_bonus: 50000.0,
            survival_bonus: 3000.0,
            death_penalty: 2000.0,
            speed_bleed_damage: 15.0,
            stagnation_secs: 15.0,
            wall_t_threshold: 40,
        }
    }
}

impl LabParams {
    pub fn load(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let json = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&json)?)
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }
}

use bevy::prelude::Resource;
