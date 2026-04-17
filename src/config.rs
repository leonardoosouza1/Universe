use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};

#[derive(Resource, Debug, Serialize, Deserialize, Clone)]
pub struct PhysicsSettings {
    pub enable_stall: bool,
    pub enable_thermal: bool,
    pub enable_aero_drag: bool,
    pub enable_downforce: bool,
    pub enable_tcs: bool,
    pub enable_abs: bool,
    pub enable_aquaplaning: bool,
    pub enable_tire_wear: bool,
}

impl Default for PhysicsSettings {
    fn default() -> Self {
        Self {
            enable_stall: true,
            enable_thermal: true,
            enable_aero_drag: true,
            enable_downforce: true,
            enable_tcs: true,
            enable_abs: true,
            enable_aquaplaning: true,
            enable_tire_wear: true,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrackProfile {
    pub name: String,
    pub seed: u64,
    pub track_width: f32,
    pub complexity: f32,
}

#[derive(Resource, Debug, Serialize, Deserialize, Clone)]
pub struct UniverseConfig {
    pub physics: PhysicsSettings,
    pub tracks: Vec<TrackProfile>,
    pub selected_track_index: usize,
    pub swarm_count: usize,
    /// Multiplicador de velocidade de simulação (1.0 = tempo real, 10.0 = 10x mais rápido)
    pub time_scale: f32,
    /// Duração máxima de cada geração em segundos de simulação.
    /// A geração encerra quando: alguém completa 5 voltas, OU este timeout é atingido.
    /// Carros mortos permanecem na cena (ocultos) mas não encerram a gen — os vivos
    /// continuam acumulando distância até o tempo acabar.
    #[serde(default = "default_generation_timeout_s")]
    pub generation_timeout_s: f32,
}

fn default_generation_timeout_s() -> f32 { 120.0 }

impl Default for UniverseConfig {
    fn default() -> Self {
        Self {
            physics: PhysicsSettings::default(),
            tracks: vec![
                TrackProfile {
                    name: "Interlagos Gateway".to_string(),
                    seed: 64,
                    track_width: 36.0,
                    complexity: 1.0,
                },
                TrackProfile {
                    name: "Dubai Oval".to_string(),
                    seed: 123,
                    track_width: 40.0,
                    complexity: 0.5,
                }
            ],
            selected_track_index: 0,
            swarm_count: 50,
            time_scale: 3.0,
            generation_timeout_s: 120.0,
        }
    }
}

impl UniverseConfig {
    pub fn load_or_create() -> Self {
        if let Ok(mut file) = File::open("universe_config.json") {
            let mut contents = String::new();
            if file.read_to_string(&mut contents).is_ok() {
                if let Ok(config) = serde_json::from_str(&contents) {
                    return config;
                }
            }
        }
        let config = Self::default();
        let _ = config.save();
        config
    }

    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        let mut file = File::create("universe_config.json")?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    pub fn current_track(&self) -> &TrackProfile {
        &self.tracks[self.selected_track_index]
    }
}
