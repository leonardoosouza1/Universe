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
    /// Tier 5.3: rotação de pistas por geração. Se false (default), usa sempre
    /// `selected_track_index`. Se true, alterna entre `tracks` por bloco de
    /// `gens_per_track` e fixa `tracks[0]` em gens de validação (env_level 0).
    /// Pode ser ativado via CLI `--rotate` ou env `UNIVERSE_ROTATE=1`.
    #[serde(default)]
    pub rotate_tracks: bool,
    #[serde(default = "default_gens_per_track")]
    pub gens_per_track: usize,
    /// FIX UI vs headless mismatch: salvas flags do último treino pra UI
    /// reproduzir mesma pista que DNA viu. Sem isso DNA treinado --easy
    /// --no-holes rodava em pista hard com holes na UI → cars ruins.
    #[serde(default)]
    pub easy_track: bool,
    #[serde(default)]
    pub no_holes: bool,
}

fn default_generation_timeout_s() -> f32 { 120.0 }
fn default_gens_per_track() -> usize { 3 }

impl Default for UniverseConfig {
    fn default() -> Self {
        Self {
            physics: PhysicsSettings::default(),
            // Ordem = progressão curricular (easy → very hard).
            // track_width: mais largo = mais perdão pra mutantes ruins.
            tracks: vec![
                // Default = procedural seed 9001. Matches DNA training seeds — UI mode loads
                // CHAMPION.dna trained on this layout instead of mismatching with hardcoded.
                TrackProfile {
                    name: "Procedural Sigma".to_string(),
                    seed: 9001,
                    track_width: 28.0,
                    complexity: 0.5,
                },
                TrackProfile {
                    name: "Dubai Oval".to_string(),
                    seed: 123,
                    track_width: 30.0,
                    complexity: 0.3,
                },
                TrackProfile {
                    name: "Interlagos Gateway".to_string(),
                    seed: 64,
                    track_width: 26.0,
                    complexity: 1.0,
                },
                TrackProfile {
                    name: "Monaco Tight".to_string(),
                    seed: 256,
                    track_width: 22.0,
                    complexity: 2.0,
                },
                TrackProfile {
                    name: "Silverstone Flow".to_string(),
                    seed: 512,
                    track_width: 24.0,
                    complexity: 1.8,
                },
                TrackProfile {
                    name: "Nurburgring Hills".to_string(),
                    seed: 768,
                    track_width: 24.0,
                    complexity: 2.5,
                },
                TrackProfile {
                    name: "Suzuka Crossing".to_string(),
                    seed: 1024,
                    track_width: 20.0,
                    complexity: 3.0,
                },
            ],
            selected_track_index: 0,
            swarm_count: 50,
            time_scale: 3.0,
            generation_timeout_s: 120.0,
            rotate_tracks: false,
            gens_per_track: 3,
            easy_track: false,
            no_holes: false,
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
        &self.tracks[self.selected_track_index.min(self.tracks.len().saturating_sub(1))]
    }

    /// Tier 5.3: retorna a pista para uma dada geração.
    /// - `rotate_tracks == false`: sempre `current_track()` (back-compat).
    /// - `is_validation_gen == true`: pinned em `tracks[0]` (env estável → comparação
    ///   honesta de campeão entre gens).
    /// - Caso contrário, rotação por bloco: idx = (gen / gens_per_track) % tracks.len().
    pub fn track_for_gen(&self, gen_idx: usize, is_validation_gen: bool) -> &TrackProfile {
        if !self.rotate_tracks || self.tracks.is_empty() {
            return self.current_track();
        }
        if is_validation_gen {
            return &self.tracks[0];
        }
        let per = self.gens_per_track.max(1);
        let idx = (gen_idx / per) % self.tracks.len();
        &self.tracks[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_with_rotate() -> UniverseConfig {
        let mut c = UniverseConfig::default();
        c.rotate_tracks = true;
        c.gens_per_track = 2;
        c
    }

    #[test]
    fn rotation_cycles_by_block() {
        let c = cfg_with_rotate();
        let n = c.tracks.len();
        // Block size = 2, so gens 0,1 → track 0; gens 2,3 → track 1; ...
        assert_eq!(c.track_for_gen(0, false).name, c.tracks[0].name);
        assert_eq!(c.track_for_gen(1, false).name, c.tracks[0].name);
        assert_eq!(c.track_for_gen(2, false).name, c.tracks[1 % n].name);
        assert_eq!(c.track_for_gen(3, false).name, c.tracks[1 % n].name);
        assert_eq!(c.track_for_gen(4, false).name, c.tracks[2 % n].name);
    }

    #[test]
    fn validation_gen_pins_track_zero() {
        let c = cfg_with_rotate();
        // Em qualquer gen, se é validation, track 0.
        for g in [0usize, 7, 13, 42] {
            assert_eq!(c.track_for_gen(g, true).name, c.tracks[0].name);
        }
    }

    #[test]
    fn rotate_false_keeps_current_track() {
        let c = UniverseConfig::default(); // rotate_tracks = false
        for g in [0usize, 5, 100] {
            assert_eq!(c.track_for_gen(g, false).name, c.current_track().name);
            assert_eq!(c.track_for_gen(g, true).name, c.current_track().name);
        }
    }
}
