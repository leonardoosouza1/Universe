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
                    track_width: 14.0,
                    complexity: 0.5,
                },
                TrackProfile {
                    name: "Dubai Oval".to_string(),
                    seed: 123,
                    track_width: 15.0,
                    complexity: 0.3,
                },
                TrackProfile {
                    name: "Interlagos Gateway".to_string(),
                    seed: 64,
                    track_width: 13.0,
                    complexity: 1.0,
                },
                TrackProfile {
                    name: "Monaco Tight".to_string(),
                    seed: 256,
                    track_width: 11.0,
                    complexity: 2.0,
                },
                TrackProfile {
                    name: "Silverstone Flow".to_string(),
                    seed: 512,
                    track_width: 12.0,
                    complexity: 1.8,
                },
                TrackProfile {
                    name: "Nurburgring Hills".to_string(),
                    seed: 768,
                    track_width: 12.0,
                    complexity: 2.5,
                },
                TrackProfile {
                    name: "Suzuka Crossing".to_string(),
                    seed: 1024,
                    track_width: 10.0,
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

// ============================================================================
// IARA — HUB DE CONSTANTES DE REFERÊNCIA
// ============================================================================
// Documentação centralizada dos parâmetros do sistema.
// Os valores abaixo ESPELHAM os que existem nos módulos originais.
//
// Para alterar um parâmetro:
//   1. Mude o valor AQUI
//   2. Mude no arquivo original correspondente (indicado nos comentários)
//
// Quando quiser que o sistema inteiro derive daqui automaticamente,
// basta trocar o `use` no módulo original por `use crate::config::iara::*`.
// ============================================================================

pub mod iara {
    // ── VISÃO (fonte: src/world.rs:12-18) ──────────────────────────────
    pub mod vision {
        // ORIGINAL
        // pub const RETINA_ROWS: usize = 14;
        // pub const RETINA_COLS: usize = 126;
        // pub const RETINA_TOTAL: usize = RETINA_ROWS * RETINA_COLS; 
        // pub const FOV_DEG: f32 = 140.0;
        // pub const RETINA_SKIP: u32 = 2;
        // pub const RETINA_MIN_PITCH: f32 = -0.04;
        // pub const RETINA_MAX_PITCH: f32 = -0.005;
        
        // EXPANDED VISION: 2.3x rows + 2x cols. See full field (sky→road).
        // User feedback: agent sees same place always, can't see curves ahead.
        // Solution: Full-field vision captures elevation changes, distant curves, road detail.
        pub const RETINA_ROWS: usize = 32;     // 14→32: sky to ground detail
        pub const RETINA_COLS: usize = 256;    // 126→256: wider FOV angular resolution
        pub const RETINA_TOTAL: usize = RETINA_ROWS * RETINA_COLS;
        pub const FOV_DEG: f32 = 140.0;        // 120→140°: better horizontal coverage
        pub const RETINA_SKIP: u32 = 1;        // was 2: include every pixel, no decimation
        // Pitch: ROW 0 = sky (curves/crest), ROW 16 = horizon, ROW 31 = asfalto
        pub const RETINA_MIN_PITCH: f32 = 0.12;    // +0.12 rad = +6.9° UP (see crest, curves ahead)
        pub const RETINA_MAX_PITCH: f32 = -0.12;   // -0.12 rad = -6.9° DOWN (see road, grass, obstacles)

        const fn compute_pitches() -> [f32; RETINA_ROWS] {
            let mut pitches = [0.0_f32; RETINA_ROWS];
            if RETINA_ROWS <= 1 {
                pitches[0] = RETINA_MIN_PITCH;
                return pitches;
            }
            let step = (RETINA_MAX_PITCH - RETINA_MIN_PITCH) / (RETINA_ROWS as f32 - 1.0);
            let mut i = 0;
            while i < RETINA_ROWS {
                pitches[i] = RETINA_MIN_PITCH + step * i as f32;
                i += 1;
            }
            pitches
        }

        /// Ângulos de pitch por linha da retina (rad), auto-calculados.
        /// Row 0 = horizonte alto (-0.10 rad), Row N-1 = chão (+0.05 rad).
        pub const RETINA_PITCHES: [f32; RETINA_ROWS] = compute_pitches();
    }

    // ── CÉREBRO (fonte: src/simulation/motor.rs, src/engine.rs) ────────
    pub mod brain {
        use super::vision::{RETINA_COLS};

        /// Calcula tamanho da sensory wave automaticamente.
        /// Estrutura: 0-127 (fóvea, periférica, telemetria, lookahead)
        ///          + 160-191 (retina horizon scan: 32 amostras)
        /// Total = 128 + 32 = 160, mas marginalia para segurança = 160 + RETINA_COLS
        const fn compute_sensory_size() -> usize {
            // Retina horizon usa índices 160 até 160+32
            // Mas deixar margem para RETINA_COLS em caso de ajustes futuros
            160 + 32 + (RETINA_COLS / 4) // +32 margem buffer
        }

        /// Sensory wave = array de canais i8.
        /// NÃO é retina direta — passa por pooling no motor.rs.
        /// Tamanho calculado automaticamente baseado em RETINA_ROWS/COLS.
        pub const SENSORY_WAVE_SIZE: usize = compute_sensory_size();

        /// Fast Brain: BitLinear 192→96 (instinct layer, pesos ternários no DNA).
        pub const L1_NEURONS: usize = 96;

        /// Cortex L2: 96→48 (deep composition).
        pub const L2_NEURONS: usize = 48;

        /// Cortex L3: 48→24 (pre-output).
        pub const L3_NEURONS: usize = 24;

        /// Slow Brain: FloatLinear 48→8 (córtex lento, roda a cada N ticks).
        pub const SLOW_BRAIN_INPUTS: usize = 48;
        pub const SLOW_BRAIN_OUTPUTS: usize = 8;
        pub const SLOW_BRAIN_TICK_INTERVAL: u32 = 8; // Roda 1x a cada 8 ticks (~8Hz @ 64Hz)
    }

    // ── GENÉTICA / DNA (fonte: src/evolution/dna_genesis.rs) ───────────
    pub mod genetics {
        use super::brain::*;

        // Versão atual do formato DNA
        pub const DNA_VERSION: u16 = 4;

        // Tamanho dos pesos (calculados a partir da arquitetura)
        pub const DNA_FAST_SIZE: usize = SENSORY_WAVE_SIZE * L1_NEURONS; // 192 × 96 = 18.432
        pub const DNA_SLOW_SIZE: usize = SLOW_BRAIN_INPUTS * SLOW_BRAIN_OUTPUTS; // 48 × 8 = 384

        // Legacy (compatibilidade com DNAs antigos)
        pub const DNA_FAST_SIZE_LEGACY: usize = 128 * 64; // 8.192
        pub const DNA_SLOW_SIZE_LEGACY: usize = 32 * 8;   // 256

        // Scars (trauma imutabilidade — Tier 6.6)
        pub const SCAR_TTL_INITIAL: u8 = 5;       // trava por 5 gerações
        pub const SCAR_FRACTION: f32 = 0.10;      // 10% dos pesos não-zero viram scar

        // Zona Protegida de axiomas (spawn.rs:17)
        pub const PROTECTED_ZONE_SIZE: usize = 819; // ~4.4% do DNA_FAST_SIZE

        // Mutação de Pesos
        pub const DEFAULT_MUTATION_RATE: f32 = 0.008; // 0.8% chance de flip por bit

        // Mutação Morfológica (Corpo)
        pub const MORPHO_MASS_RANGE: (f32, f32) = (700.0, 1400.0);
        pub const MORPHO_WHEELBASE_RANGE: (f32, f32) = (1.5, 3.0);
        pub const MORPHO_DOWNFORCE_RANGE: (f32, f32) = (0.5, 2.0);
        pub const MORPHO_GRIP_RANGE: (f32, f32) = (0.8, 1.2);
        pub const MORPHO_MUTATION_RATE: f32 = 0.05; // 5% por gene por gen
        pub const MORPHO_SIGMA_FRAC: f32 = 0.05;    // σ = 5% do range

        // Hiper-rede (Compressão de Pesos)
        pub const HYPERNET_COEFFS: usize = 64;
        pub const HYPERNET_EXPAND_SIZE: usize = 2048;
        pub const HYPERNET_MUTATION_FLIP_RATE: f32 = 0.02;
        pub const HYPERNET_SIGMA: f32 = 0.05;
        pub const HYPERNET_TERNARY_THRESH: f32 = 0.3;

        // Métricas (settings.rs:5)
        pub const FITNESS_HISTORY_CAP: usize = 100;
    }

    // ── SIMULAÇÃO (fonte: src/simulation/motor.rs) ─────────────────────
    pub mod simulation {
        // Timing
        pub const TICK_RATE: f32 = 64.0;
        pub const TIME_STEP: f32 = 1.0 / TICK_RATE;

        // Thresholds de Punição (Survival Policy)
        pub const STAGNATION_TIMEOUT_FRAMES: u32 = 960; // ~15s sem progresso = morte
        pub const WALL_DAMAGE_BASE: f32 = 4.0;
        pub const WALL_DAMAGE_MAX: f32 = 40.0;
        pub const WALL_DAMAGE_GRACE: f32 = 20.0;
        // Reduced 30→15 for narrow tracks: models need time to navigate tight corners
        pub const SPEED_BLEED_DAMAGE: f32 = 15.0;
        pub const WALL_PIN_TIME_S: f32 = 1.6;    // Segundos preso na parede para morrer
        pub const HP_GRACE_FRAMES: u32 = 480;     // Frames de proteção pós-volta

        // Recuperação
        pub const WAYPOINT_DIST_M: f32 = 25.0;   // Distância entre checkpoints (metros)
        pub const HEAL_AMOUNT: f32 = 40.0;        // HP recuperado por checkpoint

        // Axiomas (Injeção de Instinto)
        pub const AXIOM_DECAY_GENS: f32 = 50.0;  // Axiomas somem após N gerações
    }
}
