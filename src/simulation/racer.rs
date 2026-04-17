use bevy::prelude::*;
use crate::engine::SovereignEngine;

#[derive(Component)]
pub struct Racer {
    pub name: String,
    pub car_model: String,
    pub engine: SovereignEngine,
    pub is_dead: bool,
    pub amplitude: f32,
    pub phase: f32,
    pub pain_memories: Vec<PainSynapse>,
    pub prev_pos: Vec3,
    pub distance: f32,
    pub fuel: f32,
    pub hp: f32,
    pub total_laps: usize,
    pub stagnation: f32,
    pub memory: [f32; 5], // [throttle, steer, brake, prev_shift_up, prev_shift_down]
    pub visual_cortex: [i8; 64],
    // --- NOVOS CAMPOS (Sovereign Drive Evolution) ---
    pub tire_temp: f32,       // Carga térmica dos pneus 0.0-1.0
    pub engine_power: f32,    // Degradação mecânica 0.1-1.0 (1.0 = full)
    pub grass_time: f32,      // Tempo acumulado na grama (penalidade fitness)
    pub wall_time: f32,        // Tempo acumulado raspando parede (penalidade fitness)
    pub aquaplaning_timer: f32, // Timer de aquaplanagem (0.5s de perda de controle)
    pub joy_integral: f32,    // Dopamina acumulada: Σ(joy_tick * streak_mult * dt) — driver de fitness
    pub joy_streak: f32,      // Tempo contínuo em flow state (cap 5s); zera em "dor"
    pub wall_hits: u32,       // Contador de impactos na parede (3 strikes = dead)
    pub wall_cooldown: f32,   // Tempo até registrar novo hit na parede
    pub last_waypoint: usize, // Último waypoint alcançado para cura
    pub track_progress: f32,  // Progresso real ao longo da polilinha da pista (arc-length)
    pub grace_ticks: u32,     // Contador de frames desde spawn; imune a time_scale (grace real de ~10s)
}

pub struct PainSynapse {
    pub position: Vec3,
    pub pain_phase: f32,
    pub intensity: f32,
}

#[derive(Component)]
pub struct Skidmark {
    pub timer: f32,
}
