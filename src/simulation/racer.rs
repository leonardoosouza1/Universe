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
    pub reverse_timer: f32,   // >0 enquanto em manobra de ré de emergência; decrementa com dt
    pub progress_checkpoint: f32, // Último track_progress que marcou avanço real (+3m)
    pub progress_stall: f32,      // Segundos acumulados SEM ganhar 3m de pista (anti-donut)
    pub wall_ema_l: f32,          // EMA da distância à parede esquerda (decay ~1s) — sensação persistente
    pub wall_ema_r: f32,          // EMA da distância à parede direita
    pub proximity_sustain: f32,   // Segundos contínuos raspando alguma lateral (alerta sustentado)
    pub post_reverse_lockout: f32, // Cooldown pós-ré com throttle capado (evita "ré-acelera-bate")
    pub spawn_progress: f32,       // track_progress da posição de spawn — fitness usa (track_progress - spawn_progress)
    pub path: Vec<Vec3>,           // Histórico de posições (amostrado ~4Hz) — usado para desenhar traçado do campeão
    pub path_timer: f32,           // Cooldown entre amostras de path
    pub brain_trace: Vec<BrainSnapshot>, // Snapshots periódicos do cérebro (diagnóstico)
    pub trace_timer: f32,          // Cooldown entre snapshots
}

/// Frame de diagnóstico do cérebro do racer.
/// Capturado a ~2Hz e dumpeado nos reports em momentos chave (morte, campeão).
/// Revela se o sinal sensorial está chegando, se o volante responde, e
/// quanto foi atenuado entre o neurônio e a física.
#[derive(Clone)]
pub struct BrainSnapshot {
    pub t: f32,                    // Tempo de vida do carro
    pub speed_kmh: f32,            // Velocidade
    pub wall_prox_l: i8,           // Canal 85: proximidade parede esquerda (0=longe, 127=colada)
    pub wall_prox_r: i8,           // Canal 86: proximidade parede direita
    pub ttc_inv: i8,               // Canal 82: urgência de colisão frontal
    pub curv_25: i8,               // Canal 79: curvatura antecipada 25m
    pub retina_center_max: i8,     // Pico dos canais centrais da retina (10..22) — parede no centro
    pub raw_throttle: f32,         // final_intent[0] do cérebro
    pub raw_steer: f32,            // final_intent[1] (pré-atenuação)
    pub raw_brake: f32,            // final_intent[2]
    pub post_damp_steer: f32,      // steer_intent depois do risk_damp do slow brain
    pub physical_steer: f32,       // O que REALMENTE chegou no volante (com dynamic_steer_limit)
    pub steer_limit: f32,          // Fator dinâmico 1/(1+v*0.02)
    pub risk_damp: f32,            // Atenuação do slow brain (0..0.3)
    pub top_channels: [(u8, i8); 5], // Top-5 canais mais ativados (|valor|), com índice+valor
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
