#![allow(dead_code)]
#![allow(unused_variables)]
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct Metabolism {
    pub adenosine: f32, // Acúmulo de fadiga (0.0 a 1.0)
    pub recovery_rate: f32,
    pub is_sleeping: bool,
    pub cache_stress: f32,
}

impl Metabolism {
    pub fn new() -> Self {
        Self {
            adenosine: 0.0,
            recovery_rate: 0.05, // TURBO RECOVERY: 10x mais rápido
            is_sleeping: false,
            cache_stress: 0.0,
        }
    }

    pub fn update(&mut self, delta_t: f32) {
        if self.is_sleeping {
            // Sono REM ultra-rápido: limpa a fadiga quase instantaneamente
            self.adenosine -= self.recovery_rate * 20.0 * delta_t;
            if self.adenosine <= 0.0 {
                self.adenosine = 0.0;
                self.is_sleeping = false;
            }
        } else {
            // Recuperação natural enquanto pensa
            self.adenosine -= self.recovery_rate * 0.5 * delta_t;
            if self.adenosine < 0.0 { self.adenosine = 0.0; }
        }
        
        // O cache_stress (Energia) se regenera se a fadiga estiver baixa
        if self.adenosine < 0.5 {
            self.cache_stress = (self.cache_stress + 0.1 * delta_t).clamp(0.0, 1.0);
        } else {
            self.cache_stress = (self.cache_stress - 0.05 * delta_t).max(0.0);
        }
    }

    pub fn accumulate_fatigue(&mut self, amount: f32) {
        // Reduzida drasticamente a penalidade por leitura (Escala Turbo)
        self.adenosine += amount * 0.01; 
        if self.adenosine >= 1.0 {
            self.adenosine = 1.0;
            self.is_sleeping = true;
        }
    }

    pub fn can_process(&self) -> bool {
        !self.is_sleeping && self.adenosine < 0.95
    }
}
