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
            recovery_rate: 0.005,
            is_sleeping: false,
            cache_stress: 0.0,
        }
    }

    pub fn update(&mut self, delta_t: f32) {
        if self.is_sleeping {
            self.adenosine -= self.recovery_rate * 5.0 * delta_t;
            if self.adenosine <= 0.0 {
                self.adenosine = 0.0;
                self.is_sleeping = false;
            }
        } else {
            // Recuperação natural lenta
            self.adenosine -= self.recovery_rate * 0.1 * delta_t;
            if self.adenosine < 0.0 { self.adenosine = 0.0; }
        }
    }

    pub fn accumulate_fatigue(&mut self, amount: f32) {
        self.adenosine += amount;
        if self.adenosine >= 1.0 {
            self.adenosine = 1.0;
            self.is_sleeping = true;
        }
    }

    pub fn can_process(&self) -> bool {
        !self.is_sleeping && self.adenosine < 0.95
    }
}
