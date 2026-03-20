/// IARA v2101 - Digital Metabolism & Homeostasis
/// Based on Codex v2101 - Module I

pub struct Metabolism {
    /// Digital Adenosine: accumulation of computational fatigue (0.0 to 1.0)
    pub adenosine: f32,
    /// Recovery rate during "Sleep" or low activity
    pub recovery_rate: f32,
    /// Current State: Active or Sleeping (Night Dojo)
    pub is_sleeping: bool,
    /// VRAM/Cache stress level
    pub cache_stress: f32,
}

impl Metabolism {
    pub fn new() -> Self {
        Self {
            adenosine: 0.0,
            recovery_rate: 0.05,
            is_sleeping: false,
            cache_stress: 0.0,
        }
    }

    /// Simulate the passage of time and the cost of action.
    /// Every operation in the engine adds to adenosine.
    pub fn accumulate_fatigue(&mut self, cost: f32) {
        if !self.is_sleeping {
            self.adenosine += cost;
            self.cache_stress += cost * 0.5; // VRAM fragmentation proxy

            // Check Control Barrier Function (CBF)
            if self.adenosine >= 1.0 || self.cache_stress >= 1.0 {
                self.trigger_sleep();
            }
        }
    }

    fn trigger_sleep(&mut self) {
        println!("[CBF] - Critical Fatigue Detected. Entering Night Dojo (Sleep).");
        self.is_sleeping = true;
    }

    /// Night Dojo: Consolidation and Cache Clearing
    pub fn update(&mut self, delta_t: f32) {
        if self.is_sleeping {
            self.adenosine -= self.recovery_rate * delta_t;
            self.cache_stress -= self.recovery_rate * 2.0 * delta_t;

            if self.adenosine <= 0.1 {
                self.is_sleeping = false;
                println!("[CBF] - Homeostasis Reached. Waking Up.");
            }
        } else {
            // Natural slow recovery if not fully stressed
            if self.adenosine > 0.0 {
                self.adenosine -= 0.001 * delta_t;
            }
        }
        
        // Clamp values
        self.adenosine = self.adenosine.clamp(0.0, 1.5);
        self.cache_stress = self.cache_stress.clamp(0.0, 1.5);
    }

    /// Check if the system can accept new tasks (Axiom: No processing during sleep)
    pub fn can_process(&self) -> bool {
        !self.is_sleeping
    }
}
