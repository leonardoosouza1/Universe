use crate::layers::BitLinear;

#[derive(Clone)]
pub struct SovereignEngine {
    pub instinct_layer: BitLinear,
    pub current_time: f32,
}

impl SovereignEngine {
    pub fn new() -> Self {
        Self {
            instinct_layer: BitLinear::new(128, 64),
            current_time: 0.0,
        }
    }

    pub fn execute_cycle(&mut self, _sensory_wave: &[i8], _phase: f32, dt: f32) {
        self.current_time += dt;
    }

    pub fn get_intent(&self, input: &[i8]) -> Vec<i32> {
        self.instinct_layer.forward(input)
    }
}
