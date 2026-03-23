#![allow(dead_code)]
#![allow(unused_variables)]
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct CfCCell {
    pub state: Vec<f32>,
    pub hidden_size: usize,
}

impl CfCCell {
    pub fn new(hidden_size: usize) -> Self {
        Self {
            state: vec![0.0; hidden_size],
            hidden_size,
        }
    }

    pub fn inject(&mut self, inputs: &[f32]) {
        for (i, &val) in inputs.iter().enumerate() {
            if i < self.state.len() {
                // Integração Temporal: o estado atual é influenciado pelo novo pulso
                self.state[i] = self.state[i] * 0.5 + val * 0.5;
            }
        }
    }

    pub fn flow(&mut self, delta_t: f32) -> Vec<f32> {
        for val in self.state.iter_mut() {
            // Decaimento Líquido: a informação "escorre" no tempo
            *val *= (-delta_t * 0.1).exp();
        }
        self.state.clone()
    }
}
