use bevy::prelude::*;
use rand::prelude::*;

#[derive(Clone, Default)]
pub struct BitLinear {
    pub weights: Vec<i8>, 
    pub in_features: usize,
    pub out_features: usize,
    pub weight_scale: f32,
}

impl BitLinear {
    pub fn new(in_f: usize, out_f: usize) -> Self {
        let mut rng = rand::rng();
        let weights = (0..in_f * out_f).map(|_| {
            let r = rng.random_range(0..3);
            if r == 0 { -1 } else if r == 1 { 0 } else { 1 }
        }).collect();
        Self { weights, in_features: in_f, out_features: out_f, weight_scale: 1.0 }
    }

    pub fn forward(&self, input: &[i8]) -> Vec<i32> {
        let mut output = vec![0i32; self.out_features];
        for j in 0..self.out_features {
            let mut acc = 0i32;
            let offset = j * self.in_features;
            for i in 0..self.in_features {
                let w = self.weights[offset + i];
                let x = input[i] as i32;
                if w == 1 { acc += x; } else if w == -1 { acc -= x; }
            }
            output[j] = acc;
        }
        output
    }

    pub fn mutate(&mut self, rate: f32) {
        let mut rng = rand::rng();
        for w in self.weights.iter_mut() {
            if rng.random_bool(rate as f64) {
                let r = rng.random_range(0..3);
                *w = if r == 0 { -1 } else if r == 1 { 0 } else { 1 };
            }
        }
    }

    pub fn quantize(&mut self, weights: &[f32]) {
        self.weights = weights.iter().map(|&w| {
            if w > 0.3 { 1 } else if w < -0.3 { -1 } else { 0 }
        }).collect();
    }
}
