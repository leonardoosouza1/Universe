#![allow(dead_code)]
#![allow(unused_variables)]
/// IARA v2101 - SOVEREIGN WAVE-LINEAR
/// Matemática ancestral resolvendo problemas modernos.

use serde::{Serialize, Deserialize};
use std::fs::{self, File};
use std::io::{Write, Read};

#[derive(Serialize, Deserialize, Clone)]
pub struct BitLinear {
    pub weights: Vec<i8>,      
    pub weight_scale: f32,     
    pub input_dim: usize,
    pub output_dim: usize,
}

impl BitLinear {
    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        let tmp_path = format!("{}.tmp", path);
        let json = serde_json::to_string(self).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let mut file = File::create(&tmp_path)?;
        file.write_all(json.as_bytes())?;
        file.sync_all()?;
        fs::rename(&tmp_path, path)?;
        Ok(())
    }

    pub fn load_from_file(path: &str) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut json = String::new();
        file.read_to_string(&mut json)?;
        let brain: Self = serde_json::from_str(&json).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?;
        Ok(brain)
    }

    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let mut weights = vec![0; input_dim * output_dim];
        let mut rng = rand::thread_rng();
        use rand::Rng;
        for w in weights.iter_mut() {
            *w = rng.gen_range(-1..2); 
        }
        Self { weights, weight_scale: 1.0, input_dim, output_dim }
    }

    pub fn forward(&self, input: &[i8]) -> Vec<f32> {
        let mut output = vec![0.0; self.output_dim];
        for j in 0..self.output_dim {
            let mut acc: i32 = 0;
            for i in 0..self.input_dim {
                if i < input.len() {
                    let w = self.weights[j * self.input_dim + i] as i32;
                    let x = input[i] as i32;
                    acc += x * w;
                }
            }
            output[j] = (acc as f32) * self.weight_scale;
        }
        output
    }

    pub fn hebbian_update(&mut self, input: &[i8], outputs: &[f32], lr: f32) {
        for j in 0..self.output_dim {
            let out = outputs[j];
            if out.abs() < 0.05 { continue; } 

            for i in 0..self.input_dim {
                if i < input.len() {
                    let inp = input[i] as f32;
                    let grad = (inp * out) * lr;
                    let idx = j * self.input_dim + i;
                    let mut w = self.weights[idx] as f32;
                    w = (w + grad).clamp(-1.1, 1.1);
                    self.weights[idx] = if w > 0.5 { 1 } else if w < -0.5 { -1 } else { 0 };
                }
            }
        }
    }

    /// RESTAURADO: Método de aprendizado por ressonância
    pub fn myelinate(&mut self, coherence: f32) {
        self.weight_scale *= 1.0 + (coherence * 0.01);
    }
}
