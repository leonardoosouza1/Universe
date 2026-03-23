#![allow(dead_code)]
#![allow(unused_variables)]
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct TernaryMesh {
    pub states: Vec<u64>,
    pub width: usize,
    pub height: usize,
}

impl TernaryMesh {
    pub fn new(w: usize, h: usize) -> Self {
        let size = (w * h) / 32 + 1;
        Self {
            states: vec![0; size],
            width: w,
            height: h,
        }
    }

    pub fn set_state(&mut self, x: usize, y: usize, bits: u8) {
        // Lógica de estado da malha
    }

    pub fn bitwise_propagate(&mut self) {
        // Propagação de energia
    }
}

fn val_to_bits(val: u8) -> u8 {
    val & 0b11
}
