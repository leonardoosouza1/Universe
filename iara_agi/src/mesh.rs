/// IARA v2101 - Ternary Mesh (Lago de 2-bits) with SIMD-style Bitwise Propagation
/// Based on Codex Part 1 & 5: Hardware de Padaria Optimization

pub struct TernaryMesh {
    /// Each u64 represents 32 nodes (2 bits per node).
    pub states: Vec<u64>,
    pub width: usize,
    pub height: usize,
}

impl TernaryMesh {
    pub fn new(width: usize, height: usize) -> Self {
        let total_nodes = width * height;
        let storage_size = (total_nodes + 31) / 32;
        Self {
            states: vec![0; storage_size],
            width,
            height,
        }
    }

    pub fn set_state(&mut self, x: usize, y: usize, bits: u8) {
        let index = y * self.width + x;
        let u64_idx = index / 32;
        let bit_shift = (index % 32) * 2;
        self.states[u64_idx] &= !(0b11 << bit_shift);
        self.states[u64_idx] |= (val_to_bits(bits) as u64) << bit_shift;
    }

    /// SIMD Bit-Packing Optimization: Propagate energy using bitwise XOR/AND
    /// Axiom: "Resolvendo 16 nós de ressonância simultaneamente em um único ciclo de clock."
    pub fn bitwise_propagate(&mut self) {
        for i in 0..self.states.len() {
            let current = self.states[i];
            if current == 0 { continue; } // Quiescent block

            // Simulate wave propagation by shifting bits and applying logical gates
            // This spreads the '1's (energy) to adjacent bits in the u64
            let spread = current | (current << 2) | (current >> 2);
            
            // Dissipate energy (Newtonian Cooling via bitwise mask)
            // Mask out some bits to simulate decay
            self.states[i] = spread & 0xAAAA_AAAA_AAAA_AAAA; 
        }
    }
}

fn val_to_bits(val: u8) -> u8 {
    val & 0b11
}
