/// IARA v2101 - Closed-form Continuous-time (CfC)
/// Based on Codex v2101 - Module III: Temporal Flow

pub struct CfCCell {
    pub hidden_size: usize,
    // Learnable parameters (simplified for the prototype)
    pub f_theta: Vec<f32>, // Time-gate weights
    pub g_theta: Vec<f32>, // State transformation G
    pub h_theta: Vec<f32>, // State transformation H
}

impl CfCCell {
    pub fn new(hidden_size: usize) -> Self {
        Self {
            hidden_size,
            f_theta: vec![0.5; hidden_size],
            g_theta: vec![1.0; hidden_size],
            h_theta: vec![-1.0; hidden_size],
        }
    }

    /// Analytical solution: x(t) = sigmoid(-f*t) * g + (1 - sigmoid(-f*t)) * h
    pub fn flow(&self, t: f32) -> Vec<f32> {
        let mut state = vec![0.0; self.hidden_size];
        for i in 0..self.hidden_size {
            let f = self.f_theta[i];
            let g = self.g_theta[i];
            let h = self.h_theta[i];
            
            // Time gate: sigmoid(-f * t)
            let gate = 1.0 / (1.0 + (-f * t).exp());
            
            // Blend the two states G and H over continuous time
            state[i] = gate * g + (1.0 - gate) * h;
        }
        state
    }
}
