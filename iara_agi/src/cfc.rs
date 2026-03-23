#![allow(dead_code)]
#![allow(unused_variables)]
/// IARA v2101 - CfC (Closed-form Continuous-time) Cell
/// Based on Codex v2101 - Liquid Neural Networks for Temporal Flow

pub struct CfCCell {
    pub hidden_size: usize,
    pub state: Vec<f32>,
}

impl CfCCell {
    pub fn new(hidden_size: usize) -> Self {
        Self {
            hidden_size,
            state: vec![0.0; hidden_size],
        }
    }

    /// Liquid Flow: O tempo (dt) é um parâmetro de entrada direto.
    /// x(t) = sigmoid(-f(t)) * g + (1 - sigmoid(-f(t))) * h
    pub fn flow(&mut self, dt: f32) -> Vec<f32> {
        for s in self.state.iter_mut() {
            // Decaimento contínuo: A memória "derrete" se o tempo passar e nada ocorrer.
            let decay = (-0.5 * dt).exp();
            *s *= decay;
        }
        self.state.clone()
    }

    pub fn inject(&mut self, input: &[f32]) {
        for (s, &i) in self.state.iter_mut().zip(input.iter()) {
            *s = (*s + i).clamp(-2.0, 2.0);
        }
    }
}
