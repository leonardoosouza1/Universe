//! Chaos signal: blends physical instability with Kuramoto neural decoherence.
//! Output [0, 1]: 0 = calm, 1 = total chaos. Drives Gate input.

use crate::resonance::{ResonantNode, coherence};

#[derive(Debug, Clone, Copy)]
pub struct ChaosInput {
    pub yaw_rate: f32,            // rad/s
    pub slip_avg: f32,            // [0, 1]
    pub retina_temporal_var: f32, // 0..N magnitude of frame-to-frame retina change, normalized
}

#[derive(Debug, Clone, Copy)]
pub struct ChaosScore {
    pub physical: f32,  // [0, 1]
    pub neural: f32,    // [0, 1] = 1 - coherence
    pub coherence: f32, // raw kuramoto
    pub combined: f32,  // max(physical, neural)
}

pub fn score(input: &ChaosInput, oscillators: &[ResonantNode]) -> ChaosScore {
    let phys = 0.4 * (input.yaw_rate.abs() / 2.0).tanh()
             + 0.4 * (input.slip_avg / 0.5).tanh()
             + 0.2 * input.retina_temporal_var.clamp(0.0, 1.0);
    let phys = phys.clamp(0.0, 1.0);
    let coh = coherence(oscillators);
    let neural = 1.0 - coh;
    let combined = phys.max(neural);
    ChaosScore { physical: phys, neural, coherence: coh, combined }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn calm_input() -> ChaosInput {
        ChaosInput { yaw_rate: 0.0, slip_avg: 0.0, retina_temporal_var: 0.0 }
    }

    #[test]
    fn calm_state_low_chaos() {
        let nodes: Vec<ResonantNode> = (0..4).map(|_| ResonantNode::new(1.0)).collect();
        let s = score(&calm_input(), &nodes);
        assert!(s.combined < 0.2);
    }

    #[test]
    fn high_yaw_high_physical_chaos() {
        let mut i = calm_input(); i.yaw_rate = 5.0;
        let s = score(&i, &[]);
        assert!(s.physical > 0.3);
    }

    #[test]
    fn decoherent_oscillators_high_neural_chaos() {
        let mut nodes: Vec<ResonantNode> = (0..4).map(|_| ResonantNode::new(1.0)).collect();
        for (i, n) in nodes.iter_mut().enumerate() {
            n.phase = (i as f32) * (2.0 * PI / 4.0);
        }
        let s = score(&calm_input(), &nodes);
        assert!(s.neural > 0.9);
        assert!(s.combined > 0.9);
    }

    #[test]
    fn combined_takes_max() {
        let mut i = calm_input(); i.yaw_rate = 5.0;
        let nodes: Vec<ResonantNode> = (0..4).map(|_| ResonantNode::new(1.0)).collect();
        let s = score(&i, &nodes);
        assert_eq!(s.combined, s.physical);
    }
}
