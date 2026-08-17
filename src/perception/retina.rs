/// IARA Perception — retina adapter over iara-perception.
///
/// Converts the raw retina `[i8; RETINA_TOTAL]` (14 rows × 126 cols) into a
/// compact SensoryField via column max-pooling (126→16 bins), then processes
/// through a learned CfC pipeline:
///   retina → col pool (126→16) → zone pool (FAR+MID+NEAR) → CfC(16) → 4-class head
///
/// Column pooling keeps the CfC small enough for DNA (~1140 params = 4.6KB f32).
/// Evolution optimizes these weights — random init learns nothing.
///
/// Integration: motor.rs injects perception hidden state into sensory_wave
/// before execute_cycle, so the existing brain layers see temporal visual features.
use iara_perception::{
    PerceptionOutput, PerceptionPipeline, PoolMethod, Rng, Sensor, SensoryField, Zone, ZoneConfig,
};

use crate::config::iara::vision::{RETINA_COLS, RETINA_ROWS};

pub const PERCEPTION_HIDDEN: usize = 16;
pub const PERCEPTION_DECISIONS: usize = 4;
pub const PERCEPTION_BINS: usize = 16;

/// Sensor adapter: converts `[i8; RETINA_TOTAL]` to compact SensoryField.
/// Pools 126 columns → 16 bins via max-pool per row, then shifts i8→u8.
pub struct RetinaSensor;

impl RetinaSensor {
    fn sense_retina(&self, input: &[i8]) -> SensoryField {
        let cols_per_bin = RETINA_COLS / PERCEPTION_BINS;
        let mut data = vec![128u8; RETINA_ROWS * PERCEPTION_BINS];
        for row in 0..RETINA_ROWS {
            for bin in 0..PERCEPTION_BINS {
                let start = bin * cols_per_bin;
                let end = if bin == PERCEPTION_BINS - 1 {
                    RETINA_COLS
                } else {
                    (bin + 1) * cols_per_bin
                };
                let mut max_val = -128i8;
                for col in start..end {
                    let v = input[row * RETINA_COLS + col];
                    if v > max_val {
                        max_val = v;
                    }
                }
                data[row * PERCEPTION_BINS + bin] = (max_val as i16 + 128) as u8;
            }
        }
        SensoryField::new(data, RETINA_ROWS, PERCEPTION_BINS)
    }
}

impl Sensor for RetinaSensor {
    type Input = [i8];

    fn sense(&self, input: &Self::Input) -> SensoryField {
        self.sense_retina(input)
    }
}

/// 3-zone layout matching Universe's depth structure:
///   FAR  = rows 0-4  (~80m ahead) → mean pool (curve shape)
///   MID  = rows 5-9  (~40m ahead) → mean pool (transition)
///   NEAR = rows 10-13 (~20m ahead) → max pool (immediate obstacles)
fn retina_zones() -> ZoneConfig {
    ZoneConfig {
        zones: vec![
            Zone {
                name: "far",
                row_start: 0,
                row_end: 5,
                pool_method: PoolMethod::Mean,
            },
            Zone {
                name: "mid",
                row_start: 5,
                row_end: 10,
                pool_method: PoolMethod::Mean,
            },
            Zone {
                name: "near",
                row_start: 10,
                row_end: 14,
                pool_method: PoolMethod::Max,
            },
        ],
    }
}

/// Learned retina perception with DNA-evolvable weights.
///
/// Pipeline: retina [14×126 i8] → col pool [14×16] → 3-zone pool → CfC(16) → 4-class head
/// Total params: ~1140 f32 (4.6KB in DNA)
pub struct RetinaPerception {
    sensor: RetinaSensor,
    pub pipeline: PerceptionPipeline,
}

impl RetinaPerception {
    pub fn new() -> Self {
        let mut rng = Rng::new(0xCAFE_BABE_DEAD_BEEF);
        let zones = retina_zones();
        let pipeline = PerceptionPipeline::new(
            zones,
            PERCEPTION_BINS,
            PERCEPTION_DECISIONS,
            PERCEPTION_HIDDEN,
            &mut rng,
        );
        Self {
            sensor: RetinaSensor,
            pipeline,
        }
    }

    pub fn perceive(&mut self, retina: &[i8]) -> PerceptionOutput {
        let field = self.sensor.sense(retina);
        self.pipeline.perceive(&field)
    }

    pub fn inject_into_wave(
        &mut self,
        retina: &[i8],
        wave: &mut [i8],
        offset: usize,
        strength: f32,
    ) {
        let output = self.perceive(retina);
        let end = (offset + output.hidden.len()).min(wave.len());
        for (i, &h) in output.hidden.iter().enumerate() {
            if offset + i >= end {
                break;
            }
            let existing = wave[offset + i] as f32;
            let injected = existing + h.tanh() * 127.0 * strength;
            wave[offset + i] = injected.clamp(-127.0, 127.0) as i8;
        }
    }

    pub fn reset(&mut self) {
        self.pipeline.reset();
    }

    pub fn n_params(&self) -> usize {
        self.pipeline.n_params()
    }

    pub fn decision(&mut self, retina: &[i8]) -> (usize, f32) {
        let out = self.perceive(retina);
        (out.decision, out.confidence)
    }

    // ── DNA weight serialization ────────────────────────────────────────

    /// Export all weights as a flat f32 vector for DNA storage.
    /// Layout: [cfc.w_xi | cfc.w_hi | cfc.w_f | cfc.b_i | cfc.b_f | head.w | head.b]
    pub fn weights_to_vec(&self) -> Vec<f32> {
        let cfc = &self.pipeline.cfc;
        let head = &self.pipeline.decision_head;
        let mut w = Vec::with_capacity(self.n_params());
        w.extend_from_slice(&cfc.w_xi);
        w.extend_from_slice(&cfc.w_hi);
        w.extend_from_slice(&cfc.w_f);
        w.extend_from_slice(&cfc.b_i);
        w.extend_from_slice(&cfc.b_f);
        w.extend_from_slice(&head.w);
        w.extend_from_slice(&head.b);
        w
    }

    /// Load weights from a flat f32 vector (inverse of weights_to_vec).
    pub fn load_weights(&mut self, w: &[f32]) {
        let cfc = &mut self.pipeline.cfc;
        let head = &mut self.pipeline.decision_head;

        let mut off = 0;
        let n = cfc.w_xi.len();
        cfc.w_xi.copy_from_slice(&w[off..off + n]);
        off += n;
        let n = cfc.w_hi.len();
        cfc.w_hi.copy_from_slice(&w[off..off + n]);
        off += n;
        let n = cfc.w_f.len();
        cfc.w_f.copy_from_slice(&w[off..off + n]);
        off += n;
        let n = cfc.b_i.len();
        cfc.b_i.copy_from_slice(&w[off..off + n]);
        off += n;
        let n = cfc.b_f.len();
        cfc.b_f.copy_from_slice(&w[off..off + n]);
        off += n;
        let n = head.w.len();
        head.w.copy_from_slice(&w[off..off + n]);
        off += n;
        let n = head.b.len();
        head.b.copy_from_slice(&w[off..off + n]);
        let _ = off + n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Import local do módulo de teste: `cargo fix` remove imports do escopo do
    // arquivo quando só os testes os usam, porque analisa sem `cfg(test)`.
    use crate::config::iara::vision::RETINA_TOTAL;

    #[test]
    fn retina_sensor_pools_columns() {
        let retina = vec![0i8; RETINA_TOTAL];
        let sensor = RetinaSensor;
        let field = sensor.sense(&retina);
        assert_eq!(field.rows, RETINA_ROWS);
        assert_eq!(field.cols, PERCEPTION_BINS);
        assert_eq!(field.data.len(), RETINA_ROWS * PERCEPTION_BINS);
    }

    #[test]
    fn retina_sensor_max_pools_correctly() {
        let mut retina = vec![0i8; RETINA_TOTAL];
        let cols_per_bin = RETINA_COLS / PERCEPTION_BINS;
        retina[cols_per_bin - 1] = 100;
        let sensor = RetinaSensor;
        let field = sensor.sense(&retina);
        assert_eq!(field.data[0], (100i16 + 128) as u8);
        assert_eq!(field.data[1], 128);
    }

    #[test]
    fn perception_compact_output() {
        let mut perc = RetinaPerception::new();
        let retina = vec![0i8; RETINA_TOTAL];
        let output = perc.perceive(&retina);
        assert_eq!(output.logits.len(), PERCEPTION_DECISIONS);
        assert_eq!(output.probs.len(), PERCEPTION_DECISIONS);
        assert_eq!(output.hidden.len(), PERCEPTION_HIDDEN);
        let prob_sum: f32 = output.probs.iter().sum();
        assert!((prob_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn weight_roundtrip() {
        let mut perc = RetinaPerception::new();
        let weights = perc.weights_to_vec();
        assert_eq!(weights.len(), perc.n_params());

        let mut perc2 = RetinaPerception::new();
        perc2.pipeline.cfc.w_xi.fill(0.0);
        perc2.load_weights(&weights);
        let weights2 = perc2.weights_to_vec();
        assert_eq!(weights, weights2, "weights should survive roundtrip");
    }

    #[test]
    fn n_params_reasonable() {
        let perc = RetinaPerception::new();
        let n = perc.n_params();
        assert!(n > 500 && n < 3000, "expected ~1140 params, got {n}");
    }

    #[test]
    fn inject_into_wave_blends() {
        let mut perc = RetinaPerception::new();
        let retina = vec![64i8; RETINA_TOTAL];
        let mut wave = vec![0i8; 192];
        perc.inject_into_wave(&retina, &mut wave, 170, 0.5);
        let modified = wave[170..170 + PERCEPTION_HIDDEN].iter().any(|&v| v != 0);
        assert!(modified);
    }

    #[test]
    fn temporal_state_differs() {
        let mut perc = RetinaPerception::new();
        let r1 = vec![100i8; RETINA_TOTAL];
        let r2 = vec![-100i8; RETINA_TOTAL];
        let out1 = perc.perceive(&r1);
        let out2 = perc.perceive(&r2);
        let diff: f32 = out1
            .hidden
            .iter()
            .zip(out2.hidden.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.01);
    }

    #[test]
    fn zone_config_matches_retina() {
        let zones = retina_zones();
        assert_eq!(zones.n_zones(), 3);
        assert_eq!(zones.zones[2].row_end, 14);
    }

    #[test]
    fn mutation_changes_behavior() {
        let mut parent = RetinaPerception::new();
        let retina = vec![50i8; RETINA_TOTAL];
        let parent_out = parent.perceive(&retina);

        let mut child_weights = parent.weights_to_vec();
        let mut rng = Rng::new(42);
        for w in child_weights.iter_mut() {
            *w += rng.signed() * 0.1;
        }

        let mut child = RetinaPerception::new();
        child.load_weights(&child_weights);
        let child_out = child.perceive(&retina);

        let diff: f32 = parent_out
            .hidden
            .iter()
            .zip(child_out.hidden.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.01,
            "mutated child should produce different output: diff={diff}"
        );
    }

    #[test]
    fn crossover_blends_parents() {
        let mut rng = Rng::new(123);
        let parent_a = RetinaPerception::new();
        let wa = parent_a.weights_to_vec();

        let mut wb = wa.clone();
        for w in wb.iter_mut() {
            *w += rng.signed() * 0.5;
        }

        let crossed: Vec<f32> = wa
            .iter()
            .zip(wb.iter())
            .enumerate()
            .map(|(i, (&a, &b))| if i % 2 == 0 { a } else { b })
            .collect();

        let mut child = RetinaPerception::new();
        child.load_weights(&crossed);
        let retina = vec![80i8; RETINA_TOTAL];
        let out = child.perceive(&retina);
        assert_eq!(out.hidden.len(), PERCEPTION_HIDDEN);

        let diff_a: f32 = crossed
            .iter()
            .zip(wa.iter())
            .map(|(c, a)| (c - a).abs())
            .sum();
        let diff_b: f32 = crossed
            .iter()
            .zip(wb.iter())
            .map(|(c, b)| (c - b).abs())
            .sum();
        assert!(
            diff_a > 0.1 && diff_b > 0.1,
            "child should differ from both parents"
        );
    }
}
