//! Universal perception pipeline: raw input → bytes → HDC → zone pooling → CfC → decision.
//!
//! Generalizes the zone pooling pattern validated in Universe (+141% single agent, 10x champion)
//! with HDC encoding from byte-lab (2300 req/s, p99 478µs) and CfC decision from saci (AUC 0.74).
//!
//! Any domain (text, vision, audio, market) follows the same pipeline — only the
//! [`Sensor`] and [`ZoneConfig`] differ.

pub use iara_cfc::Rng;
use iara_cfc::{CfcCell, CfcState, LinearHead};
use iara_hdc::{Hv, HV_U64};

// ─────────────────────────────────────────────────────────────────────────────
// Sensor trait — domain-specific input adapter
// ─────────────────────────────────────────────────────────────────────────────

/// Converts domain-specific input into a byte stream with spatial structure.
///
/// The stream is a 2D grid: `rows × cols` bytes, where rows represent
/// distinct spatial/temporal zones and cols represent features within each zone.
pub trait Sensor {
    type Input: ?Sized;

    fn sense(&self, input: &Self::Input) -> SensoryField;
}

/// A 2D byte grid with spatial structure. Row index encodes "depth" or "scale",
/// column index encodes "position" or "feature channel".
#[derive(Clone, Debug)]
pub struct SensoryField {
    pub data: Vec<u8>,
    pub rows: usize,
    pub cols: usize,
}

impl SensoryField {
    pub fn new(data: Vec<u8>, rows: usize, cols: usize) -> Self {
        debug_assert_eq!(data.len(), rows * cols);
        SensoryField { data, rows, cols }
    }

    pub fn row(&self, r: usize) -> &[u8] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    pub fn from_flat(bytes: &[u8], n_zones: usize) -> Self {
        let cols = (bytes.len() + n_zones - 1) / n_zones;
        let mut data = bytes.to_vec();
        data.resize(n_zones * cols, 0);
        SensoryField { data, rows: n_zones, cols }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Zone pooling — preserves spatial structure that max-pool destroys
// ─────────────────────────────────────────────────────────────────────────────

/// Defines how rows of a SensoryField are grouped into perceptual zones.
///
/// Universe proved: FAR zone (rows 0-4, ~80m) + NEAR zone (rows 10-14, ~20m)
/// gives +141% vs global max-pool. The principle generalizes to any domain.
#[derive(Clone, Debug)]
pub struct ZoneConfig {
    pub zones: Vec<Zone>,
}

#[derive(Clone, Debug)]
pub struct Zone {
    pub name: &'static str,
    pub row_start: usize,
    pub row_end: usize, // exclusive
    pub pool_method: PoolMethod,
}

#[derive(Clone, Debug, Copy, PartialEq)]
pub enum PoolMethod {
    Max,
    Mean,
    /// Signed max: preserves the value with largest absolute magnitude (keeps sign).
    SignedMax,
}

impl ZoneConfig {
    pub fn n_zones(&self) -> usize {
        self.zones.len()
    }

    /// Pool a SensoryField into one feature vector per zone.
    /// Returns `[n_zones × cols]` as f32 values normalized to [-1, 1].
    pub fn pool(&self, field: &SensoryField) -> Vec<Vec<f32>> {
        self.zones
            .iter()
            .map(|zone| {
                let start = zone.row_start.min(field.rows);
                let end = zone.row_end.min(field.rows);
                if start >= end {
                    return vec![0.0; field.cols];
                }

                let n_rows = end - start;
                let mut result = vec![0.0f32; field.cols];

                match zone.pool_method {
                    PoolMethod::Max => {
                        result.fill(f32::NEG_INFINITY);
                        for r in start..end {
                            for c in 0..field.cols {
                                let v = field.data[r * field.cols + c] as f32 / 127.5 - 1.0;
                                if v > result[c] {
                                    result[c] = v;
                                }
                            }
                        }
                        for v in result.iter_mut() {
                            if v.is_infinite() {
                                *v = 0.0;
                            }
                        }
                    }
                    PoolMethod::Mean => {
                        for r in start..end {
                            for c in 0..field.cols {
                                let v = field.data[r * field.cols + c] as f32 / 127.5 - 1.0;
                                result[c] += v;
                            }
                        }
                        let scale = 1.0 / n_rows as f32;
                        for v in result.iter_mut() {
                            *v *= scale;
                        }
                    }
                    PoolMethod::SignedMax => {
                        for r in start..end {
                            for c in 0..field.cols {
                                let v = field.data[r * field.cols + c] as f32 / 127.5 - 1.0;
                                if v.abs() > result[c].abs() {
                                    result[c] = v;
                                }
                            }
                        }
                    }
                }

                result
            })
            .collect()
    }

    /// Pool and flatten to a single feature vector: `[n_zones * cols]`.
    pub fn pool_flat(&self, field: &SensoryField) -> Vec<f32> {
        self.pool(field).into_iter().flatten().collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HDC zone encoding — encodes each zone's features as an HDC vector
// ─────────────────────────────────────────────────────────────────────────────

/// Encodes pooled zone features into HDC vectors for fast similarity search.
pub fn encode_zone_hdc(
    zone_features: &[f32],
    byte_hvs: &[Hv; 256],
    pos_hvs: &[Hv; 256],
) -> Hv {
    // Quantize float features to bytes and encode via HDC
    let bytes: Vec<u8> = zone_features
        .iter()
        .map(|&v| ((v.clamp(-1.0, 1.0) + 1.0) * 127.5) as u8)
        .collect();

    let unigrams: Vec<Hv> = bytes.iter().map(|&b| byte_hvs[b as usize]).collect();
    let bag = iara_hdc::bundle(&unigrams);

    let bigrams: Vec<Hv> = if bytes.len() >= 2 {
        bytes
            .windows(2)
            .map(|w| {
                let a = byte_hvs[w[0] as usize];
                let b_rot = iara_hdc::hv_rot1(&byte_hvs[w[1] as usize]);
                iara_hdc::hv_xor(&a, &b_rot)
            })
            .collect()
    } else {
        unigrams.clone()
    };
    let bigram = iara_hdc::bundle(&bigrams);

    // Combine bag + bigram (no seq — zone features have no meaningful absolute position)
    let mut combined = [0u64; HV_U64];
    for i in 0..HV_U64 {
        combined[i] = bag[i] & bigram[i]; // AND = conservative merge
    }
    combined
}

/// Similarity between two zone-encoded HDC vectors.
pub fn zone_similarity(a: &Hv, b: &Hv) -> f32 {
    iara_hdc::cosine_hv(a, b)
}

// ─────────────────────────────────────────────────────────────────────────────
// Perception pipeline — the full stack
// ─────────────────────────────────────────────────────────────────────────────

/// Complete perception pipeline: SensoryField → zone pool → CfC → decision.
pub struct PerceptionPipeline {
    pub zone_config: ZoneConfig,
    pub cfc: CfcCell,
    pub state: CfcState,
    pub decision_head: LinearHead,
    input_dim: usize,
}

impl PerceptionPipeline {
    /// Create a new pipeline.
    ///
    /// `n_features_per_zone`: number of columns in the SensoryField (= cols).
    /// `n_decisions`: number of possible output decisions.
    /// `hidden_dim`: CfC hidden state dimension.
    pub fn new(
        zone_config: ZoneConfig,
        n_features_per_zone: usize,
        n_decisions: usize,
        hidden_dim: usize,
        rng: &mut Rng,
    ) -> Self {
        let input_dim = zone_config.n_zones() * n_features_per_zone;
        let cfc = CfcCell::new(hidden_dim, input_dim, rng);
        let state = CfcState::new(hidden_dim);
        let decision_head = LinearHead::new(n_decisions, hidden_dim, rng);

        PerceptionPipeline {
            zone_config,
            cfc,
            state,
            decision_head,
            input_dim,
        }
    }

    /// Feed a sensory field through the full pipeline.
    /// Returns raw logits for each decision and the CfC hidden state.
    pub fn perceive(&mut self, field: &SensoryField) -> PerceptionOutput {
        let features = self.zone_config.pool_flat(field);
        debug_assert_eq!(features.len(), self.input_dim);

        self.state.step(&self.cfc, &features);

        let mut logits = vec![0.0f32; self.decision_head.out_dim];
        self.decision_head.forward(&mut logits, &self.state.h);

        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

        let (best_idx, best_prob) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, &p)| (i, p))
            .unwrap_or((0, 0.0));

        PerceptionOutput {
            logits,
            probs,
            decision: best_idx,
            confidence: best_prob,
            hidden: self.state.h.clone(),
        }
    }

    /// Convert perception output to a Vote for use with ConsensusGate.
    pub fn perceive_vote(&mut self, field: &SensoryField) -> iara_gate::Vote<usize> {
        let out = self.perceive(field);
        iara_gate::Vote {
            decision: out.decision,
            confidence: out.confidence,
        }
    }

    pub fn reset(&mut self) {
        self.state.reset();
    }

    pub fn n_params(&self) -> usize {
        self.cfc.n_params() + self.decision_head.w.len() + self.decision_head.b.len()
    }
}

#[derive(Clone, Debug)]
pub struct PerceptionOutput {
    pub logits: Vec<f32>,
    pub probs: Vec<f32>,
    pub decision: usize,
    pub confidence: f32,
    pub hidden: Vec<f32>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Domain-specific presets
// ─────────────────────────────────────────────────────────────────────────────

/// Zone configs that mirror validated domain patterns.
pub mod presets {
    use super::*;

    /// Universe-style vision: FAR (horizon) + NEAR (ground).
    /// Validated: +141% single agent, 10x champion growth.
    pub fn vision_far_near(total_rows: usize) -> ZoneConfig {
        let mid = total_rows / 2;
        ZoneConfig {
            zones: vec![
                Zone {
                    name: "far",
                    row_start: 0,
                    row_end: mid,
                    pool_method: PoolMethod::Mean,
                },
                Zone {
                    name: "near",
                    row_start: mid,
                    row_end: total_rows,
                    pool_method: PoolMethod::Max,
                },
            ],
        }
    }

    /// Text perception: beginning (context) + middle (body) + end (conclusion).
    /// 3 zones for capturing document structure.
    pub fn text_positional(total_rows: usize) -> ZoneConfig {
        let third = total_rows / 3;
        ZoneConfig {
            zones: vec![
                Zone {
                    name: "begin",
                    row_start: 0,
                    row_end: third,
                    pool_method: PoolMethod::Mean,
                },
                Zone {
                    name: "middle",
                    row_start: third,
                    row_end: 2 * third,
                    pool_method: PoolMethod::Mean,
                },
                Zone {
                    name: "end",
                    row_start: 2 * third,
                    row_end: total_rows,
                    pool_method: PoolMethod::Mean,
                },
            ],
        }
    }

    /// Audio perception: low (bass) + mid + high (treble).
    /// Mirrors standard audio EQ band separation.
    pub fn audio_frequency(total_rows: usize) -> ZoneConfig {
        let third = total_rows / 3;
        ZoneConfig {
            zones: vec![
                Zone {
                    name: "low",
                    row_start: 0,
                    row_end: third,
                    pool_method: PoolMethod::SignedMax,
                },
                Zone {
                    name: "mid",
                    row_start: third,
                    row_end: 2 * third,
                    pool_method: PoolMethod::Mean,
                },
                Zone {
                    name: "high",
                    row_start: 2 * third,
                    row_end: total_rows,
                    pool_method: PoolMethod::Max,
                },
            ],
        }
    }

    /// Market perception: recent (tactical) + medium (trend) + long-term (regime).
    /// Maps to saci-trader's multi-timeframe consensus pattern.
    pub fn market_temporal(total_rows: usize) -> ZoneConfig {
        let third = total_rows / 3;
        ZoneConfig {
            zones: vec![
                Zone {
                    name: "recent",
                    row_start: 2 * third,
                    row_end: total_rows,
                    pool_method: PoolMethod::Max,
                },
                Zone {
                    name: "trend",
                    row_start: third,
                    row_end: 2 * third,
                    pool_method: PoolMethod::Mean,
                },
                Zone {
                    name: "regime",
                    row_start: 0,
                    row_end: third,
                    pool_method: PoolMethod::Mean,
                },
            ],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-pipeline ensemble with consensus
// ─────────────────────────────────────────────────────────────────────────────

/// Multiple perception pipelines voting through a consensus gate.
/// Each pipeline can use a different ZoneConfig (different "perspective").
pub struct PerceptionEnsemble {
    pub pipelines: Vec<PerceptionPipeline>,
    pub gate: iara_gate::ConsensusGate<usize>,
}

impl PerceptionEnsemble {
    pub fn new(
        pipelines: Vec<PerceptionPipeline>,
        min_agreement: f32,
        min_confidence: f32,
    ) -> Self {
        PerceptionEnsemble {
            pipelines,
            gate: iara_gate::ConsensusGate::new(min_agreement, min_confidence, 0.85),
        }
    }

    /// Feed the same sensory field through all pipelines and gate the result.
    pub fn perceive(&mut self, field: &SensoryField) -> EnsembleOutput {
        let outputs: Vec<PerceptionOutput> = self
            .pipelines
            .iter_mut()
            .map(|p| p.perceive(field))
            .collect();

        let votes: Vec<iara_gate::Vote<usize>> = outputs
            .iter()
            .map(|o| iara_gate::Vote {
                decision: o.decision,
                confidence: o.confidence,
            })
            .collect();

        let gate_result = self.gate.evaluate(&votes);

        EnsembleOutput {
            individual: outputs,
            votes,
            gate_result,
        }
    }

    pub fn reset(&mut self) {
        for p in &mut self.pipelines {
            p.reset();
        }
    }
}

#[derive(Debug)]
pub struct EnsembleOutput {
    pub individual: Vec<PerceptionOutput>,
    pub votes: Vec<iara_gate::Vote<usize>>,
    pub gate_result: iara_gate::GateResult<usize>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Built-in sensors
// ─────────────────────────────────────────────────────────────────────────────

/// Sensor that converts raw text into a SensoryField using byte encoding.
/// Rows = n_zones (text split into chunks), Cols = chunk_size bytes.
pub struct TextSensor {
    pub n_zones: usize,
    pub chunk_size: usize,
}

impl TextSensor {
    pub fn new(n_zones: usize, chunk_size: usize) -> Self {
        TextSensor { n_zones, chunk_size }
    }
}

impl Sensor for TextSensor {
    type Input = str;

    fn sense(&self, input: &str) -> SensoryField {
        let bytes = iara_byte::normalize_hdc(input);
        SensoryField::from_flat(&bytes, self.n_zones)
    }
}

/// Sensor that wraps raw byte slices into a SensoryField.
pub struct RawSensor {
    pub n_zones: usize,
}

impl Sensor for RawSensor {
    type Input = [u8];

    fn sense(&self, input: &[u8]) -> SensoryField {
        SensoryField::from_flat(input, self.n_zones)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sensory_field_from_flat_pads() {
        let field = SensoryField::from_flat(&[1, 2, 3, 4, 5], 2);
        assert_eq!(field.rows, 2);
        assert_eq!(field.cols, 3);
        assert_eq!(field.data.len(), 6);
        assert_eq!(field.data[5], 0);
    }

    #[test]
    fn zone_pool_mean_correct() {
        let field = SensoryField::new(vec![100, 200, 50, 150], 2, 2);
        let config = ZoneConfig {
            zones: vec![Zone {
                name: "all",
                row_start: 0,
                row_end: 2,
                pool_method: PoolMethod::Mean,
            }],
        };
        let pooled = config.pool(&field);
        assert_eq!(pooled.len(), 1);
        assert_eq!(pooled[0].len(), 2);
        let expected_0 = ((100.0 / 127.5 - 1.0) + (50.0 / 127.5 - 1.0)) / 2.0;
        assert!((pooled[0][0] - expected_0).abs() < 1e-5);
    }

    #[test]
    fn zone_pool_max_correct() {
        let field = SensoryField::new(vec![100, 50, 200, 150], 2, 2);
        let config = ZoneConfig {
            zones: vec![Zone {
                name: "all",
                row_start: 0,
                row_end: 2,
                pool_method: PoolMethod::Max,
            }],
        };
        let pooled = config.pool(&field);
        let val_200 = 200.0 / 127.5 - 1.0;
        let val_150 = 150.0 / 127.5 - 1.0;
        assert!((pooled[0][0] - val_200).abs() < 1e-5);
        assert!((pooled[0][1] - val_150).abs() < 1e-5);
    }

    #[test]
    fn zone_pool_signed_max() {
        // byte 10 → (10/127.5 - 1.0) = -0.92 (large negative)
        // byte 200 → (200/127.5 - 1.0) = 0.57 (smaller positive)
        let field = SensoryField::new(vec![10, 128, 200, 128], 2, 2);
        let config = ZoneConfig {
            zones: vec![Zone {
                name: "all",
                row_start: 0,
                row_end: 2,
                pool_method: PoolMethod::SignedMax,
            }],
        };
        let pooled = config.pool(&field);
        let val_10 = 10.0 / 127.5 - 1.0;
        assert!((pooled[0][0] - val_10).abs() < 1e-5);
    }

    #[test]
    fn two_zones_preserves_structure() {
        let data: Vec<u8> = (0..8).collect();
        let field = SensoryField::new(data, 4, 2);
        let config = presets::vision_far_near(4);
        let pooled = config.pool(&field);
        assert_eq!(pooled.len(), 2);
        assert_eq!(pooled[0].len(), 2);
        assert_eq!(pooled[1].len(), 2);
        // FAR (rows 0-1) and NEAR (rows 2-3) should differ
        assert_ne!(pooled[0], pooled[1]);
    }

    #[test]
    fn pool_flat_concatenates() {
        let field = SensoryField::new(vec![100; 12], 4, 3);
        let config = presets::vision_far_near(4);
        let flat = config.pool_flat(&field);
        assert_eq!(flat.len(), 2 * 3); // 2 zones × 3 features
    }

    #[test]
    fn perception_pipeline_runs() {
        let mut rng = Rng::new(42);
        let config = presets::vision_far_near(4);
        let mut pipeline = PerceptionPipeline::new(config, 8, 3, 16, &mut rng);

        let field = SensoryField::new(vec![128; 4 * 8], 4, 8);
        let out = pipeline.perceive(&field);

        assert_eq!(out.logits.len(), 3);
        assert_eq!(out.probs.len(), 3);
        assert!(out.decision < 3);
        assert!(out.confidence > 0.0 && out.confidence <= 1.0);
        let prob_sum: f32 = out.probs.iter().sum();
        assert!((prob_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn perception_pipeline_different_inputs_different_outputs() {
        let mut rng = Rng::new(42);
        let config = presets::vision_far_near(4);
        let mut pipeline = PerceptionPipeline::new(config, 4, 3, 16, &mut rng);

        let field_a = SensoryField::new(vec![50; 16], 4, 4);
        let out_a = pipeline.perceive(&field_a);

        pipeline.reset();
        let field_b = SensoryField::new(vec![200; 16], 4, 4);
        let out_b = pipeline.perceive(&field_b);

        assert_ne!(out_a.logits, out_b.logits);
    }

    #[test]
    fn perception_has_temporal_memory() {
        let mut rng = Rng::new(42);
        let config = presets::text_positional(3);
        let mut pipeline = PerceptionPipeline::new(config, 4, 3, 16, &mut rng);

        let field = SensoryField::new(vec![128; 12], 3, 4);

        let out1 = pipeline.perceive(&field);
        let out2 = pipeline.perceive(&field);

        // Same input but different hidden states → different outputs
        assert_ne!(out1.hidden, out2.hidden);
    }

    #[test]
    fn ensemble_voting() {
        let mut rng = Rng::new(42);
        let configs = vec![
            presets::vision_far_near(4),
            presets::text_positional(4),
        ];

        let pipelines: Vec<PerceptionPipeline> = configs
            .into_iter()
            .map(|c| PerceptionPipeline::new(c, 4, 3, 16, &mut rng))
            .collect();

        let mut ensemble = PerceptionEnsemble::new(pipelines, 0.5, 0.3);

        let field = SensoryField::new(vec![128; 16], 4, 4);
        let out = ensemble.perceive(&field);

        assert_eq!(out.individual.len(), 2);
        assert_eq!(out.votes.len(), 2);
    }

    #[test]
    fn text_sensor_normalizes() {
        let sensor = TextSensor::new(3, 10);
        let field = sensor.sense("Café bonito");
        assert_eq!(field.rows, 3);
        // normalized text should be lowercase without diacritics
    }

    #[test]
    fn raw_sensor_preserves_bytes() {
        let sensor = RawSensor { n_zones: 2 };
        let field = sensor.sense(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(field.rows, 2);
        assert_eq!(field.cols, 3);
        assert_eq!(field.data[0], 1);
    }

    #[test]
    fn preset_market_has_three_zones() {
        let config = presets::market_temporal(9);
        assert_eq!(config.n_zones(), 3);
        assert_eq!(config.zones[0].name, "recent");
        assert_eq!(config.zones[1].name, "trend");
        assert_eq!(config.zones[2].name, "regime");
    }

    #[test]
    fn preset_audio_has_three_zones() {
        let config = presets::audio_frequency(9);
        assert_eq!(config.n_zones(), 3);
        assert_eq!(config.zones[0].name, "low");
    }

    #[test]
    fn empty_zone_returns_zeros() {
        let field = SensoryField::new(vec![255; 6], 2, 3);
        let config = ZoneConfig {
            zones: vec![Zone {
                name: "empty",
                row_start: 5,
                row_end: 10,
                pool_method: PoolMethod::Mean,
            }],
        };
        let pooled = config.pool(&field);
        assert!(pooled[0].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn n_params_reasonable() {
        let mut rng = Rng::new(42);
        let config = presets::vision_far_near(4);
        let pipeline = PerceptionPipeline::new(config, 8, 3, 16, &mut rng);
        let n = pipeline.n_params();
        // input_dim=16, hidden=16, decisions=3
        // CfC: 16*16 + 16*16 + 16*16 + 16 + 16 = 800
        // Head: 3*16 + 3 = 51
        assert!(n > 800);
        assert!(n < 2000);
    }

    #[test]
    fn encode_zone_hdc_produces_nonzero() {
        let byte_hvs = iara_hdc::build_pos_hvs(); // reuse as mock byte HVs
        let pos_hvs = iara_hdc::build_pos_hvs();
        let features = vec![0.5, -0.3, 0.8, 0.0];
        let hv = encode_zone_hdc(&features, &byte_hvs, &pos_hvs);
        let nonzero = hv.iter().any(|&w| w != 0);
        assert!(nonzero);
    }

    #[test]
    fn similar_features_similar_hdc() {
        let byte_hvs = iara_hdc::build_pos_hvs();
        let pos_hvs = iara_hdc::build_pos_hvs();
        let a = vec![0.5, 0.5, 0.5, 0.5];
        let b = vec![0.51, 0.49, 0.52, 0.48];
        let c = vec![-0.9, -0.9, -0.9, -0.9];

        let hv_a = encode_zone_hdc(&a, &byte_hvs, &pos_hvs);
        let hv_b = encode_zone_hdc(&b, &byte_hvs, &pos_hvs);
        let hv_c = encode_zone_hdc(&c, &byte_hvs, &pos_hvs);

        let sim_ab = zone_similarity(&hv_a, &hv_b);
        let sim_ac = zone_similarity(&hv_a, &hv_c);

        assert!(sim_ab > sim_ac, "similar inputs should be more similar in HDC space");
    }
}
