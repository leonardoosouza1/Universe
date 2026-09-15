use crate::cfc::CfCCell;
use crate::config::iara::brain::SENSORY_WAVE_SIZE;
use crate::config::iara::vision::RETINA_TOTAL;
use crate::dense_layer::DenseLayer;
use crate::kan::SensoryPreFilter;
use crate::layers::BitLinear;
use crate::metabolism::Metabolism;
use crate::perception::RetinaPerception;
use crate::resonance::ResonantNode;
use crate::types::SplineNode;
use std::sync::RwLock;

/// Tier 6.8 — Mirror neurons / observational learning.
/// GLOBAL (process-wide) state: retina mean of the previous generation's champion
/// + last known motor action (throttle, steer). Populated by the pipeline at the
/// end of the gen (when there is a coronation). Each SovereignEngine queries it via
/// `SovereignEngine::champion_snapshot()`. No significant lock contention —
/// simple read per frame, written 1× per gen.
static CHAMPION_RETINA_MEAN: RwLock<Option<[f32; RETINA_TOTAL]>> = RwLock::new(None);
static CHAMPION_ACTION: RwLock<Option<[f32; 2]>> = RwLock::new(None);

/// Updates the global champion snapshot (called by the pipeline at the end of the gen).
pub fn set_global_champion_snapshot(retina_mean: [f32; RETINA_TOTAL], action: [f32; 2]) {
    if let Ok(mut g) = CHAMPION_RETINA_MEAN.write() {
        *g = Some(retina_mean);
    }
    if let Ok(mut g) = CHAMPION_ACTION.write() {
        *g = Some(action);
    }
}

/// Strategic cortex — SLOW brain whose architecture is CfC + Dense f32.
/// Runs every N ticks (subsampling), integrates context and emits direct motor actions.
///
/// Architecture:
///   input[48] → normalize [-1,1] → CfC(64, dim-expansion) → relu
///   → Dense1(64→16) → relu → Dense2(16→3) → tanh → [throttle, steer, brake]
///
/// SlowBrain keeps temporal state via the CfC, allowing continuous learning
/// of contextual patterns (risk, fatigue, etc.) without quantized ternary.
///
/// 3-dim output: [throttle, steer, brake] in [-1,1] (brake in [0,1]).
pub struct SlowBrain {
    pub cfc: CfCCell,       // 64 channels with temporal memory
    pub dense1: DenseLayer, // 64 → 16
    pub dense2: DenseLayer, // 16 → 3 (throttle, steer, brake)
    pub output: [f32; 3],   // [throttle, steer, brake] held between ticks
    pub tick_counter: u32,
    pub tick_interval: u32, // runs 1× every N ticks
}

impl SlowBrain {
    pub fn new() -> Self {
        // Symmetric init σ=0.1. It used a hand-written LCG whose final mapping was
        // `x/2^31 - 1.0`, which returns [-1,0) and not [-1,1): ALL 1024 dense1 weights
        // and the 48 of dense2 were born negative. A single-sign layer is an adder,
        // so the three outputs became the same sum (correlation +0.9999, PC1 = 100% of the
        // variance) and never positive — `throttle_cap` stayed stuck at 1.0. Measured in
        // `src/bin/slow_probe.rs`; it now comes from `rng::Lcg`, which has a symmetry test.
        let mut dense1 = DenseLayer::new(64, 16);
        let mut rng = crate::rng::Lcg::new(0xDEAD_BEEF);
        for w in dense1.weights.iter_mut() {
            *w = rng.symmetric() * 0.1;
        }
        for b in dense1.bias.iter_mut() {
            *b = rng.symmetric() * 0.01;
        }

        let mut dense2 = DenseLayer::new(16, 3);
        let mut rng = crate::rng::Lcg::new(0xCAFE_BABE);
        for w in dense2.weights.iter_mut() {
            *w = rng.symmetric() * 0.1;
        }
        for b in dense2.bias.iter_mut() {
            *b = rng.symmetric() * 0.01;
        }

        Self {
            cfc: CfCCell::new(64),
            dense1,
            dense2,
            output: [0.0; 3],
            tick_counter: 0,
            tick_interval: 8, // runs 1× every 8 ticks (~8Hz @ 64Hz)
        }
    }

    /// Injects the 48-dimension aggregated input (normalized in [-1, 1]).
    /// Only recomputes on a "cortex" tick. Performs the full forward:
    /// 1. Expands input[48] → cfc_input[64] (zero padding)
    /// 2. CfC(64) with temporal memory
    /// 3. Dense1(64→16) → relu
    /// 4. Dense2(16→3) → tanh
    /// Stores the result in output[3].
    pub fn tick(&mut self, input: &[i8; 48], dt: f32) {
        self.tick_counter = self.tick_counter.wrapping_add(1);
        if self.tick_counter % self.tick_interval != 0 {
            return;
        }

        // Expands input[48] → cfc_input[64] with zero padding
        let mut cfc_input = [0i8; 64];
        cfc_input[..48].copy_from_slice(input);

        // The cortex is sampled once every `tick_interval` body ticks. Integrate the
        // elapsed physical time, not one body's `dt`; otherwise every slow time
        // constant is stretched by the sampling factor (8x with the default).
        let elapsed_dt = dt * self.tick_interval as f32;

        // CfC update (temporal filtering)
        let mut cfc_out = [0i8; 64];
        self.cfc.update(&cfc_input, &mut cfc_out, elapsed_dt);

        // Converts CfC output to f32 [-1, 1]
        let mut cfc_f32 = [0.0f32; 64];
        for i in 0..64 {
            cfc_f32[i] = (cfc_out[i] as f32) / 127.0;
        }

        // Dense1 → ReLU
        let d1_raw = self.dense1.forward(&cfc_f32);
        let mut d1_relu = [0.0f32; 16];
        for i in 0..16 {
            d1_relu[i] = d1_raw[i].max(0.0);
        }

        // Dense2 → Tanh
        let d2_raw = self.dense2.forward(&d1_relu);
        for i in 0..3 {
            self.output[i] = d2_raw[i].tanh();
        }
    }

    /// Returns aggregated weights (Dense1 + Dense2) for DNA compatibility.
    /// Total: (64×16) + 16 + (16×3) + 3 = 1024 + 16 + 48 + 3 = 1091 weights.
    pub fn all_weights(&self) -> Vec<f32> {
        let mut all = Vec::new();
        all.extend_from_slice(&self.dense1.weights);
        all.extend_from_slice(&self.dense1.bias);
        all.extend_from_slice(&self.dense2.weights);
        all.extend_from_slice(&self.dense2.bias);
        all
    }

    /// Loads aggregated weights.
    ///
    /// Fails explicitly if the size does not match — the previous version silently
    /// ignored divergences, causing cascading corruption: when a block
    /// did not fit, the offset did not advance and the next block read data from the wrong place.
    pub fn set_all_weights(&mut self, weights: &[f32]) {
        let d1_w_size = self.dense1.weights.len();
        let d1_b_size = self.dense1.bias.len();
        let d2_w_size = self.dense2.weights.len();
        let d2_b_size = self.dense2.bias.len();
        let expected = d1_w_size + d1_b_size + d2_w_size + d2_b_size;
        assert_eq!(
            weights.len(),
            expected,
            "slow_brain expects {} values ({}w+{}b+{}w+{}b), got {}",
            expected,
            d1_w_size,
            d1_b_size,
            d2_w_size,
            d2_b_size,
            weights.len()
        );
        let mut offset = 0;
        self.dense1
            .weights
            .copy_from_slice(&weights[offset..offset + d1_w_size]);
        offset += d1_w_size;
        self.dense1
            .bias
            .copy_from_slice(&weights[offset..offset + d1_b_size]);
        offset += d1_b_size;
        self.dense2
            .weights
            .copy_from_slice(&weights[offset..offset + d2_w_size]);
        offset += d2_w_size;
        self.dense2
            .bias
            .copy_from_slice(&weights[offset..offset + d2_b_size]);
    }
}

#[cfg(test)]
mod slow_brain_tests {
    use super::SlowBrain;

    #[test]
    fn subsampled_cortex_integrates_elapsed_physical_time() {
        let mut brain = SlowBrain::new();
        brain.cfc.denoise_mag = 0.0;
        brain.cfc.fatigue_alpha = 0.0;
        let input = [127_i8; 48];

        for _ in 0..brain.tick_interval {
            brain.tick(&input, 1.0 / 64.0);
        }

        let elapsed = brain.tick_interval as f32 / 64.0;
        let expected = brain.cfc.dc_gain * (1.0 - (-brain.cfc.decay_rates[0] * elapsed).exp());
        assert!((brain.cfc.state[0] - expected).abs() < 1e-6);
    }
}

/// Cognitive engine reduced to the essential: ternary perception (instinct),
/// slow cortex (context), metabolism (processing cost) and heart
/// resonance (coherence signal that triggers myelination).
///
/// Tier 2.1 ACTIVE: CfCCell (64→64, dim-preserving) pre-processes the retina
/// slice of the sensory_wave [0..64] before the BitLinear. That gives TEMPORAL MEMORY
/// — each retinal pixel carries filtered history, enabling motion
/// detection (approaching wall vs static wall).
///
/// Design: retina pre-processor (dim-preserving), chosen over "channel
/// remap" so as not to break the existing axiomatic priming in spawn.rs (sensory_wave
/// channels 64-127 remain untouched). See melhorias/02-cerebro-dormente.md§2.1.
///
/// `cfc_state` is RUNTIME-ONLY — it does not persist in .dna v3 (reset on every spawn).
pub struct SovereignEngine {
    pub metabolism: Metabolism,
    pub instinct_layer: BitLinear, // 192 → 96 (Layer 1, sensory mixing)
    pub cortex_l2: crate::float_layers::FloatLinear, // 96 → 48 (Layer 2, AND/OR composition)
    pub slow_brain: SlowBrain,
    pub heart_resonance: ResonantNode,
    pub cfc_state: CfCCell,
    /// Tier 2.2: KAN per-channel pre-filter (128 splines × 16 coeffs).
    /// Applied BEFORE the CfC — each channel has its own response curve.
    /// Init identity (coeffs=1.0) → output ≈ input at gen 0.
    pub kan_filters: SensoryPreFilter,
    /// Tier 2.4: per-output-neuron phase-coherence tracker (dopamine-gated Hebbian).
    /// Increments for the top-K neurons that fire strongly after each forward;
    /// decay 0.99/tick; reset on a myelination event. RUNTIME-only.
    pub resonance_coherence: [f32; 96],
    pub current_time: f32,
    pub active_actions: Vec<SplineNode>,
    pub temperature: f32,
    pub final_intent: Vec<f32>,
    // ------------------------------------------------------------------
    // Tier 6.8 — Mirror neurons (per-instance override + cached snapshot)
    // ------------------------------------------------------------------
    /// Retina snapshot of the previous gen's champion (mean of the 16 frames of the
    /// winner's retina_history). Optional: when None, the engine falls back to the global
    /// CHAMPION_RETINA_MEAN. Allows a local override (e.g. tests, spawn with DNA).
    pub champion_retina_snapshot: Option<[f32; RETINA_TOTAL]>,
    /// Last known motor action of the champion (throttle, steer).
    pub champion_action: Option<[f32; 2]>,
    // ------------------------------------------------------------------
    // Tier 6.1 — Predictive coding (brain-free baseline)
    // ------------------------------------------------------------------
    /// Retina prediction for the next tick. Simple EMA over the real retina.
    /// Prediction error = |current - predicted| (mean L1) feeds curiosity
    /// as an intrinsic drive. RUNTIME-only.
    pub predicted_retina: [f32; RETINA_TOTAL],
    // ------------------------------------------------------------------
    // Trinity (Gate + Reflex + Sentinel) — Task 4 wiring
    // ------------------------------------------------------------------
    pub gate: crate::gate::GateBrain,
    pub reflex: crate::specialist::SpecialistBrain,
    pub last_chaos: f32,
    pub hard_veto_count: u32,
    pub soft_override_count: u32,
    pub gate_entropy_acc: f32,
    pub gate_entropy_n: u32,
    pub oscillators: Vec<crate::resonance::ResonantNode>,
    /// Learned retina perception — parallel CfC pathway over zone-pooled retina.
    /// Provides temporal visual features alongside the manual pooling in motor.rs.
    pub retina_perception: RetinaPerception,
    // ------------------------------------------------------------------
    // OUTPUT fatigue (mirror of OUT_ALPHA/OUT_BETA in brain.wgsl)
    // ------------------------------------------------------------------
    /// Slow per-channel output-fatigue state (96 slots = 48 L2 + 48 bypass).
    ///
    /// Mechanism identical to brain.wgsl stage 2:
    ///   out_fat[j] += OUT_ALPHA * (|raw[j]| - out_fat[j]) * dt
    ///   intent[j]  = tanh(sign(raw[j]) * max(0, |raw[j]| - OUT_BETA * out_fat[j]))
    ///
    /// OUT_ALPHA = 4.0, OUT_BETA = 0.85 — same values as the shader.
    ///   τ ≈ 0.25 s (rises in ~16 ticks to 63% of the target); fast enough
    ///   to break saturation before ~10 ticks without freezing a varying signal.
    ///
    /// RUNTIME-ONLY state: reset on spawn/death together with cfc_state.
    /// Does not persist in .dna — a newborn being starts with no output fatigue.
    pub out_fatigue: [f32; 96],
}

impl SovereignEngine {
    pub fn new() -> Self {
        Self {
            metabolism: Metabolism::new(),
            instinct_layer: BitLinear::new(SENSORY_WAVE_SIZE, 96),
            // Identity + symmetric perturbation. With the broken LCG the perturbation was
            // all negative (mean -0.025 × 96 inputs = DC bias of -2.4 per output),
            // a common mode added to ALL units before any learning.
            cortex_l2: {
                let mut l = crate::float_layers::FloatLinear::new(96, 48);
                let mut r = crate::rng::Lcg::new(0xC0DE_CAFE);
                for i in 0..4608 {
                    l.weights[i] = r.symmetric() * 0.05;
                }
                for i in 0..48 {
                    l.weights[i * 96 + i] += 1.0;
                }
                l
            },
            slow_brain: SlowBrain::new(),
            heart_resonance: ResonantNode::new(1.0),
            cfc_state: CfCCell::new(96),
            kan_filters: SensoryPreFilter::new(SENSORY_WAVE_SIZE),
            resonance_coherence: [0.0; 96],
            current_time: 0.0,
            active_actions: Vec::new(),
            temperature: 1.0,
            final_intent: vec![0.0; 96],
            champion_retina_snapshot: None,
            champion_action: None,
            predicted_retina: [0.0; RETINA_TOTAL],
            gate: crate::gate::GateBrain::new_uniform(0xDA7E),
            reflex: crate::specialist::SpecialistBrain::new(),
            last_chaos: 0.0,
            hard_veto_count: 0,
            soft_override_count: 0,
            gate_entropy_acc: 0.0,
            gate_entropy_n: 0,
            oscillators: (0..8)
                .map(|i| {
                    let freq = 0.5 + (i as f32) * (4.0 - 0.5) / 7.0;
                    crate::resonance::ResonantNode::new(freq)
                })
                .collect(),
            retina_perception: RetinaPerception::new(),
            out_fatigue: [0.0f32; 96],
        }
    }

    /// Trinity output. Returns final [throttle, steer, brake] after Gate blend + Sentinel hard veto.
    /// Caller is responsible for running `execute_cycle` first to populate cortex `final_intent`.
    /// Trinity is independent: cortex_intent is supplied by caller (typically from final_intent[0..3]).
    pub fn decide(
        &mut self,
        cortex_intent: [f32; 3],
        reflex_input: &[i8],
        safety_state: &crate::safety::SafetyState,
        chaos_input: &crate::chaos::ChaosInput,
    ) -> [f32; 3] {
        use crate::safety::{IntentVec, SafetyFlags, Sentinel};

        let chaos = crate::chaos::score(chaos_input, &self.oscillators);
        self.last_chaos = chaos.combined;

        let reflex_raw = self.reflex.forward_specialist(reflex_input);
        let sentinel_out = Sentinel::evaluate(safety_state);

        let gi: [f32; 16] = [
            chaos.combined,
            (safety_state.yaw_rate / 3.0).clamp(-1.0, 1.0),
            safety_state.slip_avg.clamp(0.0, 1.0),
            (1.0 / (1.0 + safety_state.ttc_forward.max(0.0))).clamp(0.0, 1.0),
            (safety_state.speed_kmh / 100.0).clamp(0.0, 1.0),
            chaos_input.retina_temporal_var.clamp(0.0, 1.0),
            chaos.coherence,
            (1.0 / (1.0 + safety_state.wall_left.max(0.0))).clamp(0.0, 1.0),
            (1.0 / (1.0 + safety_state.wall_right.max(0.0))).clamp(0.0, 1.0),
            if sentinel_out
                .flags
                .intersects(SafetyFlags::ESP | SafetyFlags::LANE_KEEP | SafetyFlags::REVERSE_OUT)
            {
                1.0
            } else {
                0.0
            },
            if sentinel_out
                .flags
                .intersects(SafetyFlags::TCS | SafetyFlags::AQUA | SafetyFlags::CORNER_LIM)
            {
                1.0
            } else {
                0.0
            },
            if self.hard_veto_count > 0 { 1.0 } else { 0.0 },
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        let w = self.gate.forward(&gi);
        self.gate_entropy_acc += crate::gate::GateBrain::entropy(&w);
        self.gate_entropy_n = self.gate_entropy_n.saturating_add(1);
        crate::neuro::registrar_uso(&w);

        let mut cortex_iv = IntentVec {
            throttle: cortex_intent[0],
            steer: cortex_intent[1],
            brake: cortex_intent[2],
        };
        let mut reflex_iv = IntentVec {
            throttle: reflex_raw[0],
            steer: reflex_raw[1],
            brake: reflex_raw[2],
        };
        let mut sentinel_iv = sentinel_out.soft_proposal;

        // PATHWAY ABLATION (see `neuro.rs`): zeroes the intent AFTER the gate, not the weights.
        // That way the gate keeps seeing the same world and distributing the same weights — what
        // is measured is the absence of the pathway, not a behaviour change of the mixer.
        let zero = IntentVec {
            throttle: 0.0,
            steer: 0.0,
            brake: 0.0,
        };
        match crate::neuro::ablada() {
            crate::neuro::Via::Cortex => cortex_iv = zero,
            crate::neuro::Via::Reflexo => reflex_iv = zero,
            crate::neuro::Via::SentinelaSuave | crate::neuro::Via::SentinelaTotal => {
                sentinel_iv = zero
            }
            _ => {}
        }

        let mixed = IntentVec {
            throttle: w[0] * cortex_iv.throttle
                + w[1] * reflex_iv.throttle
                + w[2] * sentinel_iv.throttle,
            steer: w[0] * cortex_iv.steer + w[1] * reflex_iv.steer + w[2] * sentinel_iv.steer,
            brake: w[0] * cortex_iv.brake + w[1] * reflex_iv.brake + w[2] * sentinel_iv.brake,
        };

        // The HARD veto only drops on total ablation. On the soft one it stays: without a veto the car
        // crashes within the first seconds and the number would measure "died early", not the pathway's contribution.
        let veto_ativo = match crate::neuro::ablada() {
            crate::neuro::Via::SentinelaTotal => None,
            _ => sentinel_out.hard_veto,
        };
        let final_iv = if let Some(veto) = veto_ativo {
            self.hard_veto_count = self.hard_veto_count.saturating_add(1);
            crate::neuro::registrar_veto();
            veto
        } else {
            if !sentinel_out.flags.is_empty() {
                self.soft_override_count = self.soft_override_count.saturating_add(1);
            }
            mixed
        };

        [
            final_iv.throttle.clamp(-1.0, 1.0),
            final_iv.steer.clamp(-1.0, 1.0),
            final_iv.brake.clamp(0.0, 1.0),
        ]
    }

    /// Tier 6.8 — queries the champion snapshot: prefers the local override,
    /// otherwise reads the process-wide global. Returns None if nothing was set.
    pub fn champion_snapshot(&self) -> Option<([f32; RETINA_TOTAL], [f32; 2])> {
        let retina = self
            .champion_retina_snapshot
            .or_else(|| CHAMPION_RETINA_MEAN.read().ok().and_then(|g| *g))?;
        let action = self
            .champion_action
            .or_else(|| CHAMPION_ACTION.read().ok().and_then(|g| *g))
            .unwrap_or([0.0, 0.0]);
        Some((retina, action))
    }

    pub fn execute_cycle(&mut self, sensory_wave: &[i8], target_phase: f32, delta_t: f32) {
        self.current_time += delta_t;

        if !self.metabolism.can_process() {
            // Tier 2.3 — Sleep consolidation hook (single call site).
            // When adenosine >= 1.0 (sleep trigger) and we have not yet consolidated
            // this cycle, drains episodic_buffer into the slow_brain via
            // ternary Hebbian. The sleep_consolidated guard is reset in update()
            // when adenosine falls below 0.5 (awake enough).
            if self.metabolism.adenosine >= 1.0 && !self.metabolism.sleep_consolidated {
                let _episodes: Vec<_> = self.metabolism.episodic_buffer.drain(..).collect();
                // Tier 6.2 LITE — Sleep consolidation disabled for new SlowBrain architecture.
                // Future: implement Hebbian consolidation via CfC + Dense state injection.
                // For now, clear buffer to prevent unbounded memory growth.
                self.metabolism.sleep_consolidated = true;
            }
            self.metabolism.update(delta_t);
            if self.metabolism.adenosine < 0.5 {
                self.metabolism.sleep_consolidated = false;
            }
            return;
        }

        // 0a. KAN PRE-FILTER (Tier 2.2): per-channel spline applied to ALL
        //     128 channels BEFORE the CfC. Each channel learns its own curve
        //     (log for wall_distance, sat for speed, step for grass, etc.).
        //     Init identity (coeffs=1.0) → behaviour ≈ pre-KAN at gen 0.
        // SENSE ABLATION (see `neuro.rs`): zeroes the channel group BEFORE the KAN, the CfC
        // and everything else. That way the ablation means "this sense does not exist" and not "this
        // sense arrives distorted" — if it came after the filter, the KAN would already have mixed the
        // ablated channel with the others and the number would measure something else.
        let mut onda_ablada = [0i8; SENSORY_WAVE_SIZE];
        let sensory_wave: &[i8] = if matches!(crate::neuro::ablada(), crate::neuro::Via::Sentido(_))
        {
            let n = sensory_wave.len().min(SENSORY_WAVE_SIZE);
            onda_ablada[..n].copy_from_slice(&sensory_wave[..n]);
            crate::neuro::ablar_sentido(&mut onda_ablada[..n]);
            &onda_ablada[..n]
        } else {
            sensory_wave
        };

        let mut kan_wave = [0i8; SENSORY_WAVE_SIZE];
        self.kan_filters.apply(sensory_wave, &mut kan_wave);

        // 0b. CfC PRE-PROCESSOR: filters retina [0..96] with temporal memory.
        let mut filtered_wave = [0i8; SENSORY_WAVE_SIZE];
        filtered_wave.copy_from_slice(&kan_wave);
        let mut filtered_retina = [0i8; 96];
        self.cfc_state
            .update(&kan_wave[0..96], &mut filtered_retina, delta_t);
        filtered_wave[0..96].copy_from_slice(&filtered_retina);

        // Tier 6.1 — Predictive coding as the primary drive (brain-free baseline).
        // Compares the current retina (post-KAN+CfC, normalized [-1,1]) with the previous
        // tick's prediction. Mean L1 error becomes an intrinsic drive (curiosity hook).
        // Prediction for the next tick = EMA (0.7 * prev + 0.3 * current) — brainless
        // baseline; replaceable by a predictive network in the future (Dreamer-like).
        let mut current_retina_f = [0.0f32; RETINA_TOTAL];
        for j in 0..RETINA_TOTAL {
            current_retina_f[j] = if j < filtered_retina.len() {
                filtered_retina[j] as f32 / 127.0
            } else {
                0.0
            };
        }
        let mut err_acc = 0.0f32;
        for j in 0..RETINA_TOTAL {
            err_acc += (current_retina_f[j] - self.predicted_retina[j]).abs();
        }
        let prediction_error = err_acc / RETINA_TOTAL as f32;
        self.metabolism.prediction_error = prediction_error;
        self.metabolism.apply_prediction_error(prediction_error);
        // EMA update for the next tick.
        for j in 0..RETINA_TOTAL {
            self.predicted_retina[j] = 0.7 * self.predicted_retina[j] + 0.3 * current_retina_f[j];
        }

        // Stage 3: 2-layer cortex L1 → tanh → L2 (Retina HD v2: 192→96→48).
        let l1_raw = self.instinct_layer.forward(&filtered_wave);
        let mut l1_act = [0i8; 96];
        for (i, &v) in l1_raw.iter().enumerate().take(96) {
            l1_act[i] = ((v.tanh() * 127.0).clamp(-127.0, 127.0)) as i8;
        }
        let l2_raw = self.cortex_l2.forward(&l1_act);

        // Output fatigue — exact mirror of brain.wgsl stage 2 (OUT_ALPHA/OUT_BETA).
        //
        // Breaks the output fixed point: intent[0] saturated at +1.0 since tick 10.
        // The CfC's input fatigue acts on the recurrent state, but does not reach L2/bypass.
        //
        // Per-channel equation for j (same as the shader, same operation order):
        //   out_fat[j] += OUT_ALPHA * (|raw[j]| - out_fat[j]) * dt
        //   intent[j]  = tanh(sign(raw) * max(0, |raw[j]| - OUT_BETA * out_fat[j]))
        //
        // The shrinkage acts on the MAGNITUDE. The previous form subtracted directly
        // (`raw - OUT_BETA*out_fat`) and was asymmetric, because `out_fat` tracks
        // |raw|: with sustained negative raw it ADDED in modulus and pushed towards
        // -0.999, while with positive raw it attenuated to 0.15·raw. The fatigue, which
        // should be neutral, biased the whole population downward.
        //
        // OUT_ALPHA = 4.0: τ ≈ 0.25 s (63% of the target in ~16 ticks @ 64 Hz).
        //   Fast enough to act before the fixed point consolidates
        //   (~10 ticks), without freezing a signal that normally varies between ticks.
        // OUT_BETA = 0.85: at steady state (out_fat → |raw|), intent →
        //   tanh(|raw| × 0.15). With |raw| = 3.8: tanh(0.57) ≈ 0.54 < 0.90 ✓.
        const OUT_ALPHA: f32 = 4.0;
        const OUT_BETA: f32 = 0.85;

        let mut final_intent = [0.0f32; 96];
        // Channels 0..48: via L2 (val * 2.0 as in the shader: acc * 2/127 * 127 = acc * 2)
        for (j, &val) in l2_raw.iter().enumerate().take(48) {
            let raw = val * 2.0;
            // Updates fatigue BEFORE applying (same order as the shader):
            self.out_fatigue[j] += OUT_ALPHA * (raw.abs() - self.out_fatigue[j]) * delta_t;
            let adapted = raw.signum() * (raw.abs() - OUT_BETA * self.out_fatigue[j]).max(0.0);
            final_intent[j] = adapted.tanh();
        }
        // Channels 48..96: L1 bypass (l1_raw[48..96] × 2.0).
        // Iterates l1_raw[48..96] with index k from 0..48; j = 48+k.
        // NOTE: do not use enumerate().skip(48) since k would start at 48 and j=96 would overflow.
        for (k, &val) in l1_raw[48..96].iter().enumerate() {
            let j = 48 + k; // k ∈ [0,48), j ∈ [48,96)
            let raw = val * 2.0;
            self.out_fatigue[j] += OUT_ALPHA * (raw.abs() - self.out_fatigue[j]) * delta_t;
            let adapted = raw.signum() * (raw.abs() - OUT_BETA * self.out_fatigue[j]).max(0.0);
            final_intent[j] = adapted.tanh();
        }
        self.final_intent = final_intent.to_vec();

        // 1b. Tier 6.8 — Mirror neurons / observational learning.
        //     Compares the current filtered retina (post-KAN+CfC) with the retinal mean
        //     of the previous gen's champion via cosine similarity. If > 0.8, injects
        //     the champion's motor action (throttle, steer) as a soft bias into
        //     final_intent[0..2] with weight MIRROR_W=0.05. Imitation without a supervised
        //     signal: when the racer finds itself in a perceptual state
        //     similar to the champion's, it gently pulls its output towards
        //     what worked. Counts firings in metabolism.mirror_activation_count
        //     (a fitness hook can read it to reward effective imitation).
        if let Some((champ_retina, champ_action)) = self.champion_snapshot() {
            let mut dot = 0.0f32;
            let mut norm_a = 0.0f32;
            let mut norm_b = 0.0f32;
            for j in 0..64 {
                let a = filtered_retina[j] as f32 / 127.0;
                let b = champ_retina[j];
                dot += a * b;
                norm_a += a * a;
                norm_b += b * b;
            }
            let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-6);
            let cosine = dot / denom;
            if cosine > 0.8 {
                const MIRROR_W: f32 = 0.05;
                if self.final_intent.len() >= 2 {
                    self.final_intent[0] =
                        (self.final_intent[0] + MIRROR_W * champ_action[0]).clamp(-1.0, 1.0);
                    self.final_intent[1] =
                        (self.final_intent[1] + MIRROR_W * champ_action[1]).clamp(-1.0, 1.0);
                }
                self.metabolism.mirror_activation_count =
                    self.metabolism.mirror_activation_count.saturating_add(1);
            }
        }

        // 2. Resonance + Kuramoto credit assignment (Tier 2.4).
        //    Fire-together-wire-together with a dopaminergic GATE (Izhikevich 2007).
        //    - Per-neuron coherence tracker adds up when a top-K neuron fires strongly.
        //    - Decay 0.99/tick (passive forgetting).
        //    - Myelination fires ONLY when (a) there is at least 1 winner neuron
        //      (coherence > 0.95) AND (b) dopamine > +0.1 (positive surprise = RPE).
        //    - Dopamine < -0.1 → SKIP (punishment: does not consolidate the active pathway).
        //    - Winners reset after the event; cap 10.0 preserved in layers.rs.
        //    - Heart resonance kept as a secondary gate of global plausibility.
        self.heart_resonance.synchronize(target_phase, delta_t);
        let heart_coherence = self.heart_resonance.get_coherence(target_phase);

        // Global decay + top-K reinforcement on final_intent.
        const TOP_K: usize = 8;
        const COHERENCE_GAIN: f32 = 0.12;
        const COHERENCE_DECAY: f32 = 0.99;
        const COHERENCE_FIRE_THRESHOLD: f32 = 0.95;
        const DOPAMINE_POSITIVE: f32 = 0.1;
        const DOPAMINE_NEGATIVE: f32 = -0.1;

        for c in self.resonance_coherence.iter_mut() {
            *c *= COHERENCE_DECAY;
        }
        // Top-K by |intent| (the neurons that fired most strongly).
        let mut ranked: [(usize, f32); 64] = std::array::from_fn(|i| (i, final_intent[i].abs()));
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for &(idx, mag) in ranked.iter().take(TOP_K) {
            // Gain proportional to firing intensity (saturated at 1.0).
            self.resonance_coherence[idx] =
                (self.resonance_coherence[idx] + COHERENCE_GAIN * mag.min(1.0)).min(1.0);
        }

        let winners = self
            .resonance_coherence
            .iter()
            .filter(|&&c| c > COHERENCE_FIRE_THRESHOLD)
            .count();

        let dopamine = self.metabolism.dopamine;
        if winners > 0 && dopamine > DOPAMINE_POSITIVE && heart_coherence > 0.5 {
            // Dopamine-gated Hebbian: consolidates coherent + rewarded pathways.
            let lr_mod = self.metabolism.learning_modulation();
            let factor = heart_coherence * lr_mod * dopamine.min(1.0);
            self.instinct_layer.myelinate(factor);
            // Reset winners to avoid repeated myelination of the same subset.
            for c in self.resonance_coherence.iter_mut() {
                if *c > COHERENCE_FIRE_THRESHOLD {
                    *c = 0.0;
                }
            }
        } else if dopamine < DOPAMINE_NEGATIVE {
            // Punishment: clears accumulated coherence of the recently active pathways
            // (avoids consolidating a wrong move — credit misassignment).
            for &(idx, _) in ranked.iter().take(TOP_K) {
                self.resonance_coherence[idx] *= 0.5;
            }
        }

        // 3. Metabolic cost
        // CRITICAL FIX: it was `0.002 / delta_t` (cost per tick = 0.12 @ 60Hz). Adenosine
        // reached 1.0 in ~130ms → trigger_sleep → the brain ignored sensors for ~18s.
        // That explained the frozen raw_steer in the brain traces: the brain was SLEEPING.
        // Correct: cost proportional to delta_t (accumulates ~0.002/s = exhausted in ~500s).
        self.metabolism.accumulate_fatigue(0.002 * delta_t);
        self.metabolism.update(delta_t);
    }
}
