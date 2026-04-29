use crate::metabolism::Metabolism;
use crate::layers::BitLinear;
use crate::float_layers::FloatLinear;
use crate::resonance::ResonantNode;
use crate::types::SplineNode;
use crate::cfc::CfCCell;
use crate::kan::SensoryPreFilter;
use std::sync::RwLock;

/// Tier 6.8 — Mirror neurons / observational learning.
/// Estado GLOBAL (process-wide): média de retina do campeão da geração anterior
/// + última ação motora conhecida (throttle, steer). Populado pelo pipeline no
/// fim da gen (quando há coroação). Cada SovereignEngine consulta aqui via
/// `SovereignEngine::champion_snapshot()`. Sem lock contention significativo —
/// leitura simples por frame, escrita 1× por gen.
static CHAMPION_RETINA_MEAN: RwLock<Option<[f32; 64]>> = RwLock::new(None);
static CHAMPION_ACTION: RwLock<Option<[f32; 2]>> = RwLock::new(None);

/// Atualiza o snapshot global do campeão (chamado pelo pipeline no fim da gen).
pub fn set_global_champion_snapshot(retina_mean: [f32; 64], action: [f32; 2]) {
    if let Ok(mut g) = CHAMPION_RETINA_MEAN.write() { *g = Some(retina_mean); }
    if let Ok(mut g) = CHAMPION_ACTION.write() { *g = Some(action); }
}

/// Córtex estratégico — cérebro LENTO.
/// Roda a cada N ticks (subsampling), integra contexto e emite bias que
/// MODULA (não substitui) o cérebro rápido. Camada 32→8 ternária.
/// Saídas:
///   0: aggression_bias   (-1..+1)  modula pedal
///   1: throttle_cap_bias (-1..+1)  reduz teto do pedal em situação de risco
///   2: risk_damp         (-1..+1)  amortece volante em alta velocidade
///   3..7: reservado para evolução descobrir uso
pub struct SlowBrain {
    pub layer: FloatLinear,
    pub output: [f32; 8],        // held entre ticks lentos
    pub tick_counter: u32,
    pub tick_interval: u32,      // roda 1× a cada N ticks
}

impl SlowBrain {
    pub fn new() -> Self {
        Self {
            layer: FloatLinear::new(32, 8),
            output: [0.0; 8],
            tick_counter: 0,
            tick_interval: 8,       // roda 1× a cada 8 ticks (~8Hz @ 64Hz) — contexto fresco
        }
    }

    /// Injeta entrada agregada. Só recomputa se for tick de "córtex".
    pub fn tick(&mut self, input: &[i8; 32]) {
        self.tick_counter = self.tick_counter.wrapping_add(1);
        if self.tick_counter % self.tick_interval != 0 { return; }
        let raw = self.layer.forward(input);
        for i in 0..8 {
            self.output[i] = (raw[i] * 2.0).tanh();
        }
    }
}

/// Motor cognitivo reduzido ao essencial: percepção ternária (instinto),
/// córtex lento (contexto), metabolismo (custo de processamento) e ressonância
/// cardíaca (sinal de coerência que dispara mielinização).
///
/// Tier 2.1 ATIVO: CfCCell (64→64, dim-preserving) pré-processa a fatia de
/// retina do sensory_wave [0..64] antes do BitLinear. Isso dá MEMÓRIA TEMPORAL
/// — cada pixel retiniano carrega histórico filtrado, permitindo detecção de
/// movimento (parede se aproxima vs parede estática).
///
/// Design: retina pre-processor (dim-preserving), escolhido sobre "channel
/// remap" para não quebrar priming axiomático existente em spawn.rs (canais
/// 64-127 do sensory_wave permanecem intocados). Ver melhorias/02-cerebro-dormente.md§2.1.
///
/// `cfc_state` é RUNTIME-ONLY — não persiste em .dna v3 (reset a cada spawn).
pub struct SovereignEngine {
    pub metabolism: Metabolism,
    pub instinct_layer: BitLinear,         // 128 → 64 (Layer 1, sensory mixing)
    pub cortex_l2: crate::float_layers::FloatLinear,  // 64 → 32 (Layer 2, composição AND/OR)
    pub slow_brain: SlowBrain,
    pub heart_resonance: ResonantNode,
    pub cfc_state: CfCCell,
    /// Tier 2.2: KAN per-channel pré-filtro (128 splines × 16 coeffs).
    /// Aplicado ANTES do CfC — cada canal tem curva de resposta própria.
    /// Init identity (coeffs=1.0) → output ≈ input no gen 0.
    pub kan_filters: SensoryPreFilter,
    /// Tier 2.4: per-output-neuron phase-coherence tracker (dopamine-gated Hebbian).
    /// Incrementa para os top-K neurônios que disparam forte após cada forward;
    /// decay 0.99/tick; reset em evento de mielinização. RUNTIME-only.
    pub resonance_coherence: [f32; 64],
    pub current_time: f32,
    pub active_actions: Vec<SplineNode>,
    pub temperature: f32,
    pub final_intent: Vec<f32>,
    // ------------------------------------------------------------------
    // Tier 6.8 — Mirror neurons (per-instance override + cached snapshot)
    // ------------------------------------------------------------------
    /// Snapshot retina do campeão da gen anterior (média dos 16 frames do
    /// retina_history do winner). Opcional: quando None, o engine cai no global
    /// CHAMPION_RETINA_MEAN. Permite override local (ex: testes, spawn com DNA).
    pub champion_retina_snapshot: Option<[f32; 64]>,
    /// Última ação motora conhecida do campeão (throttle, steer).
    pub champion_action: Option<[f32; 2]>,
    // ------------------------------------------------------------------
    // Tier 6.1 — Predictive coding (brain-free baseline)
    // ------------------------------------------------------------------
    /// Previsão da retina para o próximo tick. EMA simples sobre retina real.
    /// Erro de predição = |current - predicted| (L1 médio) alimenta curiosity
    /// como drive intrínseco. RUNTIME-only.
    pub predicted_retina: [f32; 64],
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
}

impl SovereignEngine {
    pub fn new() -> Self {
        Self {
            metabolism: Metabolism::new(),
            instinct_layer: BitLinear::new(128, 64),
            cortex_l2: crate::float_layers::FloatLinear::new(64, 32),
            slow_brain: SlowBrain::new(),
            heart_resonance: ResonantNode::new(1.0),
            cfc_state: CfCCell::new(64),
            kan_filters: SensoryPreFilter::new(128),
            resonance_coherence: [0.0; 64],
            current_time: 0.0,
            active_actions: Vec::new(),
            temperature: 1.0,
            final_intent: vec![0.0; 64],
            champion_retina_snapshot: None,
            champion_action: None,
            predicted_retina: [0.0; 64],
            gate: crate::gate::GateBrain::new_uniform(0xDA7E),
            reflex: crate::specialist::SpecialistBrain::new(),
            last_chaos: 0.0,
            hard_veto_count: 0,
            soft_override_count: 0,
            gate_entropy_acc: 0.0,
            gate_entropy_n: 0,
            oscillators: (0..8).map(|i| {
                let freq = 0.5 + (i as f32) * (4.0 - 0.5) / 7.0;
                crate::resonance::ResonantNode::new(freq)
            }).collect(),
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
        use crate::safety::{Sentinel, IntentVec, SafetyFlags};

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
            if sentinel_out.flags.intersects(
                SafetyFlags::ESP | SafetyFlags::LANE_KEEP | SafetyFlags::REVERSE_OUT
            ) { 1.0 } else { 0.0 },
            if sentinel_out.flags.intersects(
                SafetyFlags::TCS | SafetyFlags::AQUA | SafetyFlags::CORNER_LIM
            ) { 1.0 } else { 0.0 },
            if self.hard_veto_count > 0 { 1.0 } else { 0.0 },
            0.0, 0.0, 0.0, 0.0,
        ];
        let w = self.gate.forward(&gi);
        self.gate_entropy_acc += crate::gate::GateBrain::entropy(&w);
        self.gate_entropy_n = self.gate_entropy_n.saturating_add(1);

        let cortex_iv = IntentVec { throttle: cortex_intent[0], steer: cortex_intent[1], brake: cortex_intent[2] };
        let reflex_iv = IntentVec { throttle: reflex_raw[0], steer: reflex_raw[1], brake: reflex_raw[2] };
        let sentinel_iv = sentinel_out.soft_proposal;

        let mixed = IntentVec {
            throttle: w[0]*cortex_iv.throttle + w[1]*reflex_iv.throttle + w[2]*sentinel_iv.throttle,
            steer:    w[0]*cortex_iv.steer    + w[1]*reflex_iv.steer    + w[2]*sentinel_iv.steer,
            brake:    w[0]*cortex_iv.brake    + w[1]*reflex_iv.brake    + w[2]*sentinel_iv.brake,
        };

        let final_iv = if let Some(veto) = sentinel_out.hard_veto {
            self.hard_veto_count = self.hard_veto_count.saturating_add(1);
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

    /// Tier 6.8 — consulta o snapshot do campeão: prefere override local,
    /// senão lê do global process-wide. Retorna None se nada foi setado.
    pub fn champion_snapshot(&self) -> Option<([f32; 64], [f32; 2])> {
        let retina = self.champion_retina_snapshot.or_else(|| {
            CHAMPION_RETINA_MEAN.read().ok().and_then(|g| *g)
        })?;
        let action = self.champion_action.or_else(|| {
            CHAMPION_ACTION.read().ok().and_then(|g| *g)
        }).unwrap_or([0.0, 0.0]);
        Some((retina, action))
    }

    pub fn execute_cycle(&mut self, sensory_wave: &[i8], target_phase: f32, delta_t: f32) {
        self.current_time += delta_t;

        if !self.metabolism.can_process() {
            // Tier 2.3 — Sleep consolidation hook (single call site).
            // Quando adenosine >= 1.0 (trigger de sleep) e ainda não consolidamos
            // neste ciclo, drena episodic_buffer pra dentro do slow_brain via
            // Hebbian ternário. Guard sleep_consolidated é resetado em update()
            // quando adenosine cai abaixo de 0.5 (acordado o suficiente).
            if self.metabolism.adenosine >= 1.0 && !self.metabolism.sleep_consolidated {
                let episodes: Vec<_> = self.metabolism.episodic_buffer.drain(..).collect();
                if !episodes.is_empty() {
                    // Tier 6.2 LITE — Dreamer-style rollouts: 10 passadas com
                    // shuffle LCG consolidam padrões salientes durante o sono.
                    let rollouts = 10usize;
                    crate::memory::dream_rollout(
                        &mut self.slow_brain.layer,
                        &episodes,
                        rollouts,
                    );
                    println!("[DREAM] rollouts={} episodes={}", rollouts, episodes.len());
                }
                self.metabolism.sleep_consolidated = true;
            }
            self.metabolism.update(delta_t);
            if self.metabolism.adenosine < 0.5 {
                self.metabolism.sleep_consolidated = false;
            }
            return;
        }

        // 0a. KAN PRE-FILTER (Tier 2.2): spline per-channel aplicado em TODOS
        //     os 128 canais ANTES do CfC. Cada canal aprende curva própria
        //     (log p/ wall_distance, sat p/ speed, step p/ grass, etc).
        //     Init identity (coeffs=1.0) → comportamento ≈ pré-KAN no gen 0.
        let mut kan_wave = [0i8; 128];
        self.kan_filters.apply(sensory_wave, &mut kan_wave);

        // 0b. CfC PRE-PROCESSOR (Tier 2.1): filtra retina [0..64] com memória
        //    temporal sobre o sinal JÁ passado pelo KAN. Estado persiste entre
        //    ticks (reset só em spawn/death). Canais 64..127 (internal/flow/
        //    proprioceptivos) passam somente pelo KAN (sem CfC).
        let mut filtered_wave = [0i8; 128];
        filtered_wave.copy_from_slice(&kan_wave);
        let mut filtered_retina = [0i8; 64];
        self.cfc_state.update(&kan_wave[0..64], &mut filtered_retina, delta_t);
        filtered_wave[0..64].copy_from_slice(&filtered_retina);

        // Tier 6.1 — Predictive coding como drive primário (brain-free baseline).
        // Compara retina atual (pós-KAN+CfC, normalizada [-1,1]) com a predição
        // do tick anterior. Erro L1 médio vira drive intrínseco (curiosity hook).
        // Predição p/ próximo tick = EMA (0.7 * prev + 0.3 * current) — baseline
        // sem cérebro; substituível por rede preditiva no futuro (Dreamer-like).
        let mut current_retina_f = [0.0f32; 64];
        for j in 0..64 {
            current_retina_f[j] = filtered_retina[j] as f32 / 127.0;
        }
        let mut err_acc = 0.0f32;
        for j in 0..64 {
            err_acc += (current_retina_f[j] - self.predicted_retina[j]).abs();
        }
        let prediction_error = err_acc / 64.0;
        self.metabolism.prediction_error = prediction_error;
        self.metabolism.apply_prediction_error(prediction_error);
        // EMA update p/ próximo tick.
        for j in 0..64 {
            self.predicted_retina[j] = 0.7 * self.predicted_retina[j] + 0.3 * current_retina_f[j];
        }

        // 1. Percepção single-layer (Stage 3 reverted: 2-layer regrediu 90% em wide tracks).
        let raw_output = self.instinct_layer.forward(&filtered_wave);
        let mut final_intent = [0.0f32; 64];
        for (i, &val) in raw_output.iter().enumerate() {
            if i < 64 {
                final_intent[i] = (val * 2.0).tanh();
            }
        }
        self.final_intent = final_intent.to_vec();

        // 1b. Tier 6.8 — Mirror neurons / observational learning.
        //     Compara a retina filtrada atual (pós-KAN+CfC) com a média retiniana
        //     do campeão da gen anterior via cosine similarity. Se > 0.8, injeta
        //     a ação motora do campeão (throttle, steer) como soft bias em
        //     final_intent[0..2] com peso MIRROR_W=0.05. Imitação sem sinal
        //     supervisionado: quando o racer se encontra em estado perceptivo
        //     parecido com o do campeão, puxa levemente sua saída na direção do
        //     que deu certo. Conta disparos em metabolism.mirror_activation_count
        //     (fitness hook pode ler pra bonificar imitação efetiva).
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
                    self.final_intent[0] = (self.final_intent[0] + MIRROR_W * champ_action[0]).clamp(-1.0, 1.0);
                    self.final_intent[1] = (self.final_intent[1] + MIRROR_W * champ_action[1]).clamp(-1.0, 1.0);
                }
                self.metabolism.mirror_activation_count =
                    self.metabolism.mirror_activation_count.saturating_add(1);
            }
        }

        // 2. Ressonância + Kuramoto credit assignment (Tier 2.4).
        //    Fire-together-wire-together com GATE dopaminérgico (Izhikevich 2007).
        //    - Per-neuron coherence tracker soma quando neurônio top-K dispara forte.
        //    - Decay 0.99/tick (esquecimento passivo).
        //    - Mielinização dispara SÓ quando (a) há pelo menos 1 neurônio winner
        //      (coherence > 0.95) E (b) dopamina > +0.1 (surpresa positiva = RPE).
        //    - Dopamina < -0.1 → SKIP (punição: não consolida pathway ativo).
        //    - Reset dos winners após evento; cap 10.0 preservado em layers.rs.
        //    - Heart resonance mantido como gate secundário de plausibilidade global.
        self.heart_resonance.synchronize(target_phase, delta_t);
        let heart_coherence = self.heart_resonance.get_coherence(target_phase);

        // Decay global + top-K reinforcement em final_intent.
        const TOP_K: usize = 8;
        const COHERENCE_GAIN: f32 = 0.12;
        const COHERENCE_DECAY: f32 = 0.99;
        const COHERENCE_FIRE_THRESHOLD: f32 = 0.95;
        const DOPAMINE_POSITIVE: f32 = 0.1;
        const DOPAMINE_NEGATIVE: f32 = -0.1;

        for c in self.resonance_coherence.iter_mut() {
            *c *= COHERENCE_DECAY;
        }
        // Top-K por |intent| (neurônios que mais fortemente dispararam).
        let mut ranked: [(usize, f32); 64] = std::array::from_fn(|i| (i, final_intent[i].abs()));
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for &(idx, mag) in ranked.iter().take(TOP_K) {
            // Ganho proporcional à intensidade de fogo (saturado em 1.0).
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
            // Dopamine-gated Hebbian: consolida pathways coerentes + recompensados.
            let lr_mod = self.metabolism.learning_modulation();
            let factor = heart_coherence * lr_mod * dopamine.min(1.0);
            self.instinct_layer.myelinate(factor);
            // Reset winners pra evitar mielinização repetida do mesmo subset.
            for c in self.resonance_coherence.iter_mut() {
                if *c > COHERENCE_FIRE_THRESHOLD {
                    *c = 0.0;
                }
            }
        } else if dopamine < DOPAMINE_NEGATIVE {
            // Punição: limpa coerência acumulada dos pathways recém-ativos
            // (evita consolidar movimento errado — credit misassignment).
            for &(idx, _) in ranked.iter().take(TOP_K) {
                self.resonance_coherence[idx] *= 0.5;
            }
        }

        // 3. Custo Metabólico
        // FIX CRÍTICO: era `0.002 / delta_t` (cost per tick = 0.12 @ 60Hz). A adenosina
        // atingia 1.0 em ~130ms → trigger_sleep → brain ignorava sensores por ~18s.
        // Isso explicava raw_steer congelado nos brain traces: o cérebro estava DORMINDO.
        // Correto: custo proporcional a delta_t (acumula ~0.002/s = exausto em ~500s).
        self.metabolism.accumulate_fatigue(0.002 * delta_t);
        self.metabolism.update(delta_t);
    }
}
