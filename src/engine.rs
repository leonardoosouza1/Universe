use crate::metabolism::Metabolism;
use crate::layers::BitLinear;
use crate::dense_layer::DenseLayer;
use crate::resonance::ResonantNode;
use crate::types::SplineNode;
use crate::cfc::CfCCell;
use crate::kan::SensoryPreFilter;
use crate::perception::RetinaPerception;
use crate::config::iara::vision::RETINA_TOTAL;
use crate::config::iara::brain::SENSORY_WAVE_SIZE;
use std::sync::RwLock;

/// Tier 6.8 — Mirror neurons / observational learning.
/// Estado GLOBAL (process-wide): média de retina do campeão da geração anterior
/// + última ação motora conhecida (throttle, steer). Populado pelo pipeline no
/// fim da gen (quando há coroação). Cada SovereignEngine consulta aqui via
/// `SovereignEngine::champion_snapshot()`. Sem lock contention significativo —
/// leitura simples por frame, escrita 1× por gen.
static CHAMPION_RETINA_MEAN: RwLock<Option<[f32; RETINA_TOTAL]>> = RwLock::new(None);
static CHAMPION_ACTION: RwLock<Option<[f32; 2]>> = RwLock::new(None);

/// Atualiza o snapshot global do campeão (chamado pelo pipeline no fim da gen).
pub fn set_global_champion_snapshot(retina_mean: [f32; RETINA_TOTAL], action: [f32; 2]) {
    if let Ok(mut g) = CHAMPION_RETINA_MEAN.write() { *g = Some(retina_mean); }
    if let Ok(mut g) = CHAMPION_ACTION.write() { *g = Some(action); }
}

/// Córtex estratégico — cérebro LENTO com arquitetura CfC + Dense f32.
/// Roda a cada N ticks (subsampling), integra contexto e emite ações motoras diretas.
///
/// Arquitetura:
///   input[48] → normalize [-1,1] → CfC(64, dim-expansion) → relu
///   → Dense1(64→16) → relu → Dense2(16→3) → tanh → [throttle, steer, brake]
///
/// SlowBrain mantém temporal state via CfC, permitindo aprendizado contínuo
/// de padrões contextuais (risco, fadiga, etc) sem ternário quantizado.
///
/// Output de 3 dims: [throttle, steer, brake] em [-1,1] (brake em [0,1]).
pub struct SlowBrain {
    pub cfc: CfCCell,                  // 64 canais com memória temporal
    pub dense1: DenseLayer,            // 64 → 16
    pub dense2: DenseLayer,            // 16 → 3 (throttle, steer, brake)
    pub output: [f32; 3],              // [throttle, steer, brake] held entre ticks
    pub tick_counter: u32,
    pub tick_interval: u32,            // roda 1× a cada N ticks
}

impl SlowBrain {
    pub fn new() -> Self {
        // Init simétrico σ=0.1. Usava um LCG escrito à mão cujo mapeamento final era
        // `x/2^31 - 1.0`, que devolve [-1,0) e não [-1,1): TODOS os 1024 pesos do dense1
        // e os 48 do dense2 nasciam negativos. Uma camada de sinal único é um somador,
        // então as três saídas viravam a mesma soma (correlação +0,9999, PC1 = 100% da
        // variância) e nunca positivas — `throttle_cap` ficava preso em 1,0. Medido em
        // `src/bin/slow_probe.rs`; agora vem de `rng::Lcg`, que tem teste de simetria.
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
            tick_interval: 8,  // roda 1× a cada 8 ticks (~8Hz @ 64Hz)
        }
    }

    /// Injeta entrada agregada de 48 dimensões (normalizada em [-1, 1]).
    /// Só recomputa se for tick de "córtex". Realiza forward completo:
    /// 1. Expande input[48] → cfc_input[64] (padding com zeros)
    /// 2. CfC(64) com memória temporal
    /// 3. Dense1(64→16) → relu
    /// 4. Dense2(16→3) → tanh
    /// Salva resultado em output[3].
    pub fn tick(&mut self, input: &[i8; 48], dt: f32) {
        self.tick_counter = self.tick_counter.wrapping_add(1);
        if self.tick_counter % self.tick_interval != 0 { return; }

        // Expande input[48] → cfc_input[64] com padding zeros
        let mut cfc_input = [0i8; 64];
        cfc_input[..48].copy_from_slice(input);

        // CfC update (temporal filtering)
        let mut cfc_out = [0i8; 64];
        self.cfc.update(&cfc_input, &mut cfc_out, dt);

        // Converte output CfC para f32 [-1, 1]
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

    /// Retorna weights agregados (Dense1 + Dense2) para compatibilidade com DNA.
    /// Total: (64×16) + 16 + (16×3) + 3 = 1024 + 16 + 48 + 3 = 1091 pesos.
    pub fn all_weights(&self) -> Vec<f32> {
        let mut all = Vec::new();
        all.extend_from_slice(&self.dense1.weights);
        all.extend_from_slice(&self.dense1.bias);
        all.extend_from_slice(&self.dense2.weights);
        all.extend_from_slice(&self.dense2.bias);
        all
    }

    /// Carrega weights agregados. Ignora tamanho incompatível (backward-compat).
    pub fn set_all_weights(&mut self, weights: &[f32]) {
        let mut offset = 0;
        let d1_w_size = self.dense1.weights.len();
        let d1_b_size = self.dense1.bias.len();
        let d2_w_size = self.dense2.weights.len();
        let d2_b_size = self.dense2.bias.len();

        if offset + d1_w_size <= weights.len() {
            self.dense1.weights.copy_from_slice(&weights[offset..offset + d1_w_size]);
            offset += d1_w_size;
        }
        if offset + d1_b_size <= weights.len() {
            self.dense1.bias.copy_from_slice(&weights[offset..offset + d1_b_size]);
            offset += d1_b_size;
        }
        if offset + d2_w_size <= weights.len() {
            self.dense2.weights.copy_from_slice(&weights[offset..offset + d2_w_size]);
            offset += d2_w_size;
        }
        if offset + d2_b_size <= weights.len() {
            self.dense2.bias.copy_from_slice(&weights[offset..offset + d2_b_size]);
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
    pub instinct_layer: BitLinear,         // 192 → 96 (Layer 1, sensory mixing)
    pub cortex_l2: crate::float_layers::FloatLinear,  // 96 → 48 (Layer 2, composição AND/OR)
    pub cortex_l3: crate::float_layers::FloatLinear,  // 48 → 24 (Layer 3, deep abstract)
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
    pub resonance_coherence: [f32; 96],
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
    pub champion_retina_snapshot: Option<[f32; RETINA_TOTAL]>,
    /// Última ação motora conhecida do campeão (throttle, steer).
    pub champion_action: Option<[f32; 2]>,
    // ------------------------------------------------------------------
    // Tier 6.1 — Predictive coding (brain-free baseline)
    // ------------------------------------------------------------------
    /// Previsão da retina para o próximo tick. EMA simples sobre retina real.
    /// Erro de predição = |current - predicted| (L1 médio) alimenta curiosity
    /// como drive intrínseco. RUNTIME-only.
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
    // Fadiga de SAÍDA (output fatigue, espelho de OUT_ALPHA/OUT_BETA em brain.wgsl)
    // ------------------------------------------------------------------
    /// Estado lento da fadiga de saída por canal (96 slots = 48 L2 + 48 bypass).
    ///
    /// Mecanismo idêntico ao brain.wgsl etapa 2:
    ///   out_fat[j] += OUT_ALPHA * (|raw[j]| - out_fat[j]) * dt
    ///   intent[j]  = tanh(sign(raw[j]) * max(0, |raw[j]| - OUT_BETA * out_fat[j]))
    ///
    /// OUT_ALPHA = 4.0, OUT_BETA = 0.85 — mesmos valores do shader.
    ///   τ ≈ 0.25 s (sobe em ~16 ticks para 63% do alvo); rápido o suficiente
    ///   para quebrar saturação antes de ~10 ticks sem engessar sinal variado.
    ///
    /// Estado RUNTIME-ONLY: resetado em spawn/death junto com cfc_state.
    /// Não persiste em .dna — ser recém-nascido começa sem fadiga de saída.
    pub out_fatigue: [f32; 96],
}

impl SovereignEngine {
    pub fn new() -> Self {
        Self {
            metabolism: Metabolism::new(),
            instinct_layer: BitLinear::new(SENSORY_WAVE_SIZE, 96),
            // Identidade + perturbação simétrica. Com o LCG quebrado a perturbação era
            // toda negativa (média -0,025 × 96 entradas = viés DC de -2,4 por saída),
            // um modo comum somado a TODAS as unidades antes de qualquer aprendizado.
            cortex_l2: {
                let mut l = crate::float_layers::FloatLinear::new(96, 48);
                let mut r = crate::rng::Lcg::new(0xC0DE_CAFE);
                for i in 0..4608 { l.weights[i] = r.symmetric() * 0.05; }
                for i in 0..48 { l.weights[i * 96 + i] += 1.0; }
                l
            },
            cortex_l3: {
                let mut l = crate::float_layers::FloatLinear::new(48, 24);
                let mut r = crate::rng::Lcg::new(0xDEAD_BEEF);
                for i in 0..1152 { l.weights[i] = r.symmetric() * 0.05; }
                for i in 0..24 { l.weights[i * 48 + i] += 1.0; }
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
            oscillators: (0..8).map(|i| {
                let freq = 0.5 + (i as f32) * (4.0 - 0.5) / 7.0;
                crate::resonance::ResonantNode::new(freq)
            }).collect(),
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
        crate::neuro::registrar_uso(&w);

        let mut cortex_iv = IntentVec { throttle: cortex_intent[0], steer: cortex_intent[1], brake: cortex_intent[2] };
        let mut reflex_iv = IntentVec { throttle: reflex_raw[0], steer: reflex_raw[1], brake: reflex_raw[2] };
        let mut sentinel_iv = sentinel_out.soft_proposal;

        // ABLAÇÃO DE VIA (ver `neuro.rs`): zera a intenção DEPOIS do portão, não os pesos.
        // Assim o portão continua vendo o mesmo mundo e distribuindo os mesmos pesos — o que
        // se mede é a falta da via, não uma mudança de comportamento do misturador.
        let zero = IntentVec { throttle: 0.0, steer: 0.0, brake: 0.0 };
        match crate::neuro::ablada() {
            crate::neuro::Via::Cortex => cortex_iv = zero,
            crate::neuro::Via::Reflexo => reflex_iv = zero,
            crate::neuro::Via::SentinelaSuave | crate::neuro::Via::SentinelaTotal => {
                sentinel_iv = zero
            }
            _ => {}
        }

        let mixed = IntentVec {
            throttle: w[0]*cortex_iv.throttle + w[1]*reflex_iv.throttle + w[2]*sentinel_iv.throttle,
            steer:    w[0]*cortex_iv.steer    + w[1]*reflex_iv.steer    + w[2]*sentinel_iv.steer,
            brake:    w[0]*cortex_iv.brake    + w[1]*reflex_iv.brake    + w[2]*sentinel_iv.brake,
        };

        // O veto DURO só cai na ablação total. Na suave ele fica: sem veto o carro bate nos
        // primeiros segundos e o número mediria "morreu cedo", não a contribuição da via.
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

    /// Tier 6.8 — consulta o snapshot do campeão: prefere override local,
    /// senão lê do global process-wide. Retorna None se nada foi setado.
    pub fn champion_snapshot(&self) -> Option<([f32; RETINA_TOTAL], [f32; 2])> {
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

        // 0a. KAN PRE-FILTER (Tier 2.2): spline per-channel aplicado em TODOS
        //     os 128 canais ANTES do CfC. Cada canal aprende curva própria
        //     (log p/ wall_distance, sat p/ speed, step p/ grass, etc).
        //     Init identity (coeffs=1.0) → comportamento ≈ pré-KAN no gen 0.
        // ABLAÇÃO POR SENTIDO (ver `neuro.rs`): zera o grupo de canais ANTES do KAN, do CfC
        // e de tudo mais. Assim a ablação significa "este sentido não existe" e não "este
        // sentido chega distorcido" — se fosse depois do filtro, o KAN já teria misturado o
        // canal ablado com os outros e o número mediria outra coisa.
        let mut onda_ablada = [0i8; SENSORY_WAVE_SIZE];
        let sensory_wave: &[i8] = if matches!(crate::neuro::ablada(), crate::neuro::Via::Sentido(_)) {
            let n = sensory_wave.len().min(SENSORY_WAVE_SIZE);
            onda_ablada[..n].copy_from_slice(&sensory_wave[..n]);
            crate::neuro::ablar_sentido(&mut onda_ablada[..n]);
            &onda_ablada[..n]
        } else {
            sensory_wave
        };

        let mut kan_wave = [0i8; SENSORY_WAVE_SIZE];
        self.kan_filters.apply(sensory_wave, &mut kan_wave);

        // 0b. CfC PRE-PROCESSOR: filtra retina [0..96] com memória temporal.
        let mut filtered_wave = [0i8; SENSORY_WAVE_SIZE];
        filtered_wave.copy_from_slice(&kan_wave);
        let mut filtered_retina = [0i8; 96];
        self.cfc_state.update(&kan_wave[0..96], &mut filtered_retina, delta_t);
        filtered_wave[0..96].copy_from_slice(&filtered_retina);

        // Tier 6.1 — Predictive coding como drive primário (brain-free baseline).
        // Compara retina atual (pós-KAN+CfC, normalizada [-1,1]) com a predição
        // do tick anterior. Erro L1 médio vira drive intrínseco (curiosity hook).
        // Predição p/ próximo tick = EMA (0.7 * prev + 0.3 * current) — baseline
        // sem cérebro; substituível por rede preditiva no futuro (Dreamer-like).
        let mut current_retina_f = [0.0f32; RETINA_TOTAL];
        for j in 0..RETINA_TOTAL {
            current_retina_f[j] = if j < filtered_retina.len() { filtered_retina[j] as f32 / 127.0 } else { 0.0 };
        }
        let mut err_acc = 0.0f32;
        for j in 0..RETINA_TOTAL {
            err_acc += (current_retina_f[j] - self.predicted_retina[j]).abs();
        }
        let prediction_error = err_acc / RETINA_TOTAL as f32;
        self.metabolism.prediction_error = prediction_error;
        self.metabolism.apply_prediction_error(prediction_error);
        // EMA update p/ próximo tick.
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

        // Fadiga de saída — espelho exato do brain.wgsl etapa 2 (OUT_ALPHA/OUT_BETA).
        //
        // Quebra ponto fixo de saída: intent[0] saturado em +1.0 desde tick 10.
        // A fadiga de input do CfC age no state recorrente, mas não alcança L2/bypass.
        //
        // Equação por canal j (mesma que o shader, mesma ordem de operações):
        //   out_fat[j] += OUT_ALPHA * (|raw[j]| - out_fat[j]) * dt
        //   intent[j]  = tanh(sign(raw) * max(0, |raw[j]| - OUT_BETA * out_fat[j]))
        //
        // O encolhimento age sobre a MAGNITUDE. A forma anterior subtraía direto
        // (`raw - OUT_BETA*out_fat`) e era assimétrica, porque `out_fat` rastreia
        // |raw|: com raw negativo sustentado ela SOMAVA em módulo e empurrava para
        // -0,999, enquanto com raw positivo atenuava para 0,15·raw. A fadiga, que
        // deveria ser neutra, enviesava a população inteira para baixo.
        //
        // OUT_ALPHA = 4.0: τ ≈ 0.25 s (63% do alvo em ~16 ticks @ 64 Hz).
        //   Suficientemente rápido para agir antes do ponto fixo se consolidar
        //   (~10 ticks), sem engessar sinal que varia normalmente entre ticks.
        // OUT_BETA = 0.85: em regime permanente (out_fat → |raw|), intent →
        //   tanh(|raw| × 0.15). Com |raw| = 3.8: tanh(0.57) ≈ 0.54 < 0.90 ✓.
        const OUT_ALPHA: f32 = 4.0;
        const OUT_BETA:  f32 = 0.85;

        let mut final_intent = [0.0f32; 96];
        // Canais 0..48: via L2 (val * 2.0 como no shader: acc * 2/127 * 127 = acc * 2)
        for (j, &val) in l2_raw.iter().enumerate().take(48) {
            let raw = val * 2.0;
            // Atualiza fadiga ANTES de aplicar (mesma ordem do shader):
            self.out_fatigue[j] += OUT_ALPHA * (raw.abs() - self.out_fatigue[j]) * delta_t;
            let adapted = raw.signum() * (raw.abs() - OUT_BETA * self.out_fatigue[j]).max(0.0);
            final_intent[j] = adapted.tanh();
        }
        // Canais 48..96: bypass de L1 (l1_raw[48..96] × 2.0).
        // Itera l1_raw[48..96] com índice k de 0..48; j = 48+k.
        // NOTA: não usar enumerate().skip(48) pois k começaria em 48 e j=96 estouraria.
        for (k, &val) in l1_raw[48..96].iter().enumerate() {
            let j = 48 + k;  // k ∈ [0,48), j ∈ [48,96)
            let raw = val * 2.0;
            self.out_fatigue[j] += OUT_ALPHA * (raw.abs() - self.out_fatigue[j]) * delta_t;
            let adapted = raw.signum() * (raw.abs() - OUT_BETA * self.out_fatigue[j]).max(0.0);
            final_intent[j] = adapted.tanh();
        }
        self.final_intent = final_intent.to_vec();
        let _ = &self.cortex_l3; // kept struct, forward unused

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
