use crate::metabolism::Metabolism;
use crate::layers::BitLinear;
use crate::resonance::ResonantNode;
use crate::types::SplineNode;

/// Córtex estratégico — cérebro LENTO.
/// Roda a cada N ticks (subsampling), integra contexto e emite bias que
/// MODULA (não substitui) o cérebro rápido. Camada 32→8 ternária.
/// Saídas:
///   0: aggression_bias   (-1..+1)  modula pedal
///   1: throttle_cap_bias (-1..+1)  reduz teto do pedal em situação de risco
///   2: risk_damp         (-1..+1)  amortece volante em alta velocidade
///   3..7: reservado para evolução descobrir uso
pub struct SlowBrain {
    pub layer: BitLinear,
    pub output: [f32; 8],        // held entre ticks lentos
    pub tick_counter: u32,
    pub tick_interval: u32,      // roda 1× a cada N ticks
}

impl SlowBrain {
    pub fn new() -> Self {
        Self {
            layer: BitLinear::new(32, 8),
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
/// Módulos experimentais (CfC/KAN/MultiverseSpiral/Reality/RepoMiner/TernaryMesh)
/// permanecem no repositório como sandboxes de pesquisa mas não participam mais
/// do ciclo motor — dead-in-hot-path é pior que dead-on-disk: gastava tick budget
/// sem modular saída.
pub struct SovereignEngine {
    pub metabolism: Metabolism,
    pub instinct_layer: BitLinear,
    pub slow_brain: SlowBrain,
    pub heart_resonance: ResonantNode,
    pub current_time: f32,
    pub active_actions: Vec<SplineNode>,
    pub temperature: f32,
    pub final_intent: Vec<f32>,
}

impl SovereignEngine {
    pub fn new() -> Self {
        Self {
            metabolism: Metabolism::new(),
            instinct_layer: BitLinear::new(128, 64),
            slow_brain: SlowBrain::new(),
            heart_resonance: ResonantNode::new(1.0),
            current_time: 0.0,
            active_actions: Vec::new(),
            temperature: 1.0,
            final_intent: vec![0.0; 64],
        }
    }

    pub fn execute_cycle(&mut self, sensory_wave: &[i8], target_phase: f32, delta_t: f32) {
        self.current_time += delta_t;

        if !self.metabolism.can_process() {
            self.metabolism.update(delta_t);
            return;
        }

        // 1. Percepção Ternária: BitLinear → final_intent.
        let raw_output = self.instinct_layer.forward(sensory_wave);
        let mut final_intent = [0.0f32; 64];
        for (i, &val) in raw_output.iter().enumerate() {
            if i < 64 {
                final_intent[i] = (val * 2.0).tanh();
            }
        }
        self.final_intent = final_intent.to_vec();

        // 2. Ressonância: alta coerência entre fase-alvo e coração → mielinização.
        self.heart_resonance.synchronize(target_phase, delta_t);
        let coherence = self.heart_resonance.get_coherence(target_phase);
        if coherence > 0.95 {
            self.instinct_layer.myelinate(coherence);
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
