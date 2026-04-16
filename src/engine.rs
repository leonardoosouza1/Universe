use crate::metabolism::Metabolism;
use crate::layers::BitLinear;
use crate::cfc::CfCCell;
use crate::kan::KanAxiomFilter;
use crate::resonance::ResonantNode;
use crate::memory::MultiverseSpiral;
use crate::types::SplineNode;
use crate::reality::RealityEngine;
use crate::repominer::RepoMiner;
use crate::mesh::TernaryMesh;

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
    pub tick_interval: u32,      // roda 1× a cada N ticks (default 32 ≈ 0.5Hz @ 64Hz)
}

impl SlowBrain {
    pub fn new() -> Self {
        Self {
            layer: BitLinear::new(32, 8),
            output: [0.0; 8],
            tick_counter: 0,
            tick_interval: 32,
        }
    }

    /// Injeta entrada agregada. Só recomputa se for tick de "córtex".
    pub fn tick(&mut self, input: &[i8; 32]) {
        self.tick_counter = self.tick_counter.wrapping_add(1);
        if self.tick_counter % self.tick_interval != 0 { return; }
        let raw = self.layer.forward(input);
        for i in 0..8 {
            self.output[i] = (raw[i] * 0.5).tanh(); // mesmo dampener do fast
        }
    }
}

pub struct SovereignEngine {
    pub metabolism: Metabolism,
    pub instinct_layer: BitLinear,
    pub slow_brain: SlowBrain,
    pub temporal_cfc: CfCCell,
    pub axiom_guard: KanAxiomFilter,
    pub heart_resonance: ResonantNode,
    pub memory_spiral: MultiverseSpiral,
    pub reality_engine: RealityEngine,
    pub repo_miner: RepoMiner,
    pub sensory_mesh: TernaryMesh,
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
            temporal_cfc: CfCCell::new(64),
            axiom_guard: KanAxiomFilter::new(),
            heart_resonance: ResonantNode::new(1.0),
            memory_spiral: MultiverseSpiral::new(),
            reality_engine: RealityEngine::new(8, 8),
            repo_miner: RepoMiner::new(),
            sensory_mesh: TernaryMesh::new(8, 8),
            current_time: 0.0,
            active_actions: Vec::new(),
            temperature: 1.0,
            final_intent: vec![0.0; 64],
        }
    }

    /// Execução por Ondas e Ressonância (Codex v2101)
    pub fn execute_cycle(&mut self, sensory_wave: &[i8], target_phase: f32, delta_t: f32) {
        self.current_time += delta_t;
        
        if !self.metabolism.can_process() {
            self.metabolism.update(delta_t);
            return;
        }

        // 1. Percepção Ternária: BitLinear → Saída Direta (SEM filtros atenuantes)
        let raw_output = self.instinct_layer.forward(sensory_wave);
        
        // Pipeline direto: BitLinear output → final_intent
        // Normaliza para range [-1, 1] usando tanh suave
        let mut final_intent = [0.0f32; 64];
        for (i, &val) in raw_output.iter().enumerate() {
            if i < 64 {
                final_intent[i] = (val * 10.0).tanh(); // Ganho massivo para reflexos instantâneos
            }
        }
        self.final_intent = final_intent.to_vec();

        // 2. Sistemas auxiliares (memória/aprendizado — NÃO filtram saída motora)
        self.heart_resonance.synchronize(target_phase, delta_t);
        let coherence = self.heart_resonance.get_coherence(target_phase);
        
        self.temporal_cfc.inject(&raw_output);
        let _ = self.temporal_cfc.flow(delta_t);

        // 3. Mielinização: alta coerência em alta velocidade grava aprendizado
        if coherence > 0.95 {
            self.instinct_layer.myelinate(coherence);
            self.memory_spiral.store_concept("FlowState", target_phase, self.current_time, coherence, "SOVEREIGN");
        }

        // 4. Custo Metabólico
        self.metabolism.accumulate_fatigue(0.002 / delta_t.max(0.001));
        self.metabolism.update(delta_t);
    }
}
