#![allow(dead_code)]
#![allow(unused_variables)]
use serde::{Serialize, Deserialize};
use std::fs::{self, File};
use std::io::{Write, Read};

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

#[derive(Serialize, Deserialize, Clone)]
pub struct SovereignEngine {
    pub metabolism: Metabolism,
    pub instinct_layer: BitLinear,
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
}

impl SovereignEngine {
    pub fn new_scaled(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        println!("[ENGINE] Allocating Hyper-Brain: {} inputs, {} hidden", input_size, hidden_size);
        
        Self {
            metabolism: Metabolism::new(),
            instinct_layer: BitLinear::new(input_size, hidden_size),
            temporal_cfc: CfCCell::new(hidden_size),
            axiom_guard: KanAxiomFilter::new(),
            heart_resonance: ResonantNode::new(1.0),
            memory_spiral: MultiverseSpiral::new(),
            reality_engine: RealityEngine::new(64, 64),
            repo_miner: RepoMiner::new(),
            sensory_mesh: TernaryMesh::new(64, 64),
            current_time: 0.0,
            active_actions: Vec::new(),
            temperature: 1.0,
        }
    }

    /// MÉTODO SOBERANO: Salvar o Cérebro Completo
    pub fn save_dna(&self, path: &str) -> std::io::Result<()> {
        let tmp_path = format!("{}.tmp", path);
        let json = serde_json::to_string(self).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        
        let mut file = File::create(&tmp_path)?;
        file.write_all(json.as_bytes())?;
        file.sync_all()?;
        fs::rename(&tmp_path, path)?;
        
        println!("[DNA] Sovereign Consciousness archived: {}", path);
        Ok(())
    }

    /// MÉTODO SOBERANO: Carregar o Cérebro Completo
    pub fn load_dna(path: &str) -> std::io::Result<Self> {
        let mut file = File::open(path)?;
        let mut json = String::new();
        file.read_to_string(&mut json)?;
        let engine: Self = serde_json::from_str(&json).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })?;
        println!("[DNA] Sovereign Consciousness restored from {}", path);
        Ok(engine)
    }

    pub fn execute_cycle(&mut self, sensory_wave: &[i8], target_phase: f32, delta_t: f32) {
        self.current_time += delta_t;
        let energy_intensity: f32 = sensory_wave.iter().map(|&x| (x as f32).abs()).sum::<f32>() / sensory_wave.len() as f32;
        let stress_factor = 1.0 + (energy_intensity / 127.0);

        if !self.metabolism.can_process() {
            self.metabolism.update(delta_t);
            return;
        }

        let intent_amplitudes = self.instinct_layer.forward(sensory_wave);
        self.heart_resonance.synchronize(target_phase, delta_t);
        let coherence = self.heart_resonance.get_coherence(target_phase);
        self.temporal_cfc.inject(&intent_amplitudes);
        let _liquid_flow = self.temporal_cfc.flow(delta_t);

        if coherence > 0.95 {
            self.instinct_layer.myelinate(coherence);
        }

        self.metabolism.accumulate_fatigue((0.001 * stress_factor) / delta_t.max(0.001));
        self.metabolism.update(delta_t);
    }
}
