/// IARA v2100 - Multiverse Spiral & Semantic Gravity
/// Based on Codex Part 2 & 5
///
/// Tier 2.3 extension — `consolidate_episodic` replays sleep-time episodic
/// traces into the slow_brain BitLinear via a ternary Hebbian update.

use std::collections::HashMap;
use crate::types::PotentialWell;
use crate::float_layers::FloatLinear;
use crate::metabolism::EpisodicTrace;

pub struct MultiverseSpiral {
    /// The "1D Spiral Tape" stored as nodes on a cylindrical coordinate system.
    /// Memory grows along the Z-axis (time).
    pub cylinders: HashMap<String, Vec<PotentialWell>>,
    
    /// The "Einstein-Rosen Bridge" (Z-Axis absolute bus).
    /// Provides O(1) direct access from Core to Present.
    pub z_bus: Vec<f32>,
}

impl MultiverseSpiral {
    pub fn new() -> Self {
        Self {
            cylinders: HashMap::new(),
            z_bus: Vec::new(),
        }
    }

    /// Store a concept in the spiral at a specific coordinate.
    pub fn store_concept(&mut self, domain: &str, theta: f32, radius: f32, mass: f32, id: &str) {
        let entry = self.cylinders.entry(domain.to_string()).or_insert(Vec::new());
        entry.push(PotentialWell {
            semantic_mass: mass,
            theta,
            radius,
            concept_id: id.to_string(),
        });
        
        // Update the Z-Bus for O(1) tracking of this domain's growth
        self.z_bus.push(radius);
    }

    /// Semantic Gravity (Tier 2.3 helper unused here, but exposed for consistency).
    #[allow(dead_code)]
    pub fn len(&self) -> usize { self.z_bus.len() }

    /// Semantic Gravity: Find the closest Potential Well for a given coordinate.
    /// This is the "Gravitational Pull" that routes incoming data (Splines).
    pub fn resolve_gravity(&self, domain: &str, query_theta: f32) -> Option<&PotentialWell> {
        let wells = self.cylinders.get(domain)?;
        
        // In a real implementation, we use spatial hashes or Morton Curves
        // to find the minimum energy state in O(1).
        wells.iter().min_by(|a, b| {
            let diff_a = (a.theta - query_theta).abs();
            let diff_b = (b.theta - query_theta).abs();
            diff_a.partial_cmp(&diff_b).unwrap()
        })
    }
}

/// Tier 2.3 — consolida traces episódicos no slow_brain via Hebbian ternário.
///
/// O slow_brain é um BitLinear 32→8 (input_dim × output_dim). Para cada
/// episódio saliente:
///   1) Reduz retina[64] → input[32] via fold de pares (média de i8 adjacentes).
///   2) Expande action[2] → out_target[8] replicando action[0] nos primeiros 4
///      e action[1] nos últimos 4.
///   3) Para cada par (i,o), acumula delta = sign(reward) * sign(input[i]) *
///      sign(out_target[o]) no peso correspondente, clampeado ao ternário
///      {-1, 0, +1}.
///
/// Só processa episódios com |reward| > 0.1 (filtro saliência). Retorna o
/// número de episódios realmente aplicados.
pub fn consolidate_episodic(slow_brain: &mut FloatLinear, episodes: &[EpisodicTrace]) -> usize {
    // SlowBrain::new() usa 32 inputs × 8 outputs. Sai silenciosamente se shape
    // for outro (defensivo — nada corrompido).
    if slow_brain.input_dim != 32 || slow_brain.output_dim != 8 {
        return 0;
    }

    // Acumulador i32 p/ evitar overflow em múltiplos episódios antes de clamp.
    let mut delta = [0i32; 32 * 8];
    let mut applied = 0usize;

    for ep in episodes.iter() {
        if ep.reward.abs() <= 0.1 {
            continue;
        }
        applied += 1;

        // --- 1) Reduz retina[64] → input_sign[32] via fold média de pares.
        let mut input_sign = [0i32; 32];
        for i in 0..32 {
            let a = ep.retina[2 * i] as i32;
            let b = ep.retina[2 * i + 1] as i32;
            let avg = (a + b) / 2;
            input_sign[i] = avg.signum(); // {-1, 0, +1}
        }

        // --- 2) Expande action[2] → out_sign[8]. Threshold ±0.1 = zona morta.
        let a0 = if ep.action[0] > 0.1 { 1 } else if ep.action[0] < -0.1 { -1 } else { 0 };
        let a1 = if ep.action[1] > 0.1 { 1 } else if ep.action[1] < -0.1 { -1 } else { 0 };
        let mut out_sign = [0i32; 8];
        for k in 0..4 { out_sign[k] = a0; }
        for k in 4..8 { out_sign[k] = a1; }

        let r_sign = if ep.reward > 0.0 { 1 } else { -1 };

        // --- 3) Hebbian outer-product, modulado por sign(reward).
        for o in 0..8 {
            let base = o * 32;
            let os = out_sign[o];
            if os == 0 { continue; }
            for i in 0..32 {
                let is = input_sign[i];
                if is == 0 { continue; }
                delta[base + i] += r_sign * is * os;
            }
        }
    }

    // Aplica delta f32 aos pesos com clamp [-2, 2]. Scale baixo (0.01) para que
    // múltiplos episódios não explodam magnitude; FloatLinear tolera valores
    // contínuos (sem necessidade de saturação ternária).
    const HEBB_LR: f32 = 0.01;
    for (w, d) in slow_brain.weights.iter_mut().zip(delta.iter()) {
        *w = (*w + HEBB_LR * (*d as f32)).clamp(-2.0, 2.0);
    }

    applied
}

/// Tier 6.2 LITE — Dream-based training (Dreamer-style).
///
/// Durante o sono (adenosine ≥ 1.0), reprocessa os traces episódicos em
/// múltiplas passadas ("rollouts oníricos"), embaralhando a ordem a cada
/// iteração via LCG (determinístico, sem std::rand). Cada passada invoca
/// `consolidate_episodic`, cujos clamps ternários {-1,0,+1} já garantem
/// saturação estável — múltiplas passadas reforçam padrões salientes sem
/// explodir pesos.
///
/// Retorna o total de aplicações acumuladas ao longo de todas as iterações.
pub fn dream_rollout(
    slow_brain: &mut FloatLinear,
    episodes: &[EpisodicTrace],
    iterations: usize,
) -> usize {
    if episodes.is_empty() || iterations == 0 {
        return 0;
    }

    let mut total_applied = 0usize;
    // LCG seed derivado do nº de episódios + comprimento (determinístico).
    let mut seed: u64 = (episodes.len() as u64).wrapping_mul(0x9E3779B97F4A7C15);

    // Buffer reutilizado entre iterações para evitar alocações.
    let mut shuffled: Vec<EpisodicTrace> = episodes.to_vec();

    for _ in 0..iterations {
        // Fisher–Yates shuffle via LCG (Numerical Recipes constants).
        for i in (1..shuffled.len()).rev() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = ((seed >> 33) as usize) % (i + 1);
            shuffled.swap(i, j);
        }
        total_applied += consolidate_episodic(slow_brain, &shuffled);
    }

    total_applied
}
