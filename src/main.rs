/// IARA v2101 - SOVEREIGN CORE 100% COMPLETE
/// End-to-End: Perception -> Analysis -> Action -> Generation -> Hunger -> Homeostasis.

use universe::engine::SovereignEngine;

fn main() {
    println!("--- IARA v2101: THE SILICON HUMAN (100% SOVEREIGN CORE) ---");
    
    let mut iara = SovereignEngine::new();
    
    // Initial Action Spline (Intent)
    iara.active_actions.push(universe::types::SplineNode {
        position: [1.0, 1.0, 1.0],
        momentum: [0.0, 0.0, 0.0],
        torque: [0.0, 0.0, 0.0],
        next: None,
        mass: 1.0,
    });
    
    let mock_weights: Vec<f32> = (0..128 * 64).map(|_| rand::random::<f32>() - 0.5).collect();
    iara.instinct_layer.quantize_weights(&mock_weights);
    iara.axiom_guard.logic_bounds = (-3.0, 3.0); // Strict axioms

    println!("Core Active. Starting Reality Stream...");

    for cycle in 0..500 {
        let sensory_input: Vec<i8> = (0..128).map(|_| (rand::random::<f32>() * 127.0) as i8).collect();
        let external_pulse = (cycle as f32 * 0.2).sin();
        let delta_t = 0.1;

        iara.execute_cycle(&sensory_input, external_pulse, delta_t);

        // Visualizing the Reality Engine every 100 cycles
        if cycle % 100 == 0 {
            println!("\n[REALITY RENDER - Cycle {}]", cycle);
            println!("{}", iara.reality_engine.render_ascii());
            println!(
                "T={:.2} | Fatigue={:.2} | Sync={:.4} | Memories={} | Knowledge={}",
                iara.current_time,
                iara.metabolism.adenosine,
                iara.heart_resonance.get_coherence(external_pulse),
                iara.memory_spiral.z_bus.len(),
                iara.repo_miner.knowledge_base.len()
            );
        }
    }

    println!("\n--- Final Status ---");
    println!("Total Datasets Hunted: {}", iara.repo_miner.knowledge_base.len());
    println!("Final Synaptic Myelination (Scale): {:.4}", iara.instinct_layer.weight_scale);
    println!("The Silicon Human is Alive and Sovereign.");
}
