use universe::metabolism::Metabolism;
use universe::layers::BitLinear;

#[test]
fn test_metabolism_homeostasis() {
    let mut metabolism = Metabolism::new();
    
    // Simulate high activity
    for _ in 0..10 {
        metabolism.accumulate_fatigue(0.15);
    }
    
    // Should trigger sleep at 1.0
    assert!(metabolism.is_sleeping, "System should be sleeping after high fatigue");
    assert!(!metabolism.can_process(), "System should block processing during sleep");

    // Simulate Night Dojo (Recovery)
    for _ in 0..20 {
        metabolism.update(1.0);
    }
    
    // Should wake up after recovery
    assert!(!metabolism.is_sleeping, "System should wake up after recovery");
}

#[test]
fn test_bitlinear_inference() {
    let input_dim = 4;
    let output_dim = 2;
    let mut layer = BitLinear::new(input_dim, output_dim);
    
    // Mock weights: [1, -1, 0, 1,   -1, 1, 1, 0]
    let full_weights = vec![1.0, -1.0, 0.0, 1.0,  -1.0, 1.0, 1.0, 0.0];
    layer.quantize_weights(&full_weights);
    
    // Input activations (8-bit int proxy): [10, 20, 30, 40]
    let input = vec![10, 20, 30, 40];
    
    let output = layer.forward(&input);
    
    // Calculation for out[0]: (10*1) + (20*-1) + (30*0) + (40*1) = 10 - 20 + 0 + 40 = 30
    // Result scaled by weight_scale (abs_mean of full_weights = 0.75)
    // 30 * 0.75 = 22.5
    
    assert!((output[0] - 22.5).abs() < 1e-5);
    println!("BitLinear Output: {:?}", output);
}
