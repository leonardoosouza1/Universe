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
    layer.quantize(&full_weights);
    
    // Input activations (8-bit int proxy): [10, 20, 30, 40]
    let input = vec![10, 20, 30, 40];
    
    let output = layer.forward(&input);

    // Calculation for out[0]: (10*1) + (20*-1) + (30*0) + (40*1) = 10 - 20 + 0 + 40 = 30
    // 30 / 127.0 (norm_factor) = 0.23622047
    //
    // Achado (fora do escopo de P1.8, descoberto ao rodar `cargo test --workspace`
    // como gate final do conserto): esta asserção assumia `output[0]` multiplicado
    // por `weight_scale` (abs_mean dos pesos = 0.75) — mas `BitLinear::forward`
    // (src/layers.rs:39-52) hoje NÃO multiplica `weight_scale` no forward (comentário
    // do próprio campo, layers.rs:22: "Forward NÃO multiplica scale (peso já absorve
    // magnitude)"). `weight_scale` só é lido/gravado no boundary do disco
    // (`quantize_for_save`/`load_from_quantized`, ver layers.rs:83-98), não no forward
    // em si. A asserção antiga (`22.5/127.0`) nunca foi atualizada quando esse
    // comportamento mudou (git log mostra o último commit real em `layers.rs` como
    // "fix(dna): cortex quantize scale 63.5 (was 127, saturated weights >1.0)",
    // anterior a esta sessão) — corrigida aqui para refletir o forward atual.
    assert!((output[0] - (30.0 / 127.0)).abs() < 1e-5);
    println!("BitLinear Output: {:?}", output);
}
