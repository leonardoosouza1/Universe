use universe::layers::BitLinear;
use universe::metabolism::Metabolism;

#[test]
fn test_metabolism_homeostasis() {
    let mut metabolism = Metabolism::new();

    // Simulate high activity
    for _ in 0..10 {
        metabolism.accumulate_fatigue(0.15);
    }

    // Should trigger sleep at 1.0
    assert!(
        metabolism.is_sleeping,
        "System should be sleeping after high fatigue"
    );
    assert!(
        !metabolism.can_process(),
        "System should block processing during sleep"
    );

    // Simulate Night Dojo (Recovery)
    for _ in 0..20 {
        metabolism.update(1.0);
    }

    // Should wake up after recovery
    assert!(
        !metabolism.is_sleeping,
        "System should wake up after recovery"
    );
}

#[test]
fn test_bitlinear_inference() {
    let input_dim = 4;
    let output_dim = 2;
    let mut layer = BitLinear::new(input_dim, output_dim);

    // Mock weights: [1, -1, 0, 1,   -1, 1, 1, 0]
    let full_weights = vec![1.0, -1.0, 0.0, 1.0, -1.0, 1.0, 1.0, 0.0];
    layer.quantize(&full_weights);

    // Input activations (8-bit int proxy): [10, 20, 30, 40]
    let input = vec![10, 20, 30, 40];

    let output = layer.forward(&input);

    // Calculation for out[0]: (10*1) + (20*-1) + (30*0) + (40*1) = 10 - 20 + 0 + 40 = 30
    // 30 / 127.0 (norm_factor) = 0.23622047
    //
    // Finding (out of P1.8's scope, discovered while running `cargo test --workspace`
    // as the final gate of the fix): this assertion assumed `output[0]` multiplied
    // by `weight_scale` (abs_mean of the weights = 0.75) — but `BitLinear::forward`
    // (src/layers.rs:39-52) today does NOT multiply `weight_scale` in the forward
    // (the field's own comment, layers.rs:22: "Forward does NOT multiply scale
    // (weight already absorbs magnitude)"). `weight_scale` is only read/written at the
    // disk boundary (`quantize_for_save`/`load_from_quantized`, see layers.rs:83-98),
    // not in the forward itself. The old assertion (`22.5/127.0`) was never updated
    // when that behavior changed (git log shows the last real commit in `layers.rs` as
    // "fix(dna): cortex quantize scale 63.5 (was 127, saturated weights >1.0)",
    // prior to this session) — fixed here to reflect the current forward.
    assert!((output[0] - (30.0 / 127.0)).abs() < 1e-5);
    println!("BitLinear Output: {:?}", output);
}
