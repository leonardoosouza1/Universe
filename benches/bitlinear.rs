use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use universe::layers::BitLinear;
use universe::specialist::SpecialistBrain;

fn make_layer(input_dim: usize, output_dim: usize, seed: u64) -> BitLinear {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut layer = BitLinear::new(input_dim, output_dim);
    // BitLinear became CortexLayer and now stores f32 weights at runtime (ternary
    // survives only on disk, in the genome). The old -1..=1 i8 draw no longer typechecks.
    for w in layer.weights.iter_mut() {
        *w = rng.random_range(-1.0f32..=1.0f32);
    }
    layer.weight_scale = 0.5;
    layer
}

fn make_input(dim: usize, seed: u64) -> Vec<i8> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..dim).map(|_| rng.random_range(-127i8..=127i8)).collect()
}

fn bench_forward_128x64(c: &mut Criterion) {
    let layer = make_layer(128, 64, 0xC0FFEE);
    let input = make_input(128, 0xBEEF);
    let mut g = c.benchmark_group("bitlinear/fast_brain_128x64");
    g.throughput(Throughput::Elements(128 * 64));
    g.bench_function("forward", |b| {
        b.iter(|| {
            let out = layer.forward(black_box(&input));
            black_box(out);
        })
    });
    g.finish();
}

fn bench_forward_32x8(c: &mut Criterion) {
    let layer = make_layer(32, 8, 0xC0FFEE);
    let input = make_input(32, 0xBEEF);
    let mut g = c.benchmark_group("bitlinear/slow_brain_32x8");
    g.throughput(Throughput::Elements(32 * 8));
    g.bench_function("forward", |b| {
        b.iter(|| {
            let out = layer.forward(black_box(&input));
            black_box(out);
        })
    });
    g.finish();
}

fn bench_swarm_tick(c: &mut Criterion) {
    let layer = make_layer(128, 64, 0xC0FFEE);
    let inputs: Vec<Vec<i8>> = (0..80)
        .map(|i| make_input(128, 0xBEEF + i as u64))
        .collect();
    let mut g = c.benchmark_group("bitlinear/swarm_80cars_tick");
    g.throughput(Throughput::Elements(80 * 128 * 64));
    g.bench_function("sequential", |b| {
        b.iter(|| {
            for input in &inputs {
                let out = layer.forward(black_box(input));
                black_box(out);
            }
        })
    });
    g.finish();
}

fn bench_specialist(c: &mut Criterion) {
    let mut brain = SpecialistBrain::new();
    // SpecialistBrain moved from BitLinear to FloatLinear layers and this benchmark
    // was never updated, so it stopped compiling (f32 weights, and no `weight_scale`
    // field). `cargo build` does not build benches, which is why it rotted unnoticed;
    // `cargo clippy --all-targets` in CI is what surfaced it.
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    for b in [&mut brain.steer, &mut brain.throttle, &mut brain.brake] {
        for w in b.weights.iter_mut() {
            *w = rng.random_range(-0.5f32..=0.5f32);
        }
    }
    let input = make_input(128, 0xBEEF);
    let mut g = c.benchmark_group("bitlinear/specialist_3x32");
    g.throughput(Throughput::Elements(3 * 32));
    g.bench_function("forward", |b| {
        b.iter(|| {
            let out = brain.forward_specialist(black_box(&input));
            black_box(out);
        })
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_forward_128x64,
    bench_forward_32x8,
    bench_swarm_tick,
    bench_specialist
);
criterion_main!(benches);
