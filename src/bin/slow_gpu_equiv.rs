//! Equivalência CPU × GPU do CÉREBRO LENTO.
//!
//! O agente que escreveu `src/gpu/slow_brain.wgsl` não pôde rodar na GPU, então o kernel
//! está COMPILADO e NÃO VERIFICADO. Neste projeto já houve cinco relatórios de prova
//! fabricada — inclusive uma tabela de validação de uma execução que deu segfault e um
//! teste que comparava zero com zero. Então a regra aqui é: o número tem que sair de uma
//! execução real, e o teste tem que ser capaz de FALHAR.
//!
//! Duas armadilhas específicas que este arquivo evita de propósito:
//!
//!  1. **Comparar zero com zero.** Se os pesos nascerem zerados, os dois lados dão 0 e a
//!     equivalência é vácuo. Aqui os pesos vêm de `rng::Lcg::symmetric()` e o programa
//!     ABORTA se a saída da CPU não tiver variação — sem sinal, não há o que validar.
//!  2. **Testar só um tick.** O lento roda 1 vez a cada 8 ticks e tem estado recorrente
//!     (CfC + fadiga + contador). Um único tick não exercita nem o subsample nem a
//!     memória. Rodamos centenas de ticks com entrada variando.
//!
//! Um único `wgpu::Device` no processo: abrir dois no mesmo dispositivo já travou aqui.

use std::sync::Arc;
use universe::engine::SlowBrain;
use universe::gpu::slow_brain::SlowBrainGpu;
use universe::rng::Lcg;

const N_BEINGS: usize = 8;
const TICKS: usize = 400;
const DT: f32 = 1.0 / 64.0;

fn main() {
    // ── Dispositivo compute-only, sem janela ────────────────────────────────
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        ..Default::default()
    });
    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    })) {
        Some(a) => a,
        None => {
            eprintln!("sem adaptador GPU");
            std::process::exit(2);
        }
    };
    println!("GPU: {}", adapter.get_info().name);

    let (device, queue) = match pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("slow-equiv"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    )) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("request_device falhou: {e:?}");
            std::process::exit(2);
        }
    };
    let device: wgpu::Device = device;
    let queue: wgpu::Queue = queue;
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    // ── Pesos: um conjunto DIFERENTE por ser ────────────────────────────────
    // Seres com pesos distintos pegam erro de indexação por ser (`base = i * stride`),
    // que é exatamente o bug que já apareceu aqui: `fat_base = being * 96` num buffer
    // de 192 slots fazia o ser N ler os slots do ser N−1.
    let mut cpus: Vec<SlowBrain> = Vec::with_capacity(N_BEINGS);
    let mut all_weights: Vec<f32> = Vec::new();
    for b in 0..N_BEINGS {
        let mut brain = SlowBrain::new();
        let mut rng = Lcg::new(0x5107_0000 + b as u64);
        let n = brain.all_weights().len();
        let w = rng.weights(n, 0.35);
        brain.set_all_weights(&w);
        all_weights.extend_from_slice(&w);
        cpus.push(brain);
    }
    println!(
        "{} seres · {} pesos/ser · {} ticks",
        N_BEINGS,
        all_weights.len() / N_BEINGS,
        TICKS
    );

    let gpu = SlowBrainGpu::new_shared(device.clone(), queue.clone(), N_BEINGS);
    gpu.upload_weights(&all_weights, N_BEINGS);

    // ── Entrada: varia no tempo E entre seres ───────────────────────────────
    let make_input = |t: usize, b: usize| -> [i8; 48] {
        let mut inp = [0i8; 48];
        for (i, slot) in inp.iter_mut().enumerate() {
            let period = 23.0 + (i as f32) * 5.0 + (b as f32) * 3.0;
            *slot = (((t as f32) / period * std::f32::consts::TAU).sin() * 110.0) as i8;
        }
        inp
    };

    let mut worst_abs = 0.0f32;
    let mut worst_tick = 0usize;
    let mut sum_abs = 0.0f64;
    let mut n_comparisons = 0usize;
    let mut cpu_samples: Vec<[f32; 3]> = Vec::new();

    for t in 0..TICKS {
        let inputs: Vec<[i8; 48]> = (0..N_BEINGS).map(|b| make_input(t, b)).collect();

        for (b, brain) in cpus.iter_mut().enumerate() {
            brain.tick(&inputs[b], DT);
        }
        let gpu_out = gpu.forward(&inputs, N_BEINGS, DT);

        if gpu_out.len() != N_BEINGS {
            eprintln!(
                "GPU devolveu {} saídas, esperava {}",
                gpu_out.len(),
                N_BEINGS
            );
            std::process::exit(1);
        }

        for b in 0..N_BEINGS {
            for k in 0..3 {
                let d = (cpus[b].output[k] - gpu_out[b][k]).abs();
                sum_abs += d as f64;
                n_comparisons += 1;
                if d > worst_abs {
                    worst_abs = d;
                    worst_tick = t;
                }
            }
        }
        cpu_samples.push(cpus[0].output);

        if t % 80 == 0 {
            println!(
                "  t={:3}  cpu[0] = {:+.6} {:+.6} {:+.6}   gpu[0] = {:+.6} {:+.6} {:+.6}",
                t,
                cpus[0].output[0],
                cpus[0].output[1],
                cpus[0].output[2],
                gpu_out[0][0],
                gpu_out[0][1],
                gpu_out[0][2],
            );
        }
    }

    // ── Guarda contra validação em vácuo ────────────────────────────────────
    // Se a CPU não produziu variação, comparar com a GPU não prova nada.
    let mut max_amplitude = 0.0f32;
    for k in 0..3 {
        let mx = cpu_samples.iter().map(|o| o[k]).fold(f32::MIN, f32::max);
        let mn = cpu_samples.iter().map(|o| o[k]).fold(f32::MAX, f32::min);
        max_amplitude = max_amplitude.max(mx - mn);
    }
    println!("\namplitude do sinal na CPU: {max_amplitude:.6}");
    if max_amplitude < 1e-3 {
        eprintln!("ABORTA: a CPU não produziu sinal — a comparação seria vácuo (0 vs 0).");
        std::process::exit(1);
    }

    let mean = sum_abs / n_comparisons as f64;
    println!("comparações: {n_comparisons}");
    println!("erro absoluto médio: {mean:.3e}");
    println!("pior erro absoluto:  {worst_abs:.3e}  (tick {worst_tick})");

    // Tolerância: a diferença esperada é só de ordem de operações em f32 ao longo de
    // somas de 64 e 16 termos, com tanh/exp nos dois lados. 1e-4 é folgado para isso e
    // apertado o bastante para pegar erro de índice, de layout ou de equação.
    const TOL: f32 = 1e-4;
    if worst_abs <= TOL {
        println!("\nEQUIVALENTE — pior erro {worst_abs:.3e} ≤ {TOL:.0e}");
    } else {
        println!("\nDIVERGENTE — pior erro {worst_abs:.3e} > {TOL:.0e}");
        std::process::exit(1);
    }
}
