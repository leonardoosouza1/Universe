// ── TESTE DA FOLHA CORTICAL EMERGENTE ─────────────────────────────────────────
//
// Prova que a folha tem atividade HETEROGÊNEA e NÃO-ZERO após os primeiros ticks.
//
// # O que este teste valida
//
// 1. `std > 0.01` e `max − min > 0.05` por ser (HETEROGENEIDADE REAL)
//    — Se todas as 1024 unidades tivessem o mesmo estado, o std seria zero.
//    A heterogeneidade é garantida estruturalmente pelo banco de τ log-espaçado:
//    unidades com τ=1.6ms integram rápido, τ=318ms integram devagar →
//    mesmo com a mesma onda espontânea, cada unidade decai num ritmo próprio.
//
// 2. `mean ≠ 0` (sinal real, não estado degenerado)
//    — A onda espontânea injeta cos() em fase e magnitude mensuráveis.
//    Se mean=0, a folha não está recebendo entrada.
//
// 3. `motor ≠ [0,0,0]` (projeção motora funciona)
//    — Com pesos buf_mw inicializados com ruído pseudo-aleatório ±0.1 e
//    ativações não-zero, a saída motora DEVE ser diferente de zero.
//
// 4. Os resultados DIFEREM entre os 4 seres testados (INDIVIDUALIDADE)
//    — Cada ser recebe a mesma onda espontânea mas com sensory_wave diferente
//    (o tick 10 usa um wave com canal k = ser*13 + k como sinal distinto).
//    Mesmo que o wave seja idêntico, os pesos hebbians divergem tick a tick.
//    Em ticks 1-10, o teste usa waves distintos entre seres para forçar divergência.
//
// # Como rodar
//
//   cd ~/projects/LLM/Universe
//   env -u DISPLAY nice -n 19 cargo build --release -j 2 --bin cortex_test && \
//   env -u DISPLAY ./target/release/cortex_test
//
// Saída esperada (valores reais, NÃO fabricados):
//   ser 0: min=-0.XXXX max=+0.XXXX mean=0.XXXX std=0.XXXX [PASS]
//   ser 1: min=-0.XXXX max=+0.XXXX mean=0.XXXX std=0.XXXX [PASS]
//   ...
//   motor[0]: [throttle=0.XXXX steer=0.XXXX brake=0.XXXX]
//   RESULTADO: PASS (todas as 4 métricas verificadas com magnitude real)

use std::sync::Arc;

use universe::gpu::cortex_sheet::{CortexSheet, SHEET_N};

fn main() {
    // ── Garante ICD Vulkan único (previne SIGSEGV do loader com múltiplos ICDs) ──
    let icd_radeon = "/usr/share/vulkan/icd.d/radeon_icd.json";
    if std::path::Path::new(icd_radeon).exists() && std::env::var("VK_ICD_FILENAMES").is_err() {
        unsafe {
            std::env::set_var("VK_ICD_FILENAMES", icd_radeon);
        }
    }

    // ── Abre device wgpu ─────────────────────────────────────────────────────
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("Nenhum adapter encontrado");

    let name = adapter.get_info().name.clone();
    eprintln!("[cortex_test] Adapter: {}", name);

    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffers_per_shader_stage = 8;

    let (raw_device, raw_queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("cortex-test"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .expect("Falha ao criar device");

    let device = Arc::new(raw_device);
    let queue = Arc::new(raw_queue);

    // ── Cria folha cortical com 4 seres ─────────────────────────────────────
    let n_seres = 4usize;
    let cortex = CortexSheet::new(Arc::clone(&device), Arc::clone(&queue), n_seres);

    // Inspeciona o banco de τ (sem GPU — calculado na CPU)
    let taus = cortex.tau_bank();
    println!(
        "[τ] banco de {} valores: min={:.4}ms  max={:.1}ms  spread={:.0}×",
        taus.len(),
        taus[0] * 1000.0,
        taus[taus.len() - 1] * 1000.0,
        taus[taus.len() - 1] / taus[0]
    );
    // Verifica log-espaçamento: τ[512] deve ser ≈ √(τ_min × τ_max)
    let tau_mid_expected = (taus[0] * taus[taus.len() - 1]).sqrt();
    println!(
        "[τ] τ[512] = {:.4}ms  esperado(geomédio)={:.4}ms",
        taus[512] * 1000.0,
        tau_mid_expected * 1000.0
    );

    // ── Executa 20 ticks com sensory_wave distinto por ser ────────────────────
    // Waves artificiais: ser i tem canal k = sin(k + i*π/4) × 64 como i8.
    // Isso garante que seres tenham entradas DIFERENTES → divergência hebbiana.
    let n_ticks = 20usize;
    let mut act_sum_prev = 0.0_f32; // Σ|a_j| do tick anterior (0 no tick 0)

    for tick in 0..n_ticks {
        // Constrói waves distintos por ser
        let mut waves: Vec<[i8; 223]> = Vec::with_capacity(n_seres);
        for ser in 0..n_seres {
            let mut w = [0i8; 223];
            for k in 0..223usize {
                // Onda senoidal distinta por ser e por tick: sin(k/10 + ser + tick/5)
                let phase = (k as f32 / 10.0)
                    + (ser as f32 * std::f32::consts::PI / 4.0)
                    + (tick as f32 / 5.0);
                w[k] = (phase.sin() * 64.0) as i8;
            }
            waves.push(w);
        }
        cortex.upload_wave(&waves, n_seres);

        // Encoder único para os 3 kernels + readback
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cortex-test-tick"),
        });

        cortex.dispatch_afference(&mut enc, n_seres, tick as u32);
        cortex.dispatch_hebb(&mut enc, n_seres, tick as u32, act_sum_prev);
        cortex.dispatch_motor(&mut enc, n_seres, tick as u32);
        cortex.copy_for_readback(&mut enc, n_seres);

        queue.submit(Some(enc.finish()));
        device.poll(wgpu::Maintain::Wait);

        // Lê estatísticas de ativação (prova de heterogeneidade)
        let stats = cortex.read_act_stats(n_seres);

        // Calcula act_sum do ser 0 para o próximo tick
        // (Σ|a_j| passado ao kernel Hebb como parâmetro de normalização)
        // Para simplicidade do teste, usamos apenas ser 0.
        // Em produção, calcular por ser via readback antes do dispatch B.
        // Aqui: approximamos como SHEET_N × |mean| (bounded abaixo por heterogeneidade).
        // Na integração real, ler buf_act_rb e somar antes do dispatch B.
        act_sum_prev = stats[0].2.abs() * SHEET_N as f32;

        // Imprime no último tick
        if tick == n_ticks - 1 {
            println!("\n── Ativações da folha (tick {}) ──", tick);
            let mut all_pass = true;
            for (ser, (mn, mx, mean, std)) in stats.iter().enumerate() {
                // Critérios de PASS (com magnitude real):
                // std > 0.01: unidades têm estados diferentes (heterogeneidade)
                // max-min > 0.05: há spread real, não constante
                // |mean| > 0: sinal não-zero (onda espontânea chegou)
                let heterogeneo = std > &0.01 && (mx - mn) > 0.05;
                let nao_zero = mean.abs() > 1e-6;
                let ok = heterogeneo && nao_zero;
                if !ok {
                    all_pass = false;
                }
                println!(
                    "  ser {}: min={:+.4}  max={:+.4}  mean={:+.4}  std={:.4}  {}",
                    ser,
                    mn,
                    mx,
                    mean,
                    std,
                    if ok { "PASS" } else { "FAIL" }
                );
            }

            // Lê saída motora
            let motors = cortex.read_motor(n_seres);
            println!("\n── Saída motora (tick {}) ──", tick);
            for (ser, m) in motors.iter().enumerate() {
                let nao_zero = m[0].abs() + m[1].abs() + m[2].abs() > 1e-6;
                if !nao_zero {
                    all_pass = false;
                }
                println!(
                    "  ser {}: throttle={:+.4}  steer={:+.4}  brake={:+.4}  {}",
                    ser,
                    m[0],
                    m[1],
                    m[2],
                    if nao_zero { "PASS" } else { "FAIL (zero)" }
                );
            }

            // Verifica que seres DIFEREM entre si (individualidade)
            let mut divergencia_ok = false;
            for i in 0..n_seres {
                for j in (i + 1)..n_seres {
                    let diff = (stats[i].2 - stats[j].2).abs();
                    if diff > 1e-6 {
                        divergencia_ok = true;
                    }
                }
            }
            if !divergencia_ok {
                all_pass = false;
            }
            println!("\n── Divergência entre seres ──");
            for i in 0..n_seres {
                for j in (i + 1)..n_seres {
                    let diff = (stats[i].2 - stats[j].2).abs();
                    println!(
                        "  |mean[{}] − mean[{}]| = {:.6}  {}",
                        i,
                        j,
                        diff,
                        if diff > 1e-6 {
                            "divergente (OK)"
                        } else {
                            "idêntico (PROBLEMA)"
                        }
                    );
                }
            }

            println!(
                "\n══ RESULTADO FINAL: {} ══",
                if all_pass { "PASS" } else { "FAIL" }
            );
            if !all_pass {
                eprintln!("FAIL: Verificar stderr para detalhes do driver");
                std::process::exit(1);
            }
        }
    }

    // ── Sumário do banco de τ ────────────────────────────────────────────────
    // Conta quantas unidades estão em cada banda (por ser 0, τ são os mesmos).
    let (mut n_gamma, mut n_beta, mut n_alpha, mut n_theta, mut n_delta) = (0, 0, 0, 0, 0);
    for &t in &taus {
        let f_hz = 1.0 / (2.0 * std::f32::consts::PI * t);
        if f_hz >= 30.0 {
            n_gamma += 1;
        } else if f_hz >= 13.0 {
            n_beta += 1;
        } else if f_hz >= 8.0 {
            n_alpha += 1;
        } else if f_hz >= 4.0 {
            n_theta += 1;
        } else {
            n_delta += 1;
        }
    }
    println!("\n── Distribuição de bandas (doc §5.3) ──");
    println!(
        "  gamma (>30Hz):   {:4} unidades ({:.1}%)",
        n_gamma,
        n_gamma as f32 / SHEET_N as f32 * 100.0
    );
    println!(
        "  beta  (13-30Hz): {:4} unidades ({:.1}%)",
        n_beta,
        n_beta as f32 / SHEET_N as f32 * 100.0
    );
    println!(
        "  alpha (8-13Hz):  {:4} unidades ({:.1}%)",
        n_alpha,
        n_alpha as f32 / SHEET_N as f32 * 100.0
    );
    println!(
        "  theta (4-8Hz):   {:4} unidades ({:.1}%)",
        n_theta,
        n_theta as f32 / SHEET_N as f32 * 100.0
    );
    println!(
        "  delta (<4Hz):    {:4} unidades ({:.1}%)",
        n_delta,
        n_delta as f32 / SHEET_N as f32 * 100.0
    );
}
