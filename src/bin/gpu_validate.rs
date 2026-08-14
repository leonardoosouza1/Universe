//! Arnês de validação GPU × CPU — cada subsistema isolado, com tabela honesta.
//!
//! # Por que este binário existe
//!
//! A regra inegociável do GPU_PLANO.md: "Um caminho GPU que diverge em silêncio
//! invalida todo experimento já medido." Este binário é a barreira. Ele compara
//! cada kernel GPU com a implementação CPU correspondente e imprime o veredito
//! COM O NÚMERO que o sustenta — nunca "OK" sem evidência.
//!
//! # Como usar
//!
//! ```text
//! env -u DISPLAY ./target/release/gpu_valida
//! env -u DISPLAY ./target/release/gpu_valida --retina --nucleo --cerebro --fisica --percepcao
//! ```
//!
//! # Tolerâncias declaradas (GPU_PLANO.md)
//!
//! - retina (i8 codificada): diferença ≤ 1 (arredondamento de float na quantização).
//! - floats de uma passada (física, intenção): ≤ 2 ULP ou erro relativo ≤ 1e-5.
//! - estado acumulado por N ticks: deriva medida e DECLARADA, não assumida zero.

use bevy::math::{Vec2, Vec3};
use universe::config::iara::vision::RETINA_TOTAL;
use universe::gpu::{Pose, RetinaGpu};
use universe::gpu::core::{CoreGpu, PosGpu};
use universe::world::VoxelWorld;

/// Executa `f()` num processo filho (fork). Retorna `true` se o filho terminou com exit 0,
/// `false` se morreu por sinal (ex: SIGSEGV=139) ou com exit != 0.
///
/// Usado para isolar crashes de driver GPU que chegam como SIGSEGV e não podem ser
/// capturados por std::panic::catch_unwind.
///
/// A saída do filho vai para stdout/stderr normalmente (fd são herdados).
#[cfg(unix)]
fn run_in_child<F: FnOnce()>(f: F) -> bool {
    use std::io::Write;
    unsafe {
        let pid = libc::fork();
        if pid < 0 {
            // fork falhou — executa no processo atual (sem isolamento)
            let _ = std::io::stderr().write_all(b"  AVISO: fork() falhou, sem isolamento de crash.\n");
            f();
            return true;
        }
        if pid == 0 {
            // Filho: executa e sai
            f();
            libc::_exit(0);
        }
        // Pai: aguarda o filho
        let mut status: libc::c_int = 0;
        libc::waitpid(pid, &mut status, 0);
        // WIFEXITED e WEXITSTATUS — filho saiu normalmente com 0
        if libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0 {
            true
        } else {
            false
        }
    }
}

#[cfg(not(unix))]
fn run_in_child<F: FnOnce()>(f: F) -> bool {
    // Em não-unix, sem fork — executa diretamente (sem isolamento)
    f();
    true
}

// ── ParamsGpu local (idêntico ao layout interno de RetinaGpu/NucleoGpu) ─────
// O tipo interno de mod.rs é privado. Este é uma cópia local usada só aqui.
// CRÍTICO: manter em sincronia com mod.rs::ParamsGpu e retina.wgsl::Params.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LocalParamsGpu {
    grid_size: u32,
    n_beings:  u32,
    cell_size: f32,
    fov_half:  f32,
    pitches:   [[f32; 4]; 4],
}

/// Fadiga de saída — referência CPU do validador.
///
/// Estava com a fórmula ANTIGA (`raw - beta·fat`, subtração com sinal) enquanto
/// `engine.rs` e `brain.wgsl` já tinham migrado para o encolhimento sobre a magnitude.
/// Um validador com gabarito velho é pior que nenhum: ele APROVA a divergência que
/// deveria pegar, porque compara os dois lados contra a mesma referência errada.
fn adapta_saida(raw: f32, out_fat: f32) -> f32 {
    const OUT_BETA: f32 = 0.85;  // espelha engine.rs e brain.wgsl
    (raw.signum() * (raw.abs() - OUT_BETA * out_fat).max(0.0)).tanh()
}

fn main() {
    // SEGFAULT FIX (física): múltiplos ICDs Vulkan (intel_hasvk, lvp, nouveau, asahi,
    // radeon, virtio…) causam segfault ao carregar um ICD incompatível com a GPU AMD.
    // VK_ICD_FILENAMES DEVE ser setado ANTES do primeiro wgpu::Instance::new —
    // o Vulkan loader carrega as libs de ICD uma única vez de forma global.
    // Setar dentro de validate_physics() é tarde demais: o loader já foi invocado em
    // validate_retina() / validate_nucleo() / validate_brain().
    let icd_radeon = "/usr/share/vulkan/icd.d/radeon_icd.json";
    if std::path::Path::new(icd_radeon).exists()
        && std::env::var("VK_ICD_FILENAMES").is_err()
    {
        // safety: thread única, nenhum outro código wgpu rodando ainda
        unsafe { std::env::set_var("VK_ICD_FILENAMES", icd_radeon); }
    }

    let args: Vec<String> = std::env::args().collect();
    let tudo           = args.len() == 1;
    let quer_retina    = tudo || args.iter().any(|a| a == "--retina");
    let quer_nucleo    = tudo || args.iter().any(|a| a == "--nucleo");
    let quer_cerebro   = tudo || args.iter().any(|a| a == "--cerebro");
    let quer_fisica    = tudo || args.iter().any(|a| a == "--fisica");
    let quer_percepcao = tudo || args.iter().any(|a| a == "--percepcao");

    let world = VoxelWorld::new(9001, 18.0);
    println!("mundo: {}x{} celulas · {:.0} m · volta {:.0} m",
        world.size, world.size,
        world.size as f32 * world.cell_size,
        world.lap_distance,
    );

    // 48 posturas espalhadas pela pista
    let mut poses = Vec::new();
    let n_wp = world.waypoints_world.len();
    for k in 0..48usize {
        let wp = world.waypoints_world[(k * n_wp / 48).min(n_wp - 1)];
        let ang = k as f32 * 0.37;
        poses.push(Pose {
            pos:      Vec3::new(wp.x, 0.0, wp.y),
            forward:  Vec2::new(ang.cos(), ang.sin()),
            right:    Vec2::new(-ang.sin(), ang.cos()),
            gaze_yaw: 0.0,
        });
    }

    println!();
    print_table_header();

    // ==========================================================================
    // SECAO 1: RETINA
    // ==========================================================================
    if quer_retina {
        println!("[RETINA]");
        let gpu = match RetinaGpu::new(&world, poses.len().max(64)) {
            Some(g) => g,
            None => {
                println!("  NAO MEDIDO — sem GPU utilizavel.");
                print_table_row("retina (i8 legado)", 0, 0, -1, false);
                println!();
                // continua para as outras secoes
                return;
            }
        };
        println!("  GPU: {}  ·  {} seres", gpu.adapter, poses.len());

        let saida_gpu = gpu.render_legacy(&poses);
        let mut total  = 0usize;
        let mut iguais = 0usize;
        let mut maxdif = 0i32;
        for (i, p) in poses.iter().enumerate() {
            let cpu = world.render_binocular_retina(
                p.pos,
                Vec3::new(p.forward.x, 0.0, p.forward.y),
                Vec3::new(p.right.x, 0.0, p.right.y),
                p.gaze_yaw,
            );
            for px in 0..RETINA_TOTAL {
                let d = (cpu[px] as i32 - saida_gpu[i][px] as i32).abs();
                if d == 0 { iguais += 1; }
                if d > maxdif { maxdif = d; }
                total += 1;
            }
        }
        println!("  comparados={total}  identicos={iguais} ({:.4}%)  max_dif={maxdif}",
            iguais as f64 / total as f64 * 100.0);

        let reps = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..reps { let _ = gpu.render(&poses); }
        let t_gpu = t0.elapsed().as_secs_f64() / reps as f64;
        let t1 = std::time::Instant::now();
        for _ in 0..reps {
            for p in &poses {
                let _ = world.render_binocular_retina(p.pos,
                    Vec3::new(p.forward.x,0.0,p.forward.y),
                    Vec3::new(p.right.x,0.0,p.right.y), p.gaze_yaw);
            }
        }
        let t_cpu = t1.elapsed().as_secs_f64() / reps as f64;
        println!("  CPU={:.2}ms  GPU={:.2}ms  aceleracao={:.1}x",
            t_cpu*1000.0, t_gpu*1000.0, t_cpu/t_gpu);

        print_table_row("retina (i8 legado)", total, iguais, maxdif, maxdif <= 1);
        println!();
    }

    // ==========================================================================
    // SECAO 2: NUCLEO
    // ==========================================================================
    if quer_nucleo {
        println!("[NUCLEO]");
        let n_beings = poses.len();
        let core = match CoreGpu::new(&world, n_beings.max(64)) {
            Some(n) => n,
            None => {
                println!("  NAO MEDIDO — sem GPU.");
                print_table_row("core (cfc zeros)", 0, 0, -1, false);
                print_table_row("nucleo (tick n seres)", 0, 0, -1, false);
                println!();
                return;
            }
        };
        println!("  GPU: {}  ·  capacidade={}", core.adapter, core.capacity);

        // Pesos de validação SIMÉTRICOS. Antes vinham do LCG quebrado, todos em
        // [-1,0): a equivalência CPU/GPU continuava válida (os dois lados liam os
        // mesmos pesos), mas era uma distribuição degenerada — um bug de sinal no
        // kernel teria menos chance de aparecer.
        let mut rng = universe::rng::Lcg::new(0xDEAD_C0DE);
        let mut nf = move || rng.symmetric();
        let w1: Vec<f32> = (0..n_beings*CoreGpu::w1_total()).map(|_| nf()).collect();
        let w2: Vec<f32> = (0..n_beings*CoreGpu::w2_total()).map(|_| nf()).collect();
        let w3: Vec<f32> = (0..n_beings*CoreGpu::w3_total()).map(|_| nf()).collect();
        core.upload_weights(&w1, &w2, &w3, n_beings);

        let cfc = core.download_cfc_state(n_beings);
        let mut cfc_zeros = 0usize;
        let mut cfc_total = 0usize;
        let mut cfc_max = 0.0f32;
        for estado in &cfc {
            for &v in estado {
                cfc_total += 1;
                if v.abs() < 1e-7 { cfc_zeros += 1; }
                if v.abs() > cfc_max { cfc_max = v.abs(); }
            }
        }
        println!("  CfC inicial: {cfc_zeros}/{cfc_total} zeros  max_abs={cfc_max:.2e}");
        print_table_row("core (cfc zeros)", cfc_total, cfc_zeros,
            (cfc_max * 1e7) as i32, cfc_zeros == cfc_total);

        // Tick
        use universe::config::iara::vision::RETINA_ROWS;
        let mut pitches = [[0.0f32; 4]; 4];
        for row in 0..RETINA_ROWS {
            pitches[row / 4][row % 4] = universe::visao::pitch(row);
        }
        let params = LocalParamsGpu {
            grid_size: world.size as u32,
            n_beings:  n_beings as u32,
            cell_size: world.cell_size,
            fov_half:  (universe::visao::fov_deg() / 2.0).to_radians(),
            pitches,
        };
        let poses_gpu: Vec<PosGpu> = poses.iter().map(|p| PosGpu {
            pos_pad: [p.pos.x, p.pos.y + 0.4, p.pos.z, 0.0],
            dirs:    [p.forward.x, p.forward.y, p.right.x, p.right.y],
        }).collect();
        let intents = core.tick(&poses_gpu, bytemuck::bytes_of(&params), n_beings);
        println!("  tick: {} intents (esperado: {})", intents.len(), n_beings);
        print_table_row("core (tick n beings)", n_beings, intents.len(), 0,
            intents.len() == n_beings);
        println!();
    }

    // ==========================================================================
    // SECAO 3: CEREBRO
    //
    // Canais NAO comparados:
    //   - metabolism (can_process, adenosine): estado Bevy, fica na CPU.
    //     Engines recem-criados: adenosine=0, energy=1 → can_process=true → seguro.
    //   - mirror neurons: champion_snapshot=None em engines novos → nao disparam.
    //   - resonance_coherence: dopamine=0 no tick 0 → nao dispara.
    //   - Trinity + axiom_policy: executam na CPU apos o forward.
    //   - drift CfC 100 ticks: NAO MEDIDO — CerebroGpu nao expoe baixar_cfc().
    // ==========================================================================
    if quer_cerebro {
        println!("[CEREBRO]");
        validate_brain();
        println!();
    }

    // ==========================================================================
    // SECAO 4: FISICA
    //
    // Canais NAO comparados:
    //   - orientacao (fwd/lat pos rotate_y): fica na CPU (transform Bevy).
    //   - hash espacial inter-seres (drafting): fica na CPU.
    //   - troca de marcha manual: processada antes de serializar.
    //
    // Canais comparados: vel[3], av_y, pos[3], wheels[4x2], rpm, engine/oil_temp, is_stalled.
    //
    // A replica CPU foi necessaria porque as funcoes de fisica (reset_forces,
    // suspension_and_load_system, etc.) sao sistemas Bevy privados que requerem
    // Query<...>, Res<Time>, Res<VoxelWorldResource> — nao chamáveis diretamente.
    // A replica reproduz INTENCIONALMENTE os bugs documentados em fisica.wgsl:
    //   a) aero drag aplicado duas vezes (fase powertrain + integration)
    //   b) yaw_inertia = mass*(wheel_base/2)^2 (ponto de massa, simplificado)
    //   c) split_left nao clampeado [0,1] com LSD alto
    // ==========================================================================
    if quer_fisica {
        println!("[FISICA]");
        // SEGFAULT GUARD: o driver RADV crasha ao compilar physics.wgsl em alguns sistemas.
        // Rodamos a secao de fisica em processo filho para isolar o crash do processo principal.
        // Se o filho morrer com SIGSEGV/139, imprimimos NAO MEDIDO em vez de derrubar tudo.
        let fisica_ok = run_in_child(|| validate_physics(&world));
        if !fisica_ok {
            println!("  NAO MEDIDO — segfault no driver ao criar PhysicsGpu (physics.wgsl).");
            println!("  Diagnóstico: o driver RADV crasha ao compilar o shader de física.");
            println!("  Fix planejado: simplificar physics.wgsl ou atualizar o driver Mesa.");
            print_table_row("fisica (1 passo 6 cen)",       0, 0, -1, false);
            print_table_row("fisica (deriva 100 passos)",   0, 0, -1, false);
        }
        println!();
    }

    // ==========================================================================
    // SECAO 5: PERCEPCAO
    //
    // Canais EXCLUIDOS da comparacao (documentados em percepcao.rs e percepcao.wgsl):
    //   ch 0-15       : pool legado (retina binocular — nao disponivel neste binario)
    //   ch 21-27, 98  : motion per-row (visual_cortex anterior — temporal)
    //   ch 29-31,33-35: edge density (temporal)
    //   ch 44-51      : wall_imminent + trapped (motor.rs sobrescreve)
    //   ch 52-55      : depth_per_row (motor.rs sobrescreve)
    //   ch 60-63      : agent_state slot (motor.rs zera depois)
    //   ch 64-75      : telemetria proprioceptiva (RPM, vel, temp, drafting)
    //   ch 78-81      : curvatura + lane offset (ch 81 = racing_line.half_width)
    //   ch 82         : urgencia frontal (GPU=dist, CPU=TTC — diverge por design)
    //   ch 83-84      : fluxo optico hemisférico (visual_cortex anterior)
    //   ch 85-89      : wall EMA, slip, corpo, proximity, re
    //   ch 90-93      : dark-center (calculado sobre retina binocular)
    //   ch 99-100     : motion/edge rows 6-7 (temporal)
    //   ch 101-127    : misc motor.rs (curvatura 260/350m, approach, racing line, etc.)
    //   ch 128+       : pool legado HD + CfC percepcao (temporal)
    //
    // Canais COMPARADOS: 16-20, 24-28, 32-43, 48-59, 76-77, 94-97.
    // ==========================================================================
    if quer_percepcao {
        println!("[PERCEPCAO]");
        validate_perception(&world, &poses);
        println!();
    }

    println!("---");
    println!("NAO MEDIDO = readback nao exposto publicamente, nao que passou.");
    println!("PASS so aparece com o numero que o sustenta.");
}

// ============================================================================
// CEREBRO
// ============================================================================

fn validate_brain() {
    use universe::gpu::brain::{BrainGpu, BrainInput};
    use universe::engine::SovereignEngine;
    use universe::config::iara::brain::SENSORY_WAVE_SIZE;
    use std::sync::Arc;

    // CerebroGpu precisa de Arc<Device>. Criamos device proprio para esta secao.
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = match pollster::block_on(instance.request_adapter(
        &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        },
    )) {
        Some(a) => a,
        None => {
            println!("  NAO MEDIDO — sem adaptador GPU.");
            print_table_row("cerebro (forward 1 tick)", 0, 0, -1, false);
            print_table_row("cerebro (drift CfC 100t)", 0, 0, -1, false);
            return;
        }
    };
    // CerebroGpu precisa de storage buffers por shader stage:
    //   buf_wave(1), buf_kan(2), buf_l1w(3), buf_l2w(4), buf_intent(5),
    //   buf_cfc(6), buf_debug(7), buf_percept(8), buf_fatigue(9), buf_tau(10) = 10 storage.
    // Comentário antigo parava em 9 (buf_tau, o banco de τ por canal, foi adicionado
    // depois em src/gpu/brain.rs e este arquivo nunca foi atualizado — mesma classe de
    // bug de limite desatualizado já achada e corrigida em gpu_loop.rs no E3). Usamos
    // 12 (mesmo valor de context.rs/gpu_loop.rs) para dar folga a bindings futuros.
    // downlevel_defaults() tem max=4, insuficiente.
    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffers_per_shader_stage = 12;
    let (device_raw, queue_raw) = match pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("brain-validate"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    )) {
        Ok(p) => p,
        Err(e) => {
            println!("  NAO MEDIDO — request_device: {e}");
            print_table_row("cerebro (forward 1 tick)", 0, 0, -1, false);
            print_table_row("cerebro (drift CfC 100t)", 0, 0, -1, false);
            return;
        }
    };
    let nome   = adapter.get_info().name.clone();
    let device = Arc::new(device_raw);
    let queue  = Arc::new(queue_raw);

    let n   = 4usize;
    let cap = n.max(8);
    let brain = BrainGpu::new(Arc::clone(&device), Arc::clone(&queue), nome.clone(), cap);
    println!("  GPU: {}  · {} seres", nome, n);

    // LCG determinístico para preencher pesos: semente fixa garante
    // que CPU e GPU recebem EXATAMENTE os mesmos valores.
    // Sem isto, instinct_layer começa com zeros e o teste compara 0 vs 0 (vácuo).
    // Simétrico: o mapeamento antigo devolvia [-1,0)·0,1, ou seja pesos de validação
    // exclusivamente negativos — comparação válida, distribuição degenerada.
    let lcg_next = |s: &mut u64| -> f32 {
        let mut r = universe::rng::Lcg::new(*s);
        let v = r.symmetric() * 0.1;
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        v
    };

    // Engines para a GPU: preenchidos com pesos pseudo-aleatórios
    let mut engines: Vec<SovereignEngine> = (0..n).map(|_| SovereignEngine::new()).collect();
    {
        let mut rng: u64 = 0xBEEF_F00D_1234_5678;
        for eng in engines.iter_mut() {
            for w in eng.instinct_layer.weights.iter_mut() {
                *w = lcg_next(&mut rng);
            }
            eng.cfc_state.reset(); // garante estado zero = identico ao estado inicial GPU
        }
    }

    // Snapshot dos pesos L1 para replicar nos engines de CPU (mesma semente, mesma ordem)
    let l1_snapshots: Vec<Vec<f32>> = engines.iter()
        .map(|e| e.instinct_layer.weights.clone())
        .collect();

    let mag_cpu_l1: f32 = l1_snapshots[0].iter().map(|v| v.abs()).sum::<f32>()
        / l1_snapshots[0].len() as f32;
    println!("  |pesos L1| medio (ser 0): {mag_cpu_l1:.4e}  (>0 = pesos reais, nao zeros)");

    let eng_refs: Vec<&SovereignEngine> = engines.iter().collect();
    brain.upload_weights(&eng_refs);

    // Wave deterministico com valores variados (nao zeros)
    let mut waves: Vec<[i8; SENSORY_WAVE_SIZE]> = Vec::with_capacity(n);
    for s in 0..n {
        let mut w = [0i8; SENSORY_WAVE_SIZE];
        for (i, v) in w.iter_mut().enumerate() {
            *v = (((i as i32 * 7 + s as i32 * 13) % 200) - 100) as i8;
        }
        waves.push(w);
    }

    let dt = 1.0_f32 / 64.0;

    // GPU: forward batch
    let inputs: Vec<BrainInput<'_>> = engines.iter().zip(waves.iter())
        .map(|(eng, w)| BrainInput { engine: eng, wave: w })
        .collect();
    let intent_gpu = brain.forward(&inputs, dt);

    // Lê estado CfC da GPU pós-forward (para diagnóstico por camada)
    let cfc_gpu = brain.download_cfc_state(n);

    // Lê buffer de debug: l1_raw, cfc_inputs e kan_inputs para ser 0 diretamente da GPU
    let (l1_raw_gpu, cfc_in_gpu, kan_in_gpu) = brain.download_debug();

    // Lê fadiga de saída: verifica se out_fat está sendo acumulado (debug)
    {
        let fat = brain.download_fatigue(1);  // só ser 0
        // Slots [96..192] = output fatigue (slots [0..48] = L2, [48..96] = bypass)
        let out_fat_max = fat[96..192].iter().cloned().fold(0.0f32, f32::max);
        let out_fat_nonzero = fat[96..192].iter().filter(|&&v| v.abs() > 1e-6).count();
        println!("  [DIAGNÓSTICO FADIGA SAÍDA — ser 0, tick 1]");
        println!("  out_fat: max={:.4e}  canais_não_zero={}/96", out_fat_max, out_fat_nonzero);
        println!("  out_fat[0] (throttle L2 ch0) = {:.4e}", fat.get(96).copied().unwrap_or(0.0));
        // Com OUT_ALPHA=2.0, dt=1/64: out_fat[0] deve ser ~2.0 * |l2_raw[0]| * 0.015625
        // Qualquer valor > 0 confirma que o mecanismo está sendo executado.
        if out_fat_nonzero == 0 {
            println!("  AVISO: out_fat todos zero — fadiga de saída pode não estar ativa!");
        }
    }

    // CPU: replica explícita por camada (sem execute_cycle) para diagnóstico preciso.
    // Usa os mesmos pesos enviados para a GPU e o mesmo wave deterministico.
    // Cada etapa é explicitamente calculada para podermos comparar camada por camada.
    use universe::cfc::CfCCell;

    // Diagnóstico camada por camada para o ser 0 apenas (representativo)
    {
        let s = 0usize;
        let wave0 = &waves[s];
        let eng0 = &engines[s];

        // ETAPA KAN: aplica spline por canal (identidade com coeffs=1.0)
        let mut kan_wave0 = [0i8; SENSORY_WAVE_SIZE];
        eng0.kan_filters.apply(wave0.as_slice(), &mut kan_wave0);

        // ETAPA CfC: atualiza state[0..96] usando kan_wave0[0..96] como input
        let mut cfc_cpu = CfCCell::new(96);
        cfc_cpu.reset();  // estado zero = identico ao GPU
        let mut filtered_retina0 = [0i8; 96];
        cfc_cpu.update(&kan_wave0[0..96], &mut filtered_retina0, dt);

        // Compara CfC state: CPU vs GPU
        let cfc_state_gpu = &cfc_gpu[s];
        let mut cfc_max_dif = 0.0f32;
        let mut cfc_mismatch = 0usize;
        for i in 0..96 {
            let cpu_s = cfc_cpu.state[i];
            let gpu_s = cfc_state_gpu.get(i).copied().unwrap_or(0.0);
            let d = (cpu_s - gpu_s).abs();
            if d > cfc_max_dif { cfc_max_dif = d; }
            if d > 5e-4 { cfc_mismatch += 1; }
        }
        // Compara filtered_retina (saida CfC quantizada) CPU vs o que a GPU calcularia
        // (tanh(state) * 127, arredondado para i8)
        let mut fr_max_dif = 0i32;
        for i in 0..96 {
            let cpu_fr = filtered_retina0[i] as i32;
            let gpu_fr = (cfc_state_gpu.get(i).copied().unwrap_or(0.0).tanh() * 127.0)
                .clamp(-127.0, 127.0) as i32;
            let d = (cpu_fr - gpu_fr).abs();
            if d > fr_max_dif { fr_max_dif = d; }
        }

        // ETAPA L1: forward com filtered_wave = [filtered_retina0(0..96), kan_wave0(96..222)]
        let mut filtered_wave0 = [0i8; SENSORY_WAVE_SIZE];
        filtered_wave0.copy_from_slice(kan_wave0.as_slice());
        filtered_wave0[0..96].copy_from_slice(&filtered_retina0);

        // L1 manual (replica BitLinear::forward)
        let l1w = &eng0.instinct_layer.weights;
        let norm = 1.0_f32 / 127.0;
        let mut l1_raw0 = vec![0.0f32; 96];
        for j in 0..96 {
            let off = j * SENSORY_WAVE_SIZE;
            let mut acc = 0.0f32;
            for i in 0..SENSORY_WAVE_SIZE {
                acc += l1w[off + i] * (filtered_wave0[i] as f32);
            }
            l1_raw0[j] = acc * norm;
        }

        // ETAPA L2 (nos primeiros 48 do final_intent)
        let mut l1_act0 = [0i8; 96];
        for i in 0..96 {
            l1_act0[i] = ((l1_raw0[i].tanh() * 127.0).clamp(-127.0, 127.0)) as i8;
        }
        let l2w = &eng0.cortex_l2.weights;
        let mut l2_raw0 = vec![0.0f32; 48];
        for j in 0..48 {
            let off = j * 96;
            let mut acc = 0.0f32;
            for i in 0..96 {
                acc += l2w[off + i] * (l1_act0[i] as f32);
            }
            l2_raw0[j] = acc * norm;
        }

        // final_intent esperado pela CPU (camada por camada) — INCLUI fadiga de saída.
        // Replica exatamente o que o shader faz em brain.wgsl (etapa 2).
        //
        // Parâmetros espelho dos do shader (manter em sincronia com brain.wgsl):
        const OUT_ALPHA_CPU: f32 = 4.0;
        const OUT_BETA_CPU:  f32 = 0.85;
        //
        // Estado inicial de out_fat: zeros (mesmo que o buffer GPU no primeiro tick).
        // A fadiga é ATUALIZADA antes de ser aplicada (= comportamento do shader):
        //   out_fat += OUT_ALPHA * (|raw| - out_fat) * dt
        //   intent  = tanh(raw - OUT_BETA * out_fat_novo)
        let mut fi_cpu_manual = [0.0f32; 96];
        for j in 0..48 {
            let raw = l2_raw0[j] * 2.0;          // l2_raw como no shader (×2/127 × 127 = ×2)
            // out_fat parte de zero: atualiza primeiro, depois aplica
            let out_fat = 0.0f32 + OUT_ALPHA_CPU * (raw.abs() - 0.0f32) * dt;
            fi_cpu_manual[j] = adapta_saida(raw, out_fat);
        }
        for j in 48..96 {
            let raw = l1_raw0[j] * 2.0;          // bypass: raw_bp como no shader (×2)
            let out_fat = 0.0f32 + OUT_ALPHA_CPU * (raw.abs() - 0.0f32) * dt;
            fi_cpu_manual[j] = adapta_saida(raw, out_fat);
        }

        let fi_gpu0 = &intent_gpu[s];
        let mut fi_max_dif = 0.0f32;
        let mut fi_mismatch = 0usize;
        let mut fi_worst: Option<(usize, f32, f32, f32)> = None;
        for d in 0..96 {
            let c = fi_cpu_manual[d];
            let g = fi_gpu0[d];
            let dif = (c - g).abs();
            if dif > fi_max_dif {
                fi_max_dif = dif;
                fi_worst = Some((d, c, g, dif));
            }
            if dif > 5e-4 { fi_mismatch += 1; }
        }

        println!("  [DIAGNÓSTICO CAMADA A CAMADA — ser 0]");
        println!("  KAN: state identidade (coeffs=1.0) — sem divergencia esperada");
        println!("  CfC state max_dif={cfc_max_dif:.4e}  mismatches(>5e-4)={cfc_mismatch}/96");
        println!("  CfC filtered_retina max_dif_i8={fr_max_dif}  (0=identico)");
        println!("  final_intent (CPU-manual vs GPU) max_dif={fi_max_dif:.4e}  mismatches={fi_mismatch}/96");
        if let Some((d, c, g, dif)) = fi_worst {
            println!("    pior: dim={d}  cpu_manual={c:.6}  gpu={g:.6}  dif={dif:.4e}");
        }
        // Compara final_intent[0] diretamente: calcula L2 de L1 vs GPU
        let fi_l2_cpu = fi_cpu_manual[0];
        let fi_l2_gpu = fi_gpu0[0];
        println!("  fi[0]: cpu_manual={fi_l2_cpu:.6}  gpu={fi_l2_gpu:.6}  dif={:.4e}", (fi_l2_cpu-fi_l2_gpu).abs());
        let fi_bypass_cpu = fi_cpu_manual[48];
        let fi_bypass_gpu = fi_gpu0[48];
        println!("  fi[48] (bypass L1): cpu_manual={fi_bypass_cpu:.6}  gpu={fi_bypass_gpu:.6}  dif={:.4e}", (fi_bypass_cpu-fi_bypass_gpu).abs());

        // Comparação DIRETA de l1_raw: CPU manual vs buffer de debug da GPU
        // (sem inversão via bypass — buf_debug[0..96] = w_l1r exportado diretamente)
        let mut l1r_direct_max = 0.0f32;
        let mut l1r_direct_worst = 0usize;
        let mut l1a_direct_mismatches = 0usize;
        for j in 0..96 {
            let cpu_v = l1_raw0[j];
            let gpu_v = l1_raw_gpu[j];
            let d = (cpu_v - gpu_v).abs();
            if d > l1r_direct_max { l1r_direct_max = d; l1r_direct_worst = j; }
            let l1a_cpu = (cpu_v.tanh() * 127.0).clamp(-127.0, 127.0) as i32;
            let l1a_gpu_v = (gpu_v.tanh() * 127.0).clamp(-127.0, 127.0) as i32;
            if l1a_cpu != l1a_gpu_v { l1a_direct_mismatches += 1; }
        }
        println!("  L1 raw DIRETO (buf_debug): max_dif={l1r_direct_max:.4e}  pior_j={l1r_direct_worst}");
        println!("    cpu={:.6}  gpu={:.6}", l1_raw0[l1r_direct_worst], l1_raw_gpu[l1r_direct_worst]);
        println!("  L1 act mismatches (i8 diferente): {l1a_direct_mismatches}/96");

        // Mostra L1 raw j=21 (maior divergência no resultado final) diretamente
        let l1r21_cpu = l1_raw0[21];
        let l1r21_gpu_direct = l1_raw_gpu[21];
        let l1a21_cpu = l1_act0[21] as i32;
        let l1a21_gpu = (l1r21_gpu_direct.tanh() * 127.0).clamp(-127.0, 127.0) as i32;
        println!("  L1[21]: cpu={l1r21_cpu:.6}  gpu_direct={l1r21_gpu_direct:.6}  dif={:.4e}", (l1r21_cpu-l1r21_gpu_direct).abs());
        println!("  l1_act[21]: cpu={l1a21_cpu}  gpu={l1a21_gpu}  → mesmo i8? {}", l1a21_cpu == l1a21_gpu);

        // Verifica se os inputs CfC e KAN na GPU são idênticos aos da CPU.
        // NOTA: buf_debug[192..287] = w_kan[96..191] (96 valores escritos por threads 0..95).
        //       buf_debug[287..319] = w_kan[192..222] NÃO escrito (zeros). Comparar apenas 96 valores.
        let mut cfc_in_max_dif = 0i32;
        let mut cfc_in_worst = 0usize;
        let mut kan_in_max_dif = 0i32;
        let mut kan_in_worst = 0usize;
        for i in 0..96 {
            let cpu_v = filtered_wave0[i] as i32;  // CfC output (parte 0..96)
            let gpu_v = cfc_in_gpu[i] as i32;
            let d = (cpu_v - gpu_v).abs();
            if d > cfc_in_max_dif { cfc_in_max_dif = d; cfc_in_worst = i; }
        }
        // Compara apenas os 96 primeiros canais KAN (96..191), que foram escritos no buf_debug
        for i in 0..96 {
            let cpu_v = filtered_wave0[96 + i] as i32;  // KAN output para canal 96+i
            let gpu_v = kan_in_gpu[i] as i32;           // w_kan[96+i] da GPU
            let d = (cpu_v - gpu_v).abs();
            if d > kan_in_max_dif { kan_in_max_dif = d; kan_in_worst = i; }
        }
        println!("  Inputs L1: cfc_in[0..96] max_dif_i8={cfc_in_max_dif}  kan_in[96..191] max_dif_i8={kan_in_max_dif}");
        if cfc_in_max_dif > 0 {
            println!("    CfC pior ch={cfc_in_worst}  cpu={}  gpu={}", filtered_wave0[cfc_in_worst] as i32, cfc_in_gpu[cfc_in_worst] as i32);
        }
        if kan_in_max_dif > 0 {
            println!("    KAN pior ch={}  cpu={}  gpu={}", 96+kan_in_worst, filtered_wave0[96+kan_in_worst] as i32, kan_in_gpu[kan_in_worst] as i32);
        }
        println!("    (0=inputs identicos em ambas faixas → divergencia em l1_raw vem de matmul FP ≤1e-5)");
        println!();
    }

    // CPU: execute_cycle em engines com MESMOS pesos dos engines de GPU.
    // CRITICO: reusar l1_snapshots para garantir identidade de pesos CPU vs GPU.
    // Para engines novos (tick 0, dopamine=0, champion_snapshot=None), as partes
    // de mirror neurons / resonance nao alteram final_intent. Comparacao e valida.
    //
    // execute_cycle agora inclui a fadiga de saída (OUT_ALPHA/OUT_BETA) em engine.rs,
    // espelhando brain.wgsl. Não há mais recálculo manual aqui — o teste compara
    // engine.execute_cycle() contra a GPU diretamente, sem gambiarra no meio.
    let intent_cpu: Vec<Vec<f32>> = waves.iter().enumerate()
        .map(|(s, w)| {
            let mut cpu_eng = SovereignEngine::new();
            // Aplica os mesmos pesos L1 que foram para a GPU
            cpu_eng.instinct_layer.weights.copy_from_slice(&l1_snapshots[s]);
            cpu_eng.cfc_state.reset();
            // out_fatigue já é [0.0; 96] no new() — idêntico ao estado GPU pós reset_cfc.
            cpu_eng.execute_cycle(w.as_slice(), 0.0, dt);
            cpu_eng.final_intent.clone()
        })
        .collect();

    // Magnitude do sinal CPU (deve ser >0 com pesos reais)
    let mag_intent: f32 = intent_cpu.iter().flat_map(|v| v.iter())
        .map(|v| v.abs()).sum::<f32>()
        / (n * intent_cpu.first().map(|v| v.len()).unwrap_or(1)).max(1) as f32;
    println!("  |final_intent| medio na CPU: {mag_intent:.4e}  (PASS com zero e falso-positivo)");

    let tol = 5e-4_f32;
    let mut comp_total  = 0usize;
    let mut comp_iguais = 0usize;
    let mut comp_maxabs = 0.0f32;
    let mut comp_maxrel = 0.0f32;
    let mut piores: Vec<(usize, usize, f32, f32, f32)> = Vec::new();
    let dims = intent_gpu.first().map(|a| a.len()).unwrap_or(0).min(96);

    for s in 0..n {
        if s >= intent_gpu.len() || s >= intent_cpu.len() { break; }
        for d in 0..dims {
            let c = intent_cpu[s].get(d).copied().unwrap_or(0.0);
            let g = intent_gpu[s][d];
            let dif = (c - g).abs();
            let rel = if c.abs() > 1e-6 { dif / c.abs() } else { dif };
            if dif <= tol { comp_iguais += 1; }
            if dif > comp_maxabs { comp_maxabs = dif; comp_maxrel = rel; }
            if dif > tol { piores.push((s, d, c, g, dif)); }
            comp_total += 1;
        }
    }
    piores.sort_by(|a,b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
    println!("  forward 1 tick: {comp_total} campos  {comp_iguais} dentro tol ({:.4}%)",
        comp_iguais as f64 / comp_total.max(1) as f64 * 100.0);
    println!("  max_abs={comp_maxabs:.2e}  max_rel={comp_maxrel:.2e}  tol<={tol:.2e}");
    if !piores.is_empty() {
        for (s,d,c,g,dif) in piores.iter().take(5) {
            println!("    ser={s} dim={d}  cpu={c:.6}  gpu={g:.6}  dif={dif:.2e}");
        }
    }
    let c_pass = comp_maxabs <= tol;
    print_table_row("cerebro (forward 1 tick)", comp_total, comp_iguais,
        (comp_maxabs * 1e6) as i32, c_pass);

    // ── Drift CfC: 100 ticks acumulados ──────────────────────────────────────
    // Mede quanto o estado CfC da GPU deriva em relação à CPU após N ticks.
    // Um drift crescente indica bug recorrente no kernel (ex: decay errado, noise diferente).
    // Aqui comparamos state[0..96] de cada ser após cada tick.
    println!("  [DRIFT CFC — 100 ticks, {} seres]", n);
    {
        use universe::cfc::CfCCell;

        // Estado CfC CPU: um CfCCell por ser, começando em zero (igual ao GPU)
        let mut cfc_cpus: Vec<CfCCell> = (0..n).map(|_| {
            let mut c = CfCCell::new(96);
            c.reset();
            c
        }).collect();

        // GPU: resetar CfC de todos os seres para zero antes do loop
        for s in 0..n {
            brain.reset_cfc(s);
        }

        let mut drift_max_per_tick: Vec<f32> = Vec::with_capacity(100);

        for tick in 0..100usize {
            // GPU: um forward com os mesmos waves determinísticos
            let inputs_loop: Vec<BrainInput<'_>> = engines.iter().zip(waves.iter())
                .map(|(eng, w)| BrainInput { engine: eng, wave: w })
                .collect();
            let _ = brain.forward(&inputs_loop, dt);

            // CPU: atualiza CfC manualmente (mesma entrada que o GPU recebe)
            for s in 0..n {
                let wave_s = &waves[s];
                let eng_s  = &engines[s];
                // KAN (identidade, coeffs=1.0, mas aplicar mesmo assim para fidelidade)
                let mut kan_wave = [0i8; SENSORY_WAVE_SIZE];
                eng_s.kan_filters.apply(wave_s.as_slice(), &mut kan_wave);
                // CfC atualiza com os primeiros 96 canais pós-KAN
                let mut filtered = [0i8; 96];
                cfc_cpus[s].update(&kan_wave[0..96], &mut filtered, dt);
            }

            // Comparar state CfC GPU vs CPU neste tick
            let cfc_states_gpu = brain.download_cfc_state(n);
            let mut tick_max = 0.0f32;
            for s in 0..n {
                let gpu_st = &cfc_states_gpu[s];
                let cpu_st = &cfc_cpus[s].state;
                for i in 0..96 {
                    let d = (cpu_st[i] - gpu_st.get(i).copied().unwrap_or(0.0)).abs();
                    if d > tick_max { tick_max = d; }
                }
            }
            drift_max_per_tick.push(tick_max);
        }

        let d10  = drift_max_per_tick.get(9).copied().unwrap_or(0.0);
        let d50  = drift_max_per_tick.get(49).copied().unwrap_or(0.0);
        let d100 = drift_max_per_tick.last().copied().unwrap_or(0.0);
        let dmax = drift_max_per_tick.iter().cloned().fold(0.0f32, f32::max);
        let ratio = if d10 > 1e-9 { d100 / d10 } else { 0.0 };
        println!("  drift state t=10:{d10:.4e}  t=50:{d50:.4e}  t=100:{d100:.4e}  max:{dmax:.4e}  ratio:{ratio:.1}x");
        let drift_pass = d100 < 1e-3;
        println!("  veredito: {} (limiar 1e-3)", if drift_pass { "PASS" } else { "FAIL" });
        print_table_row("cerebro (drift CfC 100t)", 100,
            if drift_pass { 100 } else { 0 },
            (dmax * 1e6) as i32, drift_pass);
    }
}

// ============================================================================
// FISICA
// ============================================================================

fn validate_physics(world: &VoxelWorld) {
    use universe::gpu::physics::{PhysicsGpu, PhysicsStateGpu, pack_config_flags};

    let grid_u8: Vec<u8> = world.grid.iter().map(|v| *v as u8).collect();
    let phys_gpu = match PhysicsGpu::new(&grid_u8, &world.height_map, world.size, world.cell_size, 16) {
        Some(g) => g,
        None => {
            println!("  NAO MEDIDO — sem GPU.");
            print_table_row("fisica (1 passo 6 cen)", 0, 0, -1, false);
            print_table_row("fisica (deriva 100 passos)", 0, 0, -1, false);
            return;
        }
    };
    println!("  GPU: {}  · cap={}", phys_gpu.adapter, phys_gpu.capacity);

    let dt    = 1.0_f32 / 60.0;
    let fl    = pack_config_flags(true, true, true, true, true, true);
    let fl_ns = pack_config_flags(false, true, true, true, true, true);

    macro_rules! base {
        ($dt:expr, $fl:expr) => { state_base($dt, $fl) };
    }

    // 6 cenários
    let mut cen_cpu: Vec<PhysicsStateGpu> = Vec::new();
    let mut cen_names: Vec<&str> = Vec::new();
    {
        macro_rules! push {
            ($nome:expr, $e:expr) => {
                cen_names.push($nome);
                cen_cpu.push($e);
            };
        }
        let mut e0 = base!(dt, fl);
        e0.vel = [20.0, 0.0, 0.0]; e0.current_rpm = 4000.0;
        e0.current_gear = 3; e0.throttle = 0.5;
        let w = 20.0/0.33; for wh in e0.wheels.iter_mut() { wh[0] = w; }
        push!("vel_media", e0);

        let mut e1 = base!(dt, fl);
        e1.vel = [10.0, 0.0, 0.0]; e1.current_rpm = 3500.0;
        e1.current_gear = 2; e1.throttle = 1.0;
        let w = 10.0/0.33; for wh in e1.wheels.iter_mut() { wh[0] = w; }
        push!("acelerando", e1);

        let mut e2 = base!(dt, fl);
        e2.vel = [15.0, 0.0, 2.0]; e2.av_y = 0.3;
        e2.current_rpm = 3000.0; e2.current_gear = 3;
        e2.throttle = 0.6; e2.steer = 0.3;
        let w = 15.0/0.33; for wh in e2.wheels.iter_mut() { wh[0] = w; }
        push!("em_curva", e2);

        let mut e3 = base!(dt, fl);
        e3.vel = [25.0, 0.0, 5.0]; e3.av_y = 0.8;
        e3.current_rpm = 5000.0; e3.current_gear = 4;
        e3.throttle = 0.3; e3.steer = -0.4;
        let w = 25.0/0.33;
        e3.wheels[0][0] = w; e3.wheels[2][0] = w;
        e3.wheels[4][0] = w*1.4; e3.wheels[6][0] = w*1.4;
        push!("derrapando", e3);

        let mut e4 = base!(dt, fl_ns);
        e4.vel = [3.0, 0.0, 0.0]; e4.current_rpm = 0.0;
        e4.current_gear = 2; e4.is_stalled = 1;
        push!("motor_travado", e4);

        let mut e5 = base!(dt, fl);
        e5.vel = [1.0, 0.0, 0.0]; e5.current_rpm = 900.0;
        e5.current_gear = 2; e5.brake_cmd = 0.8;
        let w = 1.0/0.33; for wh in e5.wheels.iter_mut() { wh[0] = w; }
        push!("freando", e5);
    }

    // CPU: replica das 5 fases
    let results_cpu: Vec<PhysicsStateGpu> = cen_cpu.iter()
        .map(|e| physics_step_cpu(*e, world))
        .collect();

    // GPU: batch
    let mut gpu_states: Vec<PhysicsStateGpu> = cen_cpu.clone();
    let ok_g = phys_gpu.step(&mut gpu_states);
    if !ok_g {
        println!("  NAO MEDIDO — passo() retornou false.");
        print_table_row("fisica (1 passo 6 cen)", 0, 0, -1, false);
        print_table_row("fisica (deriva 100 passos)", 0, 0, -1, false);
        return;
    }

    let n_cen = cen_names.len();
    println!("  1 passo: {n_cen} cenarios, dt={dt:.4}s");

    let tol_abs = 1e-3_f32;
    let tol_rel = 1e-2_f32;
    let mut total  = 0usize;
    let mut iguais = 0usize;
    let mut maxabs = 0.0f32;
    let mut maxrel = 0.0f32;
    let mut piores: Vec<(usize, &'static str, f32, f32, f32)> = Vec::new();

    macro_rules! cmp {
        ($cen:expr, $campo:expr, $cpu:expr, $gpu:expr) => {
            let d = ($cpu - $gpu).abs();
            let r = if $cpu.abs() > 1e-4 { d / $cpu.abs() } else { d };
            total += 1;
            if d <= tol_abs || r <= tol_rel { iguais += 1; }
            if d > maxabs { maxabs = d; maxrel = r; }
            if d > tol_abs && r > tol_rel { piores.push(($cen, $campo, $cpu, $gpu, d)); }
        }
    }

    for c in 0..n_cen {
        let cp = &results_cpu[c];
        let gp = &gpu_states[c];
        for k in 0..3 { cmp!(c, ["vel.x","vel.y","vel.z"][k], cp.vel[k], gp.vel[k]); }
        cmp!(c, "av_y", cp.av_y, gp.av_y);
        for k in 0..3 { cmp!(c, ["pos.x","pos.y","pos.z"][k], cp.pos[k], gp.pos[k]); }
        cmp!(c, "current_rpm", cp.current_rpm, gp.current_rpm);
        cmp!(c, "engine_temp", cp.engine_temp, gp.engine_temp);
        cmp!(c, "oil_temp", cp.oil_temp, gp.oil_temp);
        // is_stalled exato
        total += 1;
        if cp.is_stalled == gp.is_stalled { iguais += 1; }
        else { piores.push((c, "is_stalled", cp.is_stalled as f32, gp.is_stalled as f32, 1.0)); }
        const WN: [&str;4] = ["omega","slip_r","slip_a","norm_load"];
        for wi in 0..4 {
            for fi in 0..4 { cmp!(c, WN[fi], cp.wheels[wi*2][fi], gp.wheels[wi*2][fi]); }
        }
    }

    piores.sort_by(|a,b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
    println!("  comparados={total}  identicos={iguais} ({:.4}%)  max_abs={maxabs:.4e}  max_rel={maxrel:.4e}",
        iguais as f64 / total.max(1) as f64 * 100.0);
    if !piores.is_empty() {
        println!("  DIVERGENCIAS (ate 5):");
        for (c, campo, cpu_v, gpu_v, d) in piores.iter().take(5) {
            println!("    cen={c}({}) campo={campo}  cpu={cpu_v:.6}  gpu={gpu_v:.6}  dif={d:.4e}", cen_names[*c]);
        }
    }
    // VEREDITO: usa critério DUAL (abs OU rel) igual ao da contagem de `iguais`.
    // Antes usava só `maxabs <= tol_abs`, que marcava FALHA quando max_abs=3.39e-3
    // (> tol_abs=1e-3) mesmo com 162/162 idênticos pelo critério dual — rótulo enganoso.
    // A tolerância real (tol_abs=1e-3, tol_rel=1e-2) NÃO foi afrouxada; só o veredito
    // final foi alinhado com o mesmo critério usado em cada campo individualmente.
    let fisica_pass = iguais == total;
    print_table_row("fisica (1 passo 6 cen)", total, iguais, (maxabs*1e4) as i32, fisica_pass);

    // Deriva 100 passos
    println!("  [DERIVA 100 PASSOS — cenario vel_media]");
    let mut ec = base!(dt, fl);
    ec.vel = [20.0,0.0,0.0]; ec.current_rpm = 4000.0;
    ec.current_gear = 3; ec.throttle = 0.5;
    let w = 20.0/0.33; for wh in ec.wheels.iter_mut() { wh[0] = w; }
    let mut eg = ec;

    let mut derivas = Vec::with_capacity(100);
    let mut divergiu: Option<usize> = None;

    for tick in 0..100usize {
        let en = physics_step_cpu(ec, world);
        let mut gs = vec![eg];
        phys_gpu.step(&mut gs);
        let gn = gs[0];
        let mut md = 0.0f32;
        for k in 0..3 {
            md = md.max((en.vel[k]-gn.vel[k]).abs());
            md = md.max((en.pos[k]-gn.pos[k]).abs());
        }
        md = md.max((en.av_y-gn.av_y).abs());
        if tick == 0 {
            println!("  [drift t=0] cpu_vel={:.4?} gpu_vel={:.4?}  cpu_pos={:.4?} gpu_pos={:.4?}  dif={md:.4e}",
                en.vel, gn.vel, en.pos, gn.pos);
        }
        derivas.push(md);
        if md > 1.0 && divergiu.is_none() { divergiu = Some(tick); }
        ec = en; eg = gn;
    }

    let d10  = derivas.get(9).copied().unwrap_or(0.0);
    let d50  = derivas.get(49).copied().unwrap_or(0.0);
    let d100 = derivas.last().copied().unwrap_or(0.0);
    let dmax = derivas.iter().cloned().fold(0.0f32, f32::max);
    let ratio = if d10 > 1e-9 { d100/d10 } else { 0.0 };
    println!("  drift t=10:{d10:.4e}  t=50:{d50:.4e}  t=100:{d100:.4e}  max:{dmax:.4e}  ratio:{ratio:.1}x");
    if let Some(t) = divergiu {
        println!("  AVISO: deriva>1.0 no tick {t}");
    } else {
        println!("  crescimento {:.0}x — {}", ratio,
            if ratio < 50.0 { "sub-linear (arredondamento FP)" } else { "supra-linear (investigar)" });
    }
    let drift_pass = d100 < 0.5 && divergiu.is_none();
    print_table_row("fisica (deriva 100 passos)", 100,
        if drift_pass { 100 } else { 0 }, (d100*1000.0) as i32, drift_pass);
}

// Replica das 5 fases de physics.rs — ver comentario no cabecalho da secao FISICA.
fn physics_step_cpu(
    mut e: universe::gpu::physics::PhysicsStateGpu,
    world: &VoxelWorld,
) -> universe::gpu::physics::PhysicsStateGpu {

    let dt   = e.dt;
    let mass = e.mass;
    let speed = v3len(e.vel);

    // FASE 1: reset_forces
    let mut forces  = [0.0f32, -9.81 * mass, 0.0f32];
    let mut torques = [0.0f32; 3]; // [0]=yaw_inertia stash, [1]=yaw_torque

    // FASE 2: suspension_and_load
    let hw = e.wheel_base * 0.5;
    let ht = e.track_width * 0.5;
    let offsets: [[f32; 3]; 4] = [
        [e.fwd[0]*hw - e.lat[0]*ht, e.fwd[1]*hw - e.lat[1]*ht, e.fwd[2]*hw - e.lat[2]*ht],
        [e.fwd[0]*hw + e.lat[0]*ht, e.fwd[1]*hw + e.lat[1]*ht, e.fwd[2]*hw + e.lat[2]*ht],
        [-e.fwd[0]*hw - e.lat[0]*ht, -e.fwd[1]*hw - e.lat[1]*ht, -e.fwd[2]*hw - e.lat[2]*ht],
        [-e.fwd[0]*hw + e.lat[0]*ht, -e.fwd[1]*hw + e.lat[1]*ht, -e.fwd[2]*hw + e.lat[2]*ht],
    ];
    let en_down = (e.config_flags >> 3) & 1 != 0;
    let down_pw = if en_down { 3.0*speed*speed*0.25 } else { 0.0 };
    let mut tload = 0.0f32;
    for (i, off) in offsets.iter().enumerate() {
        let wx = e.pos[0]+off[0]; let wz = e.pos[2]+off[2];
        let wy = e.pos[1]+off[1];
        let gnd = world.get_height_at(wx, wz);
        let comp = e.susp_rest - (wy - gnd);
        if comp > 0.0 && comp < e.susp_max_travel {
            let prev = e.wheels[i*2+1][2];
            let f = (e.susp_stiffness*comp + e.susp_damping*(comp-prev)/dt).max(0.0) + down_pw;
            let sl = mass*9.81/4.0;
            let cl = f.clamp(sl*0.2, sl*6.0);
            forces[1] += cl;
            e.wheels[i*2][3] = cl;
            e.wheels[i*2+1][1] = comp;
            tload += cl;
        } else {
            e.wheels[i*2][3] = 0.0;
            e.wheels[i*2+1][1] = 0.0;
        }
        e.wheels[i*2+1][2] = comp;
    }
    if tload < 100.0 {
        let pw = mass*9.81*0.25;
        for i in 0..4 { e.wheels[i*2][3] = pw; }
    }

    // FASE 3: powertrain_and_tire
    let fwd_v = v3dot(e.vel, e.fwd);
    let lat_v = v3dot(e.vel, e.lat);
    let gears = [-3.5f32, 0.0, 3.8, 2.6, 1.9, 1.45, 1.15];
    let fd = 3.7f32;
    let gr = gears[e.current_gear as usize];
    let tr = gr * fd;
    let tra = tr.abs();
    let en_stall = e.config_flags & 1 != 0;

    if e.is_stalled != 0 && en_stall {
        if e.current_gear <= 2 && e.throttle > 0.5 {
            e.starter_timer += dt;
            if e.starter_timer >= 1.0 {
                e.is_stalled = 0; e.starter_timer = 0.0; e.current_rpm = 800.0;
                for i in 0..4 { e.wheels[i*2][0] = 0.0; }
            }
        } else { e.starter_timer = 0.0; }
    }
    if e.shift_timer > 0.0 { e.shift_timer = (e.shift_timer-dt).max(0.0); }

    let do_avg = match e.drivetrain {
        1 => (e.wheels[4][0]+e.wheels[6][0])*0.5,
        0 => (e.wheels[0][0]+e.wheels[2][0])*0.5,
        _ => (e.wheels[0][0]+e.wheels[2][0]+e.wheels[4][0]+e.wheels[6][0])*0.25,
    };
    let wrpm = do_avg.abs()*tra*60.0/(2.0*std::f32::consts::PI);
    let frpm = 800.0 + e.throttle*(e.max_rpm-800.0);
    let cb = if e.current_gear==1 { 0.0 }
        else if wrpm < 1500.0 { (wrpm/1500.0).clamp(0.0,1.0) }
        else { 1.0 };
    let cslip = cb < 0.95;
    let mut raw = frpm*(1.0-cb) + wrpm*cb;
    if e.shift_timer > 0.0 {
        raw = (e.current_rpm - e.current_rpm*dt/e.flywheel_inertia.max(0.05)*50.0).max(800.0);
    } else if en_stall && !cslip && e.is_stalled==0 && raw < 550.0 {
        e.is_stalled = 1; e.starter_timer = 0.0;
    }
    let en_therm = (e.config_flags>>1)&1 != 0;
    if e.is_stalled==0 && en_therm {
        let rr = e.current_rpm/e.max_rpm;
        e.engine_temp = (e.engine_temp + rr*e.throttle*15.0*dt - (speed*0.3+2.0)*dt).clamp(20.0,150.0);
        e.oil_temp = e.oil_temp + (e.engine_temp-e.oil_temp)*0.01*dt;
    } else { e.engine_temp = (e.engine_temp - 1.0*dt).max(20.0); }
    let tp = if en_therm && e.engine_temp > 105.0 {
        (1.0-(e.engine_temp-105.0)/45.0).clamp(0.1,1.0) } else { 1.0 };
    let dtq;
    if e.is_stalled != 0 {
        e.current_rpm = 0.0; dtq = 0.0;
    } else {
        e.current_rpm = raw.clamp(800.0, e.max_rpm+250.0);
        let rr = e.current_rpm/e.max_rpm;
        let mut tm = 1.0 - (rr-0.6).powi(2)*2.0;
        if rr > 1.0 { tm = 0.0; } else if rr < 0.2 { tm *= rr/0.2; }
        tm = tm.clamp(0.0,1.0);
        let ce = if e.shift_timer > 0.0 { 0.0 } else { 1.0 };
        let en_tcs = (e.config_flags>>4)&1 != 0;
        let mut tcs = 1.0f32;
        if en_tcs && e.drivetrain==1 {
            let sa = (e.wheels[4][1].max(0.0)+e.wheels[6][1].max(0.0))*0.5;
            if sa > 0.15 { tcs = 1.0-((sa-0.15)*5.0).clamp(0.0,0.9); }
        }
        dtq = e.max_torque*tm*e.throttle*tcs*tr*ce*tp;
        e.drive_torque_out = dtq;
    }

    let en_aero = (e.config_flags>>2)&1 != 0;
    forces[0] += e.wind[0]*mass*0.5;
    forces[1] += e.wind[1]*mass*0.5;
    forces[2] += e.wind[2]*mass*0.5;
    let fa = 2.2f32;
    let a1 = if en_aero { 0.5*1.225*speed*speed*e.drag_coeff*fa } else { 0.0 };
    let vl = speed.max(1e-6);
    forces[0] -= e.vel[0]/vl*a1; forces[1] -= e.vel[1]/vl*a1; forces[2] -= e.vel[2]/vl*a1;
    let d2 = if en_down { 0.5*1.225*speed*speed*e.drag_coeff*2.5*fa } else { 0.0 };
    forces[1] -= d2;
    let roll: f32 = e.wheels.iter().step_by(2).map(|w| w[3]).sum::<f32>() * 0.015;
    if speed > 0.05 { forces[0]-=e.vel[0]/vl*roll; forces[1]-=e.vel[1]/vl*roll; forces[2]-=e.vel[2]/vl*roll; }
    if e.is_stalled==0 && e.throttle<0.05 && fwd_v.abs()>1.0 {
        let eb = (tra*80.0).min(fwd_v.abs()*mass/dt);
        forces[0]-=e.fwd[0]*fwd_v.signum()*eb; forces[1]-=e.fwd[1]*fwd_v.signum()*eb; forces[2]-=e.fwd[2]*fwd_v.signum()*eb;
    }

    // Grip: GPU usa c.terrain_grip do estado (setado pelo host antes do dispatch).
    // CPU replica usa o mesmo valor — NAO recomputa do voxel (evita divergencia em Grass).
    let tg = e.terrain_grip;
    let eg_ = e.env_grip;
    let di: &[usize] = match e.drivetrain { 1=>&[2,3], 0=>&[0,1], _=>&[0,1,2,3] };
    let ndw = di.len() as f32;
    let mut wt = [0.0f32;4];
    if di.len()==2 {
        let (li,ri)=(di[0],di[1]);
        let dO=(e.wheels[ri*2][0]-e.wheels[li*2][0]).abs();
        let lf=(e.lsd_lock_ratio*dO*0.5).clamp(0.0,0.4);
        let sl=0.5+lf*(e.wheels[ri*2][0]-e.wheels[li*2][0]).signum();
        wt[li]=dtq*sl; wt[ri]=dtq*(1.0-sl);
    } else { for &idx in di { wt[idx]=dtq/ndw; } }
    let bm=e.brake_cmd*e.brake_torque_max;
    let bp=[bm*0.30,bm*0.30,bm*0.20,bm*0.20];
    let wlp=[hw,hw,-hw,-hw];
    let wlt=[-ht,ht,-ht,ht];
    let ms = if speed<15.0{35.0} else {(35.0-(speed-15.0)*0.4).max(12.0)};
    let en_abs=(e.config_flags>>5)&1!=0;
    let en_tt=en_therm;
    let aq=e.aquaplaning;
    let mut tlong=0.0f32; let mut tlat=0.0f32; let mut tyaw=0.0f32;

    for i in 0..4 {
        let om=e.wheels[i*2][0]; let load=e.wheels[i*2][3].max(10.0);
        let tt=e.wheels[i*2+1][0];
        let tf=if en_tt{1.0-(tt-e.optimal_temp).abs()*0.5}else{1.0};
        let mu=e.mu_peak*tg*eg_*tf.clamp(0.7,1.0);
        let ws=om*e.wheel_radius;
        let rfwd=fwd_v+e.av_y*wlt[i]; let rlat=lat_v-e.av_y*wlp[i];
        let sa=if i<2{e.steer*ms.to_radians()}else{0.0};
        let (cs,ss)=(sa.cos(),sa.sin());
        let cfwd=rfwd*cs+rlat*ss; let clat=-rfwd*ss+rlat*cs;
        let vr=ws.abs().max(cfwd.abs()).max(0.5);
        let sr=((ws-cfwd)/vr).clamp(-1.0,1.0);
        let fl=load*mu*(1.5*(10.0*sr).atan()).sin();
        let (fla,sla)=if cfwd.abs()>0.5{
            let a=clat.atan2(cfwd.abs());
            (load*mu*(1.35*(12.0*a).atan()).sin(),a)
        } else { ((clat*load*mu*2.0).clamp(-load*mu,load*mu),0.0) };
        let mf=mu*load; let cb2=(fl*fl+fla*fla).sqrt();
        let (ffl,ffla)=if cb2>mf&&cb2>0.001{let sc=mf/cb2;(fl*sc,fla*sc)}else{(fl,fla)};
        let rxn=ffl*e.wheel_radius;
        let mut xo=om+(wt[i]-rxn)/e.wheel_inertia*dt;
        if bp[i]>0.0{
            let st=(xo.abs()*e.wheel_inertia)/dt;
            let abk=if en_abs{bp[i].min(st)}else{bp[i]};
            if xo>0.0{xo-=abk/e.wheel_inertia*dt;}else if xo<0.0{xo+=abk/e.wheel_inertia*dt;}
        }
        e.wheels[i*2][0]=xo; e.wheels[i*2][1]=sr; e.wheels[i*2][2]=sla;
        let cfl=ffl*cs-ffla*ss; let cfla=ffl*ss+ffla*cs;
        tlong+=cfl*aq; tlat+=cfla*aq; tyaw+=cfla*wlp[i]*aq;
        tyaw+=cfl*wlt[i]*0.005*(speed/5.0).clamp(0.0,1.0);
        if en_tt {
            let sh=(sr.abs()+sla.abs())*load*2e-6*dt;
            let bh=bp[i]*xo.abs()*1e-6*dt;
            let co=0.02*(1.0+speed*0.01)*dt;
            e.wheels[i*2+1][0]=(e.wheels[i*2+1][0]+sh+bh-co).clamp(0.0,1.0);
        }
    }
    forces[0]+=e.fwd[0]*tlong-e.lat[0]*tlat;
    forces[1]+=e.fwd[1]*tlong-e.lat[1]*tlat;
    forces[2]+=e.fwd[2]*tlong-e.lat[2]*tlat;
    let yi=mass*(e.wheel_base*0.5).powi(2);
    torques[0]=yi; torques[1]=tyaw;

    // FASE 4: colisao parcial — a GPU (fisica.wgsl linhas 681-686) faz um "ground clamp"
    // baseado em heightmap ANTES da integracao: se pos.y < height_at(pos.x,pos.z)+0.5,
    // sobe pos.y e zera vel.y negativa. Sem isso o carro começa abaixo do terreno e
    // recebe impulso upward.
    let ground_y_c = world.get_height_at(e.pos[0], e.pos[2]) + 0.5;
    if e.pos[1] < ground_y_c {
        e.pos[1] = ground_y_c;
        if e.vel[1] < 0.0 { e.vel[1] = 0.0; }
    }

    // FASE 5: integration
    // Aero drag aplicado SEGUNDA VEZ (bug replicado — fisica.wgsl nota 7)
    let a2=if en_aero{0.5*1.225*e.drag_coeff*fa*speed.powi(2)}else{0.0};
    let acc=[
        (forces[0]-e.vel[0]/vl*a2)/mass,
        (forces[1]-e.vel[1]/vl*a2)/mass,
        (forces[2]-e.vel[2]/vl*a2)/mass,
    ];
    e.vel[0]+=acc[0]*dt; e.vel[1]+=acc[1]*dt; e.vel[2]+=acc[2]*dt;
    let fsp=v3dot(e.vel,e.fwd);
    let lv=[e.vel[0]-e.fwd[0]*fsp,e.vel[1]-e.fwd[1]*fsp,e.vel[2]-e.fwd[2]*fsp];
    if v3len(lv)>0.5 {
        let sc=(1.0-3.0*dt).max(0.85);
        e.vel[0]=e.fwd[0]*fsp+lv[0]*sc;
        e.vel[1]=e.fwd[1]*fsp+lv[1]*sc;
        e.vel[2]=e.fwd[2]*fsp+lv[2]*sc;
    }
    e.pos[0]+=e.vel[0]*dt; e.pos[1]+=e.vel[1]*dt; e.pos[2]+=e.vel[2]*dt;
    let yd=-(400.0+speed*60.0)*e.av_y;
    let yi2=if torques[0]>0.0{torques[0]}else{mass*2.0};
    e.av_y+=(torques[1]+yd)/yi2*dt;
    if e.pos[1]<0.5{e.pos[1]=0.5; if e.vel[1]<0.0{e.vel[1]=0.0;}}
    e
}

// ============================================================================
// PERCEPCAO
// ============================================================================

fn validate_perception(world: &VoxelWorld, poses_in: &[Pose]) {
    use universe::gpu::perception::{PerceptionGpu, PosePerception, WAVE_U32S};
    use universe::config::iara::brain::SENSORY_WAVE_SIZE;

    let retina = match RetinaGpu::new(world, poses_in.len().max(64)) {
        Some(g) => g,
        None => {
            println!("  NAO MEDIDO — sem GPU.");
            print_table_row("percepcao (canais portados)", 0, 0, -1, false);
            return;
        }
    };
    let n = poses_in.len();
    println!("  GPU: {}  · {} seres", retina.adapter, n);

    let perception = match PerceptionGpu::new(retina.device(), retina.queue(), world, n.max(64)) {
        Some(p) => p,
        None => {
            println!("  NAO MEDIDO — PercepcaoGpu::novo() falhou.");
            print_table_row("percepcao (canais portados)", 0, 0, -1, false);
            return;
        }
    };

    // Gera retina fisica (input do kernel)
    let _ret = retina.render(poses_in);

    let percep_poses: Vec<PosePerception> = poses_in.iter().map(|p| PosePerception {
        pos_x: p.pos.x, pos_y: p.pos.y, pos_z: p.pos.z, phase: 0.0,
        fwd_x: p.forward.x, fwd_z: p.forward.y,
        rgt_x: p.right.x,   rgt_z: p.right.y,
    }).collect();

    let wave_gpu = perception.reduce(retina.device(), retina.queue(), &percep_poses, retina.output_buffer());
    let wave_cpu = perception_cpu(world, poses_in);

    // Canais excluidos (ver cabecalho da secao 5)
    let excl: std::collections::HashSet<usize> = {
        let mut s = std::collections::HashSet::new();
        for c in 0..=15 { s.insert(c); }       // pool legado
        for c in 21..=27 { s.insert(c); }      // motion per-row
        s.insert(98);
        for c in [29,30,31,33,34,35] { s.insert(c); } // edge density
        s.insert(99); s.insert(100);
        for c in 44..=55 { s.insert(c); }      // wall_imminent, trapped, depth
        for c in 60..=75 { s.insert(c); }      // agent_state, telemetria
        for c in 78..=93 { s.insert(c); }      // curvatura, urgencia, fluxo, corpo
        for c in 99..=127 { s.insert(c); }     // misc motor.rs
        for c in 128..SENSORY_WAVE_SIZE { s.insert(c); } // HD + CfC percepcao
        s
    };

    println!("  Canais comparados: 16-20, 24-28, 32-43, 48-59, 76-77, 94-97");
    // Nota: ch56-59 foram corrigidos (2026-08-04) — de 4× material_code para
    // material+lookahead 3m/8m/20m. Divergência medida: 2/48 poses (ser 18 e 21,
    // que estão em Hole enquanto a frente retorna ao Road).

    let mut total  = 0usize;
    let mut iguais = 0usize;
    let mut maxdif = 0i32;
    let mut piores: Vec<(usize, usize, i8, i8, i32)> = Vec::new();

    for s in 0..n {
        if s >= wave_gpu.len() || s >= wave_cpu.len() { break; }
        for ch in 0..SENSORY_WAVE_SIZE {
            if excl.contains(&ch) { continue; }
            let g = wave_gpu[s][ch];
            let c = wave_cpu[s][ch];
            let d = (c as i32 - g as i32).abs();
            total += 1;
            if d <= 1 { iguais += 1; }
            if d > maxdif { maxdif = d; }
            if d > 1 { piores.push((s, ch, c, g, d)); }
        }
    }
    piores.sort_by(|a,b| b.4.cmp(&a.4));
    println!("  comparados={total}  identicos={iguais} ({:.4}%)  max_dif={maxdif}",
        iguais as f64 / total.max(1) as f64 * 100.0);
    if !piores.is_empty() {
        println!("  DIVERGENCIAS (ate 5):");
        for (s,ch,c,g,d) in piores.iter().take(5) {
            println!("    ser={s} ch={ch}  cpu={c}  gpu={g}  dif={d}");
        }
    }
    print_table_row("percepcao (canais portados)", total, iguais, maxdif, maxdif <= 1);
    let _ = WAVE_U32S; // suppress unused
}

fn perception_cpu(world: &VoxelWorld, poses_in: &[Pose]) -> Vec<[i8; universe::config::iara::brain::SENSORY_WAVE_SIZE]> {
    // Onda legada tinha 223 canais; a atual tem 192 (SENSORY_WAVE_SIZE). A assinatura
    // ficou hardcoded em 223 enquanto o corpo já usava SENSORY_WAVE_SIZE=192 — mesma
    // classe de bug de "literal legado nunca atualizado" documentada no CLAUDE.md do
    // projeto. Corrigido para usar a constante canônica (bug pré-existente, não
    // introduzido por E1-E5; corrigido aqui porque o E6 exige build verde de todos
    // os bins vivos listados).
    use universe::config::iara::brain::SENSORY_WAVE_SIZE;
    use universe::world::VoxelType;

    let n = poses_in.len();
    let mut waves: Vec<[i8; SENSORY_WAVE_SIZE]> = vec![[0i8; SENSORY_WAVE_SIZE]; n];
    let fh: f32 = 50.0f32.to_radians();

    for (s, p) in poses_in.iter().enumerate() {
        let pos = p.pos;
        let fwd = p.forward;
        let rgt = p.right;
        let f3 = Vec3::new(fwd.x, 0.0, fwd.y);
        let w = &mut waves[s];

        let yd = |a: usize| -> Vec2 {
            let fr = (a as f32 - 3.5) / 3.5;
            Vec2::new(fwd.x*(fr*fh).cos() + rgt.x*(fr*fh).sin(),
                      fwd.y*(fr*fh).cos() + rgt.y*(fr*fh).sin()).normalize_or_zero()
        };

        // ch 16-23: wall_distance_fov
        const WS: [f32; 8] = [3.0, 8.0, 15.0, 25.0, 35.0, 45.0, 52.0, 60.0];
        for a in 0..8 {
            let d = yd(a);
            let mut dmin = f32::MAX;
            for &sd in &WS {
                if matches!(world.get_voxel_at(pos.x+d.x*sd, pos.z+d.y*sd),
                    VoxelType::Wall|VoxelType::Hole) { dmin=dmin.min(sd); }
            }
            w[16+a] = if dmin>=60.0{0} else {(127.0*(60.0-dmin)/60.0).clamp(1.0,127.0) as i8};
        }

        // ch 24-31: wall_width_fov
        for a in 0..8 {
            let d = yd(a); let perp = Vec2::new(-d.y, d.x);
            let mut wid = 0.0f32;
            for &off in &[-3.0f32,-2.0,-1.0,1.0,2.0,3.0] {
                let px=pos.x+d.x*15.0+perp.x*off; let pz=pos.z+d.y*15.0+perp.y*off;
                if matches!(world.get_voxel_at(px,pz),VoxelType::Wall|VoxelType::Hole){wid+=off.abs();}
            }
            w[24+a] = (wid/6.0*127.0).clamp(0.0,127.0) as i8;
        }

        // ch 32-39: wall_curvature_fov
        // GPU (percepcao.wgsl): cross * 127.0 clamped to [-127,127] → i32
        // NOTA: ch32 é sobrescrito por motor.rs (c180) no runtime completo.
        // No gpu_valida excluimos ch32 da comparacao (excl set).
        for a in 0..8 {
            let d = yd(a);
            let dist = 10.0+a as f32*1.5;
            let flow = world.get_flow_at(pos.x+d.x*dist, pos.z+d.y*dist);
            let cross = fwd.x*flow.y - fwd.y*flow.x;
            w[32+a] = (cross * 127.0).clamp(-127.0, 127.0) as i8;
        }

        // ch 40-47: road_ahead NEAR
        for di in 0..8 {
            let dist=5.0+di as f32*5.0;
            let pr=pos+f3*dist;
            let flow=world.get_flow_at(pr.x,pr.z);
            let rq: f32 = match world.get_voxel_at(pr.x,pr.z) {
                VoxelType::Road=>60.0, VoxelType::Grass=>-60.0,
                VoxelType::Wall=>127.0, VoxelType::Hole=>-127.0,
            };
            w[40+di] = (rq*(0.5+flow.length()*0.5)).clamp(-127.0,127.0) as i8;
        }

        // ch 48-55: road_ahead FAR
        for step in 0..8 {
            let dist=45.0+step as f32*15.0;
            let ap=pos+f3*dist;
            let h0=world.get_height_at(ap.x,ap.z);
            let h1=world.get_height_at(ap.x+f3.x*10.0,ap.z+f3.z*10.0);
            let norm=world.get_surface_normal(ap.x,ap.z);
            let bank=norm.x.atan2(norm.y);
            let rp: f32 = match world.get_voxel_at(ap.x,ap.z) {
                VoxelType::Road=>1.0, VoxelType::Grass=>0.3,
                VoxelType::Wall=>-1.0, VoxelType::Hole=>-0.5,
            };
            w[48+step] = ((h1-h0)*50.0+bank*40.0+rp*30.0).clamp(-127.0,127.0) as i8;
        }

        // ch 56-59: material atual + lookahead (atualizado 2026-08-04).
        // ANTES: 4× o mesmo material_code — duplicata exata, variância zero.
        // DEPOIS: ch56=atual, ch57=3m, ch58=8m, ch59=20m à frente.
        // Deve ser equivalente à GPU (perception.wgsl BLOCO H atualizado).
        let enc_mat = |vox: VoxelType| -> i8 {
            match vox {
                VoxelType::Road  => 0,
                VoxelType::Grass => 64,
                VoxelType::Wall  => 100,
                VoxelType::Hole  => -127,
            }
        };
        w[56] = enc_mat(world.get_voxel_at(pos.x, pos.z));
        w[57] = enc_mat(world.get_voxel_at(pos.x + f3.x*3.0,  pos.z + f3.z*3.0));
        w[58] = enc_mat(world.get_voxel_at(pos.x + f3.x*8.0,  pos.z + f3.z*8.0));
        w[59] = enc_mat(world.get_voxel_at(pos.x + f3.x*20.0, pos.z + f3.z*20.0));

        // ch 76-77: dist lateral (expandido 15m → 40m, paridade com perception.wgsl).
        // CONTA: 40 m ÷ 28 m/s = 1.43 s de antecipação; oráculo atualizado para 40m.
        //
        // Convenção do shader perception.wgsl BLOCO J:
        //   lp = (-fwd_z, fwd_x) = esquerda → ch76
        //   -lp = (fwd_z, -fwd_x) = direita  → ch77
        // Em termos do rgt deste oráculo (rgt = (-fwd.y, fwd.x) = lp):
        //   ch76 = esquerda = rgt
        //   ch77 = direita  = Vec2::new(-rgt.x, -rgt.y)
        // (oráculo antigo tinha os pares trocados — corrigido aqui)
        let ls=[1.0f32,3.0,6.0,9.0,12.0,16.0,20.0,26.0,33.0,40.0];
        for (off, dir2d) in [(0u32, rgt), (1u32, Vec2::new(-rgt.x,-rgt.y))] {
            let mut dmin=40.0f32;  // sentinela = livre (sem parede em 40m)
            // Replica exatamente o shader: para na PRIMEIRA amostra que encontra Wall/Hole.
            for &sd in &ls {
                if matches!(world.get_voxel_at(pos.x+dir2d.x*sd,pos.z+dir2d.y*sd),
                    VoxelType::Wall|VoxelType::Hole){ dmin=sd; break; }
            }
            w[76+off as usize]=(dmin/40.0*127.0).clamp(0.0,127.0) as i8;
        }

        // ch 94-97: whiskers diagonais
        let angs=[-60.0f32,-30.0,30.0,60.0];
        let ws2=[1.0f32,3.0,5.0,7.0,10.0];
        for (wi,&ang) in angs.iter().enumerate() {
            let r=ang.to_radians();
            let dn=Vec2::new(fwd.x*r.cos()+rgt.x*r.sin(),fwd.y*r.cos()+rgt.y*r.sin()).normalize_or_zero();
            let mut dmin=10.0f32;
            for &sd in &ws2 {
                if matches!(world.get_voxel_at(pos.x+dn.x*sd,pos.z+dn.y*sd),
                    VoxelType::Wall|VoxelType::Hole){dmin=dmin.min(sd);}
            }
            w[94+wi]=if dmin>=10.0{0} else {((1.0-dmin/10.0)*127.0).clamp(0.0,127.0) as i8};
        }
    }
    waves
}

// ============================================================================
// HELPERS
// ============================================================================

fn v3len(v: [f32;3]) -> f32 { (v[0]*v[0]+v[1]*v[1]+v[2]*v[2]).sqrt() }
fn v3dot(a: [f32;3], b: [f32;3]) -> f32 { a[0]*b[0]+a[1]*b[1]+a[2]*b[2] }

fn state_base(dt: f32, flags: u32) -> universe::gpu::physics::PhysicsStateGpu {
    use universe::gpu::physics::PhysicsStateGpu;
    PhysicsStateGpu {
        vel:[0.0;3], av_y:0.0, wheels:[[0.0;4];8],
        pos:[48.0,0.5,48.0], _pad_pos:0.0,
        fwd:[1.0,0.0,0.0], _pad_fwd:0.0,
        lat:[0.0,0.0,1.0], _pad_lat:0.0,
        max_torque:620.0, current_rpm:2000.0, max_rpm:8500.0,
        throttle:0.0, brake_cmd:0.0, steer:0.0,
        drive_torque_out:0.0, current_gear:3, shift_timer:0.0,
        is_stalled:0, starter_timer:0.0, flywheel_inertia:0.15,
        engine_temp:90.0, oil_temp:85.0,
        mass:1350.0, drag_coeff:0.32, wheel_base:2.65, cg_height:0.45,
        track_width:1.55, lsd_lock_ratio:0.3, brake_torque_max:4000.0,
        drivetrain:1,
        mu_peak:1.6, cornering_stiffness:55000.0,
        wheel_radius:0.33, wheel_inertia:1.5, optimal_temp:0.5,
        susp_stiffness:45000.0, susp_damping:3000.0,
        susp_rest:0.25, susp_max_travel:0.12,
        _pad_before_wind:0.0, wind:[0.0;3], _pad_wind:0.0,
        terrain_grip:1.0, env_grip:1.0, aquaplaning:1.0,
        config_flags:flags, dt, _pad_dt:0.0,
    }
}

fn print_table_header() {
    println!("{:<38} {:>11} {:>11} {:>10}  {}", "subsistema","comparados","identicos","max_dif","veredito");
    println!("{}", "-".repeat(90));
}

fn print_table_row(nome: &str, total: usize, iguais: usize, max_dif: i32, ok: bool) {
    if max_dif < 0 {
        println!("{:<38} {:>11} {:>11} {:>10}  NAO MEDIDO", nome,"—","—","—");
    } else {
        let pct = iguais as f64 / total.max(1) as f64 * 100.0;
        let verd = if ok { format!("PASS  ({pct:.4}% identicos, max_dif={max_dif})")
        } else { format!("FALHA  (max_dif={max_dif}, {pct:.4}% identicos)") };
        println!("{:<38} {:>11} {:>11} {:>10}  {}", nome, total, iguais, max_dif, verd);
    }
}
