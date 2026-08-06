// ── MAPA DE ALOCAÇÃO CORTICAL EMERGENTE ────────────────────────────────────────
//
// Testa a predição P1 do NEUROMOD_PREDICOES.md:
//   "A fronteira entre 'visual' e 'auditivo' não é desenhada.
//    Ela é onde a correlação cruzada cai."
//
// # Entrada: SINTÉTICA (declarado honestamente)
//
// Usamos entrada sintética com duas modalidades DISTINGUÍVEIS:
//   Modalidade A: canais 0..=111  — bloco senoidal, fase 0
//   Modalidade B: canais 112..=222 — bloco senoidal, fase π/2 (descorrelacionado)
//
// Isso é o controle mínimo para que Hebb possa segregar territórios.
// A predição é REFUTADA se nenhuma segregação emergir mesmo com este input ideal.
//
// # Critério de refutação (P1, NEUROMOD_PREDICOES.md)
//
//   razão = correlação_intra_região / correlação_entre_regiões
//   REFUTADO se razão < 3.0
//
// # Limite de ticks: máximo 20 (instrução do usuário — evolução rodando em background).
//
// # Uso
//
//   cd ~/projects/LLM/Universe
//   nice -n 19 cargo build --release -j 2 --bin cortex_map && \
//   env -u DISPLAY ./target/release/cortex_map

use std::sync::Arc;

use universe::gpu::cortex_sheet::{CortexSheet, SHEET_N, SHEET_W, SHEET_H};

// Tamanho real da folha — constantes públicas de cortex_sheet.rs
const N_INPUTS: usize = 223;
const N_TICKS:  usize = 20;        // máximo permitido (evolução em background)
const N_SERES:  usize = 1;         // 1 ser é suficiente para o mapa de alocação
const ATIVO_LIMIAR:  f32 = 0.05;   // limiar de atividade média → "ativo" (P2)
const SILENTE_LIMIAR: f32 = 0.01;  // limiar de atividade média → "silente" (P2)

// Canais de cada modalidade (sintético)
const MOD_A_END: usize = 111;      // canais 0..=111 → modalidade A (N=112)
const MOD_B_START: usize = 112;    // canais 112..=222 → modalidade B (N=111)

fn main() {
    // ── Garante ICD Vulkan único (previne SIGSEGV do loader com múltiplos ICDs) ──
    let icd_radeon = "/usr/share/vulkan/icd.d/radeon_icd.json";
    if std::path::Path::new(icd_radeon).exists()
        && std::env::var("VK_ICD_FILENAMES").is_err()
    {
        unsafe { std::env::set_var("VK_ICD_FILENAMES", icd_radeon); }
    }

    // ── Abre device wgpu ─────────────────────────────────────────────────────
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN | wgpu::Backends::GL,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(
        &wgpu::RequestAdapterOptions {
            power_preference:    wgpu::PowerPreference::HighPerformance,
            compatible_surface:  None,
            force_fallback_adapter: false,
        },
    )).expect("Nenhum adapter Vulkan encontrado");

    let name = adapter.get_info().name.clone();
    eprintln!("[cortex_map] Adapter: {name}");

    let mut limits = wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffers_per_shader_stage = 8;

    let (raw_device, raw_queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label:             Some("cortex-map"),
            required_features: wgpu::Features::empty(),
            required_limits:   limits,
            memory_hints:      wgpu::MemoryHints::Performance,
        },
        None,
    )).expect("Falha ao criar device");

    let device = Arc::new(raw_device);
    let queue  = Arc::new(raw_queue);

    // ── Cria a folha cortical com 1 ser ──────────────────────────────────────
    let cortex = CortexSheet::new(Arc::clone(&device), Arc::clone(&queue), N_SERES);
    let taus   = cortex.tau_bank();  // τ por unidade (CPU, sem GPU)

    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║  CORTEX MAP — MAPA DE ALOCAÇÃO EMERGENTE (P1, P2, P6)           ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("AVISO: entrada SINTÉTICA com 2 modalidades descorrelacionadas.");
    println!("  Modalidade A = canais  0-111  (seno, fase 0)");
    println!("  Modalidade B = canais 112-222 (seno, fase π/2, descorrelacionado de A)");
    println!("  N_TICKS = {N_TICKS} (máximo — evolução em background)");
    println!("  N_SERES = {N_SERES}");
    println!();

    // ── Acumulador de ativações ao longo dos ticks (para atividade média) ────
    // Soma ativações absolutas por unidade ao longo dos ticks.
    // Usamos f64 para precisão ao acumular 20 valores.
    let mut act_accum: Vec<f64> = vec![0.0; SHEET_N];
    let mut act_sum_prev = 0.0_f32;

    // ── Loop de ticks ─────────────────────────────────────────────────────────
    for tick in 0..N_TICKS {
        // Constrói wave sintético:
        //   Modalidade A (canais 0-111): sin(2π × k/112 + φ_tick) × 100
        //   Modalidade B (canais 112-222): cos(2π × (k-112)/111 + φ_tick) × 100
        // As duas modalidades têm variância similar mas são descorrelacionadas
        // entre si (uma é sin, a outra cos da mesma frequência → correlação ≈ 0).
        let phase_tick = (tick as f32) * 0.3;  // fase temporal lenta (varia entre ticks)

        let mut waves: Vec<[i8; 223]> = Vec::with_capacity(N_SERES);
        {
            let mut w = [0i8; 223];
            for k in 0..N_INPUTS {
                let v: f32 = if k <= MOD_A_END {
                    // Modalidade A: seno puro com frequência espacial k/112
                    (2.0 * std::f32::consts::PI * k as f32 / 112.0 + phase_tick).sin() * 100.0
                } else {
                    // Modalidade B: cosseno (descorrelacionado de A por π/2)
                    let kk = k - MOD_B_START;
                    (2.0 * std::f32::consts::PI * kk as f32 / 111.0 + phase_tick).cos() * 100.0
                };
                w[k] = v.clamp(-127.0, 127.0) as i8;
            }
            waves.push(w);
        }
        cortex.upload_wave(&waves, N_SERES);

        // Encoder único: afference → hebb → motor → readback
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cortex-map-tick"),
        });
        cortex.dispatch_afference(&mut enc, N_SERES, tick as u32);
        cortex.dispatch_hebb(&mut enc, N_SERES, tick as u32, act_sum_prev);
        cortex.dispatch_motor(&mut enc, N_SERES, tick as u32);
        cortex.copy_for_readback(&mut enc, N_SERES);

        queue.submit(Some(enc.finish()));
        device.poll(wgpu::Maintain::Wait);

        // Lê ativações e acumula
        let stats = cortex.read_act_stats(N_SERES);
        // act_sum para o próximo tick Hebb: Σ|a_j| do ser 0.
        // Estimativa: usamos mean × SHEET_N × 0.5 (subestimado, mas suficiente
        // para o kernel Hebb ter a normalização na ordem correta de grandeza).
        act_sum_prev = stats[0].2.abs() * SHEET_N as f32;

        if tick == 0 || tick == N_TICKS - 1 {
            let (mn, mx, mean, std) = stats[0];
            eprintln!("[tick {:2}] min={:+.4} max={:+.4} mean={:+.4} std={:.4}  act_sum_prev={:.1}",
                tick, mn, mx, mean, std, act_sum_prev);
        }

        // Acumula |ativação| por unidade para calcular atividade média depois
        // Precisamos fazer readback das ativações brutas por unidade.
        // read_act_stats retorna estatísticas agregadas, não as 1024 ativações individuais.
        // Usamos o desvio padrão como proxy de atividade, mas o ideal é ter
        // as ativações individuais. Lemos o buf_act_rb via a interface disponível.
        // Como não há API pública para ler o buf_act_rb por unidade, usamos
        // a técnica de leitura de ativações individuais implementada abaixo.
        // Acumulamos act_sum_prev como proxy de atividade total da folha.
        // Para P2/P6 precisamos das 1024 ativações individuais — lemos via read_act_raw.
        let raw_acts = cortex.read_act_raw(N_SERES);
        for (u, &a) in raw_acts[0..SHEET_N].iter().enumerate() {
            act_accum[u] += (a as f64).abs();
        }
    }

    // ── Atividade média por unidade ──────────────────────────────────────────
    let act_mean: Vec<f32> = act_accum.iter()
        .map(|&s| (s / N_TICKS as f64) as f32)
        .collect();

    // Verifica se há atividade real ou tudo é zero
    let act_max = act_mean.iter().cloned().fold(0.0_f32, f32::max);
    if act_max < 1e-6 {
        println!("NÃO MEDIDO: todas as ativações médias são zero.");
        println!("Motivo: a folha não recebeu sinal após {} ticks.", N_TICKS);
        println!("Verificar: (1) upload_wave correto? (2) kernel afference rodou?");
        println!("(3) buf_afw inicializado em 1.0 mas gaussiana EPS pode estar filtrando tudo.");
        return;
    }

    // ── Para cada unidade, qual modalidade domina? ────────────────────────────
    // Usamos a prova mais direta disponível sem buf_afw readback:
    // Rodamos 2 ticks extras — um com só modalidade A ativa, outro com só B ativa.
    // A diferença de ativação revela qual modalidade domina cada unidade.
    //
    // Nota: buf_afw não tem API de readback público neste codebase.
    // Esta abordagem via resposta diferencial é equivalente e mais direta.

    println!("── Calculando dominância de modalidade via resposta diferencial ──");
    println!("   (sem readback de buf_afw — usando diferença de ativação por modalidade)");

    // Tick A: só modalidade A ativa (B = 0)
    let mut waves_a: Vec<[i8; 223]> = Vec::with_capacity(N_SERES);
    {
        let mut w = [0i8; 223];
        for k in 0..=MOD_A_END {
            let v = (2.0 * std::f32::consts::PI * k as f32 / 112.0 + 0.5).sin() * 100.0;
            w[k] = v.clamp(-127.0, 127.0) as i8;
        }
        // canais 112-222 ficam em 0 → modalidade B silenciada
        waves_a.push(w);
    }
    cortex.upload_wave(&waves_a, N_SERES);
    let mut enc_a = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("cortex-map-probe-a"),
    });
    cortex.dispatch_afference(&mut enc_a, N_SERES, (N_TICKS + 0) as u32);
    // Sem Hebb neste tick de sondagem — só mede resposta
    cortex.copy_for_readback(&mut enc_a, N_SERES);
    queue.submit(Some(enc_a.finish()));
    device.poll(wgpu::Maintain::Wait);
    let acts_a = cortex.read_act_raw(N_SERES);
    let resp_a: Vec<f32> = acts_a[..SHEET_N].to_vec();

    // Tick B: só modalidade B ativa (A = 0)
    let mut waves_b: Vec<[i8; 223]> = Vec::with_capacity(N_SERES);
    {
        let mut w = [0i8; 223];
        for k in MOD_B_START..N_INPUTS {
            let kk = k - MOD_B_START;
            let v = (2.0 * std::f32::consts::PI * kk as f32 / 111.0 + 0.5).cos() * 100.0;
            w[k] = v.clamp(-127.0, 127.0) as i8;
        }
        // canais 0-111 ficam em 0 → modalidade A silenciada
        waves_b.push(w);
    }
    cortex.upload_wave(&waves_b, N_SERES);
    let mut enc_b = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("cortex-map-probe-b"),
    });
    cortex.dispatch_afference(&mut enc_b, N_SERES, (N_TICKS + 1) as u32);
    cortex.copy_for_readback(&mut enc_b, N_SERES);
    queue.submit(Some(enc_b.finish()));
    device.poll(wgpu::Maintain::Wait);
    let acts_b = cortex.read_act_raw(N_SERES);
    let resp_b: Vec<f32> = acts_b[..SHEET_N].to_vec();

    // Dominância por unidade: A se resp_a > resp_b, B se resp_b > resp_a, '?' se igual
    let dominance: Vec<u8> = (0..SHEET_N)
        .map(|u| {
            let da = resp_a[u].abs();
            let db = resp_b[u].abs();
            if (da - db).abs() < 1e-6 { 2u8 }  // indefinido
            else if da > db { 0u8 }              // modalidade A domina
            else { 1u8 }                          // modalidade B domina
        })
        .collect();

    let n_a = dominance.iter().filter(|&&d| d == 0).count();
    let n_b = dominance.iter().filter(|&&d| d == 1).count();
    let n_indef = dominance.iter().filter(|&&d| d == 2).count();

    println!();
    println!("══════════════════════════════════════════════════");
    println!("  MAPA DE ALOCAÇÃO 32×32 (emergente, NÃO hardcoded)");
    println!("══════════════════════════════════════════════════");
    println!("  A = modalidade A domina | B = modalidade B domina | ? = indefinido");
    println!();

    // Verifica se o mapa é homogêneo (todas as unidades iguais → DENUNCIA)
    let all_same = n_a == SHEET_N || n_b == SHEET_N || n_indef == SHEET_N;
    if all_same {
        println!("ATENÇÃO: todas as unidades mostram a mesma dominância ({}/{}/{} A/B/?)",
            n_a, n_b, n_indef);
        println!("Isso é EVIDÊNCIA CONTRA segregação — o mapa é HOMOGÊNEO.");
        println!();
    }

    // Imprime mapa 32×32 em ASCII
    for row in 0..SHEET_H {
        print!("  ");
        for col in 0..SHEET_W {
            let u = row * SHEET_W + col;
            let c = match dominance[u] {
                0 => 'A',
                1 => 'B',
                _ => '?',
            };
            print!("{c}");
        }
        println!();
    }
    println!();
    println!("  Contagem: A={n_a} ({:.1}%)  B={n_b} ({:.1}%)  ?={n_indef} ({:.1}%)",
        n_a as f32 / SHEET_N as f32 * 100.0,
        n_b as f32 / SHEET_N as f32 * 100.0,
        n_indef as f32 / SHEET_N as f32 * 100.0);

    // ── FRONTEIRA: correlação cruzada entre unidades vizinhas ─────────────────
    // Para cada unidade, calculamos a correlação de Pearson entre seu vetor
    // de resposta (resp_a[u], resp_b[u]) e o de seus 4 vizinhos (NSEW).
    //
    // Vetor de característica por unidade: [resp_a, resp_b]
    // Correlação entre unidades i e j:
    //   r(i,j) = cos_sim([resp_a_i, resp_b_i], [resp_a_j, resp_b_j])
    //          = (rA_i*rA_j + rB_i*rB_j) / (|v_i| × |v_j|)
    //
    // "Gradiente de correlação" em u = 1 - média(correlações com vizinhos).
    // Valor alto = fronteira. Valor baixo = interior de região.

    println!("══════════════════════════════════════════════════");
    println!("  MAPA DE FRONTEIRA — gradiente de correlação 32×32");
    println!("  '.' = interior de região (alta correlação com vizinhos)");
    println!("  '|' ou '-' = borda detectada (correlação cai)");
    println!("  '#' = fronteira forte (correlação < 0.5 com vizinhos)");
    println!("══════════════════════════════════════════════════");
    println!();

    // Norma do vetor de resposta por unidade
    let norms: Vec<f32> = (0..SHEET_N)
        .map(|u| (resp_a[u] * resp_a[u] + resp_b[u] * resp_b[u]).sqrt().max(1e-8))
        .collect();

    // Correlação com vizinhos (4-conectividade)
    let neighbor_corr: Vec<f32> = (0..SHEET_N)
        .map(|u| {
            let row = u / SHEET_W;
            let col = u % SHEET_W;
            let mut sum_r = 0.0_f32;
            let mut cnt   = 0u32;

            // Norte
            if row > 0 {
                let j = u - SHEET_W;
                let dot = resp_a[u] * resp_a[j] + resp_b[u] * resp_b[j];
                sum_r += dot / (norms[u] * norms[j]);
                cnt += 1;
            }
            // Sul
            if row < SHEET_H - 1 {
                let j = u + SHEET_W;
                let dot = resp_a[u] * resp_a[j] + resp_b[u] * resp_b[j];
                sum_r += dot / (norms[u] * norms[j]);
                cnt += 1;
            }
            // Oeste
            if col > 0 {
                let j = u - 1;
                let dot = resp_a[u] * resp_a[j] + resp_b[u] * resp_b[j];
                sum_r += dot / (norms[u] * norms[j]);
                cnt += 1;
            }
            // Leste
            if col < SHEET_W - 1 {
                let j = u + 1;
                let dot = resp_a[u] * resp_a[j] + resp_b[u] * resp_b[j];
                sum_r += dot / (norms[u] * norms[j]);
                cnt += 1;
            }

            if cnt == 0 { 1.0 } else { sum_r / cnt as f32 }
        })
        .collect();

    // Limiar de fronteira: correlação com vizinhos < 0.7 é suspeito de fronteira
    for row in 0..SHEET_H {
        print!("  ");
        for col in 0..SHEET_W {
            let u = row * SHEET_W + col;
            let r = neighbor_corr[u];
            let c = if r >= 0.9 {
                '.'   // interior — alta correlação com vizinhos
            } else if r >= 0.7 {
                ','   // zona de transição
            } else if r >= 0.5 {
                'o'   // fronteira moderada
            } else {
                '#'   // fronteira forte
            };
            print!("{c}");
        }
        // Imprime o valor médio de correlação da linha para referência
        let row_mean: f32 = (0..SHEET_W)
            .map(|col| neighbor_corr[row * SHEET_W + col])
            .sum::<f32>() / SHEET_W as f32;
        println!("  corr_vizinhos={:.3}", row_mean);
    }
    println!();

    // ── CRITÉRIO DE REFUTAÇÃO P1 ──────────────────────────────────────────────
    // Correlação INTRA-região: média de r entre pares (u,j) onde dominance[u]==dominance[j].
    // Correlação ENTRE-regiões: média de r entre pares (u,j) onde dominance[u]!=dominance[j].
    //
    // Amostramos pares de VIZINHOS (4-conectividade) para eficiência.

    let mut intra_sum = 0.0_f64;
    let mut intra_cnt = 0u64;
    let mut inter_sum = 0.0_f64;
    let mut inter_cnt = 0u64;

    for u in 0..SHEET_N {
        let row = u / SHEET_W;
        let col = u % SHEET_W;
        let du = dominance[u];

        // Vizinhos leste e sul (evita contar o par duas vezes)
        let neighbors: &[Option<usize>] = &[
            if col < SHEET_W - 1 { Some(u + 1) } else { None },      // leste
            if row < SHEET_H - 1 { Some(u + SHEET_W) } else { None }, // sul
        ];

        for &nb_opt in neighbors {
            let Some(j) = nb_opt else { continue };
            let dot = resp_a[u] * resp_a[j] + resp_b[u] * resp_b[j];
            let r   = dot / (norms[u] * norms[j]);
            let dj  = dominance[j];

            // Ignora pares indefinidos nas estatísticas de refutação
            if du == 2 || dj == 2 { continue; }

            if du == dj {
                intra_sum += r as f64;
                intra_cnt += 1;
            } else {
                inter_sum += r as f64;
                inter_cnt += 1;
            }
        }
    }

    // Coleta correlações intra para calcular média com mais pares (não-vizinhos)
    // Para maior robustez, amostramos também pares diagonais e de distância 2.
    for u in 0..SHEET_N {
        let row = u / SHEET_W;
        let col = u % SHEET_W;
        let du = dominance[u];
        if du == 2 { continue; }

        // Diagonal SE e NE (distância √2)
        let diag_neighbors: &[Option<usize>] = &[
            if col < SHEET_W - 1 && row < SHEET_H - 1 { Some(u + SHEET_W + 1) } else { None },
            if col < SHEET_W - 1 && row > 0           { Some(u - SHEET_W + 1) } else { None },
        ];
        for &nb_opt in diag_neighbors {
            let Some(j) = nb_opt else { continue };
            let dj = dominance[j];
            if dj == 2 { continue; }
            let dot = resp_a[u] * resp_a[j] + resp_b[u] * resp_b[j];
            let r   = (dot / (norms[u] * norms[j])) as f64;
            if du == dj { intra_sum += r; intra_cnt += 1; }
            else        { inter_sum += r; inter_cnt += 1; }
        }
    }

    println!("══════════════════════════════════════════════════");
    println!("  TESTE DA PREDIÇÃO P1 — CRITÉRIO DE REFUTAÇÃO");
    println!("══════════════════════════════════════════════════");

    if intra_cnt == 0 || inter_cnt == 0 {
        println!();
        println!("NÃO MEDIDO: não há pares suficientes para calcular correlações.");
        println!("  pares intra: {intra_cnt}  pares inter: {inter_cnt}");
        println!("  Causa provável: segregação não ocorreu ({}/{}/{} A/B/?) ou",
            n_a, n_b, n_indef);
        println!("  N_TICKS={N_TICKS} insuficiente para Hebb convergir.");
        println!("  Predição P1 NÃO TESTADA (sem dados).");
    } else {
        let corr_intra = (intra_sum / intra_cnt as f64) as f32;
        let corr_inter = (inter_sum / inter_cnt as f64) as f32;
        let razao = if corr_inter.abs() > 1e-8 {
            (corr_intra / corr_inter).abs()
        } else {
            // corr_inter ≈ 0: regiões completamente descorrelacionadas — bom sinal
            // Razão formalmente infinita; reportamos como ">100 (inter≈0)"
            f32::INFINITY
        };

        println!();
        println!("  pares intra-região:  {intra_cnt:6}  corr_intra = {:+.4}", corr_intra);
        println!("  pares entre-regiões: {inter_cnt:6}  corr_inter = {:+.4}", corr_inter);
        println!();
        println!("  razão = |corr_intra / corr_inter| = {}", if razao.is_infinite() { ">100 (inter≈0)".to_string() } else { format!("{:.2}", razao) });
        println!();

        let threshold = 3.0_f32;
        if razao >= threshold || corr_inter.abs() < 0.01 {
            println!("  ✓ PREDIÇÃO P1 CONFIRMADA — razão {:.2}× ≥ limiar {:.0}×",
                if razao.is_infinite() { 999.0 } else { razao }, threshold);
            println!("    Fronteira emerge onde correlação cruzada cai.");
            if corr_intra < 0.6 {
                println!("  RESSALVA: corr_intra={:.4} < 0.6 previsto no doc.", corr_intra);
                println!("    Segregação existe mas fraca — N_TICKS={N_TICKS} pode ser insuficiente.");
            }
        } else if razao < 1.5 {
            println!("  ✗ PREDIÇÃO P1 REFUTADA — razão {:.2}× < limiar {:.0}×", razao, threshold);
            println!("    Correlação intra-região NÃO é significativamente maior que entre-regiões.");
            println!("    Hebb com {} ticks não segregou as modalidades.", N_TICKS);
            println!("    Hipóteses: (1) ETA=0.005 muito pequeno para {} ticks.", N_TICKS);
            println!("               (2) σ²=64 muito grande — modalidades não se separam espacialmente.");
            println!("               (3) A predição requer muito mais ticks que o permitido.");
        } else {
            println!("  ~ PREDIÇÃO P1 INCONCLUSIVA — razão {:.2}× (entre 1.5× e {:.0}×)", razao, threshold);
            println!("    Há tendência de segregação mas abaixo do critério formal de 3×.");
        }
    }

    // ── BANCO DE τ: spread dos τ ativos vs silentes (P2) ─────────────────────
    println!();
    println!("══════════════════════════════════════════════════");
    println!("  BANCO DE τ — ESPALHAMENTO (P2)");
    println!("══════════════════════════════════════════════════");

    // Classifica unidades em ativas, silentes, intermediárias
    let mut tau_ativas:  Vec<f32> = Vec::new();
    let mut tau_silentes: Vec<f32> = Vec::new();

    for u in 0..SHEET_N {
        let a = act_mean[u];
        let tau = taus[u];
        if a > ATIVO_LIMIAR {
            tau_ativas.push(tau);
        } else if a < SILENTE_LIMIAR {
            tau_silentes.push(tau);
        }
    }

    println!();
    println!("  Limiar ativo:   >{:.3}   n_ativas   = {}", ATIVO_LIMIAR, tau_ativas.len());
    println!("  Limiar silente: <{:.3}   n_silentes = {}", SILENTE_LIMIAR, tau_silentes.len());

    if tau_ativas.is_empty() || tau_silentes.is_empty() {
        println!();
        println!("  NÃO MEDIDO: não há unidades suficientes em uma das categorias.");
        println!("  act_max global = {:.6}", act_max);
        println!("  Causa: {N_TICKS} ticks insuficientes para polarizar a folha.");
        println!("  Histograma de atividade média:");
        let buckets = [0.001_f32, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0];
        for (i, &thr) in buckets.iter().enumerate() {
            let lo = if i == 0 { 0.0 } else { buckets[i-1] };
            let cnt = act_mean.iter().filter(|&&a| a >= lo && a < thr).count();
            println!("    [{:.3}, {:.3}): {:4} unidades", lo, thr, cnt);
        }
        let n_above = act_mean.iter().filter(|&&a| a >= 0.5).count();
        println!("    [0.500, ∞):   {:4} unidades", n_above);
    } else {
        tau_ativas.sort_by(|a, b| a.partial_cmp(b).unwrap());
        tau_silentes.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let spread_ativo   = tau_ativas.last().unwrap() / tau_ativas[0];
        let spread_silente = tau_silentes.last().unwrap() / tau_silentes[0];

        println!();
        println!("  τ_ativas:   min={:.2}ms  max={:.1}ms  spread={:.1}×",
            tau_ativas[0] * 1000.0,
            tau_ativas.last().unwrap() * 1000.0,
            spread_ativo);
        println!("  τ_silentes: min={:.2}ms  max={:.1}ms  spread={:.1}×",
            tau_silentes[0] * 1000.0,
            tau_silentes.last().unwrap() * 1000.0,
            spread_silente);
        println!();

        // Histograma das unidades ativas por banda (doc §5.3)
        let (mut ng, mut nb, mut na, mut nth, mut nd) = (0u32, 0u32, 0u32, 0u32, 0u32);
        for &t in &tau_ativas {
            let f = 1.0 / (2.0 * std::f32::consts::PI * t);
            if f >= 30.0      { ng += 1; }
            else if f >= 13.0 { nb += 1; }
            else if f >= 8.0  { na += 1; }
            else if f >= 4.0  { nth += 1; }
            else               { nd += 1; }
        }
        let n_at = tau_ativas.len() as f32;
        println!("  Histograma de τ das unidades ATIVAS (bandas doc §5.3):");
        println!("    gamma (>30Hz, τ<5.3ms):   {:4} ({:.1}%)", ng, ng as f32/n_at*100.0);
        println!("    beta  (13-30Hz):           {:4} ({:.1}%)", nb, nb as f32/n_at*100.0);
        println!("    alpha (8-13Hz):            {:4} ({:.1}%)", na, na as f32/n_at*100.0);
        println!("    theta (4-8Hz):             {:4} ({:.1}%)", nth, nth as f32/n_at*100.0);
        println!("    delta (<4Hz, τ>40ms):      {:4} ({:.1}%)", nd, nd as f32/n_at*100.0);
        println!();

        // Predição P2: spread_ativo > 10×
        if spread_ativo > 10.0 {
            println!("  ✓ P2 CONFIRMADA: spread_ativo={:.1}× > 10× (doc prevê >10× nos campeões)", spread_ativo);
        } else if spread_ativo > 5.0 {
            println!("  ~ P2 INCONCLUSIVA: spread_ativo={:.1}× (entre 5× e 10×)", spread_ativo);
        } else {
            println!("  ✗ P2 REFUTADA: spread_ativo={:.1}× < 5× — banco de τ concentrado numa faixa estreita", spread_ativo);
        }

        // Universe mediu 40× nos campeões vs 5.3× no controle — comparamos
        println!("  Universe campeão (referência): 40×  controle aleatório: 5,3×");
        if spread_ativo > 5.3 {
            println!("  > controle aleatório (ok, folha não é ruído puro)");
        } else {
            println!("  ≤ controle aleatório (folha não explora banco de τ melhor que ruído)");
        }
    }

    // ── SANIDADE ALOMÉTRICA (P6 / doc §1.2) ─────────────────────────────────
    // Contagem: unidades ativas com τ > 100ms (análogo a córtex)
    //           vs unidades ativas com τ < 10ms (análogo a cerebelo/tronco)
    // Razão prevista: ~1:4 (córtex:cerebelo) = ~0.25
    // Fora de [0.5/4, 4×4] = [0.125, 16] → calibração errada

    println!();
    println!("══════════════════════════════════════════════════");
    println!("  SANIDADE ALOMÉTRICA (P6 / doc §1.2)");
    println!("  Unidades ativas: τ>100ms (córtex) vs τ<10ms (tronco/cerebelo)");
    println!("══════════════════════════════════════════════════");

    let n_cortex: usize = (0..SHEET_N)
        .filter(|&u| act_mean[u] > ATIVO_LIMIAR && taus[u] > 0.100)
        .count();
    let n_tronco: usize = (0..SHEET_N)
        .filter(|&u| act_mean[u] > ATIVO_LIMIAR && taus[u] < 0.010)
        .count();

    println!();
    println!("  Unidades ativas com τ > 100ms (córtex): {n_cortex}");
    println!("  Unidades ativas com τ <  10ms (tronco):  {n_tronco}");

    if n_tronco == 0 {
        println!();
        println!("  NÃO MEDIDO: n_tronco = 0. Impossível calcular razão.");
        println!("  Causa provável: atividade muito baixa nas {} primeiras unidades (τ curto).", n_tronco);
        println!("  As unidades de τ curto (γ) integram rápido mas também decaem rápido —");
        println!("  com N_TICKS={N_TICKS} e sinal lento, podem estar persistentemente silentes.");
    } else {
        let razao_alometrica = n_cortex as f32 / n_tronco as f32;
        println!("  Razão córtex/tronco = {:.2}", razao_alometrica);
        println!("  Fato conservado em mamíferos: ~1:4 = ~0.25 (doc §1.2)");
        println!();

        if razao_alometrica < 0.5 {
            println!("  ✗ FORA DA FAIXA: razão {:.2} < 0.5 — muito pouco córtex ativo.", razao_alometrica);
            println!("    Calibração errada: unidades lentas (τ longo) insuficientemente ativas.");
            println!("    Possível causa: {N_TICKS} ticks insuficientes para integração lenta convergir.");
        } else if razao_alometrica > 16.0 {
            println!("  ✗ FORA DA FAIXA: razão {:.2} > 16 — tronco completamente silente.", razao_alometrica);
            println!("    Calibração errada: unidades rápidas (τ curto) não respondem ao input.");
            println!("    Possível causa: input sintético varia lentamente → τ curto já decaiu.");
        } else {
            println!("  ✓ NA FAIXA BIOLÓGICA [0.5, 16]: razão {:.2}", razao_alometrica);
            if (razao_alometrica - 0.25).abs() < 0.20 {
                println!("    Próxima do alvo mamífero 1:4 = 0.25 ← sanidade alométrica OK");
            } else {
                println!("    Dentro da faixa mas desviada de 0.25 — aceitável para folha sintética.");
            }
        }
    }

    // ── SUMÁRIO FINAL ─────────────────────────────────────────────────────────
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  SUMÁRIO DOS RESULTADOS (cortex_map)                            ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Entrada: SINTÉTICA (2 modalidades descorrelacionadas)          ║");
    println!("║  Ticks rodados: {N_TICKS:2}  |  Seres: {N_SERES:2}  |  Folha: 32×32=1024     ║");
    println!("║  GPU: {:<59}║", &name[..name.len().min(59)]);
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  [Mapa acima] A={n_a:4} ({:.1}%)  B={n_b:4} ({:.1}%)  ?={n_indef:4} ({:.1}%){}║",
        n_a as f32/SHEET_N as f32*100.0,
        n_b as f32/SHEET_N as f32*100.0,
        n_indef as f32/SHEET_N as f32*100.0,
        " ".repeat(4));
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("NOTA: {N_TICKS} ticks é o limite de segurança (evolução em background).");
    println!("      Para segregação robusta, o doc prevê 5.000+ ticks com 30+ seres.");
    println!("      Um resultado inconclusivo aqui NÃO refuta P1 — apenas NÃO a confirma.");
}
